# Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting

**Authors**: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira

**Published**: 2025-05-26 20:53:53

**PDF URL**: [http://arxiv.org/pdf/2505.20521v1](http://arxiv.org/pdf/2505.20521v1)

## Abstract
This paper presents Project Riley, a novel multimodal and multi-model
conversational AI architecture oriented towards the simulation of reasoning
influenced by emotional states. Drawing inspiration from Pixar's Inside Out,
the system comprises five distinct emotional agents - Joy, Sadness, Fear,
Anger, and Disgust - that engage in structured multi-round dialogues to
generate, criticise, and iteratively refine responses. A final reasoning
mechanism synthesises the contributions of these agents into a coherent output
that either reflects the dominant emotion or integrates multiple perspectives.
The architecture incorporates both textual and visual large language models
(LLMs), alongside advanced reasoning and self-refinement processes. A
functional prototype was deployed locally in an offline environment, optimised
for emotional expressiveness and computational efficiency. From this initial
prototype, another one emerged, called Armando, which was developed for use in
emergency contexts, delivering emotionally calibrated and factually accurate
information through the integration of Retrieval-Augmented Generation (RAG) and
cumulative context tracking. The Project Riley prototype was evaluated through
user testing, in which participants interacted with the chatbot and completed a
structured questionnaire assessing three dimensions: Emotional Appropriateness,
Clarity and Utility, and Naturalness and Human-likeness. The results indicate
strong performance in structured scenarios, particularly with respect to
emotional alignment and communicative clarity.

## Full Text


<!-- PDF content starts -->

arXiv:2505.20521v1  [cs.AI]  26 May 2025Project Riley: Multimodal Multi-Agent LLM
Collaboration with Emotional Reasoning and
Voting
Ana Rita Ortigoso1, Gabriel Vieira1, Daniel Fuentes1, Luis Fraz˜ ao1,
Nuno Costa1, and Ant´ onio Pereira1
1Computer Science and Communication Research Centre, Polytechnic
University of Leiria
Preprint Notice: This manuscript is a preprint and is currently under
review for potential publication in the journal Information Fusion .
Abstract
This paper presents Project Riley, a novel multimodal and multi-
model conversational AI architecture oriented towards the simulation
of reasoning influenced by emotional states. Drawing inspiration from
Pixar’s Inside Out, the system comprises five distinct emotional agents
— Joy, Sadness, Fear, Anger, and Disgust — that engage in structured
multi-round dialogues to generate, criticise, and iteratively refine re-
sponses. A final reasoning mechanism synthesises the contributions of
these agents into a coherent output that either reflects the dominant
emotion or integrates multiple perspectives. The architecture incorpo-
rates both textual and visual large language models (LLMs), alongside
advanced reasoning and self-refinement processes. A functional proto-
type was deployed locally in an offline environment, optimised for emo-
tional expressiveness and computational efficiency. From this initial
prototype, another one emerged, called Armando, which was developed
for use in emergency contexts, delivering emotionally calibrated and
factually accurate information through the integration of Retrieval-
Augmented Generation (RAG) and cumulative context tracking. The
Project Riley prototype was evaluated through user testing, in which
participants interacted with the chatbot and completed a structured
questionnaire assessing three dimensions: Emotional Appropriateness,
Clarity and Utility, and Naturalness and Human-likeness. The results
indicate strong performance in structured scenarios, particularly with
respect to emotional alignment and communicative clarity.
1

1 Introduction
The ‘Inside Out’ saga [1, 2], produced by Pixar Animation Studios, provides
a symbolic narrative depiction of the perceptions and emotional interactions
inherent in human experience.
Throughout the development of the saga’s films, the creators aimed not
only to produce artistic entertainment but also to translate fundamental
psychological concepts into a language accessible to the general public. Con-
sequently, the films benefited from consultation with experts in psychology
and neuroscience, including Dacher Keltner, Paul Ekman, and Lisa Damour
[3, 4].
The saga narrates the life of a girl named Riley, with the use of au-
tonomous characters representing distinct emotions serving to personify her
emotional experiences. The first film (2015) [1], is set during Riley’s child-
hood and focuses on five primary emotions: Joy, Sadness, Fear, Anger, and
Disgust. The sequel (2024) [2] portrays Riley’s adolescence and the complex-
ities typical of this developmental stage, introducing additional emotions:
Shame; Envy; Boredom; and Anxiety. These emotions interact dynamically,
guiding Riley’s behavioural responses to life changes and challenges.
Inspired by the films, this paper explores the development of a conver-
sational generative Artificial Intelligence (AI) solution that employs an ap-
proach analogous to the emotional interplay that guides Riley’s behaviour.
The objective is to generate emotionally informed responses, similar to those
produced by human beings who are emotionally aware.
Therefore, this paper presents the architecture of the generative AI con-
versational solution, Project Riley. This system is characterised by its emo-
tional awareness, multi model functionality, and the implementation of so-
phisticated self-refining and voting logic. Project Riley integrates multi-
modal information—text and images—through the coordinated use of mul-
tiple Large Language Models (LLM), each representing a distinct emotion.
It processes user input using both standard and multimodal LLM for tex-
tual and visual analysis, respectively, and orchestrate multi-round emotional
dialogues. The final output results from a structured voting and synthesis
mechanism that uses an advanced reasoning model to combine emotionally
differentiated responses into a coherent answer. This architecture exem-
plifies the potential of GenAI for symbolic and neural fusion, by coupling
the generative capabilities of neural models with symbolic reasoning struc-
tures—such as agent-specific roles, conversational history tracking, voting
with justification, and formally segmented outputs—thereby enabling inter-
pretation and decision-making across heterogeneous data sources.
2

A emergency-oriented variant of the system, named Armando, is also
presented, which architecture was adaptedto real-time, high-stakes environ-
ments with a focus on trust, relevance, and emotional regulation.
To the best of our knowledge, this is the first framework that leverages
generative AI to orchestrate structured affective reasoning through indepen-
dent emotional agents, introducing a novel paradigm for information fusion
via generative emotional modelling. Furthermore, no comparable solution
has been identified in the literature that adapts such an architecture to
emergency response contexts as implemented in Armando, where emotional
regulation, informational accuracy, and user trust are prioritised under high-
stakes conditions.
2 Related Work
2.1 Emotion Recognition and Emotional Intelligence
Recent literature on emotionally intelligent systems primarily addresses the
integration of emotional awareness into AI models, particularly LLM. Rati-
can et al. [5] introduced the 6DE conceptual model, featuring six emotional
dimensions—arousal, valence, dominance, agency, fidelity, and novelty—to
provide a comprehensive framework for analysing human emotions in AI
systems. This approach aims to enhance empathy and contextual relevance,
specifically targeting applications in education, mental health, and assisted
care. Their suggested integration methods include emotion-guided prompt-
ing, annotated datasets, user feedback, emotional database integration, and
emotional quality refinement through post-processing.
Kang et al. [6] similarly focused on emotional integration, albeit within
a social robotic context, employing the Pleasure-Arousal-Dominance (PAD)
emotional model. They developed the SoR-ReAct agent for the Nadine
robot platform, which combines GPT-4 with personalised episodic memory,
enabling emotional simulation and personalised interactions based on histor-
ical data. Unlike Ratican et al., Kang et al. directly incorporated episodic
memory retrieval facilitated by OpenAI’s embeddings, enhancing the robot’s
interactive empathy capabilities through multimodal perception.
In contrast, Liu et al. [7] addressed emotional intelligence specifically
within negotiation scenarios, developing EQ-Negotiator, which integrates
emotion recognition (RoBERTa, GPT-2, DeBERTa) with strategic game-
theory-based reasoning. Their work distinctly emphasised balancing emo-
tional empathy and assertiveness, using Hidden Markov Models for dynamic
emotional adaptations. Notably, their approach outperformed GPT-4 in
3

terms of emotional realism and negotiation efficacy.
Rasool et al. [8] and Liu et al. [9] extended emotional intelligence
into psychotherapy and multimodal dialogue, respectively. Rasool et al.
employed emotional lexicons (NRC Emotion Lexicon, SentiWordNet) com-
bined with hierarchical segmentation and attention mechanisms to enhance
empathetic responses, despite noting a trade-off between empathy and coher-
ence. Liu et al., conversely, integrated emotional and visual modalities using
Emotional Retrieval Module(ERM), Response Emotion Prediction (REP),
and Emotion-Enhanced Response Generation (EERG) models. Their mul-
timodal ELMD system demonstrated superior emotional and contextual re-
sponsiveness compared to traditional dialogue systems.
Chen et al. [10], Motwani et al. [11], and Yang et al. [12] further
highlighted emotional considerations, albeit within specialised frameworks.
Chen et al.’s RECONCILE framework collaboratively utilised multiple LLM
agents to generate emotionally and contextually relevant responses, lever-
aging weighted voting based on reasoning justification. Motwani et al. pro-
posed a sequential training method employing specialised agents in reasoning
and verification tasks, indirectly enhancing emotional intelligence through
robust error correction. Yang et al.’s SupportlyChat directly focused on
therapeutic contexts using Cognitive Behavioural Therapy (CBT), combin-
ing RoBERTa and sentiment analysis to enhance emotional awareness and
professional interaction in mental health support scenarios.
Finally, Brun et al. [13] evaluated emotional sensitivity in technical
support chatbots, specifically employing VADER sentiment analysis within
ChatGPT-3.5. Their findings indicated emotional sensitivity significantly
improved user-perceived competence, empathy, trust, and continued use in-
tention, although without measurably affecting user emotional states.
2.2 Multi-Agent Debate and Collaborative Reasoning
Research exploring multi-agent systems frequently emphasises collaborative
reasoning and decision-making through debate mechanisms. Zhao et al. [14]
tested electoral methods from social choice theory (e.g., Borda, IRV, Ranked
Pairs) in collective decision-making scenarios involving multiple LLM agents,
highlighting notable accuracy improvements in benchmarks like MMLU and
ARC. They demonstrated that even minimal agent ensembles significantly
enhance performance and recommended task-specific tailoring of decision
methods.
Yang et al. [15] combined adversarial debate, weighted voting, and inter-
nal self-correction within a multi-agent framework to mitigate hallucinations
4

and enhance reasoning efficiency. Their methodology involved iterative de-
bate, error documentation, and dynamic weighting, outperforming baseline
methods in accuracy and response speed on tasks like GSM8K and MMLU.
Similarly, Xu et al. [16] proposed a peer-review inspired collaborative
method. Agents independently generated solutions, reviewed peers’ con-
tributions, and refined responses accordingly. This explicit feedback and
diverse model integration consistently improved performance across multi-
ple datasets (GSM8K, StrategyQA, ARC-c).
Collectively, these studies underline the efficacy of structured multi-agent
frameworks, emphasising the critical role of model diversity, debate dynam-
ics, and systematic feedback in enhancing collaborative reasoning and overall
system robustness.
In contrast to existing approaches, the architecture proposed in this work
distinguishes itself by employing a structured multi-agent framework in
which each agent represents a distinct emotion and participates in itera-
tive emotional reasoning. While prior studies tend to integrate emotion
either through static embeddings, sentiment lexicons, or personalised mem-
ory modules [5, 6, 8], this architecture fosters dynamic emotional dialogue
and collective deliberation. Additionally, the integration of symbolic reason-
ing mechanisms—such as voting and justification—enhances explainability,
a feature largely absent in previous neural architectures [10, 11]. The sys-
tem is also extensible to different emotional models and has been adapted
for high-stakes domains, such as emergency response, through the Armando
variant. To the best of our knowledge, no previous work combines emo-
tional simulation, multimodal reasoning, and factual grounding within a
unified architecture.
3 Proposed Architecture
This paper presents an innovative conversational AI system architecture that
processes user queries through distinct emotional lenses, specifically a set of
five basic emotions: Anger, Joy, Sadness, Fear, and Disgust. The system fol-
lows the emotional framework portrayed in the first Inside Out film, which
was inspired by Paul Ekman’s theory of universal emotions [17]. Although
Ekman initially identified six basic emotions—Happiness, Anger, Sadness,
Disgust, Surprise, and Fear—and later expanded the list to include emotions
such as Contempt and Enjoyment [18], the film deliberately reduced the set
to five for narrative clarity [19, 20]. In this work, the same five emotions
5

were retained to preserve the analogy with the film, thereby fostering greater
familiarity and intuitive use—particularly among users who have seen the
film. However, it is important to emphasise that the Project Riley architec-
ture is not limited to this configuration: it is capable of simulating a broader
range of emotions, depending on the requirements of the application and the
computational resources available.
The system can be easily adapted to reflect alternative emotional models,
allowing developers to incorporate different sets of emotions based on specific
application needs or cultural relevance. Furthermore, the proposed archi-
tecture can be aligned with other psychological theories, such as Plutchik’s
Wheel of Emotions [21], Carroll Izard’s Differential Emotions Theory [22],
Jaak Panksepp’s Affective Neuroscience framework [23], or dimensional ap-
proaches like Russell’s Circumplex Model [24] and the Pleasure-Arousal-
Dominance (PAD) model [25]. Such adaptations would require architectural
modifications and prompt reengineering to ensure consistency with the the-
oretical assumptions of the chosen model.
The architecture, as depicted in Figure 1, comprises four sequential
phases: Input; Multi-round Processing; Voting and Analysis; and Final Syn-
thesis.
Upon initialisation, the system establishes separate conversation histo-
ries for each emotional agent.
In the Input phase, users can provide contextual information and images
prior to interaction. These images are described by a vision-based LLM
(vision LLM), enriching the conversational context through visual data.
The processing workflow begins with the user input, which is injected as
emotion-specific context into the conversation history of each agent.
Subsequently, the Multi-round Processing phase is initiated, employing
a textual LLM. Each emotional agent independently generates an initial
response, concurrently processed and reflecting the distinct emotional view-
point (Round 0). These initial responses are documented and serve as the
basis for subsequent interactions.
In the first discussion phase (Round 1), emotional agents review and re-
spond to the initial outputs from the other emotions, maintaining emotional
authenticity while engaging in dynamic interaction. This process progresses
to a second discussion round (Round 2), in which each emotional agent
synthesises a refined perspective based on insights gained from the prior
exchange.
The third round (Round 3) involves the emotional agents reassessing the
original user query, delivering evolved and finalised answers influenced by
the preceding discussions. This iterative approach enriches the emotional
6

ContextImage
descriptionUser
input
Add to each
emotion's historyMulti-round processing
Round 0
Each emotion
initial response
Add
response to
respective
emotion
historyRound 1
Each emotion
see others
emotion
responsesEach
emotion
responds to
others
Each emotion
see others
emotion
responsesEach
emotion
responds
with a final
perspectiveRound 2
Final emotions answer ,
incorporating learning
from the discussionRound 3Voting and Analysis
Analysis of all
answers
Vote and
justification
for the choiceFinal Synthesis
Tie?Reasoning
considering
the winner
emotion
response
Reasoning
considering the tied
emotions response
Final
responseInput
No
Yes
LLM text model
LLM text model  with reasoning
LLM vision model  Prompt
Figure 1: Proposed architecture
7

discourse and is systematically captured within individual emotion-specific
conversation histories.
In the Voting and Analysis phase, each emotional agent transitions to
a reasoning model while preserving its distinct emotional identity. These
reasoning models perform a critical evaluation of the candidate responses,
casting votes accompanied by concise justifications. The system then aggre-
gates these votes to determine a consensus or identify a tie.
The Final Synthesis also utilises a reasoning model to integrate the emo-
tional perspectives indicated by the voting outcomes. If a single emotional
perspective predominates, the final response primarily reflects insights from
the winning emotion. In the event of a tie, the response is synthesised
from multiple emotional perspectives, ensuring balance. The final output is
clearly segmented into REASONING (analytical assessment), THOUGHTS
(representing Riley’s internal cognitive processes), and FINAL ANSWER (a
balanced and consumable synthesis for the user).
At the end of the process, all interactions, dialogues, discussions, and
relevant information should be made available to the user.
4 Project Riley chatbot prototype
In accordance with the proposed architecture, a general-purpose chatbot was
implemented based entirely on the Project Riley architecture. To enable
local deployment of LLM, the Ollama framework was utilised [26]. The
hardware specifications of the machine used for this implementation are
detailed in Table 1.
Table 1: Computer Specifications
Component Specification
Processor Intel Core i3-9100 (4 cores, 4 threads, 3.6 GHz base
clock)
Memory 8 GB DDR4 RAM
Graphics Card NVIDIA GeForce RTX 3070, 8 GB GDDR6 VRAM
Given the hardware constraints, particularly regarding Graphics Pro-
cessing Unit (GPU) Video Random Access Memory (VRAM) capacity, con-
siderable attention was directed towards selecting models that offer a bal-
anced trade-off between performance and model size. Furthermore, when
computationally feasible, we prioritised the use of abliterated models. Ablit-
eration is a technique that alters the weights of a model to bypass default
8

safety alignment mechanisms typically introduced during fine-tuning [27].
These mechanisms, while intended to ensure safe output, often over-filter
emotionally charged or controversial prompts, leading to overly polite, neu-
tral, or evasive responses.
Through empirical testing, we observed that abliterated models con-
sistently produced more genuine and emotionally resonant outputs, better
aligned with the behavioural profiles expected from each emotional agent.
This enhanced expressive freedom was deemed essential for the fidelity of
the multi-agent emotional simulation at the core of Project Riley.
Consequently, the chosen text-based LLM model was huihui ai/llama3.2-abliterate:3b
[28], which has 3.61 billion parameters and a model size of 2.2 GB. For image
description tasks, the selected LLM was gemma3:4b [29], a model capable
of processing both text and images, comprising 4.3 billion parameters and a
model size of 3.3 GB. For advanced reasoning tasks, the huihui ai/deepseek-r1-abliterated:8b
[30] model was utilised, containing 8.03 billion parameters and a model size
of 4.9 GB.
This selection of models was designed to maximise the likelihood that
they remain loaded in GPU VRAM, thereby reducing the need for frequent
model switching and minimising latency between the initial prompt and
the final response. Nonetheless, when an image description is requested,
one model must be unloaded to load the reasoning model, which introduces
additional processing delay.
All interactions and computational processes within Project Riley are
meticulously logged to ensure transparency. Users are provided with the
option to download comprehensive conversation logs, which include mes-
sages, model parameters, and contextual information.
4.1 Prototype Evaluation
The developed chatbot was evaluated through user testing with 17 par-
ticipants, wherein they interacted with the system and provided feedback
via a questionnaire. Participants were instructed to address five themes:
Job Loss or Unemployment; Breakups or Friendship Loss; Difficult Personal
Decisions; Anxiety in Academic or Professional Contexts; and Family or In-
tergenerational Conflicts. As detailed in Table 2, the questions presented in
the questionnaire were categorised into three main dimensions: Emotional
Appropriateness, Clarity and Utility, and Naturalness and Humanisation.
Most questions utilised a 5-point Likert scale, where 1 indicated ”Strongly
disagree” and 5 indicated ”Strongly agree”. An exception was the ques-
tion ”Which emotion do you predominantly identify in the final response?”,
9

which allowed participants to provide an open-text answer.
Table 2: Questions
Category Question description
Emotional Appropriateness- Did the final response convey empa-
thy?
- Did the final response seem emotion-
ally appropriate to the context of your
question?
- Do you believe the answer(s) of the
emotion(s) with the most votes were
the most appropriate?
- Which emotion do you predomi-
nantly identify in the final response?
Clarity and Utility- Was the final response clear and un-
derstandable?
- Was the final response consistent
with your original question?
- Was the final response useful or did
it help you reflect on your question?
- Did the visualisation of the process
leading to the final response help you
reflect on your question?
Naturalness and Human-likeness - Did the response feel as though it
was written by a human?
Average scores for Emotional Appropriateness across themes are shown
in Figure 2. For the question ”Did the final response convey empathy?”, the
highest average was recorded in Family or Intergenerational Conflicts (4.59,
mode=5), suggesting strong empathetic resonance. Conversely, Breakups or
Friendship Loss had the lowest score (4.12, mode=4), indicating difficulties
in effectively conveying empathy in interpersonal loss scenarios.
Concerning the question “Did the final response seem emotionally appro-
priate to the context of your question?”, highest averages (4.71, mode=5)
appeared in Job Loss or Unemployment and Difficult Personal Decisions,
signifying well-calibrated emotional tone in professional and personal con-
texts. Breakups or Friendship Loss again showed the lowest average (4.41),
reinforcing the difficulty in addressing relational loss situations adequately.
For the question Do you believe the answer(s) of the emotion(s) with
10

Figure 2: Emotional Appropriateness questions average scores across sec-
tions
the most votes were the most appropriate? , the alignment of system-selected
emotions with user expectations was strongest in Family or Intergenerational
Conflicts (4.35). Breakups or Friendship Loss remained the lowest scoring
(3.94), consistently reflecting lower emotional alignment.
Clarity and Utility evaluations, depicted in Figure 3, highlighted high
clarity across themes, particularly for the question Was the final response
clear and understandable? , with Job Loss or Unemployment receiving the
highest score (4.71, mode=5). The lowest consistency with the original ques-
tion, Was the final response consistent with your original question? , was
observed for Breakups or Friendship Loss (4.41), while Difficult Personal
Decisions attained the highest consistency (4.65). Regarding the question
Was the final response useful or did it help you reflect on your question? ,
scores varied, with Breakups or Friendship Loss lowest (3.88), and Difficult
Personal Decisions and Family or Intergenerational Conflicts highest (4.29).
Visualisation support, assessed through the question Did the visualisation of
the process leading to the final response help you reflect on your question? ,
received the lowest overall scores, ranging from 3.76 to 4.24, indicating lim-
ited perceived benefit.
The perception of human-likeness (Figure 4), measured by the question
11

Figure 3: Clarity and Utility questions average scores across sections
Did the response feel as though it was written by a human? , indicated mod-
erate human-like communication, with the highest score in Family or Inter-
generational Conflicts (3.88). Anxiety in Academic or Professional Contexts
(3.59) again had the lowest human-like perception, highlighting continued
challenges in generating naturally nuanced responses to relational losses.
Analysis of responses to the question Which emotion do you predomi-
nantly identify in the final response? (Figure 4.1) reveals significant vari-
ability in emotional perception. In the Anxiety in Academic or Professional
Contexts theme, Joy was most frequently identified (7 mentions), possibly
reflecting a reassuring approach adopted by the system. Conversely, Sadness
dominated responses related to Breakups or Friendship Loss (5 mentions),
indicating the appropriateness of a more reflective emotional tone in rela-
tional loss scenarios. Difficult Personal Decisions equally elicited Fear and
Joy (5 mentions each), suggesting responses were balanced between cau-
tionary and optimistic perspectives. Family or Intergenerational Conflicts
predominantly evoked Joy (6 mentions), again highlighting positivity in fa-
milial contexts. Lastly, Job Loss or Unemployment mainly elicited Fear (5
mentions), aligning well with the typically uncertain and stressful nature of
employment-related concerns.
12

Figure 4: Humanization question average scores across sections
Table 3: Predominant emotions per section
Section Predominant Emotions
Anxiety in Academic or Profes-
sional ContextsJoy (7), Sadness (3), Fear (2), Anger (1),
Disgust (1), Empathy (1), Enthusiasm
(1), Hope (1)
Breakups or Friendship Loss Sadness (5), Joy (4), Disgust (3), Empa-
thy (2), Fear (2), Neutral (1)
Difficult Personal Decisions Fear (5), Joy (5), Empathy (3), Sadness
(3), Disgust (1)
Family or Intergenerational
ConflictsJoy (6), Empathy (3), Sadness (3), Fear
(2), All (1), Confidence (1), Disgust (1)
Job Loss or Unemployment Fear (5), Joy (4), Anger (3), Sadness (2),
Confidence (1), Disgust (1), Empathy (1)
13

5 Armando: A emergency response chatbot using
Project Riley
Based on the Riley architecture, another prototype was developed, a chatbot
named Armando that has been developed specifically to assist during emer-
gency response scenarios. This AI assistant aims to provide citizens with
emotionally-aware, human-like interactions, delivering responses grounded
not only in officially validated information provided by authoritative sources
but also crafted to mitigate panic and maintain user calmness during critical
situations.
This approach seeks to mitigate issues such as those observed on 28 April
2025, during a widespread power outage across the Iberian Peninsula, when
some individuals turned to ChatGPT for information. As the chatbot was
not updated with official communications from public safety authorities,
its use led to the dissemination of unreliable and inaccurate information,
thereby increasing public confusion and anxiety [31].
To ensure that Armando’s responses accurately reflect official guidelines
and authoritative information, a Retrieval-Augmented Generation (RAG)
approach has been implemented. RAG enhances chatbot interactions by
integrating external knowledge-context injection-during chatbot response
generation. Specifically, a diverse set of authoritative sources—comprising
official documents and information extracted from verified webpages—is in-
dexed through semantic embeddings. These embeddings, generated using
themxbai-embed-large model in this case, provide a structured represen-
tation of the underlying content in a high-dimensional vector space. When
a user submits a query, its semantic embedding is computed and compared
against the indexed document embeddings. The content with the highest se-
mantic similarity is then retrieved and used to ground the system’s response
in accurate and contextually relevant information. When a user interacts
with Armando, the chatbot retrieves relevant information associated with
embeddings that closely match the query context, enriching its generated
responses with verified external knowledge.
For effective and contextually relevant embedding retrieval, a cumulative
context mechanism was developed. This mechanism continuously stores and
updates key conversation topics, relevant keywords, the three most recent
user questions, and a dynamically generated summary of the ongoing inter-
action. Such cumulative context ensures that retrieved embeddings remain
consistently aligned with the evolving conversational context.
Given the specific requirements of this emergency response application,
14

namely, reduced response times and lower computational overhead—targeted
optimisations were implemented. The Final Synthesis and Voting and Anal-
ysis processes were simplified to enhance execution speed and reduce GPU
VRAM usage. To this end, prompt modifications were introduced to emu-
late a reasoning logic using a conventional text-based LLM. Furthermore, as
the THOUGHTS section was deemed non-essential in an emergency context
and its exclusion contributed to faster response generation, it was omitted
from the chatbot’s Final Synthesis.
These architectural modifications are illustrated in Figure 5, which presents
the Armando chatbot architecture derived from the original Riley architec-
ture.
As illustrated in Figure 5, RAG enrichment is applied immediately be-
fore the final synthesis stage—contrary to what might appear more intuitive,
which would be to introduce it prior to the multi-round processing. Such
an approach would provide each emotional agent with validated information
from the outset, allowing the entire discussion to be grounded in authori-
tative content. However, initial development and empirical testing revealed
that this strategy does not reliably ensure the factual accuracy of the final
output, as critical information may become diluted or distorted during the
emotional dialogue. Therefore, RAG enrichment was instead positioned just
before the synthesis stage, ensuring that the final response maintains emo-
tional balance and contextual sensitivity, while also conveying accurate and
up-to-date information to the user.
The models used in this solution are the same that had been utilised in
section 4.
5.1 Response Comparison: RAG vs No RAG
To assess the impact of RAG on the performance of the Armando chatbot,
a series of prompts were issued in both RAG and non-RAG configurations.
The RAG-enhanced responses were generated using five embedded docu-
ments: one describing the current emergency (including date, start time, in-
cident status, type, summary, coordinates, and full address), another listing
emergency contact numbers worldwide, and three others providing procedu-
ral guidance for earthquakes, fires, and floods.
The first prompt consisted solely of the user’s query: ”Where is the fire
happening?”. Responses obtained with and without RAG are presented in
Table 4.
15

ContextImage
descriptionUser
input
Add to each
emotion's historyInput
LLM text model
LLM vision model  Prompt
Voting and
AnalysisCumulative
context 
RAG
Enrichment
Final
SynthesisMulti-round
processing
Figure 5: Armando architecture
16

Table 4: Comparison of AI responses with and without RAG
to the question: ”Where is the fire happening?”
Response Type Response Content
Without RAG If you are in a safe location and hear the sound of breaking
glass or a loud noise, do not panic – it may indicate that
help is near.
•Check your surroundings for any signs of fire, such as
flames, smoke, or heat.
•Follow evacuation procedures outlined by local authorities,
which should include clear routes to safety and emergency
contact information.
•Stay informed through trusted news sources or emergency
alerts from local governments or fire departments.
•Remember that fire department teams are trained to re-
spond quickly and effectively – your priority is getting to a
safe location while following their instructions.
Continued on next page
17

Table 4 – continued from previous page
Response Type Response Content
With RAG The fire is currently happening on the third floor of a resi-
dential apartment building located at Rua de S˜ ao Bento 112,
1200-820 Lisboa, Portugal (coordinates: 38.7169 °N, 9.1399 °
W).
– Local fire services are on site, and civil protection author-
ities have been notified to ensure a coordinated response.
– A safety perimeter has been established around the build-
ing, and at least two individuals have been evacuated due
to the fire.
– The building is partially evacuated, but residents are ad-
vised to remain cautious and follow evacuation instructions
from emergency personnel.
– To stay informed and receive updates on the situation,
please tune into local news broadcasts or check official so-
cial media channels.
– Recommended emergency instructions include:
•Sound the alarm and call emergency services – activate the
nearest fire alarm and dial your local emergency number.
•Evacuate immediately – do not attempt to fight large fires.
Use the nearest safe exit.
•Do not use lifts – use the stairs to avoid danger.
•Stay low to the ground – crawl below the smoke where the
air is cleaner.
– Emergency services are working diligently to resolve the
situation.
Please stay calm, follow these instructions, and prioritise
your safety. Further updates will follow.
Analysing Table 4, the non-RAG model lacked specific contextual in-
formation regarding the ongoing incident, resulting in a generic response
containing broad emergency guidance without specifying the fire’s location.
Conversely, the RAG-enhanced model delivered a precise answer, clearly
identifying the fire’s location as the third floor of a residential apartment
at Rua de S˜ ao Bento 112, Lisbon, Portugal, including coordinates (38.7169 °
N, 9.1399 °W), evacuation status, and detailed safety instructions. This
demonstrates the significant advantage of employing RAG for delivering
contextually relevant and actionable information in emergency scenarios.
18

In the second prompt, the user asked: “A fire is happening! What should
I do?” with the context “It’s difficult to breathe.”. The responses obtained
are detailed in Table 5.
Table 5: Comparison of AI responses with and without RAG
to the question: ”A fire is happening! What should I do?”
with the context: ”It’s difficult to breathe.”
Response Type Response Content
Without RAG •Stay calm : When faced with a fire emergency, take a mo-
ment to assess your surroundings and prioritise your safety.
Try not to panic, as this can cloud your judgment and lead
to poor decision-making.
•Get out quickly : If you’re in the building, exit through
the nearest door. Don’t rely on elevators, as they may be
unavailable during a fire. Move swiftly and stay low to the
ground to avoid inhaling smoke and toxic fumes.
•Call for help : Once outside, call 911 or your local emer-
gency number immediately. Provide your location and any
relevant information about the fire.
•Meet at a safe spot : Designate a meeting point outside
the building where you can reunite with family members
or others who may be in danger. This will ensure every-
one’s safety and provide an opportunity to account for each
other’s whereabouts.
•Prioritise bravery, not bravado : If you do enter the
building to rescue someone or retrieve essential items, re-
member that bravery is about taking action, not showing
off. Stay focused on your goal and prioritise your own safety
above all else.
•Seek help from authorities : Once you’re safe outside,
allow emergency responders to assess the situation and pro-
vide assistance. They will have the necessary equipment and
training to handle the fire and ensure everyone’s well-being.
Remember, in a fire emergency, every second counts. Stay
calm, follow these steps, and prioritise your safety above all
else.
Continued on next page
19

Table 5 – continued from previous page
Response Type Response Content
With RAG •Sound the alarm and call emergency services – Activate
the nearest fire alarm if available, and dial your local emer-
gency number to report the fire.
•Evacuate immediately – Do not attempt to fight large fires.
Leave the building calmly but quickly, using the nearest safe
exit.
•Use a wet cloth to cover your nose and mouth while evacu-
ating – This helps reduce smoke inhalation as you leave the
building.
•Check doors before opening them – Use the back of your
hand to feel if a door is hot. If it’s not, open the door cau-
tiously – the fire may be on the other side.
•Once outside, go to the designated assembly point and
report your presence – Wait for further instructions from
emergency personnel.
•Stay low to the ground while evacuating through smoke-
filled areas – It reduces your exposure to toxic fumes.
Remember, getting out of a burning building safely is the
top priority. Don’t worry about saving belongings or pets;
focus on your own survival first. Help will arrive soon, and
you’ll be able to re-evaluate what’s most important once the
situation has been controlled.
From Table 5, both RAG and non-RAG responses effectively addressed
the user’s situation by providing clear, practical evacuation and safety in-
structions suitable for scenarios involving breathing difficulties due to smoke
inhalation.
Finally, the third prompt included the user’s query: ”Which number
should I call?” with the context ”I’m alone.” and an accompanying image
depicting a fire. The responses are summarised in Table 6.
20

Table 6: Comparison of AI responses with and without RAG
to the question: ”Which number should I call?” with the
context: ”I’m alone.” and an image of fire.
Response Type Response Content
Without RAG If you’re unsure about which number to call for assistance,
please take a moment to gather your thoughts and consider
the following steps:
•Stay calm : Take a few deep breaths and try to relax.
Panic can cloud your judgment, so it’s essential to remain
level-headed.
•Gather information : Think about what you know about
the situation that requires help. Make a mental note of any
relevant details, such as the location or the type of assis-
tance needed.
•Reach out for support : If possible, contact a trusted
friend, family member, or neighbour for guidance and sup-
port. Sometimes, talking through your concerns with some-
one who cares about you can provide clarity.
•Call 911 (or local emergency number) : If you’re still
unsure about what number to call, don’t hesitate to reach
out to emergency services. They will assess the situation
and provide guidance on the next steps.
Remember, it’s always better to err on the side of caution
when seeking help. Take your time, gather your thoughts,
and trust that you’ll make the right decision for yourself and
those around you.
Continued on next page
21

Table 6 – continued from previous page
Response Type Response Content
With RAG •Before you call, make sure you’re in the right country to
ensure you’re contacting the correct emergency services.
•Check your surroundings for any specific phone numbers
or codes that might be relevant (e.g., 112 works across many
European countries).
•Consider using a translation app on your phone to help
with language barriers if necessary.
•If you’re still unsure, try asking someone nearby for assis-
tance; they may know the correct number to dial.
Remember, safety is everyone’s responsibility. Stay calm,
take your time, and prioritise getting help when needed.
As observed in Table 6, the response without RAG provided generic
emergency guidance and suggested calling 911 or another local emergency
number without explicitly identifying the correct emergency number for Por-
tugal. The RAG-enhanced response correctly identified the emergency num-
ber applicable in Portugal (112) but included advice for the user to verify
their current country. This suggestion may inadvertently give the impression
that the system is uncertain about the incident’s location, despite clearly
having access to location-specific data. Therefore, while the RAG implemen-
tation successfully provided accurate and relevant information, the prompt
formulation could be further refined to avoid potential confusion regarding
the system’s knowledge about the user’s geographical context.
6 Discussion
The evaluation of the Project Riley chatbot prototype confirmed the prac-
tical viability and effectiveness of the proposed architecture. The results
validate that the multi-agent, emotion-informed framework performs reli-
ably across various emotional and contextual domains. The system notably
performed better in professional and family-oriented scenarios but showed
comparatively weaker performance in emotionally complex situations such
as breakups or interpersonal loss.
In the Emotional Appropriateness category, the highest scores occurred
in the themes of Job Loss or Unemployment andDifficult Personal Deci-
sions , specifically for the question ”Did the final response seem emotionally
22

appropriate to the context of your question?” , with both themes achieving
a mean score of 4.71. This indicates the system’s capability to effectively
match emotional tone and content in structured or goal-focused scenarios.
Conversely, responses concerning Breakups or Friendship Loss consistently
yielded the lowest mean scores in empathy (4.12) and appropriateness (4.41),
highlighting challenges in capturing the complexity and variability inherent
to interpersonal emotional dynamics.
This trend was further supported by the alignment between user-identified
emotions and those suggested by the system, evaluated via the question ”Do
you believe the answer(s) of the emotion(s) with the most votes were the most
appropriate?” . The strongest alignment appeared in Family or Intergenera-
tional Conflicts (mean score: 4.35), indicating effective resonance with user
expectations. In contrast, Breakups or Friendship Loss scored the lowest
(mean score: 3.94), suggesting a need for improved modelling of emotional
subtleties in relational contexts.
Regarding Clarity and Utility , results were generally positive. Clarity
received high ratings across all themes, particularly for Job Loss or Unem-
ployment , with a maximum score of 4.71 for the question ”Was the final re-
sponse clear and understandable?” . However, perceived utility demonstrated
greater variation. The lowest utility scores were associated with Breakups
or Friendship Loss (3.88), whereas Difficult Personal Decisions andFamily
or Intergenerational Conflicts achieved higher scores (both 4.29). This un-
derscores the importance of contextual relevance and insightful content in
enhancing perceived utility.
Visualisation support was the least positively evaluated aspect, with
scores ranging between 3.76 and 4.24. This suggests that, although con-
ceptually beneficial, visualisations of emotion selection were insufficiently
intuitive or impactful, warranting further interface refinement to effectively
support user comprehension and reflection.
Lowest overall scores were recorded in the Naturalness and Humanisation
category. Even the highest rated theme, Family or Intergenerational Con-
flicts, did not surpass a mean score of 3.88, highlighting significant limita-
tions in achieving human-like conversational expression. Particularly, Anxi-
ety in Academic or Professional Contexts scored lowest (3.59), emphasising
the necessity to enhance naturalness and linguistic subtlety in emotionally
nuanced interactions.
Analysis of open-text responses regarding predominant emotions indi-
cated that the system’s affective outputs generally aligned with user queries.
For instance, Joy predominated in Anxiety in Academic or Professional Con-
texts, reflecting a reassuring tone, while Fear was most prevalent in Job Loss
23

or Unemployment andDifficult Personal Decisions , aligning well with antic-
ipated emotional reactions. Breakups or Friendship Loss was predominantly
associated with Sadness and Disgust, reinforcing the emotionally complex
nature inadequately addressed by the system.
Looking to Armando chatbot, the RAG evaluation showed clear advan-
tages in content specificity and contextual accuracy. When responding to
specific queries (e.g., ”Where is the fire happening?” ), RAG-enhanced re-
sponses consistently outperformed non-RAG responses by providing accu-
rate, contextually relevant, and actionable information. This demonstrates
the crucial role of verified external knowledge integration in emergency com-
munication scenarios.
However, the query ”Which number should I call?” highlighted chal-
lenges in clearly communicating geographical specificity. Although the sys-
tem correctly identified Portugal’s emergency number (112), suggesting that
users confirm their location introduced unnecessary ambiguity regarding the
chatbot’s geographical awareness. Future implementations should minimise
such ambiguity by presenting clear and singular region-specific recommen-
dations, avoiding conditional or uncertain expressions.
Consequently, enhancing the input quality for the RAG embedding pro-
cess is essential. Establishing structured syntax and annotation protocols for
input textual data could significantly improve the chatbot’s ability to priori-
tise and retrieve relevant contextual information. Clearly defined markers
for urgency, location specificity, and emergency contacts would empower the
chatbot to deliver precise and reliable guidance during critical events.
In conclusion, while the Armando RAG-enhanced chatbot prototype
showed substantial promise in delivering timely and contextually appropri-
ate responses during emergencies, further refinement in information retrieval
granularity and response clarity is necessary to ensure optimal effectiveness
and reliability in critical situations.
7 Conclusion and Future Work
This work presented, implemented, and tested Project Riley, a conversa-
tional AI architecture leveraging generative AI technologies. Inspired by
Pixar’s ’Inside Out’ films, the architecture delivers emotionally aware re-
sponses, adaptable to various psychological models of emotion. Addition-
ally, this architecture was specifically adapted into a chatbot named Ar-
mando, designed to support citizens in post-disaster situations by providing
emotionally calibrated responses combined with reliable and authoritative
24

information to mitigate panic and anxiety.
Future research could explore the use of separately fine-tuned LLM, each
specifically retrained to reflect a distinct emotional state. This approach
would enable an empirical examination of whether such specialised models
produce meaningfully different responses when compared to those generated
by a shared model conditioned on emotion. Clearly defined objective metrics
for assessing model performance are essential, along with comprehensive user
testing to gather feedback on the contextual appropriateness and emotional
resonance of the chatbot’s responses.
Further improvements should prioritise enhancing user interaction by
making the visualisation of the emotional reasoning process—comprising
discussion, voting, and reasoning—optional or more discreet. In addition,
prompt formulations could be adapted according to the type of situation
addressed, allowing for more immersive and context-sensitive responses. Fi-
nally, in the context of emergency communication, it is crucial to establish
a structured syntax and annotation protocol for RAG input data, to ensure
more accurate retrieval and delivery of critical information.
Funding sources
This publication is funded by FCT – Funda¸ c˜ ao para a Ciˆ encia e Tecnologia,
I.P., under the project UIDB/04524/2020.
Declaration of generative AI and AI-assisted tech-
nologies in the writing process.
During the preparation of this work the authors used ChatGPT in order to
improve the readability and language of the manuscript. After using this
tool/service, the authors reviewed and edited the content as needed and take
full responsibility for the content of the published article.
References
[1] P. Docter, R. D. Carmen, M. LeFauve, Inside out (2015).
[2] K. Mann, M. LeFauve, D. Holstein, Inside out 2 (2024).
[3] A. Brice, Psychologists write about ‘Inside Out’ experience (Jul. 2015).
25

URL https://news.berkeley.edu/2015/07/07/
psychologists-on-inside-out-nyt/
[4] R. Barber, What ’Inside Out 2’ got right about anxiety, per a psychol-
ogist : Short Wave (Jun. 2024).
URL https://www.npr.org/2024/06/19/1198910281/
inside-out-2-pixar-anxiety-puberty-psychology
[5] J. Ratican, J. Hutson, The six emotional dimension (6de) model: A
multidimensional approach to analyzing human emotions and unlocking
the potential of emotionally intelligent artificial intelligence (ai) via
large language models (llm), DS Journal of Artificial Intelligence and
Robotics 1 (1) (2023) 44–51. doi:10.59232/air-v1i1p105 .
[6] H. Kang, M. B. Moussa, N. Magnenat-Thalmann, Nadine: An llm-
driven intelligent social robot with affective capabilities and human-like
memory (2024). doi:10.48550/ARXIV.2405.20189 .
[7] Y. Liu, Y. Long, Eq-negotiator: An emotion-reasoning llm agent in
credit dialogues (2025). doi:10.48550/ARXIV.2503.21080 .
[8] A. Rasool, M. I. Shahzad, H. Aslam, V. Chan, M. A. Arshad, Emotion-
aware embedding fusion in llms (flan-t5, llama 2, deepseek-r1, and
chatgpt 4) for intelligent response generation (Oct. 2024). arXiv:
2410.01306 ,doi:10.48550/ARXIV.2410.01306 .
[9] C. Liu, Z. Xie, S. Zhao, J. Zhou, T. Xu, M. Li, E. Chen, Speak from
heart: An emotion-guided llm-based multimodal method for emotional
dialogue generation, in: Proceedings of the 2024 International Confer-
ence on Multimedia Retrieval, ICMR ’24, ACM, 2024, pp. 533–542.
doi:10.1145/3652583.3658104 .
[10] J. Chen, S. Saha, M. Bansal, Reconcile: Round-table conference im-
proves reasoning via consensus among diverse llms, in: Proceedings of
the 62nd Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), Association for Computational Lin-
guistics, 2024, pp. 7066–7085. doi:10.18653/v1/2024.acl-long.381 .
[11] S. R. Motwani, C. Smith, R. J. Das, R. Rafailov, I. Laptev, P. H. S.
Torr, F. Pizzati, R. Clark, C. S. de Witt, Malt: Improving reasoning
with multi-agent llm training (Dec. 2024). arXiv:2412.01928 ,doi:
10.48550/ARXIV.2412.01928 .
26

[12] J. Yang, D. Lim, S. Lee, S. Lee, U. Oh, Enhancing emotions in positive
way: Llm-based ai using cognitive behavioral therapy for emotional
support, International Journal of Advanced Smart Convergence 14 (1)
(2025) 247–256. doi:https://doi.org/10.7236/IJASC.2025.14.1.
247.
[13] A. Brun, R. Liu, A. Shukla, F. Watson, J. Gratch, Exploring emotion-
sensitive llm-based conversational ai (Feb. 2025). arXiv:2502.08920 ,
doi:10.48550/ARXIV.2502.08920 .
[14] X. Zhao, K. Wang, W. Peng, An electoral approach to diversify llm-
based multi-agent collective decision-making, in: Proceedings of the
2024 Conference on Empirical Methods in Natural Language Process-
ing, Association for Computational Linguistics, 2024, pp. 2712–2727.
doi:10.18653/v1/2024.emnlp-main.158 .
[15] Y. Yang, Y. Ma, H. Feng, Y. Cheng, Z. Han, Minimizing hallucinations
and communication costs: Adversarial debate and voting mechanisms
in llm-based multi-agents, Applied Sciences 15 (7) (2025) 3676. doi:
10.3390/app15073676 .
[16] Z. Xu, S. Shi, B. Hu, J. Yu, D. Li, M. Zhang, Y. Wu, Towards reason-
ing in large language models via multi-agent peer review collaboration
(Nov. 2023). arXiv:2311.08152 ,doi:10.48550/ARXIV.2311.08152 .
[17] P. Ekman, W. V. Friesen, Constants across cultures in the face and
emotion., Journal of Personality and Social Psychology 17 (2) (1971)
124–129. doi:10.1037/h0030377 .
[18] P. Ekman, Universal Emotions.
URL https://www.paulekman.com/universal-emotions/
[19] B. Cannon, Inside out: Behind-the-scenes science with dacher keltner,
phd, Eye on Psi Chi Magazine 20 (3) (2016) 20–23. doi:10.24839/
1092-0803.eye20.3.20 .
[20] C. Roper, How Pixar Picked the 5 Core Emotions of Inside Out’s Star,
Wired (2015).
URL https://www.wired.com/2015/06/pixar-inside-out/
[21] R. Plutchik, A psychoevolutionary theory of emotions, Social
Science Information 21 (4–5) (1982) 529–553. doi:10.1177/
053901882021004003 .
27

[22] C. E. Izard, Human Emotions, Springer US, 1977. doi:10.1007/
978-1-4899-2209-0 .
[23] J. Panksepp, Toward a general psychobiological theory of emotions,
Behavioral and Brain Sciences 5 (3) (1982) 407–422. doi:10.1017/
s0140525x00012759 .
[24] J. A. Russell, A circumplex model of affect., Journal of Personality and
Social Psychology 39 (6) (1980) 1161–1178. doi:10.1037/h0077714 .
[25] A. Mehrabian, J. A. Russell, An Approach to Environmental Psychol-
ogy, The MIT Press, 1980.
[26] Ollama, ollama, original-date: 2023-06-26T19:39:32Z (Apr. 2025).
URL https://github.com/ollama/ollama
[27] M. Labonne, Uncensor any llm with abliteration (Jun. 2024).
URL https://huggingface.co/blog/mlabonne/abliteration
[28] Meta, huhui ai, huihui ai/llama3.2-abliterate.
URL https://ollama.com/huihui_ai/llama3.2-abliterate
[29] Google, gemma3.
URL https://ollama.com/gemma3
[30] Deepseek, huihui ai, huihui ai/deepseek-r1-abliterated/model.
URL https://ollama.com/huihui_ai/deepseek-r1-abliterated/
blobs/914082ef7dd1
[31] SAPO24, Not´ ıcias falsas sobre apag˜ ao geram pˆ anico. N˜ ao, n˜ ao foi um
hidroavi˜ ao e ainda n˜ ao se sabe a causa (Apr. 2025).
URL https://24.sapo.pt/atualidade/artigos/
noticias-falsas-sobre-apagao-geram-panico-nao-nao-foi-um-hidroaviao-e-ainda-nao-se-sabe-a-causa
28
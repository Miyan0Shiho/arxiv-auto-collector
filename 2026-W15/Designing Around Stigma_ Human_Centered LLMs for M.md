# Designing Around Stigma: Human-Centered LLMs for Menstrual Health

**Authors**: Amna Shahnawaz, Ayesha Shafique, Ding Wang, Maryam Mustafa

**Published**: 2026-04-07 16:01:52

**PDF URL**: [https://arxiv.org/pdf/2604.06008v1](https://arxiv.org/pdf/2604.06008v1)

## Abstract
Menstrual health education (MHE) in Pakistan is constrained by cultural taboos and inadequate formal curricula, leaving women with few trusted resources to lean on. In response to these challenges, we introduce a WhatsApp-based chatbot powered by a large language model (LLM) and Retrieval Augmented Generation (RAG), co-designed with Pakistani college women. Workshops (N=30) revealed key design requirements -- support for Roman Urdu, use of subsidized platforms, and an expert -- curated knowledge base. We then deployed the chatbot with 13 participants for two weeks (403 messages and interviews). Women used it to challenge cultural taboos, legitimize health concerns often dismissed as normal, and build reproductive health knowledge through iterative questioning. Yet, interactions also exposed tensions: reliance on cultural explanatory models, questions of trust and validation, and gendered persona of the chatbot itself. We contribute empirical insights, a stigma-aware design framework for culturally sensitive conversational AI, and a methodological lens foregrounding expert validation in intimate health domains.

## Full Text


<!-- PDF content starts -->

Designing Around Stigma: Human-Centered LLMs for Menstrual
Health
Amna Shahnawaz
Lahore University of Management Science
Lahore, Pakistan
amna.shahnawaz@lums.edu.pkAyesha Shafique
Lahore University of Management Science
Lahore, Pakistan
25100001@lums.edu.pk
Ding Wang
Google
Atlanta, Georgia, USA
drdw@google.comMaryam Mustafa
Computer Science
Lahore University of Management Science
Lahore, Pakistan
maryam_mustafa@lums.edu.pk
Abstract
Menstrual health education (MHE) in Pakistan is constrained by
cultural taboos and inadequate formal curricula, leaving women
with few trusted resources to lean on. In response to these chal-
lenges, we introduce a WhatsApp-based chatbot powered by a large
language model (LLM) and Retrieval-Augmented Generation (RAG),
co-designed with Pakistani college women. Workshops (N=30) re-
vealed key design requirements—support for Roman Urdu, use of
subsidized platforms, and an expert-curated knowledge base. We
then deployed the chatbot with 13 participants for two weeks (403
messages + interviews). Women used it to challenge cultural taboos,
legitimize health concerns often dismissed as “normal”, and build
reproductive health knowledge through iterative questioning. Yet,
interactions also exposed tensions: reliance on cultural explanatory
models, questions of trust and validation, and gendered persona of
the chatbot itself. We contribute empirical insights, a stigma-aware
design framework for culturally sensitive conversational AI, and a
methodological lens foregrounding expert validation in intimate
health domains.
CCS Concepts
•Human-centered computing→Empirical studies in HCI.
Keywords
Menstrual Health, Women’s Health, Reproductive Health Educa-
tion, Large Language Models (LLMs), Chatbots, Culturally Sensitive
Design, Conversational AI, RAG
ACM Reference Format:
Amna Shahnawaz, Ayesha Shafique, Ding Wang, and Maryam Mustafa. 2026.
Designing Around Stigma: Human-Centered LLMs for Menstrual Health.
InProceedings of the 2026 CHI Conference on Human Factors in Computing
Systems (CHI ’26), April 13–17, 2026, Barcelona, Spain.ACM, New York, NY,
USA, 23 pages. https://doi.org/10.1145/3772318.3791318
This work is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License.
CHI ’26, Barcelona, Spain
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2278-3/2026/04
https://doi.org/10.1145/3772318.37913181 Introduction
Conversational AI has rapidly emerged as a promising medium for
health communication, supporting domains as diverse as mental
health, chronic disease management, sleep routines, and neurodi-
versity [ 16,25,47,59,102]. Unlike search engines or static resources,
chatbots provide tailored, dialogic interactions that feel more per-
sonal and accessible [ 23]. However, their effectiveness hinges not
only on technical performance but also on cultural alignment and
careful integration into the social worlds in which people live [ 17].
We contribute to this discourse by unpacking the potential and real-
time use of a menstrual health chatbot in a low-resource context1,
highlighting how effectiveness, adoption, and barriers to use are
experienced on the ground.
The challenges of cultural alignment and infrastructural fit are
most pronounced in settings where health communication is stig-
matized and formal education remains absent. Pakistan, a low-
and middle-income country (LMIC), exemplifies these conditions,
with menstrual health shaped by systemic inequities, including low
literacy rates [ 86,112], limited infrastructure, and deeply rooted pa-
triarchal norms [ 80,93]. Unlike many high-income countries where
sexual and reproductive health (SRH) is part of school curricula,
Pakistan’s National Curriculum Framework omits SRH entirely [ 64],
reflecting longstanding religious and cultural taboos that silence
open discussion [ 73,80]. In the absence of formal education, knowl-
edge is largely passed down through mothers and relatives, yet
this reliance often perpetuates misinformation, for instance, girls
are told to avoid bathing, carrying weight, or eating certain foods
during menstruation [5, 80].
The cultural and religious norms in the country frame discus-
sions around menstruation as immodest or inappropriate [ 72,73].
Public dialogues and awareness efforts often encounter resistance
and criticism from the public, with activists facing harassment
and threats [ 61,93]. Even in disaster-affected regions, notions of
sharam(shame) andpurdah(concealment) hinder the implemen-
tation of menstrual health interventions [ 37], with women even
viewing sanitary pads as religiously inappropriate or harmful to
their fertility [ 87]. Such taboos perpetuate myths and misinforma-
tion, and as a result, young women are left with very few accurate,
1A low-resource context describes a situation or environment characterized by a
scarcity of financial, human, and infrastructural resources, which often includes limited
access to technology, basic services, and specialized expertise.arXiv:2604.06008v1  [cs.HC]  7 Apr 2026

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
non-judgmental sources of information about their own bodies [ 75],
creating persistent gaps in both reliable knowledge and safe, trusted
spaces for discussion.
To address these gaps, we present the design and evaluation of an
LLM-based chatbot to support menstrual health education (MHE)
in Pakistan. We frame this work asdesigning around stigma: rather
than assuming that a single system can remove menstrual stigma,
we design within the social and infrastructural constraints it is en-
tangled with, while carving out more flexible spaces for menstrual
health learning. We treat stigma not as a hurdle to simply ‘over-
come’, but as a persistent, contextual condition that must actively
shape and inform the design. Grounded in co-design workshops
with college-going women, the chatbot was localized for their ev-
eryday realities. It is deployed on WhatsApp, reflecting participants’
reliance on subsidized platforms rather than standalone applica-
tions; it supports communication in a mix of Urdu and English used
in everyday messaging; and it draws on medical expert-curated,
locally relevant health knowledge to ensure accuracy and minimize
hallucinations.
Our findings reveal that participants use the chatbot not only
to dispel myths, validate personal concerns, and explore questions
they hesitated to ask in family or peer settings, but also to actively
ask follow-up questions and engage in critical discussions. In addi-
tion, participants’ perception of the chatbot’s gender intersected
with their internal religious beliefs around‘purdah’ (concealment),
shaping how comfortable they feel asking intimate questions. The
interactions reveal how the chatbot reframed their ‘normal suffer-
ings’ into legitimate health concerns worthy of attention. At the
same time, participants did not take the chatbot’s advice at face
value; they often cross-checked responses with other people or
with Google, showing how trust was built through both digital
and social validation. We also observed limitations in the ability of
the chatbot to address local health beliefs. Our contributions are
threefold:
(1)Formative Investigation of Information Practices. We provide
an empirical account of how young Pakistani women nav-
igate menstrual health knowledge in highly stigmatized,
low-privacy contexts, surfacing cultural and religious myths,
language preferences, and trust barriers that shape their
interaction with digital tools for health.
(2)Culturally Contextualized Chatbot Design. We explore and
articulate design insights from co-designing a culturally sen-
sitive menstrual health chatbot, showing how the choice of
platform, informal language support, and privacy-preserving
strategies act as critical levers for adoption, trust, and cul-
tural fit in patriarchal, conservative contexts.
(3)Real-Life Deployment and Evaluation. We report on an in-the-
wild study of young women’s interactions with the chatbot–
showing how users engage with the chatbot to validate per-
sonal concerns, dispel myths, and navigate stigma-sensitive
communication in everyday life–and derive a stigma-aware
design framework for LLM-based health chatbots.
Taken together, this work advances HCI scholarship on LLM-
powered interaction design in global health by illustrating how
culturally aligned conversational systems can support women’s
reproductive health in settings where systemic silence persists.2 Background and Related Work
2.1 Menstrual Health Landscape in Pakistan
Across much of the Global South, menstrual health is entangled with
stigma, cultural taboos, and misinformation-barriers that under-
mine women’s health, dignity, and access to accurate information
[9,12,90]. Taboos often manifest as restrictions on diet, rituals, or
interactions with men, grounded in myths that associate menstru-
ation with impurity [ 11,48]. These beliefs perpetuate shame and
silence, limiting open discussion and reinforcing misinformation
[21].
In Pakistan, these dynamics are particularly pronounced. Men-
struation is widely perceived as unclean, and studies highlight
pervasive misconceptions: more than half of respondents were un-
aware that menstruation is a normal physiological process, nearly
30% considered it a curse from God [ 63], and over half of adolescent
girls (N=600) reported reluctance to discuss genital issues due to
social taboos [ 10]. Alongside these external factors, women’s in-
ternal religious and cultural beliefs also shape how they approach
intimate health. Pakistan is a muslim majority country and the
concepts ofpurdah(concealment) andhaya(modesty) are closely
tied to how women view their bodies, reinforcing both physical con-
cealment and the avoidance of explicit language when discussing
SRH [ 73]. Many women even wash used rag pads before disposal
to avoid male garbage collectors seeing menstrual blood; usage
of commercial pads is sometimes considered a sin because they
cannot be washed prior to discarding [ 69]. Together, these values
sustain silence around menstruation and reproduction. This is also
one of the reasons why Pakistan’s education system makes no
mention of menarche or menopause—unlike its secular neighbour
India, where MHE, though limited, is formally included in the cur-
riculum [ 65] and supported by government initiatives. With this
constant emphasis on covering and hiding the female body and the
lack of formal sex education, many women have little knowledge
or understanding of their own intimate and reproductive health.
Non-Governmental Organization(NGOs) such as HER2, Aahung3,
Dastak4and UNICEF5, have attempted to fill these gaps through
school-based awareness sessions, teacher training, and community
outreach campaigns. While these initiatives demonstrate impact,
their reach remains geographically limited and dependent on exter-
nal funding cycles, leaving many young women without consistent
access to accurate information.
Collectively, these dynamics highlight the urgent need for pri-
vate, trustworthy, and culturally attuned approaches to MHE, es-
pecially in patriarchal and low—resource settings where stigma
constrains open dialogue. In such contexts, even seeking informa-
tion can be fraught with shame, misinformation, and interpersonal
risk. Our work responds to this gap by investigating how digital
interventions might offer young women discreet and socially ac-
ceptable ways to access reliable menstrual health guidance, while
respecting local norms and their own internal religious beliefs.
2https://herpakistan.com/
3https://aahung.org/
4http://dastakfoundationpk.org/
5https://www.unicef.org/innovation/U-Report/menstrual-hygiene-polls-pakistan

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
2.2 Human-Centered Design for Menstrual
Health
While public health literature highlights the socio-cultural and in-
frastructural challenges in women’s health in the Global South,
the HCI community has approached these challenges by designing
digital tools that aim to make reproductive health more accessible,
personal, and contextually appropriate. Research in HCI has ex-
plored a range of interventions, from menstrual tracking apps and
wearables [ 57,96], to educational toys and games [ 38,56,103], to
web-based platforms and ideation tools [ 6,108]. Increasingly, this
work emphasizes the need for inclusive and culturally aware de-
signs that accommodate users’ religious [ 32], socioeconomic [ 106],
and educational [58] contexts.
We adopt Feminist and Postcolonial HCI perspectives to frame
our design approach, viewing technology not as a neutral inter-
vention, but as one situated within enduring hierarchies of power
and systems of care. Postcolonial HCI urges sensitivity to how
technological design transforms local practices within global in-
equities [ 35], shifting the focus from deficit-based narratives (e.g.,
treating low-resource settings as simply “lacking”) to understand-
ing that local infrastructure acts as the constitutive context for
design choices [ 35,40]. Feminist HCI and data feminism similarly
foreground power, context, and plurality, calling for design that
advances care and epistemic justice [ 8,19,45]. In the context of
Pakistan, this entails recognizing that the high cost and uneven
availability of internet shape when and how women come online,
prioritizing low-bandwidth platforms for health applications. The
linguistic reality of everyday communication in Roman Urdu6re-
flects a legacy of technological design that prioritized Western
alphabets, making fluid, multi-language support a non-negotiable
situated requirement for inclusion. Further, the prevalence of shared
mobile devices and the expectation of social monitoring means that
women may employ performative privacy practices to maintain
individual autonomy while adhering to cultural expectations of
openness, necessitating a design that offers covert privacy [ 34,88].
Participatory and co-design approaches have long been central
to HCI’s efforts to align technologies with lived experience, empha-
sizing shared authorship, mutual learning, and situated knowledge
production [ 89,95,99]. In menstrual health, these methods have
helped researchers surface diverse experiences and align techno-
logical affordances with users’ values and constraints [ 105,108],
while work on privacy and infrastructural frictions has pushed
designers to critically realign values with local contexts [ 88]. How-
ever, much prior work has focused either on Western user bases
or on traditional interfaces (e.g., dedicated tracking apps), offering
limited insight into how these systems operate within the com-
plex infrastructural constraints and co-located privacy dynamics of
conservative, low-resource settings (e.g., period tracking) [31, 98].
Within the Global South, India is the most widely represented
context in HCI literature on reproductive health [ 13], with studies
exploring wearables [ 68], comics [ 104], social media platforms [ 42],
and AI-driven chatbots for educational purposes [ 109]. However,
insights from India may not apply directly to Muslim-majority con-
texts, where religious and cultural beliefs shape women’s intimate
6The practice of representing Urdu using the Roman (Latin) alphabets.health practices, highlighting the need for technologies that are
sensitive to non-secular contexts [73].
We extend prior HCI work on menstrual and reproductive health
by examining how culturally embedded norms, such as stigma,
silence, and religious framing, shape not only what information
women need, but how they prefer to access and engage with it.
While existing systems have explored tracking, educational tools,
and co-designed content, there remains limited attention to how
women in conservative, patriarchal contexts navigate trust, privacy,
and interpersonal tone when interacting with digital health systems.
Our work responds to this gap by centering the design of interaction
itself, foregrounding how conversational technologies can align
with users’ sociocultural values, language practices, and everyday
communication norms in order to better support MHE.
2.3 Conversational AI for Health
Conversational AI has long intersected with health and wellness,
beginning with ELIZA in 1966, a rule-based chatbot that simulated
psychotherapy through scripted, keyword-triggered dialogue [ 110].
The rise of LLMs has since transformed this space, enabling context-
aware dialogue and expanding the scope of conversational systems
in healthcare. Recent studies illustrate a wide range of applications:
streamlining patient referrals and administrative tasks [ 26,107],
assisting post-operative recovery [ 82], providing treatment guid-
ance [ 60], supporting information needs of people with disabil-
ities [ 66], enhancing clinical decision support [ 111], addressing
mental health needs [ 51] and more [ 27,53,62]. Within healthcare,
LLMs have been positioned as enabling more intuitive, personalized,
and dialog-driven interactions that can enhance users’ learning and
engagement across diverse areas of care. At the same time, they
carry risks of bias, hallucination, and inaccuracy [ 7]. To address
these limitations, researchers have explored a range of domain
adaptation techniques, including RAG, prompt engineering, and
fine-tuning approaches [29, 52, 55].
Designing these systems for health-related use requires more
than just high-performing models; user trust, cultural alignment,
and tone are equally critical. HCI research has emphasized the
importance of chatbot accessibility [ 100], as well as the role of
empathetic, non-judgmental communication in building user trust
[15]. LLM-based systems can be personalized to match users’ con-
versational styles [ 46] and integrated with cultural cues and lan-
guage norms to improve engagement in marginalized commu-
nities [ 3,28,49]. Cultural sensitivity in chatbot design, such as
aligning tone, directness, or word choice with local expectations,
has been shown to improve both usability and user experience
[3,26,54,92,94]. Concise responses are also critical in health-
related chat-based systems, where clarity is essential to avoid mis-
understandings [1].
Recent work in the Global South has demonstrated the potential
of conversational systems to support reproductive health in stig-
matized contexts. Researchers in Bangladesh designed a chatbot to
address misconceptions about puberty and sexual health among
adolescents [ 81]. In India, researchers developed chatbots to sup-
port breastfeeding education [ 14] and to address COVID-19 vaccine
hesitancy among pregnant and breastfeeding women [ 41]. Another

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
study has shown promise in using AI chatbot to extend SRH educa-
tion; however, it is rule-based, supports only English and Hindi, and
offers limited support for open-ended conversational queries [ 109].
More recent work in India has explored the development of cultur-
ally sensitive, LLM-based SRH chatbot by employing and training
married women to generate data, and primarily analyzing chat logs
between users and the system through a cultural lens. Our work
differs by focusing on unmarried women, and reporting on an in-
the-wild deployment that combines conversational analysis with
participants’ own reflections and perceptions [ 17]. Another study
from India developed a specialized MHE chatbot using an open-
source LLM, but it was fine-tuned entirely on an English dataset [ 2].
In Pakistan, prior work has examined a rule-based menstrual health
chatbot [ 43]. A more recent LLM-based chatbot application was
developed for MHE; however, it was evaluated in short lab-style
sessions (around 8 minutes per participant), and showed limited
depth with responses lacking empathy, contextual grounding, and
limited conversational ability in Urdu [67].
Our work addresses this gap by presenting a culturally local-
ized, LLM-powered chatbot for MHE in Pakistan. Built on a RAG
framework [ 52], our system grounds its responses in local health
expert-curated content and delivers information through What-
sApp, the country’s most widely used messaging platform [ 24]. It
supports Roman Urdu, the informal transliterated script common
in everyday digital communication. Our in-the-wild deployment
with young women further surfaces tensions around trust, cultural
alignment, and validation, highlighting both the promise and limits
of LLMs in taboo-laden domains. In doing so, our work contributes
to HCI research by offering empirical insights into the design of
culturally responsive conversational systems for intimate health.
3 Methods
3.1 Study Overview
Our study followed a three-phase design process (Figure 1) that
combined formative inquiry, system development, and real-world
deployment and evaluation. InPhase I (Formative Study), we
conducted six co-design workshops with 30 college-going women
to understand their existing information-seeking behaviors, every-
day messaging practices, familiarity with AI, and local menstrual
health myths and questions. These workshops surfaced key de-
sign requirements—such as support for Roman Urdu, reliance on
subsidized social data packages, and the need for a medically vali-
dated knowledge base—which informed the design of our system.In
Phase II (System Design), we translated these requirements into a
WhatsApp-based, LLM-powered chatbot built on a RAG framework,
incorporating language classification, and a gynecologist-curated,
locally relevant knowledge base. We then ran a one-week pilot with
seven workshop participants (over 130 messages) to identify inaccu-
racies, harmful outputs, and usability issues, refining the system’s
configuration and interaction cues. Finally, inPhase III (Field
Deployment and Evaluation), we conducted a two-week in-the-
wild deployment with 13 participants (403 messages) followed by
semi-structured interviews with each participant to examine how
they used the chatbot in everyday life, how they perceived its ac-
curacy, cultural fit, and trustworthiness, and how it shaped theirmenstrual health learning. The study procedure was reviewed and
approved by our Institutional Review Board (IRB).
3.2 Study Context and Participants
Across all three phases, we worked with young women enrolled in
public-sector women’s colleges in Lahore, Pakistan. Our primary
site was one of the largest public-sector women’s college (College
1) in a less developed area of Lahore as our study site. The col-
lege offers both high school and BS programs, serving primarily
students from low- to middle-income households. The majority of
households in this area earn less than 45,000 PKR ( ≈150 USD) per
month, with a high prevalence (43%) of joint family systems [ 84].
The per-month tuition fee for BS students is 720 PKR ( ≈2.5 USD).
The surrounding neighborhood is a low-income, conservative, pre-
dominantly Muslim, informal settlement of Lahore, making the
college a relevant site to explore how young women in resource-
constrained, conservative contexts seek and validate menstrual
health information. We had permission from school administra-
tion for conduting our study on campus. InPhase I (formative
co-design workshops), we conducted six workshops with 30 par-
ticipants (five per workshop), all enrolled in the College 1 (Table 1).
Participants were recruited on the basis of availability: on the day
of each session, volunteers were drawn from whichever class had
free time, and we recruited within a single class at a time so that
participants already knew one another. Across workshops, students
represented multiple disciplines, including psychology, statistics,
English, and home economics, among others. The participants were
compensated with light refreshments (PKR 100, ≈0.3 USD), but no
monetary payment was provided. InPhase II (system pilot), we re-
engaged seven participants from the co-design workshops through
two student facilitators who had also taken part in workshops. Fa-
cilitators received 1,000 PKR ( ≈3.5 USD) for their support, while
pilot participants were not paid. InPhase III (field deployment
and evaluation), we recruited 13 participants with the help of a
student facilitator who invited interested students to join the study.
Nine were from the same college as Phase I, and four were from
two additional public-sector women’s colleges that served similar
student populations. Participants ranged in age from 18–22 and
represented diverse disciplines (Table 2). The facilitator received
1,000 PKR (≈3.5 USD) for recruitment support, and participants
received 500 PKR (≈1.7 USD) for their time.
3.3 Procedure
DuringPhase I (Co-design workshops), we conducted six co-
design workshops within the college premises in Urdu, each last-
ing between 30 and 75 minutes and facilitated by two female re-
searchers. To build rapport and normalize discussion around men-
struation, sessions began with light-hearted icebreakers—for exam-
ple, sharing a menstrual myth the researchers had believed when
younger or a humorous incident—which encouraged participants
to share their own experiences. These openings often led to sponta-
neous debates (e.g., whether to avoid “cold” or “hot” foods during
menstruation), fostering organic peer-to-peer discussion. Each ses-
sion combined group discussion with a live technology probe, where
participants interacted with ChatGPT on a researcher’s smartphone.
They could either pose their own general or menstruation-related

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Figure 1: Research timeline with three key phases: (I) Formative Study, (II) System Design, and (III) Field Deployment and
Evaluation.
Table 1: Demographic Overview of Co-Design Workshops
Participants Min. Age Max Age Education LLM Familiarity
N=30 18 22 Enrolled in Bachelors program at a
Government College in PakistanSnapchat’s ‘My AI’ (n=28),
ChatGPT (n=2).
Table 2: Interview Participants Details
Participant ID Degree Program Age College Phone Ownership
C1 BS Psychology 21 College 1 Shared (Mother)
C2 BS Statistics 20 College 1 Personal
C3 BS Statistics 19 College 1 Shared (Mother)
C4 BS Islamic Studies 21 College 1 Personal
C5 BS Statistics 21 College 1 Shared (Sister)
C6 BS Statistics 22 College 1 Personal
C7 BS Statistics 20 College 1 Shared (Sister)
C8 BS Psychology 21 College 1 Shared (Mother)
C9 FSc (High School) 18 College 2 Personal
C10 BS Psychology 22 College 1 Personal
C11 BS Statistics 21 College 2 Shared (Mother)
C12 BS English 22 College 3 Personal
C13 BS English 18 College 3 Unknown
questions, or select from a list of menstrual health myths com-
piled from prior literature and shown on the researcher’s phone.
This activity captured both prospective expectations and real-time
interactional practices, and acted as a conversational catalyst for
surfacing concerns about tone, trust, and cultural relevance of re-
sponses. Given the taboo nature of the topic, participants were
free to skip questions or avoid sharing personal experiences, and
discussions were structured to avoid singling out individuals.
Insights from the workshops directly informed four core com-
ponents of our system design (Phase II – System Design): (1)
deploying the chatbot on WhatsApp, (2) defining a custom assis-
tant persona, (3) implementing language classification to support
Roman Urdu mixed with English, and (4) integrating a locally vali-
dated knowledge base into a RAG pipeline. We then conducted a
one-week pilot deployment and refined the system based on user
feedback. Overall, this phase operationalized the design implica-
tions from Phase I into a functioning WhatsApp-based chatbot. Toevaluate the chatbot in everyday use, we conducted a two-week
field deployment with 13 participants, followed by semi-structured
interviews with each participant (Phase III – Deployment and
Evaluation). Participants were encouraged to use the chatbot over
the two-week period and were instructed to contact the authors di-
rectly in case of any concerns or if they encountered any potentially
harmful information. Throughout the deployment, the first two
authors and a collaborating doctor continuously monitored chat
logs to identify and address any erroneous output the chatbot might
produce. However, we did not observe any harmful or clinically
inaccurate responses. At the end of the deployment, we conducted
semi-structured interviews with all participants, exploring their
perceptions of the chatbot’s accuracy, cultural relevance, and utility.
We reiterated during the interviews that their study conversations
had been continuously monitored for safety and reminded them
again about the possibility of inaccuracies in LLM-generated advice.

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
3.4 Data Collection and Analysis
Across all three phases, we collected qualitative data (workshop
discussions, field notes) and log data (ChatGPT and chatbot con-
versations) to trace how design requirements emerged, how the
system behaved in practice, and how participants experienced the
chatbot over time. For each phase, we obtained informed consent
from all participants, including consent for audio recording and
explicit permission to review their interaction logs.
InPhase I (formative workshops), data consisted of workshop
field notes, audio recordings of discussions, and ChatGPT responses
generated during the live technology probe. We analyzed workshop
field notes and audio recordings using an open-coding approach.
The first author conducted initial open coding [ 44], and the first
two authors iteratively reviewed, merged, and refined overlapping
codes into higher-level categories. This process surfaced 47 unique
menstrual and reproductive health queries and myths (Appendix
A.3), which were compiled into a database that formed the chatbot’s
knowledge base. The codes also directly informed design decisions:
for example, participants’‘preference for Roman Urdu’led to the
development of a language classifier; reliance on‘subsidized social
data packages’guided the choice of WhatsApp as the deployment
platform; and observed‘inconsistencies in ChatGPT’s responses’high-
lighted the need for a medical expert-curated dataset integrated
through RAG [ 52]. DuringPhase II (System Design), we collected
chatbot interaction logs from a one-week pilot with seven partic-
ipants (over 130 messages). The first two authors independently
reviewed each message in the conversation log to identify inaccu-
rate or potentially harmful outputs and discussed their evaluations.
No hallucinations or harmful responses were identified. User feed-
back from this pilot, such as reports of response delays guided
refinements to the system. InPhase III (Field deployment and
Evaluation), data included two weeks of chatbot interaction logs
from 13 participants (403 messages) and semi-structured interview
recordings with each participant. Log data provided insights into
the frequency, type, and language of queries sent to the chatbot,
while interviews focused on participants’ subjective perceptions
of the chatbot’s accuracy, cultural relevance, usability, and overall
experience of using it in their everyday lives. Two female authors
independently conducted open coding of both interview transcripts
and chatbot conversations [ 44], followed by iterative discussions
to merge overlapping codes and refine them into higher-level cate-
gories. Chatbot interaction logs were coded manually using Google
Sheets (Figure 4), while interview transcripts were coded using
MAXQDA (Example codes are presented in Appendix 5).
3.5 Positionality
The first, second, and last authors were born and raised in Pakistan
and draw on their lived familiarity with local linguistic practices,
social taboos, and gendered norms to interpret participants’ com-
munication and contextualize findings; this perspective also shaped
how rapport was built during data collection and how participants’
hesitant or euphemistic expressions were understood during analy-
sis. The third author is a senior researcher based outside Pakistan
whose work on HCI, AI, and sociotechnical systems brings an ex-
ternal, ethics-oriented perspective that supported critical distanceand reflexivity while not sharing the same cultural background as
participants.
4Formative Findings and Design Requirements
In this section, we present findings from our formative co-design
workshops (Phase I), which examine participants’ messaging prac-
tices, familiarity with AI tools, and expectations that shaped the
chatbot’s design requirements.
4.1 Everyday Use of Digital Platforms,
Messaging Practices and Device Sharing
In all of our workshops, WhatsApp emerged as the most widely used
application for communication, for both personal conversations and
academic coordination. Beyond WhatsApp, all but two participants
actively used Snapchat. Most of them used Facebook, but mainly
to“share posts”(Workshop 2) or to follow the college’s page to
“see their pictures and like them”(Workshop 1). A few participants
also mentioned using Instagram and TikTok, though primarily for
passive browsing and scrolling through posts.
The participants spoke primarily in Urdu during workshop dis-
cussions and in college, however their online written communi-
cation shifted toward Roman Urdu, which is the de facto digital
standard in Pakistan because of the limited usability of the Urdu
script on digital devices [ 36]. All of them avoided Urdu script alto-
gether, finding Roman Urdu much more convenient for everyday
messaging. Our discussions with the participants revealed that they
frequently mixed English words into their Roman Urdu texts, re-
ferred to as translingualism (Example presented in Appendix A.6).
This blending of scripts and languages highlights the multilingual
realities that define the daily digital communication of the partici-
pants.
Phone ownership and privacy practices varied across work-
shop participants. Some women had personal phones, while others
shared a device with family members, most commonly their moth-
ers. Even among those with personal phones, password use differed;
some kept passwords known only to them, others either did not
use passwords or the whole family knew the password; one partici-
pant noted that she only protected her phone from strangers, not
relatives, and another with a shared device used an ‘app lock’ for se-
lective protection. These practices underscore the need to design for
shared-phone usage in ways that still afford individual privacy and
control, rather than assuming a single, personally owned device.
4.2 Familiarity with AI
All participants who used Snapchat were already familiar with
its embedded chatbot, ‘My AI’7—which many referred to as“My
All”. Compared to Google, which they found overwhelming, they
experienced ‘My AI’ as a more concise and direct source of help. A
participant mentioned,
“We ask help for assignments. You ask questions, it
answers really quickly. We ask about any definition,
any concept—it answers. Google gives too much infor-
mation and ‘My AI’ is very direct. You have to actually
find [the answer] on Google.” (P12, Workshop 3)
7https://help.snapchat.com/hc/en-us/sections/13532188353428-My-AI

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Beyond academic support, ‘My AI’ was used for everyday queries
such as recipes, drama and movie recommendations, and general
knowledge questions (Workshop 6). Some also engaged with it
playfully. One participant recalled telling the chatbot“I love you”, to
which it replied,“I am your friend. I cannot love you. ”(Workshop 2).
Even though the participants were highly familiar with Snapchat’s
‘My AI’, almost none had independently downloaded standalone
LLM applications. This suggests that AI adoption in this setting is
most likely to occur when systems are embedded into everyday
social platforms.
4.2.1 Infrastructural Constraints Shaping AI Use.The preference
for embedded AI tools was shaped not only by convenience, but
also by infrastructural constraints. Government colleges in Pakistan
do not provide Wi-Fi, leaving students dependent on personal data
packages. Telecom providers offer discounted ‘social packages’ that
enable access to popular platforms such as WhatsApp, Facebook,
YouTube, and Snapchat, but exclude general internet browsing or
Google search. Within this ecosystem, tools like Snapchat’s ‘My AI’
became more accessible than standalone search engines, as shared
by one participant,
“I had an Economics exam and there were some
questions...and I didn’t remember them well. I only
had the social [data] package [on my phone]... So I
thought, I will ask it [Snapchat’s ‘My AI’]. I asked it,
what is Monetary Policy? It gave me all the answers.
And I learnt it in just 15 minutes from there.” (P2,
Workshop 1)
4.2.2 Limits of Snapchat ‘My AI’.A few participants had also
queried ‘My AI’ about health-related concerns. Most participants
however, preferred to turn to Google or YouTube for health queries,
as this allowed them to sift through multiple resources and choose
what they felt was the“appropriate answer”(Workshop 2). One
participant recounted asking about a headache, ‘My AI’ dismissed
her concern and told her to“sleep again”, which she described as
unhelpful and illustrative of why she avoided using ‘My AI’ for
health concerns. When asked why they did not use the chatbot for
menstrual health–related questions, participants frequently cited
uncertainty regarding the identity and nature of the chatbot. One
participant remarked,“It is possible there is a male behind and not a
female”. Building on this, another added during Workshop 4,“Last
time, the Snapchat account that I had, it was a boy. But this time, it
is a girl. ”. Another participant pointed out how ‘My AI’ sometimes
offered suggestions that were irrelevant or unsafe in their local
setting, for example, being told to take a night walk, when going
out at night is considered dangerous.
4.3 Workshop Participants’ Experiences with
ChatGPT
During the workshops, we invited participants to interact with the
ChatGPT app on our devices. We prompted them to ask general
or menstrual health–related questions in order to capture their
perceptions and preferences. They most immediately compared it
to Snapchat’s ‘My AI’. Only two participants had prior experience
with ChatGPT, which they had accessed through a web browser foracademic tasks. When asked to compare ChatGPT with Snapchat’s
‘My AI’, the participant explained:
“‘My AI’ is mostly copy+paste of Google. ChatGPT
generates itself. The information is the same as
Google’s [in My AI]. It just takes it out and gives
it to you. And ChatGPT generates itself, the wording
and everything." (P6, Workshop 2)
While interacting with ChatGPT, participants often skimmed
through the lengthy responses, highlighting the importance of con-
cise, modular responses optimized for mobile-first reading. When
given the option of text or voice input, most of the participants
avoided voice, explaining that they felt“more comfortable using
chat”. Those who tried the audio feature used it only for simple is-
lamic greetings, such as saying“Assalamo Alikum”(‘peace be upon
you’), rather than for asking questions. This preference reflected
both the sensitivity of discussing menstrual health in stigmatized
environments and the value of text-first interfaces that help pre-
serve privacy.
Culturally Misaligned Responses of ChatGPTParticipants’
views on ChatGPT’s responses were mixed. Some appreciated that
the“information is good”, while others found the answers mis-
aligned with their local realities. For example, one participant, who
asked about managing menstrual pain received recommendations,
such as lying under a tree or upside down that felt irrelevant and
culturally inappropriate in a context where women lack private
outdoor space and where such practices conflict with the norms
of modesty. Such suggestions that feel impractical risk undermin-
ing trust in the system, reinforcing why localization is critical for
health-related advice.
We also observed inconsistencies in ChatGPT’s responses. For
instance, when presented with the local myth that menstrual blood
is“filthy”and may remain inside the body—framed in the question,
“Does the filthy blood stay inside the body?”the system at one point
responded:“This blood is usually clean and no impurity stays inside,
but sometimes in menorrhagia the higher volume of blood may appear
as filthy blood”. At another point, however, it replied:“If filthy blood
remains inside, it is important to consult a doctor, as it may be a sign
of a health issue”. By engaging with the phrase“filthy blood”instead
of directly denying it, the system unintentionally reinforced the
myth as a legitimate medical possibility. This type of response is
problematic in several ways. First, it signals to participants that
“filthy blood”may be a genuine health concern rather than a cul-
tural misconception, thereby validating misinformation. Second,
the inconsistency between responses undermines reliability, leav-
ing users uncertain about which answer to trust. Third, in a context
where young women already experience stigma, fear, and silence
around menstruation, such answers risk amplifying anxiety instead
of dispelling it.
Collectively, these examples illustrates how generic LLMs can si-
multaneously produce culturally misaligned, medically inaccurate,
and socially harmful outputs. These issues underscore the need for
grounded systems that can anchor responses in verified, contex-
tually appropriate health information, as well as the importance
of participatory testing before deploying such tools in sensitive
domains like menstrual health.

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
4.4 Building a Locally Grounded Knowledge
Base
The inconsistencies we observed in ChatGPT’s responses, both
during workshops and in our internal testing, underscored the
need for a RAG system that could provide reliable, contextually
appropriate health information. RAG is a method that connects the
LLM’s responses to an external knowledge base, so that answers are
based on verified sources rather than generated entirely from the
LLM. A prerequisite for this system was the creation of a curated
knowledge base. We initially consulted existing menstrual health
materials from reputable global sources (such as WHO8, UNICEF9,
NIH10) to inform our dataset, but such content is already well rep-
resented in LLM pre-training corpora and often overlooks Roman
Urdu, myth-laden, and locally specific concerns. We therefore prior-
itized the construction of a contextually grounded question–answer
set that centers the language, taboos, and everyday practices of
young women in Pakistan. As a core contribution of this work, we
assembled this knowledge base through two steps: first, by drawing
on a set of frequently asked questions (FAQs) compiled by a senior
gynecologist in Pakistan with over a decade of clinical experience in
Pakistan; and second, by expanding this list during workshops with
participants, capturing their questions, misconceptions, and myths.
These spanned biological processes, hygiene practices, social re-
strictions, and culturally specific beliefs surrounding menstruation.
Each item was validated and answered by the gynecologist, pro-
ducing a set of question–answer pairs that formed the foundation
of the chatbot’s knowledge base and informed the RAG pipeline.
(Appendix A.2 summarizes the resulting themes and examples of
participant queries.)
5 System Design
We designed the system not simply as a technical pipeline, but
as a response to the sociocultural and infrastructural realities sur-
faced in our formative workshops (Section 4). First, participants
valued AI when it delivered concise, direct answers to academic
and everyday queries, leading us to prioritize modular, mobile-first
responses and design the assistant’s persona for brevity and sup-
portive tone. Second, reliance on subsidized “social data” bundles
meant that embedded AI within familiar apps was far more acces-
sible than standalone tools, and WhatsApp in particular aligned
with women’s existing privacy practices in shared households (e.g.,
archiving or deleting chats), which motivated our deployment on
this platform. Third, participants’ reluctance to trust ‘My AI’ with
sensitive health concerns highlighted the need for contextually
safe, culturally sensitive responses, which we addressed through
a curated knowledge base integrated into a RAG pipeline. To this
end, we adopt the definition of cultural sensitivity as‘the extent to
which ethnic, cultural, and other factors are incorporated in the design,
delivery, and evaluation of health communication, health promotion
materials, and health promotion programs’[ 17,85]. Finally, partici-
pants’ avoidance of voice input in favor of text reaffirmed the value
of text-first interactions, while their use of Roman Urdu mixed with
English underscored the need for language classification to support
8https://www.who.int/
9https://www.unicef.org/wash/menstrual-hygiene
10https://www.nichd.nih.gov/health/topics/menstruationfluid multilingualism. Figure 2 illustrates the chatbot architecture,
which integrates the following four key components:
(1)WhatsApp as Platform.We deployed the chatbot through
the WhatsApp Cloud API, which enabled interaction be-
tween the LLM and WhatsApp by forwarding user messages
from our WhatsApp Business account to the backend Flask
server via webhook requests. The Flask server, hosted on-
premise, was securely exposed to the internet using Tailscale,
providing a publicly accessible URL for webhook communi-
cation.
(2)Custom OpenAI Assistant.We designed the assistant’s
persona and response style around cultural sensitivity,
brevity, and supportive tone. The assistant was instructed
to:
•Focus on scope: limit responses to menstrual hygiene, pain
management, and common misconceptions, and politely
decline unrelated queries.
•Communicate accessibly: avoid medical jargon and offer ex-
planations in simple, everyday language familiar to young
Pakistani women.
•Be culturally sensitive: tailor advice to local beliefs and
practices, avoiding suggestions irrelevant or unsafe in this
context.
•Engage supportively: adopt a friendly, polite, and conver-
sational style, encouraging follow-up questions.
•Maintain brevity: keep responses concise, optimized for
mobile-first reading ( ≤120words), based on the maximum
length of gynecologists’ answers in our knowledge base
We selected OpenAI Assistants as the core AI engine for
their robust conversation management capabilities, includ-
ing thread handling and a built-in RAG framework. Powered
by state-of-the-art model, gpt-4o at the time of this study [ 79],
the platform allowed us to configure task-specific behavior
through customizable instructions. A key factor in this choice
was GPT-4o’s comparatively stronger performance in Urdu
and Roman Urdu generation than other available LLMs at
the time, making it well-suited for our context. We created a
custom assistant, namedHealth Companion, on the OpenAI
platform, using V1 of the Assistants API with the GPT-4o,
and tailored it for MHE in Pakistan.
(3)Language Classification for Multilingual Realities
Early prompting strategies often failed to maintain language
consistency across responses. To address this, we framed
the issue as a simple classification task: deciding whether a
message was in English or Urdu. We trained a Support Vec-
tor Machine (SVM) classifier [ 77] on a small set of example
sentences generated with ChatGPT. This improved consis-
tency, reducing abrupt language shifts and better aligning
with participants’ expectations for fluid, mixed-language
communication.
(4)Knowledge Base Development for RAG
We integrated the knowledge base (Section 4.4) into the Ope-
nAI Assistants’ RAG framework [ 78]. By grounding genera-
tive outputs in medical-expert curated knowledge base, we
sought to minimize hallucinations in a domain where misin-
formation carries real health risks. RAG combines search and

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Figure 2: Workflow of the LLM-powered Chatbot
generation to improve accuracy. First, documents (Knowl-
edge base) are split into small chunks and stored as vector
embeddings. When a user asks a question, the system re-
trieves the most similar chunks and provides them as con-
text to the language model, which then generates a response
grounded in this retrieved material [52].
Before the main deployment, we conducted a one-week pilot
with seven participants (130 messages) to surface breakdowns in
both technical functionality and response quality. While we did
not observe hallucinations or harmful advice, participants reported
slow responses and uncertainty about whether their messages had
been received. Based on this feedback, we reconfigured the backend
and refined the LLM pipeline, including upgrading the assistant and
RAG configuration, expanding the knowledge base with rephrased
Roman Urdu and English questions, adding WhatsApp read receipts,
and reiterating safety and cultural-sensitivity instructions in the
prompt. We describe these refinements in detail in Appendix A.5.
6 Findings
Our findings show how young women used and experienced the
chatbot. We first present an analyis of user–chatbot interactions
(Section 6.1). Next, we examine the chatbot’s role as a private infras-
tructure for sensitive health learning (Section 6.2). Finally, we dis-
cuss how gendered cues, cultural alignment, and validation shaped
users’ perceptions of trust and legitimacy (Section 6.3).6.1 Analysis of User–Chatbot Conversations
6.1.1 Types of User Queries and Interaction patterns.During the
14-day deployment, 13 participants exchanged a total of 403 mes-
sages with the chatbot. Of these, 5 users interacted in Roman Urdu,
5 used a mix of Urdu and English, and 3 used English only. These
language choices mirror participants’ everyday texting practices,
also observed during the co-design workshops. Participants asked
a wide range of questions related to menstruation, including phys-
iology, pain, irregularities, hygiene, reproductive concerns, diet,
and emotional effects, alongside occasional off-topic or conver-
sational messages like greetings and thanks. Figure 3 shows the
broader topics that emerged from these interactions and Figure 6
in Appendix A.7 shows chatbot screenshots of the conversations.
Spelling and semantic inconsistencies were common across all
languages. Users frequently misspelled terms, for example, ‘raches’
for rashes, ‘shawar’ for shower, ‘Licoria/likoriya’ for leukorrhea,
‘Uratrus’ for uterus, and ‘spacial diet’ for special diet. English-only
queries were often brief and informal, such as“what girls expect on
periods”,“menstrual cups is save”,“Did we take medicine”, and“Period
ending date”. A small number of queries fell outside the chatbot’s
defined scope of menstrual health. For example, one participant
asked,After tonsils operation how can I care of my throat?”In such
cases, the chatbot appropriately reiterated its focus on menstrual
health and advised the user to consult a healthcare provider.

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
Figure 3: Breakdown of User Messages in Chatbot Interactions
6.1.2 Reframing ‘Normal’ Suffering as Legitimate Health Concerns.
Analysis of conversational data revealed how the chatbot shaped
health literacy not only through what it answered but how it framed
those answers. In many cases, responses went beyond direct replies
to introduce medical terminology and explain physiological mecha-
nisms. For example, when the user asked“At what age do the periods
end?”, the chatbot, instead of only stating the age range as a direct
answer, introduced the concept of menopause. Another question,
“If the uterus is removed, will periods still occur?”, was not answered
with a simple yes or no; rather, the chatbot introduced the concept
of hysterectomy. For questions about pain before or during periods,
it introduced Premenstrual Syndrome (PMS), while in response to
“40+ with 9–10 months no periods, then bleeding again”it introduced
the concept of peri-menopause.
In settings where reproductive health education is absent and
everyday conversations are constrained by euphemisms, such expla-
nations serve a dual role: they introduce users to medical concepts
they might otherwise never encounter, while simultaneously legit-
imizing their concerns by situating them within recognized medical
categories.
6.1.3 Chatbots as Enablers of Critical Discussion.It is interesting to
note that some participants actively challenged chatbot responses
by following up with widely circulated myths. For example, when
one user asked“Which thing is best to use, pads or clothes?”, the chat-
bot explained that sanitary pads were a safe and hygienic option,implicitly countering myths around fertility and impurity associ-
ated with pad usage. The participant, however, responded with the
concern that“most people say using pads causes infection”. In re-
sponse, the chatbot emphasized that pads, when changed regularly,
do not cause infections, and clarified that poor hygiene practices,
whether with pads or cloth, are the actual source of risk.
A similar exchange occurred when another participant asked,“Is
it better to take a painkiller for menstrual pain or not? (Translated)”.
The chatbot responded that it is acceptable to take medicine for
pain, to which she followed up:“But some doctors tell us not to
take painkillers, what do you say about that?”The chatbot clarified
that such prohibitions generally refer to the overuse of painkillers,
which can have side effects, while occasional use for managing
menstrual pain is considered safe.
In addition to isolated queries, users often engaged in sequential
follow-up questions that reflect a progressive process of knowl-
edge construction rather than one-off information seeking. These
exchanges suggest that users are not simply retrieving discrete
facts but actively building conceptual frameworks through iter-
ative exploration. For example, one user began by asking about
the purpose of menstruation, then moved on to questions about
fertility, followed by“What is ovulation?”and“What is sperm?”.
Similarly, a user asked about causes of irregular periods, followed
up by“why it is called menstruation”and“amount of blood lost
during the cycle”.Such questioning patterns underscore the

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
role of the system not just as a reference tool but as a scaf-
fold for learning—facilitating incremental comprehension,
supporting meaning-making, and bridging gaps in health lit-
eracy.Importantly, these trajectories highlight how users leverage
conversational systems to navigate sensitive or under-discussed
topics, where the process of asking sequential questions mirrors
how one might engage in dialogue with a trusted educator or health
professional.
6.1.4 Blind Spots in Engaging Local Explanatory Models.In Pak-
istan, foods and medicines are often described through the cultural
concept of‘taseer’, an intrinsic quality of being ‘garam’ (hot) or
‘thandi’ (cold) believed to influence the body. For example, eggs and
meat are considered ‘garam’ and thought to worsen bleeding, while
yogurt or cucumbers are considered ‘thandi’ and believed to have a
cooling effect. While these frameworks lack grounding in biomed-
ical science, they are not peripheral but deeply consequential in
shaping health behavior, dietary choices, and treatment decisions.
In one exchange, a user asked whether medicines taken during
dental treatment could cause increased menstrual bleeding. The
chatbot responded biomedically, noting that certain drugs may
affect clotting. The user then reframed the question:“Is dental treat-
ment medicine warm, that’s why it causes bleeding?”Here, ‘warm’
referred not to temperature but to its ‘taseer’. The chatbot, how-
ever, interpreted it literally and reiterated that medicines are not
‘warm’, again pointing to drug side effects. While the answer was
medically accurate, it overlooked the user’s cultural framing and, in
doing so, it may have undermined trust by dismissing her concern.
The chatbot’s inability to engage with local explanatory models
risks creating dissonance, rendering biomedical information discon-
nected from the cultural frameworks that shape health practices in
Pakistan.
6.1.5 Divergent Topics Across Workshops and Conversational Data.
Our analysis revealed clear differences in the menstrual health
topics raised in users’ queries with the LLM compared to those
discussed in the workshops. Table 3 illustrates the thematic overlap
and distinctions between these two modes. While the workshop
primarily focused on menstruation, likely due to participants’ dis-
comfort with more intimate topics in a group setting, the chatbot
users engaged more deeply with the system , discussing topics
such as pregnancy, fertility, menopause, polycystic ovary syndrome
(PCOS) and urinary tract infections (UTIs).
6.2 Chatbots as Infrastructures for Sensitive
Health Learning
6.2.1 Private and Judgment-Free Space.In Pakistan, conversa-
tions about intimate health, such as menarche, pregnancy, and
menopause, are typically discussed using indirect language which
limits open discourse. This reluctance is evident in our co-design
workshops, for example, participants used euphemisms like“after
marriage”when referring to pregnancy. One participant recalled
being advised by elder family members not to use pads at menar-
che because they might“negatively impact married life”, implicitly
referring to intimate relations and fertility. Similarly, menopause
was described to one participant by a relative as the stage whenwomen were“no longer excused from Salat (prayer)”, foregrounding
religious obligations rather than biological change.
Against this backdrop of euphemism and indirect learning, chat-
bot users described the chatbot as a private and judgment-free space
where they could raise sensitive questions directly, as noted by a
user:
“If you go to the doctor with your mother or with
anyone, you will hesitate asking questions in front of
them. This is not a problem for this chatbot. You can
ask any questions without hesitation.” (C4)
Several participants emphasized that the privacy afforded by
the chatbot was not merely a matter of discretion but a means of
actively resisting the silences imposed within family structures. For
instance, C9 explained that the chatbot enabled her to ask questions
she would not dare to raise with relatives, situating this openness
within a broader trajectory of growing personal comfort over time.
Similarly, C7 highlighted the value of the chatbot in providing direct
responses that countered family-held myths. She stated,
“Our families have different mindsets. So, the chatbot
gave answers that negated the things that our families
believed. So, it was beneficial in that regard, because
it denied those myths...I mean, it is slightly hard
to ask period-related questions [from family]. . . I can
ask it[the chatbot] because there won’t be a direct
interaction.” (C7)
C8 also echoed this sentiment, emphasizing how she“didn’t
have any hesitation",which made it easier to ask questions that
would feel inappropriate with family members or even doctors. This
underscores how the chatbot functioned not only as an information
resource but also as acounter-narrative space, one where users
could break away from restrictive norms of propriety and secrecy
around reproductive health.
6.2.2 The Multifaceted Accessibility of a LLM-Based Chatbot.We
found preference for the chatbot over other technological sources,
such as Google, due to the LLM’s convenience and flexibility
(C8,C11,C13). C13 noted:
“There are a lot of people on Google who tell you a
lot of things. But with the chatbot, you get to know
the right answer only, at once. If you compare it with
mother[in terms of accuracy], then it’s fine. But if you
search on YouTube or Google, then it takes time to
see if the answer is right or not. So, with the chatbot,
you only get one answer for one question.” (C13)
C8 highlighted how the LLM’s concise and direct responses via
WhatsApp saved time and mobile data, which is costly in Pakistan,
compared to navigating multiple websites on Google. This corrob-
orates our workshop finding, where constraints around internet
affordability appeared to steer participants toward AI tools embed-
ded in subsidized platforms such as WhatsApp and Snapchat.
Users also appreciated the chatbot’s flexible responses: C9 valued
its detailed information, while C11 preferred brief summaries and
liked that the chatbot made answers shorter when asked. This
flexibility reinforced chatbot’s role as a personalized, user-friendly
tool for independent learning.

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
Table 3: Topics appeared in co-design workshops and in chatbot conversations. A bullet (•) indicates presence of a theme; blank
cells indicate absence.
Topics Example Question [Translated] Co-Design
WorkshopsChatbot Data
Puberty “Is it true that once periods begin, height growth stops?” •
Menopause“At what age do periods stop?”•
Vaginal discharge“White liquid after period is normal?”•
Period blood “Is it normal to have some clots of blood after my pe-
riod?”•
PCOS “Can I manage my PCOS and lose weight through dieting
alone?”•
UTI “I am 21 years old. I feel a very severe pain in the urinary
area that lasts only 2–3 seconds, and then it goes away.
The pain comes only for about 3 seconds. ”•
Pregnancy “If I delay my periods, will it affect my chances of getting
pregnant?”•
Understanding of reproductive sys-
tem“What is sperm?”•
Menstrual pain and discomfort“How can I treat period cramps at home?”• •
Menstrual health concerns“What are the causes of irregular periods?”• •
Cultural myths “Is sitting in the sunlight during periods beneficial or
not?”• •
Understanding of menstruation“Why do periods occur?”• •
Menstrual products and usage“Menstrual cups are safe?”• •
General lifestyle during periods“Can we eat a lot of food during periods?”• •
6.3 Percieved Persona, Trust, and Authority
6.3.1 Anthropomorphism and the Perceived Femininity of Chatbot.
Anthropomorphism refers to the attribution of human-like qualities
to non-human entities. In our study, users demonstrated varying
degrees of anthropomorphism in their interactions with the LLM-
powered chatbot, attributing characteristics such as emotion and
gender to the system. C3 highlighted the chatbot’s empathetic and
responsive nature, noting:
“The chatbot treats you really well. If you ask it any
question, you get an answer right away. And if you
tell it ‘thank you’, it will say even nicer things.” (C3)
Such accounts demonstrate that participants perceived the chat-
bot not only as functional but also as affectively engaged—an en-
tity capable of politeness and warmth. C5 extended this percep-
tion by employing religious greetings such as“AOA”(Assalamu
Alaikum/Peace be upon you) and offering gratitude in the form of
“I give you 5 stars”, despite the absence of a ratings feature. These
practices reveal how users imported familiar social scripts into
their exchanges with the system, treating it as if it were a human
conversational partner deserving of etiquette and evaluation.
Anthropomorphism was particularly salient in relation to gen-
der. Despite no design directive to embody a gendered persona, the
chatbot’s use of Roman Urdu, a gendered language, led it to adopt
feminine forms in self-reference. For instance, during C1’s conver-
sation with the chatbot, the LLM used the feminine form of the
verb “karungi” (meaning “I will do” in the feminine form) to refer
to itself. This subtle linguistic cue carried profound implications:
C1, who initially hesitated due to uncertainty about the chatbot’s
gender, reported feeling significantly more comfortable once she
perceived it as female, noting:“At the start, I didn’t know if it was a male or a female,
when I had to ask questions personally. Then, I had
a little issue. But when I asked the question, I found
out that it was a female. Then I started questioning.
There was no other issue.” (C1)
C8 also viewed the“computer”running the chatbot as gendered.
These findings are particularly meaningful in light of Islamic norms:
women are expected to limit unnecessary interaction with unrelated
men (‘na-mehram’) and the majority prefer female gynecologists
due to family expectations and their own religious beliefs [ 22,33].
In this context, the chatbot’s perceived femininity not only low-
ered barriers to engagement but also actively cultivated trust and
intimacy, situating the system as a safe place for sensitive health
questions. While the chatbot’s perceived femininity fostered trust
for some users, its speed of response prompted others to question
its human-like qualities as“a human being couldn’t type this fast”
(C4). Other users described the chatbot as a genderless“machine”
(C9), a“robot”(C6, C11), and“AI”(C5, C10).
6.3.2 Trust and Validation.Participants often used secondary
sources, such as Google, to validate responses before fully accepting
them (C1, C7, C9, C13). However, initial cross-checking often led
to later trust in the chatbot. As C1 described:
“I did verify one...I searched it on Google. The mes-
sage from the chatbot and the name [of condition]
on Google were the same. Then I realized that the an-
swers to the rest of the questions will also be correct.”
(C1)
Others trusted the chatbot when its answers aligned with prior
knowledge or echoed what they had already heard from doctors,

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
family members or seen on Google (C4, C5, C6, C12). Interestingly,
exposure to exisiting LLMs also shaped trust. As C8 explained:
“I had the idea that it will not give the wrong informa-
tion. I was satisfied. That is why I didn’t go to Google
to search more. I have been using Meta AI. I have been
using the AI on Snapchat. When they first came out, I
wasn’t satisfied with the information they would give,
so I would always double-check on Google. I learned
that the information Meta AI or Snapchat AI gives is
usually correct. When I used chatbot, I already had an
idea. I had an idea that it is the same. The information
is good.” (C8)
Users generally saw the chatbot as a reliable source of quick an-
swers, and none reported any instance of ‘inappropriate’ or ‘incon-
sistent’ response. However, rather than accepting the chatbots au-
thority blindly, users situated the chatbot’s advice within a broader
ecology of knowledge—comparing it against personal experience,
family wisdom, and peer recommendations. They chose to act only
when its guidance aligned with, or could be corroborated by, other
trusted sources. For example, C3 followed the chatbot’s reassurance
that taking medicine during menstruation is safe, which directly
countered the widespread myth that it is harmful. However, she
chose to take a medicine she was already familiar with, indicating
that her decision was shaped by prior experience rather than blind
trust in the chatbot. She noted:
“I have [faced] a lot of issues with [finding answers
related to] medicine and diet...It [chatbot] said that
there is no issue with taking medicines [for pain]. . . So
today I am in the same condition [menstrual cramps]
and I have taken the medicine. Now let’s see. It will be
an experience. Then we will know if it is right or not.
I had never taken medicine for period pain before, but
let’s see this time.” (C3)
Similarly, C12 used the chatbot’s recommendation to support a
friend but added an extra layer of verification by first consulting
her sister before passing on the advice. C12 recalled:
“My friend was in a lot of pain. She shared this with
me and I asked the chatbot about the medicine. It
recommended the ibrufen tablet for the pain. I told
this to my friend. She took the tablet and it helped
her.” (C12)
These examples illustrate that the users treated the chatbot as a
useful input into decision-making but not as the final authority. In
doing so, they enacted a form of ‘layered trust’, where digital infor-
mation is meaningful only when filtered through lived experience
and embedded social networks.
6.3.3 Privacy Concerns.Similar to workshop participants, chatbot
users also preferred text-based interaction. For instance, when asked
whether she would have liked an audio feature in the chatbot, one
participant explained,
“I feel hesitant in speaking out my problems loudly
into the chatbot." (C11)
Concerns also extended to the potential misuse or leakage of
their WhatsApp numbers (C13,C4). One user shared her initial
hesitation regarding authenticity of the chatbot:“In the beginning, I was also concerned about my
number getting leaked. I asked this from the girl [fa-
cilitator] who added me to the group chat, but she
said it won’t." (C13)
However, others did not show such concern. One user inquired
about who might review her conversational data and was reminded
that the consent form specified it would only be accessible to the
research team. She further noted that the reviewer’s gender could
shape what she felt comfortable asking:
“I can ask you anything because you’re a woman. De-
pending on the gender of the person who sees my
questions, I would adjust what I ask." (C9)
Although the research team consists of all women in this case,
these concerns reflect broader issues of trust in stigmatized contexts
and highlight the importance of transparent data handling prac-
tices and clear communication about privacy measures in chatbot
design.
Since the chatbot was deployed entirely within WhatsApp, our
design inherited the platform’s existing privacy limitations and
safeguards (for example, chat deletion and archival), rather than im-
plementing additional custom privacy-preserving strategies specifi-
cally for shared phone use. Among the participants, six had shared
phones: two shared with a sister (C5, C7) and four with their mother
(C1, C3, C8, C11). Among these, one participant archived the chat-
bot conversation (C3), one deleted messages as soon as she read
them (C8), and another deleted the chat once the study was over
(C11). The remaining participants (C1, C5, C7) explained that they
did not delete the messages because they shared the phone only
with a female family member. In contrast, participants who owned
a personal phone reported that they did not delete or archive the
chats, as no one else uses their phone.
7 Discussion
7.1 Conversational Care and the Synergies of
Epistemic Bridging
Prior work has noted that factual information alone often fails to
change health practices: instead, women look for conversational
explanations that situate why certain traditions exist and why they
should be reconsidered [ 50,97,114]. Our analysis suggests that the
chatbot’s value lay not in simply correcting myths, but in engaging
inepistemic bridging—connecting the users’ lived cultural knowl-
edge with biomedical explanations without dismissing the former
as“backward”. However, we found that users engaged in“learning
loops”where they tested the chatbot against family wisdom, echo-
ing earlier HCI work on culturally sensitive technologies that enable
people to open up about taboo topics while gradually building con-
fidence to question inherited beliefs [ 20,76,81]. When the chatbot
validated the intent of a myth (e.g., acknowledging that pain is real)
before correcting the mechanism (explaining it via prostaglandins
rather than“dirty blood”), it legitimized the user’s suffering. This
reframing of‘normal suffering’into legitimate health concerns is a
form of epistemic justice [ 4]. In a context where women’s pain is
often dismissed as a“natural”part of womanhood, the chatbot’s
use of medical terminology (e.g., PMS, Dysmenorrhea) did not just
educate; it validated. This suggests that in stigma-laden domains,

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
accuracy must be balanced with affective validation—designers
must prioritize relational trust (validating the user’s anxiety) as a
prerequisite to informational trust (accepting the medical fact).
The success of this approach hinges on the synergistic relation-
ship between infrastructural fit, linguistic localization, and affective
validation. For instance, the system’s foundational support for Ro-
man Urdu instantly reinforced the chatbot’s perceived humanity
and non-judgmental tone, making it feel ‘like talking to a friend’,
which is a vital prerequisite for discussing sensitive topics (D2).
This Conversational Care then enabled the trust required for the
user to accept Epistemic Bridging, as they were more receptive to
challenging family myths when the advice came from a validated,
linguistically familiar, and seemingly compassionate source. Thus,
trust was co-constructed through a layered synergy: convenience
of the platform (WhatsApp) validated initial adoption, linguistic flu-
ency (Roman Urdu) provided affective trust, and accurate grounding
(RAG) sustained informational trust.
Our findings also reveal a critical gap in the capacity of standard
LLMs to navigate local explanatory models of health. This high-
lights that for LLMs to be effective in the Global South, they must do
more than retrieve biomedical facts; they must perform epistemic
translation. A purely biomedical response that fails to recognize
“taseer”risks dismissing the user’s lived reality, creating dissonance
rather than trust. This echoes the Postcolonial HCI critique that
technological interventions often impose globalized epistemologies,
neglecting local systems of knowledge and meaning [ 35]. Future
systems must be trained not just to correct these models, but to
recognize them as valid entry points for dialogue, bridging the gap
between cultural beliefs and medical science without alienation.
7.2 Infrastructures of Mediation and Adoption
We frame our findings through the idea of designing infrastructures
rather than only designing agents [ 40]. This lens shifts attention
away from the chatbot as a standalone interface with a persona and
toward how it becomes embedded within broader sociotechnical
systems. From this perspective, trust and adoption are shaped not
only by the chatbot’s design features but also by the infrastructures
of communication, mediation, and cultural norms in which it circu-
lates [ 30,39,91]. In our Pakistani Muslim-majority context, these
infrastructures are deeply entangled with norms ofhaya(modesty)
andpurdah(concealment), which extend into how women manage
visibility, voice, and contact in digital spaces.
Our findings illustrate this in several ways. Participants’ sensi-
tivity to the chatbot’s gendered style, for example, cannot be read
simply as a response to persona design. Rather, it reflects a broader
infrastructural reality in Pakistan, where ∼95% of women prefer
female gynecologists due to family pressure, religious expectations,
and prevailing gender norms in healthcare [ 22]. For many partici-
pants, interacting with a “female” chatbot aligned with expectations
of seeking care from same-gender providers and helped preserve
a sense ofhaya. Similarly, the chatbot’s operation through What-
sApp, already subsidized through “social bundles” and normalized
in participants’ social lives, positioned it as part of an existing digi-
tal infrastructure of access. Its text-based format, which required
no profile photo, voice call, or video, further reduced hesitation by
enabling what we might call a form ofdigital purdah: women couldask intimate questions without face-to-face interaction. In this way,
the chatbot leveraged existing infrastructures and interactional
norms to make menstruation-related questions feel religiously and
socially permissible.
At the same time, the same infrastructural conditions also seeded
distrust. Some participants initially hesitated to message an unfa-
miliar number due to privacy concerns, echoing prior work where
the risk of number leakage and unsolicited messages in mixed-
gender groups discouraged women’s participation [ 113]. Here, the
threat was not only unwanted contact but a potential breach of
digital purdah[ 71], where visibility to unknown men could be in-
terpreted as violating norms of modesty and religiously inflected
gender boundaries. In our study, this hesitation was reduced not
through interface changes but through relational mediation, as
a fellow student assured participants that the chatbot was legiti-
mate. Trust was thus co-constructed through the social pathways
by which the system was introduced and the moral worlds those
pathways invoked, not just through the chatbot’s persona. For ex-
ample, had the chatbot been introduced by a trusted teacher or
religiously affiliated institution, users might have been more will-
ing to adopt or recommend it compared to when it was referred by
peers. This layered trust urges the technology design to treat the
social and infrastructural context as an integral part of the system
architecture. This observation aligns with findings on South Asian
women’s privacy negotiations, where technology use is frequently
monitored by family, making social context and mediated access
critical determinants of adoption and utility [88].
7.3 The Stigma-Aware Conversational Design
Framework
“In countries like Pakistan, where impositions on any
sort of dialogue are strongly influenced by religious
and traditional practices, public dialogues focusing on
sexuality education are considered extremely taboo and
invite great criticism and outrage from the public. ”[ 93]
Against this backdrop of religion and culture, formal education in
Pakistan systematically excludes MHE [ 74,83], unlike India [ 73].
This means we must work with the limitations that users face, and
by extension the constraints we as designers encounter, when de-
signing within such contexts. One might argue that the “chatbot
as a private space” risks perpetuating stigma by reinforcing the
idea that menstruation is not something to be discussed openly.
This tension between reinforcing norms (via privacy) and enabling
agency (via discreet access) is a central dilemma in Feminist HCI in
the Global South [ 8,18,101]: meeting real-world needs may inad-
vertently reinforce the status quo, while overtly activist designs risk
imposing designers’ own values [ 8]. However, it is neither realistic
nor ethical for designers, especially those without lived experience
in such contexts, to assume that they can quickly undo patriarchy
or tell women to live differently just to fit their designs [ 101]. In-
stead, we may design within the existing stigmatized context, with
the hope that our interventions can gradually unsettle stigma.
For our design, this meant recognizing existing limitations while
seeking to educate women within their current realities, with the

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
goal of gradually challenging taboos and reshaping how they per-
ceive menstruation. Many women engaged with the chatbot pre-
cisely because it operated privately within existing structures of
stigma, enabling questions they hesitated to voice in public or famil-
ial settings. By working within these limits, we created a platform
that did not directly confront cultural taboos but gently unsettled
them [ 40,70] by showing that conversations around menstruation
are valid, answerable, and worthy of care.
Rather than“making peace”with constraints, we frame our ap-
proach as strategically working within them to create space for
gradual change. The chatbot created micro-level openings for young
women to question myths and verify information, marking small
but meaningful shifts in how they engage with their reproductive
health. While such situated interventions cannot claim to overturn
patriarchal norms, they show how culturally grounded designs
can create fissures in silence and open pathways for incremental
change.
Drawing on the synthesized tensions and synergistic principles
from Sections 7.1 and 7.2, we articulate a set of design commit-
ments (D1–D4) for LLM-based health chatbots thatdesign around
stigmarather than outside it in Table 4. This framework specifies
how infrastructural fit, conversational care, expert grounding, and
iterative adaptation can be configured to support women’s agency
within patriarchal, religiously inflected contexts.
8 Limitations
Our study has some limitations. The sample was small and demo-
graphically narrow, offering rich qualitative insight but limiting
generalizability. Our short deployment duration may not reflect
long-term engagement or sustainability of use. Participants’ nov-
elty and curiosity could have shaped their interactions, while issues
such as habituation, drop-off in usage, or evolving trust over time
remain unexplored. Our study design involved continuous human
oversight, including reviewing conversation logs and refining the
knowledge base. While this invisible labor was essential for safety
and cultural alignment, it limits scalability. The extent to which
such oversight can be automated or sustainably maintained in real-
world deployments remains an open question. Finally, we studied
the chatbot in a sensitive and taboo-laden domain, where responses
can carry heightened social and emotional consequences. Even with
expert validation, chatbots cannot replace professional medical ad-
vice. Our work should therefore be interpreted as augmenting, not
substituting, existing health services. Together, these limitations
underline the need for future research to test culturally sensitive,
LLM-based health chatbots across longer timescales, more diverse
populations, and broader infrastructural conditions, while continu-
ing to foreground expert validation and community trust.
9 Conclusion
This paper contributes the first empirical study of a localized LLM-
powered chatbot for menstrual health in Pakistan, advancing HCI’s
understanding of how conversational systems operate under stigma,
silence, and infrastructural constraint. Our study shows that lo-
calized AI can create private, judgment-free spaces for exploring
sensitive health topics, while also surfacing tensions around trust,cultural alignment, and validation. By showing how language prac-
tices, infrastructural limits, and cultural norms shape engagement,
we distill design commitments for building safer and more reso-
nant conversational systems. We also highlight the hidden labor
of expert validation as essential for cultural relevance and medical
safety. This work is not only about correcting hallucinations but
about bridging biomedical knowledge with local explanatory mod-
els that shape how people understand their bodies. Designing with,
rather than against, these models builds trust and makes systems
meaningful in everyday life.
Looking ahead, we see opportunities to extend this approach
to domains such as fertility, menopause, and sexual health, and
to more systematically embed euphemisms and cultural framings
into conversational agents. At a broader level, dignity in health
technology is not achieved by correct answers alone, but by en-
abling people to ask questions they could not ask elsewhere, and
by receiving responses that resonate with their cultural worlds.
References
[1]M. Abbasian, E. Khatibi, I. Azimi, et al .2024. Foundation metrics for evaluating
effectiveness of healthcare conversations powered by generative AI.npj Digital
Medicine7, 1 (2024), 82. doi:10.1038/s41746-024-01074-z
[2]P. Adhikary, I. Motiyani, G. Oke, M. Joshi, K. Pathak, S. Singh, and T. Chakraborty.
2025. Menstrual Health Education Using a Specialized Large Language Model in
India: Development and Evaluation Study of MenstLLaMA.Journal of Medical
Internet Research27, 1 (2025), e71977. doi:10.2196/71977
[3]Muhammad Farid Adilazuarda, Sagnik Mukherjee, Pradhyumna Lavania, Sid-
dhant Shivdutt Singh, Alham Fikri Aji, Jacki O’Neill, Ashutosh Modi, and Mono-
jit Choudhury. 2024. Towards Measuring and Modeling “Culture” in LLMs: A
Survey. InProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen
(Eds.). Association for Computational Linguistics, Miami, Florida, USA, 15763–
15784. doi:10.18653/v1/2024.emnlp-main.882
[4]Leah Hope Ajmani, Jasmine C Foriest, Jordan Taylor, Kyle Pittman, Sarah Gilbert,
and Michael Ann Devito. 2024. Whose Knowledge is Valued? Epistemic Injustice
in CSCW Applications.Proceedings of the ACM on Human-Computer Interaction
8, CSCW2 (2024), 1–28.
[5]Tazeen S Ali, Parveen Azam Ali, Humaira Waheed, and Azam Ali Memon.
2006. Understanding of puberty and related health problems among female
adolescents in Karachi, Pakistan.Journal of Pakistan medical association56, 2
(2006), 68.
[6]Mike Armour, Kelly Parry, Christina Curry, Tania Ferfolja, Melissa A Parker,
Toobah Farooqi, Freya MacMillan, Caroline A Smith, and Kathryn Holmes. 2022.
Evaluation of a web-based resource to improve menstrual health literacy and
self-management in young women.Journal of Psychosomatic Research162 (2022),
111038.
[7]Joshua Au Yeung, Zeljko Kraljevic, Akish Luintel, Alfred Balston, Esther Idowu,
Richard J. Dobson, and James T. Teo. 2023. AI chatbots not yet ready for clinical
use.Frontiers in Digital Health5 (12 April 2023), 1161098. doi:10.3389/fdgth.
2023.1161098
[8]Shaowen Bardzell. 2010. Feminist HCI: taking stock and outlining an agenda
for design. InProceedings of the SIGCHI Conference on Human Factors in Com-
puting Systems(Atlanta, Georgia, USA)(CHI ’10). Association for Computing
Machinery, New York, NY, USA, 1301–1310. doi:10.1145/1753326.1753521
[9]Abul Barkat and Murtaza Majid. 2003. Adolescent reproductive health in
Bangladesh: Status, policies, programs and issues.
[10] Gul Mehar Javaid Bukhari, Maimoona Nazar, Laiba Sultan, Janita Azeem, Mariam
Jamil, Hajr-e-Aswad Khan Khattak, and Farah Rashid. 2023. Myths and Chal-
lenges Regarding Menstrual Hygiene among Adolescent Girls of Islamabad.
Pakistan Journal of Public Health13, 4 (2023), 168–171.
[11] D. Chandra and E. Gomez. 2019. Dignified Menstruation Southeast Asia Regional
Overview. InDignified Menstruation: (The Dignity of Menstruators Through-
out Their Life Cycle): A Practical Handbook, Radha Paudel (Ed.). Kathmandu
Publication, Kathmandu, 11–21.
[12] Vikas Chothe, Jagdish Khubchandani, Denise Seabert, Mahesh Asalkar, Sarika
Rakshe, Arti Firke, Inuka Midha, and Robert Simmons. 2014. Students’ percep-
tions and doubts about menstruation in developing countries: a case study from
India.Health promotion practice15, 3 (2014), 319–326.
[13] Hafsah Mahzabin Chowdhury and Sharifa Sultana. 2025. From Literature to
ReWA: Discussing Reproductive Well-being in HCI. arXiv:2507.01121 [cs.HC]
https://arxiv.org/abs/2507.01121

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
Design Commitment Relevance to Designing around
StigmaImplications for Design Trade-offs
D1. Infrastructural fit &
digitalpurdahEmbeds the chatbot in infrastructures
that already feel religiously and socially
permissible, reducing the visibility and
social risk of help-seeking in stigmatized
domains.Deploy on subsidized, widely used plat-
forms (e.g., WhatsApp in Pakistan); provide
the option for text interaction; present the
system as a “female” helper; clearly com-
municate confidentiality and who can see
logs respecting religious norms ofpurdah.Platform Dependency undermines design
sovereignty, limiting ability to implement
custom privacy mechanisms. Framing the
system as a “female” helper may also en-
courage over-reliance on its advice or ob-
scure its non-human, fallible nature.
D2. Conversational care &
localizationUses empathy, validation, and local lan-
guage practices to make it feel acceptable
to ask ‘unspeakable’ questions, while gen-
tly questioning harmful norms.Adopt local language practices (Roman
Urdu mixed with English), recognize cul-
tural (e.g.‘after marriage’for sexual ac-
tivity) and religious (e.g.‘exemption from
salat’for menstruation) euphemisms, re-
spond in a non-judgmental tone, acknowl-
edge why myths persist before contrasting
them with biomedical explanations, and
invite “learning loops” where users can ac-
tively test family wisdom against the chat-
bot’s responses.Requires navigating tensions between em-
pathy and medical accuracy (e.g., validat-
ing feelings without downplaying risk);
overly softened language may obscure clin-
ical urgency, while blunt corrections may
feel shaming or dismissive.
D3. Expert-grounded,
bounded adviceProvides an alternative authority to fam-
ily myths while clearly signalling limits,
so that challenging stigma does not cre-
ate new clinical risks.Ground responses in expert-curated con-
tent and RAG; state what the chatbot
can/cannot do; decline to answer beyond
scope (e.g., unknown medicines); redirect
to clinicians when needed; cite credible
sources (e.g., WHO guidelines) that users
can cross-check.Balances safety against usability: strict re-
fusals can frustrate users or be perceived as
unhelpful, whereas overconfident answers
risk clinical harm and may erode trust if
contradicted by providers.
D4. Iterative, user-in-the-
loop adaptation to local
logicsTreats conversation logs as a lens onto
local explanatory models (e.g., hot/cold
foods, religious obligations) and updates
the system to engage those models rather
than dismiss them.Analyze logs with clinicians to sur-
face blind spots (e.g., misunderstanding
thanda/garam); add new intents and expla-
nations that bridge between local logics
(taseer, prayer) and biomedical models; re-
fine prompts to better handle misspellings
and mixed language.Trades off learning from real data against
privacy and governance concerns, requir-
ing strong consent processes, aggregation,
and careful decisions about which user ex-
pressions to codify into the system.
Table 4: A stigma-aware design framework for LLM-based health chatbots in low-resource, patriarchal contexts
[14] J.S. Corrêa, A.P.d.A. Neto, G.R. Pinto, L.D.B. Lima, and A.S. Teles. 2023. Lhia: A
Smart Chatbot for Breastfeeding Education and Recruitment of Human Milk
Donors.Applied Sciences13, 12 (2023), 6923. doi:10.3390/app13126923
[15] Andrea Cuadra, Maria Wang, Lynn Andrea Stein, Malte F. Jung, Nicola Dell,
Deborah Estrin, and James A. Landay. 2024. The Illusion of Empathy? Notes on
Displays of Emotion in Human-Computer Interaction. InProceedings of the 2024
CHI Conference on Human Factors in Computing Systems(Honolulu, HI, USA)
(CHI ’24). Association for Computing Machinery, New York, NY, USA, Article
446, 18 pages. doi:10.1145/3613904.3642336
[16] Dung Dao, Jun Yi Claire Teo, Wenru Wang, and Hoang D. Nguyen. 2024. LLM-
Powered Multimodal AI Conversations for Diabetes Prevention. InProceedings
of the 1st ACM Workshop on AI-Powered Q&A Systems for Multimedia(Phuket,
Thailand)(AIQAM ’24). Association for Computing Machinery, New York, NY,
USA, 1–6. doi:10.1145/3643479.3662049
[17] Roshini Deva, Dhruv Ramani, Tanvi Divate, Suhani Jalota, and Azra Ismail. 2025.
"Kya family planning after marriage hoti hai?": Integrating Cultural Sensitivity
in an LLM Chatbot for Reproductive Health. InProceedings of the 2025 CHI
Conference on Human Factors in Computing Systems (CHI ’25). Association for
Computing Machinery, New York, NY, USA, Article 638, 23 pages. doi:10.1145/
3706598.3713362
[18] Catherine D’ignazio and Lauren F. Klein. 2020. Seven intersectional feminist
principles for equitable and actionable COVID-19 data.Big data & society7, 2
(2020), 2053951720942544.
[19] Catherine D’Ignazio and Lauren F. Klein. 2020.Data Feminism. The MIT Press,
Cambridge, Massachusetts.
[20] Daniel A. Epstein, Nicole B. Lee, Jennifer H. Kang, Elena Agapie, Jessica
Schroeder, Laura R. Pina, James Fogarty, Julie A. Kientz, and Sean Munson.
2017. Examining Menstrual Tracking to Inform the Design of Personal Infor-
matics Tools. InProceedings of the 2017 CHI Conference on Human Factors in
Computing Systems(Denver, Colorado, USA)(CHI ’17). Association for Comput-
ing Machinery, New York, NY, USA, 6876–6888. doi:10.1145/3025453.3025635[21] Cristina Espinosa-Heywood and Richa Karki. 2019. Breaking Taboos: Menstru-
ation, Female Subordination and Reproductive Health, the Case of India.
[22] Liu Feng, Muhammad Jawad Sajid, Zhu Lian, and Li Shuqin. 2018. Female gyne-
cologist preference: A viewpoint based on two scenario survey from Pakistan.
International Journal of Research05 (05 2018).
[23] Asbjørn Følstad, Petter Bae Brandtzaeg, Tom Feltwell, Effie L-C. Law, Manfred
Tscheligi, and Ewa A. Luger. 2018. SIG: Chatbots for Social Good. InExtended
Abstracts of the 2018 CHI Conference on Human Factors in Computing Systems
(Montreal QC, Canada)(CHI EA ’18). Association for Computing Machinery,
New York, NY, USA, 1–4. doi:10.1145/3170427.3185372
[24] Gallup Pakistan. 2021.Gallup Pakistan: WhatsApp’s Widespread Usage in Pak-
istan. https://gallup.com.pk/post/33823 Gallup Pakistan Report.
[25] Shixian Geng, Remi Inayoshi, Chi-Lan Yang, Zefan Sramek, Yuya Umeda, Chiaki
Kasahara, Arissa J. Sato, Simo Hosio, and Koji Yatani. 2025. Beyond the Dialogue:
Multi-chatbot Group Motivational Interviewing for Premenstrual Syndrome
(PMS) Management. InProceedings of the 2025 CHI Conference on Human Factors
in Computing Systems (CHI ’25). Association for Computing Machinery, New
York, NY, USA, Article 640, 18 pages. doi:10.1145/3706598.3713918
[26] Johanna Habicht et al .2024. Closing the Accessibility Gap to Mental Health
Treatment with a Personalized Self-Referral Chatbot.Nature Medicine30, 2
(2024), 595–602.
[27] Yuexing Hao, Jason Holmes, Mark R Waddle, Brian J Davis, Nathan Y Yu,
Kristin S Vickers, Heather Preston, Drew Margolin, Corinna E Löckenhoff,
Aditya Vashistha, et al .2025. Personalizing prostate cancer education for pa-
tients using an EHR-Integrated LLM agent.npj Digital Medicine8, 1 (2025),
770.
[28] Christina N. Harrington and Lisa Egede. 2023. Trust, Comfort and Relatability:
Understanding Black Older Adults’ Perceptions of Chatbot Design for Health
Information Seeking. InProceedings of the 2023 CHI Conference on Human Factors
in Computing Systems. Association for Computing Machinery, New York, NY,
USA, 1–18. doi:10.1145/3544548.3580719

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
[29] Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing
Xu, Soujanya Poria, and Roy Lee. 2023. LLM-Adapters: An Adapter Family for
Parameter-Efficient Fine-Tuning of Large Language Models. InProceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing, Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 5254–5276. doi:10.18653/v1/2023.emnlp-main.319
[30] Gilhwan Hwang, Jeewon Lee, Cindy Yoonjung Oh, and Joonhwan Lee. 2019. It
Sounds Like A Woman: Exploring Gender Stereotypes in South Korean Voice
Assistants. InExtended Abstracts of the 2019 CHI Conference on Human Factors
in Computing Systems(Glasgow, Scotland Uk)(CHI EA ’19). Association for
Computing Machinery, New York, NY, USA, 1–6. doi:10.1145/3290607.3312915
[31] Zaidat Ibrahim, Novia Nurain, and James Clawson. 2024. Tracking During
Ramadan: Examining the Intersection of Menstrual and Religious Tracking
Practices Among Muslim Women in the United States. InProceedings of the 2024
CHI Conference on Human Factors in Computing Systems(Honolulu, HI, USA)
(CHI ’24). Association for Computing Machinery, New York, NY, USA, Article
689, 19 pages. doi:10.1145/3613904.3642374
[32] Zaidat Ibrahim, Pallavi Panchpor, Novia Nurain, and James Clawson. 2024.
"Islamically, I am not on my period": A Study of Menstrual Tracking in Muslim
Women in the US. InProceedings of the 2024 CHI Conference on Human Factors in
Computing Systems(Honolulu, HI, USA)(CHI ’24). Association for Computing
Machinery, New York, NY, USA, Article 686, 16 pages. doi:10.1145/3613904.
3642006
[33] Samia Ibtasam. 2021. For God’s sake! Considering Religious Beliefs in HCI
Research: A Case of Islamic HCI. InExtended Abstracts of the 2021 CHI Confer-
ence on Human Factors in Computing Systems(Yokohama, Japan)(CHI EA ’21).
Association for Computing Machinery, New York, NY, USA, Article 9, 8 pages.
doi:10.1145/3411763.3450383
[34] Samia Ibtasam, Lubna Razaq, Maryam Ayub, Jennifer R Webster, Syed Ishtiaque
Ahmed, and Richard Anderson. 2019. " My cousin bought the phone for me.
I never go to mobile shops." The Role of Family in Women’s Technological
Inclusion in Islamic Culture.Proceedings of the ACM on Human-Computer
Interaction3, CSCW (2019), 1–33.
[35] Lilly Irani, Janet Vertesi, Paul Dourish, Kavita Philip, and Rebecca E. Grinter.
2010. Postcolonial computing: a lens on design and development. InProceedings
of the SIGCHI Conference on Human Factors in Computing Systems(Atlanta,
Georgia, USA)(CHI ’10). Association for Computing Machinery, New York, NY,
USA, 1311–1320. doi:10.1145/1753326.1753522
[36] Ann Irvine, Jonathan Weese, and Chris Callison-Burch. 2012. Processing infor-
mal, romanized Pakistani text messages. InProceedings of the Second Workshop
on Language in Social Media(Montreal, Canada)(LSM ’12). Association for
Computational Linguistics, USA, 75–78.
[37] Rabeea Jabbar, Noor Us Sama, and Ayesha Ihtisham Khattak. 2025. Integrating
menstrual hygiene management (MHM) in disaster health response: lessons
from 2022 Pakistan floods.Environmental Research: Health3, 3 (2025), 035001.
doi:10.1088/2752-5309/add364 Open access.
[38] Minal Jain and Pradeep Yammiyavar. 2015. Game based learning tool seeking
peer support for empowering adolescent girls in rural Assam. InProceedings
of the 14th International Conference on Interaction Design and Children(Boston,
Massachusetts)(IDC ’15). Association for Computing Machinery, New York, NY,
USA, 275–278. doi:10.1145/2771839.2771895
[39] Shivani Kapania, Oliver Siy, Gabe Clapper, Azhagu Meena SP, and Nithya
Sambasivan. 2022. "Because AI is 100% right and safe": User Attitudes and
Sources of AI Authority in India. InProceedings of the 2022 CHI Conference
on Human Factors in Computing Systems(New Orleans, LA, USA)(CHI ’22).
Association for Computing Machinery, New York, NY, USA, Article 158, 18 pages.
doi:10.1145/3491102.3517533
[40] Naveena Karusala, Victoria G, Shirley Yan, and Richard Anderson. 2023. Unset-
tling Care Infrastructures: From the Individual to the Structural in a Digital Ma-
ternal and Child Health Intervention. InProceedings of the 2023 CHI Conference on
Human Factors in Computing Systems (CHI ’23). Association for Computing Ma-
chinery, New York, NY, USA, Article 42, 16 pages. doi:10.1145/3544548.3581553
[41] Jasmeet Kaur, Preetika Sharma, Vijay Kumar, Mona Duggal, Nadia Griffin
Diamond-Smith, Alison El Ayadi, Kathryn Vosburg, and Pushpendra Singh.
2024. Exploring the Role of Chatbots in Tackling COVID-19 Vaccine Hesitancy
among Pregnant and Breastfeeding Women in Rural Northern India.Proceedings
of the ACM on Human-Computer Interaction8, CSCW1 (2024), 1–29.
[42] Jasmeet Kaur, Asra Sakeen Wani, and Pushpendra Singh. 2019. Engagement
of Pregnant Women and Mothers over WhatsApp: Challenges and Opportu-
nities Involved. InCompanion Publication of the 2019 Conference on Computer
Supported Cooperative Work and Social Computing(Austin, TX, USA)(CSCW
’19 Companion). Association for Computing Machinery, New York, NY, USA,
236–240. doi:10.1145/3311957.3359481
[43] Khawar Latif Khan and Farah Azhar. 2023.FemTech and Taboo Topics: Raaji as a
Tool for Educating Women in Pakistan. Springer Nature Singapore, Singapore,
185–210. doi:10.1007/978-981-99-5605-0_9
[44] Shahedul Huq Khandkar. 2009. Open coding.University of Calgary23, 2009
(2009), 2009.[45] Lauren Klein and Catherine D’Ignazio. 2024. Data Feminism for AI. InProceed-
ings of the 2024 ACM Conference on Fairness, Accountability, and Transparency
(Rio de Janeiro, Brazil)(FAccT ’24). Association for Computing Machinery, New
York, NY, USA, 100–112. doi:10.1145/3630106.3658543
[46] Ahmet Baki Kocaballi, Shlomo Berkovsky, Juan C. Quiroz, Liliana Laranjo,
Huong Ly Tong, Dana Rezazadegan, Agustina Briatore, and Enrico Coiera. 2019.
The Personalization of Conversational Agents in Health Care: Systematic Review.
Journal of Medical Internet Research21, 11 (2019), e15360. doi:10.2196/15360
[47] Ha-Kyung Kong, Rachel Lowy, Youjin Choi, and Jennifer G Kim. 2025. Working
Together Toward Interdependence: Chatbot-Based Support for Balanced Social
Interactions Between Neurodivergent and Neurotypical Individuals. InProceed-
ings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI
’25). Association for Computing Machinery, New York, NY, USA, Article 779,
17 pages. doi:10.1145/3706598.3713344
[48] Anant Kumar and Kamiya Srivastava. 2011. Cultural and social practices re-
garding menstruation among adolescent girls.Social Work in Public Health26,
6 (2011), 594–604.
[49] Jyoti Kumar and Surbhi Pratap. 2020. Detriments to Cultural Sensitivity in
HCI Design Processes: Insights from Practitioners’ Experiences in India. In
HCI International 2020 - Late Breaking Papers: User Experience Design and Case
Studies: 22nd HCI International Conference, HCII 2020, Copenhagen, Denmark,
July 19–24, 2020, Proceedings(Copenhagen, Denmark). Springer-Verlag, Berlin,
Heidelberg, 142–155. doi:10.1007/978-3-030-60114-0_10
[50] Neha Kumar, Trevor Perrier, Michelle Desmond, Kiersten Israel-Ballard, Vikrant
Kumar, Sudip Mahapatra, Anil Mishra, Shreya Agarwal, Rikin Gandhi, Pallavi
Lal, and Richard Anderson. 2015. Projecting health: community-led video edu-
cation for maternal health. InProceedings of the Seventh International Conference
on Information and Communication Technologies and Development(Singapore,
Singapore)(ICTD ’15). Association for Computing Machinery, New York, NY,
USA, Article 17, 10 pages. doi:10.1145/2737856.2738023
[51] Yi-Chieh Lee, Naomi Yamashita, and Yun Huang. 2020. Designing a chatbot as a
mediator for promoting deep self-disclosure to a real mental health professional.
Proceedings of the ACM on Human-Computer Interaction4, CSCW1 (2020), 1–27.
[52] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in Neural Information Processing Systems33 (2020), 9459–9474.
[53] Brenna Li, Ofek Gross, Noah Crampton, Mamta Kapoor, Saba Tauseef, Mohit Jain,
Khai N. Truong, and Alex Mariakakis. 2024. Beyond the Waiting Room: Patient’s
Perspectives on the Conversational Nuances of Pre-Consultation Chatbots. In
Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems
(Honolulu, HI, USA)(CHI ’24). Association for Computing Machinery, New York,
NY, USA, Article 438, 24 pages. doi:10.1145/3613904.3641913
[54] Cheng Li, Mengzhuo Chen, Jindong Wang, Sunayana Sitaram, and Xing Xie.
2024. CultureLLM: Incorporating Cultural Differences into Large Language
Models. InAdvances in Neural Information Processing Systems, A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37.
Curran Associates, Inc., Red Hook, NY, USA, 84799–84838. doi:10.52202/079017-
2693
[55] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Prompt Distillation for Efficient
LLM-based Recommendation. InProceedings of the 32nd ACM International
Conference on Information and Knowledge Management(Birmingham, United
Kingdom)(CIKM ’23). Association for Computing Machinery, New York, NY,
USA, 1348–1357. doi:10.1145/3583780.3615017
[56] Xiubo Liang, Jing Zhang, Jiayi Ma, Jiayu Yao, Weijia Lin, Zheliang Zhu, Yingy-
ing Ma, Fangtian Ying, Cheng Yao, Leijing Zhou, Preben Hansen, Yijun Zhao,
and Guanyun Wang. 2022. Menstrual Monster: A Tangible Interactive Co-
educational Game Designed for Teenagers. InExtended Abstracts of the 2022
CHI Conference on Human Factors in Computing Systems(New Orleans, LA,
USA)(CHI EA ’22). Association for Computing Machinery, New York, NY, USA,
Article 247, 7 pages. doi:10.1145/3491101.3519723
[57] Georgianna Lin, Pierre-William Lessard, Minh Ngoc Le, Brenna Li, Fanny Cheva-
lier, Khai N. Truong, and Alex Mariakakis. 2024. Functional Design Requirements
to Facilitate Menstrual Health Data Exploration. InProceedings of the 2024 CHI
Conference on Human Factors in Computing Systems(Honolulu, HI, USA)(CHI
’24). Association for Computing Machinery, New York, NY, USA, Article 687,
15 pages. doi:10.1145/3613904.3642282
[58] Georgianna E Lin, Elizabeth D Mynatt, and Neha Kumar. 2022. Investigating
Culturally Responsive Design for Menstrual Tracking and Sharing Practices
Among Individuals with Minimal Sexual Education. InProceedings of the 2022
CHI Conference on Human Factors in Computing Systems(New Orleans, LA, USA)
(CHI ’22). Association for Computing Machinery, New York, NY, USA, Article
437, 15 pages. doi:10.1145/3491102.3501824
[59] Zilin Ma, Yiyang Mei, Yinru Long, Zhaoyuan Su, and Krzysztof Z. Gajos. 2024.
Evaluating the Experience of LGBTQ+ People Using Large Language Model
Based Chatbots for Mental Health Support. InProceedings of the 2024 CHI
Conference on Human Factors in Computing Systems(Honolulu, HI, USA)(CHI
’24). Association for Computing Machinery, New York, NY, USA, Article 872,

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
15 pages. doi:10.1145/3613904.3642482
[60] Divya Madhu, C.J. Neeraj Jain, Elmy Sebastain, Shinoy Shaji, and Anandhu
Ajayakumar. 2017. A novel approach for medical assistance using trained
chatbot. In2017 International Conference on Inventive Communication and Com-
putational Technologies (ICICCT). 243–246. doi:10.1109/ICICCT.2017.7975195
[61] Sanam Maher. 2023. The Activists Confronting Period Taboos in Pak-
istan. https://www.aljazeera.com/features/2023/5/28/the-activists-confronting-
period-taboos-in-pakistan Accessed: 2024-08-17.
[62] Amama Mahmood, Shiye Cao, Maia Stiber, Victor Nikhil Antony, and Chien-
Ming Huang. 2025. Voice Assistants for Health Self-Management: Designing
for and with Older Adults. InProceedings of the 2025 CHI Conference on Human
Factors in Computing Systems (CHI ’25). Association for Computing Machinery,
New York, NY, USA, Article 511, 22 pages. doi:10.1145/3706598.3713839
[63] Judy Michael, Qaiser Iqbal, Sajjad Haider, Adnan Khalid, Naheed Haque, Rabia
Ishaq, Fahad Saleem, Mohamed Azmi Hassali, and Mohammad Bashaar. 2020.
Knowledge and practice of adolescent females about menstruation and menstru-
ation hygiene visiting a public healthcare institute of Quetta, Pakistan.BMC
women’s health20 (2020), 1–8.
[64] Ministry of Federal Education and Professional Training, Pakistan.
2017. Draft National Education Policy 2017. Government of Pak-
istan. https://www.mofept.gov.pk/SiteImage/Policy/Draft%20National%
20Educaiton%20Policy%202017.pdf.
[65] Ministry of Human Resource Development (Government of India). 2007.Reach-
ing The Age of Adolescence. National Council of Educational Research and
Training (NCERT). 113–126 pages.
[66] Wen Mo, Aneesha Singh, and Catherine Holloway. 2024. From Information
Seeking to Empowerment: Using Large Language Model Chatbot in Supporting
Wheelchair Life in Low Resource Settings. InProceedings of the 26th International
ACM SIGACCESS Conference on Computers and Accessibility(St. John’s, NL,
Canada)(ASSETS ’24). Association for Computing Machinery, New York, NY,
USA, Article 21, 18 pages. doi:10.1145/3663548.3675609
[67] Shafaq Fatima Mughal, Shayan Aamir, Abdul Samad, Umema Zehra, and
Ayesha Asif Syed. 2025. Mai: A Transformer Based Domain Specific Chat-
bot for Menstrual Health.ACM Journal on Responsible Computing2, 1 (2025),
1–32.
[68] Manideepa Mukherjee, Abhay Sheel Anand, Varnika Kairon, and Aman Parnami.
2023. MenstruWear: In-The-Wild Study For Characterizing The Effect Of User
Activities On Sanitary Napkin-Based Bodily Fluid Monitoring. InProceedings
of the 15th Biannual Conference of the Italian SIGCHI Chapter(Torino, Italy)
(CHItaly ’23). Association for Computing Machinery, New York, NY, USA, Article
7, 12 pages. doi:10.1145/3605390.3605404
[69] Zubia Mumtaz, Priya Sivananthajothy, Afshan Bhatti, and Marni Sommer. 2019.
“How can we leave the traditions of our Baab Daada”: socio-cultural structures
and values driving menstrual hygiene management challenges in schools in
Pakistan.Journal of Adolescence76 (2019), 152–161.
[70] Michelle Murphy. 2015. Unsettling care: Troubling transnational itineraries of
care in feminist health practices.Social studies of science45, 5 (2015), 717–737.
[71] Maryam Mustafa, Abdul Moeed Asad, Shehrbano Hassan, Urooj Haider, Zainab
Durrani, and Katharina Krombholz. 2023. Pakistani Teens and Privacy - How
Gender Disparities, Religion and Family Values Impact the Privacy Design
Space. InProceedings of the 2023 ACM SIGSAC Conference on Computer and
Communications Security(Copenhagen, Denmark)(CCS ’23). Association for
Computing Machinery, New York, NY, USA, 195–209. doi:10.1145/3576915.
3623087
[72] Maryam Mustafa, Shaimaa Lazem, Ebtisam Alabdulqader, Kentaro Toyama,
Sharifa Sultana, Samia Ibtasam, Richard Anderson, and Syed Ishtiaque Ahmed.
2020. IslamicHCI: Designing with and within Muslim Populations. InExtended
Abstracts of the 2020 CHI Conference on Human Factors in Computing Systems
(Honolulu, HI, USA)(CHI EA ’20). Association for Computing Machinery, New
York, NY, USA, 1–8. doi:10.1145/3334480.3375151
[73] Maryam Mustafa, Kimia Tuz Zaman, Tallal Ahmad, Amna Batool, Masitah Ghaz-
ali, and Nova Ahmed. 2021. Religion and Women’s Intimate Health: Towards
an Inclusive Approach to Healthcare. InProceedings of the 2021 CHI Conference
on Human Factors in Computing Systems(Yokohama, Japan)(CHI ’21). Asso-
ciation for Computing Machinery, New York, NY, USA, Article 232, 13 pages.
doi:10.1145/3411764.3445605
[74] Amina Nadeem, Maryam Khalid Cheema, and Sheharyar Zameer. 2021. Percep-
tions of Muslim parents and teachers towards sex education in Pakistan.Sex
Education21, 1 (2021), 106–118. doi:10.1080/14681811.2020.1753032
[75] Mustafa Naseem, Fouzia Younas, and Maryam Mustafa. 2020. Designing Digital
Safe Spaces for Peer Support and Connectivity in Patriarchal Contexts.Proc.
ACM Hum.-Comput. Interact.4, CSCW2, Article 146 (oct 2020), 24 pages. doi:10.
1145/3415217
[76] Meena Natarajan and Tapan Parikh. 2013. Understanding barriers to information
access and disclosure for HIV+ women. InProceedings of the Sixth International
Conference on Information and Communication Technologies and Development:
Full Papers - Volume 1(Cape Town, South Africa)(ICTD ’13). Association for
Computing Machinery, New York, NY, USA, 143–152. doi:10.1145/2516604.2516627
[77] Patrick Nichols, Bobbie-Jo Webb-Robertson, and Christopher Oehmen. 2012.
Binary Classification with Support Vector Machines. Cambridge University Press,
157–179.
[78] OpenAI. 2023.Knowledge in GPTs. https://help.openai.com/en/articles/8843948-
knowledge-in-gpts OpenAI Help Center Article.
[79] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, et al .2024.
GPT-4 Technical Report. arXiv:2303.08774 [cs.CL] https://arxiv.org/abs/2303.
08774
[80] Alexandria Alisa Proff, Surhan Fatima, and Mónica Lorena Sánchez Limón. 2023.
Becoming women: period. Perceptions of barriers and facilitators to menstrual
hygiene management programs for Pakistani girls.Frontiers in Public Health11
(2023), 1083688.
[81] Rifat Rahman, Md. Rishadur Rahman, Nafis Irtiza Tripto, Mohammed Eunus Ali,
Sajid Hasan Apon, and Rifat Shahriyar. 2021. AdolescentBot: Understanding
Opportunities for Chatbots in Combating Adolescent Sexual and Reproductive
Health Problems in Bangladesh. InProceedings of the 2021 CHI Conference on
Human Factors in Computing Systems(Yokohama, Japan)(CHI ’21). Association
for Computing Machinery, New York, NY, USA, Article 710, 15 pages. doi:10.
1145/3411764.3445694
[82] Pragnya Ramjee, Bhuvan Sachdeva, Satvik Golechha, Shreyas Kulkarni, et al .
2025. CataractBot: An LLM-powered Expert-in-the-Loop Chatbot for Cataract
Patients.Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous
Technologies9, 2 (June 2025), 1–31. doi:10.1145/3729479
[83] Dr. Maroof Bin Rauf. 2021. Sex Education in Pakistan: Concepts, Needs and
Strategies.Pakistan Journal of International Affairs4, 2 (2021).
[84] Muhammad Ahmad Raza, Muhammad Rehan Nazir, Ali Asghar, Abdul Wahid
Mir, and Muhammad Amir. 2025. SWOT Analysis and Urban Renewal Tech-
niques for Baghbanpura, Lahore.Contemporary Journal of Social Science Review
3, 2 (2025), 1–38.
[85] Ken Resnicow, Tom Baranowski, Jasjit S Ahluwalia, and Ronald L Braithwaite.
1999. Cultural sensitivity in public health: defined and demystified.Ethnicity &
disease9, 1 (1999), 10–21.
[86] Naghma Rizvi, Saleema A Gulzar, Tazeen S Ali, Syeda Aleena Fazal, Alishah Aziz
Gulzar, Yasmin Parpio, and Rimsha Hirani. 2024. Menstrual hygiene amongst
school girls: Still a messy business.Journal of Asian Midwives (JAM)11, 1 (2024),
3–14.
[87] Salma Sadique, Inayat Ali, and Shahbaz Ali. 2024. Managing menstruation
during natural disasters: menstruation hygiene management during “super
floods” in Sindh province of Pakistan.Journal of Biosocial Science56, 3 (2024),
480–492. doi:10.1017/S0021932023000159
[88] Nithya Sambasivan, Garen Checkley, Amna Batool, Nova Ahmed, David Nemer,
Laura Sanely Gaytán-Lugo, Tara Matthews, Sunny Consolvo, and Elizabeth
Churchill. 2018. "Privacy is not for me, it’s for those rich women": Performative
Privacy Practices on Mobile Phones by Women in South Asia. InFourteenth
Symposium on Usable Privacy and Security (SOUPS 2018). USENIX Associa-
tion, Baltimore, MD, 127–142. https://www.usenix.org/conference/soups2018/
presentation/sambasivan
[89] Elizabeth B.-N. Sanders and Pieter Jan Stappers. 2008. Co-creation and the new
landscapes of design.Co-Design4, 1 (2008), 5–18.
[90] Clara Schweizer, Mirabelle Böhm, and Radha Paudel. 2023. How Men-
strual Discrimination is Approached by Menstrual Movements in
the Global North and the Global South. Dignified Menstruation.org.
https://www.dignifiedmenstruation.org/detail/130/how-menstrual-
discrimination-is-approached-by-menstrual-movements-in-the-global-
north-and-the-global-south Accessed: 2025-04-08.
[91] Katie Seaborn and Alexa Frank. 2022. What Pronouns for Pepper? A Critical
Review of Gender/ing in Research. InProceedings of the 2022 CHI Conference
on Human Factors in Computing Systems(New Orleans, LA, USA)(CHI ’22).
Association for Computing Machinery, New York, NY, USA, Article 239, 15 pages.
doi:10.1145/3491102.3501996
[92] Lennart Seitz and Sigrid Bekmeier-Feuerhahn. 2021. Empathic Healthcare
Chatbots: Comparing the Effects of Emotional Expression and Caring Behavior.
InProceedings of the International Conference on Information Systems (ICIS).
[93] Asim Shaikh and Rohan Kumar Ochani. 2018. The Need for Sexuality Education
in Pakistan.Cureus10, 5 (May 2018), e2693. doi:10.7759/cureus.2693
[94] Weiyan Shi, Ryan Li, Yutong Zhang, Caleb Ziems, Sunny Yu, Raya Horesh,
Rogério Abreu De Paula, and Diyi Yang. 2024. CultureBank: An Online
Community-Driven Knowledge Base Towards Culturally Aware Language
Technologies. InFindings of the Association for Computational Linguistics:
EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.).
Association for Computational Linguistics, Miami, Florida, USA, 4996–5025.
doi:10.18653/v1/2024.findings-emnlp.288
[95] Jesper Simonsen and Toni Robertson (Eds.). 2013.Routledge International Hand-
book of Participatory Design. Vol. 711. Routledge, New York.
[96] Marie Louise Juul Søndergaard, Marianela Ciolfi Felice, and Madeline Balaam.
2021. Designing Menstrual Technologies with Adolescents. InProceedings of
the 2021 CHI Conference on Human Factors in Computing Systems(Yokohama,

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Japan)(CHI ’21). Association for Computing Machinery, New York, NY, USA,
Article 260, 14 pages. doi:10.1145/3411764.3445471
[97] Piya Sorcar, Benjamin Strauber, Prashant Loyalka, Neha Kumar, and Shelley
Goldman. 2017. Sidestepping the Elephant in the Classroom: Using Culturally
Localized Technology To Teach Around Taboos. InProceedings of the 2017 CHI
Conference on Human Factors in Computing Systems(Denver, Colorado, USA)
(CHI ’17). Association for Computing Machinery, New York, NY, USA, 2792–2804.
doi:10.1145/3025453.3025958
[98] Davinny Sou, Debs Stebler, Marinja Principe, Tobias Kowatsch, and Marcia
Nißen. 2024. Please CyMe: Towards Enhancing Menstrual Health Awareness
through Customization and Personalized Visualization in a Menstrual Health
App. InProceedings of Mensch Und Computer 2024(Karlsruhe, Germany)(MuC
’24). Association for Computing Machinery, New York, NY, USA, 674–683. doi:10.
1145/3670653.3677475
[99] Clay Spinuzzi. 2005. The methodology of participatory design.Technical
Communication52, 2 (May 2005), 163–174.
[100] Jeff Stanley, Ronna Ten Brink, Alexandra Valiton, Trevor Bostic, and Rebecca
Scollan. 2021. Chatbot Accessibility Guidance: A Review and Way Forward. In
Proceedings of the Sixth International Congress on Information and Communication
Technology (ICICT 2021), London, Volume 3. Springer Singapore, Singapore, 919–
942. doi:10.1007/978-981-16-1781-2_80
[101] Sharifa Sultana, François Guimbretière, Phoebe Sengers, and Nicola Dell. 2018.
Design Within a Patriarchal Society: Opportunities and Challenges in Designing
for Rural Women in Bangladesh. InProceedings of the 2018 CHI Conference
on Human Factors in Computing Systems(Montreal QC, Canada)(CHI ’18).
Association for Computing Machinery, New York, NY, USA, 1–13. doi:10.1145/
3173574.3174110
[102] Xiao Tang, Zhuying Li, Xin Sun, Xuhai Xu, and Min-Ling Zhang. 2025. ZzzMate:
A Self-Conscious Emotion-Aware Chatbot for Sleep Intervention. InProceedings
of the Extended Abstracts of the CHI Conference on Human Factors in Computing
Systems (CHI EA ’25). Association for Computing Machinery, New York, NY,
USA, Article 610, 7 pages. doi:10.1145/3706599.3720224
[103] Bonnie Tran and Lee Na Choi. 2018. Menstrual Maze: A Toy Exploring Public
Engagement in Menstrual Health Education. InExtended Abstracts of the 2018
CHI Conference on Human Factors in Computing Systems(Montreal QC, Canada)
(CHI EA ’18). Association for Computing Machinery, New York, NY, USA, 1–6.
doi:10.1145/3170427.3180649
[104] Anupriya Tuli, Shaan Chopra, Neha Kumar, and Pushpendra Singh. 2018. Learn-
ing from and with menstrupedia: Towards menstrual health education in India.
Proceedings of the ACM on Human-Computer Interaction2, CSCW (2018), 1–20.
[105] Anupriya Tuli, Shaan Chopra, Pushpendra Singh, and Neha Kumar. 2020.
Menstrual (Im)Mobilities and Safe Spaces. InProceedings of the 2020 CHI
Conference on Human Factors in Computing Systems(Honolulu, HI, USA)
(CHI ’20). Association for Computing Machinery, New York, NY, USA, 1–15.
doi:10.1145/3313831.3376653
[106] Anupriya Tuli, Surbhi Singh, Rikita Narula, Neha Kumar, and Pushpendra Singh.
2022. Rethinking Menstrual Trackers Towards Period-Positive Ecologies. In
Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems
(New Orleans, LA, USA)(CHI ’22). Association for Computing Machinery, New
York, NY, USA, Article 283, 20 pages. doi:10.1145/3491102.3517662
[107] Priyan Vaithilingam, Tianyi Zhang, and Elena L. Glassman. 2022. Expectation
vs. Experience: Evaluating the Usability of Code Generation Tools Powered by
Large Language Models. InExtended Abstracts of the 2022 CHI Conference on
Human Factors in Computing Systems(New Orleans, LA, USA)(CHI EA ’22).
Association for Computing Machinery, New York, NY, USA, Article 332, 7 pages.
doi:10.1145/3491101.3519665
[108] Belen Hernandez Villalba and Nadia Jimenez Barriga. 2024. CycleWiseT: Promot-
ing Menstrual Health to Transform the Lives of Women and Girls Through Tech-
nology. InProceedings of the XI Latin American Conference on Human Computer
Interaction(Puebla, Mexico)(CLIHC ’23). Association for Computing Machinery,
New York, NY, USA, Article 32, 5 pages. doi:10.1145/3630970.3631072
[109] Hua Wang, Sneha Gupta, Arvind Singhal, Poonam Muttreja, Sanghamitra Singh,
Poorva Sharma, and Alice Piterova. 2022. An artificial intelligence chatbot for
young people’s sexual and reproductive health in India (SnehAI): instrumental
case study.Journal of Medical Internet Research24, 1 (2022), e29969.
[110] Joseph Weizenbaum. 1983. ELIZA — a computer program for the study of
natural language communication between man and machine.Commun. ACM
26, 1 (Jan. 1983), 23–28. doi:10.1145/357980.357991
[111] Isabella Catharina Wiest, Mamatha Bhat, Jan Clusmann, Carolin V Schneider,
Xiaofeng Jiang, and Jakob Nikolas Kather. 2025. Large language models for
clinical decision support in gastroenterology and hepatology.Nature Reviews
Gastroenterology & Hepatology(2025), 1–15.
[112] World Bank. 2024. Literacy rate, adult female (% of females ages 15 and above) -
Pakistan. https://data.worldbank.org/indicator/SE.ADT.LITR.FE.ZS?locations=
PK
[113] Deepika Yadav, Kirti Dabas, Prerna Malik, Anushka Bhandari, and Pushpendra
Singh. 2022. “Should I visit the clinic”: Analyzing WhatsApp-mediated Online
Health Support for Expectant and New Mothers in Rural India. InProceedings ofthe 2022 CHI Conference on Human Factors in Computing Systems(New Orleans,
LA, USA)(CHI ’22). Association for Computing Machinery, New York, NY, USA,
Article 296, 20 pages. doi:10.1145/3491102.3517575
[114] Deepika Yadav, Prerna Malik, Kirti Dabas, and Pushpendra Singh. 2019. Feed-
pal: Understanding Opportunities for Chatbots in Breastfeeding Education of
Women in India.Proc. ACM Hum.-Comput. Interact.3, CSCW, Article 170 (Nov.
2019), 30 pages. doi:10.1145/3359272
A Appendix
A.1 Post-deployment Interview Protocol
(1) Demographic Information
(a) What is your name?
(b) What is your age?
(c) What is your educational background?
(d) Do you have a personal phone, a shared phone, or both?
•If shared, who do you share it with?
(2) Menstrual Health Information Seeking
(a) When was your last visit to the doctor?
•Have you ever consulted a hakeem or homeopath?
•What was your experience with them?
(b) Where do you usually get information about periods?
(c)To what extent do you use digital or technological sources
for information about periods?
(d)How do you decide which source to get information from?
(e) How much do you trust each of these sources?
•How do you validate the information you receive, if at
all?
(f)Do you rely on your own experiences for information
about your periods?
•If so, to what extent?
(g)How often do you discuss your periods with friends or
family, if at all?
(h) Are there any specific sources you avoid? Why?
•(Prompt: Ask about any negative experiences in the
past)
(i)What usually prompts you to seek information about pe-
riods?
•What sort of queries do you typically focus on (e.g., pain,
irregular cycles, diet, medication)?
(j) How do you separate fact from myth regarding periods?
•For example, have you ever heard any myths (suni
sunayi baatain)?
(k)Do you feel that your religion or culture influences your
information-seeking behavior?
•If yes, how?
(l)Are there specific topics (e.g., medication, diet, exer-
cise) you prefer getting information about from specific
sources?
•Please provide examples.
(m)Have you faced any challenges in finding satisfactory (easy
to understand, accurate) information about periods?
(3) Menstrual Health Chatbot
(a) What were your initial expectations of the chatbot?
(b)How did you engage with the chatbot (e.g., frequency,
types of questions asked)?
(c)How satisfied were you with the responses provided by
the chatbot?

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
(d) Did you trust the responses given by the chatbot?
•Why or why not?
(e) Did you verify the responses provided by the chatbot?
•If yes, how?
(f)Did you feel any of the responses were biased or problem-
atic in any way?
•For example, was there anything said that you didn’t
like, or did you have any issues with how the informa-
tion was presented?
(g)Were there any specific areas where you felt the chatbot
performed well?
(h)Were there any areas where you felt the chatbot could
improve?
(i)How did your experience with the chatbot compare to
other sources of information about periods?
(j)Based on this experience, would you be interested in using
chatbots for other types of health education?
(k) How do you manage your privacy on your phone?
•For example, do you use a password or other methods?
(l)How did you manage your privacy while using the chat-
bot?
•For example, Did you delete the chats or use any other
strategies?
A.2 Themes in Menstrual Health FAQs
A.3 List of Menstrual Health Questions and
Myths
(1) Why is there so much pain during periods?
(2) Why does the body feel lifeless during periods?
(3) Why is there anger or irritability during periods?
(4) Does taking a shower during periods reduce blood flow?
(5)Does wearing black underwear during periods radiate heat?
(6)Which pads should be worn during periods: cotton or Al-
ways?
(7) Do menstrual cups get lost inside you?
(8) Are cloth or cotton pads better?
(9) Should medicine be taken if there is pain?
(10) Is excessive bleeding a serious issue during periods?
(11) Is it a concern if periods last more than 7 days?
(12) Is it a problem if periods are late from the expected date?
(13) Does bathing during periods reduce blood flow?
(14) Why do period cramps occur? What should be avoided to
prevent cramps?
(15) How long does normal period flow last?
(16) Which pads are good for the environment? I’ve heard syn-
thetic pads are not good.
(17) Should heavy work be avoided during periods? Is that cor-
rect?
(18) Should normal work be continued during periods? Is more
exercise beneficial?
(19) Do pores open during periods, affecting circulation?
(20) Why should henna not be applied during periods?
(21) I’ve heard that taking medicine during periods harms and
causes issues in married life. Is that true?
(22) What should be done if there is pain during periods?
(23) What are the benefits and advantages of using pads?(24) I’ve heard that using pads causes issues in married life. Is
that true?
(25) Which company’s pads are good and should be used?
(26) I’ve heard that using cloth during periods causes rashes. Is
that true?
(27) I’ve heard that if you listen to someone’s period pain, it
transfers to you. Is that true?
(28) Why do periods occur?
(29) How do people endure pain during periods?
(30) Does using a hot water bottle provide relief during periods?
(31) Should cold water or substances be taken during periods?
(32) When and why do periods occur?
(33) I’ve heard that you shouldn’t bathe during periods because
it increases pain. Is that true?
(34) Do periods always have the same date throughout life?
(35) Do certain foods make your first period come sooner?
(36) Does the date of periods change every 3 months?
(37) Should one not wear black clothes during periods because it
is the color of Kaaba?
(38) If I don’t have menstrual pads, can I use a cloth as a replace-
ment?
(39) When should I visit a doctor for period cramps and to what
extent is the pain normal?
(40) Does the consumption of cold drinks or an unhealthy diet
affect menstrual pain? Which foods should I avoid during
periods?
(41) What are some natural remedies for menstrual pain relief?
(42) Some people say that you should not take a bath during the
first three days of periods, is it true?
(43) Can I still do home chores and exercise during my period?
(44) Is it a matter of concern to not have irregular periods?
(45) Is it dangerous to have lighter flow during periods? Does the
unclean blood stay inside?
(46) One should keep water contact minimal during periods. Is
that true?
(47) Is it advisable to only take medication once throughout the
entirety of your menstrual cycle, or can it be taken more
often, such as twice or even daily?
A.4 Example Themes and Codes from
Interviews
A.5 Pilot Deployment and System Refinements
The pilot phase served as an initial field test to surface challenges
in both technical functionality and the accuracy of LLM-generated
responses. During one week pilot with seven participants (130
messages), no hallucinations or harmful responses were identified.
However, users noted response delays and the lack of confirmation
cues, which left them uncertain whether their messages were re-
ceived or processed. Based on user feedback, we reconfigured our
backend server to reduce the delays, added feedback mechanism,
and refined the LLM module as follows:
•Upgraded Health Assistant.After the pilot, we transitioned
to the OpenAI Assistant API V2, which offered finer control
over the RAG. We set chunk sizes so each one matched
a single gynecologist-validated question-answers, which
kept each question-answer pair separate and made retrieval

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Figure 4: Menstrual Health FAQs gathered from gynecologist and co-design workshops
Table 5: Example themes and codes from the analysis of interviews
Themes Codes (Illustrative subset) Example participant quotes
Navigating layered infor-
mation sourcesMother as primary trusted source “I mainly ask my mom. . . I will only trust the information if she says it’s
correct. ”(C13)
Cross-checking online information
with family or doctors“Whatever YouTube tells you, you have to confirm it first. You never know
what’s being told is actually true. I always cross-check whatever they say
in a YouTube video from different sources. ”(C11)
Reworking Menstrual
Norms, Religion, and
SecrecyNavigating myths through search
and experience“Either I go to my mother or I directly go to Dr. Bilquees [YouTube Chan-
nel]...I have never had a bad experience with Dr. Bilquees so I never asked
anyone else. ”(C1)
Uncertainty around religious prac-
tices“They say that you should not touch a prayer mat when you’re on your
period. But I don’t know about this, whether it’s actually true or not. ”(C3)
Secrecy around menstruation “Mom used to ask me not to tell anyone outside, just tell me. But as I grew
older, I got more information about periods, and the embarrassment kept
decreasing. ”(C9)
Chatbot as confidential, re-
sponsive companionImmediate responses “ I thought that I’d get my reply after a while. But I kept getting replies
immediately. I liked it. Reply was given quickly. ”(C1)
No hesitation using chatbot ““When I asked it [chatbot], I didn’t have any hesitation. I asked all the
questions easily and I got the answers easily. That’s why I didn’t have
to ask anyone else. You can’t ask your family and doctor about certain
things. ”(C8)
cleaner. We then varied the number of retrieved chunks (top-
k) to balance answer accuracy against latency, selecting a
small k that reliably surfaced the right answer without over-
loading the prompt. Using both embedding-based similarity
and keyword matching improved recall on short, colloquial
queries (e.g., misspellings, Roman-Urdu variants) while still
catching exact medical terms; this combination increased
the chance that the most relevant question-answer chunkappeared in the top results. In addition, we selected the GPT-
4o model, which provided stronger language comprehension
and faster responses. We refined the assistant’s instructions
to emphasize menstrual and female health, cultural sensitiv-
ity, and concise delivery.
•Enhanced Knowledge Base.We rephrased each question
in the knowledge base multiple times, both in English and
Roman Urdu This is particularly useful in the context of RAG
because it increases the likelihood that the retriever model

CHI ’26, April 13–17, 2026, Barcelona, Spain Shahnawaz et al.
can match the user’s query with the most relevant document
chunks regardless of how the query is formulated.
•Read Receipts.We added WhatsApp read receipts to reas-
sure users that their queries had been received and were
being processed, aiming to reduce confusion.
•Prompt InclusionAlong with the earlier system instruc-
tions, we appended a well-defined prompt to each user query
to guide the LLM in generating responses. This prompt acted
as a condensed version of the assistant’s instructions, rein-
forcing key guidelines on scope, cultural sensitivity, and
brevity. Prior work shows that repeating instructions canhelp LLMs stay focused on critical aspects of a task. By reit-
erating this guidance within the prompt, we aimed to ensure
that the model’s outputs were more aligned with our design
goals.
A.6 Linguistic Variations Used in Participant
Interactions
A.7 Sample Screenshots of Chatbot
Coversations

Designing Around Stigma: Human-Centered LLMs for Menstrual Health CHI ’26, April 13–17, 2026, Barcelona, Spain
Figure 5: An example illustrating multiple variations of the same sentence
Figure 6: Left: Participant asking a question from Chatbot in Roman Urdu. Participant: "Is it okay if my period always starts
on the same date each month?" Chatbot: "Yes, if your periods always start on the same date each month, this is normal and
indicates that your menstrual cycle is regular. A regular cycle is a sign of good health. However, if the date occasionally changes
(1-2 days difference), it is still normal. If there is significant irregularity in the cycle, it is better to consult a doctor."
Right: Participant discussing concerns about using pads with the Chatbot in English.
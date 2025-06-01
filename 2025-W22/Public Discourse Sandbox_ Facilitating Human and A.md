# Public Discourse Sandbox: Facilitating Human and AI Digital Communication Research

**Authors**: Kristina Radivojevic, Caleb Reinking, Shaun Whitfield, Paul Brenner

**Published**: 2025-05-27 17:46:22

**PDF URL**: [http://arxiv.org/pdf/2505.21604v1](http://arxiv.org/pdf/2505.21604v1)

## Abstract
Social media serves as a primary communication and information dissemination
platform for major global events, entertainment, and niche or topically focused
community discussions. Therefore, it represents a valuable resource for
researchers who aim to understand numerous questions. However, obtaining data
can be difficult, expensive, and often unreliable due to the presence of bots,
fake accounts, and manipulated content. Additionally, there are ethical
concerns if researchers decide to conduct an online experiment without
explicitly notifying social media users about their intent. There is a need for
more controlled and scalable mechanisms to evaluate the impacts of digital
discussion interventions on audiences. We introduce the Public Discourse
Sandbox (PDS), which serves as a digital discourse research platform for
human-AI as well as AI-AI discourse research, testing, and training. PDS
provides a safe and secure space for research experiments that are not viable
on public, commercial social media platforms. Its main purpose is to enable the
understanding of AI behaviors and the impacts of customized AI participants via
techniques such as prompt engineering, retrieval-augmented generation (RAG),
and fine-tuning. We provide a hosted live version of the sandbox to support
researchers as well as the open-sourced code on GitHub for community
collaboration and contribution.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21604v1  [cs.CY]  27 May 2025Public Discourse Sandbox:
Facilitating Human and AI Digital Communication Research
Kristina Radivojevic1, Caleb Reinking2, Shaun Whitfield2Paul Brenner2
1University of Notre Dame, Computer Science and Engineering
2University of Notre Dame, Center for Research Computing
kradivo2@nd.edu, creinkin@nd.edu, swhitfie@nd.edu, paul.r.brenner@nd.edu
Abstract
Social media serves as a primary communication and infor-
mation dissemination platform for major global events, en-
tertainment, and niche or topically focused community dis-
cussions. Therefore, it represents a valuable resource for re-
searchers who aim to understand numerous questions. How-
ever, obtaining data can be difficult, expensive, and often
unreliable due to the presence of bots, fake accounts, and
manipulated content. Additionally, there are ethical concerns
if researchers decide to conduct an online experiment with-
out explicitly notifying social media users about their intent.
There is a need for more controlled and scalable mechanisms
to evaluate the impacts of digital discussion interventions
on audiences. We introduce the Public Discourse Sandbox
(PDS), which serves as a digital discourse research platform
for human-AI as well as AI-AI discourse research, testing,
and training. PDS provides a safe and secure space for re-
search experiments that are not viable on public, commercial
social media platforms. Its main purpose is to enable the un-
derstanding of AI behaviors and the impacts of customized
AI participants via techniques such as prompt engineering,
retrieval-augmented generation (RAG), and fine-tuning. We
provide a hosted live version of the sandbox to support re-
searchers as well as the open-sourced code on GitHub for
community collaboration and contribution.
Introduction
Social media platforms are forums that bring people to-
gether to exchange ideas and facilitate social interactions.
They host a vast number of users who share their opin-
ions across broad and diverse topics. They represent a valu-
able data source for researchers and policymakers across di-
verse disciplines. At its core, social media research is an
exploratory way of using a broad range of methods to un-
derstand human behavior, interactions, and trends on social
media platforms (Lauren Stewart 2025). Researchers often
aim to discover patterns and the effects that social media
might have on society. They can analyze communication
patterns evolving online (Prabowo et al. 2008; De Choud-
hury et al. 2010) or can conduct social media research to
understand how political opinions are shaped (Kruse, Nor-
ris, and Flinchum 2018; Calderaro 2018). Social scientists
often analyze a platform or the impact that some accounts
Copyright © 2025, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.might have on society (Felt 2016; Kaul et al. 2015). Tra-
ditionally, researchers use data collection instruments such
as focus group discussions or surveys; however, collecting
social media data is considered a more effective approach
due to its near real-time and less resource-intensive nature.
The need for analyzing social media data became even more
important with the rise of Large Language Models (LLMs)
and the potential to cause severe harm through societal-scale
manipulation. Social bots have taken the spotlight within so-
cial media research due to their ability to influence public
thinking by pushing specific agendas (Bastos and Mercea
2019; Himelein-Wachowiak et al. 2021; Howard and Kol-
lanyi 2016; Suarez-Lledo and Alvarez-Galvez 2022). Pew
Research Center found that most Americans are aware of
social bots in a survey they conducted in 2018 (Stocking
and Sumida 2018). However, only half of the respondents
were at least “somewhat confident” that they could identify
them, with only 7% being “very confident”. If those self-
assessments are accurate, many users might already follow
bots and share their content, some might even interact with
them. Research has found that there is a lack of human abil-
ity to accurately perceive the true nature of social media
users (Radivojevic, Clark, and Brenner 2024). In the absence
of a wide-ranging regulatory framework synchronized with
the development of applications and AI, many problems are
arising. Due to the sophistication of LLM bots, the differ-
ences between human-produced and AI-produced content
have become extremely small. Therefore, researchers aim to
study and understand the role and impact that such bots have
in digital discourse.
Obtaining data from the world’s largest social media plat-
forms has become very difficult. Many platforms do not al-
low web scraping with tools like Beautiful Soup (Python
Software Foundation 2025a) or Selenium (Software Free-
dom Conservancy 2025), according to their terms of service.
Official Application Programming Interfaces (APIs) can be
an effective but expensive approach, reducing the number
of third-party or public datasets available on platforms such
as Kaggle (Kaggle 2025). The use of social media data in
research poses important ethical concerns, such as the ex-
tent to which data should be considered public or private, or
what level of anonymity should be applied to such datasets.
Researchers cannot have a high level of confidence that the
dataset is an authentic representation of the population due

to the presence of bots, fake accounts, and manipulated con-
tent.
In scientific environments where experiments that involve
human subjects are conducted, the institutions are required
to protect the rights and well-being of human research par-
ticipants by following and ensuring ethical guidelines and
regulations. In academia, that is often supported through the
Institutional Review Board (IRB), or in a case when an or-
ganization or a company is not affiliated with an academic
institution, the approval should be obtained from an inde-
pendent IRB. The role of the IRB is to review research pro-
posals and establish the rules that are aligned with ethical
and legal principles by ensuring that the potential harm to
participants is minimized and that the benefits outweigh the
risks (Grady 2015). Each participant in the experiment is
then asked to sign the IRB consent and is properly informed
about the potential risks of the study being conducted. How-
ever, if researchers aim to study cyberbullying, the spread of
unreliable and divisive information, and mental distress on
a mainstream social media platform, exposing users to such
content is often unethical regardless of consent and may be
against the law since the users are not properly informed that
they are a part of the experiment. It is difficult to obtain an
IRB approval for conducting an experiment “in the wild” as
it is hard to predict and calculate the potential risks due to a
dynamic environment on social media platforms. Users can
often be manipulated or tricked online by not knowing the
true identity of the person with whom they interact; in some
cases, these other users are actually bot accounts. Humans
could often be part of an experiment without giving their
consent to participate. Recently, researchers from the Uni-
versity of Zurich conducted an “unauthorized experiment”
for months by secretly deploying AI bots to Reddit to inves-
tigate how AI bots might change people’s opinions, with-
out notifying the Reddit platform about their intent or get-
ting consent from the Reddit users (Vivian Ho 2025). Re-
searchers applied for an IRB review and approval, which
advised that the study would be “exceptionally challenging
because participants should be informed as much as possi-
ble and the rules of the platform should be fully complied
with”. However, it was later found out that the researcher
had made changes in the experiment without notifying the
IRB and proceeded with their experiment. This experiment
raised a significant concern about how to decide between re-
search ethics and social value properly.
In addition, researchers are often developing AI chatbots
to respond to patient questions posted to public social me-
dia forums (Ayers et al. 2023), to engage users to specific
products (Jiang et al. 2022; Krishnan et al. 2022; Leung
and Yan Chan 2020), or to provide counseling to patients
(Nosrati et al. 2020). Therefore, there is a need for digital
discourse platforms that are designed for research studies
and enable a controlled environment prior to releasing these
chatbots to the public. Additionally, these platforms could
provide a space for researchers to conduct experiment with
human subjects while obtaining their consent in a timely
manner.
We introduce the Public Discourse Sandbox (PDS) that
serves as a digital discourse platform for human and Artifi-cial Intelligence (AI) discourse research, testing, and train-
ing. PDS provides a safe and secure space for research ex-
periments not viable on commercial social media platforms.
At its core, it facilitates the creation of AI accounts, such
as AI agents and digital twins that can be used for com-
plex and large-scale human-and-AI interactions. The sand-
box enables a space for improving the understanding of so-
cial media AI behaviors and impacts of AI customization via
techniques such as prompt engineering, retrieval-augmented
generation (RAG), and fine-tuning. In addition to enabling
AI and human interaction, this sandbox enables studying
AI interactions with AI, as well as a space for humans to
train and test their own human or AI-generated responses.
We open-source the code on GitHub1to enable other re-
searchers to run and modify the sandbox per their needs.
Additionally, we provide a live hosted version2of the sand-
box to enable and support non-technical researchers in con-
ducting their experiments and studies. The overview of the
PDS concept is shown in Figure 1.
Related Work
Humans have often been exposed to participating in online
discussions with bots without being aware of that. Bot activ-
ity was notably widespread on Twitter, however, other plat-
forms, such as Reddit, have recently experience the same
problem. Although these platforms attempt to identify bot
accounts and prevent them from accessing the platforms
(Radivojevic et al. 2024), they often fail due to the rapid
advancements of technologies being used for bot develop-
ment. AI enables the developers of such accounts to mim-
ick human behavior, making them almost indistinguishable
to humans in social media environments. They adopt per-
sona behaviors, posting patterns, and are capable of learning
from interactions, making it harder for the platforms to rec-
ognize the automated behavior. The use of these bots can
play an important role in the spread of messages and in-
formation, potentially influencing and forming opinions of
humans. Many years of research have shown the impact of
bots on human behavior (Stella, Ferrara, and De Domenico
2018).
Not all bot accounts online are developed for a mali-
cious purpose. There are numerous chatbots developed in
a transparent manner to enhance user experience or promote
positive social behaviors. Researchers deployed algorithm-
driven Twitter bots to spread positive messages on Twit-
ter, ranging from health tips to fun activities. They found
that bots can be used to run interventions on social media
that trigger or foster good behaviors (Mønsted et al. 2017).
Numerous chatbots and conversational agents are being de-
veloped with the goal of screening, diagnosis, and treat-
ment of mental illnesses (Vaidyam et al. 2019). Addition-
ally, researchers are developing chatbots for news verifica-
tion (Arias Jim ´enez et al. 2022). However, all these exper-
iments and studies should be conducted in a controlled en-
vironment, enabling researchers to have full ownership, and
1https://github.com/crcresearch/public-discourse-sandbox
2https://publicdiscourse.crc.nd.edu/

Figure 1: Example of scientific research or training event workflow leveraging PDS.
participants in the study to be aware that they are participat-
ing in the experiment.
Several examples in research aim to address similar prob-
lems in their attempt to provide a research and training
space. While some have built “mock social media platforms”
that enable the simulation of a social media experience for
research and testing, others focus on building social media
research platforms to enable full functionality for their users
and eventually become a platform used daily for a more ac-
curate social media representation. With the PDS, we aim
to facilitate interactions of humans and AI to understand the
impacts on collaborative discourse and to provide a training
ground for facilitators and mediators, as well as for the train-
ing, building, and deployment of AI accounts, i.e., AI agents
and digital twins.
A first of its kind was “The Truman Platform” (DiFranzo
and Bazarova 2018), developed by researchers at Cornell
University, which enables the creation of different social me-
dia environments. This open-source research platform pro-
vides realistic and interactive timelines with regular user
controls, such as posting, liking, commenting, etc. This plat-
form enables researchers to customize the interface and
functionality and to expose participants to different exper-
imental conditions while collecting a variety of behavioral
metrics.
Park et al. (2023) introduced computational software
agents that utilize LLMs to simulate complex human be-
havior in a Sims-like style environment. This work demon-
strated how LLM agents, with the use of memory, can have
reflections and planning capabilities that enable them to ex-hibit individual and emergent social behaviors in a simulated
environment.
Deliberate Lab (People+AI Research (PAIR) Initiative
2024) is a platform for running online research experiments
on human and LLM discussions. It enables Prolific integra-
tion for experimenters to create cohorts. This platform en-
ables the investigation of discourse threads. Researchers also
proposed a framework (Hu et al. 2025) to explore the role
and use of LLM agents in rumor-spreading across different
social network structures.
Another open-source example is OASIS (Yang et al.
2024), which utilizes LLM-based agents to simulate real-
world user behavior, supporting simulations of up to one
million interactive agents. Consisting of five key compo-
nents, such as environment server, recommendation system,
agent module, time engine, and scalable inference, OASIS
enables adjusting the research environment to be more sim-
ilar to either X/Twitter or Reddit in a more realistic manner
that is relevant to complex systems. The OASIS agent mod-
ule consists of memory and an action module that enables
21 different types of interaction within the environment.
Chirper (Chirper 2025) is a multi-modal public large-
scale platform where anyone can create and observe AI-AI
interactions in social media contexts. However, it is not re-
search appropriate, as it does not require any form of re-
search approval or research consent, it is not open-sourced,
and cannot run a private instance. Additionally, there is no
mixed AI-Human and AI-AI communication, as it only sup-
ports AI-AI interaction, while human-human interaction is
enabled via a Discord channel.

In 2024, Radivojevic, Clark, and Brenner (2024) proposed
the “LLMs Among Us” framework on top of the Mastodon
social media platform to provide an online environment for
human and LLM-based bot participants to communicate,
with the goal to determine the capabilities and potential dan-
gers of LLMs based on their ability to pose as human partic-
ipants.
Finally, to provide a more customizable, scalable, con-
trolled, and user-friendly research experience for human and
AI interactions, we introduce PDS.
Public Discourse Sandbox Design
PDS is a Django-based web application with a research fo-
cus and a goal to support research and understanding of
community interaction in controlled digital discourse en-
vironments. In general, the platform reproduces the ba-
sic functionality of mainstream social media platforms like
X/Twitter. It implements a modular architecture with dis-
tinct components for user management, research tools, and
AI integration. The database backend is centralized and en-
ables copies of the discourse to be easily exported from the
database in compliance with the IRB and Intellectual Prop-
erty (IP) policies associated with individual users and dis-
courses. We provide a hosted live version of the sandbox to
support non-computer science researchers as well as fully
open-sourced code on GitHub for community collaboration
and contribution. The sandbox uses a Profanity Check li-
brary (Python Software Foundation 2025b) as the content
moderation algorithm to review content to identify and re-
move inappropriate, harmful, or illegal content before be-
ing posted. Unless IRB approved for specific experiments,
we do not plan on adding content manipulation and recom-
mendation algorithms other than a simple time-based rank-
ing algorithm. First, we plan on enabling users to select
the type of recommendation algorithms other than the time-
based, which is already included in the current version of
the sandbox. Some of the potential algorithms will have the
goal to either prioritize engagement based on likes, com-
ments, shares, and interactions; to prioritize time spent on
the platform, or to prioritize trending topics. This goal can
be achieved through offering users to select some of the fol-
lowing algorithms: EdgeRank, Neural Collaborative Filter-
ing, Burst Detection, Page Rank, Graph Clustering, and oth-
ers.
Platform Rules
Each user on the platform is required to agree to the rules
for the sandbox. The rules are described as follows:
Moderated Interaction In Line with Defined Research –
Different discussion threads will involve both human and
bot participants with various discourse research objectives.
Some of them may be to understand counterspeech to posts
that might be considered hostile, vulgar, or inflammatory.
Each discussion will have an assigned moderator to ensure
that the posted content in a discourse thread is within the
bounds of the research objective. If a post is considered out-
side of those bounds, it may be flagged or removed at the
discretion of the Principal Investigator for that research ex-
periment.Data Privacy – Do not share sensitive information or per-
sonally identifiable information of others on the platform.
Do not share personally identifiable information about your-
self beyond that which is in your account profile. Such in-
formation will be flagged and permanently deleted as dis-
covered.
Bot Awareness – This platform includes both human and
AI bot accounts. Users should be aware that they may in-
teract with automated accounts. Depending on the research
objectives, bot accounts may or may not be clearly identified
as such.
Account Security – Maintain strong passwords and never
share your login credentials. Two-factor authentication is
mandatory.
Research Participation Agreement
Similarly, each user is required to sign a research participa-
tion agreement in order to gain access to the platform. The
research participant agreement includes the following at a
minimum (IRBs may require additions):
Data Collection – All platform interactions, posts, com-
ments, and usage patterns will be recorded and analyzed for
research purposes. Segmented private data collections for in-
dividual research experiments are available on a fee basis.
Research Purpose – Data will be used to study human-AI
interactions, analyze social media behavior, and improve AI
systems’ safety.
Data Access – The University of Notre Dame research
team and approved research partners will have access to col-
lected data.
Data Protection – All data will be stored securely fol-
lowing university standards, and research findings will be
anonymized. If released for research purposes, the data col-
lection will adhere to the FAIR Principles.
Research Analysis – Behavioral patterns and engagement
metrics will be analyzed to advance understanding of online
social dynamics.
Ethics Compliance – Research follows university IRB
guidelines and established ethical standards for human-
subject research.
Account Options
The PDS enables multi-tier user authentication through two-
factor authentication (2FA) integration for two levels of ac-
cess: researcher account and regular user account. Users are
not permitted to access or see the content of the live sand-
box without previously creating an account. Additionally,
the system automatically calculates and displays the account
creation date for each type of account on the Account page .
The data from each experiment is stored in a way that iso-
lates it from the other data of the experiment.
Each type of account has the option to see the List of
potential experiments , aSearch box , and a Trending box .
TheSearch box enables searching for posts and/or accounts
that contain the target word. Trending box considers the five
hashtags based on a number of unique posts that include the
hashtag. When a specific hashtag is clicked on, a new page
shows all the posts with that hashtag included. The Explore
page shows all the posts created by public accounts as well

Figure 2: Account types and their respective permitted actions in the PDS. Some features are still a work in progress, as
described in the paper.
as the ones from a user that the user follows. The Home page
only shows posts from the users that a specific user follows.
Researcher Account To request researcher access, they
are asked to provide basic information, such as username,
email, password, along with the researcher’s information re-
lated to their position title, research institution, department,
and a brief description on how they intend to use the sand-
box. This type of account enables the creation of social me-
dia posts and research experiments, the management of re-
search participants, and the creation and deployment of AI
accounts. The researcher has the permission to create one or
multiple research experiments and to invite participants to
join the experiment(s). In addition, the researcher must pro-
vide a description of each experiment, as well as upload an
IRB form relevant to the experiment.
When inviting participants to join the experiment, the re-
searcher makes sure that the participant receives an invita-
tion email with all the relevant information related to the
experiment. Within the experiment under the researcher ac-
count, four different permission levels result in four differ-
ent types of researcher accounts: owner, collaborator, con-
tent moderator, and regular user. The owner of the experi-
ment, i.e., the researcher, has full control of the experiment,
meaning that they can configure the experiment details as
well as accounts and their roles. Collaborator, who is a co-
researcher of the experiment, has the permission to invite/re-
move regular users and content moderators, as well as to
moderate the content in terms of making sure that the rules
set by the owner of the experiment are being followed. Con-
tent moderator can delete threads, comments, ban, and re-
port regular users. Finally, regular users get invited by theresearcher to join the experiment, or if expressing an interest
in joining the experiment, get approved by the researcher.
During the experiment creation process, the researcher
will have the ability to select whether the experiment is pri-
vate or public. Currently, the sandbox only allows private
experiments. If private, the experiment is invite-only. The
researcher has the option to remove a participant from the
experiment in case they violate the guidelines, as well as to
report them for significantly impacting the experiment de-
sign.
Regular User Account Similarly, a regular user is asked
to provide basic information such as username, email, and
password. If not invited to join the experiment initially, a
regular user can create an account and join the sandbox. In
that case, the user can only see the content and actions of
users who are part of public experiments.
Regular user has the following actions available: scroll,
create posts with up to 280 characters, create hashtags, like,
undo like, create comments, like comments, repost posts,
follow other users, see other public accounts, see content of
followed accounts, and report other users. On the Account
Settings page, user can manage 2FA devices, upload and up-
date their profile and banner photo, and add a background
description (if allowed and instructed by the researcher in
case of participating in the experiment), as well as see the
list of all the experiments that a user is part of. When a
user receives likes, comments, or replies to their own posts,
a real-time notification will be shown to the user. Notifica-
tions can also be visible on the notification page. When new
unseen notifications are detected, the number of unseen no-
tifications is shown to the user. The notification page has five

Figure 3: Technical Architecture of the PDS.
filters: all, likes, comments, reposts, and follows. Posts can
be created on the Home orExplore page in the post box or by
clicking the post button, which opens a post creation dialog
box.
AI Account Deployment
AI accounts, such as AI agents and digital twins, can be en-
rolled in the experiment by the researcher as a part of the ex-
periment. PDS supports two types of AI accounts: internal
(currently available) and external (will be implemented as a
part of future work). Each type of AI account has the follow-
ing actions available: create posts with up to 280 characters
and create hashtags.
The internal AI accounts can be generated directly from
the PDS web application and require less technical AI setup
and configuration. This type of AI account consists of a per-
sonification prompt, an OpenAI-compatible API endpoint,
and an API key for access. It utilizes a Celery task queue as
a notification system related to new posts and replies being
created in the sandbox. Internal AI accounts are, however,
limited in their scalability and customization due to utilizing
the system’s default prompting template.
Technical Architecture
The PDS system is designed as a group of Docker contain-
ers, which contributes to enhancing portability and interop-
erability. The system can be deployed on any server that has
Docker and Docker Compose installed. The Docker Com-
pose configuration file defines the set of services, persistent
data directories, network access rules, and application con-
figuration. The application stack is based on the production-
ready template (Cookiecutter Django 2025), which embod-
ies many twelve-factor application principles (Adam Wig-
gins 2017). Two significant configurable external options are
an email provider and an LLM inference provider. The sys-tem currently supports OpenAI-compatible inference end-
points and tokens. Each agent can override the system’s in-
ference endpoint with their own endpoint, model, and token,
which allows for a variety of inference providers.
The web application is Django 5.x, utilizing Django’s full
stack capabilities as a Model View Template (MVT) frame-
work. The agents operate as task on a task queue to provide
asynchronous and scalable processing. For this purpose, the
PDS uses Celery (Ask Solem and Contributors 2023) as a
task queue. Agent tasks operate in parallel with one another,
but with the full context available on the PDS. In the future,
it is planned for agents to be able to access data and take
actions via API, allowing for external agents to be enrolled.
The tech stack includes the following: Docker for the or-
chestration, PostgreSQL for the database management, Gu-
nicorn for the web application server, Django for the web ap-
plication framework, NGINX for the media file server, Trae-
fik for the reverse proxy, Redis as the message broker, and
Celery for the task queue. The technical architecture can be
seen in Figure 3.
System Events Flow
When a human user creates a post or replies to existing posts,
it triggers an event emitted to each agent that is selected to
act. Each agent then executes their response logic in an in-
dependent process. The response logic is determined by the
researcher or a trainer of the agent/bot. Based on this logic
as well as their persona, they can decide whether any action
is appropriate to take. The actions include liking the post,
reposting it under their own account, or replying to the post.
An agent may decide to perform more than one of these
actions. If the agent decides to reply, then it formulates a
prompt to its inference engine, writes the reply, and submits
it to the system. The agent’s response turn will complete un-
til another event triggers it to act. System flow can be seen
in Figure 4.

Figure 4: PDS System Events Flow.
Implementation - Potential Use Cases
PDS features and functions were selected and designed
based on two primary use cases, internal to our research in-
stitution and in collaboration with other research and train-
ing organizations.
Experiments and Studies
Research teams can develop their chatbots under their own
rules and conditions and bring their external API to the sand-
box. For example, a researcher might want to investigate hu-
man and AI discourse and its effects on information spread
and conversation dynamics. For that purpose, they could de-
ploy the AI bots within the sandbox. Then, they could in-
vite human participants to join the experiment. An important
aspect of this sandbox is that it enables conducting experi-
ments in an environment where the risks of unaware human
participant exposure are removed. They could then assign
collaborator and moderator roles to their team members who
can help them conduct the experiment/study. After running
the experiment at their own pace and terms, they can export
the dataset for the analysis.
Trainings
The sandbox enables the creation of personas and more gen-
eral digital twins that can be used for complex and large-
scale bot interactions. Other than improving the understand-
ing of AI social media behaviors and impacts, the sandbox
also provides a training ground in cases that counter divi-
sive and harmful content online to protect discourse. Addi-
tionally, some teams that would usually provide in-person
training sessions on how to identify problematic online be-
havior could use the sandbox to reduce the training cost and
provide an approach based on exposure to the examples of
such behaviors. They could, for example, change the discus-
sion dynamics to train facilitators/mediators/trainers’ ability
to identify a problematic behavior in early stages, as the data
produced during the training process could be used for pat-
tern prediction, along with human expertise.Future Work
We will make reasonable improvements and add additional
features to improve both the researcher and user experi-
ence. Our primary goal will be to improve the AI account
deployment feature to stay relevant with the current fast-
paced commercial tool developments, such as Model Con-
text Protocol (MCP 2025). Additionally, we plan to enable
the implementation and selection of multiple algorithms rel-
evant for specific research needs in the future. The external
AI account will interact with the PDS via API, supporting
more advanced and complex researcher-customized AI ac-
counts. API endpoints will be inspired by the X/Twitter API
v2 (Twitter - Developer Platform 2025), although they will
not be fully compatible. This can aid technical researchers in
further adapting their AI accounts developed within the PDS
for X/Twitter implementation. The API will be used for post-
exploration as well as for all actions described above. To stay
up to date with the relevant content in the sandbox, external
AI accounts will be capable of connecting to receive server-
sent events in real-time (such as exploring posts, replies, and
likes) with the goal of simulating real user features. Addi-
tionally, a researcher will be able to populate the sandbox by
adding data if available. That dataset, if in possession of the
researcher, should consist of the information that can be used
to create digital twins of accounts on social media platforms.
The model choice, prompt preparation, personification, and
wake/sleep times will be determined by the bot author as the
bot acts as an external user. As part of future work, we also
plan on adding the following functionalities to AI accounts:
like, undo like, create a comment, like a comment, repost
posts, repost comments, follow other users, see the content
of followed accounts, and report other users. External AI
accounts will be highly scalable due to distributed AI host-
ing and computation. Additionally, there will be an option
for a user to see the list of private experiments in progress
and the ability to reach out to the researcher with the intent
of joining the experiment. If approved, the regular user will
receive an invitation from the researcher to join the exper-
iment. To access the new experiment, the main credentials
will remain the same, however, an additional security code

created by the researcher will be added as an authentication
method. Users will then be able to participate in multiple ex-
periments, however, they will be required to follow the rules
relevant to the specific experiments they are part of.
We will test the features and design of the sandbox to meet
regular users’ as well as researchers’ needs. First, we will
utilize the contact information collected via the mailing list
during the previous initial version of the sandbox to test the
regular user experience. To invite non-technical and techni-
cal researchers to use the platform, we will reach out to the
mailing list provided during the proposal of this project. We
will also take direct feedback on the platform via the feed-
back and feature request button . Additionally, we will seek
to conduct regular surveys to look for areas of improvement
based on the needs. Each survey will be created in a way to
better understand each feature available in the sandbox.
Conclusion
Integrating AI accounts into a live public social media en-
vironment can be technically complex as well as ethically
questionable without prior testing. There is a need for plat-
forms that can enable researchers to study numerous re-
search questions related to AI and human interaction. To
protect users on mainstream social media platforms from ex-
posure to AI research experiments without getting their con-
sent and protect their data, while still conducting research
and answering important societal questions, we introduce
PDS. The PDS addresses the need through robust guidelines
and responsible research practices via a user-friendly and
scalable environment. Our hosted live version of the sand-
box, as well as open-source code available on GitHub, can
be of great use for both non-technical and technical digital
discourse researchers.
Acknowledgments
We would like to acknowledge Plurality Institute, Civic
Health Project, and the Center for Research Computing for
their financial support on this project.
References
Adam Wiggins. 2017. The Twelwe-Factor App. https:
//www.12factor.net/. Accessed: 2025-04-16.
Arias Jim ´enez, B.; Rodr ´ıguez-Hidalgo, C.; Mier-Sanmart ´ın,
C.; and Coronel-Salas, G. 2022. Use of chatbots for news
verification. In Communication and Applied Technologies:
Proceedings of ICOMTA 2022 , 133–143. Springer.
Ask Solem and Contributors. 2023. Celery - Distributed
Task Queue. https://docs.celeryq.dev/en/stable/getting-
started/introduction.html. Accessed: 2025-04-16.
Ayers, J. W.; Poliak, A.; Dredze, M.; Leas, E. C.; Zhu, Z.;
Kelley, J. B.; Faix, D. J.; Goodman, A. M.; Longhurst, C. A.;
Hogarth, M.; et al. 2023. Comparing physician and artificial
intelligence chatbot responses to patient questions posted
to a public social media forum. JAMA internal medicine ,
183(6): 589–596.
Bastos, M. T.; and Mercea, D. 2019. The Brexit botnet and
user-generated hyperpartisan news. Social science computer
review , 37(1): 38–54.Calderaro, A. 2018. Social media and politics. The SAGE
handbook of political sociology , 2: 781–795.
Chirper. 2025. Chirper AI - AI Life Simulation. https://
chirper.ai/. Accessed: 2025-04-16.
Cookiecutter Django. 2025. Cookiecutter Django’s Doc-
umentation. https://github.com/cookiecutter/cookiecutter-
django. Accessed: 2025-04-16.
De Choudhury, M.; Sundaram, H.; John, A.; and Seligmann,
D. D. 2010. Analyzing the dynamics of communication in
online social networks. Handbook of social network tech-
nologies and applications , 59–94.
DiFranzo, D.; and Bazarova, N. 2018. The Truman Plat-
form: Social Media Simulation for Experimental Research.
InICSWM Workshop” Bridging the Lab and the Field.
https://socialmedialab. cornell. edu/the-truman-platform .
Felt, M. 2016. Social media and the social sciences: How
researchers employ Big Data analytics. Big data & society ,
3(1): 2053951716645828.
Grady, C. 2015. Institutional review boards: Purpose and
challenges. Chest , 148(5): 1148–1155.
Himelein-Wachowiak, M.; Giorgi, S.; Devoto, A.; Rahman,
M.; Ungar, L.; Schwartz, H. A.; Epstein, D. H.; Leggio, L.;
and Curtis, B. 2021. Bots and misinformation spread on so-
cial media: implications for COVID-19. Journal of medical
Internet research , 23(5): e26933.
Howard, P. N.; and Kollanyi, B. 2016. Bots,# strongerin,
and# brexit: Computational propaganda during the uk-eu
referendum. arXiv preprint arXiv:1606.06356 .
Hu, T.; Liakopoulos, D.; Wei, X.; Marculescu, R.; and
Yadwadkar, N. J. 2025. Simulating Rumor Spreading
in Social Networks using LLM Agents. arXiv preprint
arXiv:2502.01450 .
Jiang, H.; Cheng, Y .; Yang, J.; and Gao, S. 2022. AI-
powered chatbot communication with customers: Dialogic
interactions, satisfaction, engagement, and customer behav-
ior.Computers in Human Behavior , 134: 107329.
Kaggle. 2025. Kaggle Datasets. https://www.kaggle.com/
datasets. Accessed: 2025-04-16.
Kaul, A.; Chaudhri, V .; Cherian, D.; Freberg, K.; Mishra, S.;
Kumar, R.; Pridmore, J.; Lee, S. Y .; Rana, N.; Majmudar,
U.; et al. 2015. Social media: The new mantra for managing
reputation. Vikalpa , 40(4): 455–491.
Krishnan, C.; Gupta, A.; Gupta, A.; and Singh, G. 2022. Im-
pact of artificial intelligence-based chatbots on customer en-
gagement and business growth. In Deep learning for social
media data analytics , 195–210. Springer.
Kruse, L. M.; Norris, D. R.; and Flinchum, J. R. 2018. So-
cial media as a public sphere? Politics on social media. The
Sociological Quarterly , 59(1): 62–84.
Lauren Stewart. 2025. Social Media Research: Analysis of
Social Media Data. https://atlasti.com/research-hub/social-
media-research. Accessed: 2025-04-16.
Leung, C. H.; and Yan Chan, W. T. 2020. Retail chatbots:
The challenges and opportunities of conversational com-
merce. Journal of Digital & Social Media Marketing , 8(1):
68–84.

MCP. 2025. Model Context Protocol. https://
modelcontextprotocol.io/introduction. Accessed: 2025-04-
16.
Mønsted, B.; Sapie ˙zy´nski, P.; Ferrara, E.; and Lehmann, S.
2017. Evidence of complex contagion of information in so-
cial media: An experiment using Twitter bots. PloS one ,
12(9): e0184148.
Nosrati, S.; Sabzali, M.; Heidari, A.; Sarfi, T.; and Sabbar, S.
2020. Chatbots, counselling, and discontents of the digital
life. Journal of Cyberspace Studies , 4(2): 153–172.
Park, J. S.; O’Brien, J.; Cai, C. J.; Morris, M. R.; Liang, P.;
and Bernstein, M. S. 2023. Generative agents: Interactive
simulacra of human behavior. In Proceedings of the 36th
annual acm symposium on user interface software and tech-
nology , 1–22.
People+AI Research (PAIR) Initiative. 2024. Deliberate
Lab. https://github.com/PAIR-code/deliberate-lab. Ac-
cessed: 2025-04-16.
Prabowo, R.; Thelwall, M.; Hellsten, I.; and Scharnhorst, A.
2008. Evolving debates in online communication: a graph
analytical approach. Internet Research , 18(5): 520–540.
Python Software Foundation. 2025a. Beautiful Soup. https:
//pypi.org/project/beautifulsoup4/. Accessed: 2025-04-16.
Python Software Foundation. 2025b. Profanity Check. https:
//pypi.org/project/profanity-check/. Accessed: 2025-04-16.
Radivojevic, K.; Clark, N.; and Brenner, P. 2024. Llms
among us: Generative ai participating in digital discourse.
InProceedings of the AAAI Symposium Series , volume 3,
209–218.
Radivojevic, K.; McAleer, C.; Conley, C.; Kennedy, C.;
and Brenner, P. 2024. Social Media Bot Policies: Eval-
uating Passive and Active Enforcement. arXiv preprint
arXiv:2409.18931 .
Software Freedom Conservancy. 2025. Selenium. https:
//www.selenium.dev/. Accessed: 2025-04-16.
Stella, M.; Ferrara, E.; and De Domenico, M. 2018. Bots
increase exposure to negative and inflammatory content in
online social systems. Proceedings of the National Academy
of Sciences , 115(49): 12435–12440.
Stocking, G.; and Sumida, N. 2018. Social media bots draw
public’s attention and concern. Pew Research Center .
Suarez-Lledo, V .; and Alvarez-Galvez, J. 2022. Assessing
the role of social bots during the COVID-19 pandemic: in-
fodemic, disagreement, and criticism. Journal of Medical
Internet Research , 24(8): e36085.
Twitter - Developer Platform. 2025. Twitter API v2: Early
Access. https://developer.x.com/en/docs/x-api/early-access.
Accessed: 2025-04-16.
Vaidyam, A. N.; Wisniewski, H.; Halamka, J. D.; Kashavan,
M. S.; and Torous, J. B. 2019. Chatbots and conversational
agents in mental health: a review of the psychiatric land-
scape. The Canadian Journal of Psychiatry , 64(7): 456–464.
Vivian Ho. 2025. Reddit slams ‘unethical experi-
ment’ that deployed secret AI bots in forum. https:
//www.washingtonpost.com/technology/2025/04/30/reddit-
ai-bot-university-zurich/. Accessed: 2025-05-15.Yang, Z.; Zhang, Z.; Zheng, Z.; Jiang, Y .; Gan, Z.; Wang,
Z.; Ling, Z.; Chen, J.; Ma, M.; Dong, B.; et al. 2024. Oasis:
Open agents social interaction simulations on one million
agents. arXiv preprint arXiv:2411.11581 .
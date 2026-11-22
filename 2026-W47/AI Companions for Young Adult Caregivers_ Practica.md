# AI Companions for Young Adult Caregivers: Practical Advice, Emotional Depth, and the Limits of Metric Evaluation

**Authors**: Trizal Garg, Xuelong Fu, Dhruv Shah, Jin Huang, Shrigopal Maheshwari, Atiksh Aggarwal, Yash Sinha, Dhruv Kumar, Siddharth Mehrotra

**Published**: 2026-11-17 18:00:46

**PDF URL**: [https://arxiv.org/pdf/2611.23147v1](https://arxiv.org/pdf/2611.23147v1)

## Abstract
Young adult caregivers (YACs) are individuals aged 18 to 25 currently taking care of a family member, partner, or friend facing chronic illness. AI companions designed specifically for YACs remain underexplored in prior works. We present Carebot, it uses an open-weight LLM (Gemma 3-27B) as its base model, on which we compared fine-tuning, retrieval-augmented generation (RAG), and a combination of both. The fine-tuning was performed based on an existing dataset of conversations with YACs and a psychologist. Three well-being experts evaluated these configurations and preferred the fine-tuned model for the warmth of its responses and its actionable advice tailored to YACs. Based on their choice, we adopted the fine-tuned model as Carebot's final configuration and used it for all subsequent evaluations. An LLM judge preferred this Carebot's responses over the base model's in 85% of evaluated pairs. We then conducted a user study with 90 YACs, half of whom interacted with the base model and half with Carebot. The user study revealed a discrepancy between metric evaluation and actual user perception. Participants who used Carebot were more likely to characterize its replies as overly practical and brief, noting that it favored advice over emotional support. Conversely, participants who used the base model indicated that its responses were more emotionally oriented than practical. Although, fine-tuning made Carebot's replies more practical, but they were not emotionally supportive for users. We argue that support tools for YACs require situation-sensitive emotional depth and aligned evaluation frameworks to study their effectiveness.

## Full Text


<!-- PDF content starts -->

AI Companions for Young Adult Caregivers: Practical Advice, Emotional Depth,
and the Limits of Metric Evaluation
TRIZAL GARG,BITS Pilani
XUELONG FU,University of Cambridge
DHRUV SHAH,BITS Pilani
JIN HUANG,University of Cambridge
SHRIGOPAL MAHESHWARI,BITS Pilani
ATIKSH AGGARWAL,BITS Pilani
YASH SINHA,BITS Pilani
DHRUV KUMAR,BITS Pilani
SIDDHARTH MEHROTRA,BITS Pilani
Young adult caregivers (YACs) are individuals aged 18 to 25 currently taking care of a family member, partner, or friend facing
chronic illness. AI companions designed specifically for YACs remain underexplored in prior works. We present Carebot, it uses an
open-weight LLM (Gemma 3-27B) as its base model, on which we compared fine-tuning, retrieval-augmented generation (RAG), and
a combination of both. The fine-tuning was performed based on an existing dataset of conversations with YACs & a psychologist.
Three well-being experts evaluated these configurations and preferred the fine-tuned model for the warmth of its responses and its
actionable advice tailored to YACs. Based on their choice, we adopted the fine-tuned model as Carebot’s final configuration and used
it for all subsequent evaluations. An LLM judge preferred this Carebot’s responses over the base model’s in 85% of evaluated pairs.
We then conducted a user study with 90 YACs, half of whom interacted with the base model and half with Carebot. The user study
revealed a discrepancy between metric evaluation and actual user perception. Participants who used Carebot were more likely to
characterize its replies as overly practical and brief, noting that it favored advice over emotional support. Conversely, participants
who used the base model indicated that its responses were more emotionally oriented than practical. Although, fine-tuning made
Carebot’s replies more practical, but they were not emotionally supportive for users. We argue that support tools for YACs require
situation-sensitive emotional depth and aligned evaluation frameworks to study their effectiveness.
CCS Concepts:•Human-centered computing →Empirical studies in HCI;Natural language interfaces;•Computing method-
ologies→Natural language processing.
Additional Key Words and Phrases: young adult caregivers, large language models, conversational agents, fine-tuning, retrieval-
augmented generation, LLM-as-a-judge, user study, evaluation practice
1 Introduction
Family members provide much of the day-to-day care for relatives who live with illness, disability or frailty. This care
often affects the caregiver’s own health, finances and time . Researchers and services have responded with support for
caregivers, such as information programs, respite care and peer support groups [ 2]. More recently, this support has
moved to chatbots based on LLMs, which generate fluent replies to open questions [ 16,43]. These caregiving tools,
however, target caregivers of all ages. Young adult caregivers (YACs) are people aged 18 to 25 who provide regular
care to a family member, partner, or friend [ 18,31]. They take on this role while they finish their education, start work
Authors’ Contact Information: Trizal Garg, BITS Pilani; Xuelong Fu, University of Cambridge; Dhruv Shah, BITS Pilani; Jin Huang, University of
Cambridge; Shrigopal Maheshwari, BITS Pilani; Atiksh Aggarwal, BITS Pilani; Yash Sinha, BITS Pilani; Dhruv Kumar, BITS Pilani; Siddharth Mehrotra,
BITS Pilani.
Manuscript submitted to ACM 1
arXiv:2611.23147v1  [cs.HC]  17 Nov 2026

2 Garg et al.
and build their own adult lives [ 18]. This group of people, often faces emotional, mental, and practical challenges that
are not always visible or well-supported. In a survey of 295 young adult carers aged 14 to 25 in the United Kingdom,
almost half reported a mental health problem [ 9]. Dang’s work with student YACs in the Netherlands found that they
perceived slightly less balance in life than students without caregiving duties [19].
A recent review of digital tools for young caregivers found only six studies and limited effects on mental health [ 48].
We found no study that builds a generative AI companion for YACs. This gap matters because a general-purpose chatbot
can misjudge what a young caregiver needs. One way to catch such mismatches is to evaluate a companion before it
reaches users. Common method is review by LLM-as-a-judge, in which a second LLM scores or compares replies [ 54].
Neither method involves YACs, so it is not known whether these checks predict how YACs experience a companion.
We present Carebot, an AI companion for YACs, using Gemma-3-27B. It uses two techniques, fine-tuning (FT) and
retrieval-augmented generation (RAG), to tailor the model to YACs. We compared four configurations of the system,
which were the base model, FT alone, RAG alone, and RAG combined with FT. Human experts reviewed a sample of
responses from each configuration and based on that review FT alone was selected. Carebot, the final version tested
with YACs, is the FT alone configuration. The fine-tuning dataset consists of 10,000 synthetic conversations between a
caregiver and a supportive companion, generated with GLM 5.2. To guide the generation, the model was given real
caregiver transcripts as few-shot examples. The conversations balance gender and span 20 countries, 23 care-recipient
conditions and the ten caregiving categories of Dang’s taxonomy [ 18]. We ran a user study with 90 YACs. Half of them
interacted with the base model and half of them with Carebot and then answered a questionnaire about their experience.
We also did an metric evaluation where we reviewed the replies manually, and alongside this we used an LLM judge
that compared replies from both the base and Carebot. We then compared this metric evaluation with the user-study.
In the metric evaluation, human experts preferred Carebot’s replies over those of the base model because, on 500
synthetic held-out examples, Carebot scored higher than base on three metrics out of five, namely BERTScore (0.576),
BLEU (0.014) and ROUGE-L (0.195). An LLM judge (GLM 5.2) also favored Carebot’s replies in 85% of decisive pairwise
verdicts. The YACs who used Carebot were more likely to describe its replies as practical. However, the participants who
used the base model more often described its replies as polite but not actionable. This work makes three contributions.
•We present Carebot, an AI companion for YACs built on Gemma-3-27B (Section 3.1).
•We compare four configurations that combine FT and RAG. The comparison shows how FT and RAG changed
replies as judged by human experts (Section 3.1).
•We report a between-subjects user study with YACs aged 17 to 24 ( 𝑛=90) to study the effectiveness of Carebot
(Sections 3.2 and 4.2).
2 Related Work
2.1 Young Adult Caregivers and Their Support Needs
Levine et al. gave one of the first accounts of YACs as a distinct group [ 31]. They reanalyzed two national caregiver
surveys from the United States and focused on caregivers aged 18 to 25. They estimated that YACs make up 12 to 18%
of adult caregivers. Over half of these YACs were male, and their mean age was 21. The YACs in these surveys also
reported unmet needs, including needs for information.
Becker and Sempik surveyed 295 young adult carers aged 14 to 25 in the United Kingdom about their health, education
and work [ 9]. Almost half of the respondents reported a mental health problem. Lacey et al. used five rounds of a British
household panel to follow young adults over time [ 28]. Those aged 16 to 29 who became caregivers reported fewer
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 3
close friends in the short term. Chevrier et al. reviewed 23 studies of young adult carers [17]. They found that studies
identify YACs in different ways, which makes prevalence estimates vary widely. They also found few or no support
services designed for this age group.
Dang’s doctoral thesis gives a broad account of YAC support needs [ 18]. The thesis includes a systematic review and
a survey of life balance and well-being. It also includes a study of the positive aspects of caregiving, and interviews and
usability tests of a web-based support tool. In the survey, student YACs in the Netherlands reported slightly less balance
in life than students without caregiving duties [ 19]. In the interviews, 13 student YACs said they wanted professionals
who listen without judging them [ 20]. They also wanted more information on how to give emotional support to the
person they care for. They were neutral about using the web-based tool in the future. The thesis concludes that support
should address both the difficulties and the rewards of caregiving [ 18]. Thomas studied young carers in New Delhi and
Mumbai through interviews and a school survey [ 52]. The study found that some children in India provide intensive
care with little recognition or support.
This work differs from these studies in two ways. First, most of the studies above come from the United States and
Europe, and several cover wider age ranges, from 14 to 29. Our user study included participants from across India.
Second, these studies describe YACs and their needs, but none of them has tested a conversational AI system for this
group.
We draw on Dang’s thesis even though it studied students in the Netherlands. The thesis focuses on YACs specifically
and examines their support needs from several angles. Studies from other countries suggest that young carers face
similar difficulties wherever they live. Becker compared research and policy on young carers in the United Kingdom,
Australia, the United States and Sub-Saharan Africa [ 8]. He concluded that young carers have much in common
irrespective of where they live. Lewis et al. surveyed 2,099 carers aged 15 to 17 in six European countries, including
the Netherlands [ 32]. In each country, carers reported lower well-being than their peers without caring duties. Most
carers also reported little formal support for their caring role. Raji et al. reviewed 20 studies of young carers in low- and
middle-income countries [ 44]. We therefore treat the needs that Dang identified as a reasonable starting point for YACs
in India. These needs include professionals who listen without judgment and information on how to give emotional
support.
2.2 Conversational Agents for Emotional and Caregiving Support
Fitzpatrick et al. ran a randomized controlled trial of Woebot, a conversational agent that delivers cognitive behavioral
therapy [ 23]. Woebot follows scripted rules and does not use an LLM. Later work studied companions built on LLMs. Ma
et al. analyzed posts and comments from an online community of Replika, an LLM-based companion app which does
not cater to YAC’s[ 35]. They reported problems with consistency, memory and harmful content, and some described
overdependence. Laestadius et al. analyzed 582 posts by Replika users and described how emotional dependence on
the chatbot harmed some users’ mental health [ 29]. Maeda and Quan-Haase argued that human-like features, such
as personal pronouns and affirming language, lead users to attribute agency to chatbots and trust them [ 36]. They
described the resulting one-sided relationship as parasocial and linked it to harms such as the disclosure of sensitive
data. Sharma et al. took a different approach and used AI to assist human peer supporters [ 49]. Their system gave
supporters feedback on how empathic their replies were. In a randomized trial with 300 supporters on a peer support
platform, it raised conversational empathy by 19.6% overall.
A small set of studies built LLM tools for caregivers. Cheng and Ng grounded a GPT-4o chatbot in a guidebook on
dementia caregiving [ 16]. Twenty-one family caregivers used it for two weeks and rated it easy to use and helpful.
Manuscript submitted to ACM

4 Garg et al.
Parmanto et al. fine-tuned small open models and connected them to a knowledge base on dementia caregiving [ 43].
Their caregiving language model outperformed GPT-3.5 on the metrics they reported. Jo et al. studied CareCall, an
LLM system that makes check-in calls to socially isolated people [ 27]. Their interviews with 34 users, operators and
developers showed that the system eased loneliness. It also led to tensions among stakeholders about what the system
should do. Shankar et al. reviewed digital tools for caregivers aged 25 or younger [ 48]. They found only six studies,
limited effects on mental health, and much lower adherence to self-guided tools than to structured programs.
Miner et al. tested four smartphone assistants and found that they responded inconsistently to crisis statements [ 38].
None of the assistants referred users to a helpline for depression. Moore et al. tested LLMs and commercial therapy
bots against clinical standards of care [ 39]. The models expressed stigma toward some conditions and responded
inappropriately to some crisis prompts. Chandra et al. surveyed 283 adults in the United States with lived experience
of mental health challenges [ 15]. From their accounts, the authors built a taxonomy of 19 harmful AI behaviors and
21 negative psychological impacts. Ajmani et al. collected accounts from 53 people who had used conversational AI
during a mental health crisis and interviewed 16 mental health experts [ 4]. People turned to AI when they lacked access
to professionals or feared burdening others. The authors recommend designing such systems as a bridge to human
support. Arnaiz-Rodriguez et al. audited five LLMs on about 2,000 crisis messages [ 5]. Replies to self-harm and suicidal
ideation received the lowest scores. The authors ran their experiments outside the United States and argue that generic
or US-centric contact information undermines trust in such settings. Sambasivan et al. made a related argument about
algorithmic fairness research, which they found to be West-centric in its subgroups, values and methods [ 46]. Their
interviews in India showed that several assumptions of this research do not hold there.
None of the conversational agents above was built for or tested with YACs. The companion tested models with
prompts or studied companion users, peer supporters and adults with lived experience of mental health challenges.
The two dementia tools serve family caregivers of people with dementia, and CareCall serves socially isolated people.
Shankar et al. is the only work above that focuses on young caregivers [ 48]. Their review summarizes existing digital
tools and does not build or evaluate an LLM system. Carebot is designed for YACs caring for a loved one with any health
condition. Like Parmanto et al., we combined FT with retrieval in some configurations. In our comparison, retrieval did
not improve the replies, so the final Carebot uses FT alone.
2.3 Existing Caregiver Support and Evaluation of AI Companions
Existing support for young carers includes online resources, local services, and peer support. In the United Kingdom,
Carers Trust supports young adult carers aged 16 to 25 through local organizations offering practical and emotional
support, advice, and activities [ 11,12]. The Children’s Society, through its Include programme, provides information on
education, employment, well-being, and carers’ rights, along with resources for finding local young-carer services [ 51].
The NHS also provides guidance on accessing support through schools, healthcare professionals, and local services [ 40].
Action for Children’s Sidekick, initially developed for young carers, provides confidential text and email support from
human coaches. It primarily serves young people aged 13 to 18, with some eligibility extending to age 25 [1].
In India, Carers Worldwide India works with local organizations to provide peer support groups, counselling, liveli-
hood assistance, and access to government entitlements for unpaid family caregivers [ 13,14]. Its programmes support
family caregivers more broadly rather than focusing specifically on YACs [ 14]. These services provide informational,
practical, and interpersonal support, although their availability and target populations vary across countries.
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 5
3 Methodology
Our study had two parts. In Part A, we built the carebot, compared four different configurations offline. In Part B, we
compared the base model with Carebot in two ways, through an metric evaluation by human experts and an LLM judge,
and through a user study with YACs (Section 3.2).
3.1 Part A. System Design
Part A had five steps. We selected a base model, built a training corpus, fine-tuned the model, and added an optional
retrieval component. We then compared the resulting configurations offline.
3.1.1 Architecture.Carebot is a chat application embedded in an online survey. Before the chat begins, the user answers
survey questions about themselves and their caregiving situation. The questions are adapted from Dang’s thesis [ 18].
They cover, for example, age, country, the relationship to the care recipient, the care recipient’s condition and the
time spent on care in Appendix A. The answers form a caregiving profile, which the application inserts into the
system prompt. The system prompt is the fixed set of instructions that the application sends to the model with every
conversation (Appendix B). It asks the model to act as a warm, empathic and supportive assistant that focuses on
practical, everyday help and keeps answers concise but caring. The profile also records whether the user prefers a
quick response. If so, the model gives advice at once. Otherwise, it first asks one or two clarifying questions, which the
prompt calls scaffolding. Each user message also carries a short instruction to keep the response concise unless the
user asks for details. In the conditions with retrieval, the system adds passages from a document store to the prompt
(Section 3.1.5). The application runs on Streamlit, and the fine-tuned model is hosted on Fireworks AI. Figure 1 gives an
overview of the design.
3.1.2 Base Model Selection.We first compared six open models on Human and LLM Mental Health Conversations, a
public dataset previously used by Bird et al. [ 10]. Only the fine-tuned Gemma 3-27B gave more empathetic and less
formal replies while staying coherent (Table 10 in Appendix E). We therefore built Carebot on Gemma-3-27B-IT, the
instruction-tuned release of this model.
3.1.3 Dataset Composition.We generated a synthetic corpus of 10,000 conversations between a caregiver and a
supportive companion using GLM 5.2. The generation prompts used transcripts from real caregivers as few-shot
examples, taken from the sixty hours of interview data in Dang’s thesis [ 18]. The corpus is balanced across the ten
caregiving categories in Dang’s thesis [ 18], 20 countries, 23 conditions of the care recipient, and gender. The conditions
are the health problems of the person receiving care, such as dementia, cancer and substance use. Two percent of
the conversations show the companion referring a user in crisis to safety resources. A crisis is an immediate risk to
the safety of the user or the care recipient. Examples include thoughts of suicide or self-harm, abuse at home and
medical emergencies. Appendix E describes the deduplication and quality checks. We split the corpus into 9,500 training
examples and 500 held-out examples, with 50 per caregiving category and a reference reply for each. We also obtained
54 question and answer passages from real interviews with caregivers. Together with the 500 held-out examples, they
form an evaluation set of 554 inputs.
3.1.4 Fine-Tuning.We fine-tuned Gemma-3-27B-IT with low-rank adaptation (LoRA) [ 25] on the 9,500 training
conversations (Section 3.1.3). LoRA trains a small set of added parameters and leaves the original weights unchanged.
The loss covered only the assistant’s turns, so tokens from user messages and padding did not contribute to training.
Manuscript submitted to ACM

6 Garg et al.
Fig. 1. Overview of Carebot. The user’s profile and preference for a quick reply shape the system prompt. The fine-tuned model
either asks clarifying questions or gives advice. In the conditions with retrieval, a vector database supplies passages from career and
caregiving documents.
Fine-tuning had to teach the model the warm and brief style of the training conversations without harming its fluency.
The model selection experiment showed this risk. Most fine-tuned models in that experiment produced incoherent
or at times inappropriate replies (Section 3.1.2). We therefore chose settings that keep the fine-tuned model close to
Gemma-3-27B-IT. We used a LoRA rank of 8, a single epoch and learning rate of10−5as hyperparameters. The training
loss measures how poorly the model predicts the assistant’s turns in the training data. It fell from 5.34 to 1.71 over
training. Appendix A lists the remaining settings.
3.1.5 Retrieval Component.The retrieval store holds web documents on young caregiving and career guidance,
embedded with the all-mpnet-base-v2 sentence encoder [ 45,50] in a FAISS index [ 21]. Several documents describe
services in the United Kingdom and the Netherlands (Table 6 in Appendix A). The first setup inserted three unlabeled
passages into the prompt. The model then treated other caregivers’ interview statements as the user’s own history, a
context leak. The revised setup labels each passage by its source and tells the model not to attribute retrieved content to
the user. It also gates retrieval and adds up to three passages only when their cosine similarity to the user message is at
least 0.60. This threshold is met for about 20% of messages. We did not include retrieval in the final version of Carebot
as most documents in the store come from particular countries. Their information is often specific and may not apply
to users elsewhere. Retrieved passages may therefore be irrelevant to users in other countries. The metric evaluation
also found that retrieval did not improve replies on average (Section 4.1).
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 7
Table 1. The four conditions in the metric evaluation.
Condition Model Retrieval
Base Gemma-3-27B-IT None
RAG Gemma-3-27B-IT Labeled, not gated
FT (Carebot) Fine-tuned None
FT + gated RAG Fine-tuned Labeled, gated
3.1.6 Evaluated Conditions.Table 1 lists the four evaluated conditions: Base, Base + RAG, FT and FT + RAG. Based on
the results presented in Section 4.1, Carebot is the FT condition, the fine-tuned model without retrieval, and the user
study used this configuration.
3.1.7 Metric Evaluation.We gathered three tiers of evidence. Tier one used reference-based metrics, tier two used LLM
judges, and tier three used human review by the research team. The three tiers give scores or counts. Human review
gives counts of preferred replies and written notes on the reasons for each preference. We treated human review as the
primary tier, so its verdicts took precedence when the tiers disagreed.
•Reference-based metrics.The first tier compared each generated reply with the reference reply for the same
input on the 500 synthetic held-out examples. Table 9 in Appendix E describes each metric. BLEU [ 42], ROUGE-L
[33] and METEOR [ 7] measure overlap in wording. BERTScore [ 53] and BLEURT-20 [ 47] measure similarity
in meaning, so they also give credit for paraphrases. These metrics show how closely a reply resembles the
reference reply. They do not measure whether users find the reply helpful. The reference replies also come from
the same synthetic corpus as the training data. The metrics therefore reward replies that match the style of that
corpus.
•LLM judges.In LLM-as-a-judge evaluation, an LLM scores or compares replies in place of human raters [ 54].
We used GLM 5.2 and DeepSeek-V4-Pro as judges on the 54 real inputs, which have no reference replies.We
chose judges from model families other than Gemma, which underlies every condition, because LLM judges can
favor their own outputs [ 41]. DeepSeek-V4-Pro served as an independent check on GLM 5.2, because GLM 5.2
also generated the training corpus. In tier two, each judge compared the replies of every condition. We kept a
verdict only when the judge chose the same reply in both orders. This protocol controls for position bias, the
tendency of a judge to favor the reply shown first. The judging prompt also stated that length is not a merit.
•Human expert review.In tier three, the human expert team reviewed samples of replies that the team had
selected at random. The reviewers did not know which condition had produced each reply. Each of the three
rounds used 15 items per condition. Round one compared the fine-tuned version against the base model. Round
two examined fine-tuned against fine-tuned with retrieval after we fixed the context leak (Section 3.1.5). Round
three compared fine-tuned against fine-tuned with retrieval again after gating the relevant context. In each
round, reviewers chose the reply they preferred and wrote notes on their reasons.
3.2 Part B. User Study Design
3.2.1 Participants and Recruitment.We evaluated Carebot with 90 YACs, 45 who interacted with the base model and
45 with Carebot. Most were students enrolled in a university (87%). They most often cared for a grandparent (24%), a
parent (10%), a sibling (9%) or a friend (9%), and several named more than one person. Table 2 describes the participants
Manuscript submitted to ACM

8 Garg et al.
Table 2. Participants in the analysis set by condition. Counts show the number of participants, with percentages of the condition in
parentheses.
Base (𝑛=45) Carebot (𝑛=45)
Age
Mean age in years 19.5 18.7
Median age in years 19 18
Age range in years 17 to 23 17 to 24
Occupation
Student 35 (78%) 43 (96%)
Student and working part-time 4 (9%) 1 (2%)
Working professional 6 (13%) 1 (2%)
Use of AI chatbots
Daily 22 (49%) 26 (58%)
A few times a week 19 (42%) 16 (36%)
Rarely 3 (7%) 3 (7%)
Never or first time 1 (2%) 0 (0%)
in each condition. Participants in both groups had a mean age below 20, and most used AI chatbots daily or a few times
a week. The Carebot group had more students, at 96% compared with 78% in the base group.
The study compared two models. In the base model, participants interacted with Gemma-3-27B without FT or RAG.
In the Carebot condition, participants interacted with the fine-tuned only version of Gemma-3-27B.
3.2.2 Procedure.Each participant used the Carebot web application, which connected them to the model. Before the
chat, participants filled in a form about their caregiving situation, including the health condition of the person they care
for. The application passed these answers to the model so that it could personalize its replies. They then interacted with
the model about topics of their own choice. After the conversation, they completed an online questionnaire about their
experience, which Appendix C lists in full. The application did not store the conversations, so all user study results in
this paper come from the questionnaire.
3.2.3 Measures.The first part of the questionnaire described the participant. It asked for the participant’s age, occupa-
tion, frequency of AI chatbot use and caregiving status. The caregiving item asked whether the participant currently
helped care for someone who was ill, disabled, elderly or living with a health or mental condition. We counted partici-
pants who reported any level of help with care as having caregiving duties (Table 7 in Appendix C). Conversation items
covered an overall rating (1 to 10), a tone rating (1 to 5), response length, the balance between emotional support and
practical advice, whether a response felt particularly right or wrong, and a comparison with general AI tools such as
ChatGPT or Claude.
3.2.4 Statistical Analysis.We compared the base and Carebot conditions on nine outcomes from the questionnaire
described in Section 3.2.3. Two of them are ratings, namely the overall rating from 1 to 10 and the tone rating from 1 to
5. The other seven are yes or no outcomes, which record whether a participant chose a given answer, such as finding
the replies too practical (Table 7). For ratings, we used the Mann-Whitney U test, which checks whether one group
tends to rate higher without assuming normally distributed ratings [ 37]. For yes or no outcomes, we used Fisher’s
exact test, which compares how often each group gave an answer and stays valid in small samples [ 22]. For yes or no
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 9
outcomes, we report conditional maximum likelihood odds ratios (ORs) with exact 95% confidence intervals (CIs) [ 3].
We adjusted the p-values of the nine outcomes in Table 5 with the Holm correction, which limits chance findings when
many outcomes are tested [24]. We used a significance level of 0.05.
4 Results
We first report the metric comparison of the four conditions, including Carebot against the base model. We then report
the user study.
4.1 Part A. Metric Comparison
This section reports how the four conditions in Table 1 performed across the three tiers of the metric evaluation
(Section 3.1.7). We report the reference-based metrics, the pairwise verdicts of the LLM judges, the problems that
retrieval raised, the human expert review and the final selection.
Reference-based metrics.Table 3 reports the five metrics for the four conditions. Fine-tuning brought the replies closer
to the reference replies. The FT condition scored highest of the four conditions on BERTScore, BLEU and ROUGE-L. Its
BERTScore was 0.5756 against 0.5346 for the base model, and its ROUGE-L was 0.1950 against 0.1570. BLEU scores
stayed below 0.014 for every condition, because a free reply rarely repeats the exact word sequences of a reference.
METEOR and BLEURT-20 favored the base model, with 0.2954 against 0.2577 for METEOR and 0.4735 against 0.4470 for
BLEURT-20 . A likely reason is that the longer base replies cover more of the content of the reference replies. Retrieval
made almost no difference. Adding retrieval changed each metric by less than 0.012 for the base model and by less than
0.005 for the fine-tuned model. The reference replies come from the same synthetic corpus as the training data. These
gains therefore suggest that fine-tuned learned the wording and style of that corpus. They do not show whether users
prefer its replies, which the user study tests (Section 4.2).
Pairwise judging.We ran two pairwise comparisons (Section 3.1.7), the base model against the fine-tuned model
and the fine-tuned model against the fine-tuned model with gated retrieval. Table 4 reports the results. GLM 5.2, the
primary judge, preferred the fine-tuned model to the base model in 39 of the 46 pairs it counted (85%). It discarded 8 of
the 54 pairs. Retrieval added passages to the prompt for only 28 of the 54 inputs, so the comparison with retrieval used
these 28 inputs. On them, GLM 5.2 preferred the fine-tuned model without retrieval in 14 of 23 counted pairs (61%) and
discarded 5. DeepSeek-V4-Pro, the secondary judge, preferred each model equally often, 14 times each, and discarded
26 of 54 pairs. In the comparison with retrieval, it preferred the fine-tuned model without retrieval in 9 of 15 counted
pairs (60%) and discarded 13.
The two judges differed in position bias, which is a tendency to prefer the reply shown first whatever its content. An
unbiased judge would choose the first reply about half of the time. Counting each order separately, GLM 5.2 chose
the first reply 50% of the time in the comparison with retrieval, which shows no measurable bias. Counting the same
way, DeepSeek-V4-Pro chose the first reply 70% of the time against the base model. A judge with this bias often picks a
different reply when the order is swapped, so the pair is discarded. This explains why DeepSeek-V4-Pro discarded far
more pairs than GLM 5.2.
Retrieval.Retrieval raised two problems during evaluation. First, the initial setup caused context leaks, in which
the model treated other caregivers’ interview statements as the user’s own history. Labeling each passage by its
source removed the leaks we observed. Second, gated retrieval at first appeared to raise the single-response score
Manuscript submitted to ACM

10 Garg et al.
Table 3. Reference-based metrics for the four conditions on the 500 synthetic held-out examples. Higher scores mean greater similarity
to the reference replies. Bold marks the highest score for each metric.
Condition BERTScore BLEU ROUGE-L METEOR BLEURT-20
Base 0.5346 0.0075 0.15700.2954 0.4735
RAG 0.5440 0.0067 0.1582 0.2842 0.4652
FT (Carebot)0.5756 0.0135 0.19500.2577 0.4470
FT + RAG 0.5746 0.0123 0.1942 0.2535 0.4451
Table 4. Pairwise judging on the 54 real inputs. Each row compares condition A (named first) with condition B (named second). For
each input, the judge read one reply from A and one from B and picked the better one. The judge saw each pair twice, with the order
swapped the second time. A condition wins when the judge picked its reply both times. If the judge picked differently the two times,
the pair is discarded. The win share of B is the percentage of wins that went to B. For example, GLM 5.2 picked FT over Base 39 times
and Base over FT 7 times, so the win share of FT is 39 of 46, or 85%. The last two rows use only the 28 inputs for which retrieval added
passages.
Comparison (A against B) Judge A wins B wins Discarded Win share of B
Base against FT (Carebot) GLM 5.2 7 39 8 85%
Base against FT (Carebot) DeepSeek-V4-Pro 14 14 26 50%
FT against FT + RAG GLM 5.2 14 9 5 39%
FT against FT + RAG DeepSeek-V4-Pro 9 6 13 40%
of the fine-tuned model by 0.16 points on the 1 to 5 scale. The 54 real inputs were also put in the retrieval store, so
retrieval often returned the user’s own message. After we excluded these self-matches, the gain shrank to 0.04 points
(Appendix E). The gate fired on 28 of the 54 real inputs. As the pairwise results above show, both judges preferred the
fine-tuned model without retrieval on these inputs (Table 4).
Human review.Human expert team reviewed samples of replies in three rounds. Round one compared the fine-tuned
against the base model. Reviewers clearly preferred the fine-tuned replies. They described the base replies as good but
generic and the fine-tuned replies as warmer and less templated. Round two examined retrieval after the context-leak
fix. Reviewers found that retrieval helped only when it surfaced relevant information and degraded replies when it
added unrelated or generic content. Round three compared the fine-tuned model with and without gated retrieval.
Reviewers preferred the fine-tuned model without retrieval on 11 of 15 items and with retrieval on 2, with 2 ties. Human
review therefore agreed with both debiased judges on retrieval.
Selection.We selected the fine-tuned model without retrieval for Carebot. It scored highest on BERTScore and
ROUGE-L, and GLM 5.2 preferred it to the base model. Both judges and the human experts also preferred it to its variant
with gated retrieval.
4.2 Part B. User Study
4.2.1 Participants.Table 2 describes the 90 participants in the analysis set, 45 per condition. All participants were
aged 17 to 24. The two groups differed in composition. About seven in ten base participants reported caregiving duties,
compared with about half of Carebot participants.
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 11
0 5 10 15 20 25 30 35 40 45Base
Carebot6
2218
1321
10
Number of participantsToo emotional About right Too practical
Fig. 2. Answers to the response balance item by condition, with 45 participants per condition. Participants who used Carebot chose
“too practical” nearly four times as often as participants who used the base model. Base participants more often chose “too emotional”.
4.2.2 Overall Ratings.After the conversation, participants gave Carebot and the base model an overall rating on a scale
from 1 to 10, where higher scores mean a more favorable rating. This single item captures how positively participants
judged the companion as a whole. The mean overall rating was 7.44 (median 8) for the base model and 7.38 (median
7) for Carebot, so participants in both conditions rated their companion fairly positively and almost identically. We
compared the two groups with the Mann-Whitney U test, which checks whether ratings in one group tend to be higher
than in the other without assuming that ratings follow a bell-shaped distribution. The difference was not significant
(𝑝=0.30), which means that a gap this small could easily arise by chance. Low ratings of 5 or below were also similar,
coming from 6 of 45 base participants (13%) and 4 of 45 Carebot participants (9%) (𝑝=0.74).
4.2.3 Response Balance.The clearest difference concerned the balance between emotional and practical content. The
questionnaire asked whether the replies gave the right balance of emotional support and practical advice. Figure 2
shows the answers. About half of Carebot participants found the replies too practical (22 of 45, 49%), compared with
about one in eight base participants (6 of 45, 13%). This difference was significant (Fisher’s exact test, 𝑝< 0.001) and
remained significant after the Holm correction (adjusted𝑝=0.005).
Base participants more often chose “too emotional” (21 of 45, 47%) than Carebot participants (10 of 45, 22%). This
difference is significant only before correction ( 𝑝=0.026), and the outcome was not part of the corrected family of nine
tests, so we interpret it with caution. The share choosing “about right” did not differ significantly (18 of 45, 40%, for
base compared with 13 of 45, 29%, for Carebot;𝑝=0.38).
4.2.4 Tone and Descriptions of Replies.Table 5 lists all nine outcomes. Apart fromresponse balance, no difference
remained significant after the Holm correction (all adjusted 𝑝≥ 0.21), so we treat the following results as exploratory.
Base participants rated the tone higher, with a mean of 4.09 compared with 3.73 on a scale from 1 to 5 ( 𝑝= 0.026,
adjusted 𝑝=0.21). They more often described the replies as extremely polite and helpful, with 17 of 45 (38%) compared
with 7 of 45 (16%) ( 𝑝=0.031). Carebot participants more often said that nothing in the replies stood out. This was the
case for 21 of 45 Carebot participants (47%) and 11 of 45 base participants (24%) ( 𝑝=0.047). Carebot participants also
more often found the replies too short or lacking depth (18 of 45 against 9 of 45, 𝑝=0.065). Some participants had used
general-purpose assistants. Among them, 15 of 40 base participants (38%) rated the companion as better, compared with
10 of 39 Carebot participants (26%) (𝑝=0.33).
Manuscript submitted to ACM

12 Garg et al.
Table 5. User study outcomes in the analysis set. Ratings show means, with medians in parentheses for the overall rating. Yes or no
outcomes show counts and percentages. ORs above 1 mean the outcome was more common with Carebot. Holm-adjusted p-values
cover the nine outcomes in this table.
Outcome Base Carebot OR (95% CI)𝑝Holm𝑝
Overall rating, 1 to 10 7.44 (8) 7.38 (7) n/a 0.30 0.89
Tone rating, 1 to 5 4.09 3.73 n/a 0.026 0.21
Replies too practical 6/45 (13%) 22/45 (49%) 6.08 (2.02, 21.15)<0.001 0.005
Extremely polite and helpful 17/45 (38%) 7/45 (16%) 0.31 (0.09, 0.91) 0.031 0.21
Nothing stood out 11/45 (24%) 21/45 (47%) 2.67 (1.01, 7.39) 0.047 0.28
Too short or lacked depth 9/45 (20%) 18/45 (40%) 2.64 (0.95, 7.78) 0.065 0.32
A reply felt right 26/45 (58%) 18/45 (40%) 0.49 (0.19, 1.22) 0.14 0.56
Better than ChatGPT or Claude 15/40 (38%) 10/39 (26%) 0.58 (0.19, 1.66) 0.33 0.89
Overall rating of 5 or below 6/45 (13%) 4/45 (9%) 0.64 (0.12, 2.92) 0.74 0.89
4.2.5 Summary of User Study Results.Participants rated the two companions similarly overall, so the user study did not
show that fine-tuning improved the experience of YACs. The one difference that remained significant after correction
for multiple comparisons wasresponse balance: Carebot participants were significantly more likely to describe its replies
as too practical. The base model’s higher tone and politeness ratings did not remain significant after correction, and the
Carebot participants’ tendency to find replies short or lacking depth was not significant.
5 Discussion
This section interprets the main result of the study. The metric evaluation favored Carebot, yet participants rated it no
higher overall and more often found its replies too practical. We examine why the two evaluations diverged.
5.1 Why Metric Evaluation and User Ratings Differed
First, a system can reply well to one isolated message and still fall short in a conversation, where each reply must
build on earlier ones. Lee et al. found that better performance without interaction does not always lead to better
interaction with people [ 30]. In our metric evaluation, the offline methods scored single replies to single messages,
while participants judged whole conversations about their own situations.
Second, LLM judges can favor text written in their own style [ 41]. In our metric evaluation, GLM 5.2 wrote the
training corpus, the reference replies came from that corpus, and GLM 5.2 also served as the primary judge. The
reference metrics and the primary judge therefore partly measured agreement with GLM 5.2. DeepSeek-V4-Pro had
no part in the training data, and it split evenly between Carebot and the base model, although its strong position bias
makes this split hard to interpret.
Third, observers and recipients can judge empathy differently. Clinicians who read patient questions rated chatbot
answers as more empathetic than physician answers [ 6]. People who took part in conversations rated chatbots as less
empathetic than human partners [ 34]. In our metric evaluation, the human reviewers and the LLM judges evaluated
replies to queries that YACs had written. However, the YACs involved in the user study rated replies to their own
concerns.
Fourth, a model tested on data that resembles its training data can appear stronger than it is in real use [ 26]. In our
metric evaluation, the offline inputs were synthetic messages and passages from caregiver interviews, while the YACs
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 13
in the user study raised their own concerns. A model tuned and tested on the offline inputs may therefore fit poorly
with what YACs asked.
5.2 What the Metric Evaluation Missed
The metric evaluation did not measure three aspects that could have mattered to YACs or to their safety.
Firstly, it did not measure the balance between emotional support and practical advice. A reply can be warm and give
useful advice, and still fail a user who only wanted someone to listen. In the user study, the question on this balance
showed the clearest difference between Carebot and the base model, yet none of the measures in our metric evaluation
asked about it. This gap matters for many YACs, who said in Dang’s interviews that they wanted professionals who
listen without judging them [20].
Secondly, it did not capture whether the replies were long and substantial enough. Fine-tuning taught Carebot the
brief style of the training corpus. LLM judges tend to prefer longer replies whatever their quality, a tendency known
as length bias [ 54], so the pairwise prompt told the judges not to rate a reply higher only because it was longer. This
removed a known bias, but it may also have hidden a signal that participants used. Carebot participants more often chose
the answer “too short or lacked depth” (18 of 45 against 9 of 45), although this difference was not statistically significant
(Table 5). That answer combines brevity with a lack of substance, such as little engagement with the participant’s
situation, and because we did not store conversations, we cannot tell which of the two participants meant.
Thirdly, it did not test whether crisis responses were locally relevant. A participant reported that the base model
answered a disclosure of suicidal thoughts with helplines for the United States and Canada. No part of our metric
evaluation tested whether crisis referrals point to services in the user’s country. These gaps share a common source:
YACs had no role in the metric evaluation.
5.3 Alternative Interpretations
The results also allow two other interpretations, which concern the system prompt and the choice of configuration.
First, the system prompt may explain part of the shift. Both conditions used the same prompt, which asked for
practical, everyday help and concise answers (Appendix B). The fine-tuned model, trained on brief replies, may have
followed these instructions more closely than the base model. Varying the prompt while holding the model fixed would
separate the effect of the prompt from the effect of fine-tuning.
Second, a different configuration would likely not have changed the main result. Retrieval added passages to only
about one in five user messages (Section 3.1.5), and most of those passages gave practical guidance on careers and
services (Table 6). Adding retrieval to Carebot would therefore have left most replies unchanged and made the rest
more practical. Retrieval without fine-tuning would have kept the style of the base model, which nearly half of its users
found too emotional. We did not test these configurations with users, but reviewers preferred Carebot without retrieval
on 11 of 15 items (Section 4.1).
5.4 Limitations
Majority of our YACs were male university students. We also assigned conditions by time slot, so the two groups
differed in caregiving status and age. A future study should assign equal YACs aged 18 to 25 to conditions at random. It
could also test a version of Carebot that first asks whether the user wants to be heard or wants advice. The application
did not store conversations, so we cannot link ratings to specific replies or verify the reported crisis response. A longer
deployment with consented logging would address this. GLM 5.2 wrote the training corpus and reference replies and
Manuscript submitted to ACM

14 Garg et al.
also served as the primary judge, and the second judge showed a strong position bias. No YACs took part in the metric
evaluation. Future work could fine-tune on replies that YACs write or rate, and test which LLM judges agree with YAC
ratings. Finally, Carebot has no source of local information, and we did not test whether crisis referrals point to services
in the user’s country. Retrieval from local sources, such as services and helplines in India, could improve both.
6 Conclusion
We presented Carebot, a companion for young adult caregivers built on a fine-tuned Gemma-3-27B-IT, and compared
its metric evaluation with how it fared with the people it was built for. On paper, Carebot did well: three well-being
experts chose the fine-tuned version over RAG and the combined approach, and an LLM judge preferred its replies
to the base model’s in 85% of pairs. However, the 90 YACs in our user study saw it differently. Many found Carebot’s
replies short and too practical, offering advice when they wanted to feel heard, while the base model came across as
more emotionally attuned. Expert review and automated judges are useful, but they cannot stand in for the people a
system is meant to help. Teams building support companions for YACs should involve YACs in the evaluation itself,
letting them define what a good reply looks like and when practical advice is welcome.
Ethical Considerations Statement
The young people in this study care for family members, often while also studying, working, or simply growing up, and
many of them are under real strain. That shaped how we handled both the people and their words.
The study was approved by the Institutional Review Board of University of Amsterdam (ref. 2025-898). The 54
interview passages we use for retrieval and evaluation come from an existing dataset collected under Dang [18]. Everyone
we interviewed was 18–25 years old and gave written informed consent, which covered the reuse of anonymised
excerpts in research of this kind. They knew they could skip questions, stop at any point, or withdraw their data up to
October 01, 2026.
Before using any passage, we removed names, places, dates, and anything about the family member’s condition that
could identify them. The people being cared for never agreed to take part, so we were just as careful with their details
as with the participants’ own. The passages stay on our secure institutional storage and only the research team can
access them. We aren’t releasing them. Instead, we will share code, prompts, rubrics, and synthetic examples used in
this study.
We are also wary of what a system like this could do. A companion that sounds warm can still give bad advice about
medication or benefits, miss signs that someone is in crisis, or become a substitute for people who could actually help.
Our prototype hasn’t been used by caregivers outside the study. One of our main findings is that standard metrics can
score a response well while missing exactly these problems. A good score shouldn’t be read as a sign that the system is
safe, and anyone building on this work should involve young carers and the organisations that support them from the
start.
Generative AI Usage Statement
We acknowledge the use of AI tools like Claude and ChatGPT for assistance in coding and in improving the grammatical
precision and presentation of this paper. Within the study itself, GLM 5.2 generated the synthetic training corpus
(Section 3.1.3). GLM 5.2 and DeepSeek-V4-Pro served as the LLM judges described in Section 3.1.7. The experimental
design, analysis, and all scientific conclusions accurately represent the original contributions of the authors.
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 15
References
[1]Action for Children. 2026. Do You Need Someone to Talk to? Text Sidekick for Support. Retrieved November 4, 2026 from https://sidekick.
actionforchildren.org.uk/
[2]Ronald D. Adelman, Lyubov L. Tmanova, Diana Delgado, Sarah Dion, and Mark S. Lachs. 2014. Caregiver Burden: A Clinical Review.JAMA311, 10
(2014), 1052–1059. doi:10.1001/jama.2014.304
[3] Alan Agresti. 2013.Categorical Data Analysis(3 ed.). John Wiley & Sons, Hoboken, NJ.
[4]Leah Hope Ajmani, Arka Ghosh, Benjamin Kaveladze, Eugenia Kim, Keertana Namuduri, Theresa Nguyen, Ebele Okoli, Jessica Schleider, Denae
Ford, and Jina Suh. 2026. Seeking Late Night Life Lines: Experiences of Conversational AI Use in Mental Health Crisis. InProceedings of the 2026
ACM Conference on Fairness, Accountability, and Transparency (FAccT ’26). Association for Computing Machinery, New York, NY, USA, 1263–1290.
doi:10.1145/3805689.3812256
[5]Adrian Arnaiz-Rodriguez, Miguel Baidal, Erik Derner, Jenn Layton Annable, Mark Ball, Mark Ince, Elvira Perez Vallejos, and Nuria Oliver. 2026.
Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models.JMIR Mental Health13 (2026), e88435.
doi:10.2196/88435
[6]John W. Ayers, Adam Poliak, Mark Dredze, Eric C. Leas, Zechariah Zhu, Jessica B. Kelley, Dennis J. Faix, Aaron M. Goodman, Christopher A.
Longhurst, Michael Hogarth, and Davey M. Smith. 2023. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions
Posted to a Public Social Media Forum.JAMA Internal Medicine183, 6 (2023), 589–596. doi:10.1001/jamainternmed.2023.1838
[7] Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.
InProceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization. Association for
Computational Linguistics, Ann Arbor, Michigan, 65–72. https://aclanthology.org/W05-0909/
[8]Saul Becker. 2007. Global Perspectives on Children’s Unpaid Caregiving in the Family: Research and Policy on ‘Young Carers’ in the UK, Australia,
the USA and Sub-Saharan Africa.Global Social Policy7, 1 (2007), 23–50. doi:10.1177/1468018107073892
[9]Saul Becker and Joe Sempik. 2019. Young Adult Carers: The Impact of Caring on Health and Education.Children & Society33, 4 (2019), 377–386.
doi:10.1111/chso.12310
[10] Jordan J. Bird, David Wright, Alexander Sumich, and Ahmad Lotfi. 2024. Generative AI in Psychological Therapy: Perspectives on Computational
Linguistics and Large Language Models in Written Behaviour Monitoring. InProceedings of the 17th International Conference on PErvasive Technologies
Related to Assistive Environments (PETRA ’24). Association for Computing Machinery, New York, NY, USA, 322–328. doi:10.1145/3652037.3663893
[11] Carers Trust. 2026. About Young Adult Carers. Retrieved November 4, 2026 from https://carers.org/about-caring/about-young-adult-carers
[12] Carers Trust. 2026. Getting Support If You Are a Young Carer or Young Adult Carer. Retrieved November 4, 2026 from https://carers.org/getting-
support-if-you-are-a-young-carer-or-young-adult-carer
[13] Carers Worldwide. 2019. Impact Report 2019. Retrieved November 4, 2026 from https://carersworldwide.org/images/publications/2019-Impact-
Report.pdf
[14] Carers Worldwide. 2026. Carers Worldwide India: Our Sister Organisation. Retrieved November 4, 2026 from https://carersworldwide.org/our-
work/carers-worldwide-india
[15] Mohit Chandra, Suchismita Naik, Denae Ford, Ebele Okoli, Munmun De Choudhury, Mahsa Ershadi, Gonzalo Ramos, Javier Hernandez, Ananya
Bhattacharjee, Shahed Warreth, and Jina Suh. 2025. From Lived Experience to Insight: Unpacking the Psychological Risks of Using AI Conversational
Agents. InProceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’25). Association for Computing Machinery,
New York, NY, USA, 975–1004. doi:10.1145/3715275.3732063
[16] Sheung-Tak Cheng and Peter H. F. Ng. 2025. The PDC30 Chatbot—Development of a Psychoeducational Resource on Dementia Caregiving Among
Family Caregivers: Mixed Methods Acceptability Study.JMIR Aging8 (2025), e63715. doi:10.2196/63715
[17] Basilie Chevrier, Kristopher Lamore, Aurélie Untas, and Géraldine Dorard. 2022. Young Adult Carers’ Identification, Characteristics, and Support: A
Systematic Review.Frontiers in Psychology13, Article 990257 (2022). doi:10.3389/fpsyg.2022.990257
[18] Srishti Dang. 2024.Young Hands, Caregiving Hearts: Caregiving Experiences and Support Needs of Young Adult Caregivers in General and towards a
Digital Support Tool. Ph. D. Dissertation. University of Groningen, Groningen, The Netherlands. doi:10.33612/diss.1124066292
[19] Srishti Dang, Anne Looijmans, Giovanni Lamura, and Mariët Hagedoorn. 2024. Perceived Life Balance among Young Adult Students: A Comparison
between Caregivers and Non-Caregivers.BMC Psychology12, 1, Article 18 (2024). doi:10.1186/s40359-023-01500-z
[20] Srishti Dang, Anne Looijmans, Nikita Sharma, Annemarie Braakman Jansen, and Mariët Hagedoorn. 2024. Exploring the Support Needs of Young
Adult Caregivers, Their Issues, and Preferences towards a Web-Based Tool: A Mixed-Method Study.BMC Digital Health2, 1, Article 3 (2024).
doi:10.1186/s44247-023-00057-6
[21] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and
Hervé Jégou. 2026. The Faiss Library.IEEE Transactions on Big Data12, 2 (2026), 346–361. doi:10.1109/TBDATA.2025.3618474
[22] R. A. Fisher. 1935. The Logic of Inductive Inference.Journal of the Royal Statistical Society98, 1 (1935), 39–54. doi:10.2307/2342435
[23] Kathleen Kara Fitzpatrick, Alison Darcy, and Molly Vierhile. 2017. Delivering Cognitive Behavior Therapy to Young Adults With Symptoms of
Depression and Anxiety Using a Fully Automated Conversational Agent (Woebot): A Randomized Controlled Trial.JMIR Mental Health4, 2 (2017),
e19. doi:10.2196/mental.7785
Manuscript submitted to ACM

16 Garg et al.
[24] Sture Holm. 1979. A Simple Sequentially Rejective Multiple Test Procedure.Scandinavian Journal of Statistics6, 2 (1979), 65–70. https:
//www.jstor.org/stable/4615733
[25] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank
Adaptation of Large Language Models. InInternational Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9
[26] Ben Hutchinson, Negar Rostamzadeh, Christina Greer, Katherine Heller, and Vinodkumar Prabhakaran. 2022. Evaluation Gaps in Machine Learning
Practice. InProceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency(Seoul, Republic of Korea)(FAccT ’22). Association
for Computing Machinery, New York, NY, USA, 1859–1876. doi:10.1145/3531146.3533233
[27] Eunkyung Jo, Daniel A. Epstein, Hyunhoon Jung, and Young-Ho Kim. 2023. Understanding the Benefits and Challenges of Deploying Conversational
AI Leveraging Large Language Models for Public Health Intervention. InProceedings of the 2023 CHI Conference on Human Factors in Computing
Systems (CHI ’23). Association for Computing Machinery, New York, NY, USA, 1–16. doi:10.1145/3544548.3581503
[28] Rebecca E. Lacey, Giorgio Di Gessa, Baowen Xue, and Anne McMunn. 2023. Inequalities in Associations between Young Adult Caregiving and
Social Relationships: Evidence from the UK Household Longitudinal Study.Journal of Adolescence95, 7 (2023), 1293–1310. doi:10.1002/jad.12202
[29] Linnea Laestadius, Andrea Bishop, Michael Gonzalez, Diana Illenčík, and Celeste Campos-Castillo. 2024. Too human and not human enough: A
grounded theory analysis of mental health harms from emotional dependence on the social chatbot Replika.New Media & Society26, 10 (2024),
5923–5941. doi:10.1177/14614448221142007
[30] Mina Lee, Megha Srivastava, Amelia Hardy, John Thickstun, Esin Durmus, Ashwin Paranjape, Ines Gerard-Ursin, Xiang Lisa Li, Faisal Ladhak,
Frieda Rong, Rose E. Wang, Minae Kwon, Joon Sung Park, Hancheng Cao, Tony Lee, Rishi Bommasani, Michael S. Bernstein, and Percy Liang. 2023.
Evaluating Human-Language Model Interaction.Transactions on Machine Learning Research(2023). https://openreview.net/forum?id=hjDYJUn9l1
[31] Carol Levine, Gail Gibson Hunt, Deborah Halper, Andrea Y. Hart, Jessica Lautz, and David A. Gould. 2005. Young Adult Caregivers: A First Look at
an Unstudied Population.American Journal of Public Health95, 11 (2005), 2071–2075. doi:10.2105/AJPH.2005.067702
[32] Feylyn Mercedies Lewis, Saul Becker, Thomas Parkhouse, Stephen Joseph, Valentina Hlebec, Maja Mrzel, Rosita Brolin, Giulia Casu, Licia Boccaletti,
Sara Santini, Barbara D’Amen, Marco Socci, Renske Hoefman, Nynke de Jong, Agnes Leu, Daniel Phelps, Elena Guggiari, Lennart Magnusson, and
Elizabeth Hanson. 2023. The First Cross-National Study of Adolescent Young Carers Aged 15–17 in Six European Countries.International Journal of
Care and Caring7, 1 (2023), 6–32. doi:10.1332/239788222X16455943560342
[33] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. InText Summarization Branches Out. Association for Computational
Linguistics, Barcelona, Spain, 74–81. https://aclanthology.org/W04-1013/
[34] Tingting Liu, Salvatore Giorgi, Ankit Aich, Allison Lahnala, Brenda Curtis, Lyle H. Ungar, and João Sedoc. 2025. The Illusion of Empathy: How AI
Chatbots Shape Conversation Perception. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. Association for the Advancement
of Artificial Intelligence, 14327–14335. doi:10.1609/aaai.v39i13.33569
[35] Zilin Ma, Yiyang Mei, and Zhaoyuan Su. 2023. Understanding the Benefits and Challenges of Using Large Language Model-based Conversational
Agents for Mental Well-being Support.AMIA Annual Symposium Proceedings2023 (2023), 1105–1114.
[36] Takuya Maeda and Anabel Quan-Haase. 2024. When Human-AI Interactions Become Parasocial: Agency and Anthropomorphism in Affective
Design. InThe 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’24). Association for Computing Machinery, New York, NY,
USA, 1068–1077. doi:10.1145/3630106.3658956
[37] H. B. Mann and D. R. Whitney. 1947. On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other.The Annals of
Mathematical Statistics18, 1 (1947), 50–60. doi:10.1214/aoms/1177730491
[38] Adam S. Miner, Arnold Milstein, Stephen Schueller, Roshini Hegde, Christina Mangurian, and Eleni Linos. 2016. Smartphone-Based Conversational
Agents and Responses to Questions About Mental Health, Interpersonal Violence, and Physical Health.JAMA Internal Medicine176, 5 (2016),
619–625. doi:10.1001/jamainternmed.2016.0400
[39] Jared Moore, Declan Grabb, William Agnew, Kevin Klyman, Stevie Chancellor, Desmond C. Ong, and Nick Haber. 2025. Expressing stigma and
inappropriate responses prevents LLMs from safely replacing mental health providers. InProceedings of the 2025 ACM Conference on Fairness,
Accountability, and Transparency (FAccT ’25). Association for Computing Machinery, New York, NY, USA, 599–627. doi:10.1145/3715275.3732039
[40] NHS. 2024. Help for Young Carers. Page last reviewed 26 February 2024. Retrieved November 4, 2026 from https://www.nhs.uk/social-care-and-
support/support-and-benefits-for-carers/help-for-young-carers/
[41] Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM Evaluators Recognize and Favor Their Own Generations. InAdvances in Neural
Information Processing Systems, Vol. 37. Neural Information Processing Systems Foundation, Inc., 68772–68802. doi:10.52202/079017-2197
[42] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. In
Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Philadelphia,
Pennsylvania, USA, 311–318. doi:10.3115/1073083.1073135
[43] Bambang Parmanto, Bayu Aryoyudanta, Timothius Wilbert Soekinto, I Made Agus Setiawan, Yuhan Wang, Haomin Hu, Andi Saptono, and
Yong Kyung Choi. 2024. A Reliable and Accessible Caregiving Language Model (CaLM) to Support Tools for Caregivers: Development and Evaluation
Study.JMIR Formative Research8 (2024), e54633. doi:10.2196/54633
[44] Naseem Akorede Raji, Shaun Liverpool, Fariba Bannerman, and Ken A. Fletcher. 2025. Experiences and Outcomes of Young Carers Living
in Low- and Middle-Income Countries: A Scoping Review and Narrative Synthesis.Journal of Child and Family Studies34, 1 (2025), 83–94.
doi:10.1007/s10826-024-02992-y
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 17
[45] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. InProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).
Association for Computational Linguistics, Hong Kong, China, 3982–3992. doi:10.18653/v1/D19-1410
[46] Nithya Sambasivan, Erin Arnesen, Ben Hutchinson, Tulsee Doshi, and Vinodkumar Prabhakaran. 2021. Re-imagining Algorithmic Fairness in
India and Beyond. InProceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’21). Association for Computing
Machinery, New York, NY, USA, 315–328. doi:10.1145/3442188.3445896
[47] Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020. BLEURT: Learning Robust Metrics for Text Generation. InProceedings of the 58th Annual
Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Online, 7881–7892. doi:10.18653/v1/2020.acl-
main.704
[48] Ravi Shankar, Amaevia Lim, and Qian Xu. 2026. Effectiveness of Digital Tools in Supporting Young Caregivers: A Systematic Review.Early
Intervention in Psychiatry20, 7, Article e70215 (2026). doi:10.1111/eip.70215
[49] Ashish Sharma, Inna W. Lin, Adam S. Miner, David C. Atkins, and Tim Althoff. 2023. Human–AI collaboration enables more empathic conversations
in text-based peer-to-peer mental health support.Nature Machine Intelligence5, 1 (2023), 46–57. doi:10.1038/s42256-022-00593-2
[50] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. MPNet: Masked and Permuted Pre-training for Language Understanding. InAdvances
in Neural Information Processing Systems, Vol. 33. 16857–16867. https://proceedings.neurips.cc/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-
Abstract.html
[51] The Children’s Society. 2026. Advice and Support for Young Carers. Retrieved November 4, 2026 from https://www.childrenssociety.org.uk/
information/young-people/young-carers
[52] Teemol Thomas. 2025.Hidden in Plain Sight: A Mixed-Methods Study of Young Carers in New Delhi and Mumbai. Ph. D. Dissertation. University of
East Anglia, Norwich, UK. https://ueaeprints.uea.ac.uk/id/eprint/103071/
[53] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating Text Generation with BERT. In
International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=SkeHuCVFDr
[54] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao
Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. InAdvances in Neural Information
Processing Systems, Vol. 36. Neural Information Processing Systems Foundation, Inc., 46595–46623. doi:10.52202/075280-2020
A Model Configuration
Fine-tuning used the Unsloth library with the PEFT and SFTTrainer libraries from Hugging Face. The fine-tuned model
used LoRA with rank 8, one epoch and a learning rate of10−5. The 27B models were trained on NVIDIA A100 (40 GB)
or H100 (94 GB) GPUs. Smaller models and test runs used a GPU with 24 GB of memory. Evaluation runs served the
models with vLLM, and the fine-tuned model for the application is hosted on Fireworks AI.
Retrieval used chunks of 1,000 characters with an overlap of 100 characters, embedded with all-mpnet-base-v2 and
stored in a FAISS index. The embeddings are normalized, and the same encoder embeds documents and queries. The
first setup inserted the top three passages from maximal marginal relevance search. The gated setup inserts up to three
passages whose cosine similarity to the query is at least 0.60. Table 6 lists the documents in the first store.
B System Prompt
The application sent the main instruction prompt below to the model with every conversation. It filled the field
{user_profile} with the user’s caregiving profile and the field {context} with the retrieved passages. Each user
query also carried the additional instruction in the second box.
Manuscript submitted to ACM

18 Garg et al.
Main instruction prompt
You are Carebot, a warm, empathic, supportive assistant. You speak kindly and clearly, avoid medical
or legal claims, and focus on practical, everyday help. Reflect feelings, validate, suggest small
steps and resources, and encourage seeking trusted adults or professionals when appropriate. Keep
answers concise but caring. Use the user’s profile to personalize your tone and suggestions:
{user_profile}
If the user prefers a quick response, immediately give your advice. If not, employ scaffolding
techniques and ask one or two questions to better understand the situation before giving advice. Keep
this scaffolding process concise.
You may use the retrieved context to help you answer the question, if relevant:
{context}
Here is the user query:
Additional instruction prompt for each user query
Keep your response concise, except when asked for details
Each user query also carries the instruction “Keep your response concise, except when asked for details”.
Table 6. Documents in the first retrieval store. The second store caps the first document at 300 chunks and adds five caregiver-specific
sources.
Title Source
Find a fulfilling career that does good 80,000 Hours
Framework for career guidance for young people Euroguidance
Professionalism of career guidance Euroguidance
Young carers: Shaping our future The Children’s Society
Balancing caregiving and career: tips for starting your new job strong Bright Horizons
Help for young carers NHS
Do you know your rights in youth care? Jeugdstem
C Survey Instrument
Table 7 lists the questionnaire items with the answer options observed in the two conditions. Items marked as free text
had no fixed options. The questionnaire also asked for consent, the participant’s full name and an email address.
D Sensitivity Analysis
Table 8 reports the analysis of all responses received. P-values carry the Holm correction across the outcomes in the
table. The exception is the item on short replies, for which the table gives the uncorrected p-value from Fisher’s exact
test. The balance p-value refers to the item as a whole.
In covariate-adjusted ordinal models, the OR for response balance was 11.7 (95% CI 4.3 to 31.8). The OR for the
overall rating was 2.0 (𝑝=0.12).
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 19
Table 7. Questionnaire items and answer options.
Item Answer options
Age Number
Gender Male, female
Current primary occupation or status Student, working professional, student and working part-time
Do you currently help look after a family member or
loved one who is ill, disabled, elderly, or struggling with
health or mental conditions?Yes, a lot (high involvement caregiver), yes, a little, or left blank
If yes, who are you helping care for? Free text
How often do you use AI chatbots (e.g. ChatGPT, Claude,
Gemini)?Daily, a few times a week, rarely, never or first time
What specific topics or questions did you ask the AI
model?Free text
Overall rating of the AI companion 1 to 10
Tone, empathy, and accuracy Extremely, very, moderately, slightly, or not at all polite or helpful
Did Carebot give the right balance of emotional support
and practical advice?Far too emotional, not practical enough, somewhat emotional, not
practical enough, about right, somewhat practical, not emotional
enough, far too practical, not emotional enough
Was there one response that felt particularly right or
wrong?Yes, it felt right, yes, it felt wrong, no, nothing stood out, with
optional free text
Response length and clarity Just right, too short or lacked depth, too long or wordy
How does this model compare to general AI tools like
ChatGPT or Claude?Much better, better, similar, worse, much worse, I haven’t used them
Comparison to a human companion Free text
What changes or tweaks would make this AI better for
young adult caregivers?Free text
Table 8. Outcomes of the base and Carebot participants. Ratings show means, with the median in parentheses for the overall rating.
Outcome Base Carebot𝑝
Overall rating, 1 to 10 7.38 (8) 7.62 (7) 0.95
Tone rating, 1 to 5 4.02 3.79 0.49
Balance, too emotional 36% 18%
Balance, about right 47% 26%
Balance, too practical 17% 56%<0.001
Too short or lacked depth 25% 38% 0.17
Comparison with ChatGPT or Claude, 1 to 5 3.23 2.89 0.49
E Details of Part A
E.1 Model Selection
Human and LLM Mental Health Conversations pairs questions from users with reference answers. We used 3,307
examples for training and 200 for evaluation, split at random with a fixed seed. We fine-tuned Gemma 3 with 4B and
27B parameters, Qwen 3 with 30B parameters and GPT-oss with 20B parameters. We also evaluated GPT-oss with 120B
parameters and DeepSeek V3 without fine-tuning. Fine-tuning raised the overlap scores of most models but lowered
BLEURT for all of them (Table 10). Gemma 3 27B changed little on any metric. When we read the outputs, the other
fine-tuned models produced replies that were unresponsive, incoherent or at times inappropriate.
Manuscript submitted to ACM

20 Garg et al.
Table 9. Reference-based metrics in the first tier. Each metric compares a generated reply with the reference reply for the same input.
Higher scores mean greater similarity.
Metric Type What it measures
BLEU Wording overlap Share of word sequences of up to four words in the reply that also
occur in the reference, with a penalty for short replies
ROUGE-L Wording overlap Length of the longest sequence of words that occurs in the same order
in both texts, with gaps allowed
METEOR Wording overlap Matches of single words, including word stems and synonyms, with a
penalty when matched words appear in a different order
BERTScore Meaning similarity Similarity of the words in the two texts, based on representations from
the language model BERT that depend on the surrounding words
BLEURT-20 Meaning similarity Score from a model based on BERT and trained to predict human
ratings of how well a text matches a reference
E.2 Training Corpus
Requests for data from two earlier caregiver studies were unsuccessful, which led us to generate the corpus. Its gender
balance corrects the skew toward women that Dang’s thesis reports as a limitation [ 18]. We removed exact and near
duplicates with MinHash, a method that estimates how much two texts overlap. Pairs with an estimated Jaccard
similarity of 0.70 or more counted as near duplicates. An audit confirmed that the assistant always played the supportive
companion. A rule-based pass then removed punctuation habits typical of LLM output, such as em dashes. A second
audit found no near duplicates between the training and held-out sets. The rebuilt examples with retrieved context left
the other 8,450 training examples and the held-out split unchanged.
E.3 Retrieval Store
The first store held 1,347 vectors, made of 1,293 document chunks and the 54 interview passages. For the interview
passages, the caregiver’s words form the retrievable text. One career guide supplied 1,019 of the 1,293 document chunks,
about 76%. Many retrievals therefore returned general career advice whatever the user had asked. The second store
capped this guide at 300 chunks, for 771 vectors in total. It added five sources on respite care, self-care, dementia
behaviour, the emotional side of caregiving and long-distance caregiving. All retrieval conditions reported in this paper
use labeled passages.
E.4 Self-Retrieval in Evaluation
Gated retrieval at first appeared to raise the single-response score of Carebot by 0.16 points. The 54 real evaluation
inputs are also passages in the store, so retrieval often returned the user’s own message. After we excluded these
self-matches, the apparent gain shrank to 0.04 points. Production users are not passages in the store, so this artifact
affects evaluation and not the deployed system.
E.5 Tables
Table 2 reports the age, gender, occupation and use of AI chatbots of the 90 participants in the analysis set, by condition.
Table 10 reports the model selection experiment. Table 3 reports the reference-based metrics for the four conditions on
the 500 synthetic held-out examples.
We report these scores for completeness, since reply length confounds them.
Manuscript submitted to ACM

AI Companions for Young Adult Caregivers 21
Table 10. Model selection on Human and LLM Mental Health Conversations. Parameters give model size. Higher is better for every
metric.
Model Parameters BERTScore BLEU ROUGE-L METEOR BLEURT
Gemma 3, base 4B 0.5014 0.0020 0.0970 0.2384 0.4708
Gemma 3, fine-tuned 4B 0.5297 0.0035 0.1491 0.2122 0.4493
Gemma 3, base 27B 0.5023 0.0029 0.0980 0.2439 0.4633
Gemma 3, fine-tuned 27B 0.5020 0.0024 0.0982 0.2427 0.4627
Qwen 3, base 30B 0.4984 0.0020 0.1010 0.2361 0.4666
Qwen 3, fine-tuned 30B 0.5352 0.0039 0.1535 0.1979 0.4644
GPT-oss, base 20B 0.4720 0.0014 0.0875 0.2219 0.4238
GPT-oss, fine-tuned 20B 0.4944 0.0005 0.1136 0.2031 0.3427
GPT-oss, base 120B 0.4780 0.0011 0.0761 0.2142 0.4159
DeepSeek V3, base 671B 0.5176 0.0050 0.1240 0.2480 0.4772
Manuscript submitted to ACM
# An AI Teaching Assistant for Motion Picture Engineering

**Authors**: Deirdre O'Regan, Anil C. Kokaram

**Published**: 2026-04-06 13:30:11

**PDF URL**: [https://arxiv.org/pdf/2604.04670v1](https://arxiv.org/pdf/2604.04670v1)

## Abstract
The rapid rise of LLMs over the last few years has promoted growing experimentation with LLM-driven AI tutors. However, the details of implementation, as well as the benefit in a teaching environment, are still in the early days of exploration. This article addresses these issues in the context of implementation of an AI Teaching Assistant (AI-TA) using Retrieval Augmented Generation (RAG) for Trinity College Dublin's Master's Motion Picture Engineering (MPE) course. We provide details of our implementation (including the prompt to the LLM, and code), and highlight how we designed and tuned our RAG pipeline to meet course needs. We describe our survey instrument and report on the impact of the AI-TA through a number of quantitative metrics. The scale of our experiment (43 students, 296 sessions, 1,889 queries over 7 weeks) was sufficient to have confidence in our findings. Unlike previous studies, we experimented with allowing the use of the AI-TA in open-book examinations. Statistical analysis across three exams showed no performance differences regardless of AI-TA access (p > 0.05), demonstrating that thoughtfully designed assessments can maintain academic validity. Student feedback revealed that the AI-TA was beneficial (mean = 4.22/5), while students had mixed feelings about preferring it over human tutoring (mean = 2.78/5).

## Full Text


<!-- PDF content starts -->

1
An AI Teaching Assistant for Motion Picture
Engineering
Deirdre O’Regan, Anil C. Kokaram,Senior Member, IEEE
The rapid rise of LLMs over the last few years has promoted growing experimentation with LLM-driven
AI tutors. However, the details of implementation, as well as the benefit in a teaching environment, are
still in the early days of exploration. This article addresses these issues in the context of implementation
of an AI Teaching Assistant (AI-TA) using Retrieval Augmented Generation (RAG) for Trinity College
Dublin’s Master’s Motion Picture Engineering (MPE) course. We provide details of our implementation
(including the prompt to the LLM, and code1), and highlight how we designed and tuned our RAG
pipeline to meet course needs. We describe our survey instrument and report on the impact of the AI-TA
through a number of quantitative metrics. The scale of our experiment (43 students, 296 sessions, 1,889
queries over 7 weeks) was sufficient to have confidence in our findings. Unlike previous studies, we
experimented with allowing the use of the AI-TA in open-book examinations. Statistical analysis across
three exams showed no performance differences regardless of AI-TA access (p >0.05), demonstrating
that thoughtfully designed assessments can maintain academic validity. Student feedback revealed that
the AI-TA was beneficial (mean = 4.22/5), while students had mixed feelings about preferring it over
human tutoring (mean = 2.78/5).
©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any
current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new
collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other
works.
Deirdre O’Regan and Anil C. Kokaram are with the Department of Electronic and Electrical Engineering, Trinity College
Dublin, Ireland. (e-mail: dregan@tcdie; anil.kokaram@tcd.ie).
This work was sponsored by the HEA Human Capital Initiative (HCI) Computational and Data-Centric Engineering Programme
(C4-WP4). We also acknowledge the generous support of YouTube to the work of our Sigmedia group.
1https://github.com/sigmedia/ai-teaching-assistant
April 7, 2026 Accepted for publication in IEEE Signal Processing MagazinearXiv:2604.04670v1  [eess.IV]  6 Apr 2026

2
INTRODUCTION: EMBRACINGTHELLM
Since the rise of transformer-based Large Language Models (LLMs) [1], AI chatbots like OpenAI’s
ChatGPT2have gained widespread popularity among students for academic support. Rather than resisting
what has become widespread student practice, educators have an opportunity to integrate AI thoughtfully.
By developing course-specific AI Teaching Assistants (AI-TAs), instructors can harness the benefits of
LLMs, address generic AI chatbots’ limitations, and foster valuable discussions about AI in the classroom.
What Is an AI-TA?
An AI-TA (or AI tutor) is a conversational system customized for a specific course or subject domain.
Typically LLM-driven, well-designed AI-TAs provide on-demand support with pedagogical guardrails
that promote critical thinking rather than simply dispensing answers. A popular implementation approach
employs Retrieval Augmented Generation (RAG) [2], a technique widely adopted for educational AI
applications [3]. RAG addresses a key limitation of LLMs: they can only generate responses based on
training patterns, which may be outdated or lack domain-specific knowledge. As Figure 1 shows, the RAG
pipeline retrieves relevant excerpts from instructor-curated materials, which the LLM uses to generate
responses with source citations. This grounds AI-TA responses in instructor-curated materials rather than
generic internet sources, and the citations let students trace AI-generated answers back to authoritative
course content, promoting AI literacy alongside knowledge acquisition.
RAG-Based 
AI 
Teaching 
Assistant 
(AI-TA)
LLM
Conversational 
UI
Course 
Materials
query, 
relevant 
context
response 
with 
source 
citations
Student
query
query
response
with 
source
citations
Fig. 1. A typical RAG-based AI-TA: Based on the student’s query, the AI-TA searches course materials, retrieves relevant context
(i.e., excerpts plus source metadata), and forwards the query plus context to an LLM which generates a response grounded in
the materials and including source citations.
An AI-TA for Motion Picture Engineering (MPE)
MPE is a Master’s course at Trinity College Dublin preparing students for careers in cinematic post-
production, visual effects, and video streaming. Building on Digital Signal Processing (DSP) foundations,
2https://chatgpt.com
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

3
it covers video manipulation, compression, and transcoding, while exploring neural network-driven ad-
vances in the field. Students gain hands-on experience with Foundry’s industry-standard Nuke software3
and hear from guest lecturers at Netflix, YouTube, Meta, Sky, VideoLAN, and others. MPE attracts
students with a wide technical background, ranging from Deep Learning and DSP practitioners to complete
newcomers in both fields. Its assessment structure of take-home assignments and online open-book exams
makes it vulnerable to LLM disruption. Instead of resisting this trend, in Spring 2025 we embraced it
by deploying a RAG-based AI-TA for MPE (43 students, 296 sessions, 1,889 queries over 7 weeks).
While several institutions have previously deployed RAG-based AI tutors [4]–[7], detailed implemen-
tation guidance remains limited. This article guides educators through our implementation approach,
addressing needs that may resonate—from navigating institutional constraints and pedagogical prompt
design to evaluating effectiveness—with code1and survey instrument, facilitating replication and adapta-
tion. We investigated: Would students benefit from it? Would they prefer it to a human teaching assistant?
What would happen if we allowed its use in an open-book exam?
Student feedback revealed that the AI-TA was beneficial (mean = 4.22/5), and its responses helpful
(mean = 4.19/5), while students had mixed feelings about preferring it over human tutoring (mean
= 2.78/5). Mixed-methods evaluation uncovered important nuance: students reported positive personal
experiences yet expressed concerns about its general use in assessments. Statistical analysis across
three open-book exams showed no performance differences regardless of AI-TA access (p >0.05),
demonstrating that assessments designed specifically for AI-TA access can maintain academic integrity.
Other key implementation insights include: (1) custom AI-TAs provide greatest value when accessing
specialized knowledge unavailable in general tools like ChatGPT; (2) Microsoft Azure-based deployments
effectively address institutional data governance and compliance requirements; (3) pedagogically informed
prompt engineering proved effective in practice; (4) students strategically concentrate usage around
assessments rather than maintaining steady engagement; (5) anonymity, while ethically desirable, limits
opportunities for personalized support and iterative improvement based on student feedback.
RAG ‘N’ ROLL: OURIMPLEMENTATION
The RAG pipeline has two phases: (1) indexing—source materials are chunked, converted to vector
embeddings, and stored in a searchable database; (2) query time—the query is similarly embedded and
used to retrieve relevant chunks, which provides context to the LLM. The LLM generates a response
grounded in this context, with source citations. This approach offers key educational advantages including
3https://foundry.com/products/nuke
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

4
domain specificity without LLM retraining, transparency through verifiable citations, and widely purported
hallucination4reduction. We now turn to the specifics of our implementation, highlighting the critical
design choices that shaped it.
Platform Selection and Architecture
While numerous off-the-shelf RAG solutions exist (e.g. OpenAI GPTs, Agenthost, Open-Custom-
GPT5), our particular requirements presented unique challenges: preserving students’ anonymity, safe-
guarding their chat conversations in case of accidental private information disclosure, and complying with
Trinity’s data protection policies. After careful analysis, we decided to custom-build our RAG-based AI-
TA using components from Microsoft Azure and Microsoft Foundry6, for the following reasons:
•Data protection and compliance:Azure is Trinity’s approved enterprise cloud platform. Microsoft’s
Azure OpenAI service offers Azure-hosted versions of OpenAI’s models with configurable content-
safety filters, customer-controlled data residency and erasure, and a guarantee that customers’ data
won’t be used for model training.
•Development and customization:Microsoft Foundry provides RAG-specific tools and templates
that accelerate development, while Azure provides secure infrastructure for hosting a custom-built
web client, facilitating iterative refinement based on student feedback.
Figure 2 illustrates both our AI-TA’s architecture, and its student query workflow. The system comprises
five main parts, each involving key implementation decisions:
1)AI Models:We chose Azure OpenAI’s gpt-4o-mini LLM for its balance of capability and cost-
effectiveness, along with Azure OpenAI’s text-embedding-ada-002 for vectorization. While an
embedding model capable of image understanding may have benefited MPE’s visual content,
available options required third-party services raising data protection concerns.
2)Azure AI Search:Provides “hybrid + semantic” search combining vector and keyword-based (i.e.,
hybrid) retrieval across the indexed course materials, with semantic re-ranking of search results.
We adopted Microsoft Foundry’s default content chunking settings, prioritizing rapid deployment
over extensive optimization. This usefully yielded slide-level citation granularity for lecture slide
PDFs. We configured k=10 (retrieve the 10 most relevant results) based on exploratory testing
4LLMs “hallucinate” when they generate text that sounds plausible but is factually incorrect or unsupported by real data
5OpenAI GPTs: https://openai.com/index/introducing-gpts/, Agenthost: https://agenthost.ai, Open-Custom-GPT: https://github.
com/SamurAIGPT/Open-Custom-GPT
6Microsoft Azure: https://azure.microsoft.com/, Microsoft Foundry: https://ai.azure.com
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

5
Azure 
OpenAI 
(1)
Azure 
AI 
Search
(2)
Microsoft 
Foundry
Custom 
Web 
Application 
(Azure 
Web 
App)
(5)
Azure 
SQL 
Database
(4)
query
formatted 
response
with 
citations
query, 
chat 
history
session 
data, 
chat 
history
Azure 
Prompt 
Flow
(3)
Preprocessing
Steps
Index 
Lookup
Generate 
Context
Prompt 
LLM
query
search 
results
Prompt 
Template
{date_time}
{context}
{chat_history}
{query}
response
with 
citations
prompt
response
with 
citations
LLM
gpt-4o-mini
prompt
query, 
date 
& 
time,
chat 
history
Embedding 
Model
text-embedding-ada-002
context
search 
results
Student
Hybrid 
Index 
(Course 
Content)
query
Embed 
Query 
& 
Extract 
Keywords
Retrieve
Search 
Results
Fig. 2. Our AI-TA’s implementation architecture and core student workflow. To use this system, create a Project in Microsoft
Foundry, and from there: (1) deploy Azure OpenAI models; (2) upload course materials and deploy an Azure AI Search instance
to create a Hybrid Index; (3) deploy an Azure Prompt Flow. In Microsoft Azure, deploy: (4) an Azure SQL Database; (5) our
custom web application as an Azure Web App. Our web application code, Prompt Flow, Prompt Template, and detailed setup
instructions are available on GitHub1. You will need to adapt our Prompt Flow and Prompt Template for your specific needs.
with representative queries, finding this provided good precision-recall balance for MPE’s content
density.
3)Azure Prompt Flow:Designed using Microsoft Foundry, this orchestrates the RAG pipeline which
manages query processing, context retrieval, Prompt Template population, and LLM inference.
Figure 2 shows the Prompt Flow we designed specifically for MPE. Preprocessing steps address
two unexpected challenges: (a) content-safety filter false positives (e.g., “How do I install Nuke”
triggered violent content detection), which we mitigated through phrase replacement; (b) time-
dependent queries (e.g., exam schedules, deadlines) required capturing the current date and time,
which we injected into the Prompt Template with instructions for temporal reasoning. Once con-
figured, we deployed the Prompt Flow on an Azure Virtual Machine via Microsoft Foundry, which
provided a secure endpoint for application integration.
4)Azure SQL Database:Stores students’ anonymous session data and conversation history, enabling
anonymous telemetry to protect student privacy while facilitating cohort-level analysis.
5)Custom Web Application:Provides anonymous log in, a conventional chat interface with mo-
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

6
bile responsiveness, configurable-length conversation history within browser sessions (limited to
the 10 most recent exchanges for our MPE deployment), markdown-rendered responses, LaTeX
mathematics display, and one-click code copying. This was deployed as an Azure Web App.
Our Prompt and Its Pedagogical Foundations
Our AI-TA was designed to operationalize elements of dialogic pedagogy [9] through explicit prompt
instructions emphasizing collaborative knowledge construction. The Prompt Template (available on GitHub7)
directed the LLM to “encourage deeper understanding by posing follow-up questions” such as “Are you
interested in the mathematical formula or the practical application?” and to guide students “step-by-step
instead of just stating the answer” while providing “worked examples where relevant”. For multi-turn
conversations, it instructed the LLM to “maintain conversation flow by recalling previous interactions”
and “summarize key points before continuing” when students built on earlier questions.
The Prompt Template and web application were designed as a cohesive system to address cognitive load
and self-directed learning: the prompt mandated full-path, granular source citations (e.g., slide- or page-
level for precise traceability) and markdown/LaTeX formatting, and the application rendered the formatted
content faithfully while maintaining conversation history for incremental knowledge construction.
Operational Workflow
Figure 3 illustrates the lecturer’s workflow for ingesting course materials into the AI-TA’s Hybrid Index.
For MPE, course materials included lecture slides, announcements, Matlab and BlinkScript8scripts, along
with audio transcriptions from lectures. Although Spring 2025’s MPE lectures were recorded in Microsoft
Teams, which has built-in transcription, we used OpenAI’s Whisper Turbo9instead, as it appeared to
produce better results in informal testing.
MPE DEPLOYMENT: BUILDITANDTHEYWILLCOME?
For the first 6 weeks of the MPE semester, we prepared the AI-TA, uploading and indexing course
materials, conducting exploratory testing and tuning. We launched it to students on week 7 to coincide with
Trinity’s lecture-free Reading Week. All 43 MPE students received anonymous access via Blackboard10
7https://github.com/sigmedia/ai-teaching-assistant/blob/main/tools/azure-prompt-flow-examples/mpe-experiment-2025/chat
with context.jinja2
8Nuke’s proprietary scripting language
9https://github.com/openai/whisper
10Trinity’s Learning Management System (LMS)
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

7
Azure 
Open 
AI
Microsoft 
Foundry 
UI
Lecturer
Azure 
AI 
Search
Course 
Materials 
(Multi-Modal)
Transform 
Course 
Materials
Upload 
Data
Create 
New 
Index
Extract 
& 
Chunk 
Content
Embedding 
Model
text-embedding-ada-002
Embed 
Chunks 
& 
Extract 
Keywords
text-based 
documents
Create 
Hybrid
Search 
Index 
Wait
Microsoft 
Foundry 
UI
Connect 
Prompt 
Flow 
to 
Index
Hybrid 
Index 
(Course 
Content)
Test 
& 
Deploy 
Prompt 
Flow
Fig. 3. Workflow for ingesting course materials: First, transform all course materials to text-based documents, then using
Microsoft Foundry’s UI: upload the documents, create a new index, connect the Prompt Flow to the index, and test and deploy
the Prompt Flow.
alongside a Participant Information Leaflet (PIL) explaining that AI-TA use was voluntary and anonymous,
and that their grades would not be affected by whether they chose to use it or not11. It also cautioned
against disclosing private information, and to verify all responses.
The AI-TA remained online for 7 weeks until the end of semester and was updated 2–4 times weekly
with lecture materials and transcripts following MPE’s delivery schedule. Figure 4 shows daily student
engagement and overall statistics.
Initial engagement was low, but usage increased following an in-class reminder (Class Practice for
Exam 2), subsequently correlating with assessment deadlines and exams. Assignment 1, which involved
implementing video matting techniques in Nuke and reporting on results in an IEEE-style paper, generated
queries about report-writing and structuring. For Exam 2, an invigilated open-book quiz delivered via
Blackboard, students were permitted to query the AI-TA alongside their notes, but not other AI chatbots
(e.g., ChatGPT). The exam period produced a usage spike: 42 active sessions accounting for 564 (78%)
of that day’s 724 queries. While session counts don’t map directly to individual students, the volume
suggests widespread AI-TA use during the exam. In week 6, students sat Exam 3—another Blackboard
quiz without AI-TA access but with notes permitted. Preceding days generated queries about likely exam
questions, which the system appropriately declined to predict while offering study guidance instead.
Assignment 2’s deadline in week 7 produced a modest final usage peak. Open-book Exam 1 is not
shown in Figure 4, as it was administered prior to the AI-TA’s launch.
11Students provided informed consent by checking a box on the AI-TA’s login page acknowledging they had read the PIL, as
approved by Trinity’s Research Ethics Review Process
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

8
7140316232423812101473935105724
8491720553136073321592862045186712010180282670330
Student Queries Per Day
Class Practice for Exam 2(Fri)Assignment 1 Due Date(Tues)Exam 2(Wed)Graded Lab(Thurs)Class Practice for Exam 3(Thurs)Exam 3(Fri)Assignment 2 Due Date(Wed)Week 1Week 2Week 3Week 4Week 5Week 6Week 7Total Student QueriesTotal AI-TA ResponsesAvg Queries Per Student2961,8891,87643.9Total SessionsAvg Messages Per Session12.7Student Queries Per Day 
Fig. 4. AI-TA engagement: Number of MPE student queries per day over 7 weeks. Annotations show correlations between
usage peaks and assessment-related events. Overall engagement statistics are displayed above the chart.
STUDENTOUTCOMES ANDPERCEPTIONS
To assess the AI-TA’s impact, we examined both student outcomes and perceptions by comparing
student performance across three open-book exams administered under different AI-TA availability con-
ditions and analyzing student perceptions through a course exit survey. The study cohort comprised 43
students enrolled in MPE during Spring 2025. Demographics reflected MPE’s typical composition: 27
male students (63%), 16 female (37%); aged 20–24: 30 (70%), 25–29: 12 (28%), 30+: 1 (2%); prior
backgrounds in Engineering: 26 (60%), Computer Science: 16 (37%), other fields: 1 (2%). 36 of the 43
students (84%) answered the course survey.
Student Performance
To evaluate whether AI-TA access affected exam performance—a key concern for academic integrity,
we compared student scores across three open-book exams (all invigilated quizzes delivered via Black-
board). In all three exams, students could reference their own notes but not other AI chatbots. The three
conditions were: Exam 1 (administered before AI-TA launch), Exam 2 (AI-TA permitted during exam),
and Exam 3 (AI-TA not permitted during exam). Exam 2 was designed with AI-TA access in mind,
emphasizing higher-order thinking skills that could not be directly answered through AI queries alone.
Mean exam scores were: Exam 1 (µ= 53.5%), Exam 2 (µ= 52.5%), and Exam 3 (µ= 52.3%).
We performed a paired permutation test with the Student’s t-test statistic [10], accounting for individual
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

9
student differences while controlling for multiple comparisons. All comparisons yieldedp >0.05(Exam
1 vs 2:p= 0.94; Exam 1 vs 3:p= 0.92; Exam 2 vs 3:p= 0.94), indicating no significant performance
differences across the three exam conditions. These results suggest that with appropriately designed
assessments, AI-TA availability does not compromise open-book exam validity.
Survey Results
The course exit survey comprised 24 mandatory questions, with 6 assessing student perceptions of the
AI-TA. Our survey instrument and anonymized data are publicly available on GitHub12. We compare
our results with the MoodleBot study [4], which implemented a RAG-based AI chatbot for a Bachelor’s
Computer Science course (Database & Information Systems) at a German university and surveyed 46
participants with a 65% response rate. We designed three questions specific to our deployment alongside
three adapted from their survey. Our students received 1% of their overall course grade as incentive
to complete the survey. Questions Q1–Q5 employed 5-point Likert scales with an additional “Not
Applicable” option:
Q1 I found the AI Teaching Assistant beneficial in supporting my learning in the course.
Q2 The AI Teaching Assistant’s answers to my questions were helpful and informative.
Q3 The AI Teaching Assistant enhanced my overall experience during the open book exam compared
to relying solely on printed notes or memory (i.e. during the second quiz held March 19th, 2025).
Q4 Given a choice, I would prefer interacting with the AI Teaching Assistant rather than with a real
tutor.
Q5 I would continue to use the AI Teaching Assistant in the future as part of other courses.
Q6 Tell us more about what you thought of the AI Teaching Assistant. Do you think it should be
allowed in open book exams? Did it help you to understand any topics which were otherwise
difficult to grasp from the lecturer’s explanations? If you didn’t use it, why not? Is there anything
else you want to mention about it?
Questions Q2, Q4, and Q5 were adapted from the MoodleBot study with key modifications: we added
a “Not Applicable” (N/A) option, and our Q4 options ranged from “Never” to “Always” (testing our
hypothesis that students’ preferences for the AI-TA over a real tutor might be situational). We assigned
numerical scores (Strongly Disagree/Never = 1; Strongly Agree/Always = 5) to Q1–Q5 responses to
enable quantitative comparison with the MoodleBot study, calculating population mean (µ) and standard
deviation (σ) while excluding N/A responses.
12https://github.com/sigmedia/ai-teaching-assistant/tree/main/tools/evaluation/exit-survey/mpe-experiment-2025
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

10
Results indicated strongly positive perceptions: Q1 (µ= 4.22, σ= 0.79, N= 36), Q2 (µ= 4.19, σ=
0.81, N= 36), Q3 (µ= 4.44, σ= 0.83, N= 36), and Q5 (µ= 4.08, σ= 0.86, N= 36). Q4 showed
mixed responses (µ= 2.78, σ= 1.08, N= 32), suggesting students view the AI-TA as complementary
to rather than a replacement for human instruction, appreciating one option or the other depending on
the situation. Figure 5 visualizes the Likert distributions for questions Q1–Q5.
“I think its answers are not as good as chatgpt”“…It was especially useful when I needed quick explanations or different ways of understanding topics that didn’t quite click during lectures…”
“…While it's great for learning and revision, allowing it during exams could blur the line between testing understanding and just getting the AI to solve things for me.”“…The AI assistant helps me quickly locate relevant knowledge points in the slides, making it easier for me to review the material.”“Yes, it should be allowed in open book exams. It helped me to understand Markov Random Fields, which was confusing for me at the beginning.”
Fig. 5. Student perceptions of the AI-TA from the course exit survey (question texts summarized for brevity). We calculated
the mean,µ, and standard deviation,σ, for Q1–Q5.Ndenotes the number of respondents to each question (excluding “Not
Applicable” responses). Interesting quotes from open-ended Q6 responses are also shown.
Table I compares our results with the MoodleBot study for equivalent questions (Q2, Q4, Q5). All
differences in means and standard deviations were below 6%, demonstrating consistency across the two
implementations despite differences in course content, institutional, and educational context.
To extract richer contextual insights from open-ended Q6 responses, we assigned a sentiment rating to
each response and tallied approval for AI-TA availability in open-book exams. Student quotes in this article
were lightly edited for spelling/grammar (unedited responses are available as supplementary material12).
Sentiment distribution was as follows: Positive 30.6% (11), Positive with Caveats/Reservations 38.9%
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

11
TABLE I
COMPARISON OFAI-TAWITHMOODLEBOTSURVEYRESULTS
Question (text summarized) AI-TA MoodleBot Difference
µ±σ N µ±σ N∆µ(%)∆σ(%)
Q2: Answers were helpful & informative4.19±0.81364.4±0.7730 -4.8 +5.2
Q4: Would prefer over a real tutor2.78±1.08322.7±1.1430 +3.0 -5.3
Q5: Would use in future courses4.08±0.86364.23±0.8530 -3.5 +1.2
∆µand∆σrepresent percentage differences in population means and standard deviations respectively.
(14), Neutral/Balanced/Limited Use 13.9% (5), and Negative/Skeptical 16.7% (6). Exam use approval
was distributed as follows: Yes 33.3% (12), No 38.9% (14), Not Clear 27.8% (10).
Students appreciated the AI-TA for clarifying complex concepts not fully grasped in lectures (e.g.,
Markov Random Fields), and for quickly locating specific information without hunting through slides,
validating the usefulness of granular source citations. However, four students noted answer quality
concerns ranging from vague responses to incorrect information, while several expressed concerns about
dependency, over-reliance, or reduced motivation to learn deeply. Students’ cautious stance on exam
approval contrasts with Q3 responses where 89% agreed/strongly agreed that it enhanced their exam
experience, reflecting nuanced concerns about broader AI adoption in assessment contexts despite pos-
itive individual experiences. Students who expressed reservations worried that AI-TA availability could
undermine motivation to master material or provide unfair advantages to less conscientious students, with
one candidly noting using it “mainly so that I’m not disadvantaged by everyone else using it.” Students
proposed mitigation strategies including limiting prompts per student, deducting marks per query, or
offering bonus points for non-usage.
CROSS-STUDYCOMPARISON, LESSONSLEARNED ANDRECOMMENDATIONS
Our seven-week deployment revealed the educational potential and practical challenges of AI-TAs. We
compare our experience with similar systems, distill key lessons learned, limitations, and offer guidance
for educators considering similar deployments.
Comparing Implementations Across Contexts
We compare our AI-TA with five other AI tutors: MoodleBot [4], CS50 Duck [5], HiTA [6], OwlMentor
[7], and Smeaton [8] across diverse contexts, including undergraduate Computer Science (CS50 Duck,
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

12
HiTA, MoodleBot), Master’s Educational Technology (OwlMentor), undergraduate Business & Marketing
(Smeaton), and Master’s MPE (ours). Most deployed during active courses while MoodleBot, developed
for a 700+ student course, was evaluated post-course with 46 students who had previously completed it.
Cohort sizes varied substantially: CS50 Duck (50,000+ users), HiTA (400), Smeaton (143), MoodleBot
(46), our AI-TA (43), OwlMentor (16).
Engagement Patterns and Performance Outcomes:Usage varied substantially: HiTA averaged
51.9 queries/student over Fall 2023 semester, our AI-TA 43.9 queries/student over 7 weeks, Smeaton
19.5 queries/student over 12 weeks, MoodleBot 3.5 queries/student in post-course evaluation. CS50
Duck reported 50,000+ users with 5–15 prompts per active user daily during Fall 2023, though direct
comparison is complicated by its scale and continuous availability across multiple courses. Despite
differences in deployment durations and contexts, a consistent pattern emerged: strategic concentration
around assessments rather than steady engagement. Our AI-TA received 38.3% of total queries during
one exam, Smeaton documented 26.2% in 24 hours before the final exam, and OwlMentor showed
declining mid-course engagement (150 to 71 events) with renewal before exams (121 events). Our study
uniquely examined AI-TA access during assessments, finding no significant performance differences
(p >0.05) across three exams with varied AI-TA availability. HiTA reported narrowing performance gaps
post-deployment, though with potential confounds, while OwlMentor found usage frequency showed no
significant correlation with learning outcomes (p= 0.581).
Student Perceptions:Positive perceptions emerged across implementations despite varying survey
instruments: HiTA (97.9% beneficial), CS50 Duck (88% always/frequently helpful), our deployment
(83.4% agreed/strongly agreed the AI-TA was beneficial for learning; 86.1% agreed/strongly agreed
responses were helpful and informative). Our survey’s adaptation of MoodleBot questions enabled some
direct comparison (see Table I). OwlMentor revealed declining Perceived Usefulness during deployment,
with higher self-efficacy students showing lower usage—suggesting AI tools may particularly benefit
students lacking initial confidence.
AI Model Choices:Most RAG-based systems converged on OpenAI’s GPT family with text-embedding-
ada-002 for vectorization. CS50 Duck and our AI-TA hosted models on Azure (CS50: GPT-4; ours:
GPT-4o-mini), while others accessed OpenAI’s API directly. Smeaton’s deployment diverged, adopting
Agenthost with incrementally fine-tuned GPT-3.5 instead of RAG (we include this as a notable example
of active-course AI-TA deployment using an alternative architectural approach).
Costs:Our 49-day deployment cost approximately EUR 841 (EUR 0.45/query), with EUR 6.23
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

13
(EUR 0.003/query) of this attributed to token-based AI usage13. Therefore, fixed-tier Azure infrastructure
(virtual machine, search engine, web application plan, database, etc) dominated our expenses. Post-
deployment analysis suggests that we over-provisioned the virtual machine, with at least 40% cost reduc-
tion possible through right-sizing infrastructure in future deployments, with further economies possible
through system tuning and extension for multi-course deployment. Only CS50 Duck and MoodleBot
reported costs for comparison: CS50 Duck reports $0.05/query (cost components not detailed); MoodleBot
reports $0.01–$2.33/message accounting for API token costs only.
Other Lessons Learned
Competing With Commercial Chatbots:Despite significant prompt engineering and retrieval tuning
effort, informal testing often revealed ChatGPT’s responses to be as good as our AI-TA’s. ChatGPT is
improving rapidly and is now extremely difficult to beat on public knowledge, even for specialized topics
like MPE. However, RAG-based AI-TAs excel in two scenarios: (1) private institutional knowledge (e.g.,
“when is my 3rd quiz scheduled?”), and (2) recent specialized information that commercial chatbots have
not yet incorporated. For instance, our AI-TA knew updated FFmpeg14command-line syntax from course
materials while ChatGPT did not, which was critical since students needed this for their assignments.
Pros and Cons of Anonymity:Smeaton, OwlMentor, and our AI-TA employed usage anonymity.
Smeaton found students asked questions they wouldn’t ask in lectures, with similar benefits noted in our
survey where one student wrote: “Sometimes I have loads of dumb questions, that’s when it becomes more
helpful.” This judgment-free space was encouraging given students’ reticence to speak up in class this
year. However, anonymity also prevented collaborative discussions on AI-TA features and improvements,
and made targeted individual support impossible. Future work will explore whether aggregate query
analysis can identify challenging concepts for in-situ teaching adaptation or iterative course refinement.
Limitations
Technical:Our text-only RAG implementation prevented image understanding, which would have been
valuable for MPE’s visual content. We relied on exploratory testing rather than formal accuracy evaluation,
expecting that evaluation by the lecturer for a few key queries was sufficient. Post-deployment analysis
revealed that low-context queries from mid-conversation (e.g., “what is the main reason”) yielded poor
context retrieval results. The operational workflow for updating course materials required approximately
1 hour per update, representing significant administrative effort given 2–4 updates weekly for MPE.
13Our costs include some exploratory development with shared infrastructure
14http://ffmpeg.org
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

14
Study Design, Biases and Comparability:Our quasi-experimental design lacked random assignment
and control groups, precluding strong causal inferences that would be possible with a randomized
controlled trial. Our small scale (43 students) limits statistical power versus large implementations (CS50
Duck: 50,000+ users). Our survey used positively-worded statements (e.g., “enhanced my open-book
exam experience”) likely inflating agreement rates—a limitation shared with MoodleBot and OwlMentor.
Our 84% response rate may reflect incentive bias (1% grade for participation), while anonymous access
prevented correlating usage with outcomes. Our Master’s-level cohort (63% male; 70% aged 20–24; 97%
Engineering/Computer Science) limits generalizability. Deployment context differences further complicate
cross-study comparisons: our in-course deployment versus MoodleBot’s post-course volunteer evaluation
involve distinct participant motivations and selection effects.
Recommendations for Implementers
Platform Selection, Cost Optimization, and Data Privacy:Clarify your institution’s data governance
requirements early, as these determine available options. With flexible data policies, off-the-shelf platforms
like Agenthost (used by Smeaton [8]) may provide richer features and reduced operational burden. For
stricter data protection or residency (e.g., GDPR compliance), Azure-hosted solutions may be preferable,
and costs can be optimized by right-sizing fixed-tier services. Regardless of platform choice, consider
user-facing privacy measures. Analysis of 100 random AI-TA queries revealed students avoided entering
personally identifiable information; our system displayed “Do not disclose any information which could
identify an individual,” suggesting clear privacy notices effectively encourage data caution.
Anonymity Trade-offs:Consider anonymity’s pedagogical and practical implications. While a judgment-
free environment may encourage questions students wouldn’t ask publicly, anonymity complicates indi-
vidual follow-up and targeted support while preventing correlation of usage with outcomes.
Evaluation and Assessment Design:Employ experimental designs with control groups when feasi-
ble, and use mixed-methods evaluation combining quantitative and qualitative data. Report participant
demographics and deployment contexts to enable meaningful cross-study comparisons. When permitting
AI-TA access during assessments, design questions requiring critical evaluation, higher-order thinking,
or responses difficult to generate via AI prompting.
CONCLUSION
This article demonstrates that educators can successfully deploy RAG-based AI-TAs by addressing insti-
tutional, pedagogical, and technical considerations, with our implementation code and survey instrument1
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

15
enabling replication and adaptation. Statistical analysis across three open-book exams revealed no sig-
nificant performance differences regardless of AI-TA access (p >0.05), demonstrating that thoughtfully
designed assessments can maintain academic integrity. Students rated the AI-TA highly beneficial (mean
= 4.22/5), though expressed nuanced concerns about broader assessment use despite positive personal
experiences. We plan to extend our AI-TA to additional Engineering courses with improvements including
operational workflow automation, formal accuracy evaluation, assessment usage protocols, prompt flow
improvement for low-context queries, and aggregate query analysis to inform adaptive teaching. The
rapid evolution of LLM capabilities suggests exciting opportunities ahead for educators willing to engage
thoughtfully with these technologies.
ACKNOWLEDGMENTS
Many thanks to Trinity’s John Squires for support with Microsoft Foundry, to Mark Linanne, Vibhoothi,
Amir Bijedic, and Harun ˇSiljak for technical and ethics support, to Evelyn Fox and Colm Lawlor for
data protection recommendations, and to the Sigmedia group for testing the AI-TA.
REFERENCES
[1] A. Vaswani et al., “Attention is all you need,” inAdvances in Neural Information Processing Systems, 2017, pp. 5998–6008.
[Online]. Available: https://papers.nips.cc/paper/7181-attention-is-all-you-need
[2] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” inAdvances
in Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474. [Online]. Available:
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[3] Z. Li et al., “Retrieval-augmented generation for educational application: A systematic survey,”Comput.
Educ. Artif. Intell., vol. 8, pp. 100417, Jun. 2025, doi: 10.1016/j.caeai.2025.100417. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S2666920X25000578
[4] A. T. Neumann et al., “An LLM-driven chatbot in higher education for databases and information systems,”
IEEE Trans. Educ., vol. 68, no. 1, pp. 103–116, Feb. 2025, doi: 10.1109/TE.2024.3467912. [Online]. Available:
https://doi.org/10.1109/TE.2024.3467912
[5] R. Liu et al., “Teaching CS50 with AI: Leveraging generative artificial intelligence in computer science education,”
inProc. 55th ACM Tech. Symp. Comput. Sci. Educ.(SIGCSE), Portland, OR, USA, Mar. 2024, pp. 750–756, doi:
10.1145/3626252.3630938. [Online]. Available: https://doi.org/10.1145/3626252.3630938
[6] C. Liu, L. Hoang, A. Stolman, and B. Wu, “HiTA: A RAG-based educational platform that centers educators in the
instructional loop,” inArtificial Intelligence in Education. AIED 2024. Lecture Notes in Computer Science, vol. 14830, A.
M. Olney, I. A. Chounta, Z. Liu, O. C. Santos, and I. I. Bittencourt, Eds. Cham, Switzerland: Springer, 2024, pp. 405–412,
doi: 10.1007/978-3-031-64299-9 37. [Online]. Available: https://doi.org/10.1007/978-3-031-64299-9 37
[7] D. Th ¨us, S. Malone, and R. Br ¨unken, “Exploring generative AI in higher education: A RAG system to enhance student
engagement with scientific literature,”Front. Psychol., vol. 15, pp. 1474892, 2024, doi: 10.3389/fpsyg.2024.1474892.
[Online]. Available: https://doi.org/10.3389/fpsyg.2024.1474892
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine

16
[8] A. Smeaton, “Student use of an LLM incrementally fine-tuned to behave like a teaching assistant in busi-
ness and marketing,” inUsing GenAI in Teaching, Learning and Assessment in Irish Universities, A. E. Schalk
Quintanar and P. Rooney, Eds. Trinity College Dublin and University College Cork, 2025. [Online]. Available:
https://ucclibrary.pressbooks.pub/genai/chapter/ unknown -4/. doi: 10.33178/x841-2fsr.
[9] R. Alexander,A Dialogic Teaching Companion. London, UK: Routledge, 2020.
[10] G.-C. Pascutto, “bootstrap.py - resampling analysis for multiple comparisons,” Computer software, Version 1.0, GNU
Affero GPL v3, Feb. 2011. [Online]. Available: https://www.sjeng.org/ftp/bootstrap.py
April 7, 2026 Accepted for publication in IEEE Signal Processing Magazine
# Towards Requirements Engineering for RAG Systems

**Authors**: Tor Sporsem, Rasmus Ulfsnes

**Published**: 2025-05-12 13:30:44

**PDF URL**: [http://arxiv.org/pdf/2505.07553v1](http://arxiv.org/pdf/2505.07553v1)

## Abstract
This short paper explores how a maritime company develops and integrates
large-language models (LLM). Specifically by looking at the requirements
engineering for Retrieval Augmented Generation (RAG) systems in expert
settings. Through a case study at a maritime service provider, we demonstrate
how data scientists face a fundamental tension between user expectations of AI
perfection and the correctness of the generated outputs. Our findings reveal
that data scientists must identify context-specific "retrieval requirements"
through iterative experimentation together with users because they are the ones
who can determine correctness. We present an empirical process model describing
how data scientists practically elicited these "retrieval requirements" and
managed system limitations. This work advances software engineering knowledge
by providing insights into the specialized requirements engineering processes
for implementing RAG systems in complex domain-specific applications.

## Full Text


<!-- PDF content starts -->

Towards Requirements Engineering for RAG Systems
Tor Sporsem
tor.sporsem@sintef.no
SINTEF & NTNU
Trondheim, NorwayRasmus Ulfsnes
rasmus.ulfsnes@sintef.no
SINTEF & NTNU
Trondheim, Norway
ABSTRACT
This short paper explores how a maritime company develops and
integrates large-language models (LLM). Specifically by looking
at the requirements engineering for Retrieval Augmented Gener-
ation (RAG) systems in expert settings. Through a case study at
a maritime service provider, we demonstrate how data scientists
face a fundamental tension between user expectations of AI per-
fection and the correctness of the generated outputs. Our findings
reveal that data scientists must identify context-specific "retrieval
requirements" through iterative experimentation together with
users because they are the ones who can determine correctness.
We present an empirical process model describing how data sci-
entists practically elicited these "retrieval requirements" and man-
aged system limitations. This work advances software engineering
knowledge by providing insights into the specialized requirements
engineering processes for implementing RAG systems in complex
domain-specific applications.
KEYWORDS
Requirements Engineering, Retrieval Augmented Generation (RAG),
GenAI, RE4AI, case study, maritime industry
1 INTRODUCTION
Kat, a maritime engineer at Marcomp, has received a question from
a ship’s captain. Her job is to help Marcomp’s customers understand
and apply international regulations that ships must follow. This
time, instead of drafting the answer herself, she uses a recently
provided Large Language Model (LLM). She clicks the “generate
reply” button, and within seconds, a response appears on her screen.
Smiling, she says:
"Incredible right? It has searched through all the 500,000
previous answers we have given, found similar ones
and written a new answer based on all the old ones.
Now, let’s check if we can trust it"
She reads through the generated answer and notices this would
have been a great answer a few years ago. However, new rules have
been implemented, making the generated answer incorrect.
"Well, I know this rule is quite new and there are
probably no previous answers on similar questions
after the new rule came into effect."
Kat deletes the generated text and writes up her own answer.
This highlights a fundamental problem of "plugging in" an LLM
into an organization’s local data (knowledge base) – using a re-
trieval augmented generation (RAG) system. The knowledge base,
which contains 500,000 answers to questions about ships from the
past 15 years, holds no answers where the new rule has been in
effect. When the LLM is instructed to predict its answer basedon previous answers, it cannot identify which are outdated and
consequently generates incorrect answers.
There are already studies describing how software engineering
(see e.g. Kalinowski et al. [10]; Wan et al. [28]) and requirements
engineering (see e.g. Amershi et al. [2]; Kim et al. [11]) change when
developing AI systems. However, RAG is gaining in popularity
because of its pragmatic way of integrating LLM into organizations’
databases without having to train their own [ 4]. With this as a
backdrop, we ask the following research question: How do developers
elicit requirements for RAG systems?
In this short paper, we present preliminary findings from a case
study that shows how a team of developers and data scientists
implemented RAG in a maritime company.
2 BACKGROUND
2.1 Retrieval augmented generation
While LLMs excels at various tasks, including software engineering
[23] and across experimental applications in different fields [ 5],
there remains a gap in their training data that leads to inconsistent
results [ 7]. Re-training an LLM is impractical for small applications
due to the extensive time required (months) and smaller organi-
zations’ resource constraints [ 4]. RAG proposed by Facebook AI
Research in 2020, offers a potential solution that avoids retraining
costs while leveraging external data [ 15]. The RAG process involves
four key steps:
(1)Indexing - Converting existing knowledge into a machine-
understandable knowledge base, a vector database with em-
beddings, that are semantic representations of the words,
and their relation to other words in the documents.
(2)Retrieval - Finding relevant stored information in the knowl-
edge base when presented with an external query.
(3)Augmentation - Combining the incoming query with the
retrieved information from the knowledge base.
(4)Generation - Using the augmented information to produce
an LLM output.
Current literature on RAG primarily focuses on technical aspects,
such as selecting appropriate indexing, embedding strategies, and
retrieval technologies [ 25] (e.g., ScaNN). Some studies evaluate RAG
efficiency across use cases like QA, text generation, summarization,
and SW development and maintenance [ 3]. Another line of work
examines technical implementation challenges, identifying failure
points related to missing content, lack of relevant docs, or incom-
plete answers [ 4]. This research also highlights the need for ML
skills. While various ENG strategies and optimization approaches
are being explored, literature on RE and AI development indicates
that in expert domains, close user involvement is often necessary
[9; 27].arXiv:2505.07553v1  [cs.SE]  12 May 2025

EASE 2025, 17–20 June, 2025, Istanbul, Turkey Tor Sporsem and Rasmus Ulfsnes
2.2 Requirements Engineering for AI
While research on the development of RAG is still in its infancy,
research on how to develop ML systems has been extensively re-
searched in the last few years [ 1]. Similar to ML systems, when
developing RAG systems a fundamental requirement is that the
knowledge base is rich and preferably complete [ 4]. This means
that the LLM should be able to find relevant information in the
knowledge base on which to build its generation, and reduce halluci-
nation. Research on machine learning has shown that a knowledge
base with large amounts of data and a high degree of diversity in-
creases the chance that the ML system can handle a query [ 26]. For
example, if an ML system is designed for doctors, the knowledge
base needs to include a diverse variety of diagnoses and as many
characteristics of each diagnosis as possible [ 13]. A lack of diversity
means a risk of missing diagnoses, which can be critical [ 26]. The
number of examples in the knowledge base is thus not as significant
as the diversity in examples [ 26]. Moreover, this principle applies
across medical [ 13], human resources [ 24], or technical domains [ 2].
The quality and breadth of information determine how effectively
the system handles common and specialized queries.
Another fundamental challenge when engineering AI systems
lies in establishing clear correctness criteria for system outputs [ 9].
This difficulty stems from the inherent ambiguity in determining
what constitutes a "correct" prediction, as correctness can mani-
fest in various forms and interpretations [ 16]. Unlike systems with
binary or limited output possibilities, LLM outputs rarely have a
single definitive correct answer [ 22]. Even when addressing factual
questions with a RAG system with a trusted knowledge base, mul-
tiple valid expressions of the same information exist. Additionally,
the notion of correctness in these systems may evolve as real-world
conditions change over time. This variability in what constitutes
correctness presents significant challenges for developing evalua-
tion frameworks and benchmarks and performance benchmarks
for LLMs [ 6]. This leads to the question of who can judge if an
output of a RAG system is correct or not and if an output is "correct
enough" to be of value.
Research already demonstrates that eliciting requirements for
AI systems introduces unique challenges [ 10;26]. Given the black
box nature of most AI models, requirements engineers struggle to
specify precise requirements, often producing specifications consid-
ered ’too high-level’ or ’vague’ [ 1]. Rather than focusing on precise
initial requirements, it is recommended to first hypothesize possi-
ble outcomes from available data [ 28], then refining requirements
through experimentation [ 8;9]. However, we still lack empirical
industry cases documenting how developers and data scientists
elicit requirements for RAG systems.
3 METHODS
Study design & case description. The research literature on SE and
RE benefits from case studies that provide insights into how these
processes unfold in the industry. Case studies are particularly suit-
able for evidence-based recommendations for practitioners [ 19].
Following this, we embarked on a case study[ 17] following Mar-
comp’s AI software development and present initial findings from
one project.Table 1: Collected data
Data source Type of data
Idea Workshop 7 hours observation during workshop
Data Architect 7 hours observation & 2 interviews
AI Solution Engineer 3 interviews
8 Case handlers (users) 28 hours of observation & 4 interviews
Marcomp is a major maritime sector service provider with ap-
proximately 3,700 employees worldwide. Their primary business
involves verifying vessels’ compliance with regulations; success-
ful verification results in certificates that ships need for obtaining
marine insurance and operating internationally. As a customer of
Marcomp, shipping companies also have access to a 24/7 helpdesk
where they can get support from Marcomp’s top engineers on every-
thing from technical questions to regulatory questions. Marcomp
receives hundreds of questions every day from customers around
the world and has five offices in different time zones to operate the
help desk, with approximately 60 case handlers. Marcomp main-
tains an internal development department that creates software for
Marcomp employees and external customers.
Data collection and analysis. To completely understand the case,
we interviewed and observed developers and users (case handlers).
Interviews are a valuable source of information as they allow in-
depth conversation with experts. Still, it is only one of many po-
tential methods to collect data in field studies [ 14]. Interviews fall
in a category of methods that Lethbridge et al. [14] have labeled
inquisitive techniques, in that a researcher must actively engage
with interviewees to get information from them. A second category,
"observational techniques, " includes watching professionals at work.
Each approach has strengths and limitations; interview data may
be less accurate than observational data, while observation may
trigger the Hawthorne effect, where people modify their behavior
when they know they’re being watched [14].
In the summer of 2023, we observed an idea development work-
shop with managers, case handlers, and data scientists who decided
to explore if LLMs could be useful for case handlers in answering
customer questions. Then, a few months later, in late 2023, we ob-
served four case handlers to understand their work process and
software use. We spent three full days with each of them, observing
how they answered customer questions and observed their meet-
ings and lunch discussions. Additionally, we interviewed four other
case handlers. In early 2024, we interviewed two data scientists (a
Data Architect and an AI Solution Engineer) to understand their
approach and working methods. Again, in late 2024, we revisited
the four case handlers and data scientists to observe and interview
them after launching the RAG system. All data is summarized in
table 1.
For our preliminary data analysis presented in this paper, we
started with a grounded theory approach [ 20], combined with Sea-
man’s [ 18] guidelines for open-ended coding and memoing. Based
on this initial analysis, we noticed different phases emerge from
the process of developing the RAG system. We then looked at tem-
poral bracketing strategy from Langley [ 12] to guide the analysis
further. We identified five different phases (summarized in figure 2):

Towards Requirements Engineering for RAG Systems EASE 2025, 17–20 June, 2025, Istanbul, Turkey
1) Knowledge modeling and experimentation, 2) Retrieval strategy,
3) Retrievable data management, 4) Monitoring and operation, and
5) Expectation management.
4 RESULTS
The release of ChatGPT in November 2022 created tremendous
excitement among businesses, with managers impressed by its
capabilities. This wave also caught Marcomp. Their senior man-
agement was determined to prevent competitors from gaining an
edge by adopting this technology first. This led to middle managers
rapidly developing ideas for AI implementation, with funding for AI
projects becoming readily available. Fortunately for Marcomp, they
began building their AI expertise in 2017 by forming a data science
team to investigate the growing opportunities in Deep Learning.
4.1 Knowledge Modeling and experimentation
During a two-day workshop in 2023, case handlers’ managers and
data scientists developed the idea that LLMs could extract value
from Marcomp’s accumulated data. They recognized that their
database of 500,000 preserved case handler responses could be a
valuable resource to support their current team of 60 case handlers.
They assumed that the knowledge needed to answer questions was
contained in these 500,000 previous answers and could be utilized
by an LLM. They discussed creating a RAG system that combined
a GPT-4 large language model with the knowledge base of 500,000
answers (see figure 1). This system was designed to match incoming
customer questions with similar previous responses and generate
new answers based on these historical examples.
But were 500,000 previous answers enough for the model to
generate correct answers (in other words, was the knowledge base
suitable)? The data scientists understood that defining the correct-
ness of the model’s generated answers would not be straightfor-
ward. ’Correct’ could mean different things in different contexts.
Additionally, the data scientists needed the domain knowledge of
case handlers to determine correctness. They decided the best ap-
proach was to develop the RAG system and begin experimenting
with the case handlers, allowing them to judge the ’correctness’ of
the answers.
“This is an experiment; it will be interesting to see
if this GPT model can actually generate sensible an-
swers that convince the case handlers.” – Heimdal, AI
Solution Engineer
Their worry with this approach was that an early release of a
poorly performing model might disappoint users, who could then
dismiss the entire system.
“We can keep talking forever, but if we are going to
figure out if this can work, then we just need to start.”
– Heimdal, AI Solution Engineer
The initial feedback from test users was positive. Junior case han-
dlers were particularly enthusiastic, seeing this as a tool that could
compensate for their limited experience and knowledge. They could
now benefit from all previous answers created by their senior col-
leagues. Additionally, since Marcomp operates globally, most case
handlers are not native English speakers, and many felt they couldnow produce replies that better communicated their message in
English.
4.2 Retrieval strategies
Looking back to the scenario in the introduction, with Kat having
to delete the generated answer because a new rule came into effect,
one can see a significant data requirement challenge: changing rules
over time . When a new rule takes effect or an existing rule changes,
the database of previous answers lacks examples reflecting these
updates. For instance, if a new rule became effective in 2018, all
previously stored answers might be inaccurate depending on the
change in the rule.
“The model will make mistakes because it only has
examples from the old rules and not the new ones.” –
Magnus, Data Architecht
Since they had a fixed set of previous answers, and it would take
time for case handlers to generate new responses that followed
updated rules, they needed another solution. Their idea was to add
an always-updated rulebook to the existing database of answers.
However, they discovered that linking specific ships to relevant
rules was more complicated than anticipated.
“I did what turned out to be a silly project at the start
where I just took all the rules and ran them through a
standard vectorization solution, but it didn’t work be-
cause then you get that dilemma where you ask ques-
tions about passenger ships, it responds with things
that apply to container ships. It doesn’t distinguish
the contexts.” – Magnus, Data Architecht
The rules were too similar for the model to connect them to the
specific ship in question correctly. As a result, the data scientists
abandoned trying to solve this challenge by adding more data.
Instead, they developed what they called a "filtering" function. This
meant the case handler could filter out all the previous answers
according to a year. So, if a rule took effect in 2018, they could
instruct the model to exclude all answers prior to 2018. This meant
that in order for case handlers to generate a good answer, they
had to support the model in filtering out the irrelevant previous
answers.
“To accomplish this [applying correct filters], they
need many years of experience [as a case handler].
... They need to have a complete overview of all the
rules to understand which rules apply to the ship in
question.” – Magnus, Data Architecht
Unfortunately, rules were not the only contextual challenge.
Case handlers quickly discovered that the model struggled with
unfamiliar contexts and sometimes created fictional responses.
Sometimes, it hallucinates. It picks up information
that is not necessarily relevant for that specific ques-
tion. – Karl, Case handler
A hallucinating LLM indicates a gap in the previous-answers-database.
“Every ship is different, even sister ships have differ-
ences. This is because different people manage the
ships, different management styles, different opera-
tional areas, cargoes, and flags [nationalities]. ... So
probably it [the RAG system] would take the answer

EASE 2025, 17–20 June, 2025, Istanbul, Turkey Tor Sporsem and Rasmus Ulfsnes
from an already answered question, which is of sim-
ilar age, similar type of vessel, similar flag, but not
necessarily the same management.” – Emil, Case han-
dler
Since the contextual factors are numerous and can combine
in countless ways, creating almost infinite scenarios. Questions
will emerge with combinations of contextual elements not found
in previous answers. When these new combinations do not exist
in the database, the model does not recognize the context and
lacks a framework for handling the question. Case handlers have
a significant advantage over the model when dealing with this
complexity—they can gather information from beyond computer
systems. We observed them calling surveyors who had inspected the
vessel to hear what they observed. Or they consulted colleagues in
other departments who had dealt with similar cases to get necessary
explanations. In this way, case handlers obtained vital information
not stored in the database of previous answers or any computer
system. Case handlers doubted the model’s ability to manage high
complexity because they knew it would not access information
about all these factors.
To address the context problem, the data scientists enhanced the
filtering function to include multiple ship characteristics such as
nationality, type of vessel, age, etc. This allowed case handlers to
filter previous answers that shared similar contextual factors. When
case handlers clicked "generate reply," the system produced text
based on these filtered previous answers, increasing the likelihood
of providing a correct prediction.
“When there’s an advanced context or when the con-
text is very specific, then it [the RAG system] doesn’t
seem to give a useful answer” – Karl, Case handler
The filtering function was the first instance where the data scien-
tists had to give up trying to solve their data requirement challenge
by adding more data. Instead, they developed filtering as compen-
sation for this limitation. This filtering function allowed the case
handlers to provide the missing contextual information and deter-
mine which previous answers should inform the generated response.
As a result, case handlers could input the relevant contextual data
themselves for the model to generate an appropriate answer.
4.3 Engineering for (Ir)retrievable Data
Despite the filtering function addressing some retrieval problems,
other challenges remained. The nature of rules is that they require
interpretation, and for certifying companies like Marcomp this
means that every answer must comply with the prevailing inter-
pretation norms among classification societies. These norms are
established in joint forums among classification companies and
regulatory authorities. Deviating from these norms could damage
Marcomp’s reputation and, in the worst case, they could lose their
authority to issue certificates.
During one of our observations, a case handler received a ques-
tion she recognized as having a strategic motive behind it.
“The straight forward "correct" answer would favor
the wishes of the captain here. However, in this con-
text, we need to uphold the consensus on how this
rule is interpreted and write a more strategic answer.”
– Kat, Case handler
Figure 1: The RAG system as developed by Marcomp. First, a
question—e.g., from a ship’s captain—is sent to a case handler.
The case handler sets filters to help the RAG system retrieve
relevant past answers. These, along with the question, form
the input for the LLM. The generated answer is then reviewed
and often revised by the case handler before sending a final
answer back to the captain.
The case handler pointed out that even when the model creates
text that looks flawless, it might fail to capture the key interpre-
tation elements and potentially undermine established consensus
in the industry. Case handlers preserve this established precedent
in their responses. Senior case handlers serve a vital function in
these scenarios. Their years of experience allow them to recognize
complicated rules and understand how they should be interpreted
according to international agreements.
Again, unable to adequately solve this data requirement, the data
scientists had to compensate with a different approach. They en-
sured that the human case handler always checks every generated
response before sending it, even for the simplest cases. While it
would theoretically be possible for the RAG system to recognize
interpretation norms from previous answers and generate correct
responses, this was not practically possible with the existing data-
base of 500,000 answers.
4.4 Managing Expectations
To identify the RAG system’s correctness and find measures to
improve it, like the filters, the data scientists first had to release the
system and see how case handlers used it in real situations. The
data scientists stressed that finding the key contextual elements
and determining which ones were missing from previous answers
completely depended on observing case handlers’ reactions to early
versions of the system. This approach helped them discover what
data was missing from the database and what would be impossible
to include. These insights guided their development of the filters.
“We know that the model [RAG system] will make
errors, it is inevitable. So we have to think completely
differently than normal, we have to think about the

Towards Requirements Engineering for RAG Systems EASE 2025, 17–20 June, 2025, Istanbul, Turkey
Figure 2: An iterative five-stage process model for eliciting
"retrieval requirements" (RR). First, data scientists explore
the available data in the knowledge base. Second, they define
which parts should be retrievable as input to the LLM. Third,
they work with users to assess the RAG system’s output and
design filtering functions for tailored retrieval. Fourth, they
monitor the live system to identify new "retrieval require-
ments". Fifth, they continuously manage user expectations
as the system evolves.
consequences when it is wrong, and set up safeguards
for that. [...] We have to somehow get the system
started even though it’s not perfect, and then based
on that we can initiate continuous improvement.” –
Heimdal, AI Solution Engineer
However, this approach put the data scientists in a difficult posi-
tion of managing users’ high expectations while needing to release
an early, imperfect version of the model. While observing the case
handlers, we noticed that they did not feel threatened by introduc-
ing the RAG system, despite initial comments that it could take over
their job. After testing the system, they doubted it could match their
performance due to the complex contexts and strategic elements in-
volved in answering questions. Case handlers shared that their high
expectations were unmet in interviews and observations. Many
were initially impressed by their first experience with ChatGPT
and hoped for the same level of impact again.
“I feel that this AI is kind of a toddler. ... I’m like a God
to him” – Emil, Case handler
Despite users’ high expectations, the data scientists knew the
system would make mistakes because data requirements were diffi-
cult to meet, especially in early versions. They made some efforts
to manage the users’ high expectations but eventually gave up. The
hype surrounding LLMs seemed too strong and all-encompassing.
They still chose to release it to users because they needed to dis-
cover perceived correctness of generated answers and ways to
continuously improve it.
“I have given up trying to explain this [to the users],
and just started implementing.” – Heimdal, AI Solu-
tion Engineer
5 TOWARDS REQUIREMENTS ENGINEERING
FOR RAG
In addressing our research question: how do developers elicit require-
ments for RAG-systems? , we propose a five-stage iterative process
that outlines how developers elicit "retrieval requirements" for RAGsystems (figure 2). We identify four sequential steps: 1) Knowledge
modeling and experimentation, 2) Retrieval Strategy, 3) Retriev-
able data management, and 4) Monitoring and operation. With an
underlying continuous activity of expectation management (5).
This process shares commonalities with existing machine learn-
ing processes [ 2], particularly regarding the (1) Knowledge modeling
and experimentation stage, where developers must evaluate what
information exists in the organization’s knowledge base and which
potential LLMs might be suitable for implementation.
However, our findings reveal two distinctive stages: Retrieval
strategy andRetrievable data management . The (2) retrieval strategy
stage enables data scientists and developers to define which data
the LLM can access when generating predictions—in other words,
what it can and cannot retrieve as part of the prompt. This phase in-
volves searching and matching the incoming questions or problems
with the existing knowledge base at runtime. This highlights a key
difference between RAG and traditional ML: RAG allows flexible,
runtime control [ 4], while traditional ML requires model retraining
to handle input changes [26].
The (3) Retrievable data management stage addresses knowledge
that can be effectively retrieved, providing case handlers with func-
tions like retrieval filters or discarding generated content entirely.
In this stage, developers create functions that empower users to
address missing content, poorly ranked documents, and extraction
failures as identified by Barnett et al. [4].
The final stage, (4) Monitoring and operation , parallels the moni-
toring step described by Amershi et al. [2], emphasizing the need
to evaluate system performance and determine if adjustments to
theRetrieval strategy are necessary due to changes in Retrievable
data management .
Throughout the process, (5) expectation management is a foun-
dational element. This emerges from uncertainty about user per-
formance expectations and the RAG system’s ability to generate
"correct" answers. As shown in ML development research [ 2;21],
these systems heavily depend on user feedback and input to achieve
satisfactory performance. Our findings demonstrate how data scien-
tists elicited "retrieval requirements" through experiments and field
studies, incrementally implementing additional filters to enhance
the retrieval of previous answers, discovering necessary filter types
through iterative implementation cycles.
6 CONCLUSION & LIMITATIONS
Based on our preliminary findings, eliciting "Retrieval Require-
ments" is essential when developing RAG systems to ensure output
correctness. This short paper presents an empirically based process
for developing RAG systems and practical examples of how Re-
trieval Requirements can be identified. We plan to continue study-
ing Marcomp’s RAG system development to uncover additional
aspects and refine our proposed development process.
There are three major limitations to this study. First, as this
is a case study, generalization was not the objective. Although
our findings and the proposed process model for developing RAG
systems appear logical and potentially applicable to other cases,
a single case study cannot confirm their broader applicability. To
achieve generalizability, a quantitative study would be preferable.
Alternatively, conducting multiple case studies across different

EASE 2025, 17–20 June, 2025, Istanbul, Turkey Tor Sporsem and Rasmus Ulfsnes
industrial contexts could strengthen the conclusions and reveal
patterns across settings.
Second, construct validity poses a challenge due to the imma-
ture state of terminology surrounding AI, ML, and RAG systems
among practitioners. Informants often use different terms for the
same concept or the same term for different concepts, complicating
the research process. This required us to frequently validate our
interpretations of their language. To improve construct validity in
this emerging topic, we urge researchers to develop empirically
grounded definitions of the concepts studied. Addressing construct
validity explicitly is essential when researching new areas such as
requirements engineering for RAG systems.
Third, we have not yet measured the effects of the RAG system at
Marcomp, meaning we cannot determine whether its development
has been a success. In future work, we plan to study the system’s
impact on productivity and knowledge sharing among case han-
dlers. We are particularly interested in whether the RAG system
can handle most simpler questions automatically, thereby remov-
ing routine tasks from case handlers. At the same time, we aim to
examine how this might affect junior case handlers, who typically
rely on these simpler tasks for early career learning. Additionally,
we want to investigate whether automating simple tasks changes
the nature of questions juniors direct to seniors, potentially altering
knowledge-sharing dynamics.
7 ACKNOWLEDGMENTS
We thank the Norwegian Research Council for funding this research
(grant number: 309631 & 355691).
REFERENCES
[1]Ahmad, K., Abdelrazek, M., Arora, C., Bano, M., and Grundy, J. (2023). Require-
ments practices and gaps when engineering human-centered Artificial Intelligence
systems. Applied Soft Computing , 143:110421.
[2]Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N.,
Nushi, B., and Zimmermann, T. (2019). Software Engineering for Machine Learning:
A Case Study. In 2019 IEEE/ACM 41st International Conference on Software Engineering:
Software Engineering in Practice (ICSE-SEIP) , pages 291–300.
[3]Arslan, M., Ghanem, H., Munawar, S., and Cruz, C. (2024). A Survey on RAG with
LLMs. Procedia Computer Science , 246:3781–3790. Publisher: Elsevier.
[4]Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., and Abdelrazek, M. (2024).
Seven Failure Points When Engineering a Retrieval Augmented Generation System. In
Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering - Software
Engineering for AI , pages 194–199, Lisbon Portugal. ACM.
[5]Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee,
P., Lee, Y. T., Li, Y., Lundberg, S., Nori, H., Palangi, H., Ribeiro, M. T., and Zhang,
Y. (2023). Sparks of Artificial General Intelligence: Early experiments with GPT-4.
arXiv:2303.12712 [cs].
[6]Chen, J., Lin, H., Han, X., and Sun, L. (2024). Benchmarking Large Language
Models in Retrieval-Augmented Generation. Proceedings of the AAAI Conference on
Artificial Intelligence , 38(16):17754–17762. Number: 16.
[7]Dell’Acqua, F., McFowland III, E., Mollick, E. R., Lifshitz-Assaf, H., Kellogg, K.,
Rajendran, S., Krayer, L., Candelon, F., and Lakhani, K. R. (2023). Navigating the
jagged technological frontier: Field experimental evidence of the effects of ai on
knowledge worker productivity and quality.
[8]Giray, G. (2021). A software engineering perspective on engineering machine
learning systems: State of the art and challenges. Journal of Systems and Software ,
180:111031.
[9]Ishikawa, F. and Yoshioka, N. (2019). How Do Engineers Perceive Difficulties
in Engineering of Machine-Learning Systems? - Questionnaire Survey. In 2019
IEEE/ACM Joint 7th International Workshop on Conducting Empirical Studies in Industry
(CESI) and 6th International Workshop on Software Engineering Research and Industrial
Practice (SER&IP) , pages 2–9. ISSN: 2575-4793.
[10] Kalinowski, M., Mendez, D., Giray, G., Alves, A. P. S., Azevedo, K., Escovedo, T.,
Villamizar, H., Lopes, H., Baldassarre, T., Wagner, S., Biffl, S., Musil, J., Felderer, M.,
Lavesson, N., and Gorschek, T. (2024). Naming the Pain in Machine Learning-Enabled
Systems Engineering. arXiv:2406.04359 [cs].[11] Kim, M., Zimmermann, T., DeLine, R., and Begel, A. (2018). Data Scientists in
Software Teams: State of the Art and Challenges. IEEE Transactions on Software
Engineering , 44(11):1024–1038.
[12] Langley, A. (1999). Strategies for Theorizing from Process Data. Academy of
Management Review , 24(4):691–710. Publisher: Academy of Management.
[13] Lebovitz, S., Levine, N., and Lifshitz-Assaf, H. (2021). Is AI ground truth really
true? The dangers of training and evaluating AI tools based on experts’ know-what.
MIS Quarterly , 45(3, SI):1501–1526.
[14] Lethbridge, T. C., Sim, S. E., and Singer, J. (2005). Studying Software Engineers:
Data Collection Techniques for Software Field Studies. Empirical Software Engineering ,
10(3):311–341.
[15] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.-t., and Rocktäschel, T. (2020). Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems ,
33:9459–9474.
[16] Lipton, Z. C. (2018). The Mythos of Model Interpretability: In machine learning,
the concept of interpretability is both important and slippery. Queue , 16(3):31–57.
[17] Runeson, P. and Höst, M. (2008). Guidelines for conducting and reporting case
study research in software engineering. Empirical Software Engineering , 14(2):131.
[18] Seaman, C. B. (1999). Qualitative methods in empirical studies of software
engineering. IEEE Transactions on software engineering , 25(4):557–572.
[19] Stol, K.-J. (2024). Teaching Theorizing in Software Engineering Research. In
Mendez, D., Avgeriou, P., Kalinowski, M., and Ali, N. B., editors, Handbook on Teaching
Empirical Software Engineering , pages 31–69. Springer Nature Switzerland, Cham.
[20] Stol, K.-J., Ralph, P., and Fitzgerald, B. (2016). Grounded theory in software
engineering research: a critical review and guidelines. In Proceedings of the 38th
International Conference on Software Engineering , pages 120–131, Austin Texas. ACM.
[21] Tanweer, A., Gade, E., Krafft, P. M., and Dreier, S. (2021). Why the data revolution
needs qualitative thinking. Harvard Data Science Review , 3.
[22] Ulfsnes, R., Mikalsen, M., and Barbala, A. M. (2024a). From generation to applica-
tion: Exploring knowledge workers’ relations with GenAI. In ICIS 2024 Proceedings .
[23] Ulfsnes, R., Moe, N. B., Stray, V., and Skarpen, M. (2024b). Transforming Software
Development with Generative AI: Empirical Insights on Collaboration and Workflow.
In Nguyen-Duc, A., Abrahamsson, P., and Khomh, F., editors, Generative AI for
Effective Software Development , pages 219–234. Springer Nature Switzerland, Cham.
[24] van den Broek, E., Sergeeva, A., and Huysman, M. (2021). When the machine
meets the expert: an ethnography of developing AI for hiring. MIS Quarterly , 45(3,
SI):1557–1580.
[25] Veturi, S., Vaichal, S., Jagadheesh, R. L., Tripto, N. I., and Yan, N. (2024). RAG based
Question-Answering for Contextual Response Prediction System. arXiv:2409.03708.
[26] Vogelsang, A. and Borg, M. (2019). Requirements Engineering for Machine Learn-
ing: Perspectives from Data Scientists. In 2019 IEEE 27th International Requirements
Engineering Conference Workshops (REW) , pages 245–251.
[27] Waardenburg, L. and Huysman, M. (2022). From coexistence to co-creation:
Blurring boundaries in the age of AI. Information and Organization , 32(4):100432.
Publisher: Elsevier.
[28] Wan, Z., Xia, X., Lo, D., and Murphy, G. C. (2021). How does Machine Learning
Change Software Development Practices? IEEE Transactions on Software Engineering ,
47(9):1857–1871.
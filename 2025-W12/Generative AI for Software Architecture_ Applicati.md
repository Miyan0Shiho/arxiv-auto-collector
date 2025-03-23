# Generative AI for Software Architecture. Applications, Trends, Challenges, and Future Directions

**Authors**: Matteo Esposito, Xiaozhou Li, Sergio Moreschini, Noman Ahmad, Tomas Cerny, Karthik Vaidhyanathan, Valentina Lenarduzzi, Davide Taibi

**Published**: 2025-03-17 15:49:30

**PDF URL**: [http://arxiv.org/pdf/2503.13310v1](http://arxiv.org/pdf/2503.13310v1)

## Abstract
Context: Generative Artificial Intelligence (GenAI) is transforming much of
software development, yet its application in software architecture is still in
its infancy, and no prior study has systematically addressed the topic. Aim: We
aim to systematically synthesize the use, rationale, contexts, usability, and
future challenges of GenAI in software architecture. Method: We performed a
multivocal literature review (MLR), analyzing peer-reviewed and gray
literature, identifying current practices, models, adoption contexts, and
reported challenges, extracting themes via open coding. Results: Our review
identified significant adoption of GenAI for architectural decision support and
architectural reconstruction. OpenAI GPT models are predominantly applied, and
there is consistent use of techniques such as few-shot prompting and
retrieved-augmented generation (RAG). GenAI has been applied mostly to initial
stages of the Software Development Life Cycle (SDLC), such as
Requirements-to-Architecture and Architecture-to-Code. Monolithic and
microservice architectures were the dominant targets. However, rigorous testing
of GenAI outputs was typically missing from the studies. Among the most
frequent challenges are model precision, hallucinations, ethical aspects,
privacy issues, lack of architecture-specific datasets, and the absence of
sound evaluation frameworks. Conclusions: GenAI shows significant potential in
software design, but several challenges remain on its path to greater adoption.
Research efforts should target designing general evaluation methodologies,
handling ethics and precision, increasing transparency and explainability, and
promoting architecture-specific datasets and benchmarks to bridge the gap
between theoretical possibilities and practical use.

## Full Text


<!-- PDF content starts -->

Generative AI for Software Architecture.
Applications, Trends, Challenges, and Future Directions
Matteo Espositoa, Xiaozhou Lia, Sergio Moreschinia,b, Noman Ahmada, Tomas Cernyc, Karthik Vaidhyanathand, Valentina
Lenarduzzia, Davide Taibia
aUniversity of Oulu, Finland
bTampere University, Finland
cUniversity of Arizona, USA
dSoftware Engineering Research Center, IIIT Hyderabad, India
Abstract
Context . Generative Artificial Intelligence (GenAI) is transforming much of software development, yet its application in soft-
ware architecture is still in its infancy, and no prior study has systematically addressed the topic.
Aim . Systematically synthesize the use, rationale, contexts, usability, and future challenges of GenAI in software architecture.
Method . Multivocal literature review (MLR), analyzing peer-reviewed and gray literature, identifying current practices, models,
adoption contexts, reported challenges, and extracting themes via open coding.
Results : This review identifies a significant adoption of GenAI for architectural decision support and architectural reconstruc-
tion. OpenAI GPT models are predominantly applied and there is consistent use of techniques such as few-shot prompting
and retrieved-augmented generation (RAG). GenAI has been applied mostly to the initial stages of the Software Development
Life Cycle (SDLC), such as Requirements-to-Architecture and Architecture-to-Code. Monolithic and microservice architectures
were the main dominant targets. However, rigorous testing of GenAI outputs was typically missing from the studies. Among
the most frequent challenges are model precision, hallucinations, ethical aspects, privacy issues, lack of architecture-specific
datasets, and the absence of sound evaluation frameworks.
Conclusions : GenAI shows significant potential in software design, but there are several challenges on its way toward greater
adoption. Research efforts should target designing general evaluation methodologies, handling ethics and precision, increasing
transparency and explainability, and promoting architectural-specific datasets and benchmarks to overcome the gap between
theoretical possibility and practical use.
Keywords: Generative AI, Software Architecture, Multivocal Literature Review, Large Language Model, Prompt Engineering,
Model Human Interaction, XAI
1. Introduction
Generative AI (GenAI) is driven by the need to create, inno-
vate, and automate complex tasks that traditionally require
human creativity. It empowers businesses and individuals to
unlock new possibilities, fostering innovation and improving
productivity.
In software engineering, GenAI is revolutionizing the way
developers design, write, and maintain code. Given its po-
tential and benefits, the integration of GenAI within the do-
main of software engineering has gained increasing attention
as it has a transformative potential to enhance and automate
various aspects of the software development lifecycle.
Email addresses: matteo.esposito@oulu.fi (Matteo Esposito),
xiaozhou.li@oulu.fi (Xiaozhou Li), sergio.moreschini@oulu.fi
(Sergio Moreschini), noman.ahmad@oulu.fi (Noman Ahmad),
tcerny@arizona.edu (Tomas Cerny),
karthik.vaidhyanathan@iiit.ac (Karthik Vaidhyanathan),
valentina.lenarduzzi@oulu.fi (Valentina Lenarduzzi),
davide.taibi@oulu.fi (Davide Taibi)Although GenAI has shown its capabilities in areas such as
code generation, software documentation, and software test-
ing [14, 1], its application in software architecture remains
an emerging area of research, with ongoing debates about its
effectiveness [4], reliability [23], and best practices [32]. Re-
searching the application of GenAI in software architecture
is crucial because it has the potential to transform the way
complex systems are designed, optimized, and maintained.
However, practitioners and researchers continue to get
challenged when understanding the implications, limita-
tions, and potential benefits of GenAI for architectural tasks.
To catalyze research in this area, they need a roadmap on var-
ious research directions, applications, trends, challenges, and
future directions.
To better understand existing research in this area, this
study investigates the current state of research and practice
on the use of GenAI in software architecture .
Specifically, we conducted a Multivocal Literature Review
(MLR) to synthesize the findings from academic literature
and gray literature sources, including industry reports, blog
Preprint submitted to Journal of Systems and Software March 18, 2025arXiv:2503.13310v1  [cs.SE]  17 Mar 2025

posts, and technical documentation [2]. In particular, our
goal is to understand how GenAI is used in software archi-
tecture and what the underlying rationales, models, and us-
age approaches are, as well as the context and practical use
cases where GenAI has been adopted for software architec-
ture. Moreover, we also aim at understanding research gaps
highlighted by the literature, to provide an overview of possi-
ble research directions to practitioners and researchers.
Despite the growing adoption of GenAI in software engi-
neering, several factors justify the need for a systematic in-
vestigation into its role in software architecture:
•Emerging and Underexplored Research Area : Although
GenAI has been widely adopted in software develop-
ment tasks, its role in software architecture remains un-
derdeveloped [14]. Studies suggest that while GenAI
models can help in architectural modeling and decision-
making, their contributions are still in the early stages of
research and adoption [4].
•Lack of Systematic Evidence on Effectiveness and Reliabil-
ity: Existing work reports inconsistent findings regarding
the reliability of GenAI for architectural decisions [23].
Some studies indicate its potential in architectural mod-
eling and automation, while others highlight challenges
such as hallucinations, interpretability, and alignment
with established architectural principles [32].
•Need for a Comprehensive Synthesis of Both Academic
and Gray Literature : Given the rapid evolution of GenAI
models, gray literature, such as industry reports and
practitioner blogs, provides valuable but fragmented
knowledge that needs systematic integration [3].
•Unclear Best Practices and Guidelines for Adoption : Al-
though strategies such as prompt engineering, Retrieval-
Augmented Generation (RAG), and fine-tuning have
been explored, there is no consensus on best practices
for effectively using GenAI in different software architec-
ture tasks [6, 7]. A structured review can help identify
and formalize these practices for both researchers and
practitioners [33].
•Increasing Industry Interest in Architectural Automation :
Enterprises are increasingly exploring AI-assisted archi-
tectural decision-making tools, yet there is still limited
understanding of their practical benefits and risks [41].
The demand for explainable AI in architecture, and in
particular in safety-critical domains, highlights the need
for a systematic evaluation of the literature [43].
•Identifying Open Challenges : Multiple research ques-
tions remain open on multiple aspects. Examples are se-
curity vulnerabilities introduced by AI-driven modifica-
tions [23], biases in architectural decision making [15],
or ethical implications of AI-generated architectural de-
cisions [9]. This work will help illuminate open chal-
lenges highlighted by practitioners and researchers.
The main contributions of this study are as follows.•A comprehensive synthesis of the existing literature and
industry reports to provide an overview of how GenAI is
used in software architecture.
•A classification of the GenAI models adopted for Soft-
ware Architecture based on data extracted following the
open coding approach [4].
•Identification of Common Applications, benefits, and
challenges of the application of GenAI in software archi-
tecture.
•Identification of research gaps and open research ques-
tions that provide recommendations for future studies
and practical adoption.
•Industry Relevance By incorporating the gray literature,
we bridge the gap between research and practice, ensur-
ing that our findings are aligned with real-world applica-
tions.
Paper Structure: Section 2 presents the related work. Sec-
tion 3 describes the study design. Section 4 presents the re-
sults obtained, and Section 5 discusses them. Section 6 high-
lights the threats to the validity of our study. Finally, Section 7
draws the conclusion.
2. Related Work
Different works have been done to understand the extent
to which large language models have been applied in soft-
ware engineering. Fan et al. [5] performed a survey to iden-
tify how LLMs have been leveraged by different steps in the
software engineering lifecycle. The work highlights that while
much emphasis has been given to implementation, particu-
larly code generation, not much work has been done in the
area of using LLMs for requirements and design. This is fur-
ther emphasized by Hou et al. [6], where the authors per-
formed a systematic literature review to understand the us-
age of LLMs in software engineering with a particular focus
on how LLMs have been leveraged to optimize processes and
outcomes. The authors analyzed 395 research articles and
concluded that similar to the previous study, most of the ap-
plications of LLMs have been on software development. It is
also important to note that the work only selected four rele-
vant academic literature that leverage LLMs for software de-
sign. Thereby emphasizing the need for a multi-vocal liter-
ature review. Ozkaya [7] provided a pragmatic view into us-
ing LLMs for Software Engineering tasks by enlisting the op-
portunities, associated risks, and potential challenges. The
work points out challenges such as bias, data quality, privacy,
explainability, etc, while describing some of the opportuni-
ties with respect to specification generation, code generation,
documentation, etc.
There have also been various secondary studies focusing
on the use of LLMs for specific aspects of Software Engineer-
ing. For instance, Jiang et al. [8] performed a systematic lit-
erature review to understand the use of LLMs for code gen-
2

Table 1: Classification and Comparison of Related Systematic Studies
Legend :SLR - Systematic Literature Review; SMS - Systematic Mapping Study; MLR - Multivocal Litterature Review; Hol Holistic Review
ReferenceSystematic
Study
TypeMain Focus Area Identified Challenges Key Findings
Hou et al. [6] SLRProcess optimization using
LLMsLimited software design
applicationsMajority use in software development phases, underscoring
the need for multi-vocal studies.
Ozkaya [7] HolRisks and opportunities of
LLMs in SEBias, data quality, privacy,
explainabilityHighlights potential in specification, code, and documenta-
tion generation tasks.
Jiang et al.
[8]SLR LLMs for code generationBridging research-
practice gapTaxonomy developed; outlined research-practice gaps and
future opportunities.
Wang et al.
[9]SLRLLM applications in soft-
ware testingIntegration challengesExtensive LLM usage in testing highlighted; discussed prac-
tical integration barriers.
Marques
et al. [10]HolChatGPT in requirements
engineeringData accuracy and rele-
vanceProvided a detailed overview of current use, challenges, and
identified future directions.
Santos et al.
[11]SLRGenerative AI impact on SE
lifecycleOveremphasis on devel-
opment/testing phasesConfirmed dominance of development/testing; suggested
expansion to other SE phases.
Saucedo and
Rodríguez
[12]SMSAI for migration to microser-
vicesAccuracy of unsupervised
learning methodsHighlighted clustering as a prevalent AI technique for mi-
grating monolithic to microservices architecture.
Fan et al. [5] Hol LLMs in SE lifecycleLimited exploration in re-
quirements/designEmphasis predominantly on code generation; limited atten-
tion to early SE phases.
Our Work MLRGenerative AI specifically for
software architectureScarcity of comprehen-
sive reviews; dominance
of grey literatureProvides comprehensive insights, bridging academic and in-
dustry perspectives in generative AI applied to software ar-
chitecture.
eration. The authors selected and analyzed around 235 arti-
cles and developed a taxonomy of LLMs for code generation.
Further, the work points out critical challenges and identifies
opportunities to bridge the gap between research and prac-
tice of using LLMS for code generation. Wang et al. [9], on
the other hand, performed a systematic literature review to
identify the different types of work that have used LLMs for
software testing. It identified and analyzed 102 relevant stud-
ies that have used LLMs for software testing from both the
software testing and LLMs perspectives. Marques et al. [10]
performed a comprehensive study to understand the appli-
cation of LLMs (in particular ChatGPT) in requirements en-
gineering. The work highlights the state of use of ChatGPT
in requirements engineering and further lists the challenges
and potential future work that needs to be performed in this
direction. A secondary study to identify the impact of GenAI
on software development activities was performed by Santos
et al. [11]. Like other secondary studies on using LLMs for
software engineering, this study also highlighted that most of
the work has been centered around development and testing.
While to the best of our knowledge, there is a lack of sec-
ondary study on the use of GenAI applied to software archi-
tecting practices, there have been some work that leverages
LLMs for various software architecting practices.
Alsayed et al. [13] developed MicroRec, an approach
that leverages state-of-the-art deep learning techniques and
LLMs to recommend microservices to developers. The ap-
proach allows developers to search for microservices in ser-
vice registries using natural language queries. An approach
that leverages GenAI, in particular LLMs, to suggest archi-tectural patterns from requirements was proposed by Gus-
trowsky et al. [14]. The proposed solution fine-tunes the
Llama 2 LLM on a custom dataset of requirements and archi-
tectural patterns. The evaluation demonstrated an accuracy
of 70% on the test set. Kaplan et al. [3], on the other hand,
proposed an approach that combines knowledge graphs and
LLMs to support effective discovery and access to software
architecture research knowledge.
Apart from the works that leverage GenAI, particularly
LLMs, there have also been works that applied various
AI techniques to software architecting processes/practices.
Saucedo and Rodríguez [12] performed a systematic mapping
study to understand the use of AI for migrating monolithic
systems to microservice-based systems. The study identified
unsupervised learning, particularly clustering, as one of the
most popular AI techniques used for migration based on ob-
servations from 22 primary studies.
Despite the active exploration of LLMs for a variety of soft-
ware engineering (SE) tasks, particularly code generation,
testing, requirements engineering, etc, there is a dearth of a
comprehensive literature review dedicated to LLM for soft-
ware architecture. Further, many of the works related to using
GenAI for software design or software architecture are more
available in the grey literature. Hence, in this work, we per-
formed a multi-vocal literature review to identify the existing
landscape of using GenAI for software architectural practices
and processes.
3

Deﬁnition of research
questions
Search of the
literature and
snowballingGray
literature
Peer reviewed
literatureGoogle
search
Initial set of
documents
(1054)
Reading the
retrieved literature
Application of
inclusion/exclusion
criteria
Data extraction
Data synthesis
Data interpretationgoal
raw data
results
Answers to the research questions
Data SequenceData
ﬂowLegend
ActivityDeﬁnitive set
of documents
(46)Selection of data
sources and search
termsDeﬁnition of the MLR
goal
Documentresearch
questionsFigure 1: Study Workflow
3. Methodology
This section addresses the methodology, defining the goal
and research questions. It also provides the search and se-
lection process, as well as inclusion and exclusion criteria for
both peer-reviewed and gray literature. Our search strategy is
presented in Figure 1.
3.1. Goal and Research Questions
The goal of this MLR is to provide a comprehensive
overview of GenAI’s role in software architecture, from its cur-
rent state to its prospects. We aim to contribute significantly
to the body of knowledge in software engineering, providing
actionable insights to researchers and practitioners.
To carry out this research, we conducted a multivocal re-
view of the literature [2]. Based on the objectives of our study,
we defined the following research questions (RQs).RQ 1
How is Generative AI utilized in software architecture
and what are the underlying rationales, models, and
usage approaches?
•RQ 1.1. (Why ) For what purposes are Generative AI
models used in software architecture?
•RQ 1.2. (What ) Which Generative AI models have
been used?
•RQ 1.3. (How ) How has Generative AI been applied?
In this RQ, we aim to investigate the integration of GenAI
technologies in the domain of software architecture to high-
light the motivations behind the adoption of these technolo-
gies, the specific models that have been employed, and the
practical applications in software architecture. We try to un-
derstand the underlying rationale behind the adoption of
AI models and how they contribute in practice to architec-
tural design, maintenance, and process optimization ( RQ 1.1).
Therefore, researchers and practitioners can better assess the
impact and potential of GenAI in their specific contexts.
However, an in-depth investigation of the adopted GenAI
models can provide a catalog of the technologies that have
been implemented, providing a detailed landscape of the
tools available to software architects ( RQ 1.2).
Other important aspects to be considered are the strategies
for implementing GenAI technologies in architectural prac-
tices, focusing on the types of projects that benefit from them,
and the outcomes of these integrations ( RQ 1.3).
RQ 2
In what contexts is Generative AI used for software ar-
chitecture?
•RQ 2.1. (Where ) In which phase of the software de-
velopment life cycle is Generative AI applied?
•RQ 2.2. (For what ) Which architectural styles or pat-
terns are targeted?
•RQ 2.3. (For what ) Which architectural quality and
maintenance tasks are targeted?
•RQ 2.4. Which architectural analysis or modeling
methods have been used to validate Generative AI
outputs?
Once GenAI technologies have been investigated in the do-
main of software architecture, the next step is to explore the
environments and scenarios where GenAI is integrated map-
ping the conditions or settings in which these technologies
are applied. Therefore, researchers and practitioners could
better identify opportunities where GenAI can be used ef-
4

fectively, improving the architectural design process, and ad-
dressing complex challenges. In particular, we identified the
stages of the software development life cycle where GenAI
tools are the most beneficial, such as requirements, design,
implementation, testing, or maintenance, providing insight
for the continuous integration of AI throughout the develop-
ment life cycle ( RQ 2.1). Another important aspect is to spec-
ify for which architectural styles or design patterns (e.g., mi-
croservices, monolithic architectures) a GenAI model is more
effective and advantageous in improving design coherence
and system scalability ( RQ 2.2). Moreover, since the benefit of
adopting a new model should always be validated, it is neces-
sary to evaluate and validate the results produced by GenAI,
and architectural analysis or modeling methods have been
used ( RQ 2.3).
RQ 3
To which use cases has Generative AI been applied?
Exploring the environments and scenarios where GenAI is
integrated led to identifying use cases where it has been im-
plemented to highlight versatility and adaptability in differ-
ent cases to solve specific problems, contribute to innova-
tion, and drive industry advancements ( RQ 3).
RQ 4
What future challenges are identified for the use of
Generative AI in software architecture?
As a last RQ, we investigate the future challenges of GenAI
in software architecture for which researchers and practition-
ers should work in the next years ( RQ 4).
3.2. Search Strategy
In this Section, we report the process we adopted for col-
lecting the peer-reviewed papers and the gray literature con-
tributions to be included in our revision.
3.2.1. Search Terms
The search string contained the following search terms:
Search String
(“generative AI” OR “gen AI” OR gen-AI OR genAI OR
“large language model*” OR “small language model*”
OR LLM OR LM OR GPT* OR Chatgpt OR Claude* OR
Gemini* OR Llama* OR Bard* OR Copilot OR
Deepseek)
AND
(“software *architect*” OR “software design*” OR
“software decompos*” OR“software structur*”)In our search string, we used different terms for GenAI,
such as gen AI, gen-AI, or genAI, to increase research effi-
ciency. We used an asterisk character (*), such as software
architect*, to get all possible term variations, such as plurals
and verb conjugations. To increase the likelihood of finding
papers that addressed our goal, we applied the search string
to the title and abstract.
3.2.2. Bibliographic Sources
For retrieving the peer-reviewed paper, we selected the list
of relevant bibliographic sources following Kitchenham and
Charters recommendations [15] since these sources are rec-
ognized as the most representative in the software engineer-
ing domain and are used in many reviews. The list includes:
ACM Digital Library, IEEEXplore Digital Library, Scopus, Web
of Science . For contributions to the gray literature, we ex-
tracted data from Google, Google Scholar, and Bing [2].
3.2.3. Inclusion and Exclusion Criteria
We defined the inclusion and exclusion criteria to be ap-
plied to the title and abstract (T/A), the full text (F), or both
cases (All), as reported in Table 2.
Table 2: Inclusion and Exclusion Criteria
ID Criteria Step
I1 Papers should specifically use LLM or Generative AI for Soft-
ware architecture*All
E1 Not in English T/A
E2 Duplicated / extension has been included T/A
E3 Out of topic All
E4 Non peer-reviewed papers T/A
E5 Not accessible by institution T/A
E6 Papers mentioning software architecture for running LLM or
Gen-aiF
E7 Papers before 15.3.2022 when the initial release of GPT-3.5 is
release public**F
*The papers should genuinely be talking about LLM and SA, not just
mentioning the buzzword in abstracts/discussion
**https://platform.openai.com/docs/models
We only included a paper that specifically uses LLM or
GenAI for Software architecture (T/A), defined these terms
(F), reported causes or factors of this phenomenon (F), pro-
posed approaches or tools for their measurement (F), and
recommended any techniques or approaches for remedia-
tion (F).
In the exclusion criteria, we excluded a paper that was not
written in English (T/A), was duplicated, or had an extension
already included in the review (T/A), they were beyond the
scope (All), or was not accessible by an institution (T/A).
3.2.4. Search and Selection Process for the Peer-Reviewed Pa-
pers (white)
We conducted the search and selection process in Febru-
ary 2025 and included all available publications until this
5

period. The application of the search terms returned 621
unique white papers as reported in Table 5.
•Testing the applicability of the inclusion and exclusion
criteria: Before implementing the inclusion and exclu-
sion criteria, we evaluated their applicability [16] in ten
randomly chosen articles from the retrieved paper (as-
signed to all authors).
•Applying inclusion and exclusion criteria to the title and
abstract: We used the same criteria for the remaining 611
articles. Two authors read each paper, and if there was
any disagreement, a third author participated to resolve
the disagreement. We included a third author for 30 pa-
pers. The interrater agreement through the Cohen co-
efficient kshowed a 71% agreement corresponding to a
substantial agreement. Based on the title and abstract,
we selected 45 of the original 621 papers.
•Full reading: We performed a full read of the 45 papers
included by title and abstract, applying the inclusion
and exclusion criteria defined in Table 2 and assigning
each article to two authors. We involved a third author
for eight papers to reach a final decision. Based on this
step, we selected 19 papers as possibly relevant contri-
butions (Cohen’s kcoefficient 64%: substantial agree-
ment).
•Snowballing: The snowballing process [17] involved: 1)
the evaluation of all articles that cited the recovered ar-
ticles and 2) the consideration of all references in the re-
covered articles. The snowball search was performed in
February 2025. We found that 23 articles were included
in the final set of publications. Since our search and
selection process was conducted immediately after the
notification of the International Conference on Software
Architecture (ICSA) 2025, we waited for the pre-print of
all accepted papers to be available to avoid not including
some potentially interesting contributions.
•Quality and Assessment Criteria: Before proceeding with
the review, we checked whether the quality of the se-
lected articles was sufficient to support our goal and
whether the quality of each article reached a certain
quality level. We perform this step according to the pro-
tocol proposed by Dybå and Dingsøyr [18]. To evaluate
the selected articles, we prepared a checklist (Table 3)
with a set of specific questions. We rank each answer, as-
signing a score on a five-point Likert scale (0=poor, 4=ex-
cellent). A paper satisfied the quality assessment criteria
if it achieved a rating higher than (or equal to) 2. Among
the 39 papers included in the review of the search and se-
lection process, only 37 fulfilled the quality assessment
criteria, as reported in Table 5.
Starting from the 413 unique papers, following the process,
we finally included 37papers as reported in Table 5.Table 3: Quality Assessment Criteria - Peer-Reviewed Papers (white)
QAs QA
QA1 Is the paper based on research (or is it merely a “lessons learned”
report based on expert opinion)?
QA2 Is there a clear statement of the aims of the research?
QA3 Is there an adequate description of the context in which the re-
search was carried out?
QA4 Was the research design appropriate to address the aims of the
research?
QA5 Was the recruitment strategy appropriate for the aims of the re-
search?
QA6 Was there a control group with which to compare treatments?
QA7 Was the data collected in a way that addressed the research issue?
QA8 Was the data analysis sufficiently rigorous?
QA9 Has the relationship between researcher and participants been
considered to an adequate degree?
QA10 Is there a clear statement of findings?
QA11 Is the study of value for research or practice?
Response scale: 4 (Excellent), 3 (Very Good), 2 (Good), 1 (Fair), 0 (Poor)
3.2.5. Search and Selection Process for the Grey Literature
The search was carried out in September 2024 and in-
cluded all publications available until this period. The ap-
plication of the search terms returned 433 unique contribu-
tions to the grey literature as reported in Table 5.
•Testing the applicability of inclusion and exclusion cri-
teria. We used the same method adopted in the search
and selection process for the peer-reviewed papers (10
papers as test cases)
•Applying inclusion and exclusion criteria to title and ab-
stract. We applied the criteria to the remaining 423 pa-
pers. Two authors read each paper, and if there were dis-
agreements, a third author participated in the discussion
to resolve them. For 25 articles, we include a third au-
thor. Of the 433 initial papers, we included 77 based on
title and abstract (Cohen’s kcoefficient 81%: almost per-
fect agreement).
•Full reading. We fully read the 77 articles included by ti-
tle and abstract, applying the criteria defined in Table 2
and assigning each to two authors. We involve a third au-
thor for one paper to reach a final decision (Cohen’s kco-
efficient 88%: almost perfect agreement). Based on this
step, we selected five papers as possibly relevant contri-
butions.
•Snowballing. The snowball search was carried out in
February 2025. We found that four articles were included
in the final set of publications.
•Quality and Assessment Criteria. Different from peer-
reviewed literature, grey literature does not go through
a formal review process, and therefore, its quality is less
controlled. To evaluate the credibility and quality of the
6

Table 4: Quality Assessment Criteria - Grey literature
Criteria Questions Possible Answers
Authority of the producer Is the publishing organization reputable? 1: reputable and well known organization
0.5: existing organization but not well known, 0: unknown
or low-reputation organization
Is an individual author associated with a reputable organization? 1: true
0: false
Has the author published other work in the field? 1: Published more than three other work
0.5: published 1-2 other works, 0: no other works pub-
lished.
Does the author have expertise in the area? (e.g., job title principal
software engineer)1: author job title is principal software engineer, cloud en-
gineer, front-end developer or similar
0: author job not related to any of the previously mentioned
groups. )
Methodology Does the source have a clearly stated aim? 1: yes
0: no
Is the source supported by authoritative, documented references? 1: references pointing to reputable sources
0.5: references to non-highly reputable sources
0: no references
Does the work cover a specific question? 1: yes
0.5: not explicitly
0: no
Objectivity Does the work seem to be balanced in presentation 1: yes
0.5: partially
0: no
Is the statement in the sources as objective as possible? Or, is the
statement a subjective opinion?1: objective
0.5 partially objective
0: subjective
Are the conclusions free of bias or is there vested interest? E.g., a tool
comparison by authors that are working for a particular tool vendor1=no interest
0.5: partial or small interest
0: strong interest
Are the conclusions supported by the data? 1: yes
0.5: partially
0: no
Date Does the item have a clearly stated date? 1: yes
0: no
Position w.r.t. related sources Have key related GL or formal sources been linked to/discussed? 1: yes
0: no
Novelty Does it enrich or add something unique to the research? 1: yes
0.5: partially
0: no
Outlet type Outlet Control 1: high outlet control/ high credibility: books, magazines,
theses, government reports, white papers
moderate outlet control/ moderate credibility: annual re-
ports, news articles, videos, Q/A sites (such as StackOver-
flow), wiki articles
0: low outlet control/low credibility: blog posts, presenta-
tions, emails, tweets
7

sources selected from the grey literature and to decide
whether to include a source from the grey literature or
not, we extended and applied the quality criteria pro-
posed by Garousi et al. [2] (Table 4), considering the au-
thority of the producer, the methodology applied, ob-
jectivity, date, novelty, impact, and outlet control. Two
authors assessed each source using the aforementioned
criteria, with a binary or 3-point Likert scale, depending
on the criteria itself. In case of disagreement, we discuss
the evaluation with the third author, who helped provide
the final assessment. We finally calculated the average of
the scores and rejected sources from the grey literature
that scored less than 0.5 on a scale ranging from 0 to 1.
Table 5: Search and Selection Process
Step # Papers
Retrieval from white sources (unique papers) 621
-Reading by title and abstract -576
-Full reading - 30
-Snowballing + 24
-Quality assessment - 2
Primary studies 37
Retrieval from grey literature sources (unique papers) 433
-Reading by title and abstract -356
-Full reading - 76
-Snowballing + 4
Primary studies 9
3.3. Data Extraction
Starting from the initial 1054 unique papers (621 white
and 443 grey ), following the process, we finally included 46
papers (37 white and 9 grey) as reported in Table 5. The data
extraction form, together with the mapping of the informa-
tion needed to answer each RQ, is summarized in Table 6. We
extracted the data following the open coding approach [4],
in which two authors extracted the information, and we in-
volved a third author in case of disagreement. This data is
exclusively based on what is reported in the papers, without
any kind of personal interpretation.
4. Results
In this Section, we report the results to answer our RQs.
4.1. Study Context
This sub-section provides an overview of the study context
in the reviewed research, including the types of studies con-
ducted, the balance between white and gray literature, and
the categories of published works.
Most of the works we considered belong to white litera-
ture (78%) while 22% to the gray (Table 7). Case studies are
the most common type (37%), followed by method proposalsTable 6: Data Extraction
Data RQ Outcome
Work category
naList of Category
Methods List of methodological approaches
Author-First and last name
-Affiliation
Publication Sources-Peer-reviewed literature (white)
-Grey literature
-Publication name
-Publication type (e.g., journal)
-Publication year
GenAI usageRQ1.1 Purpose (why)
RQ1.2 Model (what)
RQ1.3 How
GenAI usage contextRQ2.1 Where
RQ2.2 For what
RQ2.3 Architecture analysis or modeling
method
Use case RQ3-List of use cases
-Analyzed systems
-Programming languages
Future Challenges RQ4 List of challenges
Table 7: White and Grey Literature Distribution
Code PaperID %
White OS[OS[1], OS[2], OS[3], OS[4], OS[5], OS[6], OS[7],
OS[8], OS[9], OS[10], OS[11], OS[12], OS[13],
OS[14], OS[15], OS[16], OS[17], OS[18], OS[19],
OS[20], OS[21], OS[22], OS[23], OS[24], OS[25],
OS[26], OS[27], OS[28], OS[29], OS[30], OS[31],
OS[32], OS[33], OS[34], OS[35], OS[36]]78
Grey OS[OS[37], OS[38], OS[39], OS[40], OS[41], OS[42],
OS[43], OS[44], OS[45], OS[46]]22
(29%) and experiments (14%). Tool reviews (10%) and proof-
of-concept (PoC) studies (3%) represent real experience in
some articles. Surprisingly, we only included a few position
papers (3%) and vision papers (1%) (Table 8). Most of them
are (52%) full papers, followed by short papers (15%) and a
few theses (7%) Table 9. Finally, according to Figure 2 show-
ing the publication source trend, GenAI in SA was promi-
nently discussed and featured in the gray literature during
the start of the hype (2023), but the white literature became
prominent the year after consolidating in 2025 as the main
publication source for the topic.
4.2. Generative AI for Software Architecture: How is it used
(RQ 1)
Here, we present how GenAI is currently applied in SA in
terms of purpose, models used, and techniques for perfor-
mance improvement, such as prompt engineering practices
and the level of human interaction.
Architectural decision support is the purpose most fre-
quently investigated in the reviewed studies, appearing in
8

Table 8: Study Type
Code PaperID %
Case Study [OS[2], OS[3], OS[4], OS[5], OS[6],
OS[7], OS[9], OS[14], OS[15], OS[16],
OS[17], OS[18], OS[20], OS[21], OS[24],
OS[25], OS[26], OS[27], OS[28], OS[29],
OS[30], OS[31], OS[32], OS[34], OS[35],
OS[36]]37
Experiment [OS[1], OS[37], OS[38], OS[8], OS[10],
OS[11], OS[19], OS[22], OS[25], OS[32]]14
Exploratory Study [OS[38]] 1
Method Proposal [OS[1], OS[37], OS[5], OS[6], OS[7],
OS[8], OS[9], OS[10], OS[11], OS[12],
OS[16], OS[17], OS[18], OS[20], OS[21],
OS[26], OS[28], OS[33], OS[34], OS[36]]29
PoC [OS[12], OS[28]] 3
Survey [OS[14]] 1
Tool Review [OS[39], OS[40], OS[41], OS[43],
OS[44], OS[45], OS[46]]10
Table 9: Study Category
Code PaperID %
Blog Post [OS[39], OS[44], OS[45], OS[46]] 9
Full Paper [OS[1], OS[37], OS[38], OS[3], OS[4], OS[6],
OS[7], OS[8], OS[10], OS[11], OS[12],
OS[13], OS[15], OS[16], OS[17], OS[20],
OS[22], OS[25], OS[26], OS[30], OS[32],
OS[33], OS[34], OS[36]]52
Industry Report [OS[31]] 2
Position Paper [OS[42]] 2
Short Paper [OS[2], OS[14], OS[19], OS[23], OS[24],
OS[27], OS[35]]15
Thesis [OS[18], OS[21], OS[29]] 7
Vision Paper [OS[5], OS[9], OS[28]] 7
White Paper [OS[41], OS[43]] 4
Youtube Video [OS[40]] 2
30% of them. This suggests that the primary focus of cur-
rent research on GenAI in software architecture is its appli-
cation in assisting architectural decision-making. For exam-
ple, [OS[3]] use GenAI to generate microservice names, while
[OS[21]] use it to support software design and requirement
engineering, and [OS[16]] use it to guide software architects
in making architectural decisions. Similarly, the second most
frequent purpose for using GenAI in the case of reverse en-
gineering for architectural reconstruction appears in 22% of
the cases. On the other hand, the least explored uses are Re-
verse Engineering for Traceability ([OS[10]]) and Migration &
Re-engineering ([OS[31]]), each of which appeared only in 2%
of the studies (Table 11 - RQ 1.1).
0% 20% 40% 60% 80% 100%202320242025
Grey WhiteFigure 2: Publication Source Trend
1. RQ 1.1(Why GenAI in SA)
LLMs are primarily used for architectural decision
support (30%) and reverse engineering (37%), with
less focus on tasks like migration, re-engineering, and
traceability.
OpenAI GPT models are the ones that rule the roost and
were utilized in 62% of the articles, followed by Google’s mod-
els (9%) (Table 12 - RQ 1.2). Surprisingly, the recently pub-
lished open-source model DeepSeek is already implied in two
works. It is also worth noting that on-demand cloud-based
models are by far the favorable option in place of on-premises
due to their resource requirements.
2. RQ 1.2(GenAI Model Used)
OpenAI GPT models dominate (62%) the research
landscape, while alternatives such as Google LLMs
and LLaMA models are significantly less employed.
Among the techniques to enhance the capabilities and per-
formance of GenAI, Fine-Tuning is applied in 12% of the stud-
ies, that is, some researchers have chosen to fine-tune LLMs
for specific architectural tasks with additional training. In
particular, [OS[38]] used Fine-Tuning to align the LLM in gen-
erating serverless functions. Retrieval-augmented generation
(RAG), including proprietary variants, is applied in 22% of the
studies, suggesting that applying external knowledge sources
is a common method to improve LLM performance in soft-
ware architecture contexts. For example, [OS[5]] used RAG
and Fine-Tuning to retrieve architecture knowledge manage-
ment information and align such models to their needed task.
A large percentage of studies (37%) did not report any data
on model improvements, 18% categorically reported that no
improvements were applied, and the models were run as they
were. 10% of the studies did not explicitly state whether im-
provements were applied. The split in this instance shows
that while fine-tuning and RAG methods are explored, most
studies do not document their method of improvement or
9

Table 10: Publication Sources
Sources Name Type Count Years
AIM Research Research Institution 1 -
Communications in Computer and Information Science Book Series 1 -
Design Society Society Publication 1 -
Electronics (Switzerland) Journal 1 -
European Conference on Pattern Languages of Programs, People and Practices Proceedings 1 -
European Conference on Software Architecture Conference 1 2024
Human-Computer Interaction Journal 1 -
IEEE International Conference on Software Quality Reliability and Security Companion (QRS-C) Proceedings 1 2023
IEEE International Conference on Data and Software Engineering (ICoDSE) Proceedings 1 2023
IEEE International Requirements Engineering Conference (RE) Conference 1 2024
IEEE International Conference on Software Architecture (ICSA) Conference 12 2024, 2025
IEEE International Conference on Software Architecture Companion (ICSA-C) Conference 3 2024
IEEE Software Journal 1 -
IEEE/ACM Workshop on Multi-disciplinary Open and RElevant Requirements Engineering (MO2RE) Workshop 1 2024
Information Technology Journal 1 -
Institutional Website Website 7 -
International Conference on Software Engineering Proceedings 1 -
International Workshop on Designing Software Workshop 1 2024
Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lec-
ture Notes in Bioinformatics)Book Series 1 -
Medium Online Media 2 -
Methods Journal 1 -
SN Computer Science Journal 1 -
Studies in Computational Intelligence Book Series 1 -
YouTube Online Media 1 -
apply the off-the-shelf models without any modifications
(Table 13 - RQ 1.3).
Similarly, prompt engineering is also used to quickly align
LLMs to a new task [19]. The most widely used technol-
ogy is the few-shot prompt, present in 31%. This shows that
researchers use numerous examples to a great extent to al-
low LLMs to produce more precise and contextual architec-
tural output. In contrast, one-shot prompting is the least
used, with the technique mentioned in only 2% of the re-
search suggesting that a single occurrence is infrequent in
this field. Zero-shot prompting occurs in 12% of the stud-
ies, at moderate frequency, where the researchers solely uti-
lize the pre-training knowledge of the model without addi-
tional context. As an example, [OS[43]] employed the three
techniques to evaluate LLM applications in modernizing the
architecture of legacy systems. Finally, in the spectrum of
reasoning enhancements, Chain-of-thought (CoT) prompt-
ing appears only in 8% of the cases. [OS[36]] employs such a
technique when evaluating an LLM-based pipeline from re-
quirements to code.
Moreover, 23% of the articles explicitly state that no type of
prompt engineering has been used, while 13% do not provide
any information. Furthermore, 12% of the articles did not in-
dicate whether or not a prompting strategy had been used, so
the data set was somewhat unclear. Hence, we can infer thatmost articles do not use explicit prompting techniques or at
least do not report them (Table 13 Figure 3 - RQ 1.3).
Most studies involve some form of human interaction with
the model (80%), and this indicates that our community is
prone to involve human observation, validation, or supple-
mentation when using LLMs for software architecture pur-
poses. This indicates that fully autonomous AI-driven archi-
tectural decisions are not yet prevalent, but human partici-
pation is still significant in guiding, validating, or improving
LLM-generated results. For example, [OS[6]] leverages hu-
man interaction by providing a chat-based environment to
provide AI-based support to novice architects to refine design
decisions.
On the other hand, 14% of the studies explicitly state
that no human interaction existed and that the models ex-
isted without direct human intervention. A smaller indus-
try (6%) failed to state whether it considered human interac-
tion. The breakdown shows a high preference for interactive
approaches, validating that LLMs in software development
are used primarily as auxiliary tools and not as standalone
decision-makers (Table 13 - RQ 1.3).
10

Table 11: Purpose of the LLM - (RQ 1.1)
Code PaperID Count %
Architectural Decision Support [OS[2], OS[3], OS[39], OS[40], OS[4], OS[6], OS[7],
OS[14], OS[16], OS[18], OS[43], OS[44], OS[45],
OS[21], OS[22], OS[23], OS[33]]17 37
Reverse Engineering/Architectural Reconstruction [OS[8], OS[41], OS[15], OS[19], OS[20], OS[26],
OS[28], OS[30], OS[46]]10 22
Architecture Generation [OS[1], OS[5], OS[9], OS[12], OS[17], OS[29], OS[34],
OS[36]]8 17
Quality Assessment [OS[8], OS[19], OS[20], OS[26]] 4 9
Software Comprehension [OS[25], OS[27], OS[32]] 3 7
Requirement Engineering [OS[24], OS[35]] 2 4
Migration & Re-engineering [OS[31]] 1 2
Reverse Engineering/Traceability [OS[10]] 1 2
3. RQ 1.3(How GenAI is used)
Few-shot prompting (31%) is the most common tech-
nique, RAG (22%) is frequently used for model en-
hancement, and 80% of the studies involve human in-
teraction, emphasizing the assistive rather than au-
tonomous role of LLM.
4.3. Generative AI for Software Architecture: In which context
(RQ 2)
This section presents the different contexts in which GenAI
is applied within the software architecture. Specifically, we
examine its role across various phases of the Software De-
velopment Lifecycle (SDLC), the architectural styles and pat-
terns it supports, and the validation methods used to assess
its outputs.
Regarding the use of GenAI across SDLC (Table 14 and Fig-
ure 4 - RQ 2.1), the requirement-to-architecture (Req-to-Arch)
is used most frequently, as mentioned in 40% of the papers.
This suggests that LLMs are frequently used to fill in the re-
quirement and architectural design gap, to assist in map-
ping textual specifications into formal architectural represen-
tations. In fact, [OS[2]] leveraged GenAI for collaborative ar-
chitectural design to assist practitioners in designing the SA
from requirements. Similarly, [OS[3]] used ChatGPT to gener-
ate microservice names (architecture) based on the require-
ments.
Following this, Architecture-to-Code (Arch-to-Code) is also
a compelling use case, accounting for 32% of the research.
This indicates a significant focus on using LLMs to automate
or help in mapping architectural designs to implementation-
level code. Following the same logic, [OS[38]] used GenAI to
generate a serverless function (code) from the architectural
specification. A peculiar instance and the least explored one,
nonetheless, is Architecture-to-Architecture (Arch-to-Arch)
transitions, which only 3% of the research covers, indicating
the lack of the current community interest in enhancing, mi-
grating, or converting architectures using LLMs. In line withthis, [OS[20]] refactored the architectural smells using LLMs
such as GPT-4 and LLaMA while [OS[26]] used Gemini 1.5 and
GPT-4o to recommend resolutions of architectural violations.
On the other hand, code-to-architecture (13%) and
requirement-to-architecture-to-code (12%) are fairly repre-
sented. The former is indicative of efforts toward reverse
engineering existing codebases for architectural purposes.
Consistently with this approach, [OS[7]] experimented with
developing LLM-based architecture agents that could im-
prove architecture decision-making starting from code, while
[OS[41]] presented its LLM-based tool to perform the archi-
tectural reconstruction.
The requirement-to-architecture-to-code illustrates ef-
forts to optimize the entire process from requirements to ar-
chitecture to code generation. Using this SDLC arch, [OS[40]]
presented in its video tutor an LLM-based copilot of such an
SDLC arch. Similarly, in a position paper, [OS[23]] presented
an LLM-based assisted architectural design and implementa-
tion based on software requirements.
The distribution of studies indicates that the significant
use of LLMs is at the beginning of the SDLC, e.g., during re-
quirement analysis as well as architectural design, with less
effort going toward changing or reorganizing existing archi-
tectures.
4. RQ 2.1(SDLC Phases)
LLMs are most frequently applied in the Requirement-
to-Architecture (40%) and Architecture-to-Code (32%)
transitions, while Architecture-to-Architecture (3%) is
the least explored.
Concerning the architectural styles and patterns to which
LLMs have been applied, monolithic architectures are men-
tioned most frequently, appearing in 12% of the articles (Ta-
ble 15 - RQ 2.2). This suggests that LLMs are applied primarily
in the understanding, analysis, or modernization of mono-
lithic systems. In fact, [28] used LLM to perform architectural
recovery from a legacy monolithic system to understand the
11

Table 12: LLM Models - (RQ 1.2)
Model Family Model PaperID Count % (Model) % (Family)
OpenAIGPT [OS[1], OS[38], OS[2], OS[3], OS[39], OS[40], OS[4], OS[5], OS[6], OS[7],
OS[8], OS[9], OS[10], OS[11], OS[12], OS[13], OS[14], OS[15], OS[16],
OS[17], OS[18], OS[43], OS[19], OS[20], OS[45], OS[21], OS[22], OS[24],
OS[25], OS[26], OS[27], OS[28], OS[29], OS[30], OS[31], OS[32], OS[34],
OS[35], OS[36]]39 23
62GPT-4 [OS[1], OS[38], OS[4], OS[5], OS[7], OS[8], OS[10], OS[11], OS[12], OS[13],
OS[14], OS[16], OS[17], OS[20], OS[21], OS[25], OS[26], OS[28], OS[30],
OS[31], OS[34]]21 13
ChatGPT [OS[38], OS[2], OS[3], OS[39], OS[40], OS[18], OS[43], OS[19], OS[45],
OS[22], OS[29], OS[35], OS[36]]14 8
GPT-3 [OS[4], OS[5], OS[6], OS[14], OS[15], OS[24], OS[27], OS[30], OS[32]] 9 5
GPT-3.5 [OS[4], OS[5], OS[6], OS[15], OS[24], OS[27], OS[30], OS[32]] 8 5
GPT-4o [OS[1], OS[7], OS[8], OS[10], OS[11], OS[26], OS[34]] 7 4
GPT-4o-mini [OS[1], OS[7], OS[8]] 3 2
GPT-2 [OS[4], OS[5]] 2 1
GPT-3.4 [OS[14]] 1 1
GPT-4 Turbo [OS[25]] 1 1
Google’s LLMBard [OS[40], OS[17], OS[43], OS[19], OS[30], OS[46]] 6 4
9Gemini [OS[26], OS[29], OS[46]] 3 2
Google Bard [OS[43], OS[19], OS[30]] 3 2
Bert [OS[5]] 1 1
Gemini 1.5 [OS[26]] 1 1
Google Gemini [OS[29]] 1 1
LLaMALLaMA [OS[37], OS[9], OS[10], OS[12], OS[42], OS[18], OS[20]] 7 4
8LLaMA-3 [OS[37], OS[18]] 2 1
Llama 3.1 [OS[10]] 1 1
LLaMA-2 [OS[37]] 1 1
Code Llama [OS[42]] 1 1
Codellama 13b [OS[10]] 1 1
DeepSeekDeepSeek-Coder [OS[38]] 1 1
1
DeepSeek-V2.5 [OS[1]] 1 1
CodeQwenCodeQwen [OS[1], OS[38]] 2 1
2
CodeQwen1.5-7B [OS[1]] 1 1
GitHub Copilot Copilot [OS[43], OS[44], OS[21], OS[29]] 4 2 2
MistralMistral [OS[37], OS[24]] 2 1
2
Mistral 7b [OS[24]] 1 1
T0/T5 DerivativesT5 [OS[4], OS[5], OS[42]] 3 2
6Flan-T5 [OS[4], OS[5]] 2 1
T0 [OS[4], OS[5]] 2 1
CodeT5 [OS[42]] 1 1
CodeWhisperer [OS[21]] 1 1
MiscellaneousAdobe Firefly [OS[44]] 1 1
1Claude AI [OS[31]] 1 1
Codex [OS[42]] 1 1
Codium [OS[43]] 1 1
Cursor [OS[43]] 1 1
Falcon [OS[9]] 1 1
k8sgpt [OS[43]] 1 1
Mutable.AI [OS[43]] 1 1
N.A [OS[41], OS[23]] 2 1
Phi-3 [OS[37]] 1 1
Replit [OS[43]] 1 1
Robusta ChatGPT bot [OS[43]] 1 1
Tabnine [OS[43]] 1 1
Unknown [OS[33]] 1 1
Yi [OS[9]] 1 112

Human 
InteractionPrompt 
EngineeringModel
EnhancementFigure 3: How GenAI is used
LLM models SDLC
Adobe FireflyBardBertChatGPTClaude  AICodeWhispererCursorFalconGitHub Copilotk8sgptLLaMAMistralN.ARobustaT0T5TabnineYi
Arch-to-ArchArch-to-CodeCode-to-ArchReq-to-ArchReq-to-Arch-to-Code
Figure 4: Sankey Plot connecting LLM Models to SDLC Phase
program. Similarly,
As expected, microservices also have a strong appearance,
and studies investigating their architectural aspects in 6% of
the studies.
The purpose of preserving the microservice architecture
varies. For example, [OS[22]] uses LLM for analyzing the code
of a microservice-based system to answer architectural ques-
tions related to its designs (program comprehension). Simi-
larly, [OS[19]] focused on the identification of antipatterns in
a microservice-based system.
Other trends, such as Self-Adaptive Architecture, Server-
less, Layered Architecture, and Model-Based Architecture,
only appear erratically, each in 2% of the studies, showing lowresearch interest in these architectural styles.
An overwhelming 65% of the research failed to include any
data on architectural styles or trends, and it can be inferred
that the majority of the work carried out on LLMs within soft-
ware architecture does not necessarily correlate their conclu-
sions or base the focus on a certain architectural style.
Such an asymmetrical distribution demonstrates that al-
though the focus is given to some of the architectural schools,
especially monolithic and microservices, others are left unex-
plored regarding the application of LLMs.
Similarly to programming languages, we can represent SA
via many architectural languages (AL). Among such AL, UML
(Unified Modeling Language) is most commonly applied as a
notation in 17% of the studies (Table 16 - RQ 2.2) thus assess-
ing UML as the still dominant modeling language for studies
studying LLM due to its versatility in software design and ar-
chitecture documentation [19]. For example, [OS[34]] used
LLM to generate UML component diagrams from informal
specifications.
The remaining modeling approaches, i.e., C4, ADR (Ar-
chitecture Decision Records), SysML, and Knowledge Graphs
(KG), each appear only in a mere 2% of the studies, indicat-
ing little exploration of other architectural modeling nota-
tions. In particular, [OS[4]] uses ADR while using LLM to gen-
erate architectural design decisions with LLM. In contrast,
[OS[12]] investigated automating architecture generation us-
ing LLMs in Model-Based Systems Engineering using SysML
as the modeling language. [OS[41]] used KG for LLM-based
architectural reconstruction. Finally, [OS[14]] used a combi-
nation of UML and C4 for LLM-based assisted architectural
decision-making.
Most of the studies (74%) did not report any data on the use
of architectural modeling languages, suggesting that much
research on LLM in software architecture does not neces-
sarily use or elaborate formal modeling approaches. The
prevalence of UML and the non-wider deployment of rival
13

Table 13: How GenAI is used
Code PaperID Count %Prompt EngineeringUnspecified [OS[37], OS[9], OS[18], OS[22],
OS[25], OS[29], OS[33]]7 15%
Chain-of-
Thought[OS[7], OS[10], OS[42], OS[36]] 4 9%
Few-Shot [OS[1], OS[38], OS[4], OS[6], OS[8],
OS[11], OS[12], OS[13], OS[16],
OS[17], OS[43], OS[20], OS[21],
OS[26], OS[31], OS[34]]16 35%
One-Shot [OS[31]] 1 2%
Zero-Shot [OS[38], OS[4], OS[6], OS[43],
OS[31], OS[32]]6 13%
Total 34 74 %Model enhancementsUnspecified [OS[1], OS[7], OS[8], OS[10],
OS[11], OS[12], OS[13], OS[42],
OS[16], OS[17], OS[21], OS[22],
OS[26], OS[29], OS[31], OS[32],
OS[33], OS[34]]18 39%
Fine-Tuning [OS[37], OS[38], OS[4], OS[5],
OS[43], OS[36]]6 13%
Proprietary
RAG[OS[41]] 1 2%
RAG [OS[37], OS[5], OS[6], OS[9],
OS[41], OS[18], OS[43], OS[20],
OS[24], OS[25]]10 22%
Total 35 76 %Human Model InteractionYes [OS[1], OS[37], OS[38], OS[2],
OS[39], OS[40], OS[5], OS[6],
OS[7], OS[8], OS[9], OS[10],
OS[11], OS[12], OS[13], OS[14],
OS[15], OS[16], OS[17], OS[18],
OS[19], OS[44], OS[20], OS[45],
OS[21], OS[22], OS[23], OS[24],
OS[25], OS[26], OS[28], OS[29],
OS[46], OS[31], OS[32], OS[33],
OS[34], OS[35], OS[36]]39 85%
Model used as-is [OS[2], OS[3], OS[4], OS[5], OS[41],
OS[14], OS[15], OS[43], OS[19],
OS[23], OS[24], OS[27], OS[28],
OS[30], OS[42], OS[35], OS[39],
OS[40], OS[44], OS[45], OS[46]]21 46%
model languages suggest there is still sufficient scope for ex-
tension research combining LLMs and architected presenta-
tion forms.
On the topic of architectural design language, five stud-
ies reported using some form of Model Driven Engineering
(MDE) (Table 17 - RQ 2.2). More specifically, [OS[1]] used MDE
for the generation of the IoT architecture, while [OS[11]] for
low code platform consistency, [OS[33]] generation of UML
component diagrams, [OS[16]] mapping of the source code
to architecture, and [OS[26]] architectural conformance rec-
ommendation, each of which occurs in 14. 3% of the articles.
However, 86% of the articles did not contain information on
the utilization of MDE, so while there is evidence of research
that uses LLM for MDE applications, the topic is still fairly
unexplored compared to other architectural activities.Table 14: Use of LLMs in the Software Development Life Cycle - (RQ 2.1)
Code PaperID Count %
Req-to-Arch [OS[2], OS[3], OS[39], OS[40],
OS[4], OS[5], OS[6], OS[9], OS[12],
OS[14], OS[16], OS[17], OS[18],
OS[43], OS[44], OS[45], OS[21],
OS[23], OS[24], OS[46], OS[33],
OS[34], OS[35], OS[36]]24 40
Arch-to-Code [OS[1], OS[37], OS[38], OS[40],
OS[8], OS[10], OS[11], OS[13],
OS[42], OS[43], OS[44], OS[45],
OS[21], OS[22], OS[23], OS[29],
OS[31], OS[32], OS[36]]19 32
Code-to-Arch [OS[7], OS[41], OS[15], OS[19],
OS[25], OS[27], OS[28], OS[30]]8 13
Req-to-Arch-to-Code [OS[40], OS[43], OS[44], OS[45],
OS[21], OS[23], OS[36]]7 12
Arch-to-Arch [OS[20], OS[26]] 2 3
Table 15: Use of LLMs for Architectural Style and Patterns - (RQ 2.2)
Code PaperID Count %
N.A. [OS[37], OS[2], OS[39],
OS[40], OS[4], OS[5],
OS[6], OS[7], OS[9],
OS[10], OS[12], OS[13],
OS[15], OS[42], OS[16],
OS[17], OS[18], OS[43],
OS[44], OS[20], OS[45],
OS[21], OS[23], OS[24],
OS[25], OS[26], OS[29],
OS[46], OS[32], OS[34],
OS[35], OS[36]]32 68
Monolithic [OS[41], OS[14], OS[19],
OS[27], OS[28], OS[30],
OS[31]]7 15
Microservices [OS[3], OS[8], OS[22]] 3 6
Design Patterns [OS[33]] 1 2
Layered Architecture [OS[28]] 1 2
Model-Based Architecture [OS[11]] 1 2
Self-Adaptive Architecture [OS[1]] 1 2
Serverless [OS[38]] 1 2
5. RQ 2.2(Architectural Styles and Practices)
LLMs mainly target monolithic (12%) and microser-
vices architectures, with 65% of studies omitting style
details. UML dominates (17%), while alternatives (2%)
and MDE (14.3%) remain underexplored. Most stud-
ies (74%) lack formal architectural modeling.
Concerning quality aspects, 38% of the works explicitly dis-
cuss antipattern detection utilizing methods such as LLM-
based architectural smell refactoring, AI-based detection,
and rule-based learning ( RQ 2.2). In particular, [OS[19]] and
[OS[20]] use LLM to detect antipattern.
Concerning refactoring as a means of removing smells and
improving overall software quality, [OS[16]] and [OS[33]] use
LLM to aid in refactoring efforts. Moreover, [OS[13]] are the
only authors who use an external tool (EM-Assist) to aid in
14

Table 16: Architectural Modelling Language - (RQ 2.2)
Code PaperID Count %
N.A. [OS[1], OS[37], OS[38], OS[3],
OS[39], OS[5], OS[6], OS[7], OS[8],
OS[9], OS[10], OS[11], OS[13],
OS[15], OS[42], OS[16], OS[18],
OS[43], OS[19], OS[44], OS[20],
OS[45], OS[21], OS[22], OS[24],
OS[25], OS[26], OS[27], OS[28],
OS[29], OS[30], OS[46], OS[31],
OS[32], OS[33]]37 74
UML [OS[2], OS[40], OS[14], OS[17],
OS[23], OS[34], OS[35], OS[36]]8 17
ADR [OS[4]] 1 2
C4 [OS[14]] 1 2
Knowledge Graph [OS[41]] 1 2
SysML [OS[12]] 1 2
Table 17: Model-Driven Engineering (MDE) - (RQ 2.2)
Code PaperID Count %
N.A. [OS[1], OS[37], OS[38],
OS[3], OS[39], OS[40], OS[4],
OS[5], OS[6], OS[7], OS[8],
OS[10], OS[41], OS[11],
OS[12], OS[13], OS[14],
OS[15], OS[42], OS[16],
OS[17], OS[18], OS[43],
OS[19], OS[44], OS[20],
OS[45], OS[21], OS[23],
OS[24], OS[25], OS[26],
OS[27], OS[28], OS[29],
OS[30], OS[46], OS[31],
OS[32], OS[33], OS[34],
OS[35], OS[36]]30 86
IoT Architecture Generation [OS[1]] 1 3
Low-code Platform Consistency [OS[11]] 1 3
UML Component Diagram
Generation[OS[33]] 1 3
Source Code to Architecture
Mapping[OS[16]] 1 3
Architectural Conformance
Recommender[OS[26]] 1 3
refactoring, in conjunction with LLMs.
Similarly, studies that perform architectural reconstruc-
tion rely on LLM to achieve this. More specifically, [OS[15]]
used LLM to map code components to a specific architec-
ture, while [OS[28]] used LLM to recover the deductive soft-
ware architecture. Finally, only [OS[20]] reported the use of
external tools, validating the observation that LLMs are in-
creasingly being used to recover architectural knowledge and
are decreasing in strictly classical tools.Table 18: Architecture Analysis Method - Adopted Generative AI Outputs Val-
idation Methods - (RQ 2.3)
Code PaperID Count %
N.A. [OS[1], OS[37], OS[38], OS[3],
OS[39], OS[40], OS[4], OS[5],
OS[6], OS[7], OS[8], OS[10],
OS[41], OS[11], OS[12], OS[13],
OS[14], OS[15], OS[42], OS[16],
OS[17], OS[18], OS[43], OS[19],
OS[44], OS[20], OS[45], OS[21],
OS[23], OS[24], OS[25], OS[26],
OS[27], OS[28], OS[29], OS[30],
OS[46], OS[31], OS[32], OS[33],
OS[34], OS[35], OS[36]]43 93
ATAM [OS[9]] 1 2
SAAM [OS[2]] 1 2
Static Analysis [22] 1 2
6. RQ 2.3(Architectural Quality andMaintenance Tasks)
38% of studies use LLMs for antipattern detection,
refactoring ([OS[16]], [OS[33]]), and architectural re-
construction ([OS[15]], [OS[28]]). Few integrate exter-
nal tools, suggesting LLMs are replacing traditional re-
covery methods.
93% of the studies report that no information was provided
on the LLM model output validation techniques (Table 18 -
RQ 2.3) while only three of them report how they evaluated the
LLM model output. In particular, [OS[9]] used ATAM (Archi-
tecture Tradeoff Analysis Method), while [OS[2]] used SAAM
(Software Architecture Analysis Method) and [OS[22]] used
static analysis. Hence, our findings suggest that formal as-
sessment methods are still not in common practice, and most
studies do not explicitly validate their AI-generated architec-
tural designs.
7. RQ 2.4(Validation Methods)
ATAM, SAAM, and static analysis are the only valida-
tion methods reported, while 93% of the studies do
not report any evaluation strategy, indicating a lack
of systematic validation for AI-generated architectural
output.
4.4. Generative AI for Software Architecture: In which cases
(RQ 3)
This subsection presents the specific use cases in which
GeneAI has been applied to the software architecture. We ex-
amine the types of systems analyzed, the domains in which
LLMs are deployed, and the programming languages associ-
ated with these use cases. Table 19 presents the use cases and
systems addressed in the research papers that apply GenAI to
software architecture. According to Table 19, Requirements
15

and Architectural Snippets are the most common subject,
appearing in 16.1% of research papers, which indicates that
LLMs are widely tested in fragments of architectural informa-
tion [5, 24]. Enterprise and Property Software and IoT and
Smart Systems also attract significant interest, indicating ap-
plications in industrial and network environments. For ex-
ample, [OS[31]] used LLMs to re-engineer a legacy system at
Volvo Group. Since it is challenging to retrieve large-scale
open-source systems or to evaluate prioritized mobile appli-
cations and embedded systems, our findings evidenced how
such domains are underrepresented in our study. For exam-
ple, [OS[37]] experimented with retrieval-augmented gener-
ation (RAG) to evaluate green software patterns starting from
architectural documents of Instagram, WhatsApp, Dropbox,
Uber, and Netflix. Similarly, [OS[28]] investigated the archi-
tectural reconstruction of an Android app. Finally, 29% of the
research articles did not specify a precise use case, that is, po-
sition or vision articles.
8. RQ 3(Use Cases)
LLMs are most frequently applied to architectural re-
quirements and snippets (16.1%), with notable usage
in enterprise software and IoT systems (12.9%), while
large-scale, mobile, and embedded systems are less
explored.
Table 20 presents the programming languages of the use
cases examined. As is evident from Table 20, the most fre-
quent language is Java (9%), reflecting that Java systems are
leading the research on LLM applications in software archi-
tecture. Other languages, including JavaScript, Python, UML,
and natural language (NL), occur to a smaller extent, reflect-
ing a mix of implementation and design-level notation.
A significant 38% of the articles did not report the program-
ming language of the use case, and this is an area of reporting
that hinders the measurement of LLM uptake by the technol-
ogy stacks. The presence of legacy languages such as COBOL
(1%) suggests that there is research on legacy systems, but
only in a very limited subset of cases. These results show
that although Java is the most mentioned language, there is
no domination of any language, and the granularity of imple-
mentation decision details differs among studies.
9. RQ 3(Programming Languages)
Java (9%) is the language most commonly used in
LLM-driven architectural studies, but 38% of the stud-
ies do not specify a programming language, highlight-
ing a gap in reporting on implementation details.4.5. Generative AI for Software Architecture: Future Chal-
lenges (RQ 4)
This subsection presents the key challenges identified in
the original studies. Such challenges highlight limitations
in model reliability, ethical concerns, and the quality of AI-
generated outputs, which need to be addressed for broader
adoption.
Future challenges in GenAI research for SA include the ac-
curacy of LLM (15%), which is the most cited problem, sug-
gesting that maintaining accurate and reliable output is a pri-
mary challenge. LLM hallucinations (8%) are also a primary
challenge, indicating the need for mechanisms to prevent in-
correct or misleading model responses (Table 21 - RQ 4).
Ethics-related concerns (7%), privacy (7%), and human in-
teraction with LLM (5%) indicate that researchers are aware
of the need to align AI-produced outputs with responsible
and interpretable practices. In fact, [OS[18]] highlights ethi-
cal considerations as a major challenge in the use of GenAI for
software architecture. Although technology offers promising
advances, issues such as bias in AI-generated architectural
decisions and the lack of transparency in model reasoning
pose significant risks. Ensuring fairness and accountability in
AI-driven architectural solutions remains an open challenge,
particularly when AI systems are deployed in critical domains
like healthcare or finance. Meanwhile, [OS[44]] and [OS[43]]
echo these concerns, adding that privacy considerations fur-
ther complicate the adoption of AI in architecture. The risk
of accidentally leaking design information through LLM out-
put raises the need for stronger data protection mechanisms.
Addressing these challenges requires a combination of regu-
latory frameworks, improved model interpretability, and ro-
bust security measures to make GenAI a reliable tool for soft-
ware architects.
Quality of generated code, maintainability, scalability, and
security concerns are also mentioned, although each cate-
gory individually represents a limited number of studies. In
addition, 15% of the studies did not mention any future chal-
lenges altogether, which implies that there are studies that do
not explicitly articulate the threats or weaknesses of the im-
plementation of LLM in software architecture.
In general, original studies reveal that accuracy, hallucina-
tions, and ethics are the most critical issues, with generated
code and AI-human interaction issues continuing to be areas
of debate. The fact that future challenges are not more fully
reported in certain studies indicates a need for more serious
consideration of LLM limitations in software architecture re-
search.
10. RQ 4(Programming Languages)
LLM accuracy (15%) and hallucinations (8%) are the
main concerns, alongside ethics, privacy, and AI-
human interaction, while code quality and security
are less focused.
16

Table 19: Use Cases and Systems Analyzed - (RQ 3)
Category PaperID %
Social Media and Large-Scale Systems [OS[37]] 3.2
Architectural documents of Instagram, WhatsApp, Dropbox, Uber, Netflix
Educational and Research Platforms [OS[10]] 6.5
BigBlueButton, JabRef, TEAMMATES, TeaStore
Cloud and Open-Source Solutions [OS[46], OS[32], OS[10], OS[20]] 9.7
Google Jump-Start Solution, Hadoop HDFS, MediaStore, Multiple Open-Source Projects
IoT and Smart Systems [OS[26], OS[1], OS[17], OS[12]] 12.9
IoT Reference Architectures, Smart City IoT System, Smartwatch App, Remote-Controlled Autonomous
Car
Mobile and Layered Applications [OS[28]] 3.2
Layered App (Android)
Low-Code and Microservices Architectures [OS[11], OS[8], OS[22]] 9.7
Low-Code Development Platforms, Microservices in GitHub, TrainTicket Microservice Benchmark
Monolithic and Traditional Architectures [OS[2]] 6.5
Monolithic, Single Component
Enterprise and Proprietary Software [OS[16], OS[29], OS[36], OS[31]] 12.9
Proprietary Enterprise Scenarios, Ordering System, SuperFrog Scheduler, Volvo SCORE System
Requirement and Architectural Snippets [OS[3], OS[38], OS[4], OS[5], OS[24],
OS[30], OS[27]]16.1
Requirement Snippets, Snippets of Code, Snippet of Architectural Design Records, Architectural Snippets
Automotive and Embedded Systems [OS[15]] 3.2
PX4 (Drone Software)
Text-Based and Specialized Systems [OS[35], OS[34]] 6.5
Text/Aviation System, Software Engineering Exam Traces
N.A. (Not Specified) [OS[39], OS[40], OS[6], OS[7], OS[9],
OS[41], OS[13], OS[14], OS[42], OS[18],
OS[43], OS[19], OS[44], OS[45], OS[21],
OS[23], OS[25], OS[33]]29.0
5. Discussion
This section discusses the challenges implied from or high-
lighted in the identified literature and elaborates on future
directions. Additionally, it summarizes the different perspec-
tives identified in white and gray literature.
One must note that the manuscript identified a high
concentration of studies on architectural decision support
(30%) and reverse engineering/reconstruction (37%), which
is much higher than the average and suggests what the cur-
rent trend is in our community. GenAI, in its current state,
can only provide basic, high-level design blueprints but re-
quires extensive detailing for more nuanced architectural de-
cisions [39]. However, there are multiple open challenges as
we elaborate next.
5.1. Open challenges
This section elaborated on open challenges.
Evaluation of decision support: Most work on decision
support or architecture reconstruction [5, 4, 6, 25, 26, 28,
37, 8] mentioned their contributions had a shallow evalua-
tion and required broader empirical evaluations to confirmfinding generalization. This might suggest that the scien-
tific community should prioritize long-term studies or exper-
iments on a large number of projects since many works on
simplified settings might set promises that might not be fea-
sible.
Context-awareness: Multiple works [5, 4, 6] cope with Ar-
chitecture Decision Records (ADR). They see architecture as
a set of key design decisions, and one of the important parts
of architectural knowledge management is capturing archi-
tectural design decisions, and this is typically done using
lightweight documents called ADR [5]. However, ADRs follow
inconsistent writing styles, and LLMs are not able to compre-
hensively capture the Design Decisions as per human-level
proficiency. This is because of missing contextual informa-
tion from diverse sources. Dhar et al. [5, 4] noted hardware
limitations to their model training.
Finetuning generated results: Reliance on GenAI might be
difficult in the current form. For instance, Arun et al. [38]
share an example where GPT-4 was most often time thor-
ough in adhering to function requirements; occasionally, it
produces code that is more challenging to adjust with mi-
nor changes, and such a situation might become difficult on
evolving system settings.
17

Table 20: Use Case Programming Language - (RQ 3)
Code PaperID Count %
N.A. OS[OS[1]] [OS[1], OS[37],
OS[39], OS[40], OS[6],
OS[7], OS[8], OS[9], OS[10],
OS[11], OS[12], OS[13],
OS[14], OS[42], OS[16],
OS[17], OS[18], OS[43],
OS[19], OS[44], OS[45],
OS[21], OS[23], OS[25],
OS[26], OS[31], OS[32],
OS[33], OS[34]]26 38%
Java [OS[38], OS[20], OS[22],
OS[27], OS[28], OS[29],
OS[30]]7 9%
NL [OS[3], OS[4], OS[5],
OS[24]]4 5%
JavaScript [OS[38], OS[29]] 2 3%
Python [OS[38], OS[36]] 2 3%
UML [OS[2], OS[35]] 2 3%
C++ [OS[15]] 1 1%
COBOL [OS[41]] 1 1%
Node.js [OS[29]] 1 1%
React [OS[29]] 1 1%
TypeScript [OS[38]] 1 1%
Unknown [OS[46]] 1 1%
Evaluation Metrics: Ensuring the effectiveness of software
agents requires continuous refinement and the use of au-
tomated evaluation metrics. However, there is an absence
of standard automated evaluation metrics for evaluating the
quality of the generated products [24]. Constraining agents
and new robust metrics are needed to detect and prevent po-
tential hallucinations or unwanted behaviors [18].
Evaluation Benchmarks: While there are dedicated
leaderboards for code generation tasks covering various types
of programming problems such as EvoEval, Evoevalplus, etc.,
there is a lack of standardized datasets and benchmarks for
architecture-specific data. Perhaps this is also one of the rea-
sons why there is more focus on code generation and main-
tenance as opposed to requirements and design. We be-
lieve that this requires a concerted effort of both the practi-
tioners and research community to create dedicated leader-
boards and standard benchmark data for different architec-
ture tasks such as architecture knowledge management, mi-
gration, refactoring, and traceability.
Explainability: Creating visual models and graphs is vital
for effectively visualizing and communicating the design of
complex systems [18]. Consequently, it is still not possible for
LLMs to generate a graphical UML depiction of complex sit-
uations [17]. Fujitsu [41] pioneered the launch of a software
analysis and visualization service. Their service targets enter-
prise and organizational modernization by investigating and
analyzing software, visualizing black-box application struc-
tures and characteristics, and generating design documents
using GenAI. The result aims to improve understanding ofTable 21: Future Challenges - (RQ 4)
Code PaperID Count %
LLM Accuracy [OS[38], OS[10], OS[15],
OS[16], OS[17], OS[18],
OS[32], OS[44], OS[43]]9 16%
N.A. [OS[31], OS[40], OS[46],
OS[41], OS[24], OS[5],
OS[27], OS[19], OS[14]]9 16%
LLM Hallucinations [OS[37], OS[6], OS[18],
OS[29], OS[35]]5 9%
Ethical Considerations [OS[18], OS[44], OS[43],
OS[2]]4 7%
Privacy [OS[18], OS[44], OS[43],
OS[35]]4 7%
Architectural Solution Validation [OS[9], OS[28], OS[15]] 3 5%
Data Privacy [OS[44], OS[43], OS[35]] 3 5%
Generated Code Maintenability [OS[22], OS[42]] 2 4%
Generated Code Quality [OS[23], OS[4], OS[15]] 3 5%
LLM Human Interaction [OS[9], OS[33], OS[2]] 3 5%
Traceability [OS[12], OS[21], OS[36]] 3 5%
Generated Code Security [OS[38], OS[39]] 2 4%
LLM Output Generalizability [OS[25], OS[3]] 2 4%
Reduced Human Creativity [OS[39], OS[45]] 2 4%
Pattern Recognition Accuracy [OS[37], OS[30]] 2 4%
Intellectual Property [OS[2]] 1 2%
current systems and facilitate the creation of optimal mod-
ernization plans.
Verification and Formal Methods: Apart from visualiza-
tion, there is an alternative pathway that bypasses human
experts. Formal methods need to be employed with GenAI
[35] to ensure results comply with what is needed in the fi-
nal product. Yet such methods are non in place for any of
the works we analyzed. Chandraraj [39] observed that GenAI
in software architecture can pose security challenges. The AI
might miss crucial aspects like securing the API endpoints,
enforcing data encryption protocols, or overlooking vital net-
work security measures such as firewalls or intrusion detec-
tion systems. The verification becomes extremely important
in such cases.
Semantic relationships between two artifacts: Fuchs et
al. [10] tackled the challenge of semantic relationships be-
tween two artifacts, which could be two documents or two
microservices. Tracing relationships might be challenging,
and formal definitions of artifact dependencies (rather than
symptoms) might be essential. For instance, when it comes
to cloud-native, an attempt for the taxonomy of microser-
vice dependencies has been set [20]. According to Quevedo
et al. [22] practitioners indicate that one of the most signif-
icant barriers to the evolution of cloud-native systems is the
missing system-centric perspective that allows one to reason
about the system evolution and see change implications or
understand design trade-offs. This indicates that researchers
must consider coping with decentralized codebases.
Prompt engineering and complex systems: While docu-
mentation generation might be a task for toy projects, real
systems might consist of hundreds of decentralized units
18

[22]. KPMG [43] emphasizes the importance of prompt en-
gineering. The process of overcoming GenAI’s challenges
while reaping its advantages has sparked a rapidly growing
field known as prompt engineering. When we consider com-
plex systems, access to in-detail documentation is essential
for evolution; however, with large systems, traditional docu-
mentation should be personalized for different user roles and
contexts. Traditional one-fit-for-all documentation in com-
plex systems would produce hundreds of pages. GenAI can
serve as living documentation interacting contextually with
various experts. As observed by Quevedo [22], prompt en-
gineering becomes a pivotal strategy in guiding the model
toward accurate and meaningful answers, supplementing
modern system documentation based on prompts. However,
as noted, GenAI can sometimes deduce answers that exceed
the specificity of the question, yet at other times, it may over-
shoot and provide fabricated or incomplete responses.
Emerging technologies: We hear about ChatGPT 4.5,
DeepSeek, and future models or agents. Still, there is the chal-
lenge of handling large documents or code due to the con-
text window limitations of generative models, which measure
input capacity in tokens (words or parts of words) [37]. If
we consider that cloud-native solutions for Uber, Netflix, X,
or others have hundreds of microservices and decentralized
codebases, we must assume new models might need to ac-
commodate realistic industrial systems.
Hallucination, Bias, and System Evolution: GenAI intro-
duces challenges related to bias, information hallucination,
transparency, and potential over-reliance on AI [29]. Model
hallucination and value misalignment lead to issues such as
irrelevant outputs and misalignment with engineering val-
ues, hindering the effectiveness of LLMs [12] It must be noted
that the innovative potential of AI is limited by the extent and
variety of data it has been trained on. Chandraraj [39] sug-
gests that if the AI is tasked with architecting a software so-
lution for cutting-edge technology, it may find it challenging
to offer innovative solutions. The reason is that it might not
have been trained on sufficient data pertinent to this field.
These issues must be articulated to practitioners when em-
ploying AI tools. Hallucination challenges can become diffi-
cult for evolving systems as models could become biased by
the past and suggest irrelevant proposals. To tackle this, the
earlier challenge proposed formal methods and verification
alongside explainability. Moreover, others have proposed re-
liance on evaluation metrics.
Architectural Degradation: With GenAI, one may have the
perception that the developers can directly use the generated
output. However, this cannot be the case as it may lead to
technical debt [21]. The GenAI tools should be seen as ac-
tive assistants [39] to engineers as they move through each
phase of the development life-cycle. Moreover, it is essen-
tial for developers to have a fundamental understanding of
the output generated by the AI tools, which is why UML-
like models should be provided to facilitate product adoption
[17]. Chandraraj [39] points out that when given more de-
tails, the AI adds more complexity to the solution. This over-
complication or over-engineering can make the developmentprocess harder than it needs to be (i.e., suggests serverless
over monolithic architecture).
5.2. Implications and future directions
AI-assisted programming [25] is an excellent opportunity
for short-term future direction. Yet, the products have to be
explainable, especially in terms of architecture decisions; this
correlates with the need for AI products to generate models
or graphs like UML sketches to explain to practitioners the
proposed products. We elaborate on multiple directions and
implications.
Formal Verification and AI-Driven Compliance Check-
ing: It is easy to start using GenAI tools; however, maximizing
the potential can be challenging [29]. Moreover, it might be
difficult to control the tools, and results need to be checked as
practitioners can easily accept suggestions relying on AI as an
oracle. Still, there were observed limits of GenAI to complex
tasks with resulting products in less usable [29]. This leads to
comprehension issues, which we mentioned with challenges
to generate UML-like models or diagrams to guide develop-
ers on explainability.
Integration across SDLC phases: Advancement can be
claimed once a complete single integrated GenAI for engi-
neering product development engages in all SDLC phases.
Currently, we see pieces of the puzzle not necessarily related
to the previous phases. An advancement would be to create
a framework guiding the integration of all the various tools
contributing to one entity or process [21].
Evolution, Continuous Architecture, Integration with
DevOps: Once we deploy GenAI to manage code, there must
be reinforcement learning for architecture optimization, and
this must take into account the current trends in software
systems such as cloud-native that employs decentralized ar-
chitecture [21]. Future perspectives might consider tooling
that adjusts the systems to their usage, integrating with De-
vOps by monitoring user requests and trends by tracing and
taking into account available hardware resources or their fi-
nancial costs. However, GenAI support for system evolution
must cope with hallucinations architecture degradation, and
given we are currently dealing with the pieces of a puzzle with
GenAI tools rather than a comprehensive framework for the
complete SDLC, there is a long path to this.
Documentation might become legacy: While writing doc-
umentation can be expedited by GenAI [29], will this still be
needed in the future? AI can provide interactive documen-
tation by reverse engineering the code or using other static
analysis approaches like those presented by Quevedo et al.
[22]. Currently, documentation generation requires human
intervention to ensure the usefulness, correctness, and valid-
ity of the text, and hallucination in evolving systems can be
difficult to overcome [22].
Who manages what was generated: Model-driven devel-
opment had one core problem: no one wanted to manage the
code that was generated, and when one did, the model gen-
eration would not work when the system evolves as it would
override the changes. There are similar questions to asks for
19

AI-generated code [21, 36]. Experimental productivity and
quality comparison studies between human-generated and
AI-generated code in a realistic environment are needed [29].
We need to prevent architectural degradation, and thus, ar-
chitectural metrics need to be in place.
Replacement of human experts: AI replacing humans can
be approached once we overcome trust and establish evalua-
tion metrics. For instance, Prakash [21] suggests GenAI helps
developers by that 25% to write code efficiently, fix bugs,
and improve software quality. However, it is important to be
aware of the challenges and ethical considerations associated
with GenAI [45]. AI algorithms are trained on data, and this
data can be biased. This bias can be reflected in the output of
AI models. GenAI tools can make mistakes, and they should
not be used to replace human judgment. It is also essential
to consider the ethical implications of AI-generated architec-
tural patterns and designs before using them.
Project management by human experts: Literature often
mentions that the discipline will move towards a field where
human experts manage projects where GenAI agents can pro-
totype or deliver tasks for them to manage [29]. This suggests
the opportunity for research on AI tools for project manage-
ment.
Focus on cross-team decentralized collaboration with AI:
Future vision must be elaborated on human-centered cross-
team collaboration. For instance, in microservices, we deal
with a lot of co-changes that involve various teams [21]. One
cannot ignore the fact that most current systems run on de-
centralized architecture connecting codebases where consis-
tency is essential when changes take place to limit ripple ef-
fects. Moreover, many issues emerging from the GenAI im-
pact are caused by the neglect of the socio-technical prob-
lems and human needs and values [29]. Could GenAI fa-
cilitate communication across teams when co-changes must
take place? Chandraraj [39] suggest that GenAI might over-
look team dynamics and organizational culture in its archi-
tectural suggestions. For example, it might propose a com-
plex solution without considering the team’s abilities or the
availability of developers with technical skills. It could also
suggest a solution that technically works but doesn’t align
with the organization’s broader objectives.
5.3. Differences between white and gray literature findings
Our study, being an MLR, covered both the white and
gray literature to explore GenAI for Software Architecture.
The findings revealed notable differences between these two
sources.
More specifically, the white literature, including peer-
reviewed conference papers and journal papers, mainly ad-
dresses formalizing and generalizing the contribution of
LLMs to formal software architecture processes. The white
literature focused on LLMs to automate or facilitate archi-
tectural decision making ,traceability , and model-driven
development. Such studies tend to present systematic exper-
iments, propose new methods, or present conceptual foun-
dations to bring LLM into software architecture activities.
Moreover, it has a tendency to investigate empirical aspectsof LLM use, such as how good they are at generating architec-
tural fragments or determining architectural conformance to
predefined standards.
The gray literature comprises blog posts, industry reports,
preprints, and white papers and has a more pragmatic and
timely focus. LLMs are typically being researched as work
productivity tools in contrast to science objects of intense in-
vestigation. Many sources in the gray literature portray LLMs
as assistants that assist in making ongoing software develop-
ment efforts more straightforward, that is, architecture re-
construction, mapping requirements to architectures, and
generating documentation . The ability of LLMs to act as
architectural design copilots, providing quick recommenda-
tions or insight versus delving deeper into analytical reason,
is predominantly what these resources highlight. In contrast
to white literature, gray literature features industry-led use
cases, for example, using LLMs to plan modernization, auto-
mate software lifecycles, and extract knowledge from current
codebases.
The main difference is in the assessment approach : the
white literature rigorously analyzes the performance of LLM
through empirical research, controlled experiments, and case
studies, while the gray literature must suffice with anecdotal
evidence or high-level summaries without formal endorse-
ment. Moreover, the white literature is more interested in
probing theoretical questions, such as the interpretability
and trustworthiness of architectural knowledge generated by
LLM. In contrast, gray literature tends to be positive and in-
troduces LLMs as enablers without critically addressing their
limitations.
In general, both types of literature promote knowledge of
LLM implementation in software architecture but differ with
respect to the purpose and level of critique. The white liter-
ature is more research-focused and methodologically clear,
and its purpose is to refine and establish LLM integration
within the architecture process. The gray literature offers a
rapid path to industry learning, whose goal is adoption, tool
reviews, and short-term benefits. Since technological hype is
a mixture of academic and industry interests, we performed
this MLR to capture both worlds and to present a comple-
mentary view of the state of the art.
6. Threats to Validity
The results of an MLR may be subject to validity threats,
mainly concerning the correctness and completeness of the
survey. We have structured this Section as proposed by
Wohlin et al. [17], including construct, internal, external, and
conclusion validity threats.
Construct validity . Construct validity is related to the gen-
eralization of the result to the concept or theory behind the
study execution [17]. In our case, it is related to the poten-
tially subjective analysis of the selected studies. As recom-
mended by Kitchenham’s guidelines [15], data extraction was
performed independently by two or more researchers and, in
case of discrepancies, a third author was involved in the dis-
cussion to clear up any disagreement. Moreover, the quality
20

of each selected paper was checked according to the protocol
proposed by Dybå and Dingsøyr [18].
Internal validity . Internal validity threats are related to
possible wrong conclusions about causal relationships be-
tween treatment and outcome [17]. In the case of secondary
studies, internal validity represents how well the findings rep-
resent the findings reported in the literature. To address these
threats, we carefully followed the tactics proposed by [15].
External validity . External validity threats are related to
the ability to generalize the result [17]. In secondary studies,
external validity depends on the validity of the selected stud-
ies. If the selected studies are not externally valid, the synthe-
sis of its content will not be valid either. In our work, we were
not able to evaluate the external validity of all the included
studies.
Conclusion validity . Conclusion validity is related to the
reliability of the conclusions drawn from the results [17]. In
our case, threats are related to the potential non-inclusion of
some studies. To mitigate this threat, we carefully applied the
search strategy, performing the search in eight digital libraries
in conjunction with the snowballing process [17], considering
all the references presented in the retrieved papers, and eval-
uating all the papers that reference the retrieved ones, which
resulted in one additional relevant paper. We applied a broad
search string, which led to a large set of articles, but enabled
us to include more possible results. We defined inclusion and
exclusion criteria and applied them first to the title and ab-
stract. However, we did not rely exclusively on titles and ab-
stracts to establish whether the work reported evidence of ar-
chitectural degradation. Before accepting a paper based on
title and abstract, we browsed the full text, again applying our
inclusion and exclusion criteria.
7. Conclusions
This study presents the results of a multivocal review of the
literature investigating the topic of LLM and GenAI applica-
tions in the domain of software architecture. It investigated
the various perspectives of such practices, including the ra-
tionales for applying different LLM models and approaches,
application contexts in the software architecture domain,
use cases, and potential future challenges. From four well-
recognized academic literature sources and the three most
popular search engines, it extracted 38 academic articles and
8 gray literature articles. The analyzed results show that LLMs
have mainly been applied to support architectural decision-
making and reverse engineering, with the GPT model being
the most widely adopted. Meanwhile, a few-shot prompting
is the most commonly adopted technique when human in-
teraction is involved in most studies. Requirement-to-code
and Architecture-to-code are the SDLC phases where LLMs
are mostly applied, while monolith and microservice archi-
tectures are the ones that draw the most attention in terms of
structured refactoring and anti-pattern detection. Further-
more, the LLM use cases spread from enterprise software
and IoT systems to large-scale mobile and embedded sys-
tems where Java is the most commonly used programminglanguage in such studies. However, LLMs also suffer from is-
sues such as accuracy and hallucinations, with other broader
issues that need to be addressed in the future. The study sys-
tematically summarizes the current practice of LLM adoption
in the software architecture domain, which shows clearly that
LLM can contribute greatly to helping software architects in
various aspects. It is optimistic that LLM, with fast-paced iter-
ative updates, can continue to contribute to this domain with
even more astonishing outcomes.
Acknowledgment
The research presented in this article has been partially
funded by the Business Finland Project 6GSoft, by the
Academy of Finland project MUFANO/349488 and by the Na-
tional Science Foundation (NSF) Grant No. 2409933.
Data Availability Statement
We provide our raw data, and the MLR workflow in our
replication package hosted on Zenodo1.
Declaration of generative AI and AI-assisted technologies in
the writing process
During the preparation of this work the author used Chat-
GPT in order to improve language and readability. After us-
ing this service, the authors reviewed and edited the content
as needed and take full responsibility for the content of the
publication.
References
[1] M. Esposito, F . Palagiano, V . Lenarduzzi, D. Taibi, Beyond Words: On
Large Language Models Actionability in Mission-Critical Risk Analysis,
in: Proceedings of the 18th ACM/IEEE International Symposium on Em-
pirical Software Engineering and Measurement, ESEM 2024, Barcelona,
Spain, October 24-25, 2024, ACM, 2024, pp. 517–527.
[2] V . Garousi, M. Felderer, M. V . Mäntylä, Guidelines for including grey liter-
ature and conducting multivocal literature reviews in software engineer-
ing, Information and Software Technology 106 (2019) 101–121.
[3] A. Kaplan, J. Keim, M. Schneider, A. Koziolek, R. Reussner, Combining
knowledge graphs and large language models to ease knowledge access
in software architecture research (2024).
[4] J. Corbin, A. Strauss, Basics of Qualitative Research: Techniques and Pro-
cedures for Developing Grounded Theory, 3 ed., SAGE Publications, Inc.,
2008.
[5] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo, J. M.
Zhang, Large language models for software engineering: Survey and
open problems, in: 2023 IEEE/ACM International Conference on Soft-
ware Engineering: Future of Software Engineering (ICSE-FoSE), IEEE,
2023, pp. 31–53.
[6] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy,
H. Wang, Large language models for software engineering: A system-
atic literature review, ACM Transactions on Software Engineering and
Methodology 33 (2024) 1–79.
[7] I. Ozkaya, Application of large language models to software engineering
tasks: Opportunities, risks, and implications, IEEE Software 40 (2023)
4–8.
1https://doi.org/10.5281/zenodo.15032395
21

[8] J. Jiang, F . Wang, J. Shen, S. Kim, S. Kim, A survey on large language
models for code generation, arXiv preprint arXiv:2406.00515 (2024).
[9] J. Wang, Y. Huang, C. Chen, Z. Liu, S. Wang, Q. Wang, Software testing
with large language models: Survey, landscape, and vision, IEEE Trans-
actions on Software Engineering 50 (2024) 911–936.
[10] N. Marques, R. R. Silva, J. Bernardino, Using chatgpt in software re-
quirements engineering: A comprehensive review, Future Internet 16
(2024) 180.
[11] P . d. O. Santos, A. C. Figueiredo, P . Nuno Moura, B. Diirr, A. C. Alvim,
R. P . D. Santos, Impacts of the usage of generative artificial intelligence
on software development process, in: Proceedings of the 20th Brazilian
Symposium on Information Systems, 2024, pp. 1–9.
[12] A. Saucedo, G. Rodríguez, Migration of monolithic systems to microser-
vices using ai: A systematic mapping study, in: Anais do XXVII Congresso
Ibero-Americano em Engenharia de Software, SBC, 2024, pp. 1–15.
[13] A. S. Alsayed, H. K. Dam, C. Nguyen, Microrec: Leveraging large lan-
guage models for microservice recommendation, in: Proceedings of
the 21st International Conference on Mining Software Repositories, MSR
’24, 2024, p. 419–430.
[14] B. Gustrowsky, J. L. Villarreal, G. H. Alférez, Using generative artificial
intelligence for suggesting software architecture patterns from require-
ments, in: K. Arai (Ed.), Intelligent Systems and Applications, Springer
Nature Switzerland, Cham, 2024, pp. 274–283.
[15] B. Kitchenham, S. Charters, Guidelines for performing systematic liter-
ature reviews in software engineering, 2007.
[16] B. Kitchenham, P . Brereton, A systematic review of systematic review
process research in software engineering, Information & Software Tech-
nology 55 (2013) 2049–2075.
[17] C. Wohlin, Guidelines for snowballing in systematic literature studies
and a replication in software engineering, in: EASE 2014, 2014.
[18] T. Dybå, T. Dingsøyr, Empirical studies of agile software development:
A systematic review, Inf. Softw. Technol. 50 (2008) 833–859.
[19] M. Esposito, F . Palagiano, V . Lenarduzzi, D. Taibi, On Large Language
Models in Mission-Critical IT Governance: Are We Ready Yet?, arXiv
preprint arXiv:2412.11698 (2024).
[20] A. S. Abdelfattah, T. Cerny, M. S. H. Chy, M. A. Uddin, S. Perry, C. Brown,
L. Goodrich, M. Hurtado, M. Hassan, Y. Cai, et al., Multivocal study
on microservice dependencies, Journal of Systems and Software (2025)
112334.
[21] L. Lelovic, A. Huzinga, G. Goulis, A. Kaur, R. Boone, U. Muzrapov, A. S.
Abdelfattah, T. Cerny, Change impact analysis in microservice systems:
A systematic literature review, Journal of Systems and Software (2024)
112241.
Original Studies
[OS1] B. Adnan, S. Miryala, A. Sambu, K. Vaidhyanathan, M. De Sanc-
tis, R. Spalazzese, Leveraging llms for dynamic iot systems gener-
ation through mixed-initiative interaction, in: 2025 IEEE 22nd Inter-
national Conference on Software Architecture Companion (ICSA),
2025.
[OS2] A. Ahmad, M. Waseem, P . Liang, M. Fahmideh, M. S. Aktar, T. Mikko-
nen, Towards human-bot collaborative software architecting with
chatgpt, in: Proceedings of the International Conference on Evalua-
tion and Assessment in Software Engineering (EASE ’23), ACM, New
York, NY, USA, 2023, p. 7.
[OS3] S. Arias, A. Suquisupa, M. F . Granda, V . Saquicela, Generation of
Microservice Names from Functional Requirements: An Automated
Approach, Springer Nature Switzerland, Cham, 2024, pp. 157–173.
[OS4] R. Dhar, K. Vaidhyanathan, V . Varma, Can llms generate architectural
design decisions? - an exploratory empirical study, in: 2024 IEEE
21st International Conference on Software Architecture (ICSA), 2024,
pp. 79–89.
[OS5] R. Dhar, K. Vaidhyanathan, V . Varma, Leveraging generative ai for
architecture knowledge management, in: 2024 IEEE 21st Interna-
tional Conference on Software Architecture Companion (ICSA-C),
2024, pp. 163–166.
[OS6] J. A. Díaz-Pace, A. Tommasel, R. Capilla, Helping novice architects
to make quality design decisions using an llm-based assistant, in:European Conference on Software Architecture, Springer, 2024, pp.
324–332.
[OS7] J. A. Diaz-Pace, A. Tommasel, R. Capilla, Y. E. Ramirez, Architecture
exploration and reflection meet llm-based agents, in: 2025 IEEE
22nd International Conference on Software Architecture Compan-
ion (ICSA), 2025.
[OS8] C. E. Duarte, Automated microservice pattern instance detection
using iac and llms, in: 2025 IEEE 22nd International Conference on
Software Architecture Companion (ICSA), 2025.
[OS9] T. Eisenreich, S. Speth, S. Wagner, From requirements to architec-
ture: An ai-based journey to semi-automatically generate software
architectures, in: Proceedings of the 1st International Workshop on
Designing Software, 2024, pp. 52–55.
[OS10] D. Fuchß, H. Liu, T. Hey, J. Keim, A. Koziolek, Enabling architecture
traceability by llm-based architecture component name extraction
(2025).
[OS11] N. Hagel, N. Hili, A. Bartel, A. Koziolek, Towards llm-powered consis-
tency in model-based low-code platforms, in: 2025 IEEE 22nd Inter-
national Conference on Software Architecture Companion (ICSA),
2025.
[OS12] O. Von Heissen, F . Hanke, I. Mpidi Bita, A. Hovemann, R. Dumitrescu,
et al., Toward intelligent generation of system architectures, DS 130:
Proceedings of NordDesign 2024, Reykjavik, Iceland, 12th-14th Au-
gust 2024 (2024) 504–513.
[OS13] J. Ivers, I. Ozkaya, Will generative ai fill the automation gap in soft-
ware architecting?, in: 2025 IEEE 22nd International Conference on
Software Architecture Companion (ICSA), 2025.
[OS14] J. Jahi´ c, A. Sami, State of practice: Llms in software engineering and
software architecture, in: 2024 IEEE 21st International Conference
on Software Architecture Companion (ICSA-C), 2024, pp. 311–318.
[OS15] N. Johansson, M. Caporuscio, T. Olsson, Mapping source code
to software architecture by leveraging large language models, in:
A. Ampatzoglou, J. Pérez, B. Buhnova, V . Lenarduzzi, C. C. Venters,
U. Zdun, K. Drira, L. Rebelo, D. Di Pompeo, M. Tucci, E. Y. Naka-
gawa, E. Navarro (Eds.), Software Architecture. ECSA 2024 Tracks and
Workshops, Springer Nature Switzerland, Cham, 2024, pp. 133–149.
[OS16] J. a. J. Maranhão, E. M. Guerra, A prompt pattern sequence approach
to apply generative ai in assisting software architecture decision-
making, in: Proceedings of the 29th European Conference on Pat-
tern Languages of Programs, People, and Practices, EuroPLoP ’24,
Association for Computing Machinery, New York, NY, USA, 2024.
[OS17] R. Lutze, K. Waldhör, Generating specifications from requirements
documents for smart devices using large language models (llms),
in: M. Kurosu, A. Hashizume (Eds.), Human-Computer Interaction,
Springer Nature Switzerland, Cham, 2024, pp. 94–108.
[OS18] B. M. Rivera Hernández, J. M. Santos Ayala, J. A. Méndez Melo, Gen-
erative ai for software architecture (2024).
[OS19] J. Miño, R. Andrade, J. Torres, K. Chicaiza, Leveraging genera-
tive artificial intelligence for software antipattern detection, in:
S. Li (Ed.), Information Management, Springer Nature Switzerland,
Cham, 2024, pp. 138–149.
[OS20] G. Pandini, A. Martini, A. Nedisan Videsjorden, F . Arcelli Fontana,
An exploratory study on architectural smell refactoring using large
language models, in: 2025 IEEE 22nd International Conference on
Software Architecture Companion (ICSA), 2025.
[OS21] M. Prakash, Role of Generative AI tools (GAITs) in Software Develop-
ment Life Cycle (SDLC)-Waterfall Model, Massachusetts Institute of
Technology, 2024.
[OS22] E. Quevedo, A. S. Abdelfattah, A. Rodriguez, J. Yero, T. Cerny, Evaluat-
ing chatgpt’s proficiency in understanding and answering microser-
vice architecture queries using source code insights, SN Computer
Science 5 (2024) 422.
[OS23] P . Raghavan, Ipek ozkaya on generative ai for software architecture,
IEEE Software 41 (2024) 141–144.
[OS24] G. Rejithkumar, P . R. Anish, J. Shukla, S. Ghaisas, Probing with preci-
sion: Probing question generation for architectural information elic-
itation, in: 2024 IEEE/ACM Workshop on Multi-disciplinary, Open,
and RElevant Requirements Engineering (MO2RE), 2024, pp. 8–14.
[OS25] K. R. Larsen, M. Edvall, Investigating the impact of generative ai on
newcomers’ understanding of software projects, 2024.
[OS26] R. Rubei, A. Di Salle, A. Bucaioni, Llm-based recommender systems
22

for violation resolutions in continuous architectural conformance,
in: 2025 IEEE 22nd International Conference on Software Architec-
ture Companion (ICSA), 2025.
[OS27] S. A. Rukmono, L. Ochoa, M. R. Chaudron, Achieving high-level soft-
ware component summarization via hierarchical chain-of-thought
prompting and static code analysis, in: 2023 IEEE International Con-
ference on Data and Software Engineering (ICoDSE), 2023, pp. 7–12.
[OS28] S. A. Rukmono, L. Ochoa, M. Chaudron, Deductive software archi-
tecture recovery via chain-of-thought prompting, in: Proceedings of
the 2024 ACM/IEEE 44th International Conference on Software En-
gineering: New Ideas and Emerging Results, ICSE-NIER’24, Associa-
tion for Computing Machinery, New York, NY, USA, 2024, p. 92–96.
[OS29] L. Saarinen, Generative ai in software develop-ment, Information
Technology (2024).
[OS30] C. Schindler, A. Rausch, Formal software architecture rule learning:
A comparative investigation between large language models and in-
ductive techniques, Electronics 13 (2024).
[OS31] V . Singh, C. Korlu, O. Orcun, W. K. Assunçao, Experiences on using
large language models to re-engineer a legacy system at volvo group,
in: IEEE International Conference on Software Analysis, Evolution
and Reengineering (SANER), 2025.
[OS32] M. Soliman, J. Keim, Do large language models contain software ar-
chitectural knowledge? an exploratory case study with gpt, in: 2025
IEEE 22nd International Conference on Software Architecture Com-
panion (ICSA), 2025.
[OS33] V . Supekar, P . MIT WPU, R. Khande, Improving software engineering
practices: Ai-driven adoption of design patterns (2024).
[OS34] A. Tagliaferro, S. Corbo, B. Guindani, Leveraging llms to automate
software architecture design from informal specifications, in: 2025
IEEE 22nd International Conference on Software Architecture Com-
panion (ICSA), 2025.
[OS35] S. Tang, X. Chen, H. Xiao, J. Wei, Z. Li, Using problem frames
approach for key information extraction from natural language re-
quirements, in: 2023 IEEE 23rd International Conference on Soft-
ware Quality, Reliability, and Security Companion (QRS-C), 2023, pp.
330–339.
[OS36] B. Wei, Requirements are all you need: From requirements to code
with llms, in: 2024 IEEE 32nd International Requirements Engineer-
ing Conference (RE), IEEE, 2024, pp. 416–422.
[OS37] N. Ahuja, Y. Feng, L. Li, A. Malik, T. Sivayoganathan, N. Balani,
S. Rakhunathan, F . Sarro, Automatically assessing software architec-
ture compliance with green software patterns, in: 9th International
Workshop on Green and Sustainable Software (GREENS’25), 2025.
[OS38] S. Arun, M. Tedla, K. Vaidhyanathan, Llms for generation of architec-
tural components: An exploratory empirical study in the serverless
world, arXiv preprint arXiv:2502.02539 (2025).
[OS39] K. Chandraraj, Generative ai in software architec-
ture: Don’t replace your architects yet, Medium, 2023.
URL: https://medium.com/inspiredbrilliance/
generative-ai-in-software-architecture-dont-replace-your-architects-yet-cde0c5d462c5 ,
accessed: 2025-03-02.
[OS40] D. W. Reach, The future of software architecture: Diagrams as code
(dac), YouTube, 2023. URL: https://www.youtube.com/watch?
v=4Q5koGd1XGA , accessed: 2025-03-02.
[OS41] Fujitsu, Fujitsu launches gen ai software analysis and visualiza-
tion service to support optimal modernization planning, Press
Release, 2025. URL: https://www.fujitsu.com/global/about/
resources/news/press-releases/2025/0204-01.html , ac-
cessed: 2025-03-02.
[OS42] T. Sharma, Llms for code: The potential, prospects, and problems,
in: 2024 IEEE 21st International Conference on Software Architec-
ture Companion (ICSA-C), 2024, pp. 373–374.
[OS43] K. Martelli, H. Cao, B. Cheng, Generative ai and the soft-
ware development lifecycle (sdlc), KPMG Report, 2023. URL:
https://kpmg.com/kpmg-us/content/dam/kpmg/pdf/2023/
KPMG-GenAI-and-SDLC.pdf , accessed: 2025-03-02.
[OS44] A. Nandi, Gen ai in software development: Revolution-
izing the planning and design phase, AIM Research,
2024. URL: https://aimresearch.co/council-posts/
gen-ai-in-software-development-revolutionizing-the-planning-and-design-phase ,
accessed: 2025-03-02.[OS45] S. Paradkar, Software architecture and design in the age of
generative ai: Opportunities, challenges, and the road ahead,
Medium, 2023. URL: https://medium.com/oolooroo/
software-architecture-in-the-age-of-generative-ai-opportunities-challenges-and-the-road-ahead-d410c41fdeb8 ,
accessed: 2025-03-02.
[OS46] R. Seroter, Would generative ai have made me a better software ar-
chitect? probably, Richard Seroter’s Blog, 2023. Accessed: 2025-03-
02.
23
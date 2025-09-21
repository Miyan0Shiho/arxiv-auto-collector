# Accelerating Discovery: Rapid Literature Screening with LLMs

**Authors**: Santiago Matalonga, Domenico Amalfitano, Jean Carlo Rossa Hauck, Martín Solari, Guilherme H. Travassos

**Published**: 2025-09-16 14:01:44

**PDF URL**: [http://arxiv.org/pdf/2509.13103v1](http://arxiv.org/pdf/2509.13103v1)

## Abstract
Background: Conducting Multi Vocal Literature Reviews (MVLRs) is often time
and effort-intensive. Researchers must review and filter a large number of
unstructured sources, which frequently contain sparse information and are
unlikely to be included in the final study. Our experience conducting an MVLR
on Context-Aware Software Systems (CASS) Testing in the avionics domain
exemplified this challenge, with over 8,000 highly heterogeneous documents
requiring review. Therefore, we developed a Large Language Model (LLM)
assistant to support the search and filtering of documents. Aims: To develop
and validate an LLM based tool that can support researchers in performing the
search and filtering of documents for an MVLR without compromising the rigor of
the research protocol. Method: We applied sound engineering practices to
develop an on-premises LLM-based tool incorporating Retrieval Augmented
Generation (RAG) to process candidate sources. Progress towards the aim was
quantified using the Positive Percent Agreement (PPA) as the primary metric to
ensure the performance of the LLM based tool. Convenience sampling, supported
by human judgment and statistical sampling, were used to verify and validate
the tool's quality-in-use. Results: The tool currently demonstrates a PPA
agreement with human researchers of 90% for sources that are not relevant to
the study. Development details are shared to support domain-specific adaptation
of the tool. Conclusions: Using LLM-based tools to support academic researchers
in rigorous MVLR is feasible. These tools can free valuable time for
higher-level, abstract tasks. However, researcher participation remains
essential to ensure that the tool supports thorough research.

## Full Text


<!-- PDF content starts -->

Accelerating Discovery: Rapid Literature
Screening with LLMs
Santiago Matalonga1*, Domenico Amalfitano2†,
Jean Carlo Rossa Hauck3†, Mart´ ın Solari4†,
Guilherme H. Travassos5†
1*Computing, Engineering and Physical Science, University of the West
of Scotland, High Street, Paisley, PA1 2BE, Renfrewshire, United
Kingdom.
2Department of Electrical Engineering and Information Technology
(DIETI), University of Naples Federico II, Via Claudio, 21, Napoli,
80125, NA, Italy.
3Departamento de Inform´ atica e Estat´ ıstica, Universidade Federal de
Santa Catarina, R. Rua Delfino Conti, s/n, Florianopolis, 88040-370,
Santa Catharina, Brasil.
4Facultad de Ingenier´ ıa, Universidad ORT Uruguay, Cuareim 1451,
Montevideo, 11100, Uruguay.
5Programa de Engenharia de Sistemas e Computa¸ c˜ ao, Univesidade
Federal do Rio de Janheiro, Avenida Hor´ acio Macedo 2030, Bloco H,
Centro de Tecnologia, Rio de Janheiro, 21941-914, RJ, Brasil.
*Corresponding author(s). E-mail(s): santiago.matalonga@uws.ac.uk;
Contributing authors: domenico.amalfitano@unina.it;
jean.hauck@ufsc.br; martin.solari@ort.edu.uy; ght@cos.ufrj.br;
†These authors contributed equally to this work.
Abstract
Background:Conducting Multi-Vocal Literature Reviews (MVLRs) is often
time and effort-intensive. Researchers must review and filter a large number
of unstructured sources, which frequently contain sparse information and are
unlikely to be included in the final study. Our experience conducting an MVLR on
Context-Aware Software Systems (CASS) Testing in the avionics domain exem-
plified this challenge, with over 8,000 highly heterogeneous documents requiring
1arXiv:2509.13103v1  [cs.SE]  16 Sep 2025

review. Therefore, we developed a Large Language Model (LLM) assistant to
support the search and filtering of documents.
Aims: To develop and validate an LLM-based tool that can support researchers
in performing the search and filtering of documents for an MVLR without com-
promising the rigor of the research protocol.
Method:We applied sound engineering practices to develop an on-premises
LLM-based tool incorporating Retrieval Augmented Generation (RAG) to pro-
cess candidate sources. Progress towards the aim was quantified using the Positive
Percent Agreement (PPA) as the primary metric to ensure the performance of
the LLM-based tool. Convenience sampling, supported by human judgment and
statistical sampling, were used to verify and validate the tool’s quality-in-use.
Results:The tool currently demonstrates a PPA agreement with human
researchers of90%for sources that are not relevant to the study. Development
details are shared to support domain-specific adaptation of the tool.
Conclusions:Using LLM-based tools to support academic researchers in rigor-
ous MVLR is feasible. These tools can free valuable time for higher-level, abstract
tasks. However, researcher participation remains essential to ensure that the tool
supports thorough research.
Keywords:Large Language Model, Multivocal literature review, Software Engineering
1 Introduction
Multi-Vocal Literature Reviews (MVLRs) are valuable research methods for gathering
evidence from the state of practice in software engineering [1]. The method has become
an increasingly popular way to tackle industry practices in the empirical software
engineering research community. From a research design perspective, when perform-
ing MVLRs, the sources the researcher is working with are unstructured and vary in
the depth of technical details. Therefore, the researcher typically works with mate-
rials that have sparse information density. This is time-consuming, as a significant
amount of effort must be invested in understanding and evaluating sources that will
not be considered during the analysis and reporting phase of the study. Despite this
time-consuming process, MVLRs remain a relevant research method, particularly for
domains where the desired evidence isn’t readily available in peer-reviewed academic
literature. The inherent trade-off is justified by the unique value of the information
uncovered, which can significantly advance the research domain.
Our research team has been developing theory and evidence on the challenges of
testing Context-Aware Software Systems (CASS), with a high interest in researching
how these systems are being tested in the industry. Consequently, we designed and
conducted a Multi-Vocal Literature Review (MVLR) to explore how the automotive
industry was testing CASS [2]. Within these MVLR in the Automotive domain, the
effort/value tradeoff issue of MVLR was evident yet manageable. However, when the
team began instantiating the research protocol for the avionics industry, it encountered
a domain with significantly more variation in sources, which presented an industry with
a complex supply chain and different industry sub-sectors. The researchers’ result was
2

a substantial increase in potential sources of interest by several orders of magnitude
compared to the automotive industry (see Section 2).
Therefore, we decided to explore the application of Large Language Models (LLMs)
to support the screening and selection of sources. This paper presents the design, devel-
opment, and validation of an on-premises LLM-based tool to support researchers in
searching and filtering sources that present unstructured information. As mentioned,
this is typical of MVLR studies, where researchers deal with sources that lack a struc-
tured internal organization. To advance our research agenda, we aimed to produce a
tool that would support the researchers in the MVLR without compromising the rigor
of the MVLR protocol. As such, we took special care that:
•The LLM adheres to the criteria outlined in the research protocol. This means
that the tool closely follows the inclusion and exclusion criteria defined in the
protocol, and care was taken to minimize the risk of hallucinations;
•The precision and recall risks associated with the tool are comparable to the
risks introduced by human researchers. This means that our starting point is
that even for highly cohesive teams, there is a certain degree of variation in how
humans interpret selection criteria. Our engineering process utilized measures
of statistical agreement to ensure that the LLM tool’s implementation of the
inclusion/exclusion criteria was comparable to how the research team understood
them. In short, the bias introduced by the tool should be similar to the bias
introduced by any member of our research team.
The result is an on-premises LLM-based tool that can screen and process multiple
sources, determining whether human researchers should invest time in researching
them. The effectiveness of the tool in discarding sources is comparable to that of human
researchers, as calculated through the Positive Percent Agreement (PPA) between
humans and the LLM-based tool of90%.
This paper contributes to the research body of LLM to support academic origi-
nal research by providing a reusable tool for other MVLRs. This tool frees valuable
researcher time that can be made available for higher-level activities. Furthermore, the
development process followed open science principles, ensuring reproducibility. Sup-
porting materials include all artifacts of the engineering process (prompts, code, data,
and analysis) for other teams to replicate our results in their research domains of
interest. The tool is based on open-source LLMs and can be run on-premises, without
relying on proprietary LLMs or commercial API providers.
This paper is organized as follows: section 2, frames the motivation for developing
the tool in light of the research agenda and its challenges. Section 3, presents back-
ground research of the applications of LLM to support different aspects of secondary
studies, including MVLRs. Section 4, details the design and development of the LLM-
based tool. Section 5, presents the empirical results that show how the tool can be
used to filter sources for our MVLR. Section 6 presents some interesting points that
have to be taken into consideration for researchers trying to adapt the tool to their
research domain (including a discussion of threats to validity). Finally, in the section
7 present our conclusions and future work.
3

2 Problem Characterization and Motivation
This section aims to provide a brief context for the research agenda that led to the
development of the LLM-based tool. As we mentioned in the introduction, our research
interest lies in advancing the understanding and developing theories to support the
Testing of CASS. This research line has been ongoing since 2015, [3, 4]. As a result,
a significant portion of this team’s research efforts has been dedicated to identifying
approaches in peer-reviewed literature that align with this perspective. Needless to
say, we do not claim that these are the only systematic reviews concerning the testing
of CASS. Other groups have conducted secondary studies on the topic [5]. However,
thus far, our research has uncovered very limited evidence of testing approaches that
encompass all these criteria [6].
When examining the scientific peer-reviewed literature, a key issue with the lim-
ited number of sources covered by these secondary studies is their restricted capacity
for generalization. These works argue that the core problem stems from the lack of
standardization in the terminology of our discipline. For instance, terms like ”test-
ing,” which one might expect to have a clear and consistent definition, are subject to
varying interpretations. For example, the ISO 29119 [7] defines reviews as ”static test-
ing,” whereas the ISTQB [8] defines testing as the dynamic execution of a test item.
Consequently, researchers must exercise significant judgment when selecting relevant
works. A related challenge involves the interpretation of the term ”context.” It is an
overloaded term with numerous—albeit correct—applications that do not pertain to
the CASS domain. Combined with the limitations of keyword-based search engines,
this term overloading affects the precision and recall ratios of secondary studies. As a
result, it is common to observe secondary studies that screen numerous sources, only
to select a relatively small number of them. When performing MVLR, the problems
of the nomenclator remain, but the unstructured nature of non-peer-reviewed infor-
mation exacerbates them. Not only do researchers have to exercise judgment to filter
sources relevant to their review, but they also have to do so with sources that contain
very little information density.
This is a necessary trade-off that researchers accept to advance our research agenda.
In the following subsection, we aim to demonstrate that when transitioning from
the Automotive industry to the avionics industry, the effort required to overcome
information challenges would have effectively halted the advancement of research.
2.1 An instance of the MVLR Research Protocol for the
Automotive Domain
In this section, we present the key sections of the Automotive domain MVLR protocol
that are relevant to understanding the need for the LLM-based tool when compared
with the instance for the Avionics domain.
Aim:By conducting these MVLRs, we aim touncover evidence on howindustries
deploying CASSreport their work with the dynamic testing process regarding such
software systems.As a result, the first step was to identify these industries. Of those
industries, we started with the Automotive Industry [2]. The full protocol is available
in [9].
4

Searching for sources. In searching, we utilized a catalog of companies within
the automotive industry. This catalog was initially populated based on our existing
knowledge and further enhanced using Search String I (see Table 1). Sources were
subsequently identified through the application of Search String II, with the results
contributing to an iterative refinement of the company catalog. Notably, the search
was intentionally restricted to PDF documents. This decision was made because PDFs
published by these companies will likely contain curated and deliberate information
that the companies wish to share publicly.
Table 1Search Strings used in the Automotive instance of the MVLR protocol
Search String I Search String II
software test* AND automotive filetype:pdf software test* AND [company] filetype:pdf
Inclusion criteria: The following three inclusion criteria were developed for this
protocol.
IC1 The documents must be published in PDF format.
IC2 The document describes how the software system:
a) is tested (example reports), or
b) can be tested (future proposal), or
c) should be tested (standards).
IC3 The document focuses on testing an automotive software system fitting the
academic definition of CASS.
It is essential to note that, for the Automotive instance, the search was conducted
manually using the Google Search Engine, and the filtering was performed through
agreements, where each source was reviewed by three researchers, who then voted on
whether to include it in the study. We reviewed the first 100 Google search results for
each company, and accommodated Google’s algorithm variation by searching with all
our users and then merging the results. This entailed manually screening almost 1800
Google search result titles, which led to the identification of 120 potential PDFs. The
subsequent voting process narrowed this list to 36, which were then processed for data
extraction and analysis. As it is common with MVLR, the sources varied widely in
shape, size, and format. This inherent variability presents a significant challenge for
researchers tasked with identifying, filtering, and consolidating evidence.1
While 120 sources remained a feasible undertaking, researchers with experience
in executing MVLR will recognize the substantial redundancy inherent in this pro-
cess. Multiple researchers review the sources to mitigate bias, and significant time
is allocated to voting and discussion to achieve agreement in inclusion/exclusion
decisions.2
1Our attempt at categorizing the PDF types in the sample revealed a spectrum from short advertising
documents to extensive master’s theses (see [2]).
2The final number of studies included in [2] was 20. As sources are rejected during data extraction too.
5

2.2 Challenges with the Avionics Industry.
The situation became much more complex when we focused on the avionics domain.
First, the number of companies in the avionics domain is vastly bigger. The industry
is divided into large, complex supply chains that are far larger than the main known
brands. We made the scoping decision only to include those companies for which anon-
military manned aircraft will fly and be branded after a company name. This resulted
in 175 companies (for comparison, the automotive search included 18 companies). This
lead to our first design decision.
Automate the search process through the Google API. Initial attempts to automate
this process revealed that our population of potential documents would yield approxi-
mately 8,000-10,000 results. These possible sources would share the same complexities
for the automotive domain:
•Sparse information density
•Most sources would be rejected; yet, the team should still review them.
To confirm these hypotheses, we explored several sampling approaches. Figure 1
illustrates the outcomes of one such approach, where we tried to identify sampling
strategies by providing a bespoke classification of the sources. The overall conclusion
was that, regardless of the stratification of the sources, the majority of the mate-
rial would not be relevant for the research goals. Therefore, identifying the potential
sources would be effort-intensive3.
These initial results highlight the tradeoff between the perception of time required
to filter the sources and the benefits gained from our research. Our experience with
MVLR biases our perception of time to be proportional to the number of sources,
3We documented all sampling attempts in the Jupyter notebook ’SamplingExcercise’, available in our
replication package: https://git-lab.cos.ufrj.br/contextaware/llm
Fig. 1Initial assessment of potential sources for the avionics domain MVLR. A) shows potential
sources grouped by a custom classification. B) shows potential sources grouped by country of origin
of the related company
6

Table 2Indicative results of Screening early Search
results for the avionics industry
Population Sampled Include Doubt No
8482 1506 60 108 1338
and also multiplied by the number of researchers assigned to each source, where in
the past we have worked with three researchers per source to work out voting and
disagreements. The initial screening found that less than 4% (60/1,506) of the results
were potentially relevant. And even so, the quality of the material in those documents
was dubious, indicating that the final sample would likely be made up of an even
smaller percentage. Therefore, without a different approach that could automate part
of the screening and filtering task, the research line would have essentially come to
a halt. Consequently, we turned our attention to utilizing Large Language Models
to review and select (or, as the section 4 will show, discard) the sources so that our
attention and effort will be dedicated to reviewing sources that would be much more
likely to have relevant information for our research.
3 Related works on the applications of LLM to
support research and higher-abstraction tasks
Recent advancements in deep learning and natural language processing (NLP) have
sparked increased interest in automating Systematic Literature Reviews (SLRs). Large
Language Models (LLMs), particularly general-purpose models such as ChatGPT,
have been studied across various disciplines—including medicine, healthcare, educa-
tion, and the social sciences—for their potential to assist at different stages of the
SLR process [10]. Prior research has reported the use of LLMs to support tasks such
as formulating research questions, refining Boolean search strings, screening titles and
abstracts, applying inclusion and exclusion criteria, extracting data from primary stud-
ies, and synthesizing evidence. For example, Alshami et al. [11] and Wang et al. [12]
investigated LLMs’ ability to construct effective search queries, while others [13–16]
focused on their performance in automating the study selection process. Further con-
tributions, like Gupta et al. [17], explored their use in generating novel review ideas
and synthesizing insights from selected studies.
Within Software Engineering (SE), empirical research on this topic is com-
paratively limited but growing. Alchokr et al. [18] examined the application of
deep-learning-based language models to support SLRs in SE, highlighting their abil-
ity to cluster and filter relevant studies and reduce reviewer workload. Watanabe et
al. [19] proposed a supervised text classification approach to facilitate updating exist-
ing reviews, offering early empirical evidence of automation benefits in SE-specific
study selection tasks.
In addition to LLM-focused work, other studies have explored AI-based support
for SLR-related tasks in software engineering (SE) through various lenses. Girardi
et al. [20], for instance, conducted a systematic review of deep learning methods for
7

software defect prediction, indirectly highlighting the integration of data-driven tech-
niques into empirical SE research pipelines. Similarly, Necula et al. [21] presented a
systematic mapping study on applying NLP techniques in requirements engineering,
demonstrating the broader applicability of language models across SE subdomains.
Al-Shalif et al. [22] further reviewed metaheuristic feature selection strategies for text
classification, providing insights relevant to automating inclusion/exclusion decisions.
Despite promising results, several studies concur that LLMs should be utilized as
complementary tools, rather than replacements for human decision-making. Concerns
include limited transparency, hallucinated content, reliance on potentially outdated
training data, and lack of access to closed-access sources [23–26]. Huotala et al. [27],
for instance, found that LLM-based abstract screening did not outperform human
reviewers in SE. Felizardo et al. [28] offered a more comprehensive evaluation, applying
ChatGPT-4 to replicate the study selection phase of two SLRs in SE. While the
model achieved encouraging accuracy levels (75.3% and 86.1%), the authors emphasize
the risks of false exclusions and sensitivity to prompt design, recommending its use
primarily as a secondary reviewer or support mechanism for novice researchers.
In the context of LLMs in MVLR, Khraisha et al. [14] performed a feasibility
study to evaluate GPT-4’s performance in screening and extracting data for secondary
studies. While their results indicated significant variation in the tool’s effectiveness,
they highlighted the importance of tailoring prompts to each specific task.
Despite this growing body of work, the use of LLMs in the context ofMultivocal
Literature Reviews(MLRs) orGrey Literature Reviews(GLRs) remains largely unex-
plored. Most empirical evaluations to date have focused on structured, peer-reviewed
sources, with limited attention given to heterogeneous or informal content types that
are typical of grey literature. In summary, while preliminary evidence supports the
usefulness of LLMs in specific tasks related to literature review, their systematic
integration into multivocal review processes remains an open area for research. In
particular, we have observed that most research aims to evaluate LLMs’ capacity to
perform tasks, using datasets from previous (and published secondary studies). To
the best of our knowledge, this is the first tool specifically engineered to support an
ongoing research project.
3.1 LLM prompt best practices
Large Language Models (LLMs) hold significant potential to create high-quality digital
content. However, LLMs frequently present challenges in controlling the quality of
the responses [29]. The practice of prompting generative models allows end-users to
creatively assign novel tasks to LLM-based systems in an ad hoc manner, merely
by articulating them. Nonetheless, for most end-users, formulating effective prompts
remains predominantly reliant on trial and error [29].
Thus, prompt engineering has become an important technique for enhancing the
capabilities of pre-trained large-scale language models. This approach involves the
strategic formulation of task-specific instructions, known as prompts, to guide the
models’ generation of outputs without requiring modifications to their internal param-
eters. The relevance of this technique is particularly evident due to its transformative
8

impact on the adaptability and functional versatility of LLMs across different domains
and application contexts [30].
Many prompt engineering approaches have been proposed to improve interaction
with LLMs or to obtain better results [30–32], such as: (i) focusing on the amount of
response examples included in the prompts between zero-shot to few-shot prompting;
(ii) according to the form of reasoning and logic of the prompts, such as chain-of-
thought (CoT), self-consistency, logical CoT; (iii) varying in the approach to reduce
hallucinations such as react prompting, chain-of-verification or retrieval-augmented
generation (RAG); (iv) in the way of generating programming code such as scratchpad
prompting or program-of-thoughts (PoT) prompting; (v) for personalizing content
generation such as expert prompting, role-play prompting, among others [31, 33–35].
Therefore, selecting the best prompt technique depends on several factors: user intent,
model understanding, domain vocabulary, clarity and specificity, and constraints such
as response size or expected response format [36].
Even when choosing the correct prompt technique for a specific case, the best LLM
responses are often not obtained on the first attempt, requiring effort to refine the text
submitted to the models to achieve better results. Thus, prompts need to be refined
in several ways, through pure experimentation, reusing a library of prompts, reverse
engineering, using the models themselves to optimize the prompts, trying to reduce
the computational cost of processing the response, or decomposing a complex prompt
and chaining it into a sequence of subtasks [32, 37].
However, applying good prompting techniques and refining the quality of prompts
may often not be enough to prevent LLMs from producing so-called hallucina-
tions, which consist of generating content that appears factual to the reader but is
ungrounded [38]. This hallucination potential is inherent to LLMs, which are exposed
to massive amounts of text data during training, allowing them to achieve impres-
sive linguistic fluency and also to extrapolate information from the training data [38].
An alternative within prompt engineering to mitigate hallucinations is the Retrieval-
Augmented Generation (RAG) approach [39]. RAG enhances LLM-generated text by
combining the generation of the answer with an earlier retrieval step, in which external
expert knowledge is used to supplement the information. This integration improves
the accuracy of information retrieval and enables the model to provide more precise
answers to queries involving up-to-date information or very specific facts [35].
Given its ability to improve filtering, classification, synthesis, and extraction of text
content, prompt engineering approaches have also been used to support systematic
literature review activities. Zero-shot prompting strategies have been used to assist in
formulating research questions and creating initial eligibility criteria [24], in developing
search strategies [26], in screening titles and abstracts [13, 27] and in data extraction
[24]. One-shot and few-shot prompting are also used to screen titles and abstracts,
providing ”seed papers” for the LLM [18, 27]. In this sense, Chain-of-Thought prompt-
ing approaches have also been applied in screening titles and abstracts, where models
are encouraged to decompose systematic literature review activities into smaller steps
[27]. Role-play Prompting/Expert Prompting have also been employed, for example,
to instruct the LLM to assume a specific role in systematic literature reviews, such as
that of a “researcher who screens titles of scientific articles” or a domain “expert” [13].
9

4 The LLM-Based Tool for Searching and Filtering
in MVLRs
This section presents the development process of the on-premises LLM-based tool that
we specifically designed, built, and validated to support the execution of a Multivo-
cal Literature Review (MVLR). The tool was developed to automate two central and
effort-intensive phases of the MVLR protocol: (i) the identification of potentially rele-
vant documents from grey literature and (ii) their preliminary screening based on the
inclusion and exclusion criteria defined in our research protocol. The ultimate goal
was to enable scalability without compromising methodological rigor, ensuring that
the automated decisions remain consistent with those made by human researchers.
To this end, we defined and automated a structured process for the review. This
process was designed to reflect our MVLR protocol, in order to make sure that the
results obtained from the tool would be useful for continuing our research. Details
concerning the design assumptions and implementation strategies are discussed in the
following subsections.
4.1 LLM-Based Tool Design
The tool supports two primary activities within an MVLR: 1) a document retrieval
phase, guided by domain-specific search strings executed via web search engines, and 2)
a filtering phase, where each document is evaluated against protocol-driven criteria. As
illustrated in Figure 2, the tool implements these activities using a pipeline structure
composed of two main components, each corresponding to one of these activities.
The first component, theDocument Retrieval Component, receives as input one or
more structured search strings, typically derived from a PICOC formulation. It uses
Google as the underlying search engine to retrieve publicly available documents in
PDF format, which are then passed to the next stage of the process.
The second component, theLLM-Based Filtering Component, takes as input the
documents retrieved in the previous phase and evaluates each of them individually.
A Retrieval-Augmented Generation component (RAG) is employed to process each
document and incorporate the contents into the knowledge of the LLM model. The
goal is to determine whether the contents of the document satisfy the inclusion criteria
and do not meet any exclusion criteria, as defined in the MVLR protocol. This decision
process is delegated to a large language model, which evaluates the pertinence of the
content of the documents to the research.
In addition to the two core functional requirements, the key measure of success
was that the tool should not introduce more bias than what is typically associated
with human reviewers applying the same protocol. These requirements were assessed
during the Prompt Engineering phase of the process and are explained in section 5.
Furthermore, we added the restriction that the tool had to be executable on locally
available and resource-limited hardware, without relying on commercial cloud services.
It also had to be compatible with open-source technologies to ensure transparency,
replicability, and full adherence to the principles of open science.4
4Source code, datasets with execution results and Jupyter notebooks used for data analysis are available
in: https://git-lab.cos.ufrj.br/contextaware/llm
10

Fig. 2The LLM-Based Tool Pipeline Design
4.2 LLM-Based Tool Implementation
The implementation of the LLM-based tool reflects the structure and objectives
defined during the design phase and was instantiated in the context of a Multivo-
cal Literature Review focused on the avionics domain. The protocol was defined to
investigatehow context-aware software systems are tested in the civil aviation indus-
try, and it included explicit inclusion and exclusion criteria, as well as domain-specific
terminology derived from a PICOC-based formulation. These elements guided the
configuration of both tool components.
The implementation consists of two independent scripts. The first automates the
retrieval of grey literature documents from the web, while the second filters the
retrieved documents by assessing their relevance concerning the MVLR protocol. The
following subsections describe the implementation details of each component.
4.2.1 Document Retrieval Component Implementation
The automated search component of the tool was designed to retrieve grey litera-
ture documents in PDF format by querying the Google Custom Search API. Queries
were constructed to reflect two specific dimensions of the PICOC framework:Popu-
lationandIntervention, instantiated through domain-specific controlled vocabularies
selected according to the scoping decisions defined in the MVLR protocol.
The sets of terms used for query formulation are reported below:
•Population terms:Aviation,Aeronautics,Aerospace,Flight Science,
Air Transportation,Aeromechanics,Air Navigation,Avionics,Airspace
Management.
•Intervention terms:testing,test,verification,validation,quality
assessment.
To support the search within the avionics domain, we developed a Java-based
tool that automates the retrieval and download of PDF documents using the Google
Custom Search API. The tool executes structured queries based on predefined
combinations of domain-specific and testing-related keywords.
11

It accepts two sets of search terms—one for domain concepts (e.g.,Aviation,Aero-
nautics) and one for software testing activities (e.g.,testing,validation)—and supports
two query strategies: (i) a basic strategy combining a single term from each category
usingAND, and (ii) a broaderOR-based strategy that merges all domain terms with a
single testing-related keyword. The final query string includes theintext:directive
and restricts results to PDF format usingfiletype:pdf.
Once the query is built, the tool interacts with the Google API to iteratively
retrieve result pages in batches of ten, extracting candidate URLs from the JSON
response. The overall retrieval procedure is described in detail in Algorithm 1.
To comply with API constraints such as rate limits and pagination, the tool incre-
ments thestartindex to access subsequent result pages. It also validates HTTP
responses, filters by MIME type (application/pdf), and handles download failures
gracefully, accounting for invalid links, redirects, or paywalled content.
Since the Google Custom Search API limits each query to a maximum of 100
results, we adopted a structured formulation strategy to increase coverage. For each
intervention term, we generated a Boolean query that combines all population terms
using the OR operator and joins the result with the intervention term using AND. This
led to five distinct queries, each executed independently to retrieve up to 100 PDF
results. This approach maximized recall within the API constraints while preserving
thematic coherence across queries.
The retrieval process enabled scalable and reproducible collection of domain-
specific grey literature, resulting in a corpus of technical and industrial documents on
software testing in aeronautics. This dataset was used in the subsequent phases of the
multivocal review.
4.2.2 LLM-Based Filtering Component Implementation
The filtering component of the tool was designed to assess the relevance of documents
retrieved from the automated search phase, based on a set of predefined inclusion and
exclusion criteria.
The relevance of each document was assessed based on a set of predefined inclusion
and exclusion criteria. These criteria reflected our scoping decisions and were itera-
tively refined during the prompt engineering process to align with the objectives of
the MVLR.
Documents were considered for inclusion if they met one or more of the following
conditions:
IC1 The document concerns an aircraft that is manned or piloted;
IC2 The aircraft operates within the civil aviation domain;
IC3 The document indicates the existence of digital components or software in the
aircraft;
IC4 The document describes the design, execution, or reporting of the testing of
aircraft systems;
IC5 The document describes software testing techniques, technologies, processes, or
standards, and;
IC6 The described context is applied in the industry.
Conversely, documents were excluded if they met any of the following conditions:
12

Algorithm 1Automated Retrieval of Domain-Specific PDFs links
1:Input:
•populationTerms← {”Aviation”, ”Aeronautics”, ”Aerospace”, ”Flight Sci-
ence”, ”Air Transportation”, ”Aeromechanics”, ”Air Navigation”, ”Avionics”,
”Airspace Management”}
•interventionTerms← {”testing”, ”test”, ”verification”, ”validation”, ”qual-
ity assessment”}
2:Output:Downloadable PDFs links in .CSV and search logs
3:for allinterventionininterventionTermsdo
4:Construct query:
5:(intext:"term 1" OR ... OR intext:"term n") AND
intext:"intervention" AND intext:"software" filetype:pdf
6:Create output folder and initialize TXT and CSV log files
7:startIndex←1
8:whilemore results are availabledo
9:Compose API URL withstartIndexand query
10:Send HTTP GET request and parse JSON response
11:for allitems in responsedo
12:Extractlinkand write to TXT log
13:iflinkends with ‘.pdf”then
14:Validate content type and HTTP response
15:ifvalidthen
16:Log success in CSV
17:else
18:Log failure in CSV
19:end if
20:end if
21:end for
22:startIndex←startIndex+ 10
23:end while
24:end for
EC1 The document is an operating or installation manual;
EC2 The content focuses on military applications;
EC3 The subject is related to spacecraft;
EC4 The document only describes static analysis techniques, or;
EC5 The document refers exclusively to the design or execution of hardware-only
avionics systems.
Together, these criteria expressed our scoping decisions regarding the study. The
wording of the inclusion and exclusion criteria underwent multiple iterations during
the prompt engineering phase (see section 5.2), as we refined how to present them to
the LLM.
Algorithm 2 presents a high-level overview of the process. Essentially, the process
has three stages. Access and download the content, then encode it. Finally, submit the
content along with the question and prompt to the LLM for evaluation. The process
13

received three inputs: the list links to the PDF documents (in CSV format) that are
the results of the execution of theDocument Retrieval Component, theprompt, which
provides the context of the research, including the inclusion and exclusion criteria,
and thequestionthat offers the intervention for the LLM. Bothpromptandquestion
iteratively evolved as we fine-tune the behavior of the LLM-based filtering components
(see section 5.2).
Algorithm 2Algorithm for the LLM-Based component
1:Input:
2:1. CSV from search script with the List of Documents, format (ID, URL)
2. Prompt #Provides the context, IC and EC
3. Question # Provides the expectation
3:Output:
1. A set of Documents Downloaded and evaluated
2. An evaluation log with a decision for each document
4:for allDocument doc in the CSV of documentsdo
5:Access doc (wait a return for 12 seconds)
6:Download the document
7:ifdocument is downloadedthen
8:Encode()
9:Convert the PDF document to text and normalize it.
10:Collect relevant information from the embedded content.
11:Filter()
12:SendPrompt,Question, and encoding to the LLM for it to make an
evaluation about the pertinence of the Document.
13:ifanswer isYESorDOUBT then
14:Save the document in the folderPDF
15:else
16:Discard the document
17:end if
18:Register the answer in the evaluation log
19:else
20:Register NOT AVAILABLE in the evaluation log
21:end if
22:reset environment
23:end for
The implementation of this algorithm in Python can be quite straightforward.
However, communicating with the LLMs has two key points, which we describe in the
following subsections.
14

RAG and encoding
In the context of RAG, the document must be prepared so that the LLM can process
it. As mentioned, we sought out to use open-source technologies that could be executed
within the constraints of our computing hardware.
Our selection ofLlamaandmxbai-embed-large, both open-source, eliminates licens-
ing constraints that might otherwise impede the publication or sharing of our
methodology. Furthermore, all computing is executed locally, preventing any data
transmission to third parties, and showing that LLM can contribute to science with
relatively low-end hardware. It also conveys the reproducibility of our work, allowing
other researchers to replicate, build upon, and extend our findings easily.
PDF content is extracted and structured into plain text without reliance on exter-
nal services. Algorithm 3 outlines this stage of the RAG pipeline, where document
data is parsed and prepared for embedding. The resulting text, together with its
embeddings, feeds into the LLM’s inference stage. One key insight from this pro-
cess was to introduce overlap during sentence chunking (Lines 16–23), enhancing
contextual continuity across chunks and improving inference quality. To meet hard-
ware constraints and adhere to open science principles, we selected an open-source
foundational model that could run on available machines, avoiding reliance on pro-
prietary cloud-based services such as OpenAI ChatGPT. These hardware constraints
(see section 5.2 for the hardware specifications) determined the maximum model size
that could be employed for the filtering task. We selected two models: one for generat-
ing embeddings (mxbai-embed-large5) and one for assessing the relevance of sources
(dolphin-llama36).
Prompt Engineering
Our prompting strategy encompasses various techniques, including Role-Play and
Expert prompting (where we define a role to be assumed), Few-Shot prompting (where
we include examples of expected outputs), and Chain-of-Thought prompting (where
we explicitly define the rules and reasoning steps to be followed). This strategy allowed
for mapping the inclusion and exclusion criteria, as described in the research protocol,
to the instructions represented in the prompt. However, the tailoring required a series
of interactive trials to ensure the prompt and question were adequate to support the
filtering. Thisprompt engineeringprocess ran from November 2024 to February 2025.
Therefore, it is possible to observe five prompt and question versions, which evolved
based on the indications of suitability provided by the collected measures. The evolu-
tion of the questions (see table 3) and prompts was due to the level of detail and the
combination of inclusion and exclusion criteria used to influence the LLM’s behavior.
As it happened with the Questions, the single-shot prompt also evolved during the
prompt engineeringprocess. Table 7 presents a mapping between the questions, the
versions of the prompt, and the platform, and samples with which the results were
evaluated. After each modification of the prompt or questions, we would execute the
Algorithm against a suitable sample (see section 5.2 for the details of the evaluation
process and evolution of the PPA metric).
5https://ollama.com/library/mxbai-embed-large
6https://ollama.com/library/dolphin-llama3
15

Algorithm 3PDF Text Extraction and Chunking
Require:pdf name▷Filename of the PDF to process
Ensure:Chunked text is written tovault.txt
1:file path←pdf name
2:iffile pathis validthen
3:Openpdf filefromfile path
4:pdf reader←PyPDF2 reader ofpdf file
5:text←empty string
6:for allpage inpdf reader.pagesdo
7:extracted←page.extract text()
8:ifextractedis not emptythen
9:Appendextractedtotext
10:end if
11:end for
12:Normalize whitespace intext
13:Splittextintosentencesat punctuation boundaries
14:chunks←empty list
15:current chunk←empty string
16:for allsentence insentencesdo
17:iflength(current chunk+sentence) ¡ 1000then
18:Appendsentencetocurrent chunk
19:else
20:Appendcurrent chunktochunks
21:current chunk←sentence
22:end if
23:end for
24:ifcurrent chunkis not emptythen
25:Appendcurrent chunktochunks
26:end if
27:Write eachchunkas a line invault.txt
28:end if
For exemplification, we present the differences between V0.0 (see table 4) and V4.1
(see table 5) of the prompt7. The evolution of the prompt enabled the submission of
a detailed set of instructions to the LLM.
We conducted an exploratory analysis to assess the influence of the temperature
parameter. All versions were initially executed with a temperature of 0.5. In subsequent
runs involving versions 3 and 4, the temperature was lowered to 0.1. The results
indicated that this parameter had a limited impact, introducing a minor trade-off: a
marginal improvement in response confidentiality for a slight increase in processing
time.
7Major versions of the prompts used are available in the replication package https://git-
lab.cos.ufrj.br/contextaware/llm together with comparison of the main changes among each major version
16

User Questions Description
UQ0 Context-Aware Software Testing
UQ1 Does the document regard the testing of context-aware software systems?
UQ2 Would you select this document to support your activity of avionics context-
aware software systems?
UQ3 Is this document relevant and suitable for supporting the testing of avionics
context-aware software systems?
UQ4 Would you choose this document to support testing context-aware avionics soft-
ware systems in the industry?
Table 3versions of the User Question
You are a software tester specialized in selecting documents talking about testing context-aware
software systems of aircraft. You always select a document when all of these five rules are satisfied:
1 - An aircraft manned or piloted;
2 - An aircraft operating within civil aviation;
3 - The document indicates the existence of digital components or software in the aircraft;
4 - The document describes the design, execution, or reporting of the testing of aircraft systems;
5 - The document describes software testing techniques, software testing technologies, software
testing processes, or software testing standards.
You always reject a document when any of these four rules are satisfied:
1 - The document is an Operating or installation manual;
2 - The document describes Military applications;
3 - The document describes Spacecraft;
4 - The document describes only static analysis techniques.
If your suggested confidence level is>92, the< response >is *YES*.
If your suggesting confidence is<85, the< response >is *NO*.
If your suggested confidence level is>85 and<92, the< response >is *DOUBT*.
You always start your answer by informing the< response >, your confidence level in the range
of 0-100, and a brief explanation about your decision.
Table 4Prompt V0.0
5 Empirical observation of the prompt and
questions and their ability to screen sources
This section presents the iterative process we followed to evaluate the tools behavior
and drive the prompt engineering process. To articulate the purpose of this process
we used the Goal-Question-Metric approach, as reported below.
Analysean LLM-based filtering component, and its prompts.For the Purpose
ofcharacterizing its behavior.With respect toits capacity to judge the discard
of an information source in a way that is comparable to a human researcher
(using the Positive Percent Agreement metric)From the viewpoint ofsoft-
ware engineering researchersIn the context ofsupporting the researchers in
eliminating irrelevant sources of information for a multivocal literature review
regarding the testing of Context-Aware Software Systems of manned aircraft.
17

**Context**: You are an expert in context-aware software testing. You must choose software
testing documents to support testing context-aware avionics software systems for manned civil
aircraft in the industry. You consistently and professionally follow instructions and criteria to
support your choice and provide an answer.
**Instructions**: 1. Clear all of your previous document evaluations.
2. Evaluate the documents base on the following 13 rules:
- Rule 1: The document concerns a manned or piloted aircraft.
- Rule 2: The document concerns an aircraft operating within civil aviation.
- Rule 3: The document indicates the aircraft’s software.
- Rule 4: The document describes the design, execution, or reporting of the testing of avionics
software systems.
- Rule 5: The document describes techniques, technologies, processes, or standards for avionics
software testing.
- Rule 6: The document describes the planning, design, execution, or reporting of testing avionics
software systems.
- Rule 7: The document describes an application in the industry.
- Rule 8: The document is not an operating or installation manual.
- Rule 9: The document does not describe instruments, equipment, or toolkits to support software
testing in general.
- Rule 10: The document does not describe military applications.
- Rule 11: The document does not describe space aircraft or airspace applications.
- Rule 12: The document does not describe formal verification and validation methods.
- Rule 13: The document does not describe static analysis or verification techniques.
3. Provide your answer Observing a **Response Criteria** and using an **Output Template**.
**Response Criteria**: Set the ‘< choice >‘ to ”*YES*” if the software testing document satisfies
all 13 rules.
Set the ‘< choice >‘ to ”*NO*” if the software testing document does not satisfy any of the 13
rules.
Set the ‘< choice >‘ to ”*DOUBT*” if you cannot decide based on the rules
Justify your decision by filling in an ‘< explanation >‘ with two short phrases extracted from the
software testing document.
Set the ‘< confidencelevel >‘ with a 0 - 100% value to indicate your decision confidence.
**Output Template**: ‘< choice >‘; ”Confidence = ”; ‘< confidencelevel >‘; ‘< explanation >‘
**Examples of Output**:
- *YES*; Confidence = 94%; The document explains how to test context-awareness software
testing.
- *DOUBT*; Confidence = 91%; The document regards model-based testing to support the gen-
eration of context-awareness test cases.
- *NO*; Confidence = 82%; The document explains how to use formal methods to test software
systems.
Table 5Prompt V4.1
As previously stated, the goal and use case for the tool is to minimize research
bias, or at a minimum, ensure that any introduced bias is comparable to that of a
cohesive human research team. As such, this section describes the closed-loop iterative
process that was followed during the Prompt-Engineering process. For each change to
thepromptand/orquestion, we would execute the tool against a suitable sample of
PDFs and evaluate an objective metric to understand if the change brought us closer
to the goal.
Section 5.1 presents the rationale for adopting the Positive Percent Agreement
metric as our objective metric of choice, we evaluated other, maybe more frequent used,
18

agreement metrics before deciding that the Positive Percent Agreement Metric was
the one that most closely aligned with the goals. Following this, Section 5.2 presents
quantitative evidence demonstrating the ability of various prompt versions to classify
sources without introducing extraneous bias.
5.1 Evaluating metrics for observing the behavior of the
LLM-based tool
To determine whether the behavior of the LLM-based tool would contribute to our
goal, we explored inter-rater agreement statistical methods and applied them to assess
the progress and capacity of the LLM-based tool in selecting relevant sources.
While this is not the first work to employ LLM-based tools for secondary studies
(see Section 3), a salient feature embedded in our engineering process is the deliberate
avoidance of treating researcher judgment as absolute ground truth. We acknowledge
that even within a cohesive team like ours, the application of rules such as those stated
in the inclusion and exclusion criteria can lead to variations in judgment among team
members.
In short, we required a measure of agreement that: 1) would not rely on a predefined
ground truth; 2) would accommodate the inherent variability in judgment; and 3)
would specifically focus on evaluating the rejected papers (i.e.,Novotes).
Consequently, we explored several agreement statistics methods to guide the
development of our tool:
•Direct Perceptual Agreement: This can be calculated as the overall agree-
ment between two raters. However, this straightforward measure assumes that one
rater possesses the ground truth and does not account for randomness or varia-
tion in the decision. Therefore, we did not consider it suitable for our engineering
process.
•Cohen’s Kappa[40]: This statistic quantifies the agreement between two raters,
considering all possible judgment categories and adjusting for the possibility of
random agreement. Empirically, an agreement level above 30% generally indicates
genuine agreement beyond chance. While we initially considered Cohen’s Kappa,
we discarded its use because it directly compares two raters and considers all
voting categories (in our case: Yes, No, Doubt). Our tool’s primary goal, however,
was to discard sources that would not be relevant to our research (i.e.,Novotes
only).
•Fleiss’s Kappa[41], While we used Fleiss’ Kappa to demonstrate that our team
voting was consistent and exceeded the threshold of randomness, this metric is
not suitable for evaluating the output from the LLM-Based tool, as it is designed
for multi-rater agreement within a group, not for assessing an individual system’s
performance.
•Positive Percent Agreement(PPA), [42], is a statistical measure that quanti-
fies the proportion of positive cases correctly identified by a particular assessment
method. The PPA specifically focuses on the agreement within the positive cat-
egory. While its primary application is often seen in evaluating the efficacy of
diagnostic tests in correctly identifying the presence of a condition, its underly-
ing focus on a specific outcome category also applies to our context of evaluating
19

agreement on ’No’ votes. While the PPA directly measures agreement on the pos-
itive category, it implicitly acknowledges the potential for variation from other
output categories. Therefore, by focusing on the PPA for theNocategory, we
gain a targeted metric that reflects the consistency with which our human review-
ers agreed on the irrelevance of sources. A higher PPA in this context signifies
a greater degree of confidence that the LLM tool is effectively replicating the
reviewers’ ability to identify and discard irrelevant sources correctly.
5.2 Validation: Quality in use
This section presents the evolution of the PPA metric as we iterated through the devel-
opment stages of the tool outlined in the previous sections. As we mentioned before,
a key design consideration was the constrain of utilising readily available hardware
within our research team (and not relying on API-based solutions). To illustrate this,
the following hardware platforms were used for evaluating the different versions of the
promptandquestion:
•Platform 1 / Laptop with low-end GPU:Intel Core i7-EVO, 32 GB RAM,
1 TB SSD. Nvidia T500 4 GB GPU. OS: Windows 11.
•Platform 2 / GPU Desktop:Intel Core i7, 32 GB RAM, 1 TB SSD. Nvidia
RTX 4060 8 GB GPU. OS: Windows 11.
•Platform 3 / Workstation:2×Intel Xeon E5-2650 v3 2.30GHz, 192 GB RAM,
3 TB SSD. Nvidia RTX 4090 24 GB GPU. OS: Linux.
It is important to clarify that the presentation of these platforms is not intended
as a performance benchmark, but rather to demonstrate the feasibility of deploying
LLM tools on relatively low-end hardware platforms. Table 6 coveys how, as our
confidence in the tool grew, we also created more representative statistical samples
of the entire dataset. The decision to run larger datasets on more powerful hardware
aimed to reduce the turnaround time between each execution. Table 7 presents in
which platform each dataset was executed. The data in the Processing time column
indicates the execution time it took the team to obtain feedback from the LLM on
each platform.
Sample nameDevelopment Evaluation Validation Full
Purpose Programming Programming
and Prompt
EngineeringValidation in
UseContinuing
research agenda
Sampling
MethodConvenience
SamplingRandom / not
RepresentativeStatistical Ran-
dom / Represen-
tativeFull dataset
Size Small 1 to 10
PDFsMedium 58
PDFsLarge 368 PDFs Full dataset 8482
PDFs
Table 6Definition, sampling method, and purpose of each sample dataset
20

User
QuestionPrompt Version /
Releases8Platform Sample Processing Time
(min)
UQ0 0.0-0.4 1 Development<20
UQ1 1.0-1.8 1, 2 Development,
EvaluationIn Platform 2:
<5 (Development),
<18 (Evaluation)
UQ2 2.0-2.3 2, 3 Evaluation,
ValidationIn Platform 3:
<6 (Evaluation),
<23 (Validation)
UQ3 3.0-3.1 2, 3 Evaluation,
ValidationIn Platform 3:
<6 (Evaluation),
<23 (Validation)
UQ4 4.0-4.1 3 Validation, Full<23 (Validation),
<2160 (36 hours)
(Full)
8This column identifies the various executions the LLM-based tool underwent during development
and prompt engineering. While the naming strategy was generally a Major.Minor convention, seman-
tic information was occasionally added to version names during exploratory phases, as exemplified in
Fig. 3. For a detailed account of all prompt versions, readers are referred to the replication package.
Table 7Details of User Questions and Processing Times
5.2.1 Benchmark agreement of Inclusion and Exclusion criteria for
the avionics section
As mentioned in section 5.2, we iteratively sampled from the full dataset to ensure
each sample was more representative (see Table 6).
To apply the PPA statistics within our engineering process, we drew a random
sample of58sources, which is theEvaluationsample in Table 6. The sources in
theEvaluationsample were then independently assessed by three researchers on our
team, who assigned one of three votes: Yes (indicating relevance), Doubt (indicating
uncertainty), or No (indicating irrelevance). Subsequently, we employed a predefined
aggregation protocol9to determine the final inclusion status of each source. This
process yielded a dataset of58sources, each classified by three expert researchers,
representing a plausible set that could have progressed to the subsequent stage of
our MVLR protocol. As an internal measure of inter-rater reliability for the initial
voting, we calculated Fleiss’ Kappa, which yielded a value of0.49, indicating moderate
agreement among the researchers. And both confirm how, even within a cohesive team,
including and excluding criteria are subject to varying interpretations.
This dataset of58sources was internally designated asTeamAgreementand
served as the input for calculating the PPA for all iterations of the LLM-based
tool. Specifically, following each iterative development or modification of the tool (as
detailed above), we computed the PPA to evaluate its performance. An increasing
PPA value indicated a progressive alignment of the tool’s output with the consensus
reached by our research team. While the guiding metric is the PPA(No) metric, we
calculated and presented the PPA for the other two voting options. Figure 3 presents
the results of the selected version through the iterative process.
9For instance, a source was classified as ”Included” if all three researchers voted ”Yes,” or if at least
two voted ”Yes” and the third voted ”Doubt.” A unanimous ”Doubt” resulted in a ”Doubt” classification.
If exactly two researchers voted ”Doubt” and the third voted either ”Yes” or ”No,” the outcome was also
”Doubt.” All other voting combinations resulted in a ”No” classification.
21

Fig. 3PPA progression across versions with the 58 PDFs in theEvaluationsample
A local maximum was observed with version PromptV2.3, yielding a
PPA(No) = 0.79 . This result led to subsequent analysis and explorations of the
effects of code changes (PromptV2.3 BaseUpdated) and temperature adjustments
(PromptV2.3 Temp0.5). The outcomes, shown in Fig. 3, indicate some performance
variation across these attempts, yet none achieved the target metric.
Furthermore, we reviewed all the collected results in search of voting patterns
resulting from the different versions of the LLM. With this analysis, we were able to
identify a consistent voting pattern by the LLM across different iterations for a subset
of sources. For these sources, the LLM had consistently maintained the same Vote
through the other iterations. This analysis also required us to review the sources while
interacting with the LLM through the different prompt versions. Overall, we started to
realized that the interaction with the LLM was shaping our collective understanding
of the Inclusion and Exclusion Criteria (IC/EC).
Consequently, we selected a sample ofninepapers from theEvaluationdataset.
This sample intentionally included sources where the LLM had exhibited consistent
voting alongside a random selection of other sources. The same three researchers were
then asked to re-evaluate their initial votes for theseninepapers, without being
informed of the LLM’s voting consistency on any PDFs with a voting that differed
from theTeamAgreement. This review process resulted in a revised consensus for
the original58 sources in theEvaluationsample, which we namedTeamA-
greement V1. This refined dataset served as the benchmark against which the final
versions of the LLM were evaluated. In Fig. 4, the final version of the tool is shown in
comparison withTeamAgreement V1. The reader will note the difference in per-
formance for versionPromptV2.3 BaseUpdatedbetween Fig 3 and Fig 4, conveying
22

how our collective decisions regarding the 58 PDFs in theEvaluationchanged without
interaction with the LLM (we expand on the implications of this in section 6).
Fig. 4PPA progression across versions after reevaluating the 58 PDFs in theEvaluationsample
The results forPromptV4.1 Temp0.1warrant specific discussion. While this
version attained the highestPPA(No)score, it failed to register any agreement for
PPA(Yes)orPPA(Doubt). Upon inspection, we found that all sources were classified
as either ’Yes’ or ’No’, resulting in a single strong disagreement between the execution
and TeamAgreementV1. Although we hypothesize this deterministic behavior is due to
the low temperature setting, it serves as an important indicator of the potential risks
in automated source selection. In this case, one source out of the 58 in the Evaluation
sample would have been discarded (potentially erroneously) before subsequent MVLR
stages. This scenario, explored further in Section 6, represents an example of the
fundamental trade-off that motivated this research: weighing the value of saved time
and effort against the research risk posed by potential false negatives.
5.2.2 Evaluation of the LLM version against the Large
representative sample
The process described in the previous section provided sufficient confidence that the
LLM’s votes were consistent with those of the researchers. To evaluate this consistency
more rigorously, we drew a statistically representative sample from the total population
of identified sources. For a population size of 8482 potential sources, a sample size of
368 was determined to be necessary to achieve a 95% confidence level with a 5% margin
23

of error, assuming a population proportion of 50%10. This sample was then processed
by the LLM-based tool, and each of the three researchers was assigned 73 or 74 sources
for individual assessment. The Positive Percent Agreement (PPA) calculated between
the LLM’s votes and the aggregated votes of the researchers for this sample was96%,
indicating a very high degree of consistency between the manual review process and
the LLM-based approach.
The distribution of votes for this representative sample is detailed in Table 8
Category Doubt Yes N/A No
LLM Vote 0 11 95 262
Table 8LLM votes and human votes for
the representative sample
5.3 Final run for research
The final step was submitting all identified sources to the LLM-based tool. This run
will allow us to continue our research into CASS Testing. The results are presented in
Table 9 and Fig. 5.
Category Doubt Yes N/A Not
LLM Vote 13 224 2798 5447
Table 9Distribution of final votes by the
team and LLM-Tool votes
6 Discussion
Our iterative engineering process for the LLM-based tool yielded several noteworthy
insights, which are discussed in detail below.
Prompt Engineering. Engineering involves applying knowledge to solve prob-
lems within a specific domain. In the context of Prompt Engineering, however,
the knowledge base remains relatively limited. As detailed in Section 3.1, a litera-
ture review was conducted to identify best practices for defining and improving our
prompts. Nevertheless, when evaluating the evolution of these prompts and their
impact on our chosen metric (PPA) using a controlled set of58source samples, the
process appeared significantly driven by trial and error. This observation suggests that
currentPrompt Engineeringpractices lack a robust foundation of established knowl-
edge. Despite this, our analysis did reveal several potentially valuable insights for the
further development of this practice. These insights are derived from observing the
versions of the LLM-based tool that demonstrated the most substantial effect on our
10Sample Size Calculator at Calculator.net
24

Fig. 5Distribution of the final run
guiding metric, as well as the tweaks that did not affect the model’s performance
against our goal.
The effects of ambiguous positive and negative statements in the Prompt
confused the LLM model.A significant improvement in performance during our
development process occurred when theexclusion criteriawere rephrased as direct
statements. Table 10 illustrates this difference. Initially, the prompt was structured
similarly to typical inclusion/exclusion criteria. However, a shift towards direct and
assertive statements yielded substantially better model performance. As shown in the
aftercolumn of Table 10, the Exclusion Criteria concerning operations manuals was
rewritten, aligning with the instruction “Evaluate the documents base on the following
13 rules”. We hypothesize that the model is confused by the instruction with a negative
outlook (i.e., the wordreject), yet it receives a set of affirmations (i.e.,the document
is).
Before After
...You alwaysrejecta document when any of
these five rules are satisfied:
1 - The document is an Operating or installa-
tion manual;...You always select a document when all of these
eleven rules are satisfied:
...
7 – The document is not an Operating or instal-
lation manual;
Table 10Prompt comparison between positive and negative statements
We note that determining the cause of this behavior is not within the scope of our
research. Furthermore, our concept of performance improvement, specifically concern-
ing the prompt and output, is closely tied to the LLM tool’s ability to reject irrelevant
25

papers. Nonetheless, we include this observation to contribute to the growing, yet still
preliminary, body of evidence surrounding prompt engineering approaches. We argue
that this reinforces our earlier point regarding the current state of prompt engineering,
which has not yet developed the characteristics of a mature engineering discipline.
Template Prompt for other teams using our toolAs mentioned, the tool
development closely followed the restrictions in the research protocol. However, we
have discussed at length what would be needed to change to use the tool in another
research domain. We leave here a proposed prompt template for others to use our tool
in their research. This prompt is based on our lessons learned and our current prompt,
and adopting the presentation from [33]:
[Expert] You are a{Role}specialized in selecting documents talking about
{Subject}.
[Instruction] You always select a document when all of these{Number of Inclu-
sion Criteria}rules are satisfied:
{Numbered list of Inclusion Criteria}1 –{Inclusion Criterion}; 2 -{Inclusion
Criterion};{n}-{Inclusion Criterion}.
[Instruction] You always reject a document when any of these{Number of
Inclusion Criteria}rules are satisfied:{Exclusion Criterion}1 –{Exclusion
Criterion}; 2 -{Exclusion Criterion}; n -{Exclusion Criterion}.
[One Shot Answer] If your suggested confidence level is>92,<response>is
set to ’*YES*’. If your suggested confidence is<85,<response>is set to
’*NO*’. If your suggested confidence level is>85 and<92,<response>is set
to ’*DOUBT*’.
[Instruction] You always start your answer by informing the<response>, your
confidence level in the range of 0-100%, and a brief explanation about your
decision.
Regarding Model Selection, Prompts, and Model Benchmarking.The
primary focus of our research lies in CASS testing; consequently, benchmarking the
performance of different LLM models falls outside our immediate scope. Our engage-
ment with LLMs is that of users leveraging an available technology. As such, our initial
model selection was primarily dictated by the computational resources available to
run a model locally. Therefore, the key criterion for choosingDolphin-llama311was
its feasibility for local deployment on our workstations.
Throughout our experimentation with this model and the iterative process of
Prompt Engineering, we hypothesize that there was a strong interdependency between
the prompt design and the specific LLM employed. Our repeated observations sug-
gested that simply transitioning to a newer model within the same family would not
necessarily yield improved performance without corresponding adjustments to the
prompt. To illustrate this, we conducted a small exploratory experimentation. For
this, we used the58source samples and made no changes to the source code, main-
taining the same values for parameters such as temperature (set at 0.1), as well as the
question and prompts. We chose three models that a search showed were variations
11https://ollama.com/library/dolphin-llama3
26

of the one we used throughout the development. These models are:llama3in its 8B
parameter version,llama3.1in its 8B parameter version, and the latest updated ver-
sion ofdolphin-llama3, also in its 8B parameter version (which is safe to assume is
the most similar model to the one used throughout development).
Our results demonstrate that the performance of the tool varies significantly
depending on the underlying model used for the judgments (see Figure 6). For
researchers seeking to utilize our tool, these results reinforce the argument that the
question and prompt must be carefully evaluated and tailored to the research domain
to ensure that the tool’s results do not introduce unnecessary bias into the selection
process.
Fig. 6PPA for the execution of the tool with different underlying models without changes to Prompt
or Question
Similar to the previous point, our observations regarding prompt engineering and
model behavior are shared with the community to promote the advancement of LLMs
as supportive tools in specific application domains. These observations should not
be interpreted as a comprehensive analysis or benchmarking of LLM capabilities.
Further research focusing on the nuances of prompt engineering methodologies and
comparative model performance would be necessary to draw broader conclusions.
Processing time and hardware constrains. The overall processing time for
the full sample is related to the underlying hardware. However, our intention in exper-
imenting with different underlying platforms stems from pragmatism rather than
optimization. As mentioned previously, our choice of foundational LLM model was
driven and restricted by the capacity of our workstations to execute it locally. Since
27

each source (i.e., PDF) is processed individually, we were able to run the sample with
the hardware we had access to.
We also noted that the execution time of some steps remained fairly constant
regardless of the underlying execution platforms. For example, with minor variations
to account for network conditions, downloading each PDF takes roughly the same
amount of time on any of the platforms.
Improvements in hardware are most noticeable in the inference request to the LLM.
Optimization, state-of-the-practice RAG, and fit for purpose of the
LLM-toolAs mentioned in 4.2.2, we encoded the PDF content into plain text
before sending it to the LLM for processing. We acknowledge that Retrieval Aug-
mented Generation (RAG) best practices evolved rapidly during our development
cycle. For instance, current best practices often recommend encoding sources into
vector databases optimized for large language model (LLM) processing.
While this alternative approach could potentially improve execution times, our
primary goal was to demonstrate that the tool could free up researchers’ time without
injecting bias. Given the available hardware, the execution times were acceptable,
allowing us to prioritize validating the tool and ensuring the quality of its responses
for our research purposes.
6.1 Limitations and threats to validity
In this section, we discuss several limitations and threats to the validity of using the
tool to support research.
Effects of feedback and auto-influence.A potential threat to the construct
validity of our study lies in the evolving understanding of the research domain, the
quality and content of the available sources, and the application ofinclusion and exclu-
sion criteriathroughout the iterative prompt engineering process. As we refined the
prompts and reviewed the model’s output on different samples, we engaged in discus-
sions. This led to a shift in our comprehension of the sources, and the interpretation
of the inclusion and exclusion criteria inevitably changed. This creates a reinforcing
loop: our developing knowledge informs subsequent prompt iterations, and the results
from those iterations, in turn, further shape our understanding. This reinforced loop,
where the researchers are active participants within the development and evaluation
cycle, introduces a degree of subjectivity that is difficult to mitigate entirely. Our
evolving understanding directly influenced the model’s development, and conversely,
the model’s performance provided feedback that refined our domain knowledge.
Sampling and effects of Distributed Denial of Service (DDoS) Gateways
. A potential threat to the external validity of our findings arises from limitations in
data acquisition due to website access restrictions, such as DDoS protection gateways.
We observed instances where our automated scraping process was blocked, resulting
in the retrieval of PDF documents containing error messages (see ”Count NA” in
Figure 6.1). When processed by our script and the LLM, these PDFs often led to a
’No’ judgment. As the content downloaded in the PDF is not relevant to the research.
However, this classification does not necessarily reflect the actual relevance of the
source, as the LLM tool could not access and process thecontentthat was indexed
by Google and for which the link was initially generated.
28

(a) Screenshot of the PDF as
downloaded by the tool
(b) Screenshot of the Captcha
page when a human access the
URL
(c) Screenshot of source web page
[43] when a human completes the
captcha
Fig. 7Images (a), (b), and (c) show the effects of the Captcha on the Automated script (a), and
the different content that a human can process (b) and (c).
Human reviewers, in contrast, could identify these ’Not Available’ cases. Our anal-
ysis of the 368-source sample revealed that all instances marked as ’Not Available’ by
the researchers corresponded to inaccessible content for the LLM (i.e., true unavail-
ability). However, the converse was not always true; the LLM could not definitively
determine the underlying reason for a ’Not Available’ response.
In our 368-source sample, this issue accounted for approximately 10% of the
reviewed links. Assuming a similar proportion within the full population of 8482 poten-
tial sources, many false negatives (irrelevant classifications due to access issues) could
be present in the LLM’s output. Identifying these true ’Not Available’ cases would
necessitate a manual inspection of the ’No’ classifications, partially undermining the
intended efficiency gains of the automated approach.
Consequently, the acceptance of these potential false negatives represents a trade-
off made to accelerate the overall process of automatically reviewing a large volume of
material. While this limitation may introduce some degree of error, it is a necessary
compromise for substantially reducing the manual effort required for the secondary
study.
Reproducibility of the research and effects of the guiding MVLR proto-
col. Several factors contribute to the rigor and potential for replication of our work.
Firstly, our LLM-based tool has been specifically developed and tuned for our par-
ticular research domain within CASS testing. Secondly, we have made the data and
source code used in our analysis publicly available to enhance transparency and facil-
itate scrutiny. The central principle underpinning the rigor and relevance of our tool
for its intended purpose – supporting our main research aims in CASS testing – is the
strict adherence to our established research protocol in every step automated within
the LLM-based tool. We argue that researchers seeking to replicate our results can
29

readily utilize the provided source code for the tool. However, as discussed in the pre-
ceding section, the prompt requires careful tuning to align with the specific domain’s
inclusion and exclusion criteria, as well as the selected LLM model. Furthermore, the
successful application and validation of such a tool necessitate a foundation of sound
experimental practices, exemplified by the research protocol we employed.
7 Conclusion
This paper presents the development and validation of an LLM-based tool designed to
support researchers in executing an MVLR on the topic of Context-Aware Software
System testing for the avionics industry.
Throughout our engineering process, we prioritized that the tool should not pose
a threat to the execution of the MVLR. At the very least, whatever bias the tool
was intended to introduce should be comparable to the bias introduced by human
researchers. As the task entrusted to this tool was to discard sources that would not
be relevant to the topic of interest of the MVLR, we turned to Agreement statistical
methods to guide the engineering of the tool. We incorporated the notion that even
within cohesive teams, agreement cannot be perfect, and therefore, a degree of varia-
tion must be accounted for. The tool shows high levels of agreement with researchers
of90%.
To further assure this, the engineering process iteratively and systematically sam-
pled the potential sources and incrementally evaluated the tool with the different
samples. This increased our confidence that the tool would perform as intended when
used to advance our scientific interest in context-aware testing. As a result, we pre-
sented a tool capable of processing at least 8,482 sources, and we are confident in
its results, which will inform our continued research on CASS (section 5.3). Thereby
releasing precious human time without introducing bias into the selection process.
In addition, this paper highlights several key observations that are important
when considering the use of the presented tool, or any LLM-based tool, for scientific
research. Specifically, during the prompt engineering phase, we noticed that refining
the prompt to enhance the tool’s reliability created a reinforcing loop. This process, in
turn, reshaped our understanding of our inclusion and exclusion criteria. This created
another feedback loop, as the development of the tool helped us focus on our target
and avoid false positives.
Finally, we have emphasized that the tool’s development relied on a rigor-
ous research protocol. This protocol, which has supported previous MVLR, guided
our decisions and established the operational boundaries for the LLM-based tool.
Researchers looking to reproduce the results of this paper should also establish a
research protocol and ensure a clear understanding of their inclusion and exclusion
criteria. As we firmly believe that it is human researchers who must guide the direc-
tion of the LLM-based assistant, the rigor, relevance, and importance of the results
are strongly grounded in sound research practices.
30

8 Statements and Declarations
•Funding: Prof. Travassos is a Brazilian Research Council (CNPq) Researcher
(grant 305701/2022-3) and a State Scientist FAPERJ – Carlos Chagas Filho
Foundation for Research Support of the State of Rio de Janeiro (grant E-
26/204.310/2024)
•Conflict of interest: The authors declare that they have no conflict of interest.
•Data availability: All data used to write this paper are open-source and available
at: https://git-lab.cos.ufrj.br/contextaware/llm/
•Code availability: All code used to write this paper is open source and available
at: https://git-lab.cos.ufrj.br/contextaware/llm/
•Author contribution: All authors have contributed equally to the design, method-
ology, and writing of this manuscript.
References
[1] Garousi, V., Felderer, M., M¨ antyl¨ a, M.V.: The need for multivocal literature
reviews in software engineering. In: Proceedings of the 20th International Con-
ference on Evaluation and Assessment in Software Engineering - EASE ’16, pp.
1–6. ACM Press, New York, New York, USA (2016). https://doi.org/10.1145/
2915970.2916008
[2] Matalonga, S., Amalfitano, D., Solari, M., Rossa Hauck, J.C., Travassos, G.H.:
Testing context-aware software systems from the voices of the automotive indus-
try. IEEE Transactions on Industrial Informatics21(5), 1551–3203 (2025) https:
//doi.org/10.1109/TII.2025.3529918
[3] Matalonga, S., Rodrigues, F., Travassos, G.H.: Characterizing testing methods
for context-aware software systems: Results from a quasi-systematic literature
review. Journal of Systems and Software131, 1–21 (2017) https://doi.org/10.
1016/j.jss.2017.05.048
[4] Santos, I.d.S., Andrade, R.M.d.C., Rocha, L.S., Matalonga, S., Oliveira, K.M.,
Travassos, G.H.: Test case design for context-aware applications: Are we there
yet? Information and Software Technology88, 1–16 (2017) https://doi.org/10.
1016/j.infsof.2017.03.008
[5] Siqueira, B.R., Ferrari, F.C., Souza, K.E., Camargo, V.V., Lemos, R.: Testing of
adaptive and context-aware systems: approaches and challenges. Software Testing,
Verification and Reliability (2021) https://doi.org/10.1002/stvr.1772
[6] Matalonga, S., Amalfitano, D., Doreste, A., Fasolino, A.R., Travassos, G.H.:
Alternatives for testing of context-aware software systems in non-academic set-
tings: results from a Rapid Review. Information and Software Technology, 106937
(2022) https://doi.org/10.1016/j.infsof.2022.106937
[7] ISO/IEC/IEEE 29119-1:2022: Software and Systems Engineering Software testing
31

Part 1:Concepts and definitions. ISO/IEC/IEEE 29119-1:2013 (2022) https://
doi.org/10.1109/IEEESTD.2022.9698145
[8] Thompson, G., Morgan, P., Samaroo, A., Kurowski, J., Williams, P., Salmon,
M.: Software Testing, 5th edn. BCS, The Chartered Institute for IT, Swindon,
England (2024)
[9] Matalonga, S., Amalfitano, D., Solari, M., Rossa Hauck, J.C., Travassos,
G.H.: Testing Context-Aware Software Systems in the Automotive Domain: A
Multi Vocal Literature Review Protocol and Dataset. Zenodo (2023). https:
//doi.org/10.5281/ZENODO.8346839 . https://zenodo.org/doi/10.5281/zenodo.
8346839 Accessed 2023-10-26
[10] Waseem, M., Ahmady, A., Liang, P., Fehmi, M., Abrahamsson, P., Mikko-
nen, T.: Conducting systematic literature reviews with chatgpt. In: Proceedings
of the 17th International Symposium on Empirical Software Engineering and
Measurement (ESEM ’23). ACM, ??? (2023). https://doi.org/10.1145/3611322.
3611318
[11] Alshami, A., Elsayed, M., Ali, E., Eltoukhy, A.E., Zayed, T.: Harnessing the
power of chatgpt for automating systematic review process: Methodology, case
study, limitations, and future directions. Systems11(7), 1–7 (2023)
[12] Wang, S., Scells, H., Zuccon, G.: Can chatgpt write a good boolean query for
systematic review literature search? In: Proceedings of the 46th International
ACM SIGIR Conference on Research and Development in Information Retrieval,
pp. 1426–1436 (2023). https://doi.org/10.1145/3539618.3591730
[13] Guo, E., Gupta, M., Deng, J., Park, Y.-J., Paget, M., Naugler, C.: Automated
paper screening for clinical reviews using large language models: Data analysis
study. J Med Internet Res26, 48996 (2024) https://doi.org/10.2196/48996
[14] Khraisha, Q., Put, S., Kappenberg, J., Warraitch, A., Hadfield, K.: Can large
language models replace humans in systematic reviews? evaluating gpt-4’s effi-
cacy in screening and extracting data from peer-reviewed and grey literature
in multiple languages. Research Synthesis Methods15(4), 1–11 (2024) https:
//doi.org/10.1002/jrsm.1715
[15] Robinson, K.A.,et al.: Are chatgpt and large language models “the answer” to
bringing us closer to systematic review automation? Systematic Reviews12(1),
72 (2023) https://doi.org/10.1186/s13643-023-02243-z
[16] Wilkins, J.,et al.: A systematic review of chatgpt and other conversational large
language models in healthcare. Journal of Medical Internet Research26, 22769
(2024) https://doi.org/10.2196/22769
[17] Gupta, B., Mufti, T., Sohail, S.S., Madsen, D.: Chatgpt: A brief narrative review.
32

Cogent Business & Management10(1), 2275851 (2023) https://doi.org/10.1080/
23311975.2023.2275851
[18] Alchokr, R., Borkar, M., Thotadarya, S., Saake, G., Leich, T.: Supporting system-
atic literature reviews using deep-learning-based language models. In: Proceedings
of the 1st International Workshop on Natural Language-Based Software Engi-
neering. NLBSE ’22, pp. 67–74. Association for Computing Machinery, New York,
NY, USA (2023). https://doi.org/10.1145/3528588.3528658
[19] Watanabe, W.M., Felizardo, K.R., Candido, A., Ferreira de Souza, Campos Neto,
J.E., Vijaykumar, N.L.: Reducing efforts of software engineering systematic
literature reviews updates using text classification. Information and Software
Technology128, 106395 (2020) https://doi.org/10.1016/j.infsof.2020.106395
[20] Girardi, D., Minku, L., Cavalcanti, A., Ferrara, P.: Deep learning for software
defect prediction: A systematic literature review. Journal of Systems and Software
199, 111561 (2023) https://doi.org/10.1016/j.jss.2023.111561
[21] Necula, M., Petcu, D., Stefan, A.: Natural language processing in requirements
engineering: A systematic mapping study. Electronics13(11), 2055 (2024) https:
//doi.org/10.3390/electronics13112055
[22] Al-Shalif, A., Aljarah, I., Mirjalili, S.: A comprehensive review of metaheuristic-
based feature selection for text classification. PeerJ Computer Science10, 2084
(2024) https://doi.org/10.7717/peerj-cs.2084
[23] Anghelescu, A., Gheorghe, G., Suciu, B.A., Suciu, G.: Chatgpt utility in health-
care education, research, and practice: Systematic review on the promising
perspectives and valid concerns. Balneo and PRM Research Journal14(4), 1–9
(2023) https://doi.org/10.12680/balneo.2023.614
[24] Mahuli, S.A., Rai, A., Mahuli, A.V., Kumar, A.: Application chatgpt in conduct-
ing systematic reviews and meta-analyses. British Dental Journal235(2), 90–92
(2023) https://doi.org/10.1038/s41415-023-6132-y
[25] Najafali, D., Camacho, J.M., Reiche, E., Galbraith, L.G., Morrison, S.D., Doraf-
shar, A.H.: Truth or lies? the pitfalls and limitations of chatgpt in systematic
review creation. Aesthetic Surgery Journal43(8), 654–655 (2023) https://doi.
org/10.1093/asj/sjad093
[26] Qureshi, R., Shaughnessy, D., Gill, K.A.R., Robinson, K.A., Li, T., Agai, E.:
Are chatgpt and large language models “the answer” to bringing us closer to
systematic review automation? Systematic Reviews12(1), 72 (2023) https://doi.
org/10.1186/s13643-023-02243-z
[27] Huotala, A., Kuutila, M., Ralph, P., M¨ antyl¨ a, M.: The promise and challenges
33

of using llms to accelerate the screening process of systematic reviews. In: Pro-
ceedings of the 28th International Conference on Evaluation and Assessment
in Software Engineering. EASE ’24, pp. 262–271. Association for Comput-
ing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3661167.
3661172
[28] Felizardo, K.R., Lima, M.S., Deizepe, A., Conte, T.U., Steinmacher, I.: Chatgpt
application in systematic literature reviews in software engineering: an evalua-
tion of its accuracy to support the selection activity. In: Proceedings of the 18th
ACM / IEEE International Symposium on Empirical Software Engineering and
Measurement (ESEM ’24) (2024). https://doi.org/10.1145/3674805.3686666
[29] Dang, H., Mecke, L., Lehmann, F., Goller, S., Buschek, D.: How to Prompt?
Opportunities and Challenges of Zero- and Few-Shot Learning for Human-AI
Interaction in Creative Applications of Generative Models (2022). https://arxiv.
org/abs/2209.01390
[30] Sahoo, P., Singh, A.K., Saha, S., Jain, V., Mondal, S., Chadha, A.: A System-
atic Survey of Prompt Engineering in Large Language Models: Techniques and
Applications (2025). https://arxiv.org/abs/2402.07927
[31] Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., Neubig, G.: Pre-train, prompt,
and predict: A systematic survey of prompting methods in natural language
processing55(9) (2023) https://doi.org/10.1145/3560815
[32] Marvin, G., Hellen, N., Jjingo, D., Nakatumba-Nabende, J.: Prompt engineering
in large language models. In: Jacob, I.J., Piramuthu, S., Falkowski-Gilski, P. (eds.)
Data Intelligence and Cognitive Informatics, pp. 387–402. Springer, Singapore
(2024). https://doi.org/10.1007/978-981-99-7962-2 30
[33] Xu, B., Yang, A., Lin, J., Wang, Q., Zhou, C., Zhang, Y., Mao, Z.: ExpertPrompt-
ing: Instructing Large Language Models to be Distinguished Experts (2025).
https://arxiv.org/abs/2305.14688
[34] Chen, B., Zhang, Z., Langren´ e, N., Zhu, S.: Unleashing the potential of prompt
engineering in large language models: a comprehensive review. arXiv preprint
arXiv:2310.14735 (2023) https://doi.org/10.48550/arXiv.2310.14735
[35] Son, M., Won, Y.-J., Lee, S.: Optimizing large language models: A deep dive into
effective prompt engineering techniques. Applied Sciences15(3) (2025) https:
//doi.org/10.3390/app15031430
[36] Ekin, S.: Prompt Engineering For ChatGPT: A Quick Guide To Techniques,
Tips, And Best Practices. Institute of Electrical and Electronics Engineers
(IEEE) (2023). https://doi.org/10.36227/techrxiv.22683919.v1 . http://dx.doi.
org/10.36227/techrxiv.22683919.v1
34

[37] Claris´ o, R., Cabot, J.: Model-driven prompt engineering. In: 2023 ACM/IEEE
26th International Conference on Model Driven Engineering Languages and Sys-
tems (MODELS), pp. 47–54 (2023). https://doi.org/10.1109/MODELS58315.
2023.00020
[38] Tonmoy, S., Zaman, S., Jain, V., Rani, A., Rawte, V., Chadha, A., Das, A.: A
comprehensive survey of hallucination mitigation techniques in large language
models. arXiv preprint arXiv:2401.013136(2024)
[39] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H.,
Lewis, M., Yih, W.-t., Rockt¨ aschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Proceedings of the 34th Interna-
tional Conference on Neural Information Processing Systems. NIPS ’20. Curran
Associates Inc., Red Hook, NY, USA (2020)
[40] Cohen, J.: A coefficient of agreement for nominal scales. Educational
and Psychological Measurement20(1), 37–46 (1960) https://doi.org/10.1177/
001316446002000104
[41] Fleiss, J.L.: Measuring nominal scale agreement among many raters. Psychologi-
cal Bulletin76(5), 378–382 (1971) https://doi.org/10.1037/h0031619
[42] Fletcher, R.H., Fletcher, S.W.: Clinical Epidemiology, 4th edn. Lippincott
Williams and Wilkins, ISBN 0-7817-5215-9. Philadelphia, PA (2005)
[43] Dubinin, D.V., Kochegurov, A.I., Geringer, V.E.: Improving the criteria for
quality assessment of image processing algorithms. Journal of Physics: Confer-
ence Series1862(1), 012011 (2021) https://doi.org/10.1088/1742-6596/1862/1/
012011
35
# Maat: The Agentic Legal Research Assistant for Competition Protection

**Authors**: Basant Mounir, Farida Madkour, Amira Abdelaziz, Asmaa Sami

**Published**: 2026-05-26 17:38:26

**PDF URL**: [https://arxiv.org/pdf/2605.27331v1](https://arxiv.org/pdf/2605.27331v1)

## Abstract
Competition law experts conducting legal research must review extensive volumes of cases, decisions, and judicial reports to identify precedents and assess key elements in competition and merger cases. Although general research assistants such as Claude and ChatGPT and legal assistants such as SaulLM-7B and LegalGPT are increasingly used to assist legal research, they remain inadequate for competition law analysis: they lack specialized domain expertise, provide insufficient official citations, or hallucinate competition law cases. We propose Maat, a ReAct agent that orchestrates tools corresponding to different tasks of the research process. Designed iteratively with competition law experts, Maat grounds cases and findings in official sources using RAG for reliability, provides rich in-line citations, falls back to web search when database coverage is insufficient, and prompts the user for clarification when queries are ambiguous. Maat significantly outperforms all baseline assistants on case-specific tasks and performs within range of the top baseline on theoretical question tasks. The dataset used is available on GitHub.

## Full Text


<!-- PDF content starts -->

Maat: The Agentic Legal Research Assistant for Competition
Protection
Basant Mounir∗
basant.mounir@gmail.com
Cairo, EgyptFarida Madkour∗
faridamadkour@aucegypt.edu
Cairo, Egypt
Amira Abdelaziz∗
amira@mathematik.uni-marburg.de
Cairo, EgyptAsmaa Sami∗
asmaasami2011@gmail.com
Cairo, Egypt
Abstract
Competition law experts conducting legal research must review
extensive volumes of cases, decisions, and judicial reports to identify
precedents and assess key elements in competition and merger
cases. Although general research assistants such as Claude and
ChatGPT and legal assistants such as SaulLM-7B and LegalGPT are
increasingly used to assist legal research, they remain inadequate
for competition law analysis: they lack specialized domain expertise,
provide insufficient official citations, or hallucinate competition
law cases. We propose Maat, a ReAct agent that orchestrates tools
corresponding to different tasks of the research process. Designed
iteratively with competition law experts, Maat grounds cases and
findings in official sources using RAG for reliability, provides rich
in-line citations, falls back to web search when database coverage is
insufficient, and prompts the user for clarification when queries are
ambiguous. Maat significantly outperforms all baseline assistants on
case-specific tasks and performs within range of the top baseline on
theoretical question tasks. The dataset used is available on GitHub1.
CCS Concepts
•Applied computing →Law;•Computing methodologies →
Agentic systems;•Information systems→Question answering.
Keywords
Competition Protection, Legal, Cases, Search, Database, Violation,
Retrieval-Augmented Generation, Agentic, AI Assistant
1 Introduction
Competition protection promotes fair competition for a healthy
economic environment [ 29]. Experts monitor markets, receive com-
plaints, and initiate proceedings, requiring them to research cases
across jurisdictions [ 11,13], analyze hundreds of pages of case
documents, and expand on competition law concepts [ 26] using
sources such as the EU Competition Commission, Concurrences
[6], and OECD reports [ 29]. Due to the time-consuming nature
of this process, experts increasingly turn to AI assistants for rea-
soning, summarization, and question-answering [ 7,25,37]. These
assistants include general research assistants such as ChatGPT [ 28]
∗This research paper has been prepared by its authors and does not represent the
views, opinions, or official position of the Central Bank of Egypt, as their employees.
The contents are provided for informational purposes only and shall not create, give
rise to, or be construed as creating any legal obligation, liability, or responsibility on
the part of the Central Bank of Egypt and it assumes no responsibility or liability for
the content, conclusions, analyses, or recommendations contained in this paper.
1https://github.com/baahmed/maat-datasetand Claude Sonnet [ 1], and multilingual legal assistants such as
SaulLM-7B [ 5] and LegalGPT [ 19]. However, general research assis-
tants are inadequate for competition law research: they hallucinate
cases and case information [ 9,14], possess shallow domain knowl-
edge [ 2,9], and lack page-level in-line citations for verification [ 12].
Legal assistants provide responses that are not grounded in official
sources, omit granular citations, and perform poorly on questions
requiring complex reasoning about legal principles [32, 34].
The literature has explored more specialized systems. Much
of it focuses on case retrieval or question answering in isolation
[17,20,24,35,41], while systems that integrate multiple tasks
are deficient in fallback mechanisms and multi-turn clarification
[36,39]. Crucially, none of the reviewed systems is grounded in
a dedicated competition law case database, leaving the domain’s
cross-jurisdictional retrieval and source-level verification require-
ments unaddressed. To our knowledge, no prior academic work
has developed and openly evaluated a purpose-built AI research
assistant grounded in an official, dedicated competition law case
database with cross-jurisdictional retrieval, structured tool routing,
and expert evaluation.
To address these gaps, we introduce Maat, a multi-turn agentic
system for competition law research, named after the ancient Egyp-
tian goddess of law, order, and justice [ 38]. Designed iteratively
with competition law experts, Maat has a structured tool-routing
architecture where a ReAct-prompted agent [ 40] dynamically se-
lects among specialized tools based on query type and database
coverage. Our contributions are as follows:
•a ReAct agentic system with a structured tool-routing archi-
tecture with explicit fallback logic when database coverage is
insufficient and human-in-the-loop to clarify vague queries,
•an interface to interact with the agentic research system
conversationally, get page-level in-line citations from offi-
cial sources, and inspect which tools are executed at each
reasoning step,
•a competition cases dataset consolidating EU regional cases
and German national cases, enriched by metadata extracted
by a Large Language Model (LLM) to power semantic case
search,
•a blind comparative expert evaluation demonstrating that
Maat significantly outperforms all assistants under consider-
ation in case-specific questions and performs within range
of the top baseline in theoretical questions.
1arXiv:2605.27331v1  [cs.AI]  26 May 2026

, , Basant Mounir, Farida Madkour, Amira Abdelaziz, and Asmaa Sami
2 Data
The dataset is constructed from two sources: the EU Competition
Commission and the Bundeskartellamt, chosen for their jurisdic-
tional and financial market relevance to Egypt.
2.1 European Union Regional Cases
For regional cases across the EU, there are two separate API end-
points for antitrust cases and merger cases [ 8]. Relevant metadata
fields are selected, including case ID, description, violation, compa-
nies, sector, decision text link, date of release, and language of the
text.
The dataset was then cleaned. Cases with empty links to the de-
cision text were removed due to inaccessibility to the RAG pipeline.
To normalize the violations, older articles were mapped to their
newer counterparts. For example, for cartel cases, Article 85 EEC
was mapped to Article 101 TFEU. The sectors of four cases were
missing, which were filled manually. A comma-separated list of
companies was extracted per case to be readable in Python. Only
cases in English were kept. Other columns have passed data quality
checks for missing values and normalized formats.
2.2 German National Cases
For the Bundeskartellamt cases listed in [ 4], there was no direct
API or dataset source; the dataset had to be created. The strategy
was to compile decision text links of Bundeskartellamt cases and
extract metadata from the link structure and the decision texts
using an LLM. The links of the case decision texts were studied
for patterns. Cases from the Bundeskartellamt start with the prefix
bundeskartellamt.de/SharedDocs/Entscheidung/ and were retrieved
using the Wayback Machine API [15].
The same aforementioned case metadata was extracted. From
the link structure, the case ID, violation, decision release year, and
language could be deduced. The remainder of the fields, specifically
the case title, the sector, and the list of companies involved, were
extracted by prompting the OpenAI gpt-4o-mini model via few-
shot examples to guide the model. Since Germany is part of the
EU, it also builds its economic classification system for market
definition based on NACE [ 3]. To verify LLM extraction quality,
a stratified random sample of 34 cases was manually reviewed;
extraction accuracy was 85% for case titles, 97% for sectors, and
84% for company lists.
2.3 Combined Case Dataset
The final combined datasets of both the regional EU cases and
the national German cases consist of 1609 cases. It contains the
following fields:
•case_id: the unique identifier of the case,
•case_title: a description of the case,
•jurisdiction : the covered geography of the case (either
EU or Germany),
•violation: the legal basis under consideration in the case,
•sector : the code and name of the NACE section that includes
the sector in which the violation was committed,
•companies: a list of companies involved in the case,
•pdf_url: the link to the case decision text,•language : the language in which the case decision text is
written,
•decision_date : the date on which the case decision text
was released.
2.3.1 Case Decision Documents Ingestion Pipeline.To prepare the
case decision documents for querying in the RAG pipeline, the
cases dataset must be indexed appropriately. LlamaIndex [22] is the
framework used to process the documents for Maat. The PDFReader
loads a decision document from its respective link into a Document
object. The SentenceSplitter chunks the document into sets of
Nodes , each consisting of approximately 1024 tokens, with 20 over-
lap tokens [ 23]. Each Node contains document-related metadata,
such as the link of the document and the page from which the Node
comes. This document-level metadata is crucial for in-line citations.
The chunks are embedded using the OpenAI
text-embedding-ada-002 model [ 21] and stored in the Qdrant
vector database.
3 Architecture
3.1 Agent Design
At the heart of Maat is a ReAct agent that mimics the expert research
workflow through a thought-action-observation loop [ 40]. The
system prompt of Maat, developed by meta-prompting, describes
the loop structure, the tools available, the routing rules, and the
constraints.
3.1.1 Tools.The following tools were defined based on the ob-
served expert research workflow: database_search ,web_search ,
answer_case,answer_theoretical, andask_clarification.
These tools can be chained together to completely answer the
user’s question. For example, if the user asks"What was the market
definition in the case AT.39398?"and the user has never referenced
that case before, a database_search operation is executed first. If
the case was not found in the database, then this tool fails over to
theweb_search tool. If no such case was found on the web, the
agent concludes that no such case exists. However, if the case was
found, then it is fetched into the memory layer of the agent. Next,
the tool answer_case is called to answer the question asked by the
user about that case. Finally, the answer is returned. At every turn,
the agent reasons about user intent, selects a tool, and observes the
result.
3.1.2 Memory Layer.The agent uses both in-session and external
memory to infuse the appropriate context in its prompts. In in-
session memory, the agent records its ReAct loops in the scratchpad
to keep track of its line of thought when making new decisions.
The chat history is also stored to help the agent understand its
interactions with the users. Finally, the cases retrieved so far along
with their metadata are stored in session_cases .session_cases
are crucial to answer questions such as"What was the violation in
the first case?"; the first case recorded is fetched and the question
about it is answered using RAG [ 18] as explained in answer_case .
The main external memory component used is the vector database
which holds most of the EU regional and German national cases.
2

Maat: The Agentic Legal Research Assistant for Competition Protection , ,
Figure 1: Maat System Architecture. The Maat Goddess image is obtained from [10].
3.2 Tools
3.2.1 Database Search.If a search intention is detected in the query,
the agent first tries to find matching cases in the database. The query
itself is translated into a query vector composed of six dimensions
that are a subset of the fields in the database: case_id ,case_title ,
jurisdiction,violation,sector, andcompanies.
The extraction of those dimensions from the query is performed
by prompting the OpenAI gpt-4o-mini model, which is a rec-
ommended efficient model in terms of both cost and latency for
structured data extraction [ 27]. The prompt includes few-shot ex-
amples per field; schema-constrained dimensions ( jurisdiction ,
violation,sector) are validated against predefined values.
Since all six dimensions of the query vector are columns in the
cases database, the query vector is used as a filter to match relevant
cases to the query itself. The filtering of case ID is straightforward,
since matches should have the exact specified ID in the query vector
as the case. Exact matches are also found for the jurisdiction, viola-
tion, and sector fields, whose values come from a predefined schema.
Companies are matched by checking if the companies list from the
query vector is a subset of the companies list of the case under con-
sideration. Lastly, case titles are matched using cosine similarities.
Case titles of both the query vector and the case under consider-
ation are embedded using the OpenAI text-embedding-3-small
model, whose family has been used by other AI legal systems in
[16] and [ 31]. If the cosine similarity of the embeddings exceeds
the threshold of 0.85, the titles are considered a match. The thresh-
old was tuned by testing on the case question dataset described
in the evaluation section. If there are more than five cases in the
result set, the cosine similarity of the query embedding and the case
embeddings are computed, and only the top five cases are kept.
3.2.2 Web Search.Sometimes, cases are not captured because they
have been released before the database is refreshed or because thereare no snapshots on the WayBack Machine yet [ 15]. In this case,
the agent searches the web for fresh cases.
As with database_search , the user question first gets translated
to the query vector comprising the same six dimensions, which will
be used to filter web results. To perform the search, Perplexity Sonar
[30] is prompted to answer the user question by providing up to
five recent cases using its online search feature. Because assistants
may hallucinate [ 14] and unofficial sources used by assistants may
misinterpret cases, the web is searched for each title in the result
set retrieved by Perplexity to find matching official sources from
the European Competition Commission or the Bundeskartellamt,
depending on the jurisdiction of the query. The Serper Google
Search API [ 33] is used to search the web to gather official links for
each of the retrieved titles. Titles with no official sources found are
assumed hallucinated and are removed from the result set. To ensure
the remaining case titles are indeed relevant to the initial query,
the descriptions of the official sources for each title are translated
into a six-dimensional query vector for post-search filtering. The
query vector of each case title is checked against the query vector
of the user question. If there is a mismatch, the corresponding case
title is removed from the result set. In the end, the titles remaining
are of existing cases whose metadata is a match for the original
user query.
3.2.3 Answer Case-Specific Questions.When a user asks a question
about a specific case, the agent first checks if the case in question ex-
ists in the session_cases list. If not, a search operation is invoked
before continuing to retrieve the case information. The chunks of
the case in question are loaded in active_case from the database.
Then, the RAG [ 18] pipeline is invoked. The case chunks are ranked
according to the cosine similarities of each chunk and the embed-
ded user question. Based on experimentation using the question
bank described in the next section, the number of top chunks to
be retrieved was set to8. The texts, pages, and links of the top
8chunks are injected into a meta-prompted question-answering
3

, , Basant Mounir, Farida Madkour, Amira Abdelaziz, and Asmaa Sami
Table 1: Average Expert Answer Ratings (out of 5)
General Research LLMs Legal LLMs
Task Category Maat Claude Sonnet 4.6 GPT-5.5 LegalGPT SaulLM-7B
Theoretical 2.9 3.5 2.4 2.2 1.1
Case-specific 4.6 2.4 2.0 2.1 0.7
prompt alongside the question and chat history to generate the
final answer with in-line citations.
3.2.4 Answer Theoretical Questions.When a user asks a general
question about the theory of competition law, such as"What are the
forms of abuse of dominance?", answer_theoretical is invoked. A
prompt was engineered iteratively with competition law experts
to construct the knowledge base around key concepts and termi-
nologies of competition law. Perplexity Sonar was the base LLM
of choice due to its deep research capabilities [ 30]. The question is
injected into this prompt to guide Perplexity’s reasoning depth and
direction. Furthermore, the domain of search was restricted to a set
of official and reliable sources only provided by the experts. At the
end of the response, based on the chat history, Maat also recom-
mends questions for deeper research in its role as a brainstorming
assistant.
3.2.5 Ask for Clarification.If the question is unclear, the scratch-
pad, chat history, and question are analyzed to assess the infor-
mation gap that needs to be addressed before the agent answers
the question. For example, if the user asks"What was the market
definition of the case?"and there are multiple retrieved cases by a
search operation, then the agent will first ask the user to clarify
which case is the question about. The human-in-the-loop then
provides enough clarity for the agent to answer the question.
3.3 User Interface
Maat is a web application accessible via standard browsers with-
out additional installation. The Flask library is used to design an
interactive user experience with a dedicated user input box and
submit button to enter questions. As the user question is passed to
the agent in the backend, the user can see a progress indicator,"The
assistant is thinking..."that adjusts to show which tools the agent
calls during the question-answering process. The user can track
the conversation in the scrollable chat pane and explore in-line
citations for each part of the answer, allowing for a smooth answer
verification experience.
4 Evaluation
A blind comparative evaluation was conducted in which the per-
formance of Maat was compared against state-of-the-art general
research assistants, ChatGPT (GPT-5.5) [ 28] and Claude Sonnet 4.6
[1], and specialized multilingual legal research assistants, SaulLM-
7B [5] and LegalGPT [ 19]. The experts prepared two question banks:
fifty theoretical questions (e.g.,"Explain price-fixing") and twenty
case-specific questions (e.g.,"List abuse of dominance cases in the
financial sector"). Seven questions from the theoretical test bank
and eight questions from the case-specific test bank formed the testset; the remainder were used for prompt engineering and hyperpa-
rameter tuning. Responses were scored from 1 (poor) to 5 (excellent)
based on five criteria, each worth one point: coherence of legal logic,
usage of correct legal taxonomy, illustration of concepts, quality
of references, and appropriate inclusion of in-line citations. Nine
competition law experts contributed to the evaluation process.
4.1 Results
Table 1 reports average expert ratings per task type. Maat sub-
stantially outperformed all baselines on case-specific questions
and performed within range of Claude Sonnet 4.6, the strongest
baseline, on theoretical questions; a Friedman test confirmed these
differences are statistically significant ( 𝑝<0.05 ). Maat and Claude
Sonnet 4.6 both demonstrated structured legal reasoning and il-
lustrative examples, though both lacked citation granularity and
source diversity, with Claude Sonnet 4.6 performing marginally
better on these criteria. On case-specific questions, legal-domain
assistants performed poorly, with experts observing hallucinated
cases, unofficial sources, missing in-line citations, and weak com-
petition law reasoning. This reflects both slower research cycles in
legal AI and the lack of competition law specialization of these mod-
els. General research assistants retrieved more relevant cases but
shared the same citation and reasoning limitations. Maat addresses
these shortcomings by design: it retrieves case documents directly
from official sources, generates answers via RAG with page-level
in-line citations, and applies structured prompt engineering for
principled competition law reasoning.
5 Conclusion
We introduced Maat, a ReAct agent for competition law research
that integrates database retrieval, web fallback, case and theo-
retical question-answering, and human-in-the-loop clarification.
Grounded in Bundeskartellamt and EU Commission cases with
in-line citations for verifiability, Maat performed within range of
the top baseline in theoretical question tasks and outperformed
all baselines across case-specific tasks. Future work will extend
jurisdictional coverage and enhance theoretical source diversity
and citation granularity.
Acknowledgments
The authors thank the EU Competition Commission and the Bun-
deskartellamt for making the cases publicly accessible, and Ahmed
Abdelaal, Abdelrahman Abdelhalim, Ahmed Kamel, and Waad
Hegazy for paper review and involvement throughout the develop-
ment of Maat.
4

Maat: The Agentic Legal Research Assistant for Competition Protection , ,
GenAI Usage Disclosure
The authors utilized Claude and ChatGPT to debug code written
for Maat. Furthermore, the authors used the aforementioned GenAI
tools for grammar checking, minor paper structuring, and draft
revisions. All GenAI responses were checked and edited for accu-
racy and appropriateness. GenAI was not used to contribute to the
intellectual content. The authors claim full responsibility for all
contributions and results of this publication.
References
[1]Anthropic. 2026.Claude Sonnet 4.6 System Card. Technical Report. https:
//www.anthropic.com/claude-sonnet-4-6-system-card
[2]Farid Ariai, Joel Mackenzie, and Gianluca Demartini. 2025. Natural language pro-
cessing for the legal domain: A survey of tasks, datasets, models, and challenges.
Comput. Surveys58, 6 (2025), 1–37.
[3]Felix Beuter, Johannes Gussenbauer, Elias Minther, Viktoria Szabo, and Susanne
Wegner. 2025.Approaches to Automated NACE Coding of German Business Activity
Descriptions. Springer Nature Switzerland, Cham, 179–211. doi:10.1007/978-3-
032-10004-7_10
[4]Bundeskartellamt. 2024. Entscheidungen [Decisions]. https://www.
bundeskartellamt.de/SharedDocs/Entscheidung/. Official decision database
of the German Federal Cartel Office, published pursuant to § 5 UrhG..
[5]Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo,
Caio Corro, Andre FT Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Mor-
gado, et al .2024. Saullm-7b: A pioneering large language model for law.arXiv
preprint arXiv:2403.03883(2024).
[6]Concurrences. [n. d.]. Concurrences: Competition Law Review. https://www.
concurrences.com/en/
[7]Aniket Deroy, Kripabandhu Ghosh, and Saptarshi Ghosh. 2024. Applicability of
large language models and generative models for legal case judgement summa-
rization.arXiv preprint arXiv:2407.12848(2024).
[8]Directorate-General for Competition, European Commission. 2026. EU Com-
petition Case Search. https://competition-cases.ec.europa.eu/search. Offi-
cial European Commission database for antitrust, cartel, merger, and state aid
cases distributed in JSON format. License: European Commission Reuse Notice
(Dec. 2011/833/OJ)..
[9]Rajaa El Hamdani, Thomas Bonald, Fragkiskos D Malliaros, Nils Holzenberger,
and Fabian Suchanek. 2024. The factuality of large language models in the legal
domain. InProceedings of the 33rd ACM International Conference on Information
and Knowledge Management. 3741–3746.
[10] Eternal Space. [n. d.]. Maat (Goddess). https://commons.wikimedia.org/wiki/File:
Maat_(Goddess).png Licensed under CC BY-SA 4.0.
[11] European Commission. [n. d.]. Antitrust and Cartels: Procedures. https://
competition-policy.ec.europa.eu/antitrust-and-cartels/procedures_en
[12] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large lan-
guage models to generate text with citations. InProceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing. 6465–6488.
[13] Douglas H. Ginsburg and Tim Eicke (Eds.). 2023.Judicial Review of Competition
Cases. Concurrences. Multi-jurisdictional comparative study.
[14] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A
survey on hallucination in large language models: Principles, taxonomy, chal-
lenges, and open questions.ACM Transactions on Information Systems43, 2 (2025),
1–55.
[15] Internet Archive. [n. d.]. Wayback Machine APIs. https://archive.org/help/
wayback_api.php. Documents the Wayback Availability JSON API and CDX
Server API..
[16] Figarri Keisha, Prince Singh, Pallavi, Dion Fernandes, Aravindh Manivannan,
Ilham Wicaksono, Faisal Ahmad, and Wiem Ben Rim. 2025. All for law and
law for all: Adaptive RAG Pipeline for Legal Research. arXiv:2508.13107 [cs.CL]
https://arxiv.org/abs/2508.13107
[17] Radhika V Kulkarni, Avish Agrawal, Aryan Vimal, Rohan Barde, Raghav Bajaj,
and Khursheed Gaddi. 2025. Legal Case Search: An AI-Powered Legal Search
Engine. InInternational Conference on ICT for Sustainable Development. Springer,
354–363.
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[19] LexiAI. [n. d.]. Legal GPT. https://chatgpt.com/g/g-jxqQ0lepc-legal-gpt.
[20] Bulou Liu, Yiran Hu, Qingyao Ai, Yiqun Liu, Yueyue Wu, Chenliang Li, and
Weixing Shen. 2023. Leveraging event schema to ask clarifying questions for
conversational legal case retrieval. InProceedings of the 32nd ACM international
conference on information and knowledge management. 1513–1522.[21] LlamaIndex. 2025. Embeddings. https://developers.llamaindex.ai/python/
framework/module_guides/models/embeddings/ LlamaIndex Developer Docu-
mentation..
[22] LlamaIndex. 2025. Introduction to RAG. https://developers.llamaindex.ai/python/
framework/understanding/rag/ LlamaIndex Developer Documentation..
[23] LlamaIndex. 2025. Loading Data (Ingestion). https://developers.llamaindex.ai/
python/framework/understanding/rag/loading/ LlamaIndex Developer Docu-
mentation..
[24] Daniel Locke and Guido Zuccon. 2022. Case law retrieval: problems, methods,
challenges and evaluations in the last 20 years.arXiv preprint arXiv:2202.07209
(2022).
[25] Antoine Louis, Gijs Van Dijck, and Gerasimos Spanakis. 2024. Interpretable
long-form legal question answering with retrieval-augmented large language
models. InProceedings of the AAAI conference on artificial intelligence, Vol. 38.
22266–22275.
[26] OECD. 2024.The Standard and Burden of Proof in Competition Law Cases. Techni-
cal Report. OECD Competition Committee. https://doi.org/10.1787/0199f63f-en
[27] OpenAI. 2024. GPT-4o mini: Advancing Cost-Efficient Intelligence. https://
openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/.
[28] OpenAI. 2026.GPT-5.5 System Card. Technical Report. OpenAI. https://openai.
com/index/gpt-5-5-system-card/ Accessed: May 2026.
[29] Organisation for Economic Co-operation and Development. [n. d.]. OECD. https:
//www.oecd.org/en.html
[30] Perplexity AI. 2025. Meet New Sonar. Perplexity Blog. https://www.perplexity.
ai/hub/blog/meet-new-sonar
[31] Nicholas Pipitone and Ghita Houir Alami. 2024. LegalBench-RAG: A
Benchmark for Retrieval-Augmented Generation in the Legal Domain.
arXiv:2408.10343 [cs.AI] https://arxiv.org/abs/2408.10343
[32] Albert Sadowski and Jaroslaw A Chudziak. 2025. On verifiable legal reasoning:
A multi-agent framework with formalized knowledge representations. InPro-
ceedings of the 34th ACM International Conference on Information and Knowledge
Management. 2535–2545.
[33] Serper. 2025. Serper: Google Search API. https://serper.dev/
[34] Utkarsh Ujwal, Sai Sri Harsha Surampudi, Sayantan Mitra, and Tulika Saha. 2024.
" Reasoning before Responding": Towards Legal Long-form Question Answering
with Interpretability. InProceedings of the 33rd ACM International Conference on
Information and Knowledge Management. 4922–4930.
[35] Rahman SM Wahidur, Sumin Kim, Haeung Choi, David S Bhatti, and Heung-No
Lee. 2025. Legal query rag.IEEE Access(2025).
[36] Ziqi Wang and Boqin Yuan. 2025. L-MARS: Legal multi-agent workflow with
orchestrated reasoning and agentic search.arXiv preprint arXiv:2509.00761(2025).
[37] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models.Advances in neural information processing systems35
(2022), 24824–24837.
[38] Wikipedia contributors. [n. d.]. Maat. https://en.wikipedia.org/wiki/Maat
[39] Rujing Yao, Yang Wu, Chenghao Wang, Jingwei Xiong, Fang Wang, and Xi-
aozhong Liu. 2025. Elevating legal LLM responses: harnessing trainable logical
structures and semantic knowledge with legal reasoning. InProceedings of the
2025 Conference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers).
5630–5642.
[40] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.
arXiv preprint arXiv:2210.03629(2022).
[41] Ruizhe Zhang, Qingyao Ai, Yueyue Wu, Yixiao Ma, and Yiqun Liu. 2023. Diverse
legal case search.arXiv preprint arXiv:2301.12504(2023).
5
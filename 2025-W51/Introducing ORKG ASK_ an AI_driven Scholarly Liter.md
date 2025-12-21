# Introducing ORKG ASK: an AI-driven Scholarly Literature Search and Exploration System Taking a Neuro-Symbolic Approach

**Authors**: Allard Oelen, Mohamad Yaser Jaradeh, Sören Auer

**Published**: 2025-12-18 11:25:14

**PDF URL**: [https://arxiv.org/pdf/2512.16425v1](https://arxiv.org/pdf/2512.16425v1)

## Abstract
As the volume of published scholarly literature continues to grow, finding relevant literature becomes increasingly difficult. With the rise of generative Artificial Intelligence (AI), and particularly Large Language Models (LLMs), new possibilities emerge to find and explore literature. We introduce ASK (Assistant for Scientific Knowledge), an AI-driven scholarly literature search and exploration system that follows a neuro-symbolic approach. ASK aims to provide active support to researchers in finding relevant scholarly literature by leveraging vector search, LLMs, and knowledge graphs. The system allows users to input research questions in natural language and retrieve relevant articles. ASK automatically extracts key information and generates answers to research questions using a Retrieval-Augmented Generation (RAG) approach. We present an evaluation of ASK, assessing the system's usability and usefulness. Findings indicate that the system is user-friendly and users are generally satisfied while using the system.

## Full Text


<!-- PDF content starts -->

Introducing ORKG ASK: an AI-driven Scholarly
Literature Search and Exploration System Taking
a Neuro-Symbolic Approach
Allard Oelen1[0000−0001−9924−9153], Mohamad Yaser
Jaradeh2[0000−0001−8777−2780], and Sören Auer1,2[0000−0002−0698−2864]
1TIB – Leibniz Information Centre for Science and Technology, Hannover, Germany
{allard.oelen,auer}@tib.eu
2L3S Research Center, Leibniz University of Hannover, Hannover, Germany
jaradeh@l3s.de
Abstract.As the volume of published scholarly literature continues
to grow, finding relevant literature becomes increasingly difficult. With
the rise of generative Artificial Intelligence (AI), and particularly Large
Language Models (LLMs), new possibilities emerge to find and explore
literature. We introduce ASK (Assistant for Scientific Knowledge), an
AI-driven scholarly literature search and exploration system that fol-
lows a neuro-symbolic approach. ASK aims to provide active support
to researchers in finding relevant scholarly literature by leveraging vec-
tor search, LLMs, and knowledge graphs. The system allows users to
input research questions in natural language and retrieve relevant arti-
cles. ASK automatically extracts key information and generates answers
to research questions using a Retrieval-Augmented Generation (RAG)
approach. We present an evaluation of ASK, assessing the system’s us-
ability and usefulness. Findings indicate that the system is user-friendly
and users are generally satisfied while using the system.
Keywords:AI-Supported Digital Library·Intelligent User Interface·
Large Language Models·Scholarly Search System
1 Introduction
Analyzing scholarly literature is a key aspect of research. However, due to the
ever-increasing body of scholarly publications, finding scholarly literature be-
comes increasingly difficult [12]. Consequently, finding literature consumes a
substantial portion of researchers’ time [15]. Because of the recent developments
in generative Artificial Intelligence (AI), and specifically Large Language Mod-
els (LLMs), new possibilities arise to extract knowledge from scholarly articles,
helping researchers to find relevant literature in the flood of publications.
In this article, we present ORKG ASK (Assistant for Scientific Knowledge),
hereafter referred to as ASK, a next-generation scholarly search and exploration
system. ASK aims to provide support to researchers in finding relevant schol-
arly literature. ASK takes a Neuro-Symbolic approach which consists of threearXiv:2512.16425v1  [cs.IR]  18 Dec 2025

2 Oelen et al.
Q: Wh y ar e some...A: Lit er atur e sho ws...Ent er r esear ch questionGener at e 
embeddingsR etrie v al of t op n 
r ele v ant document sGener at e answ erV ect or database......12nLLMLit er atur e corpusR ele v ant document sR esear ch questionFind r esear ch y ou ar e looking f or
Answ er question 
fr om cont e xtPr epar ed pr omptPr omptLLM as gener at orbrain~ 7 6M ar ticlesAbstr act s and 
optionally full-t e xtQ: Wh y ar e some...SearchAugment pr ompt s wit h 
n document cont e xt
Fig. 1.Explainer depicting our RAG (Retrieval-Augmented Generation) approach for
scholarly search. TheRetrievalstep ranks articles by their relevance to the question.
TheAugmentedstep injects the previously retrieved context in the prompt. TheGen-
erationstep prompts the LLM and displays the answer.
key components, namely Vector Search and LLMs for the neural aspect and
Knowledge Graphs (KGs) for the symbolic part. We build upon our previously
presented work where we demonstrated the basic ASK infrastructure [16]. In
this paper, we expand on our previous work by providing an in-depth explana-
tion of the approach, technical details of the implementation, and a extensive
evaluation. In brief, ASK functions as follows: a user of ASK formulates their
information needs as a research question. Afterward, a list of relevant articles
is displayed. For each article, an automatically extracted answer for the previ-
ouslyaskedquestionisdisplayedtotheuser.Finally,thesymbolicaspectensures
usersareabletonarrowdownthesearchspacebyprovidingsemanticfilters.This
provides both the precision of symbolic approach and the flexibility of a neural
approach. ASK is running as a publicly available production service online.3
The system takes a Retrieval-Augmented Generation (RAG) approach to
support the previously described workflow. RAG [14] is commonly used to inter-
twine LLM extractions with information retrieval systems, as depicted in Fig-
ure 1. Firstly, the Vector Search component ranks documents based on their
relevance (Retrieval) for a research question. Secondly, relevant context is col-
lected (i.e., the paper abstract and, if available, full-text) from the previous step
(Augmented). Finally, the LLM generates answers and displays this to the user
(Generation). A screenshot showing the ranked articles, search query, and gen-
erated LLM responses is displayed in Figure 2. This work introduces the follow-
ing contributions: i) presents an LLM-supported open-source scholarly literature
search and exploration service, ii) describes a scholarly RAG system leveraging
LLMs, vector search, and KGs, and iii) provides insights from the design and
development process, supported with an evaluation.
2 Related Work
The landscape of scholarly search systems can be categorized into two groups,
domain-agnostic and domain-specific systems. Prominent examples of domain-
3https://ask.orkg.org

ASK: an AI-driven Scholarly Literature Search and Exploration System 3
Fig. 2.Screenshot of the ASK search results page. The nodes with (N)FR correspond
to the implementation of the (Non-)functional requirements, as listed in Table 1.
agnosticsystemsincludeGoogleScholar,SemanticScholar,andScopus[6].Well-
known domain-specific systems include PubMed for the medical domain and
ACM Digital Library for the computer science domain. At a high level, these
systemsfunctionsimilarly:relevantarticlesarereturnedforasetofuser-provided
keywords. A new generation of scholarly search systems tries to automatically
extract relevant information from articles. Among others, those systems include
Elicit, Consensus, and Scispace [3]. Those systems are similar in that sense that
they are not open source, making it difficult to determine what models are em-
ployed and what underlying data corpus is used. This makes the reproducibility
of results, for example for a systematic literature review, a challenging task. Ad-
ditionally, the trustworthiness of generated responses by Large Language Models
(LLMs) becomes increasingly problematic when the underlying technologies are
not transparent. To the best of our knowledge, the previously mentioned systems
typically use a RAG (Retrieval-Augmented Generation) [14] approach, as also
mentioned by Bolanos et al. [3]. This roughly resembles the approach of ASK.
However, a notable difference between ASK and the other systems is the open
nature of ASK. All source code is openly available online, and we clearly com-
municate which data corpus we use and which models we employ. This makes
ASK more suitable for reproducible literature searches.
LLMs gained a lot of traction after the seminal work of Vaswani et al.
which paved the way for transformer models like BERT, RoBERTa, and more
scientifically-orientedmodelslikeSciBERT[2].Modelsthenquicklystartedgrow-
ing in size and capabilities such as GPT3 [4]. From this point on, very large mod-
els started to show impressive performance across a vast spectrum of language
tasks. Despite their impressive capabilities, LLMs face several key challenges.
There are concerns about the use of LLMs in fields requiring high precision, such

4 Oelen et al.
as healthcare [19] and law, where inaccuracies can have serious consequences. A
particular issue is the phenomenon of “hallucinations”, where models generate
statements that, while plausible, are entirely fabricated and lack factual basis [1].
In the context of the scholarly domain, LLMs need to be more accurate
for widespread usage [20]. CORE-GPT [18] is an effort to combine LLMs like
ChatGPT with open-access research to provide a more trustworthy and credi-
ble scientific question answering. Furthermore, Van Dis et al. highlighted that
researchers need to pay extra attention when using LLMs for research purposes
specifically when applying them to literature comprehension and summarization
tasks [21]. Our work with ASK positions itself in the middle, trying to trans-
parently show where the answers came from and at the same highlight to users
that the answers are automatically generated via a language model and as such
need to be reviewed by a human. ASK bridges the missing part of search sys-
tems which is the natural language expression and connects it with the advanced
capabilities of LLMs to get the best of both worlds.
3 Approach
In this section we present our approach for scholarly search and discuss the
system requirements, which essentially cultivated in the ASK system.
3.1 RAG for Scholarly Search
For a scholarly search system, parametric knowledge of LLMs should be limited
andnotusedasamainsourceofinformation.Parametricknowledgeistheknowl-
edge that models encode within their vast number of parameters, and with it,
LLMs are able to answer questions. Relying solely on such knowledge can lead
to hallucinations and the generation of inaccurate information. For the afore-
mentioned reasons, ASK relies on Retrieval-Augmented Generation (RAG) [14]
to combine the parametric knowledge of the models and the non-parametric
knowledge stored in vector stores to generate accurate and related text.
Non-Parametric MemoryAlsoreferredtoasSemanticSearch.Non-parametric
memory is a part of the procedure that extends the knowledge reservoir of pre-
trained language models to the requirements of individual applications. This
type of memory provides various benefits to scholarly search: i) Customizability:
the knowledge base contains only the items of interest and as such direct the
LLM to answer only in relation to documents indexed within the vector store.
ii) Updatable knowledge base: since there is no need to retrain the language
model with every new document added, new documents can be easily indexed
and added to the already-existing knowledge base. iii) Complex filtering capa-
bilities: the vector store also offers the flexibility to further filter or refine search
results based on metadata or other available criteria.
Inorderfortheretrievercomponenttowork,firstasetofdocumentsneedsto
beprocessedandindexedinsideavectorstore.Thevectorstoreispopulatedwith

ASK: an AI-driven Scholarly Literature Search and Exploration System 5
a semantical representation of documents, via embeddings. ASK uses the Nomic
embedding model4which has an embedding size of 768 and a context window
sequence length of 8K token. The choice of the embedding model is based on its
advanced multilingual capabilities, efficient parameter utilization via MoEs, and
its long context handling ability. Finally, the collection of documents retrieved is
then passed down to the parametric memory component for further processing.
Parametric MemoryThe parametric memory component solely relies on the
languagemodelitself.Usingbothparametricandnon-parametricmemoriesside-
by-side has multiple benefits for LLM-based applications: i) Tailored responses:
rather than posing a general query to the model and expecting an answer, the
model now receives the query and the context in which it is supposed to look
up the answer. ii) Reduced hallucinations: LLMs are notorious for hallucinating
content [17]. With specifying the context, the model is forced to rely on the text
that exists within its prompt and not within its own parametric knowledge. iii)
Instruction following: the parametric knowledge within the models allows for
custom instructions to generate apt responses depending on the use case.
ASK utilizes custom-made prompts containing placeholders that take pieces
of information from the non-parametric memory and are then used by the LLM.
ASK uses the small variant of Mistral LLM5[9]. The usage of a relatively small
modelreducestherequiredcomputationalresourcesandloadingtimes,whilestill
being well capable of following instructions. The Mistral model has a 32K tokens
context window which is suitable for passing the full text of articles and getting
specific answers. Inferencing with LLMs can be resource-intensive and is usually
the bottleneck when it comes to performance in production systems. For this
particularreason,acachingmechanismisemployedwiththeparametricmemory.
Caching is applied on partial hits (i.e., for single cells), making it possible to
return single responses partially from the cache and partially from the LLM.
The implemented caching mechanism reduces the loading times significantly and
prevents calling the LLM, which in turn benefits computational efficiency.
3.2 System Requirements
To provide guidance during development, we formulate a set of system require-
ments, as listed in Table 1. The requirements are divided into functional and
non-functional requirements. For brevity reasons, we list the high-level require-
ments only. The functional requirements focus on literature search, information
extraction, and the ability to filter and organize information. The two most
important non-functional requirements are the reproducibility options and the
focus on barrier-free access via various accessibility features.
4In particular “nomic-embed-text-v1.5” which utilizes Matryoshka Representation
Learning [11].
5ASK uses Mistral 7B Instruct v0.2 with no sliding-window attention.

6 Oelen et al.
Table 1.Listoffunctionalandnon-functionalsystemrequirements,outliningthehigh-
level key concepts to guide the system development.
ID Title Requirement Rationale
Functional requirements
FR1Literature search The system shall allow users to find schol-
arly literature for research questions.To provide a scholarly search and explo-
ration system.
FR2Information ex-
tractionThe system shall display automatically ex-
tracted information from found literature.To ensure users get a quick overview of the
literature so relevancy can be assessed.
FR3Answer synthesis Thesystemshallprovideasummarizedan-
swer to research questions.To provide a clear answer to the research
question based on a set of articles.
FR4Result filtering The system shall allow users to set filters
for finding related semantic concepts.To narrow down the search space and pro-
vide a more fine-grained search.
FR5Bibliography
managerThe system shall provide a bibliography
manager to store related literature.To ensure collections can be stored and to
allow importing existing articles.
Non-functional requirements
NFR1Reproducibility The system shall always produce repro-
ducible responses.To ensure the system is suitable for schol-
arly research and is transparent to users.
NFR2Accessibility The system shall follow accessibility guide-
lines to ensure accessibility for all users.To ensure that users with disabilities can
use all features.
NFR3Usability The system shall be easy-to-use and can be
operated with a minimal learning curve.To provide an alternative to existing schol-
arly search systems.
NFR4Maintainability The system shall follow established coding
standards to facilitate maintainability.To ensure the system can be employed as
a sustainable service.
NFR5Interoperability The system shall be interoperable with ex-
isting bibliography managers.To ensure literature can be imported and
exported to existing systems.
4 Implementation
In this section, we present the implementation details to realize the ASK system.
We present the functional and non-functional requirements, the LLM setup for
various use cases, the dataset used to populate the index of the search compo-
nent, and technical details about the implementation.
4.1 Requirements Realization
In Figure 2, a screenshot depicts the ASK interface. We will now discuss how
the previously listed system requirements are implemented within this interface.
Functional RequirementsThe literature search (FR1) is implemented by
providing a large search box on the homepage from where users can get started
by entering a research question. The research result page shows the question and
a list of results ranked by relevance (Figure 2 node FR1). The LLM-extracted
answerisdisplayedinnodeFR2,alongsideadditionalcolumns thatareextracted
as well. Users can modify those columns to extract specific information by click-
ing theEdit columnsbutton. For the answer synthesis, a summarized answer is
displayed at node FR3. Citations within this summarized answer point to the
results listed below. Results can be filtered in the box displayed at node FR4. Fi-
nally, items can be added to a bibliography collection by clicking the bookmark
icon as displayed in node FR5.

ASK: an AI-driven Scholarly Literature Search and Exploration System 7
## pr oper ties 
{pr oper ties} 
## paper cont ent 
{ cont ent}R esear ch Question: { question}
# Abstr act s: {abstr act s}
# Answ er wit h inline-citations as [#]: P1: Inf ormation e xtr action pr omptP2: Answ er synt hesis pr omptSyst em pr omptSyst em pr omptR ole definitionR ole definitionInput specInput specT askT askOutput detailsOutput detailsOutput syntaxOutput syntax
User inputUser inputRA G cont e xtPrimerRA G cont e xtUser pr omptUser pr omptY ou ar e an analysis-suppor t bot t hat oper at es on scholarly 
document s and f ollo ws instructions.  
Y ou will get as an input: t he r esear ch paper cont ent and a 
set of pr oper ties/crit eria t o look f or .  
Y ou will e xtr act t he v alues corr esponding t o t he list of 
pr o vided pr edicat es.  
Limit t he v alues t o only t he cont ent wit hout pr efixing it 
wit h t he paper contains or t he t e xt sa ys.  
Y our output should AL W A YS be lik e t his: 
<e xtr action><k e y>PROPERTY</k e y><v alue> V ALUE</
v alue></e xtr action> ......Y ou ar e an analysis-suppor t bot of scholarly document s 
t hat f ollo ws instructions.  
Y ou will gener at e a compr ehensiv e answ er t o t he giv en 
r esear ch question (but no mor e t han t hr ee/f our sent ences) 
solely based on t he cont ent pr o vided.  
Cit e t he number of t he cont ent r ef er enced f or each claim 
lik e t his: [1] f or a single r ef er ence or [2][3] f or multiple 
r ef er ences.  Emphasiz e br e vity ,  f ocusing on essential 
details and omitting unnecessar y inf ormation.  A v oid 
adding t he question in t he answ er and Do not include an y 
not es or comment s
Fig. 3.Sample prompts for different RAG use cases within ASK. The system prompts
provide the instructions to the LLM (Prompt P1 trimmed for brevity reasons). The
user prompt includes the user input and RAG context. Values highlighted in red are
placeholders used to inject user values into the prompt. Additionally, P2 uses a primer
to improve the answer of the LLM.
Non-Functional RequirementsIn addition to being open source, ASK also
provides data to ensure results are reproducible (NFR1). We created the repro-
ducibility menu (see Figure 2, node NFR1) that provides for all LLM-generated
content: i) the prompts, i) the model, iii) the parameters (such as temperature,
seed, etc.), and iv) the context used for the generation (full text or abstract).
The transparency helps users to better assess the results’ correctness and en-
ables them to reproduce the same results themselves. Among other features,
this sets ASK apart from other services, which are often proprietary and lack
transparency. Accessibility is another key aspect of ASK. Any user, including
those with disabilities, should be able to use the system barrier-free (NFR2).
We integrated various features to facilitate accessibility. Firstly, the interface is
responsive, making sure that the interface is usable at large zoom levels, benefit-
ing users with visual impairments. As an additional benefit, the service can also
be operated on different screen sizes, such as tablets and mobile phones. A dark
mode (as displayed on the right bottom of Figure 2) replaces all light colors with
dark ones, and can be enabled to reduce eye strain. Secondly, ARIA attributes
are added to facilitate screen reader usage, benefiting users with visual impair-
ments [5]. Finally, the interface is internationalized, meaning that the service can
be operated in different languages and different regions. The LLM responses are
provided in multiple languages as well, opening up new methods to find related
literature in the preferred language of the user, and making literature search
more inclusive. In the end, the various accessibility features benefit all users.
The interface is designed to look intuitive and modern benefiting usability
(NFR3). To ensure the system is maintainable (NFR4) and can be operated as
a production service, it is implemented using the latest technologies and code
standards. Details about the implementation are described in subsection 4.4.

8 Oelen et al.
To make the system interoperable with other reference managers (NFR5), we
adopted the Citation Style Language (CSL)6throughout the system.
4.2 LLM Setup
A construct of an LLM chain is implemented. A chain is the combination of
three components: i) Prompt, ii) Model, and a iii) Parser. The prompt is an
aggregation of the system and user prompt for a particular task (see Figure 3).
Before the invocation of the model, the relevant information retrieved by the
non-parametric memory is injected and formatted into the prompt. Secondly,
the model is the LLM and potentially any LoRAs [8] that need to be applied to
the language model7. Lastly, a parser is a function that gets called on the output
(i.e., the response of the LLM) and is then parsed, sanitized, and formatted to be
used in other parts of the application. We note that we did not need to employ
custom-trained or fine-tuned LLMs for ASK at this stage. As the LLM is mostly
used as a means to perform information extraction, and in turn text generation,
a fine-tuned model would not necessarily result in higher-quality results.
4.3 Dataset
ASK uses the CORE [10] dataset of open-access research papers as the basis of
its indexed corpus. The CORE data is automatically crawled from open-access
repositories and publisher websites. This means that there are quality-related
issues that require some curation before any ingestion operation takes place. Be-
fore indexing the CORE data in the vector store of ASK, a pre-processing phase
was implemented to choose, based on a set of heuristics, which items and articles
are suitable to be added. This process involves checking if the articles have valid
titles and abstracts (i.e., non-empty and have a length greater than a threshold).
The abstracts proved to be the most impacting factor within this process. The
data import process is a continuous process as the CORE data is growing with
time and other sources are also integrated within the ASK system. In total, we
imported 76.4M articles from the CORE dataset, excluding items that do not
follow the previously mentioned requirements. Of the imported articles, 36.9%
have a DOI and 25% have full-text available.
In addition to the CORE data, we imported a subset of BMBF-conform
(German Federal Ministry of Education and Research) research reports related
to autonomous driving, containing approximately 310 reports.8The ASK service
is operated by the German National Library of Science and Technology (TIB)
and by importing this dataset, we demonstrate how ASK can be leveraged to
explore the library’s special collections. In the future, we plan to import more
of such special collections.
6https://citationstyles.org
7ASK does not implement any LoRAs at the moment. However, this technique can
be integrated to further customize the model for domain-specific use cases.
8https://ask.orkg.org/search?query=&filter=AND[0][source][inList][0]
=TIB%2520Forschungsberichte%2520Autonomes%2520Fahren

ASK: an AI-driven Scholarly Literature Search and Exploration System 9
4.4 Technical Details
Thesystemisdevelopedusingamicroservicessetup9andisavailableasafreeon-
line service. The services is divided into the frontend and backend. The frontend
is written in TypeScript with React. It uses the Next.js framework, adopting the
server components paradigm where suitable. Furthermore, it uses Tailwind for
styling and HeroUI as a component library. The use of standardized technologies
increases the maintainability, as described in NFR3. The frontend is available as
open-source software and is published with a permissive MIT license.10
The backend is mainly written in Python leveraging the FastAPI framework.
The backend adopts a modular approach where each functionality is in its own
module, which improves maintainability and extensibility, as described in NFR4.
Other components to serve the language models, vectorized documents, and
cache items are part of the backend but are written in different languages and
are used as turn-key solutions. The source code of the backend and various
components are available publicly under the MIT license.11Furthermore, ASK
utilizes Qdrant12as a vector store and the TGI13engine for serving LLMs and
inferencing in production. The containers are managed via Podman.
5 Evaluation
Wenowdiscussthesystemevaluation,whichisdividedintotwoparts:subjective
user evaluations and objective data analysis.
5.1 Subjective Evaluation
ASK is publicly released as a scholarly information retrieval service and is being
activelyusedbyresearchers.Togatherfeedbackfromreal-worldsystemusers,we
integrated a lightweight feedback collection component into the user interface.
The feedback component appears on question pages. The component consists of
twodifferentsetsofquestions.Thefirstsetevaluatesthehelpfulness,correctness,
and completeness of the displayed question and its answers. The second set
of questions asks users about their general feedback on the ASK system. The
questions consist of two standardized and unmodified UMUX-Lite [13] questions
and one to assess whether users are satisfied with ASK. User satisfaction is
another commonly used method to assess usability. The operational feedback
is collected on a running basis, previous results consisting of a small number of
responses(approximately3%)havebeenpublishedalreadyinademoarticle[16].
Participation in the operation feedback questionnaire is on an opt-in basis and
users can close the feedback popup if they do not wish to participate.
9Server configuration: 1TB of RAM, 15TBs of SSD storage, 128 CPU cores, and seven
GPU cards (Nvidia L4 4x24GB, Nvidia L40S 2x46GB, and Nvidia H100 1x80GB).
10https://gitlab.com/TIBHannover/orkg/orkg-ask/frontend
11https://gitlab.com/TIBHannover/orkg/orkg-ask/backend
12https://qdrant.tech
13https://huggingface.co/text-generation-inference

10 Oelen et al.
Fig. 4.Question specific results for operational feedback collection.
The question-specific form has been filled out 1,212 times in the period from
June 15, 2024, until January 15, 2025. Based on browser fingerprinting, it was
completed by 1,032 different users. The results are displayed in Figure 4. As can
be observed, the results of the helpfulness of answers vary among users. This
means that participants experience different levels of relevance for the answers,
which can be explained by their expectations of the system. The correctness of
answers is voted as more neutral. Meaning that users might had difficulty assess-
ing the correctness, or thought answers were neither fully correct nor incorrect.
Finally, the results for completeness are similar to correctness, indicating that
most users had no strong opinion about the completeness of the answers.
The results of UMUX-lite questions to assess the general usability of the sys-
tem are displayed in Figure 5. A total of approximately 443 users filled out this
evaluation. As the questions were optional, the number of users differs slightly
from the numbers in the figure. The numbers result in a calculated UMUX score
of 65.7 on a scale of 0-100, where higher scores indicate better usability. Incom-
plete partial responses were discarded in the final UMUX calculation, resulting
in 409 included responses. As the results show, ASK does not always meet the
users’ requirements. However, the majority of users do agree that the system
itself is easy to use. This indicates the design decisions to make the system easy
to use have proven to be effective. Finally, Figure 6 displays the user satisfac-
tion outcomes. In total, 363 users answered this question. As can be observed,
average user satisfaction leans toward more positive than negative opinions.
User ExperimentIn addition to the operational feedback, we conducted a
small-scale user experiment to compare ASK to an established literature search
system, specifically Google Scholar. We were particularly interested in the per-
ceived task load differences between ASK and Google Scholar. For this, we de-
signed a within-subject study where a total of 9 participants had to answer a
set of four predefined research questions, two per condition. Most of the partic-
ipants are engaged in academic research, have either a master or PhD degree,
and have used ASK before, but were not involved in the development of ASK.
The majority of participants (7 out of 9) searches for academic articles at least
on a weekly basis. To counteract sequence bias, the two conditions were evalu-
ated by participants in random order. To answer the questions, the participants
had to use at least two references per answer, the references had to be provided

ASK: an AI-driven Scholarly Literature Search and Exploration System 11
Fig. 5.UMUX-Lite results with a score of 65.7.
How satisfied ar e y ou 
with ORK G Ask?Fig. 6.General user satisfac-
tion of ASK.
by the search system. Additionally, they had to manually verify the correctness
of the answer from the source article, consequently, they were not allowed to
copy-paste LLM-generated answers. For each condition, they had to indicate
their perceived task load, measured with an unweighted NASA Task Load Index
(TLX) scale [7]. Additionally, the time to answer a question was recorded.
The results of the comparisons between the ASK and Scholar search con-
dition are displayed in Table 2. As can be observed, the ASK condition has a
considerably lower perceived task load compared to the Scholar condition. Re-
garding the required time, there was an outlier that took more than 113 minutes
to answer the questions via ASK, while only taking 24 minutes to answer via
Scholar. For completeness, the timing results are displayed with and without
the outlier. When the outlier is disregarded, the required time to answer ques-
tions with ASK is lower than with Scholar. Although ASK and Google Scholar
might not be directly comparable, making the conclusions less definitive, the
comparison gives insights into the future direction of providing alternatives to
established scholarly search systems.
5.2 Objective Evaluation
Togaininsightsintohowtosystemisused,wecollecteddatausingwebanalytics.
We analyzed user interaction data over the period starting from May 15, 2024,
Table 2.Showing a comparison between the ASK and Scholar condition regarding
task load and required time. Task load is listed as percentage and time in seconds.
ASK Scholar
Mean SD Mean SD
Task load (TLX) (%) 26.7616.65 61.316.59
Time (s) 984.41456.671241.39496.36
Time with outlier (s) 1628.791979.78 1267.49470.86

12 Oelen et al.
Table 3.Web analytics data and user interaction statistics of the ASK production
service. Measured from May 15, 2024, until February 1, 2025.
Analytics
Visits 74,145
Returning visits 26,354
Pageviews 219,189
Duration visit 4:01m
Bounce rate 3%Events
Queries asked 67,949
Downloads 7,595
Outlinks 19,723
Custom filters added 723
Custom columns added 415
Load more (1 page) 5,067
Load more (2 pages) 2,149
Load more (>2 pages) 5,010Device usage
Desktop 76,9%
Smartphone 21,2%
Other 1,9%
until February 1, 2025. Analytics data was recorded via Matomo14and used
browser fingerprinting to distinguish between users. Additionally, specific events
in the interface were logged to determine how frequently they were used. A
summarized overview of the analytics and user interaction statistics is presented
in Table 3. Visits are defined as users visiting the service who have not visited
a page in the last 30 minutes. The bounce rate of 3% is rather low, indicating
that users are actually using the system, and not just visiting a single page
and then leaving the service. A considerable amount of users are using ASK on
devices other than desktops. This indicates that the service indeed works well
for different screen sizes, part of our accessibility requirement NFR2.
The recorded events gain more insights into what features are actually being
used. The downloads and out-links relate to the number of articles that are being
visited from ASK. This includes following links to the PDF, publisher landing
pages, or data repositories. The number of added custom filters is low. Also,
the number of columns added is rather low, meaning that most users did not
use the functionalities to extract custom data from articles. The low number
of custom extractions either means that users were already satisfied with the
default extracted properties, or they did not understand how to use the feature.
Finally, the load more numbers show the number of times a user clicks the “Load
more” button at the bottom of the page, indicating they are interested in finding
more related work.
6 Discussion
ASK is meant as a scholarly search system, helping researchers to find and ex-
plore scholarly literature. Although ASK indeed also answers research questions
while performing a search, first and foremost ASK is a scholarly information
retrieval system. Therefore, finding literature is the main objective of ASK. We
consider the question-answering feature as a means to find related work, not as
anobjectivebyitself.ThismeansthatthepreviouslydiscussedRAGmethodisa
key aspect of our approach and using LLMs in isolation, even when pre-trained
14https://matomo.org

ASK: an AI-driven Scholarly Literature Search and Exploration System 13
or fine-tuned on scholarly articles, would not be sufficient to meet our search
goal. Therefore, in the user experiment, we specifically focused on comparing
ASK to a scholarly search system and not a question-answering system.
As previously discussed, when employing LLMs, one has to keep in mind that
LLMs tend to hallucinate. Hallucinations cannot be completely eliminated, even
when using models with a higher number of parameters (i.e., more powerful and
capable models). Indeed, for the use case of ASK, hallucinations can also occur,
but to a large extent do not pose major challenges. As the question-answering
capability is secondary to the main goal of information retrieval, users do not
solely rely onthe extracted information to find relevant literature. As mentioned,
it is only a means to assess the literature’s relevance. After potentially relevant
articles are discovered, users are expected to perform rigorous analysis of the
listed work, as is also expected when using traditional scholarly search systems.
To communicate this to users, a warning message is displayed in the interface,
informing users that all information needs to be manually checked.
The evaluation presents the results of operational feedback and data col-
lected from a production environment. Consequently, data is collected in an
uncontrolled environment, meaning that individual responses and data might be
inaccurateorincomplete.Forexample,theusers’intentionsoffillingouttheeval-
uation forms are unknown, which could be only for testing purposes or without
thoroughly reading the questions. Also, for the feedback, no demographics were
collected, limiting the possibility of an in-depth analysis of the results. Browser
fingerprinting was used to distinguish among different users, which means that
uniqueuserscannotbefullyaccuratelydetermined.Therefore,itshouldbenoted
that the evaluation results might contain data from the same users, even though
they are reported as different users. However, we do consider the results to be
relevant nevertheless, and when aggregated, provide valuable insights.
Finally, we will discuss future work. The usefulness of the approach as a
search system heavily depends on the quality and the size of the underlying lit-
erature corpus. We plan to extend our literature repository by including more
articles from different sources, and by parsing content semantically, for exam-
ple by performing author name disambiguation, all contributing toward more
semantic search. As mentioned in section 2, ASK sets itself apart from similar
existing services by providing an open-source service and focusing on literature
search reproducibility. A quantitative performance comparison with these ser-
vices is out of scope for this work, but is an interesting future research direction.
ASK’s transparency features provide a key advantage over these services, but a
comparison regarding the quality of the responses provides helpful insights into
other aspects of our approach.
7 Conclusion
In this work, we presented ASK, a scholarly search and exploration system. We
examined how AI can be leveraged for literature searches by exploring the direc-
tion of using vector search and LLMs to provide active support to users while

14 Oelen et al.
conducting a literature search. ASK showcases how such a literature search sys-
tem can operate. Furthermore, we focused on the ability of LLMs to provide
value to researchers while performing a literature search. To this end, we lever-
aged a RAG (Retrieval-Augmented Generation) approach to find and explore
scholarly literature. Using a RAG approach, LLMs are used for information ex-
traction and text generation, making a large literature corpus explorable using
AI technologies. The RAG approach provides a literature search where source
information is more easily traceable while partially mitigating common LLM
limitations, such as hallucinations and limited context sizes. Finally, we investi-
gatedwhetherAI-driven tools,suchasASK,arepotentialalternativestoalready
established scholarly search tools. By presenting the ASK service and its respec-
tive user study, we captured researchers’ attitudes toward the new approach and
concluded there is indeed potential and interested audience for such new tools.
AcknowledgementsThis work was co-funded by NFDI4DataScience (ID:
460234259), and by the TIB Leibniz Information Centre for Science and Tech-
nology. We want to thank the entire ORKG team for their contributions to the
ORKG platform, including research and development efforts.
Bibliography
[1] Bang, Y., Cahyawijaya, S., Lee, N., Dai, W., Su, D., Wilie, B., Lovenia,
H., Ji, Z., Yu, T., Chung, W., et al.: A multitask, multilingual, multimodal
evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv
preprint arXiv:2302.04023 (2023)
[2] Beltagy, I., Lo, K., Cohan, A.: Scibert: A pretrained language model for
scientific text (2019),https://arxiv.org/abs/1903.10676
[3] Bolanos, F., Salatino, A., Osborne, F., Motta, E.: Artificial Intelligence
for Literature Reviews: Opportunities and Challenges. arXiv preprint
arXiv:2402.08565 (2024)
[4] Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-
Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M.,
Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S.,
Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever,
I., Amodei, D.: Language models are few-shot learners (2020),https://
arxiv.org/abs/2005.14165
[5] Craig, J., Cooper, M., Pappas, L., Schwerdtfeger, R., Seeman, L.: Accessible
rich internet applications (wai-aria) 1.0. W3C Working Draft (2009)
[6] Gusenbauer, M., Haddaway, N.R.: Which academic search systems are suit-
able for systematic reviews or meta-analyses? Evaluating retrieval qualities
of Google Scholar, PubMed, and 26 other resources. Research Synthesis
Methods11(2), 181–217 (2020).https://doi.org/10.1002/jrsm.1378
[7] Hart,S.G.:NASA-taskloadindex(NASA-TLX);20yearslater.Proceedings
of the Human Factors and Ergonomics Society pp. 904–908 (2006).https:
//doi.org/10.1177/154193120605000909

ASK: an AI-driven Scholarly Literature Search and Exploration System 15
[8] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
Chen, W.: Lora: Low-rank adaptation of large language models (2021),
https://arxiv.org/abs/2106.09685
[9] Jiang,A.Q.,Sablayrolles,A.,Mensch,A.,Bamford,C.,Chaplot,D.S.,delas
Casas,D.,Bressand,F.,Lengyel,G.,Lample,G.,Saulnier,L.,Lavaud,L.R.,
Lachaux, M.A., Stock, P., Scao, T.L., Lavril, T., Wang, T., Lacroix, T.,
Sayed, W.E.: Mistral 7b (2023),https://arxiv.org/abs/2310.06825
[10] Knoth, P., Herrmannova, D., Cancellieri, M., Anastasiou, L., Pontika, N.,
Pearce, S., Gyawali, B., Pride, D.: Core: A global aggregation service for
open access papers. Nature Scientific Data10(1), 366 (Jun 2023)
[11] Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan,
V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., Farhadi, A.: Ma-
tryoshka representation learning (2024),https://arxiv.org/abs/2205.
13147
[12] Landhuis, E.: Scientific literature: Information overload. Nature535(7612),
457–458 (2016)
[13] Lewis, J.R., Utesch, B.S., Maher, D.E.: UMUX-LITE: When there’s no
time for the SUS. In: SIGCHI Conference on Human Factors in Computing
Systems. pp. 2099–2102 (2013)
[14] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela,
D.: Retrieval-augmented generation for knowledge-intensive nlp tasks. In:
Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Ad-
vances in Neural Information Processing Systems. vol. 33, pp. 9459–9474.
Curran Associates, Inc. (2020)
[15] Niu, X., Hemminger, B.M., Lown, C., Adams, S., Brown, C., Level, A.,
McLure, M., Powers, A., Tennant, M.R., Cataldo, T.: National study of
information seeking behavior of academic researchers in the United States.
Journal of the American Society for Information Science and Technology
61(5), 869–890 (2010)
[16] Oelen, A., Jaradeh, M.Y., Auer, S.: Orkg ask: A neuro-symbolic scholarly
search and exploration system. Joint Proceedings of Posters, Demos, Work-
shops, and Tutorials of the 20th International Conference on Semantic Sys-
tems (2024),https://ceur-ws.org/Vol-3759/paper7.pdf
[17] Perković, G., Drobnjak, A., Botički, I.: Hallucinations in llms: Understand-
ing and addressing challenges. In: 2024 47th MIPRO ICT and Electronics
Convention (MIPRO). pp. 2084–2088. IEEE (2024)
[18] Pride, D., Cancellieri, M., Knoth, P.: Core-gpt: Combining open access re-
search and large language models for credible, trustworthy question answer-
ing (2023),https://arxiv.org/abs/2307.04683
[19] Shen, Y., Heacock, L., Elias, J., Hentel, K.D., Reig, B., Shih, G., Moy, L.:
Chatgpt and other large language models are double-edged swords (2023)
[20] Susnjak, T., Hwang, P., Reyes, N.H., Barczak, A.L.C., McIntosh, T.R.,
Ranathunga, S.: Automating research synthesis with domain-specific large
language model fine-tuning (2024),https://arxiv.org/abs/2404.08680

16 Oelen et al.
[21] Van Dis, E.A., Bollen, J., Zuidema, W., Van Rooij, R., Bockting, C.L.:
Chatgpt: five priorities for research. Nature614(7947), 224–226 (2023)
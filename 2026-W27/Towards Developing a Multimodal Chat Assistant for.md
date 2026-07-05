# Towards Developing a Multimodal Chat Assistant for University Stakeholders: RAG-based Approach

**Authors**: Md Abu Hanif Shaikh, Abdullah Al Shafi

**Published**: 2026-07-01 16:03:03

**PDF URL**: [https://arxiv.org/pdf/2607.01115v1](https://arxiv.org/pdf/2607.01115v1)

## Abstract
University stakeholders often face difficulties in accessing timely and reliable information, especially in developing countries, where there are very few intelligent support systems. Existing rule-based chatbots are unable to handle complex, domain-specific queries and are not well-equipped to adapt to evolving institutional policies. As a fill-in-the-gap solution, we present the multimodal university chatbot with retrieval-augmented generation. The system combines the large language model with semantic retrieval to produce context-based responses from institution-centric resources, such as the university handbook. The system accepts text and image queries through the vision-language model and applies quantized inference for rapid deployment on constrained hardware. A scalable backend built with FastAPI, adjoined with a responsive frontend developed with Next.js, ensures real-time usability. Our multimodal evaluation demonstrates that the system maintains strong satisfaction scores across both text and image queries, despite increased response time for visual inputs. Furthermore, quantitative evaluation shows that hallucination is reduced from 31.7% to 6.6% in our proposed RAG-based system, confirming the effectiveness of retrieval grounding.

## Full Text


<!-- PDF content starts -->

2025 28th International Conference on Computer and Information T echnology (ICCIT)
19-21 December 2025, Cox’s Bazar, Bangladesh
T owards Developing a Multimodal Chat Assistant
for University Stakeholders: RAG-based Approach
Md Abu Hanif Shaikh, and Abdullah Al Shafi
Institute of Information and Communication T echnology ,
Khulna University of Engineering & T echnology , Khulna, Bangladesh
hanif@kuet.ac.bd, and abdullah@iict.kuet.ac.bd
Abstract —University stakeholders often face diﬀi-
culties in accessing timely and reliable information,
especially in developing countries, where there are
very few intelligent support systems. Existing rule-
based chatbots are unable to handle complex, domain-
specific queries and are not well-equipped to adapt
to evolving institutional policies. As a fill-in-the-gap
solution, we present the multimodal university chatbot
with retrieval-augmented generation. The system com-
bines the large language model with semantic retrieval
to produce context-based responses from institution-
centric resources, such as the university handbook. The
system accepts text and image queries through the
vision-language model and applies quantized inference
for rapid deployment on constrained hardware. A
scalable backend built with F astAPI, adjoined with a
responsive frontend developed with Next.js, ensures
real-time usability . Our multimodal evaluation demon-
strates that the system maintains strong satisfaction
scores across both text and image queries, despite
increased response time for visual inputs. F urther-
more, quantitative evaluation shows that hallucination
is reduced from 31.7% to 6.6% in our proposed RAG-
based system, confirming the effectiveness of retrieval
grounding.
Index T erms —Retrieval Augmented Generation, Vi-
sion Language Model, University Chat Assistant, Large
Language Model, V ector Store
I. Introduction
Instead of searching through lengthy documents, people
generally prefer to receive an instant and summarized
response [ 1]. That is why chatbots are getting popular day
by day in various sectors like education, clinical decision
support system, customer service, etc [ 2]. The same
holds for universities and their stakeholders like teachers,
administrative oﬀicers, students, staff, and potential appli-
cants. University chatbots are very common in developed
countries like the USA, UK, Canada, but almost unheard
of in developing countries like Bangladesh [ 2]. Hence,
building such a chatbot is indeed an important step toward
the modernization of academic support infrastructure,
bridging the digital divide, and enhancing student access
to information in resource-constrained educational envi-
ronments.
F urthermore, traditional rule-based chatbot systems
and static knowledge bases frequently cannot handle com-
plex queries involving subtle nuances in institutional poli-cies, dynamic changes in course offerings, or personalized
responses tied to user context [ 2]. This gap necessitates
the urgent development of intelligent systems capable of
reasoning, retrieval, and autonomous action.
Our proposed system addresses this challenge by lever-
aging retrieval-augmented generation (RAG) [ 3], an ad-
vanced paradigm in AI that integrates large language
models (LLMs) with retrieval mechanisms. This ap-
proach enables the chatbot to retrieve relevant university-
specific documents (e.g., academic policies, course cata-
logs, F AQs), which may change dynamically , without a
computationally expensive fine-tuning process. The core
of our system uses RAG to find and combine informa-
tion from documents, giving accurate answers based on
context. The backend uses F astAPI for handling many
requests at once, and the frontend ensures a smooth user
experience with Next.js and React. The main contribu-
tions of the work are listed below:
i. Development of a RAG-powered university chat as-
sistant that processes both text and image queries
utilizing a vision language model (VLM).
ii. T o support this chat assistant, a retrieval pipeline
utilizing semantic embeddings and chromaDB was
constructed to provide context-aware, accurate re-
sponses from institutional documents.
iii. Implementation of quantized inference and asyn-
chronous F astAPI backend for real-time, resource-
eﬀicient operation.
iv. Deployment of our proposed system as a publicly
accessible chat assistant at chat.kuet.ac.bd with
an interactive interface, feedback system, and demon-
strated ability to handle hallucination issues.
The rest of the paper is organized as follows, Section
II compares some existing works with limitations in the
context of Bangladesh. The architecture of our suggested
RAG-based system is described in Section III . The exper-
imental setup and simulation are presented in Section IV .
The ethical issues are discussed in Section V. Finally , a
conclusion is drawn in Section VI .
II. Related work
Research on university chatbots has developed from
early research on rule-based or semantically oriented mod-
els into advanced AI and hybrid architectures that increase
Authors’ version, published version’s DOI: 10.1109/ICCIT68739.2025.11490128arXiv:2607.01115v1  [cs.CL]  1 Jul 2026

T ext Combined 
prompt with 
relevant contexts Embedding 
model Vector 
store Relevant 
contexts Collection 
of chunks 
VLM Output Univ. 
Handbook Documents Document 
loader Text
splitting  Semantic 
Indexing Retrieval 
Augmentation 
Query 
Image Figure 1: Architecture of our developed system.
levels of student support and streamline administrative
tasks and academic resource accessibility .
At the very beginning, the paper [ 4] proposes a univer-
sity F AQ chatbot using AIML and latent semantic analysis
to provide eﬀicient, interactive, and accurate responses to
student queries. After that, Patel et al. present UNIBOT
[5], a web-based university chatbot developed with a
custom algorithm to provide fast, human-like responses
to student queries. Using the Rasa framework and BER T
and RNN models with an accuracy of 97.1% in identifying
intention, Nguyen et al. [ 6] developed an AI assistant for
admission processes and therefore reduced the workload of
admission personnel.
In the study of Balderas et al. [ 7], a chatbot system
was developed to facilitate communication with university
students in emergency situations, thus ensuring greater
campus safety . The study [ 8] reports the development and
comparative study of five NLP-based chatbot models for
counseling on university campuses, showing that sequen-
tial neural network models outclass TF-IDF and pattern-
matching approaches.
Parrales-Bravo et al. [ 9] have developed machine learn-
ing and NLP techniques to create a chatbot that is de-
ployed in T elegram, assisting students of the University of
Guayaquil in queries regarding enrollment and payment.
It shows extensive usability and positive acceptance among
the student population.
In [ 3], a hybrid chatbot integrating rule-based, retrieval-
based, and generative approaches is proposed for admin-
istrative support in education, achieving higher accuracy
and user satisfaction than stand-alone models. Neupane
et al. [ 10] present BARKPLUG V.2, a RAG-based chatbot
that improves access to university resources by deliv-
ering accurate, context-specific answers, achieving high
performance and satisfactory usability . The paper [ 11]
introduces EduChatGPT, a personalized AI chatbot for
education that integrates GPT-4 with real-time student
data to deliver tailored learning support. It employs a
modular architecture combining natural language process-ing, data integration, and adaptive feedback mechanisms.
Experimental evaluation demonstrates improvements in
student engagement and personalized learning outcomes.
The work highlights the potential of generative AI to en-
hance individualized academic support in higher education
contexts.
T o the end, though there are several university chat-
bots leveraging various techniques for developed countries,
there is a scarcity of such system in the developing and
under-developed countries including Bangladesh. W e have
tried to mitigate the gap by designing a web-based chat
application for university stakeholders in the context of
Bangladesh.
III. Proposed RAG-based System
As shown in Fig. 1, an AI system was designed to
accept multimodal input (text and image) to generate
coherent and context-sensitive responses. The system uti-
lizes cutting-edge vision language models, embedding tech-
niques, and document retrieval techniques. Its foremost
purpose is the understanding of queries posed in natural
language or image form, and then providing a descriptive
response by either querying from pre-defined knowledge
sources (University handbook) or image analysis on a real-
time basis.
A. System Overview
The system is configured as a web application with
RESTful APIs support. It receives input from the user
in the form of natural-language text or an image with
text and dynamically selects the appropriate pipeline de-
pending on the content of the input. The architecture has
three main modules: (1) inference engine; (2) knowledge
retrieval and augmentation; and (3) feedback collection
and monitoring.
1) Multimodal Inference Engine: The user inputs are
processed by a large-scale VLM llava-1.5-7b1, which is
capable of multimodal processing. The model is loaded on
1https://huggingface.co/liuhaotian/llava-v1.5-7b

(a) Chat interface
 (b) F eedback form
Figure 2: Interactive user interface of our system.
a quantized level to minimize memory consumption and
to make the inference faster, while maintaining suﬀicient
accuracy for the description and interpretation tasks. A
user may send either an image or a textual query , or both
at the same time. The system, in case both modalities
exist, builds a structured prompt incorporating the user’s
query and the image for context-aware visual reasoning.
T ext-only queries undergo language-only processing, with
an optional step of supplementing in-context information
from a domain-specific knowledge base. Whatever the sce-
nario might be, the model excels at instruction following
and natural language generation.
2) Knowledge Base Construction: T o enhance the
model’s responses with domain-specific knowledge, a static
document corpus is preprocessed and indexed. The
pipeline includes:
Document loading: The first step of the system
concerns the ingestion of documents, where raw textual
data is extracted from the knowledge source. In the
consideration of this study , the oﬀicial KUET university
handbook, in PDF format, was taken as the primary
source. Each page of the document was parsed to obtain
line-level text. T o reduce noise, regular expressions were
applied to remove irrelevant elements such as roman-
numeral headers, numeric indices, page markers, etc., none
of which have semantic information at all. In this way ,
only meaningful content was retained, thus maximizing
the relevance and accuracy of the other stages of the RAG
pipeline.
T ext splitting: Since LLMs are unable to process
arbitrarily long sequences of text, the loaded document
was thus divided into smaller, semantically coherent units.
At first, the text is recursively attempted to be split at log-
ical boundaries (paragraphs, sentences) before considering
character-based splits. A chunk size of 512 characters,
with an overlap of 64, is ideal for contextual continuitybetween adjacent chunks; ensuring that important phrases
near the boundaries are shared in multiple chunks and
therefore minimize information loss during retrieval. This
balance between granularity and semantic integrity guar-
antees that chunks are rich enough contextually and small
enough computationally to be handled for embedding
generation by the system.
Semantic Indexing: Once text splitting was done,
each chunk of text was converted into a dense vector em-
bedding to express semantic meaning. Embeddings were
created using the all-miniLM-L6-v22model that yields
384-dimensional vectors optimized for semantic similarity
tasks. The vectors were then put into a Chroma vector
store for storage due to its extreme lightness, ability to
persist data, and fast similarity search mechanisms. Per-
sisting the embeddings saved the system from recomputing
the embeddings across sessions, reducing latency . The
embeddings clustered semantically similar text chunks
together in vector space, thereby allowing the retrieval of
pertinent knowledge at query time.
3) Retrieval: The retriever acts as a bridge between user
queries and the knowledge base itself. When the query is
submitted, it must be embedded into the same seman-
tic space using the same embedding model to maintain
uniformity in representation. Then, the query embedding
is contrasted with the stored document embeddings by
their cosine similarity to select the highly relevant chunks.
Here, the top-3 were retrieved, as a good balance between
relevance and eﬀiciency was considered.
4) Augmentation via Context-Query F usion: After re-
trieving the most relevant top-3 chunks based on the
user’s query , concatenates these segments to create one
contextual block. This contextual block is then joined
with the original query to create an augmented prompt,
2https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

(a) text only
(b) image with text
Figure 3: Simulation of our system.
which is then passed to the language model. The final
prompt template is as follows:
“Use the following university-verified information
to answer. If the information is insuﬀicient, say
‘I cannot answer the question. ’
Retrieved Context 1: …
Retrieved Context 2: …
Retrieved Context 3: …
User Query: …”
Such prompt construction forces the model to be aware
of domain-specific knowledge retrieved from the underly-
ing corpus, thus sharpening its generation of grounded,
context-aware responses, and reducing hallucinations.
5) Performance Monitoring and F eedback Col lection:
F eedback can be provided via a form that accepts details
such as name, email, and comments on the outputs pro-
duced. This feedback is logged into a CSV file and later
used for assessment and enhancement of the model.B. Deployment
The proposed chat assistant is publicly deployed at
chat.kuet.ac.bd on an Intel Core i7-7700 CPU at 3.0
GHz (four CPUs), with 32 GB of main memory , and an
NVIDIA GeF orce GTX 1080 (8GB), running with Python
3.12.
IV. Experimental setup and simulation
In this section, we discuss the technologies that we have
used together with the simulation of our system.
A. Experimental setup
The proposed system is built using a Next.js front end
and a F astAPI back end. F or visual–language processing,
it employs the LLaV A-1.5-7B model, while ChromaDB
serves as the vector database. The generative AI compo-
nents are implemented using PyT orch. Documents are in-
gested through the PyPDFLoader module and segmented

Figure 4: Demonstration of how our system handle hallucination.
Figure 5: Distribution of query response time during the
trial phase.
using the RecursiveCharacterT extSplitter. Semantic simi-
larity search is used as the retrieval mechanism, supported
by the all-MiniLM-L6-v2 embedding model. The overall
retrieval-augmented generation (RAG) workflow is orches-
trated through the LangChain framework.
B. Interactive User Interface
In Fig. 2a, the interactive chat user interface of our
system has been shown, where users can ask queries (text
and/or image) and get their required information. Users
can also provide their feedback for further improvement of
our system using the feedback form shown in Fig. 2b .
C. Simulation
The simulation of our system is demonstrated in Fig.
3. In Fig. 3a, the user asks a domain-specific query
(KUET-related) only in text, and our RAG-based system
is capable of giving a context-aware reply . And the user
can also make a query based on an uploaded image, and
our system is also capable of answering such a query , which
is shown in Fig. 3b .
D. Handling Hal lucination
Fig. 4shows how our system can handle the LLM hal-
lucination problem. Hallucination means that LLMs give
wrong answers confidently , although there is no evidence
[12]. It is a very common limitation of a general LLM
models.However, as we are incorporating context and no
context related to the query was found, the LLM answersthat it doesn’t have the information instead of giving a
wrong answer.
E. Query Response Time
Fig. 5illustrates the distribution of query response
times recorded during the trial, based on 160 queries. The
response times range from 1.23 to 44.49 seconds, with a
mean of 13.57 seconds and a standard deviation of 10.87
seconds. The histogram reveals a right-skewed distribu-
tion, indicating that most queries were answered quickly ,
but some took significantly longer. Despite GPU accel-
eration, response times remain significantly high because
the model with 7 billion parameters requires substantial
computation. The GTX 1080 can handle it, but it is not
the best choice for large-scale inference. This limitation in
deployment plays a crucial role in the latency experienced,
and hence, the response time is longer. Also, the CPU-
based preprocessing and the vector searches take up more
time. Moreover, sequential processing lowers the usage of
the GPU.
F. Multimodal Performance Evaluation
The feedback form filled out by 52 trial participants
yielded a satisfaction score of 4.26/5 on average, indicating
improved access to academic resources. Besides that, we
also analyzed user satisfaction with the different query
modalities. The system managed to reach the average
score of 4.56/5 for text-only queries and 3.95/5 for image-
based queries. The average response time for image
queries was 16.4 seconds, about 4.6 seconds longer than
for text-only ones, which was caused by the overhead of
multimodal processing.
G. RAG Effectiveness Analysis
T o provide quantitative evidence of system effectiveness,
we evaluated the hallucination rate, retrieval accuracy ,
user satisfaction, and latency . Based on 160 domain-
specific queries during the trial phase, the baseline LLaV A-
1.5-7B model exhibited a hallucination rate of 31.7%,
whereas our RAG-enhanced model reduced this to 6.6%,
which is visually illustrated in Fig. 6. Hallucination
annotations were validated independently by two review-
ers, achieving a substantial inter-annotator agreement
with Cohen’s κ= 0.84, confirming the reliability of the
evaluation [ ?]. The semantic retriever achieved 78.3%

Figure 6: Demonstrating how our RAG-enhanced ap-
proach reduces baseline hallucination rate by about 25%.
top-1 and 92.1% top-3 accuracy . As the top-3 were used
in our system, retrieval accuracy remains high for most
university-related queries.
V. Data Priv acy, Secure Access Control, and
Compliance
Institutional documents and user queries are dealt with
under very strict security measures. Retrieval functions
involve only JWT-based authorization, and the entire
communication of the client and server is protected with
HTTPS. Sensitive data like emails or IDs are already
masked before logging, and uploaded images are processed
only in memory unless permission is granted. F urther-
more, very strict CORS policies are in place that allow
requests from only the trusted university domains. Not
only do these actions assure privacy , but they also comply
with the institutional data governance policies.
VI. Conclusions
The study shows the effectiveness of combining RAG
with multimodal inference to create a smart university
chat assistant that can handle both text and image queries.
By using semantic embeddings, vector storage, and re-
trieval pipelines, the system provides responses based on
context. This reduces the reliance on static knowledge
bases and helps avoid the hallucinations often seen in large
language models. The chat assistant is deployed as a
publicly available web application, demonstrating its prac-
tical use for various university stakeholders. This method
emphasizes how domain-specific, retrieval-augmented chat
assistants can make information more accessible, stream-
line administrative tasks, and help close digital gaps in
educational institutions, especially in developing areas.
Though our system answers most of the domain-specific
questions correctly , there are still several limitations, like
multihop reasoning, latency issues, etc. Sometimes we also
face the contextual integration problem, thus resulting ina very generic response. In the future, we would like
to extend our system to be more reliable using advanced
RAG variants like agentic RAG, multi-hop RAG, etc.
References
[1] C. McGrath, A. F arazouli, and T. Cerratto-Pargman, “Gener-
ative ai chatbots in higher education: A review of an emerging
research area,” Higher Education , vol. 89, no. 6, pp. 1533–1549,
2025.
[2] K. Peyton, S. Unnikrishnan, and B. Mulligan, “A review of
university chatbots for student support: F aqs and beyond,”
Discover Education , vol. 4, no. 1, p. 21, 2025.
[3] Z. Li, Z. W ang, W. W ang, K. Hung, H. Xie, and F. L. W ang,
“Retrieval-augmented generation for educational application:
A systematic survey ,” Computers and Education: Artificial
Intel ligence , p. 100417, 2025.
[4] B. R. Ranoliya, N. Raghuwanshi, and S. Singh, “Chatbot
for university related faqs,” in 2017 International Conference
on Advances in Computing, Communications and Informatics
(ICACCI) . IEEE, 2017, pp. 1525–1530.
[5] N. P . Patel, D. R. Parikh, D. A. Patel, and R. R. Patel,
“Ai and web-based human-like interactive university chatbot
(unibot),” in 2019 3rd International Conference on Electronics,
Communication and Aerospace T echnology (ICECA) . IEEE,
2019, pp. 148–150.
[6] T. T. Nguyen, A. D. Le, H. T. Hoang, and T. Nguyen, “Neu-
chatbot: Chatbot for admission of national economics univer-
sity ,” Computers and Education: Artificial Intel ligence , vol. 2,
p. 100036, 2021.
[7] A. Balderas, R. F. García-Mena, M. Huerta, N. Mora, and J. M.
Dodero, “Chatbot for communicating with university students
in emergency situation,” Heliyon , vol. 9, no. 9, 2023.
[8] G. Attigeri, A. Agrawal, and S. V. Kolekar, “Advanced nlp mod-
els for technical university information chatbots: Development
and comparative analysis,” IEEE Access , vol. 12, pp. 29 633–
29 647, 2024.
[9] F. Parrales-Bravo, R. Caicedo-Quiroz, J. Barzola-Monteses,
J. Guillén-Mirabá, and O. Guzmán-Bedor, “Csm: a chatbot
solution to manage student questions about payments and en-
rollment in university ,” IEEE Access , vol. 12, pp. 74 669–74 680,
2024.
[10] S. Neupane, E. Hossain, J. Keith, H. T ripathi, F. Ghiasi,
N. A. Golilarz, A. Amirlatifi, S. Mittal, and S. Rahimi, “F rom
questions to insightful answers: Building an informed chatbot
for university resources,” in 2024 IEEE F rontiers in Education
Conference (FIE) . IEEE, 2024, pp. 1–9.
[11] I. Han, H. Ji, S. Jin, and K. Choi, “Mobile-based artificial in-
telligence chatbot for self-regulated learning in a hybrid flipped
classroom,” Journal of Computing in Higher Education , pp. 1–
25, 2025.
[12] H. Liu, W. Xue, Y. Chen, D. Chen, X. Zhao, K. W ang, L. Hou,
R. Li, and W. Peng, “A survey on hallucination in large vision-
language models,” arXiv:2402.00253 , 2024.
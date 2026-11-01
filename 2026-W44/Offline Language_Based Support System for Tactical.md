# Offline Language-Based Support System for Tactical Decision-Making in Air Combat Operations

**Authors**: Joao P. A. Dantas

**Published**: 2026-10-26 15:46:03

**PDF URL**: [https://arxiv.org/pdf/2610.35635v1](https://arxiv.org/pdf/2610.35635v1)

## Abstract
An offline language-based support tool is presented to assist pilot training and tactical decision-making in air combat operations. The system, named MATE (Mission Assistant for Training and Evaluation), applies retrieval-augmented generation (RAG) to answer natural language questions using content extracted from doctrinal and tactical manuals. Built entirely with open-source components, it performs text preprocessing, embedding generation, semantic search, and language generation locally, ensuring full offline functionality. The tool was evaluated by three Brazilian Air Force fighter pilots, who rated the answers produced with three open-weight language models (Llama 3.1 8B, Qwen 2.5 14B, and Mistral NeMo 12B) to five tactical questions derived from a publicly available air combat manual. Average ratings ranged from 4.00 to 4.40 on a five-point scale, with Mistral NeMo 12B obtaining the highest score, and the pilots rated the response time of the tool as fully adequate compared to manual search. The relevance of the retrieved content was identified as the main point for improvement. Potential applications are discussed in onboard mission systems, flight simulators, and training environments.

## Full Text


<!-- PDF content starts -->

Offline Language-Based Support System for Tactical
Decision-Making in Air Combat Operations
Joao P. A. Dantas∗
Decision Support Systems Subdivision
Institute for Advanced Studies
São José dos Campos, São Paulo, Brazil, 12228-001
jpdantas@ita.br
Abstract
An offline language-based support tool is presented to assist pilot training and
tactical decision-making in air combat operations. The system, named MATE (Mis-
sion Assistant for Training and Evaluation), applies retrieval-augmented generation
(RAG) to answer natural language questions using content extracted from doctrinal
and tactical manuals. Built entirely with open-source components, it performs text
preprocessing, embedding generation, semantic search, and language generation
locally, ensuring full offline functionality. The tool was evaluated by three Brazilian
Air Force fighter pilots, who rated the answers produced with three open-weight
language models (Llama 3.1 8B, Qwen 2.5 14B, and Mistral NeMo 12B) to five
tactical questions derived from a publicly available air combat manual. Average
ratings ranged from 4.00 to 4.40 on a five-point scale, with Mistral NeMo 12B
obtaining the highest score, and the pilots rated the response time of the tool as fully
adequate compared to manual search. The relevance of the retrieved content was
identified as the main point for improvement. Potential applications are discussed
in onboard mission systems, flight simulators, and training environments.
1 Introduction
Air combat missions often require fast and reliable access to tactical information that is usually
found in lengthy and complex manuals. During planning or even in flight, pilots and mission
planners may need to consult this content quickly. However, doing so manually is often slow and
impractical, especially in environments where internet access is not available or allowed. The ability
to interact with technical documents in a natural and efficient way is increasingly valuable in defense
contexts [Doll et al., 2024].
Recent progress in natural language processing (NLP), especially with large language models (LLMs),
has made it possible to build systems that answer user questions in plain language using information
from documents [Brown et al., 2020, OpenAI, 2023]. However, most of these tools depend on cloud
services, which limits their use in sensitive or disconnected environments such as military aircraft or
secure simulators.
To address this gap, we present an offline language-based support tool that uses retrieval-augmented
generation (RAG) [Lewis et al., 2020] to provide helpful answers based on doctrinal manuals. The
system combines document search with a local language model, allowing users to ask questions in
natural language and receive relevant, context-aware answers. It is built entirely with open-source
tools and works fully offline, making it suitable for use in onboard systems, simulators, and training
environments. The system is modular and can be adapted to different types of documents.
∗www.joaopadantas.com
Proceedings of the Latinx in AI Workshop @ NeurIPS-25.
arXiv:2610.35635v1  [cs.CL]  26 Oct 2026

The main contribution of this work is the design and implementation of a tactical tool that runs
completely offline and helps users access sensitive documents that should not be processed online.
We show how it works using real examples from air combat manuals and explain how it can support
decision-making and training in air combat situations. We also report an expert validation in which
Brazilian Air Force fighter pilots rated the quality of the answers produced by the tool with three
different open-weight language models.
2 Related Work
NLP has been increasingly applied in military contexts to enhance decision-making, intelligence
analysis, and operational efficiency [Teže and Nazaruka, 2024]. Early initiatives, such as DARPA’s
TIPSTER program, aimed to improve information retrieval and extraction for defense applications,
laying the groundwork for subsequent advancements in military NLP systems [Defense Advanced
Research Projects Agency, 1991].
Recent developments have seen the integration of NLP in various military operations. For instance,
DARPA’s Transformative Apps (TransApps) program developed and rapidly fielded secure mobile
applications for deployed military units, showing how software tools can be delivered directly to
field operations [Defense Advanced Research Projects Agency, 2010]. Similarly, the MiTAP system,
developed by the Mitre Corporation, utilizes NLP to gather, translate, and present information for
monitoring global events, aiding in biosecurity and defense intelligence [Damianos et al., 2002].
In the field of intelligence analysis, NLP has been employed to process vast amounts of unstructured
data. Primer, an AI company, has developed NLP tools for the U.S. Special Operations Command to
analyze and identify disinformation campaigns, showcasing the role of NLP in modern information
warfare [Knight, 2020]. Additionally, the COA-GPT model employs large language models to
generate courses of action in military operations, highlighting the potential of NLP in strategic
planning [Goecks and Waytowich, 2024].
Applications of AI and language interfaces in air combat have also been explored. For example, the
Air Combat Evolution (ACE) program by DARPA aims to develop collaboration between AI and
human pilots, allowing pilots to act as “battle managers” for unmanned aircraft [Halpern, 2022].
Moreover, speech recognition systems have been implemented in advanced fighter aircraft, such as
the F-35 Lightning II, to allow pilots to manage subsystems such as communications and navigation
by voice, thereby reducing pilot workload [Schutte, 2007].
3 Methodology
The proposed system, MATE (Mission Assistant for Training and Evaluation), is designed to provide
offline, natural language access to tactical knowledge by combining document retrieval and generative
language modeling. It follows a modular architecture with five key elements: document indexing,
semantic retrieval, language model interaction, user interface, and offline deployment.
Document Indexing:The pipeline starts by preprocessing and indexing doctrinal and tactical man-
uals stored locally. Supported formats (.txt, .pdf, .docx) are converted into plain text and split into
overlapping segments of 1,200 characters, with an overlap of 250 characters between consecutive
segments. The multilingual paraphrase-multilingual-MiniLM-L12-v2 model from Sentence-
Transformers [Reimers and Gurevych, 2019, 2020] encodes each segment into a 384-dimensional
embedding. The embeddings are L2-normalized and stored in a FAISS inner-product index [Johnson
et al., 2021], so that the similarity search corresponds to cosine similarity. A metadata file links each
segment to its source document.
Semantic Retrieval:When a user submits a natural language question, the system computes its
embedding using the same model and performs a nearest-neighbor search over the FAISS index. The
top-kmost similar segments ( k= 5 ) are selected and concatenated, in order of similarity, into a textual
context limited to 4,000 characters, which is passed as input to the language model together with the
question. All retrieval logic is implemented in Python using the FAISS and SentenceTransformers
libraries.
Language Model Interaction:The context and user query are forwarded to a locally deployed
LLM via the Ollama API [Ollama, 2023], which handles model loading, memory management, and
2

inference entirely offline. Any open-weight model available in Ollama can be used; in this work,
the tool was evaluated with Llama 3.1 8B [Grattafiori et al., 2024], Qwen 2.5 14B [Qwen Team,
2024], and Mistral NeMo 12B [Mistral AI, 2024]. A FastAPI [Ramírez, 2018] backend handles
requests from the interface or external applications, builds the prompt from a system instruction, the
retrieved context, and the user’s question, and streams the generated response back to the interface.
In the evaluation reported in Section 4, the system instruction asked the model to answer directly and
concisely, based strictly on the reference chapter, using standard aviation terminology, and to state
explicitly when a question fell outside the scope of the chapter.
User Interface:The interface is built using Tkinter [Python Software Foundation, 2024], a
lightweight GUI library for Python. It allows users to type questions in natural language and
receive AI-generated answers in real time, with a selector for the installed language models, Ask,
Clear, and Stop controls, keyboard shortcuts, and error handling. All interactions with the backend
occur through HTTP requests to the local FastAPI server. An optional voice input module based on
Whisper [Radford et al., 2023] and PyAudio is also available, but it was disabled during the evaluation
reported in Section 4. Figure 1 shows a screenshot of the user interface, where the assistant answers a
tactical question using doctrinal content offline.
Figure 1: Screenshot of the MATE graphical interface.
Offline Deployment:All components are designed to run entirely offline, supporting use in air-
gapped environments. The embedding model, FAISS index, and the language model are all stored
locally. The system can be packaged using tools like PyInstaller [PyInstaller Development Team,
2025] to create a standalone executable, and no internet connection is required at runtime. This
ensures suitability for sensitive or classified military applications. Figure 2 summarizes the overall
architecture and data flow of the system, showing how user queries are processed through semantic
retrieval and local language generation.
4 Expert Validation
To assess whether the proposed system can support tactical decision-making, an evaluation was
conducted with Brazilian Air Force fighter pilots. As we cannot include or disclose classified military
documents in this study, the knowledge base was built from a publicly available reference: Chapter
5, “Section Tactics: Two-versus-One”, of Robert L. Shaw’s manual “Fighter Combat: Tactics and
Maneuvering” [Shaw, 1985]. This chapter provides comprehensive descriptions of section-level
tactics, two-versus-one engagements, and the tactical principles relevant to the evaluation.
3

Doctrinal
DocsIndexing
(Embeddings
+ FAISS)Semantic
SearchQuestionUser
Interface
Context +
QuestionLLM
(Local API)Answer
Figure 2: Overview of the offline tactical assistant system architecture.
Five questions were derived from the chapter to reflect typical queries a pilot or mission planner
might have. They cover the benefits of mutual support and the evolution of section doctrine, from the
Fighting Wing to the Double Attack and the Loose Deuce:
Q1:What is the primary advantage of flying as a section instead of a single fighter?
Q2:What are the main advantages of the Fighting Wing formation for new pilots?
Q3:What are the main weaknesses of the Fighting Wing doctrine?
Q4:What tactical innovation did the Double Attack introduce compared to Fighting Wing?
Q5:How does the Loose Deuce differ from the Double Attack in offensive engagement?
Each question was submitted to the tool three times, once with each of the evaluated models, using
the same indexed chapter and retrieval pipeline. The evaluation then followed three steps:
1)Participant Profile: Each pilot reported their operational experience, flight hours, and familiarity with
LLMs.
2)Tactical Evaluation: Each pilot rated the 15 answers (five questions, three models) on a 5-point Likert
scale [Boone and Boone, 2012], from 1 (very inadequate) to 5 (very adequate).
3)General Evaluation: Each pilot rated five tool-level aspects on the same scale and could leave free-text
suggestions.
The panel consisted of three fighter pilots with, on average, 6.7 years of operational experience
and 933 flight hours. They reported moderate familiarity with LLMs (occasional use and basic
knowledge), mostly through ChatGPT, Gemini, and Llama.
Table 1: Ratings of the answers to the tactical questions (mean ±standard deviation over three pilots,
1–5 scale).
Question Llama 3.1 8B Qwen 2.5 14B Mistral NeMo 12B
Q1 – Advantage of flying as a section 4.67±0.58 4.00±1.00 4.67±0.58
Q2 – Advantages of the Fighting Wing 4.67±0.58 4.33±0.58 4.67±0.58
Q3 – Weaknesses of the Fighting Wing 3.00±0.00 3.00±0.00 3.00±0.00
Q4 – Innovation of the Double Attack 3.33±0.58 4.00±1.00 4.67±0.58
Q5 – Loose Deuce vs. Double Attack 5.00±0.00 4.67±0.58 5.00±0.00
Average over questions 4.13 4.00 4.40
Table 1 shows the ratings of the answers to the tactical questions. All three models produced answers
rated as adequate for most questions, with averages between 4.00 and 4.40. Mistral NeMo 12B
obtained the highest average, matching or exceeding the other models in every question, while the
largest difference between models appeared in Q4, where Llama 3.1 8B received 3.33 and Mistral
NeMo 12B received 4.67. Q3, which asks about the weaknesses of the Fighting Wing doctrine,
received a rating of 3 from every pilot for every model. Since this pattern does not depend on the
model, it more likely reflects the content supplied by the retrieval step than the generation step itself.
4

Table 2: General evaluation of the tool (mean±standard deviation over three pilots, 1–5 scale).
Evaluation question Rating
Relevance of retrieval: did the information used in the answer correspond to the
correct content from the reference material?3.67±0.58
Conceptual and terminological accuracy: did the model use correct tactical
terms consistent with operational doctrine?4.33±1.15
Integration of retrieved context: did the model properly incorporate the re-
trieved text into its explanation?3.67±0.58
Clarity and coherence: was the answer clear, logical, and well structured? 4.33±0.58
Response time: was the generation time appropriate compared to manually
locating the information in the reference text?5.00±0.00
This interpretation is consistent with the general evaluation in Table 2. The aspects related to retrieval,
namely the relevance of the retrieved content and its integration into the answer, received the lowest
ratings (3.67), whereas terminological accuracy and clarity were rated 4.33. Response time received
the maximum rating from all pilots, indicating that querying the tool is faster than locating the
information manually in the reference text. One factor that may affect retrieval is the context limit:
with segments of 1,200 characters and a limit of 4,000 characters, only about three of the five retrieved
segments reach the model in full, so relevant passages ranked lower may be truncated or left out.
These results should be interpreted as exploratory. The panel was small, the questions covered a
single chapter of one manual, and the number of answers per model does not support statistical
comparisons between models or a meaningful analysis of inter-rater agreement [Koo and Li, 2016].
Nevertheless, they indicate that a fully offline RAG pipeline running on open-weight models can
produce answers that experienced fighter pilots consider adequate for most tactical questions, and
they point to retrieval as the main component to improve.
5 Conclusion and Future Work
This work presented MATE, an offline tactical assistant that combines document retrieval with a local
language model to support decision-making and training in air combat operations. The tool allows
users to ask questions in natural language and receive answers based on doctrinal manuals, even in
environments without internet access. It was built using open-source tools and follows a modular
architecture, making it suitable for integration into simulators and mission planning systems.
An evaluation with three Brazilian Air Force fighter pilots showed that the answers produced with
three open-weight models were rated as adequate for most tactical questions, with Mistral NeMo
12B obtaining the highest average rating, and that the response time of the tool was considered fully
adequate compared to manual search. The lowest ratings were associated with the relevance and
integration of the retrieved content.
Future work will focus on five main improvements:broader evaluation, with more pilots, more
questions, and additional tactical scenarios and manuals, enabling inter-rater agreement analysis;
retrieval quality, improving how manuals are segmented, how relevant passages are selected, and
how much context is passed to the model;response time, using more capable hardware to support
larger models;interactivity, adding dialogue history to support follow-up questions and integrating
the tool with simulation environments; andvoice interface, optimizing the voice module, which is
currently too heavy for routine use during flight or simulator-based training.
Offline tools like the one presented in this work may eventually help pilots access relevant tactical
knowledge during training, operational missions, or post-mission debriefings. Although still in its
early stages, this type of system has the potential to serve as a valuable support tool in simulators or
onboard aircraft, particularly in air combat operations that demand quick decision-making under high
pressure and without reliable connectivity.
5

Code Availability
The source code of MATE is publicly available at https://github.com/jpadantas/mate . The
repository includes the indexing, retrieval, backend, and interface components described in Section 3,
together with the configuration used in the evaluation. The reference manual used in the evaluation is
copyrighted and is not distributed with the code.
References
H. N. Boone and D. A. Boone. Analyzing Likert Data.The Journal of Extension, 50(2):Article 48,
2012. doi: 10.34068/joe.50.02.48.
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,
G. Sastry, A. Askell, et al. Language Models are Few-Shot Learners. InAdvances in Neural
Information Processing Systems, volume 33, pages 1877–1901, 2020. URL https://arxiv.org/
abs/2005.14165.
L. E. Damianos, J. M. Ponte, S. Wohlever, F. Reeder, D. S. Day, D. G. Wilson, and L. Hirschman.
MiTAP for Biosecurity: A Case Study.AI Magazine, 23(4):13–29, 2002.
Defense Advanced Research Projects Agency. TIPSTER Text Program. https://www-
nlpir.nist.gov/related_projects/tipster/, 1991. Accessed on 3 May 2025.
Defense Advanced Research Projects Agency. Transformative Apps (TransApps) Program. https:
//www.darpa.mil/research/programs/transformative-apps , 2010. Accessed on 3 May
2025.
T. Doll, D. Kallfass, D. Kunde, and M. Möbius. Natural Language AI for Military Decision Support
and Swarm Control for Autonomous UAS Trained in a Combat Simulation. InNATO STO
Meeting Proceedings STO-MP-MSG-207. NATO Science and Technology Organization, 2024.
ISBN 978-92-837-2491-9. URL https://www.sto.nato.int/document/natural-language-
ai-for-military-decision-support-and-swarm-control-for-autonomous-uas-
trained-in-a-combat-simulation/.
V . G. Goecks and N. Waytowich. COA-GPT: Generative Pre-trained Transformers for Accelerated
Course of Action Development in Military Operations.arXiv preprint arXiv:2402.01786, 2024.
URLhttps://arxiv.org/abs/2402.01786.
A. Grattafiori et al. The Llama 3 Herd of Models.arXiv preprint arXiv:2407.21783, 2024. URL
https://arxiv.org/abs/2407.21783.
S. Halpern. The Rise of A.I. Fighter Pilots.The New Yorker, Jan. 2022. URL https:
//www.newyorker.com/magazine/2022/01/24/the-rise-of-ai-fighter-pilots.
J. Johnson, M. Douze, and H. Jégou. Billion-scale similarity search with GPUs.IEEE Transactions
on Big Data, 7(3):535–547, 2021. doi: 10.1109/TBDATA.2019.2921572.
W. Knight. The AI Company Helping the Pentagon Assess Disinfo Campaigns.WIRED, Oct.
2020. URL https://www.wired.com/story/ai-helping-pentagon-assess-disinfo-
campaigns.
T. K. Koo and M. Y . Li. A Guideline of Selecting and Reporting Intraclass Correlation Co-
efficients for Reliability Research.Journal of Chiropractic Medicine, 15(2):155–163, 2016.
doi: 10 .1016/j .jcm.2016.02.012. URL https://www.sciencedirect.com/science/article/
pii/S1556370716000158.
P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih,
T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks. InAdvances in Neural Information Processing Systems, volume 33, pages 9459–9474,
2020. URLhttps://arxiv.org/abs/2005.11401.
Mistral AI. Mistral NeMo.https://mistral.ai/news/mistral-nemo, 2024.
6

Ollama. Ollama.https://ollama.com, 2023. Accessed on 3 May 2025.
OpenAI. GPT-4 Technical Report.arXiv preprint arXiv:2303.08774, 2023. URL https://
arxiv.org/abs/2303.08774.
PyInstaller Development Team. PyInstaller: Freeze Python Applications into Stand-Alone Executa-
bles.https://pyinstaller.org, 2025. Accessed on 3 May 2025.
Python Software Foundation. Tkinter Documentation. https://docs.python.org/3/library/
tkinter.html, 2024. Accessed on 3 May 2025.
Qwen Team. Qwen2.5 Technical Report.arXiv preprint arXiv:2412.15115, 2024. URL https:
//arxiv.org/abs/2412.15115.
A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever. Robust Speech
Recognition via Large-Scale Weak Supervision. InProceedings of the 40th International Con-
ference on Machine Learning, volume 202 ofProceedings of Machine Learning Research, pages
28492–28518, 2023.
S. Ramírez. FastAPI.https://fastapi.tiangolo.com, 2018. Accessed on 3 May 2025.
N. Reimers and I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-
Networks. InProceedings of the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 3982–3992. Association for Computational Linguistics, 2019. URL
https://aclanthology.org/D19-1410.
N. Reimers and I. Gurevych. Making Monolingual Sentence Embeddings Multilingual using Knowl-
edge Distillation. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 4512–4525. Association for Computational Linguistics,
2020. URLhttps://aclanthology.org/2020.emnlp-main.365.
J. Schutte. Researchers fine-tune F-35 pilot-aircraft speech system. Air Force Print News,
https://www.af.mil/News/Article-Display/Article/125449/researchers-fine-
tune-f-35-pilot-aircraft-speech-system/, 2007. Accessed on 3 May 2025.
R. L. Shaw.Fighter Combat: Tactics and Maneuvering. Naval Institute Press, Annapolis, MD, 1985.
V . Teže and ¯E. Nazaruka. Future Directions in Defence NLP: Investigating Research Gaps for
Low-Resource Languages. InDigital Business and Intelligent Systems (Baltic DB&IS 2024),
volume 2157 ofCommunications in Computer and Information Science, pages 93–105. Springer,
2024. doi: 10.1007/978-3-031-63543-4_7.
7
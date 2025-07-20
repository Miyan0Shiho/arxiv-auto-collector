# An Empirical Study of Multi-Agent RAG for Real-World University Admissions Counseling

**Authors**: Anh Nguyen-Duc, Chien Vu Manh, Bao Anh Tran, Viet Phuong Ngo, Luan Le Chi, Anh Quang Nguyen

**Published**: 2025-07-15 12:49:42

**PDF URL**: [http://arxiv.org/pdf/2507.11272v1](http://arxiv.org/pdf/2507.11272v1)

## Abstract
This paper presents MARAUS (Multi-Agent and Retrieval-Augmented University
Admission System), a real-world deployment of a conversational AI platform for
higher education admissions counseling in Vietnam. While large language models
(LLMs) offer potential for automating advisory tasks, most existing solutions
remain limited to prototypes or synthetic benchmarks. MARAUS addresses this gap
by combining hybrid retrieval, multi-agent orchestration, and LLM-based
generation into a system tailored for real-world university admissions. In
collaboration with the University of Transport Technology (UTT) in Hanoi, we
conducted a two-phase study involving technical development and real-world
evaluation. MARAUS processed over 6,000 actual user interactions, spanning six
categories of queries. Results show substantial improvements over LLM-only
baselines: on average 92 percent accuracy, hallucination rates reduced from 15
precent to 1.45 percent, and average response times below 4 seconds. The system
operated cost-effectively, with a two-week deployment cost of 11.58 USD using
GPT-4o mini. This work provides actionable insights for the deployment of
agentic RAG systems in low-resource educational settings.

## Full Text


<!-- PDF content starts -->

An Empirical Study of Multi-Agent RAG for
Real-World University Admissions Counseling
Anh Nguyen-Duc1,2, Chien Vu Manh2, Bao Anh Tran2, Viet Phuong Ngo2,
Luan Le Chi2, and Anh Quang Nguyen3
1University of South-Eastern Norway, Bo I Telemark, Norway
angu@usn.no
2University of Transport Technology, Hanoi, Vietnam
vumanhchien101@gmail.com,anhtranms03@gmail.com,phuong.nv147@gmail.com,luanlc@utt.edu.vn
3University of Transport and Communications, Hanoi, Vietnam
anhnq@utc.edu.vn
Abstract. This paper presents MARAUS (Multi-Agent and Retrieval-
Augmented University Admission System), a real-world deployment of
a conversational AI platform for higher education admissions counsel-
ing in Vietnam. While large language models (LLMs) offer potential for
automating advisory tasks, most existing solutions remain limited to
prototypes or synthetic benchmarks. MARAUS addresses this gap by
combining hybrid retrieval, multi-agent orchestration, and LLM-based
generation into a system tailored for real-world university admissions.
In collaboration with the University of Transport Technology (UTT) in
Hanoi, we conducted a two-phase study involving technical development
and real-world evaluation. MARAUS processed over 6,000 actual user
interactions, spanning six categories of queries. Results show substantial
improvements over LLM-only baselines: on average 92% accuracy, hallu-
cination rates reduced from 15% to 1.45%, and average response times
below 4 seconds. The system operated cost-effectively, with a two-week
deployment cost of 11.58 USD using GPT-4o mini. This work provides
actionable insights for the deployment of agentic RAG systems in low-
resource educational settings.
Keywords: RAG ·Multi Agent Systems ·hybrid RAG ·admission
counseling
1 Introduction
The rapid advancement of generative AI (GenAI), especially Large Language
Models (LLMs), is transforming multiple sectors—ranging from healthcare to fi-
nanceandpublicadministration.Thelastyearshavewitnessedanunprecedented
proliferation of GenAI models with the potential to transform educational prac-
tice [14,8,1,5]. Automated feedback, personalised tutoring, content generation,
and conversational search are now technically feasible at scale [1]. Yet the em-
pirical literature has not kept pace with the release cadence of new foundation
models: most claims of effectiveness rest on small demonstrations or syntheticarXiv:2507.11272v1  [cs.SE]  15 Jul 2025

2 Anh Nguyen-Duc et al.
benchmarksratherthancontrolledstudieswithauthenticeducationaltasks.Em-
pirical can fill this gap by providing evidence-based design knowledge —replicable
artefacts, datasets, and evaluations—that inform both practitioners and future
model builders.
Admission tasks at higher education is a manual-intensive activity with thou-
sands of high-stakes enquiries into a short time window. Applicants and parents
askaboutcut-offscores,quotas,scholarshipcriteria,ortheimpactofspecialpoli-
cies such as regional bonuses and ethnic-minority incentives. A typical composite
question looks like: “I have 23 transcript points, I belong to Region 1 (mountain-
ous) and priority group 2. How many points do I actually have for admission?” .
Answering correctly requires chaining multiple institutional rules, year-specific
cut-off tables, and applicant metadata—not merely general knowledge. Manual
hotlines and email desks are slow and error-prone; web FAQs cannot cover the
long-tail of personalised scenarios. LLMs could, in principle, deliver instant, di-
alogic explanations, but off-the-shelf systems (e.g., public ChatGPT or Claude)
tend to hallucinate, confuse policy versions, or overlook language-specific details,
especially in low-resource languages such as Vietnamese or Thai.
Retrieval-Augmented Generation (RAG) pipelines reduce hallucinations in
LLMs by grounding responses in externally retrieved knowledge [4]. Recent ad-
vancements enhance this architecture through cross-encoder re-ranking, graph-
structured retrieval, and dynamic prompt optimization [6,7]. Building on these
foundations, multi-agent LLM systems are emerging as a promising direction
for complex reasoning tasks such as university admission counseling. In such
systems, distinct agents can specialize in subtasks—e.g., retrieval refinement,
regulatory verification, or user context modeling—while coordinating via shared
memory or structured prompts to produce coherent answers.
Despite their potential, several open challenges remain. First, domain speci-
ficity is a critical concern: it is unclear whether hybrid retrieval combined with
multi-agent coordination can consistently extract the correct institutional poli-
cies or regulatory clauses required to address nuanced, context-sensitive queries.
Second,languagerepresentationremainsabottleneck,particularlyforlow-resource
settings. It is uncertain whether Vietnamese-specific or multilingual embedding
models offer sufficient semantic fidelity compared to their English-centric coun-
terparts. Finally, prompt engineering strategies—such as few-shot exemplars,
chain-of-thought reasoning, or delegated verifier agents—require further empir-
ical validation to assess their effectiveness in minimizing hallucinations while
preserving concise, trustworthy outputs in real-world deployments.
In this paper, we present MARAUS, a lightweight, domain-specific QA plat-
form tailored for university admission scenarios. Unlike general-purpose LLMs
such as ChatGPT, MARUAS can handle specialized queries that require domain
logic, score calculations, and local data access. Our experimental results demon-
strate that MARUAS outperforms existing LLM-based QA systems in both pre-
cision and hallucination control. Furthermore, MARUAS has been deployed in
a real-world university admission setting, validating its practical applicability.

Title Suppressed Due to Excessive Length 3
2 Background
This section reviews key concepts underpinning the design of our system, fo-
cusing on recent advances in LLM-based Q&A educational systems and recent
RAG techniques.
2.1 Educational Q&A systems
In the admissions context, RAG-based systems have been developed to manage
diverse institutional queries, enhancing both the precision and responsiveness of
university application support. For instance, one implementation that combines
GPT-3.5 with LlamaIndex has demonstrated superior performance over conven-
tional FAQ systems, particularly in addressing complex student inquiries [3].
Similarly, HICON AI adopts a RAG-based approach to deliver tailored college
admissions advice by segmenting applicants into advisory profiles and incorpo-
rating resume analysis for individualized suggestions [12]. In Vietnamese admis-
sion context, Bui et al. consolidate educational FAQs, curricular resources, and
data from learning management systems (LMS) into a cohesive knowledge graph
[2]. This graph-driven design enhances both intent detection and policy align-
ment. As shown in Table 1, our proposed approach MARAUS adopting novel
strategies basing on current available GPT models with superior performance
comparing to existing studies.
Reference Architecture Retrieval LLM Model
Bui et al. 2024 [2] Knowledge Graph Semantic Similarity URA
Singla et al. 2024 [12] Singular RAG Semantic Similarity LlaMA2
Ehrenthal et al., 2024 Unknown Singular RAG PaLM 2 for Text
Z. Chen et al. 2024 [3] Singular RAG Semantic Similarity Text-davinci-003, GPT-3.5-
Turbo
This study Multi-agents Keyword, semantic
and Re-rank with
LLMGPT-4o mini
Table 1. Overview of Retrieval-Augmented Generation configurations used in recent
studies.
2.2 Retrieval Strategies for RAG Systems
Retrieval forms the backbone of RAG systems, providing the contextual infor-
mation that language models use to generate relevant and accurate responses.
Broadly, retrieval approaches fall into three categories: keyword-based, semantic
vector-based, and hybrid methods.
Keyword-based retrieval, such as BM25 [10], ranks documents based on term
frequency and exact matches. These methods are efficient, interpretable, and
widely used in production through engines like Lucene and ElasticSearch. How-
ever, they often fail when queries use synonyms, informal language, or region-
specific phrasing—as is common in Vietnamese higher education contexts.

4 Anh Nguyen-Duc et al.
Semantic retrieval methods address these limitations by embedding text into
dense vector spaces using pretrained models like SBERT [9] and MPNet [13].
Tools like FAISS perform fast approximate nearest-neighbor search in this space,
enabling robust retrieval for paraphrased or imprecise queries. Nonetheless, se-
mantic similarity alone can retrieve passages that are topically related but con-
textually irrelevant.
To balance precision and recall, hybrid retrieval strategies combine both lex-
ical and semantic signals. In our system, top- kresults are retrieved using both
FAISS and BM25, then re-ranked using a GPT cross-encoder. This cross-encoder
jointlyprocessesthequeryandeachcandidatepassagetoassignrelevancescores,
improving the accuracy of context selection. This approach is particularly ef-
fective in disambiguating overlapping programme names or policy terms, and
ensures that retrieved content is both semantically aligned and task-relevant.
3 Research Approach
3.1 Research Design
We employ an in-depth single–case study design [11]. Case studies are regarded
as the most suitable empirical strategy when (i) the phenomenon under inves-
tigation cannot be separated from its real-world context, and (ii) the goal is
to build rich, explanatory theory rather than derive statistical generalisations.
Our objective is to understand howandto what extent RAG grounded in Large
Language Models (LLMs) can support real-world admission activities. Data col-
lection was between Jan 2025 and July 2025 and conducted in two phases:
–Phase 1 - Technological experimentation: the admission workflow was ex-
plored, requirements were collected, datasets were collected, models were
experimented with and technologically optimized.
–Phase 2 - Process experimentation: A prototype of AI conversational system
was built and adopted by real users. Evaluation metrics were collected in 2
weeks.
3.2 Our case
University of Transport Technology (UTT) in Hanoi, Vietnam, has several
characteristics for our case study. The university handles over 10.000applicant
enquiries annually across peak periods, typical for medium-sized public univer-
sities in Vietnam. They have large volume of digital conversations - via three
channels, Facebook, Zalo and Direct contact, there are more than 4.5 GB of
conversational data since 2022. Most importantly, UTT’s senior leadership com-
mitted to providing sustained access to operational logs, historical queries, and
key admission personnel throughout the design and evaluation phases.
The existing enquiry-handling process at UTT is decentralised, manually
intensive, and distributed across multiple communication channels. It unfolds
as follows: (1) static publication, admission information are first published on

Title Suppressed Due to Excessive Length 5
the university’s website, Facebook page, and printed leaflets. This information
is updated annually, often within tight timelines dictated by the Ministry of
Education and Training (MOET), (2) query intake. Prospective students and
parents submit questions through five main channels: (i) telephone hotline, (ii)
university e-mail, (iii) Facebook Messenger, (iv) on-campus consultation booths,
and (v) informal messaging platforms (e.g., Zalo, SMS), (3) Manual triage and
assignment. A limited number of admission officers—typically fewer than 10 dur-
ing peak weeks—monitor all incoming messages. Questions are manually cate-
gorised, sometimes assigned to relevant departments (e.g., financial aid), and
queued for response. Duplication is high: some FAQs (e.g., cut-off scores, tu-
ition fees) appear hundreds of times in a single day. (4) - Knowledge transfer,
officers consult a shared internal Google Drive folder or archived email chains
for reference documents, then draft individual responses. In practice, replies are
often copy-pasted or rephrased. However, inconsistencies frequently arise due to
non-synchronised document versions or incomplete updates.
Despite genuine efforts by staff, the current workflow suffers from three struc-
tural bottlenecks: (1) high-volume repetition of FAQ-type queries, (2) lack of
integrated knowledge management, and (3) inability to personalise responses to
individual applicants. These factors make UTT an ideal site to assess whether
RAG systems can meaningfully improve both the accuracy and responsiveness
of admission counselling under realistic operational constraints.
3.3 Threats to Validity
As with any single-case empirical study, several threats to validity must be con-
sidered [11]. One key threat lies in how chatbot performance is operationalised.
We provided standard experimental metrics, i.e. accuracy, response time and
user rate with real-user data by recording the number of correct and incorrect
responses observed during actual interactions. Another limitation concerns the
generalisability of the findings, as the study was conducted within a single in-
stitution. Institutional practices related to admissions, data governance, and IT
infrastructure vary considerably across contexts. However, UTT’s characteris-
tics—a mid-sized public university with decentralised communication and lim-
ited infrastructure—are broadly representative of many Vietnamese and Asian/
African higher education institutions. We describe UTT’s specific workflow in
Section 3.2. to support analytical generalization. Reproducibility also presents a
potential threat. To mitigate this, we have documented all configuration param-
eters in detail and publicly released our codebase and test dataset. This enables
other researchers and practitioners to replicate or adapt our work in similar
environments.
4 MARAUS development
This section presents the design, implementation, and experimental evaluation
of MARAUS (Multi-Agent and Retrieval Augmented University Admission Sys-
tem), a hybrid Retrieval-Augmented Generation (RAG) platform for handling

6 Anh Nguyen-Duc et al.
diverse user queries in university admission contexts. MARAUS integrates multi-
agent orchestration, semantic retrieval, hybrid re-ranking, and large language
model (LLM) generation, with a focus on precision, interpretability, and hallu-
cination control.
4.1 System architecture
The core of MARAUS is a multi-agent coordinator that classifies incoming
queries into four distinct processing pipelines (Figure 1):
–Information search agent: Executes keyword-based and semantic retrieval for
informational queries (e.g., program details).
–Scorecalculationagent:Handlesnumericcomputationforscore-relatedqueries,
extractingstructuredattributestocomputetranscriptpoints,prioritybonuses,
and total eligibility scores.
–Recommendation agent: for queries predicting program eligibility, such as
“With XX transcript points, can I pass [program]?” or “What programs can
I pass with XX exam points?”, the system extracts information
–General Query agent: Applies a fallback hybrid RAG strategy when query
classification confidence is low
Fig. 1.An overview of the MARAUS system
4.2 Pre-processing
All textual data undergoes rigorous preprocessing. We remove boilerplate con-
tent, HTML artifacts, and near-duplicates using a Jaccard similarity filter ( <
0.9). Text normalization includes lowercasing, diacritic normalization, and tok-
enization via VnCoreNLP . Personally identifiable information (PII) is redacted
using regular expression patterns targeting phone numbers, emails, and national

Title Suppressed Due to Excessive Length 7
IDs. Additionally, entropy-based filters detect and exclude nonsensical input or
prompt-injection attempts.
Document corpora are segmented into 8.412overlapping context windows
(500-tokensize,100-tokenstride),embeddedwiththe XAI-encode-all-mpnet-base-v2
model ( 768dimensions, 28 ms/chunk on CPU). The resulting vectors are stored
in a FAISS IndexFlatIP , occupying 565 MBof RAM. Full index construction
completes in 41 son an 8-core machine.
The training set includes 1,376 FAQ pairs and98 document chunks (ap-
prox. 600–700 tokens per chunk), curated from UTT’s internal systems (Mon-
goDB, Google Drive). Chunk size was empirically tuned to balance context re-
tention against LLM token constraints.
4.3 Hybrid Retrieval in MARAUS
We employ a hybrid RAG pipeline, combining dense vector retrieval and sparse
keyword search.
Semantic Retrieval . At runtime, user queries are embedded into a 768-
dimensional vector using Xenova/all-mpnet-base-v2 . FAISS retrieves the top-
k(k=15) similar FAQ entries and the top document chunk based on cosine
similarity. A filtering threshold of 0.9 ensures high semantic relevance.
Keyword Retrieval and Fusion . To handle spelling variations and rare
keyword queries, an ElasticSearch 8.11 BM25 index runs in parallel. The union
of BM25 and FAISS outputs is passed to a hybrid re-ranking stage.
Re-Ranking with LLM . We deploy GPT-4o mini as a zero-shot cross-
encoder for relevance scoring. Each candidate passage is concatenated with the
user query in a structured prompt, and GPT-4o mini assigns a normalized rele-
vance score [0,1]. The top-2 passages are retained. This step reduces false posi-
tives by 38 %compared to raw FAISS retrieval.
4.4 Post processing
The final prompt comprises of (1) an instruction block discouraging speculation,
(2) the two top-ranked passages, explicitly cited by ID, and (3) the user query.
Output text generation is performed using GPT-4o with temperature=0.7 ,top
p=0.9,anda max tokens=350 limit.Astreaminginterfacereturnsthefirsttoken
within 350 mson average. HallucinationMitigation . Acustom post-processor
enforcescitationintegrity.Ifgeneratedanswerslackatleastonepassagecitation,
they are discarded and regenerated with penalized decoding parameters. This
mechanism reduces hallucination rates from 15 %(LLM-only) to 1.45 %in the
hybrid pipeline.
4.5 Experimental Evaluation
We evaluate MARAUS across multiple configurations: LLM-only, RAG+Re-
rank,andHybridRAG.Metricsincludeprecision,recall,F1-score,responsetime,

8 Anh Nguyen-Duc et al.
and hallucination rate. All experiments were conducted on an 8-core Intel Xeon
machine with 32 GBRAM. The experiments were on a set of 100 pairs of ques-
tions and answers with grounded truth provided by UTT. As shown in Table
2, the Hybrid RAG pipeline outperforms baselines across all metrics, achieving
near-perfect precision and significantly reducing hallucination rates, while main-
taining sub-4-second response times. These results demonstrate the effectiveness
of multi-agent coordination, hybrid retrieval, and LLM-controlled re-ranking in
domain-specific QA systems, helping us to define the retrieval approach for MA-
RAUS
Table 2. MARAUS Experimental Results Summary
Metric LLM Only RAG+Re-rank Hybrid RAG
Precision 0.70 0.90 0.985
Recall 0.65 0.85 0.89
F1-score 0.67 0.87 0.91
Response Time (s) 7.0 4.0 3.75
Hallucination Rate 15% 6% 1.45%
Key Observations
KO1– Multi-agent RAG pipelines can significantly enhance the reliabil-
ity and factual grounding of LLMs in complex, real-world advisory tasks,
reducing hallucination while maintaining high accuracy.
5 Result
This section presents the data extracted from two weeks publishing MARAUS,
evaluation metrics and our findings.
5.1 Data
Data was collected from 2-week running of the Q & A system for UTT during
summer2025.Therearetypicallysixtypesofuserquestions,reflectingincreasing
levels of complexity and interaction. The first type is Simple Keyword Retrieval,
where the system extracts direct answers from FAQs or databases based on key-
words, such as "Does the university offer medical programs?". The second type
involves Intent and Entity Recognition, where the chatbot identifies the cate-
gory of the question and extracts key details like majors, scores, or locations, for
example: "Is 25 points enough for Computer Science?". The third type is An-
swer Generation, where the chatbot generates new responses for questions not
directly covered in the knowledge base, such as: "Give me a brief introduction
to Artificial Intelligence." The fourth type is Logical Reasoning and Calculation,

Title Suppressed Due to Excessive Length 9
where the system performs tasks like calculating admission scores or applying
priority points, for example: "I have 25.5 points from my high school record, in
a KV1 area, can I get into Computer Science?" The fifth type is Personalization
and Multi-turn Conversation, where the chatbot maintains context across mul-
tiple turns, such as remembering a user mentioned they are from Quang Tri in
one message and responding appropriately to follow-up questions about priority
scoring. Finally, the sixth type is Handling Ambiguous or Subjective Questions,
which requires the chatbot to analyze user sentiment, social context, and implied
goals, such as: "Which major is suitable for introverts?". The total of 6079 pairs
of Questions and Answers were manually inspected by coauthors of this paper.
5.2 Evaluation metrics
We used the following metrics to evaluate MARAUS:
–Number of questions (items) per day: Each item refers to a pair of question
and answer extracted from the MARAUS chat logs.
–Number of tokens per day: The total number of input and output tokens
recorded from the OpenAI platform.
–Accuracy: The ratio of correct answers to the total number of questions. A
correct answer is defined as one that provides neither incorrect information
nor irrelevant responses to user questions.
–Perceived user satisfaction: The rating given by university staff members
regarding the chatbot’s performance.
5.3 Findings
Number of token used ranges from 350k to 1.4mil token every day. With the
average input token per message is 1748 tokens and average output token per
message is 60 tokens. The cost for two weeks actual running of the bot is 11.58
USD with GPT-4o mini or 57.90 USD with GPT-4o.
As shown in Figure 2, the analysis of MARAUS chatbot interactions over
the observed period shows a high proportion of correct answers across all days,
with daily accuracy consistently ranging from 87% to 94%. The total number of
questions per day varies significantly, from as few as 180 interactions on 05/07
to as many as 797 interactions on 24/06. On average the accuracy is 92%. The
distribution of wrong answers are different across categories of questions relating
Personalization and Multi-turn conversation and Handling ambiguous and sub-
jective questions. There are also many wrong answers due to a misunderstanding
of the questions by users when the use of language is vague or inaccurate.
User satisfaction averaged 4.5/5 across 7 admission officers and test users.
Officers reported a significant reduction in repetitive question handling and
favoured the explainability provided by chunk citations. The officers also appre-
ciated the personalised responses tailored to score ranges and programm prefer-
ences.

10 Anh Nguyen-Duc et al.
Fig. 2.Systems’ accuracy
Key Observations
KO2–Domain-specializedorchestrationthatintegratesretrieval,reason-
ing, and personalized interaction is essential for addressing diverse user
intents beyond standard FAQ matching in practical deployments
6 Discussion
Recent studies have explored various approaches to enhance the accuracy and
relevance of AI chatbots using RAG techniques combined with large language
models (LLMs). For instance, Singla et al. developed the HiCON chatbot, in-
tegrating Meta Llama 2 with RAG to handle four types of user questions [12].
Their system achieved an average accuracy of 80%, demonstrating the potential
of combining LLMs with retrieval pipelines to improve response quality. Sim-
ilarly, Chen et al. proposed a system based on ChatGPT-3.5 with RAG using
LlamaIndex [3]. Their approach significantly improved the baseline performance,
increasing the average accuracy from 41.4% (ChatGPT-3.5 alone) to 89.5% with
their RAG-enhanced chatbot, and achieving a peak accuracy of 94.7% in specific
scenarios.
Within the Vietnamese context, Bui et al. explored a knowledge graph-based
approach for question answering in the education domain. While promising in

Title Suppressed Due to Excessive Length 11
concept, their work remains at a preliminary stage, with limited experimental
data and no large-scale deployment or stress testing reported.In contrast, our
proposedsystemoffersacost-effectiveandscalablesolution.Byleveragingmulti-
agent strategies and off-the-shelf LLM components combined with RAG, our
approach enables feasible real-world adoption without the overhead of maintain-
ing complex knowledge graphs. Furthermore, the system is tested with a larger
and more diverse dataset, covering six categories of user questions—including
reasoning, personalization, and handling subjective queries—thereby increasing
its applicability in real-world university counselling services.
7 Conclusion
This study presents MARAUS, an operational, multi-agent RAG-based chat-
bot designed for university admissions counseling. Unlike prior studies with
small-scale validation, MARAUS was tested in a real-world deployment, engag-
ing directly with thousands of applicants and university staff over an intensive
admissions period. Our findings show that MARAUS outperforms traditional
LLM-only approaches and prior RAG-based systems. The integration of hybrid
retrieval, LLM-based re-ranking, and multi-agent task specialization contributed
significantly to these outcomes. The study also highlights practical considera-
tions for deployment in low-resource language settings, where domain-specific
knowledge, policy variations, and language nuances pose additional challenges.
Future work will extend MARAUS to other universities and explore broader
domains such as financial aid advising or career counseling. We also plan to
refine the system’s personalization capabilities and investigate adaptive learning
from continuous user feedback to further enhance long-term performance.
References
1. Alasadi, E.A., Baiz, C.R.: Generative AI in Education and Research:
Opportunities, Concerns, and Solutions. Journal of Chemical Education
100(8), 2965–2971 (Aug 2023). https://doi.org/10.1021/acs.jchemed.3c00323,
https://doi.org/10.1021/acs.jchemed.3c00323, publisher: American Chemical So-
ciety
2. Bui, T., Tran, O., Nguyen, P., Ho, B., Nguyen, L., Bui, T., Quan,
T.: Cross-Data Knowledge Graph Construction for LLM-enabled Educa-
tional Question-Answering System: A Case Study at HCMUT. In: Proceed-
ings of the 1st ACM Workshop on AI-Powered Q&A Systems for Mul-
timedia. pp. 36–43. AIQAM ’24, Association for Computing Machinery,
New York, NY, USA (Jun 2024). https://doi.org/10.1145/3643479.3662055,
https://doi.org/10.1145/3643479.3662055
3. Chen, Z., Zou, D., Xie, H., Lou, H., Pang, Z.: Facilitating university ad-
mission using a chatbot based on large language models with retrieval-
augmented generation. Educational Technology & Society 27(4), 454–470 (2024),
https://www.jstor.org/stable/48791566, publisher: International Forum of Educa-
tional Technology & Society, National Taiwan Normal University, Taiwan

12 Anh Nguyen-Duc et al.
4. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Proceedings of the 34th Inter-
national Conference on Neural Information Processing Systems. pp. 9459–9474.
NIPS ’20, Curran Associates Inc., Red Hook, NY, USA (Dec 2020)
5. Li, Z., Wang, Z., Wang, W., Hung, K., Xie, H., Wang, F.L.:
Retrieval-augmented generation for educational application: A sys-
tematic survey. Computers and Education: Artificial Intelligence
8, 100417 (Jun 2025). https://doi.org/10.1016/j.caeai.2025.100417,
https://www.sciencedirect.com/science/article/pii/S2666920X25000578
6. Mavromatis, C., Karypis, G.: GNN-RAG: Graph Neural Retrieval for Large Lan-
guage Model Reasoning (May 2024). https://doi.org/10.48550/arXiv.2405.20139,
http://arxiv.org/abs/2405.20139, arXiv:2405.20139 [cs]
7. Mishra, P., Mahakali, A., Venkataraman, P.S.: SEARCHD - Advanced
Retrieval with Text Generation using Large Language Models and
Cross Encoding Re-ranking. In: 2024 IEEE 20th International Con-
ference on Automation Science and Engineering (CASE). pp. 975–
980 (Aug 2024). https://doi.org/10.1109/CASE59546.2024.10711642,
https://ieeexplore.ieee.org/abstract/document/10711642, iSSN: 2161-8089
8. Mittal, U., Sai, S., Chamola, V., Sangwan, D.: A Comprehen-
sive Review on Generative AI for Education. IEEE Access 12,
142733–142759 (2024). https://doi.org/10.1109/ACCESS.2024.3468368,
https://ieeexplore.ieee.org/abstract/document/10695056
9. Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks (Aug 2019). https://doi.org/10.48550/arXiv.1908.10084,
http://arxiv.org/abs/1908.10084, arXiv:1908.10084 [cs]
10. Robertson, S., Zaragoza, H.: The Probabilistic Relevance Framework:
BM25 and Beyond. Foundations and Trends ®in Information Re-
trieval 3(4), 333–389 (Dec 2009). https://doi.org/10.1561/1500000019,
https://www.nowpublishers.com/article/Details/INR-019, publisher: Now Pub-
lishers, Inc.
11. Runeson, P., Höst, M.: Guidelines for conducting and reporting case study research
in software engineering. Empirical Software Engineering 14(2), 131 (Dec 2008).
https://doi.org/10.1007/s10664-008-9102-8, https://doi.org/10.1007/s10664-008-
9102-8, number: 2
12. Singla, A.D., Tripathi, S., Victoria, A.H.: HICON AI: Higher Ed-
ucation Counseling Bot. In: 2024 4th International Conference on
Pervasive Computing and Social Networking (ICPCSN). pp. 779–
784 (May 2024). https://doi.org/10.1109/ICPCSN62568.2024.00131,
https://ieeexplore.ieee.org/document/10607729
13. Song, K., Tan, X., Qin, T., Lu, J., Liu, T.Y.: MPNet: Masked and Permuted
Pre-training for Language Understanding. In: Advances in Neural Information
Processing Systems. vol. 33, pp. 16857–16867. Curran Associates, Inc. (2020),
https://proceedings.neurips.cc/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-
Abstract.html
14. Su, J., Yang, W.: Unlocking the Power of ChatGPT: A Framework
for Applying Generative AI in Education. ECNU Review of Educa-
tion 6(3), 355–366 (Aug 2023). https://doi.org/10.1177/20965311231168423,
https://doi.org/10.1177/20965311231168423, publisher: SAGE Publications Ltd
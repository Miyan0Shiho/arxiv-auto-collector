# A Survey of Personalization: From RAG to Agent

**Authors**: Xiaopeng Li, Pengyue Jia, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng Du, Xiangyang Li, Yong Liu, Huifeng Guo, Ruiming Tang, Xiangyu Zhao

**Published**: 2025-04-14 11:57:52

**PDF URL**: [http://arxiv.org/pdf/2504.10147v1](http://arxiv.org/pdf/2504.10147v1)

## Abstract
Personalization has become an essential capability in modern AI systems,
enabling customized interactions that align with individual user preferences,
contexts, and goals. Recent research has increasingly concentrated on
Retrieval-Augmented Generation (RAG) frameworks and their evolution into more
advanced agent-based architectures within personalized settings to enhance user
satisfaction. Building on this foundation, this survey systematically examines
personalization across the three core stages of RAG: pre-retrieval, retrieval,
and generation. Beyond RAG, we further extend its capabilities into the realm
of Personalized LLM-based Agents, which enhance traditional RAG systems with
agentic functionalities, including user understanding, personalized planning
and execution, and dynamic generation. For both personalization in RAG and
agent-based personalization, we provide formal definitions, conduct a
comprehensive review of recent literature, and summarize key datasets and
evaluation metrics. Additionally, we discuss fundamental challenges,
limitations, and promising research directions in this evolving field. Relevant
papers and resources are continuously updated at
https://github.com/Applied-Machine-Learning-Lab/Awesome-Personalized-RAG-Agent.

## Full Text


<!-- PDF content starts -->

A Survey of Personalization: From RAG to Agent
XIAOPENG LI∗,City University of Hong Kong, Hong Kong
PENGYUE JIA∗,City University of Hong Kong, Hong Kong
DERONG XU, City University of Hong Kong, Hong Kong and University of Science and Technology of China, China
YI WEN, City University of Hong Kong, Hong Kong
YINGYI ZHANG, City University of Hong Kong, Hong Kong and Dalian University of Technology, China
WENLIN ZHANG, City University of Hong Kong, Hong Kong
WANYU WANG, City University of Hong Kong, Hong Kong
YICHAO WANG, Noah’s Ark Lab, Huawei, China
ZHAOCHENG DU, Noah’s Ark Lab, Huawei, China
XIANGYANG LI, Noah’s Ark Lab, Huawei, China
YONG LIU, Noah’s Ark Lab, Huawei, Singapore
HUIFENG GUO, Noah’s Ark Lab, Huawei, China
RUIMING TANG†,Noah’s Ark Lab, Huawei, China
XIANGYU ZHAO†,City University of Hong Kong, Hong Kong
Personalization has become an essential capability in modern AI systems, enabling customized interactions that align with individual
user preferences, contexts, and goals. Recent research has increasingly concentrated on Retrieval-Augmented Generation (RAG)
frameworks and their evolution into more advanced agent-based architectures within personalized settings to enhance user satisfaction.
Building on this foundation, this survey systematically examines personalization across the three core stages of RAG: pre-retrieval,
retrieval, and generation. Beyond RAG, we further extend its capabilities into the realm of Personalized LLM-based Agents, which
enhance traditional RAG systems with agentic functionalities, including user understanding, personalized planning and execution,
and dynamic generation. For both personalization in RAG and agent-based personalization, we provide formal definitions, conduct a
comprehensive review of recent literature, and summarize key datasets and evaluation metrics. Additionally, we discuss fundamental
challenges, limitations, and promising research directions in this evolving field. Relevant papers and resources are continuously
updated at the Github Repo1.
CCS Concepts: •Information systems →Personalization .
Additional Key Words and Phrases: Large Language Model, Retrieval-Augmented Generation, Agent, Personalization
1https://github.com/Applied-Machine-Learning-Lab/Awesome-Personalized-RAG-Agent
* Equal contribution.
†Corresponding authors.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
1arXiv:2504.10147v1  [cs.IR]  14 Apr 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
ACM Reference Format:
Xiaopeng Li∗, Pengyue Jia∗, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng Du, Xiangyang
Li, Yong Liu, Huifeng Guo, Ruiming Tang†, and Xiangyu Zhao†. 2018. A Survey of Personalization: From RAG to Agent. In Proceedings
of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX). ACM, New York, NY,
USA, 25 pages. https://doi.org/XXXXXXX.XXXXXXX
1 INTRODUCTION
Large Language Models (LLMs) have revolutionized AI-driven applications by enabling natural language understanding
and generation at an unprecedented scale. However, these models often suffer from issues such as outdated responses and
hallucinations, which severely hinder the accuracy of information generation. Retrieval-Augmented Generation (RAG)
has emerged as a promising framework that integrates retrieved information from external corpora, such as external
APIs [ 13,36], scientific repositories [ 86,124] or domain-specific databases [ 4,31], ensuring more knowledge-grounded
and up-to-date outputs.
Its versatility has led to significant applications across various domains, including question answering [ 115], enterprise
search [ 16] and healthcare [ 143], etc. Among these applications, one particularly notable area is in agent workflows,
where RAG enhances autonomous systems by providing context-aware, dynamically retrieved, and reliable knowledge.
This is because each stage of the RAG process closely mirrors key aspects of an agent’s workflow, as shown in Figure 1.
For instance, the query rewriting phase in RAG, which involves semantic understanding and parsing, aligns with the
semantic comprehension stage in agent workflows. Likewise, RAG’s retrieval phase, which focuses on extracting the
most relevant documents, corresponds to the planning and execution phases of an agent, where decisions are made
based on retrieved knowledge. Finally, the generation phase in RAG parallels an agent’s execution stage, where actions
are performed based on the given task. This structural alignment suggests that the architecture of RAG is fundamentally
converging with agent workflows, solidifying its position as a key facilitator of intelligent and autonomous systems.
Although the structural alignment between RAG and agent workflows highlights their deepening convergence, a
critical next step in enhancing these intelligent systems lies in personalization. Personalization is a key driver toward
achieving more adaptive and context-aware AI, which is fundamental for the progression toward Artificial General
Intelligence (AGI). It plays an essential role in applications such as personalized reasoning [ 39,149], adaptive decision-
making [ 72], user-specific content generation [ 109,151], and interactive AI systems [ 73,92]. However, existing research
lacks a comprehensive comparative analysis of personalized RAG and agentic approaches. Current surveys primarily
focus on general RAG methodologies [ 32,35] or agent-related literature [ 63,131,167], without systematically exploring
their implications for personalization. While recent works such as [ 68,168] discuss personalization, they predominantly
address personalized generation within LLMs or specific downstream tasks, overlooking how personalization can be
effectively integrated into RAG and agent workflows.
Motivated by the above issues, this survey aims to provide a comprehensive review of the integration of person-
alization into RAG and agentic RAG frameworks to enhance user experiences and optimize satisfaction. The key
contributions of this work can be summarized as follows:
•We provide an extensive exploration of the existing literature on how personalization is integrated into various
stages of RAG (pre-retrieval, retrieval, and generation) and agentic RAG (understanding, planning, execution, and
generation).
•We summarize the key datasets, benchmarks, and evaluation metrics used in existing research for each subtask to
facilitate future studies in the respective domains.
2

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
PersonalityRAGAgent
Fig. 1. Correlation between personalization and RAG with agent flow.
•We also highlight the limitations of current research and suggest future directions for personalized RAG, emphasizing
potential advancements to address existing challenges.
The outline of this survey is as follows: we introduce what is personalization (Sec. 2) and explain how personalization
is adopted into RAG pipeline (Sec. 3). Then, we present a literature review on where to integrate personalization
within different stages of RAG and agentic RAG workflows (Sec. 4) and discuss the key datasets and evaluation metrics
used in existing research (Sec.5). Lastly, we present a discussion on the limitations of current research and future
directions (Sec. 6).
2 WHAT IS PERSONALIZATION
Personalization in current research refers to the tailoring of model predictions or generated content to align with
an individual’s preferences. In the context of RAG and agents, personalization involves incorporating user-specific
information at various stages of the RAG pipeline or within agents. User personalization can be categorized into the
following types:
•Explicit User Profile: Explicitly presented user information, including biographical details, attributes ( e.g.,age, location,
gender, education), and social connections ( e.g.,social networks).
•User Historical Interactions: Behavioral data, including browsing history, clicks, and purchases, which help infer user
interests and preferences to improve personalization.
3

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
Table 1. Overview of Personalized RAG and Agent.
Field Sub-field Subsub-field Papers
Pre-retrievalQuery
RewritingLearning to
Personalized Query RewriteCLE-QR [60], CGF [38], PEARL [80]
LLM to
Personalized Query RewriteLeast-to-Most Prompting [173], ERAGent [112], CoPS [174], Agent4Ranking [61], FIG [22], BASES [99]
Query
ExpansionTagging-based query
expansionGossple [10], Biancalana and Micarelli [12], SoQuES [15], Zhou et al. [172]
Else Lin and Huang [66], Bender et al. [9], Axiomatic PQEC [79], WE-LM [144], PSQE [14], PQEWC [7]
Others Bobo [33], Kannadasan and Aslanyan [52], PSQE [8]
RetrievalIndexing PEARL [80], KG-Retriever [21], EMG-RAG [137], PGraphRAG [5]
RetrievalDense
RetrievalMeMemo [138], RECAP [71], LAPDOG [43], Gu et al. [37], PersonaLM [77], UIA [155], XPERT [125], DPSR [157],
RTM [11], Pearl [80], MemPrompt [74], ERRA [23], MALP [160], USER-LLM [84], PER-PCS [120]
Sparse
RetrievalOPPU [121], PAG [101], Au et al. [5], UniMS-RAG [128], Deng et al. [29],
Prompt-based
RetrievalLAPS [50], UniMP [140], Shen et al. [111]
Others Salemi et al. [103], PersonalTM [65], Zhang et al. [165]
Post-retrieval PersonaRAG [156], Pavliukevich et al. [89], UniMS-RAG [128], Salemi and Zamani [106], Zhang et al. [164], AutoCompressors [24], FIT-RAG [76]
GenerationGeneration from
Explicit PreferencesDirect
PromptingP2[49], Character Profiling [154] OpinionQA [107], Kang et al. [51], Liu et al. [67], Cue-CoT [129], TICL [26]
Profile-Augmented
PromptingGPG [158], Richardson et al. [101], ONCE [70], LLMTreeRec [163], KAR [145], Matryoshka [58]
Personalized-Prompt
PromptingLi et al. [57], RecGPT [166], PEPLER-D [59], GRAPA [94], SGPT [28], PFCL [152]
Generation from
Implicit PreferencesFine-tuning-Based
MethodsPLoRA [165], LM-P [142], MiLP [165], OPPU [122], PER-PCS [120], Review-LLM [91],
UserIdentifier [78], UserAdapter [171], HYDRA [175], PocketLLM [90], CoGenesis [161]
Reinforcement
Learning-Based
MethodsP-RLHF [62], P-SOUPS [47], PAD [20], REST-PG [104], Salemi et al. [103], RewriterSlRl [57],Kulkarni et al. [54]
From RAG to AgentPersonalized
UnderstandingIn user-profile
understandingXu et al. [148], Abbasian et al. [2],
In agent’s role
understandingRoleLLM [139], Character-LLM [110], Wang et al. [134],
In agent’s user-role
joint understandingSocialBench [18], Dai et al. [27], Ran et al. [96], Wang et al. [126], Tu et al. [123], Neeko [153]
Personalized Planning
and ExecutionMemory
ManagementEMG-RAG [137], Park et al. [87], Abbasian et al. [2], RecAgent [133], TravelPlanner+ [114], PersonalWAB [17], VOYAGER [127], MemoeryLLM [136]
Tool and API Calling VOYAGER [127], Zhang et al. [159], PUMA [17], Wang et al. [126], PenetrativeAI [148], Huang et al. [44], [87], MetaGPT [40], OKR-Agent [169]
Personalized
GenerationAlignment with
User FactCharacter-LLM [110], Wang et al. [135], Dai et al. [27]
Alignment with User
PreferencesWang et al. [139], Ran et al. [96], Wang et al. [134], Chen et al. [18]
•User Historical Content: Implicit personalization derived from user-generated content, such as chat history, emails,
reviews, and social media interactions.
•Persona-Based User Simulation: The use of LLM-based agents to simulate and generate personalized interactions.
Integrating this personalized information at various stages of the RAG and agent workflows enables dynamic
alignment with human preferences, thereby making responses more user-centric and adaptive.
3 HOW TO ADOPT PERSONALIZATION
We define the process of introducing personalization within the RAG pipeline as follows:
𝑔=G(R(Q(𝑞,𝑝),C,𝑝),prompt,𝑝,𝜃) (1)
where𝑝denotes personalized information, and the process unfolds in three steps. In the pre-retrieval phase , query
processing (Q) refines the query 𝑞using personalized information, such as through query rewriting or expansion.
During the retrieval phase , the retriever (R) leverages𝑝to fetch relevant documents from the corpus ( C). Finally, in
thegeneration phase , the retrieved information, combined with 𝑝and structured using the given prompt, id fed into
the generator (G) with parameter 𝜃to produce the final response 𝑔. It is evident that personalized information directly
influences multiple stages of the RAG pipeline. In this survey, we consider the agent system as a specialized application
of the RAG framework, where personalization is incorporated in a manner similar to the RAG framework.
4

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
User history 
User query !Query rewriteRephrased wordingQuery expansionAdded related conceptsOther methodBest phone camera•Huawei user•Photographer•…Top-rated mobile phones for photography ofHuaweibrand
Smartphones with best camera, image stabilization, and 4K video supportBest for professional useBest for selfies
User inputPre-retrievalRetrieval
Fig. 2. Overview of the personalized pre-retrieval stage.
4 WHERE TO ADOPT PERSONALIZATION
4.1 Pre-retrieval
4.1.1 Definition .Pre-retrieval is a crucial step in information retrieval systems, where the original user query is
enhanced or modified before the retrieval process to improve the relevance and quality of the search results, as shown
in Figure 2. This process often incorporates additional contextual or personalized information to better align the query
with the user’s intent. The process can be formalized as follows:
𝑞∗=Q(𝑞,𝑝) (2)
where𝑝and𝑞denote the personalized information and original query, and 𝑞∗is the optimized query after query
reformulation.
4.1.2 Query Rewriting .Query rewriting in RAG at the pre-retrieval stage refers to the process of reformulating user
queries to enhance retrieval effectiveness by improving relevance, disambiguating intent, or incorporating contextual
information before retrieving documents from an external knowledge source. The literature on personalized query
rewriting can be broadly classified into two primary categories: (1) Direct Personalized Query Rewriting and (2) Auxiliary
Personalized Query Rewriting.
(1). Direct Personalized Query Rewriting. The first category focuses on personalized query rewriting by using
direct models. For example, Cho et al . [25] presents a personalized search-based query rewrite system for conversational
AI that addresses user-specific semantic and phonetic errors. Nguyen et al . [82] apply reinforcement learning techniques
to improve query rewriting in online e-commerce systems, leveraging distilled LLMs for personalized performance.
CLE-QR [ 60] explores query rewriting in Taobao’s search engine to enhance user satisfaction through customized query
adaptation. CGF [ 38] introduces a constrained generation framework that allows for more flexible and personalized
query rewriting in conversational AI. Li et al . [57] investigate learning methods to rewrite prompts for personalized
5

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
text generation, improving the relevance and engagement of AI-generated content. Additionally, PEARL [ 80] discusses
personalizing large language model-based writing assistants through the integration of generation-calibrated retrievers,
enhancing AI-generated content.
(2). Auxiliary Personalized Query Rewriting. The second category emphasizes personalized query rewriting by
using auxiliary mechanisms, such as retrieval, reasoning strategies, and external memory. Zhou et al. [173] propose a
least-to-most prompting strategy that aids in complex reasoning within LLMs, which can be adapted for personalized
text generation. ERAGent [ 112] enhance retrieval-augmented LLMs to improve personalization, efficiency, and accuracy,
indirectly supporting personalized query rewriting for content generation. CoPS [ 174] integrate LLMs with memory
mechanisms to create more personalized search experiences, which also influences content generation through better
query understanding. Further, Agent4Ranking [ 61] employs multi-agent LLMs to perform semantic robust ranking,
including personalized query rewriting to improve search rankings. FIG [ 22] combine graph-based methods with LLMs
to query rewrite, improving personalized content generation and conversational interactions. Lastly, BASES [ 99] employ
LLM-based agents to simulate large-scale web search user interactions, contributing to the development of personalized
query rewriting strategies for content generation.
4.1.3 Query Expansion .Query expansion enhances retrieval systems by expanding a user’s original query with
additional terms, synonyms, or refined structure to better capture intent. This improves the relevance and scope
of retrieved documents. Recent advancements in LLMs have reinvigorated this field [ 46,48,132], leveraging their
comprehension and generation abilities to expand queries using encoded knowledge or external retrieval, with notable
success. Personalized query expansion, a subset, incorporates user-specific data to tailor results, boosting performance
and customizing the search experience.
(1). Tagging-based Query Expansion. By 2009, studies began incorporating tagging information to enhance
personalized query expansion. For instance, Gossple [ 10] introduced the TagMap and TagRank algorithms, which
dynamically selected tags from personalized networks constructed using the cosine similarity of user-item tag distances,
improving recall performance. Similarly, Biancalana and Micarelli [12]recorded user queries and visited URLs, leveraging
social bookmarking to extract relevant tags and build a personalized three-dimensional co-occurrence matrix. Based
on this, multiple semantically categorized expanded queries were generated to better reflect user interests. Further
advancements include SoQuES [ 15], which integrated tag semantic similarity with social proximity, and a graph-
based approach [ 172] that utilized Tag-Topic models and pseudo-relevance feedback for term weighting, tailoring the
expansion process to individual user preferences.
(2). Else. Apart from tagging-based techniques, early research on Personalized Query Expansion primarily focused
on modeling user personalization based on search history [ 66], social networks, or preferences derived from friendship
networks [ 9]. The Axiomatic PQEC framework [ 79] formalized expansion rules using both local (user behavior-driven)
and social (network-driven) strategies. In 2017, WE-LM [ 144] advanced this paradigm by modeling multi-relational
networks with word embeddings across tag-word relationships, refining associations through affinity graphs. Later,
PSQE [ 14] further improved tagging-based methods using utf-iuf user profiling, integrating a tag similarity graph with
user profiles in the online phase to compute expansion terms relevant to user interests in real-time, achieving dynamic
personalized expansion. In addition, PQEWC [ 7] leveraged clustering and contextual word embeddings to optimize
query expansions dynamically.
6

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Retrieval
Dense RetrievalIndexing
Chunking Sparse Retrieval
Term 1 Term 2
Term 3 Term 4
Prompt-based
Retrieval
QueryOthersPost-Retrieval
Reranking Summarization
Compr ession Indexing Encoding
chunking
pieces
Fig. 3. Overview of the personalized retrieval stage.
4.1.4 Others .Besides query rewriting and query expansion, other personalized query-related research focuses on
areas like query disambiguation and query auto-completion [ 116]. Bobo [ 33] allows users to input contextual terms
reflecting their domain knowledge. In 2019, a method [ 52] applied fastText embeddings from recent queries to rank
candidates. In addition, PSQE [ 8] employed synthetic user profiles from Wikipedia and word2vec embeddings for query
disambiguation.
4.1.5 Discussion .While both query rewriting and query expansion aim to align user input with system understanding
to enhance retrieval quality, their roles in personalization differ in fundamental ways.Understanding the distinct
operational characteristics and application scenarios of each technique is essential for designing effective personalized
retrieval systems. The key takeaways are listed as follows:
•Query rewriting is most beneficial when the original query is ambiguous , underspecified, or misaligned with
retrieval intents, particularly in conversational or multi-turn settings.
•Query expansion is most effective when the original query is relevant but incomplete — i.e., when it needs to be
semantically broadened to cover additional relevant concepts.
4.2 Retrieval
4.2.1 Definition .The retrieval process involves finding the most relevant documents 𝐷∗from a corpusCbased on a
query𝑞∗, as shown in Figure 3. To incorporate personalization, additional user-specific information 𝑝is integrated into
the retrieval function R. This allows the retrieval process to tailor the selected documents to align with individual user
preferences or contexts, thereby enhancing the relevance and personalization of the generated outputs.
𝐷∗=R 𝑞∗,C,𝑝(3)
In the retrieval process, personalization can primarily be introduced by focusing on three steps: indexing, retrieval,
and post-retrieval. These steps ensure efficient and accurate retrieval of relevant documents or knowledge, while
tailoring the process to individual user preferences. Below, we provide a detailed explanation of each step.
4.2.2 Indexing .Indexing organizes knowledge base data into a structured format to facilitate efficient retrieval. Within
the RAG pipeline, documents are either chunked or entirely encoded into representations before being integrated
into searchable systems [ 30,117]. Conventional encoding methods employ either sparse encoding techniques ( e.g.,
7

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
TF-IDF [ 95], BM25 [ 102]) or dense encoding approaches leveraging pre-trained models, such as BERT [ 1], Siamese
Encoders [98], or LLM-based encoders [64, 147].
To introduce personalization at the indexing stage, PEARL [ 80] generates user embeddings by encoding personal
history data with models like DeBERTa. These embeddings are subsequently clustered to create personalized shared
indices. Other approaches integrate knowledge graphs into indexing to enhance retrieval performance. For example,
KG-Retriever [ 21] employs a Hierarchical Index Graph, consisting of a knowledge graph layer and a collaborative
document layer, to improve RAG retrieval. EMG-RAG [ 137] incorporates personalized memory within an editable
knowledge graph, enabling dynamic retrieval. Similarly, PGraphRAG [ 5] leverages user-centric knowledge graphs to
enhance personalization in retrieval tasks.
4.2.3 Retrieval .The Retrieval step matches a user query with the indexed knowledge base to fetch relevant candidates.
It can be broadly categorized into four different types: (1) Dense Retrieval, (2) Sparse Retrieval, (3) Prompt-based Retrieval,
and (4) Others.
(1). Dense Retrieval. Dense retrieval methods often use vector embeddings and similarity metrics ( e.g.,cosine
similarity) and achieve personalization by encoding user preferences, context, or interactions into query or document
embeddings, enabling tailored results through similarity-based matching. For instance, MeMemo [ 138] retrieves
personalized information by matching user-specific embeddings with document vectors, focusing on private, on-device
text generation. Similarly, RECAP [ 71] and LAPDOG [ 43] enhance personalized dialogue generation by encoding
queries and user profiles as dense vectors and retrieving top-N results, ensuring user-specific context drives the
responses. In chatbots, Gu et al . [37] integrates conversational context and user profiles to align retrieved responses
with user personas. PersonaLM [ 77] employs group-wise contrastive learning, training its retrieval model to align user
queries with domain-specific text fragments, thereby improving personalization. UIA [ 155] employs dual encoders to
retrieve documents tailored to user preferences. XPERT [ 125] incorporates temporal events and user interactions into
embeddings, enabling large-scale retrieval across millions of items.
Dense retrieval also enhances specific applications like e-commerce, medical assistance, and language models. DPSR
[157] and RTM [ 11] encode user queries and product information to personalize product searches dynamically. Pearl [ 80]
and MemPrompt [ 74] retrieve personalized content by leveraging historical user data and memory-assisted mechanisms.
ERRA [ 23] uses review embeddings as dense queries for recommendations. In medical assistance, MALP [ 160] and USER-
LLM [ 84] integrate short- and long-term user interactions into embeddings for contextualized, personalized responses.
Finally, PER-PCS [ 120] retrieves relevant information using individual user histories, enhancing the personalization
capabilities of large language models.
(2). Sparse Retrieval. Sparse retrieval methods often rely on term-based matching ( e.g.,BM25) and apply personal-
ization by assigning higher weights to terms or keywords that are more relevant to the user. OPPU [ 121] uses the BM25
algorithm to select the k most relevant records from the user’s historical data for the current query. Similarly, PAG
[101] incorporates user input and profiles to enhance summarization and retrieval, aligning sparse representations
with personalization objectives for large language models. Au et al . [5] uses BM25 search algorithms to find entries
related to the target user or neighboring users through the graph structure. UniMS-RAG [ 128] combines sparse and
dense retrieval by leveraging multi-source knowledge, such as dialogue context and user images, to refine personalized
responses in dialogue systems. Lastly, Deng et al . [29] apply sparse retrieval to support fact-based queries, considering
user queries and preferences to enhance answer generation for e-commerce applications.
8

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
(3). Prompt-based Retrieval. Prompt-based retrieval leverages prompts to guide retrieval from the model or external
sources and introduces personalization by crafting user-specific prompts that guide the retrieval process. These prompts
may include explicit user preferences, historical interactions, or detailed instructions that reflect the user’s unique
requirements. By embedding this personalized context directly into the prompt, the retrieval process can dynamically
adjust to capture and return results that are most relevant to the user. LAPS [ 50] focuses on multi-session conversational
search by storing user preferences and dialogue context, then using prompts to retrieve relevant information tailored to
the user’s biases and categories of interest. UniMP [ 140] employs user interaction histories as input to prompt-based
retrieval, enabling personalized recommendations for multi-modal tasks, such as vision-language applications, by
aligning prompts with user behavioral data. In contrast, Shen et al . [111] explores the use of LLMs to extract empathy
and narrative styles from user-provided stories, but this work primarily focuses on style extraction and does not
explicitly involve a retrieval component.
(4). Others. Reinforcement learning-based retrieval personalizes the process by optimizing retrieval policies based
on user feedback, learning user preferences over time to adjust strategies. Salemi et al . [103] combines models like BM25,
RbR, and dense retrieval, refining them with reinforcement learning (RL) and knowledge distillation (KD) to adapt to user
profiles for personalized outputs. Parameter-based retrieval leverages pre-trained model parameters to implicitly store
and retrieve user-specific information, allowing direct retrieval from the model without traditional indices. PersonalTM
[65] generates document identifiers (Document IDs) using a Transformer model, encoding query, history, and document
relationships into its parameters for personalization. Similarly, Zhang et al . [165] uses parameterized representations to
integrate user queries and histories, tailoring responses to individual preferences.
4.2.4 Post-retrieval .Current Post-Retrieval methods primarily focus on refining retrieved documents or responses
to improve relevance and coherence, current methodologies could be categorized into three parts (1) Re-ranking, (2)
Summarization, and (3) Compression.
(1). Re-ranking. Re-ranking enhances personalized content generation by prioritizing more relevant documents
at the top. PersonaRAG [ 156] extends RAG by integrating user-centric agents, such as the Live Session Agent and
the Document Ranking Agent, to refine document ranking and improve overall performance. Pavliukevich et al .
[89] propose a cross-encoder BERT model for re-ranking external knowledge within a personalized context. UniMS-
RAG [ 128] introduces a scoring mechanism that evaluates retrieved documents and outputs by optimizing the retriever.
Besides, it includes an evidence attention mask, enabling re-ranking during inference and applying it to personalized
datasets. Salemi and Zamani [106] present an iterative approach to optimizing ranking results based on the expectation-
maximization algorithm, with performance validated in personalized scenarios.
(2). Summarization. Summarization refers to the process of summarizing retrieved information to enhance perfor-
mance. For instance, Zhang et al . [164] introduced a role-playing agent system to summarize retrieved history in order
to improve the final Personalized Opinion Summarization process.
(3). Compression. Compression involves condensing embeddings or retrieved content to enhance efficiency and
effectiveness. Approaches like AutoCompressor [ 24] compress contextual embeddings into shorter semantic represen-
tations, and FIT-RAG [ 76] introduces a self-knowledge recognizer along with a sub-document-level token reduction
mechanism to minimize tokens within RAG pipeline. However, few studies have specifically explored personalized
fields, highlighting a promising direction for future research.
9

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
4.2.5 Discussion .Indexing, retrieval, and post-retrieval methods each play a critical role in ensuring efficient and
personalized information processing, with specific applications and trade-offs. Indexing focuses on organizing knowledge
bases for efficient retrieval, using techniques such as sparse encoding methods like TF-IDF and BM25, which are efficient
but limited in understanding semantics, and dense encoding methods like BERT and DeBERTa, which provide better
semantic understanding but require significant computational resources. These methods are widely used in tasks like
question answering and personalized recommendation systems. Retrieval involves matching user queries with relevant
documents and can be categorized into dense retrieval, which provides high semantic understanding and personalization
but is computationally expensive; sparse retrieval, which is efficient and interpretable but less capable of handling
semantics; prompt-based retrieval, which is highly flexible and adaptable to user needs but requires careful engineering
of prompts; and advanced methods like reinforcement learning-based approaches, which dynamically adapt to user
feedback but are complex to implement. This step is essential in applications like personalized dialogue systems, search
engines, and e-commerce. Post-retrieval methods refine retrieved results to enhance relevance and coherence through
re-ranking, which improves personalization and prioritizes relevant content but increases computational overhead;
summarization, which simplifies complex information for better user understanding but risks losing critical details; and
compression, which reduces computational costs by condensing information but remains underexplored in personalized
contexts. Together, these methods provide a comprehensive pipeline for delivering efficient, relevant, and personalized
outputs, balancing their strengths in semantic understanding, relevance, and flexibility with challenges related to
computational costs and implementation complexity.
4.3 Generation
4.3.1 Definition .Personalized generation incorporates user-specific retrieved documents 𝐷∗, task-specific prompt
𝑝𝑟𝑜𝑚𝑝𝑡 , and user preference information 𝑝via the generatorGparameterized by 𝜃to produce tailored content 𝑔∗
aligned with individual preference, where the flow is shown in Figure 4. The generation process can be formulated as
𝑔∗=G 𝐷∗,prompt,𝑝,𝜃. (4)
Personalized generation can be achieved by incorporating explicit and implicit preferences. Explicit preference-driven
methodologies utilize direct input signals ( e.g.,𝐷∗,prompt , and𝑝), to tailor outputs to specific user preferences.
Conversely, implicit preference-encoded approaches embed personalized information within the parameters 𝜃of the
generator model, during training, thereby facilitating preference alignment without the necessity for explicit runtime
inputs.
4.3.2 Generation from Explicit Preferences .Integrating explicit preferences into LLMs facilitates personalized
content generation. Explicit preference information encompasses user demographic information ( e.g.,age, occupation,
gender, location), user behavior sequences (reflecting historical behavioral patterns), and user historical output texts
(capturing writing style and tone preferences). The injection of explicit preferences for personalized generation can be
categorized into three types: (1) Direct-integrated Prompting, (2) Summary-augmented Prompting, and (3) Adaptive
Prompting.
(1). Direct-integrated Prompting. Integrating user explicit preferences into language models through prompting
enables the prediction of users’ intent and behavioral patterns, facilitating personalized content generation. For instance,
P2[49], Character Profiling [ 154], and OpinionQA [ 107] integrate personalized data into LLMs through prompting for
role-playing task, thereby aligning the model’s responses with specified user profiles. Kang et al . [51] and Liu et al . [67]
10

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Implicit Pr eference
InjectionExplicit Pr eference
Injection
Personalized Input
User Info
Retrieved Docs
Generation
Personalized
Generation
Preference Internalization
Reinforcement
LearningFine-tuningPrompt
Fig. 4. Overview of the personalized generation stage.
integrate interaction histories into LLMs via prompting to predict user rating for candidate items. Cue-CoT [ 129] employs
chain-of-thought reasoning to infer user needs from contextual cues, enabling personalized responses to in-depth
dialogue questions. Additionally, TICL [ 26] proposes a trial-and-error framework that critiques initial LLM-generated
responses, derives explanations and integrates these negative examples into prompts to improve personalization
alignment.
(2). Summary-augmented Prompting. Direct integration of personalized information via prompting struggles with
ambiguous intent signals: Lengthy interaction histories introduce noise that obscures critical behavioral patterns [ 69],
while sparse behavioral data lacks sufficient context for LLMs to derive meaningful user preferences. To address these
issues, recent approaches focus on summarizing user personalized intents and integrating them into prompts. For in-
stance, GPG [ 158] extracts key user habits and preferences from personal contexts, enabling fine-grained personalization.
Similarly, LLMs are employed to generate task-specific summaries of user preferences, enhancing retrieval-augmented
personalized generation capabilities [ 101]. In recommendation systems, ONCE [ 70], LLMTreeRec [ 163], and KAR [ 145]
leverage historical user-item interactions to summarize user preferences. Furthermore, Matryoshka [ 58] generates user
preference summaries by dynamically retrieving and synthesizing historical data.
(3). Adaptive Prompting. Manually designing personalized prompts demands both expert knowledge and significant
labor, motivating the development of automated methods for personalized prompt generation. For example, Li et al . [57]
trains a personalized prompt rewriter via supervised and reinforcement learning. RecGPT [ 166] and PEPLER-D [ 59]
leverage prompt tuning to generate personalized prompts, enhancing sequential and explainable recommendations,
respectively. GRAPA [94] integrates semantic and collaborative signals from user-item interaction graphs with graph
neural networks to generate context-aware personalized prompts. SGPT [ 28] employs prompt tuning to jointly model
common and group-specific patterns, bridging generalized and personalized federated learning paradigms. Furthermore,
PFCL [ 152] achieves multi-granularity human preference modeling: coarse-grained prompts distill shared knowledge,
while fine-grained prompts adapt to individual user characteristics.
4.3.3 Generation from Implicit Preferences .Unlike explicit preference modeling, which captures user preferences
through textual input, implicit preference-based methods incorporate personalization through internal parameters.
This personalization is achieved either through Parameter-Efficient Fine-tuning (PEFT) techniques, such as LoRA [ 42],
11

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
or reinforcement learning-based approaches for preference alignment [ 20,57]. Based on these strategies, we classify
existing methods into two categories: (1) Fine-tuning-Based Methods and (2) Reinforcement Learning-Based Methods.
(1). Fine-tuning Based Methods. For fine-tuning methods, LoRA is the most widely adopted since it is resource-
efficient and enables rapid adaptation without compromising model performance. PLoRA [ 165] introduces a personalized
knowledge integration framework that combines task-specific LoRA with user-specific knowledge. Similarly, LM-P [ 142]
personalizes information via LoRA by incorporating User ID as a personalization factor. MiLP [ 165] employs Bayesian
optimization to determine the optimal personalization injection configuration, including LoRA settings, to effectively
capture and utilize user-specific information. OPPU [ 122] and PER-PCS [ 120] follow a similar approach, leveraging user
history data for fine-tuning LoRA-based personalization. However, PER-PCS differs by incorporating a gating module
that selects the appropriate LoRA, enabling fine-grained personalization. Additionally, Review-LLM [ 91] integrates
LoRA for supervised fine-tuning in the task of personalized review generation.
Beyond LoRA-based approaches, alternative pipelines have been proposed for personalized generation. UserIden-
tifier [ 78] introduces a user-specific identifier, significantly reducing training costs while enhancing personalized
demonstration. UserAdapter [ 171] proposes user-independent prefix embeddings, leveraging prefix tuning for per-
sonalization. Meanwhile, HYDRA [ 175] achieves implicit personalization by training user-specific headers. Recently,
researchers have also explored fine-tuning personalized model on edge devices [ 90] and collaborative learning between
small and large language models to enable more personalized generation [161].
(2). Reinforcement Learning Based Methods. Apart from fine-tuning based methods, recent research has explored
reinforcement learning based techniques to personalize text generation by aligning outputs with user preferences.
P-RLHF [ 62] has been proposed to jointly learn a user-specific and reward model to enable text generation that aligns
with a user’s styles or criteria. P-SOUPS [ 47] models multiple user preferences as a Multi-Objective Reinforcement
Learning (MORL) problem, decomposing preferences into multiple dimensions, each trained independently. PAD [ 20]
aligns text generation with human preferences during inference by utilizing token-level personalized rewards to guide
the decoding process. REST-PG [ 104] introduces a framework that trains large language models to reason over personal
data during response generation. This approach first generates reasoning paths to enhance the LLM’s reasoning ability
and then employs Expectation-Maximization Reinforced Self-Training to iteratively refine the model based on its
high-reward outputs. Additionally, Salemi et al . [103] incorporate reinforcement learning into the RAG pipeline to
improve retrieval accuracy, thereby enhancing the personalization of generated content. Other applications include
RewriterSlRl [ 57], which has been introduced to generate text via RL-based personalized prompt rewriting using
API-based LLMs, and Kulkarni et al . [54] , who explore the use of reinforcement learning to optimize RAG for improving
the relevance and coherence of chatbot responses in specialized domains, ultimately enhancing user satisfaction and
engagement.
4.3.4 Discussion .Personalized generation can be adopted via both explicit and implicit preference injection, yet
they exhibit distinct characteristics that make them suitable for different scenarios. In explicit preference-based
generation, personalization is clearly defined through user profile descriptions, contextual information, and similar
inputs, which are incorporated into generators via prompts. A key advantage of this approach is explainability, as
the personalized information is explicitly provided and easily traceable. Despite leveraging provided preferences and
internal knowledge, explicit preference injection’s personalization is constrained by model capabilities and irrelevant
information interference. In contrast, implicit preference-based generation internalizes personalized information into
12

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
AppliedMachineLearning LabCityU
1From Personalized RAG to Personalized AgentPersonalized Pre-retrievalTheprocess of enhancing or modifying user queries before retrievalPersonalized RetrievalTheprocess of finding the most relevant information from corpus based on queryPersonalized GenerationThe process of creating tailored content aligned with individual preferencesPersonalizedUnderstandingThe process of interactions with user preferences by profiling and role modelingPersonalized Planning and ExecutionThe process of integrating real-time memory and tools to align external knowledgePersonalized Generationuser-profileagent’s roleMemoryTool /APIUser FactUser PreferencesThe process of generating outputs that enforce user preferences fact consistency
Agent
Agent
Agent
Understand
Invoke
Align
Fig. 5. Overview of transition from personalized RAG to personalized agent.
the generator’s parameters through scene-specific personalized data, thereby adapting the model for more fine-grained
personalization. However, these methods typically incur substantial training and computational costs, as they require
fine-tuning the generator’s internal parameters. Therefore, selecting between these approaches should be guided by the
specific application scenario and resource constraints.
4.4 From RAG to Agent
4.4.1 Definition .A personalized LLM-based agent is a system designed to dynamically incorporate user context,
memory, and external tools or APIs to support highly personalized and goal-oriented interactions [ 19,45,146], and
solve problems in a goal-oriented manner [ 63,113]. From the previously introduced stages of RAG, we observe that the
evolution of personalized RAG reveals a structural convergence with agent architectures. We analyze them from three
key perspectives:
•Personalized Understanding : This phase within the agent parallels the query understanding and rewriting process
of RAG as outlined in Section 4.1. However, it extends beyond static semantic parsing by incorporating dynamic user
profiling [ 139] and role modeling [ 110]. This integration enables the agent to dynamically align interactions with
implicit user preferences, facilitating personalized responses and task-specific adaptations [96].
13

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
•Personalized Planning and Execution : This phase in agents mirrors RAG’s retrieval process in Section 4.2 yet it
advances beyond static document retrieval by incorporating real-time memory management [ 87] and sophisticated
tool and API calling [ 127]. This approach ensures the dynamic alignment of external knowledge with personalized
constraints, such as integrating medical history in healthcare agents [2], to deliver context-aware and user-specific
outcomes.
•Personalized Generation : This phase in agents mirrors RAG’s generative process in Section 4.3 but transcends
static template-based generation by integrating user preference and fact alignment. Agents dynamically enforce user
preferences and ensure fact consistency through role-specific mechanisms ( e.g.,social adaptability in conversational
agents [ 2]), enabling outputs to evolve in harmony with personalized and situational constraints rather than relying
solely on predefined generative frameworks.
In general we frame agent architectures as “ personalized RAG++ ”, where persistent memory [ 137] replaces static
indexes, and tool APIs [ 17] serve as dynamic knowledge connectors, enabling complicated, human-aligned interactions
beyond one-shot retrieval, as shown in Figure 5. This progression highlights that as RAG systems incorporate deeper
personalization—requiring user-state tracking, adaptive tool usage, and context-aware generation, they inherently
adopt agent-like capabilities.
4.4.2 Personalized Understanding .Personalized understanding refers to an agent’s ability to accurately interpret
user inputs by integrating user intent recognition and contextual analysis. This process ensures interactions that are
both meaningful and contextually appropriate. The rationale behind this classification lies in its capacity to address
three core aspects of understanding: recognizing user intent, analyzing context, and leveraging user profiles. Each of
these aspects plays a distinct role in improving the agent’s performance.
(1). User-profile Understanding. In user-profile understanding, an agent’s personalized ability primarily depends
on its capacity to accurately model and understand the user’s preferences, context, and intentions. Xu et al . [148]
proposes a framework in which LLMs are designed to understand the physical world, thereby facilitating a deeper
connection between the agent and its environment, which is essential for accurate task execution. Abbasian et al . [2]
further expands this understanding by emphasizing the importance of personalization in health agents, where the
user’s profile directly influences the behavior and decisions of the agent. This user understanding is foundational to
ensuring that the AI agent performs tasks in a way that aligns with individual user needs.
(2). Role Understanding. In agent’s role understanding, the role of the agent within these environments is also
crucial. Recent studies focus on enhancing role-playing capabilities within LLMs. Wang et al . [139] introduce RoleLLM,
a benchmark that aims to elicit and refine the role-playing abilities of LLMs, demonstrating how role understanding
influences agent performance in conversational tasks. Similarly, Shao et al . [110] present Character-LLM, a trainable
agent framework for role-playing, which tailors its responses based on predefined roles. Wang et al . [134] introduce a
method for evaluating personality fidelity in role-playing agents through psychological interviews, aiming to enhance
the realism and consistency of AI-driven characters. This role understanding allows for more contextually appropriate
interactions, increasing the relevance and utility of AI agents across various applications.
(3). User-role Joint Understanding. In agent’s user-role joint understanding, the intersection of user and role
understanding is explored through frameworks that evaluate and enhance the social and personality aspects of LLMs.
SocialBench Chen et al . [18] provides a sociality evaluation framework for role-playing agents. Dai et al . [27] , and
14

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[96] extend this by incorporating multi-modal data and personality-indicative information, respectively, which allows
agents to better adapt to both user and role understanding in dynamic environments. Furthermore, Wang et al . [126]
offers a perspective on how role and environment understanding can improve user experience. Tu et al . [123] contribute
by providing a benchmark specifically for evaluating role-playing agents in the Chinese context, adding a cultural
dimension to role understanding. Finally, Neeko [153] further advances role-based interactions.
4.4.3 Personalized Planning and Execution .Personalized planning and execution refer to the process of designing
and implementing strategies or actions that are specifically tailored to an individual’s unique context, and goals
[44,87,114,159]. It requires agents to dynamically integrate long-term memory, real-time reasoning, and external tool
utilization [ 40,41,169], as demonstrated in healthcare decision support [ 2] and travel planning scenarios [ 17]. We
analyze two fundamental components that enable this personalization in the following.
(1). Memory Management. Effective memory systems allow agents to integrate users’ historical preferences, behav-
ioral patterns, and contextual habits, enhancing their ability to make planning and tailor interactions to user-specific
needs [ 17,127,136]. The EMG-RAG framework [ 137] combines editable memory graphs with retrieval-augmented
generation to maintain dynamic user profiles, while Park et al . [87] implements memory streams and periodic reflection
mechanisms to simulate human-like behavior. In healthcare applications, Abbasian et al . [2] integrates multimodal
user data through specialized memory modules to optimize treatment recommendations. For recommendation systems,
RecAgent [ 133] employs hierarchical memory structures to model user interaction patterns across multiple domains.
Recent advances like TravelPlanner+ [ 114] demonstrate how memory-augmented LLMs achieve higher relevance in
personalized itinerary generation compared to generic planners.
(2). Tool and API Calling. The integration of external tools expands agents’ capabilities beyond pure linguistic
reasoning, enabling agents to interact with users and perform personalized tasks [ 17,126,127,148,159]. For instance,
VOYAGER [ 127] establishes a paradigm for lifelong skill acquisition through automatic API curriculum learning and
skill library construction. In robotics, Zhang et al . [159] develops a bootstrapping framework where LLMs guide robots
in tool-mediated skill discovery, enabling a high success rate in novel object manipulation tasks. The PUMA framework
[17] demonstrates how personalized web agents can achieve performance gains in e-commerce tasks through adaptive
API orchestration. For mobile interaction, Wang et al . [126] implements few-shot tool learning to handle diverse UI
operations with minimal training data. These approaches highlight the importance of tool grounding mechanisms [ 44]
that translate linguistic plans into executable API sequences while maintaining personalization constraints.
This synthesis highlights that modern agent systems achieve enhanced personalization through two primary
strategies: 1) Memory-augmented architectures, which leverage editable memory graphs [ 137], reflection mechanisms
[87], and hierarchical memory structures [ 133] to dynamically adapt to user preferences across various domains; and 2)
Tool and API integration, which expand agent capabilities by balancing generalization with specialization. Future work
may explore improving the contextual relevance and adaptability of memory systems while optimizing real-time tool
interaction for seamless task execution.
4.4.4 Personalized Generation .Based on the foundation of personalized planning and execution mechanisms,
which enable agents to adapt strategies to user-specific contexts [ 44,159], the next critical concern lies in personalized
generation. This capability ensures that generated outputs not only align with factual correctness but also resonate with
users’ unique preferences, personality traits, and situational needs. Personalized generation bridges the gap between
15

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
adaptive reasoning and human-aligned outcomes, allowing agents to produce contextually relevant and emotionally
appropriate responses.
(1). Alignment with User Fact. Alignment with User Fact emphasizes the accuracy, consistency, and factual ground-
ing of personalized responses, ensuring they remain trustworthy across diverse user interactions. This is particularly
challenging in personalized agents, where maintaining character authenticity while avoiding hallucinations requires
balancing creativity with factual adherence. Recent advances address these challenges through improved training
frameworks and evaluation metrics. For instance, Character-LLM [ 110] integrates memory-augmented architectures to
reduce hallucinations while preserving character-specific traits. Wang et al . [135] investigate quantization effects on
personality consistency in edge-deployed agents and stabilize outputs under computational constraints. Dai et al . [27]
ensures multimodal consistency (text-image) in role-playing. These works highlight the importance of architectural
innovations and rigorous evaluation in achieving reliability.
(2). Alignment with User Preferences. Alignment with user preferences ensures that generated outputs reflect
individualized personalities, values, and interaction styles. This requires agents to dynamically interpret implicit user
cues and adapt responses accordingly. Wang et al . [139] benchmarks role-specific alignment. Ran et al . [96] improves
personality fidelity via psychological scale datasets. Wang et al . [134] quantifies alignment via psychological interviews.
Chen et al. [18] evaluates social adaptability in conversations.
4.4.5 Discussion .The architectural evolution from RAG to personalized agents introduces significant advancements
in human-AI interaction but also surfaces critical challenges that warrant further investigation.
Personalized Understanding , while enabling interpretation of user intent and context, faces limitations in real-
time adaptability and generalization. Current approaches like RoleLLM [ 139] and Character-LLM [ 110] demonstrate
robust role-specific comprehension but struggle with dynamic user state tracking, particularly when handling evolving
preferences or multi-session interactions. Furthermore, cultural specificity in benchmarks like CharacterEval [ 123]
reveals gaps in global applicability, as agents trained on region-specific data often fail to generalize across diverse
sociocultural contexts. Future work could explore hybrid architectures that combine continuous learning mechanisms
with privacy-preserving federated learning to address these adaptability constraints while maintaining user trust.
Personalized Planning and Execution , achieves remarkable task specialization through memory management
and tool integration, yet suffers from scalability issues in complex environments. While frameworks like EMG-RAG
[137] and VOYAGER [ 127] effectively manage user-specific constraints, their reliance on predefined API taxonomies
limits emergent tool discovery in novel scenarios. The "cold-start" problem persists in domains requiring rapid skill
acquisition, as seen in healthcare applications [ 2], where delayed API responses can compromise decision-making
efficacy. A promising direction involves developing meta-reasoning architectures that dynamically prioritize memory
recall versus tool invocation based on situational urgency and confidence thresholds.
Personalized Generation balances factual accuracy with preference alignment but risks over-fitting, where
excessive finetuning to user profiles may reinforce cognitive biases. Techniques address surface-level alignment but lack
mechanisms for ethical boundary detection. For instance, agents might inadvertently propagate harmful stereotypes
when mirroring user preferences without critical oversight. Future systems could integrate value-aligned reinforcement
learning with human-in-the-loop validation to preserve authenticity while preventing detrimental customization.
16

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2. Datasets and metrics for personalized RAG and Agent.
FieldMetrics
CategoryMetrics Datasets
Pre-retrievalTextual
QualityBLEU, ROUGE, EM Avocado Research Email Collection [57, 85], Amazon review[57, 83], Reddit comments[57, 118], Amazon ESCI dataset[82, 97], PIP
Information
RetrievalMAP, MRR, NDCG,
Precision, Recall, RBPAOL[88, 174], WARRIORS[99], Personalized Results Re-Ranking benchmark [6], del.icio.us [9, 15, 144, 172],
Flickr [9, 108], CiteULike [10, 14], LRDP [12], Delicious [141], Bibsonomy [79],
Wikipedia [8, 33]
Classification Accuracy, Macro-F1 SCAN [56, 173], AITA WORKSM[53, 80], Robust04 [61]
OthersXEntropy, PMS, Image-Align,
PQEC, Prof overlapAmazon ESCI dataset[82, 97], PIP, Bibsonomy [79]
RetrievalTextual
QualityBLEU, ROUGE, Dis, PPL TOPDIAL [130], Pchatbot [93], DuLemon [150]
Information
RetrievalRecall, MRR, Precision, F1 LiveChat [34], Pchatbot [93], DuLemon [150]
Classification Accuracy, Succ TOPDIAL [130], PersonalityEvd [119], DuLemon [150], PersonalityEdit [75]
OthersFluency, Coherence,
Plausibility, ES, DD, TPEI, PAEPersonalityEvd [119], PersonalityEdit [75]
GenerationTextual
QualityBLEU, ROUGE, Dis,
PPL, METEORLaMP [105], Long LaMP [55], Dulemon [150], PGraphRAG [5], AmazonQA/Products [29], Reddit [170], MedicalDialogue [162]
Classification Accuracy, F1, Persona F1 LaMP [105], Long LaMP [55], Dulemon [150], AmazonQA/Products [29], Reddit [170], MedicalDialogue [162]
Regression MAE, RMSE LaMP [105], Long LaMP [55], PGraphRAG [5]
OthersFluency, Mean Success Rate,
Median Relative ImprovementsPersonalized-Gen [3]
AgentTextual
QualityBLEU, ROUGE, METEOR,
CIDEr, EM, Fluency,
Coherence, Instruction
Adherence, Consistency
related metricsRICO [126], RoleBench [139], Shao et al. [110], Socialbench [18], MMRole-Data [27], ROLEPERSONALITY [96], ChatHaruhi [134],
Character-LLM-Data [153], Knowledge Behind Persona [41], Wang et al. [137], Wang et al. [135], Zheng et al. [169]
Information
RetrievalRecall, F1, Precision Knowledge Behind Persona [41]
ClassificationAccuracy, Failure Rate,
Classification Accuracy,
Preference Rate,
CorrectnessMIT-BIH Arrhythmia Database [148], VirtualHome [44], Socialbench [18], ARC [100],AGIEval [100],HellaSwag [100],MedMCQA [100],
AQUA-RAT [100],LogiQA [100],LSAT-AR [100],LSAT-LR [100],LSAT-RC [100],SAT-English [100],SAT-Math [100], PersonalWAB [17],
TravelPlanner+ [114]
OthersPass@k, Executability,
Productivity, Plausibility of the
StoryHong et al. [40], Zheng et al. [169]
5 EVALUATION AND DATASET
In the evolving landscape of personalization, from RAG to advanced Agent-based systems, the evaluation of models
relies heavily on diverse datasets and metrics tailored to specific tasks. This survey categorizes metrics into several key
types: Textual Quality metrics ( e.g.,BLEU, ROUGE, METEOR) assess the fluency and coherence of generated outputs;
Information Retrieval metrics ( e.g.,MAP, MRR, Recall) evaluate the accuracy and relevance of retrieved information;
Classification metrics ( e.g.,Accuracy, F1) measure task-specific correctness; Regression metrics ( e.g.,MAE, RMSE)
quantify prediction errors; and Other metrics ( e.g.,Fluency, Pass@k) address domain-specific or task-unique aspects
like plausibility or executability. These metrics span pre-retrieval, retrieval, generation, and agent-based personalization
approaches, reflecting their varied objectives. To provide a comprehensive overview, we compile an extensive list of
datasets across these fields, as detailed in Table 2. These datasets, paired with their respective metrics, enable researchers
to benchmark and refine personalized systems, from enhancing query rewriting to enabling autonomous agents in
physical and virtual environments.
6 CHALLENGES AND FUTURE DIRECTIONS
Personalized RAG and agent-based systems still face several critical challenges that warrant further exploration. We list
them as follows:
•Balancing Personalization and Scalability : Integrating personalization data (such as preferences, history, and
contextual signals) into RAG processes often increases computational complexity, making it difficult to maintain
17

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
efficiency and scalability across large-scale systems. Future work could explore lightweight, adaptive embeddings
and hybrid frameworks that seamlessly fuse user profiles with real-time contexts.
•Evaluating Personalization Effectively : Current metrics like BLEU, ROUGE, and human evaluations fall short
in capturing the nuanced alignment of outputs with dynamic user preferences, lacking tailored measures for per-
sonalization efficacy. Developing specialized benchmarks and metrics that assess long-term user satisfaction and
adaptability is crucial for real-world applicability.
•Preserving Privacy through Device–Cloud Collaboration: Personalized retrieval often involves processing
sensitive user data, raising privacy concerns, especially with the increased global emphasis on data protection
regulations, such as the European Union’s General Data Protection Regulation (GDPR). Consequently, a promising
approach is the collaborative integration of on-device small Language models which handle sensitive personal data
locally, with cloud-based LLM, which provides broader contextual knowledge.
•Personalized Agent Planning: Current research on agent planning remains mainly in its early stages, with much
of the work focusing on building foundational frameworks such as GUI agents [ 81] and the application of agents
across diverse domains [ 131]. Notably, the incorporation of personalized approaches has yet to be widely adopted.
Exploring how to integrate personalized support into existing frameworks to enhance user experience represents a
promising and valuable direction for future investigation.
•Ensuring Ethical and Coherent Systems : Bias in data processing, privacy concerns in user profiling, and coherence
across retrieval and generation stages remain unresolved. Future directions should prioritize ethical safeguards,
privacy-preserving techniques, and cross-stage optimization to build trustworthy, unified personalized systems.
7 CONCLUSION
In this paper, we explore the landscape of personalization from Retrieval-Augmented Generation (RAG) to advanced
LLM-based Agents, detailing adaptations across pre-retrieval, retrieval, and generation stages while extending into
agentic capabilities. By reviewing recent literature, datasets, and metrics, we highlight the progress and diversity in
enhancing user satisfaction through tailored AI systems. However, challenges such as scalability, effective evaluation, and
ethical concerns underscore the need for innovative solutions. Future research should focus on lightweight frameworks,
specialized benchmarks, and privacy-preserving techniques to advance personalized AI. Relevant papers and resources
are also compiled online for ease of future research.
REFERENCES
[1] 2021. BERT: a review of applications in natural language processing and understanding. arXiv preprint arXiv:2103.11943 (2021).
[2]Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Conversational health agents: A personalized llm-powered agent
framework. arXiv preprint arXiv:2310.02374 (2023).
[3]Bashar Alhafni, Vivek Kulkarni, Dhruv Kumar, and Vipul Raheja. 2024. Personalized Text Generation with Fine-Grained Linguistic Control. In
Proceedings of the 1st Workshop on Personalization of Generative AI Systems (PERSONALIZE 2024) . 88–101.
[4] Amazon. [n. d.]. Amazon Customer Review Dataset. Online dataset. https://nijianmo.github.io/amazon/
[5]Steven Au, Cameron J Dimacali, Ojasmitha Pedirappagari, Namyong Park, Franck Dernoncourt, Yu Wang, Nikos Kanakaris, Hanieh Deilamsalehy,
Ryan A Rossi, and Nesreen K Ahmed. 2025. Personalized Graph-Based Retrieval for Large Language Models. arXiv preprint arXiv:2501.02157
(2025).
[6]Elias Bassani, Pranav Kasela, Alessandro Raganato, and Gabriella Pasi. 2022. A multi-domain benchmark for personalized search evaluation. In
Proceedings of the 31st ACM International Conference on Information & Knowledge Management . 3822–3827.
[7]Elias Bassani, Nicola Tonellotto, and Gabriella Pasi. 2023. Personalized query expansion with contextual word embeddings. ACM Transactions on
Information Systems 42, 2 (2023), 1–35.
[8] Oliver Baumann and Mirco Schoenfeld. 2024. PSQE: Personalized Semantic Query Expansion for user-centric query disambiguation. (2024).
18

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[9]Matthias Bender, Tom Crecelius, Mouna Kacimi, Sebastian Michel, Thomas Neumann, Josiane Xavier Parreira, Ralf Schenkel, and Gerhard Weikum.
2008. Exploiting social relations for query expansion and result ranking. In 2008 IEEE 24th International Conference on Data Engineering Workshop .
IEEE, 501–506.
[10] Marin Bertier, Rachid Guerraoui, Vincent Leroy, and Anne-Marie Kermarrec. 2009. Toward personalized query expansion. In Proceedings of the
Second ACM EuroSys Workshop on Social Network Systems . 7–12.
[11] Keping Bi, Qingyao Ai, and W Bruce Croft. 2021. Learning a fine-grained review-based transformer model for personalized product search. In
Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval . 123–132.
[12] Claudio Biancalana and Alessandro Micarelli. 2009. Social tagging in query expansion: A new way for personalized web search. In 2009 International
Conference on Computational Science and Engineering , Vol. 4. IEEE, 1060–1065.
[13] Microsoft Bing. [n. d.]. Bing Search Engine . https://www.bing.com
[14] Mohamed Reda Bouadjenek, Hakim Hacid, and Mokrane Bouzeghoub. 2019. Personalized social query expansion using social annotations.
Transactions on Large-Scale Data-and Knowledge-Centered Systems XL (2019), 1–25.
[15] Mohamed Reda Bouadjenek, Hakim Hacid, Mokrane Bouzeghoub, and Johann Daigremont. 2011. Personalized social query expansion using
social bookmarking systems. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval .
1113–1114.
[16] Domenico Bulfamante. 2023. Generative enterprise search with extensible knowledge base using ai . Ph. D. Dissertation. Politecnico di Torino.
[17] Hongru Cai, Yongqi Li, Wenjie Wang, ZHU Fengbin, Xiaoyu Shen, Wenjie Li, and Tat-Seng Chua. [n. d.]. Large Language Models Empowered
Personalized Web Agents. In THE WEB CONFERENCE 2025 .
[18] Hongzhan Chen, Hehong Chen, Ming Yan, Wenshen Xu, Xing Gao, Weizhou Shen, Xiaojun Quan, Chenliang Li, Ji Zhang, Fei Huang, et al .2024.
Socialbench: Sociality evaluation of role-playing conversational agents. arXiv preprint arXiv:2403.13679 (2024).
[19] Jiangjie Chen, Xintao Wang, Rui Xu, Siyu Yuan, Yikai Zhang, Wei Shi, Jian Xie, Shuang Li, Ruihan Yang, Tinghui Zhu, et al .2024. From persona to
personalization: A survey on role-playing language agents. arXiv preprint arXiv:2404.18231 (2024).
[20] Ruizhe Chen, Xiaotian Zhang, Meng Luo, Wenhao Chai, and Zuozhu Liu. 2024. Pad: Personalized alignment of llms at decoding-time. arXiv
preprint arXiv:2410.04070 (2024).
[21] Weijie Chen, Ting Bai, Jinbo Su, Jian Luan, Wei Liu, and Chuan Shi. 2024. Kg-retriever: Efficient knowledge indexing for retrieval-augmented large
language models. arXiv preprint arXiv:2412.05547 (2024).
[22] Zheng Chen, Ziyan Jiang, Fan Yang, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, and Aram Galstyan. 2023. Graph meets LLM: A novel
approach to collaborative filtering for robust conversational understanding. arXiv preprint arXiv:2305.14449 (2023).
[23] Hao Cheng, Shuo Wang, Wensheng Lu, Wei Zhang, Mingyang Zhou, Kezhong Lu, and Hao Liao. 2023. Explainable recommendation with
personalized review retrieval and aspect learning. arXiv preprint arXiv:2306.12657 (2023).
[24] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. arXiv preprint
arXiv:2305.14788 (2023).
[25] Eunah Cho, Ziyan Jiang, Jie Hao, Zheng Chen, Saurabh Gupta, Xing Fan, and Chenlei Guo. 2021. Personalized search-based query rewrite system
for conversational ai. In Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI . 179–188.
[26] Hyundong Cho, Karishma Sharma, Nicolaas Jedema, Leonardo FR Ribeiro, Alessandro Moschitti, Ravi Krishnan, and Jonathan May. 2025. Tuning-
Free Personalized Alignment via Trial-Error-Explain In-Context Learning. arXiv preprint arXiv:2502.08972 (2025).
[27] Yanqi Dai, Huanran Hu, Lei Wang, Shengjie Jin, Xu Chen, and Zhiwu Lu. 2024. Mmrole: A comprehensive framework for developing and evaluating
multimodal role-playing agents. arXiv preprint arXiv:2408.04203 (2024).
[28] Wenlong Deng, Christos Thrampoulidis, and Xiaoxiao Li. 2024. Unlocking the potential of prompt-tuning in bridging generalized and personalized
federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 6087–6097.
[29] Yang Deng, Yaliang Li, Wenxuan Zhang, Bolin Ding, and Wai Lam. 2022. Toward personalized answer generation in e-commerce via multi-
perspective preference modeling. ACM Transactions on Information Systems (TOIS) 40, 4 (2022), 1–28.
[30] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and
Hervé Jégou. 2024. The Faiss library. (2024). arXiv:2401.08281 [cs.LG]
[31] ESPN. [n. d.]. ESPN Sports Statistics Dataset. Online dataset.
[32] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms:
Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining .
6491–6501.
[33] Byron J Gao, David C Anastasiu, and Xing Jiang. 2010. Utilizing user-input contextual terms for query disambiguation. In Coling 2010: Posters .
329–337.
[34] Jingsheng Gao, Yixin Lian, Ziyi Zhou, Yuzhuo Fu, and Baoyuan Wang. 2023. LiveChat: A large-scale personalized dialogue dataset automatically
constructed from live streaming. arXiv preprint arXiv:2306.08401 (2023).
[35] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 2 (2023).
[36] Google. [n. d.]. Google Search . https://www.google.com
19

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
[37] Jia-Chen Gu, Hui Liu, Zhen-Hua Ling, Quan Liu, Zhigang Chen, and Xiaodan Zhu. 2021. Partner matters! an empirical study on fusing personas
for personalized response selection in retrieval-based chatbots. In Proceedings of the 44th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 565–574.
[38] Jie Hao, Yang Liu, Xing Fan, Saurabh Gupta, Saleh Soltan, Rakesh Chada, Pradeep Natarajan, Chenlei Guo, and Gökhan Tür. 2022. CGF: Constrained
generation framework for query rewriting in conversational AI. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing: Industry Track . 475–483.
[39] Nicola Henze, Peter Dolog, and Wolfgang Nejdl. 2004. Reasoning and ontologies for personalized e-learning in the semantic web. Journal of
Educational Technology & Society 7, 4 (2004), 82–97.
[40] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou,
et al. 2023. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352 3, 4 (2023), 6.
[41] WANG Hongru, Minda Hu, Yang Deng, Rui Wang, Fei Mi, Weichao Wang, Yasheng Wang, Wai-Chung Kwan, Irwin King, and Kam-Fai Wong.
[n. d.]. Large Language Models as Source Planner for Personalized Knowledge-grounded Dialogues. In The 2023 Conference on Empirical Methods in
Natural Language Processing .
[42] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank
adaptation of large language models. ICLR 1, 2 (2022), 3.
[43] Qiushi Huang, Shuai Fu, Xubo Liu, Wenwu Wang, Tom Ko, Yu Zhang, and Lilian Tang. 2024. Learning retrieval augmentation for personalized
dialogue generation. arXiv preprint arXiv:2406.18847 (2024).
[44] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. 2022. Language models as zero-shot planners: Extracting actionable knowledge
for embodied agents. In International conference on machine learning . PMLR, 9118–9147.
[45] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. 2024. Understanding
the planning of LLM agents: A survey. arXiv preprint arXiv:2402.02716 (2024).
[46] Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui Wang, and Michael Bendersky. 2023. Query expansion by prompting large language models.
arXiv preprint arXiv:2305.03653 (2023).
[47] Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Am-
manabrolu. 2023. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. arXiv preprint arXiv:2310.11564
(2023).
[48] Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li, Changying Hao, Shuaiqiang Wang, and Dawei Yin. 2024. MILL: Mutual Verification with
Large Language Models for Zero-Shot Query Expansion. In Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) . 2498–2518.
[49] Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. 2023. Evaluating and inducing personality in pre-trained
language models. Advances in Neural Information Processing Systems 36 (2023), 10622–10643.
[50] Hideaki Joko, Shubham Chatterjee, Andrew Ramsay, Arjen P De Vries, Jeff Dalton, and Faegheh Hasibi. 2024. Doing personal laps: Llm-augmented
dialogue construction for personalized multi-session conversational search. In Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval . 796–806.
[51] Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, and Derek Zhiyuan Cheng. 2023. Do llms
understand user preferences? evaluating llms on user rating prediction. arXiv preprint arXiv:2305.06474 (2023).
[52] Manojkumar Rangasamy Kannadasan and Grigor Aslanyan. 2019. Personalized query auto-completion through a lightweight representation of the
user context. arXiv preprint arXiv:1905.01386 (2019).
[53] Anjuli Kannan, Karol Kurach, Sujith Ravi, Tobias Kaufmann, Andrew Tomkins, Balint Miklos, Greg Corrado, Laszlo Lukacs, Marina Ganea, Peter
Young, and Vivek Ramavajjala. 2016. Smart Reply: Automated Response Suggestion for Email. In Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (San Francisco, California, USA) (KDD ’16) . Association for Computing Machinery, New York,
NY, USA, 955–964. https://doi.org/10.1145/2939672.2939801
[54] Mandar Kulkarni, Praveen Tangarajan, Kyung Kim, and Anusua Trivedi. 2024. Reinforcement learning for optimizing rag for domain chatbots.
arXiv preprint arXiv:2401.06800 (2024).
[55] Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra, Alireza Salemi, Ryan A Rossi, Franck Dernoncourt, Hanieh Deilamsalehy, Xiang Chen, Ruiyi
Zhang, Shubham Agarwal, et al .2024. Longlamp: A benchmark for personalized long-form text generation. arXiv preprint arXiv:2407.11016 (2024).
[56] Brenden Lake and Marco Baroni. 2018. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent
networks. In International conference on machine learning . PMLR, 2873–2882.
[57] Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, and Michael Bendersky. 2024. Learning to rewrite prompts for personalized text generation.
InProceedings of the ACM Web Conference 2024 . 3367–3378.
[58] Changhao Li, Yuchen Zhuang, Rushi Qiang, Haotian Sun, Hanjun Dai, Chao Zhang, and Bo Dai. 2024. Matryoshka: Learning to Drive Black-Box
LLMs with LLMs. arXiv preprint arXiv:2410.20749 (2024).
[59] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Personalized prompt learning for explainable recommendation. ACM Transactions on Information
Systems 41, 4 (2023), 1–26.
[60] Sen Li, Fuyu Lv, Taiwei Jin, Guiyang Li, Yukun Zheng, Tao Zhuang, Qingwen Liu, Xiaoyi Zeng, James Kwok, and Qianli Ma. 2022. Query rewriting
in taobao search. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management . 3262–3271.
20

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[61] Xiaopeng Li, Lixin Su, Pengyue Jia, Xiangyu Zhao, Suqi Cheng, Junfeng Wang, and Dawei Yin. 2023. Agent4ranking: Semantic robust ranking via
personalized query rewriting using multi-agent llm. arXiv preprint arXiv:2312.15450 (2023).
[62] Xinyu Li, Ruiyang Zhou, Zachary C Lipton, and Liu Leqi. 2024. Personalized language modeling from personalized human feedback. arXiv preprint
arXiv:2402.05133 (2024).
[63] Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, et al .2024. Personal
llm agents: Insights and survey about the capability, efficiency and security. arXiv preprint arXiv:2401.05459 (2024).
[64] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage
contrastive learning. arXiv preprint arXiv:2308.03281 (2023).
[65] Ruixue Lian, Sixing Lu, Clint Solomon, Gustavo Aguilar, Pragaash Ponnusamy, Jialong Han, Chengyuan Ma, and Chenlei Guo. 2023. PersonalTM:
Transformer memory for personalized retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 2256–2260.
[66] Shan-Mu Lin and Chuen-Min Huang. 2006. Personalized optimal search in local query expansion. In Proceedings of the 18th Conference on
Computational Linguistics and Speech Processing . 221–236.
[67] Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is chatgpt a good recommender? a preliminary study. arXiv
preprint arXiv:2304.10149 (2023).
[68] Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Jieming Zhu, Minda Hu, Menglin Yang, and Irwin King. 2025. A Survey of Personalized
Large Language Models: Progress and Future Directions. arXiv preprint arXiv:2502.11528 (2025).
[69] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How
language models use long contexts. Transactions of the Association for Computational Linguistics 12 (2024), 157–173.
[70] Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. 2024. Once: Boosting content-based recommendation with both open-and closed-source
large language models. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining . 452–461.
[71] Shuai Liu, Hyundong J Cho, Marjorie Freedman, Xuezhe Ma, and Jonathan May. 2023. RECAP: retrieval-enhanced context-aware prefix encoder
for personalized dialogue response generation. arXiv preprint arXiv:2306.07206 (2023).
[72] Tyler Lu and Craig Boutilier. 2011. Budgeted social choice: From consensus to personalized decision making. In IJCAI , Vol. 11. 280–286.
[73] Zhengyi Ma, Zhicheng Dou, Yutao Zhu, Hanxun Zhong, and Ji-Rong Wen. 2021. One chatbot per person: Creating personalized chatbots based on
implicit user profiles. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval . 555–564.
[74] Aman Madaan, Niket Tandon, Peter Clark, and Yiming Yang. 2022. Memory-assisted prompt editing to improve GPT-3 after deployment. arXiv
preprint arXiv:2201.06009 (2022).
[75] Shengyu Mao, Xiaohan Wang, Mengru Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Ningyu Zhang. 2024. Editing Personality for Large Language
Models. In CCF International Conference on Natural Language Processing and Chinese Computing . Springer, 241–254.
[76] Yuren Mao, Xuemei Dong, Wenyi Xu, Yunjun Gao, Bin Wei, and Ying Zhang. 2024. Fit-rag: black-box rag with factual information and token
reduction. arXiv preprint arXiv:2403.14374 (2024).
[77] Puneet Mathur, Zhe Liu, Ke Li, Yingyi Ma, Gil Keren, Zeeshan Ahmed, Dinesh Manocha, and Xuedong Zhang. 2023. Personalm: Language model
personalization via domain-distributed span aggregated k-nearest n-gram retrieval augmentation. In Findings of the Association for Computational
Linguistics: EMNLP 2023 . 11314–11328.
[78] Fatemehsadat Mireshghallah, Vaishnavi Shrivastava, Milad Shokouhi, Taylor Berg-Kirkpatrick, Robert Sim, and Dimitrios Dimitriadis. 2021.
Useridentifier: Implicit user representations for simple and effective personalized sentiment analysis. arXiv preprint arXiv:2110.00135 (2021).
[79] Philippe Mulhem, Nawal Ould Amer, and Mathias Géry. 2016. Axiomatic term-based personalized query expansion using bookmarking system. In
International Conference on Database and Expert Systems Applications . Springer, 235–243.
[80] Sheshera Mysore, Zhuoran Lu, Mengting Wan, Longqi Yang, Steve Menezes, Tina Baghaee, Emmanuel Barajas Gonzalez, Jennifer Neville, and Tara
Safavi. 2023. Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers. arXiv preprint arXiv:2311.09180
(2023).
[81] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al .2024. Gui agents:
A survey. arXiv preprint arXiv:2412.13501 (2024).
[82] Duy A Nguyen, Rishi Kesav Mohan, Van Yang, Pritom Saha Akash, and Kevin Chen-Chuan Chang. 2025. RL-based Query Rewriting with Distilled
LLM for online E-Commerce Systems. arXiv preprint arXiv:2501.18056 (2025).
[83] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In
Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language
processing (EMNLP-IJCNLP) . 188–197.
[84] Lin Ning, Luyang Liu, Jiaxing Wu, Neo Wu, Devora Berlowitz, Sushant Prakash, Bradley Green, Shawn O’Banion, and Jun Xie. 2024. User-llm:
Efficient llm contextualization with user embeddings. arXiv preprint arXiv:2402.13598 (2024).
[85] Douglas Oard, William Webber, David Kirsch, and Sergey Golitsynskiy. 2015. Avocado research email collection. Philadelphia: Linguistic Data
Consortium (2015).
[86] U.S. National Library of Medicine. [n. d.]. PubMed: A Free Resource for Biomedical Literature . https://pubmed.ncbi.nlm.nih.gov/
[87] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive
simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology . 1–22.
21

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
[88] Greg Pass, Abdur Chowdhury, and Cayley Torgeson. 2006. A picture of search. In Proceedings of the 1st International Conference on Scalable Information
Systems (Hong Kong) (InfoScale ’06) . Association for Computing Machinery, New York, NY, USA, 1–es. https://doi.org/10.1145/1146847.1146848
[89] Vadim Igorevich Pavliukevich, Alina Khasanovna Zherdeva, Olesya Vladimirovna Makhnytkina, and Dmitriy Viktorovich Dyrmovskiy. [n. d.].
Improving RAG with LoRA finetuning for persona text generation. ([n. d.]).
[90] Dan Peng, Zhihui Fu, and Jun Wang. 2024. Pocketllm: Enabling on-device fine-tuning for personalized llms. arXiv preprint arXiv:2407.01031 (2024).
[91] Qiyao Peng, Hongtao Liu, Hongyan Xu, Qing Yang, Minglai Shao, and Wenjun Wang. 2024. Review-LLM: Harnessing Large Language Models for
Personalized Review Generation. arXiv:2407.07487 [cs.CL] https://arxiv.org/abs/2407.07487
[92] Hongjin Qian, Zhicheng Dou, Yutao Zhu, Yueyuan Ma, and Ji-Rong Wen. 2021. Learning implicit user profile for personalized retrieval-based
chatbot. In proceedings of the 30th ACM international conference on Information & Knowledge Management . 1467–1477.
[93] Hongjin Qian, Xiaohe Li, Hanxun Zhong, Yu Guo, Yueyuan Ma, Yutao Zhu, Zhanliang Liu, Zhicheng Dou, and Ji-Rong Wen. 2021. Pchatbot: a
large-scale dataset for personalized chatbot. In Proceedings of the 44th international ACM SIGIR conference on research and development in information
retrieval . 2470–2477.
[94] Xiaoru Qu, Yifan Wang, Zhao Li, and Jun Gao. 2024. Graph-enhanced prompt learning for personalized review generation. Data Science and
Engineering 9, 3 (2024), 309–324.
[95] A. Rajaraman and J.D. Ullman. 2011. Mining of Massive Datasets . Cambridge University Press. https://books.google.co.uk/books?id=OefRhZyYOb0C
[96] Yiting Ran, Xintao Wang, Rui Xu, Xinfeng Yuan, Jiaqing Liang, Deqing Yang, and Yanghua Xiao. 2024. Capturing minds, not just words: Enhancing
role-playing language models with personality-indicative data. arXiv preprint arXiv:2406.18921 (2024).
[97] Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas, Anlu Xing, and Karthik
Subbian. 2022. Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search. (2022). arXiv:2206.06588
[98] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084 (2019).
[99] Ruiyang Ren, Peng Qiu, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Hua Wu, Ji-Rong Wen, and Haifeng Wang. 2024. Bases: Large-scale web search user
simulation with large language model based agents. arXiv preprint arXiv:2402.17505 (2024).
[100] Matthew Renze and Erhan Guven. 2024. Self-reflection in llm agents: Effects on problem-solving performance. arXiv preprint arXiv:2405.06682
(2024).
[101] Chris Richardson, Yao Zhang, Kellen Gillespie, Sudipta Kar, Arshdeep Singh, Zeynab Raeesy, Omar Zia Khan, and Abhinav Sethy. 2023. Integrating
summarization and retrieval for enhanced personalization via large language models. arXiv preprint arXiv:2310.20081 (2023).
[102] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends ®in Information
Retrieval 3, 4 (2009), 333–389.
[103] Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024. Optimization methods for personalizing large language models through retrieval
augmentation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 752–762.
[104] Alireza Salemi, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, Tao Chen, Zhuowan Li, Michael Bendersky, and Hamed Zamani. 2025.
Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation. arXiv preprint arXiv:2501.04167 (2025).
[105] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024. LaMP: When Large Language Models Meet Personalization. In
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) . 7370–7392.
[106] Alireza Salemi and Hamed Zamani. 2024. Learning to Rank for Multiple Retrieval-Augmented Models through Iterative Utility Maximization.
arXiv preprint arXiv:2410.09942 (2024).
[107] Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models
reflect?. In International Conference on Machine Learning . PMLR, 29971–30004.
[108] Rossano Schifanella, Alain Barrat, Ciro Cattuto, Benjamin Markines, and Filippo Menczer. 2010. Folks in folksonomies: social link prediction from
shared metadata. In Proceedings of the third ACM international conference on Web search and data mining . 271–280.
[109] Noor Shaker, Georgios Yannakakis, and Julian Togelius. 2010. Towards automatic personalized content generation for platform games. In Proceedings
of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment , Vol. 6. 63–68.
[110] Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu. 2023. Character-llm: A trainable agent for role-playing. arXiv preprint arXiv:2310.10158 (2023).
[111] Jocelyn Shen, Joel Mire, Hae Won Park, Cynthia Breazeal, and Maarten Sap. 2024. HEART-felt Narratives: Tracing Empathy and Narrative Style in
Personal Stories with LLMs. arXiv preprint arXiv:2405.17633 (2024).
[112] Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, and Min Xu. 2024. Eragent: Enhancing retrieval-augmented language models with
improved accuracy, efficiency, and personalization. arXiv preprint arXiv:2405.06683 (2024).
[113] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. 2025. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.
arXiv preprint arXiv:2501.09136 (2025).
[114] Harmanpreet Singh, Nikhil Verma, Yixiao Wang, Manasa Bharadwaj, Homa Fashandi, Kevin Ferreira, and Chul Lee. 2024. Personal Large Language
Model Agents: A Case Study on Tailored Travel Planning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing:
Industry Track . 486–514.
[115] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the
domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for
Computational Linguistics 11 (2023), 1–17.
[116] Mingyang Song and Mao Zheng. 2024. A Survey of Query Optimization in Large Language Models. arXiv preprint arXiv:2412.17558 (2024).
22

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[117] Spotify. 2023. Annoy: Approximate Nearest Neighbors in C++/Python. https://github.com/spotify/annoy
[118] Stuck_In_the_Matrix. 2015. Reddit Public Comments (2007-10 through 2015-05). (2015). https://www.reddit.com/r/datasets/comments/3bxlg7/i_
have_every_publicly_available_reddit_comment/
[119] Lei Sun, Jinming Zhao, and Qin Jin. 2024. Revealing Personality Traits: A New Benchmark Dataset for Explainable Personality Recognition on
Dialogues. arXiv preprint arXiv:2409.19723 (2024).
[120] Zhaoxuan Tan, Zheyuan Liu, and Meng Jiang. 2024. Personalized pieces: Efficient personalized large language models through collaborative efforts.
arXiv preprint arXiv:2406.10471 (2024).
[121] Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2024. Democratizing large language models via personalized
parameter-efficient fine-tuning. arXiv preprint arXiv:2402.04401 (2024).
[122] Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2025. Democratizing Large Language Models via Personalized
Parameter-Efficient Fine-tuning. arXiv:2402.04401 [cs.CL] https://arxiv.org/abs/2402.04401
[123] Quan Tu, Shilong Fan, Zihang Tian, and Rui Yan. 2024. Charactereval: A chinese benchmark for role-playing conversational agent evaluation.
arXiv preprint arXiv:2401.01275 (2024).
[124] Cornell University. [n. d.]. arXiv: An Open Access Repository for Research . https://arxiv.org/
[125] Hemanth Vemuri, Sheshansh Agrawal, Shivam Mittal, Deepak Saini, Akshay Soni, Abhinav V Sambasivan, Wenhao Lu, Yajun Wang, Mehul
Parsana, Purushottam Kar, et al .2023. Personalized retrieval over millions of items. In Proceedings of the 46th International ACM SIGIR Conference
on Research and Development in Information Retrieval . 1014–1022.
[126] Bryan Wang, Gang Li, and Yang Li. 2023. Enabling conversational interaction with mobile ui using large language models. In Proceedings of the
2023 CHI Conference on Human Factors in Computing Systems . 1–17.
[127] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An
open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023).
[128] Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei Wang, Fei Mi, Jeff Z Pan, and Kam-Fai Wong. 2024. Unims-rag: A
unified multi-source retrieval-augmented generation for personalized dialogue systems. arXiv preprint arXiv:2401.13256 (2024).
[129] Hongru Wang, Rui Wang, Fei Mi, Yang Deng, Zezhong Wang, Bin Liang, Ruifeng Xu, and Kam-Fai Wong. 2023. Cue-CoT: Chain-of-thought
prompting for responding to in-depth dialogue questions with LLMs. arXiv preprint arXiv:2305.11792 (2023).
[130] Jian Wang, Yi Cheng, Dongding Lin, Chak Tou Leong, and Wenjie Li. 2023. Target-oriented proactive dialogue systems with personalization:
Problem formulation and dataset curation. arXiv preprint arXiv:2310.07397 (2023).
[131] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al .2024. A survey
on large language model based autonomous agents. Frontiers of Computer Science 18, 6 (2024), 186345.
[132] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. arXiv preprint arXiv:2303.07678 (2023).
[133] Lei Wang, Jingsen Zhang, Hao Yang, Zhiyuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Ruihua Song, Wayne Xin Zhao, et al .2023.
User behavior simulation with large language model based agents. arXiv preprint arXiv:2306.02552 (2023).
[134] Xintao Wang, Yunze Xiao, Jen-tse Huang, Siyu Yuan, Rui Xu, Haoran Guo, Quan Tu, Yaying Fei, Ziang Leng, Wei Wang, et al .2023. Incharacter:
Evaluating personality fidelity in role-playing agents through psychological interviews. arXiv preprint arXiv:2310.17976 (2023).
[135] Yixiao Wang, Homa Fashandi, and Kevin Ferreira. 2024. Investigating the Personality Consistency in Quantized Role-Playing Dialogue Agents. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track . 239–255.
[136] Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al .[n. d.]. MEMORYLLM:
Towards Self-Updatable Large Language Models. In Forty-first International Conference on Machine Learning .
[137] Zheng Wang, Zhongyang Li, Zeren Jiang, Dandan Tu, and Wei Shi. 2024. Crafting Personalized Agents through Retrieval-Augmented Generation
on Editable Memory Graphs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing . 4891–4906.
[138] Zijie J Wang and Duen Horng Chau. 2024. MeMemo: On-device Retrieval Augmentation for Private and Personalized Text Generation. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 2765–2770.
[139] Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian
Yang, et al .2023. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. arXiv preprint arXiv:2310.00746
(2023).
[140] Tianxin Wei, Bowen Jin, Ruirui Li, Hansi Zeng, Zhengyang Wang, Jianhui Sun, Qingyu Yin, Hanqing Lu, Suhang Wang, Jingrui He, et al .
2024. Towards unified multi-modal personalization: Large vision-language models for generative recommendation and beyond. arXiv preprint
arXiv:2403.10667 (2024).
[141] Robert Wetzker, Carsten Zimmermann, and Christian Bauckhage. 2008. Analyzing social bookmarking systems: A del. icio. us cookbook. In
Proceedings of the ECAI 2008 Mining Social Data Workshop . 26–30.
[142] Stanisław Woźniak, Bartłomiej Koptyra, Arkadiusz Janz, Przemysław Kazienko, and Jan Kocoń. 2024. Personalized large language models. arXiv
preprint arXiv:2402.09269 (2024).
[143] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente Grau. 2024. Medical graph rag: Towards safe medical
large language model via graph retrieval-augmented generation. arXiv preprint arXiv:2408.04187 (2024).
[144] Xuan Wu, Dong Zhou, Yu Xu, and Séamus Lawless. 2017. Personalized query expansion utilizing multi-relational social data. In 2017 12th
International Workshop on Semantic and Social Media Adaptation and Personalization (SMAP) . IEEE, 65–70.
23

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY X. Li and P. Jia, et al.
[145] Yunjia Xi, Weiwen Liu, Jianghao Lin, Xiaoling Cai, Hong Zhu, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, and Yong Yu. 2024. Towards
open-world recommendation with knowledge augmentation from large language models. In Proceedings of the 18th ACM Conference on Recommender
Systems . 12–22.
[146] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al .2025. The rise
and potential of large language model based agents: A survey. Science China Information Sciences 68, 2 (2025), 121101.
[147] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack: Packed resources for general chinese
embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval . 641–649.
[148] Huatao Xu, Liying Han, Qirui Yang, Mo Li, and Mani Srivastava. 2024. Penetrative ai: Making llms comprehend the physical world. In Proceedings
of the 25th International Workshop on Mobile Computing Systems and Applications . 1–7.
[149] Hongyan Xu, Hongtao Liu, Pengfei Jiao, and Wenjun Wang. 2021. Transformer reasoning network for personalized review summarization. In
Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1452–1461.
[150] Xinchao Xu, Zhibin Gou, Wenquan Wu, Zheng-Yu Niu, Hua Wu, Haifeng Wang, and Shihang Wang. 2022. Long time no see! open-domain
conversation with long-term persona memory. arXiv preprint arXiv:2203.05797 (2022).
[151] Yiyan Xu, Jinghao Zhang, Alireza Salemi, Xinting Hu, Wenjie Wang, Fuli Feng, Hamed Zamani, Xiangnan He, and Tat-Seng Chua. 2025. Personalized
Generation In Large Model Era: A Survey. arXiv preprint arXiv:2503.02614 (2025).
[152] Hao Yu, Xin Yang, Xin Gao, Yan Kang, Hao Wang, Junbo Zhang, and Tianrui Li. 2024. Personalized federated continual learning via multi-granularity
prompt. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 4023–4034.
[153] Xiaoyan Yu, Tongxu Luo, Yifan Wei, Fangyu Lei, Yiming Huang, Hao Peng, and Liehuang Zhu. 2024. Neeko: Leveraging dynamic lora for efficient
multi-character role-playing agent. arXiv preprint arXiv:2402.13717 (2024).
[154] Xinfeng Yuan, Siyu Yuan, Yuhan Cui, Tianhe Lin, Xintao Wang, Rui Xu, Jiangjie Chen, and Deqing Yang. 2024. Evaluating character understanding
of large language models via character profiling from fictional works. arXiv preprint arXiv:2404.12726 (2024).
[155] Hansi Zeng, Surya Kallumadi, Zaid Alibadi, Rodrigo Nogueira, and Hamed Zamani. 2023. A personalized dense retrieval framework for unified
information access. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval . 121–130.
[156] Saber Zerhoudi and Michael Granitzer. 2024. PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents. arXiv
preprint arXiv:2407.09394 (2024).
[157] Han Zhang, Songlin Wang, Kang Zhang, Zhiling Tang, Yunjiang Jiang, Yun Xiao, Weipeng Yan, and Wen-Yun Yang. 2020. Towards personalized
and semantic retrieval: An end-to-end solution for e-commerce search via embedding learning. In Proceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval . 2407–2416.
[158] Jiarui Zhang. 2024. Guided profile generation improves personalization with llms. arXiv preprint arXiv:2409.13093 (2024).
[159] Jesse Zhang, Jiahui Zhang, Karl Pertsch, Ziyi Liu, Xiang Ren, Minsuk Chang, Shao-Hua Sun, and Joseph J Lim. [n. d.]. Bootstrap Your Own Skills:
Learning to Solve New Tasks with Large Language Model Guidance. In 7th Annual Conference on Robot Learning .
[160] Kai Zhang, Yangyang Kang, Fubang Zhao, and Xiaozhong Liu. 2023. LLM-based medical assistant personalization with short-and long-term
memory coordination. arXiv preprint arXiv:2309.11696 (2023).
[161] Kaiyan Zhang, Jianyu Wang, Ermo Hua, Biqing Qi, Ning Ding, and Bowen Zhou. 2024. Cogenesis: A framework collaborating large and small
language models for secure context-aware instruction following. arXiv preprint arXiv:2403.03129 (2024).
[162] Kai Zhang, Fubang Zhao, Yangyang Kang, and Xiaozhong Liu. 2023. Memory-augmented llm personalization with short-and long-term memory
coordination. arXiv preprint arXiv:2309.11696 (2023).
[163] Wenlin Zhang, Chuhan Wu, Xiangyang Li, Yuhao Wang, Kuicai Dong, Yichao Wang, Xinyi Dai, Xiangyu Zhao, Huifeng Guo, and Ruiming Tang.
2025. LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations. In Proceedings of the 31st International
Conference on Computational Linguistics . 886–896.
[164] Yanyue Zhang, Yulan He, and Deyu Zhou. 2025. Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large
Language Models. arXiv preprint arXiv:2503.00449 (2025).
[165] You Zhang, Jin Wang, Liang-Chih Yu, Dan Xu, and Xuejie Zhang. 2024. Personalized LoRA for human-centered text understanding. In Proceedings
of the AAAI Conference on Artificial Intelligence , Vol. 38. 19588–19596.
[166] Yabin Zhang, Wenhui Yu, Erhan Zhang, Xu Chen, Lantao Hu, Peng Jiang, and Kun Gai. 2024. Recgpt: Generative personalized prompts for
sequential recommendation via chatgpt training paradigm. arXiv preprint arXiv:2404.08675 (2024).
[167] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2024. A survey on the memory
mechanism of large language model based agents. arXiv preprint arXiv:2404.13501 (2024).
[168] Zhehao Zhang, Ryan A Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim,
et al. 2024. Personalization of large language models: A survey. arXiv preprint arXiv:2411.00027 (2024).
[169] Yi Zheng, Chongyang Ma, Kanle Shi, and Haibin Huang. 2023. Agents meet okr: An object and key results driven agent system with hierarchical
self-collaboration and self-evaluation. arXiv preprint arXiv:2311.16542 (2023).
[170] Hanxun Zhong, Zhicheng Dou, Yutao Zhu, Hongjin Qian, and Ji-Rong Wen. 2022. Less is more: Learning to refine dialogue history for personalized
dialogue generation. arXiv preprint arXiv:2204.08128 (2022).
[171] Wanjun Zhong, Duyu Tang, Jiahai Wang, Jian Yin, and Nan Duan. 2021. UserAdapter: Few-shot user learning in sentiment analysis. In Findings of
the Association for Computational Linguistics: ACL-IJCNLP 2021 . 1484–1488.
24

A Survey of Personalization: From RAG to Agent Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[172] Dong Zhou, Séamus Lawless, and Vincent Wade. 2012. Improving search via personalized query expansion using social media. Information retrieval
15 (2012), 218–242.
[173] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al .
2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625 (2022).
[174] Yujia Zhou, Qiannan Zhu, Jiajie Jin, and Zhicheng Dou. 2024. Cognitive personalized search integrating large language models with an efficient
memory mechanism. In Proceedings of the ACM Web Conference 2024 . 1464–1473.
[175] Yuchen Zhuang, Haotian Sun, Yue Yu, Rushi Qiang, Qifan Wang, Chao Zhang, and Bo Dai. [n. d.]. Hydra: Model factorization framework for
black-box llm personalization, 2024. URL https://arxiv. org/abs/2406.02888 ([n. d.]).
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
25
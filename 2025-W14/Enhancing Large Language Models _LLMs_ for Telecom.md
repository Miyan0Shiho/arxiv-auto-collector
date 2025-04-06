# Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation

**Authors**: Dun Yuan, Hao Zhou, Di Wu, Xue Liu, Hao Chen, Yan Xin, Jianzhong, Zhang

**Published**: 2025-03-31 15:58:08

**PDF URL**: [http://arxiv.org/pdf/2503.24245v1](http://arxiv.org/pdf/2503.24245v1)

## Abstract
Large language models (LLMs) have made significant progress in
general-purpose natural language processing tasks. However, LLMs are still
facing challenges when applied to domain-specific areas like
telecommunications, which demands specialized expertise and adaptability to
evolving standards. This paper presents a novel framework that combines
knowledge graph (KG) and retrieval-augmented generation (RAG) techniques to
enhance LLM performance in the telecom domain. The framework leverages a KG to
capture structured, domain-specific information about network protocols,
standards, and other telecom-related entities, comprehensively representing
their relationships. By integrating KG with RAG, LLMs can dynamically access
and utilize the most relevant and up-to-date knowledge during response
generation. This hybrid approach bridges the gap between structured knowledge
representation and the generative capabilities of LLMs, significantly enhancing
accuracy, adaptability, and domain-specific comprehension. Our results
demonstrate the effectiveness of the KG-RAG framework in addressing complex
technical queries with precision. The proposed KG-RAG model attained an
accuracy of 88% for question answering tasks on a frequently used
telecom-specific dataset, compared to 82% for the RAG-only and 48% for the
LLM-only approaches.

## Full Text


<!-- PDF content starts -->

Enhancing Large Language Models (LLMs) for
Telecommunications using Knowledge Graphs and
Retrieval-Augmented Generation
Dun Yuan1, Hao Zhou1, Di Wu2, Xue Liu1,Fellow, IEEE ,
Hao Chen3, Yan Xin3, Jianzhong (Charlie) Zhang3,Fellow, IEEE
School of Computer Science1/ Department of Electrical and Computer Engineering2, McGill University
Standards and Mobility Innovation Lab, Samsung Research America3
Emails: {dun.yuan, hao.zhou4 }@mail.mcgill.ca, xueliu@cs.mcgill.ca, di.wu5@mcgill.ca
{hao.chen1, yan.xin, jianzhong.z }@samsung.com
Abstract —Large language models (LLMs) have made significant
progress in general-purpose natural language processing tasks.
However, LLMs are still facing challenges when applied to
domain-specific areas like telecommunications, which demands
specialized expertise and adaptability to evolving standards. This
paper presents a novel framework that combines knowledge graph
(KG) and retrieval-augmented generation (RAG) techniques to
enhance LLM performance in the telecom domain. The frame-
work leverages a KG to capture structured, domain-specific infor-
mation about network protocols, standards, and other telecom-
related entities, comprehensively representing their relationships.
By integrating KG with RAG, LLMs can dynamically access
and utilize the most relevant and up-to-date knowledge during
response generation. This hybrid approach bridges the gap be-
tween structured knowledge representation and the generative ca-
pabilities of LLMs, significantly enhancing accuracy, adaptability,
and domain-specific comprehension. Our results demonstrate the
effectiveness of the KG-RAG framework in addressing complex
technical queries with precision. The proposed KG-RAG model
attained an accuracy of 88% for question answering tasks on a
frequently used telecom-specific dataset, compared to 82% for the
RAG-only and 48% for the LLM-only approaches.
Index Terms —Large Language Models, Knowledge Graphs,
Retrieval-Augmented Generation, Telecommunications.
I. I NTRODUCTION
Large language models (LLMs) have exhibited remarkable
capabilities across various natural language processing tasks,
including text generation, question answering, and knowledge
retrieval [1]. However, these models often underperform in
domain-specific applications, particularly in highly specialized
fields like telecom fields [2]. This limitation arises because
traditional LLMs are predominantly trained on general-purpose
data sources. Consequently, these models lack the specialized
understanding required to navigate the complexities of telecom-
related queries, which demand in-depth knowledge of cellular
network protocols, industry standards, and rapidly evolving
communication technologies. In addition, the telecom industry
is characterized by intricate systems and swiftly updating
standards set by organizations such as the 3rd Generation Part-
nership Project (3GPP) and the Open Radio Access Network
(ORAN) Alliance. Developing an LLM that can effectively
comprehend and reason about telecom knowledge indicatesgreat benefits for the telecom industry, which can be further
applied to network troubleshooting, project coding, standard
development, etc.
Enabling general LLMs to understand telecom-domain
knowledge is a significant challenge. A widely considered
approach is to fine-tune LLMs on specific telecom datasets.
However, there are several potential challenges for fine-tuning
large language models on domain-specific data, including high
computational costs and inefficiency in capturing the rapidly
evolving nature of telecom standards. Fine-tuning requires
substantial computational resources and time, especially for
very large models, making it impractical for frequent updates.
In addition, fine-tuned models may become outdated quickly
as new standards and protocols are developed, and there is
a risk of overfitting the fine-tuning data, leading to a loss of
generalization capability. Given the great potential of LLMs
and the challenges in telecom domain applications, this work
proposes a novel approach to enhance LLMs for the telecom
domain by integrating Retrieval-Augmented Generation (RAG)
and Knowledge Graph (KG) techniques.
In particular, RAG refers to the process of enhancing lan-
guage models by incorporating relevant information retrieved
from external knowledge sources during response generation
[3]. This allows the model to access and utilize up-to-date and
domain-specific information that is not captured in its static
training data. The RAG framework can improve an LLM’s
accuracy by enabling it to dynamically retrieve relevant, current
information from the KG during response generation. RAG has
emerged as a powerful approach to augment LLMs with domain
knowledge. For instance, the study by Roychowdhury et al. [4]
evaluates the use of RAG metrics for question answering in
telecom applications, while Medical Graph RAG [5] explores
the use of graph-based retrieval for enhancing medical LLMs’
safety. The authors of [6] demonstrate the effectiveness of
GraphRAG for improving predictions in venture capital.
Meanwhile, KG refers to the data structures that represent in-
formation in the form of entities and the relationships between
them. They are used to model complex systems and domains
by capturing the interconnections among various components.arXiv:2503.24245v1  [cs.CL]  31 Mar 2025

Fig. 1. Illustration of the KG-RAG pipeline compared to the traditional RAG pipeline. Unlike the common RAG pipeline, the KG-RAG approach leverages a
KG as a comprehensive knowledge base, providing a more structured and dynamic response capability.
In the telecom domain, KG offers a structured representation of
telecom-specific information by capturing relationships among
key entities such as network protocols, signalling standards,
and hardware components. The integration of LLMs with
KGs has also received significant attention. Research such as
KG-LLM [7] investigates the role of KGs in link prediction
tasks, while Li et al. [8] demonstrate how KG assists LLMs
in complex reasoning tasks. Sun et al. [9] question whether
LLMs could replace KGs, proposing experiments to compare
the two paradigms. Complementarily, Zhang, and Soh [10]
propose a framework for using LLMs to construct KGs through
extraction, definition, and canonicalization.
Combining KG with RAG indicates great potential to adapt
LLMs to telecom domains, contributing to generative AI-
enabled 6G networks. The core contribution of this work is
that we proposed a novel KG-RAG approach to enhance the
knowledge understanding performance for LLMs in the telecom
domain. Such a hybrid methodology leverages the strengths of
structured knowledge representation and the generative capa-
bilities of LLMs, resulting in a model that is more accurate
and better suited to adapt to the evolving nature of telecom
knowledge. By employing RAG and KG, we aim to advance the
application of LLMs in domain-specific contexts, using telecom
as a primary example.
Our proposed KG-RAG framework demonstrates significant
improvements in domain-specific performance for telecommu-
nications. For question answering tasks on the Tspec-LLM [11]
dataset, the KG-RAG model achieved an accuracy of 88%,
compared to 82% for the RAG model and 48% for the LLM-
only approach. These results show the effectiveness of theKG-RAG model that enables LLMs to better handle technical
queries with precision and up-to-date knowledge.
The rest of this paper is organized as follows: Section II
introduces the overall system model and provides a technical
background on KGs. Section III presents the KG-RAG frame-
work, detailing the retrieval mechanism, contextual retrieval,
and generation processes. In Section IV , we describe our
experimental setup, datasets, and evaluation metrics, followed
by the result tables and figures in text summarization and
question answering tasks. Finally, Section V concludes the
paper with a discussion of key findings, potential applications,
and potential future work directions.
II. S YSTEM MODEL AND TECHNICAL BACKGROUND
A. Overall System Model
The proposed system model leverages the integration of
LLMs, KGs, and RAG frameworks to enhance domain-specific
language processing within the telecom field. As shown in
Figure 1, the KG-RAG architecture is organized into several
key components. The system starts by building a domain-
specific KG that encapsulates key entities, such as network
protocols, hardware components, and signalling standards, as
well as their relationships. The KG is constructed using struc-
tured and unstructured data sources from the telecom domain,
transforming these data points into triples of the form (head
entity, relationship, tail entity). When a query is presented to the
system, the retrieval mechanism dynamically retrieves relevant
subgraphs or nodes from the KG. This step uses similarity
measures and relevance scoring to ensure that only the most
pertinent knowledge is selected for further processing. The

retrieved information provides a robust contextual base for
subsequent operations.
Finally, the retrieved knowledge is incorporated into an LLM
using a retrieval-augmented generation approach. This process
involves conditioning the LLM’s response on the retrieved
knowledge, allowing the system to generate accurate and
contextually appropriate responses to user queries. The LLM’s
generative capabilities are enhanced by the structured, domain-
specific information from the KG, resulting in responses that
are both precise and adaptable to evolving telecom standards.
This hierarchical structure ensures that domain-specific knowl-
edge is accurately captured, efficiently retrieved, and effectively
used to enhance LLM responses.
In the following, we will introduce the knowledge graph,
LLM-aided entity extraction, and link prediction.
B. Knowledge Graph
A KG is a structured representation of knowledge consisting
of entities and their relationships. It is commonly used to
model complex systems of information in fields such as natural
language processing and semantic web.
1)Entities and Relationships :KG includes the following
elements: Eas the set of all entities (nodes) in the KG, Ras
the set of all relationships (edges) between entities, and Las
the set of all literals (attribute values).
2)Triples :The fundamental building block of a KG is a
triple (h, r, t ), where h∈Eis the head entity, r∈Ris the
relationship, and t∈E∪Lis the tail, which can be either
another entity or a literal. Thus, a KG can be formally defined
as a set of triples:
KG={(h, r, t )|h∈E, r∈R, t∈E∪L} (1)
3)Graph Representation :A KG can be visualized as a
directed labelled graph G= (V, E), where V=E∪Lis the
set of vertices (entities and literals), and Eis the set of directed
edges representing relationships r∈Rbetween subjects and
objects. Each edge e∈Ecan be represented as a tuple e=
(h, r, t ).
4)Knowledge Graph Embeddings :To enable machine
learning algorithms to process KGs, entities and relationships
can be embedded into continuous vector spaces. An embedding
model maps entities and relationships to vectors as fE:E→
RdandfR:R→Rd, where dis the dimension of the
embedding space.
Additionally, scoring functions are used to measure the
plausibility of a triple (h, r, t ), e.g., the scoring function of
TransE [12] model is defined as
score (h, r, t ) =−∥fE(h) +fR(r)−fE(t)∥2
2 (2)
where fE(h)andfE(t)represent the embeddings of the head
entity hand tail entity t, respectively, and fR(r)denotes the
embedding of the relation r.
C. LLM-aided Entity extraction
Entity extraction is an important step in constructing a KG
for a telecom-specific language model. It involves identifyingkey entities and their relationships from a variety of unstruc-
tured and semi-structured data sources. In the telecom domain,
these entities include concepts like network protocols, hardware
components, signal types, frequency bands, and communication
standards, among others. Accurate extraction of these entities
enables the creation of a structured KG that serves as the
backbone for enhancing the language model’s capabilities.
We employ the LLM to extract entities from input documents
D={d1, d2, . . . , d m}. Each document is tokenized into
smaller segments using predefined prompts to detect named
entities and relationships. Given an input document di, the
LLM generates a set of entities E={e1, e2, . . . , e n}. Each
entity eiis assigned additional metadata Mi, including type
(e.g., protocol, metric, component) and semantic context.
The extraction process is formulated as
E=LLM extract(D) (3)
where LLM extract is the extraction module of the LLM, oper-
ating on the entire document set D. Each entity eimay also
exhibit multiple relationships with other entities, such that
R={(ei, ej, rij)|ei, ej∈E, rij∈ R} (4)
where Rrepresents the set of possible relationships.
D. Link Prediction
Link prediction plays a crucial role in constructing and
expanding a KG where entities such as protocols, network
components, and standards are highly interconnected. The task
involves identifying potential relationships between entities that
are either missing or can be inferred from existing connections
within the KG.
To achieve this, link prediction models are trained to maxi-
mize the scores of true triples while minimizing the scores of
false triples. This ensures that the models accurately predict
valid connections, improving the overall utility and complete-
ness of the KG. The training objective often involves a margin-
based ranking loss:
L=X
(h,r,t )∈TX
(h′,r′,t′)∈T′[γ+f(h′, r′, t′)−f(h, r, t )]+(5)
where Tis the set of positive triples, T′is the set of negative
samples, γis the margin, and [x]+ = max(0 , x)is the hinge
loss. This encourages true triples to have scores at least γhigher
than false ones. Negative samples are typically generated by
corrupting true triples (e.g., replacing the tail entity), helping
the model learn to differentiate valid from invalid facts.
III. KG-RAG F RAMEWORK
This section introduces the KG-RAG framework, developed
to enhance LLMs for domain-specific applications in telecom-
munications. By combining KG with RAG, the framework
aims to improve the accuracy and relevance of LLM-generated
responses. We provide details on the retrieval and generation
components that form the core of the KG-RAG framework.

1) Retrieval: In the KG-RAG framework, the retrieval com-
ponent is responsible for fetching relevant information from the
KG to support the language model’s response generation.
LetQrepresent the set of all possible queries, and let q∈ Q
be a specific user query. The retrieval function Rmaps this
query to a subset of the KG K
R:Q → 2K(6)
where 2Kdenotes the power set of K. The retrieval process
involves the following steps:
1)Query Encoding : Convert the user query qinto a vector
representation q∈Rdusing an encoding function ϕ:
q=ϕ(q) (7)
2)Knowledge Encoding : Encode all candidate knowledge
snippets k∈ K into vector representations k∈Rd.
3)Similarity Computation : Compute the similarity between
the query vector qand each knowledge vector kusing a
similarity measure such as cosine similarity:
sim(q,k) =q·k
∥q∥∥k∥(8)
4)Retrieval of Top- KResults : Select the top- Kknowledge
snippets that have the highest similarity scores with the
query:
Sq= arg maxK
k∈Ksim(q,k) (9)
The retrieval component effectively narrows down the vast
knowledge base to a manageable and relevant subset Sqthat
can be used by the generation component.
The relevance of each knowledge snippet to the query is
quantified using a scoring function s(q, k), which may incor-
porate additional factors like term frequency-inverse document
frequency (TF-IDF) weights, entity matching, or semantic
similarity:
s(q, k) =α·sim(q,k) +β·TF-IDF (q, k) +γ·EM(q, k)(10)
where α,β, and γare weighting coefficients, and EM (q, k)is
an entity matching score.
2) Generation: The generation component uses the retrieved
knowledge snippets Sqto produce a coherent and contextually
appropriate response rto the user query q.
The input to the language model combines the user query
and the retrieved knowledge:
I=Format (q,Sq) (11)
where Format is a function that structures the input appropri-
ately, such as by appending the knowledge snippets to the query
with special tokens to delineate them.
The language model generates the response by maximizing
the conditional probability:
P(r|I) =TY
t=1P(rt|r<t, I) (12)
where rtis the t-th token in the response, r<tare the tokensgenerated so far, and Tis the total number of tokens in the
response.
During generation, attention mechanisms allow the model to
focus on relevant parts of the input, including both the query
and the retrieved knowledge. The attention weight αt,ifor the
i-th input token at time tis computed as:
αt,i=exp(et,i)P
jexp(et,j)(13)
et,i=h⊤
t−1Waei (14)
where ht−1is the hidden state of the decoder at time t−1,
Wais a learned parameter matrix, and eiis the embedding of
thei-th input token.
The model is trained to minimize the negative log-likelihood
loss:
L=−TX
t=1logP(rt|r<t, I) (15)
To encourage the model to use the retrieved knowledge, an
auxiliary loss term can be added to penalize deviations from
the knowledge:
Ltotal=L+λ· Lknowledge (16)
where λis a hyperparameter balancing the two loss terms,
andLknowledge measures the discrepancy between the generated
response and the retrieved knowledge.
To summarize, for any given query, the processes in KG-
RAG ensure that the LLM has access to the most relevant
domain-specific knowledge from the KG. The steps are sum-
marized as follows:
•Query Encoding: Encode the query into a semantic
vector.
•Graph Retrieval: Retrieve nodes and edges from the KG
using similarity matching.
•Response Generation: Use the retrieved subgraph as
input to the LLM to generate a response.
The response generation step ensures that the KG-RAG
framework can provide accurate, grounded answers by lever-
aging the structured knowledge from the KG.
IV. P ERFORMANCE EVALUATION
A. Experiment Settings
In this work, we use OpenAI’s GPT-4o-mini [13] model
as the basic reasoning LLM for our experiments. This model
demonstrates strong capabilities in reasoning tasks, including
addressing complex queries such as telecom-related questions
and broader analytical challenges. Its integration into our
framework aligns with the nature of the datasets applied in
our work, providing robust and contextually aware responses.
Moreover, its cost-efficiency is particularly advantageous for
analyzing and performing inference on large corpus datasets
such as telecom documents.

1)Telecom Datasets :
1⃝: SPEC5G [14]: It contains 134 million words and is
a comprehensive collection of technical specifications and
documentation related to 5G wireless technology. It includes
standards from organizations such as 3GPP, ITU, and ETSI.
2⃝: Tspec-LLM [11]: Comprising 534 million words and
100 questions, Tspec-LLM is a specialized corpus of technical
telecom documents curated for training and evaluating large
language models.
3⃝: TeleQnA [15]: This dataset features 10,000 curated
question-and-answer pairs focused on the telecom domain,
providing a valuable resource for evaluation purposes. In order
to utilize the ability of the framework, we use the documents of
Tspec-LLM dataset as the knowledge database when answering
questions.
4⃝: ORAN-Bench-13K [16]: This dataset consists of 2.53
million words and 13,000 multiple-choice questions generated
from 116 O-RAN specification documents. The questions are
categorized into three difficulty levels.
2)Evaluation Tasks :
1⃝: Text Summarization : This task involves generating con-
cise summaries of technical telecom documents to extract key
information and insights. It helps in digesting large volumes of
data by highlighting the most important aspects.
2⃝: Question Answering : This task focuses on providing
accurate answers to queries based on telecom-related data.
It evaluates the model’s ability to understand questions and
retrieve relevant information.
3)Baselines :
1⃝: LLM-only : The LLM directly processes queries without
external retrieval or structured knowledge. It relies solely on
its internal knowledge.
2⃝: RAG : The RAG framework augments the LLM with a
retrieval component, fetching relevant documents or snippets
from datasets. The retrieved content provides additional context
for response generation.
3⃝: KG-RAG : The introduced method in this work integrates
KGs and the RAG pipeline.
B. Results
This subsection will present the evaluation results of our
KG-RAG framework on the text summarization and question
answering tasks. We compare the performance of three models:
LLM-only, RAG, and our proposed KG-RAG model.
1) Text summarization: The text summarization task in-
volves generating concise and informative summaries of techni-
cal telecom documents from the SPEC5G [14] dataset. The goal
is to condense complex and lengthy technical specifications into
shorter texts that retain the most important information, making
them more accessible for analysis and decision-making.
We evaluate the performance of the models using the Recall-
Oriented Understudy for Gisting Evaluation (ROUGE) metrics
[17], which measure the overlap between the generated sum-
maries and the reference summaries. Specifically, we use the
ROUGE metrics. The ROUGE score is well-suited for this taskbecause it captures the content similarity between the generated
and reference summaries, which is critical in assessing the
quality of summarization. ROUGE metrics used in this work
include:
•ROUGE-1 : Measures the overlap of unigrams between
the candidate and reference texts:
ROUGE-1 =Number of overlapping unigrams
Total unigrams in the reference
•ROUGE-2 : Measures the overlap of bigrams between the
candidate and reference texts:
ROUGE-2 =Number of overlapping bigrams
Total bigrams in the reference
•ROUGE-L : Measures the longest common subsequence
between the candidate and reference texts, providing a
measure of sequence-level similarity and capturing sen-
tence structure:
ROUGE-L =Length of LCS
Total words in the reference
Table I presents the mean and variance scores for all items
in the SPEC5G [14] dataset. The results show that the KG-
RAG model consistently outperforms both the LLM-only and
RAG models across all ROUGE metrics. Notably, the KG-
RAG model achieves a ROUGE-1 score of 0.58, which is a
significant improvement over the LLM-only model’s score of
0.53 and the RAG model’s score of 0.55. Similar improvements
can be observed in ROUGE-2 and ROUGE-L scores, indicating
that the KG-RAG model generates summaries that are more
similar to the reference summaries in terms of both content
and sequence.
2) Question answering: We assess the effectiveness of each
method in accurately answering telecom-specific questions. The
evaluation metrics used include accuracy and response time.
To evaluate the performance of each model, we utilized sets
of telecom-specific questions similar to the one above. The
questions were designed to test the models’ understanding of
key concepts in telecom such as signal processing, network
architecture, and wireless communication protocols.
TABLE I
RESULTS OF TEXT SUMMARIZATION TASK ON SPEC5G DATASET
Models ROUGE-1 ROUGE-2 ROUGE-L
LLM-only 0.53±0.02 0.31±0.02 0.44±0.03
RAG 0.55±0.03 0.34±0.02 0.45±0.02
KG-RAG 0.58±0.02 0.38±0.03 0.46±0.02
TABLE II
ACCURACY OF EACH MODEL ON DIFFERENT DATASETS
Models Tspec-LLM TeleQnA ORAN-Bench-13K
LLM-only 0.48 0.72 0.26
RAG 0.82 0.74 0.72
KG-RAG 0.88 0.75 0.80

Fig. 2. Accuarcies of different models by question difficulty on Tspec-LLM
[11]
Fig. 3. Accuarcies of different models by question difficulty on ORAN-Bench-
13K [16]
As shown in Table II, the KG-RAG model consistently
outperforms the LLM-only and RAG models across all datasets.
Specifically, on the Tspec-LLM [11] dataset, the KG-RAG
model achieves an accuracy of 0.88, a substantial improve-
ment over the LLM-only’s accuracy of 0.48 and the RAG’s
accuracy of 0.82. Similar performance gains are observed on
the ORAN-Bench-13K [16] datasets. It is worth noting that
the TeleQnA [15] datasets feature more general questions that
are not specifically tied to particular documents. Consequently,
the LLM-only approach, even without the RAG component,
does not perform as poorly compared to other methods in this
context.
We also examined the accuracy by question difficulty lev-
els on the Tspec-LLM and ORAN-Bench-13K datasets. The
dataset categorizes questions into three difficulty levels: easy,
intermediate, and hard. From Figure 2 and Figure 3, we observe
that the KG-RAG model demonstrates superior performance
across all difficulty levels. The improvement is particularly
notable for intermediate and hard questions, where domain-
specific knowledge and reasoning are critical.
V. C ONCLUSION
LLM is a promising technique for envisioned 6G networks.
This paper introduced the KG-RAG framework, which inte-grates KG with RAG to enhance LLMs for the telecommu-
nications domain. Our methodology addresses the limitations
of general-domain LLMs by grounding responses in structured
knowledge extracted from telecom-specific datasets. Our ex-
periments demonstrated that KG-RAG outperforms LLM-only
and traditional RAG models, showing significant gains across
telecom datasets for question answering and summarization
tasks. Our work highlights the potential of integrating KGs
with RAG framework to create domain-specific models that can
effectively address the challenges of complex technical queries,
providing a significant advancement in the application of LLMs
within the telecom industry. In future work, we plan to explore
the dynamic expansion of the KG by incorporating real-time
updates from telecom industry standards and also to further
compare the proposed framework with other RAG frameworks.
REFERENCES
[1] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and
J. Larson, “From local to global: A graph rag approach to query-focused
summarization,” arXiv preprint arXiv:2404.16130 , 2024.
[2] H. Zhou, C. Hu, Y . Yuan, Y . Cui, Y . Jin, C. Chen, H. Wu, D. Yuan,
L. Jiang, D. Wu et al. , “Large language model (llm) for telecommu-
nications: A comprehensive survey on principles, key techniques, and
opportunities,” arXiv preprint arXiv:2405.10825 , 2024.
[3] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel et al. , “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
Neural Information Processing Systems , vol. 33, pp. 9459–9474, 2020.
[4] S. Roychowdhury, S. Soman, H. Ranjani, N. Gunda, V . Chhabra, and
S. K. Bala, “Evaluation of rag metrics for question answering in the
telecom domain,” arXiv preprint arXiv:2407.12873 , 2024.
[5] J. Wu, J. Zhu, and Y . Qi, “Medical graph rag: Towards safe medical
large language model via graph retrieval-augmented generation,” arXiv
preprint arXiv:2408.04187 , 2024.
[6] G. Zitian and X. Yihao, “Enhancing startup success predictions in venture
capital: A graphrag augmented multivariate time series method,” arXiv
preprint arXiv:2408.09420 , 2024.
[7] D. Shu, T. Chen, M. Jin, Y . Zhang, M. Du, and Y . Zhang, “Knowledge
graph large language model (kg-llm) for link prediction,” arXiv preprint
arXiv:2403.07311 , 2024.
[8] Y . Li, R. Zhang, and J. Liu, “An enhanced prompt-based llm reasoning
scheme via knowledge graph-integrated collaboration,” in International
Conference on Artificial Neural Networks . Springer, 2024, pp. 251–265.
[9] K. Sun, Y . E. Xu, H. Zha, Y . Liu, and X. L. Dong, “Head-to-tail: How
knowledgeable are large language models (llm)? aka will llms replace
knowledge graphs?” arXiv preprint arXiv:2308.10168 , 2023.
[10] B. Zhang and H. Soh, “Extract, define, canonicalize: An llm-
based framework for knowledge graph construction,” arXiv preprint
arXiv:2404.03868 , 2024.
[11] R. Nikbakht, M. Benzaghta, and G. Geraci, “Tspec-llm: An open-source
dataset for llm understanding of 3gpp specifications,” arXiv preprint
arXiv:2406.01768 , 2024.
[12] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko,
“Translating embeddings for modeling multi-relational data,” Advances
in neural information processing systems , vol. 26, 2013.
[13] July 2024. [Online]. Available: https://openai.com/index/
gpt-4o-mini-advancing-cost-efficient-intelligence/
[14] I. Karim, K. S. Mubasshir, M. M. Rahman, and E. Bertino, “Spec5g:
A dataset for 5g cellular network protocol analysis,” arXiv preprint
arXiv:2301.09201 , 2023.
[15] A. Maatouk, F. Ayed, N. Piovesan, A. De Domenico, M. Debbah, and Z.-
Q. Luo, “Teleqna: A benchmark dataset to assess large language models
telecommunications knowledge,” arXiv preprint arXiv:2310.15051 , 2023.
[16] P. Gajjar and V . K. Shah, “Oran-bench-13k: An open source bench-
mark for assessing llms in open radio access networks,” arXiv preprint
arXiv:2407.06245 , 2024.
[17] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText summarization branches out , 2004, pp. 74–81.
# Interpretable Question Answering with Knowledge Graphs

**Authors**: Kartikeya Aneja, Manasvi Srivastava, Subhayan Das, Nagender Aneja

**Published**: 2025-10-22 02:36:35

**PDF URL**: [http://arxiv.org/pdf/2510.19181v1](http://arxiv.org/pdf/2510.19181v1)

## Abstract
This paper presents a question answering system that operates exclusively on
a knowledge graph retrieval without relying on retrieval augmented generation
(RAG) with large language models (LLMs). Instead, a small paraphraser model is
used to paraphrase the entity relationship edges retrieved from querying the
knowledge graph. The proposed pipeline is divided into two main stages. The
first stage involves pre-processing a document to generate sets of
question-answer (QA) pairs. The second stage converts these QAs into a
knowledge graph from which graph-based retrieval is performed using embeddings
and fuzzy techniques. The graph is queried, re-ranked, and paraphrased to
generate a final answer. This work includes an evaluation using LLM-as-a-judge
on the CRAG benchmark, which resulted in accuracies of 71.9% and 54.4% using
LLAMA-3.2 and GPT-3.5-Turbo, respectively.

## Full Text


<!-- PDF content starts -->

Interpretable Question Answering with
Knowledge Graphs
Kartikeya Aneja1,3, Manasvi Srivastava2,3, Subhayan Das4, and Nagender
Aneja5
1Department of Electrical and Computer Engineering, University of
Wisconsin-Madison, Madison, USAkaneja@wisc.edu
2Department of Computer Science, Banasthali Vidyapeeth, Rajasthan, India
manasvisrivastava26@gmail.com
3Ansyst Consulting, Gurgaon, India
4Huawei Ireland Research Center, Dublin, Irelandsubhayan.das@h-partners.com
5Bradley Department of Electrical and Computer Engineering, Virginia Tech,
Blacksburg, VA, USAnaneja@vt.edu
Abstract.This paper presents a question answering system that op-
erates exclusively on a knowledge graph retrieval without relying on re-
trievalaugmentedgeneration(RAG)withlargelanguagemodels(LLMs).
Instead, a small paraphraser model is used to paraphrase the entity
relationship edges retrieved from querying the knowledge graph. The
proposed pipeline is divided into two main stages. The first stage in-
volves pre-processing a document to generate sets of question-answer
(QA) pairs. The second stage converts these QAs into a knowledge graph
from which graph-based retrieval is performed using embeddings and
fuzzy techniques. The graph is queried, re-ranked, and paraphrased to
generate a final answer. This work includes an evaluation using LLM-as-
a-judge on the CRAG benchmark, which resulted in accuracies of 71.9%
and 54.4% using LLAMA-3.2 and GPT-3.5-Turbo, respectively.
Keywords:Knowledge Graph·Question Answering·RAG
1 INTRODUCTION
Question Answering (QA) systems aim to provide precise and contextually rele-
vant answers to user queries, often using structured or unstructured knowledge
sources. Recently, there has been interest in Retrieval-Augmented Generation
(RAG)[1], where a user’s query retrieves relevant text chunks, which are then
passed to an LLM for answer generation. However, these systems are heavily
reliant on unstructured documents and prone to hallucination [2, 3] and have
limited transparency.
In contrast, Knowledge Graphs (KGs) provide a structured representation of
information in the form of entities (nodes) and their relationships (edges). The
KGs can be created from both structured and unstructured data. This formal-
ism supports interpretable reasoning and contextual association, making them
International Semantic Intelligence Conference (ISIC), Germany, 20251arXiv:2510.19181v1  [cs.CL]  22 Oct 2025

particularly suitable for knowledge-intensive QA tasks. Traditional KG-based
QA systems often require complex semantic parsing and rule-based components,
limiting their scalability and adaptability.
This research presents a Knowledge Graph-based QA system that does not
rely on traditional chunk retrieval using RAG but instead constructs and queries
astructured knowledge graphgeneratedfrom QApairs. UnlikeRAGframeworks
that use text chunking and dense retrieval, this approach builds a graph-based
abstraction of knowledge, enabling semantic and entity-based search across the
document’s contents [4, 5, 6].
Our methodology consists of two primary phases.
First, QA pairs are generated from documents using a prompt-driven lan-
guage model. These pairs are then converted into a knowledge graph via entity-
relation extraction usingLLMGraphTransformerbyLangchainwithGPT-3.5-
Turbo. In the second stage, user queries are answered by retrieving relevant sub-
graphsbasedonembeddingsimilarityandfuzzymatching.Retrievedinformation
ispassedthroughalightweightparaphrasingmodeltuner007/pegasus_paraphrase
to produce coherent, human-readable answers. We evaluated the performance
using LLM-as-a-judge.
By combining structured KG retrieval with paraphrased natural language
responses, the proposed research offers an interpretable alternative to traditional
RAG pipelines. It demonstrates practical benefits in legal and technical domains
where traceability and factual consistency are critical.
Table 1.QA Graph-Based Answering Flow
Component Description
Document Input PDF containing contractual clauses
QA Pairs Extracted using prompt-based QA generation
Graph Creation GPT-3.5 identifies entities, relations
Embeddings text-embedding-3-large on nodes and types
Retrieval Cosine and fuzzy search on entities
Reranker BAAI/bge-reranker-large on triples
Answer tuner007/pegasus_paraphrase paraphrasing
2 Related Work
This section explains related work on knowledge graph-based question answer-
ing while avoiding LLM-based RAG. Related work can be categorized into two
broad approaches: (i) semantic parsing that converts a question into a logical
query, e.g., Cypher in the case of Neo4j, and executes it on the KG, and (ii)
International Semantic Intelligence Conference (ISIC), Germany, 20252

retrieval/embedding methods that can find relevant subgraphs or entities and
paths without building a full logical query.
Gu et al. [7] reviewed research on question answering over knowledge bases
for semantic-parsing and proposed joint entity linking and parsing. Most se-
mantic parsing systems target either SPARQL, a semantic query language for
data stored in Resource Description Framework (RDF) format, or use Cypher,
a standard query language for property graph databases. Du et al. [8] proposed
constructing a dual relation graph where relations are nodes, and edges connect
relations sharing head or tail entities. Their approach alternates between (1)
reasoning on the primal entity graph, (2) propagating information on the dual
relation graph, and (3) enabling interaction between the two graphs.
Zhao et al. [9] proposed linking relation phrases to relation paths instead of
single relations. The authors introduce a path ranking model that aligns both
textual (word embeddings) and structural (KG embeddings) information, utiliz-
ing a gated attention mechanism with external paraphrase dictionaries to handle
vague phrases. Yang et al. [10] highlighted the limitations of existing RAG ap-
proaches, showing that even advanced LLMs with RAG achieve only 44–63%
accuracy on dynamic, diverse questions. Although CRAG was built to bench-
mark RAG, our KG-only pipeline attains competitive accuracy on CRAG.
The proposed system leverages structured KG construction from QA pairs,
followed by embedding-based retrieval, fuzzy entity matching, reranking, and
paraphrasing. This design emphasizes interpretability, traceability, and reduced
hallucination, providing clearer reasoning paths compared to black-box RAG
methods.
3 RESEARCH METHODOLOGY
3.1 QA Generation from Document
We used two datasets: one is a PDF document, and the other is the CRAG
dataset [10]. For our PDF document, we generated question-answer (QA) pairs.
The text from the PDF is first divided into semantically meaningful units using
a rule-based segmentation strategy. This strategy leverages structural cues, such
as paragraph breaks, bullet points, clause identifiers, and length thresholds, to
ensure that each chunk represents a logically coherent segment. These segments
serveasatomicknowledgeblocksfordownstreamindexingandretrieval.Foreach
extracted chunk, the Hugging Face modeliarfmoose/t5-base-question-generator
is used to generate a list of relevant question-answer pairs automatically. These
pairs are designed to extract atomic facts and knowledge from the text and are
stored in JSON format for further processing. In the CRAG, there are two tasks
with 2706 question-answer pairs.
3.2 Knowledge Graph Creation
This section describes the methodology for constructing a Knowledge Graph
from question–answer (QA) pairs and the subsequent process of querying it for
International Semantic Intelligence Conference (ISIC), Germany, 20253

relevant information. Given an input question, the system identifies and retrieves
the associated entities and relationships from the graph. These retrieved entities
and relationships are then fed into a lightweight paraphrasing model, which
synthesizes the final answer in natural language. Table 1 presents a high-level
overview of the Knowledge Graph-based answer generation workflow.
Knowledge Graph ConstructionSets of question–answer (QA) pairs are
processed through the LLMGraphTransformer of LangChain, utilizing the GPT-
3.5 Turbo language model to extract entities and their semantic relationships.
The additional information to consider Clauses and Numerical Values as nodes
was passed to the LLMGraphTransformer as a prompt shown in Fig. 1. For this
research,setsoftwentyQApairsarepassedtoLLMGraphTransformeratatime.
prompt = "Clauses and Numerical Values, etc count as nodes"
llm_transformer = LLMGraphTransformer(llm=llm, additional_instructions=
prompt)
Fig. 1.LLM Graph Transformer setup
These nodes and relationships are used to construct a Knowledge Graph,
which is stored in a Neo4j graph database. After processing all QA pairs, the re-
sultingKnowledgeGraphencapsulatesthedocument’sknowledgeinastructured
form. In this graph, nodes represent entities, while edges denote the relationships
between them. We created two knowledge graphs, one for each dataset.
Wethengeneratedembeddingsforeachnodeandnodetypeusingtransformer-
based sentence embeddings via thetext-embedding-3-largemodel. These embed-
dings are computed internally by Neo4j using the specified model and subse-
quently stored within the graph database. In addition to individual node em-
beddings, we also computed and stored embeddings for node types. Embeddings
for node types are also important, as nodes represent specific instances (e.g.,
individual people), which may be less informative for answering general or class-
level questions—such as those referring to a group or category rather than a
single entity.
3.3 Retrieval Phase
The retrieval phase consists of three key steps:
Node-Level Semantic Matching: We first compute the cosine similarity be-
tween the embedding of the input question and the embeddings of candidate
nodes in the Knowledge Graph. Nodes with the highest similarity scores are
selected along with their respective immediate relationships.
Type-Level Generalization Retrieval: In the second step, we identify the node
type that is most semantically similar to the input question based on type em-
beddings. Once the top-matching node type is found, all nodes and relationships
International Semantic Intelligence Conference (ISIC), Germany, 20254

associated with that type are selected. This enables the retrieval of more gen-
eralized or abstract knowledge, especially useful when the question refers to a
category (e.g., "scientists") rather than a specific instance.
Fuzzy Entity Matching: Finally, we identify entities explicitly mentioned in
theinputquestionusingalightweightHuggingFacemodeldslim/bert-base-NER.
Then fuzzy matching is performed against the graph nodes. We allow an edit
distance of up to three, enabling the system to account for minor variations or
misspellings in entity names during retrieval.
3.4 Paraphrase Phase
Following the retrieval, the selected nodes and their associated relationships are
passed to a lightweight paraphrasing model. This model generates a fluent nat-
ural language response by synthesizing the retrieved information into a coherent
and contextually appropriate answer. Thetuner007/pegasus_paraphraseis used
for paraphrasing, and experiments suggest that performance can be improved
with better-trained paraphraser models.
3.5 Reranker Phase
The paraphrased answers are subsequently passed to a HuggingFace reranking
model(BAAI/bge-reranker -large), which ranks them based on their semantic
relevance to the input question. We also experimented with the reranker model
before the Paraphrase phase and found no significant difference in the accu-
racy. The top five highest-ranked responses are selected as the final answers for
evaluation.
4 EVALUATION METHODOLOGY AND RESULTS
Thissectionpresentstheevaluationresultsobtainedusingautomatedassessment
via an LLM-as-a-judge [11].
4.1 LLM-as-a-Judge Evaluation
We employed two large language models, Llama-3.2 and GPT-3.5 Turbo, as au-
tomaticevaluatorstoassessthecorrectnessofthegeneratedanswers,asshownin
Fig. 2. Specifically, for each original question used in constructing the Knowledge
Graph, we prompted the LLM to determine whether the ground-truth answer
was present among the top five candidate answers produced by our system. We
used a lenient prompt because a lightweight paraphraser was used in the answer
generation, which can sometimes result in some broken English. Using this ap-
proach, the system achieved an accuracy of89.6%by Llama-3.2 and 78.3% by
gpt-3.5-turbo for our PDF dataset. The high accuracy shows that the retrieval
from the knowledge graph performed well.
International Semantic Intelligence Conference (ISIC), Germany, 20255

For the CRAG dataset, we had 2706 QA pairs; however, 308 pairs had answer
fields marked as "invalid question." Thus, valid questions were 2398. The accu-
racy of the CRAG dataset is 71.9% by Llama-3.2 and 54.4% by GPT-3.5-turbo.
This is in comparison to the reported accuracy of 63% [10]. We also analyzed
a sample of incorrect questions and found that entity mismatch was the pri-
mary reason. The experiments demonstrate that enhancing entity-relationship
extraction can further enhance performance on the CRAG dataset. A summary
of these results is shown in Table 2
prompt = f"""
You are a generous evaluator. Your goal is to determine whether the **
predicted answer reflects the intended meaning** of the expected
answer, even if the wording is different or the match is only partial
.
- You should be lenient: allow paraphrases, synonyms, and generalizations
.
- The predicted answer does not need to match all details, just the key
idea.
- If the main meaning of the expected answer appears clearly or
recognizably in the predicted answer-even in different words-consider
it a match.
- Do not accept answers that are too vague or only weakly related.
- Ignore unrelated or repetitive sentences.
Query:
{row[queryCol]}
Expected answer (main idea to be reflected):
{row['expected']}
Predicted Answer:
{row[predictedCol]}
Does the predicted answer contain the **main idea or intent** of the
expected answer in any form (direct, indirect, partial, or
paraphrased)?
Respond only with "Yes" or "No".
"""
Fig. 2.Prompt used for LLM-as-a-judge
International Semantic Intelligence Conference (ISIC), Germany, 20256

Table 2.Accuracy of KG-based QA System under Different Evaluation Settings
Questions Asked Llama
3.2 Ac-
curacyGPT-3.5
Turbo
Accu-
racy
QAs generated at runtime using Hugging Face’s model (set of
194)89.6% 78.3%
QAs generated at runtime using Hugging Face’s model and later
perturbed (set of 194)85.6% 72.6%
QAs from standard CRAG dataset (set of 2700) 68.6% 50%
4.2 Robustness Evaluation via Question Perturbation
To assess the robustness and generalization capabilities of the system, the ques-
tions were perturbed while preserving their original intent using Llama 3.2. The
modifiedquestionswerethenre-evaluatedtodeterminewhetherthesystemcould
still retrieve correct answers. One of the examples of the perturned QA is shown
in Fig. 3. The accuracy of the perturbed questions on the PDF dataset is 85.6%
with Llama-3.2 and 72.6% with gpt-3.5-turbo.
Original Question: Which of the following is NOT an Employer's Risk under
Clause 3.1?
Answer: Contractor's Negligence is not the employer's risk.
Perturbed Question: Which is not the responsibility of the employer
according to Clause 3.1?
Answer: Contractor's Negligence is not the employer's risk. (Correct)
Fig. 3.Perturbation Sample
5 CONCLUSION
This research demonstrates a knowledge graph-based approach for question an-
swering. The pipeline combines entity and semantic search with reranking and
paraphrasing.Thesystemispracticallyusefulasitprovidesinterpretable,ranked
responses that include partially correct or contextually relevant information,
making it valuable for exploratory QA, entity discovery, and knowledge valida-
tion. Future directions include fine-tuned QA pair generation and richer relation
extraction. The code for the experiment, methods, and dataset described in this
paper is publicly available at https://github.com/kartikeyaaneja/GraphRAGDi
rectAnswer.
International Semantic Intelligence Conference (ISIC), Germany, 20257

Bibliography
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küt-
tler, M. Lewis, W.-t. Yih, T. Rocktäschelet al., “Retrieval-augmented gen-
erationforknowledge-intensiveNLPtasks,”Advances in neural information
processing systems, vol. 33, pp. 9459–9474, 2020.
[2] Z. Xu, S. Jain, and M. Kankanhalli, “Hallucination is inevitable: An in-
nate limitation of large language models,”arXiv preprint arXiv:2401.11817,
2024.
[3] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang,
A. Madotto, and P. Fung, “Survey of hallucination in natural language gen-
eration,”ACM computing surveys, vol. 55, no. 12, pp. 1–38, 2023.
[4] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[5] J. Larson and S. Truitt, “Graphrag: Unlocking llm discovery on narrative
private data,” Microsoft Research Blog, Feb 2024. [Online]. Available:
https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm
-discovery-on-narrative-private-data/
[6] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, and
S. Tang, “Graph retrieval-augmented generation: A survey,”arXiv preprint
arXiv:2408.08921, 2024.
[7] Y. Gu, V. Pahuja, G. Cheng, and Y. Su, “Knowledge base question an-
swering: A semantic parsing perspective,”arXiv preprint arXiv:2209.04994,
2022.
[8] H. Du, Q. Huang, C. Li, C. Zhang, Y. Li, and D. Zhao, “Relation-aware
question answering for heterogeneous knowledge graphs,”arXiv preprint
arXiv:2312.11922, 2023.
[9] Y. Zhao, J. Huang, W. Hu, Q. Chen, X. Qiu, C. Huo, and W. Ren, “Implicit
relationlinkingforquestionansweringoverknowledgegraph,” inFindings of
the Association for Computational Linguistics: ACL 2022, 2022, pp. 3956–
3968.
[10] X. Yang, K. Sun, H. Xin, Y. Sun, N. Bhalla, X. Chen, S. Choudhary, R. Gui,
Z. Jiang, Z. Jianget al., “CRAG-comprehensive rag benchmark,”Advances
in Neural Information Processing Systems, vol. 37, pp. 10470–10490, 2024.
[11] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin,
Z. Li, D. Li, E. Xinget al., “Judging llm-as-a-judge with mt-bench and
chatbot arena,”Advances in neural information processing systems, vol. 36,
pp. 46595–46623, 2023.
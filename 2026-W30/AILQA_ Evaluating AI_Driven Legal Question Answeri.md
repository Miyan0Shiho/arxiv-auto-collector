# AILQA: Evaluating AI-Driven Legal Question Answering Systems for the Indian Legal System

**Authors**: Shubham Kumar Nigam, Shubham Kumar Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya

**Published**: 2026-07-21 08:01:17

**PDF URL**: [https://arxiv.org/pdf/2607.18825v1](https://arxiv.org/pdf/2607.18825v1)

## Abstract
This comprehensive study introduces an advanced Artificial Intelligence for Indian Legal Question Answering (AILQA) system tailored to the Indian legal context. AILQA leverages a variety of embedding and generative models, including recent Large Language Models (LLMs), to address the unique challenges posed by the intricate and diverse nature of Indian legal texts and to enhance the accuracy and reliability of responses to legal questions. We conducted rigorous evaluations using both lexical and semantic metrics, enriched by expert legal feedback, to ensure relevance and accuracy. Our findings underscore the effectiveness of the Retrieval-Augmented Generation (RAG) paradigm in improving answer quality, particularly in complex legal domains. Additionally, we assessed performance on standardized tests such as the All India Bar Examination (AIBE), thereby providing a robust benchmark for practical applications. Under the study's evaluation protocol, some AI-generated responses received higher ratings than the available reference answers, particularly when they contained accurate and relevant supporting details. This finding is specific to the evaluated dataset and rating criteria and should not be interpreted as evidence that the models generally outperform qualified legal professionals. We also discuss the challenges encountered, such as the need for precise context and the risks of model hallucination, and propose directions for future research to further refine AI capabilities in the legal field. This study aims to pave the way for enhanced legal decision-support systems, making them more accessible and effective for legal professionals and the public alike.

## Full Text


<!-- PDF content starts -->

AILQA: Evaluating AI-Driven Legal Question
Answering Systems for the Indian Legal System
Shubham Kumar Nigam1,4*†, Shubham Kumar Mishra1†,
Noel Shallum2, Kripabandhu Ghosh3, Arnab Bhattacharya1
1*Computer Science and Engineering, Indian Institute of Technology, Kanpur,
208016, Uttar Pradesh, India.
2Law, Symbiosis Law School, Pune, 411014, Maharashtra, India.
3Computational and Data Sciences Department, Indian Institute of Science
Education and Research, Kolkata, 741246, West Bengal, India.
4School of Computer Science, University of Birmingham, Dubai, UAE.
*Corresponding author(s). E-mail(s): s.k.nigam@bham.ac.uk;
Contributing authors: skmishra20@cse.iitk.ac.in; noelshallum@gmail.com;
kripaghosh@iserkol.ac.in; arnabb@cse.iitk.ac.in;
†These authors contributed equally to this work.
Abstract
This comprehensive study introduces an advancedArtificial Intelligence forIndianLegal
QuestionAnswering or AILQA system tailored for the Indian legal context. AILQA leverages
a variety of embedding and generative models, including the latest Large Language Models
(LLMs), to address the unique challenges posed by the intricate and diverse nature of Indian
legal texts, to enhance the accuracy and reliability of legal question responses. We conducted
rigorous evaluations using both lexical and semantic metrics that are enriched by expert legal
feedback to ensure relevance and accuracy. Our findings underscore the effectiveness of the
Retrieval-Augmented Generation (RAG) paradigm in improving answer quality, particularly
in complex legal domains. Additionally, we explored the performance on standardized tests
such as the All India Bar Exam (AIBE), thus providing a robust benchmark for a practical
application. Under the study’s evaluation protocol, some AI-generated responses received
higher ratings than the available reference answers, particularly when they contained
accurate and relevant supporting detail. This finding is specific to the evaluated dataset
and rating criteria and should not be interpreted as evidence that the models generally
outperform qualified legal professionals. We also discuss the challenges encountered, such
as the need for precise context and the risks of model hallucination, and propose directions
for future research to further refine AI capabilities in the legal field. This study aims to pave
1
arXiv:2607.18825v1  [cs.CL]  21 Jul 2026

the way for enhanced legal decision-making support systems, making them more accessible
and effective for legal professionals and the public alike.
Keywords:Legal Question Answering, Retrieval Augmented Generation (RAG), Large Language
Model (LLM), Embedding Model, QA Model, Indian Legal Domain, Legal Expert Rating
1 Introduction
Question Answering (QA) is an artificial intelligence (AI) task that utilizes Natural Language
Processing (NLP) to interpret and respond to queries in natural language, mimicking human
interaction (Allam and Haggag, 2012; Choi et al., 2018). Recent advancements in deep learning
technologies, particularly with models like Generative Pretrained Transformer 3 (GPT-3)
and BERT, have significantly enhanced the capabilities of QA systems in extracting relevant
information from large, unstructured datasets (Devlin et al., 2018; Qu et al., 2019; Wang
et al., 2019; Kassner and Sch ¨utze, 2020). These systems are increasingly utilized across
diverse domains such as healthcare, customer service, and education, leading to substantial
improvements in information processing efficiency and service delivery.
In the legal domain, the implementation of QA systems has provided innovative solutions,
especially in countries with advanced AI infrastructures like the USA, UK, and Brazil. In these
jurisdictions, legal QA systems have been integrated into court proceedings and public legal
services, aiding in case summarization, evidence assessment, and even predictive judgments.
For instance, in the USA, several states are experimenting with AI to assist in minor traffic
violation adjudications and initial hearing assessments King et al. (2020). Similarly, in the UK,
AI is being piloted to navigate legal documents and suggest relevant prior cases to lawyers and
judges Collenette et al. (2023); Drake et al. (2022), thereby streamlining legal research efforts.
Despite these advancements, the adoption of AI in the Indian legal system remains in a
rudimentary stage. Unlike its Western counterparts, where AI-driven systems are gradually
becoming integral to legal operations, India has yet to fully embrace the potential of AI
technologies in the judiciary and legal education. The complex legal landscape of India,
characterized by a diverse amalgamation of civil, criminal, common, and customary laws,
presents unique challenges that have slowed AI integration. Furthermore, the use of English
as the primary language for higher judiciary and legal documentation adds another layer of
complexity, making the development of effective QA systems more challenging.
Our study aims to bridge this gap by exploring the application of QA models specifically
designed for the Indian legal domain. By focusing on criminal cases processed in English
due to resource constraints, we aim to demonstrate the potential of AI to transform legal QA
within the Indian context. This research not only addresses the technical challenges but also
highlights how such innovations could benefit the Indian judiciary and law students alike by
providing enhanced access to legal information and fostering a more efficient legal process.
Figure 1 illustrates the comparative effectiveness of AI in providing legal advice. A user
seeking advice on online threats receives distinctly different responses from a lawyer and an AI
agent. The lawyer provides a brief, to-the-point response, advising the user to file a complaint
under cyber laws and consider police assistance. Meanwhile, the AI agent offers a more
detailed solution, referencing Section 507 of the Indian Penal Code for criminal intimidation
2

UserSomeone has been threatening
me online. What can I do?
Dear Client,File a complaint under 
cyber laws. You may also 
need police assistance.
You can file a complaint under 
Section 507  of the Indian Penal Code  for 
criminal intimidation... Additionally , 
under the IT Act, 2000 , cyber threats are 
punishable... You should collect 
evidence such as screenshots and file 
a report with the cybercrime cell.
LAWYER AI AGENTFig. 1Comparison of a concise reference response and a more detailed AI-generated response to an online-threat
query. The example illustrates differences in response detail and structure; greater detail should not, by itself, be
interpreted as greater legal correctness.
and the IT Act, 2000 for cyber threats, along with practical steps like gathering evidence and
reporting to the cybercrime cell. While the lawyer’s response is concise and less descriptive.
However, with the right knowledge base and well-constructed prompts, AI has the potential to
offer legal advice of consistent quality, comparable to that given by an experienced lawyer.
This suggests that AI can help standardize legal responses, ensuring they are as thorough and
accurate as human-provided advice, regardless of the lawyer’s approach.
The contributions of our study are significant, as they pave the way for the introduction
of sophisticated AI tools in an environment where such technological advancements are yet
nascent. The following specific model combinations and evaluation strategies employed in our
research illustrate the potential of AI to exceed the capabilities of traditional legal analysis tools
in some scenarios, thereby setting a precedent for further AI adoption in Indian legal practices:
•Introduction of RAG in Indian Legal Question-answering:We pioneer the application
of the retrieval-augmented generation (RAG) paradigm for the Indian legal system. This
novel contribution involves developing and integrating RAG systems specifically tailored
to handle the complexities of India’s diverse legal framework. By enabling the model to
dynamically retrieve and utilize relevant past legal cases and statutes as part of the answer
generation process, we enhance the model’s ability to produce more accurate, contextually
relevant answers.
•Dataset and Code Availability:We contribute to the research community by making the
AILQA dataset and prediction models publicly available, thereby promoting transparency
and reproducibility in legal AI research.
•Exploration of Embedding Models:We investigate various combinations of embedding
and QA models tailored for legal question answering, and demonstrate their effectiveness in
the Indian legal domain.
•Comprehensive Evaluation Methodology:We establish a robust evaluation framework
that incorporates both lexical and semantic metrics, along with expert legal feedback, to
assess the quality of generated answers.
3

•Automated Evaluation Framework:We implement an LLM-based auto-evaluation method
to augment human expert judgment and, thus, significantly expedite the evaluation process
while maintaining accuracy.
•Statistical Analysis:We perform rigorous statistical tests to validate the performance of
different models and their outputs, to validate the experimental results statistically.
Additionally, we analyze the impact of retrieval-augmented generation (RAG) on answering
various types of questions from our legal dataset, by visualizing the results through histograms.
By providing concrete examples of hallucinations in generative models such as LLaMA2-70b
and GPT-3.5 Turbo, we aim to better understand the limitations of these models in the legal
domain. To support further research and ensure reproducibility, we have made the AILQA
dataset and the code for our prediction and explanation models publicly available via an
GitHub link1.
2 Related Work
In recent years, significant improvements in question-answering (QA) systems have been
driven by advances in machine learning models and natural language processing techniques.
Introducing models like BERT (Bidirectional Encoder Representations from Transformers)
(Devlin et al., 2018) and GPT (Generative Pre-trained Transformer) (Floridi and Chiriatti, 2020)
has revolutionized the field, enabling systems to understand and process natural language with
unprecedented accuracy. Studies such as (Devlin et al., 2018) and (Radford et al., 2019) have
demonstrated the effectiveness of these models in general QA tasks across various domains,
setting a new standard for AI-driven interaction.
Following these foundational models, researchers have explored specific adaptations and
enhancements to tailor QA systems for specialized applications. For instance, Yang et al. (2018)
introduced models that incorporate external knowledge bases to improve the contextuality of
answers in knowledge-intensive tasks.
The legal domain has been a prominent area of research for AI-driven solutions, particularly
for tasks such as Legal Judgment Prediction (LJP) and question answering. Legal Judgment
Prediction (Strickson and De La Iglesia, 2020; Xu et al., 2020; Feng et al., 2023) focuses on
predicting decisions of legal cases. Research has explored various machine learning models to
perform LJP across jurisdictions such as the European Union, China, and France (Xu et al.,
2020). In the Indian context, LJP has been attempted by developing specialized datasets like
the ILDC corpus, and utilizing hierarchical transformer models for judgment prediction (Malik
et al., 2021; Nigam et al., 2023, 2024). This work also emphasizes providing explanations
alongside predictions, an aspect crucial for transparency and trust in legal AI systems.
Recent advances have seen the application of transformer models to specific subdomains
of LJP, such as using only case facts for prediction (Nigam and Deroy, 2023). Additionally, the
performance of large language models (LLMs) on legal corpora like ILDC has been studied,
showing promising results for improving legal judgment prediction (Vats et al., 2023). Similar
approaches have been explored in other regions, such as Romania (Masala et al., 2021) and
Korea (Hwang et al., 2022), where country-specific legal corpora and benchmarks are used to
develop more specialized models.
1https://github.com/ShubhamKumarNigam/AILQA
4

Beyond LJP, question answering based on retrieval-augmented generation (RAG) has
gained traction for its ability to effectively combine retrieval mechanisms with generative
models to answer complex queries. One significant contribution in this domain is RAG-QA
Arena (Han et al., 2024), which evaluates domain robustness for long-form RAG-QA. The
authors present Long-form Robust QA (LFRQA), a dataset that integrates multiple short extrac-
tive answers into coherent, long-form narratives across seven different domains. RAG-QA
Arena further leverages model-based evaluators to benchmark the cross-domain generaliza-
tion of QA systems, offering a robust evaluation platform for future RAG-QA research. This
work is especially relevant for evaluating the effectiveness of generative models in com-
plex, multi-document scenarios and has shown that existing LLMs struggle to outperform
human-annotated long-form answers.
The dynamic nature of document relevance in RAG-based systems is further explored in the
recent work of (Hei et al., 2024), which introduces Dynamic-Relevant Retrieval-Augmented
Generation (DR-RAG). This two-stage retrieval framework aims to improve document retrieval
recall and the accuracy of answers by addressing the limitations of traditional RAG frameworks.
DR-RAG proposes a novel approach to mining relevance from both highly relevant (static)
documents and lower-relevance (dynamic) documents that may still be crucial to generating
accurate answers. By incorporating a classifier to optimize document retrieval and minimize
redundant information, DR-RAG achieves substantial improvements in multi-hop question
answering accuracy and recall. The study demonstrates the efficacy of DR-RAG across various
multi-hop QA datasets, showing enhancements in recall by 86.75% and improvements in
accuracy, exact match (EM), and F1 score by 6.17%, 7.34%, and 9.36%, respectively.
Moreover, the work of (Muludi et al., 2024) introduces the use of RAG combined with
GPT-3.5 for processing large external documents. This study focuses on using RAG to improve
the accuracy of document-based question answering by leveraging retrieval from external
knowledge sources to complement the generative capabilities of language models. The system
processes documents automatically and answers user queries by generating responses based
on the retrieved content. The study’s contributions include the creation of a dataset and per-
formance testing with the Stanford Question Answering Dataset (SQuAD). It demonstrated
the superiority of RAG, achieving significant improvements across various metrics such as
ROUGE, BERTScore, BLEU, and Jaccard Similarity. This advancement highlights the poten-
tial of RAG in real-world applications, especially in mitigating hallucinations and improving
the reliability of AI-driven question-answering systems.
Additionally, (Wiratunga et al., 2024) introduced CBR-RAG, a case-based reasoning
approach for retrieval-augmented generation in large language models for legal question
answering. This work combines the strengths of case-based reasoning and RAG to effectively
leverage past cases and external knowledge sources in generating accurate and relevant answers
to legal queries.
While these studies focus on legal judgment prediction, long-form question answering, and
improving retrieval strategies, fewer works address the challenge of query-based legal question
answering using explainable generative AI models. The task of answering legal questions
is closely related to LJP but requires additional capabilities to understand user queries and
retrieve relevant legal content.
5

Data Word Count (Avg.) No. of Documents
Judgements 4021 6942
Acts 28705 15
Articles 1557 264
Table 1Statistical overview of various Criminal Law document distributions
3 Dataset
3.1 Dataset Compilation
Our dataset is a robust aggregation of various legal documents essential for training and
evaluating the Legal Question Answering system. This collection encompasses statutory texts,
judicial decisions, and authoritative legal commentaries specific to the Indian jurisdiction. We
sourced statutory laws and regulations directly from the IndiaCode2, ensuring the inclusion of
all relevant acts listed in Table 2. Judicial opinions from the Supreme Court from the years 1947
to 2020 were meticulously compiled from IndianKanoon3, a comprehensive database offering
access to a wide range of legal documents. Furthermore, to enrich the dataset with diverse legal
discussions and analyses, we extracted articles and blogs from platforms such as Mondaq4and
LawyersClubIndia5. These sources provide contemporary interpretations and practical insights
into criminal law, enhancing the dataset’s relevance for current legal practices.
3.2 Dataset Statistics and Preprocessing
The dataset consists of approximately 7,221 legal documents, including both case law and
statutory materials. In the preprocessing phase, the dataset underwent rigorous cleaning
processes to ensure the integrity and usability of the data. This included the removal of non-
essential elements such as headers, footers, extraneous spaces, and line breaks, which could
interfere with the text processing algorithms. The cleaned documents were then analyzed to
provide statistical insights, as detailed in Table 3.2. This table presents a breakdown of the
document types, showcasing the diversity and volume of the content within the dataset, from
judicial rulings to legislative texts and insightful legal articles. These preprocessing steps were
crucial in standardizing the dataset for subsequent use in training and testing the AI models,
ensuring consistency and reliability in the data fed into our machine learning pipelines.
3.3 Test Data
To evaluate the performance of various answer generation and document retrieval models
within our legal QA system, we compiled two test datasets from the VidhiKarya website6. Test
Set 1 includes 50 legal queries, while Test Set 2 comprises 100 QA pairs. These datasets contain
legal queries with expert responses covering five key legal topics: Anticipatory Bail, Criminal
Law, Cyber Crime, Juvenile Issues, and Sex Crimes. In Test Set 2, each category contains 20
2https://www.indiacode.nic.in/
3https://indiankanoon.org/
4https://www.mondaq.com/5/India/Criminal-Law
5https://www.lawyersclubindia.com/articles/
6https://vidhikarya.com/free-legal-advice
6

QA pairs, randomly selected to represent the domain comprehensively. The answers provided
by legal experts on VidhiKarya serve as our ground truth, enabling a direct comparison between
the generated answers and expert responses. Each question is paired with its corresponding
expert answer, facilitating a straightforward evaluation of our models’ performance.
S. No. Act
1 Indian Penal Code
2 Protection of Children from Sexual Offences Act
3 Criminal Procedural Code
4 Indian Evidence Act
5 Arms Act
6 Information Technology Act
7 Narcotic Drugs and Psychotropic Substances Act
8 Contempt of Courts Act
9 Unlawful Activities Prevention Act
10 Prevention of Money Laundering Act
11 Criminal Procedure Identification Act
12 Extradition Act of 1962
13 Prisons Act of 1894
14 Prevention of Corruption Act of 1988
15 Gram Nyayalayas Act of 2008
Table 2List of Acts used as contextual data in the question-answering system.
3.3.1 All India Bar Exam (AIBE) Dataset
The All India Bar Exam (AIBE) dataset, published by Tiwari et al. (2024), serves as a critical
component to verify our methodology. This dataset encompasses questions and answers from
AIBE exams conducted over the past 12 years, from AIBE 4 to AIBE 16. It includes 1,158
multiple-choice questions covering various areas of law, which were manually verified for cor-
rectness. The questions primarily test recall abilities, with a few assessing legal reasoning skills.
The minimum passing percentage for this exam is set at 40%, providing a robust benchmark
for evaluating the effectiveness of our AI models in a standardized testing environment.
4 Methodology
4.1 System Overview
Our Legal Question Answering (QA) system employs an advanced Retrieval-Augmented
Generation (RAG) architecture that significantly enhances the capabilities of generative AI
models. This methodology integrates dynamic retrieval of contextual information from a
comprehensive legal database, which substantially augments the generative process. The
RAG system is specifically tailored to address the complex needs of legal QA by providing
precise, contextually relevant answers to legal queries and facilitating rigorous examination
7

Question Embedding Model
Text
DocumentsChunk
ChunkChunk
ChunkTop-K Relevant
Document Extracted
Split Text
Into ChunksCreate
Question
Embeddings
Create
Chunk
EmbeddingsChunkCheck for relevant document
chunks that are embedded
Question
+
 Relevant DocsGenerative Model
Write Prompt to generate
response for the question
based on the relevant docs Final Answer
Vector Store
(Store Chunk
Embeddings )Question Embedding
EmbeddingFig. 2Flowchart illustrating the Legal QA System, detailing the use of GPT-3 Ada and Instructor XL for context
extraction, and GPT-3 (Davinci), Flan-UL2, and Llama2-70B for response generation.
preparations for the legal bar exam. Figure 2 provides a visual representation of our QA
system’s workflow. This figure outlines how documents are stored in a database following
various preprocessing steps and how relevant document chunks are retrieved to answer user
queries or legal bar exam questions. When a question is posed, it is combined with the retrieved
chunk and passed to a generative model, accompanied by a defined prompt, to produce the
answer.
We have discussed the processes involved in making this architecture work for our tasks in
the sections below.
4.2 Chunking of Documents
To handle large legal documents efficiently within the computational limits of language models,
we implemented a chunking strategy. This approach divides extensive texts into manageable
pieces, each consisting of 2000 characters, with an overlap of 250 characters to maintain
narrative continuity. This method ensures that the input to our language models remains within
their token size limits and preserves the context necessary for generating accurate responses.
We utilized LangChain’s CharacterTextSplitter7for this process, optimizing the chunk
size to balance between model capacity and contextual completeness.
4.3 Embeddings Creation and Vector Store Database
In the previous section, we discussed how we divided large documents in the dataset into
smaller chunks of textual data. Now, we use embedding models to create embeddings for each
7https://python.langchain.com/v0.1/docs/modules/data connection/document transformers/character textsplitter/
8

chunk and store these embeddings in the form of a vector database. This approach is helpful in
retrieving the most contextually relevant chunks quickly and efficiently.
We have used ChromaDB8, an open-source database developed by Chroma, for storing and
using vector embeddings. The Langchain framework supports an easily integratable pipeline
for creating the vector database with the help of ChromaDB. The chunks obtained in the
previous section are passed along with the embedding model in the pipeline, and the path for
saving the embedding database is provided, allowing for seamless database creation at the
desired location.
For creating the embeddings, we utilized three models: Ada9by OpenAI, Instructor-XL
(Su et al., 2022), and Mxbai10. We chose these models as they ranked among the top embedding
models on the MTEB leaderboard11available on HuggingFace. Each model differed in size
and embedding dimension, leading to the creation of three distinct vector store databases.
This allowed us to test the quality of retrieval in our tasks. Here are some details about the
embedding models we used:
4.3.1 OpenAI’s Ada
The Ada model from OpenAI generates 1536-dimensional embeddings at a cost of $0.0004 per
1000 tokens. For our dataset of 61.6 million tokens, the total cost for embedding generation
was approximately $24.7. It can be accessed through the API token provided by OpenAI. The
documentation for usage can be found here12.
4.3.2 Instructor-XL
The Instructor-XL model produces 768-dimensional embeddings, optimized for instruction-
based embedding creation and retrieval tasks. It can generate text embeddings tailored to any
task (e.g., classification, retrieval, clustering, text evaluation, etc.) or domain (e.g., science,
finance, etc.) by simply providing the task instruction in natural language. It is available on the
HuggingFace platform13.
4.3.3 Mxbai
Mxbai is an English embedding model with an embedding dimension of 1024. It is also
available on the Hugging Face platform14. After creating the embeddings, the chunks are
stored with it’s metadata and unique id in the vector database.
4.4 Query Processing and Document Retrieval
As discussed in the previous section, the chunked data is stored in a vector database. To retrieve
relevant chunks for our question-answering tasks, we utilize LangChain’s vector data similarity
search15pipeline. When a query is submitted, the question is converted into embeddings
8https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
9https://platform.openai.com/docs/guides/embeddings
10https://www.mixedbread.ai/blog/mxbai-embed-large-v1
11https://huggingface.co/spaces/mteb/leaderboard
12https://platform.openai.com/docs/api-reference/embeddings/create
13https://huggingface.co/hkunlp/instructor-xl
14https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
15https://python.langchain.com/v0.1/docs/modules/data connection/vectorstores/
9

using the same model employed during database creation. The similarity search pipeline then
performs a cosine similarity search to identify the most relevant document chunks. In our setup,
the top three chunks, based on cosine similarity scores, are used as the context for answering
the question.
The implemented RAG architecture should be interpreted as a controlled dense-retrieval
baseline rather than a fully optimised contemporary legal RAG pipeline. It relies on fixed-
size character chunks, embedding-based cosine similarity, and top-k retrieval without hybrid
lexical–semantic retrieval, metadata filtering, query reformulation, or a dedicated reranking
stage. This deliberately simple configuration enables a controlled comparison of embedding
and generative models, while also allowing us to examine the conditions under which retrieved
context improves or degrades legal answer generation.
4.5 Answer Generation
Once the relevant context has been retrieved, as described in the previous section, we pass
the question along with the context to the generative model to produce a pertinent response.
To ensure the quality of the generated answers, we have crafted specific prompts that guide
the generative model in generating high-quality responses. We experimented with different
models, including Davinci (text-davinci-003)16, Llama2-70B (Touvron et al., 2023), Flan-UL2
(Wei et al., 2021), GPT-3.5 Turbo17, Llama3-70B (AI@Meta, 2024) and Mixtral-8x7B (Jiang
et al., 2024), and fine-tuned prompts to compare the quality of the responses.
Table 3 provides details on maximum token length, pricing, and the prompts used for each
model. The table shows that some prompts are tailored to specific models while others are
consistent across models. We iteratively refined the prompts by manually evaluating responses
to a random set of 10 questions. This iterative process allowed us to ensure the quality of the
prompts before applying them to our test data.
5 Evaluation Metrics
To ensure a comprehensive assessment of our question-answering system, we employed a
range of evaluation metrics. These metrics are designed to measure the accuracy, relevance,
and reliability of the answers generated by our system:
1.Lexical Similarity-Based Evaluation:We utilized ROUGE scores Lin (2004) (ROUGE-1,
ROUGE-2, and ROUGE-L) and the BLEU Score Papineni et al. (2002) to measure the
lexical similarity between the generated answers and the reference answers. These metrics
evaluate the overlap of n-grams and the sequence of words, providing a quantitative measure
of the linguistic quality of the generated text.
2.Semantic Similarity Based Method:To assess the semantic accuracy of the responses, we
employed the MPNET base v2 Song et al. (2020a) sentence transformer model from Hug-
gingFace18. This model projects sentences into a 768-dimensional vector space, enabling
us to perform detailed comparisons of semantic closeness between the generated answers
and the ground truth.
16https://platform.openai.com/docs/deprecations
17https://platform.openai.com/docs/models/gpt-3-5-turbo
18huggingface/sentence-transformers/all-mpnet-base-v2
10

Model Name Max Tokens Pricing Prompt
Davinci 4096 $0.02 / 1K
tokens“Your task is to answer a question as a legal assistant to the best of your
abilities, using the context given in the document. If the country is not
mentioned in the question, your response should be related to India. You
have knowledge of all laws and legal judgments of India. Be detailed
in your answer, provide relevant sections and case laws in your answer
only if you are confident that they are correct. Note that if you do not
know the answer, it is acceptable to say Sorry, I don’t know. Context:{}
Question:{}.”
Llama2-70B 4096 $0.65 / 1M
tokens“You are an honest legal advisor. Your task is to answer a question as a
legal assistant to the best of your abilities based on the context provided.
If the country is not mentioned in the question, your response should be
related to India. You have knowledge of all laws and legal judgments of
India. Be detailed in your answer, provide relevant sections and case laws
in your response only if you are confident that they are correct. If you
are unsure about an answer, truthfully say “I don’t know”. Context: {}
Question:{}”
Flan-UL2 2048 N/A “Answer the following question using the context by reasoning step
by step. If you don’t know the answer, just say Sorry, I don’t know.
Context:{}Question:{}”
GPT-3.5
Turbo4096 $0.0005 /
1K tokens“Your task is to answer a question as a legal assistant to the best of your
abilities, using the context given in the document. If the country is not
mentioned in the question, your response should be related to India. You
have knowledge of all laws and legal judgments of India. Be detailed in
your answer, provide relevant sections and case laws in your answer only
if you are confident that they are correct. Note that if you do not know
the answer, it is acceptable to say “Sorry, I don’t know.” Question: {}
Context:{}.”
Llama3-70B 8000 $0.65 / 1M
tokens“Your task is to answer a question as a legal assistant to the best of your
abilities, using the context given in the document. If the country is not
mentioned in the question, your response should be related to India. You
have knowledge of all laws and legal judgments of India. Be detailed in
your answer, provide relevant sections and case laws in your answer only
if you are confident that they are correct. Note that if you do not know
the answer, it is acceptable to say “Sorry, I don’t know.” Question: {}
Context:{}.”
Mixtral-8x7B 32K $0.50 / 1M
tokens“Your task is to answer a question as a legal assistant to the best of your
abilities, using the context given in the document. If the country is not
mentioned in the question, your response should be related to India. You
have knowledge of all laws and legal judgments of India. Be detailed in
your answer, provide relevant sections and case laws in your answer only
if you are confident that they are correct. Note that if you do not know the
answer, it is acceptable to say Sorry, I don’t know.” Question: {}Context:
{}.”
Table 3Specifications of Answer Generation Models: Max Tokens, Pricing, and Prompt Details
3.Expert Evaluation:Human evaluation was conducted by three legal evaluators who were
third- and fourth-year undergraduate law students enrolled at National Law Universities
in India. The evaluators possessed academic training in Indian law and were familiar with
legal research, statutory interpretation, and case-law analysis.
Each generated response was independently evaluated by all three evaluators using
a five-point Likert scale based primarily on legal accuracy, relevance, completeness, and
the appropriate use of statutory provisions or case law. The evaluators were provided
with the legal question, the corresponding reference answer, the generated response, and
11

information regarding the model identity and whether Retrieval-Augmented Generation
was used. Therefore, the evaluation was not conducted under blinded conditions.
After the independent evaluation, responses for which the evaluators assigned differing
ratings were discussed to establish a consensus score. Where disagreement remained, the
assessment was reviewed with a senior legal expert who is a practising legal professional,
and the final rating was determined through discussion and consensus. The resulting
consensus ratings were used in the reported expert-evaluation results.
The rating criteria were as follows:
[1]:The answer is entirely incorrect or fails to provide any answer.
[2]:The model misunderstood the question and did not offer a relevant response.
[3]:The answer is partly accurate but overlooks essential details.
[4]:A comparable, relevant answer to the ground truth.
[5]:The answer is legally accurate, directly relevant, sufficiently complete, and sup-
ported by appropriate statutory provisions or precedents where applicable. Additional detail
is rewarded only when it is correct, relevant, and useful; verbosity alone does not increase
the rating.
4.Statistical Comparison of Model Configurations:We conducted pairwise statistical
comparisons between the different experimental configurations using the per-question
semantic-similarity scores generated by the MPNET model. The resulting p-values were
used to assess whether the differences observed between model configurations were statisti-
cally significant. A p-value of 0.05 or lower was treated as evidence that the corresponding
performance difference was unlikely to have occurred by chance.
These statistical tests compare the semantic-performance distributions of the model
configurations. They should not be interpreted as measures of agreement or consistency
among the human legal evaluators.
5.Auto-Evaluation Framework:For an objective assessment of our system, we employed
an LLM-based auto-evaluation framework using LangChain’s load evaluator19. This
framework compares the generated answers to the ground truth using GPT-4 as the evalua-
tion model. It quantifies the relevance of the responses, where a score of 1 indicates high
relevance and 0 indicates high irrelevance.
These diverse metrics provide a robust framework for evaluating our system, ensuring that
it meets the rigorous standards required for effective application in the legal domain.
6 Results and Analysis
This section presents the outcomes of our experiments, which were organized into two distinct
phases to evaluate different combinations of generative and embedding models. These phases
helped us explore how different models interact and the resultant effects on the quality of
the generated answers, particularly focusing on the role of embedding models in enhancing
context extraction and the efficacy of generative models in producing accurate responses.
In Phase 1, we utilized Test Set 1, which comprised 50 legal questions, to conduct an initial
assessment of our model combinations. This set provided a preliminary understanding of how
each model performs under controlled conditions without extensive contextual diversity.
19LangChain Auto-Evaluation Documentation
12

Phase 2 expanded the evaluation to Test Set 2, consisting of 100 legal questions distributed
across five categories, as detailed in Section 3. This phase was designed to test the robustness
of the models in handling a broader array of legal questions and to assess the scalability of the
RAG architecture. Additionally, within Phase 2, we also assessed the performance of these
models on the Legal Bar Exam Dataset. This part of the evaluation was crucial for determining
the effectiveness of the RAG system in a highly specialized legal context.
The following sections provide detailed insights into the results, as well as the various
methods employed to understand the effectiveness of the RAG system in generating answers.
6.1 Results for Legal QA
Table 5 presents the performance evaluation of various generative models for legal question
answering, using a multifaceted approach. The column “Embedding Model” specifies the
embedding model employed to extract the context for each question. An empty column
indicates that the generative model produced the answer without any contextual help. Below,
we discuss the different metric scores presented in the table.
6.1.1 Lexical Based Evaluation
We employed ROUGE and BLEU scores to assess the lexical similarity between the generated
answers and the reference texts. These metrics provided insights into the precision of word
and phrase usage within the generated responses. In Table 5, for the Test Set 1 dataset, Davinci
achieves high scores with and without context for ROUGE-1, ROUGE-2, ROUGE-L, and
BLEU scores. The ROUGE and BLEU scores collectively indicate that the Ada model enhances
performance when used with Davinci compared to Instructor-XL.
Similarly, for the Test Set 2 dataset, the GPT-3.5 Turbo model achieves better ROUGE and
BLEU scores with context using Mxbai as the embedding model.
However, we cannot fully rely on these scores as they do not account for semantic meaning
or syntactic structure beyond surface-level word matching. Therefore, in subsequent sections,
we discuss other evaluation metrics that include semantic analysis and human evaluation to
provide a more comprehensive assessment.
6.1.2 Semantic Evaluation
For the semantic-based evaluation, we have used MPNET’s “all-mpnet-base-v2” (Song et al.,
2020b)20, which is a sentence-transformer model available on HuggingFace to get the similarity
scores based on the semantic similarity between generated answer and the ground truth. The
model used here is very capable in catching the semantic information of sentences and used in
variety of tasks including the sentence similarity task needed for getting similarity score for
our task.
For the Test Set 1,Llama2-70Bachieved the highest average MPNET score of0.611.
However, Llama2-70B did not perform as well when using context extracted by the Ada or
Instructor-XL models, with scores of 0.594 and 0.599, respectively. In contrast,Davinci, when
used without context, scored 0.561, and when used with context from Ada and Instructor-
XL, secured scores of 0.566 and 0.574, respectively. Lastly, theFlan-UL2model, regardless
20https://huggingface.co/sentence-transformers/all-mpnet-base-v2
13

Embedding Generative Rating Score
Model Model 1 2 3 4 5
- Davinci 0 9 13 20 8
- Llama2-70B 1 11 9 15 13
Ada Davinci 2 7 6 12 21
Instructor Davinci 2 7 11 15 15
Ada Llama2-70B 0 3 13 33 1
Instructor Llama2-70B 10 8 7 9 16
Ada Flan-UL2 11 33 5 1 0
Instructor Flan-UL2 5 36 9 0 0
Table 4Ratings from Legal Experts for Different Combinations of Embedding and Generative Models. A ‘-’ in the
‘Embedding Model’ column indicates that no embedding model was used to retrieve context for the corresponding
generative model.‘
of the embedding model used, did not show scores as high as Llama2-70B or Davinci. In
the case of the Test Set 2,Llama3-70Bindependently performed well, producing the most
semantically similar results compared to the ground truth. However, no model amongLlama3-
70B,Mixtral-8x7B, andGPT-3.5 Turboshowed a performance boost by using the context,
and generally, their performance degraded with context inclusion.
Using MPNET similarity scores, we quantitatively measured the semantic similarity
between generated answers and ground truth, providing insights into model performance with
and without contextual embedding. However, this approach lacks the nuanced understanding
and subjective judgement of human evaluation, which can more accurately capture the quality
and relevance of generated answers in a legal context. The limitations of relying solely on
automated similarity scores highlight the need for human evaluation to comprehensively assess
the performance of generative models in complex tasks like legal question answering.
6.1.3 Expert Evaluation
Legal experts reviewed the answers on a Likert scale from 1 to 5, focusing on accuracy
and relevance. The feedback indicated that certain models, especially those with advanced
generative capabilities like GPT-3 variants, often produced responses that met or exceeded the
quality of reference answers, demonstrating the potential of AI in legal expertise augmentation,
as detailed in Table 4 and average in Table 5.
6.1.4 Auto-Evaluation Framework
Finally, we utilized an LLM-based auto-evaluation framework to objectively assess the rel-
evance of the generated answers. This framework allowed for a scalable and consistent
evaluation, reinforcing findings from human expert reviews and semantic assessments. Table
6 compares model correctness across 100 evaluation questions. The Llama3-70B model out-
performed all others, both with and without the RAG, in producing answers comparable to
those of professional lawyers. However, its performance declined with RAG due to suboptimal
content extraction causing hallucinations. Conversely, the ‘Mixtral-8x7B’ model performed
better with RAG, suggesting that models with less prior knowledge benefit from additional
14

Embedding Generative Lexical Based Evaluation Semantic Evaluation Expert Evaluation
Model Model Rouge-1 Rouge-2 Rouge-L BLEU MPNET Score Rating Score
Results from Test 1 Dataset
- Davinci0.2670.0520.1580.010 0.561 3.54
- Llama2-70B 0.149 0.035 0.090 0.0070.6113.50
Ada Davinci 0.2420.0620.1470.0220.5663.74
Instructor-XL Davinci 0.229 0.053 0.139 0.016 0.574 3.68
Ada Llama2-70B 0.163 0.040 0.099 0.011 0.594 3.64
Instructor-XL Llama2-70B 0.160 0.037 0.094 0.008 0.599 3.26
Ada Flan-UL2 0.122 0.021 0.081 0.010 0.301 1.92
Instructor Flan-UL2 0.121 0.013 0.081 0.001 0.343 2.08
Results from Test 2 Dataset
- Llama3-70B 0.20 0.05 0.11 0.090.654.43
Mxbai Llama3-70B 0.22 0.06 0.12 0.1 0.63 3.37
- Mixtral-8x7B 0.19 0.05 0.11 0.09 0.624.59
Mxbai Mixtral-8x7B 0.23 0.06 0.13 0.12 0.62 4.02
- GPT-3.5 Turbo0.26 0.070.15 0.14 0.64 3.43
Mxbai GPT-3.5 Turbo0.26 0.07 0.16 0.160.62 3.55
Table 5This table compares different embedding and generative model combinations across various evaluation
metrics for Test 1 and Test 2 datasets. The highest scores in each metric are highlighted in bold. A ‘-’ in the
‘Embedding Model’ column indicates that no embedding model was used to retrieve context for the corresponding
generative model.
context. Larger models, already trained on extensive data, tend to hallucinate with added con-
text, reducing performance.
The LLM-based auto-evaluation framework proved efficient and scalable, reducing the need
for human evaluation. It provided consistent scoring and the results highlighted that smaller
models benefit from RAG, while larger models may hallucinate with added context.
Model RAG Questions with Score 1
LLAMA3-70B NO79
GPT-3.5-Turbo NO 77
LLAMA3-70B YES 75
Mixtral-8x7B YES 73
Mixtral-8x7B NO 71
GPT-3.5-Turbo YES 70
Table 6Table summarizing the auto-evaluated scores by GPT-4 for various models. The table records instances
where models achieved a perfect score of 1, indicating complete alignment with the ground truth. The columns list the
number of questions where each model scored 1, while the rows categorize the models with or without RAG system.
6.1.5 Statistical Significance Scores
Table 7 shows comparative analysis P-values for pairwise statistical comparisons between
different experimental settings based on MPNET similarity scores for Test Dataset 1. The
table is symmetric across the diagonal, hence representing in a lower triangular format. The
meaningful data (in this case, p-values) are only present in the lower half of the table, below the
main diagonal. The main diagonal and the upper half of the table (above the main diagonal) are
15

filled with placeholder symbols “-”. This means that the comparison of Model A vs. Model B
will have the same p-value as Model B vs. Model A, yielding the same statistical significance
regardless of the comparison order.
Similarly, Table 8 presents the p-values for Test Dataset 2, following the same format. The
“+” symbol in both tables indicates the use of Retrieval-Augmented Generation (RAG) for
generative models, utilizing embeddings from the model specified after the “+” sign.
•Highly Similar Models:Several comparisons in both tables, such as ‘Ada+Flan-UL2’ vs.
‘Ada+Llama2-70B’ in Table 7, and ‘Llama3-70B+Mxbai’ vs. ‘Mixtral-8x7B+Mxbai’ in
Table 8, show p-values close to 0.0000, indicating extremely high statistical significance.
This reflects substantial differences in model performance across different settings, especially
when RAG is used with certain models.
•Marginally Significant Comparisons:Table 7 has a few comparisons with p-values
slightly above 0.05, such as ‘Instructor+Llama2-70B’ vs. ‘Ada+Davinci’ with a p-value of
0.1333, indicating less pronounced differences between models. Similarly, in Table 8, the
comparison ‘Mixtral-8x7B+Mxbai’ vs. ‘GPT-3.5 Turbo+Mxbai’ has a p-value of 0.6591,
suggesting a closer similarity in their performance when using RAG.
•High P-values:In both tables, some comparisons yield very high p-values, indicating that
the differences between models are not statistically significant. For example, in Table 7,
‘Ada+Llama2-70B’ vs. ‘Davinci’ has a p-value of 0.7206. Similarly, in Table 8, ‘GPT-3.5
Turbo+Mxbai’ vs. ‘Llama3-70B’ shows a p-value of 0.1979. These high p-values suggest
that these models have similar MPNET similarity scores, and their performance may be
considered equivalent.
•Diversity in Model Performance:The wide range of p-values across both tables highlights
the diversity in model performance. Some models, especially when enhanced with RAG,
show significant differences in capabilities, while others demonstrate closer performance in
generating legal answers, depending on the configuration and datasets.
•Importance of Context:Context remains critical, particularly in legal domains. In both
tables, even marginally significant p-values (like 0.0666 for ‘Llama3-70B+Mxbai’ vs.
‘Mixtral-8x7B+Mxbai’ in Table 8) can suggest performance variations that might be vital in
specific legal applications. This reinforces the importance of using embeddings and RAG
for certain legal scenarios.
•Variability in Legal Answering Capabilities:Both tables reflect the variability in how
these models answer legal questions. Some models show clear performance differences,
while others are more closely aligned, depending on whether or not RAG is used, as shown
by varying p-values across datasets.
6.2 Analysis of Model Performance using Histograms
To better understand the impact of Retrieval-Augmented Generation (RAG) on different types
of legal questions in Test Set 2, we created a histogram comparing model performance with
and without RAG. This comparison, as shown in Figure 3, presents expert-evaluated scores for
various question types across different models.
For the GPT-3.5 Turbo model, the histogram highlights improved scores with RAG for
question types such as anticipatory bail, criminal, and juvenile cases, with a distinct shift
towards higher scores. However, in cybercrime and sex crime questions, RAG provided useful
16

Davinci Llama2-70B Ada+DavinciInstructor+
DavinciAda+
Llama2-70BInstructor+
Llama2-70BAda+
Flan-UL2
Llama2-70B 0.0786 - - - - - -
Ada + Davinci 0.2527 0.0366 - - - - -
Instructor + Davinci 0.4387 0.0948 0.4596 - - - -
Ada + Llama2-70B 0.7206 0.1237 0.2089 0.3715 - - -
Instructor + Llama2-70B 0.4678 0.2900 0.1333 0.2627 0.7035 - -
Ada + Flan-UL2 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 -
Instructor + Flan-UL2 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.2451
Table 7Comparative analysis of p-values for pairwise statistical comparisons between different generative models
with and without Retrieval-Augmented Generation (RAG) on Test Dataset 1. In the table,“+” denotes the use of RAG,
where the generative model output incorporates embeddings from the specified model following the “+” sign.
P-values are calculated based on MPNET similarity scores.
Llama3-70B Llama3-70B + Mxbai Mixtral-8x7B Mixtral-8x7B + Mxbai GPT-3.5 Turbo
Llama3-70B + Mxbai 0.0101 - - - -
Mixtral-8x7B 0.0169 0.5567 - - -
Mixtral-8x7B + Mxbai 0.0005 0.0666 0.5057 - -
GPT-3.5 Turbo 0.1979 0.4308 0.2851 0.0275 -
GPT-3.5 Turbo + Mxbai 0.0075 0.4009 0.8119 0.6591 0.0809
Table 8Comparative analysis of p-values for pairwise statistical comparisons between different generative models
with and without Retrieval-Augmented Generation (RAG) on Test Dataset 2. In the table, “+” denotes the use of RAG,
where the generative model output incorporates embeddings from the specified model following the “+” sign.
P-values are calculated based on MPNET similarity scores.
context in some instances, resulting in scores above 3, but also introduced irrelevant answers
in certain cases, leading to lower scores below 3.
The Llama3-70B model showed generally strong performance across all question types
when used without RAG. However, with RAG, the scores mostly clustered around 3 and 4.
Notably, in anticipatory bail questions, the model’s performance slightly declined, with fewer
high scores and some falling below 3.
For the Mixtral-8x7B model, the histogram indicates an overall improvement in perfor-
mance with RAG across all question types. This improvement was particularly evident in
criminal and cybercrime questions, where the number of questions scoring 4 increased sig-
nificantly. Additionally, there was a notable rise in the number of questions achieving the
maximum score of 5, especially in sex crime and juvenile cases.
These results provide important insights into how RAG affects model performance across
different legal question types, highlighting both the strengths and challenges of using this
method in legal question-answering tasks.
Our qualitative examination suggests several reasons for the observed degradation with
RAG. First, semantic similarity retrieval may return passages that share terminology with
the query but concern a legally distinct issue. Second, fixed-size chunking can separate a
statutory rule from its exceptions, definitions, or procedural conditions. Third, retrieving the
same number of chunks for every query may introduce unnecessary context for questions that
the model can already answer reliably. Finally, the absence of reranking or legal metadata
constraints can result in passages from less relevant statutes, cases, or factual settings being
included in the prompt. These failures can distract the generative model and lead to answers
that are fluent but insufficiently grounded in the applicable legal context.
17

1 2 3 4 5
Score02468101214Anticipatory_bail
Without RAG
With RAG
1 2 3 4 5
ScoreCriminal
1 2 3 4 5
ScoreCyber_crime
1 2 3 4 5
ScoreJuvenile
1 2 3 4 5
ScoreSex_crimeNumber of QuestionsNumber of Questions VS Score Plot For GPT-3.5-Turbo
1 2 3 4 5
Score02468101214Anticipatory_bail
Without RAG
With RAG
1 2 3 4 5
ScoreCriminal
1 2 3 4 5
ScoreCyber_crime
1 2 3 4 5
ScoreJuvenile
1 2 3 4 5
ScoreSex_crimeNumber of QuestionsNumber of Questions VS Score Plot For LLAMA-70b
1 2 3 4 5
Score024681012Anticipatory_bail
Without RAG
With RAG
1 2 3 4 5
ScoreCriminal
1 2 3 4 5
ScoreCyber_crime
1 2 3 4 5
ScoreJuvenile
1 2 3 4 5
ScoreSex_crimeNumber of QuestionsNumber of Questions VS Score Plot For MIXTRAL-8x7bFig. 3Comparison of legal models (GPT-3.5 Turbo, LLAMA-70B, and MIXTRAL-8x7B) across case types (Antic-
ipatory Bail, Criminal, Cybercrime, Juvenile, and Sex Crime) with and without Retrieval-Augmented Generation
(RAG). Bar plots show the number of questions per score, highlighting model performance and the impact of RAG.
6.3 Results on Legal Bar Exam Dataset
To evaluate the effectiveness of the RAG architecture with LLM models, we tested it on
the legal bar exam dataset. The architecture received the questions with options and the
corresponding extracted context to answer them. The output was the correct option for each
question.
Table 9 presents a comparative performance analysis of different models with and without
RAG. The “Total Correct” column shows the number of correct answers out of 1158 question
samples. We observed an increase in correctness for Mixtral 8x7B, Llama2-70B, and Llama3-
70B models when RAG was applied. Notably, the Llama2-70B model exhibited a significant
improvement, with correctness increasing by 5.97%.
18

These results suggest that the RAG architecture can effectively enhance performance
even with smaller or older models, without the need for any fine-tuning. In contrast, the fine-
tuning method used in the paper Tiwari et al. (2024) did not yield substantial improvements
for the legal bar exam dataset, with the Aalap model achieving only 25.56% correctness.
This demonstrates the potential of the RAG architecture to outperform traditional fine-tuning
approaches in specific tasks.
Although the reported results demonstrate the potential of LLM-based legal QA, they
do not establish that the system is sufficiently reliable for autonomous legal advice or legal
decision-making. Performance on the AIBE dataset and expert-rated question-answer pairs
evaluates specific capabilities under controlled conditions and does not capture all requirements
of professional legal practice, including factual investigation, procedural strategy, jurisdictional
variation, changes in law, and responsibility for consequential advice. Accordingly, AILQA
should presently be regarded as a research prototype and decision-support system whose
outputs require verification by qualified legal professionals.
Model RAG Correct Accuracy (%)
GPT-3.5 Turbo Yes 679 58.69
GPT-3.5 Turbo No 680 58.72
Mixtral 7x8B No 673 58.17
Mixtral 7x8B Yes 681 58.86
Llama2-70B No 529 45.72
Llama2-70B Yes 598 51.69
Llama3-70B No 822 71.05
Llama3-70B Yes823 71.13
Table 9Comparison of Model Performance on the Legal Bar Exam Dataset with and without RAG.
6.4 Examples of Legal Question Answering with RAG
This section presents examples from our test dataset to show how generative AI models and
traditional legal advice compare. Table 10 in the Appendix displays responses from different
AI models (Mixtral-8x7B, GPT-3.5 Turbo, Llama2-70B) and a human lawyer for various legal
questions.
In the table,“...” means that parts of the answers have been shortened to keep the table
concise. The full answers are longer and more detailed. We haven’t included the full context
for each question in the table to avoid making it too large. These examples highlight how
generative models with retrieval-augmented generation (RAG) can provide clearer and more
practical advice than traditional methods.
7 Hallucination
In generative AI models, especially those trained on vast, diverse datasets, “hallucinations”
refer to instances where the model generates factually incorrect or irrelevant content. This
is a significant challenge when the models are applied in domains requiring high accuracy
19

and reliability, such as legal question answering. In Table 11 in the Appendix, we provide
examples to illustrate the impact of context on the accuracy of answers and the mitigation of
hallucinations.
7.1 Influence of Context on Model Performance
The examples in Table 11 compare responses generated by our models with answers provided
by legal professionals from our ground truth dataset. It has been observed that responses
from legal experts, while accurate, often lack comprehensive detail and are sometimes overly
concise. Our AI models, equipped with well-designed prompts and carefully curated contextual
information, have the potential to generate more detailed and informative responses.
7.2 Balancing Context and Relevance
However, the integration of context into the generative process must be handled judiciously.
The table also demonstrates that while context generally enhances the quality of answers and
reduces the likelihood of hallucinations, it can also have the opposite effect if not properly
aligned with the query. Irrelevant or excessively detailed context can confuse the model,
leading to responses that are off-topic or factually incorrect.
7.3 Optimizing Contextual Information
To avoid inducing hallucinations, it is crucial to provide context that is both relevant and
concise. The quality of context directly influences the model’s ability to generate accurate and
applicable answers. This involves not only selecting the right fragments of text to serve as
context but also tuning the model to prioritize and weigh the given information effectively.
7.4 Strategic Prompt Design
Strategic prompt design also plays a pivotal role in guiding the model’s focus and filtering out
unnecessary details. By carefully crafting prompts that direct the model’s attention to the most
pertinent aspects of the provided context, we can further reduce the risk of hallucinations and
enhance the relevance of the responses.
7.5 Implications for Model Training and Deployment
These insights are crucial for refining the training and deployment strategies of AI models in
legal settings. By understanding the conditions under which hallucinations are more likely to
occur, we can better prepare models to handle complex legal questions with greater precision
and reliability. This not only improves the utility of AI in legal applications but also builds
trust among users by consistently providing reliable and pertinent information.
Through detailed analysis and strategic adjustments in the use of context and prompt design,
we aim to harness the full potential of AI in the legal domain, minimizing the drawbacks while
maximizing the benefits of these advanced technologies.
20

8 Conclusions and Future Scope
Our study critically examined the development of an Advanced Intelligent Legal Question
Answering (AILQA) system focused on the criminal law domain in India. By integrating a
variety of state-of-the-art embedding and generative QA models, we endeavored to enhance
the effectiveness and reliability of legal question answering systems. Our empirical evaluations
demonstrated that AI-generated answers often surpass the quality of responses provided by
human legal experts, showcasing the transformative potential of AI in legal settings.
Despite significant advancements, the research identified areas requiring further improve-
ment. Models like Flan-UL2 demonstrated a need for enhanced semantic capabilities to better
comprehend and process complex legal queries. Additionally, the lack of specialized legal
QA datasets poses a significant challenge for fine-tuning and optimizing AI models tailored
for legal applications. Addressing these issues is crucial for the advancement of AI in legal
question answering.
Looking forward, several avenues appear promising for advancing the field of AI in legal
question answering. The development and curation of comprehensive legal datasets, partic-
ularly for the Indian legal system, are essential. These resources will enable more targeted
training and fine-tuning of AI models. Exploring more prompting strategies, such as Chain-of-
Thought prompting, could further enhance the models’ ability to reason and generate more
accurate answers. Additionally, combining lexical, semantic, and expert evaluations will con-
tinue to provide a robust framework for assessing the performance of legal AI systems, ensuring
technological soundness while maintaining alignment with legal accuracy and relevance.
The ultimate goal of our research is to create a legal QA system that not only performs
with high accuracy but also integrates seamlessly into the legal industry, providing reliable
support for legal professionals and the public. By continuously refining AI technologies and
adapting them to meet the specific needs of the legal domain, we envision a future where AI
becomes an indispensable tool in legal practice. This study has laid a strong foundation for
future advancements in the field of legal AI, driving forward both the science and the practical
implementation of AI in the legal sector.
9 Limitations
Our study encountered several notable limitations that influenced our methodology and find-
ings, impacting the depth and applicability of our research in the legal QA domain. One
significant challenge was the resource-intensive nature of securing legal expert annotations.
Due to the high costs and substantial time required, we were limited to obtaining expert
evaluations for only a sample of 150 random documents rather than the entire dataset. This
sampling approach may have constrained the comprehensiveness and depth of our expert-based
evaluations.
The expert evaluation also has certain methodological limitations. Although every gen-
erated response was independently assessed by all three legal evaluators and disagreements
were resolved through consensus with the involvement of a senior practising legal expert, the
evaluators were aware of the model identity and whether RAG had been used. This lack of
blinding may have introduced expectation bias into some assessments. Furthermore, the princi-
pal evaluators were senior undergraduate law students rather than practising lawyers. Their
legal training made them suitable for a structured comparative evaluation, and a practising
21

legal expert participated in resolving disagreements; nevertheless, future evaluations should
involve a broader group of practising lawyers and should use blinded assessment protocols.
No formal inter-rater reliability coefficient was calculated for the present study. The
reported MPNET-based p-values assess differences between experimental model configurations
and do not measure agreement among the human evaluators. Future work should report an
ordinal inter-rater reliability measure, such as Krippendorff’s alpha, using the evaluators’
independent pre-consensus ratings.
Additionally, while Large Language Models (LLMs) proved competent in conversational
contexts, their effectiveness in handling logic or knowledge-intensive tasks like legal QA was
less convincing. The models struggled particularly with analyzing lengthy legal questions
and generating detailed answers that included explanations or relevant legal references. This
difficulty was compounded in scenarios requiring intricate legal reasoning and contextual
understanding.
Moreover, the performance of our open-source baseline model fell short of expectations.
This shortfall may have been influenced by our approach to document chunking, which
involved restricting our analysis to only 1000 characters with a 250-character overlap. This
method potentially limited the models’ ability to capture the full context of legal cases, thereby
hindering their ability to generate comprehensive and nuanced responses.
These limitations highlight the inherent challenges in applying LLMs to complex, spe-
cialized tasks such as legal QA. They underscore the necessity for ongoing research and
development efforts aimed at enhancing AI models’ capabilities to accurately interpret and
understand detailed legal documents and contexts. Future work should also explore more effec-
tive methods for integrating extensive legal texts into AI systems without compromising on
the depth or accuracy of the generated content.
10 Ethical Considerations in AI-Driven Legal Question
Answering Systems
In the Ethical Considerations section of our study on AI in the legal domain, we acknowledge
the profound ethical implications of deploying AI technologies that can significantly influence
legal outcomes and impact individuals’ lives. To address these concerns, our research utilizes
publicly available data from legal blogs, ensuring transparency by providing comprehensive
documentation of the AI models’ decision-making processes. This approach supports account-
ability and facilitates the understanding and potential challenge of AI-generated decisions.
We emphasize that the outputs of AI systems should be regarded as advisory and must be
validated by human legal experts to uphold justice and equity in legal proceedings. Our study
also incorporates continuous monitoring and evaluation of the AI systems to adapt to evolv-
ing legal standards and practices, fostering regular engagement with legal professionals to
ensure that the development of the AI system is both ethically sound and effectively aligned
with practical legal needs. This commitment to ethical considerations is crucial for building
trustworthy AI systems that enhance the legal decision-making process while safeguarding
the rights and interests of individuals. The present experiments reflect the legal corpus used
during system development, including statutes and judicial decisions available within the
stated collection period. Since Indian criminal law has subsequently undergone significant
statutory changes, a deployed version of the system would require version-aware retrieval and
22

continuous updating using authoritative legal sources. The reported results should therefore
not be interpreted as validating the system for answering all questions under the currently
applicable legal framework.
References
Allam, A.M.N., Haggag, M.H.: The question answering systems: A survey. International
Journal of Research and Reviews in Information Sciences (IJRRIS)2(3) (2012)
AI@Meta: Llama 3 model card (2024)
Collenette, J., Atkinson, K., Bench-Capon, T.: Explainable ai tools for legal reasoning about
cases: A study on the european court of human rights. Artificial Intelligence317, 103861
(2023)
Choi, E., He, H., Iyyer, M., Yatskar, M., Yih, W.-t., Choi, Y ., Liang, P., Zettlemoyer, L.: Quac:
Question answering in context. arXiv preprint arXiv:1808.07036 (2018)
Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional
transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018)
Drake, A., Keller, P., Pietropaoli, I., Puri, A., Maniatis, S., Tomlinson, J., Maxwell, J., Fussey,
P., Pagliari, C., Smethurst, H.,et al.: Legal contestation of artificial intelligence-related
decision-making in the united kingdom: reflections for policy. International Review of Law,
Computers & Technology36(2), 251–285 (2022)
Floridi, L., Chiriatti, M.: Gpt-3: Its nature, scope, limits, and consequences. Minds and
Machines30, 681–694 (2020)
Feng, G., Qin, Y ., Huang, R., Chen, Y .: Criminal action graph: a semantic representation model
of judgement documents for legal charge prediction. Information Processing & Management
60(5), 103421 (2023)
Hwang, W., Lee, D., Cho, K., Lee, H., Seo, M.: A multi-task benchmark for korean legal lan-
guage understanding and judgement prediction. Advances in Neural Information Processing
Systems35, 32537–32551 (2022)
Hei, Z., Wei, W., Ou, W., Qiao, J., Jiao, J., Zhu, Z., Song, G.: Dr-rag: Applying dynamic docu-
ment relevance to retrieval-augmented generation for question-answering. arXiv preprint
arXiv:2406.07348 (2024)
Han, R., Zhang, Y ., Qi, P., Xu, Y ., Wang, J., Liu, L., Wang, W.Y ., Min, B., Castelli, V .: Rag-qa
arena: Evaluating domain robustness for long-form retrieval augmented question answering.
arXiv preprint arXiv:2407.13998 (2024)
Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot,
D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., et al.: Mixtral of experts. arXiv preprint
23

arXiv:2401.04088 (2024)
King, T.C., Aggarwal, N., Taddeo, M., Floridi, L.: Artificial intelligence crime: An interdis-
ciplinary analysis of foreseeable threats and solutions. Science and engineering ethics26,
89–120 (2020)
Kassner, N., Sch ¨utze, H.: Bert-knn: Adding a knn search component to pretrained language
models for better qa. arXiv preprint arXiv:2005.00766 (2020)
Lin, C.-Y .: ROUGE: A package for automatic evaluation of summaries. In: Text Summarization
Branches Out, pp. 74–81. Association for Computational Linguistics, Barcelona, Spain
(2004). https://aclanthology.org/W04-1013
Muludi, K., Fitria, K.M., Triloka, J., et al.: Retrieval-augmented generation approach: Docu-
ment question answering using large language model. International Journal of Advanced
Computer Science & Applications15(3) (2024)
Masala, M., Iacob, R.C.A., Uban, A.S., Cidota, M., Velicu, H., Rebedea, T., Popescu, M.:
jurbert: A romanian bert model for legal judgement prediction. In: Proceedings of the
Natural Legal Language Processing Workshop 2021, pp. 86–94 (2021)
Malik, V ., Sanjay, R., Nigam, S.K., Ghosh, K., Guha, S.K., Bhattacharya, A., Modi, A.: ILDC
for CJPE: Indian legal documents corpus for court judgment prediction and explanation. In:
Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural Language Processing (V olume 1:
Long Papers), pp. 4046–4062. Association for Computational Linguistics, Online (2021).
https://doi.org/10.18653/v1/2021.acl-long.313 . https://aclanthology.org/2021.acl-long.313
Nigam, S.K., Deroy, A.: Fact-based court judgment prediction. arXiv preprint
arXiv:2311.13350 (2023)
Nigam, S.K., Deroy, A., Shallum, N., Mishra, A.K., Roy, A., Mishra, S.K., Bhattacharya, A.,
Ghosh, S., Ghosh, K.: Nonet at semeval-2023 task 6: Methodologies for legal evaluation.
In: Proceedings of the The 17th International Workshop on Semantic Evaluation (SemEval-
2023), pp. 1293–1303 (2023)
Nigam, S., Sharma, A., Khanna, D., Shallum, N., Ghosh, K., Bhattacharya, A.: Legal judgment
reimagined: PredEx and the rise of intelligent AI interpretation in Indian courts. In: Ku,
L.-W., Martins, A., Srikumar, V . (eds.) Findings of the Association for Computational
Linguistics ACL 2024, pp. 4296–4315. Association for Computational Linguistics, Bangkok,
Thailand and virtual meeting (2024). https://aclanthology.org/2024.findings-acl.255
Papineni, K., Roukos, S., Ward, T., Zhu, W.-J.: Bleu: a method for automatic evaluation
of machine translation. In: Isabelle, P., Charniak, E., Lin, D. (eds.) Proceedings of the
40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318.
Association for Computational Linguistics, Philadelphia, Pennsylvania, USA (2002). https:
//doi.org/10.3115/1073083.1073135 . https://aclanthology.org/P02-1040
24

Qu, C., Yang, L., Qiu, M., Croft, W.B., Zhang, Y ., Iyyer, M.: Bert with history answer
embedding for conversational question answering. In: Proceedings of the 42nd International
ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1133–
1136 (2019)
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.,et al.: Language models
are unsupervised multitask learners. OpenAI blog1(8), 9 (2019)
Strickson, B., De La Iglesia, B.: Legal judgement prediction for uk courts. In: Proceedings of
the 3rd International Conference on Information Science and Systems, pp. 204–209 (2020)
Su, H., Shi, W., Kasai, J., Wang, Y ., Hu, Y ., Ostendorf, M., Yih, W.-t., Smith, N.A., Zettlemoyer,
L., Yu, T.: One embedder, any task: Instruction-finetuned text embeddings. arXiv preprint
arXiv:2212.09741 (2022)
Song, K., Tan, X., Qin, T., Lu, J., Liu, T.-Y .: Mpnet: masked and permuted pre-training for
language understanding. In: Proceedings of the 34th International Conference on Neural
Information Processing Systems. NIPS ’20. Curran Associates Inc., Red Hook, NY , USA
(2020)
Song, K., Tan, X., Qin, T., Lu, J., Liu, T.-Y .: Mpnet: Masked and permuted pre-training for
language understanding. Advances in neural information processing systems33, 16857–
16867 (2020)
Tiwari, A., Kalamkar, P., Banerjee, A., Karn, S., Hemachandran, V ., Gupta, S.: Aalap: AI
Assistant for Legal & Paralegal Functions in India (2024)
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y ., Bashlykov, N., Batra,
S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288 (2023)
Vats, S., Zope, A., De, S., Sharma, A., Bhattacharya, U., Nigam, S., Guha, S., Rudra, K., Ghosh,
K.: Llms–the good, the bad or the indispensable?: A use case on legal statute prediction
and legal judgment prediction on indian court cases. In: Findings of the Association for
Computational Linguistics: EMNLP 2023, pp. 12451–12474 (2023)
Wiratunga, N., Abeyratne, R., Jayawardena, L., Martin, K., Massie, S., Nkisi-Orji, I., Weeras-
inghe, R., Liret, A., Fleisch, B.: Cbr-rag: case-based reasoning for retrieval augmented
generation in llms for legal question answering. In: International Conference on Case-Based
Reasoning, pp. 445–460 (2024). Springer
Wei, J., Bosma, M., Zhao, V .Y ., Guu, K., Yu, A.W., Lester, B., Du, N., Dai, A.M., Le, Q.V .:
Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021)
Wang, Z., Ng, P., Ma, X., Nallapati, R., Xiang, B.: Multi-passage bert: A globally normalized
bert model for open-domain question answering. arXiv preprint arXiv:1908.08167 (2019)
25

Xu, Z., Li, X., Li, Y ., Wang, Z., Fanxu, Y ., Lai, X.: Multi-task legal judgement prediction
combining a subtask of the seriousness of charges. In: Chinese Computational Linguistics:
19th China National Conference, CCL 2020, Hainan, China, October 30–November 1, 2020,
Proceedings 19, pp. 415–429 (2020). Springer
Yang, C., Potts, R., Shanks, D.R.: Enhancing learning and retrieval of new information: a
review of the forward testing effect. NPJ science of learning3(1), 8 (2018)
26

A Appendix
Question Answer by Lawyer Answer by Mixtral-8x7B Model Explanation
Hi, Sir If Some One
Recorded Your Video And
Blackmail For Viral . And
Next You Receive A Call
From Unknown Number
They Say He Is An Officer
From Delhi Crime Branch
They Want To Know About
Video. If You Not Share
Information They They
Make A Team For Your
Arrest. Then What Can
Individual Do?Dear Client, 1. Do Not Panic: It
Is Natural To Feel Threatened Or
Scared In Such Situations, But
It Is Important To Stay Calm
And Composed. 2. Verify The
Authenticity: If You Receive A
Call From Someone Claiming To
Be An Officer, Ask For Their
Identification Details, Including
Their Name, Badge Number, And
The Department They Belong
To...Regularly Update Your Pri-
vacy Settings On Social Media
Platforms And Avoid Accepting
Friend Requests From Unknown
Individuals. Be Mindful Of What
You Share Online To Minimize
The Risk Of Being Targeted By
Blackmailers. Thank YouI understand that you are in a distress-
ing situation. However, it is important
to remain calm and not panic. Here are
some steps you can take: 1. Do not share
any information about the video or your-
self with the unknown caller. Ask for
their identification and contact details,
and inform them that you will verify their
credentials and get back to them... 4. Do
not delete the video or any other evi-
dence related to it. It may be required for
investigation and legal proceedings. 5. If
the unknown caller continues to harass
or threaten you, inform your lawyer and
the police immediately. They can take
appropriate action to ensure your safety
and security. Remember, it is important
to take immediate action and seek legal
advice to protect yourself and your rights.
Do not share any information or engage
in any communication with the unknown
caller without proper legal guidance.The lawyer’s answer is compre-
hensive but includes advice that
may feel overwhelming or less
relevant. It covers many points
but lacks focus on urgent actions.
In contrast, the model with RAG
offers concise, targeted advice on
immediate steps like preserving
evidence and seeking legal help,
making it clearer and more action-
able in a stressful situation.
Question Answer by Lawyer Answer by GPT-3.5 Turbo Model Explanation
My Ex Girlfriend Tried
To Implicate Me In A
False Case...for Which I
Was Called To The Police
Station...i Was Threatened
And Asked For My Social
Media Account’s I’d Pass-
word At The Police Sta-
tion... Can The Police Take
Action? Please Give Me
Advise What Should I Do
So That The Person Who
Hacked My Id Is Punished
And My Old Pea Doesn’t
Even OpenDear Client It’s Crucial To Defend
Yourself By Collecting Any Proof
That Will Demonstrate Your Inno-
cence In Case Your Ex-girlfriend
Makes False Charges Against You
Once Again. This May Include
Any Texts, Emails, Or Other Doc-
uments That Show You Haven’t
Spoken To Her In A Year...If
Required, Seek Legal Counsel
To Safeguard Your Rights. It’s
Crucial To Take Precautions To
Protect Your Online Accounts,
Including Those On Social Media,
Against Future Hacking Efforts.
Among Other Things, This Can
Include Often Changing Your
PasswordsIn your situation, it is advisable to take
the following steps: 1. **File a Com-
plaint with the Cyber Cell:** You have
already taken the right step by filing a
complaint with the Cyber Cell regard-
ing the hacking of your social media
account. Provide all necessary details
and evidence to support your complaint...
5. **Consult a Legal Professional:** If
you feel that you are at risk of being
falsely implicated again, it is advisable
to consult with a legal professional who
can guide you on how to protect yourself
legally.
Remember, it is crucial to prioritize your
safety and security in such situations.The model with RAG offers more
specific and actionable advice for
your situation. It recommends fil-
ing a complaint with the Cyber
Cell, securing your accounts,
maintaining evidence, avoiding
contact with your ex-girlfriend,
and consulting a legal professional.
In contrast, the lawyer’s answer
is more general, focusing on col-
lecting proof and protecting online
accounts but lacks detailed, action-
able steps.
Question Answer by Lawyer Answer by Llama2-70B Model Explanation
One of my female fam-
ily member alleged that
I abused her on phone
call and threatened to kill
her. She has mentioned
my phone number in com-
plaint but didn’t mention
the number on which I
made a call. I have my
numbers 3 months call
details where not a sin-
gle number belongs to her
and nor I abused to any-
one. Actually I didn’t make
a call to her thatswhy I
want to make complaint
against her and file defama-
tion case for financial com-
pensation. Kindly suggest
what could be done in such
case and I also want to pun-
ish her by court ?Dear Client Let her complained
be converted into full fledged case
then in the court of laws this case
will be dismissed ( you will be pro-
nounced not guilty). After that you
can initiate the legal against her
for filing a false case and defam-
tion.In this scenario, you may want to con-
sider taking the following steps: 1. Seek
legal advice: It is essential to consult
with a lawyer who specializes in crim-
inal law to understand your rights and
options. They can help you evaluate the
evidence you have, assess the strength
of the case against you, and develop a
defense strategy... 5. Prepare for trial: If
the matter proceeds to trial, you should
be prepared to present your case and
defend yourself. Your lawyer will help
you prepare your defense, cross-examine
prosecution witnesses, and make argu-
ments on your behalf... In conclusion, it
is advisable to seek legal advice from a
qualifiedThe model with RAG provides a
structured approach for address-
ing the situation: First, seek legal
advice to understand your defense
options. Respond to any legal
summons and consult with your
lawyer before any court appear-
ance. Consider filing a counter-
complaint for defamation if the
accusations are false and have
harmed you. Gather evidence such
as phone records to support your
defense. Lastly, prepare for trial
by working with your lawyer to
present your case effectively. In
contrast, the lawyer’s answer is
more general, suggesting that you
let the case unfold in court and
then pursue defamation, without
specific guidance on immediate
steps or evidence collection.
Table 10Illustrative Examples of Legal Question Answering: Cases Where Context-Aware Generative Models
Provided More Detailed and Explanatory Answers Compared to Human Lawyers.
27

Model Question Reference Response (No Context) Response (With Context) Hallucination Analysis
Llama2-70B I have a recording
of 10 minutes out of
which only the starting
5 mins of the record-
ing is what I want to
provide as evidence,
rest is irrelevant talk.
Can it be accepted by
court if the cut audio
is provided as record.Dear client, Audio
recordings can be pre-
sented as evidence in
Indian courts. Even if
the rest of the audio
recording is irrelevant,
you can provide the
entire 10 minute audio
recording in court.
It is the court which
will determine what is
relevant and what is
not. Thank you.As a legal advisor, I must
inform you that the admis-
sibility of evidence in a
court of law is a complex
issue that depends on vari-
ous factors...
Relevant caselaws:
* State of Maharashtra
vs. Dr. Sunil Shrikant
Ambekar (2016): In this
case...
* M.C. Abraham v. State
of Kerala (2017): In this
case...Based on the information
provided, it seems that you
are seeking legal advice
regarding the admissibil-
ity of a tape-recorded con-
versation as evidence in a
court of law in India...
In the case of Dr. Motwani
vs. UOI (1995), it was held
that a tape-recorded conver-
sation is a relevant fact and
is admissible under section
8 of the Evidence Act.The relevant case laws
cited in theResponse (No
Context)section are incor-
rect, indicating that the
model generates inaccu-
rate examples, or “hallu-
cinates,” without contex-
tual information. In con-
trast, when provided with
context, the model delivers
more accurate and relevant
case examples.
Davinci I hacked a friend’s
Instagram as a prank
to show him later. But
he is taking it too seri-
ously. If he presses
charges, what are my
options?Hello Sir, You have no
options regarding this,
as you have commit-
ted an illegal act. So
If he went for Cyber
complaint you have to
represent yourself as
to why you did that.Your options depend on
the laws of the coun-
try in which the incident
occurred. In India, hacking
someone’s account without
their permission is a crimi-
nal offence under the Infor-
mation Technology Act,
2000...
In the case of State of
Maharashtra v. Prafful
Sarda, the Bombay High
Court held that the accused
was...If your friend presses
charges against you for
hacking his Instagram
account, you may be liable
under Section 43 and
Section 66 of the Informa-
tion Technology Act, 2000
(IT Act)... You may also be
liable under Section 66B
of the IT Act... to three
years or with fine which
may extend to rupees one
lakh or with both.The legal cases referenced
in theResponse (No Con-
text)section are valid but
pertain to criminal cases
involving murder, not IT
offenses. This indicates
that the model confidently
generates incorrect exam-
ples when context is not
provided.
GPT-3.5 Turbo I Am Aaromal 23
Years Old Living In
Kodungallur. I Was
Forced To Watch Pron
Videos Around Age
Of 11 And I Was
Made To Stroke Him
For So Many Days
...now I Am Going
Through Trauma Diffi-
cult To Handle Feeling
Worthlessness And No
Motivation They Stole
My Child Hood And
Pushed Me Into Dark-
ness And Seeking For
You Assistance.thank
YouDear Sir, You Cannot
Take Any Action At
This Distant Time.
Within Three Years
Of Attaining Age Of
Majority One Must
Take Legal Action
In Respect Of Illegal
ActivitiesI am sorry to hear about
the traumatic experiences
you have been through. It is
important to seek help and
support during such diffi-
cult times. In India, you
can reach out to organi-
zations like the National
Commission for Protection
of Child Rights (NCPCR)
or Childline (1098) for
assistance and guidance on
how to address the trauma
you have faced... Take care
of yourself and reach out
for the help you need.I am truly sorry to hear
about the traumatic expe-
riences you have gone
through. It is important
to seek help and sup-
port during such difficult
times... Additionally, you
may also want to consider
reporting these incidents
to the authorities so that
appropriate action can be
taken against the individ-
uals involved. Remember,
you are not alone, and there
are resources available to
help you through this chal-
lenging time.The answer generated with
context acknowledges the
customer’s concern but
fails to provide a clear
solution to the query. In
contrast, the model pro-
vides a comparatively bet-
ter response when contex-
tual data is not included.
Table 11Comparison of Legal Responses: Generative Models with and without Context vs. Human Expert Answers
28
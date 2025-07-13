# KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities

**Authors**: Hruday Markondapatnaikuni, Basem Suleiman, Abdelkarim Erradi, Shijing Chen

**Published**: 2025-07-10 12:19:03

**PDF URL**: [http://arxiv.org/pdf/2507.07695v1](http://arxiv.org/pdf/2507.07695v1)

## Abstract
Fine-tuning is an immensely resource-intensive process when retraining Large
Language Models (LLMs) to incorporate a larger body of knowledge. Although many
fine-tuning techniques have been developed to reduce the time and computational
cost involved, the challenge persists as LLMs continue to grow in size and
complexity. To address this, a new approach to knowledge expansion in LLMs is
needed. Retrieval-Augmented Generation (RAG) offers one such alternative by
storing external knowledge in a database and retrieving relevant chunks to
support question answering. However, naive implementations of RAG face
significant limitations in scalability and answer accuracy. This paper
introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome
these limitations. Inspired by the divide-and-conquer paradigm, K2RAG
integrates dense and sparse vector search, knowledge graphs, and text
summarization to improve retrieval quality and system efficiency. The framework
also includes a preprocessing step that summarizes the training data,
significantly reducing the training time. K2RAG was evaluated using the
MultiHopRAG dataset, where the proposed pipeline was trained on the document
corpus and tested on a separate evaluation set. Results demonstrated notable
improvements over common naive RAG implementations. K2RAG achieved the highest
mean answer similarity score of 0.57, and reached the highest third quartile
(Q3) similarity of 0.82, indicating better alignment with ground-truth answers.
In addition to improved accuracy, the framework proved highly efficient. The
summarization step reduced the average training time of individual components
by 93%, and execution speed was up to 40% faster than traditional knowledge
graph-based RAG systems. K2RAG also demonstrated superior scalability,
requiring three times less VRAM than several naive RAG implementations tested
in this study.

## Full Text


<!-- PDF content starts -->

KEYKNOWLEDGE RAG ( K2RAG ): A NENHANCED RAG
METHOD FOR IMPROVED LLM QUESTION -ANSWERING
CAPABILITIES∗
Hruday Markondapatnaikuni
University of Sydney
smar0241@uni.sydney.edu.auBasem Suleiman
University of New South Wales
b.suleiman@unsw.edu.auAbdelkarim Erradi
Qatar University
erradi@qu.edu.qa
Shijing Chen
University of New South Wales
arthur.chen@unsw.edu.au
ABSTRACT
Fine-tuning is an immensely resource expensive process when trying to retrain Large Language
Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have
been many different fine-tuning techniques proposed which have shown good progress in trying
to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more
intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge
expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques
where information is stored in a database and appropriate chunks of information are retrieved to
help answer the question. However there are many limitations to naive RAG implementations. This
paper proposes the KeyKnowledgeRAG ( K2RAG ) framework to address the scalability and answer
accuracy limitations associated with naive RAG implementations. This framework takes inspiration
from divide-and-conquer ideology, and combines dense and sparse vector search, knowledge graphs
and text summarization to help address these limitations. The framework also involves a data pre-
processing step to reduce training times. The MultiHopRAG dataset was used for evaluation where
the our implemented K2RAG pipeline was trained with the document corpus and then evaluated with
the test data. Results have shown that there is an improvement in answering questions compared to
common Naive RAG implementations where K2RAG achieved the highest mean answer similarity
at0.57 and also was able to answer more questions with more ground truth similarity with a highest
Q3quartile at 0.82. Additionally our proposed framework is trainable much faster due to the inclusion
of a train data corpus summarization step which reduces training times of the individual components
by93% on average. Furthermore K2RAG has shown to operate even faster than a traditional
knowledge graph based naive RAG implementation with a mean execution times reduced by up to
40% . In addition to superior question-answering capabilities, K2RAG offers being a more scalable
solution with VRAM requirements reduced by 3 times in comparison to implementations of several
naive RAG pipelines evaluated in this paper. Hence K2RAG can help companies in sophisticated
decision-making through implementing a more lightweight and robust question-answering systems
built on internal documents.
1 Introduction
Motivation: Customizing LLMs to be more knowledgeable on specific areas has been a popular task to accomplish in
recent years. A technique known as fine-tuning has been proposed and popularized through various different methods
of fine-tuning coming up over the past few years such as FLAN [ 1], LoRA [ 2] and QLoRA [ 3] which all aim to address
∗Citation :Authors. Title. Pages.... DOI:000000/11111.arXiv:2507.07695v1  [cs.CL]  10 Jul 2025

Running Title for Header
different setbacks associated with the LLM fine-tuning process such as improving the final efficacy as well as reducing
computational resources.
Fine-tuning LLMs is usually done on full LLMs and is generally not advised to fine-tune on or convert pre-finetuned
models into quantized models due to potential loss of knowledge retention and accuracy which can be attributed to
lower precision of the model’s weights. Even though existing techniques like QLoRA have shown results which indicate
minimal performance degradation upon the resulting fine-tuned quantized models which are based on their Guanaco
models, this is not applicable to all LLMs [ 3]. For example an empirical study [ 4] evaluated different LLAMA3
quantizations and fine-tune quantizations has shown a significant reduction in LLAMA3-8B scores on the MMLU
benchmarks when compared to the standard full 16-bit model and a QLoRA fine-tune-quantization to 4bits. This is
because although all LLMs follow the basic transformer architecture there are still differences in model designs which
can impact the quality of fine-tunes and hence even previously considered efficient methods such as QLoRA face their
limitations with newer models [4].
Hence a major setback from all fine-tuning methods is that they are still extremely time and computational resource
intensive in order to yield satisfactory results. Hence a new approach needed to be devised whereby LLMs could gain
access to more knowledge with more computational efficiency.
Figure 1: Evolution of RAG since 2020 [5].
A class of techniques known as Retrieval Augment Generate (RAG) systems have been devised as an alternative to
fine-tuning where they do not employ any step where a base LLM is modified as in the fine-tuning method but rather
knowledge is extracted from documents and indexed in a knowledge holding system to then retrieve [ 6]. This helps
eliminate the computationally expensive associated with fine-tuning methods and also comes with the added benefit of
having a dynamic knowledge base which could be updated quicker than fine-tuning methods.
Since 2020 the concept of RAG has since evolved to include many different implementations and approaches to
performing information retrieval to enhance LLM answering capabilities on new datasets [Figure 1]. This highlights
how the benefits gained from RAG are being researched in depth to further optimize this process.
Answer Accuracy: Naive RAG implementations often struggle with the "Needle-in-a-Haystack" problem [ 8], where
the generator model fails to provide accurate answers due to an inability to extract relevant information from irrelevant
or excessively long contexts. This issue typically arises from poorly optimized chunk sizes and retrieval processes [ 9].
Optimizing chunk size is critical and challenging, as smaller chunks risk losing essential context, while larger chunks
may introduce irrelevant information and unnecessarily increase context length [Figure 2]. Optimizing the retrieval
process to maximize the relevance of retrieved information blocks is crucial. This ensures that irrelevant data, which
could act as noise and obscure the correct information, does not compromise the generator LLM’s ability to accurately
2

Running Title for Header
Figure 2: Graph showing how LLMs prefer extracting information from top or bottom of context to answer a question
[7].
answer the question. Another limitation of naive RAG systems is that embeddings which are similar but not actually
relevant to answer the question might be retrieved [ 10] as the original sentences might be semantically similar or carry
a similar meaning but might not be right in helping answer the question in that context.
Scalability: Naive implementations of Retrieval-Augmented Generation (RAG) often rely on 16-bit floating-point
large language models (LLMs) for the generation component. However, this approach introduces significant scalability
challenges due to the increased memory demands required to host the LLM as well as longer inference times due to
using a higher precision number type. To enable more efficient scaling, it is crucial to integrate methods or techniques
that reduce the memory footprint and inference times of generator models. Quantized models offer more scalable
solutions due to less computational requirements, hence when developing RAG systems we should aim to use quantized
LLMs for more cost effective deployment as compared to a full fine-tuned LLM whose performance might be good
but is more expensive to deploy due to higher memory requirements. A quantized LLM’s role in the RAG pipeline
itself should be minimal and for means of rewriting retrieved information into a presentable fashion for the end users to
interpret.
Additionally, in naive RAG implementations, information store components, such as Knowledge Graphs, face lengthy
training times for their creation and updating. This creates scalability challenges, as it increases the downtime required to
update the information store with new knowledge in a production environment. Hence an effective data pre-processing
step should be introduced to reduce the training corpus size and hence training times while preserving as much
information as possible.
Solution: We propose the K2RAG framework which addresses the following 4 research goals based on the aforemen-
tioned answer accuracy and scalability issues characteristic of naive RAG implementations:
Goal 1. Reduce information store such as Knowledge Graph and Spare and Dense vector database creation times.
Goal 2. Reduce chances of LLMs suffering from "Needle in Haystack" Problem.
Goal 3. Increase rate of retrieving relevant passages for answering the question.
Goal 4. Alleviate time and computational cost associated with full LLMs.
K2RAG incorporates various different concepts covered in the literature in a novel way to improve the performance and
offer a more resource efficient and accurate RAG pipeline compared to naive RAG methods outlined in the literature.
3

Running Title for Header
2 Related Works
Several studies have explored and implemented advanced RAG (Retrieval-Augmented Generation) systems to overcome
the limitations of the naive RAG approach discussed in the introduction. A significant focus of these works lies in
developing novel methods to enhance vector database-centric retrieval systems, evaluated across a diverse set of metrics
pertinent to vector-based retrieval. Additionally, some studies assess the performance of RAG solutions leveraging
Knowledge Graphs [ 11], while one investigation examines the outcomes of naively combining Knowledge Graph-based
and semantic search results for question-answering tasks. These related works also highlight certain limitations, ranging
from increased resource demands to the need for novel strategies or enhancements to improve system performance.
Addressing the limitations of naive Retrieval-Augmented Generation (RAG) pipelines has led to the integration of
hybrid retrieval methods that combine sparse (keyword-based) and dense (semantic) retrievers.
Spare and Dense Retriever RAG methods: Hybrid approaches, such as the "Blended RAG" framework proposed by
Sawarkar et al. [ 12], demonstrate that combining multiple retrievers can improve the retrieval of relevant passages for
clearly defined queries. This framework integrates six distinct retrievers—including traditional sparse methods (e.g.,
BM25) and dense semantic search techniques—selected based on their performance across benchmark datasets like
Natural Questions (NQ) [13], TREC-COVID [14], SqUAD [15], and HotPotQA [16].
A key strength of this hybrid search technique is its ability to return relevant passages, aligning well with Research
Goal 3 for well-specified queries. However, these methods do not fully address Research Goal 3 when handling vaguer
questions. In such cases, the hybrid approach can retrieve extraneous or unnecessary information, thus failing to
eliminate the "needle in a haystack" problem. This shortcoming is evident in evaluations on the CoQA dataset [ 17],
where retrieval accuracies ranged only from 45% to 49%. The reliance on document metadata during retrieval—a
common feature in these systems—further complicates matters, as such metadata is often minimal in real-world
scenarios (e.g., limited to document names and creation dates), thereby increasing both the creation time of the
information store (contradicting Research Research Goal 1) and the risk of returning irrelevant passages (contradicting
Research Research Goal 3) leading to longer contexts (contradicting Research Goal 2).
Figure 3: Results of Priyanka Mandikal et al. on 2 example queries [18].
In contrast, recent work by Mandikal et al. [ 18] has advanced a hybrid retrieval model that integrates a sparse TF-IDF
retriever with a dense retriever based on fine-tuned SPECTER2 embeddings [ 19]. Their approach employs a weighted
scoring mechanism, controlled by a hyperparameter λ, to balance the contributions of the two retrieval methods.
Empirical evaluations on a medical database concerning Cystic Fibrosis indicate that while dense retrieval alone
underperforms, incorporating approximately 20% of the sparse score (i.e., setting λ= 0.2for the sparse component)
significantly improves document retrieval accuracies across different retrieval ranks [Figure 3]. Crucially, this method
indexes documents solely based on their content, thereby avoiding the pitfalls of metadata dependency. This contributes
to reducing information store creation times (Research Goal 1) and partially mitigates the retrieval of unnecessary
information (Research Goal 3), although it does not entirely resolve the challenge of handling vaguer queries.
4

Running Title for Header
In summary, while hybrid retrieval approaches represent a significant advancement by returning more relevant passages
for well-defined queries complying with Research Goal 3. However they remain imperfect in filtering out unnecessary
information for vaguer questions which can lead to longer contexts. This limitation underscores the need for further
refinement to minimize the "needle in a haystack" problem in large-scale, real-world applications and make a pipeline
which is compliant with Research Goal 2 as well.
Reranker RAG methods: To address the limitation which Hybrid-Search methods have in adhering to Research Goal
2 for vaguer questions, Reranker methods have been developed to reduce the size and improve the quality of the final
context.
Figure 4: A high level view of what a RAG pipeline with a reranker looks like.
Usually this is a process called 2-step retrieval where the first step involves retrieving documents such as through sparse
or dense retriever techniques and then a second stage where an LLM based reranker comes into play to decide which of
the documents retrieved are most relevant to the query and sorts them in order [Figure 4] before passing it to an LLM
for generation.
Prior to the formal conceptualization of Retrieval-Augmented Generation (RAG), information retrieval research
explored the use of document rerankers. For example, Noguiera et al.’s 2020 study, Document Ranking with a Pretrained
Sequence-to-Sequence Model [ 20], proposed employing a sequence-to-sequence model as a document reranker. This
work evaluated multiple fine-tuned T5 models of varying sizes, as well as a fine-tuned BERT model, on documents
initially retrieved using the sparse BM25 keyword method. The incorporation of these rerankers led to improvements
in retrieval performance across the Robust04, Core17, and Core18 datasets. However, a critical limitation of this
framework was its reliance on large models, leaving unresolved whether the observed performance gains were due to
the reranking strategy or simply the use of more powerful models.
Subsequent advancements in RAG have integrated BERT-based rerankers into new pipelines. Michael Glass et al.’s
Re2G framework [ 21] embeds a BERT-based reranker—similar to that of Noguiera et al.—within a RAG system. This
integration has shown notable improvements in metrics such as Recall (for 5 documents), overall recall, precision,
and accuracy on several datasets, as indicated by performance on the KILT leaderboard. Nevertheless, a significant
drawback of this approach is the necessity to host an additional model, which directly conflicts with Goal 4: alleviating
the monetary and computational costs associated with deploying full-scale LLMs.
Further refinement in reranking approaches is demonstrated by Xuegang Ma et al., who fine-tuned a LLaMA-2-7B
model for reranking in their RankLLaMA system [ 22]. Although RankLLaMA achieves superior reranking performance,
it again relies on a large model comparable to LLaMA-2-7B. Such reliance on resource-intensive models undermines
the overarching objective of Goal 4, as the additional computational and financial burden associated with hosting these
rerankers contradicts the need for efficiency and cost reduction.
In summary, while reranker models can mitigate the “needle in a haystack” problem (Goal 2) by filtering out least
relevant information, they are not a sustainable solution. Their requirement for additional large-scale models significantly
increases computational and monetary costs, thereby violating Goal 4. Consequently, alternative strategies that achieve
comparable retrieval performance without incurring such overhead are essential for the advancement of practical and
efficient RAG systems.
Knowledge Graph RAG methods: Knowledge graph–based solutions have been proposed as a promising alternative
for enhancing the RAG process. The core idea behind employing knowledge graphs is to address semantic search
challenges by organizing data according to intrinsic relationships. For example, although not originally applied within a
RAG context, Dawei Cheng et al.’s 2020 work on constructing knowledge graphs for a financial price prediction tool
[23] illustrates how grouping semantically related data into distinct sub-networks enables a query—such as “who quit
Apple”—to retrieve only the pertinent subset of information, rather than returning all semantically similar sentences
(e.g., both “Steve Jobs quits Apple” and “David Peter leaves Starbucks”). This targeted retrieval helps mitigate the
"needle in a haystack" problem (Goal 2) and increases the rate of retrieving relevant passages (Goal 3).
Building on these insights, recent research has directly integrated knowledge graphs within RAG frameworks. For
instance, Zhentao Xu et al. [ 24] developed a system for LinkedIn’s customer question answering service by first
5

Running Title for Header
creating a custom knowledge graph and then retrieving related nodes. The retrieved information is subsequently
refined using GPT-4 to produce coherent responses. When compared with a baseline semantic search system, this
pipeline significantly improved retrieval metrics—including Mean Reciprocal Rank, Recall, and Normalized Discounted
Cumulative Gain—as well as evaluation measures such as BLEU, METEOR, and ROUGE, thereby supporting Goals 2
and 3.
However, despite these promising advances, knowledge graph–based approaches fall short of meeting Goal 1. The
process of constructing and training knowledge graphs is computationally intensive and time-consuming, which directly
conflicts with the objective of reducing information store creation times.
Figure 5: Results comparing knowledge graph framework with semantic search methods [25].
For example, Diego Sanmartín’s 2024 work [ 25] demonstrated a Knowledge Graph RAG system that leverages LLMs
to extract entity-relationship triples from unstructured text and employs a “Chain of Explorations” retrieval strategy.
While this approach shows potential for handling vague queries, the high training times required to build and maintain
the knowledge graph significantly increase the overall overhead. Moreover, the compressed storage format—using
entity-relationship triples—can lead to a loss of detailed contextual information, thereby limiting the system’s ability to
fully expand upon relevant topics during retrieval hence resulting in even lower EM, F1, Accuracy scores with only an
improvement in Hallucination compared to a standard Semantic Search method [Figure 5].
In summary, while knowledge graph–based solutions offer clear benefits in terms of improving semantic search and
enhancing the retrieval of relevant passages (addressing Goals 2 and 3), they do not currently satisfy Goal 1 due to their
high training times and associated computational costs. Addressing these inefficiencies is essential for realizing the full
potential of knowledge graphs within efficient and practical RAG frameworks.
Combining Knowledge Graphs and Vector based Search methods: The limitations inherent in a purely knowledge
graph–based RAG framework suggest that combining the strengths of knowledge graphs with traditional semantic
search methods may yield superior results. Bhaskarjit Sarmah et al. [ 26] propose such an integrated approach through
their HybridRAG framework. This framework concatenates outputs from a knowledge graph query—implemented using
Microsoft’s GraphRAG [ 27] with content indexed by extracting entity relationships in chunk sizes of 1024—with results
from a semantic search, which similarly employs chunk sizes of 1024 and the text-embedding-ada-002 model. The
aggregated context is then provided to GPT-3.5 to generate the final answer. The dataset for this study was constructed
by web scraping earnings call documents of companies listed on the NIFTY 50, and evaluation was performed using
the RAGAS evaluation platform [28].
The HybridRAG framework demonstrated improvements in metrics such as Faithfulness, Answer Relevancy Score, and
Context Recall. These enhancements support Goal 2 by reducing the "needle in a haystack" problem and Goal 3 by
increasing the rate of retrieving relevant passages for answering questions. However, the framework underperformed
in Context Precision when compared to standalone implementations of both knowledge graph–based and vector
database–based RAG systems. This shortcoming indicates that the current integration does not fully leverage the
advantages of traditional vector search techniques, particularly hybrid search techniques, which is critical for fully
meeting Goal 3.
3 Methodology:
3.1 K2RAG Framework
To meet the research goals, the K2RAG framework implements four techniques:
•1. Knowledge Graph: K2RAG employs a knowledge graph component to organize and interconnect topics
within the corpus [Figure 8(a)]. Unlike traditional chunking approaches, which struggle to balance the trade-off
6

Running Title for Header
between providing sufficient context and avoiding irrelevant information, the knowledge graph addresses this
issue by structuring content into interconnected nodes which enables knowledge-rich and more context-aware
retrieval capable of handling complex or vague queries that vector-based search methods often struggle with.
•2. Hybrid Search: To minimize false positive embeddings—where semantically similar but irrelevant chunks
are retrieved— K2RAG integrates a hybrid retriever [Figure 8(b)]. Inspired by Mandikal et al.’s paper [ 18],
we weight dense vector and sparse vector retrieval at an optimized 80%/20% ratio. This hybrid method
significantly improves retrieval accuracy, helping reduce noise by ensuring the most relevant content is
retrieved.
•3. Summarization: To combat the "Needle-in-Haystack" issue, K2RAG incorporates summarization at
multiple stages in the pipeline [Figure 8 (B) and (E)]. To speed up indexing, documents are summarized before
being indexed into vector and knowledge graph stores. Additionally retrieved content is summarized at query
time to further refine the context provided to the LLM. By summarizing at both indexing and retrieval, we
ensure concise, high-quality input for generation.
•4. Lightweight Models: For VRAM resources efficiency, K2RAG leverages a lightweight Longformer-based
text summarizer [ 29], along with quantized LLMs to produce a low VRAM and resource-efficient pipeline
without compromising output quality.
3.1.1 Corpus Summarization Process
To address research goal 1, we decrease training times of the K2RAG framework compared to naive RAG frameworks
by summarizing the training corpus to reduce its size while preserving as much information as possible. Hence a
Longformer based model [ 29] fine tuned for long text summarization is used as it’s self-attention mechanism is highly
optimized for capturing and processing large chunks of texts compared to those seen in traditional transformer models.
This helps speed up training times of information store components used in K2RAG . To create our summarized corpus
we iterate over each article in the corpus and then summarize it using the loaded in Longformer summarizer model.
Then these results are stored into a CSV format from which then they can be loaded in for training the RAG pipeline
[Algorithm 1] [Figure 6].
Algorithm 1 Generating Summarised Corpus
1:Load in summariser model
2:Load in corpus
3:Initialise summarised corpus as empty list
4:fordocument incorpus do
5: Generate summary ofdocument using summarised model
6: Append summary tosummarised corpus list
7:end for
Summarise 
Document
Load 
corpus
For 
each 
document
Longformer 
Encoder 
Decoder 
Model
Store 
into 
list
Repeat
Export 
to 
CSV
Figure 6: How to generate summaries of corpus.
3.1.2 Indexing
The chunking strategy to generate a set of chunks Cdfrom a single document or text dis defined as:
7

Running Title for Header
Cd=Lt[
i=1
x((i−1)∗(S−O))+j|j= 1,2, . . . , S	
, Lt=Ntd−O
S−O
(1)
where Sis the desired chunk size, Ois the desired chunk overlap, Ntdis the total number of tokens in the document d.
The inner set denotes a chunk constructed from tokens x((i−1)∗(S−O))+1 tox((i−1)∗(S−O))+SandLtis the limit of
tokens to process to ensure the number of chunks of size Sand overlap Oare maximized.
We then apply the chunking strategy defined in (1)across all documents in the summarized corpus corsumand take the
union of chunks to give us the final set of chunks Cto pass to the knowledge graph and vector stores for indexing:
C=|corsum|[
d=1Cd|d∈corsum (2)
The configuration used to train all information retrievers is S= 256 , O= 20 when chunking for the dense and sparse
vector stores indexing, and S= 300 , O= 100 when chunking for knowledge graph indexing.
This was decided in accordance with Xiaohua Wang et al.’s paper for vector stores indexing which showed 512 and
256 token chunks achieved the best results in Faithfulness and Relevancy metrics [ 9] [Figure 7]. Using the chunk
size less than 512 is used to help promote in smaller contexts generated to pass to the generator LLM to address the
"Needle-in-Haystack" challenge [ 8] by ensuring enough information is contained within the chunks retrieved and
minimising noisy information.
Figure 7: Chunk sizes and their effects on Answer Faithfulness and Answer Relevancy.
3.1.3 Retrieval and Generation
For all pipelines the following system prompt was used for generation of the final answer:
Answer Generation System Prompt
Additional Information: context from retrieval
Instruction: You are a smart LLM who gives an answer to the question in as little words as possible using the
additional information provided above.
Question: question
Short Answer:
We establish the top kchunks Cqretrieved from the dense and sparse vector databases for a question qis ahybrid
retriever [Figure 8 (b)] defined as:
Cq=k[
i=1{c|scr(q, c, λ ) = max i[S]}, S={scr(q, c, λ )|c∈C} (3)
where Sis the set of scores computed for each chunk in C(2)using a scoring function scrof the question q, chunk c
and a weighting parameter λ:
8

Running Title for Header
scr(q, c, λ ) =λe(q)·e(c)
|e(q)| ∗ |e(c)|+ (1−λ)z(q)·z(c)
|z(q)| ∗ |z(c)|(4)
Where erepresents the dense embeddings model as a function and zrepresents the sparse embedding function.
The first step in the K2RAG pipeline [Figure 8] is to use a knowledge graph trained on summarized data to generate
an answer containing the identified topics relevant to the query [Figure 8(A)]. Then a Longformer model is used to
summarize the results obtained from the knowledge graph [Figure 8(B)]. Over the knowledge graph results summary,
we split it into chunks of S= 128 andO= 10 following our text chunking strategy (1). This is so that each chunk
contains information about a clear topic needed to help answer the question from which for each chunk we use a
quantized LLM to create a sub-question out of it [Figure 8(C)] with the following specialized prompt:
Question Generation System Prompt
Instruction: Your task is to create a small question out of the below information.
Information: chunk from knowledge graph results
Answer:
The sub-question is then used to query the dense and sparse vector databases using the hybrid retriever with λ= 0.8
[18] andk= 10 (4)[Figure 8(D)] where an embeddings model is used when querying the dense vector database [Figure
8(c)].
We then use the quantized LLM with the results from the hybrid retriever as context to generate the answer to the
sub-question and then store into a list containing all the sub-answers. Once completed for each chunk, we then
concatenate all the sub-answers and use the Longformer model to summarize the results so that we can reduce the size
of the context [Figure 8(D)]. Finally, the summarized sub-answers and the summarized knowledge graph results are
concatenated and passed as context to the quantized LLM to generate the final answer to the question [Figure 8(E)].
3.2 Implemented Naive RAG Pipelines
We implemented 4 common naive RAG pipelines;
•Semantic - Retrieving only from a Dense Vector Database.
•Keyword - Retrieving only from a Sparse Vector Database.
•Hybrid - Retrieving from both Dense and Sparse Vector Database.
•Knowledge Graph (KG) - Retrieving only from a Knowledge Graph.
3.2.1 Indexing
The indexing process for each information store is the same as in K2RAG but with chunks Ccreated with the
unsummarized corpus corunsum instead:
C=|corunsum |[
d=1Cd|d∈corunsum (5)
3.2.2 Retrieval and Generation
All naïve retrieval-augmented pipelines in our framework follow a two-step process: (1) retrieval of information relevant
to the input question, and (2) passage of the retrieved information to an unquantized large language model (LLM) for
answer generation. In the case of our Semantic, Keyword, and Hybrid pipelines, we use a hybrid retrieval mechanism
that combines semantic similarity and keyword matching. The weighting parameter λcontrols the balance between the
two retrieval strategies: a value of λ= 1corresponds to purely semantic retrieval [Figure 9(S)], λ= 0corresponds
to purely keyword-based retrieval [Figure 9(K)], and λ= 0.8gives higher preference to semantic retrieval while still
incorporating keyword-based signals [Figure 9(H)].
For the naive Knowledge Graph (KG) retrieval pipeline, the retrieval process differs significantly. Instead of directly
retrieving passages from the unsummarized corpus through vector search, we retrieve structured information from a
9

Running Title for Header
A
Retrieve 
from 
Knowledge 
Graph
B
Summarise 
Knowledge 
Graph 
Results
C
Generate 
sub-questions 
from 
Knowledge 
Graph 
Results 
Summary
D
Answer 
Each 
Sub-question 
and 
Summarise 
Answer
Embeddings 
Model
Quantised 
LLM
Hybrid 
Vector 
Retriever
E
Give 
final 
answer 
with 
sub-answers 
and 
Knowledge 
Graph 
Results 
Summary
Knowledge 
Graph
a
b
User 
Question
Final 
Answer
c
Figure 8: K2RAG framework overview. Component (a): Knowledge Graph. Component (b): Hybrid Retriever
containing Dense and Sparse retrievers. Component (c): Embeddings Model. Step (A): Knowledge Graph results. Step
(B): Summarise Knowledge Graph results. Step (C): Generating sub-questions from chunks. Step (D): Generating
sub-answers for sub-questions. Step (E): Generate the final answer from sub-answers and Knowledge Graph Results
Summary.
knowledge graph that has been constructed using the unsummarized corpus and a quantized LLM. In this setup, the
same quantized LLM is employed both for generating the knowledge graph and for querying it at retrieval time.
Across all pipelines, including the KG-based approach, the retrieved information is subsequently passed to an unquan-
tized LLM for final answer generation. Importantly, all pipelines share a common answer generation prompt, which
is the same as that used in K2RAG [Prompt 3.1.3]. This standardization of the answer generation step allows for a
controlled comparison across different naive retrieval strategies with K2RAG .
3.3 Evaluation Framework
3.3.1 Dataset Selection and Description
In order to perform a comprehensive evaluation on the proposed framework we have outlined two key requirements
needed from the dataset as follows:
•Dataset must contain question answer pairs in plain text format where answers are to be assumed as ground
truth answers to the question.
•The question answer pairs must be constructed in reference to a document corpus which has all the required
documents and their information from which ground truth answer to the questions are derived from.
Due to time and resource constraints we were unable to create our own dataset from scratch satisfying these requirements
and hence the MultiHop-RAG dataset [ 30] was identified and used as it satisfies the above key requirements needed to
perform the analysis.
The MultiHop-RAG dataset contains corpus data which was used for training along with a test dataset containing 2555
question answer pairs from which we performed evaluation of the pipeline. The corpus contains a 609 articles from a
10

Running Title for Header
Algorithm 2 K2RAG Retrieval and Generation
1:Load in QuantizedLLM
2:Define the generation prompt for generation
3:Define the question prompt for question creation
4:Load in embeddingsmodel
5:Load in summariser model
6:Load in knowledge graph
7:Load in BM25store
8:Load in semantic vector store
9:Obtain user input
10:Retrieve knowledge graph output from knowledge graph using user input andembeddingsmodel and
QuantizedLLM
11:Summarise knowledge graph output using summariser model and set as kgresultssummarised
12:Chunk knowledge graph output intochunks of size maximum 128 tokens with 10 tokens overlap
13:Define subanswer context as empty string
14:Define kas number of chunks to retrieve
15:forchunk inchunks do
16: Create a sub−question from the chunk using QuantizedLLM andquestion prompt
17: Retrieve top k chunks fromBM25store using sub−question
18: Initialise sub−context as empty string
19: forhybrid chunk intop k chunks do
20: Append hybrid chunk tosub−context
21: end for
22: Generate sub−answer using QuantizedLLM withgeneration prompt and passing in user input and
sub−context
23: Summarise sub−answer using summariser model and set as summarised sub −answer
24: Append summarised sub −answer tosub−answer context
25:end for
26:Summarise sub−answer context using summariser model and set as summarised −subanswer −context
27:Setfull context as concatenation of kgresultssummarised andsummarised −subanswer −context
28:Generate answer using QuantizedLLM withgeneration prompt and passing in user input andfull context
wide range of plain-text news articles data covering various topics such as entertainment, health, sports and technology
[30] collected through the mediastack REST API and processed through a pipeline involving GPT-4 in order to review
question answer pairs created from the articles by an LLM extracting factual statements from the articles and then
GPT-4 generating these question answer pairs. This diversity in topics was also another reason this dataset was selected
as it allows evaluating across multiple different domains of information to see if the pipeline can effectively answer
questions across a wide variety of questions and their topics ensuring generalization.
3.3.2 Evaluation Process
We used a K-fold evaluation framework where we split the data into K=10 folds. For each pipeline, we generated the
answers for each question in each fold and captured answering times for the questions to compare the execution times
[Figure 10] [Algorithm 3].
To evaluate the accuracy of a pipeline’s answer, we computed the similarity between the ground truth and the pipeline’s
output as a scoring function sof the plaintext ground truth and pipeline output for each question defined as:
s(o, t) =e(o)·e(t)
|e(o)| ∗ |e(t)|(6)
where ois the pipeline’s output, tis the ground truth and erepresents the embeddings model as a function to generate
the vector embedding of a plaintext input.
We randomly partitioned the dataset into ten equal folds to simulate real-world usage of the pipeline. This approach
reflects scenarios in which different users pose diverse questions to the system. Random assignment of test instances
to folds helps mitigate the risk of performance bias, which could arise if certain folds contained clusters of similar
questions that align particularly well with the strengths of the pipeline. For evaluation, we employed cosine similarity
11

Running Title for Header
1
Retrieve 
information
20%
Sparse 
Vector 
Database
80%
Dense 
Vector 
Database
2
Generate 
Final 
Answer
Quantised 
LLM
Knowledge 
Graph
100%
Dense 
Vector 
Database
100%
Sparse 
Vector 
Database
Unquantised 
LLM
User 
Question
H
Final 
Answer
S
K
KG
Figure 9: Structure of competitor pipelines. Component (S): Semantic Search - Dense Vector Retrieval. Component
(K): Keyword Search - Sparse Vector Retrieval. Component (H): Hybrid Search - Combination of Sparse and Dense
Vector Retrieval. Component (KG): Knowledge Graph Retrieval
to assess the semantic similarity between the predicted and ground truth answers, as it effectively captures the degree of
semantic similarity. Additionally, we recorded the execution time for each query to analyze the system’s efficiency.
Ground 
Truth
Model's 
Answer
Common 
System 
Prompt
Cosine 
Similarity
Embeddings 
Model
RAG 
Pipeline
MultiHop-RAG 
Test 
Dataset
Generate 
Answer
K-th 
Subset 
of 
questions 
as 
Input
Store 
Scores 
and 
Times
Fold 
1 
to 
10
Repeat
Execution 
Time
Scores 
List
Times 
List
Figure 10: Evaluation Pipeline.
12

Running Title for Header
Algorithm 3 K-Fold Evaluation Algorithm
1:Load in rag pipeline
2:Load in embeddings model
3:Load in test question −answer pairs
4:Initialize scores as empty list
5:Shuffle test points in test question −answer pairs randomly
6:Initialize folds by assigning roughly equal number of points from test question −answer pairs to 10 folds
7:forfold number ,question, groundtruth infolds do
8:start time ←current time
9: Generate model answer fromquestion using rag pipeline
10: execution time ←current time -start time
11: Generate ground truth vector using embeddings model by passing ground truth
12: Generate model answer vector using embeddings model by passing model answer
13: Calculate score by doing cosine similarity on model answer vector andground truth vector
14: Append fold number ,score ,execution time toscores list
15:end for
4 Results
4.1 Practical Implementation
The Corpus Summarization and Evaluation processes were performed on a Linux machine with an NVIDIA L4 24GB
VRAM GPU and 32GB RAM. We used the following tech stack for implementing all the pipelines;
•Embeddings Model: nomic-embed-text
•Quantized LLM: Mistral-7B-Q4
•FP16 LLM: Mistral-7B-FP16
•Summariser Model: pszemraj/led-base-book-summary from HuggingFace
•Sparse Vector Database: Okapi BM25
•Dense Vector Database: ChromaDB
•Knowledge Graph Implementation: Microsoft GraphRAG with Mistral-7B-Q4 as entity-relationship
extractor and output generator.
4.2 Corpus Summarization
Corpus summarization took roughly 25 minutes to complete. Most documents have been reduced by roughly 86% to
92% while the overall mean reduction is roughly 89% [Figure 11]. We attribute the consistent large reduction in size of
the documents to considerable white space removed in many of the documents, leading to an effective reduction.
13

Running Title for Header
1707580859095Percentage Reduction
89.02
Q1: 86.48Q3: 92.02
Lower Whisker: 78.16Upper Whisker: 98.55Box Plot of Percentage Reduction
Whiskers
Mean point
Mean
Figure 11: Boxplot of corpus size reduction.
4.3 Training Times
By performing corpus summarization, sparse and dense vector stores, and knowledge graph creation times reduced by
89% ,97% and94% respectively with significant time saved for knowledge graph creation from 18 hours to 1 hour
[Figure 12] [Table 1]. Hence, in exchange for 25 minutes of performing summarization, training times were reduced on
average by 93% across all information stores, successfully achieving Research Goal 1 in reducing creation times of
information store components, particularly the knowledge graph.
14

Running Title for Header
Trained on
Full CorpusTrained on
Summarised Corpus
Database Components010002000300040005000Creation Time4010 seconds
441 secondsDense Vector Store
Trained on
Full CorpusTrained on
Summarised Corpus
Database Components0100200300400500
356 seconds
27 secondsSparse Vector Store
Trained on
Full DataTrained on
Summarised Data
Database Components010000200003000040000500006000070000
64785 seconds
3915 secondsKnowledge Graph
Figure 12: Information store component training times with summarized and unsummarized corpus. Training on the
summarized corpus took significantly less time compared to unsummarized corpus.
Information Store Training Times
Dense Vector Full Corpus 4010s
Dense Vector Summarized Corpus 441s
Sparse Vector Full Corpus 356s
Sparse Vector Summarized Corpus 12s
Knowledge Graph Full Corpus 64785s
Knowledge Graph Summarized Corpus 3915s
Table 1: Table of Training Times
4.4 Answer Accuracy
We observe that the score distributions for the Semantic, Keyword, Hybrid Search, and KG pipelines are very similar
[Figure 13]. This suggests that although these naive pipelines utilize different components, they cannot fully exploit
15

Running Title for Header
their potential for Question Answering, which is why they are considered naive. In contrast, the distribution for our
K2RAG framework demonstrates a significantly larger Q3quartile compared to the naive frameworks, which indicates
thatK2RAG was able to answer more questions with greater similarity to the ground truths. K2RAG has a slightly
higher mean accuracy of 0.57 compared to other naive methods despite a high Q3quartile value [Table 2]. This implies
thatK2RAG is better than naive RAG pipelines in answering certain types of questions which pushed the Q3quartile
value up, but there were not enough of these types of questions to induce a greater increase in the mean similarity.
Overall, K2RAG achieved the highest mean similarity score of 0.57 and the top Q3quartile score of 0.82 [Table 2],
successfully achieving Research Goals 2 and 3 by retrieving only the most relevant information and creating better
contexts for generating accurate and information-rich answers.
0.00.20.40.60.81.0Cosine SimilarityMean: 0.55
Q1: 0.40Q3: 0.67Naive Semantic Search
0.00.20.40.60.81.0Cosine SimilarityMean: 0.56
Q1: 0.41Q3: 0.67Naive Keyword Search
0.00.20.40.60.81.0Cosine SimilarityMean: 0.56
Q1: 0.40Q3: 0.67Naive Hybrid Search
0.00.20.40.60.81.0Cosine SimilarityMean: 0.54
Q1: 0.40Q3: 0.63Naive Knowledge Graph Search
0.00.20.40.60.81.0Cosine SimilarityMean: 0.57
Q1: 0.40Q3: 0.82KeyKnowledgeRAG
Figure 13: Pipeline Answer Accuracy results. K2RAG achieved the best performance with highest mean and 3rd
Quartile value.
Pipeline Q1Mean Q3
Naive Semantic Search 0.40 0.55 0.67
Naive Keyword Search 0.41 0.55 0.67
Naive Hybrid Search 0.4 0.56 0.67
Naive Knowledge Graph Search 0.4 0.54 0.4
K2RAG 0.4 0.57 0.82
Table 2: Table of Pipeline Answer K2RAG achieved the best performance with highest mean and 3rd Quartile value.
4.5 Execution Times
Semantic Search, Keyword Search, and Hybrid Search were the fastest running pipelines. KG Search recorded the
longest mean runtime at 117.31 seconds, with a small interquartile range, indicating consistent performance [Figure
14]. The longer runtime can be attributed to the large, unsummarized knowledge graph used in retrieval. Our proposed
K2RAG framework achieved a mean runtime of 70.25 seconds, significantly faster than naive KG Search but slower
than vector based search methods such as semantic, keyword and hybrid search [Table 3]. This is expected as querying
from the knowledge graph component takes more time as it is not just a simple retrieval based on matching vectors.
16

Running Title for Header
K2RAG exhibited a wider inter-quartile range due to the sub-questions step, which increases the number of chunks as
the size of the knowledge graph results grows. Despite having more steps, K2RAG had faster execution times than KG
Search and hence successfully achieving Research Goal 4 with faster responses.
024681012Time (seconds)
Mean: 3.31
Q1: 1.59Q3: 4.46Naive Semantic Search
024681012Time (seconds)
Mean: 3.29
Q1: 1.57Q3: 4.47Naive Keyword Search
02468101214Time (seconds)
Mean: 3.92
Q1: 2.15Q3: 5.17Naive Hybrid Search
6080100120140160180Time (seconds)Mean: 117.31
Q1: 105.49Q3: 125.63Naive Knowledge Graph Search
050100150200Time (seconds)Mean: 70.25
Q1: 38.42Q3: 86.71KeyKnowledgeRAG
Figure 14: Pipeline Execution Times results. K2RAG is faster than KG Search despite more steps.
Pipeline Mean Execution Time
Naive Semantic Search 3.31s
Naive Keyword Search 3.29s
Naive Hybrid Search 3.92s
Naive Knowledge Graph Search 117.31s
K2RAG 70.25s
Table 3: Table of Execution Times. K2RAG is faster than KG Search despite more steps.
4.6 Memory Footprint
The memory footprint analysis, as presented in Table 4, highlights the significant reduction in VRAM consumption
achieved by K2RAG compared to other retrieval pipelines. Specifically, while the Semantic Search Pipeline, Keyword
Search Pipeline, and Hybrid Search Pipeline each require 14.3GB of memory, and the Knowledge Graph Search
demands an even higher 18.1GB. Meanwhile K2RAG operates with only 5GB of VRAM [Table 4].
This reduction represents a mean threefold decrease in memory usage relative to the other pipelines, demonstrating
a substantial optimization in resource efficiency. Such an improvement directly aligns with Research Goal 4, which
emphasizes minimizing computational resource requirements.
17

Running Title for Header
Pipeline Memory Footprint
Semantic Search Pipeline 14.3GB
Keyword Search Pipeline 14.3GB
Hybrid Search Pipeline 14.3GB
Knowledge Graph Search 18.1GB
KeyKnowledgeRAG 5GB
Table 4: Table of Memory Footprints
5 Discussion
5.1 Achieving Research Goal 1: Reduce information store creation and update times
K2RAG effectively reduces the creation times for Knowledge Graph and Sparse and Dense vector databases, such as
the Semantic and BM25-based Keyword databases. Performing text summarization on the training corpus led to an
average reduction in training times of up to 93% , with only 25 minutes spent on summarization.
For instance, the Semantic database creation, which originally took 4010 seconds (67 minutes) , was reduced to 441
seconds (7.35 minutes) when using the summarized corpus. Even when accounting for the summarization process,
the total time was 32.25 minutes , significantly lower than training with the full corpus. Similarly, Knowledge Graph
creation time dropped from nearly 18 hours to around 1 hour , thanks to the reuse of the summarized corpus. This
approach not only saves time and resources but also reduces the cost of LLM-based Knowledge Graph creation, which
is typically expensive to host.
5.2 Achieving Research Goal 2: Mitigate the Needle-in-a-Haystack problem
K2RAG effectively mitigates the "Needle in a Haystack" problem, which can hinder LLM performance when handling
large contexts. By integrating the Longformer-based led-base-book-summary summarizer at key stages of the retrieval
pipeline, the framework reduces context size while preserving essential information.
K2RAG maintained answer quality and even outperformed naive pipelines on certain questions, as reflected in the
higher accuracy and Q3quantile. This approach allows for improved LLM performance without a bloated context,
preventing the model from missing critical details and enhancing the final answer quality.
5.3 Achieving Research Goal 3: Improve retrieval accuracy
TheK2RAG framework enhances passage retrieval through two key components. First, the knowledge component
identifies relevant topics for answering the question. Based on these results, the system further explores each topic
within the knowledge graph to extract more concrete and relevant chunks. This approach enables the development of
more comprehensive answers. Similar to Research Goal 2, this strategy improves the quality of the final LLM-generated
responses.
5.4 Achieving Research Goal 4: Lower Temporal and Computational Overhead
While rerankers enhance passage retrieval accuracy, they require an additional large model, increasing resource demands.
Similarly, the naive Knowledge Graph RAG pipeline depends on two LLMs: Mistral-7B-FP16 (14GB VRAM) and
Mistral-7B-Q4 (4.1GB VRAM).
In contrast, the K2RAG framework streamlines the process by utilizing only Mistral-7B-Q4 (4.1GB VRAM) for
querying the Knowledge Graph, question generation, and final answer generation, along with a small summarization
model requiring just 648MB VRAM. This results in a total VRAM usage of 4.8GB, allowing for efficient RAG pipelines
on smaller machines while significantly reducing computational and monetary costs.
Moreover, K2RAG achieves faster response times and superior answer quality compared to naive RAG pipelines.
This is evident in its reduced question-answering times relative to the naive Knowledge Graph Search RAG, enabling
K2RAG to process more queries in a span of time.
18

Running Title for Header
5.5 Limitations
While our RAG framework shows a slight mean accuracy advantage over naive methods, a mean similarity of 0.57,
despite a high Q3value, indicates limited performance gains on most questions. This suggests it excels only at certain
question types, but the narrow scope restricts broader improvement. Reliance on broad knowledge graph searches may
also reduce precision. A more focused retrieval strategy could yield more precise topic pinpointing and higher accuracy.
6 Conclusion
This paper introduces the K2RAG framework, which addresses key limitations of traditional RAG methods. By
utilizing a summarized corpus, we achieved an average reduction of 93% in training times. When combined with
quantized models and an advanced retrieval process, K2RAG outperformed all tested naive RAG methods in terms
of answer accuracy, resource efficiency, and execution speed, even with more complex steps. As a result, K2RAG
offers a more accurate and lightweight RAG pipeline. While the framework demonstrated a slight improvement in
mean similarity and a higher Q3value, it primarily excelled in answering certain question types. For future work, we
recommend testing K2RAG on additional datasets to assess its performance on a wider range of topics, and enhancing
knowledge graph retrieval techniques to further improve answer accuracy across all question types by identifying a
broader set of relevant topics.
References
[1]Jason Wei, Maarten Bosma, Vincent Y . Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai,
and Quoc V . Le. Finetuned language models are zero-shot learners, 2022.
[2]Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. Lora: Low-rank adaptation of large language models, 2021.
[3]Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized
llms, 2023.
[4]Wei Huang, Xingyu Zheng, Xudong Ma, Haotong Qin, Chengtao Lv, Hong Chen, Jie Luo, Xiaojuan Qi, Xianglong
Liu, and Michele Magno. An empirical study of llama3 quantization: From llms to mllms, 2024.
[5]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024.
[6]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive nlp tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin,
editors, Advances in Neural Information Processing Systems , volume 33, pages 9459–9474. Curran Associates,
Inc., 2020.
[7]Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the middle: How language models use long contexts, 2023.
[8]Philippe Laban, Alexander R. Fabbri, Caiming Xiong, and Chien-Sheng Wu. Summary of a haystack: A challenge
to long-context llms and rag systems, 2024.
[9]Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang,
Shizheng Li, Qi Qian, Ruicheng Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing Huang. Searching for best
practices in retrieval-augmented generation, 2024.
[10] Nicholas Rossi, Juexin Lin, Feng Liu, Zhen Yang, Tony Lee, Alessandro Magnani, and Ciya Liao. Relevance
filtering for embedding-based retrieval. In Proceedings of the 33rd ACM International Conference on Information
and Knowledge Management , CIKM ’24, page 4828–4835, New York, NY , USA, 2024. Association for Computing
Machinery.
[11] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia D’amato, Gerard De Melo, Claudio Gutierrez, Sab-
rina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo,
Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, and Antoine
Zimmermann. Knowledge graphs. ACM Computing Surveys , 54(4):1–37, July 2021.
[12] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. Blended rag: Improving rag (retriever-augmented
generation) accuracy with semantic search and hybrid query-based retrievers. In 2024 IEEE 7th International
Conference on Multimedia Information Processing and Retrieval (MIPR) , pages 155–161, 2024.
19

Running Title for Header
[13] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question
answering research. Transactions of the Association for Computational Linguistics , 7:453–466, 08 2019.
[14] Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Doug Burdick, Darrin Eide,
Kathryn Funk, Yannis Katsis, Rodney Michael Kinney, Yunyao Li, Ziyang Liu, William Merrill, Paul Mooney,
Dewey A. Murdick, Devvret Rishi, Jerry Sheehan, Zhihong Shen, Brandon Stilson, Alex D. Wade, Kuansan Wang,
Nancy Xin Ru Wang, Christopher Wilhelm, Boya Xie, Douglas M. Raymond, Daniel S. Weld, Oren Etzioni, and
Sebastian Kohlmeier. CORD-19: The COVID-19 open research dataset. In Karin Verspoor, Kevin Bretonnel
Cohen, Mark Dredze, Emilio Ferrara, Jonathan May, Robert Munro, Cecile Paris, and Byron Wallace, editors,
Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020 , Online, July 2020. Association for
Computational Linguistics.
[15] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine
comprehension of text, 2016.
[16] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018.
[17] Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: A conversational question answering challenge.
Transactions of the Association for Computational Linguistics , 7:249–266, 2019.
[18] Priyanka Mandikal and Raymond Mooney. Sparse meets dense: A hybrid approach to enhance scientific document
retrieval. CoRR , abs/2401.04055, 2024.
[19] Amanpreet Singh, Mike D’Arcy, Arman Cohan, Doug Downey, and Sergey Feldman. Scirepeval: A multi-format
benchmark for scientific document representations, 2023.
[20] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document ranking with a pretrained sequence-
to-sequence model. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Compu-
tational Linguistics: EMNLP 2020 , pages 708–718, Online, November 2020. Association for Computational
Linguistics.
[21] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Naik, Pengshan Cai, and Alfio Gliozzo.
Re2G: Retrieve, rerank, generate. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz,
editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies , pages 2701–2715, Seattle, United States, July 2022. Association for
Computational Linguistics.
[22] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. Fine-tuning llama for multi-stage text retrieval.
InProceedings of the 47th International ACM SIGIR Conference on Research and Development in Information
Retrieval , SIGIR ’24, page 2421–2425, New York, NY , USA, 2024. Association for Computing Machinery.
[23] Dawei Cheng, Fangzhou Yang, Sheng Xiang, and Jin Liu. Financial time series forecasting with multi-modality
graph neural network. Pattern Recognition , 121:108218, 2022.
[24] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng Wang, and Zheng Li.
Retrieval-augmented generation with knowledge graphs for customer service question answering. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ,
volume 33 of SIGIR 2024 , page 2905–2909. ACM, July 2024.
[25] Diego Sanmartin. Kg-rag: Bridging the gap between knowledge and creativity, 2024.
[26] Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Rohan Rao, Sunil Patel, and Stefano Pasquali. Hybridrag:
Integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction. In
Proceedings of the 5th ACM International Conference on AI in Finance , ICAIF ’24, page 608–616, New York,
NY , USA, 2024. Association for Computing Machinery.
[27] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan
Larson. From local to global: A graph rag approach to query-focused summarization, 2024.
[28] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation, 2023.
[29] Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yinfei Yang.
LongT5: Efficient text-to-text transformer for long sequences. In Marine Carpuat, Marie-Catherine de Marneffe,
and Ivan Vladimir Meza Ruiz, editors, Findings of the Association for Computational Linguistics: NAACL 2022 ,
pages 724–736, Seattle, United States, July 2022. Association for Computational Linguistics.
20

Running Title for Header
[30] Yixuan Tang and Yi Yang. Multihop-RAG: Benchmarking retrieval-augmented generation for multi-hop queries.
InFirst Conference on Language Modeling , 2024.
21
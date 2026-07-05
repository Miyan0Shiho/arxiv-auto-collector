# Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts

**Authors**: Valentin J. J. Kreileder, Johannes Reisinger, Andreas Fischer

**Published**: 2026-07-02 08:12:35

**PDF URL**: [https://arxiv.org/pdf/2607.01852v1](https://arxiv.org/pdf/2607.01852v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems use the question-answering capabilities of Large Language Models (LLMs) to access information outside their parameters. We evaluate if cluster-based semantic chunking improves retrieval and answer quality compared to fixed-size and recursive chunking evaluating on long, structured academic theses using the Retrieval Augmented Generation Assessment (RAGAs) framework. RAGAs based faithfulness shows limited reliability in this setup. Performance on fixed versus document specific questions varied substantially, likely related to the formatting of documents and preprocessing. Under the tested configuration, cluster-based chunking did not outperform simpler strategies.

## Full Text


<!-- PDF content starts -->

Evaluating Chunking Strategies for
Retrieval-Augmented Generation on Academic Texts
Valentin J. J. Kreileder
Computer Science
Deggendorf Institute of Technology
Deggendorf, Germany
kreileder@gmx.netJohannes Reisinger
Computer Science
Deggendorf Institute of Technology
Deggendorf, Germany
johannes.reisinger@th-deg.deAndreas Fischer
Computer Science
Deggendorf Institute of Technology
Deggendorf, Germany
andreas.fischer@th-deg.de
Abstract—Retrieval-Augmented Generation (RAG) systems use
the question-answering capabilities of Large Language Models
(LLMs) to access information outside their parameters. We
evaluate if cluster-based semantic chunking improves retrieval
and answer quality compared to fixed-size and recursive chunk-
ing evaluating on long, structured academic theses using the
Retrieval Augmented Generation Assessment (RAGAs) frame-
work. RAGAs based faithfulness shows limited reliability in this
setup. Performance on fixed versus document specific questions
varied substantially, likely related to the formatting of documents
and preprocessing. Under the tested configuration, cluster-based
chunking did not outperform simpler strategies.
Index Terms—Chunking, Large language model, Retrieval-
Augmented Generation, Information retrieval
I. INTRODUCTION
Large Language Models (LLMs) demonstrated impressive
generative abilities, yet their responses are limited by the
information encoded in their parameters, can suffer from hal-
lucinations, and do not show their inner workings. Retrieval-
Augmented Generation (RAG) systems address these is-
sues [1]. Long documents cannot be processed as a whole
because of the embedding models and LLMs context window,
therefore, those documents need to be split into smaller
chunks. These chunks are extracted from external documents,
which can be done with different strategies. Generated chunks
are stored in a vector database before being retrieved with a
user query. The LLM uses the query and chunks to generate an
answer. The quality of the RAG-generated answer is coupled
with the retrieval quality, the source data and the chunking
strategy. Conventional strategies are fixed-sized chunking or
format based recursive chunking. Semantic chunking gained
prominence due to its potential to improve retrieval outcomes
based on sentence similarity. However, semantic strategies like
cluster based chunking are computationally more demanding
compared to traditional methods [2]. Core contributions of this
paper are as follows:
•We establish a measure combining faithfulness and an-
swer relevancy called Answer Quality Score (AQS).
•We show issues regarding the evaluation using RAGAs
on mid-range hardware.
•We show how different chunking strategies perform
within this system using the RAGAs.II. RELATEDWORK
Qu et al. demonstrate that plain fixed-size chunking is
the most cost-effective approach [2]. Their experiments were
conducted on a mixture of datasets including both original and
artificially combined documents, differing from the documents
used in this work. Their findings may not generalize to
settings where documents are significantly larger and follow
an internal structure. We also use smaller sentence encoder
under hardware constraints, further limiting comparison. Late
Chunking makes use of a long-context model to embed a
full document, then applies chunking and mean pooling to
produce chunk vectors with better surrounding context than
pipelines embedding after chunking [3]. Within the group’s
prior work, Reisinger et al. propose document-level knowledge
graphs from the hierarchical structure of chunk embeddings to
detect document versions and plagiarism, mitigating input- and
context-conflicting hallucinations without a generative judge
as done in this work [4].
III. METHODOLOGY
A. Chunking Strategies
Chunking splits data into segments an LLM can process
within its context window, and can be categorized in three
main approaches: fixed-sized, format based, semantic chunk-
ing [4].
Fixed-Sized Chunking:Fig. 1a)depicts fixed-size chunk-
ing, which splits the corpus into uniform-length segments.
Fixed boundaries ignore semantic structure, so an overlapping
sliding window is used in practice to reduce fragmentation
across chunks [2]. The chunk sizes was 150 words with an
overlap of 15.
Recursive Chunking:Fig. 1b)displays a format based strat-
egy that splits the corpus into larger segments, progressively
subdividing those segments into smaller segments based on
predefined criteria, such as word limit or structural delimiters,
combining the computational advantages of fixed-size chunk-
ing with format-aware flexibility. Common delimiters include
sentence boundaries, or explicit markers such as newlines,
paragraphs, or punctuation marks [4]. The selected chunk-size
was 150 words.arXiv:2607.01852v1  [cs.IR]  2 Jul 2026

a) Fixed-size b) Recursive c) Cluster-based
A B C D Etoo large
C D Etoo large
D E
Final chunks
C DTopic Y
E FG1. Embed all segments
ABD
EG
Topic Z CFTopic X
2. Merge segments by cosine similarityA B
D E E FB C
F GSplit document into fixed segments 1. Split on \n\n
C D
2. Recurse on \n
3. Recurse on . ! ?
A B C FFinal chunks
D E GFinal chunks
A BGGFSource document
A B C D E F GFig. 1. Used chunking methods.
Cluster-Based Chunking:Cluster-based chunking is a
method where we combine semantically similar sentences to
more coherent chunks (Fig. 1c)). We follow [2] but useall-
MiniLM-L6-v2instead of the larger encoders tested in [2],
under hardware constraints. We derive chunk size from target
sentences and slack rather than a fixed cluster count, and apply
regex sentence splitting [5]. The configuration is shown in
Table I.
TABLE I
CLUSTERING HYPERPARAMETERS.
Parameter Description Value
λpositional/semantic weight 0.25
τdistance threshold 0.5
target sents sentences per chunk 3
slack max chunk size factor 1.2
method clustering algorithm single-linkage
model sentence encoder all-MiniLM-L6-v2
B. RAGAs
For the experiment we use the Python framework RA-
GAs [6]. It makes use of an LLM to evaluate outputs based
on prompt given by RAGAs, reducing the need for human
evaluation. The evaluation pipeline is depicted in Fig. 3, using
the custom-made benchmark referred to in Section III-D.
Under VRAM constraints (16GiB), we selected llama3.2:3b
as the generator, deepseek-r1:8b as the evaluator LLM, and
all-MiniLM-L6-v2 as the embedder.
C. Measures
TF-IDF bigram cosine similarity
We use TF-IDF cosine similarity with bigrams all theses [7].
Only bigrams that appear in at least two documents are taken
into account. The bigrams left are more domain-specific within
the corpus. Stop-words are removed before calculating the n-
grams.Context F1
Context F1 is the harmonic mean between context recall and
context precision [7]. The RAGAs variants are used because
they assess semantic relevance without requiring manually
annotated ground-truth chunks [6].
Answer Quality Score
We combine the scores faithfulnessFand answer relevancy
ARgenerated into one score namedAnswer Quality Score
(AQS). Faithfulness measures the fraction of claims in the
generated answer that are supported by the retrieved context.
RAGAs extracts and verifies them with an LLM. Answer
relevancy is the mean cosine similarity between the original
question and a set of artificial questions generated by the LLM
based on the response [6].
TheAQS-score is defined as the harmonic mean betweenF
andARto penalize cases where eitherForARis low:
AQS=2·F·AR
F+AR.
This measure penalizes confident but unsupported answers
with a high answer relevancy but low faithfulness.
D. Dataset and QA set
The dataset consists of thirteen theses, with a total of ten
associated queries. Five queries ask general questions about
the author, title, and supervisors, while the other five ask
thesis-specific questions. The theses mostly follow a faculty-
standardized format. The word-count ranges from 10,232 to
26,960 with a median of approximately 16,000. No systematic
relationship between document length and retrieval or answer
performance was observed. TF-IDF cosine similarity (as pre-
sented in Fig. 2) remains low to moderate across the corpus,
indicating shared domain without substantial content overlap.
Outliers are document004, where all comparative values are
low, and documents001and005, which show the highest
bigram similarity within the corpus.

001 002 003 004 005 006 007 008 009 010 011 012 013001 002 003 004 005 006 007 008 009 010 011 012 0131.00
0.14 1.00
0.07 0.04 1.00
0.01 0.04 0.01 1.00
0.43 0.13 0.09 0.02 1.00
0.04 0.07 0.11 0.02 0.09 1.00
0.14 0.06 0.22 0.01 0.15 0.08 1.00
0.03 0.11 0.05 0.03 0.08 0.31 0.05 1.00
0.22 0.24 0.11 0.04 0.15 0.05 0.13 0.04 1.00
0.06 0.10 0.09 0.08 0.10 0.11 0.19 0.10 0.09 1.00
0.08 0.14 0.33 0.01 0.10 0.05 0.06 0.05 0.07 0.07 1.00
0.04 0.04 0.23 0.01 0.09 0.08 0.07 0.04 0.03 0.05 0.04 1.00
0.10 0.04 0.04 0.01 0.15 0.16 0.06 0.17 0.05 0.05 0.05 0.09 1.00
0.00.20.40.60.81.0Fig. 2. TF-IDF cosine similarity with bigrams as terms.
IV. RESULTS
We ran one evaluation using the configuration in Fig 3
producing 390 measurements. Faithfulness calculation failed
in 44% of cases, compared to 2-3% for the other metrics.
The evaluator either timed out or returned invalid values.
These empty values are spread relatively evenly across all
chunkers, with rates of 47% for recursive chunking, 42% for
cluster-based chunking, and for 44% fixed-sized chunking.
We proceed with interpreting the results but with caution.
All measures are computed on valid samples, resulting in a
retained sample of around 97% for context F1 and around
55% for AQS.
Input Pipeline (per thesis)
Thesis
documents
QA set
VectorDBInitialize models &
embeddings
For each thesis
Load & chunk document
Index chunks
For each QA  pair
Retrieve context
Generate answer
Collect example
Run RAGAS evaluation
Save & Cleanup
Fig. 3. Pipeline concerning the evaluation of chunking methods using RAGAS1.2 Research Objectives . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . .
. 1 1.3 Research Questions . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . .
. . 1 1.4 Structure of the Thesis . . . . .
. . . . . . . . . . . . . . . . . . . . . .
. . . . 2 2 Background 3 2.1 Cybersecurity
Landscape . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . 3
Fig. 4. Example:Fixedquestion retrieval snippet
Context F1 Results
Forfixedquestions, context f1 medians are 0 across all
chunkers. These first five questions target general information
in the preliminaries. Even after cleaning, preliminary artifacts
and dot leaders survive, polluting both indexing and retrieval.
Forfreequestions, context f1 medians reach approximately
0.5 for recursive chunking and 0.3 for fixed-sized chunking
outperforming thefixedquestions. Cluster-based chunking still
performs poorly, though with a wider IQR (interquartile range)
not as close to zero as for thefixedquestions. An example can
be found in Fig. 4.
AQS Results
AQS forfixedquestions in Fig. 6 stays low across all
chunkers. It is higher than the corresponding context f1 but still
low, with medians near zero and outliers reaching 1.00.Free
questions outperformfixedacross all chunkers. Fixed-sized
and recursive chunking both reach a median near 0.65, with
recursive showing the tightest IQR, indicating more consistent
generation quality. Cluster-based chunking shows the lowest
median of 0.40.
Retrieval and generation are decoupled in fixed questions
where retrieval scores are near zero, while generation scores
remain moderate (as seen in Fig. 5, 6), suggesting that the
judge finds some of the answers faithful and relevant, despite
the retrieved context being largely irrelevant. This could be the
case because the generator may answer using its parametric
knowledge of generic thesis structure. The RAGAs answer
relevancy rewards question-answer independent of retrieved
context, therefore resulting in high scores. Also, narrowing
reference contexts for fixed questions structurally punishes
recall.
Discussion
This intends to reflect the capabilities of small-scale self-
hosted RAG systems. Using only academic theses, while repre-
sentative of long structured documents, yields poor benchmark
scores.
V. CONCLUSION& FUTUREWORK
This paper presented the performance of three chunking
strategies on a corpus of academic theses using the RAGAs
evaluation framework and mid-range hardware to limitations
of a less cost-intensive environment, and thus relied on small
embedding, generation, and evaluation models. The quality of
retrieved chunks offixedquestions as measured by RAGAs

Recursive Cluster-based Fixed-sized0.000.250.500.751.00Context F1
n=64 n=64 n=64Fixed questions
Recursive Cluster-based Fixed-sized
n=64 n=63 n=65Free questionsFig. 5. Boxplots displaying context F1 scores for the evaluated chunking strategies.
Recursive Cluster-based Fixed-sized0.000.250.500.751.00AQS
n=34 n=34 n=37Fixed questions
Recursive Cluster-based Fixed-sized
n=31 n=40 n=32Free questions
Fig. 6. Boxplots displaying AQS scores for the evaluated chunking strategies.
was limited, as seen in Fig. 5, while being moderate for
thefreequestions. The general answer quality was in parts
good, as seen in Fig. 6, the results have to be interpreted
with caution considering the loss of faithfulness. This loss
might be due to model capacity, the highly specialized nature
of the benchmark documents. Across all configurations in
this setup, cluster-based semantic chunking did not yield any
consistent improvement with the implemented configuration
and adds computing complexity. Simpler chunking strategies
were overall more reliable though still did not yield reliable
outputs, with the best choice depending on the document
provided.
Future work could involve evaluating different clustering al-
gorithms for cluster-based semantic chunking on the academic
dataset and the use of bigger embedding models. Instead of
the RAGAs context recall proxy, we will update the QA-set
to be used with ID-based context recall. Given the divergence
between F1 and AQS on fixed questions, a follow-up study
should validate RAGAs measures against human-annotated
chunk relevance to determine whether the gap reflects genuine
retrieval issues or metric artifacts.
ACKNOWLEDGEMENT
The first author used Claude (Opus 4.7) for a final grammar
pass. No text, figures, analysis, results, or conclusions were
generated by the tool. All technical content is the authors’
own.
REFERENCES
[1] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-Augmented Generation for Large
Language Models: A Survey,” Mar. 2024, arXiv:2312.10997 [cs].
[Online]. Available: http://arxiv.org/abs/2312.10997
[2] R. Qu, R. Tu, and F. S. Bao, “Is Semantic Chunking Worth
the Computational Cost?” inFindings of the Association for
Computational Linguistics: NAACL 2025, L. Chiruzzo, A. Ritter,
and L. Wang, Eds. Albuquerque, New Mexico: Association for
Computational Linguistics, Apr. 2025, pp. 2155–2177. [Online].
Available: https://aclanthology.org/2025.findings-naacl.114/[3] M. G ¨unther, I. Mohr, D. J. Williams, B. Wang, and H. Xiao,
“Late Chunking: Contextual Chunk Embeddings Using Long-Context
Embedding Models,” Jul. 2025, arXiv:2409.04701 [cs]. [Online].
Available: http://arxiv.org/abs/2409.04701
[4] J. Reisinger, A. Fischer, and A. Igl, “Semantic Document
Graphs for Knowledge Retrieval,” in2025 2nd International
Generative AI and Computational Language Modelling Conference
(GACLM), Aug. 2025, pp. 294–298. [Online]. Available:
https://ieeexplore.ieee.org/document/11231968
[5] “sentence-transformers/all-MiniLM-L6-v2 · Hugging Face,” Jan. 2024.
[Online]. Available: https://huggingface.co/sentence-transformers/all-
MiniLM-L6-v2
[6] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, “RAGAs:
Automated Evaluation of Retrieval Augmented Generation,” in
Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demonstrations,
N. Aletras and O. De Clercq, Eds. St. Julians, Malta: Association for
Computational Linguistics, Mar. 2024, pp. 150–158. [Online]. Available:
https://aclanthology.org/2024.eacl-demo.16/
[7] C. D. Manning, P. Raghavan, and H. Sch ¨utze, “Introduction to
Information Retrieval,” Jul. 2008, iSBN: 9780511809071. [Online]. Avail-
able: https://www.cambridge.org/highereducation/books/introduction-to-
information-retrieval/669D108D20F556C5C30957D63B5AB65C
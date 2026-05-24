# A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering

**Authors**: Sereiwathna Ros, Phannet Pov, Ratanaktepi Chhor, Kimleang Ly, Wan-Sup Cho, Saksonita Khoeurn

**Published**: 2026-05-21 07:36:01

**PDF URL**: [https://arxiv.org/pdf/2605.22099v1](https://arxiv.org/pdf/2605.22099v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language model (LLM) outputs in retrieved evidence, thereby reducing hallucination and improving factual accuracy. Its efficacy, however, remains largely unexamined for low-resource, non-Latin-script languages such as Khmer. In this paper, we present a RAG-based question answering system for Khmer-language telecom-domain documents. We conduct a two-phase comparative evaluation. First, we benchmark three embedding models: BGE-M3 (567M), Jina-Embeddings-v3 (570M), and Qwen3-Embedding (597M), for dense retrieval over Khmer documents. BGE-M3 consistently performs best, achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Precision@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3 as the selected retriever, we evaluate five generator backends: Qwen3 (8B), Qwen3.5 (9B), Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT, on a curated golden dataset of 200 Khmer question-answer pairs. To quantify system performance, we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance, factual correctness, answer similarity, and answer correctness. The results show no single model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859) and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599). These findings highlight that retriever choice remains a major bottleneck for Khmer RAG, while generator strengths vary depending on whether the priority is grounding, factual precision, or semantic similarity.

## Full Text


<!-- PDF content starts -->

A Comparative Study of Language Models for Khmer
Retrieval-Augmented Question Answering
Sereiwathna Ros1, Phannet Pov2, Ratanaktepi Chhor2, Kimleang Ly3, Wan-Sup Cho4,5,
and Saksonita Khoeurn∗4,5
1Department of Computer Science, Chungbuk National University, Cheongju, South Korea
2Department of Big Data, Chungbuk National University, Cheongju, South Korea
3General Department of Information and Communication Technology, Ministry of Post and
Telecommunications, Phnom Penh, Cambodia
4Department of Management Information Systems, Chungbuk National University, Cheongju, South
Korea
5BigDatalabs Co., Ltd, Cheongju, South Korea
Abstract
Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for
grounding large language model (LLM) outputs in retrieved evidence, thereby reducing
hallucination and improving factual accuracy. Its efficacy, however, remains largely un-
examined for low-resource, non-Latin-script languages such as Khmer. In this paper,
we present a RAG-based question answering system for Khmer-language telecom-domain
documents. We conduct a two-phase comparative evaluation. First, we benchmark three
embeddingmodels—BGE-M3(567M),Jina-Embeddings-v3(570M),andQwen3-Embedding
(597M)—for dense retrieval over Khmer documents. BGE-M3 consistently performs best,
achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Preci-
sion@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3
as the selected retriever, we evaluate five generator backends—Qwen3 (8B), Qwen3.5 (9B),
Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT—on a curated
golden dataset of 200 Khmer question-answer pairs. To quantify system performance,
we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance,
factual correctness, answer similarity, and answer correctness. The results show no single
model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859)
and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380),
and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity
(0.836), and answer correctness (0.599). These findings highlight that retriever choice
remains a major bottleneck for Khmer RAG, while generator strengths vary depending on
whether the priority is grounding, factual precision, or semantic similarity.
Keywords:retrieval-augmentedgeneration, RAGevaluation, RAGASmetrics, Khmerquestion
answering, Khmer NLP, local LLMs, dense retrieval, low-resource languages
1 Introduction
Retrieval-Augmented Generation (RAG) [1] has become a common approach for question an-
swering over domain-specific document collections because it combines external retrieval with
the generative capabilities of Large Language Models (LLMs). In such systems, performance de-
pendsnotonlyonthegenerator, butalsoonthequalityofretrievalandonwhetherthegenerated
∗Corresponding author: saksonita@chungbuk.ac.kr
1arXiv:2605.22099v1  [cs.CL]  21 May 2026

answer remains grounded in the retrieved evidence. As a result, evaluating RAG is inherently
multi-dimensional: a system may fail because relevant evidence is not retrieved, because the
model does not use the retrieved context effectively, or because it hallucinates unsupported
content. Existing evaluation practices, however, remain heavily shaped by English-centric set-
tings and do not always transfer cleanly to low-resource languages or institutionally sensitive
domains [2, 3, 4].
These limitations are especially important for Khmer document Question Answering (QA).
Khmer is a low-resource language written in a complex Abugida script, with limited annotated
resources and weak standardization for word segmentation. These characteristics introduce chal-
lenges at multiple stages of a RAG pipeline, including text extraction, document preprocessing,
retrieval, and answer evaluation. In institutional settings, the problem is further compounded
by the need for trustworthy responses grounded in authoritative documents, since hallucinated
or weakly supported answers can undermine public confidence [5]. Despite growing interest in
multilingual and low-resource Natural Language Processing (NLP), it remains unclear which
retrieval models, which locally deployable generators, and which automated evaluation signals
are most suitable for Khmer-language RAG systems.
In this paper, we present a systematic study of retrieval-augmented question answering over
Khmer institutional documents. Our focus is not only on end-to-end answer quality, but also
on the interaction between retrieval, generation, and automated evaluation in a low-resource,
non-Latin-script setting. We study a privacy-preserving RAG pipeline built for locally hosted
deployment over Khmer telecom-domain documents, allowing us to examine the practical re-
quirements of document-grounded question answering under data-sovereignty constraints.
Our contributions are threefold:
1. We benchmark dense retrievers for Khmer document retrieval and analyze their effective-
ness on noisy, domain-diverse institutional text.
2. We compare five locally deployable generator models, including both general-purpose mul-
tilingual LLMs and Southeast Asian-focused models, to assess whether regional special-
ization yields measurable gains for Khmer question answering.
3. We examine six adapted RAGAS-style metrics in this setting and discuss their usefulness
and limitations for evaluating Khmer RAG pipelines.
To support this study, we construct a gold evaluation set of 200 Khmer question–answer
pairs derived from authoritative documents across multiple institutional subdomains. Using
this benchmark, we provide an empirical account of retrieval quality, answer quality, and metric
behavior in Khmer document QA. Our findings show that retriever choice has a substantial
effect on downstream performance, while the relative strengths of generator models vary across
grounding-oriented and similarity-oriented evaluation measures. More broadly, this paper high-
lights the need for language-aware RAG evaluation practices in low-resource settings, and offers
evidence that methods validated in English should not be assumed to behave identically for
Khmer.
2 Related Work
RAG combines a retriever with an LLM so that answers are generated from retrieved evidence
rather than from parametric memory alone [1]. In a typical pipeline, documents are collected,
segmented into passages, indexed, and retrieved at inference time to condition answer gener-
ation [2]. Because end-to-end performance depends jointly on retrieval quality and response
grounding, RAG has become a widely used framework for knowledge-intensive tasks. Early sys-
tems often relied on dense retrieval methods such as Dense Passage Retrieval (DPR) [6], while
more recent multilingual retrievers such as BGE-M3 [7] aim to improve transfer across languages
2

and scripts. These developments are especially relevant in low-resource settings, where retrieval
can be degraded by limited training data, Optical Character Recognition (OCR) artifacts, and
script-sensitive preprocessing challenges [8].
Traditional reference-based metrics such as Bilingual Evaluation Understudy (BLEU) and
Recall-Oriented Understudy for Gisting Evaluation (ROUGE) measure lexical overlap with gold
references [9, 10], while embedding-based metrics such as Bidirectional Encoder Representations
fromTransformersScore(BERTScore)bettercapturesemanticsimilaritythroughcontextualized
representations [11]. However, these approaches are not fully adequate for RAG because they
do not directly assess whether a response is supported by the retrieved evidence. This limitation
is particularly important in low-resource settings, where high-quality reference sets are costly to
construct and correct answers may show substantial lexical and syntactic variation.
Recent work therefore moves beyond answer similarity toward explicit assessment of ground-
ing and context use. GEval demonstrates that LLM-as-a-judge methods can support flexible
rubric-based evaluation [12]. RAGAS adapts this idea to RAG pipelines through metrics such as
faithfulness, answer relevance, and context-related measures [3, 13]. Related benchmarks such as
RGB further stress-test RAG systems under noisy retrieval and counterfactual conditions [14].
In an applied telecom setting, Roychowdhury et al. [4] likewise report that grounding-oriented
measures such as faithfulness and factual correctness align more closely with expert judgment
than similarity-based metrics. Taken together, this literature suggests that RAG evaluation
should account not only for output quality, but also for evidential support.
Although RAG and LLM evaluation have advanced rapidly, most evidence still comes from
English and other high-resource languages. For Khmer, this creates an important gap: doc-
ument processing and retrieval must contend with a non-Latin script, inconsistent word seg-
mentation, OCR noise, and limited task-specific resources. Regionally focused models such as
SEA-LION [15] and Sailor2 [16] indicate growing support for Southeast Asian languages, but
they do not by themselves establish how well retrieval models, generator models, and automated
evaluation metrics behave in Khmer institutional QA settings.
Ly et al. [17] conducted a study in which they prepared a dataset of questions and corre-
sponding Khmer answers to perform fine-tuning experiments on large language models (LLMs)
for the Khmer language. To evaluate the generated answers, the authors employed similarity-
based metrics that compare the model outputs with reference answers. Specifically, ROUGE-1,
ROUGE-2, and ROUGE-L were used to measure unigram overlap, bigram overlap, and the
longest common subsequence, respectively. These metrics provide insights into how closely the
Data Ingestion
Inference
RAGAS FrameworkDocuments
Khmer source filesRecursive chunking
with metadata extractionEmbedding
BGE-M3 modelVector database
Dense vector index
User query
Khmer input question
Embedding
BGE-M3 (Encoder)Cosine Similarity Retrieval
Top-k chunks
LLM generation
Context-conditional responseResponseRAGAS 
MetricsFaithfulness
Context relevance
Factual CorrectnessAnswer Relevance
Answer Similarity
Answer CorrectnessRAG Pipeline with RAGAS Evaluation
Figure 1: System architecture of the RAG pipeline.
3

generated responses align with the ground truth answers. In contrast, our work generates an-
swers based on retrieved contextual information rather than relying on direct comparison with
predefined target answers. Therefore, traditional similarity-based evaluation metrics may not
fully capture the quality of the responses, as they do not adequately reflect the relevance and
faithfulness of the generated answers to the provided context.
More broadly, prior work on Khmer and related low-resource language processing suggests
that language-specific preprocessing and resource constraints can materially affect downstream
system performance. Our study builds on this perspective by evaluating retrieval quality, answer
generation, and automated RAG assessment jointly in a Khmer-language institutional document
environment.
Overall, this work is positioned at the intersection of multilingual RAG, grounding-aware
evaluation, and low-resource language processing. Unlike prior work centered on English, our
focus is not only whether RAG works, but whether its retrieval and evaluation assumptions
remain reliable in a Khmer-language setting.
3 Methodology
Our experimental setup follows a standard RAG pipeline comprising a retriever followed by
a generator module. Figure 1 shows the schematic of our pipeline. The input to the system
is a user query in Khmer or English, which is processed through dense retrieval followed by
LLM-based answer generation conditioned on the retrieved context.
3.1 Dataset
We collected open-source data from websites that publish official documents related to Informa-
tion and Communication Technology (ICT). The data sources include notifications, guidelines,
laws, announcements, decrees, sub-decrees, government documents, press releases, Q&A docu-
ments, decisions, and general information. The corpus is mostly Khmer language documents
with embedded English technical terms. The focus on closed-domain institutional documents
enables this research to create a specific and verifiable knowledge domain, which is critical in
building question answering systems in a low-resource language setting.
The documents are preprocessed into Markdown format and recursively chunked into seg-
ments that preserve semantic coherence. Each chunk carries metadata including the source
document identifier and a unique chunk ID for provenance tracking. The resulting corpus con-
tains over 7,000 chunks.
For evaluation, we curate a golden dataset comprising 200 question–answer pairs derived
from the telecom domain document corpus. The questions span multiple domains and reflect
realistic citizen queries written in Khmer. Each entry consists of: (1) a question, (2) a target
answer (ground truth) composed by domain experts, and (3) domain metadata (document ID,
question ID, domain category) for stratified analysis.
3.2 Retriever Models
The retriever module is based on dense passage retrieval. An encoder-based language model
computes embeddings for both the query and the document chunks. For every query embedding,
the retriever outputs the top-kmost similar chunk embeddings using cosine similarity.
We evaluate three embedding models to determine which provides the most effective dense
retrieval for Khmer documents:
1.BGE-M3(567M) [7]: Supports multi-lingual, multi-functionality, and multi-granularity
embeddings through self-knowledge distillation, achieving competitive performance across
100+ languages including Southeast Asian scripts. Served through Ollama [18].
4

2.Jina-Embeddings-v3(570M) [19]: An embedding model supporting 89+ languages with
task-specific adapters for retrieval, classification, and semantic similarity. Served through
Ollama.
3.Qwen3-Embedding(597M) [20]: A compact embedding model from the Qwen family
designed for semantic search across multiple languages. Served through Ollama.
Document chunks are embedded in batches during the offline stage, and the complete vector
databaseisserializedforefficientruntimeloading. Atquerytime, thetop-kchunks(defaultk=3)
are selected and concatenated with similarity scores and source metadata into a structured
context string. The retriever is evaluated using standard information retrieval metrics: Hit
Rate@k, File-level Hit Rate@k, Mean Reciprocal Rank (MRR@k), and Precision@k.
3.3 Generator Models
Once the relevant context has been retrieved for a query, the query and context are passed to
the LLM for generating the response. A system prompt constrains the LLM to use only the
provided context and not generate information beyond it.
Table 1: Summary of models used in the experimental setup.
Role Model Params Runtime
Embedding BGE-M3 567M Ollama
Embedding Jina-Embeddings-v3 570M Ollama
Embedding Qwen3-Embedding 600M Ollama
Generator Qwen3 8B Ollama
Generator Qwen3.5 9B Ollama
Generator Sailor2-Chat 8B HuggingFace
Generator SeaLLMs-v3-7B 7B HuggingFace
Generator Llama-SEA-LION-v2 8B HuggingFace
Judge GPT-4o-mini – OpenAI API
We evaluate five LLM backends with different sizes and linguistic focus:
1.Qwen3 (8B)[21]: A general-purpose model from the Qwen family with broad language
coverage. Served locally via Ollama.
2.Qwen3.5 (9B)[22]: A newer-generation model from the Qwen family with improved
instruction-following capabilities. Served locally via Ollama.
3.Sailor2-8B-Chat[16]: An 8B-parameter model specifically trained for Southeast Asian
language understanding. Inference is performed using HuggingFace Transformers.
4.SeaLLMs-v3-7B-Chat[23]: A 7B chat model optimized for Southeast Asian languages.
Inference is performed using HuggingFace Transformers.
5.Llama-SEA-LION-v2-8B-IT[15]: An 8B instruction-tuned SEA-LION family model
targeting Southeast Asian multilingual use. Inference is performed using HuggingFace
Transformers.
All local inference is performed on a single machine with one NVIDIA GPU H200.
3.4 Evaluation Metrics
We focus on six adapted metrics inspired by the RAGAS framework [3]. Higher value is better
for all of them. The evaluation employs GPT-4o-mini [24] as the LLM judge and BGE-M3 [7] for
5

computing semantic similarity. We use the notation of Roychowdhury et al. [4]: given question
qand contextc(q)retrieved from the corpus, the LLM generates answera(q). The ground truth
answer is denotedgt(q).
•Faithfulness (FaiFul):Checks if the generated statements froma(q)are present in the
retrieved contextc(q)through verdicts; the ratio of valid verdictsVto total number of
statementsS(q)is the answer’s faithfulness:
FaiFul=|V|
|S(q)|(1)
•Answer Relevance (AnsRel):The average cosine similarity of the user’s questionqwith
N=3generated questions˜q i, usinga(q)as reference, is the answer relevance:
AnsRel=1
NNX
i=1sim(E(q), E(˜q i))(2)
•Context Relevance (ConRel):The semantic similarity between the questionqand the
retrieved contextc(q)computed using BGE-M3 embeddings:
ConRel= sim(E(q), E(c(q)))(3)
•Answer Similarity (AnsSim):The similarity between the embedding ofa(q)and the
embedding ofgt(q):
AnsSim= sim(E(a(q)), E(gt(q)))(4)
Table 2: Retriever evaluation results on the 200-question golden dataset. Bold values indicate
the best score per metric.↑indicates higher is better.
Metric(↑)BGE-M3 Jina v3 Qwen3-Emb
(567M) (570M) (597M)
Hit Rate@10.1700.075 0.110
Hit Rate@30.2850.135 0.175
Hit Rate@50.3550.170 0.205
Hit Rate@100.3850.205 0.225
File Hit Rate@30.7000.485 0.525
File Hit Rate@50.8050.575 0.620
File Hit Rate@100.8350.645 0.675
MRR@30.2210.099 0.141
Precision@30.1120.055 0.065
Cosine Sim (top-1) 0.7040.7590.686
•Factual Correctness (FacCor):The F1-Score of statements ina(q)classified as True
Positive (TP), False Positive (FP) and False Negative (FN) by the LLM judge with respect
togt(q):
FacCor=|TP|
|TP|+ 0.5×(|FP|+|FN|)(5)
•Answer Correctness (AnsCor):Determines correctness ofa(q)with respect togt(q)as
a weighted sum of factual correctness and answer similarity:
AnsCor=w 1×FacCor+w 2×AnsSim(6)
with weights[w 1, w2] = [0.5,0.5]in our implementation.
6

For each of the 200 questions, the evaluation pipeline: (1) retrieves the top-3 document
chunks from the vector database, (2) generates an LLM answer conditioned on the retrieved
context, (3)recordsthequestion, targetanswer, LLManswer, andretrievedchunks, and(4)com-
putes the six metrics. The evaluation is conducted separately for each of the five LLM backends.
For Qwen3 and Qwen3.5, evaluation is run via Ollama. For Sailor2-8B-Chat, SeaLLMs-v3-
7B-Chat, and Llama-SEA-LION-v2-8B-IT, inference is performed using HuggingFace Trans-
formers. The generation evaluation uses BGE-M3 as the retriever, which was identified as the
best-performing embedding model in our retriever comparison (Section 4.1).
4 Results and Discussion
The results of the RAG evaluation are presented in two parts. First, we compare the three
embedding models for retrieval quality (Table 2). Then, using BGE-M3 as the selected retriever,
we evaluate the five generator models on the six RAGAS metrics (Table 3).
4.1 Retriever Evaluation
Table 2 presents the retrieval performance of the three embedding models evaluated on the 200-
question golden dataset. We report Hit Rate (the fraction of queries for which the correct chunk
appears in the top-kresults), File-level Hit Rate (the fraction for which a chunk from the correct
source file appears), Mean Reciprocal Rank (MRR), and Precision at different values ofk.
Table 3: RAGAS metrics for the 200-question golden dataset withk=3retrieved contexts (BGE-
M3 retriever). Numbers are mean scores. Bold values indicate the best score per metric.↑
indicates higher is better.
Metric(↑)Qwen3 (8B) Qwen3.5 (9B) Sailor2 (8B) SeaLLMs (7B) Llama-SEA-LION (8B)
FaiFul0.7800.8590.758 0.846 0.556
AnsRel0.808 0.779 0.8020.8670.831
ConRel0.7180.7260.717 0.718 0.718
FacCor0.3800.303 0.258 0.352 0.217
AnsSim0.648 0.661 0.6060.8360.766
AnsCor0.521 0.480 0.4320.5990.488
BGE-M3 achieves the highest scores across all retrieval effectiveness metrics, with a Hit
Rate@3of0.285—morethandoublethatofJina-Embeddings-v3(0.135)andsubstantiallyhigher
than Qwen3-Embedding (0.175). The advantage is even more pronounced at the file level: BGE-
M3 retrieves a chunk from the correct source file 70% of the time atk=3, compared to 52.5%
for Qwen3-Embedding and 48.5% for Jina v3. Interestingly, Jina v3 achieves the highest top-1
cosine similarity (0.759), yet this does not translate into better retrieval accuracy, consistent
with cautions about interpreting raw cosine similarity as a quality signal [25]. Based on these
results, we select BGE-M3 as the retriever for the generator evaluation.
4.2 Generator Evaluation
Table 3 presents the RAGAS metrics for the five generator models on the 200-question golden
dataset using BGE-M3 as the retriever.
4.3 Discussion on Metrics
We discuss our findings about the six RAGAS metrics and their behaviour in the context of
Khmer document QA.
7

Faithfulness (FaiFul).Qwen3.5-9B achieves the highest faithfulness score of 0.859, with
SeaLLMs-v3-7B-Chat close behind at 0.846. Qwen3-8B and Sailor2-8B remain in a similar
middle range (0.780 and 0.758), whereas Llama-SEA-LION-v2-8B-IT drops markedly to 0.556.
This ranking suggests that strong multilingual or regional coverage alone is insufficient; what
matters is whether the model can stay tightly grounded in the retrieved evidence. The result is
consistent with the findings of Roychowdhury et al. [4], who report that faithfulness is generally
concordant with manual evaluation by subject matter experts.
Table 4: Summary of key findings per metric.
Metric Key Conclusion
FaiFulQwen3.5 wins: most grounded in context (0.859).
AnsRelSeaLLMs wins: strongest question-level relevance (0.867).
ConRelNear tie: retrieval quality dominates; LLM impact is small.
FacCorQwen3 wins: highest factual accuracy (0.380).
AnsSimSeaLLMs wins: closest to ground truth semantically (0.836).
AnsCorSeaLLMs wins: best overall balance of accuracy and semantics (0.599).
Answer Relevance (AnsRel).SeaLLMs-v3-7B-Chat achieves the highest answer relevance
(0.867), followed by Llama-SEA-LION-v2-8B-IT (0.831), Qwen3-8B (0.808), Sailor2-8B (0.802),
and Qwen3.5-9B (0.779). Unlike the earlier three-model comparison, the inclusion of two ad-
ditional Southeast Asian-focused models shows that regional specialization can improve topical
responsiveness to Khmer questions. However, as noted in the literature [25, 4], this metric relies
on cosine similarity between generated and original question embeddings and should therefore
be interpreted cautiously as a relative signal rather than an absolute measure.
Context Relevance (ConRel).All five models achieve nearly identical context relevance
scores, ranging only from 0.717 to 0.726, with Qwen3.5-9B slightly ahead at 0.726. This is
expected because context relevance primarily reflects retrieval quality, which is determined by
the shared BGE-M3 embedding model and is largely independent of the LLM backend. The
tight clustering confirms that retrieval quality remains a system-level bottleneck rather than a
differentiating property of the generators themselves. As observed by Roychowdhury et al. [4],
context relevance is mainly indicative and dependent on context length, making it difficult to
interpret as a standalone measure of answer quality.
Factual Correctness (FacCor).Qwen3-8B achieves the highest factual correctness (0.380),
with SeaLLMs-v3-7B-Chat second at 0.352, followed by Qwen3.5-9B (0.303), Sailor2-8B (0.258),
and Llama-SEA-LION-v2-8B-IT (0.217). The overall low scores across all models reflect the
difficulty of the task: documents contain highly specific information such as phone numbers,
URLs, and legal article numbers, where partial matches yield low F1 scores. Importantly, the
ranking differs from faithfulness: Qwen3.5-9B is best grounded in retrieved context, but Qwen3-
8B more accurately reproduces factual details from the ground truth. Together with faithfulness,
this metric has been found to be most aligned with expert evaluation [4] and remains one of the
most informative measures in our setting.
Answer Similarity (AnsSim) and Answer Correctness (AnsCor).SeaLLMs-v3-7B-Chat
achieves the highest answer similarity (0.836) and also the highest composite answer correctness
(0.599), indicating the strongest overall balance between semantic closeness to the reference
answer and factual adequacy. Llama-SEA-LION-v2-8B-IT also scores relatively high on answer
similarity (0.766) but falls back to 0.488 on answer correctness because of its weak factual
correctness. By contrast, Qwen3-8B remains competitive on answer correctness (0.521) despite
8

lower answer similarity, because its stronger factual correctness compensates for the gap. This
again shows that semantic similarity and factual accuracy are distinct dimensions of quality.
Table 4 summarizes the key conclusions.
In summary, our results indicate that of these metrics,FaiFulandFacCor(and hence
AnsCor) are perhaps best aligned with human expert judgment for our domain; scores for
AnsSim,AnsRelandConRelare subject to inherent variations as discussed above. This echoes
the findings of Roychowdhury et al. [4] for the telecom domain.
4.4 Retriever Model Selection
A key finding is that BGE-M3 substantially outperforms both Jina-Embeddings-v3 and Qwen3-
Embedding across all retrieval effectiveness metrics. Atk=3, BGE-M3 achieves more than
double the Hit Rate of Jina v3 (0.285 vs. 0.135) and a 63% higher Hit Rate than Qwen3-
Embedding (0.285 vs. 0.175). The gap widens at the file level, where BGE-M3 retrieves from
the correct source file 70% of the time compared to 52.5% (Qwen3-Embedding) and 48.5% (Jina
v3).
An interesting observation is that Jina v3 achieves the highest mean cosine similarity for
its top-1 retrieved chunk (0.759), yet produces the worst retrieval accuracy. This highlights a
known limitation of relying on raw cosine similarity as a retrieval quality indicator [25]: a model
may assign high similarity scores to semantically related but non-matching passages. BGE-M3’s
self-knowledge distillation approach appears to produce embeddings that better discriminate
between truly relevant and merely related content for Khmer documents.
Nevertheless, even the best retriever (BGE-M3) achieves a relatively modest Hit Rate@3 of
0.285, indicating that retrieval quality remains a primary bottleneck in the overall RAG pipeline.
This is reflected in the tightly clusteredConRelscores (∼0.72) observed across all five generator
models.
4.5 Language Focus vs. General Multilingual Models
With the addition of SeaLLMs-v3-7B-Chat and Llama-SEA-LION-v2-8B-IT, the comparison
between general-purpose and Southeast Asian language-focused models becomes more nuanced.
Regional specialization does not uniformly help or hurt Khmer RAG performance. SeaLLMs
is the strongest model on answer relevance, answer similarity, and answer correctness, while
Qwen3.5-9B remains best on faithfulness and Qwen3-8B remains best on factual correctness.
Within the Qwen family, an interesting trade-off remains: Qwen3.5-9B achieves higher faith-
fulness (0.859 vs. 0.780) and answer similarity (0.661 vs. 0.648), while Qwen3-8B leads on factual
correctness (0.380 vs. 0.303) and answer correctness (0.521 vs. 0.480). This suggests that the
newer, slightly larger Qwen3.5 model is better at grounding answers in retrieved context, while
Qwen3 more precisely reproduces factual details from the ground truth.
The three Southeast Asian-focused models also separate clearly from one another. SeaLLMs
appears to transfer well to Khmer answer generation, whereas Sailor2 and especially Llama-
SEA-LION struggle more on grounding-oriented metrics. Several hypotheses may explain this
spread: (1) the amount and quality of Khmer data may differ substantially across SEA-focused
pretraining corpora; (2) instruction-following ability under constrained RAG prompting may
matter more than regional coverage alone; and (3) differences in tokenizer behaviour, chat tem-
plates, and HuggingFace inference settings may influence answer style and grounding. The
broader conclusion is that regional language targeting can help, but the effect is strongly model-
specific rather than guaranteed across all models in the same family.
From a deployment perspective, model choice depends on the target objective. For appli-
cations where faithfulness to retrieved context is paramount, Qwen3.5-9B is the best choice.
For applications where factual accuracy matters most, Qwen3-8B is preferable. For applications
9

prioritizing overall answer quality and semantic closeness to the target response, SeaLLMs-v3-
7B-Chat is the strongest option among the evaluated models.
A key limitation of this study is that all reported scores are based on automated evaluation.
Although prior work in the telecom domain suggests that faithfulness and factual correctness
align more closely with expert judgment than similarity-based metrics, we did not conduct a
dedicated human evaluation for Khmer in this study. Therefore, our conclusions about metric
reliability should be interpreted as evidence from automated analysis rather than definitive
validation against human annotations.
4.6 Limitations
•Fixed retrieval.The current system uses a single dense retrieval pass with fixedk=3.
Adaptive retrieval or iterative refinement could improve context quality.
•Automated evaluation bias.All metrics rely on LLM judges and English-centric em-
bedding models, potentially introducing systematic bias for Khmer evaluation.
•Limited ground truth.The 200-question golden dataset, while carefully curated, may
not capture the full distribution of citizen queries.
•No human evaluation.We did not conduct human evaluation of answer quality, which
would provide a more reliable assessment.
•Inference setup differences.Sailor2-8B was evaluated with a different inference frame-
work (HuggingFace) than the Qwen models (Ollama), which may introduce confounding
factors.
5 Conclusion and Future Work
In this work, we presented a retrieval-augmented generation system for Khmer document ques-
tion answering using locally deployed language models. We conducted a two-fold comparative
evaluation: a retriever comparison of three embedding models, and a generator comparison of
five LLMs (Qwen3-8B, Qwen3.5-9B, Sailor2-8B, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-
v2-8B-IT) on 200 question–answer pairs across six RAGAS-inspired metrics. Our results show
that verdict-based RAGAS metrics (FaiFulandFacCor) provide more reliable evaluation sig-
nals for Khmer text than similarity-based metrics (AnsRelandAnsSim), which remain sensitive
to embedding-model choice and the inherent limitations of cosine similarity. This is consistent
with findings from the telecom domain [4] and suggests that these concerns also extend to
low-resource, non-Latin-script language settings. Among the retrievers, BGE-M3 substantially
outperformsJina-Embeddings-v3andQwen3-Embeddingonallretrievalmetrics, achievingmore
than double the Hit Rate@3 of Jina v3. Notably, high cosine similarity alone, as observed with
Jina v3, does not necessarily translate into strong retrieval accuracy, reinforcing the importance
of task-specific retriever evaluation. For generation, performance cannot be explained by a sim-
ple general-purpose-versus-regional distinction. Qwen3.5-9B achieves the highest faithfulness
(0.859), Qwen3-8B leads in factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best
on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599), whereas
Llama-SEA-LION-v2-8B-IT underperforms on faithfulness and factual correctness. These find-
ings indicate that regional specialization can benefit Khmer question answering, but its impact
is strongly model-dependent. Additionally, retrieval quality emerges as a significant bottleneck,
as reflected in the modest Hit Rate@3 (0.285) and the tightly clustered context relevance scores
(∼0.72) across all five generator models.
Future work will focus on: (1) improving retrieval through hybrid dense+sparse methods
and query expansion; (2) incorporating Khmer-specific tokenization and embedding models to
10

improve both retrieval and evaluation quality; (3) conducting human evaluation studies with
citizens to validate automated metrics against expert judgments; (4) exploring domain adapta-
tion through instruction fine-tuning of smaller models on Khmer data, which has been shown to
improve metric concordance with expert evaluation [4]; and (5) extending the system to support
cross-lingual QA where questions and documents may be in different languages.
References
[1] PatrickLewis, EthanPerez, AleksandraPiktus, FabioPetroni, VladimirKarpukhin, Naman
Goyal, Heinrich K"uttler, Mike Lewis, Wen tau Yih, Tim Rockt"aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–
9474, Red Hook, NY, USA, 2020. Curran Associates, Inc.
[2] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language
models: A survey, 2024.
[3] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated
evaluation of retrieval augmented generation. InProceedings of the 18th Conference of the
European Chapter of the Association for Computational Linguistics: System Demonstra-
tions, pages 150–158, St. Julian’s, Malta, 2024. Association for Computational Linguistics.
[4] Sujoy Roychowdhury, Sumit Soman, H G Ranjani, Neeraj Gunda, Vansh Chhabra, and
Sai Krishna Bala. Evaluation of RAG metrics for question answering in the telecom domain.
InICML 2024 Workshop on Foundation Models in the Wild, pages 1–7, Vienna, Austria,
2024. PMLR. arXiv:2407.12873.
[5] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM Computing Surveys, 55(12):1–38, 2023.
[6] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
DanqiChen, andWentauYih. Densepassageretrievalforopen-domainquestionanswering.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 6769–6781, Online, 2020. Association for Computational Linguistics.
[7] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-
embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through
self-knowledge distillation, 2024.
[8] Sara Bourbour Hosseinbeigi, Sina Asghari, Mohammad Ali Seif Kashani, Mohammad Hos-
sein Shalchian, and Mohammad Amin Abbasi. Advancing retrieval-augmented generation
for persian: Development of language models, comprehensive benchmarks, and best prac-
tices for optimization, 2025.
[9] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A method for
automatic evaluation of machine translation. InProceedings of the 40th Annual Meeting
of the ACL, pages 311–318, Philadelphia, PA, USA, 2002. Association for Computational
Linguistics.
[10] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81, Barcelona, Spain, 2004. Association for Computational
Linguistics.
11

[11] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
BERTScore: Evaluating text generation with BERT, 2019.
[12] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu.
G-eval: Nlg evaluation using gpt-4 with better human alignment. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522,
Singapore, 2023. Association for Computational Linguistics.
[13] Exploding Gradients. Ragas: Retrieval augmented generation assessment.https://gith
ub.com/vibrantlabsai/ragas, 2026. GitHub repository, accessed 2026-03-12.
[14] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models
in retrieval-augmented generation. InProceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 17754–17762, Vancouver, Canada, 2024. AAAI Press.
[15] Raymond Ng, Thanh Ngan Nguyen, Yuli Huang, Ngee Chia Tai, Wai Yi Leong, Wei Qi
Leong, Xianbin Yong, Jian Gang Ngui, Yosephine Susanto, Nicholas Cheng, Hamsaward-
hini Rengarajan, Peerat Limkonchotiwat, Adithya Venkatadri Hulagadri, Kok Wai Teng,
Yeo Yeow Tong, Bryan Siow, Wei Yi Teo, Wayne Lau, Choon Meng Tan, Brandon Ong,
Zhi Hao Ong, Jann Railey Montalan, Adwin Chan, Sajeban Antonyrex, Ren Lee, Es-
ther Choa, David Ong Tat-Wee, Bing Jie Darius Liu, William Chandra Tjhi, Erik Cam-
bria, and Leslie Teo. Sea-lion: Southeast asian languages in one network, 2025. URL
https://arxiv.org/abs/2504.05747.
[16] Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao
Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong
Feng, Xin Mao, Man Tsung Yeung, Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Ky-
dlíček, Zeyi Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Niratti-
sai Thongchim, Taechawat Konkaew, Narong Borijindargoon, Anh Dao, Matichon Manee-
gard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphai-
bun, Hoang H. Tran, Mike Zhang, Shiqi Chen, Tianyu Pang, Chao Du, Xinyi Wan, Wei
Lu, and Min Lin. Sailor2: Sailing in south-east asia with inclusive multilingual llms, 2025.
URLhttps://arxiv.org/abs/2502.12982.
[17] Kimleang Ly, Dona Valy, and Phutphalla Kong. Fine-tuning for question answering in
low-resource languages: A case study on khmer. In2024 17th International Congress
on Advanced Applied Informatics (IIAI-AAI-Winter), pages 162–165, Kitakyushu, Japan,
2024. IEEE.
[18] Ollama. Ollama: Run large language models locally.https://ollama.com, 2024.
[19] SabaSturua,IsabelleMohr,MohammadKalimAkram,MichaelGünther,BoWang,Markus
Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, and Han Xiao.
jina-embeddings-v3: Multilingual embeddings with task lora, 2024.
[20] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embed-
ding: Advancing text embedding and reranking through foundation models.arXiv preprint
arXiv:2506.05176, 2025.
[21] Qwen Team. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
[22] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February
2026. URLhttps://qwen.ai/blog?id=qwen3.5.
12

[23] Wenxuan Zhang, Hou Pong Chan, Yiran Zhao, Mahani Aljunied, Jianyu Wang, Chaoqun
Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu, Yew Ken Chia, Xin Li, and Lidong Bing. Seallms
3: Open foundation and chat multilingual large language models for southeast asian lan-
guages, 2024.
[24] OpenAI. GPT-4o mini.https://openai.com/index/gpt-4o-mini-advancing-cost-eff
icient-intelligence, 2024.
[25] HaraldSteck, ChaitanyaEkanadham, andNathanKallus. Iscosine-similarityofembeddings
really about similarity? InCompanion Proceedings of the ACM on Web Conference 2024,
pages 887–890, New York, NY, USA, 2024. Association for Computing Machinery.
A Computation of RAGAS Metrics
We refer the reader to Es et al. [3] and Roychowdhury et al. [4] for details on the metrics, but
for completeness, we summarize the prompts and steps involved in our adapted implementation.
The notation is as follows: given questionqand contextc(q)retrieved from the corpus, the LLM
generates answera(q). The ground truth answer isgt(q).
A.1 Faithfulness (FaiFul)
Faithfulness is computed in two steps. First, the LLM judge decomposesa(q)into atomic
statementsS(q)using the prompt:“Given a question and answer, create one or more statements
from each sentence in the given answer.”Second, for each statements∈S(q), the LLM judge
determines a binary verdictv(s, c(q))indicating whether the statement is supported by the
context. Faithfulness is the ratio of supported verdicts to total statements (Equation 1).
A.2 Answer Relevance (AnsRel)
The LLM judge generatesNquestions froma(q)using the prompt:“Generate a question for the
given answer.”The cosine similarity between the embedding of the original questionqand each
generated question˜q iis computed, and the average is reported as answer relevance (Equation 2).
A.3 Context Relevance (ConRel)
In our implementation, context relevance is computed directly as the cosine similarity between
the BGE-M3 embedding of the questionqand the BGE-M3 embedding of the concatenated
retrieved contextc(q). Unlike the original LLM-based extraction variant of RAGAS, this imple-
mentation does not require the judge model to extract relevant sentences from the context.
A.4 Factual Correctness (FacCor) and Answer Correctness (AnsCor)
TheLLMjudgeclassifiesstatementsfroma(q)andgt(q)intoTruePositives(TP),FalsePositives
(FP), and False Negatives (FN). Factual correctness is the F1 score (Equation 5). Answer
correctness is the weighted sum of factual correctness and answer similarity (Equation 6).
A.5 Answer Similarity (AnsSim)
Answer similarity is the cosine similarity between BGE M3-embedding ofa(q)andgt(q)(Equa-
tion 4). This metric does not involve the LLM judge.
Table 5. Single-sample metric trace in English.
13

Component Description
Summary metrics For this example, the final scores areCtxRel=0.726295,AnsSim=0.800418,Faith=1.000,
AnsRel=0.797417,FacCor=0.500, andAnsCor=0.650209.
Question and reference
answerQ:Which code cancels unwanted mobile supplementary services?GT:Dial *1200# to unsubscribe
from unwanted add-on services.
Generated answer The code to disconnect VAS without balance check, subscriber ID check, top-up, or customer-service
contact is*1200#.
Dial *1200# and press send to drop add-ons without balance or ID steps, as stated in the notice.
Faithfulness Generated statements (4):
1. The code used to unsubscribe from mobile supplementary services (VAS) iscode *1200#.
2. Code *1200# can be used without requiring balance top-up actions or identity verification.
3. To unsubscribe from supplementary services, the user dials *1200# and presses send.
4. Unsubscribing from supplementary services does not require checking balance or contacting
customer service.
Verification verdicts:Statement 1: Yes; Statement 2: Yes; Statement 3: Yes; Statement 4: Yes.
Computation:supported statements= 4, total statements= 4, thereforeFaiFul= 4/4 = 1.000.
Answer relevance Three reverse questions were generated from the answer. Their cosine similarities to the original
question were0.845994,0.735499, and0.810756, giving a mean answer-relevance score of0.797417.
Context relevance The cosine similarity between the question embedding and the retrieved-context embedding was
0.726295.
Factual correctness True positives (TP):
1. Code *1200# can be used to unsubscribe from unwanted mobile supplementary services.
False positives (FP):
1. The code to unsubscribe from mobile supplementary services (VAS) without needing a balance
check, subscriber identity verification, top-up, or customer-service contact iscode *1200#.
2. By dialing *1200# and pressing send, users can unsubscribe from supplementary services without
balance top-up or identity-verification actions, as announced in the public notice.
False negatives (FN):None.
Computation:TP= 1, FP= 2, FN= 0; precision= 1/(1 + 2) = 1/3, recall= 1/(1 + 0) = 1, and
F1= 2×(1/3)×1/(1/3 + 1) = 0.500.
Answer similarity /
correctnessAnswer similarity:cosine similarity between the generated-answer embedding and the
reference-answer embedding was0.800418.
Answer correctness formula:AnsCor= 0.5×FacCor+ 0.5×AnsSim.
Values used:FacCor= 0.500000andAnsSim= 0.800418.
Computation:AnsCor= 0.5×0.500000 + 0.5×0.800418 = 0.650209.
14
# Towards Retrieval Augmented Generation in High-Energy and Astroparticle Physics

**Authors**: Jacky Kumar, Sajan Kumar

**Published**: 2026-10-01 01:12:38

**PDF URL**: [https://arxiv.org/pdf/2610.00891v1](https://arxiv.org/pdf/2610.00891v1)

## Abstract
The high-energy and astroparticle physics literature has grown into an enormous body of knowledge. Gaining a comprehensive understanding of this literature is an important step in research, but is a challenging task. Before making meaningful advances, it is important to establish what has already been done and identify open questions. However, this process is becoming increasingly difficult given the sheer volume of publications. Conducting a focused literature survey on a narrow topic is particularly challenging. We develop an open-source Retrieval Augmented Generation (RAG) pipeline to produce concise, targeted summary reports with citations, grounded in the database of arXiv papers, enabling researchers, editors, and referees to rapidly orient themselves within any corner of the field. Specifically, we embed approximately 230K hep-ph and astro-ph.HE papers to enable a hybrid retrieval that captures both keyword matches and semantic similarity. The fused candidate set is then refined using a cross-encoder reranker. Following retrieval, the papers are passed to a Large Language Model (LLM), for which we use Google DeepMind's open-source Gemma-4-E4B model. The LLM first acts as a judge to evaluate the relevance and then synthesizes a coherent and grounded report with references. This modern AI-driven approach allows us to go beyond the capabilities of classical keyword search on arXiv or Inspire-HEP.

## Full Text


<!-- PDF content starts -->

UdeM-GPP-TH-26-314
Towards Retrieval Augmented Generation in
High-Energy and Astroparticle Physics
Jacky Kumar ,aSajan Kumarb
aPhysique des Particules, Universite de Montreal, Montreal, QC, Canada
bDepartment of Physics, University of Maryland, College Park, Maryland, USA
Abstract:The high-energy and Astroparticle physics literature has grown into an enor-
mous body of knowledge. Gaining a comprehensive understanding of this literature is an
important step in research, but is a challenging task. Before making meaningful advances,
it is important to establish what has already been done and identify open questions. How-
ever, this process is becoming increasingly difficult given the sheer volume of publications.
Conducting a focused literature survey on a narrow topic is particularly challenging. We
develop an open-source Retrieval Augmented Generation (RAG) pipeline to produce con-
cise, targeted summary reports with citations, grounded in the database of arXiv papers,
enabling researchers, editors, and referees to rapidly orient themselves within any corner
of the field. Specifically, we embed approximately 230K hep-ph and astro-ph.HE papers to
enable a hybrid retrieval that captures both keyword matches and semantic similarity. The
fused candidate set is then refined using a cross-encoder reranker. Following retrieval, the
papers are passed to a Large Language Model (LLM), for which we use Google DeepMind’s
open-sourceGemma-4-E4Bmodel. The LLM first acts as a judge to evaluate the rele-
vance and then synthesizes a coherent and grounded report with references. This modern
AI-driven approach allows us to go beyond the capabilities of classical keyword search on
arXiv or Inspire-HEP.
Keywords:AI, LLM, Embeddings, RAG
1Author list is in alphabetical order. Both authors contributed equally to this work.
arXiv:2610.00891v1  [astro-ph.HE]  1 Oct 2026

Contents
1 Introduction 2
2 Dense Embeddings of HEP-Astro Literature 3
2.1 HEP-Astro literature 3
2.2Specter2,Qwen3, andNomicembeddings 4
3 Hybrid Search Retrieval: Query→Top-kpapers 8
3.1 Dense retrieval 8
3.2 Sparse retrieval 8
3.3 Reciprocal rank fusion 9
3.4 Cross-encoder reranker 9
4 Benchmark and Evaluation 9
4.1 Benchmarks for hep-ph and astro-ph.HE 10
4.2 Evaluation 11
5 Synthesizing HEP Review Reports with RAG 13
5.1 LLM-as-a-Judge for relevance 13
5.2 Synthesizing reports with LLM 14
5.3 Example Reports 15
6 Conclusion and Future Directions 18
– 1 –

1 Introduction
Easy access to scientific knowledge is the critical foundation for its dissemination. But for
fields like high-energy physics/astroparticle physics (HEP & Astro) this is becoming very
challenging due to the rapid growth of the literature. Throughout this work, we use HEP-
Astro to refer collectively to high-energy physics and astroparticle physics. In 2025, the
arXiv database hosted more than 2.6M research papers across various disciplines. In order
to find a paper on a narrow topic, researchers mainly rely on a classical keyword search
from databases such as arXiv, Inspire-HEP, or Google Scholar to locate articles relevant to
a narrow topic of interest. However, these approaches are limited as they do not capture
the semantic meaning.
LLMs such as Claude, GPT, Gemini, and Llama have significantly changed how we
access, interpret, and generate knowledge. Despite being very broad, their parametric
knowledge is fixed at training time, so they cannot know about research articles published
afterward. When asked about material outside of this knowledge, LLMs can respond with
incomplete information or produce factually incorrect responses. Continuously updating
LLMsonnewlypublishedpapersisimpractical, giventhattheyappeareveryday. Therefore,
some commercial LLMs partially resolve this by using web search tools at inference time,
but general web search is not tailored to specific type of literature. In addition, commercial
models require subscription fees or API costs, which can be a barrier to building domain-
specific applications. Fine-tuning LLMs on domain corpora [1, 2] addresses part of this
problem, but it is computationally expensive and is hard to keep up to date.
RAG offers a powerful alternative which couples the generative capabilities of LLMs
with a dynamic retrieval mechanism that fetches semantically relevant documents from an
external source at inference time [3, 4] (see also Ref. [5] for review). The retrieved passages
aretheninjectedintotheLLMcontextwindow, groundingit’sgeneratedoutputinverifiable
evidence which could be domain specific.
Indeed, the importance of RAG has been recognized by the particle and astro physics
community, where it has been utilized for the knowledge retrieval from domain-specific
databases at several fronts. For instance, the large scale experiments have used RAG for
navigating information from internal documentation, technical reports, and answer context
aware queries, see for example DUNE-GPT [6] for DUNE experiment, MITRA [7] for CMS,
and chatlas [8] dedicated for ATLAS experiments at the LHC.Similar approaches have
also been explored in astronomy, such as Pathfinder [9], a semantic framework designed to
support literature review and knowledge discovery in the astronomical literature.
In addition to these RAG-based applications, significant effort has also been devoted
on foundational aspects upon which effective retrieval depends, in specific domains. This
includes new text embedding models such as PhysBERT [10] and Astro-HEP-BERT [11]
pretrained on large corpora of physics literature and designed to capture the semantic
nuances of HEP-Astro specific terminology. Evaluation pipelines have also been developed
to assess the quality of retrieval in scientific question answer settings [12]. In this regard,
Ref. [13] introduces an agentic hybrid RAG framework for muon collider research alongside
the dedicated benchmark for retrieval-augmented question answering .
– 2 –

Since rapidly evolving HEP-Astro literature is posing significant challenges to both
experiencedandnewresearcherstostayuptodate. AnadvancedAI-drivenmethodthatcan
efficiently retrieve and synthesize knowledge from arXiv HEP-Astro database is therefore
a valuable and timely step. To best of our knowledge, until now, not much attention has
been given to develop a RAG method for HEP-Astro subdomains hep-ph and astro-ph.HE.
In this work, we develop a RAG pipeline dedicated to these fields. Given a user’s
scientific query, using our pipeline, relevant papers are semantically retrieved from arXiv
papers embedded usingSpecter2model [14]. These papers are then provided as context to
an open-source LLM, such asGemma4[15] orLlama[16], to synthesize a concise summary
report including citations. There can be multiple applications of this pipeline. For example,
it can help researchers get insights from the literature, perform gap analysis or perhaps even
identify novel directions by knowing what has been already done. It can also facilitate the
peerreviewprocess. Westressthatourgoalisnottoreplaceexpertjudgment, buttoreduce
theeffortforliteraturesearchandprovideevidencebasedassistancetotheresearchersunder
a rapidly growing publications under the influence of AI.
This article is organized as follows, Sec. 2 is devoted to methodology, where we detail
the key ingredients of our RAG pipeline such as data preparation, creating embeddings
usingSpecter2, storing them in a vector database. Sec. 3 details the retrieval process. In
Sec. 4, we give details of benchmark and evaluation of RAG pipeline. In Sec. 5, we discuss
the report synthesis with an LLM. Finally, we conclude and briefly discuss possible future
improvements in Sec. 6.
2 Dense Embeddings of HEP-Astro Literature
In this section, we discuss the different components of our RAG pipeline. For a quick
overview of the embedding and retrieval procedures, we refer to Fig. 2 and Fig. 5, respec-
tively. In the next subsection, we first discuss our database of arXiv HEP-Astro articles,
which serves as the literature input for the embedding procedure.
2.1 HEP-Astro literature
Our ultimate goal is to be able to synthesize short review report on a given HEP-Astro topic
for the purpose of gap and novelty analysis. Such reports can also be useful for the peer-
review process and to find new research directions. Therefore, it is very important that the
report be grounded in published work and covers important previous contributions, with
appropriate references.
For this purpose, we believe that the full text of papers is not necessary, but, only the
title, abstract, andassociatedmetadataaresufficient. Indeed, thesearegoodrepresentation
of the paper and usually contain the overall motivation and new findings.
We construct our input from a Kaggle dataset, which contains all the required features
mentioned above [17]. This dataset is distributed injsonformat and updated weekly. From
this dataset, we filter papers belonging to the subdomains hep-ph and astro-ph.HE, which
align with our domain expertise and include the preprints submitted to arXiv until the
end of 2025. After filtering, our HEP and Astro knowledge base consists of total 253,606
– 3 –

Figure 1:Top-10 secondary (primary) categories when hep-ph or astro-ph.HE are primary (sec-
ondary) category papers in our database.
articles. Of these, we there are 141,160 hep-ph and 45,045 astro-ph.HE primary category
papers and 52,762 hep-ph and 22,256 astro-ph.HE secondary category papers. Note that
these numbers are not supposed to add up to 253,606, because a paper can be hep-ph
and astro-ph.HE secondary at the same time. Indeed, we checked that there are 186,305
only hep-ph related, 59,684 only astro-ph.HE related papers, and 7617 papers contain both
hep-ph and astro-ph.HE – which add up to 253,606. In Fig. 1, we show a barcharts for
the top-10 secondary categories in hep-ph primary category papers, as well as the top-10
primary categories when hep-ph is secondary. A similar barcharts for astro-ph.HE are also
shown.
2.2Specter2,Qwen3, andNomicembeddings
To create embeddings for the HEP-Astro dataset, we use theSpecter2embedding model,
which is an open-source encoder model [14]. Its pretrained weights are publicly available
at Hugging Face1.Specter2is trained on title and abstract of scientific papers in various
fields including physics. Therefore, the resulting embeddings are more suitable for scientific
document retrieval as compared to general purpose sentence encoder models.
Specter2model’s pretraining objective is the citation relationships; in other words,
papers that cite each other or are related tend to be closer in the embedding space, which
is 768 dimensional. Following pretraining on 6M citation triplets, it is also fine-tuned using
1https://huggingface.co/allenai/specter2
– 4 –

task-specific adapters trained on the SciRepEval benchmark [18]. We have added so-called
theproximityadapter on the top ofSpecter2base model to embed abstract and titles and
adhoc queryadapter to embed the query.
We will also compare the performance ofSpecter2with two general purpose embedders
Qwen3[19] andNomic[20].Qwen3– a state-of-the-art embedding, is representative of
large, generalpurposeLLMderivedembeddersandisbasedonQwen3foundationalmodels.
Qwen3spans model size of 0.6B, 4B and 8B parameters, but we useQwen3-0.6B. On the
other hand,Nomicis English text embedding supporting long context length having only
137M parameters.
Each paper is represented by a single input sequence constructed by concatenating its
title and abstract. The resulting text is embedded and stored in the ChromaDB vector
database. Metadata fields are stored separately alongside each embedding and are used for
filtering and retrieval; they are not included in the embedding input. This procedure is
depicted in Fig. 2.
HEP-Astro
LiteratureTitle and Abstract Embeddings Chroma DB
Figure 2:Procedure for creating embeddings of dataset containing title and abstract of HEP and
Astro papers and their storage in the chroma vector database.
To measure the local classification accuracy of papers in the embedding space, we now
compute thek-Nearest-Neighborhood (k-NN) purity for the three types of embedding. It
is defined by
Purity@k=1
NNX
q=11
kkX
i=11h
c
p(q)
i
=c(q)i
(2.1)
Here,Nis the total number of query papers,kis the neighborhood size,p(q)
idenotes the
i-th nearest neighbor of query paperqin the embedding space, andc(·)returns a paper’s
primary arXiv category. The indicator function1[·]equals 1 if the neighbor’s category
matches the query’s category, and 0 otherwise.
Purity@k, ranging from 0 to 1, thus measures the average of fraction of a paper’sk
nearest papers in the embedding space that share its primary category. We use cosine
similarity as a metric to define the nearest.
ConsideringN=2500, with papers drawn from hep-ph, hep-th, hep-ex, hep-lat, nucl-
th, astro-ph, and gr-qc categories, we find thatQwen3,Specter2andNomicachieve nearly
identicalk-NNpurityonourdataset. Forthethreeembeddingtypes: Purity@10∼0.66-0.67
and Purity@5∼0.68-0.69, see Tab. 1 for more details.
– 5 –

Model Parameters Dim. Purity@5 Purity@10
Specter2110 Million 768 0.68 0.66
Nomic137 Million 768 0.69 0.66
Qwen3600 Million 1024 0.69 0.67
Table 1: Purity@k(ork-NN purity) – the fraction of the nearest-neighbors that share the
query paper’s primary category. The results are shown for three types of embedding models
for sample sizes ofN=2500. Here, primary category means the first arXiv category.
Given that these quantitative gaps between models are small enough, it seems the
purity alone does not justify preferring a larger general purpose embedding models in this
case. Therefore,Specter2seems suitable on practical and domain grounds.Specter2being
a lightweight model and that can be deployed efficiently on commodity hardware, makes it
practical for local inference and large-scale indexing.
Limiting only to primary categories hep-ph and astro-ph.HE, and averaging over the
corpus (N=141,160 hep-ph,N=45,045 astro-ph primary category papers), we find
Purity@10: hep-ph=0.99,astro-ph.HE=0.97.(2.2)
Moreover, wefindthat0.94ofpapersachievePurity@10=1. ThisindicatesSpecter2reliably
locates the correct neighborhood.
Figure 3: UMAP projection ofSpecter2embeddings divided by primary and secondary
arXiv categories. The hep-ph and astro-ph.HE as primary categories are occupying largely
separate regions, shown in blue and orange, respectively. The hep-ph and astro-ph.HE as
secondary categories are shown in green and magenta regions, respectively. The axes have
no physical meaning.
In Fig. 3, we show the Uniform Manifold Approximation and Projection (UMAP) of
the full dataset containing papers with hep-ph and astro-ph.HE as primary or secondary
– 6 –

categories. Two distinct clusters emerge, corresponding to the hep-ph (blue) and astro-
ph.HE (red) primary populations, connected by a narrow bridge of overlapping points.
Papers in which hep-ph or astro-ph.HE appear as a secondary category (green and yellow,
respectively)aredistributedacrossbothprimaryclusters, withavisibleconcentrationalong
the bridge, which is consistent with these cross-listed papers sharing thematic content with
both fields.
To evaluate how well different embedding models separate subfields, we constructed a
category-balanced evaluation set drawn from our arXiv HEP corpus. We retained ten high-
energy physics and astrophysics categories: hep-ph, hep-th, hep-ex, hep-lat, nucl-th, gr-qc,
astro-ph.HE, astro-ph.SR, astro-ph.CO, and astro-ph.GA. Papers were sampled within each
primary category as evenly as possible across publication years to minimize temporal bias.
For the comparative embedding analysis, we used 2500 papers per primary category. Each
paper was represented by its title + abstract text and embedded usingSpecter2,Nomic,
andQwen3. The resulting representations were then compared using UMAP visualizations.
Figure 4: UMAP of papers from ten primary arXiv categories embedded withSpecter2,
Nomic, andQwen3. Points are colored by primary category, all models use the same
stratified sample of size 2500. The axes have no physical meaning.
This is shown in Fig. 4, which indicates clear differences in the extent to which the
embedding models organize papers by primary category.Specter2exhibits relatively well-
defined category specific regions, although several neighboring categories show substantial
overlap. This overlap is expected, as papers in the other categories are always cross-listed
with either hep-ph or astro-ph.HE as a secondary category in our dataset, which is also
illustrated by the overlap between primary and secondary categories in Fig. 1.
Nomicproduces a similarly structured embedding space, with pronounced local cluster-
ing but noticeable mixing among several categories.Qwen3also reveals a distinct category
level structure, with some categories forming compact regions and others exhibiting greater
overlap. Overall, the UMAP projections indicate that all three models capture meaning-
ful subfield structure, while the degree and location of inter-category overlap vary across
models. From now on, we will always useSpecter2embedding for retrieval and synthesis.
– 7 –

3 Hybrid Search Retrieval: Query→Top-kpapers
In this section, we give details of our retrieval method, which is also shown in a flowchart in
Fig. 5. The retrieval of semantically similar paper is refined through multistep procedure,
as described in the following subsections.
Embed query
(Specter2)
Hybrid retrieve
Dense + BM25 + RRF
Cross-encoder
rerank
Top context
papers
1) RetrievalUser Query
Chroma
vector DB
BM25
indexLLM-as-judge
Synthesis
prompt
LLM synthesis
+ citations
2) Synthesis
Figure 5:Online retrieval pipeline used to obtain the candidate pool of papers for a given query.
This is a multi-stage procedure consisting of dual retrieval: dense (viaSpecter2) and sparse (via
BM25) – fused viaRRF, followed by cross-encoder reranking. For synthesis, the LLM first judges
the relevance of the retrieved papers and selects the final list, and then generates a concise report.
3.1 Dense retrieval
The first step is to find top-ksimilar candidate papers for a given user query. For dense
systematic retrieval, the query needs to be also embedded, for which we use two comple-
mentary models. First, we useSpecter2model along withadhoc queryadapter [14] to
generate an embedding of query. Subsequently, we compare this query embedding against
precomputed embeddings, usingSpecter2along withproximityadapter, of paper stored
in offline vector database. After initial retrieval, candidate papers are ranked using cosine
similarity defined by
cos(q, d) =q·d
∥q∥∥d∥.(3.1)
Here,dandqdenote the document and query vectors, and∥v∥stands for the Euclidean
norm of a vectorv.
3.2 Sparse retrieval
Forsparselexicalretrieval, weuseBest Matching 25(BM25)model[21]. Itranksdocuments
on the basis of three factors: how often query terms appear in each paper’s abstract and
– 8 –

title, how rare those terms are across the whole HEP-Astro corpus (rare terms count more
than common ones), and the document’s length. In BM25, the latter is required because
otherwiselongdocumentscanbeartificiallyfavouredduetohigherprobabilityofoccurrence
of term. BM25 is an effective method for keyword matching.
By combining dense and sparse paradigms, our retrieval leverages both semantic simi-
larity and lexical relevance to improve the quality of candidate papers.
3.3 Reciprocal rank fusion
In order to combine the outputs from dense and sparse retrievals, we useReciprocal Rank
Fusion(RRF) [22], since scores produced by these two models are not directly comparable.
The RRF aggregates ranks instead of raw scores differing in scales. For each candidate
paperd, the fused score is defined by
RRF(d) =1
k+ rank dense(d)+1
k+ rank sparse (d).(3.2)
Here,rank r(d)denotes the rank of paperdin the list returned by retrieverr. We use
k= 60which is a standard RRF configuration [22]. The candidate papers are then sorted
according to their fused scores, producing a unified ranking.
3.4 Cross-encoder reranker
The fused candidate list is subsequently refined usingms-marco-MiniLM-L-6-v2, a cross-
encoder reranker trained on the MS MARCO passage-ranking task [23, 24].
Given a queryqand a candidate paperd, the model jointly encodes the query and the
document to predict a relevance score
r(q, d) =f θ([CLS]q[SEP]d[SEP]),(3.3)
wheref θrepresents a transformer-based model having parametersθ.[CLS]is classification
token prepended to input sequence, and[SEP]is separation token, typically needed in a
transformer model. Unlike bi-encoders, which independently encode queries and documents
before comparing their embeddings, cross-encoders jointly attend to both inputs, enabling
fine-grained interactions between query and document tokens. As a result, they provide
more accurate relevance estimates and are well suited to rerank a relatively small set of
retrieved candidates. The complete retrieval pipeline is illustrated in Fig. 5.
4 Benchmark and Evaluation
To evaluate retrieval quality, we constructed two benchmark datasets, one for hep-ph and
one for astro-ph.HE. The two differ slightly in nature, which adds variety to the evaluation,
as described below. These will be made publicly available on Hugging Face.
– 9 –

4.1 Benchmarks for hep-ph and astro-ph.HE
For each hep-ph paper’s abstract in the evaluation dataset, we generated three types of
synthetic questions, each probing a different aspect of the abstract. These questions were
generated using theLlama4-Maverick-17B-Instructmodel [25] on Amazon Bedrock. For
hep-ph, three question types are defined as follows:
•Main findings: a question about the results reported in the abstract.
•Claim verification: a question that probes a specific claim made in the abstract.
•Vague exploratory: a broader and more abstract question related to the paper’s
general topic.
The astro-ph.HE benchmark is slightly different. Three questions type for astro-ph.HE is
as follows:
•Basic:an object, instrument, or source named in the opening of the abstract.
•Intermediate:one measured feature, count, or implication from the body of the
abstract.
•Technical:the quantitative conclusion, or the model requirement that explains it.
Examples of each question type, along with the corresponding abstract, are shown for
hep-ph and astro-ph.HE in Fig. 6 and 7, respectively.
Figure 7:An example from the astro-ph.HE evaluation benchmark (source paper [27]), consisting
of an abstract and three question types (basic, intermediate, technical) used to evaluate different
retrieval and reasoning capabilities.
The benchmark datasets consist of 2040 samples for hep-ph and 2376 samples for astro-
ph.HE. They were created by prompting theLlama4-Maverick-17B-Instructmodel with the
– 10 –

Figure 6:An example from the hep-ph evaluation benchmark (source paper [26]), consisting of
an abstract and three question types (main findings, claim verification, vague exploratory) used to
evaluate different retrieval and reasoning capabilities.
title and abstract of each paper, together with three example questions of each type for the
respective category (hep-ph or astro-ph.HE).
We then evaluate whether the pipeline can correctly retrieve the source paper given a
query from this benchmark.
4.2 Evaluation
In this section, we discuss the evaluation of our pipeline based on the two benchmarks
detailed in the previous subsection. For this purpose, we employ binary Recall@kand Mean
ReciprocalRank(MRR)metrics. TheRecall@kmeasures, whethertherelevantsinglepaper
appears anywhere within top-kresults, and the MRR measures ranking quality which is
not captured by recall.
For queryqfrom a dataset of totalQqueries in the benchmark, the Recall@kis defined
as
Recall@k=1
|Q|X
q∈Q|Rq∩Tk(q)|
|Rq|.(4.1)
– 11 –

Here,R qis the set of true papers for queryq, but, since in our benchmark we associate
only a single paper for each query, this implies|R q|= 1.T k(q)denotes the set of top-k
retrieved papers and|Q|= 2404for hep-ph and|Q|= 2376for astro-ph.HE.
Suppose that for queryq, the rank of correctly retrieved paper comes out to be rank q,
the MRR is then given by
MRR=1
|Q|X
q∈Q1
rank q.(4.2)
In Tab. 2 and 3, we present the evaluation results on recall and MRR for the hep-ph and
astro-ph.HE papers fork= 3,5and10. The results are divided by the type of queries and
the method (dense only, hybrid, hybrid+rerank as discussed in Sec. 3) used in the retrieval
process.
Method CategoryRecall@kMRR@k
@3 @5 @10 @3 @5 @10
DenseClaim Verification 0.305 0.343 0.391 0.263 0.272 0.279
Main Finding 0.291 0.330 0.381 0.248 0.257 0.264
Vague Exploratory 0.055 0.071 0.100 0.042 0.045 0.049
HybridClaim Verification 0.629 0.757 0.893 0.498 0.527 0.546
Main Finding 0.594 0.698 0.836 0.474 0.498 0.516
Vague Exploratory 0.149 0.204 0.300 0.113 0.125 0.138
Hybrid + RerankClaim Verification 0.917 0.937 0.958 0.856 0.861 0.864
Main Finding 0.865 0.896 0.923 0.800 0.807 0.811
Vague Exploratory 0.332 0.374 0.422 0.272 0.282 0.288
Note:n= 2404for each category.
Table 2: Retrieval performance by method and question category for hep-ph papers.
Across both the hep-ph (Tab. 2) and astro-ph.HE (Tab. 3) benchmarks, retrieval per-
formance improves consistently and substantially as the different components are added
to the pipeline: Dense retrieval alone performs weakest, Hybrid (sparse + dense fusion)
improves markedly over Dense, and Hybrid + Rerank achieves the best results across every
metric and category.
For hep-ph, Recall@10 on Claim Verification rises from0.391(Dense) to0.893(Hy-
brid) to0.958(Hybrid + Rerank), and a similar trend holds for MRR@k, indicating that
reranking not only retrieves the correct paper more often but also ranks it closer to the top.
Vague Exploratory queries are consistently the hardest category across both methods
and both domains, with Recall@10of only0.100(Dense) and0.300(Hybrid) for hep-ph,
and even Hybrid + Rerank reaching just0.422– far below the Claim Verification and Main
Finding scores, reflecting the inherent difficulty of matching broad, under specified queries
to a single correct source paper.
– 12 –

Method DifficultyRecall@kMRR@k
@3 @5 @10 @3 @5 @10
DenseBasic 0.199 0.234 0.294 0.159 0.167 0.176
Intermediate 0.328 0.379 0.449 0.268 0.280 0.289
Technical 0.357 0.398 0.463 0.300 0.310 0.318
HybridBasic 0.458 0.562 0.712 0.357 0.381 0.401
Intermediate 0.658 0.757 0.872 0.534 0.556 0.572
Technical 0.656 0.761 0.885 0.541 0.565 0.582
Hybrid + RerankBasic 0.787 0.832 0.869 0.724 0.734 0.739
Intermediate 0.891 0.921 0.947 0.830 0.837 0.840
Technical 0.885 0.913 0.941 0.821 0.828 0.831
Note:n= 2376for each difficulty level.
Table 3: Retrieval performance by method and difficulty level for astro-ph.HE papers.
For astro-ph.HE, a similar pattern holds across difficulty levels (Basic, Intermediate,
Technical), with Hybrid + Rerank achieving Recall@10above0.86for all levels, though
interestingly Basic queries perform slightly worse than Intermediate and Technical ones
under Dense and Hybrid retrieval, suggesting that overly simple queries may lack the dis-
tinctive keywords needed for effective lexical/dense matching – an effect that reranking
largely corrects.
5 Synthesizing HEP Review Reports with RAG
We finally demonstrate the application of our retrieval pipeline for the downstream task
of generating short review reports with appropriate references for a given query. However,
although our benchmarks, each containing a single target paper, show that our pipeline
achieves high recall and MRR, report generation requires feeding multiple relevant ref-
erences to an LLM. Moreover, the top-kretrieved papers often include articles that are
semantically similar to the query but still topically off-target.
To address this, we first retrieve a large candidate pool of papers (K= 100) for each
query, and then use an LLM – as a judge, to select a final refined list (F) of references based
on relevance. For the purpose of relevance check as well as generation we useGemma-4-
E4B-ITmodel with 4.5B effective parameters from Google DeepMind. Throughout, we
will refer this asGemma4model.
Further details are given in the following subsection.
5.1 LLM-as-a-Judge for relevance
The relevance filtering step in our pipeline follows the “LLM-as-a-judge” method [28], in
which an LLM is used in place of a human to assess relevance between initially retrieved
candidates using hybrid search and cross-encoder reranking. In our setting, the LLM judge
is given the query together with a numbered list of candidate papers (title and abstract)
and is prompted to select the subset ofFmost relevant papers to the query.
– 13 –

As discussed before, this design choice is deliberate: rather than relying solely on em-
bedding similarity or lexical overlap rankings, we use the LLM’s judgment as an additional
relevance filter, exploiting its capacity for contextual and semantic reasoning about rele-
vance that rankers lack. For relevance filtering, the prompt given to LLM is:
Relevance Prompt:You are a research assistant. From the numbered list of papers
below, select theFmost relevant to the query. Return ONLY the numbers as a comma-
separated list, e.g.: 1,3,5,7.Query:q, candidate papers, Selected numbers:
Figure 8: Prompt given toGemma4model to select most relevant papers.
We note that when acting as judge, LLMs are known to exhibit certain biases, including
position, verbosity, and self-enhancement bias [28]. Self-enhancement bias, where an LLM
favours outputs generated by models from its own family, is not directly applicable to our
setting, since theGemma4evaluates retrieved original arXiv content rather than LLM
generated text.
Regarding positional bias, we empirically tested the sensitivity of relevance selections
to candidate ordering ofKretrieved papers by running the filtering prompt across different
orderings of the same candidate set: the original order, its reverse, and five randomly
reshuffled permutations (fixed random seed for reproducibility). We found selections to
be highly sensitive to ordering, with a mean pairwise Jaccard2varying a lot across the 7
2
= 21pairwise comparisons of permuted runs. We therefore adopt the union of papers
selected by LLM judge across these seven permuted runs of retrieved papers as our final
candidate set for synthesis, which improves robustness to positional bias at the cost of a
larger and query-dependent reference set size .
Concerning verbosity bias, we computed the token length of title and abstract using
Gemma4tokenizer. We find that it is moderately consistent across arXiv articles in our
dataset, with mean = 241, median = 223, and standard deviation = 111, suggesting most
inputs given toGemma4model for synthesis fall within a comparable range. Therefore, we
expect any verbosity bias effects from input length alone to be limited, though we cannot
rule out bias arising from other differences across abstracts and titles.
5.2 Synthesizing reports with LLM
StartingfromaqueryonagiventopicinHEP-Astroandendinginaconcisereport, wecom-
bine all the steps: retrieval, relevance check, and report synthesis using an LLM (Gemma4)
into Algorithm 1. See also Fig. 5. Using this algorithm, we are able to readily generate a
report on any topic within either hep-ph or astro-ph.HE. In the current implementation, the
query is a direct input to retrieval system and must therefore be well-defined. Furthermore,
the synthesis prompt provided to theGemma4model is:
2The Jaccard Index between two setsAandBis defined asJ(A, B) =|A∩B|
|A∪B|.
– 14 –

Synthesis Prompt:You are a research assistant. Write a concise synthesis report (4-
5 paragraphs) summarizing the key themes, findings, and connections across the papers
below, in the context of the query only. Properly cite papers at relevant places in the
report.Query:q, synthesis context, Synthesis:
Figure 9: Prompt given toGemma4model for report synthesis.
It is used in step 3 of the Algorithm 1. For illustration purposes, we generate only
a short report of 4-5 paragraphs in length, however, more comprehensive reports can also
be generated by increasing the number of final papers (F) selected by the LLM judge and
initial papers(K)from the hybrid retrieval.
Algorithm 1Retrieval-Augmented Report Synthesis
Require:A user queryq, initial candidate sizeK, final selection sizeF, number of random
shufflesN
Ensure:A concise report synthesis with citations.
Step 1: Initial Retrieval
— Retrieve the top100candidate papers (K) for a query using a hybrid search that
combines dense embedding similarity and BM25 matching, fuse the two rankings, and
reranks the merged list with a cross-encoder model.
Step 2: LLM-Based Relevance Filtering with Position Shuffle
— Construct a set ofN+2orderings of theKcandidates: the original order after cross-
encoder, its reverse, andN=5independent random permutations. For each orderingo,
present theKcandidates to theGemma4model as a numbered list following ordering
o, each entry showing the paper’s title and abstract.
— Using a prompt, given in Fig. 8, ask theGemma4model to select theFpapers most
relevant to the query and to return their list numbers.
— Form the filteredpaper setas the unionS
oSoof the selected sets across allorderings.
Step 3: Synthesis Generation with LLM
— Use the synthesis prompt, given in Fig. 9, containing the query together with the
title and abstract of each paper in the union.
— Ask theGemma4model to write a short report that addresses the query with
citations.
— Return the synthesis report and the union paper setS
oSo.
5.3 Example Reports
We present two example reports synthesized by our algorithm, one each for hep-ph and
astro-ph.HE. For brevity, we include only one example report per field here; additional
examples will be made available on the GitHub page of the repository.
To generate a short report on hep-ph, we selected a topic aligned with the area of
expertise of one author, enabling expert evaluation of the resulting report. Starting from
an initial retrieval ofK=100candidate papers, the LLM-judge relevance check selected
– 15 –

F=28papers through the union of reshuffled runs. These papers were then passed as
context to theGemma4model, together with a synthesis prompt, to generate the final
report. The resulting report is shown in Fig. 10.
Query:How to calculate anomalous dimensions in SMEFT to compute renormalization
group running effects?
Synthesis Report:The calculation of anomalous dimensions in the Standard Model Ef-
fective Field Theory (SMEFT) is crucial for consistently computing renormalization group
(RG) running effects, which ensures that predictions for physical observables are inde-
pendent of the arbitrary renormalization scale (µ) [arXiv:2601.19974]. These anomalous
dimensions govern how the Wilson coefficients of higher-dimension operators evolve be-
tween different energy scales. The calculation methodologies employed are diverse, ranging
from traditional loop calculations to more efficient, amplitude-based approaches. Specifi-
cally, on-shell methods have proven to be an exceptionally efficient technique for deriving
one-loop anomalous dimensions of higher-dimension operators directly from tree-level am-
plitudes, bypassing explicit loop calculations [arXiv:2005.07129, arXiv:2005.10261].
The complexity of these calculations scales with the dimension of the operators and the
loop order. For dimension-six operators, one-loop anomalous dimension matrices have
been calculated, detailing the contributions from various Standard Model couplings, such
as the Higgs self-coupling (λ) and Yukawa couplings (y) [arXiv:1308.2627]. Furthermore,
the development of higher-order calculations is ongoing; for instance, two-loop anomalous
dimensions have been described using on-shell methods [arXiv:2005.12917], and the full
two-loop RG equations for specific sectors, like baryon-number-violating operators, have
been computed [arXiv:2510.08682].
The application of these calculated anomalous dimensions is central to phenomenology. RG
running allows for the consistent combination of collider observables measured at different
energy scales, such as those in global fits [arXiv:2212.05067]. This running is necessary
to account for operator mixing, which is a key feature of SMEFT, ensuring scale inde-
pendence in Next-to-Leading Order (NLO) calculations [arXiv:2601.19974]. Specific appli-
cations include studying the impact of RG running on Higgs production processes at the
LHC [arXiv:2406.06670, arXiv:2409.19578] and analyzing flavor-violating effects induced
by Yukawa running [arXiv:2005.12283].
BeyondthestandardSMEFT,theframeworkextendstotheoriesincorporatingnewphysics,
such as sterile neutrinos (νSMEFT) [arXiv:2411.07220] or axion-like particles (ALPs)
[arXiv:2105.01078]. In these extended scenarios, the anomalous dimensions must account
for the new particle interactions, including Yukawa couplings [arXiv:2407.16751]. More-
over, the framework is being extended to higher dimensions, with theoretical work address-
ing dimension-seven [arXiv:2310.11055] and dimension-eight operators [arXiv:2203.06771,
arXiv:2409.10305], necessitating the computation of corresponding RG evolution matrices
[arXiv:2408.15378].
Figure 10:Synthesis report on hep-ph generated by our RAG pipeline fromF=28papers selected
out ofK=100retrieved candidates.
– 16 –

Query:How does the VERITAS constrain WIMP dark-matter annihilation cross sections
using gamma-ray observations?
Synthesis Report:The search for Weakly Interacting Massive Particle (WIMP) dark
matter (DM) annihilation is a primary goal in astrophysics, leveraging the prediction that
DM self-annihilation in regions of high DM density can produce detectable Very High
Energy (VHE) gamma rays [arXiv:1110.4358, arXiv:1110.6615]. Dwarf spheroidal galaxies
(dSphs)areconsideredprimetargetsfortheseindirectsearchesbecausetheyareexpectedto
possess high DM content while exhibiting low astrophysical backgrounds [arXiv:1307.8367,
arXiv:1210.4961]. VERITAS, as an imaging atmospheric Cherenkov telescope array, is
specifically designed to detect these VHE gamma rays, which can have energies up to the
DM particle mass [arXiv:2407.16518].
VERITAS constrains WIMP annihilation cross sections by setting upper limits on the
observed gamma-ray flux from these dSphs. The absence of a statistically significant
gamma-ray signal in observations of specific dSphs, such as Segue 1 [arXiv:1110.4358],
Draco, Ursa Minor, and Willman 1 [arXiv:0902.4750], allows researchers to derive conser-
vative bounds on the DM annihilation cross-section [arXiv:0902.4750]. Furthermore, joint
statistical analyses combining data from multiple dSphs have been employed to enhance
sensitivityandtightentheseconstraintsontheannihilationcrosssection[arXiv:1703.04937,
arXiv:2407.16518].
The methodology involves comparing the expected gamma-ray flux, derived from theoreti-
cal models assuming WIMP annihilation, against the upper limits established by VERITAS
observations [arXiv:1210.4961]. These constraints are crucial for testing various particle
physics models, particularly those predicting WIMPs in the mass range of 50 GeV to over
10 TeV [arXiv:1307.8367, arXiv:0910.4563]. While VERITAS has focused on dSphs, re-
lated studies also examine other targets, such as the Galactic Center [arXiv:2309.12403], to
constrain DM properties, although dSphs remain a key focus for robust cross-section limits
[arXiv:2003.13482].
In summary, VERITAS constrains WIMP annihilation cross sections by performing null
searches for VHE gamma rays originating from dSphs. By utilizing the high DM density of
these galaxies and the sensitivity of the IACT array, researchers translate non-detections
into stringent upper limits on the annihilation cross section, thereby ruling out certain
parameter spaces for WIMP models [arXiv:1703.04937, arXiv:1210.4961].
Figure 11:Synthesis report on astro-ph.HE generated by our RAG pipeline fromF=25papers
selected out ofK=100retrieved candidates.
We have checked that the above hep-ph report given in Fig. 10, which is based on the
titles and abstracts of 17 research papers, is coherent, correct, and free of hallucinations.
Furthermore, we verified that each statement in the report is accurate and is supported
by the corresponding cited paper. The report covers seminal work on the topic, while also
capturing recent progress, such as the use of on-shell methods for computing anomalous
dimensions as well as phenomenological impact of renormalization group running effects.
We therefore conclude that the report and the papers it cites provide a very good starting
point for work on the computation of anomalous dimensions in the SMEFT and beyond.
We analyses several other such report and found a similar level of quality.
– 17 –

For astro-ph.HE, we selected a topic aligned with the area of expertise of another
author, again enabling expert evaluation of the resulting report. From an initial retrieval of
K=100candidate papers, the union of the selections made by the LLM judge over several
reshuffledrunsyieldedF=25papers. ThesepaperswerethenpassedtoGemma4ascontext,
together with a synthesis prompt, to generate the synthesis report shown in Fig. 11.
Similarly, the synthesis report generated for the astro-ph.HE query, as shown in Fig. 11,
provides a clear and coherent overview of how VERITAS searches for WIMP dark matter
through the indirect detection of very-high-energy (VHE) gamma rays. Interestingly, the
report connects the theoretical motivation for the annihilation of WIMP with the obser-
vational strategy. For example, dwarf spheroidal galaxies (dSphs) were selected as targets
because of their high DM mass content and low astrophysical background. We explicitly
checked that the cited references and the factual claims of no dark matter detection are
accurate. More such reports and their expert analyses will be provided on the GitHub
repository.
6 Conclusion and Future Directions
AI has already affected the way we conduct research in HEP and Astroparticle physics.
A researcher’s day-to-day activities now include code generation, finding references, and
bouncing ideas off LLMs such as Claude, ChatGPT, and Gemini. These general-purpose
LLMs, though extremely powerful, also have limitations such as knowledge cutoffs, sub-
scription requirements, and safety concerns. For these reasons, it becomes extremely im-
portant to develop a small, open-source AI system that can serve a specific domain such as
high-energy physics without suffering from the above problems.
RAG is a powerful AI framework that allows LLMs to generate text well-grounded
in an external knowledge source. In this work, we present an open-source and modular
RAG pipeline for finding and synthesizing HEP and Astro research papers, focusing on
the hep-ph and astro-ph.HE categories. The focus on specialized subdomains hep-ph and
astro-ph.HE addresses an identified gap in existing RAG literature.
We embedded about 230K arXiv articles using theSpecter2embedding model and
stored them in a Chroma vector database. We then retrieve semantically similar papers for
a given query using a hybrid-search RAG approach and rerank them using thems-marco-
MiniLM-L-6-v2cross-encoder model. We found that this RAG pipeline achieves very high
recall and MRR, evaluated using two synthetic benchmarks in hep-ph and astro-ph.HE.
Post retrieval, a large number of papers are further refined based on their relevance to
the query with the help of the Google DeepMind’sGemma4model, which is open-source,
used as an LLM-as-a-judge ensuring bias mitigation. The final set of papers is then used as
context for theGemma4model to generate a short (4–5 paragraphs) review with citations.
Based on the author’s domain knowledge of hep-ph and astro-ph.HE, respectively, it is
found that the resulting reports to be coherent, technically sound, and well grounded in
the final list of papers.
As an application, such a method can be used by researchers, reviewers, or editors to
quickly get up to speed on the current literature, conduct a gap and novelty analysis, or
– 18 –

find new directions to work on. We see many possibilities to improve our method, such
as providing the full text of papers as context to LLM for synthesis, query decomposition
prior to retrieval and incorporating citation information of source papers into retrieval and
synthesis. Finally, developing a rigorous and scalable method for evaluating the domain-
specific reports synthesized by our pipeline, without requiring human involvement, would
be extremely valuable.
Acknowledgments
This work was financially supported by the Natural Sciences and Engineering Research
Council of Canada (J.K.). This research was enabled in part by support provided the
Digital Research Alliance of Canada (www.alliancecan.ca). Computations were performed
on the Trillium supercomputer at the SciNet [29] HPC Consortium. SciNet is funded by
Innovation, Science and Economic Development Canada; the Digital Research Alliance of
Canada; the Ontario Research Fund: Research Excellence; and the University of Toronto.
References
[1] P. Richmond, C. Papageorgakis, V. Niarchos, B. Chowdhury and P. Agarwal,FeynTune:
large language models for high-energy theory,Mach. Learn. Sci. Tech.7(2026) 025012
[2508.03716].
[2] T.D. Nguyen, Y.-S. Ting, I. Ciuca, C. O’Neill, Z.-C. Sun, M. Jabłońska et al.,AstroLLaMA:
Towards specialized foundation models in astronomy, inProceedings of the Second Workshop
on Information Extraction from Scientific Publications, T. Ghosal, F. Grezes, T. Allen,
K. Lockhart, A. Accomazzi and S. Blanco-Cuaresma, eds., (Bali, Indonesia), pp. 49–55,
Association for Computational Linguistics, Nov., 2023, DOI.
[3] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal et al.,Retrieval-augmented
generation for knowledge-intensive nlp tasks, 2021.
[4] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi et al.,Retrieval-augmented generation for
large language models: A survey, 2024.
[5] A.J. Oche, A.G. Folashade, T. Ghosal and A. Biswas,A systematic review of key
retrieval-augmented generation (rag) systems: Progress, gaps, and future directions, 2025.
[6] A. Rafique, A. Singh and R. Srinivas,Large language model integration for knowledge
retrieval and interaction for the dune experiment, 2026.
[7] A. Mallampalli and S. Dasu,Mitra: An ai assistant for knowledge retrieval in physics
collaborations, 2026.
[8]ATLAScollaboration,chATLAS: An AI Assistant for the ATLAS Collaboration, .
[9] K.G. Iyer, M. Yunus, C. O’Neill, C. Ye, A. Hyk, K. McCormick et al.,pathfinder: A
semantic framework for literature review and knowledge discovery in astronomy,The
Astrophysical Journal Supplement Series275(2024) 38.
[10] T. Hellert, J. Montenegro and A. Pollastro,Physbert: A text embedding model for physics
scientific literature, 2024.
– 19 –

[11] A. Simons,Astro-hep-bert: A bidirectional language model for studying the meanings of
concepts in astrophysics and high energy physics, 2024.
[12] X. Xu, B. Bolliet, A. Dimitrov, A. Laverick, F. Villaescusa-Navarro, L. Xu et al.,Evaluating
Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in
Astrophysics, in42nd International Conference on Machine Learning, 7, 2025 [2507.07155].
[13] R. Jiang, D. Fu, C. Jiang, T. Yang, Z. Wang, Y. Wu et al.,Agentic Hybrid RAG for
Evidence-Grounded Muon Collider Analysis,2606.10381.
[14] A. Singh, M. D’Arcy, A. Cohan, D. Downey and S. Feldman,Scirepeval: A multi-format
benchmark for scientific document representations, 2023.
[15] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak et al.,Gemma:
Open models based on gemini research and technology, 2024.
[16] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix et al.,Llama:
Open and efficient foundation language models, 2023.
[17] arXiv.org submitters,arxiv dataset, 2024. 10.34740/KAGGLE/DSV/7548853.
[18] A. Singh, M. D’Arcy, A. Cohan, D. Downey and S. Feldman,Scirepeval: A multi-format
benchmark for scientific document representations, inConference on Empirical Methods in
Natural Language Processing, 2022, https://api.semanticscholar.org/CorpusID:254018137.
[19] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang et al.,Qwen3 embedding: Advancing
text embedding and reranking through foundation models,arXiv preprint arXiv:2506.05176
(2025) .
[20] Z. Nussbaum, J.X. Morris, B. Duderstadt and A. Mulyar,Nomic embed: Training a
reproducible long context text embedder,arXiv preprint arXiv:2402.01613(2024) .
[21] S. Robertson and H. Zaragoza,The probabilistic relevance framework: Bm25 and beyond,
Foundations and Trends in Information Retrieval3(2009) 333.
[22] G.V. Cormack, C.L.A. Clarke and S. Buettcher,Reciprocal rank fusion outperforms condorcet
and individual rank learning methods, inProceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Information Retrieval, 2009.
[23] N. Reimers,Ms marco cross-encoder minilm-l-6-v2, 2021.
[24] R. Nogueira and K. Cho,Passage re-ranking with BERT, inarXiv preprint
arXiv:1901.04085, 2019.
[25] Meta AI, “The llama 4 herd: The beginning of a new era of natively multimodal ai
innovation.”https://ai.meta.com/blog/llama-4-multimodal-intelligence/, Apr.,
2025.
[26] M. Le Dall, M. Pospelov and A. Ritz,Sensitivity to light weakly-coupled new physics at the
precision frontier,Phys. Rev. D92(2015) 016010 [1505.01865].
[27] K. Belotsky, A. Berkov, A. Kirillov and S. Rubin,Clusters of black holes as point-like
gamma-ray sources,Astroparticle Physics35(2011) 28–32.
[28] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang et al.,Judging
LLM-as-a-judge with MT-bench and chatbot arena,Advances in Neural Information
Processing Systems36(2023) [2306.05685].
– 20 –

[29] C. Loken, D. Gruner, L. Groer, R. Peltier, N. Bunn, M. Craig et al.,Scinet: Lessons learned
from building a power-efficient top-20 system and data centre,Journal of Physics:
Conference Series256(2010) 012026.
– 21 –
# All for law and law for all: Adaptive RAG Pipeline for Legal Research

**Authors**: Figarri Keisha, Prince Singh, Pallavi, Dion Fernandes, Aravindh Manivannan, Ilham Wicaksono, Faisal Ahmad

**Published**: 2025-08-18 17:14:03

**PDF URL**: [http://arxiv.org/pdf/2508.13107v1](http://arxiv.org/pdf/2508.13107v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations by grounding
large language model outputs in cited sources, a capability that is especially
critical in the legal domain. We present an end-to-end RAG pipeline that
revisits and extends the LegalBenchRAG baseline with three targeted
enhancements: (i) a context-aware query translator that disentangles document
references from natural-language questions and adapts retrieval depth and
response style based on expertise and specificity, (ii) open-source retrieval
strategies using SBERT and GTE embeddings that achieve substantial performance
gains (improving Recall@K by 30-95\% and Precision@K by $\sim$2.5$\times$ for
$K>4$) while remaining cost-efficient, and (iii) a comprehensive evaluation and
generation framework that combines RAGAS, BERTScore-F1, and ROUGE-Recall to
assess semantic alignment and faithfulness across models and prompt designs.
Our results show that carefully designed open-source pipelines can rival or
outperform proprietary approaches in retrieval quality, while a custom
legal-grounded prompt consistently produces more faithful and contextually
relevant answers than baseline prompting. Taken together, these contributions
demonstrate the potential of task-aware, component-level tuning to deliver
legally grounded, reproducible, and cost-effective RAG systems for legal
research assistance.

## Full Text


<!-- PDF content starts -->

All for law and law for all:
Adaptive RAG Pipeline for Legal Research
Figarri Keisha1, Prince Singh1, Pallavi1, Dion Fernandes1,
Aravindh Manivannan1,Ilham Wicaksono1,Faisal Ahmad1
1Department of Computer Science, University College London
Abstract
Retrieval-Augmented Generation (RAG) mit-
igates hallucinations by grounding large lan-
guage model outputs in cited sources, a ca-
pability that is especially critical in the le-
gal domain. We present an end-to-end RAG
pipeline that revisits and extends the Legal-
BenchRAG baseline with three targeted en-
hancements: (i) a context-aware query transla-
tor that disentangles document references from
natural-language questions and adapts retrieval
depth and response style based on expertise
and specificity, (ii) open-source retrieval strate-
gies using SBERT and GTE embeddings that
achieve substantial performance gains (improv-
ing Recall@K by 30-95% and Precision@K
by∼2.5×forK > 4) while remaining cost-
efficient, and (iii) a comprehensive evalua-
tion and generation framework that combines
RAGAS, BERTScore-F1, and ROUGE-Recall
to assess semantic alignment and faithfulness
across models and prompt designs. Our re-
sults show that carefully designed open-source
pipelines can rival or outperform proprietary
approaches in retrieval quality, while a cus-
tom legal-grounded prompt consistently pro-
duces more faithful and contextually relevant
answers than baseline prompting. Taken to-
gether, these contributions demonstrate the po-
tential of task-aware, component-level tuning
to deliver legally grounded, reproducible, and
cost-effective RAG systems for legal research
assistance.
1 Introduction
Large Language Models (LLMs) have demon-
strated strong generative capabilities but often suf-
fer from hallucinations, which is especially detri-
mental in high-stakes fields like law, where fac-
tual inaccuracies cause significant financial and
reputational damage. Retrieval-Augmented Gener-
ation (RAG) mitigates this by grounding responses
in domain-specific documents, providing concrete
sources for the LLM to reference. Some studieson improving information retrieval, such as Legal-
BenchRAG (Pipitone and Alami, 2024), providing
datasets and benchmarks, which we are leveraging
in this study to produce an innovative end-to-end
RAG pipeline. Building on the foundations laid by
LegalBenchRAG, we attempt to explore and opti-
mise pipelines for a RAG-based legal assistant .
The goal was to assess the user’s expertise level
using sentiment analysis and tailor a response of
the appropriate level, referencing the legal docu-
ments provided to the RAG. To achieve this, we
concentrated our efforts on three main components
of the pipeline: Query Translation ,Information
Retrieval andResponse Generation . The focus
of each area/team was to optimise the processes
for their respective focus, exploring aspects such as
chunking methods, embeddings, query translation,
evaluation metrics, and LLMs selection and tuning.
Building on this design, our contributions extend
beyond prior work by systematically enhancing
each stage of the pipeline and demonstrating how
open-source methods can rival proprietary systems.
Specifically:
Contribution 1 (Context-Aware Query Transla-
tor): We design a lightweight query pre-processing
module that disentangles document references from
natural-language questions and classifies queries
by expertise ( expert vs. non-expert ) and speci-
ficity ( vague vs. verbose ). These signals guide
context-aware translation , adapting both retrieval
depth and response style to the user’s needs.
Contribution 2 (Open-source retrieval strate-
gies): We demonstrate that open-source embed-
ding models (SBERT ,GTE ) combined with file-
aware query translation can rival (and in some cases
outperform) the proprietary OpenAI model used in
LegalBenchRAG. Our tailored retrieval achieves
substantial gains, improving Recall@K by 30–
95% andPrecision@K by ∼2.5×forK > 4,
while remaining cost-efficient and reproducible.arXiv:2508.13107v1  [cs.CL]  18 Aug 2025

Contribution 3 (Evaluation and response gen-
eration): We introduce a comprehensive eval-
uation suite that combines RAGAS faithful-
ness and answer relevancy with BERTScore-F1
and ROUGE-Recall to assess semantic align-
ment, factual grounding, and completeness. Us-
ing this framework, we systematically evaluate
prompt designs and model choices (e.g., GPT-4o-
mini, LLaMA-3-8B), finding that a custom legal-
grounded prompt consistently produces more faith-
ful and contextually relevant outputs than baseline
prompting approaches.
2 Background
LegalBenchRAG proposed an information re-
trieval pipeline, measuring precision and re-
call of retrieval by experimenting with Naïve
and Recursive Text Character Split (RCTS)
chunking, using OpenAI’s embedding model
(text-embedding-3-large) and cosine similarity
search with and without reranking using Cohere’s
rerank-english-v3.0 model (Guha et al., 2023).
The paper concluded that RCTS performed better ,
while Cohere’s reranking reduced overall retrieval
performance. LegalBenchRag is a good benchmark
for retrieval performance, utilising a standard, well-
labelled corpus and query-answer (QA) pairs.
Recent works like Adaptive-RAG (Jeong et al.,
2024) and HyPa-RAG (Kalra et al., 2025) signif-
icantly influenced our system design. Adaptive-
RAG introduced query-aware reasoning paths by
using classified query complexity to guide retrieval
depth. This inspired our use of complexity predic-
tion to adapt chunk size and prompt design during
response generation. HyPa-RAG’s hybrid retrieval
and query rewriting approach, employing dense
and sparse retrievers guided by query complexity,
motivated us to explore adaptive retrieval config-
urations based on document reference relevance.
We extend these ideas with our query translation
module , splitting each input into a document refer-
ence andmain question , enabling more precisely
targeted retrieval and informed generation.
Some research collated state-of-the-art (SoTA)
RAG methods with benchmarks such as BERGEN
(Rau et al., 2024), a Python library for straightfor-
ward, reproducible end-to-end RAG experiments.
This paper proposed and benchmarked SoTA in-
formation retrieval methods such as dense-encoder
RetroMAE andSPLADE , a sparse model, both of
which we considered for our experimental setup.Due to resource constraints, we are unable to
replicate the experiments using OpenAI’s propri-
etary embedding model. However, from a practical
standpoint, adopting high-performing open-source
embedding models is more desirable. Therefore,
we focus on evaluating multiple open-source mod-
els to determine whether any can match or exceed
the retrieval performance of LegalBenchRAG. Ad-
ditionally, while LegalBenchRAG provides strong
retrieval benchmarks, it does not explore end-to-
end response generation. This leaves an open ques-
tion: How well does their retrieval method integrate
into a complete RAG pipeline incorporating im-
proved query translation and response generation
to produce more accurate responses? To investigate
this, we addressed three hypotheses:
1.Canopen-source embedding models match
or outperform the retrieval performance of
proprietary models ?
2.Doalternative similarity search methods
lead to improved retrieval results?
3.Canreranking with different encoder models
enhance retrieval effectiveness?
3 Data
3.1 Dataset
We are using the LegalBenchRAG corpus, which
was derived from the LegalBench (Guha et al.,
2023) dataset, a collaboratively constructed rea-
soning benchmark consisting of 162 tasks covering
six different types of legal reasoning. Legal ex-
perts annotated queries by highlighting relevant
text in source documents. LegalBench-RAG data
has two primary components: the original cor-
pusand the QA pairs . The corpus includes .txt
documents from our four legal domains . It ex-
cludes documents that were not requested/targeted
by at least one query in LegalBenchRAG. The QA
pairs are directly linked to the documents within
the corpus. Each query is associated with a list
of relevant snippets from source documents that
directly answer the query. For each snippet, the
file path, the exact quote, and the precise character
indices within the document are provided, ensuring
a clear reference to the source, enabling accurate
evaluation of retrieval performance.
3.2 Data Sampling
LegalBenchRAG samples 194 QA pairs per
domain and extracts relevant text files to

create a lightweight, balanced subset termed
LegalBenchRAG-mini , for experimentation. It
minimises the number of text files by selecting
unique QA pairs within the smallest possible set.
We replicated this process, but exact reproduction
was impossible due to an unknown random seed.
However, our sampling yielded a similarly sized
subset of text files and QA pairs, though not identi-
cal (see Table 1).
Dataset Corpus QA
Contract (LegalBench) 18 194
Contract (B.E.R.T) 20 194
CUAD (LegalBench) 18 194
CUAD (B.E.R.T) 17 194
MAUD (LegalBench) 29 194
MAUD (B.E.R.T) 16 194
Privacy (LegalBench) 7 194
Privacy (B.E.R.T) 7 194
Total (LegalBench) 72 776
Total (B.E.R.T) 60 776
Table 1: Corpus and QA sample sizes across four
datasets in LegalBench and B.E.R.T.
4 Experimental Setup
4.1 Query Translation
We observed that many queries in the LegalBench
dataset follow a rigid pattern, beginning with “Con-
sider . . . ; . . . ," where the text before the semi-
colon specifies a document reference . To address
this, we built a Simple Extractor (SE) that splits
queries at the semicolon, removes stopwords (e.g.,
"Non-Disclosure Agreement"), and embeds the
document reference with a sentence transformer
(all-MiniLM-L6-v2 ) to find the best-matching file
viacosine similarity . Matches are scored as 1
for correct ,-1 for incorrect , and 0if the sim-
ilarity falls below a threshold. Because CUAD
file names often diverge from the textual content
of queries, we increased the threshold to 0.55
for CUAD to reduce mismatches, while lower
thresholds ( 0.3–0.38 ) were sufficient for the other
datasets. Table 2 reports SE’s performance on the
original dataset queries with their respective thresh-
olds.
Although SE performs well when queries strictly
follow the "Consider . . . ; . . . " pattern, many
queries deviate from this format in real usage. To
test robustness, we rephrased all queries in theDataset (threshold) -1 0 1
ContractNLI (0.3) 0 15 179
CUAD (0.55) 0 114 80
MAUD (0.38) 0 0 194
PrivacyQA (0.3) 0 0 192
Table 2: Performance of the Simple Extractor (SE)
across datasets, measured by match scores at varying
thresholds.
datasets using a few-shot prompt , ensuring that the
document reference (e.g., "In the Non-Disclosure
Agreement between CopAcc and ToP Mentors
. . . ") was preserved while rewriting the question
into a more conversational style (e.g., "Is it clearly
stated that the Receiving Party has no rights . . . ?").
We employed Flan-T5 Large to generate these
more natural queries, approximating real-world sce-
narios.
Our experiments (see Table 3 in Appendix A)
show that SE’s accuracy drops on rephrased
queries, since they no longer follow the semicolon-
delimited structure. However, a Named Entity
Recognition (NER) –based approach can effec-
tively extract the document reference, resulting in
only a modest increase in -1 scores. For datasets
such as CUAD, where file names are less indica-
tive of query content, similarity scores were lower,
requiring threshold adjustments to mitigate mis-
matches. Importantly, generating rephrased vari-
ants for all four datasets substantially expanded the
LegalBench corpus, enhancing its realism for legal
document retrieval experiments.
To further enhance query understanding and
adapt downstream processing, we added a feature
extraction component that classifies queries based
onlinguistic complexity andspecificity :
1.Expertise Classification: We employed the
Dale-Chall readability formula to estimate
the linguistic complexity of each query for all
4 datasets. It was chosen for its proven ability
to assess comprehensibility in formal domains
like legal text. Unlike metrics based only on
sentence length or syllables, it uses a curated
list of familiar words, making it better suited
for structured, jargon-heavy text, an observa-
tion supported by Han et al. (2024) for legal
texts and Zheng and Yu (2017) for technical
health documents. To distinguish between
domain experts andlaypersons , we used
a threshold: scores below 8.0were labelled

non-expert , and 8.0 or above asexpert , thus
guiding the response generation module to
produce answers that are either technically de-
tailed or simplified for broader accessibility.
2.Vague vs Verbose Classification: To adapt
the retrieval strategy based on query speci-
ficity , we introduced a classification mech-
anism that labels queries as vague orver-
bose . This distinction enables the system to
dynamically adjust the number of chunks
retrieved during grounding: vague queries,
which are general or under-specified, ben-
efit from broader retrieval, while verbose
queries, often multi-part or over-specified, re-
quire more selective, targeted context win-
dows. Following HyPA-RAG (Kalra et al.,
2025), we constructed a synthetic dataset
by rephrasing LegalBench queries for all 4
datasets with Meta-LLaMA-3 to create diverse
vague and verbose variants. This dataset was
used to train DistilBERT for binary classifi-
cation. Once classified, the vague or verbose
label was passed to the retrieval module, guid-
ing chunking behaviour and context scaling
for downstream LLM processing.
4.2 Information Retrieval
1.Embedding Model: We compared three
open-source embedding models: SBERT
(all-mpnet-base-v2 ),Distilled SBERT
(all-MiniLM-L6-v2 ), and GTE-Large
(thenlper/gte-large ) against Legal-
BenchRAG’s text-embedding-3-large .
These models span a range of scales: mpnet-
base (110M) offers high accuracy, MiniLM
(22M) is lightweight and efficient, and
GTE-Large is optimised for retrieval and
reranking in RAG pipelines. This selection
allowed us to evaluate how model size and
design influence retrieval performance.
2.Pipeline: The sampled corpus was chunked
using Naïve andRCTS methods, embedded
with three models, and stored as JSON vec-
tors. Queries from the benchmarks were also
embedded, and similarity search ( cosine and
BM25 ) retrieved the top 50 chunks , referred
to as unranked for comparison purposes with
reranked chunks . However, they are ordered
by similarity scores. A reranker then re-
ordered these based on query similarity to pro-
duce reranked chunks. Precision andrecallwere calculated by comparing unranked and
reranked results against ground truth spans
across chunking–embedding–similarity com-
binations and k-values (1–50). Additionally,
we tested RetroMAE, a SoTA embedding
model from the BERGEN paper (Rau et al.,
2024), against the best-performing configura-
tion to evaluate its effectiveness in the RAG
pipeline.
3.Evaluation Approach: Following Legal-
BenchRAG, we used Precision@K andRe-
call@K based on span overlap . Retrieved
chunks were compared to benchmark QA
ground truth by file and span alignment. Pre-
cision was computed as overlap length over
chunk length, and recall as overlap over
ground truth length. With multiple chunks
and answers per query, scores were averaged
across 194 QA pairs per domain and overall,
over varying K-values (see Figure 2). After
identifying the best model, we extended the
evaluation to include a SoTA method and in-
creased K to 300 for direct comparison with
LegalBenchRAG. This extended evaluation
was run only on the top-performing model
due to resource constraints.
We explored using text-based evaluation in
addition to span-based evaluation as a way
to measure sentiment similarity between re-
trieved chunks and ground truth. Ultimately,
we decided to proceed using primarily span-
based, and used the text-based evaluation as
a sanity check by paying attention to the
behaviour of precision and recall across K-
values, such as in Figure 6 in Appendix B.
4.3 Response Generation and Evaluation
(RGE)
Unlike (Pipitone and Alami, 2024), which focuses
on retrieval, we also explore and evaluate response
generation in the legal domain. The RGE is done
in two parts: i) evaluation without complexity clas-
sifier and readability metrics, to determine the op-
timal context length, language model, and prompt
for the final run and the relevance of each metric
for our legal domain use case, and ii) evaluation of
the final response employing complexity classifier
and readability metrics.
1.Prompt Designs: We experimented with
several prompts: i) baseline is the RAG

prompt from LangChain, ii) zero-shot Chain
of Thought (CoT) prompt to assess poten-
tial reasoning improvements, and finally iii)a
custom-crafted prompt with explicit instruc-
tions to enhance accuracy, legal grounding,
and relevance.
2.Evaluation Metrics: RAGAS (RAG As-
sessment) is used, especially Answer Rele-
vancy andFaithfulness (Exploding Gradi-
ents, 2025). The original RAGAS paper pro-
posed Context Relevance, but it was later
deemed unhelpful by the authors (Es et al.,
2023). Instead, we focus on two RAGAS
reference-free metrics :Answer Relevancy ,
which compares LLM-generated questions
(from the model’s response) to the original
question using similarity scores, and Faith-
fulness , which checks if the retrieved context
supports LLM-generated claims from the re-
sponse. In addition, we consider using two
reference-based metrics :BERTScore-F1
(with LegalBERT) to measure semantic sim-
ilarity between generated answers and con-
texts, and ROUGE-Recall to assess complete-
ness through n-gram overlap, to support faith-
fulness assessment.
3.Models: GPT-4o-mini , and LLAMA-3-8B .
4.Response generation framework: Before
fitting to the final pipeline and using query
translation outputs, we want to find the op-
timal combination of prompt, model, and
k-value by generating responses for all possi-
ble combinations using the chunks retrieved
by the best model. K-values were varied (1,
3, 5, 10) to compare its effect on generation
performance.
5 Final Pipeline
The final end-to-end RAG pipeline combined query
translation, information retrieval, and response gen-
eration stages with parameters that achieved the
optimal performance during experimentation (see
Figure 1). Query translation includes query rewrit-
ing, file path extractor and complexity classifica-
tion. Any detected file paths with scores above
the dataset-specific thresholds narrow the search to
retrieve chunks only from that specific file. Oth-
erwise, the entire database is considered for doc-
ument retrieval. The feature extraction compo-
nent classifies the expertise level of the query andadapts the LLM-generated response to match the
predicted knowledge level of the user. The query
is also classified to be either vague or verbose, tai-
loring the amount of information retrieved at the
retrieval stage. The queries, along with this pre-
dicted metadata are passed to the information re-
trieval stage. At this stage, the sampled corpus is
chunked (RCTS) and embedded (SBert). Cosine
similarity is then used to retrieve the top-k most
relevant chunks for each query. The top-k value
is adjusted based on the query type (i.e. vague or
verbose). The relevant chunks are then passed on to
the response generation module, using a fine-tuned
prompt, GPT model and relevant chunks to gener-
ate a response, adapted to Expert vs. Non-Expert
classification conducted at the query translation
stage. The generated response is then evaluated us-
ing Faithfulness, answer relevancy, BERTScore-F1
and ROUGE-Recall.
6 Results and Analysis
6.1 Recall@K and Precision@K
Theinformation retrieval performance was eval-
uated based on the experiments using various mod-
els and with the use of Query Translation , compar-
ing against LegalBenchRAG benchmarks as well
as SoTA methods.
We observed a few similar findings with Legal-
BenchRAG. Firstly, RCTS performs better than
Naive chunking, although not consistently for all
model combinations. Some experiments, such as
Naive+SBERT+Cosine andNaive+GTE+Cosine ,
were in the top 5 performers, at times performing
well for individual domains. However, considering
overall performance, the top 2 model combina-
tions used RCTS chunking . Thus, overall we can
conclude that RCTS outperformed Naive chunk-
ing. Secondly, unranked results perform better
than reranked ones for cosine similarity and con-
versely for BM25 similarity , proving our Hypoth-
esis 3 inconclusive and dependent on model choice.
LegalBenchRAG also observed that unranked per-
forms better with cosine similarity. Figure 2 shows
these comparisons. Figure 2 also shows that co-
sine similarity performs better than BM25 simi-
larity search, answering Hypothesis 2 . The per-
formances may vary for each domain, which is
presented in Figure 7 in Appendix C.
RCTS+SBert+Cosine+unranked is the
best-performing model combination consid-
ering both precision and recall, followed by

Figure 1: End-to-end pipeline for filtered information retrieval and response generation. The system begins
with a sampled corpus processed via chunking and embedding, stored in vector files. Queries from sampled
QA benchmarks are optionally rewritten and analysed for complexity and user expertise. These inputs guide the
similarity search and k-filtering process to retrieve relevant information chunks for response generation. Final
evaluation is based on both precision/recall and faithfulness/relevance.
RCTS+GTE+Cosine+unranked . Compared
to LegalBenchRAG, which uses OpenAI (text-
embedding-3-large), our best model performs
slightly better on precision for all k-values
andvery similar for recall for initial k-values,
which gets lower as k-value increases (Figure
3). To address our Hypothesis 1 ,SBERT is an
open-source model that can be used reliably in this
pipeline with similar performance to the OpenAI
model, without incurring significant computational
resources.
The best model combination also outperforms
the SoTA model (RetroMAE ) for both recall and
precision (Figure 3), proving that established open-
source embedding models are still providing suf-
ficient performance for RAG pipeline as compared
to the SoTA model. However, we may need to con-
sider that SoTA models may require further finetun-
ing to achieve optimal performance, which could
surpass performances seen in these experiments.
6.2 Response Generation and Evaluation
The Faithfulness and BERT-F1 score in Figure 4
show that the custom-crafted prompt with GPT
consistently performs better than the other com-
binations across the K values. The Faithfulness
andBERT-F1 score show little variation across K
value, not providing any conclusive optimal K. Incontrast, answer relevancy seems to favour the base-
line prompt with GPT across all K values compared
to the custom-crafted prompt. Upon further analy-
sis (as shown in the Appendix D), the answer rele-
vancy metric is not well-suited to identify optimal
performance. Our analysis shows that answer rele-
vancy primarily reflects the rate of non-committal
answers, compared to the actual similarity score it-
self, downplaying the latter, which we feel is more
relevant to this analysis. A modified answer rele-
vancy metric excluding the non-committal multi-
plier showed comparable results for both the base-
line and custom-crafted prompts, reflecting the true
cosine similarities. Thus, we can briefly conclude
that the use of answer relevancy is not conclu-
sivein finding the optimal performance. However,
further studies are required in future work to as-
sess the best alternative metrics. ROUGE-Recall
indicates Llama + custom-crafted prompt to be
the best performing; however, its trend contradicts
Faithfulness andBERT-F1 , in contrary to our ex-
pectation, considering that all three metrics assess
semantic similarity between generated answers and
contexts. BERT-F1 exhibits a similar behaviour
to Faithfulness as the number of retrieved chunks
k increases, which can be attributed to its use of
contextual embeddings. Since BERT-F1 measures

Figure 2: Precision@K and Recall@K across ranked
and unranked experiments. Each curve corresponds to
a retrieval configuration (chunking method, embedding
model, and similarity search). Precision@K decreases
asKincreases, while Recall@K improves, reflecting
the trade-off between retrieving broader context and
maintaining accuracy.
semantic overlap using embeddings from a domain-
specific language model (in our case, LegalBERT),
it captures meaning even when the phrasing differs,
a critical feature in legal documents where para-
phrasing is common. The deviation of ROUGE-
Recall could be due to its reliance on surface-level
lexical overlap , which does not account for se-
mantic similarity and penalises valid paraphrasing
or abstraction (more details are in Appendix E).
Therefore, we do not pursue using ROUGE-Recall
for evaluation, and identify GPT with a custom-
crafted prompt to be an ideal choice as supported
by Faithfulness and BERT-F1.
In considering the optimal K-value , we note that
theFaithfulness stagnates after K=5 and BERT-
F1 stagnates with marginal fluctuations for all K-
values. Based on Figure 2, we can see that having
too high a K-value will result in a loss of precision
but a gain on recall for information retrieval, thus
requiring a fine balance in choosing the optimal
K-value to trade-off precision and recall. Thus, we
can consider a K around 5 , which has good Faith-
fulness, BERT-F1, precision, and decent recall.
For the second part of the evaluation, which in-
corporates the Readability and Complexity clas-
sifier (R&C) , we applied K=5 for non-expert
queries and increased K to 10 for expert queries ,
to add more details for complex questions. Figure
5 shows that including R&C does not change the
performance significantly .
Figure 3: Precision@K and Recall@K for se-
lected retrieval configurations. Comparison of the
RCTS_SBert_Cos baseline against its variant with
Query-Rewriter, the RCTS_RetroMAE_Cos model, and
the LegalBenchRAG reference. The plots illustrate how
query rewriting and embedding choice impact retrieval
quality across different values of K.
Qualitative analysis (more details are in Ap-
pendix F) confirms that the inclusion of R&C ad-
justs its tone and content based on query complex-
ity: responses to non-expert queries are concise
and free of excessive legal jargon, while responses
to expert queries are more detailed and contain
strong legal grounding, offering adaptive retrieval
strategies without compromising response quality.
7 Conclusions
LLMs often suffer from hallucinations , which is a
critical issue in the legal domain where informa-
tion is crucial. This study addresses this key issue
by successfully optimising an end-to-end RAG
pipeline for legal documents, advancing beyond
theLegalBenchRAG by integrating query trans-
lation ,retrieval andresponse generation . Us-
ingopen-source models (SBERT ,GTE-large ) en-
abled cost-effective and more accessible pipelines

Figure 4: Evaluation metrics across prompt strategies and models. Comparison of GPT-4o-mini and LLaMA-
3-8B under three prompting strategies (baseline, zero-shot CoT, and custom legal-grounded). Metrics include
faithfulness, BERTScore-F1, ROUGE-Recall, and answer relevancy, showing how prompt design and model choice
jointly affect response quality in the legal domain.
Figure 5: Impact of readability and complexity on
response generation. Evaluation of responses with and
without the readability and complexity classifier shows
how adapting output to expert versus non-expert queries
affects response style and content, while maintaining
overall response quality.
without the reliance on commercial APIs. The
pipeline incorporates a novel strategy including
query translation and complexity-driven chunking
and generation. Complexity classification (i.e. ex-
pert vs. non-expert and vague vs. verbose) has
helped to reduce hallucinations and tailor retrieval.
The simple extractor used on structured queries
helped to significantly improve the recall and pre-
cision for retrieval. For the retrieval stage, RCTS
and cosine similarity outperformed Naïve chunk-ing and BM25. Query translation significantly in-
creased recall@K andPrecision@K for informa-
tion retrieval, while adaptive top-k retrieval based
on complexity outperforms fixed-k in response
generation. Furthermore, faithfulness andBERT-
F1evaluated legal response quality better than
ROUGE . This highlights the value of adapting re-
trieval and generation according to query complex-
ity, providing a strong foundation for lightweight
legal tools without commercial APIs.
However, limited resources restricted rerank-
ing, fine-tuning, and testing higher K-values com-
pared to LegalBenchRAG. Furthermore, the le-
gal dataset corpus covered only NDAs, M&A
agreements, commercial contracts and privacy poli-
cies and did not support queries requiring informa-
tion from multiple documents. Future work could
explore advanced retrievers (e.g. ColBERT ,
SPLADE ) and multi-document queries . Addi-
tionally, domain-specific fine-tuning anduser ex-
perience studies on domain experts vs laypersons
could be explored. In general, this research shows a
practical and scalable method for implementing
RAG pipelines in the legal field, striking a balance
between accuracy and accessibility.

Limitations
TheLegalBenchRAG-mini dataset , while broad,
covering NDAs ,M&A agreements ,commercial
contracts , and privacy policies , is not exhaustive
of all types of legal documents. Additionally, each
query in this benchmark is answerable by a single
document , limiting its ability to evaluate multi-
document reasoning . It primarily tests a system’s
capability to retrieve the correct document and rel-
evant snippets within it.
Our response generation experiments were
limited to K=10 , and it could still be suboptimal
considering that Recall@K continues to improve
at larger K values. However, extending response
generation to higher K values exceeds the scope
and resource limits of our research. These con-
straints may have affected the generalisability and
upper-bound performance of our RAG system,
particularly in complex queries requiring broader
context. Further work can explore extending the
runs beyond these limitations. In-depth analysis
offaithfulness andBERT-F1’s performance satu-
ration at K=5 should be explored in future research,
as such deep meta-analysis of evaluation methods
are beyond the scope of this project. Future work
can also evaluate the usefulness of the adaptive re-
trieval performance using a complexity classifier
with human-in-the-loop validation .
References
I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Ale-
tras, and I. Androutsopoulos. 2020. LEGAL-BERT:
The muppets straight out of law school. In Find-
ings of the Association for Computational Linguistics:
EMNLP 2020 , pages 2898–2904, Online. Association
for Computational Linguistics.
S. Es, J. James, L. Espinosa-Anke, and S. Schockaert.
2023. Ragas: Automated evaluation of retrieval aug-
mented generation. CoRR , abs/2309.15217.
Exploding Gradients. 2025. GitHub - explod-
inggradients/ragas: Supercharge Your LLM Ap-
plication Evaluations. https://github.com/
explodinggradients/ragas/tree/main . Ac-
cessed: 2025-04-07.
N. Guha, M. Lamm, B. Wu, A. Zhang, J. Yin, Y . Taori,
Y . Zhang, S. S. Schoenholz, R. G. Krishnan, and C. D.
Manning. 2023. Legalbench: A collaboratively built
benchmark for measuring legal reasoning in large
language models. CoRR , abs/2308.11462.
Y . Han, A. Ceross, and J. H. M. Bergmann. 2024. The
use of readability metrics in legal text: A systematic
literature review. CoRR , abs/2411.09497.S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park.
2024. Adaptive-rag: Learning to adapt retrieval-
augmented large language models through question
complexity. CoRR , abs/2403.14403.
R. Kalra, Z. Wu, A. Gulley, A. Hilliard, X. Guan,
A. Koshiyama, and P. Treleaven. 2025. Hypa-rag: A
hybrid parameter adaptive retrieval-augmented gen-
eration system for ai legal and policy applications.
CoRR , abs/2409.09046.
N. Pipitone and G. H. Alami. 2024. Legalbench-rag: A
benchmark for retrieval-augmented generation in the
legal domain. CoRR , abs/2408.10343.
D. Rau, S. Chau, R. Kalra, A. Wang, B. Faltings, P. Tre-
leaven, and A. Koshiyama. 2024. Bergen: A bench-
marking library for retrieval-augmented generation.
CoRR , abs/2407.01102.
J. Zheng and H. Yu. 2017. Readability formulas and
user perceptions of electronic health records diffi-
culty: A corpus study. Journal of Medical Internet
Research , 19(3):e59.

A Named Entity Recognition-based
approach
To evaluate robustness on rephrased queries, we
compared the Simple Extractor (SE) , which relies
on semicolon-delimited patterns, against a Named
Entity Recognition (NER) –based method. Ta-
ble 3 reports the distribution of match scores ( −1,
0,1) across datasets after rephrasing all queries
into more natural forms.
Model Dataset -1 0 1
SE ContractNLI 0 194 0
CUAD 0 194 0
MAUD 0 194 0
PrivacyQA 0 194 0
NER ContractNLI 4 29 161
CUAD 0 180 14
MAUD 7 24 163
PrivacyQA 0 83 169
Table 3: Performance of SE and NER on rephrased
queries. Distribution of match scores ( −1: incorrect,
0: no match, 1: correct) across four datasets after
rephrasing queries into conversational style. The NER-
based approach substantially improves matching accu-
racy compared to SE, which fails under rephrasing.
B Text vs. Span-based Evaluation
To validate retrieval performance, we compared
span-based evaluation (matching retrieved text
spans to gold annotations) with text-based evalu-
ation (measuring semantic similarity between re-
trieved chunks and ground truth). While span-
based metrics are more reliable for legal retrieval,
text-based evaluation served as a sanity check.
Figure 6 illustrates Precision@K andRecall@K
trends for the ContractNLI domain under both
approaches.
C Information Retrieval Performance by
Domain
To provide finer-grained insights, we report
domain-specific Precision@K andRecall@K re-
sults across all retrieval configurations. Figure 7
presents the performance curves for ContractNLI ,
CUAD ,MAUD , and PrivacyQA , allowing com-
parison of ranked versus unranked outputs within
each dataset.
(a) Precision@K for the ContractNLI domain.
(b) Recall@K for the ContractNLI domain.
Figure 6: Comparison of text-based and span-based
evaluation in ContractNLI. Precision@K and Re-
call@K curves show that while span-based evaluation
provides stricter alignment with annotated spans, text-
based evaluation follows similar trends and was used as
a supplementary check.
D Quantitative Look into RAGAS
Answer Relevancy
On their paper and webpage, answer relevancy
is described as a calculation of how relevant the
answer is to the original question. This is done
by generating a number of artificially generated
questions from the response, and calculating the
similarity score of them to the original question ( 3
generated question by default):
answer relevancy =1
NNX
i=1Egi·Eo
∥Egi∥∥Eo∥(1)
•Egiis the embedding of the generated ques-
tioni.
•Eois the embedding of the original question.
•Nis the number of generated questions,
which is 3 by default.
A closer look at their implementation revealed
that they also do classification on the answer as
acommittal andnon-committal answer. A non-
committal answer is evasive, vague, or ambiguous.
For example, "I don’t know" or "I’m not sure" are

Figure 7: Domain-level Precision@K and Recall@K across retrieval experiments. Results are shown separately
for ContractNLI, CUAD, MAUD, and PrivacyQA. Solid lines denote ranked outputs, while dashed lines denote
unranked outputs. The curves highlight variations in retrieval effectiveness across domains and confirm that
performance differences are dataset-dependent.
noncommittal answers. If a non-committal an-
swer is detected, the final score will be multiplied
by 0.
We found out that this has a high influence
on lowering the total (average) answer relevancy
score in a batch of queries compared to the actual
similarity score. We finally recalculated answer
relevancy for samples in contractNLI document
forbaseline prompt andhuman-tuned prompt
without non-committal multiplier and found out
they both got a very high score around 0.9, and not
that different between the two. Analysis of non-
committal answers is in the qualitative analysis
part.
We conclude that the answer relevancy score
alone is not a definitive indicator of model qual-
ity, as it mainly reflects the rate of non-committal
answers . Recalculated scores excluding this factor
yield similarly high and comparable results acrossmodels. Moreover, the non-committal count de-
creases for both prompts as kgrows, suggesting
thatadditional retrieved context contributes to
more decisive responses , in line with Recall@K
trends in information retrieval.
Another parameter in answer relevancy is the
number of artificial questions generated. Our
initial hypothesis was that increasing the number
of generated questions would provide an advantage
as the number of retrieved contexts ( k) grows. To
test this, we compared the default setting ofthree
generated questions with an extended setting of
five, focusing on the ContractNLI dataset with
GPT-4o-mini .
It can be seen that the difference between scores
from 3 questions and5 questions areminuscule .
We conclude to just use the default number of 3
questions for all our evaluations.

Figure 8: Cosine similarity versus final score across kvalues. The left plot reports mean cosine similarity
excluding the non-committal multiplier, while the right plot shows final scores including the multiplier. Results
compare baseline prompts against human-tuned prompts, highlighting how prompt design interacts with scoring
criteria.
Figure 9: Non-committal responses across kvalues.
The frequency of non-committal answers decreases as
more context is retrieved, supporting the observation
that higher recall contributes to more decisive model
outputs.
E Using Rouge as an Evaluation Metric
Our initial setup included ROUGE recall (aver-
age of ROUGE-1 ,ROUGE-2 , and ROUGE-L )
as one of the core metrics to evaluate content
overlap with reference answers. However, as
shown in Figure 4, we observe that ROUGE re-
callexhibits a declining trend as the number of
retrieved chunks (k) increases, in contrast to met-
rics such as BERTScore F1 andRAGA Faithful-
ness/Relevancy , which stabilise or improve. This
discrepancy motivated an in-depth investigation of
thelimitations of ROUGE in this context.
ROUGE was originally designed to evaluate ex-
tractive summarization by computing n-gram
overlap between generated and reference texts
(Chalkidis et al., 2020). While effective for sum-
marisation and simple question-answering scenar-
ios, it falls short in evaluating abstractive orse-
mantically equivalent generation, especially when
Figure 10: Answer relevancy with different numbers
of generated questions. Comparison of default (3) ver-
sus extended (5) artificial questions on ContractNLI us-
ing GPT-4o-mini, showing how the number of generated
questions influences answer relevancy as kincreases.
the wording differs from the reference but the mean-
ing is preserved.
•Semantic vs. Lexical Similarity: ROUGE
heavily relies on surface-level lexical over-
lap, penalizing answers that are semantically
correct , but lexically divergent from the gold
answer. In contrast, as the number of retrieved
chunks (k) increases, the model has more con-
text to paraphrase orsynthesise information,
often resulting in semantically accurate but
lexically novel responses. This leads to low
ROUGE scores despite high answer quality.
•Recall-Oriented Nature: ROUGE-recall fa-
vors longer answers that capture more ref-
erence n-grams. However, in a RAG setting
with an increase in k, the model may gener-
ate more focused and concise responses due
to better context resolution. This penalises
shorter, yet more precise answers , leading
to deceptively low ROUGE recall.

•Empirical divergence: As shown in our re-
sults, ROUGE recall decreases monotoni-
cally with increasing k, whereas RAGAS
Faithfulness and Relevancy (which evalu-
ate whether the answer is supported by re-
trieved knowledge and relevant to the ques-
tion) increase or stabilise. This indicates that
ROUGE recall ismisaligned with the actual
performance of the system, potentially mis-
leading model development and tuning.
Justification for Removal: Given these limita-
tions, we argue that ROUGE recall is an inade-
quate metric for evaluating RAG-based question
answering systems where: (i) Answers are gener-
ated using paraphrased orsynthesised language.
(ii)Multiple retrievals lead to more diverse but
valid completions. (iii) Lexical similarity is not a
reliable proxy for answer quality .
Hence, for more meaningful and context-aware
evaluation, we focus on RAGAS Faithfulness and
Relevancy , which accounts for whether the answer
is grounded in retrieved content; BERTScore F1
(Exploding Gradients, 2025), which reflects seman-
tic similarity at the embedding level. By exclud-
ingROUGE recall , we avoid penalising correct
responses that diverge lexically, and better align
our evaluation metrics with the goals of faithful
andrelevant generation.
F Qualitative Analysis of GPT-4o Mini
With Readability and Complexity
From the sample responses, we observe that ap-
plying the complexity classifier yields concise an-
swers for simpler queries (e.g., Query 3 ), making
them more accessible, while expert queries re-
ceive additional detail (e.g., Queries 1 and 2 ) to
better match the required level of specificity.
Query 1:
Consider the Non-Disclosure Agreement
between CopAcc and ToP Mentors; Does
the document permit the Receiving Party
to retain some Confidential Information
even after its return or destruction?
Classification: Expert (K=10)
Response With R&C: The NDA does
not permit the Receiving Party (Mentor)
to retain any Confidential Information af-
ter its return or destruction. According to
Clause 6 , upon the Organiser’s request,
the Mentor is required to:•Discontinue all use of Confidential
Information,
•Return all materials containing
Confidential Information,
•Destroy any copies of materials
that contain Confidential Informa-
tion,
•Erase or destroy any Confiden-
tial Information stored in computer
memory or data storage,
•Remove Confidential Information
from any software or database used
by the Mentor.
Furthermore, the Mentor must provide a
written warranty within ten days con-
firming these actions. This indicates a
clear obligation to eliminate any reten-
tion of Confidential Information.
Response with K=5: The document sim-
ilarly requires the Mentor to discontinue,
return, and destroy all Confidential Infor-
mation, including computer-stored data,
and to warrant in writing that these ac-
tions are complete. There are no provi-
sions allowing retention.
Observation: The R&C response is
more detailed, emphasising the lack of
retention rights .
Query 2:
Consider the Data Use Agreement in
New York City; Does the document re-
strict the use of Confidential Information
to the purposes stated in the Agreement?
Classification: Expert (K=10)
Response With R&C: Yes, the Agree-
ment restricts the use of Confidential In-
formation to the purposes stated, specifi-
cally those in Attachment B . The Data
Recipient is further prohibited from dis-
closing identifiable information and must
obtain written approval before releas-
ing public information, reinforcing these
restrictions.
Response with K=5: Yes, the Agree-
ment limits use to Attachment B pur-
poses, and explicitly requires that any
published work must not contain identi-
fiable information.

Observation: TheR&C response high-
lights confidentiality obligations more
thoroughly.
Query 3:
Consider the Marketing Affiliate Agree-
ment between Birch First Global Invest-
ments Inc. and Mount Knowledge Hold-
ings Inc.; What is the expiration date of
this contract?
Classification: Non-Expert (K=5)
Response With R&C: The Agreement
expires on July 22, 2019 , as stated: “the
Agreement shall expire and terminate
automatically without further notice on
July 22, 2019.”
Response with K=10: The Agreement is
valid for five (5) years from its effective
date of July 22, 2014 , thus expiring on
July 22, 2019 , unless earlier terminated
with 30 days’ notice .
Observation: The R&C response is
more concise , directly stating the expira-
tion date.
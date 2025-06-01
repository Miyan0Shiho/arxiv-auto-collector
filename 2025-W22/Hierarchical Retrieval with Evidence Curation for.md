# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents

**Authors**: Jaeyoung Choe, Jihoon Kim, Woohwan Jung

**Published**: 2025-05-26 11:08:23

**PDF URL**: [http://arxiv.org/pdf/2505.20368v1](http://arxiv.org/pdf/2505.20368v1)

## Abstract
Retrieval-augmented generation (RAG) based large language models (LLMs) are
widely used in finance for their excellent performance on knowledge-intensive
tasks. However, standardized documents (e.g., SEC filing) share similar formats
such as repetitive boilerplate texts, and similar table structures. This
similarity forces traditional RAG methods to misidentify near-duplicate text,
leading to duplicate retrieval that undermines accuracy and completeness. To
address these issues, we propose the Hierarchical Retrieval with Evidence
Curation (HiREC) framework. Our approach first performs hierarchical retrieval
to reduce confusion among similar texts. It first retrieve related documents
and then selects the most relevant passages from the documents. The evidence
curation process removes irrelevant passages. When necessary, it automatically
generates complementary queries to collect missing information. To evaluate our
approach, we construct and release a Large-scale Open-domain Financial (LOFin)
question answering benchmark that includes 145,897 SEC documents and 1,595
question-answer pairs. Our code and data are available at
https://github.com/deep-over/LOFin-bench-HiREC.

## Full Text


<!-- PDF content starts -->

arXiv:2505.20368v1  [cs.IR]  26 May 2025Hierarchical Retrieval with Evidence Curation for Open-Domain Financial
Question Answering on Standardized Documents
Jaeyoung Choe, Jihoon Kim, Woohwan Jung
Department of Applied Artificial Intelligence, Hanyang University
{cjy9100, skygl, whjung}@hanyang.ac.kr
Abstract
Retrieval-augmented generation (RAG) based
large language models (LLMs) are widely used
in finance for their excellent performance on
knowledge-intensive tasks. However, standard-
ized documents (e.g., SEC filing) share simi-
lar formats such as repetitive boilerplate texts,
and similar table structures. This similarity
forces traditional RAG methods to misiden-
tify near-duplicate text, leading to duplicate
retrieval that undermines accuracy and com-
pleteness. To address these issues, we pro-
pose the Hierarchical Retrieval with Evidence
Curation (HiREC) framework. Our approach
first performs hierarchical retrieval to reduce
confusion among similar texts. It first retrieve
related documents and then selects the most
relevant passages from the documents. The evi-
dence curation process removes irrelevant pas-
sages. When necessary, it automatically gener-
ates complementary queries to collect missing
information. To evaluate our approach, we con-
struct and release a Large-scale Open-domain
Financial (LOFin) question answering bench-
mark that includes 145,897 SEC documents
and 1,595 question-answer pairs. Our code and
data are available at https://github.com/
deep-over/LOFin-bench-HiREC .
1 Introduction
Retrieval-augmented generation (RAG) (Lewis
et al., 2020) with large language models (LLMs)
have significantly improved performance in
knowledge-intensive tasks. Due to its ability to
improve both factual accuracy and timeliness, ex-
tensive research (Yepes et al., 2024; Sarmah et al.,
2024) has investigated applying RAG in searching
financial information where accurate and up-to-
date information is crucial for decision-making.
Financial documents, such as annual reports (10-
K) in SEC filings, are highly structured and follow
standardized templates across companies and peri-
ods, often containing similar tables and repetitive
Figure 1: Comparison of a naive RAG approach and
HiREC .
narratives. As shown in Figure 1, 2023 10-K re-
ports from Amazon, Meta, and Walmart exhibit
nearly identical table structures with similar titles
and indicators, differing mainly in numerical val-
ues. Consequently, when asked, What is the dif-
ference in operating income ratio between Ama-
zon and Walmart in 2023? , a conventional RAG
system may struggle to distinguish among these
similar passages, retrieving irrelevant or redundant
information that leads to inaccurate answers.
To tackle these challenges from the standard-
ized format of financial documents, we propose
the HiREC (Hierarchical Retrieval and Evidence
Curation) framework. HiREC consists of two main
components: hierarchical retrieval and evidence
curation. The hierarchical retrieval first retrieves re-
lated documents and then selects the most relevant
passages from the documents, thereby reducing
confusion from near-duplicate text. As illustrated
in Figure 1, narrowing the candidate set to doc-
uments from Amazon and Walmart enables the
system to focus on highly relevant passages. How-
ever, the prevalence of comparative questions in

financial QA often leads to incomplete retrieval of
essential evidence. To address this, the evidence
curation stage filters out irrelevant passages and
generates complementary queries when necessary.
For example if only Amazon operating income is
retrieved then a complementary query is generated
to fetch Walmart operating income so that all nec-
essary evidence is gathered for an accurate answer.
To evaluate our approach, we assess QA perfor-
mance in an open-domain setting. Existing finan-
cial question-answering benchmarks (Islam et al.,
2023; Lai et al., 2024) rely on small-scale corpora
that include at most about 1,300 documents and
very limited test sets, which do not reflect realistic
financial scenarios. We propose LOFin (Large-
scale Open-domain Financial QA), a comprehen-
sive financial question-answering benchmark built
on a large-scale corpus containing approximately
145,000 SEC filings from companies in the S&P
500. LOFin includes 1,595 open-domain question-
answering test instances and addresses challenges
in standardized document retrieval, such as near-
duplicate tables and repetitive narratives, that are
not evident in smaller datasets. In addition, we
release the entire benchmark as open-source to sup-
port future research in the field.
Experimental results indicate that HiREC im-
proves performance by at least 13% compared to
existing RAG methods. In addition, our frame-
work consistently outperforms commercial llms
with web search engines such as SearchGPT (Ope-
nAI, 2025) and Perplexity (Perplexity, 2023).
Our contributions are threefold:
•We propose a hierarchical retrieval approach
that retrieves related documents and identifies
the most pertinent passages, thereby reducing
confusion caused by near-duplicate content in
standardized financial documents.
•We introduce an evidence curation process
that filters out irrelevant passages and gener-
ates complementary queries when necessary,
effectively supplementing missing informa-
tion for accurate financial QA.
•We present LOFin, a large-scale, realistic
financial QA benchmark that exposes chal-
lenges in standardized document retrieval, and
we release it as open-source.
2 Related Works
Financial QA and RAG Financial QA has ad-
vanced significantly, with recent benchmarks em-phasizing numeric reasoning and table understand-
ing. TAT-QA (Zhu et al., 2021) and FinQA (Chen
et al., 2021) provide single-page tabular con-
texts, while DocFinQA (Reddy et al., 2024) and
DocMath-Eval (Zhao et al., 2024) extend this to
multi-page settings. However, these benchmarks
remain closed-domain, limiting their applicabil-
ity for RAG systems. Open-domain benchmarks
have also been proposed. For instance, consider Fi-
nancebench (Islam et al., 2023) and SEC-QA (Lai
et al., 2024). However, these datasets are built on
small-scale document collections and suffer from
limited test set sizes or the lack of publicly avail-
able fixed test sets.
With the emergence of financial QA, research on
financial RAG has also been progressing. There are
graph-based methods (Barry et al., 2025; Sarmah
et al., 2024) tailored to the financial domain as well
as hybrid approaches (Wang, 2024). In addition,
studies on financial document chunking offer valu-
able insights (Yepes et al., 2024).
Retrieval augmented generation RAG has
evolved in diverse ways (Gao et al., 2023; Zhang
et al., 2025). Hierarchical retrieval was proposed
for cases where sections are clearly demarcated,
typically employing a two-step document-passage
process (Arivazhagan et al., 2023; Chen et al.,
2024). The difference from previous studies is that
while they segment documents into sections, we
segment at the level of individual documents. There
are also studies that utilize filtering to enhance the
quality of the contexts input into RAG (Zhuang
et al., 2024; Wang et al., 2024b). Unlike previ-
ous approaches that required training, we manage
quality using only an LLM. Iterative retrieval is
typically proposed for multi-hop QA. A standard
iterative method uses the context retrieved in the
first step as part of the query for subsequent it-
erations (Trivedi et al., 2022; Shao et al., 2023).
Self-RAG (Asai et al., 2023) also conducts itera-
tive retrieval that includes the generation process.
However, our approach does not utilize previously
retrieved context because we specifically need to
discover missing information.
3Large-scale Open-domain Financial QA
In this section, we present LOFin, a benchmark
that overcomes the limitations of current financial
QA datasets by expanding the retrieval corpus and
increasing open-domain QA pairs (see Table 1).

3.1 Large-scale Document Collection
To reflect a real-world scenario where retrieval and
QA must be performed over a large volume of
documents, we collected a comprehensive set of
SEC filings. Specifically, we gathered 10-K, 10-Q,
and 8-K filings from the SEC EDGAR1system,
covering S&P 500 companies from October 2001
to April 2025.
We converted the HTML documents to PDF us-
ing the wkhtmltopdf2library. Following the ap-
proach of Islam et al., 2023, we used the PyMuPDF
library3to extract text at the page level from these
PDFs, excluding reports that lacked proper page
separation. The final corpus consists of 145,897
reports from 516 companies.
3.2 Open-domain QA Pair Construction
In this section, we detail our process for construct-
ing open-domain QA pairs by leveraging three es-
tablished financial QA benchmarks: FinQA (Chen
et al., 2021), Financebench (Islam et al., 2023), and
SEC-QA (Lai et al., 2024).
We begin by converting the closed-setting ques-
tions from FinQA into an open-domain format.
First, we exclude any test questions for which ev-
idence documents were not collected due to page
separation or collection period issues (35 out of
1147). Next, we transform the remaining questions
using GPT-4o by appending relevant period and
company information. For example, the question
what are the total operating expenses for 2016? is
transformed into Could you provide the total op-
erating expenses reported by Lockheed Martin for
the year 2016? , with Lockheed Martin explicitly
added to enhance context. The conversion prompt
used is provided in Appendix B.2.
Subsequently, we identify candidate evidence
pages using a two-step process. First, we com-
pute BM25 similarity scores between the FinQA
gold table context and the content of each page in
the candidate document, explicitly considering the
distinct numerical values in the table. Then, we
employ an NLI model4to measure the semantic
similarity between the FinQA gold context (exclud-
ing table content) and each page. If both methods
select the same top candidate page we accept it
as evidence and verify its correctness. Otherwise
1https://www.sec.gov/search-filings
2https://wkhtmltopdf.org/
3https://pymupdf.readthedocs.io/
4https://huggingface.co/sentence-transformers/
nli-mpnet-base-v2Benchmark Open Multi-Doc # QAs # Docs
TAT-QA 1,669 -
FinQA 1,147 -
DocFinQA 922 -
Financebench ✓ 150 368
SEC-QA ✓ ✓ N/A 1,315
LOFin ✓ ✓ 1,595 145,897
Table 1: Comparison of financial QA benchmarks. “#
QAs” is the test set size, and “# Docs” is the number
of documents in the retrieval corpus. SEC QA does
not have a fixed QA count, as it provides a question
generation framework.
we manually annotate the correct page to ensure
accurate evidence labeling (see Appendix B.3 for
further details).
Financebench is designed for an open-domain
setting and its questions are adopted without modi-
fication. In contrast both FinQA and Financebench
mainly include single-document questions which
limits their ability to evaluate multi-document re-
trieval and reasoning. To address this limitation,
we adopt multi-document question templates from
SEC-QA. We select the four that are designed for
multi-document scenarios. We then manually craft
the associated questions and annotate the answers
along with the corresponding evidence pages. This
process enhances our open-domain QA pairs with
challenges that require multi-document and multi-
hop reasoning.
Following the ACL ARR review, we finalized
the LOFin benchmark with a total of 1,595 QA
pairs (initially 1,389), reflecting the addition of
205 newly created QA pairs based on recent SEC
filings. We refer to the initial version as LOFin-
1.4k and the expanded version as LOFin-1.6k to
clearly distinguish between the two. These newly
added questions follow the same annotation pro-
tocol and are designed to promote more complex
reasoning across multiple documents. For details
of the expanded LOFin-1.6k benchmark, see Ap-
pendix A; for the SEC-QA templates used in its
construction, refer to Appendix B.4.
4 Hierarchical Retrieval with Evidence
Curation (HiREC) Framework
In this section, we introduce the HiREC framework.
Figure 2 shows the overall framework and process.
The framework comprises two main components:
hierarchical retrieval and evidence curation. Dur-
ing the hierarchical retrieval stage a hierarchical
approach retrieves passages Prthat are relevant to

Figure 2: Overview of hierarchical retrieval with evidence curation framework
the question q. In the evidence curation process the
retrieved passages are filtered to retain only those
directly pertinent and then evaluated to determine
whether they provide sufficient information to an-
swer the question. If the information is insufficient
a complementary question qcis generated to reini-
tiate the retrieval process (complementary pass).
Otherwise if the evidence is sufficient the filtered
passage set Pfis forwarded to the Answer Genera-
tor to produce the final answer (main pass). When
the maximum iteration imaxis reached evidence
curation halts and the passages retrieved using qc
are merged with the previously filtered passages to
generate the answer. The pseudocode of HiREC
is described in Algorithm 1. For the LLM-based
components of our framework, we use instruction-
style prompts tailored to each module’s objective.
Appendix D provides the full prompt templates and
design process.
4.1 Hierarchical Retrieval
Standardized documents use uniform templates
with repetitive structures and similar content, which
makes retrieving distinct and relevant passages
challenging. We address this issue using a hier-
archical approach. First we retrieve documents
relevant to the question (4.1.1) to narrow the search
space, and then we select pertinent passages within
those documents (4.1.2).
4.1.1 Document Retriever
Document indexing. Documents contain exten-
sive context, and their standardized format makes it
difficult to capture all important details in a single
vector. To address this, we extract and index key
distinguishing information when retrieving docu-
ments. In financial reports, the cover page provides
essential details such as the company name, report
type, and fiscal period. For each document d∈ D,
we generate a cover page summary d′using an
LLM (the prompt is detailed in Appendix D.1), pre-
compute its embedding with a bi-encoder (WangAlgorithm 1 HiREC framework
Require: A question q, a corpus D, maximum number of
iterations imax
1:Pf← ∅ // Initialization
2:Pr←HIERARCHICAL RETRIEVAL (q,D) // Sec 4.1
3:fori= 1...imax do
4: (Pf,qc,y)←EVIDENCE CURATION (q,Pr)// Sec 4.2
5: ifyisAnswerable then
6: return ANSWER GENERATION (q,Pf)// Sec 4.3
7: end if
8:Pr←HIERARCHICAL RETRIEVAL (qc,D)// Sec 4.1
9:Pr← P r∪ Pf
10:end for
11:return ANSWER GENERATION (q,Pr) // Sec 4.3
et al., 2024a), and index the resulting vector as
vd=ED(d′)in the document store.
Document retrieval process. Our document re-
trieval process consists of three stages. First, given
a question q, we use an LLM to convert it into a
refined query q′. This conversion reduces issues
caused by extraneous financial terms (e.g., tickers
or service names like Google or Facebook) that
may hinder effective retrieval. (Details of the trans-
formation prompt are provided in Appendix D.2.)
Second, we perform dense retrieval. We com-
pute the vector representation of q′using the same
bi-encoder employed during document indexing,
yielding vq′=ED(q′). The relevance between
q′and each document dis then determined via
sD
q′,d=v⊤
q′vd(Karpukhin et al., 2020); based on
these scores, we retrieve k′
Dcandidate documents
Dcand. Finally, we rerank these candidates using
a cross-encoder (He et al., 2021) by computing
CrossEncoderD(q′, d)and select the top kDdoc-
uments Dr. This multi-stage process ultimately
produces the final set of documents Drthat are
most relevant to the original question q.
4.1.2 Passage Retriever
Within the final set of retrieved documents Dr,
the passage retriever evaluates each passage p
by computing a score using a cross-encoder,
CrossEncoderP(q, p). It then selects the top kP
passages to form the final set Pr. By reducing the

number of passages processed by the cross-encoder,
we enable real-time computation while still taking
advantage of its superior ability to capture inter-
sentence relationships compared to a bi-encoder.
However, passage retrievers that are pretrained on
general text typically have difficulty handling fi-
nancial tables. For instance, when retrieving tables,
attributes such as titles, periods, and indicators are
more important than the numerical values, yet these
cues are not well captured. To address this, we fine-
tuned the model on table data using FinQA training
set where tables serve as evidence. Specifically, for
each question qand its associated evidence docu-
mentd, we denote the tables on the evidence page
as the evidence passage set X. For each evidence
passage p∈ X, we sample nnegnegative passages,
where negative passages are defined as tables that
appear on pages other than the evidence page. The
cross-encoder is then trained with a binary cross-
entropy loss (Nogueira and Cho, 2019) defined as
L=X
(q,p)∈X"
−log 
CrossEncoderP(q, p)
−X
p′∈P−log
1−CrossEncoderP(q, p′)#
,
where the cross-encoder applies an internal sigmoid
to produce scores in [0, 1].
4.2 Evidence Curation
Financial questions often involve comparisons be-
tween different time periods or companies. Even
when the retrieval process selects relevant passages,
some critical information may be missing. Further-
more, retrieved passages can contain irrelevant data
that hinders overall performance (Liu et al., 2024;
Xu et al., 2024). To overcome these issues, we
introduce an evidence curation process that filters
out incorrect data and fills information gaps by ini-
tiating additional retrieval when necessary. Our
process comprises three modules: a passage filter
that removes irrelevant passages (Section 4.2.1),
an answerability checker that assesses evidence
sufficiency (Section 4.2.2), and a complementary
question generator that formulates a supplementary
query if necessary (Section 4.2.3). We use an LLM
to perform all three tasks in a single response (see
Appendix D.3 for the prompt).
4.2.1 Passage Filter
The passage filter removes passages from the re-
trieved set Prthat are not relevant to the question,
yielding a filtered passage set Pfcontaining at mostk′
Ppassages. The filter considers both newly re-
trieved passages and those previously identified as
relevant from earlier iterations. This step is crucial,
as the inclusion of noisy, irrelevant passages can
lead the LLM to generate inaccurate responses.
4.2.2 Answerability Checker
The answerability checker evaluates whether the
filtered passages Pfprovide sufficient evidence to
answer the question. If they are deemed adequate,
Pfand the question are forwarded to the answer
generation stage; otherwise, the lack of sufficient
information triggers a complementary iteration to
retrieve additional data.
4.2.3 Complementary Question Generator
The complementary question generator examines
the filtered passages Pfto identify gaps in the evi-
dence needed to answer the question. It then gen-
erates a supplementary query qc, which is used as
input for the next retrieval.
4.3 Answer Generation
In the answer generation stage, the relevant pas-
sages and the original question serve as inputs to
a reasoning process that derives the final answer.
For questions requiring numerical calculations, a
Program-of-Thought (PoT) (Chen et al., 2022) rea-
soning method is employed, while for text-based
inferential questions, a Chain-of-Thought (CoT)
(Wei et al., 2022) approach is applied. This dual
strategy is particularly effective for financial docu-
ments, which are rich in numerical data and tables,
ensuring comprehensive and accurate reasoning.
See Appendix D.4 for prompt details, adapted from
DocMath-Eval (Zhao et al., 2024).
5 Experiments
5.1 Experimental Settings
Dataset. We evaluate the open-domain QA meth-
ods on the LOFin benchmark proposed in Section 3.
All main experiments in this section are conducted
on LOFin-1.4k , which contains 1,389 QA pairs.
Results on the updated benchmark LOFin-1.6k are
reported separately in Appendix A.
For the purpose of the main experimental anal-
ysis, we categorize the question-answer pairs into
three groups based on their format and context:
•Numeric (Table): The answer is a number
from a table or can be calculated from num-
bers in tables.

Dataset Numeric (Table) Numeric (Text) Textual Average
MethodPage
RecallAnswer
AccPage
RecallAnswer
AccPage
RecallAnswer
AccPage
RecallAnswer
Acck
GPT-4o (Zero-shot) - 3.82 - 2.93 - 35.00 - 13.92 -
Perplexity (Perplexity, 2023) - 2.51 - 5.13 - 24.00 - 10.55 -
Self-RAG (Asai et al., 2023) 19.96 1.86 24.18 4.03 12.75 17.00 18.96 7.63 10.0
RQ-RAG (Chan et al., 2024) 18.61 1.97 19.05 2.56 17.96 20.50 18.54 8.34 36.0
IRCoT (Trivedi et al., 2022) ♢ 28.17 19.10 34.62 27.84 12.67 20.00 25.15 22.31 20.0
HybridSearch (Wang, 2024) ♢ 26.75 19.10 32.05 30.77 14.37 27.00 24.39 25.62 10.0
HHR (Arivazhagan et al., 2023) ♢ 37.67 26.53 40.29 32.97 21.98 26.50 33.31 28.67 10.0
Dense (Karpukhin et al., 2020) ♢ 37.69 23.69 40.48 32.97 26.18 31.00 34.78 29.22 10.0
HiREC (Ours) ♢ 50.17 37.23 53.48 48.35 32.39 41.50 45.35 42.36 3.7
Gold evidence ♢ 100 64.96 100 69.23 100 63.50 100 65.90 1.3
Table 2: Main evaluation results on LOFin-1.4k . Methods marked with ♢use GPT-4o for generation. Here, k
denotes the average number of input passages used during generation, and Gold evidence shows only the correct
page. Bold indicates the highest performance.
•Numeric (Text): The answer is a number de-
rived by extracting and combining numerical
information from text.
•Textual: The answer is a textual explanation.
Examples and statistics for these categories are
in Table 12.
Implementation details. Our system leverages
several pre-trained and fine-tuned models across
its components. For the passage retriever, we fine-
tune a DeBERTa-v3 model (He et al., 2021) with
nneg= 8 for negative sampling, a batch size of
128, 3 epochs, and a learning rate of 2×10−7on
a single GeForce RTX 4090 GPU. For document
retrieval, we employ the E5 model (Wang et al.,
2024a) as the bi-encoder and use DeBERTa-v3 as
the reranker. The answer generator is powered by
OpenAI’s GPT-4o, while other LLM-based tasks
(query transformation, document summarization,
and evidence curation) are handled by Qwen-2.5-
7B-Instruct (Yang et al., 2024).
Our framework runs for a maximum of imax= 3
iterations. The document retriever retrieves kD=
5documents, and the passage retriever extracts
kP= 5passages, with the passage filter capping
the output at k′
P= 10 . Additional hyperparameters
are provided in Appendix C.1.
Baseline methods. We compare our proposed
method with several state-of-the-art RAG ap-
proaches, including RQ-RAG (Chan et al., 2024)
and Self-RAG (Asai et al., 2023). To ensure a fair
comparison with our approach—which employs
GPT-4o as the answer generator—we use the same
answer generator for the latest retrieval algorithms:
Dense, HybridSearch (Wang, 2024), HHR (Ari-vazhagan et al., 2023), and IRCoT (Trivedi et al.,
2022). In addition, we evaluate against Perplex-
ity (Perplexity, 2023), a commercial LLM service
that combines web search with LLMs. RQ-RAG,
Self-RAG, and IRCoT employ iterative LLM-based
retrieval, while Dense serves as a strong baseline,
it employs OpenAI’s text-embedding-3-small
as its encoder for DPR, with results reranked by
DeBERTa-v3. All methods construct passages by
concatenating the title and content during retrieval.
Detailed configurations for each baseline are pro-
vided in Appendix C.2.
Metrics. We evaluate numeric answers for accu-
racy by considering rounding and truncation, while
textual answers are evaluated using GPT-4o and
FAMMA prompts (Xue et al., 2024) (see Appendix
C.3). We measure retrieval performance at the page
level, using recall and precision against ground-
truth evidence pages as metrics. In particular, since
chunk locations or units can vary for Page recall,
we standardize them at the page level to ensure
consistent performance measurement.
5.2 Main Result
Table 2 shows the retrieval performance (page re-
call) and final answer accuracy for HiREC and
the baselines. Our approach outperforms all base-
lines, achieving at least 10% higher page recall and
13% higher answer accuracy than the second-best
model, Dense. The result validates the effective-
ness of our method in retrieval for standardized
documents. Furthermore, HiREC retrieves an av-
erage of only 3.7 passages, demonstrating its effi-
ciency in selecting high-quality evidence through
evidence curation.

MethodPage
PrecisionPage
RecallAnsswer
Accuracy
HiREC 21.79 45.35 42.36
w/o HR 14.75 34.16 32.76
w/o EC 4.70 41.41 36.70
w/o Fine-tuning 21.07 42.77 40.13
w/o Filter 8.43 50.19 42.08
Table 3: Ablation study
Figure 3: Comparison of company, document, page
error rates for HiREC and baselines.
Notably, in the textual category, the answer accu-
racies are higher than the page recalls for all meth-
ods. This suggests that LLMs can often answer
text-based questions correctly even when retrieval
is incomplete. Numeric questions require more
precise reasoning and the table category remains
especially challenging.
5.3 Analysis
Ablation Study Table 3 shows page precision,
page recall and answer accuracy for HiREC when
each component is removed. Hierarchical retrieval
(HR) and evidence curation (EC) denote the two
components. The setting w/o HR corresponds
to the outcome of applying the Dense method in
conjunction with EC, while the setting w/o EC
represents the initial retrieval performance of HR
(kP= 10 ).
The w/o Fine-tuning setting uses an unfine-tuned
reranker to address fairness concerns related to
table-specific prior knowledge. Despite a slight
drop in performance, HiREC still outperforms the
Dense baseline by over 10% in accuracy. Even
when Dense is paired with a fine-tuned reranker,
it only reaches an average AnswerAcc of 30.55%,
maintaining a similar performance gap.
HR is critical for enhancing retrieval accuracy be-
cause its absence produces the lowest performance.
In addition, the initial search performance of HR
exceeds that of the dense method combined with
EC. The w/o EC setting demonstrates the effective-
ness of both passage filter and the complementary
Figure 4: Recall, precision, and passages per query by
iteration. EC stands for Evidence curation.
Figure 5: Precision-recall curve (recall on X-axis, preci-
sion on Y-axis)
question generator. HR supported by EC delivers
enhanced precision, recall and answer accuracy.
Furthermore, results from the w/o Filter setting
reveal the advantages of the complement and fil-
tering. Although the complementary component
attains the highest recall score in the absence of
filtering, its accuracy is lower than that of HiREC
with filtering. This outcome is attributed to the
inclusion of incorrect information that causes con-
flicts when filtering is not applied (Xu et al., 2024).
Error type analysis. Figure 3 presents the error
count for cases in which the company indicated
in the document is incorrect for the retrieved pas-
sage of each method. A company error is defined
as an instance in which both the passage and the
document contain incorrect company information,
and the results are reported based solely on the top-
1 retrieved passage. HiREC achieves the fewest
errors by correctly identifying companies during
document retrieval, which in turn ensures accurate
passage retrieval.
Evidence curation by iteration. Figure 4 shows
that iterative evidence curation (EC) enhances page
recall and precision while reducing passages per
query compared to the initial hierarchical retrieval
(w/o EC). As iterations progress, retrieval perfor-
mance steadily improves and efficiency increases,
demonstrating evidence curation effectiveness.

MethodRetrieval Generation
Tokens Cost ($) Tokens Cost ($)
Dense - -2,666
/ 1779.3
/ 2.5
IRCoT9,610
/ 1,37233.4
/ 19.13,475
/ 17712.1
/ 2.5
HiREC4,291
/ 313∗14.9
/∗4.41,052
/ 1593.7
/ 2.2
Table 4: Cost-efficiency comparison. Each cell shows
the values for the input/output. The asterisk (∗) indicates
that, although a much smaller open-source LLM was
used, the cost is computed using GPT-4o’s pricing.
Precision-recall. Figure 5 presents the precision-
recall curve, where kvaries from 1 to 50 for base-
line methods. As expected, increasing kimproves
recall but lowers precision due to the retrieval of
less relevant passages. HiREC achieves both higher
precision and recall, consistently exceeding the
maximum values observed across all kin the base-
line range. This highlights ability of HiREC to re-
trieve relevant information more effectively while
maintaining accuracy.
Cost-efficiency. Table 4 presents the average in-
put/output token counts and total API cost in-
curred during the retrieval and generation processes.
HiREC achieves the highest performance while us-
ing significantly fewer tokens and lower cost than
the baselines by filtering out irrelevant passages be-
fore the answer generation stage. Moreover, com-
pared to IRCoT in retrieval, HiREC uses far fewer
tokens, demonstrating that its filtering and comple-
mentary question generation enable efficient itera-
tion. Finally, our results show that even a relatively
small LLM can effectively perform curation with-
out incurring high costs.
5.4 Analysis of Performance Across Various
LLM Generators
We thoroughly analyze the effectiveness of the
HiREC framework across diverse LLM generator
models. For this analysis, additional evaluations
were conducted using open-source models such as
DeepSeek-R1-Distill-Qwen-14B(Guo et al., 2025)
and Qwen-2.5-7B-Instruct(Yang et al., 2024) as
generators, in place of GPT-4o.
As shown in Table 5, HiREC consistently
demonstrates superior QA performance across var-
ious LLM generators, proving its robustness. No-
tably, even when employing smaller open-source
models, HiREC configurations outperform the
Dense baseline. HiREC +Deepseek-14B achievedMethodNumeric
(Table)Numeric
(Text)Textual Average
Dense
+ Qwen-2.5-7B 19.76 22.34 29.52 23.87
+ Deepseek-14B 25.44 31.87 35.00 30.77
+ GPT-4o 23.69 26.18 31.00 29.22
HiREC
+ Qwen-2.5-7B 27.62 33.33 36.00 32.32
+ Deepseek-14B 34.39 41.39 40.53 38.76
+ GPT-4o 37.23 48.35 41.50 42.36
Table 5: Performance comparison across various LLM
generators. Underlined values denote HiREC configura-
tions utilizing smaller generators that surpass Dense +
GPT-4o performance.
Method Financebench FinQA SEC-QA
Page
RecallAnswer
AccPage
RecallAnswer
AccPage
RecallAnswer
Acc
Self-RAG 9.00 13.33 22.93 3.24 4.59 4.72
RQ-RAG 13.67 18.67 20.32 2.88 9.38 4.72
IRCoT 9.33 20.00 32.24 22.66 4.20 7.09
HybridSearch 10.00 29.33 29.95 23.29 10.42 7.87
HHR 16.44 34.00 41.50 29.50 10.13 5.51
Dense 26.11 33.33 40.38 27.61 15.70 9.45
HiREC 40.00 50.00 52.49 41.61 19.97 14.17
Table 6: Performance results by data source.
over 9% higher average answer accuracy compared
to Dense+GPT-4o. Another key implication is that
integrating sLLMs in the retrieval stage proves to
be an effective strategy for reducing overall infer-
ence costs while maintaining strong performance.
5.5 Performance results by data source
Our analysis of various data sources aims to evalu-
ate benchmark data leakage risk and demonstrate
the robustness of the HiREC framework.
As shown in Table 6, HiREC consistently
demonstrates superior performance across all data
sources. Notably, all methods, including HiREC ,
exhibit relatively lower performance on the SEC-
QA subset, as multi-document QA tasks require
the synthesis of information from multiple sources
for accurate answers. These results suggest the
LOFin benchmark is more challenging due to multi-
document scenarios and mitigates leakage risk by
relying on retrieved evidence.
The final results for the LOFin benchmark, in-
cluding the expanded SEC-QA subset, can be
found in the Appendix.
5.6 Comparison of Commercial LLMs with
Web Search
Commercial systems such as SearchGPT and Per-
plexity combine LLMs with web search to answer

Step Content
Question What is Adobe’s year-over-year change in unadjusted operating income from FY2015 toFY2016 (in units of percents
and round to one decimal place)? Give a solution to the question by using the income statement.
Initial
Retrieval✓ADBE_2015_10K (p.59): 3,112,300 3,045,960 Operating income 903,095 412,685 422,723 Non-operating ...
✗ADBE_2015_10K (p.39): Net income of $629.6 million increased by $361.2 million, or 135%, during fiscal 2015 ...
✗ADBE_2015_10K (p.45): Amortization of purchased intangibles 68.7 ... Total operating expenses $3,148.1 ...
Evidence
CurationRelevant passages: ADBE_2015_10K (p.59) Answerable: False
Explanation: The context does not provide operating income figures for both FY2015 andFY2016 . We only have
operating income for FY2015 and some intermediate values, but no direct comparison between FY2015 andFY2016 .
Complementary question: What is Adobe’s operating income for FY2016 ?
Complementary
Retrieval✓ADBE_2016_10K (p.61): 3,148,099 3,112,300 Operating income 1,493,602 903,095 412,685 Non-operating ...
✗ADBE_2016_10K (p.35): ITEM 6. SELECTED FINANCIAL DATA ... Net income $1,168,782 $ 629,551 ...
✗ADBE_2017_10K (p.57): CONSOLIDATED STATEMENTS OF INCOME ... Operating income 2,168,095 ...
Evidence
CurationRelevant passages: ADBE_2015_10K (p.59), ADBE_2016_10K (p.61) Answerable: True
Generation Ground Truth Answer : 46% Generated Answer : 46 Correct: ✓
Table 7: Case study illustration of HiREC framework effectiveness
MethodNumeric
(Table)Numeric
(Text)Textual Average
Perplexity 2.5 5.0 37.5 15.0
SearchGPT 15.0 32.5 35.0 27.5
HiREC 30.0 65.0 45.0 46.7
Table 8: Answer accuracy for Perplexity, SearchGPT,
and HiREC on 40 samples per category.
questions using financial data. Table 8 compares
these systems with our approach where Perplexity
uses the llama-3.1-sonar-large-128k-online model
and SearchGPT uses GPT-4o. In our evaluation
based on 40 questions per category, HiREC con-
sistently outperforms these baselines especially on
numeric questions, indicating that although com-
mercial systems effectively retrieve relevant docu-
ments they often miss the precise numerical details
necessary for accurate computation.
5.7 Case Study
We analyze how evidence curation filters out unnec-
essary information and effectively retrieves missing
details, and how this impacts the final results. Ta-
ble 7 presents the retrieval and evidence curation
outcomes over iterations for a given question. The
initial pass of hierarchical retrieval retrieves pas-
sages containing operating income for Adobe for
FY2015, but the question remains unanswerable
due to missing FY2016 data. The answerability
checker detects this gap and triggers the comple-
mentary question generator, leading to a comple-
mentary pass that retrieves the missing FY2016
operating income. With complete evidence, theanswer generator successfully computes the 46%
year-over-year change, highlighting the effective-
ness of iterative refinement in HiREC.
6 Conclusion
We introduced HiREC, a retrieval-augmented
framework for question answering over standard-
ized financial documents. Our hierarchical re-
trieval reduces confusion from repeated boilerplate
content, while evidence curation filters irrelevant
passages and recovers missing information. To
evaluate under realistic open-domain conditions,
we constructed LOFin, a large-scale benchmark
with 1,595 QA pairs across 145,000 SEC filings.
The dataset includes multi-document and multi-
hop questions that go beyond prior financial QA
benchmarks. Experiments show that HiREC con-
sistently outperforms state-of-the-art baselines in
both retrieval quality and answer accuracy, while
maintaining cost efficiency through selective pas-
sage use. Overall, our findings suggest that HiREC
provides a scalable and effective solution for open-
domain QA over complex, standardized financial
documents.
Limitations
In our study, we use LLMs for query transforma-
tion, evidence curation, and answer generation,
making our approach dependent on the perfor-
mance of the LLMs. We utilize a relatively small
LLM, Qwen 2.5 7B (Yang et al., 2024), compared
to commercial models.

The benchmarks we employ, FinQA and Fi-
nanceBench, are publicly available datasets. There-
fore, it is possible that pre-trained large language
models have already been exposed to these datasets
during their training. In Table 2, GPT-4o (Zero-
shot) achieves the second-highest textual perfor-
mance.
Acknowledgements
This work was supported by the National R&D
Program for Cancer Control through the National
Cancer Center (NCC), funded by the Ministry
of Health & Welfare, Republic of Korea (RS-
2025_02264000), and by the Institute of Informa-
tion & communications Technology Planning &
Evaluation (IITP) grant funded by the Korea gov-
ernment (MSIT) (No. RS-2023-00261068, Devel-
opment of Lightweight Multimodal Anti-Phishing
Models and Split-Learning Techniques for Privacy-
Preserving Anti-Phishing).
References
Manoj Ghuhan Arivazhagan, Lan Liu, Peng Qi, Xinchi
Chen, William Yang Wang, and Zhiheng Huang.
2023. Hybrid hierarchical retrieval for open-domain
question answering. In Findings of the Associ-
ation for Computational Linguistics: ACL 2023 ,
pages 10680–10689, Toronto, Canada. Association
for Computational Linguistics.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511 .
Mariam Barry, Gaëtan Caillaut, Pierre Halftermeyer, Ra-
heel Qader, Mehdi Mouayad, Fabrice Le Deit, Dim-
itri Cariolaro, and Joseph Gesnouin. 2025. Graphrag:
Leveraging graph-based efficiency to minimize hal-
lucinations in llm-driven rag for finance data. In
Proceedings of the Workshop on Generative AI and
Knowledge Graphs (GenAIK) , pages 54–65.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval augmented genera-
tion. arXiv preprint arXiv:2404.00610 .
Wenhu Chen, Xueguang Ma, Xinyi Wang, and
William W Cohen. 2022. Program of thoughts
prompting: Disentangling computation from reason-
ing for numerical reasoning tasks. arXiv preprint
arXiv:2211.12588 .
Xinyue Chen, Pengyu Gao, Jiangjiang Song, and Xi-
aoyang Tan. 2024. Hiqa: A hierarchical contextual
augmentation rag for massive documents qa. arXiv
preprint arXiv:2402.01767 .Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena
Shah, Iana Borova, Dylan Langdon, Reema Moussa,
Matt Beane, Ting-Hao Huang, Bryan Routledge, et al.
2021. Finqa: A dataset of numerical reasoning over
financial data. arXiv preprint arXiv:2109.00122 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: In-
centivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 .
Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021.
Debertav3: Improving deberta using electra-style pre-
training with gradient-disentangled embedding shar-
ing. arXiv preprint arXiv:2111.09543 .
Pranab Islam, Anand Kannappan, Douwe Kiela, Re-
becca Qian, Nino Scherrer, and Bertie Vidgen. 2023.
Financebench: A new benchmark for financial ques-
tion answering. arXiv preprint arXiv:2311.11944 .
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. arXiv preprint
arXiv:2004.04906 .
Viet Dac Lai, Michael Krumdick, Charles Lovering,
Varshini Reddy, Craig Schmidt, and Chris Tanner.
2024. Sec-qa: A systematic evaluation corpus for
financial qa. arXiv preprint arXiv:2406.14394 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with bert. arXiv preprint
arXiv:1901.04085 .
OpenAI. 2025. searchgpt: A comprehensive guide.
https://www.openai.com/searchgpt [Accessed:
2025-02-10].
Perplexity. 2023. Perplexity.ai. https://www.
perplexity.ai/ . [Large language model].
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain. arXiv
preprint arXiv:2408.10343 .

Varshini Reddy, Rik Koncel-Kedziorski, Viet Dac Lai,
Michael Krumdick, Charles Lovering, and Chris Tan-
ner. 2024. Docfinqa: A long-context financial rea-
soning dataset. arXiv preprint arXiv:2401.06915 .
Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Ro-
han Rao, Sunil Patel, and Stefano Pasquali. 2024.
Hybridrag: Integrating knowledge graphs and vec-
tor retrieval augmented generation for efficient infor-
mation extraction. In Proceedings of the 5th ACM
International Conference on AI in Finance , pages
608–616.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. arXiv preprint
arXiv:2305.15294 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Jing Wang. 2024. Financerag with hybrid search and
reranking.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Multilin-
gual e5 text embeddings: A technical report. arXiv
preprint arXiv:2402.05672 .
Yuhao Wang, Ruiyang Ren, Junyi Li, Wayne Xin
Zhao, Jing Liu, and Ji-Rong Wen. 2024b. Rear: A
relevance-aware retrieval-augmented framework for
open-domain question answering. arXiv preprint
arXiv:2402.17497 .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for llms: A survey. arXiv
preprint arXiv:2403.08319 .
Siqiao Xue, Tingting Chen, Fan Zhou, Qingyang Dai,
Zhixuan Chu, and Hongyuan Mei. 2024. Famma:
A benchmark for financial domain multilingual
multimodal question answering. arXiv preprint
arXiv:2410.04526 .
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 tech-
nical report. arXiv preprint arXiv:2412.15115 .
Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebas-
tian Laverde, and Renyu Li. 2024. Financial report
chunking for effective retrieval augmented genera-
tion. arXiv preprint arXiv:2402.05131 .Qinggang Zhang, Shengyuan Chen, Yuanchen Bei,
Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong,
Hao Chen, Yi Chang, and Xiao Huang. 2025. A
survey of graph retrieval-augmented generation for
customized large language models. arXiv preprint
arXiv:2501.13958 .
Yilun Zhao, Yitao Long, Hongjun Liu, Ryo Kamoi,
Linyong Nan, Lyuhao Chen, Yixin Liu, Xian-
gru Tang, Rui Zhang, and Arman Cohan. 2024.
DocMath-eval: Evaluating math reasoning capabili-
ties of LLMs in understanding long and specialized
documents. In Proceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguis-
tics (Volume 1: Long Papers) , pages 16103–16120,
Bangkok, Thailand. Association for Computational
Linguistics.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and
Tat-Seng Chua. 2021. Tat-qa: A question answering
benchmark on a hybrid of tabular and textual content
in finance. arXiv preprint arXiv:2105.07624 .
Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai
Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan
Rajmohan, Dongmei Zhang, and Qi Zhang. 2024.
EfficientRAG: Efficient retriever for multi-hop ques-
tion answering. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing , pages 3392–3411, Miami, Florida, USA.
Association for Computational Linguistics.

Appendix
A Experiments on the Expanded
LOFin-1.6k Benchmark
A.1 Motivation and Dataset Extension
The initial version of the LOFin-1.4k benchmark
was heavily biased toward single-document QA,
particularly from FinQA, which limited its cover-
age of complex financial analysis scenarios. To
address this limitation and incorporate reviewer
feedback, we expanded the SEC-QA subset from
127 to 333 question-answer pairs by adding 206
new examples. All newly added QA pairs were
constructed from recent SEC filings collected up
to April 2025, and are designed to require multi-
document and multi-page reasoning. The questions
include inter-company comparisons, trend analysis,
and temporal reasoning. All examples were manu-
ally created, which prevents the possibility of data
leakage.
Dataset Source FinQA SEC-QA Financebench Total
# QAs 1,112 333 150 1,595
Ratio (%) 69.72 20.88 9.40 100
Table 9: Distribution of expanded LOFin-1.6k QA pairs
by dataset source.
A.1.1 Experiments
This section reports the results on the expanded
benchmark LOFin-1.6k , following the same set-
tings as in Section 5. The evaluation compares our
proposed model HiREC against the second-best
baseline, Dense. Table 10 presents the main results
by answer type. HiREC consistently outperforms
Dense across all categories. Table 11 shows per-
formance by data source. Although SEC-QA is a
more challenging task, HiREC still demonstrates
superior results compared to the baseline. The cat-
egory distribution by answer type is summarized
in Table 13, and the dataset composition by source
is reported in Table 9.
A.1.2 Discussion and Conclusion
The experimental results on the extended SEC-QA
subset in expanded LOFin-1.6k validate the robust-
ness and effectiveness of HiREC under realistic
multi-document and multi-page QA settings. With
this expansion, the LOFin-1.6k benchmark now
represents a significantly more challenging and
realistic task compared to existing financial QA
datasets. Overall, these results demonstrate thatHiREC effectively performs evidence-based rea-
soning and provide strong empirical support that
addresses reviewer concerns regarding dataset di-
versity, integrity, and model generalization.
B Supplementary Details on the
LOFin Benchmark
B.1 Categorization Details
Tables 12 and 13 summarize the distribution of QA
pairs in LOFin by answer type before and after the
SEC-QA expansion. The original dataset is heav-
ily skewed toward numeric-table questions, while
the expanded version introduces a larger portion of
textual and multi-hop reasoning questions, illustrat-
ing the increased complexity and diversity of the
benchmark.
B.2 FinQA Open-Domain Conversion Prompt
You are a financial AI Assistant. The following financial
questions are provided without including the company
names and document period. Rewrite the questions to
include the given information.
- Maintain the original meaning of the question while
allowing for varied expressions that enable accurate
and open-ended responses without altering the factual
content.
- The question must include the company name and the
report year.
- Ticker is used as information about the company name
and should not be included in the question.
- While rephrasing the question, integrate additional
information naturally into the sentence without using
simple connectors like ’In’, ’for’, or ’according to’.
Document information:
- company ticker: {ticker}
- document_period: {period}
Question: {question}
Output format is ##new_question:
Table 14: This prompt template is used to convert closed-
domain FinQA questions into an open-domain format
by seamlessly integrating company names and report
years into the questions, while preserving their original
meaning.
B.3 FinQA Gold Page Selection
Our evidence page annotation for FinQA relies
on a dual-method approach. Using FinQA’s qid,
we first identify candidate documents and discard

Model Numeric Table Numeric Text Textual Average
Precision Recall Accuracy Precision Recall Accuracy Precision Recall Accuracy Precision Recall Accuracy
Dense 3.47 33.88 22.92 3.77 37.73 31.50 2.90 16.18 23.93 3.38 29.27 26.12
HiREC 25.63 48.76 35.57 21.64 52.75 44.32 13.54 23.59 26.95 20.27 41.70 35.61
Table 10: Performance by answer type on the expanded LOFin-1.6k benchmark.
Model FinanceBench FinQA SEC-QA Average
Precision Recall Accuracy Precision Recall Accuracy Precision Recall Accuracy Precision Recall Accuracy
Dense 1.80 16.00 36.00 4.00 37.28 26.44 2.88 12.66 13.51 2.81 21.98 25.00
HiREC 15.67 37.33 47.33 25.44 51.18 39.44 12.74 18.67 14.85 17.95 35.73 33.63
Table 11: Performance by data source on the expanded LOFin-1.6k benchmark.
CategoryNumeric
(Text)Numeric
(Table)Textual Total
# QAs 273 916 200 1389
Ratio (%) 19.65 65.95 14.40 100
Table 12: Distribution of initial LOFin QA pairs by
category.
CategoryNumeric
(Text)Numeric
(Table)Textual Total
# QAs 273 925 397 1595
Ratio (%) 17.11 58.00 24.89 100
Table 13: Distribution of expanded LOFin QA pairs by
category.
any questions associated with documents that have
page separation issues. Within each candidate doc-
ument, we compute BM25 similarity between the
FinQA gold table context (which emphasizes dis-
tinct numerical values) and each page’s content
to select candidate evidence pages. Additionally,
an NLI model evaluates the semantic similarity
between the concatenated pre_text and post_text
(i.e., the context surrounding the table) and the
content of each page. If both methods select the
same top candidate, that page is accepted and sub-
sequently verified; otherwise, manual annotation
is performed to ensure accuracy. Out of 1147 in-
stances, 894 (78%) were automatically accepted.
However, among these accepted pages, 9 instances
(approximately 1% of 894) contained errors due to
OCR issues that merged content from two pages
into one. In the remaining 218 instances (19%),
the top pages selected by the two methods differed,
necessitating manual annotation, and an additional
35 cases (3%) were discarded during preprocessing.
This combined approach ensures robust and accu-
rate identification of the correct evidence pages for
FinQA.B.4 Question Templates of SEC-QA
•How much common dividends did {company}
pay in the last {num_year} years in US dol-
lars?
•What is the percentage difference of {com-
pany1}’s {metric} compared to that of {com-
pany2}?
•What is {company}’s overall revenue growth
over the last {num_year}-year period?
•Among {company_names}, what is the {met-
ric2} of the company that has the highest
{metric1}?
C Experimental Settings
C.1 Hyperparameters
• Candidate document count k′
D: 100
• Final document count kD: 5
• Final passage count kP: 5
• Maximum relevant passages k′
P: 10
• Maximum iterations imax : 3
• LLM temperature: 0.01
•Chunking tool: Langchain5’s RecursiveChar-
acterTextSplitter
• Chunk size: 1024
• Overlap between chunks: 30
C.2 Baselines
Perplexity: The Perplexity baseline employs the
llama-3.1-sonar-large-128k-online model,
integrating web search to supplement the context
used for answer generation.
Self-RAG: Self-RAG utilizes a 13B model in
a Long-form setting. It leverages the Contriever
model to retrieve 10 context passages for each
query, ensuring comprehensive coverage of the rel-
evant information.
5https://www.langchain.com/

RQ-RAG: text-embedding-3-small is used
in RQ-RAG to retrieve three context passages at
each retrieval step. Configured for multi-hop QA
with an exploration depth of two, it employs itera-
tive retrieval to continuously refine the contextual
information used for answer generation.
HybridSearch: HybridSearch combines BM25-
based scores with dense scores computed using the
text-embedding-3-small model. This hybrid ap-
proach adheres to its original configuration to ef-
fectively balance lexical and semantic matching.
IRCoT: IRCoT relies on a BM25-based retriever
and employs GPT-4o for chain-of-thought (CoT)
reasoning. It performs iterative retrieval by exe-
cuting up to 3 iterations, with 5 passages being
retrieved during each iteration.
HHR: HHR implements a hybrid retrieval
strategy that combines dense and sparse
retrieval methods. In this approach, the
text-embedding-3-small model is used for
dense retrieval, and the system first retrieves the
top 5 documents; within each document, the top
10 passages are then selected.
Dense: The Dense method uses the
text-embedding-3-small model to initially
retrieve the top 50 relevant passages. These
passages are subsequently reranked using a
DeBERTa-v3 model to determine their final order.
C.3 Textual Evaluation Prompt
This prompt is used to evaluate the factual correct-
ness of generated textual answers by instructing
the LLM to compare them against the ground truth
within the provided context.
You are a highly knowledgeable expert and teacher in
the finance domain.
You are reviewing a student’s answers to financial
questions.
You are given the context, the question, the student’s
answer and the student’s explanation and the ground -
truth answer.
Please use the given information and refer to the ground -
truth answer to determine if the student’s answer is
correct.
The input information is as follows:
context: ’{gold context}’
question: ’{question}’
ground-truth answer: ’{answer}’
student’s answer: ’{generated answer}’
Please respond directly as either ’correct’ or ’incorrect’.
Table 15: GPT-4o LM Evaluation Prompt TemplateD LLM Prompts
This section presents a detailed overview of the
prompting strategies employed in our framework.
Our prompt templates are designed based on com-
mon instruction formats introduced in the Prompt
Engineering Guide6, and were initially generated
using GPT-4o in an instruction-following mode.
To ensure consistency with each module’s output
format, we manually refined the initial prompts
through a lightweight post-editing process.
No explicit constraints were imposed during
prompt construction. We did not apply few-
shot prompting due to practical limitations such
as prompt length and inference cost. The fi-
nal prompts are tailored to the specific needs of
each module—query rewriting, document summa-
rization, evidence curation, and answer genera-
tion—and are provided in this section.
These prompt designs play a central role in facili-
tating efficient information processing and enabling
accurate responses to complex financial questions.
By disclosing our prompting approach in detail, we
aim to enhance the reproducibility and transparency
of our overall system.
D.1 Summarization Prompt
To assist document indexing during hierarchical
retrieval (section 4.1.1), we use a concise LLM-
based summarization prompt as follows Table 16:
You are a helpful assistant. Summarize the following
text:
Table 16: Summarization prompt
6https://www.promptingguide.ai/

D.2 Query Transformation Prompt
To reduce noise from overly specific or ambiguous
financial terms in user queries, we rewrite them into
focused retrieval queries using the prompt below
Table 17:
You are an AI that rewrites user questions about finan-
cial topics into concise meta-focused queries.
1) Identify the key financial terms or metrics in the
question.
2) Determine which type of documents typically contain
those terms.
3) Transform the user’s question into a short query refer-
encing the financial terms and the relevant documents.
4) Do not reveal the transformation process or provide
examples.
5) Output only the final rewritten query.
## Question: {Question}
### Output format
## Query: {Rewritten query}
Table 17: Financial query transformation instructions
D.3 Evidence Curation Prompt
The evidence curation module jointly per-
forms three tasks—passage filtering, answerabil-
ity checking, and complementary question genera-
tion—using a single prompt as follows Table 18:### Instruction
You are a financial expert. Evaluate the provided context
to determine if it contains enough information to answer
the given question.
1. Read the context carefully and decide if it contains
enough information to answer the question.
2. If it is answerable, set ’is_answerable: answerable’
and provide the answer in ’answer’.
3. If it is not answerable, set ’is_answerable: unanswer-
able’. Then:
- List the relevant document IDs in ’answer-
able_doc_ids’ in order of relevance (from most to least
relevant).
- Explain what specific information is missing in ’miss-
ing_information’.
- Provide a concise question in ’refined_query’ to search
for exactly that missing information.
4. Output your result strictly in the specified format
below using ’##’ headers.
###Inputs
Context:
Context1 (ID: 1): Title is {title1}. Content is {content1}
Context2 (ID: 2): Title is {title2}. Content is {content2}
...
Question: {question}
### Output format
## is_answerable: answerable or unanswerable
## missing_information: If ’unanswerable’, specify the
details or data needed; if ’answerable’, None
## answer: If ’answerable’, provide the answer; if ’unan-
swerable’, then None
## answerable_doc_ids: Provide a list of document IDs
that contain relevant information (e.g., [1, 2]). If none,
use []
## refined_query: If ’unanswerable’, provide a refined
question to obtain the missing information;
Table 18: Evidence curation instructions
D.4 Generation Prompt
We adopt two generation styles depending on the
question type. For numerical reasoning tasks, we
use Program-of-Thoughts (PoT) prompting to gen-
erate Python code (see Table 19); for textual reason-
ing, we apply Chain-of-Thought (CoT) prompting
(see Table 20).

[System Input]
You are a financial expert, you are supposed to generate
a Python program to answer the given question based on
the provided financial document context. The returned
value of the program is supposed to be the answer.
ˋˋˋpython
def solution():
# Define variables name and value based on
the given context
guarantees = 210
total_exposure = 716
# Do math calculation to get the answer
answer = (guarantees / total_exposure) *
100
# return answer
return answer
ˋˋˋ
[User Input]
Context:
Sources: {title1} - {content1}
Sources: {title2} - {content2}
...
Question: {question}
Please generate a Python program to answer the given
question. The format of the program should be the
following:
ˋˋˋpython
def solution():
# Define variables name and value based on
the given context
...
# Do math calculation to get the answer
...
# return answer
return answer
ˋˋˋ
Continue the program to answer the question. The re-
turned value of the program is supposed to be the an-
swer:
ˋˋˋpython
def solution():
# Define variables name and value based on
the given context
ˋˋˋ
Table 19: The prompt for generating a Python program
to answer a financial question.[System Input]
You are a financial expert, and you are supposed to an-
swer the given question based on the provided financial
document context. You need to first think through the
problem step by step, documenting each necessary step.
Then, you are required to conclude your response with
the final answer in your last sentence as "Therefore, the
answer is final answer.
[User Input]
Context:
Sources: {title1} - {content1}
Sources: {title2} - {content2}
...
Question: {question}
Let’s think step by step to answer the given question.
Table 20: Chain-of-Thought prompt for generating a
Python program to answer a financial question.
E Generalization to Other Domains
To evaluate the effectiveness of our method be-
yond the financial domain, we assess retrieval per-
formance on the LegalBench-RAG (Pipitone and
Alami, 2024) dataset, a legal-domain benchmark
where retrieval is particularly challenging due to
domain-specific terminology and structured doc-
uments. Using a mini question set, we conduct
retrieval over the full document corpus to simulate
realistic conditions.
As shown in Table 21, HiREC achieves the high-
est Precision@k (41.42) and Recall@k (48.50)
while using the lowest average number of retrieved
passages per query ( k= 3.2). This result high-
lights the framework’s ability to efficiently retrieve
essential content with minimal redundancy. In par-
ticular, HiREC demonstrates strong performance
on structured datasets such as MAUD and CUAD,
outperforming other models by a large margin in
both P@k and R@k.
ContractNLI is a document-level natural lan-
guage inference task in which each contract is eval-
uated against a fixed set of hypotheses. Unlike
passage-based QA tasks, it requires global reason-
ing over the full document. As a result, Dense
may benefit from broad recall, potentially retriev-
ing passages that touch on relevant hypotheses by
chance. In contrast, HiREC follows a structured ev-
idence curation pipeline that emphasizes precision,
often selecting fewer but more targeted documents.
This structural difference helps explain the smaller
performance gap between HiREC and Dense on
ContractNLI compared to other tasks.

ModelPrivacyQA ContractNLI MAUD CUAD AverageAvg.k
P@k R@k P@k R@k P@k R@k P@k R@k P@k R@k
Hybrid 21.13 35.99 12.06 34.36 1.34 2.05 7.73 21.97 10.57 23.59 5.0
HHR 20.31 35.57 19.28 54.25 8.04 14.20 8.66 25.49 14.07 32.38 5.0
Dense 31.55 52.21 28.56 78.69 11.03 20.87 10.82 29.92 20.49 45.42 5.0
HiREC 55.79 49.94 35.46 53.31 33.70 36.80 40.74 53.95 41.42 48.50 3.2
Table 21: Average Precision and Recall for the LegalBench-RAG dataset. The best performance values are
highlighted in bold and the second-best are underlined.
Overall, these results confirm the generalizabil-
ity of HiREC to non-financial domains, demonstrat-
ing robustness across diverse legal document types
and retrieval tasks.
F Failure Case Study
Retrieval Failure Case In this case, the model
fails to retrieve all relevant documents across multi-
ple fiscal years. The question explicitly asks for the
total common dividends paid over the past three
years (as of 2023), which requires retrieving infor-
mation from multiple annual filings. However, the
model only retrieves passages from the 2023 report.
This indicates difficulty in reasoning over tempo-
ral scope and aggregating evidence when relevant
information is distributed across documents.
Generation Failure Case In this case, the model
retrieves a passage containing sufficient informa-
tion to answer the question. However, the gener-
ated answer is incorrect due to faulty arithmetic
reasoning. Specifically, the free cash flow should
be computed as the difference between cash from
operations and capital expenditures. Although both
values are retrieved, the model produces an incor-
rect value, indicating limitations in table-grounded
numerical reasoning and code execution.

Step Content
Question How much common dividends did Bank of America pay in the last 3 years, as of 2023, in millions of
US dollars?
Gold
EvidencesBAC_2023_10K (p.94), BAC_2022_10K (p.94), BAC_2021_10K (p.94)
Retrieval ✗BAC_2023_10K (p.141): NOTE 13 Shareholders’ Equity
Common Stock . . .
✗BAC_2023_10K (p.150): Defined Contribution Plans
The Corporation. . .
Generation Ground Truth Answer : 25,718 Generated Answer : 2,560
Table 22: Retrieval failure: model fails to retrieve documents from multiple years required to answer a multi-year
aggregation question.
Step Content
Question According to the information provided in the statement of cash flows, what is the FY2020 free cash
flow (FCF) for General Mills? Answer in USD millions.
Gold
EvidencesGIS_2020_10K (p.51)
Retrieval ✓GIS_2020_10K (p.51): . . . $ 3,676.2 . . . Capital expenditures $ 460.8 . . .
✗GIS_2020_10K (p.17): Adjusted diluted EPS of $3.61 increased 12 percent on a constant-currency
basis . . .
✗GIS_2020_10K (p.36): Free cash flow $3,215.4 . . . Net cash provided by operating activities
conversion . . .
LLM Output cash_from_operations = 3700 # in USD millions
capex = 461 # in USD millions
Generation Ground Truth Answer : 3,215 Generated Answer : 3,239
Table 23: Generation failure: despite correct retrieval, the model fails to compute free cash flow accurately from
table values.

Category Examples
Numeric (Table) Question: What percentage of the estimated purchase price for Hologic in 2007 is
attributed to the net tangible assets?
Answer: 3.8%
Evidence: HOLX_2007_10K (p.110)
The components and initial allocation of the purchase price consist of:
Net tangible assets acquired as of September 18, 2007: $2,800
Developed technology and know how: $12,300
Customer relationship: $17,000
Trade name: $2,800
. . .
Estimated Purchase Price: $73,200.
Numeric (Text) Question: During the years 2004 to 2006, what was the average impairment on con-
struction in progress, expressed in millions, for American Tower Corporation as reported
in their 2006 financial documents?
Answer: 2.63
Evidence: AMT_2006_10K (p.106)
Construction-In-Progress Impairment Charges—For the years ended De-
cember 31, 2006, 2005 and 2004, the Company wrote-off approximately
$1.0 million, $2.3 million and $4.6 million , respectively, of construction-in-progress
costs, primarily associated with sites that it no longer planned to build.
Textual Question: What are the geographies that American Express primarily operates in as of
2022?
Answer: United States, EMEA, APAC, and LACC
Evidence: AXP_2022_10K (p.154)
Effective for the first quarter of 2022, we changed the way in which we allocate certain
overhead expenses by geographic region. As a result, prior period pretax income (loss)
from continuing operations by geography has been recast to conform to current period
presentation there was no impact at a consolidated level.
(Millions) United States EMEA APAC LACC Other Unallocated Consolidated
2022
Total revenues net of interest expense 41,3964,871 3,8352,917 (157) 52,862
Table 24: Examples of financial QA pairs categorized by question type. The highlighted segments in the evidence
indicate the spans that are most relevant to the question.
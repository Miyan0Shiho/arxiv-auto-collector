# Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems

**Authors**: Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang, Stephen Rose, Jesse C. Cresswell

**Published**: 2025-10-15 18:02:30

**PDF URL**: [http://arxiv.org/pdf/2510.13975v1](http://arxiv.org/pdf/2510.13975v1)

## Abstract
Retrieval-augmented generation (RAG) is a prevalent approach for building
LLM-based question-answering systems that can take advantage of external
knowledge databases. Due to the complexity of real-world RAG systems, there are
many potential causes for erroneous outputs. Understanding the range of errors
that can occur in practice is crucial for robust deployment. We present a new
taxonomy of the error types that can occur in realistic RAG systems, examples
of each, and practical advice for addressing them. Additionally, we curate a
dataset of erroneous RAG responses annotated by error types. We then propose an
auto-evaluation method aligned with our taxonomy that can be used in practice
to track and address errors during development. Code and data are available at
https://github.com/layer6ai-labs/rag-error-classification.

## Full Text


<!-- PDF content starts -->

Classifying and Addressing the Diversity of Errors in
Retrieval-Augmented Generation Systems
Kin Kwan Leung Mouloud Belbahri Yi Sui Alex Labach
Xueying Zhang Stephen Rose Jesse C. Cresswell
Layer 6 AI, Toronto, Canada
Abstract
Retrieval-augmented generation (RAG) is a
prevalent approach for building LLM-based
question-answering systems that can take ad-
vantage of external knowledge databases. Due
to the complexity of real-world RAG systems,
there are many potential causes for erroneous
outputs. Understanding the range of errors that
can occur in practice is crucial for robust de-
ployment. We present a new taxonomy of the
error types that can occur in realistic RAG sys-
tems, examples of each, and practical advice
for addressing them. Additionally, we curate
a dataset of erroneous RAG responses anno-
tated by error types. We then propose an auto-
evaluation method aligned with our taxonomy
that can be used in practice to track and address
errors during development. Code and data are
available at github.com/layer6ai-labs/rag-error-
classification.
1 Introduction
Retrieval-augmented generation (RAG) (Lewis
et al., 2020) has become the dominant paradigm for
applying generative large language models (LLMs)
in applications where outputs must incorporate
knowledge from outside of the model’s training set.
This is especially valuable for grounding genera-
tion in factual information to reduce fabricated con-
tent (Maynez et al., 2020; Shuster et al., 2021), and
when non-public domain knowledge is required.
RAG systems are already widely deployed in real-
world applications to provide natural language in-
terfaces to knowledge sources (Amugongo et al.,
2024), but despite their merits they can still be
error-prone in practice (Venkit et al., 2024; Magesh
et al., 2025; Grant, 2024). Due to the greater com-
plexity of RAG pipelines compared to direct LLM
generation, these errors are diverse and their causes
can be difficult to trace. Deploying a RAG pipeline,
especially in critical industries like healthcare, re-
quires understanding the variety of errors that can
occur in order to monitor and minimize them.Existing work on RAG errors has generally not
accounted for the complexity of real-world RAG
systems and their failure modes. On the data side,
widely used benchmark tasks for evaluating RAG
systems are often overly simplistic, typically fea-
turing multiple-choice questions (Mihaylov et al.,
2018; Guinet et al., 2024; Yang et al., 2024) or re-
quiring short factual answers for ease of validation
(e.g., Rajpurkar et al. (2016); Yang et al. (2018);
Joshi et al. (2017); Kwiatkowski et al. (2019)). In
practice, users expect more sophisticated answers
that thoroughly explain a topic, but as a result can
fail in more subtle ways.
On the model side, real RAG systems use com-
plex multi-step pipelines that go beyond a retriever-
generator pair (Akkiraju et al., 2024), with addi-
tional data processing steps like adaptive chunking
and reranking. Prior works that categorize RAG er-
rors have used bare-bones pipelines benchmarked
on simple datasets, and as a result have overlooked
entire classes of errors, especially those associ-
ated with pre-generation steps (Barnett et al., 2024;
Venkit et al., 2024). As a result, practitioners rely-
ing on these works may be left with an overly op-
timistic view of their RAG system’s performance,
and fail to identify the scope and severity of errors
afflicting them. To address these gaps, we use chal-
lenging public datasets and build a realistic RAG
question-answering (QA) system reflective of those
currently used in industry. We perform a deep anal-
ysis of RAG failure modes with illustrative exam-
ples, and practical advice for mitigating errors.
Finally, we develop auto-evaluation tools to clas-
sify error types according to our taxonomy. To
validate these systems, we manually annotate er-
rors producing a first-of-its-kind RAG error type
dataset. The overall purpose of our taxonomy and
auto-evaluation system is for practitioners to be
able to identify weak links and common errors in
their RAG pipelines. Hence, we provide an end-to-
end demonstration of how auto-evaluation can be
1arXiv:2510.13975v1  [cs.CL]  15 Oct 2025

Documents
Queries
Ground truthsDatasetChunker
E1,E2,E3Embedder
EmbedderSearch Reranker
E7,E8Generator
E9,E10,E11,E12,
E13,E14,E15,E16
OutputRetrieval
E4,E5,E6
Auto-eval
Figure 1: Diagram of our implemented RAG architecture, reflective of systems currently applied in industry.
Components are annotated with error types that are caused by them.
used by identifying the most common errors pro-
duced by our reference pipeline, and implementing
targeted improvements.
Our main contributions are:
1.A novel taxonomy of errors made by RAG sys-
tems, along with examples of each and recom-
mendations for addressing them;
2.An auto-evaluation system for identifying and
classifying errors according to our taxonomy;
3. A dataset of RAG errors annotated by type.
2 Related Works
RAG Error Taxonomies:Barnett et al. (2024) pro-
vide the most comparable taxonomy of RAG errors
to ours with seven types associated to select RAG
components. Here, we describe a more comprehen-
sive range of RAG errors relating toeverystep of a
realistic RAG pipeline, along with specific exam-
ples and mitigation strategies, both of which are
lacking in the prior work. Venkit et al. (2024) clas-
sify errors in public black-box QA systems from
the perspectives of human evaluators without ex-
amining specific pipeline components or attribut-
ing errors to them. Agrawal et al. (2024) focus
on RAG systems built on knowledge graphs. Yu
et al. (2024a) survey RAG evaluation benchmarks
and metrics but do not break down causes of errors
beyond the high-level dichotomy of retrieval ver-
sus generation. Other works (Huang et al., 2025;
Magesh et al., 2025) that specifically focus on hal-
lucinations overlook the nuanced ways RAG can
fail. Error taxonomies for non-RAG QA settings
also exist (Rawte et al., 2023; Huang et al., 2025).
RAG Auto-evaluation:Simple QA benchmarks
can be evaluated using exact match or overlap-
based metrics, but realistic questions and answers
require more sophisticated evaluations based in nat-
ural language understanding. Previous works have
applied metrics using LLM-as-a-Judge (Zhenget al., 2023) to automatically evaluate RAG out-
puts, including RAGChecker (Ru et al., 2024),
ARES (Saad-Falcon et al., 2024), and RAGAs (Es
et al., 2024). These works are each tailored to a
specific error classification scheme, with narrower
scope than ours. Other prior works have devel-
oped benchmark datasets with LLM-based auto-
evaluation (e.g., Zhu et al. (2024); Liu et al. (2024)),
but do not probe the internals of RAG systems to
give detailed information on error causes.
3 RAG Implementation
To demonstrate errors that can occur in realistic,
production-grade RAG systems, we designed a
modular pipeline reflecting common architectural
patterns used in industry (Bleiweiss, 2024; Alexan-
der et al., 2025) (see Figure 1). We begin with a
chunkingstage, where documents are processed to
optimize granularity and relevance. The chunks are
indexed using an embedding model, then queried
via denseretrieval, which returns top candidates
based on similarity to the input query. Retrieved
chunks arererankedusing a separate language
model given the original query as context, which
can better highlight semantic relevance. Finally,
the top-ranked chunks are fed to ageneratorLLM,
which produces the final answer. This architecture
supports component-level variation and allows for
in-depth error analysis at each stage of the pipeline.
We provide complete details on the options imple-
mented and used at each stage in Section A.
Datasets:We use the DragonBall dataset, part
of the RAGEval framework (Zhu et al., 2024), an
evaluation suite designed to rigorously assess RAG
systems across diverse domains and scenarios. The
dataset provides a broad benchmark for evaluating
RAG performance in complex settings, spanning
domains such as finance, law, and medicine. We
used both the English and Chinese portions of the
2

dataset with 3108 and 3601 questions respectively.
We also use the CLAPnq dataset for additional anal-
ysis (Rosenthal et al., 2025), which contains high-
quality reference answers and passages for 4946
natural questions relating to Wikipedia documents.
4 Error Classification
We present a practically grounded taxonomy of er-
rors that can be attributed to different parts of a
RAG system. Errors are grouped by the pipeline
stage that caused them: chunking, retrieval, rerank-
ing, or generation. For each stage and error type,
we describe the nature of the error, explain why
it occurs in RAG systems, provide real examples
from our RAG implementation, and finally give ad-
vice for how to reduce their occurrence.
Error types are not mutually exclusive; in prac-
tice multiple error types often co-occur, as errors
early in the pipeline beget later ones. Additionally,
we do not claim that our taxonomy is exhaustive,
since variations in RAG architecture will lead to
different error types. Our taxonomy focuses on er-
rors that occur within RAG systems, and hence we
exclude failures caused by adversarial inputs, faults
in the corpus, or similar anomalies.
4.1 Chunking
E1 Overchunking:Documents are split into exces-
sively small or disjointed segments, causing incom-
plete coverage of topics. Individual chunks are frag-
mented or ambiguous. Errors cascade downstream
when search fails to retrieve consecutive chunks.
Query: What platform did Sunrise Holidays in-
troduce in April 2019?
Chunk Excerpt:[Excluded from chunk: In
April 2019, Sunrise Holidays introduced an on-
line booking platform, which greatly improved its
competitiveness.]The launch of this user-friendly
platform attracted more customers [...]
Response: The platform introduced by Sunrise
Holidays in April 2019 is not specified.
Ground Truth: An online booking platform.
Cause: The retrieved chunks come from the cor-
rect document, but the description of it as anon-
line bookingplatform is cut off.
E2 Underchunking:Chunks are too large, cover-
ing multiple topics with mixed content. Irrelevant
information dilutes keywords or phrases, lowering
retrieval scores on the correct chunks. Chunks pro-
vided to the generator contain extraneous content
that can confuse the model.Query: What system does CleanCo Housekeep-
ing Services have in place to safeguard assets and
ensure financial accuracy?
Chunk Excerpt: [...] To mitigate risks such as
increasing competition, regulatory changes, and
economic uncertainties, CleanCo plans to imple-
ment risk management strategies through diversi-
fication and continuous monitoring. [...]
Response: CleanCo has a system in place to
safeguard assets and ensure financial accuracy
through its risk management framework.
Ground Truth: Unable to answer.
Cause: Retrieved chunks contain so much tan-
gential information that the generator uses unre-
lated information to answer instead of abstaining.
E3 Context Mismatch:Chunks split text at arbi-
trary points, breaking contextual links by separat-
ing definitions from the information they support.
This ambiguity causes failed retrieval downstream
where keywords are missing.
Query: Why is the Philippines typhoon prone?
Chunk A (Retrieved): The Philippines’ evident
risk to natural disasters is due to its location. [...]
Chunk B (Not Retrieved): In addition, the coun-
try faces the Pacific Ocean where 60% of the
world’s typhoons are made. [...]
Response: Due to its geographical location, cli-
mate, and topography.
Ground Truth: It faces the Pacific Ocean where
60% of the world’s typhoons are made.
Cause: Chunk B mentions “the country” but not
the Philippines by name, which led to the retriever
incorrectly assigning it low relevance.
Improvement: Straightforward heuristics include
adjusting chunk size, where larger chunks help re-
duce E1, smaller chunks help reduce E2, while
adding small overlaps mitigates E3 by preserv-
ing continuity (Safjan, 2023). However, these ap-
proaches require careful tuning and are often insuf-
ficient on their own. Adaptive chunking strategies
handle real-world variability more effectively.
Structure-aware chunking preserves document
logic by splitting along units like paragraphs or sec-
tion headers. This works especially well for cor-
pora with consistent structure, such as financial re-
ports (Yepes et al., 2024). Semantic chunking seg-
ments text based on meaning, such as by detect-
ing topic shifts using cosine distance between sen-
tence embeddings (Qu et al., 2025) or spikes in per-
plexity (Zhao et al., 2024). Hybrid strategies com-
3

bine structural and semantic cues to balance coher-
ence and topical focus. For example, S2 Chunk-
ing (Verma, 2025) integrates layout with embed-
dings and achieves strong results on document QA.
4.2 Retrieval
E4 Missed Retrieval:Relevant chunks are not re-
trieved, leading the generator to give incomplete
answers, fabricate information to fill gaps, or ab-
stain unnecessarily. Misses can occur when high-
similarity chunks only address part of the question.
Query: What were the large-scale financing ac-
tivities undertaken by Vanguard Media Group in
2019 and Energex Corp. in 2021?
Response: Vanguard Media Group’s large-scale
financing activities in 2019 included the acquisi-
tion of Silver Screen Studios for $200 million [...]
Ground Truth: Vanguard Media Group raised
$100 million in capital in 2019, while Energex
Corp. conducted large-scale financing in 2021.
Cause: The query asks about two entities, but all
top-ranked chunks relate to only one.
E5 Low Relevance:Retrieved chunks are only
loosely related to the query. Causes include an un-
balanced corpus with many chunks relating to one
topic, and vocabulary mismatch where the query
uses different terms than the corpus.
Query: In which year is the Corporate Gover-
nance Report for Green Fields Agriculture Co.?
Response: The Corporate Governance Report
does not provide a specific year, but it appears to
cover events and initiatives from 2018 and 2021.
Ground Truth: 2018.
Cause: Many chunks from the Report were as-
signed high similarity and retrieved, but did not
relate to the Report’s publication year. The gener-
ator latches onto years discussed in the content.
E6 Semantic Drift:Retrieved documents match
keywords, not the query’s intent, due to reliance on
keyword matching rather than semantic relevance.
Query: According to the judgment of Hartford,
Ashland, Court and Glenwood, Quailwood, Court,
what sentences were given to the defendants M.
Cooper and Y . Nelson?
Response: M. Cooper was sentenced to four
years imprisonment.
Ground Truth: M. Cooper was sentenced to four
years imprisonment; Y . Nelson was sentenced tothree years imprisonment.
Cause: The retrieved chunks contain information
about both defendants’ cases, but don’t include
the sentence for Y . Nelson.
Improvements:Retrieval systems present the
widest diversity of approaches out of the RAG
stages, and hence can be tuned in many ways.
First, query rewriting via keyword expansion, para-
phrasing, or semantic reformulation can improve
recall and precision by reducing vocabulary mis-
match (Ma et al., 2023). Second, hybrid retrieval
that combines sparse (e.g., BM25 (Robertson et al.,
1994)) and dense (embedding-based) methods can
improve robustness by capturing both lexical and
semantic matches (Ni et al., 2022). Third, top-
kretrieval is often insufficient when both E4 and
E5 are frequent. Adaptive thresholds can better
balance recall and precision (Sun et al., 2025).
Fourth, domain-specific or fine-tuned embedding
models can significantly boost retrieval accuracy
and reduce hallucinations (Databricks, 2025). Fi-
nally, metadata filtering enhances precision by us-
ing structured fields (e.g., source or section) to pri-
oritize relevant content (Poliakov and Shvai, 2025).
4.3 Reranking
E7 Low Recall:Although the necessary chunks
are retrieved, they are reranked too low and not
forwarded to the generation model. Low recall
produces false negatives, where relevant data is
available but unintentionally ignored.
Query: According to Greenfield, Fairview, Court
and Vandalia, Bayside, Court, whose sentencing
time is longer, H. Price or G. Torres?
Response: G. Torres’ sentencing time is longer,
four years, compared to H. Price, whose sentence
is not mentioned in the provided context.
Ground Truth: H. Price was sentenced to eight
years of imprisonment, while G. Torres was sen-
tenced to four years imprisonment.
Cause: The chunk with H. Price’s sentencing
was retrieved, but was ranked below the reranker
threshold and not passed to the generator.
E8 Low Precision:Irrelevant chunks are ranked
highly and forwarded to the generation model, lead-
ing to the generation model being confused by
noise. Low precision leads to false positives, where
the generator is overly confident in noisy context.
4

Query: On which stock exchange is Retail Em-
porium publicly traded?
Response: The New York Stock Exchange.
Ground Truth: Unable to answer.
Cause: Highly ranked but irrelevant chunks ref-
erence New York, leading the model to make an
inappropriate conclusion.
Improvements:Modern RAG pipelines increas-
ingly include a reranking stage to reorder re-
trieved chunks based on semantic relevance (Blei-
weiss, 2024). The two dominant approaches
are cross-encoders, which jointly encode query-
chunk pairs (Déjean et al., 2024), and LLM-based
rerankers, which use prompted models to assess
candidates (Sun et al., 2023). Rerankers may
still miss key context (E7) or over-rank irrelevant
chunks (E8) when domain relevance is poorly un-
derstood. Fine-tuning on in-domain data is the
most effective way to improve performance (Mor-
eira et al., 2024; Aarsen, 2025; Yu et al., 2024b),
and even small amounts of labeled data can yield
significant gains (Aarsen, 2025). More specifically,
incorporating hard negatives, which are plausible
but non-relevant passages, further helps the model
reduce false positives (E8) (Yu et al., 2024b).
4.4 Generation
E9 Abstention Failure:The model should have
abstained, but instead answers incorrectly. Absten-
tion is appropriate when the answer is not available
in the corpus, or when the query is ambiguous.
Query: How much did Green Fields Co. raise
through large-scale financing in December 2018?
Response: Green Fields Co. did not raise any
funds in December 2018.
Ground Truth: Unable to answer
Cause: The retrieved context contains informa-
tion about 2018, but not about fundraising. Hence,
the model assumes fundraising did not occur, but
this was not stated in the corpus.
E10 Fabricated Content:Although the query is
answerable, the response includes unverifiable in-
formation not grounded in the retrieved chunks nor
supported by external evidence.
Query: According to the hospitalization records
of Southport General Hospital, summarize the
past medical history of K. Mendoza.
Response: K. Mendoza has a history of Parkin-
son’s disease, with symptoms including tremors,bradykinesia, and postural instability.
Ground Truth: K. Mendoza was generally
healthy prior to the onset of his symptoms.
Cause: The model fabricates a history of Parkin-
son’s disease which is not in the context.
E11 Parametric Overreliance:The LLM relies
on its internal (parametric) knowledge rather than
retrieved documents. Internal knowledge may be
factual, as opposed to fabricated, but irrelevant.
Query: What was the total amount of liabilities
of Artistic Creations at the end of 2019?
Response: The total amount of liabilities of Artis-
tic Creations at the end of 2019 is $150M - $20M
(acquisition) - $50M (financing) - $10M (invest-
ment) + $15M (restructuring) = $85M.
Ground Truth: Unable to answer.
Cause: The model does a calculation based on
parametric financial knowledge instead of ac-
knowledging that it has insufficient information.
E12 Incomplete Answer:The response is from
the corpus and correct, but misses critical details.
Query: How did the dividend in July 2017 reflect
AMS Inc.’s strong financial performance?
Response: The dividend in July 2017 reflected
AMS Inc.’s strong financial performance, which
was driven by market demand and changes in
product prices, resulting in an operating income
of $450 million and a net profit of $60 million.
Ground Truth: The dividend demonstrated the
company’s ability to generate sufficient profits
and maintain a healthy cash flow.
Cause: The response affirms the basis of the ques-
tion, but does not analyze the situation.
E13 Misinterpretation:The generator misuses or
misrepresents retrieved content.
Query: According to the judgments of Trenton,
Springfield and Hamilton, Harrison Courts what
sentences were given to Q. Ruiz and M. Ward?
Response: M. Ward was sentenced to four years
imprisonment, and Q. Ruiz has the right to appeal
this judgment within 30 days.
Ground Truth: M. Ward was sentenced to four
years of imprisonment, and Q. Ruiz was sen-
tenced to 5 years of imprisonment.
Cause: The model misinterprets facts about two
defendants as being related and incorrectly com-
bines them for the answer.
E14 Contextual Misalignment:The response is
5

factual and comes from related information in the
corpus, but does not address the query.
Query: According to the judgment of Preston,
Lancaster, Court, summarize the facts of M. Har-
ris’s crimes.
Response: M. Harris was found guilty of homi-
cide and was sentenced to life imprisonment.
Ground Truth: M. Harris committed the crime
of intentional homicide by tampering with the
brakes of her business partner.
Cause: The response contains factual informa-
tion, but discusses sentences rather than the re-
quested facts of the crimes.
E15 Chronological Inconsistency:The response
presents events or facts in the wrong temporal order,
or confuses past, present, or future timelines.
Query: Which company, Green Fields or
JetWing, changed their board of directors earlier?
Response: Green Fields made changes to their
board of directors earlier, in December 2021,
while JetWing made changes in October 2021.
Ground Truth: JetWing made changes earlier
in September 2021, while Green Fields made
changes in December 2021.
Cause: The model provides a wrong date, and in-
correctly considers Dec. earlier than Oct.
E16 Numerical Error:The response includes in-
correct calculations, quantities, or misrepresents
numerical data from the retrieved documents.
Query: Which company reduced its liabilities
more, InnovateTech or Sparkling Clean?
Response: InnovateTech reduced its liabilities by
a larger amount, a total of $50 million, compared
to Sparkling Clean’s £500,000.
Ground Truth: InnovateTech reduced its liabili-
ties by a larger amount ($30 million) compared
to Sparkling Clean (£500,000).
Cause: The model incorrectly added together two
numbers from the context giving $50 million.
Improvements:Generation-stage errors can be
mitigated through abstention modeling, answer val-
idation, prompt engineering, and tool augmenta-
tion. For abstention failures (E9), models can ex-
plicitly use ambiguity detection (Cole et al., 2023;
Kim et al., 2023), conformal abstention (Yadkori
et al., 2024), or fine-tuning approaches like Trust-
Align (Song et al., 2025). For content fabrication
(E10), post-generation validation such as Chain-of-Verification (Dhuliawala et al., 2024) or cri-
tique modules (Asai et al., 2023) encourage fact-
checking and grounding. Errors related to context
misuse (E11-E14) often stem from noisy retrieval;
refinement modules can filter context before gen-
eration (Jin et al., 2025; Chirkova et al., 2025;
Wang et al., 2025). Query decomposition (Lin
et al., 2023) helps reduce E12 by breaking com-
plex queries into sub-questions, especially for com-
parisons, causal questions, or multi-step reason-
ing. For temporal and numerical errors (E15–E16),
structured prompting (Wei et al., 2022), timeline
reasoning (Bazaga et al., 2025), or tool-augmented
generation with code modules like PAL (Gao et al.,
2023) can boost reliability.
5 RAGEC: RAG Error Classification
To deepen our understanding of the RAG error
taxonomy and aid practitioners in applying it,
we developed a RAG Error Classification system
(RAGEC). In this section we describe its design,
curate a dataset with human annotations to validate
it, and demonstrate its use via our reference RAG
system with standard research datasets.
The primary objective of RAGEC is to identify
weak links in the RAG pipeline by classifying the
stageresponsible for errors. More granular error
types, which can co-occur and overlap, are classi-
fied secondarily. Identifying thefirststage where
an error occurred enables the developer to imple-
ment targeted improvements on the RAG pipeline.
Design.Given a RAG pipeline and dataset,
RAGEC consists of 3 steps: answer evaluation,
stage classification, and error type classification.
Foranswer evaluation, we prompt an LLM on
each datapoint individually to determine whether
the generated answer is incorrect. Included as con-
text are the original query, ground-truth answer and
documents according to the dataset, and the gen-
erated answer. This evaluation yields a subset of
Nerrexamples identified as incorrect. Forstage
classificationon an erroneous example, we apply a
rules-based approach to cascade over the stages, us-
ing specialized information to determine if an error
is present. Starting with the generation stage, we
check whether the generator had sufficient informa-
tion to answer the query by computing the chunk-
level recall and comparing to a threshold. If so,
we conclude the generation stage caused the error.
Otherwise, we continue and check if the reranker
recall dropped compared to the retrieval recall, in-
dicating that the reranker caused the error. If not,
6

Table 1: Stage classification agreement matrix
Human RAGEC Stage Classification
Annotation Chunking Retrieval Reranking Generation
Chunking49 13 5 10
Retrieval24 85 12 40
Reranking6 8 22 10
Generation4 21 6 62
we move to the final comparison between retrieval
and chunking. To differentiate these two we adapt
the idea from Matton et al. (2025) to extract key
concepts in the query. We then prompt the LLM as
to whether each concept is included in the ground
truth chunks. If query concepts fail to appear in
the chunks, it indicates the chunking stage caused
the error. Otherwise, by process of elimination, we
conclude the retriever was responsible. This design
is empirically motivated, and provided the highest
alignment with our human-annotated results. Other
designs we tested are described in Section B.4.
Forerror type classification, we take informa-
tion relevant to the identified stage, and conduct
detailed analysis using LLM-as-a-Judge (Zheng
et al., 2023). For example, on a chunking error we
only provide the query, ground truth answer, and
chunks from the ground truth document. Other in-
formation, like the retrieved chunks or generated
answer are irrelevant for classifying a chunking er-
ror type. To account for uncertainty in the classi-
fication, each erroneous example is annotated K
times (Wang et al., 2023). We then compute the
modal vote over error types and metrics that cap-
ture the self-consistency of the LLM’s predictions.
Further details on the implementation of
RAGEC, including the complete methodologies,
inputs used for each step, and LLM prompts, are
included in Section B.
Data Curation.No existing dataset details the
types of errors which can occur in RAG systems.
Hence, to validate RAGEC, we manually annotated
406 erroneous responses generated by our reference
system on the DragonBall dataset as to the responsi-
ble stage, and error types present. Annotators used
all available context including ground truth answer,
generated response, retrieved and reranked chunks,
and the corpus. First, annotators selected the ear-
liest stage in the pipeline which demonstrated er-
roneous behaviour, since errors propagate through
RAG systems. Subsequently, one or more error
types were selected within that stage. The repo
at github.com/layer6ai-labs/rag-error-classification
contains our annotated data.
Overchunk.Underchunk.Context Mis.
Missed Retriev.Low relevanceSemantic driftLow recall
Low precisionAbstention Fail.Fab. ContentParam. Overr.
Incomplete Ans.MisinterpretContext Misal.Chron. Incons.Num. Err01020304050PercentageCLAPnq
DragonBallFigure 2: Mode error type distribution per dataset.
Results & Analysis.In the first step of RAGEC,
answer evaluation, 27% (25%) of responses for the
DragonBall-EN (CLAPnq) dataset were classified
as incorrect, giving Nerr= 832 (1222) responses
to analyze. Of these, 406 queries were manually
annotated, and 377 were confirmed as erroneous,
reaching an agreement rate of 92.9%.
Forstage classification, RAGEC achieves a
57.8% agreement rate with human annotations. De-
tails of the results are shown in Table 1. We find
that human annotations more often identify the re-
trieval stage as causing errors, whereas RAGEC fa-
vors blaming the generator. While RAGEC’s agree-
ment rate is not close to perfect, we tested several
other methods to engineer the context for LLM-
based auto-evaluation that all performed worse (de-
scribed in Section B.4). Hence, RAG error classifi-
cation appears to be a challenging problem due to
the highly complex set of intermediate outputs pro-
duced by RAG pipelines. We encourage the com-
munity to further explore this topic by releasing
our reference pipeline and annotated error data.
Forerror type classification, we query the LLM
K= 10 times for each datapoint and extract the
most and second-most frequently predicted error
types. Compared to human-annotated error types,
RAGEC achieves an accuracy of 40.3%. Figure 2
shows the distribution of the modal error type for
DragonBall-EN and CLAPnq. Table 2 shows more
details for the DragonBall-EN dataset (CLAPnq is
shown in Table 5 of Section C.1). These distribu-
tions differ, showing that the nature of the dataset
can affect the types of errors committed by a given
RAG pipeline. This type of information can direct
diagnosis and triaging of problems in RAG. For ex-
ample, CLAPnq exhibited no chunking errors, as
its source documents were manually curated by the
dataset annotators into self-contained paragraphs
designed to answer the corresponding queries. In
contrast, DragonBall-EN almost never shows Para-
metric Overreliance (E11) for both English and
7

Table 2: Error-type distributions for DragonBall-EN.
Mode and 2ndMode show the original RAG pipeline,
while Impr. shows the mode after targeted improve-
ments to the RAG pipeline.
Error Classification Mode 2ndMode Impr.
E1 Overchunking 0 19 0
E2 Underchunking 32 92 10
E3 Context Mismatch 207 32 78
Percentage Chunking 29.7% 32.2% 16.4%
E4 Missed Retrieval 229 27 122
E5 Low Relevance 33 107 8
E6 Semantic Drift 0 7 0
Percentage Retrieval 31.5% 31.8% 24.3%
E7 Low Recall 53 5 28
E8 Low Precision 6 21 0
Percentage Reranking 7.1% 5.9% 5.2%
E9 Abstention Failure 6 0 5
E10 Fabricated Content 34 37 53
E11 Parametric Overreliance 4 6 0
E12 Incomplete Answer 17 22 26
E13 Misinterpretation 178 42 164
E14 Contextual Misalignment 14 18 8
E15 Chronological Inconsistency 3 3 13
E16 Numerical Error 16 6 19
Percentage Generation 32.7% 30.2% 53.9%
Total 832 444 534
Chinese (See Appendix C.2 for details). This can
be attributed to the dataset’s domain-specific and
context-dependent queries, which are unlikely to
be covered by the LLM’s parametric knowledge.
RAGEC also demonstrates that some error types
dominate their stage, like E3 for chunking, while
others like E5 primarily occur as a secondary error
along with another type. Notably,Fabricated Con-
tent(E10)—often referred to as hallucination—is
relatively rare in our analysis. This observation
stands in contrast to the focus on hallucination in
recent research and public discourse surrounding
LLMs (Ji et al., 2023). Our findings suggest that,
in the context of RAG, other types of errors such
as retrieval or chunking issues are more prevalent
and may warrant greater attention. Finally, in Ta-
ble 3 we examine the consistency of classifications
via mode frequency, the number of the Kruns that
agreed with the mode error type. For the majority
of the queries, RAGEC is quite consistent in deter-
mining error categories. Section C.3 discusses the
co-occurrence of the error categories.
Summary.RAGEC provides an entry point
for debugging and improvement of RAG pipelines,
which can be complex and opaque to developers.
Instead of manually sifting through individual gen-
erations, our system automatically captures inter-
mediate information like retrieved and reranked
chunks, and distills the information into high-level
descriptive statistics. Hence, our focus is not on nu-
merical metrics of performance, but on understand-Table 3: Distribution of mode frequency (K= 10).
Mode Frequency DragonBall CLAPnq
3 1 4
4 7 32
5 41 91
6 74 123
7 74 135
8 94 149
9 153 172
10 388 516
ing the types and causes of errors so that develop-
ers can intelligently prioritize aspects to improve.
Using RAGEC to Improve RAG.We con-
duct a case study on how RAGEC can be used
in practice to improve a RAG pipeline. We fo-
cus on DragonBall-EN where 832 errors were orig-
inally identified (73.3% correct). RAGEC indi-
cates that chunking, retrieval, and generation all
caused roughly the same proportion of errors (Ta-
ble 2, Mode), and hence are the primary candi-
dates for improvement. Since most chunking er-
rors come from Context Mismatch (E3) where con-
textual links are broken across chunks, we follow
the recommendation in Section 4.1 and modify the
chunking strategy from fixed-length segmentation
to a recursive, sentence-level segmentation of ap-
proximately the same size. In addition, RAGEC
indicates that most retrieval and reranking errors
come from Missed Retrieval (E4) and Low Recall
(E7) which are addressed by increasing the number
of chunks retrieved and passed to the generator.
With these improvements, we reran the RAG
pipeline and RAGEC. The improved pipeline re-
sulted in fewer errors overall, only 534 (82.8% cor-
rect). In Table 2 (column Impr.) we observe that
chunking-, retrieval-, and reranker-related errors
were signficantly reduced, while generator errors
stay similar as no intervention was done.
6 Conclusion
Our contributions in this work provide a framework
for practitioners working with RAG systems to un-
derstand, categorize, track, and fix model errors.
Our findings also indicate the wide range of possi-
ble avenues for improving the robustness of RAG
systems in future works. While our RAG error clas-
sification method, RAGEC, achieves a useful level
of agreement to human error annotations, it is far
from perfect, indicating that error classification re-
mains a challenging problem for LLM-based sys-
tems. To promote future research in this area, we
release our annotated dataset of error types as a re-
source for the community.
8

Limitations
Our RAG error taxonomy is more expansive than
prior works in this vein, but still is not exhaustive
of all possible errors that could occur, especially if
further stages were to be added to the RAG pipeline.
We focused on the most prevalent types observed
in practice.
The RAG pipeline we used for demonstrating
auto-evaluation for error classification reflects the
general architecture of modern pipelines used in
real-world applications. However, it was not highly
tuned for the DragonBall and CLAPnq datasets, for
example by implementing the many improvements
we listed in Section 4. Our aim was to show a
starting point, where practitioners can understand
errors in their system before making changes.
Additionally, our study focuses on single-turn
textual queries. Expanding the evaluation frame-
work to handle multi-turn conversations or multi-
modal inputs remains an important direction for fu-
ture work.
Finally, we found RAG error classification ac-
cording to our taxonomy to be a challenging predic-
tion problem. Part of this stems from the difficulty
of collecting labeled data. The error type labels we
collected require annotators to be experts in RAG
systems so that they comprehend the role and oper-
ation of each stage, and can disambiguate the error
types. Hence, crowdsourcing the annotation work
was not possible, and all annotation was done man-
ually by the authors to ensure the highest quality la-
bels. Still, LLM-as-a-Judge systems were not fully
capable of reasoning over all the intermediate infor-
mation created throughout the RAG pipeline, lead-
ing to rather low stage-classification accuracy. We
expect this to increase as LLMs become more pro-
ficient with multi-step reasoning and assimilating
information over long contexts.
References
Tom Aarsen. 2025. Training and finetuning reranker
models with sentence transformers v4. https://
huggingface.co/blog/train-reranker.
Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi,
and Huan Liu. 2024. Mindful-RAG: A Study of
Points of Failure in Retrieval Augmented Generation.
In2nd International Conference on Foundation and
Large Language Models, pages 607–611.
Rama Akkiraju, Anbang Xu, Deepak Bora, Tan
Yu, Lu An, Vishal Seth, Aaditya Shukla, Pritam
Gundecha, Hridhay Mehta, Ashwin Jha, PrithviRaj, Abhinav Balasubramanian, Murali Maram,
Guru Muthusamy, Shivakesh Reddy Annepally,
Sidney Knowles, Min Du, Nick Burnett, Sean
Javiya, and 19 others. 2024. Facts about build-
ing retrieval augmented generation-based chatbots.
arXiv:2407.07858.
John Alexander, Kristine Toliver, and Schofield
McLean. 2025. Build advanced retrieval-
augmented generation systems. https://learn.
microsoft.com/en-us/azure/developer/ai/
advanced-retrieval-augmented-generation .
Accessed: 2025-05-01.
Lameck Mbangula Amugongo, Pietro Mascheroni,
Steven Geoffrey Brooks, Stefan Doering, and Jan Sei-
del. 2024. Retrieval augmented generation for large
language models in healthcare: A systematic review.
Preprints.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. 2023. Self-RAG: Learn-
ing to Retrieve, Generate, and Critique through Self-
Reflection. InThe Twelfth International Conference
on Learning Representations.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. InProceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering-Software Engineering for AI, pages 194–
199.
Adrián Bazaga, Rexhina Blloshmi, Bill Byrne, and
Adrià de Gispert. 2025. Learning to reason over time:
Timeline self-reflection for improved temporal rea-
soning in language models. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers).
Amit Bleiweiss. 2024. Enhancing RAG
Pipelines with Re-Ranking. https:
//developer.nvidia.com/blog/
enhancing-rag-pipelines-with-re-ranking/ .
Accessed: 2025-05-01.
Nadezhda Chirkova, Thibault Formal, Vassilina
Nikoulina, and Stéphane Clinchant. 2025. Provence:
efficient and robust context pruning for retrieval-
augmented generation. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Jeremy Cole, Michael Zhang, Daniel Gillick, Julian
Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein.
2023. Selectively answering ambiguous questions.
InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages 530–
543.
Databricks. 2025. Improving Retrieval and RAG with
Embedding Model Fine-Tuning. Accessed: 2025-05-
01.
9

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu,
Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Ja-
son Weston. 2024. Chain-of-verification reduces hal-
lucination in large language models. InFindings of
the Association for Computational Linguistics ACL
2024, pages 3563–3578.
Hervé Déjean, Stéphane Clinchant, and Thibault For-
mal. 2024. A thorough comparison of cross-
encoders and llms for reranking splade.Preprint,
arXiv:2403.10407.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations.
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon,
Pengfei Liu, Yiming Yang, Jamie Callan, and Gra-
ham Neubig. 2023. PAL: Program-aided Language
Models. InProceedings of the 40th International
Conference on Machine Learning, volume 202, pages
10764–10799.
Nico Grant. 2024. Google’s AI Search errors cause a
furor online.The New York Times. Accessed: 2025-
05-01.
Gauthier Guinet, Behrooz Omidvar-Tehrani, Anoop De-
oras, and Laurent Callot. 2024. Automated evalu-
ation of retrieval-augmented language models with
task-specific exam generation. InProceedings of the
41st International Conference on Machine Learning,
volume 235.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Lan-
guage Models: Principles, Taxonomy, Challenges,
and Open Questions.ACM Trans. Inf. Syst., 43(2).
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput.
Surv., 55(12).
Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yu-
tao Zhu, Yongkang Wu, Zhonghua Li, Qi Ye, and
Zhicheng Dou. 2025. Hierarchical Document Refine-
ment for Long-context Retrieval-augmented Genera-
tion.arXiv:2505.10413.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A Large Scale Dis-
tantly Supervised Challenge Dataset for Reading
Comprehension. InProceedings of the 55th Annual
Meeting of the Association for Computational Lin-
guistics.
Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joon-
suk Park, and Jaewoo Kang. 2023. Tree of clarifica-
tions: Answering ambiguous questions with retrieval-
augmented large language models. InProceedings ofthe 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 996–1009.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural
Questions: A Benchmark for Question Answering
Research.Transactions of the Association for Com-
putational Linguistics, 7.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Kevin Lin, Kyle Lo, Joseph Gonzalez, and Dan Klein.
2023. Decomposing complex queries for tip-of-the-
tongue retrieval. InFindings of the Association for
Computational Linguistics: EMNLP 2023, pages
5521–5533.
Jintao Liu, Ruixue Ding, Linhao Zhang, Pengjun Xie,
and Fie Huang. 2024. CoFE-RAG: A Comprehen-
sive Full-chain Evaluation Framework for Retrieval-
Augmented Generation with Enhanced Data Diver-
sity.arXiv:2410.12248.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing.
Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suz-
gun, Christopher D. Manning, and Daniel E. Ho.
2025. Hallucination-Free? Assessing the Reliability
of Leading AI Legal Research Tools.Journal of Em-
pirical Legal Studies, 22(2):216–242.
Katie Matton, Robert Ness, John Guttag, and Emre Kici-
man. 2025. Walk the talk? measuring the faithfulness
of large language model explanations. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and
Ryan McDonald. 2020. On Faithfulness and Factual-
ity in Abstractive Summarization. InProceedings of
the 58th Annual Meeting of the Association for Com-
putational Linguistics, pages 1906–1919.
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
Sabharwal. 2018. Can a Suit of Armor Conduct Elec-
tricity? A New Dataset for Open Book Question An-
swering. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing.
Association for Computational Linguistics.
10

Gabriel de Souza P Moreira, Ronay Ak, Benedikt Schif-
ferer, Mengyao Xu, Radek Osmulski, and Even
Oldridge. 2024. Enhancing Q&A Text Retrieval with
Ranking Models: Benchmarking, fine-tuning and de-
ploying Rerankers for RAG.arXiv:2409.07691.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo
Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan,
Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022.
Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing.
Mykhailo Poliakov and Nadiya Shvai. 2025. Multi-
Meta-RAG: Improving RAG for Multi-hop Queries
Using Database Filtering with LLM-Extracted Meta-
data. InInformation and Communication Technolo-
gies in Education, Research, and Industrial Applica-
tions. Springer Nature Switzerland.
Renyi Qu, Ruixuan Tu, and Forrest Sheng Bao. 2025. Is
semantic chunking worth the computational cost? In
Findings of the Association for Computational Lin-
guistics: NAACL 2025. Association for Computa-
tional Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions
for machine comprehension of text. InProceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing, pages 2383–2392.
Vipula Rawte, Swagata Chakraborty, Agnibh Pathak,
Anubhav Sarkar, S.M Towhidul Islam Tonmoy,
Aman Chadha, Amit Sheth, and Amitava Das. 2023.
The Troubling Emergence of Hallucination in Large
Language Models - An Extensive Definition, Quan-
tification, and Prescriptive Remediations. InProceed-
ings of the 2023 Conference on Empirical Methods
in Natural Language Processing, pages 2541–2573.
Stephen E. Robertson, Steve Walker, Susan Jones,
Micheline Hancock-Beaulieu, and Mike Gatford.
1994. Okapi at TREC-3. InProceedings of the Third
Text REtrieval Conference, pages 109–126. NIST
Special Publication.
Sara Rosenthal, Avirup Sil, Radu Florian, and Salim
Roukos. 2025. CLAPnq: Cohesive Long-form An-
swers from Passages in Natural Questions for RAG
systems.Transactions of the Association for Compu-
tational Linguistics, 13:53–72.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxi-
ang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang,
Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.
RAGChecker: A Fine-grained Framework for Di-
agnosing Retrieval-Augmented Generation. InAd-
vances in Neural Information Processing Systems,
volume 37, pages 21999–22027.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An Automated Evalua-
tion Framework for Retrieval-Augmented GenerationSystems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354.
Krystian Safjan. 2023. From Fixed-Size to NLP Chunk-
ing – A Deep Dive into Text Chunking Techniques.
Accessed: 2025-05-01.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval Augmentation
Reduces Hallucination in Conversation. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2021.
Maojia Song, Shang Hong Sim, Rishabh Bhardwaj,
Hai Leong Chieu, Navonil Majumder, and Soujanya
Poria. 2025. Measuring and Enhancing Trustworthi-
ness of LLMs in RAG through Grounded Attribu-
tions and Learning to Refuse. InThe Thirteenth In-
ternational Conference on Learning Representations.
Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, and Ji-
awei Han. 2025. DynamicRAG: Leveraging Out-
puts of Large Language Model as Feedback for Dy-
namic Reranking in Retrieval-Augmented Genera-
tion.arXiv:2505.07233.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT Good at Search?
Investigating Large Language Models as Re-Ranking
Agents. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 14918–14937.
Pranav Narayanan Venkit, Philippe Laban, Yilun
Zhou, Yixin Mao, and Chien-Sheng Wu. 2024.
Search Engines in an AI Era: The False Promise
of Factual and Verifiable Source-Cited Responses.
arXiv:2410.22349.
Prashant Verma. 2025. S2 Chunking: A Hy-
brid Framework for Document Segmentation
Through Integrated Spatial and Semantic Analysis.
arXiv:2501.05485.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le,
Ed H. Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations.
Zilong Wang, Zifeng Wang, Long Le, Steven Zheng,
Swaroop Mishra, Vincent Perot, Yuwei Zhang,
Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-
Yu Lee, and Tomas Pfister. 2025. Speculative RAG:
Enhancing retrieval augmented generation through
drafting. InThe Thirteenth International Conference
on Learning Representations.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
11

Advances in Neural Information Processing Systems,
volume 35, pages 24824–24837.
Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, An-
drás György, Adam Fisch, Arnaud Doucet, Iuliya Be-
loshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba
Szepesvári, Ali Taylan Cemgil, and Nenad Tomasev.
2024. Mitigating LLM Hallucinations via Confor-
mal Abstention.arXiv:2405.01563.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024. CRAG - Comprehensive RAG
Benchmark. InAdvances in Neural Information Pro-
cessing Systems, volume 37, pages 10470–10490.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380.
Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebas-
tian Laverde, and Renyu Li. 2024. Financial Report
Chunking for Effective Retrieval Augmented Gener-
ation.arXiv:2402.05131.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2024a. Evaluation of retrieval-
augmented generation: A survey. InCCF Conference
on Big Data, pages 102–120. Springer.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024b. RankRAG: Unifying Context Ranking
with Retrieval-Augmented Generation in LLMs. In
Advances in Neural Information Processing Systems,
volume 37.
Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian
Qi, Simin Niu, Bo Tang, Feiyu Xiong, and
Zhiyu Li. 2024. Meta-Chunking: Learning Ef-
ficient Text Segmentation via Logical Perception.
arXiv:2410.12788.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
Maosong Sun. 2024. RAGEval: Scenario Specific
RAG Evaluation Dataset Generation Framework.
arXiv:2408.01262.
12

A RAG Pipeline and Dataset Details
This section outlines the configurable components
of our RAG pipeline and provides details on the
specific configurations used to obtain the experi-
mental results presented in the main paper. In Ta-
ble 4, we list important hyperparameters used on
the DragonBall and CLAPnq datasets. Note that
the DragonBall dataset is publicly available under
a CC BY-NC-SA 4.0 license. The CLAPnq dataset
is publicly available under an Apache 2.0 license.
Chunker.We implemented multiple chunking
strategies, including fixed-length chunking (with
and without overlap) and recursive chunking at the
sentence level. These methods enable flexible pre-
processing of documents depending on the context
granularity needed. When available, our implemen-
tation also handles pre-defined semantic chunks
provided by the dataset. For instance, the CLAPnq
dataset provides pre-chunked documents aligned
with natural discourse boundaries, which we use
directly in our experiments. For the DragonBall
dataset, we follow the original paper’s best configu-
ration and apply fixed-length chunking with a win-
dow size of 128 tokens and an overlap of 25 tokens
(Zhu et al., 2024).
Retrieval.For dense retrieval, we used mod-
els from the Hugging Face repository (e.g.,
gte-large-en-v1.5 ) to generate embeddings of
the queries and chunks. Retrieval computes vec-
tor similarity (using cosine similarity) between the
query and document chunks and selects the top- k
most relevant candidates. The value of kis treated
as a hyperparameter. We fixed k= 8 andk= 5 for
the DragonBall and CLAPnq datasets respectively.
Reranker.For the reranking stage, we use spe-
cialized LLM reranker models fine-tuned for rank-
ing tasks (e.g., rank-zephyr-7b-v1-full ), also
sourced from Hugging Face. Given a set of re-
trieved candidate chunks and the original query,
this model assigns a relevance score to each chunk.
We then select the top k′candidates (with k′< k
from retrieval) to ensure that only the most rele-
vant context is passed to the generator. The rerank-
ing step plays a critical role in filtering noisy or
marginally relevant content returned by dense re-
trieval. This two-stage process is meant to enhance
retrieval precision, thereby improving final answer
quality. We fixed k′= 5andk′= 3for the Drag-
onBall and CLAPnq datasets respectively. The sys-
tem prompt is reproduced in Listing 1.Table 4: RAG Implementation Hyperparameters
Parameter DragonBall CLAPnq
Chunker Fixed-Length Semantic
size=128, overlap=25
Embedder gte-large-en-v1.5 gte-large-en-v1.5
Search Top-k=8 Top-k=5
Reranker rank-zephyr-7b-v1-full rank-zephyr-7b-v1-full
Top-k′=5 Top-k′=3
Generator Llama-3-8B-Instruct Llama-3-8B-Instruct
"""
System:
You are RankLLM, an intelligent
assistant that can rank passages
based on their relevancy to the
query.
User:
I will provide you with {{K}} passages,
each indicated by a numerical
identifier []. Rank the passages
based on their relevance to the
search query: {{query}}
[1] {{chunk1}}
[2] {{chunk2}}
...
[K] {{chunkK}}
Search Query: {{query}}.
Rank the {{K}} passages above based on
their relevance to the search query.
All the passages should be included
and listed using identifiers, in
descending order of relevance. The
output format should be [] > [], e.g
., [2] > [1]. Only respond with the
ranking results, do not say any word
or explain.
Assistant:
"""
Listing 1: System Prompt Used for our RAG Reranker
Generator.The final stage involves generating
answers using an LLM. The reranked top k′chunks
are concatenated and provided as context to the
model alongside the query. Note that only the re-
trieved chunks (not full documents) are used as
context. Our implementation is modular enough
so that one can experiment with several LLMs
from Hugging Face and select the one that best
suits the needs. In our experiments, we used
Meta-Llama-3-8B-Instruct . The system prompt
is reproduced in Listing 2.
"""
System:
You are an assistant for answering
queries. You are given a list of
context (extracted parts of some
documents) and a query. Based on the
13

given context, provide an answer to
the query. Please be concise and to
the point. If you don't know the
answer say'I don't know!'Don't
make up an answer. Cite the document
id used. The output format should
be answer with citations. Only
respond with the answer, do not
explain.
User:
Query:
Where is the capital of France?
Context:
{document id: 1, content: Paris is the
capital of France, the largest
country of Europe with 550 000 km2)
.}{document id: 2, content: France
is a country in Europe.}
Assistant:
Answer: Paris. [1]
User:
Query:
{{query}}
Context:
{{Context}}
"""
Listing 2: System Prompt Used for our RAG Generator
B Automatic Error Classification Details
This section provides additional materials that sup-
port our main auto-evaluation analysis from sec-
tion 5. We begin with an extended description of
our auto-evaluation pipeline, RAGEC, including
details such as the system prompts we used. Ad-
ditionally, we present alternative auto-evaluation
methods which empirically had lower agreement
with our human annotations.
B.1 Answer Evaluation System Prompts
The first step of RAGEC, answer evaluation uses
an LLM to determine if a generated response is
correct, given the query, and ground-truth answer
and documents. In practice, we used GPT-4o with
the prompt in Listing 3.
"""
You are an expert evaluator. Your
task is to evaluate if a
proposed answer matches the
ground truth answer for a given
question.
You will be provided with
information between special tags
:
1. The original question <question>
2. The ground truth (correct) answer
<ground_truth>3. A proposed answer to evaluate <
proposed_answer>
4. (Optional) the proposed answer's
cited information <
ground_truth_citations>
Please evaluate the proposed answer
based on the following criteria:
- Abstain: If the proposed answer is
"I don't know" or "I don't have
enough information to answer
this question", then the label
is'abstain'.
- Accuracy: Does it contain the same
key information as the ground
truth?
- Completeness: Does it cover all
important points from the ground
truth?
- Correctness: Are there any factual
errors compared to the ground
truth?
Provide your evaluation as a JSON
object with the following fields
:
{{
\"label\": \"correct\" | \"
possible_correct\" | \"
incorrect\" | \"abstain\",
\"reasoning\": string // A very
brief explanation of your
evaluation
}}
Here are some examples:
<...>
<question>
{question}
</question>
<ground_truth>
{ground_truth}
</ground_truth>
<proposed_answer>
{proposed_answer}
</proposed_answer>
{citations}
Your evaluation:
"""
Listing 3: System Prompt Used for Answer Evaluation
B.2 Error Stage Classification
This section describes the algorithm for the identi-
fication of which stage in the RAG pipeline caused
the error.
B.2.1 Determining the ground truth chunks
The first step is to determine which chunks contain
the ground truth information from the corpus. For
14

CLAPnq, the dataset provides short ground truth
chunks along with each query; thus we use the
dataset’s chunks directly.
For DragonBall, only the document containing
the ground truth is given with the query, but these
documents are long, so we perform the chunk-
ing ourselves. We leverage an LLM to determine
which chunks are necessary to answer the query.
We provide GPT-4o-mini the query, ground truth
answer, and all the chunks from the ground truth
documents, and ask for the IDs of chunks that con-
tain information necessary to answer the query. We
perform this process 10 times and select chunks ap-
pearing more than 8 times as ground truth chunks.
The repetition accounts for the stochasticity of us-
ing an LLM, and better reflects its confidence. The
prompt is shown below in Listing 4.
"""
You are an expert evaluator for
Retrieval Augmented Generation (RAG)
systems. You will help identify
chunking-related errors in RAG
systems by analyzing the
relationship between queries, ground
truth answers, and document chunks.
Your task is to identify which
chunks from the ground truth
document are relevant to answering
the query.
## Instructions
1. Review ALL available chunks from the
ground truth document
2. Identify ALL chunks containing
information relevant to answering
the query
3. Consider both direct and indirect
relevance to the query
4. Select chunks that together provide
sufficient information to answer the
query
## Context
**Query**: {query}
**Ground Truth Answer**: {ground_truth}
**Available Chunks from Ground Truth
Document**:
{chunks}
## Output Format
Provide your answer in the following
format:
Relevant Chunks: [45_1, 45_4, 45_10]
"""
Listing 4: System prompt for determining the chunks
containing the ground truth on the DragonBall datasetB.2.2 Determining errors in the generation
stage
Given the ground truth chunks, if we know that the
chunks being passed to the generator are sufficient
to determine the answer, we can conclude the error
happened in the generation stage. Thus we look at
the fraction of the ground truth chunks being passed
to the generator. However, we must account for the
possibility that multiple chunks contain the ground
truth. In these cases, using only a subset of the
ground truth chunks may be sufficient to determine
the correct answer. To this end, we specify that
if more than half of the ground truth chunks are
passed to the generator, it is deemed a generator
error.
If there are no ground truth chunks extracted,
it is likely that the error is an abstention error as
the question may be unanswerable. Thus it is also
deemed a generator error in this case.
B.2.3 Determining errors in the reranking
stage
An error occurs in the reranking stage if the
reranker filters out some ground truth chunks which
the retriever had returned. Thus if there is any
ground truth chunk that is filtered out at this stage,
we deem the error a reranker error.
Note that it could be the case that the reranker
filters out redundant chunks containing the ground
truth, as described in the above step. However,
in practice, if both chunks containing the correct
answer to the question are retrieved, it is not likely
that the reranker would treat them differently. We
found the described method to be the most effective
in determining whether there is a reranker error.
B.2.4 Distinguishing between chunking errors
and retrieval errors
Chunking errors and retrieval errors are difficult to
distinguish, as the efficacy of the retriever is tightly
coupled to the quality of chunks. Chunking error
types E1, E2, and E3 all relate to cases where im-
portant information is missing or overshadowed in
the chunks. Thus we aim to check if each “con-
cept” in each query is preserved in at least one of
the chunks containing the ground truth.
To extract the concepts in each query, we fol-
low closely the method from Matton et al. (2025).
We ask GPT-4o-mini to extract the concepts from
each query, then for each concept, we ask the LLM
which ground truth chunks contain that concept.
We then compute the fraction of all concepts which
15

are included in at least one of the ground truth
chunks. If this fraction is less than 0.8, we deem
the error to be a chunking error. Otherwise it is
deemed a retrieval error.
The parameter 0.8 was chosen to reflect that if
the number of concepts extracted from the query is
high, it is likely that not all the concepts are needed
to answer the question. Thus we allow some leeway
for the cases where the number of concepts is high
(namely greater than 6).
The prompt for determining the concepts in the
query is given in Listing 5, and for determining
whether a concept is contained in the ground truth
chunks is given in Listing 6.
"""
Consider the following questions. Your
task is to list the set of distinct
concepts, or high-level pieces of
information, in the'Context'that
could possibly influence someone's
answer to the question. Each concept
should appear word-to-word in the
question, or a very minor rewording.
Here are three examples.
Example 1:
Question:
Compare the debt restructuring efforts
of Company A in 2018 and Company B
in 2021. Which company reduced more
liabilities through debt
restructuring?
Concept List:
Debt restructuring efforts
Company A
2018
Company B
2021
Reducing liabilities through debt
restructuring
{examples 2 and examples 3}
Please fill out the'Concept List'for
the fourth example by providing a
numbered list. You should not
restate the'Concept List'header.
You should not put dash ('-') or
numbers before each item in the list
.
Example 4
{query}
"""
Listing 5: System prompt used for determining the
concepts in each query
"""
You will be given a list of excerpts and
a concept. Your job is to determine
whether the concept given is
contained in each excerpt. Output a
line for excerpt, output "True" or "
False".Example 1:
**Concepts**: Peter
**Excerpts**:
[45_3] John is running
[45_6] Peter is walking
**Answer**:
[45_3] False
[45_6] True
Example 2:
**Concepts**: running
**Excerpts**:
[45_3] John is running
[45_6] Peter is walking
**Answer**:
[45_3] True
[45_6] False
Question:
**Concepts**: {concept}
**Excerpts**:
{excerpts}
**Answer**:
"""
Listing 6: System prompt used for determining whether
a concept is contained in the ground truth chunks
B.3 Error Type Classification
After determining the RAG stage responsible for
the error, we then prompt GPT-4o-mini K= 10
times to judge the error type, and use the distribu-
tion of results to evaluate the LLM’s confidence.
Each stage uses a different system prompt and dif-
ferent contextual information relevant to that stage.
B.3.1 Chunking error type
If stage classification determined the error to be
a chunking error, then classifying which type of
chunking error only requires the query, ground
truth answer, and chunks from the ground truth doc-
ument. The prompt for chunking errors is given in
Listing 7.
"""
You are an expert Retrieval Augmented
Generation (RAG) system evaluator.
Background:
Given a query, and a list of documents
possibly containing the answer to
the query, a RAG model has tried to
answer the query using the following
4 steps.
- Chunking: Each document in the
document list is split into smaller
chunks.
- Retrieval: A specified number of
chunks closest to the query will be
retrieved.
- Reranking: The retrieved chunks are
ranked for the second time,
according to how relevant the chunks
are to the query. Then, only a
16

specified number of reranked chunks
are retained.
- Generation: The query and reranked
chunks are passed to the generator
LLM to generate an answer.
You will be given a query and the ground
truth answer, as well as all the
chunks that belong to the document
containing the ground truth answer.
As described in the background, each
chunk will be seen as independent
text blocks by the RAG model. Given
there is an error in the chunking
step, your job is to determine the
best description of the error from
the list below.
- Overchunking: Document is split into
excessively small chunks, causing
important context to be lost.
Individual chunks appear incomplete
or ambiguous.
- Underchunking: Chunks are too large,
covering multiple topics with mixed
content. Individual chunks can be
confusing and crucial information is
diluted.
- Context Mismatch: Chunks are split at
arbitrary boundaries, disrupting the
logical structure of the document.
Key contextual links are separated
from the information they link to
Context:
**Query**: {query}
**Ground Truth**: {ground_truth}
**Document Chunks**:
{ground_truth_chunks}
**Output format**:
Your answer should only contain one of
the following,
Overchunking, Underchunking, Context
Mismatch
"""
Listing 7: System prompt used for determining
chunking error types
B.3.2 Retrieval error type
For retrieval errors, the LLM only needs the query,
the ground truth answer, the ID for the ground truth
documents, and the retrieved chunks. The prompt
for retrieval errors is given in Listing 8.
"""
You are an expert Retrieval Augmented
Generation (RAG) system evaluator.
Background:
Given a query, and a list of documents
possibly containing the answer to
the query, a RAG model has tried to
answer the query using the following
4 steps.
- Chunking: Each document in the
document list is split into smaller
chunks.- Retrieval: A specified number of
chunks closest to the query will be
retrieved.
- Reranking: The retrieved chunks are
ranked for the second time,
according to how relevant the chunks
are to the query. Then, only a
specified number of reranked chunks
are retained.
- Generation: The query and reranked
chunks are passed to the generator
LLM to generate an answer.
You will be given a query and the ground
truth answer. You will also be
given IDs of the documents
containing the ground truth and a
selection of document chunks
retrieved.
Given there is an error in the retrieval
step, your job is to determine the
best description of the error from
the list below.
- Missed Retrieval: Retrieved chunks do
not contain the relevant information
to answer the query from the ground
truth documents
- Low Relevance: Retrieved chunks are
only loosely related to the query
- Semantic Drift: Retrieved chunks
appear to match keywords but do not
align with the query's intent
Context:
**Query**: {query}
**Ground Truth**: {ground_truth}
**Ground Truth Document ID**: {
ground_truth_doc_ids}
**Retrieved Chunks**:
{retrieved_chunks}
**Output format**:
Your answer should only contain one of
the following:
Missed Retrieval, Low Relevance,
Semantic Drift
"""
Listing 8: System prompt used for determining retrieval
error types
B.3.3 Reranking error type
To determine the reranking error type, the LLM
only needs the query, the ground truth answer, the
retrieved chunks, and the reranked chunks. The
prompt for reranking error is given in Listing 9.
"""
You are an expert Retrieval Augmented
Generation (RAG) system evaluator.
Background:
Given a query, and a list of documents
possibly containing the answer to
the query, a RAG model has tried to
answer the query using the following
4 steps.
- Chunking: Each document in the
document list is split into smaller
17

chunks.
- Retrieval: A specified number of
chunks closest to the query will be
retrieved.
- Reranking: The retrieved chunks are
ranked for the second time,
according to how relevant the chunks
are to the query. Then, only a
specified number of reranked chunks
are retained.
- Generation: The query and reranked
chunks are passed to the generator
LLM to generate an answer.
You will be given a query and the ground
truth answer. You will also be
given a selection of document chunks
retrieved. The retrieved chunks
will be reranked so that only {
num_reranked_chunks} chunks are
further selected. Given there is an
error in the reranking step, your
job is to determine the best
description of the error from the
list below.
- Low Recall: Necessary chunks are
retrieved but reranked too low and
not forwarded to the generator
- Low Precision: Irrelevant chunks are
reranked highly and forwarded to the
generator, with greater importance
than the truly relevant chunks
Context:
**Query**: {query}
**Ground Truth**: {ground_truth}
**Ground Truth Document ID**: {
ground_truth_doc_ids}
**Reranked Chunks**:
{reranked_chunks}
**Output format**:
Your answer should only contain one of
the following:
Low Recall, Low Precision
"""
Listing 9: System prompt used for determining
reranking error types
B.3.4 Generation error type
To determine the generation error type, the LLM
only needs the query, the ground truth answer, in-
correct answer by RAG model and the reranked
chunks. The prompt for generation error types is
as follows.
"""
You are an expert Retrieval Augmented
Generation (RAG) system evaluator.
Background:
Given a query, and a list of documents
possibly containing the answer to
the query, a RAG model has tried to
answer the query using the following
4 steps.- Chunking: Each document in the
document list is split into smaller
chunks.
- Retrieval: A specified number of
chunks closest to the query will be
retrieved.
- Reranking: The retrieved chunks are
ranked for the second time,
according to how relevant the chunks
are to the query. Then, only a
specified number of reranked chunks
are retained.
- Generation: The query and reranked
chunks are passed to the generator
LLM to generate an answer.
You will be given a query, the ground
truth answer and an incorrect answer
to the query generated by the RAG
model. You will also be given {
num_reranked_chunks} document chunks
. Your job is to determine the
reason why the model outputs the
incorrect answer, from the list
below.
- Abstention Failure: The model should
have abstained but provided an
incorrect answer
- Fabricated Content: The response
includes information not present in
the source document chunks and is
unverifiable
- Parametric Overreliance: The response
depends on the LLM's internal
knowledge rather than the source
document chunks
- Incomplete Answer: The response is
correct but missing critical details
- Misinterpretation: The generator
misuses or misrepresents the source
document chunks
- Contextual Misalignment: The response
is correct but does not address the
query
- Chronological Inconsistency: The
response presents events or facts in
the wrong temporal order, or
confuses past, present, or future
timelines
- Numerical Error: The response includes
incorrect calculations, quantities,
or misrepresents numerical data
from the retrieved documents
Context:
**Query**: {query}
**Ground Truth**: {ground_truth}
**Incorrect Answer**: {
incorrect_rag_answer}
**Reranked Chunks**:
{reranked_chunks}
**Output format**:
Your answer should only contain one of
the following:
Abstention Failure, Fabricated Content,
Parametric Overreliance, Incomplete
Answer, Misinterpretation,
Contextual Misalignment,
Chronological Inconsistency,
18

Numerical Error
"""
Listing 10: System prompt psed for determining
generation error types
B.4 Alternative Auto-evaluation Approaches
So far this appendix has described in detail
RAGEC, our proposed error classification method.
While designing RAGEC, we explored many other
systems for LLM-based error classification, but
RAGEC empirically had the best agreement rates
with our human-annotated data. For comparison, in
this section we describe two alternative approaches
to error classification that we tested. For each alter-
native, we used GPT-4o-mini, the same model as
RAGEC above for fair comparison.
B.4.1 Single-step error type classification
The most straightforward approach to error type
classification is to directly prompt an LLM to out-
put one of the 16 defined error types given all avail-
able contextual information, including the query,
ground truth response from the dataset, generated
response, retrieved chunks, and reranked chunks.
Consistently with other methods, the 16 error types
are defined and described in the system prompt. We
apply this method directly after theanswer evalua-
tionstep of RAGEC, such that the same set of 832
potential errors from DragonBall are annotated.
Compared to RAGEC which achieved 57.8%
stage classification agreement, and 40.3% error-
type classification agreement, single-step prompt-
ing had only 41.1% and 31.1% agreement rates, re-
spectively. Note that stage classification is done
implicitly, taking the stage of the modal error type.
Single-step prompting overloads the LLM with in-
formation which must be digested and synthesized
in a single shot. Breaking the classification up by
stage helps to guide the LLM’s reasoning towards a
smaller set of possibilities, potentially with a more
focused collection of contextual information.
B.4.2 Stage-sequential error type classification
As above, we begin with theanswer evaluation
stage of RAGEC to identify which RAG responses
need to be classified. After this, we proceed se-
quentially over the stages of the RAG pipeline in
order, starting with chunking. At each stage, an
LLM is prompted with context to determine if an
error occurred at this stage, and if so, which of the
error types occurred within that stage. Once the
earliest error is identified, the evaluation stops.In more detail, for the chunking stage, the con-
text contains the query, ground truth response from
the dataset, and all chunks from the ground truth
document. Notably, we do not include information
like the generated response or retrieved chunks,
because these are not available during the chunk-
ing step and hence are not relevant to determining
chunking errors. Next, for the retrieval stage we
provide the query and ground truth response, but
now include the retrieved chunks and the ground
truth document ID. Continuing to reranking if nec-
essary, the retrieved chunks are replaced by the
reranked chunks. Because of the sequential nature
of the pipeline, retrieved chunks are not strictly nec-
essary here. Finally, if no error is identified in the
first three stages, we assume the generator caused
the error by process of elimination. The LLM is
prompted to classify the error type given the query,
ground truth response, generated response, and
reranked chunks.
Stage-sequential error type classification per-
formed better than single-step with 47.5% stage
classification agreement, and 36.3% error-type clas-
sification, but this still falls short of RAGEC by
a considerable margin. We analysed the perfor-
mance of stage-sequential error type classification
compared to human annotations and found that it
lacked the ability to correctly classify any chunk-
ing errors. Without context on the generated re-
sponse or how chunks were used downstream in
the RAG pipeline, determining that chunking was
the primary culprit is extremely difficult. This in-
formed how we designed RAGEC, which also it-
erates over stages sequentially, but in a backward
fashion, starting from generation and proceeding
back to chunking. We also provided RAGEC more
information for the chunking stage by generating
ground truth chunk labels with concept extraction.
C Additional Results
In this section we present extended results on the
co-occurrence patterns of error types, which offer
further insight into the structure and dependencies
among different failure modes observed in RAG
systems beyond what was discussed in Section 5.
C.1 CLAPnq Error Type Distribution
Dataset characteristics can influence the failure
modes of RAG systems. In Table 5, we present the
distribution of the most and second most frequent
error types as identified by RAGEC on the CLAPnq
19

Table 5: Distribution of most and second most common
error types across the CLAPnq dataset.
Error Classification Mode Second Mode
E1 Overchunking 0 0
E2 Underchunking 0 0
E3 Context Mismatch 0 0
Percentage Chunking 0% 0%
E4 Missed Retrieval 180 69
E5 Low Relevance 84 99
E6 Semantic Drift 1 16
Percentage Retrieval 21.69% 26.10%
E7 Low Recall 23 1
E8 Low Precision 1 3
Percentage Reranking 1.96% 0.57%
E9 Abstention Failure 9 23
E10 Fabricated Content 87 125
E11 Parametric Overreliance 46 98
E12 Incomplete Answer 51 38
E13 Misinterpretation 607 136
E14 Contextual Misalignment 112 87
E15 Chronological Inconsistency 4 6
E16 Numerical Error 17 4
Percentage Generation 76.35% 73.33%
Total 1222 705
dataset. Compared to the DragonBall-EN dataset
(see Table 2 in Section 5), the CLAPnq error distri-
bution is much more heavily weighted towards gen-
eration errors, with few reranking errors. The lack
of chunking errors is because the dataset comes
pre-chunked with ground truth properly included.
C.2 DragonBall-CN Error Type Distribution
To demonstrate that our method extends to lan-
guages other than English, we tested our method
on the Chinese subset of the DragonBall dataset.
Note that the documents and the queries are in Sim-
plified Chinese. As in Table 4, the parameters used
for chunking, embedding, retrieval, reranking, and
generation are all the same as DragonBall-EN, with
the exception that chunking uses 512 characters in-
stead of 128 tokens, which is more suitable for the
structure of Chinese. We also express the generator
prompt in Chinese, as listed below:
在此任务中，你需要总结出回答问题所必需
的Key Points，并使用这些Key Points来帮助
你回答问题。
请按以下格式列出Key Points：
1. ...
2. ...
依此类推，根据需要增加序号,但不超
过10个。
每个Key Points必须附带引用来源。随后，利
用这些Key Points，在“Answer:”之后生成最
终答案。
最终答案中也要引用来源。Table 6: Distribution of most and second most common
error types across the DragonBall-CN dataset.
Error Classification Mode Second Mode
E1 Overchunking 1 27
E2 Underchunking 8 75
E3 Context Mismatch 185 9
Percentage Chunking 15.11% 18.32%
E4 Missed Retrieval 296 19
E5 Low Relevance 25 89
E6 Semantic Drift 0 6
Percentage Retrieval 25.00% 18.81%
E7 Low Recall 378 19
E8 Low Precision 21 155
Percentage Reranking 31.07% 28.71%
E9 Abstention Failure 16 15
E10 Fabricated Content 93 57
E11 Parametric Overreliance 0 7
E12 Incomplete Answer 3 5
E13 Misinterpretation 213 77
E14 Contextual Misalignment 30 42
E15 Chronological Inconsistency 4 1
E16 Numerical Error 11 3
Percentage Generation 28.82% 34.16%
Total 1284 705
确保每个Key Point覆盖不同的内容。
所有答案必须使用中文进行回答。
示例问题：
新修订的《公司法》有哪些重大变化？
示例回答：
Key Points:
1.修订强化了对公司治理的监管，明确了董
事会和监事会的具体职责。[12-1]
2.引入了强制性的ESG报告披露要求。[14-2]
3.调整了公司资本制度，降低了最低注册资
本要求。[12-1]
4.为 中小企业引入了 专项扶持措
施。[15-1][13-2]
Answer:
2023年修订的《公司法》引入了几项重大变
化。首先，此次修订强化了公司治理的监管，
具体明确了董事会和监事会的职责。[12-1]其
次，引入了强制性的环境、社会及公司治理
（ESG）报告披露要求。[14-3]此外，修订还
调整了公司资本制度，降低了最低注册资本要
求。[12-2]最后，此次修订为中小企业引入了
专项扶持措施，以促进其发展。[14-2][12-1]
C.3 Error-types Co-occurrence Analysis
To understand whether certain error types tend to
appear together, we analyze the relationship be-
tween the mode and second most frequent error
types for each example for each of the RAG stages.
Figure 3 shows the co-occurrence counts be-
tween generation error types for DragonBall-EN
and CLAPnq. We observe thatMisinterpretation
20

Figure 3: Joint distribution of first and second mode error categories for the generation stage.Left:Dragonball-EN
Right:CLAPnq
(E13) often co-occurs withFabricated Content
(E10) andContextual Misalignment(E14). In-
terestingly, for CLAPnq,Misinterpretation(E13)
also frequently co-occurs withParameter Overre-
liance(E11), whereas this pattern is not observed
in DragonBall-EN. This can be explained by the
fact that CLAPnq queries are natural questions that
may exist in the model’s parametric knowledge ac-
quired during pre-training, while DragonBall con-
tains domain-specific queries that are unlikely to
appear in the pre-training data.
Despite both datasets exhibiting significant χ2
test results, we observe a notable difference in the
density of their error co-occurrence patterns. The
contingency table for the DragonBall-EN dataset
is considerably sparser, indicating that error types
tend to co-occur in fewer, more specific combina-
tions. In contrast, the CLAPnq dataset presents a
denser co-occurrence matrix, with a wider variety
of error type pairs appearing more frequently. This
difference may reflect that CLAPnq surfaces more
complex and intertwined error patterns, reinforc-
ing the need for fine-grained diagnostic evaluation
across diverse RAG applications.
C.4 Detailed Error Classification Results
Detailed error classification results comparing
RAGEC to human annotations are shown in Fig-
ure 4. For RAGEC, we show the most frequent
error type out of the K= 10 repeated LLM calls
per query. Note that humans sometimes annotated
more than one error type for a single query (though
only from a single stage), and did not indicate an
ordering between types. In such cases, each humanannotated error type appears in the counts in the fig-
ure. We find that some specific error types are pre-
ferred by humans or RAGEC, such asMissed Re-
trieval(E4) which was heavily used by humans, or
Context Mismatch(E3) which was overrepresented
in the RAGEC labels.
21

Figure 4: Joint distribution of RAGEC auto-evaluation error type classifications and human annotations for
DragonBall-EN
22
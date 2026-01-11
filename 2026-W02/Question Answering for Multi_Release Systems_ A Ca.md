# Question Answering for Multi-Release Systems: A Case Study at Ciena

**Authors**: Parham Khamsepour, Mark Cole, Ish Ashraf, Sandeep Puri, Mehrdad Sabetzadeh, Shiva Nejati

**Published**: 2026-01-05 18:44:26

**PDF URL**: [https://arxiv.org/pdf/2601.02345v1](https://arxiv.org/pdf/2601.02345v1)

## Abstract
Companies regularly have to contend with multi-release systems, where several versions of the same software are in operation simultaneously. Question answering over documents from multi-release systems poses challenges because different releases have distinct yet overlapping documentation. Motivated by the observed inaccuracy of state-of-the-art question-answering techniques on multi-release system documents, we propose QAMR, a chatbot designed to answer questions across multi-release system documentation. QAMR enhances traditional retrieval-augmented generation (RAG) to ensure accuracy in the face of highly similar yet distinct documentation for different releases. It achieves this through a novel combination of pre-processing, query rewriting, and context selection. In addition, QAMR employs a dual-chunking strategy to enable separately tuned chunk sizes for retrieval and answer generation, improving overall question-answering accuracy. We evaluate QAMR using a public software-engineering benchmark as well as a collection of real-world, multi-release system documents from our industry partner, Ciena. Our evaluation yields five main findings: (1) QAMR outperforms a baseline RAG-based chatbot, achieving an average answer correctness of 88.5% and an average retrieval accuracy of 90%, which correspond to improvements of 16.5% and 12%, respectively. (2) An ablation study shows that QAMR's mechanisms for handling multi-release documents directly improve answer accuracy. (3) Compared to its component-ablated variants, QAMR achieves a 19.6% average gain in answer correctness and a 14.0% average gain in retrieval accuracy over the best ablation. (4) QAMR reduces response time by 8% on average relative to the baseline. (5) The automatically computed accuracy metrics used in our evaluation strongly correlate with expert human assessments, validating the reliability of our methodology.

## Full Text


<!-- PDF content starts -->

Question Answering for Multi-Release Systems:
A Case Study at Ciena
Parham Khamsepour∗†, Mark Cole†, Ish Ashraf†, Sandeep Puri†, Mehrdad Sabetzadeh∗, Shiva Nejati∗
∗University of Ottawa, 800 King Edward Avenue, Ottawa ON K1N 6N5, Canada
†Ciena Corp, 7035 Ridge Road, Hanover, MD 21076, USA
Email:{parham.khamsepour, m.sabetzadeh, snejati}@uottawa.ca;{mcole, iashraf,spuri}@ciena.com
Abstract—Companies regularly have to contend with multi-
release systems, where several versions of the same software are
in operation simultaneously. Question answering over documents
from multi-release systems poses challenges because different
releases have distinct yet overlapping documentation. Moti-
vated by the observed inaccuracy of state-of-the-art question-
answering techniques on multi-release system documents, we
propose QAMR, a chatbot designed to answer questions across
multi-release system documentation. QAMR enhances traditional
retrieval-augmented generation (RAG) to ensure accuracy in the
face of highly similar yet distinct documentation for different
releases. It achieves this through a novel combination of pre-
processing, query rewriting, and context selection. In addition,
QAMR employs a dual-chunking strategy to enable separately
tuned chunk sizes for retrieval and answer generation, improving
overall question-answering accuracy.
We evaluate QAMR using a public software-engineering
benchmark as well as a collection of real-world, multi-release sys-
tem documents from our industry partner, Ciena. Our evaluation
yields five main findings: (1) QAMR outperforms a baseline RAG-
based chatbot, achieving an average answer correctness of 88.5%
and an average retrieval accuracy of 90%, which correspond to
improvements of 16.5% and 12%, respectively. (2) An ablation
study shows that QAMR’s mechanisms for handling multi-release
documents directly improve answer accuracy. (3) Compared
to its component-ablated variants, QAMR achieves a 19.6%
average gain in answer correctness and a 14.0% average gain
in retrieval accuracy over the best ablation. (4) QAMR reduces
response time by 8% on average relative to the baseline. (5) The
automatically computed accuracy metrics used in our evaluation
strongly correlate with expert human assessments, validating the
reliability of our methodology.
Index Terms—Question Answering, Large Language Models,
Retrieval-Augmented Generation, Multi-Release Systems.
I. INTRODUCTION
Systems and software engineering processes rely on tech-
nical documents, such as requirements specifications, design
documents, and installation guides. These documents support
diverse stakeholders, such as end-users, product managers,
software engineers, system architects, and customer support
teams, in understanding system functions, dependencies, and
carrying out their roles.
Companies often maintain multiple versions of their sys-
tems, with each version evolving from its predecessors to meet
client needs, support legacy users, comply with regulations,
or enable phased rollouts [1]. Such collections of coexisting
versions are referred to asmulti-release systems. Unlike a
software product line [2] – a planned family of relatedproducts that coexist without forming sequential stages of
evolution – multi-release systems progress version by version.
Each version typically has its own documentation, capturing
updates, improvements, and configuration changes specific to
that release while retaining much of the earlier content.
Large Language Models (LLMs) are now commonly used
to develop question-answering chatbots in software and sys-
tems engineering. These chatbots help stakeholders search
large document sets and get real-time answers to questions.
Among others, Abdellatif et al. [3], [4], Daniel and Cabot [5],
Chaudhary et al. [6], and Huang et al. [7] examine the use of
chatbots in software engineering, both during development and
operation, showing their potential to enhance communication
and support tasks such as bug tracking, documentation, code
generation, and runtime assistance.
One of the most widely used architectures for modern chat-
bots is retrieval-augmented generation (RAG). RAG integrates
document retrieval with LLMs to produce more accurate and
contextually relevant responses [8], [9], [10], [11]. Applying
RAG to documents associated with multiple releases of the
same system presents two challenges:(C1)RAG’s retrieval
accuracy declines due to the coexistence of numerous versions
with distinct yet overlapping content. For example, when a
query pertains to a system property that varies across releases,
the retrieval component of RAG may get confused, extracting
inconsistent information.(C2)Standard RAG cannot effec-
tively prioritize the most relevant information in the context
of multi-release systems. Here, not all documents or their
parts are necessarily linked to specific releases. To generate
accurate answers, a query needs to be reinterpreted into more
specific ones – some tied to releases, and others independent of
releases. This reinterpretation results in multiple sets of texts
to be retrieved by RAG, but not all of these texts are equally
relevant. A strategy is therefore needed to keep only the
relevant texts retrieved by the different query interpretations.
In this paper, we propose QAMR, a novel RAG-based
QuestionAnswering chatbot forMulti-Release systems that
addresses the above challenges. Our approach includes a
pre-processing step in which documents are indexed into a
separate corpus for each release, thereby preventing RAG
from mixing information from different releases. Using a
prompting technique, QAMR transforms a given user query
into multiple standalone versions that fully incorporate the
conversational history. Unlike traditional RAG, which retrievesarXiv:2601.02345v1  [cs.SE]  5 Jan 2026

a single set of texts, QAMR retrieves multiple sets, each using
a rewritten standalone query. To use only the most relevant
texts for answering the original user query, QAMR employs a
state-of-the-art LLM-based ranking strategy, RankRAG [12].
Instead of using all retrieved documents for answer generation,
RankRAG uses an LLM to rank and select the texts most con-
textually relevant to the input query before answer generation.
In addition to supporting multi-release systems, QAMR
mitigates a known limitation of RAG in software-engineering
applications [11], [6]: the sensitivity to chunk size – that is, the
size of the retrieved texts. Small chunks often improve retrieval
precision but provide limited context for answer generation,
whereas large chunks supply more context but tend to lower
retrieval accuracy. Instead of relying on a single compromise
size, QAMR employs adual-chunkingstrategy, using smaller
chunks for retrieval and larger ones for answer generation.
Contributions. (1)We introduce, QAMR, a RAG-based chat-
bot for question answering over multi-release systems. QAMR
can process structured content such as tables and schematics,
and it generates multiple interpretations of user queries to
improve retrieval accuracy. It employs RankRAG to identify
contextually relevant information for response generation. In
addition, QAMR implements a dual-chunking strategy: it uses
smaller text chunks during retrieval and larger chunks during
answer generation. This design avoids the trade-offs inherent
in traditional RAG systems that rely on a single chunk size.
(2)We evaluate QAMR on an industry dataset from Ciena
and on an open-source requirements set, REQuestA [13]. We
compare QAMR against a state-of-the-art baseline and conduct
an ablation study to isolate the effect of each component
of QAMR. To do so, we report sixLLM-as-Judgemetrics
focused on answer correctness and retrieval accuracy [14],
[15], along with response time. Our use of LLM-as-Judge
metrics is motivated by the limitations of both manual evalua-
tion and traditional word-overlap metrics such as BLEU [16],
ROUGE [17], and cosine similarity: Manual evaluation, while
capable of nuanced judgment, is difficult to scale because it
requires checking a large number of answers generated by
a proposed chatbot, its variants, and baselines, as well as the
results of repeated runs to account for randomness. Automated
metrics such as BLEU, ROUGE, and cosine similarity provide
only shallow word-level matches and fail to capture the
human-like qualities required to effectively evaluate LLM-
generated texts; prior studies already show the inadequacy of
these metrics in this regard [6], [18], [19], [20].
We implement two measures to ensure that LLM-computed
metrics serve as reliable proxies for expert human judgment.
First, our LLM-as-Judge framework, created in collaboration
with Ciena’s subject-matter experts, uses prompts that closely
reflect human evaluators’ reasoning when assessing response
quality in multi-release systems. Second, we validate the
LLM-generated metrics against evaluations by Ciena’s domain
experts, who manually assess the accuracy of responses from
an individual run of QAMR on the full industrial dataset.
Our analysis shows a strong correlation between the LLM-generated metrics and expert assessments, confirming the
reliability of our metrics.
Findings.Our evaluation results show that QAMR outper-
forms a baseline RAG-based chatbot without dual-chunking,
achieving an average of 88.5% answer correctness and 90%
retrieval accuracy. These represent average improvements of
16.5% and 12%, respectively, over the baseline.
Our ablation study leads to two main findings:First,
the components of QAMR complement one another. While
each component individually improves accuracy, only the full
QAMR – with all components included – consistently achieves
the strongest performance, yielding average gains of 19.6% in
answer correctness and 14.0% in retrieval accuracy compared
to the best ablation.Second,the ability to reinterpret user
queries across multiple system releases is critical for effective
question answering on multi-release documents, as the ablation
with this capability achieves higher answer correctness and
contextual recall compared to the ablations without it.
With respect to response time, across the two datasets Ciena
and REQuestA, QAMR reduces the average by about 9% and
8%, respectively, yielding an overall improvement of roughly
8% compared with the baseline.
Novelty.The novelty of our work is in designing a RAG-
based chatbot for multi-release systems, addressing challenges
from distinct but overlapping documentation. In addition, we
propose a chunking strategy to mitigate the trade-off between
retrieval accuracy and contextual richness by combining small
chunks for retrieval with larger chunks for answer generation.
Significance.Multi-release systems are prevalent in industry,
as organizations often have to deal with evolving products that
have multiple active versions. A practical and accurate chat-
bot can provide substantial assistance in efficiently gleaning
information of interest from the complex technical documents
associated with such systems. Another significant aspect is
our dual-chunking method, which has the potential to increase
chatbot accuracy in broader contexts, meriting further study
beyond multi-release systems.
Replication Package.All code, evaluation scripts, and exper-
imental data for our public dataset are available online [21]
II. INDUSTRYCONTEXT ANDMOTIVATION
Ciena is exploring chatbots to automate routine tasks, with a
focus on managing its complex multi-release system portfolio.
Each release at Ciena requires updating version-specific doc-
umentation that covers requirements, architecture, configura-
tion, and installation. While releases share substantial overlaps,
they also include important differences. These documents sup-
port diverse stakeholders – engineers, developers, and product
managers – who each need to query documentation across
single or multiple releases. For instance, a software developer
might ask a question concerning a specific release of a product,
e.g., “How do I update [product] to release 17?”. To answer
this question accurately, the information must be retrieved
from documents corresponding to release 17. Similarly, the
process for “What are the upgrade paths to [product] release
12?” would require consulting the documents for release 12.

Release vi's 
Corpus Preprocessing Chunking1Preprocessed 
Documents2
Embedding
FunctionContext
Chunks
Documents
Release vi
VectorizationSear ch
Chunks
Embedded
Sear ch Chunks3Fig. 1: QAMR’s corpus creation for each document release
The similarity in release-specific questions and the overlaps
in the documentation of different releases make it essen-
tial to query the appropriate documents to generate correct
responses. Alternatively, the developer could ask a generic
question whose answer is not tied to a particular release,
e.g., “How to connect to [product]’s dashboard?”. In such a
case, Ciena would want to maintain the flexibility to provide a
version-agnostic response that remains relevant across multiple
releases. In all queries and examples, original content has been
slightly modified for confidentiality, with redactions shown in
square brackets ([...]).
Our goal in this paper is to develop a chatbot using state-
of-the-art LLM technologies, enabling stakeholders at Ciena
to query structurally complex and semantically rich system
documentation while accommodating the multi-release nature
of the company’s systems. While our work is motivated by
the specific needs of Ciena, the challenges we address are
not unique to the company. The prevalence of multi-release
systems across industries necessitates the development of
methods capable of handling the complexities of constantly
evolving software documentation. Recognizing this broader
relevance, we have designed our chatbot with a focus on
generalizability.
III. CHATBOT FORMULTI-RELEASESYSTEMS
QAMR takes multi-release documents and a query as
input and generates a response based on the documents.
Section III-A describes corpus creation, and Section III-B
presents QAMR.
A. Corpus Creation
Given a set of multi-release documents, QAMR creates a
separate corpus for each release, instead of combining all
releases into a single corpus. This approach ensures that
differences between releases do not affect the LLM’s accuracy
during answer generation. Based on the user query, QAMR
selects the most relevant corpus for document retrieval (see
Section III-B). To transform the documents from a specific
release into a corpus, we follow the three steps shown in
Figure 1 and described next.
Step 1: Preprocessing.To prepare documents for LLM
processing, we extract text from all components, including
tables, schematics, and flowcharts, in HTML or SVG formats.
Redundant elements, such as headers, footers, and copyright
information, are removed to prevent QAMR from retrieving
irrelevant pages due to matching terms. Document names
are added to each page’s metadata. While not required for
Header
Footer Page NumberTable 1: Caption of Table 1
Text 10.Table 1: Caption of Table 1
__
Header 1
Header 2
Header 3
__
Text 1
Text 2
Text 3
Text 4
__
Text 5
Text 6
Text 7
Text 8__
Text 9.
Original pageProcessed PageText 9.Header  2
Text 6
Text 7Header  1 Header  3
Text 1 Text 2 Text 3 Text 4
Text 8 Text 5(a)(b)
__
Text 10.Fig. 2: Preprocessing of an individual page by QAMR
Algorithm 1QAMR’s Dual-Chunking Algorithm
InputDocs: A set of processed documents from Step 1 of Figure 1
Parameterk: Number of search chunks per page
Parameterps: Padding size
OutputS-Chunks: A set of search chunks
OutputC-Chunks: A set of context chunks
1:C-Chunks=S-Chunks←∅
2:foreverydoc 𝑖∈Docsdo
3:foreverypg𝑖 𝑗∈doc 𝑖do//Building search chunks
4:Dividepg𝑖 𝑗into𝑘search chunkssc 𝑖 𝑗1,...,sc 𝑖 𝑗𝑘with equal size
5:S-Chunks←S-Chunks∪{sc 𝑖 𝑗1,...,sc 𝑖 𝑗𝑘}
6:end for
7:foreverypg𝑖 𝑗∈doc 𝑖do//Building context chunks
8:cc 𝑖 𝑗←pg𝑖 𝑗//Initialize context chunkcc 𝑖 𝑗
9:Addpscharacters from the bottom ofpg𝑖, 𝑗−1 to the beginning ofcc 𝑖 𝑗
10:Addpscharacters from the top ofpg𝑖, 𝑗+1 to the end ofcc 𝑖 𝑗
11:metadata(𝑐𝑐 𝑖 𝑗)←the title info ofdoc 𝑖
12:C-Chunks←C-Chunks∪{cc 𝑖 𝑗}
13:end for//Each search chunksc 𝑖 𝑗𝑙is related to context chunkcc 𝑖 𝑗.
14:end for
15:returnS-Chunks,C-Chunks//Returning search and context chunks
generating answers, this metadata ensures traceability between
QAMR’s responses and the original documents. Text from
tables, diagrams, and schematics is extracted from top-left to
bottom-right, reconstructing lines to preserve both horizontal
and vertical order. Delimiters are used to separate different
sections. Figure 2 shows our preprocessing: headers, footers,
and page numbers are removed; table text is captured top-left
to bottom-right, converted to lines, and separated by “—”.
Step 2: Chunking.The processed text from Step 1 is divided
into smaller segments, known aschunks[11], [8], [6]. Chunk-
ing serves more than just fitting text into an LLM’s token limit:
For search, smaller chunks tend to increase retrieval accuracy.
And, for answer generation, opting for the largest possible
chunks (i.e., the LLM’s token limit) is not necessarily the best
option due to the “needle in a haystack” problem [22], [23],
[24], [11]. To account for these considerations, wedecouple
the search and answer-generation chunk sizes, allowing these
sizes to be configured explicitly and independently. We gen-
erate two distinct sets of chunks: one for search and another
for answer generation. We refer to this method, detailed in
Algorithm 1, asdual chunking.
Algorithm 1 takes as input a set of processed documents,
denoted asDocs, from Step 1. For each document inDocs,
the algorithm produces two outputs: a set ofsearch chunks
(S-Chunks) and a set ofcontext chunks(C-Chunks). The
(smaller) search chunks are used for semantic similarity

Query
Rewriting
Retrieval1
Context
ReductionContext
Selection4Answer
Generation5
Multiple
Standalone
Queries
Reduced Context ChunksTop-Ranked
Reduced
Context Chunks2
3Query Answer Release
Number
Dual-Chunking
Retriever
Retrieved
Context ChunksMulti-Release
Corpora 
Release
vi's 
Corpus 
Release
vi's 
Corpus .
.
.Fig. 3: Architecture of QAMR.
searches to accurately locate relevant text. After identifying
the most relevant search chunks, the corresponding (and larger)
context chunk is retrieved for each search chunk to generate
the answer. Duplicates among context chunks are removed.
Algorithm 1 has two configurable parameters:𝑘, the number
of search chunks to create per page, andps, the padding
size for building context chunks. To create search chunks,
each document page is divided into𝑘chunks, which are then
added toS-Chunks(lines 3–6). Each context chunk is created
by extending a page withpscharacters from the bottom of
its previous page at the beginning andpscharacters from
the top of its next page at the end (lines 8–10). Thus, there
is a one-to-one correspondence between context chunks and
pages. To maintain traceability, the metadata of each context
chunk is set to the title information of its corresponding
document (line 11). Finally, the created context chunk is added
toC-Chunks(line 12). Once all the documents inDocshave
been processed,S-ChunksandC-Chunksare returned (line 15).
Step 3: Vectorization.We generate embedding vectors
for each search chunk using an embedding function. The
resulting vectors numerically represent the chunks in a high-
dimensional space, capturing their semantics [25]. These vec-
tors are stored in a vector database. We note that context
chunks do not require vectorization in our approach. As shown
in Algorithm 1, there is a traceable relationship between search
chunks and context chunks. Specifically, a search chunksc 𝑖𝑗𝑙
maps to its corresponding pagepg𝑖𝑗, which, in turn, maps to
the associated context chunkcc 𝑖𝑗. Thus, when a search chunk
is deemed relevant to a query (via semantic similarity), its
corresponding context chunk can be directly identified.
B. Chatbot Architecture
Figure 3 provides an overview of QAMR, which consists
of five steps for generating a response to a user query based
on the corpora created in Section III-A. Steps 1, 3, 4, and 5 in
Figure 3 require an LLM. While different LLMs could be used
for each step, we use a single LLM instance for efficiency,
given the sequential nature of the steps. This instance is
referred to asLLM-Corein the figure and throughout the rest
of the paper. A working example [26] and the prompts of
QAMR [27] – omitted here for space – are available online.
Step 1: Query Rewriting.QAMR begins by extracting the
release number, if present, from the user query. The extracted
release number is used in both the query rewriting process and
User  Query #1:
What is the the acceptable transmit range for [product] ? 
What are the provisioning rules for it in [release] ?Query Rewriting Output #1:
Base1: What is the the acceptable transmit range for [product] ?   
Filtered1: Valid transmit range for  [product]
Release Number:  None
User  Query #2: (Follow-up Question)
Query Rewriting Output #2:
Base2: What are the provisioning rules for [product]  in [release] ?
Filtered2: [product]  provisioning rules  in [release]
Versionless2: [product]  provisioning rules
Release Number:  [release]Fig. 4: Examples of query rewriting in QAMR.
for determining the appropriate corpus for document retrieval
in Step 2. To extract the release number, QAMR prompts
LLM-Core. This enables us to correctly map variations in the
user query to the intended release. For instance, variants like
“R17.2”, “Rel 17.20”, or “Release 17.2” are all interpreted to
map to “Release 17.20”, as it appears in the documents. Using
an LLM for this mapping provides greater accuracy compared
to regular expressions which are limited by fixed patterns.
After extracting the release number (if present), QAMR
rewrites the query using the user query, the conversation
history, and the extracted release number. QAMR generates
threestandalonequeries if a release number is identified;
otherwise, it generates two. A standalone query refers to a
query that is self-contained and does not require additional
context to be understood. Query rewriting is performed using
three prompts that are available online [27]. We refer to
the three generated standalone queries asBase,Filtered, and
Versionless, as explained below.
TheBasequeryinterprets the user query in the context
of the user conversational history. In interactive settings, users
often ask questions that may lack clarity, particularly follow-
up questions that rely on prior context, using pronouns like
“it” or “they”, or phrases such as “the previous one” [28],
[6], [11]. Figure 4 illustrates two examples ofBasequeries.
The first example,Base 1, is identical to the user query, as
the query is already self-contained. In contrast, the second
example,Base 2, replaces the pronoun “it” with [product]
from the conversation history.
TheFilteredqueryremoves stop words and other common
terms from theBasequery (above), focusing on core elements.
The resulting query often aligns more closely with the content
and language used in document tables, thereby increasing
accuracy, specially when working with table-heavy documents.
For example, in Figure 4,Filtered 1andFiltered 2represent the
Filteredqueries for their respective user queries.
TheVersionlessqueryis created – only if the user query
includes a release number – by removing the release number
from theFilteredquery (above).Versionlessqueries address
two scenarios that can cause retrieval inaccuracies: (1) when
the release number is explicitly mentioned in relevant docu-
ments but absent in the user query, or (2) when the release

number is explicitly included in the user query but missing
from relevant documents. For example, in Figure 4,Version-
less2, formed by omitting the release number fromFiltered 2,
is the third standalone query generated for the second user
query. In contrast, for the first user query, which lacks a release
number, only two standalone queries are generated.
Step 2: Retrieval.In this step, we first retrieve the most rel-
evant search chunks by taking the union of the highest-scoring
(search) chunks that match a given standalone query, using two
widely used scoring methods: cosine similarity and maximal
marginal relevance search [29], [30]. The corpus to use for
retrieval is determined by the release number mentioned in
the user query (as extracted in Step 1). If no release number
is provided, the latest corpus is used to generate responses.
Since there are multiple standalone queries, multiple sets of
search chunks will be retrieved. We take the (duplicate-free)
union of these search chunks. Next, using the traceability
information retained between search and context chunks in
Algorithm 1, we map the search chunks to their corresponding
context chunks. Finally, we proceed to Step 3, described below,
with a (duplicate-free) set of the most relevant context chunks,
discarding any redundancies.
Step 3: Context Reduction.As noted earlier, large con-
text chunks, even when within an LLM’s token limit, can
create a “needle in a haystack” problem, where irrelevant
details obscure key information and hinder accurate answer
generation by LLMs [23], [22], [24], [11]. To address this,
QAMR uses few-shot, chain-of-thought (CoT) prompts [27]
to extract only the information relevant to the user query from
each retrieved context chunk. This ensures that eachindividual
chunk contains only query-relevant information. Each reduced
chunk is retained unless it is empty, indicating that the original
chunk contained no relevant information.
Step 4: Context Selection.QAMR identifies the most
relevant reduced context chunks from Step 3 based on their
usefulness in answering the corresponding standalone query.
Similar to step 3, this is achieved using a few-shot CoT
prompt [27]. The prompt we use here is inspired by the one
proposed by Yu et al. [12]. The top-ranked reduced context
chunks are then selected.
Step 5: Answer Generation.QAMR uses the top-ranked
reduced context chunks from Step 4 to answer the query. The
prompt [27] instructsLLM-Coreto rely only on these chunks
rather than its internal knowledge and to reply with “I don’t
know” if they lack sufficient information for answering the
query, reducing hallucination risk.
IV. EMPIRICALEVALUATION
Our evaluation aims to answer three research questions:
RQ1 (Accuracy).How accurate is QAMR?RQ1 assesses
the accuracy of QAMR using a proprietary dataset from Ciena
and an open-source dataset, REQuestA [13]. We compare
QAMR with the baseline described in Section IV-C using
six LLM-as-Judge metrics [14], [15], which measure answer
accuracy and retrieval accuracy. To ensure the reliability ofTABLE I: Summary statistics for datasets.
Ciena REQuestA
Number of Documents 59 6
Average Characters per Document 887,559 119,741
Average Pages per Document 531 65
Number of Questions 88 159
Average Characters per User Queries 74 65
Average Characters per Ground Truth Response 240 528
Number of Non-Textual Elements 5694 20
these LLM-generated metrics, we conduct a correlation anal-
ysis validating the metrics against expert human judgments.
RQ2 (Ablation Study).How does each step of QAMR affect
its overall accuracy?Among the five steps of QAMR, the
last step – answer generation – is essential for any question-
answering chatbot. The other four steps, however, are designed
to improve accuracy, particularly for multi-release systems.
RQ2 investigates how each of these four steps individually
contributes to the overall accuracy of QAMR.
RQ3 (Response Time).What is the execution time of
QAMR?RQ3 compares the response times of QAMR and the
baseline across the two datasets in our evaluation.
A. Datasets and Ground Truths
Our evaluation uses two datasets: Ciena’s system-
specification documents on optical networking and telecom-
munications, and REQuestA, a public dataset of aerospace,
defence, and security software requirements [13].
The Ciena dataset – the larger of the two datasets – has
multi-release characteristics, which are absent in REQuestA.
The inclusion of REQuestA in our evaluation serves three
purposes: (1) to confirm that QAMR generalizes beyond the
Ciena dataset and maintains accuracy across different types
of documents, (2) to systematically evaluate the effectiveness
of dual chunking irrespective of the presence of multi-release
characteristics, and (3) to be able to provide a public repli-
cation package since the Ciena dataset, due to its proprietary
nature, cannot be publicly released.
Table I provides summary statistics for the two datasets.
Ciena’s dataset includes 59 multi-release system specifications,
averaging 531 pages and 887,559 characters, with 5,694 non-
textual elements. In contrast, REQuestA dataset comprises six
software requirements specifications, averaging 65 pages and
119,741 characters, with only 20 non-textual elements.
The Ciena documents are accompanied by a ground truth
including 88 question-answer pairs, with questions averaging
74 characters and answers averaging 240 characters. This
dataset is part of an internal benchmark developed by the
company to evaluate chatbot initiatives. Both the questions and
answers were carefully created by subject-matter experts, none
of whom are co-authors of this paper. The REQuestA ground
truth consists of 159 well-defined, human-verified question-
answer pairs [13], with questions averaging 65 characters and
answers averaging 528 characters.
B. Evaluation Metrics
We evaluate QAMR using the following six LLM-as-Judge
metrics:answer correctness,answer relevancy,answer faith-
fulness,contextual precision,contextual recall, andcontextual

relevancy.Answer correctnessis measured with G-Eval [15],
while the other five use DeepEval [14]. All metrics rely on an
LLM as the evaluator. We describe these six metrics below:
Let𝑄be the set of user queries in our evaluation,𝑅the set
of generated answers, and𝐺the set of ground-truth answers
for𝑄. For each query𝑞∈𝑄, let𝑟 𝑞∈𝑅denote the generated
answer and𝑔 𝑞∈𝐺the ground-truth answer. Further, let
𝐶𝑞=(𝑐 1,...,𝑐𝑙)represent the ordered list of chunks used
to generate the answer for𝑞. We refer to the underlying
LLM used for computing the accuracy metrics asLLM-Eval.
Throughout the evaluation,𝑞always refers to the original user
query before query rewriting.
1) Answer Correctness:This metric usesLLM-Evaland
relies on domain-specific prompts that emulate the reasoning a
human evaluator would apply to assess the quality of responses
generated within the chatbot’s application context [15]. In col-
laboration with Ciena, we developed eight prompts, provided
online [27], designed to replicate expert reasoning. These
prompts are organized into four groups:
Factual Accuracy (two prompts)ensure that,𝑟 𝑞∈𝑅and
𝑔𝑞∈𝐺are factually consistent for each question𝑞∈𝑄.
Matching Units and Values (two prompts)ensure no mis-
match in values or units between𝑟 𝑞and𝑔𝑞, e.g., seconds used
instead of decibel-milliwatts for the receiver-damage level.
Equivalence of Core Meaning (two prompts)ensure
that differences in verbosity or in the order of information
presented in𝑟 𝑞and𝑔𝑞are disregarded, as long as𝑟 𝑞and
𝑔𝑞convey the same core meaning. For example, consider a
question𝑞asking which pins a specific product can connect
to. Suppose𝑔 𝑞explicitly lists the pin numbers, while𝑟 𝑞offers
guidelines for determining these pins, such that engineers can
identify them using those guidelines. In this example, our
prompts ensure that𝑟 𝑞is deemed correct.
Recognition of Scope (two prompts)ensure that: (1) For
out-of-scope queries (where𝑔 𝑞indicates no answer exists),𝑟 𝑞
explicitly acknowledges the absence of an answer; and (2) For
in-scope queries,𝑟 𝑞does not claim that there is no answer.
Given a set of domain-specific prompts that produce verdicts
of “correct” or “incorrect”, G-Eval assigns a score (between
0 and 1) to each query𝑞, evaluating𝑟 𝑞against𝑔𝑞[15].
2) Answer Relevancy:For a query𝑞, answer relevancy
usesLLM-Evalto classify the statements in𝑞’s generated
response,𝑟 𝑞, as relevant or irrelevant to the statements in𝑞.
Let𝐴be the total number of statements in𝑟 𝑞, and𝐵be the
number of statements in𝑟 𝑞that are relevant to𝑞. Answer
relevancy is defined as𝐵
𝐴, measuring the proportion of answer
statements that are relevant to𝑞.
3) Answer Faithfulness:For a query𝑞, answer faithfulness
usesLLM-Evalto classify statements in𝑞’s generated response,
𝑟𝑞, as being relevant or irrelevant to at least one statement
in some chunk in𝐶 𝑞=(𝑐 1,...,𝑐𝑙)used for answering𝑞.
Let𝐴be the total number of statements in𝑟 𝑞, and𝐵be the
number of statements in𝑟 𝑞that find a relevant statement in𝐶 𝑞.
Answer Faithfulness is then defined as𝐵
𝐴, which measures the
proportion of answer statements that are relevant to the chunks
used to answer𝑞.4) Contextual Precision:For a query𝑞, contextual preci-
sion measures the relevance of the chunks𝐶 𝑞=(𝑐 1,...,𝑐𝑙)
used for answering it. This metric usesLLM-Evalto classify
each chunk𝑐 𝑖as being relevant or irrelevant to𝑞by comparing
𝑐𝑖with𝑞’s ground-truth answer, i.e.,𝑔 𝑞. For each𝑐 𝑖, let𝑏𝑖be
a binary variable that is set to 1 when𝑐 𝑖is relevant to𝑔 𝑞and
to 0 otherwise. Contextual precision for𝑞is then defined as:
Contextual precision=Í𝑙
𝑖=1# of relevant chunks in sublist(𝑐1,...,𝑐 𝑖)
𝑖×𝑏𝑖
# of relevant chunks in𝐶 𝑞
Contextual precision emphasizes the relevance of higher-
ranked contexts (near 1) more than lower-ranked ones (near𝑙).
5) Contextual Recall:For a query𝑞, contextual recall uses
LLM-Evalto classify the statements in𝑞’s ground-truth answer,
𝑔𝑞, as relevant or irrelevant to the chunks𝐶 𝑞=(𝑐 1,...,𝑐𝑙)
used for answering𝑞. Let𝐴be the total number of statements
in𝑔𝑞, and let𝐵be the number of statements in𝑔 𝑞that are
relevant to some statement in some chunk𝑐 𝑖. Contextual recall
is defined as𝐵
𝐴, measuring the proportion of ground-truth
statements covered by𝐶 𝑞.
6) Contextual Relevancy:For a query𝑞, contextual rel-
evancy usesLLM-Evalto classify the statements in chunks
𝐶𝑞=(𝑐 1,...,𝑐𝑙)used for answering𝑞as being relevant or
irrelevant to𝑞. Let𝐴be the total number of statements in𝐶 𝑞,
and let𝐵be the number of statements in𝐶 𝑞that are relevant
to𝑞. Contextual relevancy is then defined as𝐵
𝐴, measuring the
proportion of chunk statements that are relevant to𝑞.
C. Ablations and Baseline
We develop five ablations of QAMR, and in addition,
compare QAMR with a state-of-the-art baseline. The five
ablations correspond to four steps of QAMR – query rewriting,
dual chunking, context reduction, and context selection – as
well as one ablation that excludes all these steps. Specifi-
cally, we construct (i) one ablation without any of the four
intermediate steps but with answer generation, and (ii) four
ablations where each includes only a single intermediate step
along with answer generation. These ablations enable us to
isolate the impact of each component of QAMR. In particular,
the ablation with query rewriting allows us to assess the
effectiveness of this step in improving answer correctness for
multi-release documents.
For comparison with existing methods, we use the ar-
chitecture of state-of-the-art RAG-based question-answering
chatbots in software engineering [11], [6], [28], [12], adapting
them to include the query rewriting and context selection steps
from QAMR. This adaptation enables the baseline to handle
multi-release documents, noting that existing chatbots are not
suitable for use in their current form with such documents.
Our baseline implementation is provided online [21].
D. Implementation
For both QAMR and the baseline, we use Meta’s
instruction-tuned Llama 3.0 model with 70B parameters
(Llama-3-70B-Instruct) [31]. Llama 3-70B balances efficiency
with strong instruction-following capabilities while remaining

TABLE II: Key parameters of QAMR and the baseline.
Chunking Retrieval
QAMR# of search chunks
per page (k)Padding size (ps) # of retrieved chunks
per standalone query# of chunks used for
answer generation
2 500 characters
4 3
BaselineChunk size
Each page up to 3000 characters is a chunk
practical for local deployment [31] – as necessitated by our
industry context. We implement QAMR and the baseline using
Python 3.10 and the Transformers library (v4.45.1) [32] for
model loading and language tokenization. To minimize com-
putational overhead, bothLLM-CoreandLLM-Evalare loaded in
a 4-bit quantized format using BitsAndBytes (v0.43.1) [33].
For document indexing, we use the BAAI/bge-base-en-v1.5
model, which provides high-quality and efficient semantic
embeddings, making it ideal for large corpora [34], [35]. The
dual-chunking approach is implemented using LangChain’s
Multi-Vector Retrievers [36], with vectors stored in a Chroma
vectorstore [37]. We use LangChain (v0.2.8) [30] as the
primary underlying framework, utilizing the LangChain Ex-
pression Language (LCEL) [38] to construct chatbot pipelines.
Our implementation is available online [21].
E. Experimental Procedure
Our experiments were conducted on a machine with two
Intel Xeon Gold 6338 CPUs, 512 GB of RAM, and one
NVIDIA A40 GPU (46 GB memory). To mitigate randomness,
each experiment was repeatedten times. In total, we posed
4,940 queries across both datasets and evaluated the responses
using the six metrics described in Section IV-B.
Table II shows the parameters for configuring QAMR and
the baseline. QAMR’s dual-chunker (Algorithm 1) requires
two parameters:kandps. We setk=2to split each
page into two search chunks, balancing search accuracy and
effectiveness by avoiding chunks that are overly large or
small. For the padding sizeps, we use 500 characters to
provide sufficient preceding and succeeding context (roughly
1/3 to 1/4 of a page). For the baseline, which uses the
same chunks for both search and answer generation, we treat
each page as a single chunk, with pages exceeding 3,000
characters split into smaller chunks. The chunks maintain a
25% overlap with the previous and next chunks, following
recommendations from prior studies on traditional (single-
chunk) RAG systems [6], [13].
As shown in Table II, both QAMR and the baseline retrieve
four chunks per standalone query; two via cosine similarity,
and the other two via maximal marginal relevance. The answer
generation step then uses the top three ranked chunks to
produce responses, consistent with prior work [6], [13].
We evaluate the five ablations discussed in Section IV-C
using the same setup for QAMR described in Table II on the
Ciena dataset and repeat each experimentten times.
F . Results
In this section, we present our findings and answer RQ1-
RQ3. To provide statistical support for our analysis, we use
0.70 0.75 0.80 0.85 0.90 0.95 1.00
PercentageContextual
RelevancyContextual
RecallContextual
PrecisionAnswer
FaithfulnessAnswer
RelevancyAnswer
Correctness
0.93
0.880.89
0.830.93
0.780.97
0.960.96
0.930.92
0.72
QAMR
Baseline(a) Ciena Dataset
0.6 0.7 0.8 0.9 1.0
PercentageContextual
RelevancyContextual
RecallContextual
PrecisionAnswer
FaithfulnessAnswer
RelevancyAnswer
Correctness
0.94
0.850.85
0.740.85
0.590.98
0.960.92
0.920.85
0.72
QAMR
Baseline
(b) REQuestA Dataset
Fig. 5: Comparison of QAMR and the baseline using the
metrics defined in Section IV-B.
Wilcoxon Rank-Sum test [39] and Vargha–Delaney effect
size ( ˆ𝐴12) [40], with a 1% significance level. A difference
is considered statistically significant when the𝑝-value falls
below this threshold. The effect size is classified as small,
medium, or large when ˆ𝐴12deviates from 0.5 by at least 0.06,
0.14, and 0.21, respectively.
RQ1 (Accuracy).We measure accuracy using the six
metrics defined in Section IV-B. Figure 5 compares the
distributions of the six metrics obtained by QAMR and the
baseline for the Ciena and REQuestA datasets. For the Ciena
dataset (Figure 5(a)), QAMR achieves higher averages than
the baseline across all six metrics. For the REQuestA dataset
(Figure 5(b)), QAMR achieves higher averages for five of the
metrics, with both QAMR and the baseline producing the same
average for answer relevancy. Statistical tests in Table III show
that QAMR significantly outperforms the baseline with a large
effect size for all metrics except answer faithfulness in the
Ciena dataset and answer relevancy in the REQuestA dataset.
Across both datasets, QAMR achieves average scores of
88.5%in answer correctness,89%in contextual precision,
87%in contextual recall, and93.5%in contextual relevancy.

TABLE III: Statistical test results comparing QAMR and the
baseline. QAMR significantly outperforms the baseline with
large effect sizes, except in the yellow highlighted cells where
no statistical significance is observed.
Ciena REQuestA
p-value ˆ𝐴12p-value ˆ𝐴12
Answer Correctness 0.002 1 (L) 0.002 1 (L)
Answer Relevancy 0.006 0.83 (L) 0.84 –
Answer Faithfulness 0.08 – 0.002 0.98 (L)
Contextual Precision 0.002 1 (L) 0.002 1 (L)
Contextual Recall 0.004 0.98 (L) 0.002 1 (L)
Contextual Relevancy 0.002 1 (L) 0.002 1 (L)
TABLE IV: Pearson correlation coefficients between metrics
in Section IV-B and adequacy labels provided by domain
experts. Green cells represent p-values≪0.01.
Dataset Metric Correlation
CienaAnswer Correctness 0.88
Contextual Precision -0.02
Contextual Recall -0.01
REQuestAAnswer Correctness 0.75
Contextual Precision 0.63
Contextual Recall 0.54
These results indicate an average improvement of16.5%in
answer correctness compared to the baseline. Averaging the
improvements across contextual precision, recall, and rele-
vancy yields a12% overall increase over the baseline.
As noted in Section I, we use LLM-as-Judge metrics instead
of expert judgment because of the scale of our evaluation
and the infeasibility of comprehensive manual assessment. An
important question that arises here is how closely our LLM-
as-Judge metrics align with expert judgment. To this end, we
asked Ciena experts (who are not authors of this study) to
review the responses generated by QAMR on the full Ciena
dataset (all 88 questions). From the 10 runs conducted for
each question in this dataset, we randomly selected one run,
and experts labelled its answer as “adequate” or “inadequate”
based on how closely it matched the ground truth. Similarly,
we randomly selected one run of QAMR on the REQuestA
dataset and engaged an independent annotator (also not an
author) to label the responses generated for the 159 questions
in that dataset, in order to determine how closely the generated
answers matched the ground truth. We then computed the
Pearson correlation coefficients between the labels provided
by humans and the LLM-as-Judge metric values, as reported
in Table IV. In this analysis, among the six metrics discussed
in Section IV-B, we considered only the metrics that com-
pare generated answers with ground truths – namely, answer
correctness, contextual precision, and contextual recall. As
shown in the table, answer correctness strongly correlates with
human judgments (0.88 for Ciena and 0.75 for REQuestA,
p-value≪0.01). For REQuestA, contextual precision and
recall also correlate positively with human judgment.
Finding 1.QAMR significantly outperforms the baseline
on all answer accuracy and retrieval accuracy metrics,
except for answer faithfulness (Ciena dataset) and answerTABLE V: Average (%) evaluation results for the ablation
study. TheBaseablation uses only the answer generation
step of QAMR. Highlighted cells mark the best-performing
ablation for each metric.
Base+ Query
Rewrite+ Dual
Chunk+ Ctx
Reduce+ Ctx
Select
Answer Correctness (%) 68.29 72.28 72.13 59.30 64.39
Answer Relevancy (%) 93.56 92.64 93.67 78.07 90.69
Answer Faithfulness (%) 95.04 94.95 96.24 97.91 95.09
Contextual Precision (%) 67.46 71.25 57.77 58.88 72.39
Contextual Recall (%) 73.26 74.76 63.93 61.45 73.25
Contextual Relevancy (%) 82.10 78.82 75.97 74.68 87.63
relevancy (REQuestA dataset). Notably, QAMR achieves
an average 16.5% increase in answer correctness and
an average improvement of 12% in contextual precision,
recall, and relevance compared to the baseline.
Finding 2.The answer-correctness metric has a strong pos-
itive correlation with human assessments for both datasets.
Take away 1.QAMR’s dual chunking identifies relevant
chunks more often than traditional chunking. Combined
with context reduction, this ensures that answer generation
is provided with only relevant information.
Take away 2.The strong correlation between the answer-
correctness metric and human judgment across both
datasets indicates that, in our application context, this met-
ric is a reliable proxy and provides a scalable alternative
to human evaluation for verifying LLM-generated texts
against the ground truth.
RQ2 (Ablation Study).Table V presents the ablation study
results obtained using the Ciena dataset. We refer to the
ablation that uses only the answer-generation step as thebase
ablation. Each of the other four ablations combines the base
with exactly one of the following QAMR’s steps: query rewrit-
ing, dual chunking, context reduction, or context selection.
Box plots showing the distributions underlying the averages
in Table V are provided in the replication package [41].
The results in Figure 5(a) and Table V show that QAMR
significantly improves the average answer correctness, answer
relevancy, contextual precision, contextual recall, and contex-
tual relevancy relative to the five ablations, with minimum
respective improvements of 19.60%, 2.16%, 20.27%, 14.59%,
and 5.70%. In addition, QAMR’s performance on the answer
faithfulness metric is better or on par with that of the ablations.
Statistical tests in Table VI show that QAMR significantly
outperforms all ablations with a large effect size for all
metrics except for the answer relevancy and answer faith-
fulness metrics. For answer relevancy, QAMR significantly
outperforms two ablations, while for answer faithfulness, it
significantly outperforms one ablation. In both cases, the
differences show large effect sizes. For all other comparisons
of answer faithfulness and relevancy, there is no statistically
significant difference between QAMR and its ablations. Thus,
according to our statistical tests, QAMR is never outperformed
by any of its ablations on any metric.

TABLE VI: Statistical test results comparing QAMR with each
of the ablations in Table VI. QAMR significantly outperforms
all ablations with large effect sizes, except in the yellow
highlighted cells where no statistical significance is observed.
Base + Query Rewrite + Dual Chunk + Ctx Reduce + Ctx Select
p-value ˆ𝐴12 p-value ˆ𝐴12 p-value ˆ𝐴12 p-value ˆ𝐴12 p-value ˆ𝐴12
Ans. Cor. 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L)
Ans. Rel. 0.064 - 0.080 - 0.002 0.75(L) 0.002 1(L) 0.020 -
Ans. Fait. 0.064 - 0.002 0.84(L) 0.080 - 0.200 - 0.014 -
Ctx. Prec. 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L)
Ctx. Rec. 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L)
Ctx. Rel. 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L) 0.002 1(L)
When comparing the base ablation with the four non-base
ablations – each incorporating a single step from QAMR –
we observe that each non-base ablation outperforms the base
ablation in at least one metric. Specifically, the query-rewriting
ablation improves answer correctness, answer faithfulness,
contextual precision, and contextual recall compared to the
base ablation; the dual-chunking ablation improves answer
correctness, answer faithfulness, and answer relevancy; the
context-selection ablation improves contextual precision, con-
textual recall, and contextual relevancy; and the context-
reduction ablation improves answer faithfulness. Notably, the
base ablation never achieves the highest average on any metric.
Finding 1.No ablation significantly outperforms QAMR
in any metric, while QAMR significantly outperforms all
ablations in four or five out of the six metrics. Further,
every non-base ablation outperforms the base in at least
one metric, showing that each step – query rewriting,
dual chunking, context reduction, and context selection –
contributes positively to the overall approach. All steps
together complement one another, enabling QAMR to
outperform every ablation.
Finding 2.The query-rewriting ablation outperforms the
other ablations that lack mechanisms for interpreting
queries across multi-release documents. Among the five ab-
lations (Table V), the query-rewriting ablation achieves the
highest answer correctness and contextual recall, as well
as the second-highest contextual precision. This highlights
the importance of query rewriting for accurate question
answering in the context of multi-release documents.
RQ3 (Response Time).Table VII compares the average re-
sponse times of QAMR and the baseline. Both yield higher av-
erage response times on the Ciena dataset than on REQuestA,
as Ciena’s documents are larger and more complex, often con-
taining multiple tables and schematics. We observe two points
from Table VII: (1) For both datasets, the average response
time of the baseline is about 8% higher than that of QAMR
(Ciena: 45/41, REQuestA: 40/37). (2) On the Ciena dataset,
the average response times of both QAMR and the baseline
are 12% higher than on REQuestA (QAMR: 41/37, baseline:
45/40), indicating that QAMR’s runtime scales proportionally
with the baseline as document complexity increases.
As shown in Table VII, for QAMR, context reduction is the
most time-consuming step, accounting for 36% of the totalTABLE VII: Average response times of QAMR vs. baseline.
Average Response Time Response Time Breakdown
per Query (in seconds) according to Steps (%)
Ciena REQuestAQuery
RewritingContext
ReductionContext
SelectionAnswer
GenerationOther
Steps
QAMR 41 37 27% 36% 23% 13% 1%
Baseline 45 40 20% N/A 33% 46% 1%
response time, followed by query rewriting and context selec-
tion at 27% and 23%, respectively. For the baseline, which
does not have a context-reduction step, answer generation is
the most time-intensive step, followed by context selection
and query rewriting. In the baseline, answer generation and
context selection are more expensive than query rewriting,
whereas the opposite is true for QAMR. This is because,
without context reduction, the baseline must process large
chunks, while context reduction provides a concise context for
answer generation and selection, thus improving efficiency.
Finding.On average, QAMR is about 8% faster than the
baseline on both datasets, primarily because it provides the
answer generation and context selection steps with a more
concise context from the context-reduction step.
Take away.Based on an internal deployment of QAMR
on Ciena’s infrastructure, experts found QAMR’s response
time to be practical.
G. Validity Considerations
Internal Validity.To ensure a fair comparison, QAMR
and the baseline used the same underlying LLM (LLM-Core),
identical prompts for shared components, and the same param-
eters for retrieval (Table II). The baseline’s single-chunking
approach was configured according to best practices from prior
studies [13], [6]. Metrics for both QAMR and the baseline
were computed using the same evaluation LLM (LLM-Eval). To
mitigate randomness, we set the temperature to 0.01 and top-
k to one, and repeated each experiment ten times [42], [43].
Regarding data leakage, we are confident that the proprietary
Ciena dataset was not part ofLLM-Core’s pre-training. While
the REQuestA dataset is open-source, both QAMR and the
baseline rely on the sameLLM-Core, so any leakage affecting
LLM-Coreimpacts both equally. As such, our comparison
remains fair and reliable.
External Validity.Our evaluation used two datasets from
different sources. The Ciena dataset contains a large col-
lection of complex, multi-release system documents with
expert-created and validated questions and ground truths. The
REQuestA dataset includes both auto-generated and manual
questions, with all auto-generated items rigorously vetted for
relevance and quality. In relation to the choice of LLM, we
used Llama 3-70B [31] for bothLLM-CoreandLLM-Eval. At
the time our experiments were conducted, our industry part-
ner’s policies prohibited use of externally hosted LLMs when
handling proprietary data. In addition, our LLM-as-Judge
evaluation would have been very costly – thousands of dollars
– due to the large number of LLM calls required to compute
each metric, had it been conducted on externally hosted LLMs.
While benchmarking other LLMs is valuable, the exclusive use

of Llama 3-70B does not pose a significant risk, as it remains
one of the most accurate instruction-following models – freely
available for research and industry use – and suitable for on-
premise deployment, offering performance competitive with
newer LLMs in text generation and question answering [44].
V. RELATEDWORK
Prior to LLMs, extractive BERT-based models were used to
automate question answering, either by integrating RAG [13]
or without it [45]. More recent approaches rely on generative
LLMs such as ChatGPT [42] and Llama [31], which not only
identify relevant text but also generate human-like responses
tailored to the user query and conversational history. For ex-
ample, Chaudhary et al. [6] develop an LLM-based chatbot to
help systems engineers with CI/CD-related queries. Similarly,
Barnett et al. [11] report on their experience applying LLM-
based RAG chatbots to case studies from different domains
and provide recommendations for improving such chatbots
in the context of software engineering. These existing RAG
architectures rely on a traditional single-chunking strategy,
similar to our evaluation baseline in Section IV.
As observed in recent literature [11], [6], [13], [28], [46],
[12], [47], RAG solutions face important limitations, including
challenges in determining the optimal chunk size, the retrieval
of irrelevant or noisy data within chunks, and vulnerability to
low-quality user queries. Our approach seeks to mitigate these
well-known limitations. In particular, our dual-chunking strat-
egy decouples the chunks used for retrieval and generation,
thus eliminating the need to select a single chunk size for both
tasks. In addition, our approach incorporates context reduction
and selection mechanisms to reduce noisy data within chunks.
Finally, the query-rewriting component in our approach, which
generates multiple standalone queries that improve and extend
the user query in various ways, helps improve resilience to
low-quality user queries compared to techniques that rewrite
the user query into a single standalone query.
In relation to the retrieval step in RAG, Liu et al. [48]
propose a hierarchical approach that, given a set of documents,
first performs document retrieval and then retrieves relevant
passages within the retrieved documents. This approach works
well when the documents are sufficiently distinct, allowing the
document-retrieval stage to meaningfully narrow the search
space. However, this hierarchical method cannot effectively
distinguish between highly overlapping multi-release docu-
ments and, similar to the other approaches discussed above,
fails to address the challenges of question answering for multi-
release documents described in Section I.
VI. LESSONSLEARNED
Lesson 1:Among the six metrics presented in Section IV-B,
answer correctness, contextual precision, and contextual recall
evaluate generated answers against ground truths, whereas
contextual relevancy, answer relevancy, and answer faithful-
ness do not. Instead, the latter metrics assess how well a
response overlaps with the user query or retrieved chunks,
rather than whether it matches the ground truth. High scoreson these metrics can be achieved by matching the query or
retrieved chunks even when the answer itself is incorrect. In
our experiments, this effect made the baseline appear com-
parable to QAMR on these three metrics, despite performing
substantially worse on the other three metrics that are ground-
truth-dependent.
Implication of Lesson 1.Treat contextual and answer
relevancy, and faithfulness as indicators of plausibility rather
than correctness. Interpret them alongside ground-truth-based
metrics, and prioritize the latter when assessing accuracy.
Lesson 2:In domain-specific and technical documents,
even small wording changes can affect chatbot accuracy. For
example, asking “What are performance monitoring consider-
ations?” yields an accurate answer, whereas replacing“per-
formance monitoring”with“PM”or“operational”– terms
not used in the documents – produces poor results. This is
because retrieval is largely literal: if the query vocabulary does
not appear in the corpus, relevant chunks are not retrieved.
Implication of Lesson 2:Apply query rewriting based on a
domain-specific glossary to align user queries with document
language by disambiguating and expanding acronyms, and
mapping terms to their glossary synonyms.
Lesson 3:The most prominent source of error in the
baseline, QAMR, and its ablations is incorrect document
retrieval, as most errors arise from the retrieved context being
either irrelevant or incomplete. The next major source of error
is hallucination, which occurs more frequently in the baseline
than in QAMR. This is because query rewriting improves
retrieval accuracy by enabling release-specific retrieval across
multi-release documents, while dual-chunking and context
reduction further reduce hallucinations by improving retrieval
accuracy and pruning irrelevant information.
Implication of Lesson 3.Apply query rewriting for release-
specific retrieval in multi-release systems, dual-chunking to
improve overall retrieval accuracy, and context reduction to
prune irrelevant information from the retrieved contexts.
VII. CONCLUSION
We presented QAMR, a RAG-based chatbot for question
answering over multi-release systems. We evaluated QAMR on
two datasets, comparing its effectiveness and efficiency against
a state-of-the-art baseline and conducting an ablation study to
assess the impact of its components, especially the release-
specific query-rewriting module. QAMR significantly outper-
forms both the baseline and its ablations in answer correctness
and retrieval accuracy. Moreover, the automated LLM-as-
Judge metrics used to evaluate answer accuracy closely align
with expert evaluations, validating the reliability of these
automated metrics as a proxy for manual judgment.
VIII. ACKNOWLEDGMENT
We gratefully acknowledge funding from Mitacs Accelerate,
Ciena, the Ontario Graduate Scholarship (OGS) program,
and NSERC of Canada under the Discovery and Discovery
Accelerator programs.

REFERENCES
[1] M. M. Lehman, “Laws of software evolution revisited,” inSoftware
Process Technology, 5th European Workshop, EWSPT ’96, Nancy,
France, October 9-11, 1996, Proceedings, ser. Lecture Notes in
Computer Science, C. Montangero, Ed., vol. 1149. Springer, 1996,
pp. 108–124. [Online]. Available: https://doi.org/10.1007/BFb0017737
[2] K. Pohl, G. B ¨ockle, and F. van der Linden,Software Product Line
Engineering - Foundations, Principles, and Techniques. Springer,
2005. [Online]. Available: https://doi.org/10.1007/3-540-28901-1
[3] A. Abdellatif, K. Badran, D. E. Costa, and E. Shihab, “A comparison
of natural language understanding platforms for chatbots in software
engineering,”IEEE Trans. Software Eng., vol. 48, no. 8, pp. 3087–3102,
2022. [Online]. Available: https://doi.org/10.1109/TSE.2021.3078384
[4] A. Abdellatif, D. Costa, K. Badran, R. Abdalkareem, and E. Shihab,
“Challenges in chatbot development: A study of stack overflow posts,”
inMSR ’20: 17th International Conference on Mining Software
Repositories, Seoul, Republic of Korea, 29-30 June, 2020, S. Kim,
G. Gousios, S. Nadi, and J. Hejderup, Eds. ACM, 2020, pp. 174–185.
[Online]. Available: https://doi.org/10.1145/3379597.3387472
[5] G. Daniel and J. Cabot, “Applying model-driven engineering to
the domain of chatbots: The xatkit experience,”Sci. Comput.
Program., vol. 232, p. 103032, 2024. [Online]. Available: https:
//doi.org/10.1016/j.scico.2023.103032
[6] D. Chaudhary, S. L. Vadlamani, D. Thomas, S. Nejati, and
M. Sabetzadeh, “Developing a llama-based chatbot for CI/CD question
answering: A case study at ericsson,” inIEEE International Conference
on Software Maintenance and Evolution, ICSME 2024, Flagstaff,
AZ, USA, October 6-11, 2024. IEEE, 2024, pp. 707–718. [Online].
Available: https://doi.org/10.1109/ICSME58944.2024.00075
[7] J. Huang, Y . Zhong, G. Y . adn Zhihan Jiang, M. Yan, W. Luan,
T. Y . adn Rui Ren, and M. Lyu, “iKnow: an intent-guided chatbot for
cloud operations with retrieval-augmented generation,” inIEEE/ACM
Automated Software Engineering (ASE 2025) Conference, Seoul, South
Korea, Nov 16-20, 2025, to appear.
[8] P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschel,
S. Riedel, and D. Kiela, “Retrieval-augmented generation for
knowledge-intensive NLP tasks,” inAdvances in Neural Information
Processing Systems 33: Annual Conference on Neural Information
Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual,
H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin,
Eds., 2020. [Online]. Available: https://proceedings.neurips.cc/paper/
2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[9] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval
augmented language model pre-training,” inProceedings of the 37th
International Conference on Machine Learning, ICML 2020, 13-18
July 2020, Virtual Event, ser. Proceedings of Machine Learning
Research, vol. 119. PMLR, 2020, pp. 3929–3938. [Online]. Available:
http://proceedings.mlr.press/v119/guu20a.html
[10] G. Izacard and E. Grave, “Leveraging passage retrieval with generative
models for open domain question answering,” inProceedings of the
16th Conference of the European Chapter of the Association for
Computational Linguistics: Main Volume, EACL 2021, Online, April 19
- 23, 2021, P. Merlo, J. Tiedemann, and R. Tsarfaty, Eds. Association
for Computational Linguistics, 2021, pp. 874–880. [Online]. Available:
https://doi.org/10.18653/v1/2021.eacl-main.74
[11] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek,
“Seven failure points when engineering a retrieval augmented generation
system,” inProceedings of the IEEE/ACM 3rd International Conference
on AI Engineering - Software Engineering for AI, CAIN 2024, Lisbon,
Portugal, April 14-15, 2024, J. Cleland-Huang, J. Bosch, H. Muccini,
and G. A. Lewis, Eds. ACM, 2024, pp. 194–199. [Online]. Available:
https://doi.org/10.1145/3644815.3644945
[12] Y . Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi,
and B. Catanzaro, “Rankrag: Unifying context ranking with retrieval-
augmented generation in llms,” inAdvances in Neural Information
Processing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada,
December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave,
A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, Eds., 2024.
[Online]. Available: http://papers.nips.cc/paper files/paper/2024/hash/
db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.html
[13] S. Ezzini, S. Abualhaija, C. Arora, and M. Sabetzadeh, “Ai-
based question answering assistance for analyzing natural-languagerequirements,” in45th IEEE/ACM International Conference on
Software Engineering, ICSE 2023, Melbourne, Australia, May 14-
20, 2023. IEEE, 2023, pp. 1277–1289. [Online]. Available: https:
//doi.org/10.1109/ICSE48619.2023.00113
[14] Confident AI, “Deepeval,” https://github.com/confident-ai/deepeval,
2024, [Online; accessed January 6, 2026].
[15] Y . Liu, D. Iter, Y . Xu, S. Wang, R. Xu, and C. Zhu, “G-eval:
NLG evaluation using gpt-4 with better human alignment,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2023, Singapore, December 6-10,
2023, H. Bouamor, J. Pino, and K. Bali, Eds. Association for
Computational Linguistics, 2023, pp. 2511–2522. [Online]. Available:
https://doi.org/10.18653/v1/2023.emnlp-main.153
[16] K. Papineni, S. Roukos, T. Ward, and W. Zhu, “Bleu: a method for
automatic evaluation of machine translation,” inProceedings of the
40th Annual Meeting of the Association for Computational Linguistics,
July 6-12, 2002, Philadelphia, PA, USA. ACL, 2002, pp. 311–318.
[Online]. Available: https://aclanthology.org/P02-1040/
[17] C.-Y . Lin, “ROUGE: A package for automatic evaluation of summaries,”
inText Summarization Branches Out. Barcelona, Spain: Association for
Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available:
https://aclanthology.org/W04-1013
[18] T. Goyal, J. J. Li, and G. Durrett, “News summarization and evaluation
in the era of GPT-3,”CoRR, vol. abs/2209.12356, 2022. [Online].
Available: https://doi.org/10.48550/arXiv.2209.12356
[19] C. Liu, R. Lowe, I. Serban, M. Noseworthy, L. Charlin, and J. Pineau,
“How NOT to evaluate your dialogue system: An empirical study of
unsupervised evaluation metrics for dialogue response generation,” in
Proceedings of the 2016 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2016, Austin, Texas, USA, November
1-4, 2016, J. Su, X. Carreras, and K. Duh, Eds. The Association for
Computational Linguistics, 2016, pp. 2122–2132. [Online]. Available:
https://doi.org/10.18653/v1/d16-1230
[20] N. Wu, M. Gong, L. Shou, S. Liang, and D. Jiang, “Large language
models are diverse role-players for summarization evaluation,” in
Natural Language Processing and Chinese Computing - 12th
National CCF Conference, NLPCC 2023, Foshan, China, October
12-15, 2023, Proceedings, Part I, ser. Lecture Notes in Computer
Science, F. Liu, N. Duan, Q. Xu, and Y . Hong, Eds., vol.
14302. Springer, 2023, pp. 695–707. [Online]. Available: https:
//doi.org/10.1007/978-3-031-44693-1 54
[21] ReplicationPackage, “QAMR replication package,” https://github.com/
parham-box/SANER-QAMR, 2025, [Online; accessed January 6, 2026].
[22] P. Laban, A. R. Fabbri, C. Xiong, and C. Wu, “Summary of a haystack:
A challenge to long-context llms and RAG systems,” inProceedings
of the 2024 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024,
Y . Al-Onaizan, M. Bansal, and Y . Chen, Eds. Association for
Computational Linguistics, 2024, pp. 9885–9903. [Online]. Available:
https://aclanthology.org/2024.emnlp-main.552
[23] S. Hassani, M. Sabetzadeh, D. Amyot, and J. Liao, “Rethinking legal
compliance automation: Opportunities with large language models,” in
32nd IEEE International Requirements Engineering Conference, RE
2024, Reykjavik, Iceland, June 24-28, 2024, G. Liebel, I. Hadar, and
P. Spoletini, Eds. IEEE, 2024, pp. 432–440. [Online]. Available:
https://doi.org/10.1109/RE59067.2024.00051
[24] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
and P. Liang, “Lost in the middle: How language models use long
contexts,”Trans. Assoc. Comput. Linguistics, vol. 12, pp. 157–173,
2024. [Online]. Available: https://doi.org/10.1162/tacl a00638
[25] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean,
“Distributed representations of words and phrases and their
compositionality,” inAdvances in Neural Information Processing
Systems 26: 27th Annual Conference on Neural Information Processing
Systems 2013. Proceedings of a meeting held December 5-8, 2013,
Lake Tahoe, Nevada, United States, C. J. C. Burges, L. Bottou,
Z. Ghahramani, and K. Q. Weinberger, Eds., 2013, pp. 3111–
3119. [Online]. Available: https://proceedings.neurips.cc/paper/2013/
hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html
[26] P. Khamsepour, M. Cole, I. Ashraf, S. Puri, M. Sabetzadeh, and
S. Nejati, “QAMR working example,” https://github.com/parham-box/
SANER-QAMR/blob/main/Example.pdf, 2025, [Online; accessed Jan-
uary 6, 2026].

[27] ——, “QAMR full prompts,” https://github.com/parham-box/
SANER-QAMR/blob/main/Prompts.pdf, 2025, [Online; accessed
January 6, 2026].
[28] X. Ma, Y . Gong, P. He, H. Zhao, and N. Duan, “Query
rewriting for retrieval-augmented large language models,”CoRR, vol.
abs/2305.14283, 2023. [Online]. Available: https://doi.org/10.48550/
arXiv.2305.14283
[29] J. G. Carbonell and J. Goldstein, “The use of mmr, diversity-based
reranking for reordering documents and producing summaries,”SIGIR
Forum, vol. 51, no. 2, pp. 209–210, 2017. [Online]. Available:
https://doi.org/10.1145/3130348.3130369
[30] Langchain, https://www.langchain.com, 2024, [Online; accessed January
6, 2026].
[31] AI@Meta, “Llama 3 model card,” https://github.com/meta-llama/llama3/
blob/main/MODEL CARD.md, 2024.
[32] T. Wolf, L. Debut, V . Sanh, J. Chaumond, C. Delangue, A. Moi,
P. Cistac, T. Rault, R. Louf, M. Funtowicz, and J. Brew, “Huggingface’s
transformers: State-of-the-art natural language processing,”CoRR, vol.
abs/1910.03771, 2019. [Online]. Available: http://arxiv.org/abs/1910.
03771
[33] HuggingFace, “Bitsandbytes library,” https://huggingface.co/docs/
bitsandbytes/main/en/index, 2024, [Online; accessed January 6, 2026].
[34] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-pack: Packaged
resources to advance general chinese embedding,”CoRR, vol.
abs/2309.07597, 2023. [Online]. Available: https://doi.org/10.48550/
arXiv.2309.07597
[35] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “MTEB: massive
text embedding benchmark,” inProceedings of the 17th Conference of
the European Chapter of the Association for Computational Linguistics,
EACL 2023, Dubrovnik, Croatia, May 2-6, 2023, A. Vlachos and
I. Augenstein, Eds. Association for Computational Linguistics, 2023,
pp. 2006–2029. [Online]. Available: https://doi.org/10.18653/v1/2023.
eacl-main.148
[36] Langchain, “Multivectorretriever,” https://python.langchain.com/v0.1/
docs/modules/data connection/retrievers/multi vector/, 2024, [Online;
accessed January 6, 2026].
[37] C. vectorstore, https://www.trychroma.com, 2024, [Online; accessed
January 6, 2026].
[38] Langchain, “Langchain expression language,” https://python.langchain.
com/docs/concepts/#langchain-expression-language-lcel, 2024, [Online;
accessed January 6, 2026].
[39] F. Wilcoxon, “Individual comparisons by ranking methods,”
inBreakthroughs in statistics: Methodology and distribution.
Springer, 1992, pp. 196–202. [Online]. Available: https:
//doi.org/10.1007/978-1-4612-4380-9 16[40] A. Vargha and H. D. Delaney, “A critique and improvement of the cl
common language effect size statistics of mcgraw and wong,”Journal
of Educational and Behavioral Statistics, vol. 25, no. 2, pp. 101–132,
2000. [Online]. Available: https://doi.org/10.3102/10769986025002101
[41] P. Khamsepour, M. Cole, I. Ashraf, S. Puri, M. Sabetzadeh, and S. Ne-
jati, “QAMR ablation study results,” https://github.com/parham-box/
SANER-QAMR/blob/main/RQ2/Ablation-Study-Results.pdf, 2025,
[Online; accessed January 6, 2026].
[42] OpenAI, “GPT-4 technical report,”CoRR, vol. abs/2303.08774, 2023.
[Online]. Available: https://doi.org/10.48550/arXiv.2303.08774
[43] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi,
Y . Wu, Y . K. Li, F. Luo, Y . Xiong, and W. Liang, “Deepseek-coder:
When the large language model meets programming - the rise of code
intelligence,”CoRR, vol. abs/2401.14196, 2024. [Online]. Available:
https://doi.org/10.48550/arXiv.2401.14196
[44] W. Chiang, L. Zheng, Y . Sheng, A. N. Angelopoulos, T. Li, D. Li,
B. Zhu, H. Zhang, M. I. Jordan, J. E. Gonzalez, and I. Stoica, “Chatbot
arena: An open platform for evaluating llms by human preference,” in
Forty-first International Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. [Online].
Available: https://openreview.net/forum?id=3MW8GKNyzI
[45] M. Borg, J. Bengtsson, H. ¨Osterling, A. Hagelborn, I. Gagner, and
P. Tomaszewski, “Quality assurance of generative dialog models in an
evolving conversational agent used for swedish language practice,” in
Proceedings of the 1st International Conference on AI Engineering:
Software Engineering for AI, CAIN 2022, Pittsburgh, Pennsylvania,
May 16-17, 2022, I. Crnkovic, Ed. ACM, 2022, pp. 22–32. [Online].
Available: https://doi.org/10.1145/3522664.3528592
[46] A. Landschaft, D. Antweiler, S. Mackay, S. Kugler, S. R ¨uping,
S. Wrobel, T. H ¨ores, and H. Allende-Cid, “Implementation and
evaluation of an additional gpt-4-based reviewer in prisma-based
medical systematic literature reviews,”Int. J. Medical Informatics, vol.
189, p. 105531, 2024. [Online]. Available: https://doi.org/10.1016/j.
ijmedinf.2024.105531
[47] Y . Zhu, H. Yuan, S. Wang, J. Liu, W. Liu, C. Deng, Z. Dou,
and J. Wen, “Large language models for information retrieval:
A survey,”CoRR, vol. abs/2308.07107, 2023. [Online]. Available:
https://doi.org/10.48550/arXiv.2308.07107
[48] Y . Liu, K. Hashimoto, Y . Zhou, S. Yavuz, C. Xiong, and P. S. Yu,
“Dense hierarchical retrieval for open-domain question answering,” in
Findings of the Association for Computational Linguistics: EMNLP
2021, Virtual Event / Punta Cana, Dominican Republic, 16-20
November, 2021, M. Moens, X. Huang, L. Specia, and S. W. Yih,
Eds. Association for Computational Linguistics, 2021, pp. 188–200.
[Online]. Available: https://doi.org/10.18653/v1/2021.findings-emnlp.19
# FRAGATA: Semantic Retrieval of HPC Support Tickets via Hybrid RAG over 20 Years of Request Tracker History

**Authors**: Santiago Paramés-Estévez, Nicolás Filloy-Montesino, Jorge Fernández-Fabeiro, José Carlos Mouriño-Gallego

**Published**: 2026-04-15 10:53:49

**PDF URL**: [https://arxiv.org/pdf/2604.13721v1](https://arxiv.org/pdf/2604.13721v1)

## Abstract
The technical support team of a supercomputing centre accumulates, over the course of decades, a large volume of resolved incidents that constitute critical operational knowledge. At the Galician Supercomputing Center (CESGA) this history has been managed for over twenty years with Request Tracker (RT), whose built-in search engine has significant limitations that hinder knowledge reuse by the support staff. This paper presents Fragata, a semantic ticket search system that combines modern information retrieval techniques with the full RT history. The system can find relevant past incidents regardless of language, the presence of typos, or the specific wording of the query. The architecture is deployed on CESGA's infrastructure, supports incremental updates without service interruption, and offloads the most expensive stages to the FinisTerrae III supercomputer. Preliminary results show a substantial qualitative improvement over RT's native search.

## Full Text


<!-- PDF content starts -->

FRAGATA: Semantic Retrieval of HPC
Support Tickets via Hybrid RAG over 20
Years of Request Tracker History
Santiago Param´ es-Est´ evez12, Nicol´ as Filloy-Montesino3, Jorge Fern´ andez-Fabeiro2and Jos´ e
Carlos-Mouri˜ no Gallego2
Abstract— The technical support team of a super-
computing centre accumulates, over the course of
decades, a large volume of resolved incidents that con-
stitute critical operational knowledge. At the Gali-
cian Supercomputing Center (CESGA) this history
has been managed for over twenty years with Request
Tracker (RT), whose built-in search engine has sig-
nificant limitations that hinder knowledge reuse by
the support staff. This paper presents Fragata, a se-
mantic ticket search system that combines modern
information retrieval techniques with the full RT his-
tory. The system can find relevant past incidents re-
gardless of language, the presence of typos, or the
specific wording of the query. The architecture is
deployed on CESGA’s infrastructure, supports incre-
mental updates without service interruption, and of-
floads the most expensive stages to the FinisTerrae III
supercomputer. Preliminary results show a substan-
tial qualitative improvement over RT’s native search.
Keywords— RAG, automation, information re-
trieval, semantic search, HPC, Request Tracker,
CESGA.
I. Introduction
High-Performance Computing (HPC) centres op-
erate heterogeneous platforms hosting hundreds of
scientific applications, compilers, parallel libraries,
resource schedulers, and distributed storage systems.
In this environment, the technical support team acts
as a critical interface between research users and
the infrastructure, resolving incidents ranging from
MPI compilation errors to storage quotas, Conda
environment configuration, or network failures on
GPU nodes. The Galician Supercomputing Center
(CESGA) has managed this workflow for over two
decades using Request Tracker (RT) [1], a widely
1A Spanish version of this paper has been submitted at Jor-
nadas SARTECO 2026.
2Galician Supercomputing Center (CESGA), Santiago de
Compostela, e-mail:sparames@cesga.es.
3Universidade de Vigo.used open-source ticketing tool.
RT offers two main interaction channels: a web in-
terface (available only to support staff) and e-mail.
CESGA users interact with the system exclusively
via e-mail, sending and replying to messages that RT
automatically converts into tickets and associated re-
sponses. This dynamic has direct implications for the
quality of the stored text, as detailed in Section III-
B.
The history accumulated by CESGA in RT spans
over twenty years of conversations between support
staff and users, constituting an invaluable opera-
tional memory: a significant proportion of the inci-
dents received today are variants of problems solved
years ago. However, RT version 4.4.1 has severe lim-
itations in its built-in search: it does not index the
full ticket body, is case-sensitive, does not tolerate ty-
pos, does not normalize morphological variants, and
lacks any notion of semantic similarity. As a result,
both veteran and newly onboarded staff struggle to
locate relevant past cases, leading to duplicated ef-
fort, loss of institutional knowledge, and increased
mean resolution time.
This paper presentsFragata, a semantic ticket
retrieval system designed and deployed at CESGA
to mitigate these limitations.Fragataap-
plies theRetrieval-Augmented Generation(RAG)
paradigm [2], an approach that improves search qual-
ity by first retrieving the most relevant documents
from a knowledge base and presenting them as con-
text. Specifically, the system combines dense re-
trieval based onembeddings, numerical vector repre-
sentations of text meaning, with classical BM25 lexi-
cal retrieval [3] andrerankingviacross-encoders[4].
The system also integrates complementary sources
(the centre’s technical documentation, scientific ap-
plication manuals, and repositories) and is deployed
on a hybrid architecture comprising a virtual ma-
chine and the FinisTerrae III supercomputer, with
incremental ingestion and hot-reload of the search
engine.
The main contributions are: (i) the design of
a reproducible protocol for extraction, normaliza-
tion, and chunking of RT’s SQL history; (ii) a hy-
brid retrieval architecture with weighted fusion and
query-aware reranking; (iii) a hot-swap mechanism
that guarantees continuous availability during re-
indexing; and (iv) operational integration with an
HPC scheduler to offload expensive ingestion stagesarXiv:2604.13721v1  [cs.IR]  15 Apr 2026

without penalizing service latency.
II. Related work
A. Neural information retrieval
Information retrieval has undergone a profound
transformation in recent years thanks totrans-
former-based models [5]. These neural net-
work architectures, which underpin models such as
BERT [6], learn to represent texts as dense numer-
ical vectors (embeddings) that capture the seman-
tic meaning of words and sentences. Two texts
with similar meaning yield vectors that are close in
the embedding space, enabling the retrieval of rele-
vant documents by measuring vector distances rather
than relying on exact word matches. In particular,
Sentence-BERT [7] adapted BERT to efficiently pro-
duce sentence-level embeddings, and dense retrieval
systems such as DPR [8] demonstrated that this ap-
proach can outperform traditional lexical search on
question-answering tasks.
However, purely dense retrieval exhibits weak-
nesses on queries containing highly specific terminol-
ogy, proper names, or technical identifiers, where ex-
act word matching remains decisive. Consequently,
hybrid approaches that combine BM25, a classical
retrieval algorithm based on term frequencies, with
dense retrieval [9] have become the practical state of
the art. These systems benefit especially from a sub-
sequentrerankingstage, in which across-encoder[4]
jointly evaluates the query and each candidate to re-
order the results with greater precision.
The RAG paradigm [2], [10] integrates this re-
trieval chain into a complete pipeline: given a query,
the system retrieves the most relevant documents
from a knowledge base. This approach leverages
large volumes of information without needing to re-
train models, making it especially well-suited to do-
mains with accumulated knowledge such as technical
support.
B. Technical support automation
The application of natural language processing
techniques to ticketing systems has been explored
in various contexts. Potharaju et al. [11] analysed
corpora of network trouble tickets to infer problems
automatically, while Zhou et al. [12] proposed ma-
chine learning methods for intelligent ticket routing.
These works focus on classification and triage, but
do not address semantic retrieval over accumulated
ticket histories.
In the HPC domain specifically, published efforts
have focused predominantly on job monitoring and
automatic diagnosis systems, whereas the systematic
exploitation of knowledge contained in support his-
tories has received less attention.Fragataoccupies
precisely this niche: high-quality semantic retrieval
over a heterogeneous corpus dominated by real tech-
nical e-mail, with the operational requirements of a
production service.III. Data sources and preparation
Figure 1 shows the overall data processing
pipeline, which starts from the direct extraction of
the RT history and culminates in the generation of
vector representations (embeddings) that enable se-
mantic search.
A. RT history dump
The limitations of RT’s search engine forced us to
approach the problem from outside the tool. We de-
veloped specific SQL queries against RT 4.4.1’s rela-
tional database that extract, for each ticket, the rele-
vant metadata (identifier, queue, department, dates)
and the entirety of the messages (transactionsand
text-typeattachments) comprising the conversation
thread between staff and users. The dump spans
approximately twenty years of activity and is ma-
terialized as JSONL1files, a format chosen for its
simplicity of streaming processing and its good fit
with incremental pipelines.
B. Normalization and chunking
As noted above, most interactions in RT occur via
e-mail. When a user replies to a message, the e-
mail client automatically includes the text of pre-
vious messages in the conversation thread. This
causes the raw messages stored in each ticket to
contain references to prior messages, leading to cu-
mulative redundancy: the same information appears
partially or fully repeated across multiple thread en-
tries. Additionally, messages contain noise typical of
e-mail: quoted headers, personal signatures, institu-
tional banners, and inconsistent formatting.
The normalization phase removes this redundant
information and associated noise by applying a se-
quence of operations: character decoding, removal of
known headers and signatures, suppression of quoted
messages from earlier in the thread, and whitespace
collapsing. The goal is to present the retrieval sys-
tem with each piece of information contained in a
ticket exactly once, without duplications that could
distort the results.
After normalization, a chunking process segments
each conversation into overlapping fragments of
bounded length. This segmentation is necessary
because embedding models have a limit on the
text length they can process, and shorter, more fo-
cused fragments produce more precise vector rep-
resentations. Each chunk preserves identity meta-
data (ticket id,conversation id,chunk id),
filtering metadata (department,last updated,
source type), and ingestion traceability metadata.
C. Complementary sources
In addition to the ticket history,Fragatacan
ingest external technical documentation relevant to
1JSONL (JSON Lines) is a text format in which each line of
the file constitutes an independent, valid JSON object. Unlike
a conventional JSON file, JSONL allows line-by-line (stream-
ing) processing without loading the entire file into memory,
making it especially suitable for incremental data pipelines.

Fig. 1: Ticket processing pipeline: from SQL extraction of the RT history to the generation ofembeddingsfor semantic search.
CESGA users: the centre’s own web pages, scientific
application manuals in PDF format, and repository
documentation (READMEs and GitHub/GitLab
wikis). Each source has a dedicated pipeline that
produces rows in the same JSONL document, allow-
ing them to be indexed jointly with the tickets and
return unified results.
IV. Hybrid retrieval architecture
The core ofFragatais a retrieval engine imple-
mented in Python on FastAPI2, instantiated from
a declarative configuration (config/rag.yaml) and
served through a lifecycle manager (EngineManager)
described in Section V.
A. Initialization and artifacts
At startup, the engine detects the available com-
pute device: CPU or GPU (via NVIDIA’s CUDA
platform). GPU usage significantly accelerates em-
bedding computation and vector searches, reducing
response and indexing times. The engine then loads
theparaphrase-multilingual-MiniLM-L12-v2
model from thesentence-transformersfamily [7],
which produces 384-dimensional embeddings with
support for over 50 languages, making it especially
well-suited to CESGA’s trilingual corpus.
The engine reconstructs the FAISS [13]3index
from on-disk artifacts. FAISS can search millions
of vectors to find those closest to a given query
vector, which constitutes the basis of semantic re-
trieval: finding the text fragments whose meaning is
most similar to the user’s question. To avoid dupli-
cating the corpus in memory, documents are recon-
structed from FAISS’s internal docstore rather than
re-reading the complete JSONL.
In parallel, a BM25 [3]4retriever is
built over the same corpus, and the
mmarco-mMiniLMv2-L12-H384-v1[14] reranker
is loaded—a cross-encoder trained on mMARCO,
2FastAPI (https://fastapi.tiangolo.com) is a modern,
high-performance Python framework for building web APIs,
with native support for asynchronous operations and auto-
matic documentation generation.
3FAISS (Facebook AI Similarity Search,https://github.
com/facebookresearch/faiss) is an open-source library de-
veloped by Meta for efficient similarity search over large-scale
dense vectors.
4BM25 (Best Matching 25) is a classical information re-
trieval algorithm that scores documents based on the fre-
quency of query terms, adjusted for document length. Unlike
embeddings, it operates on exact word matches.the multilingual version of the MS MARCO bench-
mark corpus. This model jointly evaluates the query
and each candidate document to produce a relevance
score more precise than the initial retrieval.
B. Query variants
Before retrieval, the engine generates weighted
variants of the original query to maximize the prob-
ability of finding relevant documents. The vari-
ants include: the canonically normalized form of the
query; a single-edit-distance correction when a fre-
quent candidate is detected in the corpus vocabu-
lary; an intent-based expansion that adds terms as-
sociated with the query type (definition, installation,
containers, error/troubleshooting); and a translation
via a curated Spanish/English term dictionary.
This last variant addresses a characteristic aspect
of CESGA’s environment: trilingualism. Support
tickets contain text in Spanish, English, and Gali-
cian, as users freely employ all three languages in
their communications. Translating key terms into
English, the predominant language in technical doc-
umentation, standardizes query representation and
improves matching against corpus fragments, regard-
less of the language in which they were originally
written. Galician, due to its linguistic proximity
to Spanish, implicitly benefits from both Spanish-
language variants and the multilingual embeddings
used by the system. This diversification increases
recall on short or ambiguous queries and provides
robustness against the linguistic heterogeneity of the
corpus.
C. Dense retrieval, lexical retrieval, and fusion
For each variant, two channels are executed in
parallel: a semantic search over FAISS (semantic k
candidates) and a BM25 search (lexical k). The
candidate pool sizes are dynamically adjusted when
the query is very short, when a temporal filter is
applied, or when results are restricted to a specific
department, to compensate for subsequent pruning.
The fusion of results from both channels employs
Weighted Reciprocal Rank Fusion(WRRF) [15]:
s(d) =X
r∈Rwr·1
krrf+ rank r(d),(1)
wherew rcombines the global channel weight (se-
mantic or lexical) and the query variant weight.

WRRF is robust against the different score scales of
BM25 and FAISS and the heterogeneity introduced
by the variants, since it operates solely on relative
positions (rankings) rather than absolute scores.
D. Query-aware reranking
From the fused set, a number of candidates
max(rerank topn,4·final k) is selected, optionally
augmented with a semantic rescue and, for single-
token queries, an exact lexical match rescue. The
cross-encoder scores each candidate against several
prompts (original, capitalization variants, derived
variants) and the best score is retained.
Score adjustments (boostsand penalties) based on
domain heuristics are then applied. These adjust-
ments modify each candidate’s score based on spe-
cific signals: a document is boosted when the query
explicitly suggests a department and the document
belongs to that department; non-ticket sources are
penalized in short queries where tickets are more
likely to be relevant; low-information-density frag-
ments receive a reduced score; titles covering all
query terms are favoured; and exact matches are
boosted for single-term queries. These adjustments
incorporate domain knowledge from HPC technical
support that generic neural models do not capture
on their own.
E. Ticket-level post-processing
The engine operates internally at the chunk level,
but the support staff is presented information at the
ticket level via a link to the RT service hosted at
CESGA. The API layer deduplicates byticket id,
applies adaptive overfetch when deduplication leaves
the response short, and produces clean text snippets
by removing redundant prefixes and incorporating
page titles when they provide context. This separa-
tion between internal ranking (fragment) and exter-
nal presentation (ticket) makes it possible to opti-
mize retrieval quality without polluting the informa-
tion presented to the user.
V. Incremental ingestion and hot-swap
A production search service cannot afford down-
time for re-indexing.Fragataaddresses this re-
quirement with an ingestion architecture orches-
trated by Slurm, the job scheduler of FinisTerrae III,
and an atomic engine reload mechanism.
A. Job model
Every ingestion, whether initiated by API
(/ingest/web,/ingest/pdf,/ingest/repo-docs,
/ingest/rt-weekly) or by the scheduler, material-
izes as a job with a manifest (manifest.json) and an
execution log (job.log) persisted atomically. The
manifest tracks state (queued,running,succeeded,
failed), current stage, progress, validated request,
and per-stage metrics.B. Serialized critical section
The stages that mutate global state, dataset
merge, source catalogue update, FAISS index ap-
pend, and engine reload, execute under a global mu-
tation lock. Without this serialization, two concur-
rent jobs could corrupt each other’s dataset, append
deltas over inconsistent snapshots, or reload the en-
gine with artifacts from different generations. The
accepted cost is reduced concurrency in the critical
section, which is acceptable given the centre’s usage
pattern.
C. Incremental append with atomic promotion
The incremental append module loads the active
FAISS index, adds the delta documents, validates
index and docstore growth, writes to a staging lo-
cation, re-validates, and only then atomically pro-
motes the new index and rotates the previous one
to a timestamped backup. In the online flow, any
inconsistency triggers an explicit failure rather than
silently launching a full rebuild, allowing immediate
detection and diagnosis.
D. Engine hot-swap
After a successful index mutation, the orchestra-
tor requests an engine reload. The manager uses
two separate locks: one serializes builds and the
other protects the pointer to the active engine. The
new engine is built outside the final swap, and the
changeover only takes effect if the build completes
successfully, incrementing a generation counter. If
the build fails, the service continues responding with
the previous engine, ensuring graceful degradation
without service interruption.
E. Ingestion cycle and transactional watermark
The weekly batch pipeline extracts an incremental
window from RT determined by a reference times-
tamp (watermark) plus a safety overlap. This over-
lap is necessary to ensure that tickets whose last
modification occurred very close to the boundary of
the previous window are not missed: clock differ-
ences between servers, in-flight transactions at the
cutoff time, or data propagation delays could cause
a ticket modified just before the watermark to go un-
recorded. By slightly overlapping the windows, the
system reprocesses a small number of already-known
tickets, which are discarded by deduplication, but
avoids coverage gaps.
The pipeline prepares data by department, consol-
idates a global delta, updates the index, and reloads
the engine. The watermark is only confirmed if all
previous stages have succeeded, making the tempo-
ral pointer advance a transactional operation with
respect to the engine state.
VI. Deployment and FinisTerrae III
integration
Fragatais deployed in a virtual machine + su-
percomputer topology, illustrated in Figure 2. The
virtual machine hosts the static frontend served

Fig. 2: Deployment topology: the virtual machine hosts the web service and the API, while expensive indexing stages are
offloaded to the FinisTerrae III supercomputer equipped with an NVIDIA T4 GPU.
by Nginx5and the FastAPI backend managed by
systemd. Communication between frontend and
backend is configured via CORS (Cross-Origin Re-
source Sharing), a browser security mechanism that
controls which origins (domains) are allowed to make
requests to a server different from the one that served
the page. This configuration allows the frontend and
backend to be separated into independent domains
while maintaining service security.
The expensive ingestion and re-indexing stages can
be offloaded to the FinisTerrae III supercomputer
(Figure 2) via an offload mechanism controlled by
environment variables. The remote lifecycle mate-
rializes as observable stages (resource requested,
waiting resources,running remote,sync back)
whose metadata is persisted for traceability.
The resource release policy is configurable: in au-
tomatic mode, remote success implies implicit re-
lease; in explicit cancellation mode, job cancella-
tion is also invoked on success; on workload failure,
cancellation is always attempted. This integration
makes it possible to re-index large volumes of the
RT history without saturating the virtual machine,
taking advantage of FinisTerrae III nodes during pe-
riods of low occupancy.
VII. Preliminary results and discussion
The system is currently deployed in CESGA’s in-
ternal production environment. Qualitative results
on real queries from the support team show that
Fragatasystematically retrieves old tickets that
were invisible to RT’s native search, especially in the
following scenarios:
•Natural language queries in Spanish or Galician
about incidents originally reported in English,
5Nginx (https://nginx.org) is a high-performance HTTP
server and reverse proxy, widely used to serve static content
and as an entry point that distributes requests among an ap-
plication’s internal services.thanks to dictionary-based translation and the
multilingual embedding model.
•Morphological variants or typos in scientific ap-
plication names, which RT’s strict lexical search
could not resolve.
•Intent-based queries (e.g., “how to install X” or
“error when running Y”) where query meaning
matters more than the exact words used.
•Combined searches by content and date range or
department.
The main limitations identified are the dependence
on the quality of upstream e-mail cleaning, intent
heuristics based on static dictionaries rather than
learned models, a curated ES/EN vocabulary instead
of a general translation system, and a reranking cost
that grows with the product of the number of candi-
dates and the number of query variants.
VIII. Conclusions and future work
This paper has presentedFragata, a semantic
retrieval system for HPC support tickets based on
hybrid RAG and deployed on CESGA’s infrastruc-
ture. The combination of dense retrieval via embed-
dings, BM25 lexical retrieval, WRRF-weighted fu-
sion, and cross-encoder reranking, together with an
incremental ingestion architecture with engine hot-
swap and selective offloading to the FinisTerrae III
supercomputer, has made it possible to turn over
twenty years of Request Tracker history into an ef-
fectively searchable resource, overcoming the limita-
tions of RT 4.4.1’s native search.
Future work includes the incorporation of an au-
tomatically learned intent classifier in place of static
heuristics, quantitative evaluation with a curated
query set and relevance judgments generated by the
support team itself, extension of the system to a con-
versational interface with augmented generation that
synthesizes actionable answers citing source tickets,
and generalization of the deployment to other HPC
centres interested in exploiting their own support his-

tories.
Acknowledgements
This research project was made possible through
the access granted by the Galician Supercomputing
Center (CESGA) to its supercomputing infrastruc-
ture. The supercomputer FinisTerrae III and its per-
manent data storage system have been funded by the
NextGeneration EU 2021 Recovery, Transformation
and Resilience Plan, ICT2021-006904, and also from
the Pluriregional Operational Programme of Spain
2014-2020 of the European Regional Development
Fund (ERDF), ICTS-2019-02-CESGA-3, and from
the State Programme for the Promotion of Scientific
and Technical Research of Excellence of the State
Plan for Scientific and Technical Research and In-
novation 2013-2016 State subprogramme for scien-
tific and technical infrastructures and equipment of
ERDF, CESG15-DE-3114
Additionally, this work was carried out within the
framework of the Technological Upgrade Project for
the Computing and Data Node of the Galician Su-
percomputing Center (CESGA), funded by the Re-
covery, Transformation and Resilience Plan through
the NextGenerationEU instrument of the European
Union, within the Strategic Project for Economic Re-
covery and Transformation in Microelectronics and
Semiconductors (PERTE Chip), in accordance with
Royal Decree 714/2024.
References
[1] Best Practical Solutions, “Request tracker (rt),”https:
//bestpractical.com/request-tracker.
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Se-
bastian Riedel, and Douwe Kiela, “Retrieval-augmented
generation for knowledge-intensive NLP tasks,” in
Advances in Neural Information Processing Systems
(NeurIPS), 2020.
[3] Stephen Robertson and Hugo Zaragoza, “The probabilis-
tic relevance framework: BM25 and beyond,”Founda-
tions and Trends in Information Retrieval, vol. 3, no. 4,
pp. 333–389, 2009.
[4] Rodrigo Nogueira and Kyunghyun Cho, “Pas-
sage re-ranking with BERT,” inarXiv preprint
arXiv:1901.04085, 2019.
[5] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez,  Lukasz Kaiser,
and Illia Polosukhin, “Attention is all you need,”
inAdvances in Neural Information Processing Systems
(NeurIPS), 2017.
[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova, “BERT: Pre-training of deep bidi-
rectional transformers for language understanding,” in
Proceedings of NAACL-HLT, 2019.
[7] Nils Reimers and Iryna Gurevych, “Sentence-BERT: Sen-
tence embeddings using Siamese BERT-networks,” in
Proceedings of EMNLP, 2019.
[8] Vladimir Karpukhin, Barber Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih, “Dense passage retrieval for open-domain ques-
tion answering,” inProceedings of EMNLP, 2020.
[9] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong
Yang, Ronak Pradeep, and Rodrigo Nogueira, “Pyserini:
A Python toolkit for reproducible information retrieval
research with sparse and dense representations,” inPro-
ceedings of SIGIR, 2021.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang,
“Retrieval-augmented generation for large language mod-
els: A survey,”arXiv preprint arXiv:2312.10997, 2024.[11] Rahul Potharaju, Navendu Jain, and Cristina Nita-
Rotaru, “Juggling the jigsaw: Towards automated prob-
lem inference from network trouble tickets,” inProcee-
dings of NSDI, 2013.
[12] Wenjun Zhou, Wei Xue, Ramesh Baral, Hongyuan Zha,
and Robert Welch, “Smart ticket routing by multi-
criteria learning,” inProceedings of the ACM Conference
on Information and Knowledge Management (CIKM),
2016.
[13] Jeff Johnson, Matthijs Douze, and Herv´ e J´ egou, “Billion-
scale similarity search with GPUs,”IEEE Transactions
on Big Data, vol. 7, no. 3, pp. 535–547, 2021.
[14] Luiz Henrique Bonif´ acio, Hugo Abonizio, Marzieh
Fadaee, and Rodrigo Nogueira, “mMARCO: A mul-
tilingual version of the MS MARCO passage ranking
dataset,” inarXiv preprint arXiv:2108.13897, 2022.
[15] Gordon V. Cormack, Charles L. A. Clarke, and Stefan
B¨ uttcher, “Reciprocal rank fusion outperforms Con-
dorcet and individual rank learning methods,” inPro-
ceedings of SIGIR, 2009.
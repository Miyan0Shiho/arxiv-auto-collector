# Local Hybrid Retrieval-Augmented Document QA

**Authors**: Paolo Astrino

**Published**: 2025-11-13 13:34:32

**PDF URL**: [https://arxiv.org/pdf/2511.10297v1](https://arxiv.org/pdf/2511.10297v1)

## Abstract
Organizations handling sensitive documents face a critical dilemma: adopt cloud-based AI systems that offer powerful question-answering capabilities but compromise data privacy, or maintain local processing that ensures security but delivers poor accuracy. We present a question-answering system that resolves this trade-off by combining semantic understanding with keyword precision, operating entirely on local infrastructure without internet access. Our approach demonstrates that organizations can achieve competitive accuracy on complex queries across legal, scientific, and conversational documents while keeping all data on their machines. By balancing two complementary retrieval strategies and using consumer-grade hardware acceleration, the system delivers reliable answers with minimal errors, letting banks, hospitals, and law firms adopt conversational document AI without transmitting proprietary information to external providers. This work establishes that privacy and performance need not be mutually exclusive in enterprise AI deployment.

## Full Text


<!-- PDF content starts -->

Local Hybrid Retrieval-Augmented Document QA
Paolo Astrino
Università Ca’ Foscari Venezia, Italy
Abstract
Organizations handling sensitive docu-
ments face a critical dilemma: adopt
cloud-based AI systems that offer powerful
question-answering capabilities but com-
promise data privacy, or maintain local
processing that ensures security but deliv-
ers poor accuracy. We present a question-
answering system that resolves this trade-
off by combining semantic understanding
with keyword precision, operating entirely
on local infrastructure without internet ac-
cess. Our approach demonstrates that
organizations can achieve competitive ac-
curacy on complex queries across legal,
scientific, and conversational documents
while keeping all data on their machines.
By balancing two complementary retrieval
strategies and using consumer-grade hard-
ware acceleration, the system delivers reli-
able answers with minimal errors, letting
banks, hospitals, and law firms adopt con-
versational document AI without transmit-
ting proprietary information to external
providers. This work establishes that pri-
vacy and performance need not be mutu-
ally exclusive in enterprise AI deployment.
1 Introduction
The exponential growth of digital information
has created unprecedented challenges for or-
ganizations seeking to efficiently access and
utilize knowledge stored in heterogeneous doc-
ument formats. Traditional keyword-based
search methods fail to address complex queries
or synthesize information across multiple doc-
uments (Lewis et al., 2020). Moreover, state-
of-the-art AI language models require upload-
ing sensitive data to external cloud servers,
creating significant barriers for regulated in-
dustries and organizations handling propri-
etary information (European Data Protection
Board, 2025; Privacy International, 2024).Retrieval-AugmentedGeneration(RAG)ad-
dresses these challenges by combining the gen-
erative capabilities of large language models
(LLMs) with external knowledge retrieval sys-
tems (Wang et al., 2024b; Lin et al., 2024).
However, existing RAG implementations of-
ten rely on cloud-based processing or single
retrieval strategies that limit their effective-
ness in enterprise environments (Wang et al.,
2024a). Even systems that claim "local" oper-
ation frequently depend on external API calls
for embedding generation or LLM inference,
compromising data privacy.
This paper presents a fully local RAG
system that operates entirely on owned in-
frastructure without internet access. Docu-
ment processing, embedding generation using
BGE (HuggingFace), hybrid retrieval combin-
ing BM25 and dense vectors, and answer syn-
thesis via Ollama and Llama 3.2 all execute
on-premises. The hybrid strategy was tuned
across 10 weight configurations to identify
the optimal 30% sparse / 70% dense balance.
GPU acceleration yields 4.2×faster embed-
dingand3×fasterinferenceonconsumerhard-
ware. Hallucination is quantified using LLM-
as-Judge on over 1,500 query-answer pairs,
and multi-dimensional metrics—covering re-
trieval coverage, ranking quality, extractive
fidelity, and distributional statistics—are re-
ported on commodity hardware. This work
demonstrates that high-accuracy document
question-answering can be achieved without
sacrificing data sovereignty, offering a practi-
cal solution for healthcare, finance, and legal
sectors.arXiv:2511.10297v1  [cs.CL]  13 Nov 2025

2 Related Work
2.1 Retrieval-Augmented Generation
Lewis et al. (Lewis et al., 2020) introduced
the foundational RAG architecture, establish-
ing retrieval, augmentation, and generation as
core components. Recent surveys (Lin et al.,
2024) have identified key variants including
Fusion-in-Decoder and REALM approaches,
highlightingRAG’sadvantageoverfine-tuning
through dynamic knowledge access without
model retraining.
2.2 Hybrid Retrieval Strategies
Dense retrieval using transformer-based
embeddings excels at semantic similarity
but struggles with out-of-vocabulary terms
(Reimers and Gurevych, 2019; BAAI, 2024).
Sparse methods like BM25 provide precise
lexical matching but miss conceptual rela-
tionships (Wang et al., 2024a). Recent work
demonstrates that linear combination of
sparse and dense scores often outperforms
individual methods, though optimal weight-
ing strategies remain empirically determined
(Wang et al., 2024a).
2.3 Hallucination Detection in RAG
LLM hallucination remains a critical challenge
in production RAG systems (Zhang et al.,
2023). Recent approaches leverage LLM-as-
Judge methodologies for automated detection,
though reliability varies across domains and
question types (Zheng et al., 2023). Our evalu-
ation framework builds upon these established
methodologies to assess system reliability.
2.4 Enterprise AI Security
Cloud-based LLM services raise concerns
about data sovereignty and regulatory compli-
ance (Privacy International, 2024; European
Data Protection Board, 2025). Local pro-
cessing approaches address these concerns but
introduce hardware requirements and man-
agement complexity (Anthropic, 2024). Our
work bridges this gap through secure creden-
tial management and local document process-
ingwhile maintaining accessto advancedLLM
capabilities.3 Method
3.1 System Architecture
The system employs a fully local, three-
component architecture (Figure 1). All op-
erations—document parsing, embedding com-
putation, vector indexing, retrieval, and lan-
guage model inference—execute entirely on-
premises.
Frontend Component:Implemented us-
ing HTML, CSS, and JavaScript, providing an
intuitive web interface for document upload,
management, and conversational queries. Op-
erates locally without internet connectivity.
Client Component:A Flask-based
HTTP API server that handles user interac-
tions, performs input validation, and forwards
commands to the server via TCP sockets us-
ing a custom JSON protocol. No external API
calls are made; all processing is delegated to
the local server.
Server Component:The core system re-
sponsible for local document processing, hy-
brid retrieval, RAG orchestration, and LLM
integration. Hosts HuggingFace embeddings
(BGE), maintains in-memory vector stores,
manages BM25 indices, and interfaces with Ol-
lamaforlocalLlama3.2inference—allwithout
external network requests.
Figure 1: Architecture: frontend UI, client API, and
server (retrieval, RAG core, secure credentials) with
isolated secrets and local processing.

3.1.1 Security Framework
Security is enforced through strict separation
of concerns: all sensitive credentials are man-
aged exclusively on the server side through
secure environment variable storage. The
client never accesses authentication secrets,
document content, or internal processing logic.
This design ensures that even if the client or
frontend is compromised, attackers cannot ac-
cess credentials or manipulate core functional-
ity.
3.1.2 Communication Protocol
Inter-component communication uses a
lightweight JSON-based protocol over TCP
sockets. Each request contains acommand
key with additional parameters, while re-
sponses include astatuskey indicating
outcome. This approach provides simplicity,
extensibility, and language agnosticism while
avoiding the complexity of full JSON-RPC
implementation.
3.2 Hybrid Retrieval Strategy
The system implements a hybrid approach
combining semantic and keyword-based re-
trieval (Wang et al., 2024a):
Semantic Retrieval:Utilizes Hugging-
Face embeddings (BAAI/bge-base-en-v1.5)
(BAAI,2024)withLangChain’scachedembed-
ding framework (LangChain.js, 2025) for effi-
cient vector similarity search.
Keyword Retrieval:Employs BM25
ranking for precise lexical matching of techni-
cal terms and identifiers (Wang et al., 2024a).
Ensemble Integration:LangChain’s En-
sembleRetriever (LangChain.js, 2025) com-
bines results with weights tuned through eval-
uation across 10 configurations (10%-100%
sparse weight).
3.3 Document Processing Pipeline
Documents are processed through special-
ized loaders supporting PDF (PyPDFLoader),
CSV (CSVLoader), and JSON (JSONLoader)
formats (LangChain.js, 2025). Content is
chunked using RecursiveCharacterTextSplit-
ter with optimized parameters (300-400 to-
kens, 30-50 token overlap) to maintain seman-
tic coherence while enabling efficient retrieval.
These values were selected empirically under
constrained compute and time resources; abroader sensitivity sweep (e.g., 128–1024 to-
kens with learned dynamic merging) is de-
ferred to future work once additional resources
are available.
3.4 Local Language Model Integration
Answer generation employs Ollama (Ollama
Team, 2024), an open-source platform for run-
ning large language models locally. The sys-
tem uses Llama 3.2 as the generative model,
accessed via Ollama’s local API endpoint
(localhost:11434). This design ensures com-
plete data privacy by keeping all text genera-
tion on-premises, eliminating the need for ex-
ternal API calls to cloud-based LLM services.
The generation pipeline constructs prompts
that combine: (1) formatted chat history for
conversationalcontext, (2)retrieveddocument
chunks as grounding context, and (3) the cur-
rent user question. Ollama parameters are
configured for balanced output quality (tem-
perature: 0.6, top-p: 0.95, top-k: 40, max to-
kens: 1024), optimizing for factual accuracy
while maintaining natural language fluency.
This local inference approach provides sub-
second generation latency on consumer hard-
ware while maintaining strict data sovereignty
requirements.
3.5 GPU Acceleration
ThesystemautomaticallydetectsCUDAavail-
ability and uses GPU resources for both em-
bedding generation and LLM inference (Py-
Torch, 2025). Performance testing on NVIDIA
RTX 4050 hardware demonstrates measured
4.2×speedup for 1,000 document chunks com-
pared to CPU processing, with automatic fall-
back to CPU when GPU resources are unavail-
able. Ollama similarly benefits from GPU ac-
celeration, reducing inference latency by ap-
proximately 3×compared to CPU-only execu-
tion.
4 Experimental Setup
4.1 Datasets and Methodology
We conducted comprehensive evaluation
across three benchmark datasets (Table 1):
SQuAD v1.1 contains reading comprehen-
sion questions with guaranteed answers in
Wikipedia contexts; MS MARCO v2.1 com-
prises real Bing search queries with sparse

Table 1: Benchmark datasets for retrieval evalua-
tion.
Dataset Size Source
SQuAD v1.1 10,570 Wikipedia (Rajpurkar et al., 2016)
MS MARCO v2.1 9,706 Bing (Bajaj et al., 2016)
Natural Questions 3,610 Google (Kwiatkowski et al., 2019)
relevance patterns; Natural Questions in-
cludes real Google Search queries paired with
Wikipedia articles. Evaluation employed mul-
tiple metrics including Recall@K, Mean Recip-
rocalRank(MRR),exactmatchrates, anddis-
tributional statistics across 10 hybrid weight
configurations (10%-100% sparse weight).
4.2 Evaluation Metrics
We track four categories: (i)coverage(Re-
call@K / Hit@K), (ii)ranking quality(MRR,
mean/median rank, Rank-1 count), (iii)an-
swer fidelity(Exact Match, Answer Coverage),
and (iv)reliability(Hallucination Rate, Faith-
fulness 1–5, Confidence 1–5, Success = 1 -
Hallucination). Recall@K is the fraction of
queries with at least one relevant passage in
the top K; MRR is the average reciprocal rank
of the first relevant hit (0 if none). Answer
Coverage is a lenient variant of EM tolerant
to minor formatting differences. Reliability
metrics come from the LLM-as-Judge pipeline
(Table 5). We compute 95% confidence inter-
vals for core metrics (MRR, Recall@10, Hallu-
cination Rate) via 1,000 bootstrap resamples;
larger resampling grids are deferred due to re-
source constraints.
4.3 Hallucination Evaluation Protocol
We adopt an LLM-as-Judge framework to
quantify hallucination and faithfulness.Note:
While the production system oper-
ates 100% locally, we leverage Gem-
ini’s API exclusively for offline evalu-
ation to enable scalable assessment of
1,500+ query-answer pairs across three
datasets; no production user data is
transmitted externally.For each dataset, a
stratified sample of 500 queries (uniform over
question length tertiles) is selected. This size
balances statistical precision and constrained
evaluation budget: at observed rates (0.8%–
6.2%) a Wilson 95% interval remains within
roughly±2 percentage points while keeping
API cost and labeling time manageable underlimited resources. Larger stratified expansions
(e.g., 1k–2k) are deferred to future work when
additional compute and budget are available.
For every query we store: (i) user question, (ii)
retrieved context chunks (top 5, concatenated
with provenance IDs), and (iii) model answer
under the optimal hybrid retriever.
A judging prompt (abbreviated):
SYSTEM : You are a meticulous fact - checker .
Given QUESTION , CONTEXT ( retrieved passages ), and
ANSWER :
1. Is ANSWER fully supported by CONTEXT ? ( Yes /
Partially /No)
2. List unsupported claims if any .
3. Provide a faithfulness score 1-5 (5 = fully
grounded ).
4. Provide a confidence score 1-5 reflecting
certainty .
Return JSON : {" hallucination ": true |false , "
faithfulness ": int , " confidence ": int }
A response is flagged as a hallucination if
any critical claim lacks grounding (judge re-
turns false support or unsupported claim list
non-empty). Faithfulness and confidence use
discrete 1–5 scales. We reject malformed
JSON and re-query up to two times (retry
rate <1%). Inter-judge reliability was ap-
proximated by 50 double-coded samples (same
prompt, temperature 0 vs 0.2) yielding agree-
ment: hallucination label 96%, faithfulness ex-
act 82%, within±1: 100%.
Limitations: (i) Single-model dependence
may inherit judge bias; (ii) Partial credit not
linearly mapped to downstream utility; (iii)
Context truncation risk for unusually long ag-
gregated passages. Future work: multi-judge
majority voting and human adjudication sub-
set.
5 Results
5.1 Hybrid Weight Optimization
Evaluation across 10 weight configurations
(10% to 100% sparse weight in 10% incre-
ments) identified optimal hybrid balance. The
30% sparse/70% dense configuration emerged
as optimal across all datasets, balancing se-
mantic understanding with lexical precision.
5.2 Ablation: Single-Method vs
Hybrid Retrieval
Toisolatethecontributionofhybridfusion, we
compare pure sparse (BM25 only), pure dense
(embedding only), and the optimal hybrid con-
figuration. We report core ranking and cover-
age metrics. Hybrid delivers consistent gains

Table 2: Ablation of retrieval strategies (Sparse
= BM25, Dense = embeddings, Hybrid = 30/70).
NQ = Natural Questions; single-method NQ base-
lines omitted due to resource limits. MeanRk
nearly identical for MS MARCO sparse/dense
(early-rank ties).
Dataset Method MRR Recall@10 AnsCov MeanRk
SQuAD Sparse 0.717 0.840 0.952 4.66
SQuAD Dense 0.805 0.959 0.976 2.18
SQuAD Hybrid0.805 0.974 0.980 2.18
MS MARCO Sparse 0.103 0.480 0.406 0.60
MS MARCO Dense 0.315 0.605 0.482 0.60
MS MARCO Hybrid0.250 0.620 0.4870.62
NQ Hybrid0.813 0.978 0.9872.10
Table 3: Point estimates with 95% CIs (1,000 boot-
strap resamples). Hallucination for NQ pending.
Dataset Metric Value 95% CI Notes
SQuAD Recall@10 0.974 [0.971, 0.977] Stable
SQuAD HallucRate 0.8% [0.4%, 2.0%] 500 judged
MS MARCO Recall@10 0.620 [0.610, 0.630] Sparse rel.
NQ Recall@10 0.978 [0.973, 0.982] High cov.
NQ HallucRate — — Scheduled
over both single methods—particularly im-
proving MRR on SQuAD / Natural Questions
andRecall@10onMSMARCO—showingcom-
plementary error reduction.
5.3 Statistical Reliability and
Confidence Intervals
We quantify uncertainty for principal metrics
(MRR, Recall@10, Hallucination Rate) using
non-parametric bootstrap resampling (1,000
samples) over the query set. For each dataset
andmetric, wesample|Q|querieswithreplace-
ment, recompute the metric on the resampled
set, and repeat this process 1,000 times, tak-
ing the 2.5th and 97.5th percentiles as the 95%
confidence interval (CI). For binary propor-
tions(Recall@K,HallucinationRate)wecross-
validated bootstrap intervals against Wilson
score intervals; they were consistent (differ-
ences <0.2 percentage points).
The narrow intervals for SQuAD and Nat-
ural Questions indicate stable rankings; wider
Hallucination CIs reflect smaller judged sam-
ple size (500). Future work: stratified boot-
strap by question category and paired signifi-
cance testing (e.g., randomization test) for re-
trieval method deltas.
5.4 Overall Performance
Table 4 summarizes performance across opti-
mal configurations:
(a) Aggregate rates and means
(b) Score distributions
Figure 2: Reliability metrics: (a) low hallucination
with high faithfulness/confidence; (b) distributions
concentrated at 5 with modest degradation on MS
MARCO.
The hybrid approach consistently outper-
formed single-method baselines across all
datasets. Natural Questions validation con-
firmed 30% sparse/70% dense as the optimal
weighting, balancing ranking quality with an-
swer coverage.
5.5 Reliability Assessment
LLM-as-Judge evaluation using Gemini for
both answer generation and hallucination de-
tection across 500 queries per dataset (Zheng
et al., 2023; Zhang et al., 2023).
The low hallucination rates demonstrate
reliable answer generation grounded in re-
trieved context (Zhang et al., 2023), with MS
MARCO showing slightly higher rates due to
its challenging, sparse relevance patterns (Ba-
jaj et al., 2016).
Table 4: Optimal hybrid configuration perfor-
mance (30% sparse / 70% dense).
Dataset MRR Recall@10 Answer Cov.
SQuAD 0.805 0.974 0.980
MS MARCO 0.250 0.620 0.487
Natural Questions 0.813 0.978 0.987

6 Analysis
6.1 Hybrid Weight Optimization
Figure 3 illustrates performance trends across
sparse weight configurations. MS MARCO
shows steep degradation with increased spar-
sity, while SQuAD demonstrates remarkable
resilience until 50% sparse weight.
Analysis reveals distinct performance pat-
terns across datasets:
SQuAD:Demonstrates exceptional re-
silience to weight configuration changes, main-
taining high performance until 50% sparse
weight, reflecting its structured reading com-
prehension format.
MS MARCO:Shows steep performance
degradationwithincreasedsparsity, indicating
sensitivity to semantic understanding for real-
world search queries.
Natural Questions:Exhibits balanced
performance characteristics, confirming op-
timal 30% sparse/70% dense configuration
across diverse query types.
6.2 System Strengths
The modular architecture provides extensibil-
ity through well-defined component interfaces
and clean separation of concerns. Security
through local processing and credential isola-
tion addresses enterprise compliance require-
ments. The tuned hybrid retrieval strategy
combines semantic understanding with lexical
precision, achieving competitive performance
across diverse query types.
GPU acceleration provides substantial
performance improvements for embedding-
intensive operations, while hallucination
detection ensures reliability in production
environments.
6.3 Scalability Characteristics
Performance testing revealed single-user re-
sponse times of approximately 2 seconds for
small files, with 3–4×latency increase un-
der concurrent load (10 users) (PyTorch, 2025;
Johnson et al., 2019). Large file processing
Table 5: Hallucination metrics (n=500 judged
queries per dataset). Natural Questions pending.
Dataset Hallucination Rate Faithfulness Confidence Success
SQuAD 0.8% 4.93 4.87 99.2%
MS MARCO 6.2% 4.79 4.71 96.8%(100MB PDFs) requires approximately 4 min-
utes on CPU versus 1 minute with GPU accel-
eration (PyTorch, 2025). Memory efficiency
scales linearly with document collection size,
and GPU utilization reaches optimal perfor-
mance at 85–90% capacity under heavy load.
6.4 Cost-Effectiveness Analysis
Economic evaluation of LLM-as-Judge
methodology demonstrates cost advantages
(Zheng et al., 2023): Gemini evaluation costs
approximately $0.01 per call (versus $0.03
for GPT-4), with automatic budget controls
stopping at configurable limits ($50 default)
and resume capability ensuring zero data loss
during interruptions.
6.5 System Environment and
Resources
Table 6 documents the hardware, software
stack, and runtime performance characteris-
tics to support reproducibility and contextu-
alize efficiency claims.
All experiments executed on the above envi-
ronment unless otherwise stated. Performance
may vary with alternative embedding models,
storage backends, or GPU architectures.
7 Conclusion
Thispaperpresentedasecure, localRAGchat-
bot system that addresses enterprise needs for
document-based conversational AI. The tuned
hybrid retrieval strategy (30% sparse, 70%
dense) achieved superior performance across
multiple benchmarks while maintaining data
privacy through local processing.
Key findings show that hybrid retrieval
consistently outperforms single-method ap-
proaches, GPU acceleration yields major
speedups, LLM-as-Judge reveals low error
rates, and the modular design supports secure
enterprise deployment.
The system demonstrates that effective
RAG performance can be achieved without
compromising data security, offering a prac-
tical solution for organizations requiring AI-
powered document analysis while maintaining
regulatory compliance and data sovereignty.
Limitations
Throughput is limited by synchronous re-
trieval and single-node design, causing tail

(a) SQuAD: Stable until >50%
sparse
(b) MS MARCO: High sparsity sen-
sitivity
(c) Natural Questions: Balanced
optimum
Figure 3: Hybrid weight sensitivity across datasets. Each panel summarizes retrieval quality vs sparse
weight (10%–100%): composite plots include MRR, Recall@K, answer coverage, and rank / degradation
curves. The 30% sparse / 70% dense configuration achieves near-optimal balance across all datasets;
increasing sparsity causes sharp degradation for MS MARCO, gradual decline for SQuAD, and modest
impact for Natural Questions.
Table 6: Execution Environment and Runtime Characteristics. Latencies are median unless noted.
Throughput measured on hybrid retrieval (k= 10).
Component Specification / Measurement
CPU 12-core AMD Ryzen (3.8 GHz boost)
GPU NVIDIA RTX 4050 Laptop (6GB VRAM)
System Memory 32 GB DDR5
Storage NVMe SSD (3.2 GB/s seq. read)
OS Windows 11 (64-bit)
Python 3.11.x
Core Libraries torch>=1.11, faiss-cpu>=1.7.0, langchain 0.x
ML / NLP Stack transformers>=4.20, sentence-transformers>=2.0, scikit-learn>=1.0
Data / Utils numpy>=1.21, pandas>=1.3, tqdm>=4.62, python-dotenv>=0.19, psutil>=5.8
Embedding Model BGE Base (HuggingFace)
Vector Index FAISS (L2 / Inner Product)
Sparse Scorer BM25 (in-memory)
Batch Size (Embeddings) 32 chunks (GPU), 8 (CPU fallback)
Median Query Latency 2.0 s (single user)
P90 Query Latency 3.4 s (single user)
Concurrent (10 users) 6.8 s median (async scheduling)
Embedding Speedup 4.2×GPU vs CPU (1k chunks)
Max GPU Utilization 88% (hybrid retrieval stress)
Peak Memory (10k Chunks) 5.1 GB RAM, 3.2 GB VRAM
Cost (Hallucination Eval) ~$5 per 500 judged queries (Gemini)
Secrets Management .env + server-side isolation
Reproducibility Artifacts Versioned config + cached embeddings
Figure 4: Benchmark positioning: top three hy-
brid weights vs tier bands (MS MARCO normalized,
SQuAD absolute). Chosen 30/70 mix sits solidly in
Competitive while retaining acceptable MS MARCO
performance.latency under load. Future work includes
async batching and sharded indices. Hallu-
cination judgments rely on a single LLM-as-
Judge; multi-judge ensembles and significance
testing are planned. Benchmarks are English-
only and web-skewed (SQuAD, MS MARCO,
Natural Questions); multilingual and domain-
specific evaluation is deferred. Security lacks
RBAC, PII redaction, and retrieval poisoning
detection—these will be addressed in future
iterations through embedding-space anomaly
scoring, signed chunk manifests, role-based ac-
cess policies, and red-team audit telemetry.

Ethics Statement
Although production inference is fully local,
hallucinationevaluationusedGemini’sAPIon
benchmark-derived queries only—no user data
was transmitted. This enables scalable assess-
ment but introduces external dependency in
evaluation, not deployment. Future work will
explore local judge models (e.g., fine-tuned
Llama) to eliminate this for organizations re-
quiring fully offline pipelines.
The low hallucination rates (0.8%–6.2%, Ta-
ble 5) risk over-confidence. Users may accept
answers without verification, especially when
confidence scores are high. We recommend UI
disclaimers and human review for high-stakes
decisions. LLM-as-Judge filtering is proba-
bilistic and may miss subtle omissions or flag
correct paraphrases. Deployments should log
adjudication rationales and enable audit sam-
pling.
The benchmarks are English and web-
centric, potentially under-serving specialized
or multilingual domains. Web/Wikipedia
sources under-represent minority languages
and domain-specific discourse (legal, medical).
Future evaluation will incorporate domain-
internal corpora and bias diagnostics (entity
coverage, dialectal robustness). English-only
embeddings may systematically fail for non-
English queries; mitigation includes multilin-
gual retrievers (mBGE) and language detec-
tion with dynamic routing.
Local processing reduces exposure (Privacy
International, 2024; European Data Protec-
tion Board, 2025) but enables internal data
mining. Access logging and rate limiting
should accompany deployment. Adversarial
document injection remains a risk. Current
sanitization is heuristic; future defenses in-
clude embedding-space anomaly detection and
signed chunk manifests. Releasing weight
sweeps and evaluation scripts supports trans-
parency, but omitting raw corpora may limit
replication fidelity.
Deployments should include human over-
sight, periodic audits, multilingual expansion,
drift monitoring, uncertainty signaling, and
red-team testing. While the system advances
secure retrieval for enterprise contexts, ethical
stewardship requires ongoing bias assessment,
multilingual inclusion, and safeguards againstover-reliance and adversarial misuse.
Acknowledgments
We thank the open-source community for pro-
vidingthefoundationaltoolsanddatasetsthat
made this research possible. Special acknowl-
edgment to the developers of LangChain, Hug-
gingFaceTransformers, andFAISSfortheirro-
bust implementations that enabled rapid pro-
totyping and evaluation.
References
Anthropic. 2024. Model context proto-
col.https://www.anthropic.com/news/
model-context-protocol.
Paolo Astrino. 2025. Local hybrid retrieval-
augmented document qa (code repository).
https://github.com/PaoloAstrino/Local_
RAG_. Commit: 30c52ff.
BAAI.2024. Bge: Baaigeneralembedding.https:
//github.com/FlagOpen/FlagEmbedding.
P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao,
X. Liu, R. Majumder, A. McNamara, B. Mitra,
T. Nguyen, M. Rosenberg, X. Song, A. Stoica,
S. Tiwary, and T. Wang. 2016. Ms marco: A
human generated machine reading comprehen-
sion dataset. InProceedings of the Workshop on
Cognitive Computation: Integrating neural and
symbolic approaches.
EuropeanDataProtectionBoard.2025. Aiprivacy
risks and mitigations in llms.https://www.
edpb.europa.eu/system/files/2025-04/
ai-privacy-risks-and-mitigations-in-llms.
pdf.
J. Johnson, M. Douze, and H. Jégou. 2019. Billion-
scale similarity search with gpus.IEEE Trans-
actions on Big Data, 7(3):535–547.
T. Kwiatkowski, J. Palomaki, O. Redfield,
M. Collins, A. Parikh, C. Alberti, D. Epstein,
I. Polosukhin, J. Devlin, K. Lee, K. Toutanova,
L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai,
J. Uszkoreit, Q. Le, and S. Petrov. 2019. Natu-
ral questions: A benchmark for question answer-
ingresearch.Transactions of the Association for
Computational Linguistics, 7:453–466.
LangChain.js. 2025. Introduction to
langchain.js.https://js.langchain.com/
docs/introduction.
P. Lewis, E. Perez, A. Piktus, F. Petroni,
V. Karpukhin, N. Goyal, H. Küttler, M. Lewis,
W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela.
2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. InAdvances

in Neural Information Processing Systems, vol-
ume 33.
W. Lin and 1 others. 2024. A survey on retrieval-
augmentedgenerationforlargelanguagemodels.
arXiv preprint arXiv:2410.12837.
Ollama Team. 2024. Ollama: Get up and run-
ning with large language models locally.https:
//ollama.com. Open-source platform for local
LLM inference.
Privacy International. 2024. Large
language models and data protec-
tion.http://privacyinternational.
org/explainer/5353/
large-language-models-and-data-protection.
PyTorch. 2025. Cuda semantics. PyTorch Docu-
mentation,https://docs.pytorch.org/docs/
stable/cuda.html.
P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang.
2016. Squad: 100,000+ questions for machine
comprehension of text. InProceedings of the
2016 Conference on Empirical Methods in Nat-
ural Language Processing, pages 2383–2392.
N. Reimers and I. Gurevych. 2019. Sentence-
bert: Sentence embeddings using siamese bert-
networks. InProceedings of the 2019 Conference
on Empirical Methods in Natural Language Pro-
cessing and the 9th International Joint Confer-
ence on Natural Language Processing (EMNLP-
IJCNLP), pages 3982–3992.
L. Wang, S. Wang, J. Wang, and 1 others. 2024a.
Hybrid retrieval for open-domain question an-
swering.arXiv preprint arXiv:2404.07220.
S. Wang and 1 others. 2024b. Retrieval augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997.
Y. Zhang, H. Zhang, Y. Wang, and 1 others. 2023.
Siren’s song in the ai ocean: A survey on halluci-
nation in large language models.arXiv preprint
arXiv:2309.01219.
L. Zheng, W. Chiang, Y. Sheng, and 1 others. 2023.
Judging llm-as-a-judge with mt-bench and chat-
bot arena.arXiv preprint arXiv:2306.05685.
A Implementation Details
A.1 Threat Model and Privacy
Considerations
The system assumes an honest-but-curious ad-
versary and a potentially compromised client,
but a trusted server. Attack surfaces include
client-server channel interception, prompt in-
jection via uploaded documents, and creden-
tial exfiltration. All document parsing, embed-
ding, and retrieval occur server-side; raw textnever leaves the host. API keys are loaded
from local.envfiles and never transmitted
to clients or echoed in logs. Prompt injec-
tion is mitigated via token sanitization and
length caps, though a structured allowlist is
planned. Retrievedchunksincludeprovenance
(filename + span) for auditability and forensic
inspection. Process isolation between retrieval
and generation will narrow lateral movement
surface. Logging stores only content hashes,
not raw text, reducing exposure while preserv-
ing cache traceability.
Residual risks include model inversion on
generated answers (low due to local-only cor-
pus), adversarial chunk crafting to skew hy-
brid weighting (requires future anomaly de-
tectors), and denial-of-service via pathological
uploads (mitigated by size/type checks). No
user-identifying analytics are collected; the on-
premises footprint facilitates compliance align-
ment.
A.2 Reproducibility and Artifact
Availability
To support independent verification, all config-
urations use a fixed random seed for chunking
and retrieval. Embeddings and BM25 indices
are cached on disk, enabling cold/warm start
timing replication. Evaluation scripts out-
put CSVs (per-dataset weight sweeps, hallu-
cination judgments) toevaluation/results/
with schema headers retained. Metrics are for-
malized in Section 4.2 with unambiguous for-
mulas. The hardware and software stack is
documented in Table 6, including Python 3.11,
FAISS, and BGE; any deviations should be
stated when reporting alternative results.
GPU nondeterminism is minimal (MRR
variance <0.002 over 5 runs); stricter
reproducibility can be achieved with
CUBLAS_WORKSPACE_CONFIG=:16:8. Com-
mit hashes are recorded alongside evalu-
ation CSVs (future automation planned)
to bind results to code state. Full
source code (retrieval pipeline, evaluation
scripts, hallucination judge) is available at
https://github.com/PaoloAstrino/Local_RAG_
(commit30c52ff) (Astrino, 2025).
Recommended reproduction: (1) load cor-
pus, (2) build embedding + BM25 indices,
(3) execute weight sweep script, (4) run boot-
strap script for CIs, (5) sample queries and in-

voke hallucination judge pipeline. Discrepan-
cies >1.5% absolute Recall@10 or >0.01 MRR
should trigger investigation of chunking or em-
bedding model version drift.
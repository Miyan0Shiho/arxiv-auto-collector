# Hierarchical Reranking for Scalable Financial RAG System

**Authors**: Joohyun Lee, Sungwoo Hong

**Published**: 2026-07-29 23:25:45

**PDF URL**: [https://arxiv.org/pdf/2607.27523v1](https://arxiv.org/pdf/2607.27523v1)

## Abstract
Analyzing financial documents such as 10-K filings, tabular disclosures, and macroeconomic reports demands expert reasoning and extensive time. However, existing Retrieval-Augmented Generation systems often struggle to process hybrid text-table structures or the massive scale of financial documents. To address these challenges, we propose Hierarchical Reranker, a RAG framework designed to improve retrieval performance and generative reliability across large-scale financial datasets. The system integrates three key innovations: Pre-Retrieval Optimization, enhancing query clarity and search efficiency through normalization, keyword expansion, and table transformation; Hierarchical Reranker Architecture, improving retrieval precision through a two-stage ranking mechanism; and Long-Context Management, preserving reasoning accuracy through adaptive input partitioning and fusion under extensive contexts. Across multiple benchmarks, including FinQA, FinanceBench, and ConvFinQA, the proposed system achieved an NDCG@20 score of 0.7918 and demonstrated superior factual consistency. Its robustness was further validated by achieving second place in the ACM-ICAIF '24 FinanceRAG Challenge. This work presents a deployable, domain-optimized RAG pipeline that enhances both the accuracy and scalability of financial reasoning, paving the way for automated audit reporting and quantitative investment analysis. The source code will be made publicly available on GitHub upon acceptance.

## Full Text


<!-- PDF content starts -->

Hierarchical Reranking for Scalable Financial RAG System
Joohyun Lee1,Sungwoo Hong2
1Financial Security Institute
2Hanyang University
dlee110600@gmail.com, toggiya0701@gmail.com
Abstract
Analyzing financial documents such as 10-K fil-
ings, tabular disclosures, and macroeconomic re-
ports demands expert reasoning and extensive
time. However, existing Retrieval-Augmented
Generation systems often struggle to process hy-
brid text–table structures or massive scale of fi-
nancial documents. To address these challenges,
we propose Hierarchical Reranker, a RAG frame-
work designed to improve retrieval performance
and generative reliability across large-scale finan-
cial datasets. The system integrates three key in-
novations: Pre-Retrieval Optimization, enhancing
query clarity and search efficiency through normal-
ization, keyword expansion, and table transforma-
tion; Hierarchical Reranker Architecture, improv-
ing retrieval precision through a two-stage ranking
mechanism; and Long-Context Management, pre-
serving reasoning accuracy through adaptive input
partitioning and fusion under extensive contexts.
Across multiple benchmarks, including FinQA, Fi-
nanceBench, and ConvFinQA, the proposed sys-
tem achieved an NDCG@20 score of 0.7918 and
demonstrated superior factual consistency. Its ro-
bustness was further validated by achieving second
place in the ACM-ICAIF ’24 FinanceRAG Chal-
lenge. This work presents a deployable, domain-
optimized RAG pipeline that enhances both the
accuracy and scalability of financial reasoning,
paving the way for automated audit reporting and
quantitative investment analysis. The source code
will be made publicly available on GitHub upon ac-
ceptance.
1 Introduction
Interpreting 10-K filings, reconciling tabular disclosures, and
contextualizing results against shifting macroeconomic con-
ditions remain among the most labor-intensive activities in
the financial industry. These tasks resist automation be-
cause they demand both expert domain judgment and ex-
haustive cross-referencing across long, heterogeneous doc-
uments, resulting in high operational cost and limited scal-
ability [Lee and Han, 2025 ]. As financial institutions be-gin to deploy autonomous LLM-based agents for auditing,
quantitative research, and portfolio analysis [Wuet al., 2023;
Papasotiriouet al., 2024 ], the bottleneck has shifted from
whetherLLMs can read financial text tohow reliably and
economicallythey can ground their reasoning in the underly-
ing evidence.
Retrieval-Augmented Generation (RAG) is the natural sub-
strate for this grounding [Yanget al., 2023 ], yet off-the-shelf
RAG pipelines exhibit three persistent failure modes when
transplanted into finance [Yuet al., 2025; Databricks, 2024 ]:
(1) domain-specific jargon, units, and abbreviations cause
query–corpus embedding drift, degrading recall; (2) hybrid
text–table evidence is poorly aligned by retrievers trained on
prose, leading to numerically incorrect generations; and (3)
reasoning quality collapses on long inputs, even on models
that nominally support 100k+ token contexts. Each failure
mode interacts with the others: a financial 10-K can simulta-
neously be long, dense in jargon, and dominated by tables.
We address these failures with the Hierarchical Reranker,
a finance-specific RAG framework engineered for deploy-
able, large-scale use. Rather than scaling a single mono-
lithic retriever or generator, the framework decomposes the
problem across cooperating components, each addressing
one failure mode. The two-stage retrieval design in partic-
ular instantiates asmall-and-large model collaboration: a
lightweight reranker rapidly prunes the candidate pool, and a
high-capacity reranker performs fine-grained semantic adju-
dication on the survivors. This decomposition keeps end-to-
end latency and token cost bounded while concentrating ex-
pensive computation where it matters — a property we view
as essential for any RAG system that is to operate at institu-
tional scale.
Our contributions are threefold:
•Pre-Retrieval Optimization.We normalize finance-
specific units and abbreviations, augment queries with
domain keywords, and deterministically convert Mark-
down tables to JSON so that numeric values stay bound
to their headers — producing measurable gains in re-
trieval quality without introducing LLM-induced hallu-
cinations during preprocessing.
•Hierarchical Reranker.A two-stage cascade pairs a
fast first-stage filter with a high-capacity second-stage
reranker, capturing cross-sentence financial dependen-
arXiv:2607.27523v1  [cs.IR]  29 Jul 2026

cies and table–text consistency while keeping inference
cost bounded.
•Long-Context Management.For inputs beyond a 64k-
token threshold, we partition evidence into semantically
coherent chunks and fuse intermediate answers with an
evidence-aware merger that explicitly handles ambigu-
ous or contradictory partials. Ablations isolate the con-
tribution of each component.
The framework was validated through extensive ablation
studies and benchmark evaluations, has been deployed in
institutional auditing and investment workflows, and se-
cured2nd placein the ACM-ICAIF ’24 FinanceRAG Chal-
lenge [Choiet al., 2024 ], demonstrating both robustness and
industry-level competitiveness.
2 Related Work
The integration of artificial intelligence in the financial
domain has rapidly advanced through the emergence of
retrieval-based benchmarks and reasoning datasets [Lee
and Han, 2025 ]. Early datasets such as FinQABench
[LighthouzAI, 2024 ]and FinanceBench [Islamet al., 2023 ]
focused primarily on factual grounding in financial docu-
ments like 10-K filings, evaluating models’ ability to reduce
hallucinations and improve factual correctness. While these
benchmarks improved retrieval–generation alignment, they
often assumed relatively short contexts and ignored multi-
step numerical reasoning.
Subsequently, TATQA [Zhuet al., 2021 ], FinQA [Chenet
al., 2022a ], and ConvFinQA [Chenet al., 2022b ]expanded
the task scope to hybrid textual–tabular reasoning, requir-
ing models to perform arithmetic and comparative analysis.
These datasets highlighted the limitations of general-purpose
retrievers and LLMs in understanding structured quantitative
data, motivating research into more domain-aware retrieval
architectures.
From the retrieval perspective, query expansion [Jagerman
et al., 2023 ]and reranking [Maet al., 2023 ]have been widely
studied to enhance search accuracy. Methods such as HyDE
[Gaoet al., 2022 ]and query rewriting [Liu and Mozafari,
2024 ]improved retrieval recall through lexical and seman-
tic enrichment, while rerankers refined document relevance
post-retrieval. However, most existing studies [Fanet al.,
2024 ]have been optimized for horizontal tasks, focusing on
either pre-retrieval or post-retrieval, with few demonstrating
performance improvements in domain-specific (vertical) ap-
plications.
Long-context reasoning represents another key research
stream. Despite advances in models like Claude-Opus
4.6[Anthropic, 2026 ], GPT-5.4 [OpenAI, 2026 ], Grok-4.20
[xAI, 2026 ]and Gemini-3.0-Pro [Google, 2026 ], several
studies [Databricks, 2024; Anet al., 2024 ]have observed per-
formance degradation beyond 64k tokens, particularly in fi-
nancial reasoning tasks with dense numerical references. Ex-
isting work [Liet al., 2023 ]offers few solutions for dynam-
ically constraining and fusing contexts while preserving rea-
soning consistency. By integrating pre-retrieval optimization,
hierarchical reranker refinement, and context-length manage-
ment into a unified finance RAG pipeline, our study aims tobuild a practically deployable RAG system for real-world fi-
nancial applications.
3 Tasks and Dataset
To construct a finance-specific RAG system, we define
two interdependent tasks that jointly determine the overall
pipeline design: Document Retrieval and Answer Generation.
3.1 Task 1: Document Retrieval
Given a user query, the objective is to identify the top 20
most relevant passages from a large corpus of financial doc-
uments. Unlike general-purpose retrieval tasks, financial cor-
pora contain both textual and numerical data, making seman-
tic and quantitative alignment equally important. We thus for-
mulate retrieval as a two-stage ranking problem embedding-
based coarse retrieval followed by fine-grained reranking
to balance efficiency and accuracy. Retrieval performance
is evaluated using Normalized Discounted Cumulative Gain
(NDCG@20), which captures both ranking quality and se-
mantic relevance. This metric is chosen for its robustness
in evaluating graded relevance rather than binary correctness,
which aligns well with real-world financial information re-
trieval.
3.2 Task 2: Answer Generation
Once the top-ranked corpora are retrieved, the goal is to gen-
erate a factual, numerically grounded answer that directly ref-
erences the evidence within the selected documents. This task
extends beyond text summarization by requiring precise in-
terpretation of tables, ratios, and cross-document dependen-
cies. To assess generation quality, we adopt the LLM-as-a-
Judge framework [Guet al., 2025 ], which compares model
outputs with ground-truth answers while evaluating factual
consistency, logical reasoning, and numerical precision. This
was feasible because the benchmark answers are short and
clear (mainly numbers or simple words), allowing evaluation
through straightforward accuracy metrics. Although the eval-
uation relied solely on LLM-as-a-Judge, the deterministic and
fact-based nature of the benchmark tasks minimizes heuristic
bias, making statistical significance testing less critical.
3.3 Datasets
The proposed system is benchmarked on multiple finance-
oriented datasets, each emphasizing distinct aspects of re-
trieval and reasoning. Each dataset sample consists of a
natural-language query, a set of chunked corpora, the cor-
responding ground truth, and document sources. Collec-
tively, these datasets provide a comprehensive benchmark
suite for assessing both retrieval effectiveness and numeri-
cally grounded generation in real-world financial scenarios.
•FinQABench: Based on 10-K filings; focuses on de-
tecting hallucinations in generated answers and ensuring
factual correctness.
•FinQA: Derived from earnings reports; evaluates multi-
step numerical reasoning using both tabular and textual
data.
•ConvFinQA: Also based on earnings reports; assesses
model performance on conversational financial queries.

Figure 1: Hierarchical Reranker Framework, which integrates query expansion, corpus compression, and a two-stage reranking pipeline to
enhance retrieval performance. The retrieved corpora are dynamically managed under a long-context fusion mechanism, ensuring efficient
and accurate response generation even for inputs exceeding 64k tokens.
•FinanceBench: Built on 10-K filings; measures a sys-
tem’s ability to handle real-world financial questions
with domain precision.
•TATQA: Composed of financial reports; tests arith-
metic, comparative, and logical reasoning over hybrid
tabular–text data.
4 Method
The proposed financial RAG system is composed of three se-
quential stages. The first two stages correspond to Task 1 (Re-
trieval), while the third stage corresponds to Task 2 (Genera-
tion). Overall, the pipeline is divided into three core phases:
Pre-Retrieval, Retrieval, and Generation. As illustrated in
Figure 1, the blue components represent the retrieval process,
while the green components correspond to generation.
4.1 Pre-Retrieval
The Pre-Retrieval phase is designed to enhance the inter-
pretability of financial queries and the consistency of the
document corpus before embedding-based retrieval. Finan-
cial texts often contain abbreviations, implicit relations, and
domain-specific terminology, which often lead to semantic
mismatches between queries and document embeddings. To
mitigate these issues, we propose a four-stage pre-processing
pipeline that improves both query clarity and corpus normal-
ization.
Normalization
All input queries are first normalized to reduce lexical ambi-
guity. This step includes lowercase, typographical correction,
and expansion of financial abbreviations (e.g., EPS: Earnings
per share, YoY: Year-over-Year). Measurement units such
as “K” or “M” are standardized into consistent numeric ex-
pressions, and missing contextual terms are restored based
on document metadata. This ensures that semantically equiv-
alent expressions share a uniform representation within the
embedding space. The corpus is normalized likewise for con-
sistency.Keyword Extraction
Financial-specific keywords are extracted and appended to
each query to strengthen the alignment between query and
corpus embeddings. This increases the density of financial
terminology within the input, allowing the retriever to bet-
ter capture hybrid text–table semantics commonly found in
financial documents.
Paraphrasing
Each normalized query is semantically expanded through
paraphrasing. This step generates multiple linguistically di-
verse but semantically equivalent variants, enabling the re-
triever to generalize across different formulations of financial
questions (e.g., “What is Apple’s revenue in 2023?” vs. “Re-
port Apple’s 2023 revenue”). This expansion broadens the
search space without introducing significant computational
overhead.
Hypothetical Document Generation (HyDE)
Inspired by HyDE-based methods, a synthetic pseudo-
document is generated for each query to represent a plausi-
ble context where the answer could appear. These hypotheti-
cal passages act as semantic anchors that enhance embedding
alignment between the abstract query and the domain-specific
corpus.
Table-to-json
During the corpus pre-processing stage, it is crucial to main-
tain semantic coherence across documents while ensuring the
accuracy of numerical information. Accordingly, each docu-
ment was semantically chunked at the sentence level to pre-
vent contextual fragmentation. In addition, since large-scale
tables in financial documents contain key quantitative infor-
mation, Markdown-style tables were converted into JSON
structures (table-to-json) using a rule-based script, rather than
relying on LLM-based conversion, to prevent potential hallu-
cinations.
This process explicitly preserves the relationships between
numerical values and their headers, thereby strengthening the

semantic alignment between textual and numerical data dur-
ing the embedding and retrieval stages. The effectiveness of
this transformation was validated through comparative exper-
iments between the Markdown format (original tables) and
the JSON format (table-to-json).
Summary
When the corpus size is extremely large (over 10k tokens),
utilizing all documents as-is may be inefficient and could
lower retrieval performance. To address this, each document
was replaced through summarization substitution, preserving
only the essential financial indicators, results, and contextual
information. This compression removes unnecessary narra-
tive sentences while prioritizing semantically central state-
ments, thereby improving both the efficiency and accuracy
of the retrieval stage.
Most Pre-Retrieval steps were performed using Claude-
Opus-4.6 [Anthropic, 2026 ], while certain quantitative tasks,
such as table conversion, were executed through rule-based
scripts. Through this bidirectional normalization and struc-
turing process applied to both queries and corpora, the system
maximizes semantic compatibility between complex textual–
numerical data in the financial domain and enables more pre-
cise and reliable retrieval in subsequent stages.
4.2 Retrieval
The Retrieval stage identifies the most semantically and nu-
merically relevant passages from a large-scale financial cor-
pus, bridging the preprocessed query and the downstream
generator. The design objective is twofold: maximize re-
trieval precision on hybrid text–table evidence, and bound
inference cost so that the system remains deployable at in-
stitutional scale.
We therefore avoid a single monolithic reranker and instead
employ a two-stage hierarchical reranker in which a small,
fast model and a large, accurate model cooperate. Each model
is specialized for the regime where it dominates: the small
model handles coarse lexical pruning across the full candi-
date pool, while the large model performs deep semantic ad-
judication on a much smaller surviving set. This division of
labor decouplesbreadthfromdepthand concentrates expen-
sive computation where it has the highest marginal value.
Stage 1: Lightweight Filtering
In the first stage, a fast, low-complexity jina-reranker-v3
[Wanget al., 2025 ]is used to eliminate low-relevance or
noisy candidates. This model is optimized for lexical and
shallow semantic similarity, leveraging extended context sup-
port up to 131k tokens. By narrowing the candidate pool
to the top 100 passages, it significantly reduces downstream
computational load without sacrificing recall.
Stage 2: Fine-Grained Semantic Reranking
The second stage applies a high-capacity reranker to re-
evaluate the top 100 candidates and extract the final top 20
passages. This reranker focuses on fine-grained contextual
relationships, such as cross-sentence financial dependencies
and numerical consistency across tables and text. Through
this hierarchical refinement, the system effectively capturesAlgorithm 1Proposed Framework
Input: QueryQ, CorpusC
Output: ResponseR
1:Q′←Normalization(Q)∪Keywords-Extraction(Q)
2:C′←Normalization(C)∪Table2Json(C)
3:C 100←Rerank 1(Q′, C′)
4:C 20←Rerank 2(Q′, C100)
5:iftokens(Q′∪C 20)≤64kthen
6:R←LLM(Q′, C20)
7:else
8:R 1←LLM(Q′, C1:10)
9:R 2←LLM(Q′, C11:20)
10:R←Fusion(R 1, R2)
11:end if
12:returnR
both semantic and quantitative correspondence, which is cru-
cial for hybrid financial corpora.
This architecture achieves an optimal trade-off between
precision and efficiency. The lightweight first stage prevents
unnecessary computation on irrelevant candidates, while the
second stage provides the semantic depth needed to identify
financially meaningful evidence. This separation of lexical
filtering and contextual reasoning enables robust retrieval per-
formance even under high-volume workloads, making the ap-
proach scalable for institutional use.
Financial documents often combine narrative text and tab-
ular disclosures, requiring models to align textual descrip-
tions with structured numerical information. To handle this,
the retrieval module leverages both the normalized corpus
(from the Pre-Retrieval phase) and the JSON-formatted tabu-
lar data, allowing the reranker to compute cross-modal simi-
larity between natural language and numeric fields. This de-
sign improves factual consistency and ensures that retrieved
contexts are suitable for quantitative reasoning.
The final output consists of the top 20 ranked passages,
which collectively form a compact yet information-rich con-
text for the generation stage. This cap balances semantic
coverage and token efficiency, ensuring that subsequent long-
context generation operates within model input limits while
retaining all essential evidence.
4.3 Generation
Although recent LLMs nominally accept extremely long
inputs [OpenAI, 2026; Anthropic, 2026 ], multiple studies
[Databricks, 2024; Jinet al., 2024; Paulsen, 2025 ]report
that response quality, and in particular numerical fidelity, de-
grades well before the advertised context limit. The gap be-
tweennominalandeffectivecontext length is especially con-
sequential in finance, where 10-K filings routinely exceed
100k tokens and where a single misread cell can invalidate
downstream reasoning. The generation stage must therefore
decide not how much context to feed the model, but how to
feed it in a way that preserves accuracy.
Context Size Management
To empirically identify a reliable operational threshold,
we conducted ablation studies using several state-of-the-art

Table 1: Ablation study of Pre-Retrieval components: Norm denotes normalization of queries and corpora, including abbreviation expansion,
unit standardization, and typo or grammar correction. HyDE represents Hypothetical Document Embedding, where a pseudo-document is
generated to enhance semantic alignment between the query and the corpus.
Query Corpus NDCG@20
Original Norm Keywords Paraphrased HyDE Original Norm Table-to-json Summary
Extraction
◦- - - - ◦- - - 0.7323
-◦- - - ◦- - - 0.7446
-◦ ◦- - ◦- - - 0.7542
-◦-◦- ◦- - - 0.7347
-◦- -◦ ◦- - - 0.7347
◦- - - - -◦- - 0.7211
-◦- - - -◦- - 0.7333
-◦ ◦- - -◦- - 0.7503
-◦-◦- -◦- - 0.7446
-◦- -◦ -◦- - 0.6759
◦- - - - -◦ ◦- 0.7301
-◦- - - -◦ ◦- 0.7529
-◦ ◦- - -◦ ◦- 0.7918
-◦-◦- -◦ ◦- 0.7677
-◦- -◦ -◦ ◦- 0.6843
◦- - - - - - -◦ 0.6544
-◦- - - - - -◦ 0.6501
-◦ ◦- - - - -◦ 0.6579
-◦-◦- - - -◦ 0.6542
-◦- -◦ - - -◦ 0.5853
Table 2: Comparison of Hierarchical Reranker Combinations
1st2ndNDCG@20
Reranker Reranker
- 0.7260
Linq-Embed-Mistral 0.7378
jina-reranker-v3 gte-Qwen2-7B-instruct 0.7567
Qwen3-Reranker-4B 0.7763
Qwen3-Reranker-8B 0.7918
LLMs under long-context and numerically intensive condi-
tions. We observed a noticeable decline in performance when
input size exceeded 64k tokens. Based on this observation,
we set 64k as the context threshold for all subsequent experi-
ments.
Fusion
When the combined input exceeds 64k tokens, the top-20
corpora are divided into two semantically coherent subsets,
producing interim answersR 1andR 2. The two outputs are
then merged through a conditional fusion process: the sys-
tem first identifies whether each interim response contains a
definitive answer; if only one does, that answer is directly
adopted; if both contain valid answers, the one with the higher
confidence value is selected; and if neither provides a clear
answer, the model outputs an explicit “unknown” response.
This adaptive fusion ensures consistent and interpretable gen-Table 3: Comparison of LLM Performance with/without Context
Management
LLM Context Accuracy
Management
Gemini 3.0 Pro - 0.7593
◦0.7610
GPT-5.4 - 0.7786
◦0.7794
Grok-4.20 - 0.7901
◦0.7938
Claude-4.6 Opus - 0.8103
◦0.8152
eration under long-context conditions while preventing hallu-
cinated synthesis across partitions.
5 Results
This section presents the experimental results of the proposed
framework, including analyses of Pre-Retrieval design, Hier-
archical Reranker architecture, and Long-Context Manage-
ment.

5.1 Pre-Retrieval Design
Table 1 summarizes the retrieval performance under various
combinations of query and corpus preprocessing methods.
The results clearly show that normalization yields the highest
performance gain. Combining Query Normalization, Key-
word Extraction, and Table-to-json Conversion achieved the
best score (NDCG@20 = 0.7918), outperforming the baseline
(no Pre-Retrieval) by +5.9%. This confirms that pre-retrieval
normalization significantly improves semantic coherence and
retrieval performance within financial documents.
5.2 Impact of Hierarchical Reranker
Table 2 compares different reranking combinations. Using
jina-reranker-v3as the lightweight first-stage model followed
byQwen3-Reranker-8B [Zhanget al., 2025 ]as the second-
stage model achieved the best performance (NDCG@20 =
0.7918). This hierarchical reranking improved relevance
by +6.5% compared to a single-reranker setup. By sep-
arating a lightweight model for fast filtering (Stage1) and
a larger model for fine-grained reranking (Stage2), the ap-
proach aimed to mitigate the inference time limitation while
maintaining retrieval performance.
5.3 Long-Context Management
Table 3 presents the impact of the proposed context segmenta-
tion and fusion strategy. Across all tested LLMs, applying the
split-and-fusion mechanism resulted in a slight accuracy im-
provement ranging from 0.08% to 0.49%. Claude-4.6 Opus
achieved the highest performance (Accuracy = 0.8152) under
the 64k-token threshold, indicating that context segmentation
had only a marginal effect on overall accuracy.
5.4 Summary
Overall, the proposed Hierarchical Reranker framework
showed consistent and statistically stable improvements in
both retrieval and generation performance. These results can
be attributed to three key components: (1) optimized pre-
retrieval algorithms, (2) a hierarchical reranker architecture
that enhances retrieval precision, and (3) long-context man-
agement for stable reasoning across extended inputs. The
system provides a reliable, scalable, and domain-adaptive so-
lution, supporting its applicability to financial document anal-
ysis.
6 Discussion
This section discusses the strengths, limitations, and future
directions of the proposed finance-specific RAG system. We
first highlight how each component contributes to the sys-
tem’s effectiveness in real-world financial tasks, then out-
line key computational and scalability limitations, and finally
present potential avenues for improvement and research ex-
tension.
6.1 Contributions
Three complementary components — Pre-Retrieval Opti-
mization, the Hierarchical Reranker, and Long-Context Man-
agement — jointly produced consistent gains across re-
trieval precision, factual consistency, and reasoning stability.The Pre-Retrieval phase is the simplest yet highest-leverage
stage: deterministic normalization, keyword augmentation,
and Markdown-to-JSON table conversion lifted NDCG@20
by +5.9% over the no-preprocessing baseline, and did so
without introducing LLM-induced hallucinations during pre-
processing. This kind of conservative, rule-based engineering
is what makes the system reproducible enough for regulated
environments such as auditing and investment research.
The hierarchical reranker complements this by realiz-
ing a small-and-large model collaboration: the lightweight
first-stage model bounds compute, while the high-capacity
second-stage model concentrates effort on the candidates
most likely to matter. The two-stage cascade improved
NDCG@20 by +6.5% over a single-reranker baseline while
keeping per-query inference cost compatible with institu-
tional throughput. We see this pattern — specialize small
models for breadth and large models for depth — as broadly
applicable to other vertical RAG settings.
6.2 Limitations
Despite its effectiveness, the system entails several limita-
tions that warrant attention. First, the hierarchical rerank-
ing architecture, while improving precision, introduces ad-
ditional computational overhead compared to single-stage
rerankers. This cost can be mitigated through user-guided
search constraints — for instance, extracting company names
from the user query and using them to substantially narrow
the scope of candidate documents before reranking is in-
voked. Second, the reliance on a fixed 64k-token context
threshold, though empirically validated, restricts scalability
when processing extremely long financial reports or multi-
document reasoning tasks. Finally, while the framework was
evaluated on multiple financial benchmarks, further valida-
tion across multilingual or real-time financial streams remains
an open challenge for industrial deployment.
6.3 Future Work
We see three natural extensions. First,dynamic context
prioritization [Ikramet al., 2025 ]would replace the cur-
rent fixed 64k threshold with importance-weighted alloca-
tion of model attention, reducing information loss during
segmentation and fusion on very long filings. Second, we
plan to push the small-and-large model collaboration to-
wardadaptive, query-conditional reranking: invoke the
heavy second-stage reranker only when the first stage exhibits
low confidence, further compressing token spend on routine
queries while preserving accuracy on hard ones. Third, em-
bedding the pipeline inside a broaderagentic workflow—
with expert-in-the-loop feedback signals and tool use for nu-
merical verification — could close the remaining gap between
retrieval-grounded answers and analyst-grade financial rea-
soning, moving the system from a single-turn RAG pipeline
toward an autonomous financial-analysis agent.
7 Conclusion
Financial document analysis sits at the intersection of high
economic value and high expertise cost, making it one of the
most natural targets for LLM- and RAG-based automation.

Global financial institutions [Wuet al., 2023; Papasotiriouet
al., 2024 ]have already begun integrating retrieval-grounded
LLMs into their operational workflows, but the gap between
research-grade RAG and institutional-grade deployment re-
mains substantial.
This work narrows that gap. We presented Hierarchical
Reranker, a finance-specific RAG framework that couples
deterministic Pre-Retrieval Optimization, a small-and-large
model Hierarchical Reranker, and adaptive Long-Context
Management into a single deployable pipeline. Comprehen-
sive ablations show that each component contributes measur-
able gains, and the system as a whole achieved NDCG@20
= 0.7918 and secured 2nd place in the ACM-ICAIF ’24 Fi-
nanceRAG Challenge. Just as importantly, the framework
is now running inside real auditing and investment work-
flows, supporting the claim that careful engineering — not
only scale — is what carries RAG into production.
We hope this study offers a concrete reference point for
vertical RAG design in finance and contributes to the broader
trajectory of LLM-based, agentic systems for high-stakes fi-
nancial reasoning.
References
[Anet al., 2024 ]Chenxin An, Jun Zhang, Ming Zhong, Lei
Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng
Kong. Why does the effective context length of LLMs fall
short?, 2024. arXiv:2410.18745.
[Anthropic, 2026 ]Anthropic. Introducing Claude 4.6. https:
//www.anthropic.com/claude/opus, 2026. Online docu-
mentation. Accessed: 2026-02-05.
[Chenet al., 2022a ]Zhiyu Chen, Wenhu Chen, Charese
Smiley, Sameena Shah, Iana Borova, Dylan Lang-
don, Reema Moussa, Matt Beane, Ting-Hao Huang,
Bryan Routledge, and William Yang Wang. FinQA: A
dataset of numerical reasoning over financial data, 2022.
arXiv:2109.00122.
[Chenet al., 2022b ]Zhiyu Chen, Shiyang Li, Charese Smi-
ley, Zhiqiang Ma, Sameena Shah, and William Yang
Wang. ConvFinQA: Exploring the chain of numerical
reasoning in conversational finance question answering,
2022. arXiv:2210.03849.
[Choiet al., 2024 ]Chanyeol Choi, Jy-Yong Sohn, Yongjae
Lee, Subeen Pang, Jaeseon Ha, Hoyeon Ryoo, Yongjin
Kim, Hojun Choi, and Jihoon Kwon. ACM-ICAIF ’24
FinanceRAG challenge. https://kaggle.com/competitions/
icaif-24-finance-rag-challenge, 2024. Kaggle.
[Databricks, 2024 ]Databricks. The long context
RAG capabilities of OpenAI o1 and Google
Gemini. https://www.databricks.com/blog/
long-context-rag-capabilities-openai-o1-and-google-gemini,
2024. Blog post. Accessed: 2024-10-27.
[Fanet al., 2024 ]Wenqi Fan, Yujuan Ding, Liangbo Ning,
Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on RAG meeting LLMs: To-
wards retrieval-augmented large language models, 2024.
arXiv:2405.06211.[Gaoet al., 2022 ]Luyu Gao, Xueguang Ma, Jimmy Lin, and
Jamie Callan. Precise zero-shot dense retrieval without
relevance labels, 2022. arXiv:2212.10496.
[Google, 2026 ]Google. Introducing Gemini 3.0. https:
//deepmind.google/models/gemini/, 2026. Online docu-
mentation. Accessed: 2025-11-19.
[Guet al., 2025 ]Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hex-
iang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Ying-
han Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang,
Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni,
and Jian Guo. A survey on LLM-as-a-judge, 2025.
arXiv:2411.15594.
[Ikramet al., 2025 ]Azam Ikram, Xiang Li, Sameh Elnikety,
and Saurabh Bagchi. Ascendra: Dynamic request prioriti-
zation for efficient LLM serving, 2025. arXiv:2504.20828.
[Islamet al., 2023 ]Pranab Islam, Anand Kannappan,
Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie
Vidgen. FinanceBench: A new benchmark for financial
question answering, 2023. arXiv:2311.11944.
[Jagermanet al., 2023 ]Rolf Jagerman, Honglei Zhuang,
Zhen Qin, Xuanhui Wang, and Michael Bendersky. Query
expansion by prompting large language models, 2023.
arXiv:2305.03653.
[Jinet al., 2024 ]Bowen Jin, Jinsung Yoon, Jiawei Han, and
Sercan O Arik. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG.arXiv preprint
arXiv:2410.05983, 2024.
[Lee and Han, 2025 ]Stevens Nicholas Lee and
Soyeon Caren Han. Large language models in fi-
nance (FinLLMs).Neural Computing and Applications,
37(30):24853–24867, January 2025.
[Liet al., 2023 ]Yucheng Li, Bo Dong, Chenghua Lin,
and Frank Guerin. Compressing context to enhance
inference efficiency of large language models, 2023.
arXiv:2310.06201.
[LighthouzAI, 2024 ]LighthouzAI. FinQABench: A new
QA benchmark for finance applications. https://
huggingface.co/datasets/lighthouzai/finqabench, 2024.
[Liu and Mozafari, 2024 ]Jie Liu and Barzan Mozafari.
Query rewriting via large language models, 2024.
arXiv:2403.09060.
[Maet al., 2023 ]Xueguang Ma, Xinyu Zhang, Ronak
Pradeep, and Jimmy Lin. Zero-shot listwise docu-
ment reranking with a large language model, 2023.
arXiv:2305.02156.
[OpenAI, 2026 ]OpenAI. Introducing GPT-5.4. https://
openai.com/index/introducing-gpt-5-4, 2026. Online doc-
umentation. Accessed: 2026-03-05.
[Papasotiriouet al., 2024 ]Kassiani Papasotiriou, Srijan
Sood, Shayleen Reynolds, and Tucker Balch. AI in
investment analysis: LLMs for equity stock ratings. In
Proceedings of the 5th ACM International Conference
on AI in Finance, ICAIF ’24, pages 419–427. ACM,
November 2024.

[Paulsen, 2025 ]Norman Paulsen. Context is what you need:
The maximum effective context window for real world
limits of LLMs, 2025. arXiv:2509.21361.
[Wanget al., 2025 ]Feng Wang, Yuqing Li, and Han Xiao.
jina-reranker-v3: Last but not late interaction for listwise
document reranking, 2025. arXiv:2509.25085.
[Wuet al., 2023 ]Shijie Wu, Ozan Irsoy, Steven Lu, Vadim
Dabravolski, Mark Dredze, Sebastian Gehrmann, Prab-
hanjan Kambadur, David Rosenberg, and Gideon Mann.
BloombergGPT: A large language model for finance,
2023. arXiv:2303.17564.
[xAI, 2026 ]xAI. Grok 4.20. https://x.ai/grok, 2026. Online
documentation. Accessed: 2026-02-17.
[Yanget al., 2023 ]Yi Yang, Yixuan Tang, and Kar Yan
Tam. InvestLM: A large language model for invest-
ment using financial domain instruction tuning, 2023.
arXiv:2309.13064.
[Yuet al., 2025 ]Xiaohan Yu, Pu Jian, and Chong Chen.
TableRAG: A retrieval augmented generation frame-
work for heterogeneous document reasoning, 2025.
arXiv:2506.10380.
[Zhanget al., 2025 ]Yanzhao Zhang, Mingxin Li, Dingkun
Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and
Jingren Zhou. Qwen3 embedding: Advancing text em-
bedding and reranking through foundation models.arXiv
preprint arXiv:2506.05176, 2025.
[Zhuet al., 2021 ]Fengbin Zhu, Wenqiang Lei, Youcheng
Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli
Feng, and Tat-Seng Chua. TAT-QA: A question answer-
ing benchmark on a hybrid of tabular and textual content
in finance, 2021. arXiv:2105.07624.
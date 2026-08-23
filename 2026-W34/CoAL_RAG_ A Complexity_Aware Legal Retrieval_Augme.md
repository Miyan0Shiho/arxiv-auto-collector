# CoAL-RAG: A Complexity-Aware Legal Retrieval-Augmented Generation Method

**Authors**: Jin Su, Zhuofeng Zhao, Huanhuan Wang, Hao Chen

**Published**: 2026-08-18 08:58:11

**PDF URL**: [https://arxiv.org/pdf/2608.17536v1](https://arxiv.org/pdf/2608.17536v1)

## Abstract
Legal consultation questions exhibit multi-level complexity. A single retrieval strategy often leads to over-reasoning for simple questions and poor interpretability for complex ones, making it difficult to meet the requirements for both answer quality and efficiency in high-risk scenarios. To address this issue, this paper proposes CoAL-RAG, a complexity-aware legal retrieval-augmented generation method, which constructs a multi-dimensional evaluation mechanism based on ``question essence'' and ``retrieval consistency'' to enable adaptive routing of retrieval strategies. First, the reasoning demand is quantified according to the logical structure of the question. Then, the discrepancy between semantic retrieval and keyword retrieval is utilized to indirectly reflect problem complexity, thereby selecting the most appropriate retrieval strategy and dynamically filtering contextual information. Experimental results demonstrate that the proposed method significantly outperforms baseline models not only on Chinese legal benchmarks (SocialLawQA, LawBench) but also demonstrates strong cross-jurisdictional generalization on English datasets (LexGLUE, CaseHold). Specifically, on Chinese datasets, the BLEU score improves by 42.5\% and ROUGE-L reaches 3.6 times that of knowledge graph-based methods. On English benchmarks, CoAL-RAG maintains highly competitive accuracy, achieving an optimal balance between generation quality, deep logical reasoning, and system efficiency across different legal systems.

## Full Text


<!-- PDF content starts -->

CoAL-RAG: A Complexity-Aware Legal
Retrieval-Augmented Generation Method
Jin Su1,2, Zhuofeng Zhao1,2⋆, Huanhuan Wang1,3, and Hao Chen1,3⋆⋆
1North China University of Technology, Beijing, 100144, China
2Beijing Key Laboratory of Key Technologies for AI+ Domain Applications, Beijing,
100144, China
3Beijing Key Laboratory on Integration and Analysis of Large-scale Stream Data,
Beijing 100144, China
edzhao@ncut.edu.cn
Abstract.Legal consultation questions exhibit multi-level complexity.
A single retrieval strategy often leads to over-reasoning for simple ques-
tions and poor interpretability for complex ones, making it difficult to
meet the requirements for both answer quality and efficiency in high-
risk scenarios. To address this issue, this paper proposes CoAL-RAG,
a complexity-aware legal retrieval-augmented generation method, which
constructs a multi-dimensional evaluation mechanism based on “ques-
tion essence” and “retrieval consistency” to enable adaptive routing of
retrieval strategies. First, the reasoning demand is quantified accord-
ing to the logical structure of the question. Then, the discrepancy be-
tween semantic retrieval and keyword retrieval is utilized to indirectly re-
flectproblemcomplexity,therebyselectingthemostappropriateretrieval
strategy and dynamically filtering contextual information. Experimental
results demonstrate that the proposed method significantly outperforms
baseline models not only on Chinese legal benchmarks (SocialLawQA,
LawBench) but also demonstrates strong cross-jurisdictional generaliza-
tion on English datasets (LexGLUE, CaseHold). Specifically, on Chinese
datasets, the BLEU score improves by 42.5% and ROUGE-L reaches 3.6
times that of knowledge graph-based methods. On English benchmarks,
CoAL-RAGmaintainshighlycompetitiveaccuracy,achievinganoptimal
balance between generation quality, deep logical reasoning, and system
efficiency across different legal systems.
Keywords:Legal Q&A·Retrieval-Augmented Generation·Complex-
ity Awareness·Adaptive Retrieval·Knowledge Graph
1 Introduction
Driven by breakthrough advancements of Large Language Models (LLMs) in
Natural Language Processing (NLP) [4,38], intelligent legal question answering
⋆Corresponding author.
⋆⋆Equal contribution.
arXiv:2608.17536v1  [cs.CL]  18 Aug 2026

2 J. Su et al.
(a) Evidence Gaps in Complex Reasoning Problems
(b) Noise Introduction in Simple Factual Issues
(c) Retrieval conflict
Fig. 1.Limitations of Existing Methods
is transitioning toward a new paradigm of semantic generation. Given the strin-
gent demands for accuracy and traceability in high-stakes legal scenarios [35],
integratingexternalknowledgebasescaneffectivelymitigatemodelhallucination
and knowledge obsolescence [16]. However, the complexity of legal consultation
queries varies significantly. Questions such as “What is the statutory retirement
age?” involve a single legal provision and require minimal reasoning, resulting in
relatively low complexity. In contrast, queries such as “My employer fled without
signing a labor contract after a workplace injury. Which laws have been violated
and how can I protect my rights?” involve dense legal knowledge, strong con-
ditional constraints, and multi-step reasoning, leading to substantially higher
complexity [6].
Among existing retrieval strategies in the legal domain, pure vector-based
retrieval lacks deep relational reasoning, yielding only fragmented legal provi-
sions (Fig. 1a). Conversely, the indiscriminate application of graph reasoning
leads to computational redundancy and noise interference (Fig. 1b). Further-
more, a direct hybrid of the two often results in inconsistent retrieval outcomes
and reasoning conflicts (Fig. 1c). To address these shortcomings, this paper pro-
poses CoAL-RAG (Complexity-Aware Legal RAG). Leveraging LangGraph, this
method constructs a multi-dimensional evaluation mechanism centered on “ques-
tion essence” and “retrieval consistency”. First, it quantifies the internal logical
demands of a query through a five-dimensional metric system. Second, the dif-
ference between semantic and BM25 indirectly reflects complexity. Ultimately,

CoAL-RAG: Complexity-Aware Legal RAG 3
it selects the optimal retrieval strategy based on complexity scores and dynam-
ically filters the context, effectively balancing response speed with deep logical
accuracy. The main contributions of this paper are summarized as follows:
1) A Multi-dimensional Complexity-Aware Mechanism. We propose a mecha-
nism that evaluates query complexity across multiple dimensions by integrating
the internal logic of the query with the external consistency of the retrieval.
Furthermore, we design a “retrieval consistency” algorithm based on a competi-
tion function, providing a criterion for adaptive routing characterized by both
numerical stability and probabilistic interpretability.
2) The CoAL-RAG Method. We introduce the CoAL-RAG approach, which
synergizes complexity awareness, hybrid retrieval, and knowledge graph coordi-
nation. Driven by the multi-dimensional evaluation mechanism, this framework
dynamically tailors retrieval strategies to achieve highly efficient and accurate
generation for legal question answering.
3) Extensive Cross-Jurisdictional Validation and Performance Trade-off. Ex-
periments conducted on both Chinese civil law datasets (SocialLawQA, Law-
Bench) and English common law benchmarks (LexGLUE, CaseHold) demon-
strate that our approach significantly outperforms existing baselines. CoAL-
RAG successfully bridges the reasoning gap across different legal jurisdictions,
enhancingaccuracyforcomplexquerieswhilemaintaininglow-latencyresponses.
2 Related Works
2.1 Legal Large Models
General-purpose LLMs [1,7] frequently suffer from legal hallucinations. Early
domainmodels[2,26,32]optimizedcomprehensionbutlackedgenerativecapabil-
ities. Subsequent instruction-tuned models [5,20,40] and knowledge-augmented
models [13,33] improved intent recognition and reasoning. However, they remain
inadequate for resolving highly complex, multi-step legal consultations.
2.2 Retrieval-Augmented Generation
RAG [19] mitigates knowledge lag via hybrid retrieval [23], re-ranking [9,24],
and dynamic routing based on token confidence (FLARE [17]) or generic com-
plexity classifiers (Adaptive-RAG [14]). Despite their success in open-domain
tasks, applying these methods to legal queries reveals critical limitations. Token-
confidencemetricsfailtocapturerigorousjudicialdeduction,andone-dimensional
complexity classifiers ignore the multifaceted nature of legal queries. Unlike
Adaptive-RAG’s black-boxrouting, CoAL-RAGintroduces a transparent, multi-
dimensionalcomplexityassessmentexplicitlytailoredtothehierarchical“chapter-
and-clause” structureofstatutorytexts,enablingprecisedynamicroutingaligned
with legal reasoning.

4 J. Su et al.
2.3 Knowledge Graph
Knowledge Graphs (KGs) enhance complex reasoning in RAG [15,30] via path-
based explicit chains ( RoG [22], ToG [28]) or subgraph-based structural extrac-
tion [8,12,25]. While effective generally, applying graph structures directly to
statutory tasks introduces two main challenges: (1) indiscriminate retrieval in-
troduces noise and latency [27,29]; and (2) general models like HAKE [36] fail to
capture the hierarchical “chapter-and-clause” structure of legal texts, resulting
in the loss of fine-grained judicial logic [37].
Fig. 2.Overall Framework of CoAL-RAG
3 Method
To address queries with varying levels of complexity, we propose CoAL-RAG.
The framework is built upon a hierarchical legal knowledge graph and incorpo-
rates complexity-aware modeling together with adaptive context-driven routing
to enable dynamic selection of retrieval strategies. The overall system is imple-
mented using LangGraph, as illustrated in Fig. 2.
3.1 Problem Description
To address the diverse complexity of legal queries, this paper proposes CoAL-
RAG, which enables dynamic routing via a complexity-aware mechanism. The
input is formally defined asT={Q, D, G}, whereQdenotes the user’s natural
language legal query,D={d 1, d2, . . . , d n}represents the unstructured legal cor-
pus, andG={E, R}is a hierarchical knowledge graph in the legal domain. The

CoAL-RAG: Complexity-Aware Legal RAG 5
method computes a complexity scoreC final∈[0,1]based on both query charac-
teristicsandretrievalconsistency.Guidedbyathresholdθ,itdynamicallyselects
a retrieval strategyS, constructs the corresponding contextC S, and generates
the answerA= arg max A′P(A′|Q, C S).
3.2 Hierarchical Legal KG Construction
We build a KGG(∼4.2k nodes,∼11.5k edges) across 16 statutes, defining
entities (Law, Chapter, Article, Concept) and relations (Subsumption, Refer-
ence, Conflict). The pipeline involves: 1)Extraction: LLMs extract(e s, r, eo)
triplets from corpusD, retaining metadata. 2)Fusion: LLMs resolve contradic-
tions by merging divergent entities into unified nodes. 3)Clustering: BGE-M3
and GMM-UMAP hierarchically group articles (Articles→Sections→Domains)
to mirror statutory taxonomy. 4)Deployment: Dual-indexed in Milvus and
MySQL, the KG supports version-controlled incremental updates.
3.3 Multi-dimensional Complexity Awareness Mechanism
This mechanism dynamically integrates existing retrieval methods by evaluating
query complexity across multiple dimensions to activate tailored retrieval strate-
gies accordingly. It ensures deep reasoning for complex queries while maximizing
response efficiency for simple ones.
Problem Intrinsic AssessmentSimple legal queries are assigned a low base
complexitythroughpatternmatching,whereC base∈[0.1,0.2],andtheirintrinsic
complexity is defined asC intrinsic =Cbase.
For complex legal queries, semantic features are used to categorize them into
six types: “scenario reasoning”, “conditional judgment”, “multi-condition com-
binations”, “interest protection”, “cross-domain”, and “complex enumeration”. A
base complexityC base∈[0.4,0.6]is predefined for these types, and weight vec-
torsωare assigned according to the “principle of core feature priority” (e.g., for
“scenario reasoning”, the logical dimension is prioritized). After type determina-
tion, the LLM decomposes the query into a set of subqueriesQ suband evaluates
the five complexity dimensions through a structured prompting scheme.4
Reasoning Chain Length (RCL), Knowledge Integration Require-
ment (KIR), and Domain Span (DS)are each measured relative to a single
unit, with complexity exhibiting linear growth as the number of units increases.
The calculation formula is defined in Eq. 1:
Score i= min
1.0,Vi−1
n
.(1)
WhereV irepresents the statistical baseline for each dimension:
4The prompt guides the LLM to extract subqueries, entities, and constraints to quan-
tify the five dimensions (normalized byn= 4).

6 J. Su et al.
–RCL usesV i=|Qsub|, reflecting the logical jumps required for the response;
–KIR usesV i= max(|C|,|Q sub|), whereCis the set of extracted explicit
entities, reflecting the density of legal knowledge integration;
–DS adoptsV i=|Daug|, representing the size of the identified augmented
domain set, reflecting cross-domain integration difficulty.
WhenV i≥n+ 1, the saturation value of1.0is reached, and all are judged as
complex.
The initial values forRelational Reasoning Complexity (RRC)and
Conditional Constraint Density (CCD)are set to0, with scores accu-
mulating incrementally as specific logical structures or constraints appear. The
calculation formula is defined in Eq. 2:
Score i= min
1.0,Vi
n
.(2)
WhereV irepresents the statistical benchmark for each dimension:
–RRC takesV i=|Raug|, the number of unions between explicit logic and
implicit scene relationships, reflecting the degree of logical entanglement;
–CCD usesV i=|Cconst|, representing the number of numerical or temporal
constraints, indicating the precision of boundary determination.
Both RRC and CCD start at0. WhenV i≥n, they reach the saturation value
of1.0and are both judged as complex.
Based on the multidimensional assessment, the five-dimensional weighted
scoreC 5Dis derived from the dimension scoresScore iand their dynamically
assigned weightsω i, as formulated in Eq. 3:
C5D=X
i∈{RCL,KIR,RRC,DS,CCD}ωi·Score i (3)
Combined with the baseline complexityC base, the final intrinsic complexity
is calculated via Eq. 4:
Cintrinsic =α C base+β C 5D,(4)
whereα= 0.3andβ= 0.7. The closerC intrinsicapproaches1.0, the more intrin-
sically complex the legal issue becomes, demanding higher reasoning capability.
Retrieval Consistency AssessmentBased on the hypothesis that “simple
queries exhibit high consensus across different retrieval viewpoints”, an external
feedback mechanism is introduced to detect potential ambiguity and complex-
ity by measuring discrepancies among multiple retrieval pathways. Specifically,
BM25 keyword retrieval and vector-based semantic retrieval are performed to
obtain candidate document setsD BM25andD vec, respectively. The retrieval
consistency complexityC consistency is then computed as follows:

CoAL-RAG: Complexity-Aware Legal RAG 7
Query Simplicity Index (QSI).Using the Top-1 score from BM25 as a proxy
for literal matching between the query and the knowledge base, we compute the
QSI via a sigmoid transformation and treat it as an inverse complexity indicator
(Eq. 5):
QSI =σ 
ScoreBM25
top1
=1
1 + exp 
−0.5 
ScoreBM25
top1−12.5.(5)
Here,12.5is an empirical threshold. A higherQSI(approaching1) indicates
more reliable literal matching and lower retrieval complexity.
Retrieval Divergence Index (RDI).Thismetricquantifiesthedivergencebetween
thekeywordandsemanticretrievalresultsetsbymeasuringtheiroverlap(Eq.6):
RDI = 1.0−(0.7R overlap (DBM25, Dvec) + 0.3R top3(DBM25, Dvec)).(6)
Here,R overlap (·,·)denotes the Jaccard similarity coefficient, andR top3(·,·)de-
notes the top-3 document overlap rate. A higherRDIindicates greater disagree-
ment between semantic understanding and keyword matching.
Consistency Fusion via Competitive Gating.To nonlinearly integrate the above
metrics, we define the “simple evidence energy”E simpleand “complex evidence
energy”E complexas follows (Eq. 7):
Esimple = QSIp(1−RDI)q+ε, E complex = (1−QSI)pRDIq+ε,(7)
We setp= 1.5to apply a non-linear penalty to low-confidence literal matches
(QSI), effectively filtering out weak keyword signals, andq= 0.3to maintain a
smooth response to retrieval divergence (RDI), preventing minor overlaps from
causing routing jitter. The final complexity of the retrieval consistency is com-
puted as the proportion of complex evidence, shown in Eq. 8:
Cconsistency =Ecomplex
Ecomplex +E simple.(8)
Unified Complexity ScoreTo comprehensively evaluate query complexity,
this paper integrates two aspects “problem essence” and “retrieval consistency”.
The final complexity score is formulated in Eq. 9:
Cfinal=γ C intrinsic + (1−γ)C consistency .(9)
We setγ= 0.5to assign equal importance to the query’s linguistic structure
and the retrieval system’s feedback, ensuring a balanced perspective between
internal reasoning demands and external evidence consistency.

8 J. Su et al.
Table 1.Case Study of the Complexity Awareness in CoAL-RAG
Phase Metrics Explanation Final Score
CintrinsicBase Setup(C base = 0.50) Type: Scenario Reasoning,a unified weightω= 0.25.
0.50RCL(0.75) 4 sub-queries (invention/relevance/ownership/time)
KIR(1.00) 8 entities (Zhang San/A Company/PC/patent, etc.)
DS(0.50) 3 domains (Patent / Labor / Civil Code)
RRC(0.75) 3 relations (employment / ownership / infringement)
CCD(1.00) 4 constraints (weekend/company PC/non-core/resigned)
CconsistencyQSI(0.32)Scoretop1
BM25 = 11.0(low literal match degree)0.871RDI(0.93)R top3 = 0.0,R overlap = 0.1(significant divergence between
semantic and BM25)
3.4 Dynamic Retrieval Routing and Adaptive Context Construction
Toaccommodatequeriesofvaryingcomplexity,thispaperintroducesacomplexity-
aware multi-path routing strategy. Based on three predefined thresholds(θ low=
0.25,θmedium = 0.45andθ high= 0.7), the processing pipeline is organized into
four distinct tiers:
WhenC final≤θlow, the query is classified as a simple factual question. Dense
vector retrieval is activated, relying on the large model’s intrinsic reasoning ca-
pabilities to generate answers efficiently.
Whenθ low< Cfinal≤θmedium, the query is considered semantically ambigu-
ous and requiring precise localization. A hybrid retrieval strategy combining
dense and sparse methods is employed, followed by a re-ranking module to re-
fine results and mitigate semantic drift. Iterative processing is also applied to
enhance answer accuracy.
Whenθ medium < Cfinal≤θhigh, the query is treated as moderately complex
and handled via network graph retrieval.
WhenC final≥θhigh, the query is identified as highly complex, graph–text
verification is activated. Using the logical reasoning paths extracted from the
hierarchical knowledge graph as the backbone, the legal provision fragments
retrieved through hybrid retrieval are cross-validated to eliminate conflicting
texts.
After determining the retrieval strategy, adaptive truncation based on score
cliffs is applied to further reduce tail noise. The score decline rate between
adjacent documents is defined as∆ i= (s i−si+1)/si, with a cliff threshold
σ= 0.2. The optimal truncation position is determined as the first index where
the decline exceeds20%, sok= arg min i{∆i> σ}. The resulting context set
Cctx={d 1, d2, . . . , d k}is then incorporated into the prompt template to guide
the final answer generation.
4 Experimental
The domain of social law encompasses high-frequency scenarios such as labor
contracts and work injury identification. Its well-defined structure and hierarchy
make it ideal for evaluating the adaptability of CoAL-RAG. Experiments are
conducted on the self-constructed SocialLawQA dataset and the public Law-
Bench benchmark.

CoAL-RAG: Complexity-Aware Legal RAG 9
Hyperparameter CalibrationHyperparameters (α, β, p, q, γ) and thresholds
(θ) were calibrated via grid search on an expert-annotated, stratified validation
set (N= 120). Sensitivity analysis (Sec. 5.4) shows performance remains stable
within±10%parameter variance, confirming the robustness of our complexity-
aware design.
4.1 Baselines
To evaluate the effectiveness of CoAL-RAG, we compare it with the following
baselines:1)InferencewithoutRetrieval:Directinference,Chain-of-Thought
(CoT) reasoning [31] and LawGPT_zh [40]. 2)Inference with Retrieval:
Retrieval-Augmented Generation (RAG) [19], Hybrid RAG [11], CLERAG, IR-
CoT [29], and Search-o1 [21]. 3)Unified Retrieval-and-Reranking Mod-
els: bge-reranker-v2-m3 and Qwen3-Reranker-4B. 4)Knowledge Graph Aug-
mentation:G-Retriever[12](flatgraph)andLeanRAG[34](hierarchicalgraph).
5)RL Tuning Methods: R1 [10] and Search-R1 [18]. R1 performs reasoning
based on internal knowledge, while Search-R1 interacts with a search engine dur-
ing inference. For fairness, all RL methods use the F1 score as the reward metric
and follow their original training settings. Real-world retrieval is simulated using
Google Web Search via SerpAPI, with ten retrieved documents for each method.
4.2 Evaluation Metrics
WeassessgenerationqualityonChineselegalbenchmarksusingROUGE(1/2/L),
BLEU-4, andBERTScoreto measure token overlap and deep semantic align-
ment. For English cross-jurisdictional benchmarks, we reportAccuracy(for
CaseHold) alongsideMicro-F1andMacro-F1(for LexGLUE) to evaluate
multi-class logical reasoning performance. Finally, system efficiency is measured
via Average Response Time (ART), with detailed latency analysis presented in
Section 5.2.
4.3 Datasets
Chinese Benchmarks (Civil Law):SocialLawQAis a curated dataset of 1.5k
Q&A pairs across 16 statutes, featuring a diverse complexity distribution ideal
for validating adaptive routing in real-world scenarios.LawBench[17] is an
authoritative benchmark from which we selected a 1k Q&A subset focusing on
social law to evaluate core dimensions like memory, comprehension, and appli-
cation.
English Benchmarks (Common Law): To evaluate cross-jurisdictional adapt-
ability,weutilizeLexGLUE[3](specificallysubsetsrequiringlogicaldeduction)
for broad legal NLU assessment, andCaseHold[39], a challenging multiple-
choice dataset rigorously testing long-text reasoning and legal holding identifi-
cation.

10 J. Su et al.
Table 2.Comprehensive Results.R-1/2/L:Rouge-1/2/L;BL: BLEU;BS:BERTScore;
Mi-F: Micro-F1; Ma-F: Macro-F1; Acc: Accuracy.⋆Out-of-domain. ’-’ indicates sys-
tem/language mismatch or excessive migration cost.Boldand underlined are best
results.†denotes statistical significance (p <0.05) over the strongest baseline via
paired t-test.
MethodsChinese Benchmarks (Civil Law) English Benchmarks (Common Law)
LawBench⋆SocialLawQA⋆LexGLUE CaseHold
R-1 R-2 R-L BL BS R-1 R-2 R-L BL BS Mi-F Ma-F Acc
Qwen2.5-3B-Instruct
Direct Inference 0.2523 0.0718 0.1670 0.0254 0.7220 0.2003 0.0426 0.1132 0.0298 0.7351 0.2810 0.2220 0.4954
CoT 0.2680 0.0543 0.1673 0.0312 0.7524 0.3369 0.1266 0.2159 0.0628 0.7765 0.3056 0.2452 0.5126
LawGPT_zh 0.2677 0.0691 0.2046 0.0294 0.7548 0.3461 0.1273 0.2186 0.0638 0.7866 - - -
Standard RAG 0.4022 0.2456 0.3257 0.1739 0.8002 0.3781 0.1855 0.2763 0.1240 0.7972 0.4550 0.4937 0.5211
Hybrid RAG 0.4212 0.2534 0.3339 0.1975 0.8058 0.3799 0.1777 0.2833 0.1059 0.7965 0.6782 0.6150 0.5320
CLERAG 0.4258 0.2561 0.3130 0.1588 0.8140 0.4003 0.2104 0.2692 0.1073 0.8065 0.6617 0.6020 0.6251
IRCoT 0.3244 0.1021 0.2055 0.0532 0.7800 0.3936 0.1978 0.2533 0.1015 0.7843 0.3858 0.3942 0.5102
Search-o1 0.2666 0.0772 0.1748 0.0497 0.7611 0.3345 0.1574 0.2688 0.1014 0.8005 0.3142 0.2683 0.5037
bge-rerank 0.4477 0.3166 0.3679 0.2424 0.8258 0.4299 0.2621 0.3210 0.1355 0.8301 0.5936 0.4339 0.6550
Qwen3-Rerank 0.3031 0.1284 0.2145 0.0913 0.7756 0.3644 0.1634 0.2622 0.0939 0.8010 0.3712 0.5195 0.5383
G-retriever 0.2450 0.0687 0.1753 0.0466 0.7480 0.3220 0.1258 0.2267 0.0650 0.7845 0.2603 0.4047 0.6232
LeanRAG 0.1911 0.0466 0.1137 0.0188 0.7327 0.2315 0.0629 0.1204 0.0237 0.7480 - - -
Search-R10.4953 0.35270.4430 0.27000.8419 0.43200.2738 0.3321 0.15580.8404 0.69250.6585 0.6745
CoAL-RAG (Ours)0.48320.3690†0.41770.2815†0.83420.4427†0.2560 0.33020.1684†0.81840.7186†0.65200.6885†
5 Results
5.1 Generation Quality and Generalization
Table 2 compares CoAL-RAG with baselines (utilizing Qwen2.5-3B-Instruct as
the primary base model unless otherwise specified).
ChineseBenchmarks(CivilLaw):Pureparametricmodelsperformpoorly
due to domain hallucinations. Static pipelines suffer from semantic drift, while
pure KG methods (LeanRAG) introduce noise. In contrast, CoAL-RAG achieves
the highest BLEU scores (0.2815 on LawBench, 0.1684 on SocialLawQA), de-
livering highly competitive accuracy comparable to the compute-heavy Search-
R1, with improvements in key precision metrics being statistically significant
(p <0.05).
EnglishBenchmarks(CommonLaw):EvaluatedonLexGLUEandCase-
Hold, pure parametric models predictably struggle without common-law ground-
ing (e.g., 0.4954 CaseHold Accuracy). Conversely, CoAL-RAG exhibits robust
generalization, outperforming the RL-tuned Search-R1 (0.6885 Accuracy, 0.7186
Micro-F1). Despite a marginal Macro-F1 lag due to our Civil-Law-centric KG
lacking precedent indexing, CoAL-RAG consistently surpasses generic rerankers
and Hybrid RAG without requiring costly reinforcement learning.
5.2 Efficiency Analysis
Benefiting from complexity-aware routing, CoAL-RAG avoids redundant com-
putation for simple queries, achieving average response times of 4.76s and 5.09s
on LawBench and SocialLawQA. It is∼2.2×faster than LawGPT and faster

CoAL-RAG: Complexity-Aware Legal RAG 11
Table 3.Ablation Experiment Results on the LawBench Dataset
Methods Article F1LawConcept
RecallROUGE-L BERTScore
CoAL-RAG(Ours) 0.5308 0.8146 0.4162 0.7706
w/o Intrinsic 0.4977 0.8059 0.4025 0.7688
w/o Consistency 0.5160 0.8071 0.4007 0.7676
w/o Dynamic 0.5278 0.8034 0.4029 0.7653
than complex graph methods like LeanRAG. While adding∼2s of latency com-
pared to Standard RAG, it improves LawBench BLEU by 61.8%, achieving an
optimal trade-off between generation quality and real-time system performance.
5.3 Ablation Study
To validate the core components of CoAL-RAG, we conduct ablation experi-
ments on LawBench using three variants: (1) w/o Intrinsic—removes intrinsic
complexity assessment, relying solely on retrieval consistency for routing; (2)
w/o Consistency—omits retrieval consistency assessment, using only query fea-
tures; and (3) w/o Dynamic—replaces adaptive document selection with a fixed
Top-10 set for generation. To rule out random variance, all scores are averaged
across multiple runs.
Two additional metrics, Article F1 and LawConcept Recall (measuring statu-
tory article retrieval and legal concept coverage), are introduced in the ablation
study. Table 3 presents the ablation results.
The Effectiveness of Intrinsic AssessmentThe results show that removing
intrinsic complexity assessment leads to a decline in Article F1 from 0.5308 to
0.4977 (a relative decrease of 6.24%), underscoring the importance of anticipat-
ing logical depth for identifying complex queries. ROUGE-L also decreases from
0.4162 to 0.4025, indicating that fine-grained perception of question types con-
tributes to the structural coherence and relevance of generated answers. Overall,
intrinsic complexity assessment plays a key role in accurately determining query
complexity and ensuring effective legal provision retrieval.
TheEffectivenessofRetrievalConsistencyRemovingretrievalconsistency
evaluation leads to performance declines across all metrics: Article F1 drops by
1.48 percentage points, ROUGE-L by 1.55 points, and both LawConcept Recall
and BERTScore also decrease. These results confirm that this module effectively
filters retrieval noise and improves routing accuracy through multi-perspective
consistency.
The Effectiveness of Dynamic Top-KThe w/o Dynamic variant, despite
retrieving a fixed top-10 documents, achieves the lowest LawConcept Recall

12 J. Su et al.
Table 4.Representative case analysis of CoAL-RAG under different query complexi-
ties.
QueryC final CoAL-RAG
Can I resign during probation with-
out violating the labor contract?0.31Yes. Under the Labor Contract Law, an employee may ter-
minate the contract during the probation period by giv-
ing advance notice (typically three days) to the employer
(Success. Hybrid Retrieval).
Who owns the patent if software
is developed after work hours using
company equipment?0.79If the invention is not related to the employer’s business
scope and is not part of assigned duties, the patent rights
generally belong to the individual developer rather than
the employer (Success. Graph Reasoning).
Howshouldeligibilityforgovernment
housing benefits be determined?0.26The system lacks local policy data and fails to resolve
priorities among overlapping administrative regulations.
(FailureKnowledge Gap).
(0.8034) and BERTScore (0.7653), confirming that indiscriminately increasing
document volume introduces noise. In contrast, CoAL-RAG dynamically filters
low-relevance documents, improving semantic accuracy while preserving infor-
mation density.
Ablation results confirm the synergistic effect of the three core modules:
multi-dimensional evaluation combined with dynamic filtering enables precise
identification of key evidence, effectively balancing answer quality and response
efficiency.
5.4 Sensitivity Analysis
Sensitivity analysis on a stratifiedN= 120validation set confirms the model’s
robustness: shifting routing thresholdsθ {low,med,high} or the fusion weightγby
±10%causes minimal (<1.5%) fluctuation in ROUGE-L and BERTScore. Gat-
ing exponentsp, qalso exhibit high stability, with performance variance<1.0%
acrosstestedranges(p∈[1.2,1.8], q∈[0.2,0.4]).Furthermore,AverageResponse
Time remains consistent (±0.4s) even under critical threshold variations, prov-
ing that CoAL-RAG’s efficacy stems from its structural complexity-aware logic
rather than heuristic hyperparameter over-tuning.
5.5 Case Study and Error Analysis
Table 4 demonstrates CoAL-RAG’s effectiveness across complexities. Error anal-
ysisrevealstwoprimaryfailuremodes:(1)KnowledgeGaps:Queriesinvolving
local policies (Case 3) outside the statutory KG cause reasoning voids. (2)Pri-
ority Conflicts: The model occasionally struggles to resolve hierarchical logic
among overlapping laws (e.g., General vs. Special laws). This indicates that the
routing mechanism may systematically misclassify edge cases where implicit le-
gal hierarchy is required but not explicitly encoded in the KG. Future work
should focus on integrating multi-tier policy data and enhancing legal hierarchy
awareness.

CoAL-RAG: Complexity-Aware Legal RAG 13
6 Conclusion
ThispaperproposesCoAL-RAG,amulti-dimensionalcomplexity-awareretrieval-
augmentedgenerationmethodtailoredtothevariablecomplexityoflegalqueries.
By jointly assessing reasoning depth across multiple dimensions and incorpo-
rating retrieval consistency, the approach dynamically selects optimal retrieval
strategies and adaptively constructs context. Evaluations across Chinese (Civil
Law) and English (Common Law) benchmarks confirm that this complexity-
aware mechanism is highly generalizable, mitigating the signal-to-noise trade-off
in complex cross-jurisdictional scenarios.
While CoAL-RAG balances quality and efficiency, challenges remain in dy-
namic adaptation. Future work will: (1) extend to specialized domains (e.g.,
criminal law, finance); (2) enhance cross-document reasoning to resolve conflicts
among overlapping provisions; and (3) scale to larger LLMs (e.g., 7B/14B) to
investigate performance ceilings.
References
1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L.,
et al.: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023).
https://doi.org/10.48550/arXiv.2303.08774
2. Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., Androutsopoulos, I.:
LEGAL-BERT: The muppets straight out of law school. In: Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2020. pp. 2898–2904 (2020)
3. Chalkidis, I., Jana, A., Dirschl, D., Pichler, A., Bouchikhi, Y., Vossen, N., Frank,
A., Androutsopoulos, I., Aletras, N.: LexGLUE: A benchmark dataset for legal
language understanding in English. In: Proceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (ACL). pp. 4310–4330 (2022).
https://doi.org/10.18653/v1/2022.acl-long.297
4. Chen, H., Hu, Z., Chai, J., Yang, H., He, H., Wang, X., et al.: ToolForge: A data
synthesis pipeline for multi-hop search without real-world APIs. arXiv preprint
arXiv:2512.16149 (2025). https://doi.org/10.48550/arXiv.2512.16149
5. Cui, J., Li, Z., Yan, Y., Chen, B., Yuan, L.: ChatLaw: Open-source legal
large language model with integrated external knowledge bases. arXiv preprint
arXiv:2306.16092 (2023). https://doi.org/10.48550/arXiv.2306.16092
6. Duan, X., Wang, B., Wang, Z., Ma, W., Cui, Y., Wu, D., et al.: CJRC: A reliable
human-annotated benchmark dataset for Chinese judicial reading comprehension.
In: Chinese Computational Linguistics (CCL). pp. 439–451 (2019)
7. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A.,
et al.: The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
https://doi.org/10.48550/arXiv.2407.21783
8. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph RAG
approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024)
9. Fei, Z., Shen, X., Zhu, D., Zhou, F., Han, Z., Huang, A., et al.: LawBench: Bench-
marking legal knowledge of large language models. In: Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing (EMNLP). pp.
7933–7962 (2024)

14 J. Su et al.
10. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang,
P., Bi, X., et al.: DeepSeek-R1: Incentivizing reasoning capability in LLMs via
reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
11. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.W.: Retrieval augmented lan-
guage model pre-training. In: Proceedings of the 37th International Conference on
Machine Learning (ICML). pp. 3929–3938 (2020)
12. He, X., Tian, Y., Sun, Y., Chawla, N., Laurent, T., LeCun, Y.: G-Retriever:
Retrieval-augmented generation for textual graph understanding and question an-
swering. In: Advances in Neural Information Processing Systems (NeurIPS 37). pp.
132876–132907 (2024)
13. Huang, Q., Tao, M., Zhang, C., An, Z.: Lawyer LLaMA: Enhanc-
ing LLMs with legal knowledge. arXiv preprint arXiv:2305.15062 (2023).
https://doi.org/10.48550/arXiv.2305.15062
14. Jeong, S., Baek, J., Cho, S., Hwang, S.J., Park, J.C.: Adaptive-RAG: Learning
to adapt retrieval-augmented large language models through question complexity.
In: Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics (NAACL). pp. 7350–7380 (2024)
15. Ji, S., Pan, S., Cambria, E., Marttinen, P., Yu, P.S.: A survey on knowledge graphs:
Representation, acquisition, and applications. IEEE Transactions on Neural Net-
works and Learning Systems33(2), 494–514 (2021)
16. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., et al.: Survey of hallucination
in natural language generation. ACM Computing Surveys55(12), 1–38 (2023)
17. Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., et al.: Active re-
trieval augmented generation. In: Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP). pp. 7969–7992 (2023)
18. Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., Han, J.:
Search-R1: Training LLMs to reason and leverage search engines with reinforce-
ment learning. arXiv preprint arXiv:2503.09516 (2025)
19. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al.:
Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances
inNeuralInformationProcessingSystems(NeurIPS).vol.33,pp.9459–9474(2020)
20. Li, H., Ai, Q., Chen, J., et al.: SAILER: Structure-aware pre-trained language
modelforlegalcaseretrieval.In:Proceedingsofthe46thInternationalACMSIGIR
Conference on Research and Development in Information Retrieval (SIGIR). pp.
1035–1044 (2023)
21. Li, X., Dong, G., Jin, J., Zhang, Y., Zhou, Y., Zhu, Y., Zhang, P., Dou,
Z.: Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint
arXiv:2501.05366 (2025)
22. Luo, L., Li, Y., Haffari, G., Pan, S.: Reasoning on graphs: Faithful and inter-
pretable large language model reasoning. In: International Conference on Learning
Representations (ICLR) (2024)
23. M3-Embedding Team: M3-Embedding: Multi-linguality, multi-functionality, multi-
granularity text embeddings through self-knowledge distillation. arXiv preprint
arXiv:2402.03216 (2024)
24. Ma, Y., Cao, Y., Hong, Y., Sun, A.: Large language model is not a good few-shot
information extractor, but a good reranker for hard samples! In: Findings of the
Association for Computational Linguistics: EMNLP 2023. pp. 10572–10601 (2023)
25. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., Wu, X.: Unifying large language
models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and
Data Engineering36(7), 3580–3599 (2024)

CoAL-RAG: Complexity-Aware Legal RAG 15
26. Shao, Y., Mao, J., Liu, Y., Ma, W., Satoh, K., Zhang, M., Ma, S.: BERT-PLI:
Modeling paragraph-level interactions for legal case retrieval. In: Proceedings of
theTwenty-NinthInternationalJointConferenceonArtificialIntelligence(IJCAI).
pp. 3501–3507 (2020)
27. Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E.H.: Large language
models can be easily distracted by irrelevant context. In: Proceedings of the 40th
International Conference on Machine Learning (ICML). pp. 31210–31227 (2023)
28. Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Ni, L.M., Shum, H.Y., Guo,
J.: Think-on-Graph: Deep and responsible reasoning of large language model on
knowledge graph. arXiv preprint arXiv:2307.07697 (2023)
29. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. In:
Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (ACL). pp. 10014–10037 (2023)
30. Wang, X., Yang, Q., Qiu, Y., Liang, J., He, Q., Gu, Z., Xiao, Y., Wang,
W.: KnowledGPT: Enhancing large language models with retrieval and stor-
age access on knowledge bases. arXiv preprint arXiv:2308.11761 (2023).
https://doi.org/10.48550/arXiv.2308.11761
31. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou,
D., et al.: Chain-of-Thought prompting elicits reasoning in large language models.
In: Advances in Neural Information Processing Systems (NeurIPS). vol. 35, pp.
24824–24837 (2022)
32. Xiao, C., Hu, X., Liu, Z., Tu, C., Sun, M.: Lawformer: A pre-trained language
model for Chinese legal long documents. AI Open2, 79–84 (2021)
33. Yue, S., Chen, W., Wang, S., Li, B., Shen, C., Liu, S., et al.: Disc-LawLLM:
Fine-tuning large language models for intelligent legal services. arXiv preprint
arXiv:2309.11325 (2023). https://doi.org/10.48550/arXiv.2309.11325
34. Zhang, Y., Wu, R., Cai, P., Wang, X., Yan, G., Mao, S., Wang, D., Shi, B.: Lean-
RAG: Knowledge-graph-based generation with semantic aggregation and hierar-
chical retrieval. arXiv preprint arXiv:2508.10391 (2025)
35. Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., et al.: Siren’s song in the
AI ocean: A survey on hallucination in large language models. Computational
Linguistics (2025)
36. Zhang, Z., Cai, J., Zhang, Y., Wang, J.: Learning hierarchy-aware knowledge graph
embeddings for link prediction. In: Proceedings of the AAAI Conference on Arti-
ficial Intelligence (AAAI). pp. 3065–3072 (2020)
37. Zhao, Q., Gao, T., Zhou, S., Li, D., Wen, Y.: Legal judgment prediction via het-
erogeneous graphs and knowledge of law articles. Applied Sciences12(5), 2531
(2022)
38. Zhao, W.X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., et al.: A
survey of large language models. arXiv preprint arXiv:2303.18223 (2023).
https://doi.org/10.48550/arXiv.2303.18223
39. Zheng,L.,Guha,N.,Anderson,B.R.,Henderson,P.,Ho,D.E.:Whendoespretrain-
ing help? assessing self-supervised learning for law and the CaseHOLD dataset. In:
Proceedings of the 18th International Conference on Artificial Intelligence and Law
(ICAIL). pp. 159–168 (2021). https://doi.org/10.1145/3462757.3466088
40. Zhou, Z., Shi, J., Song, P., Yang, X., Jin, Y., Guo, L., Li, Y.: LawGPT: A Chinese
legal knowledge-enhanced large language model. arXiv preprint arXiv:2406.04614
(2024). https://doi.org/10.48550/arXiv.2406.04614
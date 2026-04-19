# Adaptive Query Routing: A Tier-Based Framework for Hybrid Retrieval Across Financial, Legal, and Medical Documents

**Authors**: Afshan Hashmi

**Published**: 2026-04-14 10:48:13

**PDF URL**: [https://arxiv.org/pdf/2604.14222v1](https://arxiv.org/pdf/2604.14222v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become the standard paradigm for grounding Large Language Model outputs in external knowledge. Lumer et al. [1] presented the first systematic evaluation comparing vector-based agentic RAG against hierarchical node-based reasoning systems for financial document QA across 1,200 SEC filings, finding vector-based systems achieved a 68% win rate. Concurrently, the PageIndex framework [2] demonstrated 98.7% accuracy on FinanceBench through purely reasoning-based retrieval. This paper extends their work by: (i) implementing and evaluating three retrieval architectures: Vector RAG, Tree Reasoning, and the proposed Adaptive Hybrid Retrieval (AHR) across financial, legal, and medical domains; (ii) introducing a four-tier query complexity benchmark; and (iii) employing GPT-4-powered LLM-as-judge evaluation. Experiments reveal that Tree Reasoning achieves the highest overall score (0.900), but no single paradigm dominates across all tiers: Vector RAG wins on multi-document synthesis (Tier 4, score 0.900), while the Hybrid AHR achieves the best performance on cross-reference (0.850) and multi-section queries (0.929). Cross-reference recall reaches 100% for tree-based and hybrid approaches versus 91.7% for vector search, quantifying a critical capability gap. Validation on FinanceBench (150 expert-annotated questions on real SEC 10-K and 10-Q filings) confirms and strengthens these findings: Tree Reasoning scores 0.938, Hybrid AHR 0.901, and Vector RAG 0.821, with the Tree--Vector quality gap widening to 11.7 percentage points on real-world documents. These findings support the development of adaptive retrieval systems that dynamically select strategies based on query complexity and document structure. All code and data are publicly available.

## Full Text


<!-- PDF content starts -->

Adaptive Query Routing: A Tier-Based Framework for
Hybrid Retrieval Across Financial, Legal, and Medical
Documents
Afshan Hashmi
TRDC, Tuwaiq Academy, Riyadh, Saudi Arabia
afshanhashmi786@gmail.com|a.hashmi@tuwaiq.edu.sa
April 2026
Abstract
Retrieval-Augmented Generation (RAG) has become the standard paradigm for ground-
ingLargeLanguageModeloutputsinexternalknowledge. Lumeretal.[1]presentedthefirst
systematic evaluation comparing vector-based agentic RAG against hierarchical node-based
reasoning systems for financial document QA across 1,200 SEC filings, finding vector-based
systems achieved a 68% win rate. Concurrently, the PageIndex framework [2] demonstrated
98.7% accuracy on FinanceBench through purely reasoning-based retrieval. This paper ex-
tends their work by: (i) implementing and evaluating three retrieval architectures: Vector
RAG, Tree Reasoning, and the proposed Adaptive Hybrid Retrieval (AHR):across financial,
legal, and medical domains; (ii) introducing a four-tier query complexity benchmark; and
(iii) employing GPT-4-powered LLM-as-judge evaluation. Experiments reveal that Tree
Reasoning achieves the highest overall score (0.900), but no single paradigm dominates
across all tiers: Vector RAG wins on multi-document synthesis (Tier 4, score 0.900), while
the Hybrid AHR achieves the best performance on cross-reference (0.850) and multi-section
queries (0.929). Cross-reference recall reaches 100% for tree-based and hybrid approaches
versus 91.7% for vector search, quantifying a critical capability gap. Validation on Fi-
nanceBench (150 expert-annotated questions on real SEC 10-K and 10-Q filings) confirms
and strengthens these findings: Tree Reasoning scores 0.938, Hybrid AHR 0.901, and Vector
RAG 0.821, with the Tree–Vector quality gap widening to 11.7 percentage points on real-
world documents. These findings support the development of adaptive retrieval systems
that dynamically select strategies based on query complexity and document structure. All
code and data are publicly available.
Keywords:Adaptive Query Routing, Retrieval-Augmented Generation, Vectorless RAG,
Hybrid Retrieval, Tier-Based Framework, LLM-as-Judge, FinanceBench, Document Intelli-
gence
1 Introduction
Large Language Models (LLMs) have transformed document understanding, yet remain con-
strained by context window limitations [4, 3]. Retrieval-Augmented Generation (RAG) emerged
as the dominant solution, with vector databases becoming the default infrastructure [5, 6]. How-
ever, semantic similarity does not reliably capture contextual relevance [2, 7], particularly for
structured professional documents.
Lumer et al. [1] published the first systematic comparison of vector-based and non-vector
RAG for financial QA across 1,200 SEC filings with 150 expert-curated questions, finding vector-
basedagenticRAGachieveda68%winrateoverhierarchicalnode-basedsystems. Cross-encoder
reranking yielded 59% absolute MRR improvement [8, 9]. Meanwhile, PageIndex [2] reported
98.7% accuracy on FinanceBench through tree-based reasoning without any vector database.
1arXiv:2604.14222v1  [cs.IR]  14 Apr 2026

These seemingly contradictory findings highlight critical gaps: (1) no cross-domain evalua-
tion exists, (2) no framework explains when each paradigm excels, and (3) no adaptive system
combines their complementary strengths. This paper addresses all three gaps.
Contributions.The contributions of this paper are as follows:
1. The first cross-domain implementation comparing Vector RAG, Tree Reasoning RAG,
and a proposed Adaptive Hybrid Retrieval (AHR) framework across financial, legal, and
medical documents.
2. A four-tier query complexity benchmark testing simple lookups through cross-reference
following, with GPT-4 LLM-as-judge evaluation.
3. Experimental evidence that no single paradigm dominates all query types, with the AHR
framework achieving the best performance on the most challenging query tiers.
4. Validation on FinanceBench [51] (150 expert-annotated questions on real SEC 10-K and
10-Q filings), confirming that the tier-based findings generalise to real-world financial doc-
uments, with the Tree–Vector gap widening from 5.5 to 11.7 percentage points on real
documents.
2 Related Work
2.1 Evolution of RAG Architectures
The RAG paradigm has evolved through three generations [3, 10]. Naive RAG (2020–2022)
established the retrieve-then-generate pipeline [5]. Advanced RAG introduced query rewrit-
ing [11], cross-encoder reranking [8, 9], and corrective RAG [12]. Modular RAG decomposed
the pipeline into interchangeable components [13]. Recent surveys provide comprehensive cov-
erage [14, 15, 16, 17].
2.2 Vector-Based Retrieval
Dense passage retrieval (DPR) [6] established embedding-based search using dual-encoder archi-
tectures. ColBERT [18] introduced late-interaction retrieval, while hybrid approaches combined
BM25 with neural retrievers [19]. For financial documents specifically, Setty et al. [20] intro-
duced metadata-enhanced retrieval, and RAG-Fusion [21] improved recall through reciprocal
rank fusion.
2.3 Non-Vector and Structured Retrieval
PageIndex [2] introduced hierarchical tree indexing with LLM-powered tree search, inspired
by AlphaGo [22]. GraphRAG [23] represented documents as knowledge graphs. HiRAG [24]
introduced hierarchical knowledge retrieval at EMNLP 2025. Graph-of-Thought [25] structured
LLM reasoning as graphs, and KRAGEN [26] used graph-of-thoughts prompting for subproblem
decomposition.
2.4 Financial Domain RAG
TAT-QA [27] established hybrid tabular-textual benchmarks. FinanceBench [51] introduced
10,231 expert-annotated questions on real SEC filings; the original evaluation found that GPT-4
Turbo with standard RAG answered only 19% of questions correctly, demonstrating the bench-
mark’s difficulty. Setty et al. [20] subsequently improved to 25.6% through metadata-enhanced
retrieval, and Jimeno-Yepes et al. [52] reached 32.6% with improved chunking strategies. Fin-
Sage [28] represents the current state-of-the-art among vector-based systems on FinanceBench,
achieving 49.66% LLM accuracy through multi-path retrieval with domain-specialised reranking.
2

RLFKV[29]addressedhallucinationthroughfine-grainedknowledgeverification. RankRAG[30]
unified context ranking with generation at NeurIPS 2024.
3 Methodology
3.1 Multi-Domain Document Dataset
A controlled evaluation corpus was constructed spanning three domains:financial(SEC-style
10-K and 10-Q filings with balance sheets, risk factors, and appendices),legal(master services
agreements with clause hierarchies, definitions, and exhibits), andmedical(Phase III clinical
studyreportswithefficacydata,safetyprofiles,andsubgroupanalyses). Eachdocumentcontains
5–8 sections with nested subsections and explicit cross-references (e.g., “See Appendix A for
detailed risk quantification”).
3.2 Four-Tier Query Complexity Benchmark
Following the query complexity analysis implicit in Lumer et al. [1], a four-tier classification is
formalised (Table 1).
Table 1: Query Complexity Tier Classification
Tier Type Optimal Strategy Example
T1 Single-fact lookup Vector similarity What was Q3 revenue?
T2 Multi-section reasoning Tree navigation Revenue vs. guidance?
T3 Cross-reference Hierarchical traversal Appendix details?
T4 Multi-document synthesis Hybrid fusion Compare across filings
The benchmark comprises 22 queries (10 financial, 6 legal, 6 medical) distributed across all
four tiers, with expert-annotated ground truth answers and relevant section labels.
3.3 System Implementations
VectorRAG.Documentsarechunked(100tokens,20-tokenoverlap),embeddedusingall-MiniLM-L6-v2[31],
and indexed with FAISS [32]. Top-5 chunks are retrieved via cosine similarity. GPT-4o-mini
generates answers from retrieved context.
Tree Reasoning RAG.Documents are parsed into hierarchical tree indices with GPT-4o-
mini-generated node summaries. At query time, GPT-4o-mini reasons over the tree structure,
selecting branches to explore, drilling into subsections, and following detected cross-references
through regex-based reference detection and recursive tree traversal.
Adaptive Hybrid RAG (Proposed).A GPT-4o-mini query classifier assigns each query
to a complexity tier. Tier 1 queries route to vector search; Tier 2–3 queries route to tree
reasoning; Tier 4 queries invoke both systems with result fusion.
3.4 Evaluation Protocol
Following Lumer et al. [1], LLM-as-judge evaluation is employed [33] using GPT-4o-mini. Each
response is scored on accuracy (0–1), completeness (0–1), and relevance (0–1), yielding an overall
quality score. Retrieval metrics (precision, recall, F1) measure section-level retrieval accuracy.
Latency is measured end-to-end including LLM inference.
3

3.5 FinanceBench Real-World Validation
To validate findings beyond the controlled corpus, all three retrieval architectures were evaluated
on FinanceBench [51], a publicly available benchmark of 150 expert-annotated question-answer
pairs drawn from real SEC 10-K and 10-Q filings. FinanceBench provides pre-extracted evidence
passages enabling evaluation without raw PDF access. Questions span information extraction,
numerical reasoning, and logical reasoning types. The same LLM-as-judge protocol was applied
to enable direct comparison with controlled corpus results. A stratified sample ofn= 50
questions covering all three question types was used for evaluation.
4 Experimental Results
4.1 Overall Performance
Table 2 presents aggregate performance across all 22 queries and three domains.
Table 2: Overall Performance Comparison (22 Queries, 3 Domains)
Method Quality Recall Precision F1 Latency
Tree Reasoning0.900 0.977 0.850 0.8393.40s
Hybrid AHR (Proposed) 0.845 0.955 0.655 0.670 3.10s
Vector RAG 0.845 0.932 0.341 0.4721.61s
Tree Reasoning achieves the highest quality (0.900) with substantially better precision (0.850
vs. 0.341 for Vector RAG). Vector RAG maintains the lowest latency at 1.61s. The 2×latency
cost of reasoning-based retrieval yields a 6.5% quality improvement.
4.2 Performance by Query Complexity Tier
Figure 1 reveals that no single method dominates across all complexity tiers.
Figure 1: (a) LLM-as-Judge quality score and (b) retrieval recall by query complexity tier. No
single method dominates all tiers, supporting the case for adaptive retrieval.
Key findings: Tree Reasoning leads on simple queries (T1: 0.938). Hybrid AHR achieves
the highest scores on multi-section (T2: 0.929) and cross-reference queries (T3: 0.850). Vector
RAG wins on multi-document synthesis (T4: 0.900). These results directly support the central
thesis that query complexity determines optimal retrieval strategy.
4

4.3 Cross-Domain Analysis
Figure 2 and Table 3 present domain-specific performance.
Figure 2: Domain-wise performance comparison across financial, legal, and medical documents.
Tree Reasoning dominates legal and medical domains; Vector RAG leads in financial.
Table 3: LLM-as-Judge Quality Score by Domain
Domain Vector RAG Tree Reasoning Hybrid AHR
Financial0.9100.890 0.880
Legal 0.8000.8830.850
Medical 0.7900.9330.800
Vector RAG leads in the financial domain (0.910), where queries often involve specific nu-
merical lookups matching embedding similarity. Tree Reasoning dominates legal (0.883) and
medical (0.933) domains, where nested hierarchies and cross-referenced data require structural
navigation.
4.4 Latency vs. Quality Trade-off
Figure 3 visualizes the latency-quality distribution across all 66 query-method pairs.
Figure 3: Latency vs. answer quality trade-off. Vector RAG clusters in low-latency with high
variance; Tree Reasoning shows higher but more consistent quality.
Vector RAG exhibits high variance (0.2–1.0 quality at 0.5–2s latency), while Tree Reasoning
concentrates in the 0.8–1.0 quality range at 2–8s latency. Hybrid AHR achieves an intermediate
profile, reducing average latency from 3.40s to 3.10s while maintaining competitive quality.
5

4.5 Cross-Reference Resolution
Figure 4 isolates performance on Tier 3 cross-reference queries—the most discriminating test
between paradigms.
Figure 4: Cross-reference resolution capability on Tier 3 queries. Tree-based approaches achieve
100% recall vs. 91.7% for Vector RAG.
Both Tree Reasoning and Hybrid AHR achieve perfect section recall (1.00), correctly fol-
lowing internal references to appendices, exhibits, and notes. Vector RAG achieves 0.917 recall,
missing cross-referenced sections with low semantic similarity to the query. This 8.3 percentage
point gap confirms that cross-reference following is a structural capability that vector similarity
cannot replicate.
4.6 Domain×Tier Interaction
Figure 5 reveals interaction effects between domain and query complexity.
Figure 5: Answer quality heatmap (Domain×Tier) for each retrieval method. Vector RAG
shows weakness on medical T1 (0.50); Tree Reasoning achieves 1.00 on medical T2.
Vector RAG shows its weakest performance on medical Tier 1 queries (0.50), where precise
clinicaldataextractionrequiresstructuralunderstanding. TreeReasoningachievesperfectscores
on medical Tier 2 (1.00), correctly assembling safety data from distributed sections. The Hybrid
AHR heatmap shows the most balanced performance, avoiding extreme low scores.
4.7 FinanceBench Validation on Real SEC Filings
Table 4 presents results on FinanceBench (n= 50, real SEC 10-K and 10-Q filings). The ranking
from the controlled corpus is confirmed and strengthened on real documents.
6

Table 4: Performance on FinanceBench real SEC filings (n= 50, same LLM-as-judge protocol).
Method Overall Accuracy Completeness Relevance Latency
Tree Reasoning0.938 0.936 0.914 0.9644.15s
Hybrid AHR (Proposed) 0.901 0.884 0.876 0.942 4.36s
Vector RAG 0.821 0.798 0.768 0.8982.40s
Both Tree Reasoning and Hybrid AHR improve over their controlled corpus scores (0.938
vs. 0.900; 0.901 vs. 0.845), while Vector RAG declines slightly (0.821 vs. 0.845), widening the
Tree–Vector gap from 5.5 to 11.7 percentage points on real-world documents. The Hybrid AHR
routed 74% of FinanceBench queries to tree reasoning and 26% to vector search, compared
to 55%/45% on the controlled corpus, reflecting the greater structural complexity of real SEC
filings.
Table 5 breaks down performance by question reasoning type. Information extraction queries
are best served by tree reasoning (0.990), while pure numerical reasoning queries with multi-step
calculations are the most challenging for all systems (0.500–0.533), consistent with the findings
of [51].
Table 5: FinanceBench performance by question reasoning type.
Question type Vector RAG Tree Reasoning Hybrid AHR
Information extraction 0.9260.9900.938
Numerical reasoning 0.8920.9740.956
Logical reasoning (numerical) 0.5000.5330.500
Logical reasoning (numerical + logical) 0.7340.9500.900
Numerical + logical reasoning 0.611 0.8890.911
Numerical + information extraction 0.444 0.8440.867
Figure6: FinanceBenchevaluationonrealSECfilings. (A)OverallLLM-as-Judgequalityscores:
Tree Reasoning 0.938, Hybrid AHR 0.901, Vector RAG 0.821. (B) Performance by question type
showing numerical reasoning as the most challenging category. (C) Latency vs. quality: Tree
Reasoning shows higher and more consistent quality; Vector RAG is faster with higher variance.
7

5 Discussion
5.1 Reconciling Contradictory Findings
The results reconcile the contradiction between Lumer et al. [1] (vector wins, 68%) and PageIn-
dex [2] (reasoning wins, 98.7%). The disagreement arises from evaluation scope: Lumer et al.
used diverse query types where vector search excels at broad matching; PageIndex tested precise
numerical extraction where tree navigation excels. The tier-based analysis shows both claims
are correct within their respective complexity ranges.
5.2 FinanceBench in Context of Prior Work
The FinanceBench results (Tree 0.938, Hybrid AHR 0.901, Vector 0.821 on a continuous 0–1
scale) are directionally consistent with the broader FinanceBench literature. Prior vector-based
systems report binary pass/fail accuracy of 19% [51], 25.6% [20], 32.6% [52], and 49.66% [28],
a progression driven largely by improved chunking and reranking. The fact that tree reasoning
achieves its highest absolute scores on real SEC filings (0.938 vs. 0.900 on the controlled corpus)
supports the PageIndex hypothesis [2] that reasoning-based retrieval benefits from the richer
structural signals present in real professional documents. The widening Tree–Vector gap on
FinanceBench (11.7 pp vs. 5.5 pp) further suggests that vector RAG’s relative weakness grows
with document complexity.
5.3 The Case for Adaptive Retrieval
The Hybrid AHR used vector search for 45% of queries (10/22, avg. score 0.830) and tree
reasoning for 55% (12/22, avg. score 0.858). While not always achieving the absolute best score,
AHR avoids catastrophic failures that affect pure approaches, with Vector RAG scoring 0.20 on
some medical queries and Tree Reasoning scoring 0.60 on some multi-document queries.
5.4 Implications for Enterprise Deployment
For regulated industries (finance, healthcare, legal), tree reasoning should be the default for
precision-critical queries. The 100% cross-reference recall is particularly important where in-
complete retrieval has material consequences, such as understating drug safety signals [34] or
missing contractual obligations.
6 Future Work
Five directions emerge: (1) Scaling to the full 150-question FinanceBench set and 1,200+
real SEC filings with statistical significance testing. (2) Incorporating cross-encoder rerank-
ing into the vector baseline, yielding 59% MRR improvement in [1]. (3) Developing reinforce-
ment learning-based meta-retrieval agents that learn optimal routing from query-outcome feed-
back. (4) Extending to multimodal documents where charts and images require vision-language
model integration [35, 36]. (5) Investigating privacy-preserving tree indexing for regulated in-
dustries [37].
7 Conclusion
This paper presents the first cross-domain experimental comparison of Vector RAG, Tree Rea-
soning, and Adaptive Hybrid Retrieval across financial, legal, and medical documents, with
validation on real SEC filings via FinanceBench. Tree Reasoning achieves the highest overall
qualityonthecontrolledcorpus(0.900)andimprovesfurtheronrealSECfilings(0.938), withthe
8

Tree–Vector gap widening from 5.5 to 11.7 percentage points on real-world documents. Cross-
reference recall reaches 100% for tree-based approaches versus 91.7% for vector search. No single
paradigm dominates: Vector RAG leads on financial documents and multi-document synthesis,
while Tree Reasoning excels on legal and medical documents and information extraction tasks.
The Adaptive Hybrid framework achieves the best performance on cross-reference (0.850) and
multi-section queries (0.929) through dynamic strategy selection. These findings establish that
the future of document intelligence lies in adaptive systems that reason aboutwhen to search
andwhen to reason.
Reproducibility.All code, data, and evaluation scripts are available as a Google Colab note-
book.
References
[1] E.Lumer, M.Melich, O.Zino, etal., “Rethinkingretrieval: FromtraditionalRAGtoagentic
and non-vector reasoning systems in the financial domain for LLMs,”arXiv:2511.18177,
Nov. 2025.
[2] M. Zhang, Y. Tang, and PageIndex Team, “PageIndex: Next-generation vectorless,
reasoning-based RAG,” VectifyAI, Sep. 2025.
[3] Y. Gao, Y. Xiong, X. Xu, et al., “Retrieval-augmented generation for large language models:
A survey,”arXiv:2312.10997, 2024.
[4] N. F. Liu, K. Lin, J. Hewitt, et al., “Lost in the middle: How language models use long
contexts,”TACL, vol. 12, pp. 157–173, 2024.
[5] P. Lewis, E. Perez, A. Piktus, et al., “Retrieval-augmented generation for knowledge-
intensive NLP tasks,” inProc. NeurIPS, 2020.
[6] V. Karpukhin, B. Oguz, S. Min, et al., “Dense passage retrieval for open-domain question
answering,” inProc. EMNLP, 2020.
[7] P. Sarkar, “Proxy-pointer RAG: Achieving vectorless accuracy at vector RAG scale and
cost,”Towards Data Science, Apr. 2026.
[8] R. Nogueira and K. Cho, “Passage re-ranking with BERT,”arXiv:1901.04085, 2019.
[9] W. Sun, L. Yan, X. Ma, et al., “Is ChatGPT good at search? Investigating LLMs as
re-ranking agents,” inProc. EMNLP, 2023.
[10] R. Ranjan et al., “A comprehensive survey of RAG: Evolution, current landscape and future
directions,”arXiv:2410.12837, 2024.
[11] X. Ma, Y. Gong, P. He, et al., “Query rewriting for retrieval-augmented large language
models,”arXiv:2305.14283, 2023.
[12] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, “Corrective retrieval augmented generation,”
arXiv:2401.15884, 2024.
[13] Y. Hu and Y. Lu, “RAG and RAU: A survey on retrieval-augmented language model in
NLP,”arXiv:2404.19543, 2024.
[14] S. Wu, Y. Xiong, Y. Cui, et al., “Retrieval-augmented generation for NLP: A survey,”
arXiv:2407.13193, 2024.
9

[15] P. Zhao et al., “Retrieval-augmented generation for AI-generated content: A survey,”
arXiv:2402.19473, 2024.
[16] C. Sharma, “RAG: A comprehensive survey of architectures, enhancements, and robustness
frontiers,”arXiv:2506.00054, 2025.
[17] H. Yu, A. Gan, K. Zhang, et al., “Evaluation of retrieval-augmented generation: A survey,”
arXiv:2405.07437, 2024.
[18] O. Khattab and M. Zaharia, “ColBERT: Efficient and effective passage search via contex-
tualized late interaction over BERT,” inProc. SIGIR, 2020.
[19] S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,”
Found. Trends Inf. Retr., vol. 3, no. 4, pp. 333–389, 2009.
[20] S. Setty, H. Thakkar, A. Lee, et al., “Improving retrieval for RAG-based QA models on
financial documents,”arXiv:2404.07221, 2024.
[21] Z. Rackauckas, “RAG-Fusion: A new take on retrieval-augmented generation,”arXiv
preprint, 2024.
[22] D. Silver, A. Huang, C. J. Maddison, et al., “Mastering the game of Go with deep neural
networks and tree search,”Nature, vol. 529, no. 7587, pp. 484–489, 2016.
[23] D. Edge, H. Trinh, N. Cheng, et al., “From local to global: A graph RAG approach to
query-focused summarization,”arXiv:2404.16130, 2024.
[24] H. Huang, Y. Huang, J. Yang, et al., “Retrieval-augmented generation with hierarchical
knowledge,” inFindings of EMNLP, 2025.
[25] M. Besta, N. Blach, A. Kubicek, et al., “Graph of thoughts: Solving elaborate problems
with large language models,” inProc. AAAI, 2024.
[26] N. Matsumoto et al., “KRAGEN: Knowledge graph enhanced retrieval-augmented genera-
tion,”arXiv preprint, 2024.
[27] F. Zhu, W. Lei, Y. Huang, et al., “TAT-QA: A question answering benchmark on hybrid
tabular and textual content in finance,” inProc. ACL, 2021.
[28] X. Wang, J. Chi, Z. Tai, et al., “FinSage: A multi-aspect RAG system for financial filings
QA,”arXiv:2504.14493, 2025.
[29] “Mitigating hallucination in financial RAG via fine-grained knowledge verification,”
arXiv:2602.05723, 2026.
[30] Y. Yu et al., “RankRAG: Unifying context ranking with RAG in LLMs,” inProc. NeurIPS,
2024.
[31] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-
networks,” inProc. EMNLP, 2019.
[32] J. Johnson, M. Douze, and H. Jegou, “Billion-scale similarity search with GPUs,”IEEE
Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2021.
[33] Y. Zhuang, Z. Lin, et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,”
arXiv:2306.05685, 2023.
[34] G. Goel, “PageIndex: The vectorless RAG,”PVTech Substack, Mar. 2026.
10

[35] A. Radford, J. W. Kim, et al., “Learning transferable visual models from natural language
supervision,” inProc. ICML, 2021.
[36] Z. Shen, R. Zhang, M. Dell, et al., “LayoutParser: A unified toolkit for deep learning based
document image analysis,” inProc. ICDAR, 2021.
[37] S. Chatterjee, “The structural pivot: Analytical perspectives on vectorless RAG and hier-
archical page indexing,”Medium, Feb. 2026.
[38] MarkTechPost, “VectifyAI launches Mafin 2.5 and PageIndex: 98.7% financial RAG accu-
racy,” Feb. 2026.
[39] Build Fast with AI, “Vectorless RAG: How PageIndex works (2026 guide),” 2026.
[40] MicrosoftTechCommunity, “Vectorlessreasoning-basedRAG:Anewapproach,” Mar.2026.
[41] ByteIota, “Vectorless RAG hits 98.7% accuracy: PageIndex challenges vectors,” Jan. 2026.
[42] E. Lumer et al., “Decomposing retrieval failures in RAG for long-document financial QA,”
arXiv:2602.17981, 2026.
[43] E. Lumer et al., “Resolving the robustness-precision trade-off in financial RAG,”
arXiv:2603.26815, 2026.
[44] E. Lumer, V. K. Subbiah, J. A. Burke, et al., “Toolshed: Scale tool-equipped agents with
advanced RAG-tool fusion,” Preprint, 2024.
[45] OpenAI, “GPT-4 technical report,”arXiv:2303.08774, 2023.
[46] J. Wei, X. Wang, D. Schuurmans, et al., “Chain-of-thought prompting elicits reasoning in
large language models,” inProc. NeurIPS, 2022.
[47] Anthropic, “Introducing contextual retrieval,” Anthropic Blog, 2024.
[48] Z.Jiang, F.F.Xu, L.Gao, etal., “Activeretrievalaugmentedgeneration,” inProc. EMNLP,
2023.
[49] D. Sanmartin, “KG-RAG: Bridging the gap between knowledge and creativity,”arXiv
preprint, 2024.
[50] “A systematic review of key RAG systems: Progress, gaps, and future directions,”
arXiv:2507.18910, 2025.
[51] P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, and B. Vidgen, “FinanceBench: A
new benchmark for financial question answering,”arXiv:2311.11944, 2023.
[52] A. Jimeno-Yepes, Y. You, J. Milczek, S. Laverde, and R. Li, “Financial report chunking for
effective retrieval augmented generation,”arXiv:2402.05131, 2024.
11
# Page-Aware Retrieval-Augmented Generation for EvalLLM 2026: A Five-Variant Study on French PDFs

**Authors**: Abdelhak kelious

**Published**: 2026-09-28 09:51:52

**PDF URL**: [https://arxiv.org/pdf/2609.34776v1](https://arxiv.org/pdf/2609.34776v1)

## Abstract
We study retrieval-augmented generation (RAG) for questions about French PDF documents when both the answer and its supporting document pages are evaluated. Five system variants add dense retrieval, rank fusion, reranking, and query decomposition to a BM25 baseline. On 595 challenge questions, the complete system scores 0.4450 MRR@10 and 0.4013 Recall@10, compared with 0.3430 and 0.2994 for BM25. Dense retrieval alone and a simple lexical--dense fusion both underperform BM25. Reranking improves the hybrid system, whereas adding query decomposition produces the largest further gain, with higher latency and more detected output artifacts. The complete system slightly exceeds the reported anonymous overall mean on two answer metrics but falls below it on most page-retrieval metrics. These results identify accurate page selection, rather than semantic retrieval in isolation, as the main opportunity for improvement in this setting.

## Full Text


<!-- PDF content starts -->

Page-Aware Retrieval-Augmented Generation for EvalLLM 2026:
A Five-Variant Study on French PDFs
Abdelhak Kelious
Capsens, France
abdelhak.kelious@capsens.eu
Abstract
We study retrieval-augmented generation (RAG) for questions about French PDF documents when
both the answer and its supporting document pages are evaluated. Five system variants add dense
retrieval, rank fusion, reranking, and query decomposition to a BM25 baseline. On 595 challenge
questions, the complete system scores 0.4450 MRR@10 and 0.4013 Recall@10, compared with 0.3430
and 0.2994 for BM25. Dense retrieval alone and a simple lexical–dense fusion both underperform BM25.
Reranking improves the hybrid system, whereas adding query decomposition produces the largest further
gain, with higher latency and more detected output artifacts. The complete system slightly exceeds the
reported anonymous overall mean on two answer metrics but falls below it on most page-retrieval metrics.
These results identify accurate page selection, rather than semantic retrieval in isolation, as the main
opportunity for improvement in this setting.
Keywords:retrieval-augmented generation; page-level retrieval; French PDFs; BM25; reranking; query
decomposition.
1 Introduction
Retrieval-augmented generation (RAG) conditions answers on external evidence [ 8]. For PDF question
answering, evidence is often evaluated at the level of individual pages: a fluent answer is insufficient when
its cited page is wrong. PDF extraction noise, long documents, and names or dates that require exact
matching make this a demanding retrieval task. The EvalLLM 2026 setting requires an answer accompanied
by document–page references, connecting answer quality to evidence attribution [10, 3].
We examine a simple progression of five systems. BM25 provides a lexical baseline [ 11]; dense retrieval
adds semantic matching [ 4,13]; reciprocal rank fusion (RRF) combines both rankings [ 2]; a second-stage
reranker improves candidate ordering [ 9]; and query decomposition extends retrieval to multiple aspects of
a question [ 12]. The experiment asks whether each addition improves page-level evidence and generated
answers, and at what runtime cost. It is a system ablation, not a claim of a new retrieval algorithm.
2 System
2.1 Page-preserving indexing
We extract PDF text page by page with PyMuPDF. Each page has a document name, page number,
and extracted text. Long pages are divided into 220-word chunks with 40 words of overlap; every chunk
retains its parent page identifier. BM25 indexes the chunks, while the dense variants embed them with
intfloat/multilingual-e5-large[13]. Figure 1 shows the shared processing stages.
2.2 Retrieval variants and answer generation
Table 1 defines the runs. For hybrid retrieval, the BM25 and dense rankings are merged with RRF. For a
candidate page or chunkd, its fused score is
sRRF(d) =X
r∈{BM25,dense}1
60 + rank r(d),(1)
1
arXiv:2609.34776v1  [cs.AI]  28 Sep 2026

PDF pages
document + page IDPage chunks
220 words; overlap 40Retrieval
BM25 / dense / RRFRefinement
rerank / decomposeAnswer + pages
up to 8 candidate pages
A: BM25    B: dense    C: hybrid    D: + reranking    E: + query decompositionFigure 1: Page-preserving pipeline. Only the retrieval components differ across variants; answer generation
selects supporting pages from the retrieved candidates.
Run Retrieval RRF Rerank Decompose
A BM25 – – –
B Dense – – –
C BM25 + dense Yes – –
D BM25 + dense Yes Yes –
E BM25 + dense Yes Yes Yes
Table 1: Five component variants (corresponding to runs 1–5 in the supplied results).
where an absent candidate contributes zero. BM25 and dense retrieval each take up to 60 candidates; the
reranking variants reorder 25 candidates using BAAI/bge-reranker-v2-m3 . Variant E also decomposes the
question into short subqueries before retrieval. This description follows the reported setup; implementation
details such as subquery count and the precise aggregation of chunk scores into page scores are unavailable
in the supplied manuscript.
The generator receives at most eight candidate pages and is instructed to cite only pages needed to support
its answer. The reported generation setting is gpt-5.4-mini at temperature 0.1. The original manuscript
describes a shared answer prompt, generator, and context budget across variants, but does not provide
the prompt text or model snapshot. Consequently, an exact reproduction of the runs would require those
additional materials. Selective page citation also couples the retrieval metrics to the final citation policy:
an expected page omitted at this stage cannot receive credit even if it appeared earlier among retrieved
candidates.
3 Evaluation
We evaluate the five runs on the 595 questions in the supplied challenge test file. Each output contains an
answer and a variable-length list of cited document pages. The official report gives seven metrics. MRR@10,
Recall@10, Top1, and nDCG@10 primarily measure page retrieval and ranking; LLMaaJ, QBERT, and
QPARA Albert concern answer quality. Their exact scoring implementations are not specified in the supplied
manuscript, so we retain the official metric names and reported scores without redefining them.
4 Results and analysis
4.1 Page retrieval and answer quality
Variant E is best among our runs on all seven reported metrics (Table 2). Compared with A, its MRR@10
rises from 0.3430 to 0.4450 (+29.7%), Recall@10 from 0.2994 to 0.4013 (+34.0%), and nDCG@10 from
0.2916 to 0.3801 (+30.3%). Its QBERT increase is smaller (+2.3%). Figure 2 displays the trajectory of three
representative metrics.
The additions are not uniformly beneficial. B performs substantially worse than A across all metrics, and
C still trails A: for example, C reaches 0.2168 MRR@10 versus 0.3430 for BM25. A plausible explanation
is that semantic similarity retrieves topically related but incorrect pages; the available aggregate results
cannot establish that mechanism. D improves on C in all four page-retrieval metrics, although QBERT
2

Run LLMaaJ MRR Recall Top1 nDCG QBERT QPARA
@10 @10 @10 Albert
A: BM25 .4820 .3430 .2994 .2913 .2916 .6935 .7087
B: dense .2383 .0874 .0703 .0777 .0716 .6444 .5534
C: hybrid .4204 .2168 .2201 .1650 .1966 .6826 .6796
D: reranked .4262 .2573 .2431 .2039 .2265 .6708 .7282
E: decomposed .5940 .4450 .4013 .3592 .3801 .7093 .7961
Overall mean .5910 .4582 .4627 .3789 .4072 .6974 .8311
Overall best .8416 .7364 .7146 .6408 .6570 .7689 .9320
Table 2: Reported challenge scores. Bold identifies the best of our five runs in each column. The last two
rows are anonymous challenge-wide summary statistics, not additional runs.
A
BM25B
DenseC
HybridD
+ RerankingE
+ Decomposition0.00.10.20.30.40.50.6Challenge score
0.594
0.445
0.380LLMaaJ MRR@10 nDCG@10
Figure 2: Reported scores for five variants. Lines connect ablation stages for readability and do not imply a
continuous parameter sweep.
falls from 0.6826 to 0.6708. E produces the largest step over D: +72.9% MRR@10, +65.1% Recall@10, and
+67.8% nDCG@10. Without question-level results, these gains cannot specifically be attributed to long or
multi-aspect questions.
Relative to the anonymous overall mean, E is slightly higher in LLMaaJ (0.5940 versus 0.5910) and
QBERT (0.7093 versus 0.6974), but lower in MRR@10, Recall@10, Top1, nDCG@10, and QPARA Albert.
Relative to the reported best system, its gaps are 0.2914 MRR@10, 0.3133 Recall@10, and 0.2769 nDCG@10.
Page ranking and citation selection therefore remain substantial sources of error.
4.2 Output checks
The source manuscript reports a simple automatic screen for explicit non-answers, very short answers,
unexpected writing systems, and conversational residue (Table 3). These are counts from heuristic checks,
not manually validated error rates; categories may overlap. E has the fewest non-answers (66 versus 88 for
A) and very short answers (4 versus 9), but the most unexpected-script instances (21) and conversational
residues (92). Better coverage thus coexists with output-control problems.
4.3 Runtime and a partial energy estimate
Variant E takes 101.1 minutes versus 70.3 minutes for A, an increase of approximately 43.8% (Table 4). For
context, the source manuscript estimates client energy as E=Ptusing a constant client power of 15 W,
and associated emissions using 50 g CO 2e/kWh. These assumptions do not cover remote model inference,
embedding computation, or any server-side reranking; the emissions are therefore a partial client-side proxy,
not a full carbon footprint. D adds only 0.8 minutes over C in these measured runs, whereas E adds 22.8
minutes over D. No variance or repeated-run measurements were supplied.
3

Run Non-answers Very short Script intrusions Residues
A 88 9 11 82
B 156 30 12 54
C 111 14 14 82
D 100 12 11 89
E 66 4 21 92
Table 3: Reported counts from automated output checks over 595 questions per run.
Run Runtime (min) Client energy (kWh) Client emissions (g CO 2e)
A 70.3 .01759 .879
B 65.2 .01630 .815
C 77.5 .01937 .968
D 78.3 .01957 .978
E 101.1 .02526 1.263
Table 4: Reported runtime and partial client-side estimates under the stated constant-power assumptions.
5 Discussion and limitations
The experiment supports a narrow but useful conclusion: reranking improves this hybrid configuration on
page-ranking metrics, and the full configuration performs best among the five runs. It does not support a
general claim that hybrid retrieval beats lexical retrieval, because the unfitted fusion C underperforms A.
Fusion weights, page aggregation, and final citation selection merit separate analysis [ 1]. Evaluating the
candidate retrieval list before citation selection would help locate the bottleneck.
The results come from one challenge collection with no reported confidence intervals, per-question analysis,
or statistical significance test. The supplied source lacks the exact prompt, decomposition procedure, and
full scoring definitions. Further work should release those details; stratify questions by evidence count and
type; tune fusion on held-out data; and assess citations and output artifacts manually. Such experiments
would test whether the gains from decomposition reflect multi-hop evidence gathering or another change in
the retrieval distribution.
Future work could also test alternative RAG approaches, such as the graph-based retrieval used in
CODENS [7], and explore prompting strategies and how LLMs assess task-specific outputs[6, 5].
6 Conclusion
On this page-level French PDF task, BM25 is a strong baseline and dense retrieval alone is weak. Simple
fusion degrades the baseline, but reranking and query decomposition recover and then improve performance.
The complete system reaches 0.4450 MRR@10 and 0.4013 Recall@10 while taking longer and producing more
of some detected generation artifacts. Future improvements should target page-specific ranking, citation
control, and transparent accounting of inference cost.
References
[1]Sebastian Bruch, Siyu Gai, and Avi Ingber. An analysis of fusion functions for hybrid retrieval.ACM
Transactions on Information Systems, 2023.
[2]Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InProceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 758–759, 2009.
[3]Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate
text with citations, 2023. arXiv:2305.14627.
[4]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural Language Processing, 2020.
4

[5]Abdelhak Kelious, Mathieu Constant, and Christophe Coeur. Complex word identification: A com-
parative study between ChatGPT and a dedicated model for this task. InProceedings of the 2024
Joint International Conference on Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024), pages 3645–3653. ELRA and ICCL, 2024.
[6]Abdelhak Kelious, Mathieu Constant, and Christophe Coeur. How do large language models evaluate
lexical complexity? InProceedings of the 14th Joint Conference on Lexical and Computational Semantics
(*SEM 2025), pages 348–361. Association for Computational Linguistics, 2025.
[7]Abdelhak Kelious, Chyrine Tahri, and Eliot Bardet. Codens: Transforming code changes into living,
accessible, and queryable documentation. InProceedings of the 2026 ACM Symposium on Document
Engineering, pages 1–4, 2026.
[8]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive nlp tasks. InAdvances in Neural Information
Processing Systems, 2020.
[9] Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert, 2019. arXiv:1901.04085.
[10]Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktaschel, and
Sebastian Riedel. Kilt: a benchmark for knowledge intensive language tasks, 2020. arXiv:2009.02252.
[11]Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond.
Foundations and Trends in Information Retrieval, 3(4):333–389, 2009.
[12]Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions, 2022. arXiv:2212.10509.
[13]Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multilingual
e5 text embeddings: A technical report, 2024. arXiv:2402.05672.
5
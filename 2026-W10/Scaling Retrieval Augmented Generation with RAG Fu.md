# Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment

**Authors**: Luigi Medrano, Arush Verma, Mukul Chhabra

**Published**: 2026-03-02 18:15:09

**PDF URL**: [https://arxiv.org/pdf/2603.02153v1](https://arxiv.org/pdf/2603.02153v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems commonly adopt retrieval fusion techniques such as multi-query retrieval and reciprocal rank fusion (RRF) to increase document recall, under the assumption that higher recall leads to better answer quality. While these methods show consistent gains in isolated retrieval benchmarks, their effectiveness under realistic production constraints remains underexplored. In this work, we evaluate retrieval fusion in a production-style RAG pipeline operating over an enterprise knowledge base, with fixed retrieval depth, re-ranking budgets, and latency constraints.
  Across multiple fusion configurations, we find that retrieval fusion does increase raw recall, but these gains are largely neutralized after re-ranking and truncation. In our setting, fusion variants fail to outperform single-query baselines on KB-level Top-$k$ accuracy, with Hit@10 decreasing from $0.51$ to $0.48$ in several configurations. Moreover, fusion introduces additional latency overhead due to query rewriting and larger candidate sets, without corresponding improvements in downstream effectiveness.
  Our analysis suggests that recall-oriented fusion techniques exhibit diminishing returns once realistic re-ranking limits and context budgets are applied. We conclude that retrieval-level improvements do not reliably translate into end-to-end gains in production RAG systems, and argue for evaluation frameworks that jointly consider retrieval quality, system efficiency, and downstream impact.

## Full Text


<!-- PDF content starts -->

Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from
an Industry Deployment
Luigi Medrano
Dell Technologies
Austin, TX, USA
Luigi.Medrano@dell.comArush Verma
Dell Technologies
Austin, TX, USA
Arush.Verma@dell.comMukul Chhabra
Dell Technologies
Austin, TX, USA
mukul.chhabra@dell.com
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems commonly adopt retrieval fusion tech-
niques such as multi-query retrieval and recip-
rocal rank fusion (RRF) to increase document
recall, under the assumption that higher recall
leads to better answer quality. While these
methods show consistent gains in isolated re-
trieval benchmarks, their effectiveness under
realistic production constraints remains under-
explored. In this work, we evaluate retrieval
fusion in a production-style RAG pipeline op-
erating over an enterprise knowledge base,
with fixed retrieval depth, re-ranking budgets,
and latency constraints.
Across multiple fusion configurations, we find
that retrieval fusion does increase raw re-
call, but these gains are largely neutralized
after re-ranking and truncation. In our set-
ting, fusion variants fail to outperform single-
query baselines on KB-level Top-kaccu-
racy, with Hit@10 decreasing from0.51to
0.48in several configurations. Moreover, fu-
sion introduces additional latency overhead
due to query rewriting and larger candidate
sets, without corresponding improvements in
downstream effectiveness.
Our analysis suggests that recall-oriented fu-
sion techniques exhibit diminishing returns
once realistic re-ranking limits and context
budgets are applied. We conclude that
retrieval-level improvements do not reliably
translate into end-to-end gains in production
RAG systems, and argue for evaluation frame-
works that jointly consider retrieval quality,
system efficiency, and downstream impact.
1 Introduction
Retrieval-Augmented Generation (RAG) systems
have emerged as a widely adopted approach for
grounding large language models (LLMs) in ex-
ternal knowledge bases (Lewis et al., 2020). By
conditioning generation on retrieved documents,RAG systems aim to improve factual accuracy
and reduce hallucinations. However, their effec-
tiveness remains highly dependent on the quality
of retrieval: when relevant evidence is missing
or poorly ranked, downstream generation is con-
strained regardless of model capacity.
A common strategy for improving retrieval ro-
bustness is the use of retrieval fusion techniques,
such as multi-query retrieval and reciprocal rank
fusion (RRF) (Cormack et al., 2009). By issu-
ing multiple reformulations of a query or combin-
ing results from different retrieval passes, fusion
methods aim to increase document recall and re-
duce missed evidence. Prior work has shown that
these techniques can improve recall-oriented met-
rics in isolation, motivating their adoption in many
RAG pipelines.
Despite their popularity, the end-to-end impact
of retrieval fusion in production RAG systems re-
mains unclear. In practice, retrieval operates un-
der additional constraints, including fixed retrieval
depth, limited re-ranking capacity, context win-
dow budgets, and strict latency requirements. Un-
der these conditions, increases in raw recall may
be offset by redundancy, conflicting context, or
truncation effects introduced downstream.
In this paper, we investigate whether re-
trieval fusion improves system-level performance
in a production-style RAG setting. Rather than
proposing new fusion algorithms, our focus is on
evaluating whether recall gains observed at the
retrieval layer translate into meaningful improve-
ments after re-ranking and context selection. Our
results show that while fusion increases candidate
diversity, these gains are often neutralized once re-
alistic pipeline constraints are applied, resulting in
limited or negative impact on KB-level accuracy
and latency.
1.1 Contributions
This paper makes the following contributions:arXiv:2603.02153v1  [cs.IR]  2 Mar 2026

• We present a production-oriented evaluation
of retrieval fusion methods in a RAG pipeline
operating under fixed retrieval depth, re-
ranking budgets, and latency constraints.
• We empirically analyze the relationship be-
tween retrieval-level recall gains and down-
stream KB-level accuracy, showing that
fusion-induced recall improvements are fre-
quently neutralized after re-ranking and trun-
cation.
• We identify practical trade-offs introduced
by fusion, including increased redundancy
and additional latency overhead, that limit its
effectiveness in realistic RAG deployments.
2 Background
Retrieval-Augmented Generation (RAG) systems
were adopted in this work to support factual ques-
tion answering over large, public facing enter-
prise knowledge bases, where standalone large
language models (LLMs) are insufficient. By con-
ditioning generation on retrieved evidence, RAG
enables grounded, traceable responses under oper-
ational constraints such as access control, latency,
and evolving content.
In practice, early deployments revealed that
end-to-end performance was often limited by re-
trieval rather than generation. Short, underspec-
ified, or weakly lexicalized queries—common in
real user interactions—frequently failed to surface
the correct knowledge-base articles, even when
using strong hybrid retrievers. These retrieval
failures propagated downstream, constraining the
generator’s output and resulting in incomplete or
low-quality responses despite capable LLMs.
To address these recall failures, retrieval fusion
techniques such as multi-query retrieval and Re-
ciprocal Rank Fusion (RRF) have been proposed
to increase document coverage by aggregating re-
sults from multiple retrieval signals (Rackauckas,
2024). Prior work reports consistent improve-
ments on recall-oriented metrics, particularly for
ambiguous or short queries, motivating the adop-
tion of fusion in modern RAG pipelines.
However, in production systems, retrieved can-
didates are subject to reranking, deduplication,
and truncation to fit fixed context windows, rais-
ing questions about whether recall gains from fu-
sion survive downstream constraints. These ob-
servations motivate our study: rather than eval-
uating fusion in isolation, we examine whetherits retrieval-level benefits translate into improved
evidence availability and practical gains within a
production-scale RAG pipeline.
3 Problem Statement
Retrieval fusion is commonly adopted in
Retrieval-Augmented Generation (RAG) systems
under the assumption that increasing retrieval
recall leads to more accurate and trustworthy
answers. In enterprise environments, internal
RAG systems are deployed to reduce time-to-
resolution for support and operational queries,
making answer accuracy critical. However, fusion
also introduces additional system complexity
and latency through extra retrieval passes, fusion
logic, and downstream processing.
This work examines whether the added com-
plexity and latency of retrieval fusion produce
measurable improvements in answer quality for an
enterprise-scale internal RAG system. In produc-
tion pipelines, answer quality depends on whether
relevant knowledge-base articles appear within
the final Top-Kranked context after reranking and
truncation; improvements at the retrieval layer are
therefore only meaningful if they translate into
improved end-to-end evidence availability under
realistic enterprise constraints.
We test the following hypotheses:
•H1:Retrieval fusion does not produce a sta-
tistically significant improvement in Top-K
evidence accuracy compared to a strong hy-
brid retrieval baseline.
•H2:Recall gains introduced by fusion are
largely neutralized by downstream reranking
and context truncation.
•H3:Any benefits from fusion are confined to
recall-scarce query regimes and do not mate-
rially impact system-wide enterprise perfor-
mance.
4 Methodology
We evaluate retrieval fusion within an enterprise-
scale RAG retrieval pipeline designed to closely
mirror production deployment conditions. The
goal of this methodology is not to optimize re-
trieval performance in isolation, but to assess
whether recall gains introduced by fusion persist
through downstream ranking stages and improve
the availability of relevant evidence under realis-
tic enterprise constraints.

4.1 System Baselines
We employ a standard hybrid retrieval strategy
that combines sparse lexical matching (BM25)
with dense semantic retrieval, followed by a per-
query cross-encoder reranking stage, reflecting
common practice in large-scale information re-
trieval systems (Robertson and Zaragoza, 2009).
•Baseline: Hybrid retrieval (BM25) followed by
cross-encoder reranking (FlashRank), dedupli-
cation, and truncation to a fixed Top-Kcon-
text (Pradeep et al., 2023).
All baselines share the same document cor-
pus, chunking strategy, metadata filtering rules,
retrieval depth, reranker model, and ranking hy-
perparameters. This ensures that observed differ-
ences are attributable solely to the presence or ab-
sence of retrieval fusion. For all experiments, re-
trieval depth is fixed atK= 10, and all down-
stream reranking, fusion, and truncation opera-
tions operate over this fixed candidate budget.
4.2 Fusion Configuration
We evaluate a production-faithful two-query fu-
sion strategy. For each input query, retrieval is
performed using:
•Q1: The original user query.
•Q2: A single LLM-generated paraphrase or re-
formulation of the original query.
For each query, a hybrid retriever returns the top
Kdocument chunks. All documents are chun-
ked using a fixed size of 512 tokens and embed-
ded using the Granite embedding model. Query
reformulations (Q2) are generated from Q1 using
a LLaMA-based language model, following the
prompt strategies described inA.4.3.
Retrieval results for Q1 and Q2 are processed
independently through the same cross-encoder
reranker used in the baseline. Reciprocal Rank
Fusion (RRF) is then applied to the two re-ranked
lists to produce a fused candidate set.
Importantly, fusion is appliedafterper-query
reranking, reflecting production constraints where
retrieval outputs are ranked independently prior to
aggregation. All downstream components, includ-
ing deduplication, truncation, and evaluation, re-
main unchanged relative to the baseline, ensuring
that observed differences reflect the impact of fu-
sion rather than embedding, chunking, or query-
generation choices (see Section 6).4.3 Dataset and Ground Truth
Experiments are conducted on a synthetic dataset
ofN= 115enterprise-style support queries.
Each query is derived from a known knowledge-
base (KB) article and paired with its originating
article as ground truth, enabling controlled evalua-
tion of retrieval behavior under realistic document
distributions. Retrieval relevance is evaluated at
the KB article level, while ranking operates at the
chunk level using chunk-aware identifiers.
All retrieved text chunks and associated meta-
data are represented using standardized docu-
ment abstractions provided by the LangChain
framework, which is used to orchestrate re-
trieval and ranking components within the RAG
pipeline (Kakkerla et al., 2024). Product metadata
is used exclusively to scope retrieval and is not
used during ranking or evaluation.
4.4 Evaluation Protocol
End-to-end performance is evaluated using Top-
KKB-level hit metrics. For each query, we de-
termine whether at least one chunk belonging to
a ground-truth KB article appears within the final
Top-Kranked context after reranking and trunca-
tion. Metrics reported include Top-1, Top-3, and
Top-10 accuracy, aggregated across the evaluation
set.
This evaluation framework captures whether
retrieval and fusion strategies successfully sur-
face relevant evidence to downstream compo-
nents, without relying on subjective generation
quality judgments.
4.5 Operational Constraints
All experiments are conducted under enterprise
deployment constraints that directly influence the
effectiveness of retrieval fusion:
• Fixed reranking capacity, limiting the number of
candidates that can be meaningfully scored.
• Context window limits, requiring truncation of
ranked results prior to generation.
• Metadata-based retrieval filtering to enforce
product relevance and access control.
5 Framework Overview
Evaluating retrieval fusion in RAG systems re-
quires more than comparing retrieval metrics in
isolation. While fusion techniques are typi-
cally motivated by recall-oriented objectives, their

downstream impact depends on how retrieved
candidates interact with ranking capacity, context
limits, and operational constraints within the full
pipeline. The evaluation framework is designed to
surface these interactions and identify where and
why fusion gains may fail to translate into mean-
ingful system-level improvements.
Synthetic Questions (RAGAS)
Fusion Question Generation
(Use a prompt to generate a simi-
lar question (Q2) to the original (Q1))
Retrieve Top-KChunks
(Q1 original; optional Q2 reformulation)
Rank Results
(per-query re-rank; optional RRF fusion)
Evaluate (KB-level)
Top-1 / Top-3 / Hit@10 + Latency
Aggregate + Report
tables, plots, trade-offs
Figure 1: Simplified overview of the evaluation
pipeline.
The analysis is structured around a set of re-
curring caveats observed when introducing fusion
into production-faithful RAG pipelines, including
reranker saturation, contextual redundancy, con-
flicting evidence, and increased operational over-
head. Rather than treating these effects as sec-
ondary concerns, this work explicitly measures
their impact on whether relevant evidence sur-
vives into the final ranked context.
The experimental setup follows the retrieval
and ranking pipeline described inAppendix Aand
exhibited in the figure below (Figure 1). For each
query, a comparison is made using a strong hybrid
retrieval baseline against a two-query fusion vari-
ant, holding all downstream components fixed.
Retrieval outputs are ranked at the chunk level,
fused using Reciprocal Rank Fusion, and trun-
cated to a fixed Top-Kcontext prior to evaluation.
The primary goal is to determine whetherfusion improves the availability of correct
knowledge-base evidence after reranking and
truncation. To this end, there is a focus on Top-K
KB-level hit metrics as a proxy for downstream
utility. This choice allows the effect of fusion
within the retrieval and ranking stages to be iso-
lated, without conflating results with generation-
specific variability.
By organizing the evaluation around concrete
failure modes and measuring their impact inde-
pendently, this framework enables a more precise
assessment of when retrieval fusion is beneficial,
when it is neutralized, and when it introduces neg-
ative trade-offs.
6 Results
We present results organized by the primary
caveats identified in our evaluation framework.
For each case, we compare the fusion configu-
ration against the baseline under identical down-
stream conditions and report retrieval, ranking,
and operational outcomes. Together, these results
characterize how retrieval fusion behaves under
realistic pipeline constraints and how its observed
effects align with the hypotheses outlined in Sec-
tion 3.
Tables 1–3 present end-to-end accuracy for
three fusion prompt strategies described in Ap-
pendix A.4.3, ranging from conservative query
reformulation to explicit diversity maximization.
Results are reported using KB-level Top-Khit
metrics, reflecting whether relevant knowledge-
base evidence survives reranking and truncation
under a fixed retrieval budget.
6.1 Recall Gains Are Neutralized After
Reranking
Across all evaluated configurations, retrieval fu-
sion consistently increases upstream recall, con-
firming its intended effect at the retrieval layer.
However, once retrieved candidates are passed
through cross-encoder reranking and truncated to
a fixed Top-Kcontext, these gains are not consis-
tently reflected in downstream KB-level accuracy.
As shown in Tables 1, 2, and 3, fusion variants ex-
hibit accuracy comparable to, and in some cases
lower than, the baseline across Top-1, Top-3, and
Hit@10 metrics.
These results are consistent withH2. Un-
der fixed reranking and truncation constraints,
additional candidates introduced by fusion fre-

Method Top-1 Top-3 Hit
baseline30.43%37.39%51.30%
rerank onrrfq1 29.57%38.26%47.83%
rerank onrrf2 15.65% 23.48% 47.83%
rrfq1q2 25.22%38.26%47.83%
Table 1: Summary metrics table for Fusion Prompt 1
Method Top-1 Top-3 Hit
baseline29.57%36.52%50.43%
rerank onrrfq129.57%39.13% 46.09%
rerank onrrf2 7.83% 17.39% 46.09%
rrfq1q2 27.83%40.87%46.09%
Table 2: Summary metrics table for Fusion Prompt 2
Method Top-1 Top-3 Hit
baseline29.57%36.52%50.43%
rerank onrrfq129.57%40.00% 44.35%
rerank onrrf2 10.43% 19.13% 44.35%
rrfq1q2 27.83%40.87%44.35%
Table 3: Summary metrics table for Fusion Prompt 3
quently fail to survive into the final ranked con-
text. While increased recall expands the candi-
date pool, downstream ranking capacity appears
to limit the extent to which these gains can influ-
ence the evidence ultimately provided to the lan-
guage model.
6.2 Statistical Significance at Operational
Top-3 Depth
The differences in Top-3 accuracy across fusion
variants are small, so we run a paired significance
test to check whether these changes are likely to
be real or just noise. We treat each query as a
binary hit/miss outcome: a query is a hit if any
chunk from a ground-truth KB article appears in
the final Top-3 ranked list after reranking and trun-
cation (this matches the same Top-3 context we
pass to the LLM during generation). Since ev-
ery method is evaluated on the sameN= 115
queries, we use McNemar’s exact test on paired
outcomes when comparing each fusion method
against the baseline (re only q1). Because
we run multiple comparisons within each prompt
type, we apply Benjamini–Hochberg correction
(FDR= 0.05). We excludererank onrrfq2
from this analysis because it performs substan-
tially worse than the baseline across prompts and
does not represent a realistic deployment option.Table 4 shows that Prompt 2/3 have slightly
higher Top-3 hit rates (+2.61to+4.35percent-
age points), but none of these improvements are
statistically significant after correction (allp adj≥
0.125). In other words, fusion with certain pa-
rameters has a slightly positive impact on retrieval
accuracy, but the effect is not consistent enough
to confidently justify the extra fusion overhead in
production.
P Method Top-3∆ppp p adj
P1 rrf q1q2 38.26% +0.87 1.0000 1.0000
P1 rerank onrrfq1 38.26% +0.87 1.0000 1.0000
P2 rrf q1q2 40.87% +4.35 0.1250 0.1667
P2 rerank onrrfq1 39.13% +2.61 0.2500 0.2500
P3 rrf q1q2 40.87% +4.35 0.1250 0.1250
P3 rerank onrrfq1 40.00% +3.48 0.1250 0.1250
Table 4: Paired Top-3 significance vs baseline.
6.3 Contextual Redundancy and Conflicting
Evidence
Fusion adds very littleexactoverlap between the
Q1 and Q2 retrieval lists, which means the rewrite
(Q2) is usually pulling a different set of candidates
than the original query. That is easy to see in Ta-
ble 5: for Prompts 2 and 3 the Jaccard similar-
ity is around0.09, even though the union grows
to roughly 15 distinct sources. We report overlap
and Jaccard because they tell us whether Q2 is ac-
tually adding new material (low overlap / low Jac-
card) versus just returning the same documents in
a different order, while the union size shows how
much the candidate pool expands once we com-
bine Q1 and Q2.
Low overlap by itself is not a problem—it is
the point of generating a reformulation. The is-
sue is that many of the “new” candidates are not
truly newinformation. They often repeat the same
ideas (near-duplicate chunks from nearby KB sec-
tions) or introduce partially conflicting guidance
across related articles. That makes reranking and
truncation less stable and can push genuinely use-
ful evidence out of the Top-Kwindow.
This also helps explain why
rerank onrrfq2performs so poorly.
When we fuse Q1 and Q2 with RRF and then
rerank the fused list using Q2, the scoring can
over-align to Q2-specific phrasing (and any
drift introduced during rewriting). In practice,
strong Q1-aligned documents can get demoted or
dropped, which matches the sharp Top-1 drops
forrerank onrrfq2in Prompts 2 and 3.

Overall, fusion increases breadth, but semantic
redundancy and Q2-anchored reranking can
reduce the quality of the final context, aligning
withH2.
P Ov@10 Jac@10 Un@10 Uniq@10
P1 2.87 0.279 13.20 7.82
P2 1.09 0.087 15.29 7.77
P3 1.10 0.094 14.99 7.77
Table 5: Exact Q1–Q2 overlap (Top-10).
6.4 Limited Impact on End-to-End Evidence
Availability
Despite consistent improvements in upstream re-
call, fusion does not yield consistent gains in Top-
KKB-level accuracy across the evaluation set.
In several configurations, fusion slightly degrades
Top-1 and Top-3 accuracy relative to the baseline,
indicating that added candidates can interfere with
ranking effectiveness rather than improve it.
These findings are consistent withH1. Im-
provements at the retrieval layer do not reliably
translate into better end-to-end evidence availabil-
ity once downstream constraints are applied. Re-
sults are consistent across fusion prompt variants
and reranking anchors within this experimental
setup, suggesting that the observed behavior is not
driven by a single configuration choice.
6.5 Latency and Pipeline Overhead
Fusion introduces additional latency due to ex-
tra retrieval passes, fusion logic, and increased
downstream processing. As summarized in Ta-
ble 6, while median latency remains within ac-
ceptable operational bounds, tail latency is consis-
tently higher than the baseline across experimen-
tal runs. This effect compounds as retrieval depth
and query volume increase.
Metric Avg Time (s)
Fusion question generation 0.89
Baseline Retrieval time (Q1) 54.60
Retrieval time (Q1 + Q2 combined) 65.98
RRF time 0.012
Flashrank time (K= 10) 0.26
Table 6: Average latency components across all
datapoints.
We observe an additional 0.89s of overhead at-
tributable to secondary fusion processes, includ-
ing fusion question generation, Reciprocal Rank
Fusion, and an additional reranking pass. Re-
trieval latency also increases when processing twoqueries instead of one, reflecting the cost of addi-
tional embedding and search operations.
Although latency is not explicitly captured in
the hypotheses, these results contextualizeH1by
demonstrating that fusion incurs measurable op-
erational costs without delivering corresponding
gains in evidence accuracy. In production environ-
ments where throughput and responsiveness are
critical, this trade-off further constrains the practi-
cal value of fusion.
6.6 Query-Slice Analysis
When results are segmented by query character-
istics, fusion shows modest benefits for a small
subset of recall-scarce queries, particularly those
for which the baseline retriever fails to surface rel-
evant KB articles within the Top-Kresults. In
these cases, query reformulation occasionally sur-
faces relevant evidence that is missed by the base-
line retriever.
These observations partially supportH3. While
fusion can provide localized benefits, these gains
apply to a minority of the workload and do not
materially affect system-wide performance. As
corpus coverage and ranking capacity improve,
the marginal utility of fusion diminishes, reinforc-
ing the conclusion that its effectiveness is highly
context-dependent.
7 Conclusion
This work presents a systematic evaluation of re-
trieval fusion within a production-faithful RAG
retrieval and ranking pipeline. While fusion re-
liably improves upstream recall, our results show
that these gains are frequently neutralized by
downstream reranking and context constraints,
and in several configurations lead to reduced ev-
idence availability and increased latency.
By evaluating fusion through concrete pipeline
bottlenecks such as reranker saturation, contextual
redundancy, and operational overhead we demon-
strate that recall-oriented improvements do not au-
tomatically translate into downstream utility. In
mature enterprise systems, ranking capacity and
context selection emerge as the dominant factors
shaping end-to-end performance, often outweigh-
ing gains achieved by expanding retrieval breadth.
Our findings indicate that the benefits of fusion
are highly context-dependent and largely confined
to recall-scarce query regimes. For the majority
of enterprise workloads, fusion introduces addi-

tional system cost without delivering material im-
provements in evidence availability.While fu-
sion may yield marginal improvements in iso-
lated retrieval metrics, these gains are often
insufficient to justify the added system com-
plexity, computational overhead, and opera-
tional costs in production environments. The
modest downstream benefits observed in our
evaluation suggest that simpler, more main-
tainable architectures may offer superior cost-
benefit tradeoffs for most enterprise RAG de-
ployments.

References
Gordon V . Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms con-
dorcet and individual rank learning methods. InProceedings of the 32nd International ACM SIGIR Conference
on Research and Development in Information Retrieval, pages 758–759.
Sahil Es, Jithin James, Kumar Shridhar Pradeep, Salman Khan, and Brandon Talley. 2023. Ragas: Automated
evaluation of retrieval augmented generation.arXiv preprint arXiv:2309.15217.
Shiva Kumar Kakkerla, Krishnaveni Yalla, Pujitha Pathipaka, Yashwanthini Irukulla, Sriharshini Panthangi, and
Rohith Adupa. 2024. Question and answer model using langchain framework and generative ai. In2024
Intelligent Systems and Machine Learning Conference (ISML), pages 200–205.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
augmented generation for knowledge-intensive nlp tasks. InAdvances in Neural Information Processing Sys-
tems, volume 33, pages 9459–9474.
Kumar Shridhar Pradeep, Pablo Moreno, and Srikanta Muthukrishnan. 2023. Flashrank: Ultra-fast reranking with
lightweight language models.arXiv preprint arXiv:2311.07179.
Zackary Rackauckas. 2024. Rag-fusion: A new take on retrieval augmented generation.International Journal on
Natural Language Computing, 13(1):37–47.
Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond.Foun-
dations and Trends in Information Retrieval, 3(4):333–389.

A Experimental Pipeline and Evaluation
Methodology
This appendix documents the end-to-end pipeline
used to evaluate retrieval, fusion, and re-ranking
strategies in a Retrieval-Augmented Generation
(RAG) system operating over a large enterprise
knowledge base. The goal here is transparency:
what data went in, what intermediate artifacts
were produced, how ranking was applied, and how
metrics were computed. The main paper focuses
on outcomes; this appendix captures the mechan-
ics behind those outcomes.
A.1 Terminology and Core Objects
We use the following terminology throughout the
appendix.
A.1.1 Question Set
Thequestion setis a collection of customer-
support-style queries used to probe the retrieval
system. In this study, questions weresyntheti-
cally generated using RAGAS. Synthetic gener-
ation enables controlled experimentation at scale
while still producing queries that resemble real
support interactions (Es et al., 2023).
A.1.2 Knowledge Base (KB)
Theknowledge base (KB)is a large text cor-
pus containing curated domain knowledge, such
as troubleshooting guides, known issues, and pro-
cedures. The KB represents subject matter expert
(SME) knowledge in written form and serves as
the retrieval corpus for all experiments.
A.1.3 KB Article and Chunks
AKB articleis a single document focused on
a specific topic. For retrieval, each KB article
is split into smaller text segments calledchunks.
Each chunk is stored as aLangChainDocument
object.
Chunking is used because retrieval models gen-
erally match smaller, semantically coherent spans
more reliably than full-length documents, espe-
cially when articles are long or cover multiple
subtopics.
A.1.4 Index
OurIndexis a collection of KB articles that
are embedded from text into vectors using indus-
try standard embeddings models. The aforemen-
tioned KB article chunks are stored in this index.
During the retrieval stage, the retriever performsa similarity search between the embedded query
(using the same embeddings model) and the in-
dex.
A.1.5 Retrieval Result Document
Aretrieval resultis a LangChainDocument
representing one retrieved chunk. Each result con-
tains:
•page content: the chunk text returned by
retrieval
•metadata: fields describing the origin of
the chunk, including (at minimum) itssource
KB articleandchunk index
A.1.6 Top-KResults
For each query, the retriever returns a ranked list
of theTop-Kdocuments (chunks). K is a value
defined by us, in this case K=10. When results
from multiple queries are logged in a single list,
we keep a fixed positional convention:
Positions Meaning
1. . . KQ1 retrieval results
K+1. . .2KQ2 retrieval results
Table 7: K Retrieval Results
A.2 Dataset Construction and Ground Truth
A.2.1 Query Dataset
Each experimental run operates over a dataset
where each row corresponds to a single query in-
stance. At minimum, each row contains:
•question: the natural-language query text
(synthetically generated via RAGAS)
•ground truth: the ’true’ KB article from
which the question was generated (Section
A.2.2)
In practice, additional metadata may be present
(e.g., product category labels, query length). This
metadata is used only for filtering and corpus
scoping (Section A.4.1), and is not directly used
as a ranking signal.
A.2.2 Ground Truth Labels
Relevance is defined at theKB article level. Each
questionis associated with one or more valid
’ground truth’ KB articles:

•kb article id: one or more KB article
identifiers considered correct for that query
During evaluation, a retrieval is considered ahit
ifanyground-truth KB article is retrieved (via any
of its chunks) within the top-Kranked results.
A.3 Document Representation
A.3.1 Chunking Strategy
All KB articles are preprocessed into chunks
prior to indexing. Each chunk is represented as
a LangChainDocumentwithpage content
(text) andmetadata. The metadata includes:
•source kb: identifier of the KB article the
chunk came from
•chunk index: the chunk position within
the article
• optional fields used for filtering and access
control (if applicable)
A.3.2 Chunk-Aware Identifiers
Because multiple chunks can come from the same
KB article, we treat the chunk as the atomic re-
trieval unit. To avoid accidentally collapsing mul-
tiple relevant chunks into a single item during
fusion or re-ranking, each chunk is assigned a
unique identifier:
Formatsource kb::chunk=chunk index
ExampleKB123456::chunk=0
Table 8: Source identifier format for KB chunk
references.
This chunk-aware identifier is carried through
retrieval, fusion, and re-ranking. Importantly, it
is normal for multiple chunks from the same KB
article to appear in the same top-Klist. We want
to maintain a chunks position during re-ranking
to ensure that if multiple chunks of the same KB
article are used to formulate an answer we don’t
lose any of that information.
A.4 Retrieval Stage
A.4.1 Metadata Filtering
A lightweight metadata filtering layer is used to
scope retrieval to a relevant portion of the KB. The
purpose is operational rather than theoretical: lim-
iting the candidate set can improve precision andlatency by excluding articles that are clearly out-
side the query’s target product category (or other
high-level grouping).
Filtering is appliedbeforeretrieval (i.e., it con-
strains the searchable subset), and does not di-
rectly change scoring or ranking logic.
A.4.2 Two-Query Retrieval for Fusion
Experiments
To evaluate fusion strategies, we retrieve using
two query formulations:
•Original Query (Q1): the original question
text
•Fusion Query (Q2): an LLM-generated re-
formulation (e.g., paraphrase) intended to
surface alternate matching evidence
For each query (Q1and optionallyQ2), the
retriever returns top-Kchunks as LangChain
Documentobjects. The ordered outputs are con-
catenated using the positional convention shown
earlier.
A.4.3 Fusion Question Generation Prompts
In order to better address our problem state-
ment, we decided to perform the experiment with
three different fusion generation prompts. These
prompts are sent to the LLM as instructions along
with the original question. The LLM then gener-
ates a similar question using re-wording strategy
defined in the instruction, as well as some infor-
mation about the question (product category, etc).
<|instruction|>
You are a technical query rewriter.
You will be given a technical
query, case details and product
information.
Re-write the query without adding
details about the product or case.
Guidelines:
- Simply give the output, not your
reasoning for it
- Don’t include any extra details
other than the ones provided in the
input
- Keep it Direct and Concise
</instruction>
Figure 2: Prompt Type 1

<|instruction|>
You are a technical query rewriter.
Your role is to re-write a given
technical query for Troubleshooting.
Utilize the given information to
re-write the technical query such
that it is optimized for a RAG
system.
Guidelines:
- Simply give the output, not your
reasoning for it
- Don’t include any extra details
other than the ones provided in the
input
- Make the query precise and
detail-oriented without making it
too long.
</instruction>
Figure 3: Prompt Type 2
<|instruction|>
You are a retrieval strategist.
Your job is to generate diverse,
high-quality rewrites of a technical
query to maximize document recall in
a RAG fusion pipeline.
You must only return ONE
high-quality re-write.
Guidelines:
- Simply give the output, not your
reasoning for it
- Don’t include any extra details
other than the ones provided in the
input
</instruction>
Figure 4: Prompt Type 3
A.5 Re-Ranking and Fusion Pipeline
The experimental pipeline evaluates a production-
like ranking stack composed of multiple stages.
Not every stage is used in every configuration, but
the outputs are logged consistently to allow direct
comparison.
A.5.1 Per-Query Re-Ranking (Baseline)
To emulate an enterprise setup where retrieval out-
puts are refined before any fusion is applied, each
query’s retrieved list is re-ranked independently
using the flashrank module. This yields:
•baseline: re-ranked list for Q1 retrieval
results
•q2 baseline: re-ranked list for Q2 re-
trieval results
Each list is stored as an ordered sequence of
chunk-aware identifiers. If multiple chunks fromthe same KB article are ranked highly, they are
preserved. The re-ranked list of Q1’s retrieval re-
sults is ourbaselinedatapoint. Refer to Tables 8,9
and 10 below.
A.5.2 Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion (RRF) is applied to com-
bine multiple ranked lists into a single fused rank-
ing. In this study, RRF is applied over the already
re-ranked lists (baseline,q2 baseline)
For each ranked listL i, a chunkdat rankrcon-
tributes:
score(d) +=1
k+r
wherekis a smoothing constant.
Key properties of the implementation:
• Fusion is performed at thechunk levelusing
chunk-aware identifiers.
• Each chunk contributes at most once per list
(best rank is used if duplicates occur).
• The output is a single fused ordered list, de-
notedrrf q1q2.
A.5.3 Post-Fusion Re-Ranking
To isolate whether fusion benefits from additional
semantic refinement, we optionally re-rank the
top-Kitems from the fused list using flashrank.
This produces two variants, depending on which
query formulation is used as the re-ranking an-
chor:
•rerank onrrfq1: re-rank fused list us-
ing Q1 as the anchor
•rerank onrrfq2: re-rank fused list us-
ing Q2 as the anchor
This stage helps separate:
• the effect offusion alone, versus
• the effect offusion + additional re-ranking.
To see the evaluation of these lists, refer to Ta-
bles 8, 9 & 10 below.
A.6 Recorded Outputs
For each instance of a pair of queries (Q1, Q2)
(question), the experiment logs:
• ranked output lists for each pipeline config-
uration (per-query re-ranks, RRF fused list,
post-fusion re-ranks)

• retrieval latency measurements
• re-ranking latency measurements
• fusion question generation latency measure-
ments (when enabled)
All ranked lists are stored as ordered sequences
of chunk-aware identifiers, enabling consistent
evaluation even when chunk text or metadata
changes over time.
A.7 Evaluation Methodology
A.7.1 KB-Level Evaluation from
Chunk-Level Rankings
Although retrieval and ranking operate on chunks,
correctness is defined at the KB article level. To
evaluate results, we map each chunk-aware iden-
tifier back to its KB article by removing the chunk
suffix:
source kb::chunk=idx→source kb
A method is counted as ahitfor a query ifany
chunk belonging toanyground-truth KB article
appears within the top-Kranked results for that
method.
This design matches how a RAG based conver-
sational model works. Retrieving any strong sup-
porting passage from the correct article is usually
sufficient for downstream answer generation, even
if the exact chunk differs.
A.7.2 Metrics
For each ranking method, we compute:
•Top-1 accuracy: fraction of queries where a
ground-truth KB article appears at rank 1
•Top-3 accuracy: fraction of queries where a
ground-truth KB article appears within ranks
1–3
•Hit@10 (Top-10 hit rate): fraction of
queries where a ground-truth KB article ap-
pears within ranks 1–10
Metrics are computed per query and then aggre-
gated across the dataset.
A.7.3 Evaluation Artifacts
The primary evaluation artifact is asummary
metrics tablewith one row per method and
columns for Top-1, Top-3, and Hit@10. This is
the table used in the main paper for reporting and
comparison.Method Top-1 Top-3 Hit
baseline 0.3043 0.3739 0.5130
rerank onrrfq1 0.2957 0.3826 0.4783
rerank onrrf2 0.1565 0.2348 0.4783
rrfq1q2 0.2522 0.3826 0.4783
(a) Summary metrics table for Fusion Prompt 1
Method Top-1 Top-3 Hit
baseline 0.2957 0.3652 0.5043
rerank onrrfq1 0.2957 0.3913 0.4609
rerank onrrf2 0.0783 0.1739 0.4609
rrfq1q2 0.2783 0.4087 0.4609
(b) Summary metrics table for Fusion Prompt 2
Method Top-1 Top-3 Hit
baseline 0.2957 0.3652 0.5043
rerank onrrfq1 0.2957 0.4 0.4435
rerank onrrf2 0.1043 0.1913 0.4435
rrfq1q2 0.2783 0.4087 0.4435
(c) Summary metrics table for Fusion Prompt 3
A.8 Robustness and Failure Handling
We took steps to ensure that our iterative process
is controlled and consistent. To maintain experi-
mental integrity and avoid mixing partial outputs:
• retrieval calls are retried on transient failures
(e.g., timeouts or backend instability)
• queries that fail after the maximum retry bud-
get are skipped and logged
• downstream stages (re-ranking and fusion)
assume retrieval succeeded, preventing par-
tially populated lists from contaminating
metrics
A.9 Reproducibility
Given fixed retriever parameters, fixed re-ranker
models, and fixed fusion hyperparameters, the
pipeline is deterministic. Chunk-aware identifiers
provide a stable unit of comparison across stages
and prevent accidental collapse of evidence when
multiple chunks originate from the same KB arti-
cle.
# Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG

**Authors**: Jorge Gabín, Anxo Perez, Javier Parapar

**Published**: 2026-05-26 14:44:19

**PDF URL**: [https://arxiv.org/pdf/2605.27105v2](https://arxiv.org/pdf/2605.27105v2)

## Abstract
Retrieval-Augmented Generation (RAG) systems rely on retrieved documents being concatenated into a model's input context, making both document ordering and context size critical yet controversial design choices. Prior work reports position-based effects such as lost in the middle and related long-context phenomena. However, empirical findings remain inconsistent and hard to reproduce across models, datasets, and evaluation protocols. In this paper, we present a systematic reproducibility study that revisits these claims and examines how they evolve with contemporary LLMs under a controlled evaluation framework. We first show that topic sampling is a major source of variance: small topic sets can mask or exaggerate ordering effects. Based on repeated subset sampling across multiple topic budgets, we provide a practical calibration procedure that identifies topic counts yielding stable trends at feasible cost. Using these fixed topic sets, we then reproduce and extend results on position sensitivity, re-evaluating lost in the middle and positional biases in modern LLMs. Then, we also study a more realistic RAG scenario in which relevance is mediated by a retriever rather than oracle access to ground-truth documents. In this setting, we re-examine a recent industry study and identify discrepancies to evaluation choices such as limited topic coverage and reliance on LLM-based judges. Finally, we conduct an analysis of how retrieval order and context size affect downstream LLM performance under imperfect retrieval. Our results demonstrate that both factors interact strongly with retrieval quality and model choice, and that conclusions drawn from idealised setups do not always transfer to real-world RAG pipelines. We release all code and configurations to support reproducibility and future work on robust RAG evaluation.

## Full Text


<!-- PDF content starts -->

Lost in the Evidence? Reproducing Document Position and
Context Size Effects in RAG
Jorge Gabín
jorge.gabin@udc.es
IRLab, CITIC, Universidade da Coruña
A Coruña, Spain
Linknovate Science
Santiago de Compostela, SpainAnxo Perez
anxo.pvila@udc.es
IRLab, CITIC, Universidade da Coruña
A Coruña, SpainJavier Parapar
javier.parapar@udc.es
IRLab, CITIC, Universidade da Coruña
A Coruña, Spain
Abstract
Retrieval-Augmented Generation (RAG) systems rely on retrieved
documents being concatenated into a model’s input context, making
both document ordering and context size critical yet controversial
design choices. Prior work reports position-based effects such as
lost in the middleand related long-context phenomena. However,
empirical findings remain inconsistent and hard to reproduce across
models, datasets, and evaluation protocols. In this paper, we present
a systematic reproducibility study that revisits these claims and
examines how they evolve with contemporary LLMs under a con-
trolled evaluation framework. We first show that topic sampling
is a major source of variance: small topic sets can mask or exag-
gerate ordering effects. Based on repeated subset sampling across
multiple topic budgets, we provide a practical calibration procedure
that identifies topic counts yielding stable trends at feasible cost.
Using these fixed topic sets, we then reproduce and extend results
on position sensitivity, re-evaluatinglost in the middleand posi-
tional biases in modern LLMs. Then, we also study a more realistic
RAG scenario in which relevance is mediated by a retriever rather
than oracle access to ground-truth documents. In this setting, we
re-examine a recent industry study and identify discrepancies to
evaluation choices such as limited topic coverage and reliance on
LLM-based judges. Finally, we conduct an analysis of how retrieval
order and context size affect downstream LLM performance under
imperfect retrieval. Our results demonstrate that both factors in-
teract strongly with retrieval quality and model choice, and that
conclusions drawn from idealised setups do not always transfer to
real-world RAG pipelines. We release all code and configurations to
support reproducibility and future work on robust RAG evaluation.
CCS Concepts
•Information systems→Question answering.
Keywords
LLMs, Retrieval-Augmented Generation, RAG, Question Answering
This work is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3808569ACM Reference Format:
Jorge Gabín, Anxo Perez, and Javier Parapar. 2026. Lost in the Evidence?
Reproducing Document Position and Context Size Effects in RAG. InPro-
ceedings of the 49th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne,
VIC, Australia.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/
3805712.3808569
1 Introduction
Large Language Models (LLMs) have become fundamental compo-
nents of many industrial and research applications [ 13]. Among
them, Retrieval-Augmented Generation (RAG) is a key paradigm
for enabling LLMs to answer questions grounded in external knowl-
edge [ 7,10]. By retrieving relevant passages and injecting them into
the model’s input context, RAG systems effectively extend factual
coverage and adaptability. However, despite widespread adoption,
current implementations often fail to exploit their full potential.
In particular, critical aspects of RAG systems, such as the optimal
number of passages to retrieve (context size) and the order in which
they should be presented, remain uncertain [2, 5, 12, 17, 22].
Prior studies have analysed the context size and passage order
effect on RAG systems, reaching conflicting conclusions: some re-
port strong gains from placing the most relevant passages first [ 12],
or from preserving intra-document order [ 22]. Other authors claim
little to no sensitivity to ordering or retrieval depth, with trends
varying by model and dataset [ 2,5,12,17]. These discrepancies are
difficult to interpret because prior work often differs in the topic
sets used in their experiments, the evaluation protocol (automatic
metrics vs. LLM-based judges), and the LLMs and retrieval pipelines
under study. Thus, practitioners still face uncertainty regarding
crucial configuration choices. Given the increasing reliance of in-
dustry on these systems [ 11,19], a clear understanding of how these
design factors influence results is of paramount importance.
Motivated by these inconsistent findings, our goal is to provide
reliable and reproducible evidence on how context size and passage
ordering affect RAG performance. Specifically, we focus on two
widely used question answering benchmarks: Natural Questions
(NQ) [ 8], which is predominantly single-hop, and HotpotQA [ 20],
which requires multi-hop evidence aggregation. Our starting point
is a methodological observation: conducting experiments on full
topic sets is frequently impractical, especially when comparing mul-
tiple LLMs. However, an arbitrary selection of small topic samples
can yield unstable conclusions.
We therefore introduce a calibration procedure for selecting an
adequate topic budgetbased on the stability of performance trends.
We evaluate multiple subset sizes 𝑛and, for each size, repeatedlyarXiv:2605.27105v2  [cs.IR]  27 May 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Jorge Gabín, Anxo Perez, and Javier Parapar
sample random topic subsets to measure how sensitive ordering
and context size trends are to the specific topics chosen. To ensure
reliability, we identify the threshold where the variance of the
ΔF1between sorting strategies is sufficiently low to prevent “zero-
crossings”, instances where the relative ranking of two methods flips
due to sampling noise. This approach enables us to obtain stable
performance, allowing subsequent experiments with other models
or configurations to be conducted reliably and reproducibly.
Using this controlled evaluation framework, we then revisit
prominent claims in the literature. We reproduce and extend long-
context position-effect studies [ 2,12] under modern LLMs, and
we re-examine recent robustness claims in realistic RAG scenar-
ios where relevance is mediated by imperfect retrieval rather than
oracle access to gold documents [ 22]. Beyond reproductions, we
also assess the impact of the retrieval phase itself, testing both stan-
dard ranking models and oracle rankers, as well as more advanced
retrieval strategies, to determine how crucial retrieval quality is
for overall performance. Additionally, we examine how different
LLM architectures and sizes behave under these varying setups.
By systematically exploring these factors, we aim to clarify under
which configurations RAG systems perform reliably and when their
outputs are sensitive to design choices.
Our contributions are threefold: (1) a dataset-specific calibration
method for choosing topic sizes that yield stable conclusions when
studying ordering and context-size effects; (2) a systematic repro-
ducibility study that revisits positional-bias and RAG robustness
claims under this controlled evaluation framework and contem-
porary LLMs. (3) Finally, an extensive analysis of how retrieval
quality, ordering, context size, dataset characteristics, and model
family/scale influence Question Answering (QA) performance. We
release all code and configurations to support future reproducible
research1.
2 Related Work
Retrieval-Augmented Generation combines a retriever with a gener-
ator to answer knowledge-intensive queries by grounding outputs
in external evidence. Early and influential architectures include
REALM [ 4], and the neural RAG framework of Lewis et al. [ 10]. On
the retrieval side, sparse baselines such as BM25 and dense pipelines
that pair dual-encoder retrieval with cross-encoder reranking re-
main standard in practice [ 7,15,16]. Recent surveys synthesise com-
mon RAG components (retriever, composer, generator), pipeline
variants (e.g., joint vs. fixed retrievers), and evaluation practices,
highlighting open issues around robustness, context construction,
and measurement choices [3, 21].
A growing body of evidence shows that LLMs are sensitive to
where evidence appears in long contexts. Thelost in the middle
phenomenon shows that moving the same answer-bearing content
to different positions can change QA accuracy, often peaking near
the beginning or end and degrading in the middle [12]. Follow-up
work argues that position effects interact with how passages are
composed, including whether chunks preserve intra-document or-
der versus being concatenated purely by relevance, and reports
non-monotonic behaviour as more chunks are added [ 22]. Com-
plementary to purely positional studies, a growing line of RAG
1/gtbhttps://github.com/IRLab-UDC/Lost-in-the-Evidence-in-RAG-Virtual-Appendixrobustness work evaluates end-to-end pipelines under systematic
perturbations to retrieved evidence, including retrieval depth ( 𝑘)
and permutations of the retrieved list [ 2]. These studies often find
that average trends can appear stable, yet robustness is imperfect
and can hide substantial instance-level trade-offs when order or𝑘
changes [ 2]. Another closely related direction studies how retrieval
quality and noise shape generation. Because retrieval is imperfect,
retrieved contexts naturally contain irrelevant or weakly relevant
passages, which can distract the generator and degrade answer cor-
rectness, especially when such passages are prominent or placed in
salient positions [ 1]. More broadly, recent work on RAG evaluation
emphasises that conclusions can depend strongly on choices such
as dataset formatting, chunking/composition, and scoring method-
ology, and that resource constraints often lead to evaluations on
reduced subsets whose representativeness is rarely validated [21].
Taken together, prior work suggests that retrieval depth and
ordering interact with model biases and retrieval quality, and that
conclusions may depend on dataset/sample size, score distributions,
and prompt format. We study all these factors in our reproducibil-
ity study. We differ from past reports by (i) reproducing recent
claims about weak order/depth effects under matched settings, (ii)
isolating sources of variance to obtain stable results, and (iii) extend-
ing the analysis across ordering schemes, context sizes, retrieval
quality (BM25 vs. dense reranking vs. oracle contexts), and model
families/sizes.
3 Towards Stable Evaluation of Context
Ordering and Size
As mentioned, prior work on context size ( 𝑘) and evidence ordering
in LLM-based QA reports inconsistent trends, and it is often unclear
whether disagreements reflect genuine model behavior or instabil-
ity in evaluation protocols [ 2,5,12,22]. Because running full-topic
evaluations is often impractical (especially when comparing multi-
ple LLMs or using expensive judging schemes) many papers rely
on relatively small subsets of topics [ 2,5,12]. However, if topic
sampling itself introduces large variance, then conclusions about
order and the number of retrieved documents 𝑘can be unstable
and difficult to reproduce. In this section, we therefore introduce a
simple calibration procedure to determine anadequate topic budget
for studying context-ordering and context-size effects. The goal
is to identify the smallest number of topics that yield stable con-
clusions while keeping computational cost manageable. We then
adopt the resulting topic counts as a controlled evaluation setting
for all subsequent reproducibility experiments in the paper.
Setup.We focus on two widely used QA benchmarks: NQ [ 8],
which primarily consists of single-hop queries where the answer is
typically contained within a single passage, and HotpotQA [ 20], a
more complex dataset requiring multi-hop reasoning to aggregate
evidence across multiple distinct passages. We employ LLaMA-
3.1:8B as our experimental model, selected for its status as a modern,
widely adopted standard in recent literature, offering a balance of
high performance and manageable medium-scale parameter size.
For each query, we construct a context by retrieving the top- 𝑘
passages and concatenate them into the prompt. Similar to prior
works, we study three standard ordering schemes applied to a fixed
top-𝑘set:standard(descending retrieval score),reverse(ascending

Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
(a) HotpotQA (500 topics)
(b) NQ (500 topics)
Figure 1: Performance variability across 10 random subsets
of 500 topics for HotpotQA and NQ. The figure shows the ΔF1
between ordering strategies at different context sizes. Dots
indicate the mean ΔF1, error bars represent the standard
deviation, and shaded areas denote the minimum/maximum
values across subsets.
score), andrandom(uniform permutation). Our analysis uses F1-
token (hereafter F1) as the primary metric, since it is the official
benchmark metric and does not depend on a particular judge model.
3.1 Investigating Sources of Variability
We first quantify how sensitive ordering conclusions are to the
particular set of evaluation topics. For a fixed subset size 𝑛, we
repeatedly sample topic subsets using different random seeds. For
each subset, we evaluate all context sizes and ordering schemes and
compute ΔF1, which is the pairwise F1 difference between strategies
(e.g., reverse–standard, reverse–random, standard–random).
Figures 1a and 1b summarise this analysis over 10 random subsets
with 𝑛=500topics per dataset. We use 500 topics because it is a
(a) HotpotQA (1000 topics)
(b) NQ (2000 topics)
Figure 2: Performance variability across 10 random subsets
of 1000 and 2000 topics for HotpotQA and NQ, respectively.
In addition to Figure 1, the dashed line represents the ΔF1
using the full topics.
common evaluation budget in recent RAG work (e.g., [ 2]), and it
lets us test whether a seemingly reasonable topic size can still yield
unstable conclusions. In the plots, dots show the mean ΔF1across
subsets, error bars denote the standard deviation, and the shaded
region indicates the min–max range. The x-axis is context size and
the y-axis is ΔF1. With 𝑛=500, we can see the variability is large
enough that the apparent winner among ordering strategies can
change across samples, particularly at smaller context sizes.
This implies that conclusions about whether “order matters” can
depend heavily on which topics happen to be included in the eval-
uation, providing a plausible explanation for discrepancies across
prior reports that use different subsets and protocols. This leads us
to our next point, where we study the adequate number of topics
and queries to ensure that observed trends are robust and represen-
tative.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Jorge Gabín, Anxo Perez, and Javier Parapar
3.2 Determining an Adequate Number of Topics
We next translate our variability diagnosis into a practical calibra-
tion rule: what is the minimum number of topics required to ensure
that the comparative rankings between sorting strategies are sta-
ble? We aim to identify the smallest topic set size where evaluation
findings become representative.
We begin by establishing a ground truth reference through a sin-
gle evaluation over the full topic set for each dataset, which serves
as the upper bound for stability. Next, we systematically vary the
sample size ( 𝑛∈{ 500,1000 ,2000 ,3000 ,4000 ,5000}) and measure
the variance in ΔF1 performance between ordering schemes across
random seeds. Crucially, we focus on minimising the frequency
with which these performance deltas cross the zero line. A zero-
crossing implies that the “winning” strategy flips purely due to
sampling noise, indicating that the sample size is insufficient to
discern a reliable difference. Frequent oscillations around zero sug-
gest that the relative ranking of methods is unstable. Conversely, a
reliable sample size yields a ΔF1 that remains consistently positive
or negative, ensuring that the ordering of methods is deterministic
rather than an artefact of the specific subset chosen.
Our experiments reveal that increasing the number of topics
progressively decreases these oscillations, eventually reducing the
variance enough that the performance deltas no longer intersect
the zero line for noticeable differences. This transition marks the
practical threshold for reliable evaluation. Figure 2 reports these
results for the selected topic sizes, where the dashed line represents
the reference differences measured over the full datasets. As illus-
trated in Figures 2a and 2b, using 𝑛= 1000and 𝑛= 2000topics
for HotpotQA and NQ, respectively, keeps the standard deviation
sufficiently low to prevent most zero-crossings, thereby preserving
the correct relative ordering of system performance. For a detailed
analysis of the remaining topic sizes, we refer the reader to the
virtual appendix2.
3.3 Controlled and Reproducible Experimental
Setup
Building on the calibration results presented above, we adopt topic
sizes that yield low variance and stable ordering trends for each
dataset: 𝑛=1000topics for HotpotQA and 𝑛=2000topics for NQ.
We use these fixed topic sets throughout the remainder of the paper
to control for topic-sampling noise and enable fair comparisons.
This controlled evaluation setting supports two subsequent parts
of the paper. First, in Section 4 we revisit and reproduce key prior
claims on positional effects and order/size robustness, using the
same stable topic sizes to avoid confounding conclusions with sam-
pling variance. Second, in Section 5 we conduct our main analyses
under the same controlled protocol, isolating how context size
and ordering interact with (i) the dataset structure (single-hop vs.
multi-hop), (ii) retrieval quality, and (iii) model family and scale.
4 Reproducibility Study of Context Position
and Retrieval Robustness
In this section, we reproduce two representative lines of work
under our controlled evaluation framework. We first test whether
2Information at /gtb:https://github.com/IRLab-UDC/Lost-in-the-Evidence-in-RAG-Virtual-
Appendix/blob/master/rag_calibration_topic_budget.ipynb
Figure 3:Lost in the middlereproduction [ 12] on Am-
bigQA [14] with LLaMA-3.1:8B and Mistral-NeMo:12B.
long-context positional bias phenomena persist with contemporary
LLMs (Subsection 4.1). We then re-examine recent claims of weak
sensitivity to retrieval order and context size in end-to-end RAG,
using matched settings and stable topic sizes (Subsection 4.2).
4.1 Reproducing Context Position Effects
We study here whether reported long-context position effects re-
main visible under our controlled evaluation setting and with con-
temporary LLMs. Thelost in the middlephenomenon describes
a U-shaped trend: performance is highest when answer-bearing
evidence appears near the beginning or end of the context, and
drops when the same evidence is placed in the middle [ 12]. Sub-
sequent work,lost but not only in the middle[ 22], argues that this
degradation is not confined to the exact middle and may manifest
at multiple positions depending on the prompting and document
layout. We revisit both claims using our fixed topic sizes to reduce
topic-sampling noise. In these two works, we report accuracy to
match the evaluation metric used in both original studies.
1) Reproducing lost in the middle[ 12].We follow the original
study and evaluate on the same dataset, AmbigQA [ 14]. Reproduc-
ing the exact protocol is challenging because key dataset handling
details are not fully specified in the original paper, and AmbigQA
does not provide passage-level annotations that directly support
constructing gold-and-distractor contexts in the same way as other
standard QA benchmarks. To operationalise the setting, we treat
AmbigQA’s reference documents as the gold sources and sample
distractor passages from the Natural Questions corpus [ 8]. Con-
cretely, we build each context by inserting the gold document into
a pool of distractors and sweeping its position across the input.
Figure 3 reports results with LLaMA-3.1:8B and Mistral-NeMo:12B,
where the x-axis represents the gold documents’ position, and the
y-axis the accuracy obtained.
From Figure 3, we observe two main differences from the origi-
nally reported behaviour. First, we do not recover a clear U-shaped
curve: accuracy remains comparatively flat across positions, with
only a slight upward trend. Second, absolute scores differ from those
reported in [ 12], which is consistent with small but consequential
differences in dataset processing and document construction (the

Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
(a) NQ Top-5
 (b) NQ Top-10
(c) HotpotQA Top-8
 (d) HotpotQA Top-13
Figure 4:Lost but not only in the middlereproduction [ 22] on standard NQ and HotpotQA [ 8,20] using LLaMA-3.1:8B and
Mistral-Nemo:12B. For HotpotQA, we evaluate Top-8 and Top-13 instead of standard counts to accommodate the additional
context space required for appending all relevant passages, as nearly every query in the dataset contains multiple gold passages.
original paper notes using additional support from the dataset au-
thors). Together, these results suggest that the originally observed
position effect is not straightforwardly reproducible, considering
our setting and a modern LLM.
2) Reproducing lost but not only in the middle[ 22].We next
reproduce the study of Yu et al. [ 22]. The original experiments were
conducted on KILT (formatted versions of NQ and HotpotQA). In
our setting, we instead use their standard releases [ 8,20], which are
the default format in most QA and RAG pipelines. Moreover, both
AmbigQA (in the previous experiment) and HotpotQA can contain
multiple relevant documents/passages. Since prior work does not
specify how multi-evidence cases are positioned, we adopted here
a simple convention: the x-axis position 𝑖indicates where the first
relevant document/passage is placed, and any additional relevant
documents are inserted immediately after it. This ensures keeping
the evidence grouped and avoids mixing evidence with distractors
in a way that would confound the positional sweep. We run the po-
sitional sweep protocol using LLaMA-3.1:8B and Mistral-Nemo:12B.Figure 4 shows that performance is again nearly flat across place-
ments, providing little evidence of degradation at specific positions
for modern models under our setup.
4.2 Reproducing Order and Size Robustness
Under Retrieval
Both prior studies reproduced above deliberately guarantee that the
gold evidence is present in the context and vary only its location
with a synthetic distractor set. While useful for isolating positional
sensitivity, they are optimistic relative to real-world RAG scenarios,
where the retriever may fail to return all gold documents, and
relevance is graded rather than binary. For this reason, we now
shift to a more realistic setting, in which context order and size
emerge from imperfect retrieval. Our goal here is to reproduce
recent claims that RAG performance isweakly sensitiveto both
(i) the number of retrieved documents and (ii) the order in which
those documents are concatenated into the prompt.
Target study.In a recent industry paper, Cao et al. [ 2] evaluate
retrieval robustnessin practical RAG settings using a benchmark of
1,500 open-domain questions (500 each from NQ, HotpotQA, and

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Jorge Gabín, Anxo Perez, and Javier Parapar
(a) HotpotQA (500 topics)
(b) NQ (500 topics)
Figure 5: Context ordering and size effects using LLaMA-
3.1:8B and 500 random topics. Results differ from those in
[2], suggesting that context ordering influences RAG perfor-
mance more than previously observed.
ASQA) with Wikipedia retrieval. They vary the retrieval depth 𝑘∈
{5,10,25,50,75,100}and compare three ordering schemes applied
to the same top- 𝑘set:original rank,reversed rank, andrandom
shuffle. Their main conclusion is that average performance is often
relatively stable across orderings and retrieval depths, even though
they also report residualsample-leveltrade-offs. Importantly, they
evaluate correctness using an LLM-as-a-judge (LLaMA-3.3:70B)
rather than string-match style metrics (F1-token).
Our reproduction setup.To mirror this protocol as closely as
possible while keeping our analysis aligned with the rest of the
paper, we focus on the two datasets that we use throughout: NQ
and HotpotQA. We evaluate LLaMA-3.1:8B under the same 𝑘grid
and the same three ordering schemes (original,reversed,random) on
500 randomly sampled topics per dataset, matching the per-dataset
topic budget used by Cao et al. [ 2]. As in the target study, each
condition uses the same retrieved top- 𝑘set; only the concatenation
order changes.While Cao et al. [ 2] rely on an LLM-based judge, we report
token-level F1 as our primary metric. To verify that this choice
does not change the qualitative ordering conclusions, we ran a
pilot experiment on a subset of 𝑛=250 topics from NQ and Hot-
potQA, evaluating the same conditions with both F1 and an LLM
judge. Across all ( 𝑡𝑜𝑝-𝑘,order ) settings, the two scorers exhibited
near-identical ordering gaps, indicating that the relative separa-
tion between ordering schemes is essentially unchanged. Given
this agreement, and since F1 is the official benchmark metric and
more reproducible, we use F1 throughout for reproducibility and
comparability across sections.
Figure 5 shows the reproduced curves for NQ and HotpotQA
(500 topics each), plotting the three ordering strategies, with the
x-axis showing the context size (number of retrieved passages,
𝑘∈{ 5,10,25,50,75,100}) and the y-axis the F1-score. In contrast to
the strong average stability reported in [ 2], we observe measurable
sensitivity to both context size and ordering, with differences that
are small at low 𝑘but become more apparent as 𝑘grows. Even
under a matched configuration (same datasets, model family, 500-
topic samples, and the three orderings), reproducing the original
average stability trend proves difficult.
Looking at these results, we can see that the more fundamen-
tal issue is that both settings operate with relatively small topic
samples. When per-query variance is high, and the marginal gains
between orders are modest, conclusions about “order robustness”
can depend strongly on which topics are sampled. This motivates
our controlled evaluation framework (Section 3). In the next section,
we adopt the calibrated topic sizes identified earlier and re-evaluate
ordering and context-size effects under stable topic sets, so that
subsequent comparisons across models, retrievers, and rerankers
are not confounded by topic-sampling noise.
5 Controlled RAG Analysis: Ordering, Depth,
Retrieval Quality, and Models
Having established a stable evaluation protocol and revisited key
prior claims, we now conduct targeted analyses to characterise
when and why context size and ordering matter in practical RAG
pipelines. Unlike idealised position-sweep settings, the evidence
available to the generator is mediated by imperfect retrieval, and
the marginal utility of adding more context depends on both the
ranking quality and how the model allocates attention across long
inputs. Using the fixed topic sizes for NQ and HotpotQA identified
in Section 3, we study four research questions:
•RQ1.How do different document ordering strategies and
context sizes affect QA performance? (Subsection 5.1)
•RQ2.How much performance is attributable to the LLM
versus the retrieved evidence quality, as estimated by closed-
book and oracle contexts? (Subsection 5.2)
•RQ3.How does retrieval quality (BM25 alone versus BM25
with dense reranking) interact with context ordering and
size? (Subsection 5.2)
•RQ4.How do model family and scale influence sensitivity
to ordering and context size? (Subsection 5.3)

Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
(a) HotpotQA (1000 topics)
(b) NQ (2000 topics)
Figure 6: Effect of document ordering and context size on
model performance for HotpotQA and NQ. Results show the
impact of different ordering strategies and context size.
5.1 Exploring Context Size and Ordering Effects
(RQ1)
This experiment investigates how context size and document or-
dering influence RAG performance under the controlled setup de-
fined earlier ( 𝑛= 1000topics for HotpotQA and 𝑛= 2000for
NQ). We vary two parameters: 𝑖) the number of retrieved passages
and 𝑖𝑖) their ordering within the prompt to examine their com-
bined effect on information utilisation and answer accuracy. We
again compare the three ordering schemes:standard,reverse, and
random. Each strategy is evaluated across context sizes (ranging
from 𝑘∈{ 5,10,25,50,75,100}), allowing us to assess the trade-off
between context length and retrieval order.
Figure 6 shows the results for this experiment, plotting F1 on
the y-axis against the context size on the x-axis. The results display
a dataset-dependent behaviour. On HotpotQA, model performance
is more sensitive to ordering: the reverse scheme increasingly out-
performs other strategies at larger contexts, indicating that placing
stronger passages later in the prompt helps counter position sensi-
tivity in multi-hop settings. In contrast, on NQ, performance riseswith 𝑘and remains comparatively order-stable, suggesting that
single-hop questions benefit primarily from additional recall rather
than from a particular order.
Examining the overall trends and addressing RQ1, we demon-
strate that the effectiveness of additional evidence depends on its
positioning in the prompt in multi-hop settings. Additional pas-
sages help only when high-value passages are placed in positions
the model prefers. These observations are consistent with prior
work [ 6,22]. On single-hop QA, multi-passage generative readers
such as Fusion-in-Decoder (FiD) show that supplying more rele-
vant passages generally improves accuracy [ 6]. Conversely, when
contexts become long, order-aware layouts such as OP-RAG [ 22],
which preserve intra-document chunk order instead of relevance,
improve answer quality and exhibit an inverted-U relationship
between performance and context size [22].
5.2 Impact of Retrieval Quality and Reranking
(RQ2 and RQ3)
RQ2.To quantify how much performance derives from the re-
trieved context versus the LLM itself, we propose and evaluate a
wide range of configurations that progressively increase evidence
quality while holding the LLM fixed (LLaMA-3.1:8B). In this analy-
sis, we focus on HotpotQA because it providesgold passageandgold
sentenceannotations for supporting evidence, which are required
to construct oracle contexts and to separate passage-level from
sentence-level upper bounds. We evaluate the next configurations:
•Closed-book (no retrieval): the model answers without
any external passages, our lower bound on knowledge and
reasoning without context.
•BM25 retrieval: standard top- 𝑘BM25, evaluated under the
three prompt orders to capture order sensitivity: standard,
reverse, and random.
•Oracle-passages:an upper-bound selection that includes
all gold-relevant passages as context.
•Oracle-sents:a stricter upper bound that includes only the
exact sentences from the gold-relevant passages that contain
the facts needed to answer.
•Oracle-passages+BM25 (standard): a strategy that places
first all gold-relevant passages, then fills remaining context
slots by using the top non-relevant BM25-ranked passages
in the standard ranking order up to size𝑘.
•Oracle-passages+BM25 (reverse): places the BM25-selected
non-relevant passages first, in ascending BM25 score order,
and appends the gold-relevant passages at the end.
Figure 7 reports F1 on the y-axis versus context size ( 𝑘) on the
x-axis for HotpotQA across the eight configurations. Looking at the
results, two main trends emerge. First, the closed-book baseline is
the weakest, underscoring that model base knowledge is insufficient
for this multi-hop setting. Second, both oracle-passages (all gold-
relevant passages) and oracle-sents (only the sentences containing
the answer) form the upper bound, clustering a bit above 0.70 F1
across 𝑘. This isolates the upper bound attributable to evidence
quality, not the LLM capacity. Within BM25 retrieval, the three
ordering schemes start close together at small 𝑘, then diverge mildly
as𝑘grows: reverse pulls ahead of standard and random for larger
contexts, and all three curves stabilise after 50 passages. This aligns

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Jorge Gabín, Anxo Perez, and Javier Parapar
Figure 7: Comparison of closed-book, standard retrieval
(BM25), and oracle ranking results on HotpotQA. The oracle
ranking illustrates the upper bound achievable with a perfect
ordering, showing the relative contributions of the LLM and
the retrieval component to overall RAG performance.
with prior position effects: positioning the key passages at the end
is beneficial in long contexts. Moreover, the gap between BM25 and
oracle results quantifies the remaining potential for retrieval and
ordering optimisation.
In Figure 7, we can also see that the oracle with BM25-noise in-
troduction conditions further differentiates ordering from evidence
quality. For Oracle+BM25 (standard), performance decreases as 𝑘
increases: once the oracle content is present, adding more BM25
material in standard rank adds noise and displaces the crucial evi-
dence from border positions. In contrast, Oracle+BM25 (reverse)
consistently dominates its standard counterpart and remains the
better approach for all 𝑘≥ 10, suggesting that even when gold
passages are present, placing them at the tail better aligns with the
model’s reasoning.
RQ3.We now study how retrieval quality interacts with order-
ing and context size. Beyond BM25, we add a reranking stage that
first retrieves a candidate pool with BM25 and then re-scores the
top 100 of those candidates using E5 embeddings [ 18] with cosine
similarity, finally presenting the top 𝑘to the LLM. We choose E5
because it is a strong, open sentence-embedding model trained
for retrieval-style objectives, offers solid zero-shot performance
across domains, is efficient for large candidate pools, and improves
reproducibility [18].
Figure 8 plots F1 (y-axis) against context size 𝑘(x-axis) for BM25
and reranking (BM25+E5), each under the three ordering schemes
(standard, reverse, random). In these new results, with reranking,
small contexts ( 𝑘∈{ 5,10}) achieve noticeably higher performance
than BM25 alone on both datasets, indicating that higher-quality
top-𝑘reduces dependency on long prompts. As 𝑘grows, HotpotQA
shows a decline under reranking, as lower-quality passages added
(a) HotpotQA (1000 topics)
(b) NQ (2000 topics)
Figure 8: Impact of retrieval model on RAG performance. Re-
sults illustrate the performance of BM25 and dense rerank-
ing, showing how improved retrieval interacts with ordering
strategies, influencing the consistency of generated answers.
to the end of the prompt dilute the multi-hop structure. However,
NQ performance as 𝑘grows remains roughly flat. For long contexts
(𝑘≥ 50), curves under BM25+E5 tend to approach the BM25
levels, and the gaps between the three order strategies narrow. This
suggests that better evidence quality decreases order sensitivity
and that further gains are limited by prompt length rather than
ranking.
Taken together, these trends indicate that stronger retrieval
favours shorter contexts: when the first few passages are high
quality, adding more tends to offer diminishing or negative returns
(especially on multi-hop), and the specific prompt order matters
less. Conversely, under weaker retrieval, longer contexts and their
order remain consequential. This quantifies the interplay in RQ4:
improvements from reranking reduce the need for long contexts
and reduce ordering effects, whereas BM25 alone benefits more
from reverse ordering at higher𝑘.

Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
5.3 Trends Across Model Sizes and
Architectures (RQ4)
All prior experiments use LLaMA-3.1 8B, which stands as a strong,
widely adopted baseline that offers competitive QA performance
at manageable cost. We now test whether ordering and context-
size effects generalise across different LLM families and scales. To
summarise our study without overcrowding plots, for all the models,
we report ΔF1=F1reverse−F1standard for the different context sizes.
This highlights the marginal effect of reversing the order at each
context size and facilitates cross-model comparison. Figure 9 shows
the results difference of reverse and standard ordering strategies for
HotpotQA (Figure 9a) and NQ (Figure 9b) across all architectures
and sizes. Looking at the overall results, different key patterns
emerge.
First, robustness to document ordering is strongly dataset depen-
dent. In HotpotQA, Figure 9a shows a consistent increase in ΔF1
with 𝑘for most models: reversing the order helps more as contexts
become longer, in line with the position effects observed earlier.
Larger models (e.g., LLaMA-3.1:70B) tend to have ΔF1 values that
are closer to zero and smoother, indicating reduced sensitivity. Mid-
sized models (e.g., LLaMA-3.1:8B, Mistral-Nemo:12B) exhibit larger
positive oscillations at higher 𝑘. Interestingly, the smallest model
(Gemma-3:4B) remains relatively stable around small positive val-
ues. For NQ (Figure 9b), ΔF1 is generally small near zero across
𝑘, reflecting the order stability seen in the main results. Smaller
models (e.g., Gemma-3:4B) can now show more variability, while
larger models produce lower values. This suggests that, for predom-
inantly single-hop queries, increasing evidence quantity dominates
ordering, and capacity mainly reduces variance rather than flipping
the sign of the effect.
Second, architectural design and model capacity affect this sen-
sitivity but not uniformly. Larger models show lower variance
(smoother ΔF1) across both datasets, consistent with better context
integration. However, they do not eliminate order effects in multi-
hop settings. Regarding model architectures, we can see that across
LLaMA-3.1, Mistral-Nemo, and Gemma-3, the mid-sized variants
show larger positive ΔF1 on HotpotQA at higher 𝑘(reverse helps),
whereas their larger models are less sensitive. On NQ, all three fam-
ilies are comparatively order-stable across k. These observations
imply that ordering sensitivity arises from an interaction between
dataset properties and how each architecture manages long-context
reasoning, rather than from model size alone.
Finally, we analyse overall performance, setting aside the oracle
model, which unsurprisingly achieves the highest scores but de-
clines as more BM25 noise is introduced. We observe that the larger
models, LLaMA-3.1:70B and Gemma-3:27B, consistently achieve
the best F1 results across both datasets. Their performance remains
stable across different topic sizes, indicating that increased model
capacity contributes not only to improved accuracy but also to
greater robustness against fluctuations in context size.
6 Conclusions
In this work, we revisit how context size and evidence ordering
shape QA performance in RAG under a controlled, reproducible
evaluation framework. We show that topic sampling is a major
source of variance: conclusions about ordering and context size can
(a) HotpotQA (1000 topics)
(b) NQ (2000 topics)
Figure 9: Trends in model robustness to context ordering
across different architectures and sizes. The left y-axis and
bars show the variations in F1 ( ΔF1) between reverse and stan-
dard ordering, while the right y-axis and lollipops indicate
the best F1 achieved by each model, regardless of ordering.
shift when experiments are run on small topic subsets. Using Hot-
potQA and Natural Questions, we introduce a practical calibration
procedure based on repeated subset sampling and fix topic sizes that
yield stable trends at a feasible cost. Under this controlled setting,
several patterns emerge. On single-hop NQ, performance generally
improves as 𝑘increases and is comparatively insensitive to order.
On multi-hop HotpotQA, larger contexts help mainly when high-
value evidence is placed in positions the model preferentially uses.
Dense reranking also boosts small- 𝑘performance and narrows or-
dering sensitivity. Larger models are typically more stable overall,

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Jorge Gabín, Anxo Perez, and Javier Parapar
but they do not eliminate ordering effects in multi-hop settings. Al-
together, reliable evaluation is essential for interpreting order/size
effects, and practitioners should prioritise retrieval quality and evi-
dence placement over simply increasing context size. We release
all code and configurations to support reproducible, order-aware
RAG evaluation.
Computational resources
Experiments were conducted using a private infrastructure, which
has a carbon efficiency of 0.432 kgCO 2eq/kWh. A cumulative of 160
hours of computation was performed on hardware of type A100
SXM4 80 GB (TDP of 400W), generating an estimated 22.6 kgCO 2eq,
with 0% directly offset [9].
Acknowledgments
All authors acknowledge funding from the Ministry of Science,
Innovation and Universities of the Government of Spain (project
PID2022-137061OB-C21, MCIN/AEI/10.13039/501100011033), as well
as from the Department of Education, Science, Universities, and
Vocational Training of the Xunta de Galicia (grant GRC ED431C
2025/49). CITIC, as a center accredited for excellence within the
Galician University System and a member of the CIGUS Network,
receives subsidies from the Department of Education, Science, Uni-
versities, and Vocational Training of the Xunta de Galicia. Addition-
ally, it is co-financed by the EU through the FEDER Galicia 2021-27
operational program (Ref. ED431G 2023/01).
References
[1]Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The
Distracting Effect: Understanding Irrelevant Passages in RAG. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Moham-
mad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna,
Austria, 18228–18258. doi:10.18653/v1/2025.acl-long.892
[2]Shuyang Cao, Karthik Radhakrishnan, David Rosenberg, Steven Lu, Pengxiang
Cheng, Lu Wang, and Shiyue Zhang. 2025. Evaluating the Retrieval Robustness
of Large Language Models. (05 2025). doi:10.48550/arXiv.2505.21870
[3]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[4]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: retrieval-augmented language model pre-training. InProceedings
of the 37th International Conference on Machine Learning (ICML’20). JMLR.org,
Article 368, 10 pages.
[5]Jan Hutter, David Rau, Maarten Marx, and Jaap Kamps. 2025. Lost but Not Only
in the Middle - Positional Bias in Retrieval Augmented Generation. InAdvances
in Information Retrieval - 47th European Conference on Information Retrieval, ECIR
2025, Lucca, Italy, April 6-10, 2025, Proceedings, Part I (Lecture Notes in Computer
Science, Vol. 15572), Claudia Hauff, Craig Macdonald, Dietmar Jannach, Gabriella
Kazai, Franco Maria Nardini, Fabio Pinelli, Fabrizio Silvestri, and Nicola Tonellotto
(Eds.). Springer, 247–261. doi:10.1007/978-3-031-88708-6_16
[6]Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InProceedings of the
16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume, Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty (Eds.).
Association for Computational Linguistics, Online, 874–880. doi:10.18653/v1/
2021.eacl-main.74
[7]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn,Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online,
6769–6781. doi:10.18653/v1/2020.emnlp-main.550
[8]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. doi:10.1162/tacl_a_00276
[9]Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres.
2019. Quantifying the Carbon Emissions of Machine Learning.arXiv preprint
arXiv:1910.09700(2019).
[10] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[11] Oscar Lithgow-Serrano, David Kletz, Vani Kanjirangat, David Adametz, Marzio
Lunghi, Claudio Bonesana, Matilde Tristany-Farinha, Yuntao Li, Detlef Rep-
plinger, Marco Pierbattista, Stefania Stan, and Oleg Szehr. 2025. Assessing RAG
System Capabilities on Financial Documents. InProceedings of The 10th Work-
shop on Financial Technology and Natural Language Processing, Chung-Chi Chen,
Genta Indra Winata, Stephen Rawls, Anirban Das, Hsin-Hsi Chen, and Hiroya
Takamura (Eds.). Association for Computational Linguistics, Suzhou, China,
124–147. doi:10.18653/v1/2025.finnlp-2.9
[12] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[13] Microsoft. 2025.RAG and the Future of Intelligent Enterprise Applications.
White Paper. Microsoft. https://cdn-dynmedia-1.microsoft.com/is/content/
microsoftcorp/microsoft/final/en-us/microsoft-product-and-services/March-
2025-rag-and-the-future-of-intelligent-enterprise-applications.pdf
[14] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020.
AmbigQA: Answering Ambiguous Open-domain Questions. InProceedings of the
2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
5783–5797.
[15] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
ArXivabs/1901.04085 (2019). https://api.semanticscholar.org/CorpusID:58004692
[16] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Found. Trends Inf. Retr.3, 4 (April 2009), 333–389.
doi:10.1561/1500000019
[17] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2025. Is Relevance
Propagated from Retriever to Generator in RAG?. InAdvances in Information
Retrieval - 47th European Conference on Information Retrieval, ECIR 2025, Lucca,
Italy, April 6-10, 2025, Proceedings, Part I (Lecture Notes in Computer Science,
Vol. 15572), Claudia Hauff, Craig Macdonald, Dietmar Jannach, Gabriella Kazai,
Franco Maria Nardini, Fabio Pinelli, Fabrizio Silvestri, and Nicola Tonellotto
(Eds.). Springer, 32–48. doi:10.1007/978-3-031-88708-6_3
[18] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and
Furu Wei. 2024. Multilingual e5 text embeddings: A technical report.arXiv
preprint arXiv:2402.05672(2024).
[19] Anbang Xu, Tan Yu, Min Du, Pritam Gundecha, Yufan Guo, Xinliang Zhu, May
Wang, Ping Li, and Xinyun Chen. 2024. Generative AI and Retrieval-Augmented
Generation (RAG) Systems for Enterprise. InProceedings of the 33rd ACM In-
ternational Conference on Information and Knowledge Management(Boise, ID,
USA)(CIKM ’24). Association for Computing Machinery, New York, NY, USA,
5599–5602. doi:10.1145/3627673.3680117
[20] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[21] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2025. Eval-
uation of Retrieval-Augmented Generation: A Survey. InBig Data, Wenwu Zhu,
Hui Xiong, Xiuzhen Cheng, Lizhen Cui, Zhicheng Dou, Junyu Dong, Shanchen
Pang, Li Wang, Lanju Kong, and Zhenxiang Chen (Eds.). Springer Nature Singa-
pore, Singapore, 102–120.
[22] Tan Yu, Anbang Xu, and Rama Akkiraju. 2024. In Defense of RAG in the Era
of Long-Context Language Models.ArXivabs/2409.01666 (2024). https://api.
semanticscholar.org/CorpusID:272368207
# Masking or Mitigating? Deconstructing the Impact of Query Rewriting on Retriever Biases in RAG

**Authors**: Agam Goyal, Koyel Mukherjee, Apoorv Saxena, Anirudh Phukan, Eshwar Chandrasekharan, Hari Sundaram

**Published**: 2026-04-07 17:10:15

**PDF URL**: [https://arxiv.org/pdf/2604.06097v1](https://arxiv.org/pdf/2604.06097v1)

## Abstract
Dense retrievers in retrieval-augmented generation (RAG) systems exhibit systematic biases -- including brevity, position, literal matching, and repetition biases -- that can compromise retrieval quality. Query rewriting techniques are now standard in RAG pipelines, yet their impact on these biases remains unexplored. We present the first systematic study of how query enhancement techniques affect dense retrieval biases, evaluating five methods across six retrievers. Our findings reveal that simple LLM-based rewriting achieves the strongest aggregate bias reduction (54\%), yet fails under adversarial conditions where multiple biases combine. Mechanistic analysis uncovers two distinct mechanisms: simple rewriting reduces bias through increased score variance, while pseudo-document generation methods achieve reduction through genuine decorrelation from bias-inducing features. However, no technique uniformly addresses all biases, and effects vary substantially across retrievers. Our results provide practical guidance for selecting query enhancement strategies based on specific bias vulnerabilities. More broadly, we establish a taxonomy distinguishing query-document interaction biases from document encoding biases, clarifying the limits of query-side interventions for debiasing RAG systems.

## Full Text


<!-- PDF content starts -->

Masking or Mitigating?
Deconstructing the Impact of Query Rewriting on Retriever Biases in RAG
Agam Goyal♠,∗,‡, Koyel Mukherjee♣,‡, Apoorv Saxena♡, Anirudh Phukan♢,
Eshwar Chandrasekharan♠, Hari Sundaram♠
♠Siebel School of Computing and Data Science, University of Illinois Urbana-Champaign
♣Adobe Research♡Inception Labs♢Indian Institute of Science
{agamg2, eshwar, hs1}@illinois.edu, komukher@adobe.com
Abstract
Dense retrievers in retrieval-augmented genera-
tion (RAG) systems exhibit systematic biases—
including brevity, position, literal matching,
and repetition biases—that can compromise re-
trieval quality. Query rewriting techniques are
now standard in RAG pipelines, yet their im-
pact on these biases remains unexplored. We
present the first systematic study of how query
enhancement techniques affect dense retrieval
biases, evaluating five methods across six re-
trievers. Our findings reveal that simple LLM-
based rewriting achieves the strongest aggre-
gate bias reduction (54%), yet fails under ad-
versarial conditions where multiple biases com-
bine. Mechanistic analysis uncovers two dis-
tinct mechanisms: simple rewriting reduces
bias through increased score variance, while
pseudo-document generation methods achieve
reduction through genuine decorrelation from
bias-inducing features. However, no technique
uniformly addresses all biases, and effects vary
substantially across retrievers. Our results pro-
vide practical guidance for selecting query en-
hancement strategies based on specific bias vul-
nerabilities. More broadly, we establish a tax-
onomy distinguishing query-document inter-
action biases from document encoding biases,
clarifying the limits of query-side interventions
for debiasing RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) systems
have become foundational for grounding large lan-
guage models in external knowledge, enabling ap-
plications from question-answering to document
analysis across diverse domains (Lewis et al., 2020;
Lu et al., 2022; Jiang et al., 2023; Gao et al., 2023b;
Asai et al., 2024). These systems typically employ
dense retrievers (Karpukhin et al., 2020; Thakur
et al., 2021) to identify relevant passages from doc-
∗Work done during internship at Adobe Research
‡Corresponding Authors
Figure 1:Paper Overview.Dense retrievers are suscep-
tible to various biases, leading to incorrect retrieval and
erroneous LLM outputs. Can query rewriting help? We
systematically evaluate its effects on retrieval bias.
ument collections, which are then provided as con-
text for response generation (Oche et al., 2025).
Despite their effectiveness, recent work has re-
vealed that dense retrievers exhibit systematic bi-
ases that can compromise retrieval quality (Ram
et al., 2023; Coelho et al., 2024). Fayyaz et al.
(2025) demonstrated that retrievers consistently fa-
vor documents with surface-level characteristics—
brevity, early answer positioning, lexical overlap,
and entity repetition—over documents containing
actual factual evidence. The ideal solution would
be dense retrieval models inherently robust to such
biases and capable of prioritizing semantic rele-
vance. However, achieving this through architec-
tural modifications or training objective changes
remains challenging. A more tractable first step is
understanding whether these biases persist under
query enhancement techniques (Ma et al., 2023),
which are standard in modern RAG pipelines.arXiv:2604.06097v1  [cs.IR]  7 Apr 2026

Production RAG systems routinely employ
query transformations including LLM-based rewrit-
ing for better semantic matching and pseudo-
document generation approaches like HyDE (Gao
et al., 2023a) and Query2Doc (Wang et al., 2023).
While these techniques improve retrieval effective-
ness, their interaction with systematic biases re-
mains unexplored: do they amplify existing biases
by generating surface-level features that retrievers
favor, or do specific strategies mitigate particular
bias types? Understanding this interaction is crit-
ical for deploying robust and fair RAG systems,
particularly in high-stakes domains such as medi-
cal, sociotechnical, educational, and enterprise set-
tings (Ryan et al., 2025; Wang et al., 2025; Goyal
et al., 2025; Shimgekar et al., 2025; Zhan et al.,
2025; Goyal et al., 2026; Pal et al., 2026).
In this work,we conduct the first systematic
investigation into how query rewriting tech-
niques affect retrieval biases. We evaluate five
enhancement methods across six retrievers using
the controlled framework of Fayyaz et al. (2025).
This framework’s treatment-control methodology,
which uses paired documents that differ only in
specific surface characteristics while maintaining
factual equivalence, enables us to isolate and mea-
sure the effect of query rewriting on individual bias
types rather than confounding bias reduction with
overall retrieval improvements. We find that query
enhancement effects are highly differential: simple
LLM rewriting achieves the strongest aggregate
bias reduction (54%) but fails under adversarial
conditions, while pseudo-document methods pro-
vide more modest but mechanistically robust im-
provements. We establish a taxonomy distinguish-
inginteraction biasesamenable to query-side inter-
ventions fromencoding biasesrequiring retriever-
level modifications, providing practical guidance
for bias-aware RAG deployment.
2 Related Works
Bias in Dense Retrieval.Dense retrieval has
emerged as a powerful alternative to traditional
sparse methods like BM25 for information retrieval
tasks (Karpukhin et al., 2020; Thakur et al., 2021),
showing strong performance on retrieval bench-
marks. However, despite these advances, recent
work has begun to uncover systematic biases in
how dense retrievers encode and rank documents,
including position biases (Coelho et al., 2024)
and lexical biases (Ram et al., 2023). Similarly,BehnamGhader et al. (2023) show that Dense Pas-
sage Retrieval (DPR) models fail to retrieve state-
ments requiring reasoning beyond surface-level
similarity. In terms of downstream consequences of
these biases, Huang et al. (2026) and Samuel et al.
(2026) show how these biases can affect the cover-
age, presentation, and fairness of the outputs from
generative search systems. Most relevant to our
work, Fayyaz et al. (2025) introduced ColDER, a
controlled benchmark for systematically measuring
various types of retrieval biases that biases persist
across multiple dense retriever architectures and
can compound adversarially.Our work extends
this line of research by investigating whether and
how query-side interventions can mitigate these
documented biases.
Query Enhancement in RAG.RAG sys-
tems (Lewis et al., 2020; Gao et al., 2023b) have
become foundational for knowledge-intensive NLP
tasks, with query rewriting now a standard pipeline
component (Ma et al., 2023). Several query
enhancement techniques have been proposed to
bridge the semantic gap between queries and docu-
ments. HyDE (Gao et al., 2023a) generates pseudo-
documents and retrieves based on their embeddings
rather than the original query; Query2Doc (Wang
et al., 2023) concatenates LLM-generated pseudo-
documents to expand queries; and various other
simple LLM-based query rewriting techniques have
been proposed in prior work (Jagerman et al., 2023;
Mao et al., 2024; Kim et al., 2025). While these
methods improve retrieval effectiveness, their im-
pact on systematic retrieval biases remains un-
explored. As RAG systems are increasingly de-
ployed in high-stakes domains, understanding how
retrieval biases propagate through these pipelines
becomes critical.Our work fills this gap by ana-
lyzing how query enhancement techniques interact
with known retrieval biases.
3 Preliminaries, Data, and Methods
3.1 Problem Formulation
We investigate whether and how query rewriting
techniques affect systematic biases in dense retriev-
ers. We adopt the treatment-control ColDeR bench-
mark contributed by Fayyaz et al. (2025), which
uses carefully constructed document pairs using
the Re-DocRED dataset (Tan et al., 2022) that dif-
fer only in specific surface characteristics while
maintaining factual equivalence.

Given a query qand a pair of documents
(D1, D2)where D1contains a bias-inducing char-
acteristic and D2serves as the control, we measure
retrieval preferences using the similarity scores as-
signed by a dense retriever S. This controlled de-
sign enables us to isolate the effect of query rewrit-
ing on individual bias types, separating bias miti-
gation from general retrieval performance improve-
ments. Then, for each document pair, we compute
the difference in retrieval scores:
δ=S(q, D 1)− S(q, D 2)
where S(q, D) represents the retriever’s similarity
score between the generated embeddings for the
query qand document D. Positive δvalues indi-
cate bias toward surface-level characteristics, while
values near zero suggest factual grounding. We
apply different query enhancement techniques to
transform the original query qintoq′, and mea-
sure how this transformation affects the distribu-
tion of δacross query-document pairs. For each
bias type, we collect N= 200 paired documents
{(D 1, D2)}N
ialong with associated queries. We
now describe each bias type in detail.
3.2 Bias Types
We evaluate four systematic biases identified by
Fayyaz et al. (2025) in dense retrievers:
(i)Brevity Bias:Retrievers prefer shorter docu-
ments over longer ones containing equivalent
evidence. Document D1contains the same
factual information as D2but with signifi-
cantly fewer tokens.
(ii)Literal Bias:Retrievers prioritize exact lexi-
cal matches over semantic equivalence. Doc-
ument D1uses identical or highly overlap-
ping terminology with the query, while D2
expresses the same information using para-
phrases or synonyms.
(iii) Position Bias:Retrievers favor content ap-
pearing early in documents over information
at later positions. In document pairs, D1
places the relevant information at the begin-
ning whileD 2positions it toward the end.
(iv) Repetition Bias:Retrievers prefer documents
with repeated entity mentions. Document D1
contains multiple mentions of key entities,
while D2mentions them fewer times despite
containing equivalent factual content.
We refer the reader to Fayyaz et al. (2025) for
rigorous definitions of these bias types.3.3 Dense Retrievers
We evaluate six state-of-the-art dense retrievers
representing diverse architectural approaches:
(1) Contrastive Bi-Encoders:Contriever-
MSMARCO (Izacard et al., 2021), COCO-DR
Base MSMARCO (Yu et al., 2022), Dragon
RoBERTa (Lin et al., 2023), DRAGON+ (Lin
et al., 2023)
(2) Generative Pretraining:RetroMAE MS-
MARCO FT (Xiao et al., 2022)
(3) Token-Level Interaction:ColBERTv2 (San-
thanam et al., 2022)
This diverse set represents the major paradigms
in modern dense retrieval—contrastive learning,
diverse data augmentation, and late interaction
architectures—enabling us to assess whether query
rewriting effects on biases are architecture-specific
or systemic across different retrieval approaches.
3.4 Query Enhancement Techniques
We evaluate four query enhancement approaches
that represent the spectrum of techniques used in
modern RAG systems.
(1) LLM-based Query Rewriting:We use an
LLM to reformulate the original query for better
semantic matching. Given query q, we prompt the
model to produce the rewritten queryq rewrite .
(2) HyDE:Following Gao et al. (2023a), we
leverage the Hypothetical Document Embeddings
(HyDE) approach, which uses an LLM to generate
pseudo-documents that might hypothetically con-
tain the answer to a query rather than using the
query directly for retrieval. For a given query q, we
prompt a language model Mto generate a pseudo-
document ˆd=M(prompt(q)) where prompt(q)
instructs the model to generate a passage that would
contain the answer to query q. The generated hypo-
thetical document ˆdis then embedded and used as
the query representation for retrieval. The key in-
sight of HyDE is that hypothetical documents often
provide richer semantic context than short queries,
potentially capturing vocabulary, style, and rela-
tional patterns that better align with actual target
documents.
(3) Query2Doc (Q2D):We adopt the Query2Doc
approach introduced by Wang et al. (2023), which
augments queries with LLM-generated pseudo-
documents rather than replacing them entirely. For
a given query q, we prompt a language model M
to generate a pseudo-document ˆdthat might be
relevant to the query. This generated passage is

then concatenated with the original query to form
an expanded query representation ˆqQ2D=q⊕ ˆd
where ⊕denotes concatenation. The key distinc-
tion from HyDE is that Query2Doc preserves the
original query signal while enriching it with gen-
erated context. Since our retrieval scenario is in-
herently zero-shot, we do not utilize few-shot ex-
emplars for generation of pseudo-documents like
the original Query2Doc paradigm. This augmented
query qQ2Dis then embedded and used for retrieval.
In line with recent efforts toward domain-
specific LLMs in RAG systems (Zhang et al.,
2024), we test whether domain adaptation can im-
prove query rewriting for bias mitigation. Specif-
ically, we continually pretrain Mon the origi-
nal Re-DocRED document for a given document
pair from ColDER, before applying HyDE or
Query2Doc. Specifically, we use Low-Rank Adap-
tation (LoRA) (Hu et al., 2022) to adapt the model
to documentdby minimizing:
LCPT=−|d|X
i=1logP θ(wi|w<i, d)
where θrepresents the LoRA parameters and wi
denotes the ithtoken in document d. The hypoth-
esis is that this adapted model Mdomain familiar
with domain-specific vocabulary and factual pat-
terns may generate pseudo-documents that better
align with actual evidence, potentially reducing re-
liance on surface-level features. We use LoRA with
rankr= 8, α= 16 , and train for 10epochs per
document. This gives us our final two techniques:
(4) HyDE with Continual Pretraining (HyDE-
CPT):We use the domain-adapted model Mdomain
to use to generate psuedo-document ˆdCPT =
Mdomain (prompt(q)) following the HyDE prompt,
and then use the embedding of ˆdCPTfor retrieval.
(5) Query2Doc with Continual Pretraining
(Q2D-CPT):We concatenate the original query
to the pseudo-document generated by the domain-
adapted model Mdomain to obtain ˆqQ2D-CPT =
q⊕ˆdCPTand use its embedding for retrieval.
To ensure that the effects we observe are ro-
bust across the LLM used for query enhance-
ment, we test open-source models from two fam-
ily of models: Gemma 3 (Team et al., 2025)
(google/gemma-3-12b-it ) and Qwen3 (Yang
et al., 2025) (Qwen/Qwen3-4B-Instruct-2507).Method Mean|t|Sig. Biases Reduction
Baseline 8.72±5.32 21/24 –
Rewrite 4.02±2.17 13/24 +53.9%
HyDE 6.95±4.73 20/24 +20.3%
HyDE-CPT 6.78±4.79 19/24 +22.3%
Q2D 6.15±5.96 16/24 +29.5%
Q2D-CPT 6.07±5.72 17/24 +30.4%
Table 1: Summary of retrieval bias across query en-
hancement methods. Mean |t|-statistic averaged across
all retrievers and bias types. Sig. Biases: number
of retriever-bias combinations showing significant bias
(p <0.05 after Bonferroni correction). Reduction: per-
centage decrease in|t|compared to Baseline.
3.5 Evaluation Metrics
For each bias type and query enhancement tech-
nique, we measure retrieval preferences using
paired t-tests (Ross and Willson, 2017) on score
differences ∆ = [δ 1, . . . , δ N]across all query-
document pairs. We reportmean |t|-statistic, the
absolute value of the t-statistic measuring the
strength of bias, as well as the percentage decrease
in mean |t|-statistic compared to the vanilla query.
Furthermore, we compute p−values to check for
statistical significance of the bias at α= 0.05 with
Bonferroni correction (Bonferroni, 1935). Lower
|t|-statistic values indicate reduced bias, with val-
ues approaching zero suggesting factual grounding
rather than reliance on surface characteristics.
4 Results
We now present results for query enhancements
using Gemma-3-12B-IT, and defer the results for
Qwen3-4B-Instruct to Appendix B since we ob-
serve very similar trends overall.
4.1 Overall Trend
Table 1 summarizes retrieval bias across all query
enhancement methods, aggregated over six retriev-
ers and four bias types (24 total retriever-bias com-
binations). All query enhancement techniques re-
duce bias relative to the baseline, though the mag-
nitude varies substantially. Most strikingly, simple
LLM-based query rewriting achieves the largest
reduction (53.9%), cutting the mean |t|-statistic
from 8.72 to 4.02 and reducing the number of sta-
tistically significant biases from 21 to 13 out of
24 combinations. The pseudo-document genera-
tion methods—HyDE and Query2Doc—provide
more modest improvements, with reductions rang-
ing from 20.3% to 30.4%. Query2Doc variants out-
perform HyDE variants, suggesting that preserving

the original query signal while augmenting with
generated content is more effective than replacing
the query entirely. Continual pretraining (CPT) of-
fers marginal gains over the base methods (HyDE-
CPT: +2.0% over HyDE; Q2D-CPT: +0.9% over
Q2D), indicating that domain adaptation primarily
improves factual grounding rather than bias miti-
gation. However, these aggregate statistics mask
important variation across bias types that we exam-
ine in detail below.
4.2 Reduction in Retrieval Bias
Figure 2 presents the mean |t|-statistic across bias
types and query enhancement methods, averaged
over all retrievers. We observe several patterns:
Simple rewriting is surprisingly effective.Con-
trary to our intuition, simple LLM-based query
rewriting achieves the strongest overall bias re-
duction, decreasing the average |t|-statistic from
8.72 to 4.02. This simple baseline outperforms
more sophisticated pseudo-document generation
methods across all four bias types. The result sug-
gests that much of the bias in dense retrieval may
stem from the surface-level characteristics of user
queries themselves—their brevity, specific lexical
choices, or syntactic patterns—which even basic
reformulation can address.
Pseudo-document methods show bias-specific
trade-offs.HyDE and Query2Doc show notably
different bias reduction profiles. HyDE reduces
brevity bias substantially (15.57 →11.10) but
slightlyexacerbatesposition bias (5.28 →5.39).
We hypothesize that HyDE-generated documents,
which replace the query entirely, may introduce
their own positional patterns that interact ad-
versely with document encodings. In contrast,
Query2Doc—which preserves the original query
while augmenting it—achieves the strongest posi-
tion bias reduction of any method (5.28 →2.81,
a 47% decrease). This suggests that retaining the
original query signal provides an anchoring effect
that prevents the introduction of new biases.
Brevity bias remains the most severe.Across
all methods, brevity bias exhibits the highest |t|-
statistics, indicating that dense retrievers’ prefer-
ence for shorter documents is particularly robust to
query-side interventions. Even the best-performing
method (Rewrite) leaves a residual bias of 6.62,
compared to near-complete mitigation for position
bias (2.81 withQ2D). This asymmetry suggests
Brevity LiteralPosition
RepetitionBaseline
Rewrite
HyDE
HyDE-CPT
Q2D
Q2D-CPT15.57 8.89 5.28 5.17
6.62 4.15 2.96 2.37
11.10 6.46 5.39 4.86
10.65 6.57 5.46 4.44
11.91 4.87 2.81 4.99
11.70 5.01 2.90 4.67Bias Reduction by Type
0 5 10 15
Mean |t|-statisticFigure 2:Mean |t|-statistic by bias type and method
(averaged across 6 retrievers).Lower values indicate
reduced bias. All |t|values were averaged irrespective
of statistical significance.
that brevity bias may be more deeply encoded in
document representations, requiring retriever-level
interventions rather than query modifications alone.
Literal matching responds well to most tech-
niques.All query enhancement techniques sub-
stantially reduce literal matching bias, with an aver-
age reduction of 39.12%. This suggests that these
techniques dilute exact lexical overlap between
short queries and documents, shifting retrieval from
lexical toward semantic matching.
Repetition bias proves resistant to pseudo-
document methods.While simple rewriting cuts
repetition bias by 54% (5.17 →2.37), HyDE and
Query2Doc show minimal improvement (5.17 →
4.86 and 4.99, respectively). This asymmetry sug-
gests that LLM-generated pseudo-documents may
inadvertently repeat query terms—a natural conse-
quence of generating text conditioned on answer-
ing the query—thereby perpetuating rather than
mitigating repetition bias. See Appendix A for
examples of generated pseudo-documents which
confirms the same.

ContrieverCoCoDRDRAGONDRAGON+ RetroMAEColBERTv2Baseline
Rewrite
HyDE
HyDE-CPT
Q2D
Q2D-CPT9.45 4.09 10.72 9.73 8.28 10.08
4.11 2.07 4.59 4.60 3.74 5.03
7.03 7.66 8.47 8.35 4.74 5.46
6.81 7.68 8.19 8.03 4.62 5.35
7.32 7.89 4.58 4.01 7.62 5.45
7.22 7.61 4.23 4.22 7.73 5.43Retrieval Bias Across Types
0 2 4 6 8 10
Mean |t|-statisticFigure 3:Mean |t|-statistic by retriever and query en-
hancement method, averaged across all four bias
types.Lower values indicate reduced bias. Sim-
ple rewriting consistently reduces bias, while pseudo-
document methods show differential effects.
4.3 Variation Across Dense Retrievers
Figure 3 reveals the heterogeneity in how different
retrievers respond to various query enhancement
techniques.
Retriever-specific effects.We observe that sim-
ple query rewriting achieves consistent bias re-
duction across all six retrievers, with mean |t|-
statistics ranging from 2.07 (CoCoDR) to 5.03 (Col-
BERTv2). This consistency stands in stark contrast
to the pseudo-document methods, which exhibit
highly retriever-dependent effects. For example,
Q2D dramatically reduces bias for DRAGON and
DRAGON+ (10.72 →4.58 and 9.73 →4.01, re-
spectively), yetincreasesbias for CoCoDR (4.09
→7.89) and RetroMAE (8.28 →7.62). Similarly,
HyDE substantially harms CoCoDR (4.09 →7.66),
nearly doubling its bias level. These findings sug-
gest that the interaction between pseudo-document
generation and retriever architecture is non-trivial
and potentially unpredictable.
Architectural patterns.Several architecture-
specific trends emerge from these results. The
DRAGON variants—both trained with diverse aug-
mentation strategies—respond exceptionally well
to Query2Doc, achieving bias levels comparable to
simple rewriting (4.01–4.58 vs. 4.59–4.60). We hy-
pothesize that these retrievers, having been exposed
to diverse query-document pairings during train-
ing, are better equipped to leverage the augmentedMethod RetroMAE Contriever CoCoDR
Baseline 0.0% 0.8% 2.0%
Rewrite 0.0% 0.8% 3.6%
HyDE 2.8% 4.0% 20.0%
HyDE-CPT 2.8% 5.2% 26.0%
Q2D 2.8% 3.2% 18.0%
Q2D-CPT 3.2% 3.6% 19.2%
Best improv.— 6.5×13×
Table 2:FOILdataset accuracy across retrievers.
CoCoDR shows the strongest response to query en-
hancement, with HyDE-CPT achieving 26% accuracy
(13×over baseline). Simple rewriting provides minimal
improvement across all retrievers.
context that Query2Doc provides. In contrast,
CoCoDR exhibits anomalous behavior: despite
having the lowest baseline bias (4.09), it is the only
retriever for which all pseudo-document methods
increase bias. This may reflect CoCoDR’s continu-
ous contrastive distillation objective, which could
make it sensitive to the distributional shift intro-
duced by LLM-generated content. ColBERTv2’s
late interaction architecture yields remarkably sta-
ble results across all enhancement methods (5.03–
5.46), suggesting that token-level matching may
be inherently more robust to query formulation
changes than single-vector representations.
4.4 Effect of Query Enhancement on
Interplay Between Bias Types
While Section 4 demonstrated that query rewriting
techniques reduce individual bias types, a natural
question arises:do these improvements transfer to
scenarios where multiple biases act concurrently?
To investigate this, we evaluate on the FOILsub-
set of ColDeR, which presents an adversarial set of
250document pairs where document D1exploits
multiple bias-inducing features (brevity, answer po-
sition, lexical overlap, repetition) while lacking the
answer, and D2contains the correct evidence em-
bedded within unrelated context. We report results
for RetroMAE, Contriever, and CoCoDR here, to
capture the spectrum of performances.
From Table 2, we see that simple query rewrit-
ing provides minimal improvement on FOILde-
spite achieving the largest reduction in individ-
ual bias metrics (Section 4). In contrast, pseudo-
document generation methods show substantial
gains—on CoCoDR, HyDE achieves 10 ×improve-
ment (20.0%) and Query2Doc achieves 9 ×im-
provement (18.0%) over baseline. These meth-
ods fundamentally alter the query representation

through semantic expansion and length normaliza-
tion, thereby providing more robust resistance to
adversarial bias combinations. CPT provides addi-
tional benefit over vanilla techniques, with HyDE-
CPT achieving the strongest performance at 26.0%
(13×over baseline), which suggests that document-
specific fine-tuning addresses not just relevance
matching but also enhances bias robustness.
These findings motivate a deeper investigation:
what distinguishes methods that genuinely reduce
bias sensitivity from those that merely obscure it?
We address this question through a deeper mecha-
nistic analysis in the following section.
5Mechanistic Analysis of Bias Reduction
We now investigate why query-enhancement tech-
niques might succeed or fail at mitigating biases.
We conduct an analysis to measure how rewriting
affects the correlation between retrieval scores and
bias-inducing features.
If a retriever exhibits bias toward a particular
document feature (e.g., shorter length, answer posi-
tion), we expect retrieval scores to correlate with
that feature independent of relevance. Our intuition
is that effective bias mitigation should reduce this
correlation, and we operationalize this intuition by
computing the Spearman correlation ρbetween re-
trieval scores and bias-inducing features across the
ColDER benchmark.
For each query q(original or rewritten) and doc-
ument d, we compute the retrieval score S(q, d)
and extract the corresponding bias featuref(q, d):
(i)Brevity bias:Document length in tokens,
f(q, d) =|d|.
(ii)Literal matching bias:Jaccard similar-
ity (Jaccard, 1901) between query and doc-
ument term sets. Let T(q) andT(d) denote
the sets of unique terms in query qand docu-
mentd, respectively. Then, we define:
f(q, d) =|T(q)∩ T(d)|
|T(q)∪ T(d)|
(iii) Position bias:Normalized answer position
within the document, where pos(a, d) denotes
the character offset of answer span ain docu-
mentd:
f(q, d) =pos(a, d)
|d|∈[0,1]Method Brevity Literal Position Repetition
Baseline|ρ q| 0.36 0.43 0.11 0.21
Rewrite +10% −7% −373% −112%
HyDE +56% +20% −2% −59%
HyDE-CPT +49% +19% +54% −9%
Q2D +48% +36% −55% −13%
Q2D-CPT +53% +32% +23% −21%
Table 3:Feature-score decorrelation analysis. |ρq|
represents baseline Spearman correlation for query q.
Values show percentage reduction in |ρ|after rewriting
query q(positive = reduced sensitivity, negative =
increased sensitivity). Results averaged across all 6
dense retrievers studied.
(iv)Repetition bias:Average term frequency of
query terms within the document. Let tf(t, d)
denote the number of occurrences of term tin
documentd:
f(q, d) =1
|T(q)|X
t∈T(q)tf(t, d)
We then compute ρc=Spearman(S c, f)for
each query condition c. A retriever with no system-
atic bias would yieldρ≈0, while higher absolute
correlations indicate stronger bias. We quantify
bias reduction as:
∆ρ=|ρ original| − |ρ rewritten|(1)
where ∆ρ >0 indicates that rewriting reduced the
retriever’s sensitivity to the bias-inducing feature.
Findings:Table 3 presents the mean correlation
reduction ∆ρacross six dense retrievers. We ob-
serve several key patterns:
Brevity bias.All rewriting methods substan-
tially reduce correlation with document length.
Pseudo-document methods achieve the largest
gains (HyDE: +56%, Q2D: +48%), consistent with
our hypothesis that generating longer text shifts the
query representation away from favoring brief doc-
uments. This finding aligns with Section 4, where
all methods reduced brevity bias.
Literal matching bias.Query2Doc variants
achieve substantially larger reductions in literal
bias compared to HyDE. We attribute this to a
key mechanistic difference: Query2Doc preserves
original query terms while adding semantic con-
text, effectivelydilutingrather thanreplacinglexi-
cal signals. HyDE’s complete query replacement
may introduce new lexical artifacts from the gener-
ated pseudo-document. Notably, simple rewriting
slightlyincreasesliteral bias ( −7%), suggesting

that paraphrasing without content expansion actu-
ally reinforces lexical overlap as the rewritten query
may use more “retrieval-friendly” vocabulary.
Repetition bias.All approachesincreasesensitiv-
ity to repetition, with simple rewriting showing the
most severe degradation. This represents a system-
atic failure that explains the resistance of repeti-
tion bias to pseudo-document methods. Analysis
of generated pseudo-documents shows that LLMs
naturally produce repetitive text patterns when an-
swering queries, which correlates with documents
containing repeated query terms. The |t|-statistic
improvements likely reflect increased score vari-
ance rather than true debiasing.
Position bias.Position bias reveals the strongest
mechanistic differences. Simple rewriting greatly
position sensitivity, yet achieved strong |t|-statistic
reduction in Section 4. This paradox suggests that
simple rewriting reduces position bias through in-
creased retrieval noise rather than principled decor-
relation. In contrast, HyDE-CPT achieves genuine
decorrelation. Only Q2D-CPT combines strong
|t|-statistic reduction with positive decorrelation,
suggesting that continual pretraining is essential
for robust position bias mitigation.
Reconciling |t|-statistics and decorrelation:
These findings reveal a crucial insight thatreducing
overall bias (as measured by |t|-statistics) does not
imply reducing sensitivity to bias-inducing features.
We identify two distinct mechanisms:
(i) Variance-based reduction:Simple rewriting
achieves strong |t|-statistic improvements whilein-
creasingfeature-score correlations for position and
repetition biases. The mechanism appears to be
increased retrieval score variance—noisier scores
reduce the statistical power to detect systematic
preferences, lowering |t|-statistics without address-
ing underlying bias sensitivity. This also explains
why simple rewriting fails entirely on FOIL(Ta-
ble 2); when biases act together, the underlying
sensitivities may compound further.
(ii) Decorrelation-based reduction:Pseudo-
document methods, particularly the CPT variants,
achieve bias reduction through principled decor-
relation from bias-inducing features. Specifically,
HyDE-CPT and Q2D-CPT show positive decorre-
lation for brevity, literal matching,andposition
biases simultaneously. This more robust mecha-
nism explains their gains on FOIL, where genuine
insensitivity to bias features provides resistance to
adversarial combinations.To further understand why simple rewriting
shows contrasting effects, we analyzed the lin-
guistic transformations applied by the LLM-based
rewriting (Table 4 in Appendix C). Our key obser-
vation is that these transformations are predomi-
nantlysyntactic rather than semantic, introducing
sufficient variance to reduce the paired t-statistic
without actually reducing the retriever’s sensitivity
to bias-inducing features.
6 Discussion
Implications for researchers.Our findings sug-
gest a taxonomy of retrieval biases based on their
responsiveness to query-side interventions.(i)
Query-document interaction biases—such as literal
matching—arise from how queries and documents
are compared and can be mitigated through query
transformation.(ii) Document encoding biases—
particularly position bias—appear embedded in
document representations and persist regardless of
query formulation, likely requiring retriever-level
modifications. Additionally, our decorrelation anal-
ysis reveals that aggregate metrics can obscure dif-
ferent underlying mechanisms, and we recommend
future work also report feature-score correlations.
Finally, the increase in repetition sensitivity across
all methods represents a systematic failure mode
that warrants further investigation.
Implications for practitioners.When select-
ing query enhancement techniques, practitioners
should consider specific bias vulnerabilities rather
than aggregate metrics. Simple rewriting achieves
strong overall reduction but fails when multiple
biases act together. Critically, enhancement effects
are also retriever-dependent, so techniques should
be validated on specific retrievers before deploy-
ment. For high-stakes domains, we recommend
CPT variants despite computational overhead, as
our findings reveal that decorrelation-based reduc-
tion generalizes better to adversarial conditions
than variance-based reduction.
7 Conclusion
In this work, we present the first systematic study
of how query enhancement techniques affect dense
retrieval biases in RAG systems. Our evaluation
reveals that while all techniques reduce aggregate
bias, they operate through fundamentally differ-
ent mechanisms. Simple rewriting achieves strong
overall reduction but increases sensitivity to bias-
inducing features; pseudo-document methods with

continual pretraining achieve more robust improve-
ments through genuine decorrelation. We estab-
lish a taxonomy distinguishing query-document
interaction biases, which yield to query-side inter-
ventions, from document encoding biases, which
likely require retriever-level modifications. Our
findings provide practical guidance for deploying
bias-aware RAG systems, and highlight the con-
tinued need for retrieval paradigms that prioritize
semantic relevance over superficial characteristics.
8 Limitations
(1) Benchmark and retriever scope:Our eval-
uation relies on the ColDER benchmark, which
provides controlled document pairs derived from
Re-DocRED. While this controlled setup enables
precise bias measurement, it may not fully capture
the distribution of biases in real-world retrieval sce-
narios where multiple subtle biases co-occur in less
structured ways. Additionally, we evaluate only
dense bi-encoder and late-interaction retrievers, pri-
marily because of their widespread utility and ef-
fectiveness today. Therefore, our findings may not
generalize to sparse retrievers (e.g., BM25), hybrid
systems, or emerging retrieval paradigms such as
generative retrieval.
(2) Query enhancement coverage:We evalu-
ate five query enhancement techniques using two
open-source LLMs (Gemma-3 and Qwen3). Other
methods such as Step-Back Prompting, multi-
query fusion, or chain-of-thought augmented re-
trieval may exhibit different bias profiles. How-
ever, our key contribution is not tied to a specific
query-enhancement type. Furthermore, proprietary
LLMs with stronger instruction-following capabili-
ties might produce pseudo-documents with differ-
ent characteristics, potentially affecting bias out-
comes, which we could not evaluate in this work
due to cost constraints.
(3) Downstream evaluation:Our analysis fo-
cuses on retrieval-stage biases measured through
score differences and correlations. We do not eval-
uate how these biases propagate to downstream
generation quality in end-to-end RAG systems. A
retrieval bias that appears significant in isolation
may be attenuated, or amplified, by the downstream
language model, and understanding this interaction
remains an important direction for future work.Ethical Considerations
Our work aims to understand and mitigate biases
in dense retrieval systems, which we view as a net
positive for the responsible and fair deployment
of RAG systems. However, we acknowledge that
detailed characterization of retrieval biases could
theoretically be misused to craft adversarial doc-
uments that exploit these biases to manipulate re-
trieval rankings. However, we believe the benefits
of transparency outweigh this risk, as awareness
of these biases enables practitioners to implement
appropriate safeguards.
More broadly, the biases we study have implica-
tions for fairness in information access for users of
modern generative search systems. For example,
brevity bias may systematically disadvantage com-
prehensive sources, and literal matching bias may
favor keyword-filled content over semantically rich
sources. These effects are particularly concerning
in high-stakes domains such as legal, sociotechni-
cal, and medical information retrieval, where bi-
ased retrieval could lead to incomplete or skewed
information being surfaced to end users (Liu et al.,
2023). Our findings therefore underscore the impor-
tance of bias-aware evaluation in retrieval system
development.
References
Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei
Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and
Wen-tau Yih. 2024. Reliable, adaptable, and at-
tributable language models with retrieval.arXiv
preprint arXiv:2403.03187.
Parishad BehnamGhader, Santiago Miret, and Siva
Reddy. 2023. Can retriever-augmented language
models reason? the blame game between the re-
triever and the language model. InFindings of the
Association for Computational Linguistics: EMNLP
2023, pages 15492–15509, Singapore. Association
for Computational Linguistics.
Carlo E Bonferroni. 1935. Il calcolo delle assicurazioni
su gruppi di teste.Studi in onore del professore
salvatore ortu carboni, pages 13–60.
João Coelho, Bruno Martins, Joao Magalhaes, Jamie
Callan, and Chenyan Xiong. 2024. Dwell in the
beginning: How language models embed long docu-
ments for dense retrieval. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), pages 370–377,
Bangkok, Thailand. Association for Computational
Linguistics.
Mohsen Fayyaz, Ali Modarressi, Hinrich Schuetze, and
Nanyun Peng. 2025. Collapse of dense retrievers:

Short, early, and literal biases outranking factual evi-
dence. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9136–9152, Vienna,
Austria. Association for Computational Linguistics.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023a. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. 2023b. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1).
Agam Goyal, Xianyang Zhan, Yilun Chen, Koustuv
Saha, and Eshwar Chandrasekharan. 2025. MoMoE:
Mixture of moderation experts framework for AI-
assisted online governance. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 12645–12660, Suzhou,
China. Association for Computational Linguistics.
Agam Goyal, Xianyang Zhan, Charlotte Lambert, Kous-
tuv Saha, and Eshwar Chandrasekharan. 2026. Vastu:
Value-aligned social toolkit for online content cura-
tion.arXiv preprint arXiv:2601.12491.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.ICLR, 1(2):3.
Michelle Huang, Agam Goyal, Koustuv Saha, and Es-
hwar Chandrasekharan. 2026. Answer bubbles: In-
formation exposure in ai-mediated search.arXiv
preprint arXiv:2603.16138.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Paul Jaccard. 1901. Étude comparative de la distribution
florale dans une portion des alpes et des jura.Bull
Soc Vaudoise Sci Nat, 37:547–579.
Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui
Wang, and Michael Bendersky. 2023. Query expan-
sion by prompting large language models.arXiv
preprint arXiv:2305.03653.
Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu,
Yue Fang, Zhiyuan Wang, Jinyi Tang, Hongxin Ding,
Xu Chu, Junfeng Zhao, and 1 others. 2023. Hykge: A
hypothesis knowledge graph enhanced framework for
accurate and reliable medical llms responses.arXiv
preprint arXiv:2312.15883.Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Daehui Kim, Deokhyung Kang, Jonghwi Kim, Sangwon
Ryu, and Gary Lee. 2025. GuRE:generative query
REwriter for legal passage retrieval. InProceedings
of the Natural Legal Language Processing Workshop
2025, pages 424–438, Suzhou, China. Association
for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz,
Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun
Chen. 2023. How to train your dragon: Diverse
augmentation towards generalizable dense retrieval.
arXiv preprint arXiv:2302.07452.
Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023.
Evaluating verifiability in generative search engines.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 7001–7025.
Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won
Hwang, and Alexey Svyatkovskiy. 2022. ReACC:
A retrieval-augmented code completion framework.
InProceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 6227–6240, Dublin, Ireland.
Association for Computational Linguistics.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 5303–5315, Singa-
pore. Association for Computational Linguistics.
Shengyu Mao, Yong Jiang, Boli Chen, Xiao Li, Peng
Wang, Xinyu Wang, Pengjun Xie, Fei Huang, Hua-
jun Chen, and Ningyu Zhang. 2024. RaFe: Ranking
feedback improves query rewriting for RAG. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 884–901, Miami, Florida, USA.
Association for Computational Linguistics.
Agada Joseph Oche, Ademola Glory Folashade,
Tirthankar Ghosal, and Arpan Biswas. 2025. A sys-
tematic review of key retrieval-augmented generation
(rag) systems: Progress, gaps, and future directions.
arXiv preprint arXiv:2507.18910.
Olivia Pal, Agam Goyal, Eshwar Chandrasekharan, and
Koustuv Saha. 2026. The hidden toll of social me-
dia news: Causal effects on psychosocial wellbeing.
arXiv preprint arXiv:2601.13487.

Ori Ram, Liat Bezalel, Adi Zicher, Yonatan Belinkov,
Jonathan Berant, and Amir Globerson. 2023. What
are you token about? dense retrieval as distributions
over the vocabulary. InProceedings of the 61st An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 2481–
2498, Toronto, Canada. Association for Computa-
tional Linguistics.
Amanda Ross and Victor L Willson. 2017. Paired sam-
ples t-test. InBasic and advanced statistical tests:
Writing results sections and creating tables and fig-
ures, pages 17–19. Springer.
Michael J Ryan, Danmei Xu, Chris Nivera, and
Daniel Campos. 2025. Enronqa: Towards person-
alized rag over private documents.arXiv preprint
arXiv:2505.00263.
Saron Samuel, Benjamin Van Durme, and Eugene
Yang. 2026. Does reasoning make search more fair?
comparing fairness in reasoning and non-reasoning
rerankers. InEuropean Conference on Information
Retrieval, pages 305–318. Springer.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. Col-
BERTv2: Effective and efficient retrieval via
lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 3715–3734, Seat-
tle, United States. Association for Computational
Linguistics.
Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal,
Violeta J Rodriguez, Paul A Bloom, Hari Sundaram,
and Koustuv Saha. 2025. Detecting early and implicit
suicidal ideation via longitudinal and information
environment signals on social media.arXiv preprint
arXiv:2510.14889.
Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and
Sharifah Mahani Aljunied. 2022. Revisiting Do-
cRED - addressing the false negative problem in
relation extraction. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 8472–8487, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane
Rivière, and 1 others. 2025. Gemma 3 technical
report.arXiv preprint arXiv:2503.19786.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogenous benchmark for zero-shot evalua-
tion of information retrieval models.arXiv preprint
arXiv:2104.08663.
Baiqiang Wang, Qian Lou, Mengxin Zheng, and Dong-
fang Zhao. 2025. Pir-rag: A system for private in-
formation retrieval in retrieval-augmented generation.
arXiv preprint arXiv:2509.21325.Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423, Singapore. Association for Com-
putational Linguistics.
Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao.
2022. RetroMAE: Pre-training retrieval-oriented lan-
guage models via masked auto-encoder. InProceed-
ings of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 538–548, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Yue Yu, Chenyan Xiong, Si Sun, Chao Zhang, and
Arnold Overwijk. 2022. Coco-dr: Combating dis-
tribution shifts in zero-shot dense retrieval with con-
trastive and distributionally robust learning.arXiv
preprint arXiv:2210.15212.
Xianyang Zhan, Agam Goyal, Yilun Chen, Eshwar
Chandrasekharan, and Koustuv Saha. 2025. Slm-
mod: Small language models surpass llms at content
moderation. InProceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Asso-
ciation for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pages
8774–8790.
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gon-
zalez. 2024. RAFT: Adapting language model to
domain specific RAG. InFirst Conference on Lan-
guage Modeling.
A Examples of Generated Query
Enhancements
In this section we provide examples of query en-
hancements across methods.
Vanilla Queries
1. When was The Private Life of Helen of Troy
published?
2. Where is Los Angeles located in or next to a
body of water?
3. Where was Cesare Mori born?
LLM Rewritten Queries
1. What is the publication date of The Private
Life of Helen of Troy?
2. What body of water is Los Angeles located
near or in?

3. What is the birthplace of Cesare Mori?
HyDE Queries
1. The “Private Life of Helen of Troy” was
published in 1920 by the renowned author,
Robert Graves. This groundbreaking work, a
reimagining of the Trojan War legend, offered a
fresh perspective on Helen’s character and the
events surrounding the war.
2. Los Angeles is located on the southern coast
of California, a state situated on the Pacific
Ocean. The city is situated on the shores of the
Pacific Ocean, with the Pacific Ocean bordering
its western side. The city is also bordered by
the Los Angeles Basin, a large inland basin that
is partially surrounded by mountains.
3. Cesare Borgia, the Italian Renaissance figure,
died in Rome, Italy on 18th of January 1507.
He was a powerful and controversial figure who
served as Cardinal and later as the Duke of
Valentinois. His death was a significant event in
the history of the Borgia family and the Italian
Renaissance.
Query2Doc Queries
1. When was The Private Life of Helen of Troy
published?
The “Private Life of Helen of Troy” was published
in 1920 by the renowned author, Robert Graves.
This groundbreaking work, a reimagining of the
Trojan War legend, offered a fresh perspective
on Helen’s character and the events surrounding
the war.
2. Where is Los Angeles located in or next to a
body of water?
Los Angeles is located on the southern coast of
California, a state situated on the Pacific Ocean.
The city is situated on the shores of the Pacific
Ocean, with the Pacific Ocean bordering its
western side. The city is also bordered by the
Los Angeles Basin, a large inland basin that is
partially surrounded by mountains.
3. Where was Cesare Mori born?
Cesare Borgia, the Italian Renaissance figure,
died in Rome, Italy on 18th of January 1507.
He was a powerful and controversial figure who
served as Cardinal and later as the Duke of
Valentinois. His death was a significant event in
the history of the Borgia family and the Italian
Renaissance.
B Results for Qwen3
To verify that our findings are not specific to the
choice of language model used for query enhance-
ment, we replicate our main experiments using
Qwen3-4B-Instruct as an alternative to Gemma-3-
12B-IT. Figure 4 presents the complete bias analy-sis across all retrievers and bias types.
The results show consistent patterns, with simple
rewriting achieveing substantial bias reduction on
individual metrics, while HyDE and Query2Doc
show similar moderate improvements. This con-
sistency across model architectures and scales (4B
vs. 12B parameters) suggests that our findings
reflect fundamental properties of query transfor-
mation strategies rather than artifacts of a specific
LLM’s generation characteristics.
C Further Analysis of Simple
LLM-Based Query Rewriting
Metric Baseline Rewrite
Avg. query length (words) 7.3 8.3
Length change (words) — +1.0
Term preservation (%) 100 57
New terms introduced — 3.8
Unique vocabulary 420 489
Entity preservation (%) 100 72.5
Table 4:Query transformation characteristics across
enhancement methods.Simple rewriting makes min-
imal syntactic changes (high term/entity preservation).
This explains why rewriting reduces measured bias vari-
ance but fails on adversarial FOILexamples.
Table 4 reveals that simple LLM-based rewriting
produces minute changes, with queries increase by
only 1 word on average (7.3 →8.3 words), preserv-
ing 57% of original terms while introducing 3.8
new terms. The transformations are therefore pre-
dominantlysyntactic rather than semantic. For ex-
ample,“When was X published?”becomes“What
is the publication date of X?”The key entity names
(e.g.,“Lake Ewauna, ” “Miami Sound Machine”)
remain intact in 72.5% of cases.
This explains the paradox we observed in our re-
sults in Section 4 and Section 5: syntactic reformu-
lation introduces sufficient variance to reduce the
paired t-statistic—a measure ofconsistencyin bias
direction—without actually reducing the retriever’s
sensitivity to bias-inducing lexical features. There-
fore, when multiple biases compound in the FOIL
setting, this surface-level variance fails to provide
robustness.
D Compute Resources
All experiments on open-source models were run
on internal organization GPU servers equipped
with 2xNVIDIA H100 and 3xNVIDIA A40.

ContrieverBrevityContrieverLiteralContrieverPosition ContrieverRepetitionCoCoDRBrevityCoCoDRLiteral CoCoDRPositionCoCoDR
RepetitionDRAGONBrevity DRAGONLiteral DRAGONPosition DRAGON
RepetitionDRAGON+BrevityDRAGON+LiteralDRAGON+PositionDRAGON+RepetitionRetroMAEBrevityRetroMAELiteralRetroMAEPosition RetroMAERepetitionColBERTv2BrevityColBERTv2Literal
ColBERTv2PositionColBERTv2RepetitionBaseline
Rewrite
HyDE
HyDE-CPT
Q2D
Q2D-CPT14.6 9.8 7.7 6.2 9.8 3.8 0.4 2.1 13.9 9.9 12.8 6.2 17.9 10.7 4.3 5.8 16.7 8.9 1.5 5.7 20.6 10.1 4.1 5.1
5.9 4.3 2.6 2.5 5.0 1.9 0.2 1.3 5.4 5.0 5.6 2.3 5.9 5.4 3.8 3.1 5.8 3.8 1.5 2.2 10.1 4.6 3.6 1.9
6.6 8.2 6.7 6.5 23.8 4.4 1.7 1.3 9.1 8.7 12.1 5.5 15.0 8.8 3.4 6.0 3.6 7.9 2.0 5.5 10.9 1.8 5.8 4.1
6.1 7.2 8.0 6.7 23.7 4.8 1.5 1.0 8.9 9.3 11.1 4.5 12.6 10.3 3.2 5.0 3.0 7.2 2.7 6.1 10.7 1.2 6.8 3.8
7.2 6.8 6.8 6.4 22.6 4.8 1.4 2.1 6.4 5.4 0.8 7.3 2.7 7.2 0.1 6.8 21.9 1.3 0.5 5.0 8.3 2.5 5.8 4.0
7.2 9.4 8.4 6.0 20.7 5.5 1.5 1.7 5.5 5.5 1.1 4.8 4.1 7.0 0.3 7.0 26.4 2.1 0.3 5.1 9.9 1.8 7.7 4.4Complete Bias Analysis (Qwen)
(Lower Values = Less Biased)
0 5 10 15 20 25
|t|-statisticFigure 4:Complete bias analysis using Qwen3-4B-Instruct for query enhancement.Each cell shows the
|t|-statistic measuring retrieval bias strength across retrievers and biases. Lower values indicate reduced bias.
Results demonstrate consistent patterns with Gemma-3-12B-IT, confirming generalizability of our findings.
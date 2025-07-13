# The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora

**Authors**: Chen Amiraz, Yaroslav Fyodorov, Elad Haramaty, Zohar Karnin, Liane Lewin-Eytan

**Published**: 2025-07-10 08:38:31

**PDF URL**: [http://arxiv.org/pdf/2507.07543v1](http://arxiv.org/pdf/2507.07543v1)

## Abstract
Cross-lingual retrieval-augmented generation (RAG) is a critical capability
for retrieving and generating answers across languages. Prior work in this
context has mostly focused on generation and relied on benchmarks derived from
open-domain sources, most notably Wikipedia. In such settings, retrieval
challenges often remain hidden due to language imbalances, overlap with
pretraining data, and memorized content. To address this gap, we study
Arabic-English RAG in a domain-specific setting using benchmarks derived from
real-world corporate datasets. Our benchmarks include all combinations of
languages for the user query and the supporting document, drawn independently
and uniformly at random. This enables a systematic study of multilingual
retrieval behavior.
  Our findings reveal that retrieval is a critical bottleneck in cross-lingual
domain-specific scenarios, with significant performance drops occurring when
the user query and supporting document languages differ. A key insight is that
these failures stem primarily from the retriever's difficulty in ranking
documents across languages. Finally, we propose a simple retrieval strategy
that addresses this source of failure by enforcing equal retrieval from both
languages, resulting in substantial improvements in cross-lingual and overall
performance. These results highlight meaningful opportunities for improving
multilingual retrieval, particularly in practical, real-world RAG applications.

## Full Text


<!-- PDF content starts -->

The Cross-Lingual Cost:
Retrieval Biases in RAG over Arabic-English Corpora
Chen Amiraz Yaroslav Fyodorov Elad Haramaty Zohar Karnin Liane Lewin-Eytan
Technology Innovation Institute
{chen.amiraz,yaroslav.fyodorov,elad.haramaty,zohar.karnin,liane.lewineytan}@tii.ae
Abstract
Cross-lingual retrieval-augmented generation
(RAG) is a critical capability for retrieving and
generating answers across languages. Prior
work in this context has mostly focused on gen-
eration and relied on benchmarks derived from
open-domain sources, most notably Wikipedia.
In such settings, retrieval challenges often re-
main hidden due to language imbalances, over-
lap with pretraining data, and memorized con-
tent. To address this gap, we study Arabic-
English RAG in a domain-specific setting using
benchmarks derived from real-world corporate
datasets. Our benchmarks include all combina-
tions of languages for the user query and the
supporting document, drawn independently and
uniformly at random. This enables a systematic
study of multilingual retrieval behavior.
Our findings reveal that retrieval is a critical
bottleneck in cross-lingual domain-specific sce-
narios, with significant performance drops oc-
curring when the user query and supporting
document languages differ. A key insight is
that these failures stem primarily from the re-
triever’s difficulty in ranking documents across
languages. Finally, we propose a simple re-
trieval strategy that addresses this source of
failure by enforcing equal retrieval from both
languages, resulting in substantial improve-
ments in cross-lingual and overall performance.
These results highlight meaningful opportuni-
ties for improving multilingual retrieval, par-
ticularly in practical, real-world RAG applica-
tions.
1 Introduction
Retrieval-Augmented Generation (RAG) has
emerged as the widely accepted approach for
grounding large language models (LLMs) in ex-
ternal knowledge, with most research and devel-
opment focused on high-resource languages, most
notably English. However, many real-world ap-
plications, especially in corporate contexts, relyon multilingual corpora, where content spans both
high- and low-resource languages. For example,
internal knowledge management systems in govern-
mental or legal domains often store content in both
a high-resource language like English and the lo-
cal language, while customer support systems may
receive queries in the local language that require
retrieving information from a corpus that mixes
technical content in both languages. These scenar-
ios introduce cross-lingual complexity, where users
interact in a low-resource language while relevant
information resides in a corpus containing docu-
ments in multiple languages. Prior work has shown
that system performance in such cross-lingual set-
tings tends to lag behind monolingual setups, due
to challenges across both retrieval and generation
(Wu et al., 2024; Sharma et al., 2025; Park and
Lee, 2025). In this work, we focus on the English-
Arabic setting – a representative and important case
of high- and low-resource language interaction.
Prior work has primarily focused on the gen-
eration component (Liu et al., 2025; Chirkova
et al., 2024), often using multilingual benchmarks
derived from Wikipedia, the predominant open-
domain source. However, evaluating retrieval in
this context poses challenges due to several inher-
ent characteristics: language imbalances, overlap
with pretraining data, and the fact that much of
Wikipedia’s knowledge is embedded in the model’s
parametric memory. In contrast, our work focuses
on the less explored retrieval component within a
bilingual, domain-specific setting representative of
real-world corporate applications.
To this end, we construct benchmarks from UAE
corporate datasets with parallel English-Arabic doc-
uments. Each benchmark includes a balanced set of
English and Arabic queries, with answers grounded
in a single language. The languages of the user
query and supporting document are selected inde-
pendently, enabling a systematic analysis of cross-
lingual biases. Our findings show that retrieval
1arXiv:2507.07543v1  [cs.CL]  10 Jul 2025

quality is a key driver of end-to-end performance,
with the cross-language setting posing the greatest
challenge. Further analysis reveals that the main
source of error stems not from the language mis-
match between the user query and the supporting
document, but from limitations in cross-lingual
retrieval within a shared multilingual embedding
corpus space.
Finally, we propose a simple mitigation strat-
egy tailored to the identified error source, which
retrieves an equal number of passages from each
language, resulting in significant improvements in
cross-lingual retrieval. The effectiveness of such
a basic intervention suggests that there remains
substantial room for advancements in this area.
2 Related Work
Cross Lingual Information Retrieval (CLIR) is a
critical capability for accessing knowledge across
language boundaries, and has gained renewed at-
tention with the rise of cross-lingual retrieval-
augmented generation (RAG) systems. These sys-
tems typically operate in two phases, retrieval and
answer generation. CLIR has historically been
done via translation (see Galuš ˇcáková et al. (2021)
and references within). With the rise of dense re-
trieval, most leading techniques avoid direct trans-
lation and instead embed queries and documents of
different languages into the same space (Chen et al.,
2024; Louis et al., 2025; Wang et al., 2024; Asai
et al., 2021b). The improved performance over
retrieval tasks was also verified to occur in RAG
for question answering by Chirkova et al. (2024)
that show an advantage to these direct methods over
translation coupled with monolingual retrieval. The
different retrieval techniques vary in their training
method and data collections, yet all follow the same
pattern of embedding the query and the document.
They fall into the broader area of Cross Lingual
Allignment, where the objective is to align repre-
sentations of different languages (Hämmerl et al.,
2024). This broader area and the specifics of the
different models are outside the scope of this paper.
For the answer generation phase, the challenge
comes from the fact that (1) the user language may
not be the same as the retrieved document(s) lan-
guage, and (2) the documents may be written in
multiple languages. Liu et al. (2025) provide a
benchmark containing questions that require rea-
soning. They show that the language difference be-
tween the user and document languages can causeissues such as answers in the wrong language. They
also show that documents of different languages
make cross document reasoning more challeng-
ing. Ranaldi et al. (2025) show a simple yet ef-
fective method for overcoming both issues; they
use a translation service to translate the query and
documents to english, then translate the answer.
In contrast, Wu et al. (2024) show (on different
benchmarks) that this translation-based method
breaks down when using lower-quality translation
systems, such as medium-scale LLMs. Chirkova
et al. (2024) provide practical solutions to the issue
of a different user and document language; they
highlight comments that when added to the sys-
tem prompt, result in improved performance. Qi
et al. (2025) focus on generation in cross-lingual
RAG settings, addressing the influence of retrieved
passages both when they are relevant, regardless
of their language, or when distracting passages in
different languages are provided in the context.
Several studies have examined bias in both re-
trieval and generation, namely the preference for
high-resource languages like English over low-
resource ones such as Arabic. Wu et al. (2024)
evaluate end-to-end RAG performance across mul-
tiple LLMs and show that high-resource languages
consistently outperform low-resource ones in both
monolingual and cross-lingual settings. They also
find that, when relevant documents exist in mul-
tiple languages, English passages are more likely
to be selected. Sharma et al. (2025) manually con-
structs a small benchmark over a synthetic cor-
pus to avoid the influence of the parametric mem-
ory, and observe a consistent bias favoring the user
query language in both stages. Park and Lee (2025)
analyze language preferences in both retrieval and
generation, highlighting a strong bias toward high-
resource languages, especially when the query and
document languages match. English is noted as
an exception, often outperforming even monolin-
gual configurations – an effect attributed to English
dominance in pretraining data.
Most prior work on multilingual RAG, including
those cited here, relies on Wikipedia-based datasets
and derived benchmarks such as MKQA (Longpre
et al., 2021), XOR-QA (Asai et al., 2021a), and
MLQA (Lewis et al., 2020). However, Wikipedia
introduces several inherent properties: it is signifi-
cantly richer in English content, has been typically
used during the pretraining of both retrievers and
generators, and much of its factual knowledge is
encoded in the model’s parametric memory. All
2

these factors impact cross-lingual behavior, and in
particular, the behavior and influence of retrieval.
Chirkova et al. (2024), while focusing on bench-
marks derived from Wikipedia, explicitly acknowl-
edge that retrieval performance in multilingual spe-
cialized domains remains under-explored.
Thus, our work addresses a gap that has received
limited attention by focusing on the retrieval com-
ponent in a domain-specific, bilingual corporate set-
ting involving a high- and low-resource language
pair (English-Arabic). It uses clean multilingual
corpora with well-aligned content across both lan-
guages, which are unlikely to have been seen dur-
ing pretraining and represent realistic and practical
RAG use cases.
3 Evaluation Pipeline
We use a cross-lingual basic RAG setup focused
on English and Arabic. Given a query in either
language, its goal is to generate an answer in the
same language. The corpus includes documents
in both languages, and each query is associated
with a ground-truth answer found in one language
only. The other language may contain partial or no
relevant information.
Our RAG pipeline consists of the standard com-
ponents: retrieval, re-ranking, and answer gener-
ation. Retrieval is performed using dense vector
search over a bilingual corpus. We experiment with
the multilingual embedding models BAAI BGE-
M31(referred to as BGE-M3 from now on) and
Multilingual-E5-Large2(referred to as M-E5 ), both
of dimension 1024 , along with the BGE-v2-M33re-
ranker. These models were chosen because they are
considered among the top-performing open-source
retrievers and re-ranker, and have been widely
adopted in previous works. For answer generation,
we use Qwen-2.5-14B-Instruct4, a large language
model with strong multilingual capabilites. Dur-
ing inference, the 20 most relevant passages are
retrieved for a given question, then re-ranked based
on their relevance and utility for answer generation.
The top-5 ranked passages are used to augment the
prompt provided to the LLM for answer generation
(using Prompt A.1).
1https://huggingface.co/BAAI/bge-m3
2intfloat/multilingual-e5-large
3https://huggingface.co/BAAI/
bge-reranker-v2-m3
4https://huggingface.co/Qwen/Qwen2.
5-14B-Instruct3.1 Metrics
An effective RAG system requires success at three
stages: retrieving a relevant passage, preserving it
through re-ranking, and leveraging it in generation
to produce an accurate answer. We analyze the
overall end-to-end performance, as well as each
component in isolation: retrieval, re-ranking, and
generation.
The end-to-end performance and the generation
component are evaluated using an answer quality
metric, which we refer to as accuracy, based on a
semantic equivalence to ground-truth answers pro-
vided by our benchmarks (see next section). Specif-
ically, we adopt an LLM-as-a-judge approach to
assess correctness, using Claude 3.5 Sonnet to de-
termine whether a generated answer matches the
ground-truth reference (see Prompt A.2), follow-
ing recent work by Zheng et al. (2023). Although
LLM-based judgments have faced critique, partic-
ularly for relevance assessment (Soboroff, 2024),
prior studies have shown a high correlation with
human evaluations in QA contexts. Moreover, the
common alternative of strict lexical match is even
less reliable in a multilingual setting, as discussed
for example in Qi et al. (2025), making a semantic
measure more appropriate.
To further support this choice, we validated the
metric through human evaluation with native speak-
ers of the tested languages, confirming over 95%
agreement between human and automated ratings
for both English and Arabic (see Appendix A.1.1
for more details). Given our focus on semantic
similarity with respect to the ground truth, we find
LLM-as-a-judge to be a practical and reliable mea-
sure.
For evaluating the retrieval component, we mea-
sure whether the ground-truth answer can be in-
ferred from each retrieved passage. We obtain these
relevance judgments using Claude 3.5 Sonnet with
Prompt A.3. Based on these relevance labels, we
report Hits@20, indicating whether a relevant pas-
sage appears among the top 20retrieved results.
For reranking, we apply the same procedure and
report Hits@5 to measure whether relevant pas-
sages appear among the top results of the reranked
list. Measuring the presence of relevant passages
among the top results is particularly important in
a RAG setting, as it reflects whether downstream
components have access to the required evidence.
The validity of these metrics is supported by their
correlation with downstream accuracy, as detailed
3

in Appendix A.1.2.
3.2 Our Benchmarks
We focus on a corporate setting and construct two
benchmarks, each based on a separate corpus. Both
benchmarks are derived from public websites that
contain parallel content in English and Arabic. The
first benchmark, Legal , is based on the UAE Leg-
islation website5, which contains 390 laws, with
each law described in separate documents in En-
glish and Arabic. The second benchmark, Travel ,
is based on the UAE Ministry of Foreign Affairs
website6, which offers travel-related information
for multiple countries, such as visa requirements
and embassy contacts. For each country, the infor-
mation is presented in two parallel documents, one
per language.
Having each document available in both lan-
guages is essential for our experimental design.
In order to build a corpus for each of these two
use cases, we assign a document language to each
document uniformly at random during corpus con-
struction, ensuring that every document appears in
exactly one language within the corpus. The result-
ing Legal corpus includes roughly 1.5M words,
while the Travel corpus contains around 150K
words. After building and indexing this bilin-
gual corpus, we proceed to create the benchmark.
We used DataMorgana (Filice et al., 2025), a syn-
thetic question–answer generation tool, to create
query–answer pairs per document, ensuring that
each question could be answered using that docu-
ment alone. The language of each query–answer
pair (the user language ) is also selected uniformly
at random and independently of the document lan-
guage, resulting in a benchmark that supports sys-
tematic evaluation across all language combina-
tions, and allows to identify the source of bias.
The final benchmarks include around 1.6K ques-
tion–answer pairs for Legal and 2K for Travel. De-
tails of the DataMorgana configuration we used
to generate our benchmarks are provided in Ap-
pendix A.27.
4 Experiments
We present four experiments, each structured with
a description, results, and key conclusions. The
5https://uaelegislation.gov.ae/
6https://www.mofa.gov.ae/ar-ae/travel-updates
7The benchmarks will be made publicly available upon
publication of this paper.first experiment demonstrates that retrieval is a ma-
jor bottleneck in our bilingual setting. The second
identifies the root cause of the retrieval issues as the
cross-lingual mismatch between the user language
and the document language. The third experiment
attributes the performance drop to the presence of
a bilingual corpus. Finally, the fourth experiment
proposes and evaluates a mitigation strategy to ad-
dress this issue.
4.1 Retrieval is a Critical Bottleneck
Table 1 presents the results of our first experiment,
using the metrics described in Section 3.1. We first
measured accuracy without retrieval augmentation
for each benchmark. Then, for each of our two
embedding models, we evaluated the performance
of each system component as well as the overall
end-to-end performance.
Specifically, we report Hits@20 for the retrieval
phase. For reranking, we report Hits@5 only on ex-
amples where retrieval achieved Hits@20 equal to
1, meaning a passage with the answer was passed to
the reranker. For generation, we report answer ac-
curacy only on examples where reranking achieved
Hits@5 equal to 1, namely where a passage con-
taining the answer was included in the prompt. This
analysis helps identify how each phase contributes
to the overall end-to-end accuracy.
The Legal benchmark represents a domain-
specific setting, where questions involve niche top-
ics, so the LLM cannot rely on its parametric mem-
ory alone to answer them, as shown by the low
accuracy achieved without RAG. This is further
confirmed by comparing the end-to-end score in Ta-
ble 1 with the product of retrieval score, reranking
score conditioned on successful retrieval, and gen-
eration score conditioned on successful reranking.
These values are nearly identical, indicating that
the generation phase cannot compensate for fail-
ures earlier in the pipeline. The table shows similar
results for the Travel benchmark, although the over-
all accuracy for this case is slightly higher than the
product of the component-level conditional scores.
This is likely because the Travel corpus includes
less specialized knowledge, making it better repre-
sented in the LLM’s parametric memory, as also
reflected by the performance gap without retrieval.
Looking more closely at the individual compo-
nents, the reranker performs the best of the three.
For both benchmarks with the BGE-M3 embedder,
the probability of retrieval failure is comparable
to that of generation. With the M-E5 embedder,
4

Benchmark No-RAG Embedder Retrieval Reranking Generation End-to-End
Legal 27±3%BGE-M3 82±2% 89±2% 79±2% 60±2%
M-E5 67±2% 87±2% 79±3% 50±2%
Travel 37±3%BGE-M3 89±1% 97±1% 88±2% 80±2%
M-E5 76±2% 97±1% 85±2% 68±2%
Table 1: No-RAG baseline and RAG component-wise and end-to-end performance. For each benchmark, we
first report the baseline answer accuracy using only the user question without retrieval augmentation, referred to
as No-RAG. Then, for each embedding model, we report the retriever Hit@20, the reranker Hit@5 conditioned
on successful retrievals, the generation answer accuracy conditioned on successful rerankings, and the overall
end-to-end answer accuracy. Each value is presented with its 95% confidence interval.
the retrieval gap is even larger than the genera-
tion gap, showing a 12% difference on the Legal
benchmark and 9% on Travel. Moreover, for each
benchmark, reranking and generation performance
are stable across embedders. However, changing
retrievers has a significant effect on end-to-end ac-
curacy. These results, taken together, highlight that
the retriever is a critical bottleneck and motivate us
to focus our efforts on it.
4.2 Cross-Lingual Combinations are the Most
Challenging
Next, we compare the retrieval and end-to-end per-
formances on each of the four user-document lan-
guage combinations. The results for the BGE-M3
andM-E5 embedders are presented in Tables 2a
and 2b, respectively.
The tables reveal that cross-lingual scenarios,
where the user query and the supporting document
are in different languages, consistently underper-
form compared to same-language settings in both
retrieval and end-to-end performance. For the BGE-
M3embedder, a significant decline in retrieval per-
formance is observed only when the user language
is English and the document language is Arabic,
with drops of 33% in the Legal benchmark and
14% in Travel compared to the same-language con-
figuration. A similar pattern appears in the final
accuracy, with decreases of 38% and 12%, respec-
tively. Notably, the reverse cross-lingual setting
does not exhibit any statistically significant degra-
dation for BGE-M3 .
In contrast, the M-E5 embedding exhibits an
even larger performance drop across both cross-
lingual settings. Specifically, retrieval Hit@20 de-
creases by 42% on the Legal benchmark and by
33% on Travel, compared to their same-language
counterparts. These retrieval declines also propa-
gate to the end-to-end accuracy, resulting in dropsof 38% for Legal and 36% for Travel.
In what follows we dive deeper to discover the
cause behind this gap.
4.3 The Source of the Cross-Lingual Failure
Notice that in our current setup, referred to from
now on as the direct setting, multilinguality arises
from both the corpus and the queries. This cre-
ates two main challenges. First, the retriever must
handle cross-lingual queries, ranking documents in
language X given a query in language Y . Second,
it must rank documents across languages without
introducing bias toward either a high-resource lan-
guage or the user language.
To determine which of these challenges is pri-
marily responsible for the observed failures, we
conducted the following experiment. We modified
the retriever from the direct setting to search only
within the correct language. Specifically, for a doc-
ument language X, the language-oracle retriever
returns the top results exclusively in language X,
completely excluding language Y . Hence, while
the user query can still be in a different language,
all retrieved passages, and therefore all subsequent
stages of re-ranking and generation, operate ex-
clusively on monolingual documents. This setup
allows us to isolate the challenges of multilingual
query embeddings while avoiding issues related
to multilingual document embeddings. We stress
that the oracle is used only for analysis purposes,
since in practice we do not have access to the doc-
ument language ahead of time. The first two bars
in each subfigure of Figure 1 present the Hit@20
performance of the direct andlanguage-oracle re-
trievers, broken down by user-document language
combinations, as well as overall.
We observe two clear phenomena. First, the
language-oracle retriever achieves nearly identical
performance across all user and document language
5

BenchmarkUser
Lang.Doc
Lang.Retrieval
Hit@20End-to-End
Accuracy
LegalArabic Arabic 92±3% 67±5%
Arabic English 89±3% 68±4%
English Arabic 56±5% 30±5%
English English 86±3% 68±4%
Same-lang. 89±2% 68±3%
Cross-lang. 72±3% 49±3%
TravelArabic Arabic 93±2% 85±3%
Arabic English 91±2% 78±4%
English Arabic 79±4% 72±4%
English English 93±2% 84±3%
Same-lang. 93±2% 84±2%
Cross-lang. 85±2% 75±3%
(a)BGE-M3 embedderBenchmarkUser
Lang.Doc
Lang.Retrieval
Hit@20End-to-End
Accuracy
LegalArabic Arabic 88±4% 67±5%
Arabic English 51±4% 38±4%
English Arabic 41±5% 22±4%
English English 88±3% 70±4%
Same-lang. 88±2% 68±3%
Cross-lang. 46±3% 30±3%
TravelArabic Arabic 90±3% 86±3%
Arabic English 55±4% 38±4%
English Arabic 64±4% 62±4%
English English 95±2% 86±3%
Same-lang. 92±2% 86±2%
Cross-lang. 59±3% 50±3%
(b)M-E5 embedder
Table 2: Performance across language combinations. Results are presented for each embedder, benchmark and
for each of the four possible user–document language combinations. In addition, we report same-language and
cross-language scores, defined as the mean scores over combinations where the user and document languages match
or differ, respectively.
pairs, suggesting there are essentially no failures re-
lated to the query embeddings. In contrast, the gap
between the direct andlanguage-oracle retrievers
can be substantial in many cross-lingual cases. This
indicates that the main source of failure lies in the
multilingual document embeddings, specifically
the retriever’s ability to rank documents across lan-
guages.
The results suggests that while semantic simi-
larity is well captured within a single language,
the embeddings struggle in cross-lingual settings.
For instance, BGE-M3 appears to favor English
passages when the user query is in English, while
M-E5 may exhibit a tendency to prefer passages in
the same language as the user query.
4.4 Mitigating Cross-lingual Failings
These results raise an important question: can
multilingual retrievers be used reliably on mixed-
language corpora without further tuning? To ex-
plore this question, we introduce the balanced re-
triever, which enforces an equal selection of doc-
uments from each language by retrieving 10 pas-
sages in Arabic and 10 in English. We evaluate this
approach in the same experimental setup described
earlier. The third and final bar in each subfigure
of Figure 1 presents its results. Notice that while
thelanguage-oracle retriever is infeasible, it serves
as an upper bound for what the balanced retrieval
could achieve. In practice, we observe that the
balanced retriever does not significantly degrade
performance on same-language cases while provid-
ing substantial improvements in cross-lingual cases.
Notably, the balanced retriever yields more con-sistent results across the different combinations of
user and document languages, unlike the direct set-
ting, which favors the same-language combinations
at the expense of cross-language ones. Moreover,
this strategy leads to a significant improvement in
overall retrieval accuracy across benchmarks and
embedders, with consistent gains of around 3-5%
forBGE-M3 and approximately 20% for M-E5 .
These findings suggest that even simple strate-
gies can help mitigate retrieval biases, indicating
that debiasing is feasible despite inherent biases in
the embedding model.
5 Conclusions
This work highlights retrieval as a critical bot-
tleneck in multilingual RAG systems applied to
domain-specific corpora. While prior studies have
identified and focused on generation as the main
limitation in cross-lingual RAG, their conclusions
are primarily based on Wikipedia-derived bench-
marks. Since multilingual retrievers such as BGE-
M3 and multilingual-E5-large are trained on sim-
ilar open-domain data, they exhibit strong perfor-
mance in those settings. In contrast, our domain-
specific benchmarks expose substantial retrieval
weaknesses that remain obscured in such evalua-
tions, underscoring the need to revisit cross-lingual
retrieval in practical, real-world RAG scenarios.
Our analysis shows that performance degrades
most in cross-lingual settings where the user and
document languages differ, with drops of 30–50%
compared to same-language configurations. Using
an oracle retriever restricted to the correct language,
6

(a) Legal benchmark – BGE-M3 embedder
Ar QueryAr Doc.Ar Query
En Doc.En QueryAr Doc.En QueryEn Doc.Overall405060708090100Hit@20 (%)
Direct
Oracle
Balanced (b) Legal benchmark – M-E5 embedder
Ar QueryAr Doc.Ar Query
En Doc.En QueryAr Doc.En QueryEn Doc.Overall405060708090100Hit@20 (%)
Direct
Oracle
Balanced
(c) Travel benchmark – BGE-M3 embedder
Ar QueryAr Doc.Ar Query
En Doc.En QueryAr Doc.En QueryEn Doc.Overall405060708090100Hit@20 (%)
Direct
Oracle
Balanced (d) Travel benchmark – M-E5 embedder
Ar QueryAr Doc.Ar Query
En Doc.En QueryAr Doc.En QueryEn Doc.Overall405060708090100Hit@20 (%)
Direct
Oracle
Balanced
Figure 1: Retrieval Hit@20 scores across benchmarks and embedders. Each figure corresponds to a specific
combination of benchmark and embedding. Bars represent retrieval Hit@20 scores in percentages, with 95%
confidence intervals shown as black error lines. Different retrieval policies are distinguished by color and texture.
Results are grouped by benchmark segments defined by the user-document language combination, as well as the
overall benchmark retrieval performance.
we isolate the primary source of failure as the re-
triever’s difficulty in ranking documents across lan-
guages. That is, while the retriever performs well
within a single language, it struggles when com-
paring passages across languages, often favoring
those in the query’s language. We further observe
that different embedders exhibit weaknesses in dif-
ferent cross-lingual settings. This highlights the
potential to improve training by explicitly target-
ing cross-lingual robustness and narrowing the gap
with same-language performance.
Lastly, we demonstrate that a simple mitigation
of retrieving a balanced number of documents per
language, can significantly improve cross-lingual
performance and even improve overall results. This
finding points to meaningful opportunities for miti-gating multilingual retrieval biases, particularly in
real-world applications. However, applying such a
balancing approach in practical settings with non-
uniform language distributions or more than two
languages remains an open challenge and requires
further investigation.
Acknowledgments We thank our colleagues at
AI71, and in particular Abdelrahman Ibrahim, Amr
Ali Abugreedah, Mohamad Salah, Saleem Hamo,
Kirollos Sorour, Imran Moqbel, and Anas AlHelali,
whose native proficiency in Arabic was instrumen-
tal in the annotation process used to validate the
evaluation metrics in our pipeline.
7

References
Akari Asai, Jungo Kasai, Jonathan H. Clark, Kenton
Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2021a.
XOR QA: Cross-lingual Open-Retrieval Question
Answering. arXiv preprint . ArXiv:2010.11856.
Akari Asai, Xinyan Yu, Jungo Kasai, and Hanna Ha-
jishirzi. 2021b. One question answering model for
many languages with cross-lingual dense passage re-
trieval. Advances in Neural Information Processing
Systems , 34:7547–7560.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. In Findings of the Associa-
tion for Computational Linguistics ACL 2024 , pages
2318–2335.
Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault
Formal, Stéphane Clinchant, and Vassilina Nikoulina.
2024. Retrieval-augmented generation in multi-
lingual settings. In Proceedings of the 1st Work-
shop on Towards Knowledgeable Language Models
(KnowLLM 2024) , pages 177–188.
Simone Filice, Guy Horowitz, David Carmel, Zohar
Karnin, Liane Lewin-Eytan, and Yoelle Maarek.
2025. Generating diverse QA benchmarks for
RAG evaluation with DataMorgana. arXiv preprint .
ArXiv:2501.12789.
Petra Galuš ˇcáková, Douglas W Oard, and Suraj Nair.
2021. Cross-language information retrieval. arXiv
preprint arXiv:2111.05988 .
Katharina Hämmerl, Jind ˇrich Libovick `y, and Alexander
Fraser. 2024. Understanding cross-lingual alignment–
a survey. arXiv preprint . ArXiv:2404.06228.
Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian
Riedel, and Holger Schwenk. 2020. MLQA: Evalu-
ating Cross-lingual Extractive Question Answering.
InProceedings of the 58th Annual Meeting of the As-
sociation for Computational Linguistics , pages 7315–
7330, Online. Association for Computational Lin-
guistics.
Wei Liu, Sony Trenous, Leonardo FR Ribeiro, Bill
Byrne, and Felix Hieber. 2025. Xrag: Cross-lingual
retrieval-augmented generation. arXiv preprint .
ArXiv:2505.10089.
Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.
Mkqa: A linguistically diverse benchmark for mul-
tilingual open domain question answering. Transac-
tions of the Association for Computational Linguis-
tics, 9:1389–1406.
Antoine Louis, Vageesh Kumar Saxena, Gijs van Dijck,
and Gerasimos Spanakis. 2025. Colbert-xm: A mod-
ular multi-vector representation model for zero-shot
multilingual information retrieval. In Proceedings of
the 31st International Conference on Computational
Linguistics , pages 4370–4383.Jeonghyun Park and Hwanhee Lee. 2025. Investigating
Language Preference of Multilingual RAG Systems.
arXiv preprint . ArXiv:2502.11175.
Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2025.
On the Consistency of Multilingual Context Uti-
lization in Retrieval-Augmented Generation. arXiv
preprint . ArXiv:2504.00597.
Leonardo Ranaldi, Barry Haddow, and Alexandra Birch.
2025. Multilingual retrieval-augmented genera-
tion for knowledge-intensive task. arXiv preprint .
ArXiv:2504.03616.
Nikhil Sharma, Kenton Murray, and Ziang Xiao. 2025.
Faux polyglot: A study on information disparity in
multilingual large language models. In Findings
of the Association for Computational Linguistics:
NAACL 2025 .
Ian Soboroff. 2024. Don’t use llms to make relevance
judgments. arXiv preprint . ArXiv:2409.15133.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Multilin-
gual e5 text embeddings: A technical report. arXiv
preprint arXiv:2402.05672 .
Suhang Wu, Jialong Tang, Baosong Yang, Ante Wang,
Kaidi Jia, Jiawei Yu, Junfeng Yao, and Jinsong Su.
2024. Not all languages are equal: Insights into
multilingual retrieval-augmented generation. arXiv
preprint arXiv:2410.21970 .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Advances in Neural Information Processing Systems ,
volume 36, pages 46595–46623. Curran Associates,
Inc.
A Appendix
A.1 Metric Evaluation
A.1.1 Answer Accuracy
To validate our answer accuracy metric across lan-
guages, we performed the following procedure.
First, we used samples from English and Arabic
editions of Wikipedia to construct two benchmarks
of 100 examples each, using DataMorgana (Filice
et al., 2025). We then applied a standard RAG
pipeline to generate answers using Falcon-3-10B.
The generated answers were then compared to ref-
erence answers using our LLM-as-a-judge-based
accuracy metric, as described in the main text. This
setting was intentionally selected to produce a mix
of correct and incorrect answers, ensuring a mean-
ingful evaluation of the metric.
8

End-to-End Accuracy
Retrieval Hit@20 BGE-M3 M-E5
0 10±3% 9±2%
1 79±2% 79±3%
Overall 60±2% 50±2%
Table 3: End-to-End Accuracy as Function of Our
LLM-based Hit@20. Each cell shows the average ac-
curacy along with its 95% confidence interval. Columns
correspond to retrieval embedders; rows indicate evalu-
ation segments: instances with Hit@20 = 0, Hit@20 =
1, and overall accuracy.
Independently, human annotators who are native
speakers of the respective languages were asked
to assess the similarity between the generated and
reference answers. Annotators were instructed to
label each pair as matching, not matching, or de-
bated. Our evaluation revealed that for the English
benchmark, in 95% of the cases (with 95% confi-
dence interval of ±5%) the annotators agreed with
the automatic metric (or found the matching as
debatable). For Arabic, the agreement (or debat-
able) rate was 98% ( ±2%). Hence, the annotations
corroborate the validity of the automated accuracy
metric.
A.1.2 Retrieval Hit@20
Now that we trusted our LLM-based accuracy met-
ric, we moved to validating whether our Hits@20
metric, which also uses LLM judgments, effec-
tively captures success in the retrieval step. To-
wards this goal, we analyzed the downstream accu-
racy as a function of the Hit@20 score. This anal-
ysis focused on the Legal benchmark, where the
no-RAG accuracy is relatively low (27%), making
it easier to observe the impact of retrieval quality.
Table 3 reports these results for the BGE-M3 and
M-E5 embedders.
As shown in Table 3, downstream accuracy was
indeed low when the Hits@20 metric indicates fail-
ure, confirming that our LLM-based Hits@20 re-
liably identifies cases where retrieval has failed.
Specifically, accuracy dropped to approximately
9% when no relevant passage was identified by
the metric, which is considerably lower than the
27% accuracy observed without retrieval augmenta-
tion. Furthermore, we observed consistent patterns
across retrievers: although the BGE-M3 retriever
differed markedly in overall quality from the M-E5
retriever, their downstream accuracy as a functionof retrieval quality showed only minor differences,
likely attributable to statistical noise. These find-
ings validate our Hits@20 metric as a reliable mea-
sure of retrieval effectiveness, demonstrating that
higher scores are strongly associated with improved
downstream accuracy.
A.2 DataMorgana Configuration
The following describes the configuration used to
construct both the Legal and Travel benchmarks.
In both cases, DataMorgana was configured in non-
conversational mode, supporting single-turn ques-
tion answering only.
DataMorgana allows the definition of multiple
parallel question categorizations, each selected in-
dependently of the rest of the configuration, includ-
ing other categories and the document language.
The question categorizations were defined as fol-
lows:
•Language: The user language was set to Ara-
bic in 50% of the cases and English in 50%.
•Formulation: The question was phrased as:
–Concise natural language: 40% of cases.
–Verbose natural language: 20% of cases.
–Short search query: 25% of cases.
–Long search query: 15% of cases.
•Linguistic similarity: In 50% of the cases,
the phrasing was similar to that found in the
corpus, and in the remaining 50%, it had a
greater linguistic distance.
•Question type: Questions were evenly
split between factoid (50%) and open-ended
(50%).
•User need:
–For the Legal benchmark, 50% of the
questions simulated a user seeking spe-
cific legal advice, while the other 50%
simulated a user asking out of general
curiosity.
–For the Travel benchmark, the user type
was distributed as follows: UAE user in
20% of the cases, Non-UAE user in an
additional 30%, and Undisclosed citizen-
ship in the remaining 50%.
9

A.3 Prompts
In this section, we provide all the prompts used in
our experiments. Prompt A.1 was used for answer
generation. It is based on the guidelines proposed
by Chirkova et al. (2024) for prompting RAG sys-
tems in multilingual scenarios. Prompt A.2 was
used to evaluate the accuracy of the generated an-
swer. Prompt A.3 was used to evaluate retrieval
Hit@20 and reranking Hit@5.
Prompt A.1: RAG generation
System. Answer the question based on the
given passages below.
Elaborate when answering, and if applicable
provide additional helpful information from
the passages and only from the passages. Do
not refer to the passages, just state the infor-
mation.
You MUST answer in the SAME LAN-
GUAGE as the QUESTION LANGUAGE,
regardless of the language of the passages.
Answering in the same language as the user
is asking their question is crucial to your suc-
cess. If the question is in English, the answer
must also be in English. If the question is in
Arabic, the answer must also be in Arabic.
Write all named entities in the same language
and same alphabet as the question language.
User. # Passages:
passage 1:
<Passage 1>
passage 2:
<Passage 2>
passage 3:
<Passage 3>
...
# Question: <Question>
Prompt A.2: Generated answer evaluation
Based on the question and the golden answer,
judge whether the predicted answer has
the same meaning as the golden answer.
Return your answer in the following format:
<same_meaning>True/False</same_meaning>.
<question> ... </question>
<golden_answer> ... </golden_answer>
<predicted_answer> ... </predicted_answer>Prompt A.3: Retrieval evaluation
You are given a **question**, a **ground
truth answer**, and a list of **passages**.
Your task is to return the **list of passage in-
dices** that can directly answer the question
**by containing the ground truth answer**
(i.e., the passage includes a perfect match
to the information expressed in the ground
truth).
Please follow these rules:
- A passage should be included only if it
**clearly expresses or contains the ground
truth answer**.
- Do **not include passages** that are only
loosely related or provide background infor-
mation.
- Your response **must be valid Python list
syntax**, e.g., [3, 5, 9].
- Do **not add any explanation** outside the
list.
—
**Question**: <Question>
**Ground Truth Answer**: <Answer>
**Passages**: Passage 1: <Passage 1 con-
tent>
Passage 2: <Passage 2 content>
Passage 3: <Passage 3 content>
...
10
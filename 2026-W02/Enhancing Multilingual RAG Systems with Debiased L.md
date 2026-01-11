# Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion

**Authors**: Jeonghyun Park, Byeongjeong Kim, Seojin Hwang, Hwanhee Lee

**Published**: 2026-01-06 12:01:56

**PDF URL**: [https://arxiv.org/pdf/2601.02956v1](https://arxiv.org/pdf/2601.02956v1)

## Abstract
Multilingual Retrieval-Augmented Generation (mRAG) systems often exhibit a perceived preference for high-resource languages, particularly English, resulting in the widespread adoption of English pivoting. While prior studies attribute this advantage to the superior English-centric capabilities of Large Language Models (LLMs), we find that such measurements are significantly distorted by structural priors inherent in evaluation benchmarks. Specifically, we identify exposure bias and a gold availability prior-both driven by the disproportionate concentration of resources in English-as well as cultural priors rooted in topic locality, as factors that hinder accurate assessment of genuine language preference. To address these biases, we propose DeLP (Debiased Language Preference), a calibrated metric designed to explicitly factor out these structural confounds. Our analysis using DeLP reveals that the previously reported English preference is largely a byproduct of evidence distribution rather than an inherent model bias. Instead, we find that retrievers fundamentally favor monolingual alignment between the query and the document language. Building on this insight, we introduce DELTA (DEbiased Language preference-guided Text Augmentation), a lightweight and efficient mRAG framework that strategically leverages monolingual alignment to optimize cross-lingual retrieval and generation. Experimental results demonstrate that DELTA consistently outperforms English pivoting and mRAG baselines across diverse languages.

## Full Text


<!-- PDF content starts -->

Enhancing Multilingual RAG Systems with Debiased Language
Preference-Guided Query Fusion
Jeonghyun Park, Byeongjeong Kim, Seojin Hwang, Hwanhee Lee*
Department of Artificial Intelligence, Chung-Ang University
{tom0365, michael97k, swiftie1230, hwanheelee}@cau.ac.kr
Abstract
Multilingual Retrieval-Augmented Generation
(mRAG) systems often exhibit a perceived pref-
erence for high-resource languages, particu-
larly English, resulting in the widespread adop-
tion of English pivoting. While prior studies
attribute this advantage to the superior English-
centric capabilities of Large Language Mod-
els (LLMs), we find that such measurements
are significantly distorted by structural priors
inherent in evaluation benchmarks. Specifi-
cally, we identifyexposure biasand agold
availability prior—both driven by the dis-
proportionate concentration of resources in
English—as well ascultural priorsrooted
in topic locality, as factors that hinder accu-
rate assessment of genuine language prefer-
ence. To address these biases, we propose
DeLP(DebiasedLanguagePreference), a cali-
brated metric designed to explicitly factor out
these structural confounds. Our analysis us-
ing DeLP reveals that the previously reported
English preference is largely a byproduct of
evidence distribution rather than an inherent
model bias. Instead, we find that retrievers fun-
damentally favor monolingual alignment be-
tween the query and the document language.
Building on this insight, we introduceDELTA
(DEbiasedLanguage preference–guidedText
Augmentation), a lightweight and efficient
mRAG framework that strategically lever-
ages monolingual alignment to optimize cross-
lingual retrieval and generation. Experimental
results demonstrate that DELTA consistently
outperforms English pivoting and mRAG base-
lines across diverse languages.
1 Introduction
Multilingual Retrieval-Augmented Generation
(mRAG) (Chirkova et al., 2024) generalizes
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) by retrieving evidence from multilin-
gual knowledge sources. This enables language
*Corresponding Author.
What is Multilingual RAG?
The number of ﬁles with gold passages
English queries are advantageous한국의�전통�가옥은�무엇인가? 
(What is the korean term forkorean traditional houses?)
Only Korean Documents
have the evidence
Only Thai Documents
have the evidence(What is food served inplastic bags for takeout called in Thai culture?)Structural Priors in Multilingual RAG
Gold-availability Prior Cultural Prior: Topic Locality
NoFile20 Files
Found1 File
FoundNoFileNoFile
General Topic
Probability EN
Local
Doc
ENLocal
Doc
Cultural Topic
Probability Query
TopicFigure 1: Common causes of language preference of
mRAG: gold-availability prior and cultural prior.
models to produce responses that are not only fac-
tually grounded but also sensitive to the user’s
language and linguistic context (Rau et al., 2024).
In this landscape, mRAG systems frequently ex-
hibit a significant language preference for En-
glish (Zhang et al., 2023; Park and Lee, 2025).
Consequently, English pivoting—the practice of
translating a non-English query into English be-
fore retrieval—has emerged as a surprisingly strong
heuristic that yields substantial gains across many
languages (Chirkova et al., 2024; Ranaldi et al.,
2025). Prior works have largely attributed this ad-
vantage to the "English-centric" competence of gen-
erators, such as superior reasoning in English or re-
duced translation noise, leading to research that pri-
marily intervenes at the generation stage (Chirkova
et al., 2024; Li et al., 2025; Moon et al., 2025).
However, we find that the perceived effective-
ness of English pivoting is primarily driven by
retrieval-side structural biases rather than any in-
herent linguistic preference of the model. As il-
lustrated in Figure 1, we identify two major con-
founders: agold availability priorandcultural
priors. Our analysis shows that ground-truth evi-
dence in standard benchmarks is overwhelmingly
concentrated in English resources, establishing a
dominant gold availability prior. This concentration
not only makes English the sole or primary location
where correct evidence exists, but also leads toex-
1arXiv:2601.02956v1  [cs.CL]  6 Jan 2026

posure biasof English documents during retrieval,
further amplifying English’s advantage. Together,
these effects fundamentally distort measured lan-
guage preference and inflate the apparent superi-
ority of English. In addition, we identifycultural
priorsas an equally critical factor. There are bench-
mark questions that are tied to specific geographic
or cultural contexts and contain native surface
forms (e.g., local titles, aliases, and scripts) that act
as strong retrieval anchors. When benchmarks over-
represent such locale-specific topics, languages
may appear “preferred” due to query–corpus align-
ment and environmental exposure rather than the
model’s intrinsic preference. Critically, these struc-
tural factors contaminate existing methods (Park
and Lee, 2025) for measuring language preference.
To reveal the intrinsic preference of mRAG sys-
tems, we propose Debiased Language Preference
(DeLP), a calibrated measurement that explicitly
regresses out these structural confounds. DeLP
utilizes a ridge regression framework to predict
observed language preference from structural fac-
tors (e.g., corpus size, gold availability, cultural
prior), treating the residual signal as the true, debi-
ased preference of the model. By applying DeLP,
we reveal a qualitatively different landscape: the
previously inflated preference for English largely
evaporates. Instead, our results show an increased
preference formonolingual alignment, where the
retriever performs most effectively when the query
and the target document languages match.
Building on the discovery of monolingual
alignment, we introduce DEbiased Language
preference–guided Text Augmentation (DELTA),
a lightweight query-level solution for mRAG.
DELTA leverages the debiased preference signals
from DeLP to dynamically identify intrinsic model
preference for a given query, effectively bridging
the gap between the user’s query and the languages
where the model performs most reliably. By re-
formulating the query to include these preference-
aligned multilingual anchors, DELTA preserves the
native script’s context while maximizing the ben-
efits of monolingual alignment. DELTA is highly
cost-effective, requiring no modifications to the
underlying corpus or retriever architecture. Our ex-
periments demonstrate that DELTA outperforms
naive English pivoting, proving that accounting for
the model’s true linguistic preference—rather than
following biased environmental cues—is the key
to unlocking the true potential of mRAG systems.2 The Myth of English Preference:
Structural Priors in mRAG
A dominant mRAG strategy isEnglish pivoting,
where non-English queries are translated into En-
glish to exploit the perceived superiority of English-
centric models. We hypothesize that these gains are
not necessarily indicative of model preference but
rather reflect a massive exposure bias rooted in the
structural distribution of evidence.
2.1 Experimental Setup
DatasetsWe conduct our analysis on
MKQA (Longpre et al., 2021), which pro-
vides 10k professionally translated queries. To
enable precise measurement of evidence location,
we use a 2.7K-example subset that overlaps with
KILT NQ1. Since MKQA does not provide stan-
dardized provenance for each translated instance,
using KILT allows us to inherit document-level
provenance (i.e., gold Wikipedia passage IDs),
which is essential for quantifying gold availability
across different linguistic corpora (Lewis et al.,
2020; Chirkova et al., 2024).
Models and Knowledge SourcesWe employ
BGE-m3 (Chen et al., 2024) as the multilin-
gual retriever and re-ranker. For the generation,
we use three recently released robust multilin-
gual LLMs: Qwen3-235B (Yang et al., 2025),
DeepSeek-v3.1 (Liu et al., 2024), and Gemini-2.5-
Flash (Comanici et al., 2025). We retrieve top-
50 candidate documents per query and apply re-
ranking, using the top-5 documents as contexts for
generation. In line with previous work (Chirkova
et al., 2024; Park and Lee, 2025), we use Wikipedia
editions in English and the user’s local language
to serve as the knowledge sources. Detailed corpus
statistics are provided in Appendix D.
2.2 Linguistic Superiority or Data Imbalance?
Following the MKQA protocol anchored to
KILT (Lewis et al., 2020; Chirkova et al., 2024),
we identify the location of gold passages (WPIDs)
within the multilingual Wikipedia datastore. We
report this distribution as Gold Availability, as the
number of queries whose gold passage WPID is
present in that language’s Wikipedia corpus for
each language of query. This distribution reflects
the extent of corpus-level coverage of gold evi-
dence in each language within the benchmark. We
1https://huggingface.co/datasets/facebook/
kilt_tasks
2

Gold Availability Retriever Recall Qwen3-235B-A22B Gemini-2.5-Flash DeepSeek-Chat-v3.1
lang #q ratio Base EN Base EN Base EN Base EN
en 26934 73.29% – – 70.05( EN )– 58.26( EN )– 60.77( EN )–
ar 214 0.58% 13.36 23.57 47.79 55.14 40.79 48.44 43.64 50.97
de 435 1.18% 21.62 26.40 63.81 60.72 53.52 55.17 54.16 56.92
ja 513 1.40% 16.84 25.83 46.60 59.29 44.26 53.68 44.72 56.52
ko 306 0.83% 15.62 24.81 40.14 54.57 35.97 47.67 34.21 50.49
th 187 0.51% 21.90 27.55 40.73 60.46 31.65 54.86 36.80 57.52
zh 287 0.78% 16.47 26.53 37.52 59.53 30.81 53.59 33.14 56.11
Table 1: Gold availability bias and its impact on multilingual RAG. Gold Availability measures gold-passage
coverage per language, and Retriever Recall reports Recall@50. Model columns show end-to-end accuracy. Base
denotes native-language queries, and EN denotes English-translated queries.
then relate this to retrieval performance, measured
by Recall@50—i.e., the fraction of queries whose
gold passage appears within the top-50 retrieved
candidates—under both native-language queries
and English queries (EN). We further evaluate end-
to-end mRAG performance using character 3-gram
recall between the generated and reference answers.
Details are provided in Appendix H.
Our analysis in Table 1 reveals an extreme
imbalance in the retrieval environment. English
Wikipedia provides substantially higher document
density and coverage, introducing a strongexpo-
sure bias. More critically, for a vast majority of
queries, English Wikipedia also serves as the sole
repository of ground-truth, inducing a dominant
gold availability prior. Consequently, English piv-
oting appears effective not because models prefer
English, but because of this structural skew, sustain-
ing the long-standing "myth" of English preference.
ar de es fr ja ko ru zh
21.43% 16.67% 18.52% 21.74% 14.29% 12.50% 25.00% 6.67%
Table 2: Local-gold coverage by predictedL loc.
2.3 Impact of Cultural Priors
Queries often carry cultural or regional context,
whose associated language can naturally align with
local-language evidence and be conflated with lan-
guage preference; we therefore examine where
their gold documents reside across Wikipedia lan-
guages. We first isolate queries that involve cul-
tural or regional contexts by instructing GPT-
4o-mini (Hurst et al., 2024) to predict a single
primary locale language Lloc, selecting the lan-
guage that corresponds to the query’s main ref-
erenced region or culture (Details for classifier is
in Appendix J). Table 2 reports the local-gold rate
p(gold WPID exists in local Wikipedia|L loc)for
each predicted language. Local evidence is not uni-
formly absent—across several predicted locale lan-guages, about 20% of these queries have gold pages
only in the corresponding local Wikipedia. This dis-
tribution introduces a structural bias for retrieval to
rely on locale-specific surface-form anchors (e.g.,
native titles, aliases, scripts) when they exist. As
a result, observed language preference can be in-
fluenced by locale-tied queries and their local-gold
presence, motivating an explicit cultural prior term
pcultto avoid conflating topic locality.
3 Measuring Language Preference via
Bias Calibration
The structural priors identified in Section 2 sug-
gest that existing metrics, such as MLRS (Park
and Lee, 2025), fail to distinguish between a
model’s intent to use a language and the external
necessity imposed by data distribution. To reveal
the genuine preference, we introduceDebiased
LanguagePreference (DeLP), a calibrated mea-
surement framework that explicitly regresses out
structural confounds.
3.1 Decomposing Structural Bias in mRAG
To isolate intrinsic model preference, we decom-
pose the confounding factors identified in our pre-
vious analysis into three primary priors.
Exposure prior ( pret).As observed in the ex-
posure bias of Section 2.2, high-resource cor-
pora—particularly English—dominate the top re-
trieval results regardless of the encoder’s linguistic
intent. This prior captures the "popularity bias" of
the datastore. A language that appears more fre-
quently in the candidate pool is more likely to be
retrieved, potentially leading to a false inflation
of preference. Let Lqdenote the query language
andLdthe language of a retrieved document. We
estimate pret(Ld|Lq)by calculating the average
proportion of document language Ldwithin the
top-50 candidates for queries inL q.
3

STAGE �:
Observed / Biased
Preference
Exposure prior (pret)
Gold-availabillity prior (pgold)
Cultural prior (pcult)Observed language
preference (raw)
Exposure / corpus skew+ Auxiliary
Priors   
Gold evidence concentration
Topic locality / native anchorsRegression with prior vector
& Residual calculationDocument languageQuery language
Strongly correlated
with structural priorsSTAGE �:
Structural PriorsSTAGE �:
DeLP CalibrationSTAGE �:
Debiased / Calibrated
Preference
Debiased language
preference (DeLP)
Document languageQuery language
Intrinsic preference after calibration
Monolingual alignment emergesRaw
PerformancePrior VectorFigure 2: Overview of DeLP, which measures intrinsic language preference in mRAG by regressing out exposure,
gold-availability, and cultural priors from raw preference signals.
Gold-availability prior ( pgold).Our findings in
Table 1 (Section 2.2) demonstrate that retrieval
is often forced into English because the gold evi-
dence simply does not exist elsewhere. To prevent
such uncontrollable circumstances from being mis-
taken for model preference, we explicitly model the
availability of ground-truth passages. We estimate
pgold(Lq, Ld)on the MKQA–KILT overlap as the
empirical fraction of queries in Lqfor which gold
evidence is present in theL dcorpus.
Cultural prior ( pcult).As discussed in Sec-
tion 2.3, locale-tied queries contain native sur-
face forms that act as structural anchors, natu-
rally pulling retrieval toward the corresponding
local-language evidence. This prior captures topic
locality; a retriever might show high preference
scores for a language simply because the topic
is regional. We estimate pcult(Ld)by identifying
the query’s associated locale language Llocusing
GPT-4o-miniand computing the fraction of queries
where Lloc=L d. The classifier selects the local
language of the primary referenced place or culture
(e.g., "When did Hong Kong go back to China?" →
zh), reserving en for inherently English-speaking or
genuinely global queries. For the details of how we
measure cultural prior, please refer to Appendix J.
3.2 Calibration for Genuine Preference
We calibrate the raw language preference scores
with respect to the above priors to obtain the resid-
ual signal-the component not explained by the pri-
ors, which reflects the model’s genuine language
preference. In addition to the above three main
priors, we incorporate two auxiliary structural con-
trols, namely a corpus-size prior pdband a passage-
length statistic ℓ, as additional covariates in the
prior feature vector ϕ(Lq, Ld)(defined below) to
account for language-dependent corpus scale andlength effects. Let us se(Lq, Ld)denote the ob-
served language-preference score of the encoder
efor the query language Lq, and for the language
of evidence Ld. We instantiate seby MLRS (Park
and Lee, 2025) (Table 10), which measures how
often the retriever surfaces evidence in each Ldfor
a fixed Lq. For each language pair (Lq, Ld), we
define a prior feature vectorϕ(L q, Ld)∈R7:
ϕ(Lq, Ld) =
1
log 
pret(Ld|Lq) +ϵ
log 
pdb(Ld) +ϵ
log 
ℓ(Ld) +ϵ
log 
pgold(Lq, Ld) +ϵ
log 
pcult(Ld) +ϵ
I[Lq=Ld]
.(1)
where pretis the exposure prior, pdbis the
corpus-size prior, ℓis a passage-length statistic
(e.g., median length), pgoldis the gold-availability
prior, and pcultis the cultural prior. The vector
ϕ(Lq, Ld)stacks interpretable covariates that pre-
dictsewithout invoking intrinsic model preference.
We use log-transformed priors to compress heavy-
tailed probabilities and corpus statistics, and make
linear effects more reasonable across languages.
The indicator I[Lq=Ld]allows the model to treat
same-language retrieval as a special case, ensur-
ing that monolingual matching is not forced to be
explained solely by external priors.
Ridge calibration.We fit the regression sepa-
rately for each encoder eto learn how much of its
observed score secan be attributed to structural
priors. We use ridge regularization to stabilize co-
efficients under the various priors, preventing any
single feature from disproportionately absorbing
the preference signal. Let Cbe the set of all lan-
guage pairs used for calibration. For each encoder
e, we fit a ridge regression that predicts the raw
score from priors:
4

Query
Lang.Lq=L dLq ̸=L d
en ko zh fr ja it pt es
en 50.10(56.79) – 36.67(33.94) 40.34 (33.99) 35.57(37.57) 39.61(34.18) 35.49(36.79) 36.46(36.54) 36.02(37.49)
ko 43.38 (42.21) 37.69(44.36)– 41.75(35.44) 34.90(36.84) 43.59(38.22) 34.70(36.00) 35.65(35.71) 34.77(36.24)
zh 50.60(45.81) 38.68(45.35) 37.95(35.06)– 34.73(36.73) 41.90 (36.51) 34.87(36.21) 35.84(35.91) 35.22(36.69)
fr 40.05(43.74) 41.16(47.84) 36.77(34.03) 40.50 (34.16)– 39.92(34.50) 36.00(37.31) 36.69(36.76) 36.28(37.76)
ja 49.19(45.50) 38.70(45.37) 38.40(35.69) 41.56 (35.24) 34.94(36.94)– 34.99(36.29) 35.97(36.04) 35.23(36.70)
it 39.05(41.72) 40.63(47.30) 36.85(34.12) 40.59 (34.25) 36.63(38.64) 39.86(34.44)– 37.01(37.09) 36.91(38.39)
pt 46.08(39.76) 40.55(47.23) 36.98(34.24) 40.63 (34.29) 36.50(38.52) 40.01(34.59) 36.48(37.80)– 37.73(39.21)
es 38.19(41.30) 40.71(47.39) 36.86(34.13) 40.39 (34.04) 36.30(38.31) 39.76(34.34) 36.45(37.76) 37.25(37.32)–
Table 3: DeLP scores for query-document language pairs, averaged over three encoders. Raw MLRS scores are in
parentheses. Shading indicates row-wise min-max scaling (darker = stronger preference); dashes denote Lq=L d.
ˆβe= arg min
βJe(β),
Je(β) =X
(Lq,Ld)∈C 
se(Lq, Ld)−ϕ(L q, Ld)⊤β2
+λ∥β∥2
2.(2)
where λis a regularization hyperparameter and
ϵis a small constant for numerical stability.
Debiased preference (DeLP).We define the de-
biased preference as the residual signal after remov-
ing the component explained by structural priors:
re(Lq, Ld) =s e(Lq, Ld)−ϕ(L q, Ld)⊤ˆβe.(3)
The residual re(Lq, Ld)represents the portion of
the observed score that is independent of structural
priors. To keep the overall scale comparable to the
raw score, we re-center the residuals by the global
mean of raw scoresµ e:
DeLP e(Lq, Ld) =r e(Lq, Ld) +µ e,
µe=1
|C|X
(Lq,Ld)∈Cse(Lq, Ld).(4)
By adding back µe, DeLP stays on a numeric
scale comparable to standard MLRS tables while
preserving the relative differences that define the
model’s intrinsic tendencies. To mitigate potential
encoder-specific bias, we apply our calibration pro-
cedure independently to each retriever and report
all debiased results for three multilingual encoders:
BGE-m3 (Chen et al., 2024) and two Sentence-
BERT variants (Reimers and Gurevych, 2019),
paraphrase-multilingual-MiniLM-L12-v2
andparaphrase-multilingual-mpnet-base-v2 .
We denote them as p-mMiniLM andp-mMpNet for
compactness in tables.
Emergence of Monolingual Alignment.After
calibration, we find that the preference landscapeshifts qualitatively from the raw preference as in
Table 3. The previously dominant English prefer-
ence largely disappears, and the strongest signal
consistently moves to the diagonal ( Lq=Ld). This
reveals that retrievers fundamentally favor mono-
lingual alignment—the matching of query and doc-
ument in the same language. We also observe that
queries favor the linguistically or regionally related
languages, such as Korean with Japanese. Overall,
the DeLP score suggests that much of the apparent
English preference in prior protocols was induced
by structural priors, while the residual preference
signal is dominated by query-language alignment
and interpretable related-language effects. For a
more detailed DeLP score, refer to Appendix G.
pret pgold pcult
Encoder MLRS DeLP MLRS DeLP MLRS DeLP
bge-m30.994 0.142 0.914 0.336 0.916 0.335
p-mMiniLM0.997 0.145 0.915 0.321 0.917 0.320
p-mMpNet0.996 0.131 0.917 0.311 0.920 0.310
Table 4: Pearson’s rbetween preference and priors for
before (MLRS) and after (DeLP) calibration.
Correlation Analysis.To validate DeLP, we
compute the correlation between preference scores
and priors before and after calibration as shown in
Table 4. Raw scores (MLRS) are highly correlated
with all three priors (exposure, gold-availability,
and cultural), suggesting that existing language-
preference measurements largely reflect prior-
driven preference rather than intrinsic model pref-
erence. After calibration, these correlations drop
sharply after applying DeLP. This confirms that
DeLP effectively decouples intrinsic model tenden-
cies from the structural signals that contaminate
standard benchmarks.
5

boost
boost
boost
boostRetrieving
DocumentAnswer
GenerationFigure 3: Overview of DELTA query fusion. DELTA fuses global and local query segments into a single preference-
aligned query using lightweight repetition-based weighting.
4 Debiasing mRAG through
Preference-Aligned Augmentation
Monolingual alignment in Section 3 reveals thatre-
trievers intrinsically perform bestwhen thequery
language matches the document language. This
suggests that while English pivoting provides cov-
erage (due to gold availability), it often sacri-
fices the retrieval anchors present in the user’s na-
tive tongue. Motivated by this point, we propose
DELTA (DEbiasedLanguage preference guided
TextAugmentation), a lightweight query reformu-
lation strategy that injects preference-aligned cues
into a single fused query.
4.1 Query Fusion with Native Anchors
DELTA aims to maximize the benefits of both
global coverage and local discriminative matching.
As illustrated in Figure 3, given a local question
qlocal, DELTA constructs an English pivot qgloband
extracts a set of cultural identifiers—canonical ti-
tles(tglob, tloc), aliases, and a regional hint—using
a frozen LLM (Instruction in Appendix N). These
elements are then concatenated into a single query
Qfused composed of five segments, each optionally
weighted by cultural cues’ confidence score:
•[GLOB]: The English pivotq glob.
•[LOCAL] : The original query qlocalto leverage
monolingual alignment.
•[TITLE_BRIDGE] : Paired titles (tglob, tloc)to fa-
cilitate cross-lingual mapping.
•[ALIASES] and[LOCALE_HINT] : Specific identi-
fiers that serve as stable retrieval anchors.
4.2 Repetition-based Weighting
To implement the debiased preference control,
DELTA utilizes a repetition-based weighting pol-
icy(Wang et al., 2023). We first predict a cul-tural cue (y, c, L loc), where y∈ {0,1} indicates
whether the query is culture-specific, c∈[0,1] is
the confidence score, and Llocis the language of
the key native identifiers. This cue determines how
strongly we upweight locale-specific blocks versus
the global English back-off when forming the fused
query Qfused. We map the confidence score cinto
three discrete repetition levels using two thresholds
τlowandτhigh. We then set repetition counts for
the local block[LOCAL:L loc]and the global pivot
block[GLOB]as:
rlocal=(
1 +I[c≥τ low] +I[c≥τ high], y= 1
1, y= 0
rglob=(
1 +I[c < τ low], y= 1
2, y= 0(5)
c < τ lowtriggers no additional upweighting,
τlow≤c < τ highadds one extra repetition, and
c≥τ high adds two, yielding rlocal∈ {1,2,3}
andrglob∈ {1,2} . Intuitively, we upweight high-
confidence culture-specific queries toward the local
expression to preserve culturally grounded identi-
fiers, while non-culture-specific queries mildly fa-
vor the global pivot for robust back-off. In addition,
when y=1 andc≥τ boost , we duplicate local-side
disambiguation anchors (i.e., [TITLE_BRIDGE]
and[ALIASES] ) once more to further emphasize
native surface-form anchors and reduce entity ambi-
guity. Overall, DELTA realizes preference control
via text-only weighting, and all concrete hyperpa-
rameter values are reported in Appendix K.
5 Experiments
5.1 Experimental Setup
Baselines.(1)MultiRAG: Retrieve and rerank
from multilingual datastores using the original
MKQA query, then generate the answer in the
6

Methoden ar es zh ja de ko thA VG↑Latency↓
Qwen3-235b-a22b-2507
Document Level
MultiRAG 70.05 47.7963.7637.52 46.6063.8140.14 40.73 51.30 1.38
CrossRAG 68.21 43.95 61.14 37.81 44.75 60.16 38.13 42.87 49.63 1.29
DKM-RAG 69.13 42.69 62.12 35.13 43.90 61.13 39.49 38.88 49.06 3.80
QTT-RAG70.1146.44 63.02 37.68 46.94 62.79 44.13 42.12 51.65 1.80
Query Level
English Translation - 55.14 61.94 59.53 59.29 60.72 54.57 60.46 58.81 1.17
DELTA (ours) 63.8562.5563.03 62.59 62.3862.86 63.26 62.51 62.88 1.13
Gemini-2.5-flash
Document Level
MultiRAG 58.26 40.79 55.11 30.81 44.26 53.52 35.97 31.65 43.80 1.53
CrossRAG 63.40 41.87 57.24 29.74 44.14 56.80 36.09 32.49 45.22 2.60
DKM-RAG 64.21 39.4159.2631.34 43.4557.7437.26 33.64 45.79 5.63
QTT-RAG65.3242.64 57.81 31.56 45.18 56.27 40.65 35.97 46.93 5.55
Query Level
English Translation - 48.44 55.84 53.59 53.68 55.17 47.67 54.86 52.75 1.55
DELTA (ours) 56.9756.4555.9555.83 56.1855.9856.44 56.45 56.28 1.48
Deepseek-chat-v3.1
Document Level
MultiRAG 60.77 43.64 56.22 33.14 44.72 54.16 34.21 36.80 45.46 2.56
CrossRAG 67.83 48.34 62.24 39.05 49.27 61.33 39.85 45.70 51.70 2.64
DKM-RAG 67.84 44.0762.4937.63 45.6661.6540.30 40.38 50.00 2.39
QTT-RAG68.2846.13 61.81 37.24 47.36 60.48 41.06 41.29 50.46 1.93
Query Level
English Translation - 50.97 58.32 56.11 56.52 56.92 50.49 57.5255.26 2.05
DELTA (ours) 59.8559.4658.6159.67 59.0259.2553.5156.45 58.23 1.13
Table 5: Main results (end-to-end mRAG performance). We use bge-m3 (Chen et al., 2024) for retrieval, and evaluate
with character 3-gram recall (Chirkova et al., 2024).Best, second-best A VG (mean) are computed per generator and
language. We report generation time as latency.
query language. (Chirkova et al., 2024) (2)Cross-
RAG: Run the same multilingual retrieval as Multi-
RAG, translate the retrieved passages into a single
pivot language (English). (Ranaldi et al., 2025) (3)
DKM-RAG: Translate the reranked passages into
the query language, use an LLM to produce multi-
ple refined passages. (Park and Lee, 2025) (4)QTT-
RAG: Translate cross-lingual passages and attach
translation-quality tags so the generator can decide
which contexts to trust. (Moon et al., 2025) (5)En-
glish Translation: Translate the original query into
the global pivot language.
5.2 Results and Analysis
Main Results.Table 5 shows that DELTA
achieves the best average performance for each
generator and is comparable to, or better than,
document-level frameworks that require substan-
tially higher cost due to the document’s long con-
text length. The gains are particularly pronounced
on non-English queries, indicating that preference-
aligned query augmentation is more effective than
relying on document-side transformations in mul-
tilingual settings. DELTA provides little benefit
on English queries (the encolumn in Table 5) be-
cause the local query and the global pivot becomenearly identical when Lq=en . As a result, DELTA
injects redundant segments with repeated, overlap-
ping content, which unnecessarily lengthens the
query and can dilute the useful signal for retrieval,
resulting in no gain or even a slight degradation.
Statistic Value
Queries (N) 16,828
New-hit queries (N new) 1,235
Gold best-rank (mean) 10.39
Gold best-rank (median) 5
Top-10 rate 66.23%
Rank in [10,49] rate 35.14%
Rank in [40,50] rate 4.62%
Table 6: Retrieval rank analysis for new-hit queries,
which reports the rank distribution of gold passages
newly recovered by DELTA relative to English-pivot.
Analyzing Gold Passage Recall and Ranking.
To investigate how DELTA recovers missing gold
evidence to improve the overall mRAG system, we
compare its retrieval performance against English-
pivot retrieval across seven languages (ar, de, es,
ja, ko, th, zh), totaling 16,828 queries. As shown
in Table 6, while English pivoting provides cover-
age due to English-heavy gold availability, it often
degrades native surface-form anchors—such as ti-
tles, aliases, and original scripts—that are critical
7

for precise entity matching. DELTA restores these
anchors, facilitating better alignment with English
gold documents. Among 1,235 newly recovered
queries, DELTA achieves a mean best rank of 10.39
(median 5) with a 66.2% Top-10 entry rate. This in-
dicates that DELTA does not merely rescue missed
gold pages near the cutoff but significantly elevates
them to high-ranking, actionable positions.
Method ar de es ja ko th zh Avg
Orig 63.68 62.46 63.26 63.26 62.84 63.37 63.04 63.13
+Global 71.42 72.02 71.37 71.51 71.77 71.70 71.57 71.62
+Title 68.17 67.83 67.75 67.78 67.93 68.17 68.34 68.00
+Aliases 68.14 67.46 67.43 67.38 68.61 68.15 68.13 67.90
+Locale 67.57 67.63 67.78 67.89 67.81 67.65 67.44 67.68
All cues72.99 73.01 72.48 73.26 72.88 72.93 72.70 72.89
Table 7: Cue ablations for DELTA with fixed evidence.
We incrementally add query-side cues and report end-
to-end generation accuracy.
Impact of Cues on Evidence Interpretation.To
determine if the query of DELTA improves evi-
dence interpretation in the generation stage inde-
pendently of retrieval, we conduct a cue ablation
study under a fixed-evidence setting, which is re-
ported in Table 7. By keeping retrieved passages
constant and modifying only query-side cues, we
isolate generation-time effects. Even under this
setup, the global pivot cue significantly outper-
forms the original query, indicating that concise
global English paraphrasing aids the generator in
aligning evidence. Bridge cues also provide inde-
pendent gains, showing that locale hints and identi-
fiers improve the model’s ability to select precise
evidence spans within identical passages. The best
performance across all languages is achieved by
combining all cues, suggesting that bridge cues
offer critical disambiguation and entity grounding.
Latency Analysis.To assess the efficiency of
DELTA, we report average end-to-end latency in
the rightmost column of Table 5. DELTA maintains
high efficiency by generating a single fused query
and avoiding document translation. It can even be
faster than English Translation; by incorporating
local cues and disambiguation anchors, DELTA
enables direct retrieval, reducing the overhead of
processing overly generic English-only signals.
6 Related Works
6.1 Multilingual RAG
Prior work in mRAG has explored how perfor-
mance varies with the query language (Ranaldi
et al., 2025; Longpre et al., 2021), the languageof relevant or irrelevant evidence (Qi et al., 2025;
Wu et al., 2024), as well as document ordering and
prompting strategies that affect how models con-
sume multilingual contexts (Sharma et al., 2024;
Wu et al., 2024; Shankar et al., 2024; Ki et al.,
2025). A common and effective heuristic is pivot
translation, where non-English queries are trans-
lated into English before retrieval, often produc-
ing large gains (Asai et al., 2021; Ranaldi et al.,
2025). However, much of the existing analysis of
why pivot translation helps centers on the gener-
ation stage (e.g., English-centric generation com-
petence, translation noise, and cross-lingual drift),
which motivates generator-side interventions such
as translation-aware prompting or decoding-time
control (Sharma et al., 2024; Moon et al., 2025). In
contrast, our work focuses on a retrieval-side ex-
planation: we empirically show that gold evidence
is structurally skewed toward English corpora.
6.2 Language Preference
In mRAG, language preference is shown both in
retrieval (over-retrieving high-resource languages)
and in generation (differentially using evidence by
language even under matched relevance), degrad-
ing consistency and downstream quality (Park and
Lee, 2025). Existing measurements of language
preference in mRAG commonly rely on behavioral
proxies, such as comparing outputs across query
languages via information overlap (Sharma et al.,
2024) or embedding similarity to references (Park
and Lee, 2025), and, in more controlled settings,
analyzing citation or attribution behavior as evi-
dence that language varies while other variables
are fixed (Ki et al., 2025; Qi et al., 2025). While
prior approaches offer useful signals, they miss
a key confound in mRAG: structural priors can
dominate preference scores. We therefore debias
preference by regressing out these priors and using
the residual as the preference signal.
7 Conclusion
We demonstrate that gains from English pivoting
in mRAG stem from retrieval-side evidence imbal-
ance, which biases preference measurements. We
address this with DeLP, a debiased metric that cali-
brates structural priors to reveal preference shifts
toward the query language. Leveraging DeLP, we
introduce DELTA, a lightweight query reformula-
tion strategy that fuses global and local cues into a
single query, consistently outperforming baselines.
8

Limitations
First, our debiasing targets retriever-level prefer-
ence, while generator-level preference can still re-
main. Therefore, extending debiasing to how gen-
erators consume multilingual evidence is an impor-
tant direction for future work. Second, our conclu-
sions are drawn from a Wikipedia-based mRAG
setup. Evaluating DeLP and DELTA on broader,
domain-specific multilingual corpora is therefore
necessary to assess their generalizability. Third,
DELTA controls the balance between global and lo-
cal signals using simple repetition, which is coarse.
More precise and principled weighting or adaptive
control logic could further improve effectiveness
and stability.
Ethics Statement
We conduct our experiments using publicly avail-
able multilingual datasets, knowledge sources, and
models that are widely used in the research com-
munity and released under established data-sharing
and licensing guidelines. We follow the usage pro-
tocols and license agreements specified by the orig-
inal providers. While these resources are designed
to reduce harmful biases and inappropriate content,
they may still contain artifacts of data imbalance
and may not fully represent the diversity of lan-
guages, dialects, and cultural contexts. Our work
analyzes and mitigates retrieval-side evidence im-
balance and does not involve human subject data,
user interaction logs, or the collection of person-
ally identifiable information. We encourage future
deployments to consider downstream risks such as
uneven coverage across languages and potential
disparities in answer quality for under-resourced
communities.
Acknowledgments
This work was supported by the Institute of Infor-
mation & Communications Technology Planning
& Evaluation (IITP) grant funded by the Korea
government (MSIT) [RS-2021-II211341, Artificial
Intelligent Graduate School Program (Chung-Ang
University)]. This research was supported by the
Chung-Ang University Graduate Research Scholar-
ship in 2025.
References
Akari Asai, Jungo Kasai, Jonathan H Clark, Kenton
Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2021.Xor qa: Cross-lingual open-retrieval question answer-
ing. InProceedings of the 2021 conference of the
North American chapter of the association for com-
putational linguistics: human language technologies,
pages 547–564.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault
Formal, Stéphane Clinchant, and Vassilina Nikoulina.
2024. Retrieval-augmented generation in multi-
lingual settings. InProceedings of the 1st Work-
shop on Towards Knowledgeable Language Models
(KnowLLM 2024), pages 177–188, Bangkok, Thai-
land. Association for Computational Linguistics.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and
1 others. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Dayeon Ki, Marine Carpuat, Paul McNamee, Daniel
Khashabi, Eugene Yang, Dawn Lawrie, and Kevin
Duh. 2025. Linguistic nepotism: Trading-off quality
for language preference in multilingual rag.arXiv
preprint arXiv:2509.13930.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Bo Li, Zhenghua Xu, and Rui Xie. 2025. Language drift
in multilingual retrieval-augmented generation: Char-
acterization and decoding-time mitigation.arXiv
preprint arXiv:2511.09984.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, and 1 others.
2024. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437.
Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.
MKQA: A linguistically diverse benchmark for mul-
tilingual open domain question answering.Transac-
tions of the Association for Computational Linguis-
tics, 9:1389–1406.
9

Hoyeon Moon, Byeolhee Kim, and Nikhil Verma. 2025.
Quality-aware translation tagging in multilingual rag
system. InProceedings of the 5th Workshop on Multi-
lingual Representation Learning (MRL 2025), pages
161–177.
Jeonghyun Park and Hwanhee Lee. 2025. Investigating
language preference of multilingual RAG systems.
InFindings of the Association for Computational
Linguistics: ACL 2025, pages 5647–5675, Vienna,
Austria. Association for Computational Linguistics.
Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2025.
On the consistency of multilingual context utiliza-
tion in retrieval-augmented generation. InProceed-
ings of the 5th Workshop on Multilingual Representa-
tion Learning (MRL 2025), pages 199–225, Suzhuo,
China. Association for Computational Linguistics.
Leonardo Ranaldi, Barry Haddow, and Alexandra Birch.
2025. Multilingual retrieval-augmented genera-
tion for knowledge-intensive task.arXiv preprint
arXiv:2504.03616.
David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault
Formal, Shuai Wang, Stéphane Clinchant, and Vas-
silina Nikoulina. 2024. Bergen: A benchmarking
library for retrieval-augmented generation. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 7640–7663.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.
Bhavani Shankar, Preethi Jyothi, and Pushpak Bhat-
tacharyya. 2024. In-context mixing (ICM): Code-
mixed prompts for multilingual LLMs. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 4162–4176, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Nikhil Sharma, Kenton Murray, and Ziang Xiao. 2024.
Faux polyglot: A study on information disparity
in multilingual large language models.Preprint,
arXiv:2407.05502.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423.
Suhang Wu, Jialong Tang, Baosong Yang, Ante Wang,
Kaidi Jia, Jiawei Yu, Junfeng Yao, and Jinsong Su.
2024. Not all languages are equal: Insights into mul-
tilingual retrieval-augmented generation.Preprint,
arXiv:2410.21970.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, and
Grzegorz Kondrak. 2023. Don’t trust chatgpt when
your question is not in english: a study of multi-
lingual abilities and types of llms.arXiv preprint
arXiv:2305.16339.
10

Appendix
A The Use of Large Language Models
We write the manuscript ourselves, and an LLM
(ChatGPT-5.2) is used solely for refinement—style,
clarity, and grammar. It is not used for ideation or
content generation.
B Implementation Details
We adopt the multilingual retrieval baseline of
Bergen (Chirkova et al., 2024), which retrieves evi-
dence from a datastore spanning all languages. For
generations, we follow Bergen’s prompting setup
and use the basic_translated_langspec template
(Figure 4) to produce the final mRAG response.
Building on this standardized pipeline, we conduct
a series of experiments to quantify language prefer-
ence in mRAG under the Bergen framework, which
systematically examines the key components and
practical adjustments necessary for a robust mul-
tilingual RAG baseline. We use a robust multilin-
gual LLM, qwen3-235b-a22b-2507, to translate the
user’s local question into the global pivot language,
English. We instruct GPT-4o-mini (Hurst et al.,
2024) to generate a compact lexical bundle that
supplies candidate titles, aliases, and a short dis-
ambiguation hint for the global and local segments.
All LLM calls are made using the OpenRouter API.
We conduct our experiments using an AMD EPYC
7313 CPU (3.0 GHz) paired with four NVIDIA
RTX 6000 Ada GPUs. We use Python 3.11.5 and
PyTorch 2.3.1 for the software environment.
C Language Notation
We use standard ISO 639-1 language codes to de-
note the languages in our experiments. Specifically,
en denotes English, ar represents Arabic, es cor-
responds to Spanish, zh refers to Chinese (Simpli-
fied), ja indicates Japanese, de stands for German,
ko denotes Korean, and th denotes Thai. These con-
cise codes facilitate consistent identification and
processing of language-specific data across datasets
and models in multilingual NLP research.
D Dataset Details & Statistics
Wikipedia is a widely adopted knowledge source
in both monolingual RAG and mRAG systems,
as it provides broad topical coverage and is com-
monly used to benchmark RAG pipelines. In most
experiments, we retrieve from various linguistic
data sources from (i) the KILT snapshot of EnglishWikipedia2and (ii) the Wikipedia edition in the
user’s local language3. This two-source design re-
flects a standard and practical mRAG setting where
English serves as a high-coverage reference corpus
while local-language Wikipedia captures language-
specific evidence and terminology.
We report summary statistics for the data re-
sources used in our experiments in Table 12.
MKQA is our primary evaluation dataset, and we
provide the number of examples along with the me-
dian lengths of questions and answers. We also use
Wikipedia as the external corpus for the retriever
datastore; its statistics, including the number of
passages and their median lengths, are likewise
presented in Table 12. These statistics provide an
overview of the datasets and corpora underlying
our experimental setup.
E Raw Language Preference Score
MLRS.Following the standard MultiLingual-
RankShift (MLRS) protocol, we quantify retriever-
level language preference by measuring how much
the ranks of non-query-language documents im-
prove after being translated into the query lan-
guage (Park and Lee, 2025). For each query qwith
language Lq, we first retrieve a ranked list Dqfrom
a multilingual datastore, assigning each document
d∈D qan initial rank rinit
d. We then translate docu-
ments with Ld̸=L qintoLqand re-rank them
using the same retriever, obtaining rre-rank
d. The
(non-negative) rank gain is computed as ∆rd=
max 
rinit
d−rre-rank
d,0
, and aggregated per query
as∆rq=P
d∆rd. Normalizing by the maximum
possible gain ∆rmax
q=P
d(rinit
d−1) yields the
query-level score MLRS q=∆rq
∆rmaxq×100 (or0if
∆rmax
q= 0), and the final MLRS is the average
over queries.
Results.Table 10 reports the retriever’s language
preference scores before calibration with DeLP.
Overall, we observe three consistent patterns. First,
cross-lingual retrieval ( Lq̸=Ld) generally yields
lower MLRS than monolingual retrieval, indicat-
ing that cross-lingual matching is less preferred in
most cases. Second, English emerges as a dom-
inant target language: when the retrieved docu-
ment language Ldis English, the retriever attains
near-maximum preference scores and often even
2https://huggingface.co/datasets/facebook/
kilt_wikipedia
3https://huggingface.co/datasets/wikimedia/
wikipedia
11

Generator Level Methoden ar eszhja de ko th
Deepseek-chat-v3.1DocumentMultiRAG 1.925 3.093 2.456 2.683 2.816 2.456 2.362 2.716
CrossRAG 2.005 2.897 2.706 2.822 2.968 2.328 2.778 2.653
DKM-RAG 1.955 2.710 2.262 2.572 2.841 2.227 1.733 2.843
QTT-RAG 1.671 2.347 2.054 1.348 2.475 1.663 1.3832.525
QueryEnglish Translation – 2.170 1.992 1.800 1.857 2.224 2.179 2.110
Ours0.851 0.988 1.496 1.288 1.004 0.8531.637 0.955
GeminiDocumentMultiRAG 1.524 1.518 1.502 1.546 1.595 1.494 1.519 1.581
CrossRAG 1.688 4.8210.7801.511 2.632 3.234 3.883 2.277
DKM-RAG 4.308 7.293 5.3010.7656.261 5.559 6.187 9.346
QTT-RAG 2.511 7.355 5.306 1.177 6.581 5.639 6.274 9.553
QueryEnglish Translation –1.4841.510 1.528 1.802 1.481 1.4781.539
Ours1.4831.485 1.469 1.4841.499 1.4731.490 1.481
QwenDocumentMultiRAG1.0671.905 1.367 1.657 1.147 1.343 1.141 1.410
CrossRAG 1.082 1.533 1.357 2.4751.033 0.6300.693 1.543
DKM-RAG 4.573 1.368 3.794 5.171 5.721 0.764 0.6278.364
QTT-RAG 1.401 1.813 1.698 3.263 1.494 1.350 1.037 2.352
QueryEnglish Translation – 1.086 1.347 1.141 1.095 1.108 1.214 1.226
Ours 1.069 1.036 1.272 1.1281.045 1.111 1.1231.225
Table 8: Average generation time (sec/query; lower is better).Bold= lowest, underline = second-lowestacross both
Document-Level and Query-Level rowsfor each (generator, language).
surpasses monolingual settings, consistent with
English-heavy pretraining and stronger English
representations. Third, cross-lingual preference is
partly modulated by linguistic relatedness: closely
related Romance languages (fr/it/pt/es) preserve rel-
atively high cross-lingual scores, while East Asian
pairs (ko/ja/zh) show moderate but noticeable drops
compared to monolingual baselines.
F Language Distribution of Retrieved
Documents
Table 11 reports the language composition of the
top-50 documents retrieved for each MKQA query-
language split. Across nearly all query languages,
the retrieved evidence is heavily concentrated in
English, and English often remains the most fre-
quently retrieved language even for non-English
queries. This trend is also reflected in the aggre-
gated distribution (mkqa_avg), indicating that the
observed language preference in standard mRAG
pipelines largely mirrors structural priors of the
retrieval setup rather than purely intrinsic model
preference.
G Calibration Details
Table 9 reports the detailed DeLP scores for each
of the three encoders. After removing the variance
explained by the structural priors, the calibrated ma-
trices exhibit a consistent pattern across encoders:the strongest preference concentrates on the di-
agonal ( Lq=Ld), indicating a robust shift toward
query–document language alignment rather than an
English-dominant bias. Residual cross-lingual pref-
erences remain comparatively mild and structured,
reflecting interpretable related-language effects in-
stead of exposure- or coverage-driven artifacts.
H Gold Passage Counting Protocol
We compute Table 1 on the 2,827-question sub-
set that overlaps with KILT NQ provenance. Be-
cause prior work (Park and Lee, 2025) provides
the same underlying question translated into 13
query languages, the unit counted in Table 1 is
not the number of unique questions but the num-
ber of question ×query-language instances. Hence
the total number of instances is 2,827×13 =
36,751 , and values such as “#q = 26,934 (73.29%)”
can legitimately exceed 2,827; the ratio is com-
puted as 26,934/36,751 . Gold labels originate
from KILT’s provenance, which is anchored to En-
glish Wikipedia page IDs (WPIDs). All 13 trans-
lations of the same question share the same gold
WPID set. We then assess gold availability in each
Wikipedia language edition by mapping each En-
glish WPID to a corresponding page in language
ℓusing Wikipedia/Wikidata interlanguage links
(sitelinks), and checking whether the mapped page
exists in the Wikipedia dump used to build our
12

Query Lang. EncoderL q=L d Lq ̸=L d
en ko zh fr ja it pt es
enbge-m3 49.25 – 35.87(-13.37) 39.48 (-9.77) 34.58(-14.67) 38.79(-10.46) 34.59(-14.66) 35.81(-13.43) 35.13(-14.12)
p-mMiniLM 50.26 – 37.00(-13.27) 40.94 (-9.32) 36.18(-14.08) 39.96(-10.31) 35.83(-14.43) 36.62(-13.64) 36.49(-13.78)
p-mMpNet 50.80 – 37.15(-13.65) 40.61 (-10.19) 35.94(-14.86) 40.09(-10.71) 36.04(-14.76) 36.96(-13.84) 36.43(-14.37)
kobge-m3 42.34 36.76(-5.58)– 40.69(-1.65) 34.41(-7.93) 42.45(+0.11) 34.44(-7.90) 35.29(-7.05) 34.46(-7.87)
p-mMiniLM 44.09 38.03(-6.06)– 42.37 (-1.72) 35.09(-9.00) 43.90(-0.19) 34.75(-9.34) 36.07(-8.02) 34.98(-9.11)
p-mMpNet 43.72 38.29(-5.43)– 42.19(-1.53) 35.20(-8.52) 44.43(+0.72) 34.91(-8.80) 35.59(-8.13) 34.87(-8.84)
zhbge-m3 49.67 38.52(-11.15) 37.32(-12.36)– 34.33(-15.34) 41.36 (-8.32) 34.58(-15.09) 35.71(-13.96) 34.98(-14.69)
p-mMiniLM 51.01 38.80(-12.21) 38.11(-12.90)– 34.99(-16.03) 42.20 (-8.81) 35.06(-15.95) 35.94(-15.07) 35.38(-15.64)
p-mMpNet 51.11 38.72(-12.39) 37.91(-13.20)– 34.87(-16.24) 42.13 (-8.98) 34.98(-16.13) 35.88(-15.23) 35.31(-15.80)
frbge-m3 39.45 40.48(+1.02) 36.15(-3.30) 39.95 (+0.49)– 39.47(+0.01) 35.39(-4.06) 36.25(-3.20) 35.75(-3.70)
p-mMiniLM 40.42 41.56(+1.14) 37.20(-3.22) 40.85 (+0.43)– 40.27(-0.15) 36.33(-4.09) 36.94(-3.48) 36.56(-3.86)
p-mMpNet 40.28 41.45(+1.17) 36.95(-3.33) 40.71 (+0.43)– 40.03(-0.25) 36.29(-3.98) 36.87(-3.41) 36.54(-3.74)
jabge-m3 48.61 38.44(-10.16) 38.21(-10.39) 41.15 (-7.46) 34.70(-13.91)– 34.83(-13.78) 35.86(-12.75) 35.09(-13.52)
p-mMiniLM 49.56 38.95(-10.61) 38.55(-11.01) 41.90 (-7.66) 35.19(-14.37)– 35.21(-14.35) 36.14(-13.42) 35.44(-14.12)
p-mMpNet 49.41 38.70(-10.72) 38.43(-10.98) 41.64 (-7.78) 34.94(-14.47)– 34.94(-14.47) 35.92(-13.49) 35.15(-14.26)
itbge-m3 38.38 39.88(+1.50) 36.15(-2.23) 39.83 (+1.45) 35.87(-2.51) 39.26(+0.88)– 36.39(-2.00) 36.17(-2.21)
p-mMiniLM 39.44 41.10(+1.67) 37.23(-2.21) 40.92 (+1.48) 37.08(-2.36) 40.24(+0.80)– 37.44(-2.00) 37.36(-2.08)
p-mMpNet 39.33 40.90 (+1.57) 37.18(-2.15) 41.02(+1.69) 36.94(-2.39) 40.09(+0.77)– 37.21(-2.12) 37.20(-2.12)
ptbge-m3 45.38 39.89 (-5.49) 36.22(-9.17) 39.82(-5.56) 35.78(-9.60) 39.41(-5.97) 35.81(-9.57)– 37.09(-8.29)
p-mMiniLM 46.52 41.16(-5.36) 37.33(-9.19) 41.24 (-5.28) 37.03(-9.49) 40.47(-6.05) 36.93(-9.59)– 38.21(-8.31)
p-mMpNet 46.33 40.61 (-5.72) 37.38(-8.95) 40.84(-5.49) 36.70(-9.63) 40.14(-6.19) 36.71(-9.61)– 37.88(-8.45)
esbge-m3 37.64 40.18(+2.54) 36.21(-1.43) 39.78 (+2.15) 35.68(-1.95) 39.28(+1.64) 35.90(-1.74) 36.82(-0.82)–
p-mMiniLM 38.71 41.31(+2.60) 37.29(-1.42) 40.85 (+2.14) 36.87(-1.84) 40.20(+1.49) 37.01(-1.70) 37.73(-0.98)–
p-mMpNet 38.23 40.65(+2.42) 37.09(-1.14) 40.53 (+2.30) 36.34(-1.89) 39.81(+1.58) 36.43(-1.80) 37.19(-1.04)–
Table 9: Language preference measured by DeLP. Each cell reports the debiased preference score and its delta
from the matching-language baseline ( Lq=L d). Background shading is row-wise min–max scaled (including the
diagonal cell); Darker cells indicate a stronger preference for the document language.
corpus.
A key source of confusion is that our corpus is
passage-based: each Wikipedia page is split into
multiple chunks, so a single WPID may correspond
to multiple passages. Moreover, for a given ques-
tion, KILT may provide multiple gold provenances
that map to different passages within the same
WPID. In Table 1, we use a WPID-level conven-
tion: when multiple gold passages correspond to
the same WPID for a query, we treat them as a
single gold item rather than counting them multi-
ple times. Out of the 2,827 questions in our KILT-
overlap subset, 2,404 questions have at least one
available gold WPID (i.e., a mapped gold page
exists in our Wikipedia dumps). Accordingly, the
number of questions with gold evidence satisfies
only_en+both= 2,404 in Table 1.
I Detailed Latency
Table 8 reports detailed latency measured as av-
erage generation time (sec/query; lower is better)
for each generator across languages. DELTA re-
mains consistently efficient because it produces a
single fused query and avoids document translation
overhead; in several settings, it is even faster than
English Translation, since the fused query retains
local cues and disambiguation anchors that help re-trieval focus earlier and reduce wasted computation
on overly generic English-only signals.
J Cultural Prior Measurement
To model whether a query is intrinsically tied to
a particular cultural or regional context (indepen-
dent of corpus size or retrieval exposure), we con-
struct acultural priorusing an LLM-based clas-
sifier. Starting from the English version of each
query (MKQA-en), we assign exactly onecultural
database languagefrom a fixed set of 13 languages
(en, ar, es, de, ja, ko, th, zh, fr, it, pt, ru, fi). We use
GPT-4o mini (via OpenRouter) with constrained
JSON output to enforce a single-label decision. We
instruct the classifier to choose the local language
of the primary place/culture the query is about (e.g.,
France →fr, Hong Kong/China →zh), and to se-
lectenonly when the cultural context is inherently
English-speaking (e.g., US/UK-specific) or when
the query is genuinely global/multi-country and not
tied to a single locale.
In addition to the cultural-language label, we
record lightweightcultural metadatafor analy-
sis and filtering: (i) country_or_region (a single
primary place/region), (ii) is_culture_specific
(whether the question is judged to be culture/locale-
specific), (iii) confidence (0–1), and (iv) a short
13

rationale . These fields are used only to character-
ize the dataset and to support qualitative inspection;
our core metric relies on the language label.
Finally, we define the cultural prior pcult(ℓ)as
the empirical probability that a query’s predicted
cultural language equals ℓ, i.e., the normalized fre-
quency of the single-label assignments over the
evaluation set. This prior captureswhere evidence
should existin a fair localized setting, and is incor-
porated as a structural factor alongside other priors
(e.g., exposure and gold availability) in our cali-
bration analysis. We use those cultural prior and
metadata for calibration and DELTA.
KRepetition-based weighting for DELTA
Query construction with repetition.To control
cue influence without changing the retriever or
learning parameters, we apply a deterministic rep-
etition policy while constructing the fused query
string Qfused. We use the same notation as in Eq. 5:
y∈ {0,1} indicates whether the query is culture-
specific and c∈[0,1] is the confidence score. We
repeat the local block [LOCAL:L q]rlocaltimes and
the global pivot block [GLOB]r globtimes, and con-
catenate all segments with a delimiter (“ |”) to
form a single retrieval query.
Length control.To keep retrieval budgets com-
parable across methods, we truncate the final Qfused
to a fixed maximum length (e.g., 900 characters)
after concatenation.
Deduplication.We apply conservative dedupli-
cation to avoid redundant anchors: (i) if the global
and local titles are identical, we keep only a sin-
gle[TITLE_BRIDGE] ; (ii) if alias sets match across
languages, we keep only the global alias block; and
(iii) when the query language is not English, we
always include[LOCAL:L q]at least once.
LThresholds and fixed hyperparameters.
We do not exhaustively tune (τlow, τhigh, τboost)be-
cause a full sweep is combinatorial and would cou-
ple these knobs to expensive end-to-end RAG runs.
Instead, we instantiate the confidence thresholds
with three goals: (i) discretize the continuous con-
fidence cinto a small number of stable intervals,
(ii) keep the query-length increase bounded, and
(iii) reserve upweighting for only the most reliable
culture-specific cases. Concretely, we set two cut-
offsτlow< τhighto map cinto three repetition lev-
els for the local block, rlocal∈ {1,2,3} , where τlowmarks the onset ofreliableculture-specificity and
τhighindicateshigh-confidencecases that warrant
the strongest local emphasis. In our implementa-
tion, we use τlow= 0.6 andτhigh= 0.85 , which
empirically balance coverage (triggering local up-
weighting for sufficiently confident cases) and con-
servativeness (avoiding frequent over-repetition un-
der noisy cue predictions).
For auxiliary local boosting, we use a separate
threshold τboost that applies only to the disambigua-
tion anchors ( [TITLE_BRIDGE] and[ALIASES] ),
not the full [LOCAL] query text. Specifically,
for culture-specific queries ( y= 1 ), we set
b=I[c≥τ boost]and, when b= 1 , dupli-
cate [TITLE_BRIDGE] and [ALIASES] once to
strengthen culturally grounded anchoring and re-
duce entity ambiguity
We set τboost = 0.7 so that anchor duplication is
enabled for moderately-to-high confidence culture-
specific queries, providing extra entity anchor-
ing/disambiguation without incurring the larger
length increase of repeating the entire local block.
For ridge calibration, we likewise keep a single
regularization strength λacross all encoders; this
choice is motivated by the small calibration design
(|C|language pairs with a low-dimensional prior
vector) where ridge mainly stabilizes coefficients
against correlated priors rather than serving as a
performance-tuned knob.
M Case Study
DELTA.Table 13 illustrates DELTA on a Ko-
rean query asking “when was the last time
South Korea had the Olympics.” DELTA forms
the global pivot qgloband emits it as [GLOB] ,
and places the original Korean surface form as
[LOCAL:ko] . It then injects multilingual anchors:
[TITLE_BRIDGE] contains paired Wikipedia-style
titles, while [ALIASES:GLOB] and[ALIASES:ko]
provide short alias cues in the global and local lan-
guages, respectively. Finally, [LOCALE_HINT] adds
a brief region hint with minimal disambiguation to
bias retrieval toward region-appropriate evidence.
Crucially, DELTA controls the balance be-
tween global and local signals purely through
repetition. Because this query is labeled culture-
specific ( y=1) with high confidence c=0.93 ,
the policy sets rlocal=3 (since c≥0.85 )
while keeping rglob=1(since c≥0.6 ), yield-
ing three copies of [LOCAL:ko] but only one
copy of [GLOB] . Moreover, the auxiliary local-
14

Lq=Ld Lq̸=Ld
Query Lang. Encoder en ko zh fr ja it pt es
enbge-m3 56.03 – 33.02(-23.01)33.10(-22.93)36.61(-19.42)33.36(-22.67)35.89(-20.14)35.86(-20.17)36.62 (-19.41)
p-mMiniLM 56.85 – 34.34(-22.51)34.61(-22.24)38.17 (-18.68)34.52(-22.33)37.15(-19.70)36.73(-20.12)37.96(-18.89)
p-mMpNet 57.49 – 34.45(-23.04)34.27(-23.22)37.94 (-19.55)34.67(-22.82)37.34(-20.15)37.02(-20.47)37.90(-19.59)
kobge-m3 41.15 43.49(+2.34)– 34.42(-6.73)36.42(-4.73)37.18(-3.97)35.72(-5.43)35.30(-5.85)35.93(-5.22)
p-mMiniLM 42.95 44.62(+1.67)– 36.04(-6.91)37.08(-5.87)38.47(-4.48)36.07(-6.88)36.18(-6.77)36.45(-6.50)
p-mMpNet 42.53 44.98(+2.45)– 35.85(-6.68)37.20(-5.33)39.01(-3.52)36.21(-6.32)35.65(-6.88)36.34(-6.19)
zhbge-m3 44.98 45.26(+0.28)34.52(-10.46)– 36.34(-8.64)36.05(-8.93)35.86(-9.12)35.73(-9.25)36.45(-8.53)
p-mMiniLM 46.18 45.39 (-0.79)35.46(-10.72)– 36.98(-9.20)36.77(-9.41)36.38(-9.80)36.05(-10.13)36.85(-9.33)
p-mMpNet 46.27 45.41 (-0.86)35.21(-11.06)– 36.87(-9.40)36.71(-9.56)36.28(-9.99)35.94(-10.33)36.78(-9.49)
frbge-m3 43.18 47.23(+4.05)33.29(-9.89)33.58(-9.60)– 34.07(-9.11)36.70(-6.48)36.30(-6.88)37.25(-5.93)
p-mMiniLM 44.09 48.15(+4.06)34.54(-9.55)34.52(-9.57)– 34.83(-9.26)37.65(-6.44)37.05(-7.04)38.03(-6.06)
p-mMpNet 43.96 48.14(+4.18)34.25(-9.71)34.37(-9.59)– 34.61(-9.35)37.59(-6.37)36.93(-7.03)38.01(-5.95)
jabge-m3 45.03 45.18(+0.15)35.45(-9.58)34.86(-10.17)36.71(-8.32)– 36.11(-8.92)35.88(-9.15)36.56(-8.47)
p-mMiniLM 45.80 45.54 (-0.26)35.90(-9.90)35.57(-10.23)37.18(-8.62)– 36.53(-9.27)36.25(-9.55)36.91(-8.89)
p-mMpNet 45.67 45.39 (-0.28)35.73(-9.94)35.30(-10.37)36.94(-8.73)– 36.24(-9.43)35.98(-9.69)36.62(-9.05)
itbge-m3 41.06 46.63(+5.57)33.30(-7.76)33.47(-7.59)37.92(-3.14)33.86(-7.20)– 36.44(-4.62)37.68(-3.38)
p-mMiniLM 42.11 47.69(+5.58)34.57(-7.54)34.59(-7.52)39.07(-3.04)34.80(-7.31)– 37.55(-4.56)38.83(-3.28)
p-mMpNet 41.98 47.59(+5.61)34.48(-7.50)34.68(-7.30)38.94(-3.04)34.67(-7.31)– 37.27(-4.71)38.67(-3.31)
ptbge-m3 39.19 46.64(+7.45)33.37(-5.82)33.46(-5.73)37.83(-1.36)34.02(-5.17)37.13(-2.06)– 38.61(-0.58)
p-mMiniLM 40.17 47.75(+7.58)34.67(-5.50)34.91(-5.26)39.02(-1.15)35.03(-5.14)38.25(-1.92)– 39.68(-0.49)
p-mMpNet 39.91 47.30(+7.39)34.68(-5.23)34.50(-5.41)38.70(-1.21)34.72(-5.19)38.01(-1.90)– 39.35(-0.56)
esbge-m3 40.76 46.93(+6.17)33.36(-7.40)33.42(-7.34)37.73(-3.03)33.87(-6.89)37.22(-3.54)36.88(-3.88)–
p-mMiniLM 41.81 47.90(+6.09)34.63(-7.18)34.52(-7.29)38.86(-2.95)34.76(-7.05)38.33(-3.48)37.84(-3.97)–
p-mMpNet 41.33 47.34(+6.01)34.39(-6.94)34.19(-7.14)38.34(-2.99)34.39(-6.94)37.73(-3.60)37.25(-4.08)–
Table 10: Raw language preference measured by MLRS with different re-ranking encoders for various
query–document language pairs. The Lq=L dcolumn shows scores for matching query and document lan-
guages, while the remaining columns represent cross-lingual scenarios. Parentheses indicate the change from the
Lq=L dcolumn (positive for improvement, negative for decline). The highest score per row is in bold, and the
second highest is underlined.
boost flag triggers at c≥0.7 , duplicating the
local-side anchors once more, which explains
why [TITLE_BRIDGE] and[ALIASES:ko] appear
twice, whereas [ALIASES:GLOB] remains single-
copy. Overall, this design realizes a global back-
off ([GLOB] ) with preference-aligned local empha-
sis ([LOCAL] ,[TITLE_BRIDGE] ,[ALIASES:ko] )
within a single Qfused, without modifying the re-
triever or adding model parameters.
Success Case.Table 14 presents a representative
top-1 retrieval example comparing DELTA with
a simple English-translation query for the ques-
tion “언제마지막으로대한민국이올림픽을했
었나요 . (When was the last time South Korea had
the Olympics?).” Although both methods use the
same retriever and multilingual datastore, the re-
trieved evidence differs markedly: DELTA’s fused
query contains explicit host-oriented cues (local
surface form, title/alias anchors, and a locale hint),
which increases lexical alignment with passages
that describe Olympicsheld inKorea (e.g., 개최 ,
서울 1988,평창 2018). In contrast, the English-
translation query is more underspecified and can
drift to participation-centric passages that matchbroad entities (“South Korea”, “Olympics”) but do
not emphasize hosting-related facts. As a result,
the DELTA top-1 passage provides the necessary
host evidence for inferring the most recent domes-
tically held Olympics, enabling the generator to
produce the correct answer, while the translation-
based pipeline is more likely to miss the hosting
signal and return an incorrect year/event.
Failure Case.Table 15 shows a representative
failure where the question “who is the president dur-
ing the Korean War” is underspecified: “president”
can plausibly refer to the U.S. president oversee-
ing U.S. involvement (gold: Harry S. Truman and
Dwight D. Eisenhower) or to the South Korean
president during the same period (Syngman Rhee).
In this example, DELTA’s cultural/locale cues and
title bridge steer the query towardSouth Korean
leadership, effectively resolving the ambiguity in
the wrong direction. Consequently, the top-1 re-
trieved passage focuses on Syngman Rhee and con-
tains strong lexical overlap with the localized cues
(e.g., “대한민국대통령 ”, “이승만 ”, “한국전쟁 ”),
making the generator likely to output Syngman
Rhee despite the dataset’s gold reference targeting
15

U.S. presidents. This failure highlights a limitation
of repetition- and cue-based weighting: when the
underlying intent is ambiguous, aggressively in-
jecting locale-specific anchors can over-localize re-
trieval and suppress globally relevant evidence, sug-
gesting the need for ambiguity-aware safeguards
(e.g., intent disambiguation or controlled locale
injection) for such queries.
N Prompts
As shown in Figure 4, we provide the exact prompt
templates used throughout our pipeline. Prompt
(A) specifies the RAG answer-generation instruc-
tion, with two variants depending on whether re-
trieved documents are provided, enforcing con-
cise English outputs and (when available) condi-
tioning answers on the supplied evidence. Prompt
(B) defines our cultural-context annotation step,
where an LLM assigns a single cultural database
language from a fixed set under strict locality-
oriented rules and returns lightweight metadata (re-
gion, culture-specificity, confidence, and a brief
rationale) in a structured JSON format. Prompt (C)
is used by DELTA to produce compact retrieval
anchors—English and local Wikipedia-style titles,
alias lists, and a short disambiguation hint—which
are then assembled into a fused query; this prompt
enforces a fixed JSON schema and language con-
straints to keep the generated anchors consistent
and directly usable for retrieval. Finally, in Prompt
(D), we provide the prompts used for English trans-
lation.
16

en ko ar zh fi fr de ja it pt ru es th
mkqa_en 44.12 1.60 1.19 1.30 2.54 10.03 6.90 1.44 8.32 7.67 4.85 9.90 0.13
mkqa_ko 23.07 17.35 1.99 4.81 2.04 7.90 5.96 10.36 6.16 5.06 6.85 6.85 1.58
mkqa_ar 24.93 3.30 15.29 4.07 2.10 8.30 6.53 6.64 6.80 5.71 7.78 7.65 0.89
mkqa_zh 24.70 3.17 1.76 23.22 2.01 7.47 6.17 6.27 6.08 5.24 6.37 7.27 0.27
mkqa_fi 30.32 2.27 1.63 2.33 7.92 11.11 8.20 3.78 8.77 7.18 6.51 9.42 0.58
mkqa_fr 29.90 1.48 1.25 1.55 2.50 21.44 6.96 2.06 9.40 7.96 4.77 10.55 0.19
mkqa_de 32.54 1.46 1.17 1.44 2.96 11.40 15.12 1.89 9.09 7.69 4.83 10.17 0.24
mkqa_ja 24.56 4.80 1.69 3.99 2.19 7.97 5.99 22.55 6.38 5.66 6.49 7.45 0.28
mkqa_it 28.72 1.59 1.30 1.58 2.52 12.30 6.97 1.95 17.46 8.47 5.26 11.70 0.17
mkqa_pt 28.82 1.71 1.40 1.63 2.60 11.92 6.74 2.23 10.24 13.78 5.38 13.33 0.24
mkqa_ru 27.02 2.53 1.92 1.98 2.45 8.83 6.44 2.71 7.36 6.24 23.83 8.43 0.26
mkqa_es 29.45 1.73 1.27 1.60 2.66 11.85 6.93 1.83 10.55 9.33 5.27 17.36 0.16
mkqa_th 32.39 3.10 2.10 2.96 2.53 10.00 7.40 4.43 8.06 7.43 6.80 9.70 3.10
mkqa_avg 29.27 3.55 2.61 4.04 2.85 10.81 7.41 5.24 8.82 7.49 7.31 9.98 0.62
Table 11: Language distribution of retrieved documents for each MKQA query-language split. Each row corresponds
to the query language (dataset), and each column indicates the language of the retrieved passages; values are shown
as percentages (without the %). The final row (mkqa_avg) reports the average retrieved-language distribution across
all query languages.
Dataset en ar es fi fr de ja it ko pt ru zh th
MKQA
# examples 2827 2827 2827 2827 2827 2827 2827 2827 2827 2827 2827 2827 2827
len question. 43 38 48 46 49 47 26 48 22 45 42 16 41
len answer. 11 10 11 11 11 11 8 11 6 11 12 6 12
Wikipedia
# ex. (M) 25 3.3 10 1.5 13 14 27 8.2 1.6 4.7 8.6 11 3.7
len passage. 624 585 619 833 627 720 208 650 431 619 721 206 217
Table 12: Statistics of the datasets used in our experiments. MKQA Number of examples and median lengths of
questions and answers (in Unicode characters). Wikipedia: Number of passages (in millions) and their median
lengths.
DELTA segment Instantiated content (case study) Rep.
[GLOB]when was the last time south korea had the olympics 1
[LOCAL:ko]언제마지막으로대한민국이올림픽을했었나요3
[TITLE_BRIDGE]South Korea at the Olympics /대한민국의올림픽2
[ALIASES:ko]대한민국올림픽,한국올림픽,한국의올림픽역사2
[ALIASES:GLOB]Olympics in South Korea, South Korean Olympic Games, History of South Korea Olympics 1
[LOCALE_HINT]South Korea + Last Olympic Games in South Korea 1
Table 13: DELTA case study. A Korean culture-specific query ( c=0.93 ) is converted into a single fused query Qfused
by concatenating labeled segments. Repetition counts follow Eq. 5, which upweights local cues while maintaining a
global back-off.
17

Item Content
DELTA[GLOB] when was the last time south korea had the olympics
|[LOCAL:ko] 언제마지막으로 대한민국이 올림픽을 했었나요
|[TITLE_BRIDGE] South Korea at the Olympics /대한민국의올림픽
|[ALIASES:ko]대한민국올림픽,한국올림픽,한국의올림픽역사
|[ALIASES:GLOB] Olympics in South Korea, South Korean Olympic Games,
History of South Korea Olympics
|[LOCALE_HINT] South Korea Last Olympic Games in South Korea
English TranslationWhen was the last time south korea had the olympics
Top-1 passage (DELTA)대한민국에서열린 올림픽으로는 1988년서울하계올림픽과 2018년평창동계
올림픽이널리알려져있다 .서울대회는 20세기후반대한민국의국제스포츠
행사유치와관련해자주언급되며 ,주요경기장은서울및인근지역에분산되어
운영되었다 .평창대회는강원지역을중심으로동계종목이진행되었고 ,개/폐회식
과일부경기장이 평창및주변권역에배치되었다 .두대회모두대한민국내에서
개최된사례로정리되며 ,대회의성격 (하계 /동계 )과개최지역 (서울 /평창 )이함께
기술되는경우가많다.
Top-1 passage (English Translation) 대한민국 (South Korea )은근대올림픽 (Olympics )에지속적으로참가해왔으며 ,
여러종목에서의미있는성과를거두었다 .이문서는연도별참가개요 ,선수단
규모 ,주요종목에서의메달기록과같은정보를중심으로구성된다 .예를들어
양궁 ,태권도 ,쇼트트랙등에서의성과가요약되고 ,대회별대표선수나주목할
만한기록이덧붙여지기도한다 .또한특정대회에서의종합순위변화나메달수
추이처럼참가및성과를설명하는통계적서술이포함될수있다.
Gold answer2018년평창동계올림픽. (The 2018 PyeongChang Winter Olympics.)
English translation answer1988년서울하계올림픽. (The 1988 Seoul Summer Olympics.)
DELTA answer대한민국에서개최된올림픽으로 1988년서울하계올림픽과 2018년평창동계
올림픽이언급되므로 ,질문에서묻는 “마지막으로 ”개최된올림픽은 2018년평창
동계올림픽이다. (The 2018 PyeongChang Winter Olympics.)
Table 14: Case study: DELTA vs. English translation (top-1 retrieval).
Item Content
DELTA (misled)[GLOB] who is the president during the korean war
|[TITLE_BRIDGE] President of South Korea during the Korean War /한국전쟁중
대한민국대통령
|[ALIASES:ko]이승만,이승만대통령,대통령이승만
|[ALIASES:GLOB] Syngman Rhee, Rhee Syngman, President Rhee, Rhee
|[LOCALE_HINT] Korea (Korean Peninsula) President during Korean War era
Top-1 passage (DELTA) 이승만 (Syngman Rhee)은1948년부터 1960년까지대한민국의대통령으로재임한정치인이다 .
대한민국정부수립이후초대대통령으로선출되었으며 ,냉전초기한반도의분단체제속에
서정부운영을주도했다 .재임기간에는한국전쟁 (1950–1953)시기가포함되며 ,전쟁전후의
정치적갈등과대외관계가함께언급된다 .관련문서들은대체로이승만의생애 ,대통령재임
기간,당대의국내정치상황과외교적맥락을중심으로개괄한다.
Gold answer해리S.트루먼;드와이트D.아이젠하워(Harry S. Truman; Dwight D. Eisenhower)
DELTA answer이승만(Syngman Rhee)
Table 15: Failure case (top-1 retrieval).
18

(A) RAG Answer Generation
Goal: Answer as concisely as possible in {lang}.
With Documents:
System: Extract relevant information from provided documents and answer briefly. Reply in {lang}.
User: Background: {docs} \n\nQuestion: {question}
Without Documents:
System: Answer briefly. Reply in {lang}.
User: Question: {question}
(B) Cultural Language Classifier
System (instruction):
You are annotating a FAIR multilingual retrieval setup.
Given an English query, decide the SINGLE most appropriate "cultural database language"
where the relevant evidence SHOULD exist in a fair, localized setting.
CRITICAL RULES:
- You MUST choose exactly ONE language from this fixed set:
{en, ar, es, de, ja, ko, th, zh, fr, it, pt, ru, fi}
- Prefer the LOCAL language of the primary place/culture the query is about.
- Do NOT choose ’en’ just because the query text is English.
- Choose ’en’ only if the query’s primary cultural context is inherently English-speaking
(e.g., US/UK-specific) OR the query is truly global / multi-country / not place-specific.
- If the query mentions a place that maps to one of the non-English languages,
pick that non-English language.
Examples:
- "when did hong kong go back to china" -> cultural_language="zh"
- "what is the capital of france" -> cultural_language="fr"
- "who was the first president of the united states" -> cultural_language="en"
- "compare gdp of france and germany" -> cultural_language="en" (multi-country/global)
Output (JSON only; no extra text):
{
"country_or_region": string (SINGLE primary place/region),
"cultural_language": string (exactly one from the set),
"is_culture_specific": boolean,
"confidence": number in [0,1],
"rationale": short string
}
Input:
User: Query: {query_en}
Figure 4: Prompt templates used in our pipelines: RAG generation and cultural-language classification.
19

(C) DELTA Bundle Generator
Goal: Produce title/alias anchors and a short disambiguation hint for fused query construction.
Return Format:
- SINGLE-LINE JSON object only (no markdown, no explanation).
- Keys must be EXACTLY:
en_title, local_title, aliases_en, aliases_local, extra_disambig
Constraints:
- aliases_en / aliases_local: 0..K items each
- Titles: plausible Wikipedia page titles; use null if unsure
- extra_disambig: <= 8 words
- local_title & aliases_local MUST be in {query_lang}; English fields MUST be English
- Do not add new keys
Input (User JSON):
{
"q_en": "{q_en}",
"q_orig": "{q_orig}",
"query_lang": "{query_lang}",
"country_or_region": "{country_or_region}",
"cultural_language": "{cultural_language}",
"is_culture_specific": {is_culture_specific},
"confidence": {confidence}
}
(D) English Translation
Goal: Translate the question from {lang_name} to fluent, natural English while preserving
the original meaning as much as possible.
Rules:
- Keep named entities as appropriate English forms.
- Do not add explanations or extra information.
- Return STRICT JSON with a single key "translation".
System:
You are a professional translator from {lang_name} to English.
You receive a question in the source language and must translate it into fluent,
natural English while preserving the original meaning as much as possible.
- Keep named entities as appropriate English forms.
- Do not add explanations or extra information.
Return STRICT JSON with a single key "translation".
User:
Question in {lang_name}:
{query}
Return only:
{"translation": "<the question translated into English>"}
Figure 5: Prompt templates used in our pipeline: DELTA bundle generation and English translation.
20
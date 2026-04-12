# ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals

**Authors**: Yihao Wang, Zijian He, Jie Ren, Keze Wang

**Published**: 2026-04-08 12:14:48

**PDF URL**: [https://arxiv.org/pdf/2604.06997v1](https://arxiv.org/pdf/2604.06997v1)

## Abstract
Retrieval shapes how language models access and ground knowledge in retrieval-augmented generation (RAG). In historical research, the target is often not an arbitrary relevant passage, but the exact record for a specific regnal month, where temporal consistency matters as much as topical relevance. This is especially challenging for Classical Chinese annals, where time is expressed through terse, implicit, non-Gregorian reign phrases that must be interpreted from surrounding context, so semantically plausible evidence can still be temporally invalid. We introduce \textbf{ChunQiuTR}, a time-keyed retrieval benchmark built from the \textit{Spring and Autumn Annals} and its exegetical tradition. ChunQiuTR organizes records by month-level reign keys and includes chrono-near confounders that mirror realistic retrieval failures. We further propose \textbf{CTD} (Calendrical Temporal Dual-encoder), a time-aware dual-encoder that combines Fourier-based absolute calendrical context with relative offset biasing. Experiments show consistent gains over strong semantic dual-encoder baselines under time-keyed evaluation, supporting retrieval-time temporal consistency as a key prerequisite for faithful downstream historical RAG. Our code and datasets are available at \href{https://github.com/xbdxwyh/ChunQiuTR}{\texttt{github.com/xbdxwyh/ChunQiuTR}}.

## Full Text


<!-- PDF content starts -->

ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals
Yihao Wang1, Zijian He1, Jie Ren2, Keze Wang1,
1Sun Yat-Sen University,2Shaanxi Normal University,
Correspondence:kezewang@gmail.com
Abstract
Retrieval shapes how language models access
and ground knowledge in retrieval-augmented
generation (RAG). In historical research, the
target is often not an arbitrary relevant pas-
sage, but the exact record for a specific reg-
nal month, where temporal consistency mat-
ters as much as topical relevance. This is es-
pecially challenging for Classical Chinese an-
nals, where time is expressed through terse,
implicit, non-Gregorian reign phrases that must
be interpreted from surrounding context, so
semantically plausible evidence can still be
temporally invalid. We introduceChunQi-
uTR, a time-keyed retrieval benchmark built
from theSpring and Autumn Annalsand its
exegetical tradition. ChunQiuTR organizes
records by month-level reign keys and includes
chrono-near confounders that mirror realistic
retrieval failures. We further proposeCTD
(Calendrical Temporal Dual-encoder), a time-
aware dual-encoder that combines Fourier-
based absolute calendrical context with relative
offset biasing. Experiments show consistent
gains over strong semantic dual-encoder base-
lines under time-keyed evaluation, supporting
retrieval-time temporal consistency as a key
prerequisite for faithful downstream historical
RAG. Our code and datasets are available at
github.com/xbdxwyh/ChunQiuTR.
1 Introduction
Retrieval is increasingly the interface between lan-
guage models and the world’s knowledge, most
visibly in retrieval-augmented generation (RAG)
and search-augmented assistants (Gao et al., 2023;
Lewis et al., 2020). In such systems, models ground
responses in retrieved evidence rather than rely-
ing on parametric memory alone. This evidentiary
role is central to expert workflows—literature sur-
vey, legal and policy analysis, and scientific claim
verification—where users care not only aboutwhat
an answer is, but alsowhereit comes from (Menick
et al., 2022).
What happened  in the 12th month of Duke Zhuang ’s 2nd year?魯莊公二年十二月发生了什么 ？
莊公 ·二年 ·十二月
 莊公 ·二年 ·十一月
 莊公 ·三年 ·一月
Top-3 retrieved records
1.       [NEG]  魯莊公二年十二月 ：經傳於… 
2.       [NEG]  二年冬，夫人姜氏會齊侯 … 
3.       [RECORD] 冬十有二月 ，夫人姜氏 … 
Type: commentary, same time -key
Type: adjacent -month event
Hallucinated Answer : 
夫人姜氏會齊侯 , … Type: canonical record ( Ground Truth )Reign -based calendar
Canonical  Record s (Event ) Canonical Record s
(Annals + Three Zhuan)
【春秋】 冬十有二月 ，夫人姜氏… 
【谷梁】 乙酉，宋公馮卒 。
【春秋】 二年冬，夫人姜氏會齊侯 … 
【春秋】冬十有二月 丙午齊侯衞伯 … 
# 左氏、榖梁皆以 不稱氏為貶… 
# 齊侯鄭伯詐朝于紀欲以襲之紀 … 
# 欲要姜氏則姜氏 至每舉… 
# 魯莊公二年十二月 ：經傳於…
Same key ·  lexical -nearChrono -near Commentaries
(Off-topic Answers)Duke Zhuang  · 2nd year  · 11th month Duke Zhuang  · 2nd year  · 12th month Duke Zhuang  · 3rd year · 1st month
Comment ary on 
highlighted textsOmitting the clan name  signals censure...
Qi and Zheng feigned a visit to Ji to attack it...
Whenever he wanted Lady Jiang, she  came...
Duke Zhuang, Y ear 2, Month 12: annals/commentaries on...[Chunqiu] In the twelfth month of winter, Lady Jiang ...
[Guliang] On yiyou day, Duke Feng of Song died.
[Chunqiu] In winter of Year 2, Lady J iang met the Marquis of Qi ...
[Chunqiu] In winter, month 12 , on bingwu day: Qi and Wey...
[NEG] Duke Zhuang, Year 2, Month 12 : annals/commentaries on...
[NEG] In winter of Year 2, Lady Jiang met the Marquis of Qi...
[RECORD]  In the twelfth month of winter, Lady Jiang... Lady Jiang met the Marquis of Qi...Figure 1: A query about a specific month can retrieve
same-month commentary that repeats the date phrase, or
adjacent-month near-miss events with confusable word-
ing, so a retrieval-augmented model answers fluently
but at the wrong time.
Historical research on pre-modern Chinese
sources is a canonical example of evidence-centric
retrieval (Cao et al., 2024; Zhang et al., 2024; Liu
et al., 2025). Digitized annals, commentaries, and
later annotations are now searchable, but the tar-
get is rarely an arbitrary topical snippet: it is the
passage that records what happened in a particular
month of a particular duke’s reign. As Fig. 1 illus-
trates, a query such as “What happened in Duke
Zhuang’s 2nd year, 12th month?” can easily re-
trieve (i) exegetical commentary that repeats the
same date phrase without answering the event, or
(ii) near-duplicate events from adjacent months
with highly confusable wording. In this setting,
semantic relevance is insufficient without verifying
temporal alignment to the queried month. Once
retrieval binds a downstream generator to tempo-
rally incorrect but semantically plausible evidence,
the final answer may still sound fluent while being
wrong aboutwhenthe event happened.
This motivates a more focused question that is
central to faithful historical RAG:arXiv:2604.06997v1  [cs.CL]  8 Apr 2026

(Q)How can a retriever selecttime-
consistent evidencefor queries ex-
pressed undernon-Gregorian, reign-
based chronologies?
Studying this problem is already challenging be-
cause pre-modern records typically do not provide
explicit, globally comparable (Gregorian) times-
tamps (Chen et al., 2021, 2025). Instead, they em-
ploy a ruler-centric regnal chronology: time is ex-
pressed relative to the current ruler and his regnal
year and month, so temporal reference effectively
resets across reigns and must be interpreted on a
corpus-specific timeline rather than a monotonic
calendar. Moreover, temporal phrases are often
underspecified or written in shorthand—for exam-
ple, “in summer, in the fifth month” may omit the
absolute year and only become interpretable given
the surrounding reign context. Crucially, time is
not a clean metadata field separated from content:
in annalistic writing, distinctive one-off events can
implicitly function as temporal anchors, tightly cou-
plingwhenwithwhat. As a result, retrieval cannot
rely on semantic similarity or timestamp ordering
alone; it must identify evidence that is both topi-
cally relevant and temporally consistent with the
intended regnal point or window.
To tackle this challenge, we ground our study
in a demanding case: theSpring and Autumn An-
nalsand its commentarial–exegetical corpus. We
introduceChunQiuTR, a time-keyed benchmark
built on this material, where queries and records are
expressed in a ruler-centric, non-Gregorian chronol-
ogy rather than modern timestamps. Building on
this benchmark, we propose theCalendrical Tem-
poral Dual-encoder (CTD), a time-aware dual-
encoder retriever that augments semantic match-
ing with learned calendrical structure. CTD places
each query and record at a soft location on a unified
ordered calendar axis and favors pairs that agree
not only in meaning but also in calendrical posi-
tion. Concretely, it injects an absolute calendrical
context into embeddings and adds a relative tem-
poral bias to similarity based on signed calendar
offsets, improving robustness to adjacent-month
and lexical-near confounders.
Our contributions are threefold: (i) we introduce
ChunQiuTR, a non-Gregorian, reign-keyed tem-
poral retrieval benchmark with point/gap/window
queries and leak-free splits; (ii) we proposeCTD,
a calendrically time-aware dual-encoder that com-
bines absolute context injection with relative offsetbiasing; and (iii) we show consistent improvements
over strong semantic dual-encoder baselines, espe-
cially under chrono-near and adjacent-month con-
founders, supporting the view that retrieval-time
temporal consistency is a key prerequisite for faith-
ful downstream historical RAG.
2 Related Work
2.1 Neural Information Retrieval
Lexical retrievers such as BM25 (Robertson and
Zaragoza, 2009) remain strong and interpretable,
but are limited by surface-form overlap. Neu-
ral IR methods are often grouped into neural
sparse expansion (e.g., SPLADE (Formal et al.,
2021)), dense dual-encoders trained with con-
trastive learning (e.g., DPR (Karpukhin et al.,
2020), ANCE (Xiong et al., 2021), Contriever-
style (Lei et al., 2023)), and late-interaction token
matching (e.g., ColBERT (Khattab and Zaharia,
2020)). General-purpose embedding models (e.g.,
GTR (Ni et al., 2022b), E5 (Wang et al., 2024b),
Qwen3-Embedding (Zhang et al., 2025b)) further
enable plug-and-play retrieval. However, relevance
is typically modeled as semantic similarity, which
can still confuse chrono-near near-duplicates or fail
to enforce fine-grained temporal constraints with-
out explicit temporal structure.
2.2 Temporal Information Retrieval
Temporal information retrieval (TIR) incorporates
time into ranking, ranging from timestamp-aware
priors (e.g., time-based language models Li and
Croft (2003)) to modern formulations emphasizing
temporal focus/intent (Piryani et al., 2025). Recent
work explores neural retrieval for time-sensitive set-
tings by injecting temporal signals into retrieval or
generation pipelines (Rajapakse, 2023; Zhang et al.,
2025a), as well as mechanisms that encode time
specifiers into model behaviors (Han et al., 2025).
Related lines include time-aware language mod-
els with document-dating objectives and tempo-
ral label-smoothing schemes that smooth supervi-
sion over neighboring time steps, both of which we
echo in our auxiliary temporal heads (Wang et al.,
2023; Yèche et al., 2023; Dhingra et al., 2022).
Most TIR studies target modern timestamped col-
lections under open retrieval, whereas our setting is
a micro-granular,time-keyedchronicle with dense
chrono-near near-duplicates, leading to different
supervision and evaluation objectives.
Temporal-expression extraction and normaliza-

tion are also related to our setting, including
work on historical texts and cross-lingual temporal
expression extraction/normalization (Korchagina,
2016; Cao et al., 2022; Su et al., 2025; Castro et al.,
2025; Graciotti et al., 2025). However, in many
Chunqiu passages, the ruling duke and/or regnal
year is omitted, so the target month key must be
recovered from annalistic structure and discourse
context rather than extracted as a standalone tem-
poral mention.
3 Dataset Construction
To study temporal retrieval under non-Gregorian
dating systems, we constructChunQiuTR, a
benchmark curated from authentic historical texts
centered on theSpring and Autumn Annalsand
its classical commentarial tradition. The retrieval
gallery is derived from source texts rather than AI-
generated content, and the queries are instantiated
from a small set of manually written templates.
We use LLMs only as auxiliary tools to propose
candidate splits or candidate alignments during cu-
ration; they are never used to generate, rewrite,
translate, or paraphrase historical content, and only
human-approved results enter the final benchmark.
ChunQiuTR evaluates whether models can iden-
tify botheventsand theirtemporal positionswhen
time is expressed using reign-year references rather
than explicit Gregorian dates. Fig. 2 outlines the
data construction process. We next explain why
theChunqiuis a natural testbed and how the bench-
mark is instantiated.
3.1 Chunqiu Corpus and Temporal Scheme
Why theChunqiu?TheChunqiu( 《春秋》 ,
Spring and Autumn Annals) is a terse chronicle
of the state of Lu (722–481 BCE) with a long ex-
egetical and historiographical tradition built around
it. Its entries are extremely compact and date
events only using a reign-based temporal language.
Brief formulae such as “ 元年春 ” (first year, spring)
or “夏五月 ” (summer, fifth month) omit explicit
absolute years and often leave the ruler implicit.
Because many landmark events occur only once,
mentioning the event together with such a relative
month phrase often suffices to pin down a unique
point on the timeline. These compressed records
are then expanded and re-interpreted by the three
classicalzhuan(Zuo,Gongyang,Guliang) and later
commentarial and historiographical works.
Taken together, this layered structure makestheChunqiucorpus an ideal testbed for tempo-
ral retrieval: all layers align to the same ruler-
centric timeline without explicit Gregorian dates,
and different layers often describe the same event
or nearby months with overlapping phrasing, natu-
rally producing realistic “near-miss” hard negatives.
Similar combinations of a shared chronological
backbone and dense, overlapping commentary re-
cur in many pre-modern annalistic corpora, so we
use theChunqiuas a compact starting point for
methods that can later be generalized to broader
non-Gregorian historical collections. Additional
details on sources and preprocessing are provided
in Appendix A.1.
Reign-based time keys.Unlike modern texts
that use absolute year numbering (e.g., “709 BCE”),
our sources use a ruler-centric, reign-based dating
scheme in which regnal years restart for each new
Lu duke. A regnal year may begin with a full for-
mula such as “ 元年春王正月 ” (first year, spring,
royal first month), but later entries are often short-
ened to month phrases like “ 夏五月 ” (summer,
fifth month), with the duke and year supplied by
context—effectively “in the fifth month of summer
of this duke’s current regnal year.” Because many
key events in theChunqiuare unique, the event
description plus such a relative month phrase often
pins down a single point on the reign-based time-
line. This compact temporal language contrasts
with standard time-IR corpora, where documents
carry explicit Gregorian dates that can be treated
as fixed metadata rather than temporal expressions
to be interpreted.
Importantly, assigning a month-level time key in
this corpus is not reducible to standard temporal-
expression extraction. The ruling duke and/or reg-
nal year is often omitted and must be recovered
from annalistic structure, discourse continuity, and
surrounding context. We therefore manually verify
the final time-key assignment for all records rather
than relying on fully automatic extraction.
For our benchmark, we normalize this tem-
poral language into month-level time keys τ=
(gong,year,month) to index all records, where
gong denotes the ruling duke title (Chinese char-
acter “公”, glossed as “Duke” for readability).
Month-level is the finest temporal unit consis-
tently recorded in theChunqiu; finer-grained dates
(e.g., day-level) are largely absent or sporadic and
thus cannot be normalized systematically. We as-
sign a time key to every month, including months

… 魯隱公元年一月
Duke Yin, 
Year 1, Month 1魯隱公元年三月
Duke Yin, 
Year 1, Month 3魯哀公二十六年十月
Duke Ai, 
Year 26, Month 10(1) Records -Time -Key Alignment
No event, 
No record.魯隱公元年二月
Duke Yin, 
Year 1, Month 2
… 
(2) Chrono -Near Counterfactuals
元年春…三𫝊皆無… 
In the first regnal 
year… (Why not 
explicitly say 
“assumed the 
throne ”?)三月公…父音甫邾 … 
The Duke Yin and Yi -
fu… (Who is "Yi-fu"?… 
Why did he ally with 
him? )
卒于…言大尹所弑 … 
The Duke Ai died… 
(Commentary: he was 
assassinated by Da 
Yin.)
from later historiesfrom annals & zhuan
722-481 BCE(a) P-Time Queries (probing recorded months )
No event, 
No record.
Year 1, 
Month 1Year 1, 
Month 2Year 12, 
Month 11Year 1, 
Month 1
魯隱公  (Duke Yin) 魯桓公  (Duke Huan)
Records
Hard negatives春王正月三 𫝊
皆無明文左氏
以正月為建子
得之矣而畧 …  不書葬不成喪
也注桓弑隠篡
位故喪禮不
成…  經元年春王正
月公卽位注嗣
子位定於初喪
而改元…  (b) G-Time  Queries  (probing no-event  / empty -month  cases )
(c) W-Time Queries (probing multi -month ranges with diverse events )
(3) Temporal Probe Queries
(4) Reign -Aligned Temporal Splits… … … … 
Annals  and Exegesis  of Duke YinExample -(a) What happens in    ?
Example -(b) Is there a Chunqiu  entry for    ?
Example -(c) What happens i n the three months around     ?
P-Time /G-Time /W-Time  Queries per Reign
contiguous month s (val) contiguous months  (test) Training
Gallery: masking val/test  months  in training.Q-split: per -reign contiguous months.
Training Training
冬十月公游于空澤
辛巳卒于連中
The Duke Ai died 
in Lianzhong 
during … 
三月公及邾
儀父盟于蔑
The Duke Yin 
and Yi -fu formed 
an alliance … 
元年春王正月  
In the first 
regnal year of 
Duke Yin...
元年春…  
元年春王
正月公即
位… 
冬十有一
月壬辰公
薨… 


event month      empty monthFigure 2: Overview of the ChunQiuTR construction. (Left) Time-key alignment yields event-level records,
augmented with chrono-near counterfactual hard negatives from later histories. (Right) We define P/G/W temporal
queries and leak-free, reign-aligned splits.
with no annals entry; we later instantiate them
as standardized no_event placeholders. We then
align each annals entry and its associated histo-
riographical passages to a unique τ, forming a
time-keyed gallery of short records (Section 3.2,
Appendix A.2.1). Although in theChunqiuthis
appears as a concrete gong –year–month scheme,
the pipeline itself only assumes an ordered set of
time keys with aligned texts and thus applies to
other non-Gregorian chronologies.
3.2 Record Alignment
Record–time-key alignment.Building on the
reign-based month keys τ= (gong,year,month)
defined above, we define the record units used
for retrieval as shown in Fig. 2(1). We treat each
recordas the atomic retrieval unit: a short event-
level passage aligned to a single month key τ.
For each time key τ, we gather all snippets from
the annals and the three classicalzhuanand re-
fine them into event-level record sets Dτ. This
is non-trivial: the annals often compress several
events into a single sentence, while the commen-
taries may spread one event across multiple frag-
ments, so naive sentence or paragraph boundaries
do not match historical events. We therefore use
a lightweight LLM prompt only to propose candi-
date splits and groupings under each τ, and then
manually review and correct these proposals so
that each final record corresponds to one coherent
event with its aligned commentarial material. This
yields a high-precision, time-keyed collection ofevent-level records for the entireChunqiuperiod
(Appendix A.2.2).
Chrono-near counterfactual negatives.Later
historiographical layers, such as Gu Donggao’s
Chronological Tables, re-group and paraphrase the
same Spring and Autumn events instead of in-
troducing new ones, creating naturally occurring,
easy-to-confuse off-topic variants. As shown in
Fig. 2(2), we align these sources to our reign-
based time keys using LLM-assisted candidate
matching together with fuzzy string matching (Ap-
pendix A.3). Here again, the LLM is used only
to propose candidate source passages; final align-
ments are retained only after human verification.
For each time key τ, we define Dcf
τas records from
these layers that share the same time key and de-
scribe the same situation in later paraphrase, but
arenot used as ground-truth retrieval targetsfor
queries targeting τ. These historically grounded
near-miss variants serve as hard negatives for time-
aware retrieval.
Audit and reliability.We summarize three reli-
ability decisions here and defer full details to the
appendix. First, time-key normalization is manu-
ally verified because many passages omit an ex-
plicit duke or regnal year and must be resolved
from annalistic structure and discourse continuity
rather than extracted as standalone temporal men-
tions. Second, among 1,533 non-empty months,
558 contain multiple events; after LLM candidate
grouping, only 63 required additional human cor-

rection, while the remaining 495 were accepted
without change. Third, for later-commentary align-
ment, LLMs only propose candidate matched pas-
sages, and the final human acceptance rate ranges
from 93.33% to 100% across sources. These audits
indicate that LLM assistance improves curation effi-
ciency, while final benchmark quality is controlled
by explicit human verification (Appendix B.5).
3.3 Temporal Queries and Evaluation Splits
Temporal query design.Building on the time-
keyed records {Dτ}, we design three families of
temporal queries that mirror common ways his-
torians ask about time: point queries (P-Time),
gap queries (G-Time), and local-window queries
(W-Time) as shown in Fig. 2(3). Each query is
mapped to a target interval Qion the reign-based
timeline; span (single month vs. short multi-month
window) and eventfulness (whether Qicontains
empty months) vary independently, so any fam-
ily can in principle target either eventful or empty
periods.
To make gaps and empty intervals queryable,
months with no annals entry are instantiated as
standardized no_event records keyed by τand
included in the retrieval gallery. G-Time queries
are gap-oriented: they ask which month(s) within
a reign or specified range lack recorded events and
are answered by theseno_eventrecords.
P-Time queries explicitly target a single time key
(e.g., “What happens in 鲁隐公元年三月 ?”), and
are answered by the corresponding event-bearing
record, or by an explicit no_event record when
that month is empty. W-Time queries target a local
temporal window, such as “around the time when
Xoccurs” or “in the months before/after Y,” and
are mapped to short contiguous ranges of time keys
(see Fig. 7 for representative patterns). All query
templates and concrete examples are provided in
Appendix A.4.
Reign-aligned splits.We partition the month-
level timeline into train/validation/test splits in a
reign-aware, leak-free way. For each duke’s reign,
we allocate disjoint contiguous blocks of months to
train, validation, and test, roughly in an 80/10/10
ratio, and assign all records and queries in those
months to the corresponding split. No time key,
record, or query ever appears in more than one split.
At evaluation time, validation and test queries are
drawn from their held-out reign segments, while
retrieval is always performed over the full time-keyed record gallery, so models must generalize
temporal reasoning from seen parts of a reign to
months they haveneverobserved during training.
Overall, the benchmark contains 20,172 records
and 16,226 queries (13,053 train / 1,520 validation
/ 1,653 test); detailed statistics and split visualiza-
tions are provided in Appendix A.5.
4 Methods
We first formalize ChunQiuTR as a time-keyed
retrieval task in Sec. 4.1. Sec. 4.2 then presents
our Calendrical Temporal Dual-encoder (CTD):
starting from a semantic dual-encoder score, CTD
learns a latent regnal calendar scalar and incorpo-
rates anabsolutecalendrical context and arelative
temporal bias to form the final score, as illustrated
in Fig. 3. Finally, Sec. 4.3 describes our interval-
overlap multi-positive supervision and joint train-
ing objective.
4.1 Task Formulation
We cast our ChunQiuTR benchmark as a temporal
retrieval task over a discrete, reign-based month
timeline. Building on the time-keyed records in
Sec. 3.2, we formalize all aligned historical mate-
rial as a fixed retrieval gallery
D={d j}N
j=1,
where each short Classical Chinese record djis
associated with a reign-based month key τ(dj); we
writeDτ={d j∈ D:τ(d j) =τ} for the subset
under time keyτ.
For each query qiconstructed in Section 3.3,
the benchmark specifies a target interval Qion the
same month axis and a small multi-positive ground-
truth set
Gi⊆ D.
Ground-truth records dj∈ G iare exactly those
that describe events or explicit non-events recorded
during the queried interval Qi, i.e., their time keys
τ(dj)fall within Qi. The learning objective is to
train a scoring function Sθ(qi, dj)that, for each
query, ranks its ground-truth set Giahead of the
remaining elements ofD.
4.2 CTD: Calendrical Temporal Dual-encoder
We instantiate Sθwith a standard dual-encoder re-
triever. A shared Transformer encoder fθ(·)maps
both temporal queries and candidate records into
a common embedding space, producing pooled

Score matrix
Discrete Time Key
Duke 
Min
Duke 
AiDuke 
YinSharedQuery :
    #1 "在魯莊公二年十月 … "
         (… in Duke Zhuang , Year 2, month 10)
    #2 "魯昭公九年三月在 … "
         (… in Duke Zhao , Year 9, month 3)
Record:
        #1(+) "二年冬，夫人姜氏 … "
        #1(-) "翬㑹宋公 、陳侯… "
        #2(+) "魯昭公九年三月 … "
        #2(-) "后稷封殖天下 … "
qi
d1
d2
d3
… timeline
POS POS POS1 1 0 0 1 0 1 1 0 0 1 0Pi mask(c) Interval -overlap supervision
 (b) R elative -Temporal Learning (a) A bsolute -Temporal Learning
Fixed Shared CodebookEmbedding
Time head
… … (c)  Interval -overlap sup . (a) Absolute -Temporal and (b) Relative -Temporal Learning
… … 
)(•f
)(•f
Time head……
……
………
…
…Latent Regnal Calendar
… 
… 元年春王正月
(Duke Yin, Year 1)冬十月乙亥
(Duke Xuan, Year 2)
冬天王使宰周公
(Duke Xi, Year 30)
… Duke Xuan
Duke Yin
Duke XiDiscrete  Time -Key Continuous
Latent Time Axis
PositionAbsolute
iQ
jI
Year 1,2, … Continuous Timeline
xu Latent Time Scalar
……
……
………
…
…
Fourier kernel
)(u Fourier kernel
)(u
Discrete  Time Key
xhqi
di
No overlap
()jd
()iq
time
ijabs
ijCTD
ij b s s +=
CTD
ijs
+=x x x ch h~
)( )( )(, ,m
xy
xg
x p p pΔTimeRelative
ΔTimeRelative
iqu
jdu
j i d quu,
i j q d ij u u u−=
1,1−ijuFigure 3: Overview of our Calendrical Temporal Dual-encoder (CTD). A shared Transformer dual-encoder encodes
queries and records into embeddings. Temporal heads place each text on a unified regnal calendar axis (latent
time scalar), supporting (a) absolute context injection and (b) relative biasing to form sCTD
ij. (c) Interval-overlap
supervision marks in-batch multi-positives by query–record overlap and trains a multi-positive contrastive loss.
embeddings hqi,hdj∈RH. As apurely se-
manticbaseline, we compute temperature-scaled
dot-product similarities ssem
ij=ssem(qi, dj) =
h⊤
qihdj/α,for a mini-batch of Bqueries {qi}B
i=1
andBrecords {dj}B
j=1. Building on this seman-
tic score, CTD augments the retriever with (i) an
absolutecalendrical context injected into the em-
beddings and (ii) arelativetemporal bias added to
the similarity, so that matches must agree in both
meaning and calendrical position.
4.2.1 Latent calendar scalar
Reign-based month keys are discrete identifiers and
do not directly provide a metric notion ofposition
ordistanceacross the stitched regnal calendar. To
support both absolute positioning (for context injec-
tion) and relative offsets (for biasing), we therefore
learn a continuous calendar axis where temporal
relations become measurable.
For any text x(either a query qior a record dj),
lethx∈RHdenote its pooled embedding. On top
ofhx, we attach three lightweight prediction heads
forgong,year, andmonth. Each head produces log-
its over its discrete index set, which we normalize
into distributions p(g)
x,p(y)
x,p(m)
x. Taking expecta-
tions yields soft calendrical coordinates gx, yx, mx,
which locatexon the ruler–year–month grid.
We then linearize this grid in calendar order and
normalize it to [0,1] , defining a shared latent time
scalar
ux=gx·(Y·M) +y x·M+m x
G·Y·M−1∈[0,1].Here G,Y, and Mdenote the (padded) maximum
numbers of gongs, years-per-gong, and months-
per-year used to index the unified calendar. Texts
from earlier dukes, years, or months receive smaller
uxthan those later in the chronicle, enabling both
relative distances ∆uand absolute positions to be
modeled on the same axis.
4.2.2 Absolute-temporal learning
We first exploit this signal in anabsoluteman-
ner (Fig. 3 (a)): instead of feeding discrete
(gong,year,month) indices as hard metadata, we
convert the heads’softpredictions into a continu-
ous context vector and inject it into the embedding.
Reusing p(g)
x,p(y)
x,p(m)
x, we map each calendri-
cal index to a fixed Fourier-style code and build
sinusoidal codebooks
E(g)∈RG×D t, E(y)∈RY×D t, E(m)∈RM×D t.
This fixed sinusoidal codebook provides a smooth,
non-parametric absolute-position signal, avoiding
a large learned embedding table for sparse calendri-
cal indices. Taking expectations under p(·)
xyields
a mixture representation that naturally reflects the
model’s uncertainty instead of committing to a sin-
gle hard index.
We obtain soft absolute-time contexts by taking
expectations:
c(g)
x=p(g)
xE(g),c(y)
x=p(y)
xE(y),c(m)
x=p(m)
xE(m).
Concatenating and projecting yields
cx=W ctx
c(g)
x;c(y)
x;c(m)
x
∈RH,

which we inject via a scalar-gated residual
˜hx=hx+γc x,
whereγis learned.
We compute similarities with the context-
enriched representations,
sabs
ij=˜h⊤
qi˜hdj/α,
which reduces to the semantic baseline ssem
ijwhen
γ= 0.
4.2.3 Relative-temporal learning.
Building on the absolute similarity sabs
ij, we further
use the learned calendar axis to bias matching by
relative offsets (Fig. 3 (b)). Given the latent coordi-
nates uqiandudjfor a query–record pair (qi, dj),
we form the temporal offset
∆uij=udj−uqi∈[−1,1],
so that distances along the learned timeline can
modulate how easily two texts should match. We
embed this scalar with Fourier-style features
ϕ(∆u ij)∈RDϕ,
and apply a small MLP to produce an additive tem-
poral bias
btime
ij=ϵMLP 
ϕ(∆u ij)
.
The final retrieval score is
sCTD
ij=sabs
ij+btime
ij,
where the learnable scale ϵ(initialized near zero)
keeps the bias lightweight: when ϵ= 0 , CTD
reduces to the absolute-only scorer sabs
ij, and more
generally the model can downweight this term if
the learned calendar signal is unreliable.
4.3 Learning Objectives
We train the purely semantic dual-encoder baseline
with a symmetric single-positive InfoNCE (Chen
et al., 2020) objective over ssem
ij. For CTD, we
instead optimize a temporally aware multi-positive
retrieval loss using the final scoressCTD
ij.
Interval-overlap multi-positive retrieval.As
shown in Fig. 3 (c), we treat temporal overlap as
weak supervision: each query qitargets an interval
Qi= [τmin
i, τmax
i], and each record djcarries asingle month key τ(dj)(i.e.,Ij= [τ(d j), τ(d j)]).
We mark in-batch positives by overlap,
Pi={j|Q i∩Ij̸=∅},
and optimize a multi-positive InfoNCE loss using
the final scoressCTD
ij:
Lmulti
q=−1
BBX
i=1logP
j∈Piexp
sCTD
ij
PB
k=1exp 
sCTD
ik.
The remaining in-batch records serve as negatives.
We define Lmulti
dsymmetrically by transposing 
sCTD
ij
and use
Lmulti=1
2 
Lmulti
q+Lmulti
d
.
Auxiliary calendrical classification.To stabi-
lize the absolute calendrical signal, we supervise
the gong/year/month heads on passages with cross-
entropy:
Ltime=E d∼batchP
r∈{g,y,m} CE
p(r)
d, y(r)
d
where y(r)
dare the ground-truth calendrical labels
from the aligned time keys (queries are unlabeled).
Overall objective.We jointly optimize the re-
trieval and auxiliary temporal losses with a small
weightλ time:
Ltotal=L multi+λ timeLtime.
5 Experiments
5.1 Experiment Setting
We fine-tune two dual-encoder backbones, BERT-
BASE-CHINESEand QWEN3-EMBED-0.6B, on
ChunQiuTR. Model details are deferred to Ap-
pendix B.1.1, training cost and compute settings
to Appendix B.1.2, and the full list of compared
methods to Appendix B.3.
5.2 Main Results
From Table 1, ChunQiuTR is clearly non-trivial
and strongly time-sensitive: most zero-shot
sparse, fusion, and dense retrievers lag behind
tuned BM25, while a simple temporal prior
(BM25+TimeKDE) yields a large gain over BM25
and nearly matches supervised dense models.
On encoder-based dense retrievers, in-domain
fine-tuning already outperforms BM25+TimeKDE,
generic dating auxiliaries (TempDate / TempDate-
Smooth) give little benefit, and adding our CTD

Method Pub Type R@1 R@5 R@10 MRR@10 nDCG@10
Sparse retrieval
BM25 – Sparse 0.3962 0.5209 0.5620 0.4487 0.3404
BM25+TimeKDE – Sparse 0.4943 0.6086 0.6709 0.5456 0.4222
SPLADE-IDF (ZS) arXiv’24 Sparse 0.1361 0.2765 0.3569 0.1971 0.1596
SPLADE-ℓ 0(ZS) SIGIR’25 Sparse 0.0006 0.0309 0.0587 0.0132 0.0143
Fusion / late interaction
ColBERT-JINA (ZS) MRL’24 Fusion 0.2498 0.4102 0.4743 0.3167 0.2569
ColBERT-LFM2 (ZS) arXiv’25 Fusion 0.3345 0.4567 0.4894 0.3865 0.2691
Dense retrieval (encoder-based)
mE5-Large (ZS) arXiv’24 Dense 0.2916 0.3969 0.4574 0.3389 0.2441
mE5-Large-ins (ZS) arXiv’24 Dense 0.2359 0.3545 0.4162 0.2862 0.2358
GTE-Large (ZS) arXiv’23 Dense 0.2293 0.3527 0.3890 0.2826 0.2188
BGE-Large-v1.5 (ZS) arXiv’23 Dense 0.2208 0.3430 0.4144 0.2775 0.2280
BGE-m3 (ZS) Findings ACL’24 Dense 0.2698 0.3775 0.4253 0.3135 0.2299
BERT-base (FT) NAACL’19 Dense 0.5088 0.6279 0.6727 0.5597 0.4283
BERT-base + TempDate (FT) SIGIR’23 Dense 0.5027 0.6165 0.6691 0.5508 0.4243
BERT-base + TempDate-Smooth (FT) PMLR’23 Dense 0.5051 0.6152 0.6673 0.5519 0.4244
CTD BERT-base (Ours)This work Dense0.5826 0.6721 0.7090 0.6193 0.4575
Dense retrieval (LM-based embeddings)
GTE-Qwen2-1.5B (ZS) arXiv’23 Dense 0.2783 0.4453 0.5009 0.3501 0.2613
E5-mistral-7B (ZS) ACL’24 Dense 0.2196 0.3212 0.3684 0.2619 0.2359
PQR (Qwen2.5-7B) (re) ACL’25 Dense 0.1585 0.3134 0.3805 0.2226 0.1712
PQR (Qwen3-8B) (re) ACL’25 Dense 0.0901 0.2184 0.3152 0.1481 0.1184
Qwen3-Embed-0.6B (ZS) arXiv’25 Dense 0.3376 0.4852 0.5354 0.3973 0.3107
Qwen3-Embed-4B (ZS) arXiv’25 Dense 0.4410 0.5783 0.6013 0.4985 0.3793
Qwen3-Embed-0.6B (FT) arXiv’25 Dense 0.5771 0.6376 0.6818 0.6045 0.4460
Qwen3-Embed-0.6B + TempDate (FT) SIGIR’23 Dense 0.5523 0.6425 0.6630 0.5924 0.4391
Qwen3-Embed-0.6B + TempDate-Smooth (FT) PMLR’23 Dense 0.5638 0.6346 0.6727 0.5942 0.4396
CTD Qwen3-Embed-0.6B (Ours)This work Dense0.5923 0.6485 0.6927 0.6194 0.4575
Table 1: Test-set retrieval performance on our ChunQiuTR benchmark under the official evaluation protocol.
objectives on the same backbone yields a clear
boost in early precision (around +7–8 points on
R@1). LM-based dense retrievers show a sim-
ilar pattern: zero-shot LM encoders and PQR
pipelines underperform BM25+TimeKDE, lightly
fine-tuned Qwen3-Embed-0.6B is strong, and the
CTD-enhanced variant further improves early pre-
cision and achieves the best overall scores (R@1,
MRR@10, nDCG@10), indicating that explicit
time-key supervision adds fine-grained temporal
structure beyond simple priors or auxiliary dating
heads.
Cross-corpus pilot onZizhi Tongjian.As an out-
of-domain probe, we further evaluate ChunQiuTR-
trained retrievers on two processed subsets from
Zizhi Tongjian, an annalistic general history that
also records events under non-Gregorian, reign-
based temporal expressions (Appendix B.4). For
each subset, we build a month-level gallery from
event-bearing lines and automatically instantiate
one point-style query for each unique month key
using traditional reign-year expressions, without
any additional training on the target corpus. Un-
like the full ChunQiuTR benchmark, this pilot
does not reconstruct explicit no_event months,
commentary-derived hard negatives, or the full
point/gap/window query families, and is thereforeSubset Records QueriesFT baseline CTD (ours)
MRR R@1 MRR R@1
Qi Ji (part) 268 92 0.2081 0.1848 0.2304 0.2065
Jin Ji (part) 820 119 0.1598 0.1345 0.1751 0.1597
Table 2: Cross-corpus pilot on processedZizhi Tongjian
subsets. No target-corpus training is performed.
intended as a lightweight cross-corpus transfer
probe rather than a second benchmark.
As shown in Table 2, CTD consistently improves
both MRR and R@1 on the twoZizhi Tongjian
subsets without any target-corpus fine-tuning. Al-
though this pilot is intentionally lighter than the
full ChunQiuTR setup, the trend suggests that the
temporal-consistency bias learned on ChunQiuTR
transfers beyond the source corpus and continues
to help distinguish chrono-near but temporally mis-
matched evidence.
5.3 Analysis
5.3.1 Impact of Query Type
Table 3 compares single-month ( |Q|= 1 ) and
multi-month ( |Q|>1 ) queries. Across most
methods, multi-month queries substantially boost
MRR@10 and give small gains or no change in
R@1, reflecting the fact that it is easier to hit any
correct month within a span than a single target

Method Single-month (|Q|= 1) Multi-month (|Q|>1)
R@1 MRR@10 R@1 MRR@10
BM25 0.397 0.413 0.396↓-0.0010.499↑+0.086
ColBERT-LFM2 (ZS) 0.298 0.312 0.386↑+0.0880.490↑+0.178
mE5-Large (ZS) 0.259 0.279 0.337↑+0.0780.422↑+0.143
BERT-base (FT) 0.497 0.516 0.525↑+0.0270.621↑+0.106
CTD BERT-base (Ours)0.509 0.530 0.685↑+0.1760.744↑+0.214
Qwen3-Embed-0.6B (ZS) 0.353 0.385 0.317↓-0.0360.415↑+0.031
Qwen3-Embed-0.6B (FT) 0.481 0.495 0.711↑+0.2300.757↑+0.262
CTD Qwen3-Embed-0.6B (Ours)0.491 0.513 0.733↑+0.2420.767↑+0.255
Table 3: Impact of query span on retrieval performance
on the test set. We compare single-month ( |Q|= 1 ) and
multi-month ( |Q|>1 ) queries; for multi-month queries,
the numbers in parentheses give absolute changes rela-
tive to single-month queries.
 魯莊公四年三月期間 ，《春秋》主要記錄了什麼 ？
#1 魯莊公三年三月 。《春秋》及三傳皆不載此月之事 。
#2 四年春，王二月，夫人姜氏饗齊侯于祝丘 。三月，紀伯姬卒 。
#3 四年春，王三月庚戌 ，盜殺蔡侯申 。弒君賤者窮諸人 … #1 ★四年春，王三月，楚武王荊屍 ，授師孑焉 ，以伐隨… 
#2 魯莊公三年三月 。《春秋》及三傳皆不載此月之事 。
#3 魯桓公四年三月 ：經傳於是月史事闕如 ，無專條可錄 。
在魯哀公七年十一月之後的一個月裡 ，《春秋》是否另有記事 ？
#1 魯哀公七年十一月 。《春秋》經文及三傳於此月無事可書 。
#2 ★魯哀公七年十二月 。《春秋》及三傳皆不載此月之事 。
#3 魯哀公七年九月 ：經傳於是月史事闕如 ，無專條可錄 。#1 ★魯哀公七年十二月 。《春秋》及三傳皆不載此月之事 。
#2 魯哀公七年十一月 。《春秋》經文及三傳於此月無事可書 。
#3 魯哀公七年二月 。《春秋》及三傳皆不載此月之事 。
魯昭公十七年四月之前半年之內 ，《春秋》主要記事是什麼 ？
#1 昭十七年傳稱黃帝以雲名官 ，故知縉雲黃帝時官名 … 
#2 金天國號少皥 ，身號譙周云 ：“金天氏能脩大皥之法 … 
#3 魯昭公十七年四月 。《春秋》經文及三傳於此月無事可書 。#1 ★魯昭公十七年二月 。《春秋》及三傳皆不載此月之事 。
#2 ★十有七年春 ，小邾子來朝 。
#3 魯昭公十七年十二月 。《春秋》經文及三傳於此月無事可書 。QUERY (point ):
QUERY (window ):
QUERY (window ):
Baseline
Baseline
BaselineOurs
Ours
OursWhat does the Chunqiu mainly record in Duke Zhuang ’s Year 4, Month 3?
In the month after Duke Ai ’s Year 7, Month 11, is there any further entry in the Chunqiu?#1 Duke Zhuang, Y ear 3, Month 3: no annals entry in this month.
#2 In spring of Year 4, month 2, Lady Jiang feasted the Marquis of Qi at Zhuqiu. In month 3, Ji Boji died.
#3 In spring of Year 4, month 3, gengxu: Cai Hou Shen was murdered...#1 ★ In spring of Year 4, month 3, King Wu of Chu marched against Sui...
#2 Duke Zhuang, Y ear 3, Month 3: no annals entry in this month.
#3 Duke Huan, Y ear 4, Month 3: no annals entry in this month.
#1 Duke Ai, Y ear 7, Month 11: no annals entry in this month.
#2 ★ Duke Ai, Y ear 7, Month 12: no annals entry in this month.
#3 Duke Ai, Y ear 7, Month 9: no annals entry in this month.
#2 The state title of Jintian was Shaohao; Qiao Zhou says...#1 The commentary says that the Yellow Emperor named officials after clouds...
#3 Duke Zhao, Y ear 17, M onth 4: no annals entry in this month.#1 ★ Duke Ai, Y ear 7, Month 12: no annals entry in this month.
#2 Duke Ai, Y ear 7, Month 11: no annals entry in this month.
#3 Duke Ai, Y ear 7, Month 2: no annals entry in this month.
#1 ★ Duke Zhao, Y ear 17, M onth 2: no annals entry in this month.
#2 ★ In spring of Year 17, the ruler of Xiao Zhu came to court.
#3 Duke Zhao, Y ear 17, M onth 12: no annals entry in this month.What does the Chunqiu mainly record in the half -year before Duke Zhao ’s Year 17, M onth 4?
Figure 4: Visualization of Qualitative Examples.
month. Our CTD models achieve the best perfor-
mance in both regimes, with especially large gains
on multi-month queries for the BERT backbone
(roughly +0.16 R@1) and consistent improvements
for Qwen3-Embed, indicating better temporal or-
dering under chrono-near confounds.
5.3.2 Qualitative Examples
Fig. 4 illustrates the contrast between a BERT-
based baseline and our time-aware retriever via
two queries. For a point query, the baseline is dis-
tracted by frequent no_event templates and events
from neighboring months, while our model cor-
rectly locates the target chronicle entry. For a
broader window query, the baseline’s results are
dispersed across later exegetic discussions, whereas
our model concentrates probability on the correct
local window, retrieving both the pertinent event
and an explicit no_event record. Overall, these
examples show that our retriever couples temporal
reasoning with semantic matching beyond surface
cues; moreover, temporal errors are oftenconfident
rather than uncertain, motivating retrieval-time tem-
poral constraints over downstream generation fixes
(see Appendix B.2.4 and Appendix B.2.5).
5.3.3 Ablation study
We ablate three components: the retrieval objective
Lmulti, the relative-time logit bias btime
ij, and the softVariantL multibtime
ijcx R@1 MRR@10
FT baseline – – – 0.5771—-0.6044—-
+L multi ✓– – 0.5820↑+0.00490.6107↑+0.0063
+ Bias✓ ✓– 0.5898↑+0.01270.6135↑+0.0091
+ Ctx✓–✓0.5850↑+0.00790.6134↑+0.0090
Full (Ours)✓ ✓ ✓0.5923↑+0.01520.6194↑+0.0150
Table 4: Ablation study on the test set under the same
evaluation protocol as Table 1. For each metric, the sec-
ond line reports the change relative to the FT baseline.
absolute temporal context cx. Starting from the FT
baseline, adding Lmultileads to a modest but con-
sistent improvement. This indicates that explicit
retrieval supervision enhances time-key discrim-
ination beyond standard fine-tuning. Adding ei-
ther temporal signal further improves performance.
The logit bias yields a larger gain in R@1, suggest-
ing that it effectively reshapes in-batch matching
toward chronologically plausible candidates. In
contrast, injecting temporal context achieves com-
parable improvements in MRR, indicating better
overall ranking quality. Combining the bias and
the context produces the best results, with additive
gains over each individual component. This sup-
ports their complementary roles: the bias calibrates
pairwise similarities, while the context enriches
representations with absolute-time distributional
cues.
6 Conclusion
We presentedChunQiuTR, a time-keyed tem-
poral retrieval dataset built on theSpring and
Autumn Annalsand its commentarial tradition.
ChunQiuTR operationalizes a non-Gregorian,
reign-based month timeline (gong–year–month)
and evaluates retrieval under realistic histori-
cal confounders—lexical-near same-key materi-
als, adjacent-month near-misses, and explicit
no_event months—making temporal fidelity a
first-class requirement beyond topical relevance.
We further proposedCTD, a calendrically time-
aware dual-encoder that augments semantic match-
ing with absolute context injection and relative off-
set biasing. Against strong semantic dual-encoder
baselines, CTD consistently improves retrieval
quality and reduces chrono-near confusions that
can mislead evidence-grounded systems.
Limitations
ChunQiuTR is constructed from theChunqiuan-
nals and its major commentaries, and uses a reign-
based month-level time key. This narrow scope

limits the generality of our findings: retrieval be-
haviors and error patterns may differ in other pre-
modern corpora with different calendrical conven-
tions, narrative styles, and editorial traditions, and
our results do not directly imply the same gains un-
der those settings. Extending the same construction
procedure to other dynastic corpora would require
additional source-specific normalization and time
alignment, and we leave such expansions to future
work.
Moreover, a month-level timeline cannot repre-
sent finer-grained temporal relations, and remain-
ing errors indicate that (i) near-duplicate records in
neighboring months and (ii) genuinely ambiguous
historiographical cases are still challenging even
with temporal supervision. Future work includes
extending the benchmark to broader historical cor-
pora and finer temporal granularity, incorporating
stronger reranking or evidence-checking for bor-
derline confusions, and evaluating downstream im-
pacts in end-to-end RAG pipelines.
Acknowledgments
This work was supported in part by the National
Natural Science Foundation of China (NSFC) un-
der Grant 62276283 and 62372281, in part by
the China Meteorological Administration’s Sci-
ence and Technology Project under Grant CMA-
JBGS202517, in part by Guangdong-Hong Kong-
Macao Greater Bay Area Meteorological Technol-
ogy Collaborative Research Project under Grant
GHMA2024Z04, in part by Fundamental Research
Funds for the Central Universities, Sun Yat-sen
University under Grant 23hytd006 and 23hytd006-
2, in part by Guangdong Provincial High-Level
Young Talent Program under Grant RL2024-151-2-
11, in part by the Key Development Project of the
Artificial Intelligence Institute, Sun Yat-sen Univer-
sity, and in part by The Major Key Project of PCL
(Grant No. PCL2025A17).
Ethical Considerations
Our data are derived from pre-modern Chinese his-
torical texts (theChunqiuannals and commentarial
tradition) and do not contain information about liv-
ing individuals. We use publicly available digital
editions and licensed scholarly resources; when
redistribution is permitted, we release processed
splits and annotations, otherwise we provide code
and metadata sufficient to reproduce the benchmark
from legitimate sources.These chronicles reflect historical norms and
power structures (e.g., elite-centric perspectives
and occasional descriptions of conflict or punish-
ment). We preserve such content for historical and
linguistic research rather than endorsement. We
caution against deploying models trained on this
benchmark in high-stakes decision-making, and
recommend that any downstream RAG or narrative
generation make the historical context explicit and
include appropriate uncertainty signaling.
We follow the ACL Code of Ethics and complete
the Responsible NLP checklist to the best of our
knowledge.
References
Jiahuan Cao, Dezhi Peng, Peirong Zhang, Yongxin
Shi, Yang Liu, Kai Ding, and Lianwen Jin. 2024.
TongGu: Mastering classical Chinese understanding
with knowledge-grounded large language models. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2024, pages 4196–4210, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Yuwei Cao, William Groves, Tanay Kumar Saha, Joel
Tetreault, Alejandro Jaimes, Hao Peng, and Philip Yu.
2022. XLTime: A cross-lingual knowledge transfer
framework for temporal expression extraction. In
Findings of the Association for Computational Lin-
guistics: NAACL 2022, pages 1931–1942, Seattle,
United States. Association for Computational Lin-
guistics.
Alejandro Sánchez de Castro, Lourdes Araujo, and Juan
Martinez-Romo. 2025. A novel methodology for
enhancing cross-language and domain adaptability in
temporal expression normalization.Computational
Linguistics, 51(4):1303–1335.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. 2020. A simple framework for
contrastive learning of visual representations. In
Proceedings of the 37th International Conference
on Machine Learning, volume 119 ofProceedings
of Machine Learning Research, pages 1597–1607.
PMLR.
Wenhu Chen, Xinyi Wang, William Yang Wang, and
William Yang Wang. 2021. A dataset for answer-
ing time-sensitive questions. InProceedings of the
Neural Information Processing Systems Track on
Datasets and Benchmarks, volume 1.

Ziyang Chen, Erxue Min, Xiang Zhao, Yunxin Li, Xin
Jia, Jinzhi Liao, Jichao Li, Shuaiqiang Wang, Baotian
Hu, and Dawei Yin. 2025. A question answering
dataset for temporal-sensitive retrieval-augmented
generation.Scientific Data, 12(1):1855.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language under-
standing. InProceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long and Short Papers), pages
4171–4186, Minneapolis, Minnesota. Association for
Computational Linguistics.
Bhuwan Dhingra, Jeremy R. Cole, Julian Martin
Eisenschlos, Daniel Gillick, Jacob Eisenstein, and
William W. Cohen. 2022. Time-aware language mod-
els as temporal knowledge bases.Transactions of the
Association for Computational Linguistics, 10:257–
273.
Thibault Formal, Benjamin Piwowarski, and Stéphane
Clinchant. 2021. Splade: Sparse lexical and expan-
sion model for first stage ranking. InProceedings
of the 44th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
SIGIR ’21, page 2288–2292, New York, NY , USA.
Association for Computing Machinery.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Zhichao Geng, Yiwen Wang, Dongyu Ru, and Yang
Yang. 2025. Towards competitive search relevance
for inference-free learned sparse retrievers.Preprint,
arXiv:2411.04403.
Arianna Graciotti, Leonardo Piano, Nicolas Lazzari,
Enrico Daga, Rocco Tripodi, Valentina Presutti, and
Livio Pompianu. 2025. KE-MHISTO: Towards a
multilingual historical knowledge extraction bench-
mark for addressing the long-tail problem. InFind-
ings of the Association for Computational Linguistics:
ACL 2025, pages 20316–20339, Vienna, Austria. As-
sociation for Computational Linguistics.
SeungYoon Han, Taeho Hwang, Sukmin Cho, Soyeong
Jeong, Hoyun Song, Huije Lee, and Jong C. Park.
2025. Temporal information retrieval via time-
specifier model merging. InProceedings of the 3rd
Workshop on Towards Knowledgeable Foundation
Models (KnowFM), pages 1–13, Vienna, Austria. As-
sociation for Computational Linguistics.
Rohan Jha, Bo Wang, Michael Günther, Georgios Mas-
trapas, Saba Sturua, Isabelle Mohr, Andreas Kouk-
ounas, Mohammad Kalim Wang, Nan Wang, and Han
Xiao. 2024. Jina-ColBERT-v2: A general-purpose
multilingual late interaction retriever. InProceedings
of the Fourth Workshop on Multilingual Representa-
tion Learning (MRL 2024), pages 159–166, Miami,Florida, USA. Association for Computational Lin-
guistics.
Junfeng Kang, Rui Li, Qi Liu, Yanjiang Chen, Zheng
Zhang, Junzhe Jiang, Heng Yu, and Yu Su. 2025.
PQR: Improving dense retrieval via potential query
modeling. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 13455–13469, Vi-
enna, Austria. Association for Computational Lin-
guistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR
’20, page 39–48, New York, NY , USA. Association
for Computing Machinery.
Natalia Korchagina. 2016. Building a gold standard
for temporal entity extraction from medieval german
texts. In"2016 Conference on Language Technolo-
gies and Digital Humanities".
Yibin Lei, Liang Ding, Yu Cao, Changtong Zan, An-
drew Yates, and Dacheng Tao. 2023. Unsupervised
dense retrieval with relevance-aware contrastive pre-
training. InFindings of the Association for Computa-
tional Linguistics: ACL 2023, pages 10932–10940,
Toronto, Canada. Association for Computational Lin-
guistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Xiaoyan Li and W. Bruce Croft. 2003. Time-based lan-
guage models. InProceedings of the Twelfth Inter-
national Conference on Information and Knowledge
Management, CIKM ’03, page 469–475, New York,
NY , USA. Association for Computing Machinery.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning.Preprint, arXiv:2308.03281.
Yang Liu, Lan Lan, Jiahuan Cao, Hiuyi Cheng, Kai
Ding, and Lianwen Jin. 2025. Large-scale corpus
construction and retrieval-augmented generation for

Ancient Chinese poetry: New method and data in-
sights. InFindings of the Association for Compu-
tational Linguistics: NAACL 2025, pages 779–817,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Jacob Menick, Maja Trebacz, Vladimir Mikulik,
John Aslanides, Francis Song, Martin Chadwick,
Mia Glaese, Susannah Young, Lucy Campbell-
Gillingham, Geoffrey Irving, and 1 others. 2022.
Teaching language models to support answers with
verified quotes.arXiv preprint arXiv:2203.11147.
Jianmo Ni, Gustavo Hernandez Abrego, Noah Con-
stant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang.
2022a. Sentence-t5: Scalable sentence encoders
from pre-trained text-to-text models. InFindings of
the Association for Computational Linguistics: ACL
2022, pages 1864–1874, Dublin, Ireland. Association
for Computational Linguistics.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Her-
nandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith
Hall, Ming-Wei Chang, and Yinfei Yang. 2022b.
Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages
9844–9855, Abu Dhabi, United Arab Emirates. As-
sociation for Computational Linguistics.
Bhawna Piryani, Abdelrahman Abdallah, Jamshid
Mozafari, Avishek Anand, and Adam Jatowt. 2025.
It’s high time: A survey of temporal question answer-
ing.Preprint, arXiv:2505.20243.
Thilina C. Rajapakse. 2023. Dense passage retrieval:
Architectures and augmentation methods. InPro-
ceedings of the 46th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval, SIGIR ’23, page 3494, New York, NY ,
USA. Association for Computing Machinery.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.
Xinjie Shen, Zhichao Geng, and Yang Yang. 2025. Ex-
ploring 0 sparsification for inference-free sparse re-
trievers. InProceedings of the 48th International
ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval, SIGIR ’25, page
2572–2576, New York, NY , USA. Association for
Computing Machinery.
Xin Su, Phillip Howard, and Steven Bethard. 2025.
Transformer-based temporal information extraction
and application: A review. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 28822–28841, Suzhou,
China. Association for Computational Linguistics.
Liquid AI Team. 2025. Lfm2 technical report.Preprint,
arXiv:2511.23404.Jiexin Wang, Adam Jatowt, Masatoshi Yoshikawa, and
Yi Cai. 2023. Bitimebert: Extending pre-trained lan-
guage representations with bi-temporal information.
InProceedings of the 46th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’23, page 812–821, New
York, NY , USA. Association for Computing Machin-
ery.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Improv-
ing text embeddings with large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916, Bangkok, Thai-
land. Association for Computational Linguistics.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024b. Mul-
tilingual e5 text embeddings: A technical report.
Preprint, arXiv:2402.05672.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul N. Bennett, Junaid Ahmed, and
Arnold Overwijk. 2021. Approximate nearest neigh-
bor negative contrastive learning for dense text re-
trieval. InInternational Conference on Learning
Representations.
Hugo Yèche, Alizée Pace, Gunnar Ratsch, and Rita
Kuznetsova. 2023. Temporal label smoothing for
early event prediction. InProceedings of the 40th
International Conference on Machine Learning, vol-
ume 202 ofProceedings of Machine Learning Re-
search, pages 39913–39938. PMLR.
Siyue Zhang, Yuxiang Xue, Yiming Zhang, Xiaobao
Wu, Anh Tuan Luu, and Chen Zhao. 2025a. Mrag:
A modular retrieval framework for time-sensitive
question answering. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
3080–3118.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025b. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Yuqing Zhang, Baoyi He, Yihan Chen, Hangqi Li,
Han Yue, Shengyu Zhang, Huaiyong Dou, Junchi
Yan, Zemin Liu, Yongquan Zhang, and Fei Wu.
2024. PhiloGPT: A philology-oriented large lan-
guage model for Ancient Chinese manuscripts with
dunhuang as case study. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 2784–2801, Miami, Florida,
USA. Association for Computational Linguistics.

Text (excerpt + short gloss) Source Label
夏，五月，郑伯克段于鄢。
Annals anchor: Zheng subdued Duan at Yan in month 5.《春秋》
Chunqiu
经文/annalsevent
夏，五月，郑伯克段于鄢。克之者何？杀之也。杀之则曷为谓之克？大郑伯之恶
也。段者何？郑伯之弟也。何以不称弟？当国也。
Gongyang reads ke as killing and amplifies moral blame.《春秋公羊传》
Gongyang zhuan
传文/zhuanevent
夏，五月，郑伯克段于鄢。克者何？能也。何能也？能杀也。何以不言杀？见段
之有徒众也。段，郑伯弟也. . . . . .段失子弟之道矣，贱段而甚郑伯也。
Guliang stresses armed revolt, conduct, and intensified censure.《春秋谷梁传》
Guliang zhuan
传文/zhuanevent
书曰： “郑伯克段于鄢。 ”段不弟，故不言弟；如二君，故曰克；称郑伯，讥失教
也，谓之郑志。不言出奔，难之也。
Zuo treats wording itself as a signal of layered blame.《春秋左传》
Zuo zhuan
传文/zhuanevent
案左氏云段出奔共，而公、谷皆曰杀。据隐十一年传，庄公曰： “寡人有弟不能
和协，使糊其口于四方”，则未杀明矣，公、谷之说非是。
Gu Donggao rejects the reading that Duan was killed.《春秋大事表》
Chronological
顾栋高/Qingneg
不称国讨而言郑伯，讥失教也。段不弟，故不言弟，明郑伯虽失教而段亦凶逆。
Du Yu explains how wording encodes both political and personal blame.《春秋左传注》
Zuo zhuan zhu
杜预/Jinneg
以“国讨 ”“得隽曰克 ”等例，说明称郑伯乃罪君，不称弟乃罪段，兼示兄虽失教而
弟为乱首。
Kong Yingda systematizes the case through doctrinal categories.《春秋左传疏》
Zuo zhuan shu
孔颖达/Tangneg
Table 5: Aligned materials under the reign-based time key “ 鲁隐公元年五月 ” for “郑伯克段于鄢 ”. The annals give
the anchor, the threezhuanprovide aligned expansions, and later sources yield chrono-near non-target paraphrases.
A Details of Dataset
A.1 Annals and Exegetical Layers
TheChunqiucorpus combines two tightly cou-
pled layers. The annals themselves are extremely
terse month-level records, which provide the pri-
mary temporal anchors on the Lu-state regnal time-
line. By contrast, the three classicalzhuan(Zuo,
Gongyang, andGuliang) expand the same entries
into narrative, interpretive, or doctrinal prose. In
ChunQiuTR, we treat the annals line as the an-
chor and the alignedzhuanpassages as seman-
tically richer event descriptions under the same
reign-based time key.
Table 5 illustrates this structure with the canoni-
cal case “郑伯克段于鄢 ” (Duke Yin, Year 1, Month
5). A single annals line is expanded by the three
zhuanin different ways, while later commentarialand historiographical sources further paraphrase or
reinterpret the same event. This layered organiza-
tion is central to our benchmark design: it provides
both aligned event records and naturally occurring
chrono-near but non-target passages.
A.2 From Parallel Texts to Reign-Based Time
Keys
A.2.1 Reign-based time keys
Our normalized time axis uses month-level keys of
the form
τ= (gong,year,month),
for example 「鲁隐公元年正月」 or「鲁桓
公元年三月」 . We scan the annals sequentially
while maintaining the current triple (g, y, m) . Full
reign cues initialize a new key, bare year mark-

System: You are a scholar of Chunqiu.
Prompt:  
Given the following passages under 
the same time label, group them by event. 
Passages describing the same event  should 
be placed into one group. 
Separate different groups with three 
blank lines. Preserve the original formatting 
and do not  add any extra content:  {text}
[春秋] 冬，十有二月 ，祭伯來。公子益師卒 。
[春秋公羊傳 ] 冬，十有二月 ，祭伯來。祭伯者
何？…  公子益師卒 。何以不日 ？遠也。…
[春秋穀梁傳 ] 冬，十有二月 ，祭伯來。來者… 
公子益師卒 …  
[春秋左氏傳 ] 十二月，祭伯來，非王命也 。眾父
卒。公不與小斂 ，故不書日 。
Output: 
[春秋] 冬，十有二月 ，祭伯來。
[春秋公羊傳 ] 冬，十有二月 ，祭伯來。祭伯者… 
[春秋穀梁傳 ] 冬，十有二月 ，祭伯來。來者… 
[春秋左氏傳 ] 十二月，祭伯來，非王命也 。
[春秋] 公子益師卒 。
[春秋公羊傳 ] 公子益師卒 。何以不日 ？遠也… 
[春秋穀梁傳 ] 公子益師卒 。大夫日卒 … 
[春秋左氏傳 ] 眾父卒。公不與小斂 ，故不書日 。
)1(
D
)2(
DFigure 5: Example of event-level grouping under the
reign-based time key 「鲁隐公元年十二月」 . The
model is prompted to split mixed passages and group
aligned commentary snippets by event.
ers start a new regnal year under the current ruler,
month markers update only m, and sentences with-
out new temporal cues inherit the current triple.
Table 6 shows representative cases.
A.2.2 Record–time-key alignment
We align event-level records under each normal-
ized time key using lightweight LLM suggestions
followed by manual verification. This is necessary
because a single annals line may compress multiple
events, while commentary passages may expand
one event across several fragments. Figure 5 shows
a representative case under the time key 「鲁隐公
元年十二月」.
In this case, the mixed annals line is split into
two records, one for 「祭伯来」 and one for
「公子益师卒」 , each grouped with its aligned
zhuanpassages. All such suggestions are manually
checked before entering the benchmark.
A.3 Details of Chrono-near counterfactual
negatives
A.3.1 Classical sources and perspectives
Our chrono-near counterfactual negatives draw on
several classical works that reorganize or reinter-pretChunqiuevents from distinct perspectives:
Gu Donggao’sChronological Tables( 顾栋高
《春秋大事表》 ).Gu’s Qing-dynasty compi-
lation systematically re-orders theChunqiuand the
threezhuaninto explicit chronological tables. Each
entry typically specifies the state, the reigning gong,
the year, and a short prose summary of the event,
sometimes highlighting cross-state interactions or
disagreements among the Zuo, Gongyang, and Gu-
liang traditions. Compared to the terse annals, these
tables provide a more “modernized” timeline and
condensed paraphrases of events, which we reuse
as temporally grounded, paraphrastic negatives.
Wei Liaoweng’sChunqiu Zuozhuan Yaoyi( 魏
了翁《春秋左传要义》 ).Wei’s Southern Song
work focuses on extracting the “essential meanings”
ofZuo zhuanepisodes. His prose often paraphrases
the underlying narrative, emphasizes moral and rit-
ual judgments, and occasionally re-groups several
Zuopassages into a single didactic unit. From our
perspective, these are high-level, discursive restate-
ments of the same historical events, written in a
style that differs noticeably from the base corpus.
Zuoshu annotations and sub-commentaries (注
疏).In addition, we employ theSiku Quanshu
edition ofZuozhuanannotations, which combines
multiple layers: Lu Deming’syin yi( 音义 ), Du
Yu’s Jin-dynasty commentary, Kong Yingda’s Tang-
dynastyChunqiu Zhengyi, and later Song-dynasty
notes such as Lü Zuqian’sZuozhuan shu ¯o. These
texts embed glosses, philological notes, and exeget-
ical reformulations around the same events. While
they do not always restate the full narrative, they
frequently echo key phrases, name important ac-
tors, or re-frame the event in ritual or moral terms.
Taken together, these sources provide us with
multiple “views” on the same historical episodes:
terse annal entries, narrative expansions in the
threezhuan, tabular re-organizations (Gu Dong-
gao), moral-didactic summaries (Wei Liaoweng),
and layered annotations ( 注疏 ). By aligning them
into the reign-based time-key space defined in the
main text, we obtain chrono-near passages that
are temporally co-located with our ground truth
records Dτbut often differ in wording, emphasis,
or even event granularity.

Original chronicle snippet Cue / update Normalized time key
元年，春，王正月。 Initialize a new reign-year ( 隐公元年 ),
month =正月鲁隐公元年正月
三月，公及邾仪父盟于蔑。 New month ( 三月 ), inherit current
reign-year鲁隐公元年三月
二年，春，公会戎于潜。 New year ( 二年 ) under the same ruler,
reset month to正月鲁隐公二年正月
元年，春，王正月，公即位。 New ruler detected ( 桓公 ), reset year
to元年and month to正月鲁桓公元年正月
Table 6: Representative mappings from raw chronicle phrases to normalized reign-based time keys.
A.3.2 LLM-assisted reverse matching and
fuzzy alignment
For sources like Lü Zuqian’sChunqiu ZuoZhuan
Shuos(《春秋左氏传说》 ) and certain annotation
layers, the text is often organized as short titled
sections (e.g., event summaries or topic headings)
rather than direct quotations of the base corpus.
To map these paraphrastic units back to concrete
passages in the annals andZuo zhuan, we adopt
an LLM-assisted “reverse matching” strategy, illus-
trated in Fig. 6.
Given a candidate item with a ruler name, an ap-
proximate year range, and a short event title (e.g.,
from Lü’sChunqiu ZuoZhuan Shuos), we query
a classical-Chinese LLM (DeepSeek) as a virtual
Zuo zhuanexpert. When the model can make
a judgment, it must return only the originalZuo
zhuantext segment; when it is uncertain, it must
output the sentinel token NONE . In all cases, these
LLM suggestions are further filtered and manually
checked before being accepted into our aligned
record set.
Once a plausibleZuo zhuanspan has been sug-
gested and validated, it can be matched back to
the digitized base text with simple fuzzy string
matching, which uniquely anchors the passage to
its canonical location and the corresponding reign-
based time keyτ.
A.4 Query Types and Templates
We instantiate a small set of Traditional-Chinese
natural-language templates over normalized reign-
based month keys (e.g., “ 公元年二月 ”). Queries
are divided intopointqueries, which target a single
month, andwindowqueries, which target a span
around a reference month. Point queries include
both content-oriented and existence-oriented for-
mulations, while window queries cover past, future,
around, and explicit-range retrieval. Figure 7 illus-
trates representative temporal interpretations, and
Table 7 summarizes the template groups used inTemplate groups (index range) Type
BASE (1–12) point, content-oriented
BASE (13–20) point, existence / no-event
MONTH_PAST (21–26) window, past
MONTH_FUTURE (27–31) window, future
MONTH_AROUND (32–36) window, around
MONTH_RANGE (37–41) window, range
YEAR_CURRENT (42–46) window, current year
YEAR_PAST (47–49) window, previous year
YEAR_FUTURE (50–52) window, next year
Table 7: Query template groups used in ChunQiuTR.
experiments. Empty months are handled through
the same point/window formulations via no_event
placeholders when the target month or span con-
tains no recorded event.
A.5 Statistics of Dataset
A.5.1 Raw unit-level length statistics (before
sentence splitting)
Before sentence-level segmentation, we collect raw
records from theChunqiuannals and the three tra-
ditionalzhuan(positive pool), as well as later ex-
egetical layers such asZuoshi zhuanshuoandChun-
qiu Zhengyi(negative pool). Table 8 reports basic
character-length statistics over these raw records.
The raw source pools differ substantially in
length and discourse style, especially between
canonical records and later exegetical materials,
which motivates sentence-level segmentation be-
fore retrieval construction.
Fig. 8 summarizes month-level coverage and gap
months over the normalizedChunqiutimeline. The
benchmark spans 3036 reign-based months from
Duke Yin to Duke Ai, with a substantial proportion
of months containing no recorded event, confirm-
ing that gap months are a pervasive property of the
corpus rather than an edge case.

Pool Source # Raw units Total chars (K) Avg. len. Median Min Max
Positive All (annals + threezhuan) 6641 293.7 44.22 – – –
PositiveChunqiuannals 1532 19.2 12.52 10 2 71
PositiveGongyangzhuan 1776 44.5 25.05 10 2 671
PositiveZuozhuan 1547 189.1 122.20 48 3 2658
PositiveGuliangzhuan 1786 41.0 22.93 11 2 467
Negative All exegetical layers 9227 1014.9 109.99 – – –
Negative Lü Zuqian 241 99.0 410.70 365 14 1715
Negative Kong Yingda 3653 483.5 132.36 83 0 2243
Negative Du Yu 3652 218.1 59.72 49 1 714
Negative Gu Donggao 481 39.3 81.80 62 1 597
Negative Wei Liaoweng 1200 174.9 145.79 106 10 2877
Table 8: Raw record-level character-length statistics before sentence splitting. Character counts are reported in
thousands (K). Positive records come from theChunqiuannals and the three traditionalzhuan, while negative
records come from later exegetical layers.
A.5.2 Record-level segmentation.
To construct retrieval units, we convert hetero-
geneous raw source materials into sentence-level
records. We first use an LLM to propose punctua-
tion and sentence boundaries ( 句读 ) for each raw
passage, and then apply a light rule-based splitter
over classical discourse markers such as “ 曰”, “云”,
and “传曰 ”. During this step, we also perform min-
imal normalization to reduce stylistic boilerplate,
for example by stripping framing markers such as
“正义曰 ” or formulaic quotation headers that do
not contribute substantive event content, and by
discarding extremely short fragments that contain
only a few characters.
Each resulting record inherits the reign-based
time key and source metadata of its parent unit, and
is assigned a coarse type label: event ,no_event ,
orneg_comment . For months where the annals and
the threezhuanjointly indicate that nothing was
recorded, we additionally synthesize a standardized
no_event record for that time key (e.g., “ 鲁隐公
元年二月：《春秋》经文及三传于此月无事
可书。 ”), so that retrieving an empty-month case
still requires matching the correct reign and month,
rather than collapsing all such queries to a single
global “nothing happened” entry.
This process yields 20,172 records in total,
which constitute the retrieval gallery used in our
time-aware experiments (Table 9).
A.5.3 Final benchmark splits.
The benchmark is split at the month level using an
approximate 80/10/10 partition, and all records and
queries inherit the split of their associated time key
to avoid temporal leakage. As shown in Table 9, thefinal benchmark contains 3036 month keys, 20,172
record-level retrieval units, and 16,226 queries,
with explicit breakdowns by split and record type.
A.5.4 Data sources and licensing.
Source & license.All digitized texts used in
ChunQiuTR are retrieved from Chinese Wikisource
(Siku Quanshueditions). Individual work pages are
tagged as public domain (e.g., PD-old ), while plat-
form content is provided under CC BY-SA 4.0 and
the Wikimedia Terms of Use. To facilitate compli-
ant reuse, we record page revision IDs ( oldid ) and
release the benchmark as derived metadata together
with scripts for re-downloading the raw texts.
B Details of Methods
B.1 Details of Experiment Setting
B.1.1 Model and Training Details
We fine-tune two dual-encoder backbones: BERT-
BASE-CHINESEand QWEN3-EMBED-0.6B. For
BERT-BASE, we use [CLS] pooling; for QWEN3-
EMBED-0.6B, we use last-token pooling.
Both backbones are trained with a contrastive
retrieval objective using multi-positive supervision
and explicit hard-negative training. We addition-
ally apply an auxiliary time classification loss over
the three discrete factors (gong/year/month), with
weight 0.1and label smoothing ϵ= 0.2 . For CTD,
we enable both the relative temporal bias and the
soft absolute temporal context derived from pre-
dicted time distributions.
We optimize with AdamW (weight decay 0.01)
and a linear learning-rate schedule with warmup
ratio0.1. For BERT-BASE, we train for 5 epochs

Split # months # records # queries Avg. ground-truth recs/query # event recs # no-event recs # neg. comments
Train 2424 16027 13053 7.3 5360 1209 9458
Validation 295 2049 1520 6.8 626 152 1271
Test 317 2096 1653 7.2 782 149 1165
Total 3036 20172 16226 7.2 6768 1510 11894
Table 9: Final benchmark statistics and splits over month-level time keys, record-level retrieval units, and queries.
The “Avg. ground-truth recs/query” column reports the average number of labeled relevant records per query in
each split, and the last three columns break down records by type (event,no_event, andneg_comment).
Work / layer Role in ChunQiuTR Digital source (edition) License note / release plan
Chunqiu(春秋) base corpus Chinese Wikisource (Siku
Quanshuedition; page
revisionoldidrecorded)Work pages:taggedPD-old.Platform
text:CC BY-SA 4.0 (Wikimedia Terms of
Use). We release derived metadata (time
keys, alignments, queries/qrels, indices)
and scripts to re-fetch raw texts from
recordedoldidrevisions.Zuo zhuan(左氏) base corpus
Gongyang zhuan(公羊) base corpus
Guliang zhuan(梁) base corpus
Gu Donggao,Chunqiu
Dashibiao(栋高《春秋大事
表》)chrono-near
paraphrastic negatives
Wei Liaoweng,Chunqiu
Zuozhuan Yaoyi(魏了翁《春秋
左要》)chrono-near discursive
negatives
Zuozhuanannotations /
sub-commentaries (注疏; e.g.,
音/杜注/正/)lexical-near annotation
negatives
Table 10: Text sources and licensing. All digitized texts are retrieved from Chinese Wikisource (Siku Quanshu
editions), with page revision IDs ( oldid ) recorded for traceability. See Appendix A for alignment and preprocessing.
with batch size 64 and learning rate 2×10−5, using
maximum query/passage lengths of 64/196. For
QWEN3-EMBED-0.6B, we train for 3 epochs with
effective batch size 16 and learning rate3×10−6,
using maximum query/passage lengths of 128/256;
we also enable global in-batch negatives.
We select checkpoints by validationRecall@1
and reportRecall@KandMRR@10on the test
split under the same evaluation protocol. During
evaluation, both commentary negatives and explicit
no_event records are included in the candidate
gallery.
B.1.2 Training Cost and Computational
Resources
We report the computing infrastructure and approx-
imate GPU-hours for representative fine-tuning
runs.
Hardware.BERT-BASE-CHINESEis trained on
a single GPU (either 1 ×NVIDIA RTX A6000
or 2×RTX 3090, depending on availability).
QWEN3-EMBED-0.6B uses multi-GPU distributed
training (either 2 ×RTX A6000 or 4 ×RTX 3090)
to support global in-batch negatives.
Training cost.Table 11 reports representative
wall-clock time per run and the corresponding
GPU-hours. These numbers cover end-to-end fine-Backbone Variant #GPU Time / run GPU-hours
BERT-BASE-CHINESEFT baseline 1≈15 min≈0.25
BERT-BASE-CHINESECTD (full) 1≈19 min≈0.32
QWEN3-EMBED-0.6B FT baseline 2≈45 min≈1.50
QWEN3-EMBED-0.6B CTD (full) 2≈45 min≈1.50
Table 11: Compute cost for representative fine-tuning
runs on ChunQiuTR. Time is wall-clock per run; GPU-
hours are computed as (#GPU)×(time in hours).
tuning with periodic validation; sparse baselines
and non-parametric time priors incur negligible
training cost.
Hyperparameters.We do not perform large-
scale hyperparameter sweeps. Instead, we adopt
standard fine-tuning settings for each backbone and
select the best checkpoint by validationRecall@1.
B.2 Details of Analysis
B.2.1 Details of Ablation Study
We further replicate the ablation with
bert-base-chinese to examine backbone
sensitivity. As shown in Table 12, adding multi-
positive retrieval supervision already improves
over the FT baseline, while temporal modeling
yields substantially larger gains. Among the two
temporal modules, the soft absolute temporal
context cxcontributes a stronger overall boost
than the relative-time bias btime
ij, and combining

System: You are a scholar familiar with the Zuozhuan.
Prompt:  
Ruling lord (state): {gong}
Reign year(s): {time_range}
Event subtitle in 《Chunqiu Zuozhuan Shuos 》: {event}
Based on your knowledge of the Spring and Autumn Annals ( 《春
秋》) and the Zuozhuan ( 《左傳》), infer which passage in the Zuozhuan 
this event subtitle is most likely to correspond to.
        Instructions:
        - As long as it describes the same historical event and is broadly 
similar in content, the wording does not need to be exactly identical.
        - It can be one sentence or several sentences from either the Classic 
(jing) or the Commentary (zhuan), and you may extract only the key part.
        - Please try to give the single passage from the Zuozhuan that you 
believe is most likely. Only when you have absolutely no clue and cannot 
judge at all should you output "NONE".
        
  Format requirements (very important):
        - If you can make a judgment, **output only the original Zuozhuan 
text itself**, without any explanation, analysis, or commentary.
        - Do not output book titles, years, or chapter names, and do not 
add any prefixes such as "The original text is:".
        - Do not add quotation marks, book -title marks, or brackets; only 
output the plain original text.
        - If you cannot judge, output only: ` NONE ` (all uppercase, with no 
other content and no punctuation).
Please strictly follow the above format in your answer.
Input:
{gong} =隱公
{time_range} =元年
{event} =祭仲諫鄭莊封叔叚
Output: 
初，鄭武公娶于申 ，曰武姜，生莊公及共叔段 。莊公寤生 ，驚姜氏，故名
曰「寤生」，遂惡之。愛共叔段 ，欲立之。亟請于武公 ，公弗許。及莊公即位 ，
為之請制 。公曰：「制，巖邑也，虢叔死焉 ，佗邑唯命 。」請京，使居之，謂之
京城大叔 。祭仲曰：「都城過百雉 ，國之害也 。先王之制 ，大都不過參國之一 ，
中五之一 ，小九之一 。今京不度 ，非制也，君將不堪 。」公曰：「姜氏欲之 ，焉
辟害？」對曰：「姜氏何厭之有 ？不如早為之所 ，無使滋蔓 ！蔓，難圖也。蔓草
猶不可除 ，況君之寵弟乎 ？」公曰：「多行不義 ，必自斃，子姑待之 。」
Figure 6: LLM-assisted reverse matching from para-
phrastic event titles in Lü Zuqian’sChunqiu ZuoZhuan
Shuos(《春秋左氏传说》 ) toZuo zhuanpassages.
We show the full text-only prompt given to a classical-
Chinese LLM (DeepSeek), together with one concrete
example: for rulerYinof Lu, year 1, and the subtitle “ 祭
仲谏郑庄封叔段 ” fromChunqiu ZuoZhuan Shuos, the
model proposes the most likelyZuo zhuanpassage. The
model is required to output only the originalZuotext,
or the sentinel tokenNONEwhen it cannot decide.
both gives the best performance. The larger gains
on BERT than on stronger embedding backbones
suggest that temporal supervision and structured
negatives are especially helpful when the base
retriever has more room to reduce chrono-near
confusions.
B.2.2 No-event and Hard-negative Behavior
Table 13 probes two protocol switches: whether to
keep pure no_event queries ( dq) and whether to
inject neg_comment passages as chrono-near hard
negatives ( neg). Including pure no-event queries
consistently raises scores across all methods, indi-
cating that empty-month retrieval is substantially
easier and should be controlled by protocol. ByVariantL multibtime
ijcx R@1 MRR@10
FT baseline – – – 0.5088—-0.5597—-
+L multi ✓– – 0.5178↑+0.00900.5685↑+0.0088
+ Bias✓ ✓– 0.5384↑+0.02960.5776↑+0.0179
+ Ctx✓–✓0.5620↑+0.05320.5961↑+0.0364
Full (Ours)✓ ✓ ✓0.5826↑+0.07380.6193↑+0.0596
Table 12: Ablation on the test set with
bert-base-chinese.
contrast, injecting neg_comment passages exposes
genuine robustness differences: the BERT FT base-
line drops noticeably, whereas CTD BERT-base re-
mains stable, and both Qwen-based systems change
only marginally. Overall, the benchmark contains
both an easier no-event regime and a harder exeget-
ical hard-negative regime, with CTD improving
robustness especially for weaker backbones.
B.2.3 Full Protocol Grid Results
We report results under all valid combina-
tions of the three evaluation switches: neg
(whether neg_comment passages are included in
the gallery), ne(whether explicit no_event records
are included), and dq(whether pure no_event
queries are dropped). Modes are denoted as
neg{0/1}_ne{0/1}_dq{0/1} . When ne=0 , pure
no_event queries are ill-defined for retrieval, so
onlydq=1is reported.
Tables 14 and 15 summarize validation and test
results for all queries, and separately for the point
and window families. Two trends are consistent
across settings: keeping pure no_event queries
(dq=0 ) raises aggregate scores, whereas injecting
neg_comment passages ( neg=1 ) yields a harder and
more realistic gallery. The full grid is therefore
intended mainly as a protocol reference and robust-
ness diagnostic.
Full-grid results.Tables 14 and 15 report Re-
call@K, MRR@10, and nDCG@10 on validation
and test under each mode. We report results on
allqueries, and also stratify by point andwindow
families (corresponding to our point-/gap-like vs.
window-style temporal probes in the main paper).
What this grid clarifies.Two takeaways are par-
ticularly relevant for interpreting aggregate scores.
First, dq=0 (keeping pure no-event queries) can
noticeably inflate overall metrics compared to
dq=1 , motivating our practice of reporting both
settings: dq=0 reflects the benchmark’s intended
scope (event monthsandexplicit empty months),

隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7{隐公元年五月 } 发
生了什么事 ？Reasoning -Type Time -Evolution Fact [Time Specifier]
Point, content -oriented
Point, No-event month{隐公元年六月 } 发
生了什么事 ？隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7
隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7Past/Future  window  
(1 month)
Past/Future  window
(multiple months){隐公元年六月 } 的 {前一
个月 } 发生了什么事 ？
{隐公元年六月 } 的 {前两
个月内 } 发生了什么事 ？隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7
{隐公元年五月 } 的 {附近一个
月内 } 发生了什么事 ？隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7Around  window
Explicit month range
隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7隐公 /元年 /五月 隐公 /元年 /六月 隐公 /元年 /四月 隐公 /元年 /三月 隐公 /元年 /七月
Duke Yin, Year 1, Month 3 Duke Yin, Year 1, Month 4 Duke Yin, Year 1, Month 5 Duke Yin, Year 1, Month 6 Duke Yin, Year 1, Month 7{隐公元年四月 } 到 {隐公元年
六月 } 发生了什么事 ？
Target Month Anchor  Month Target Window Retrieval Inference Target Month Anchor  Month Target Window Retrieval InferenceFigure 7: Representative temporal query types in ChunQiuTR, including point queries, past/future windows, around
windows, and explicit ranges over reign-based month keys.
MethodIncluding pure no-event queries(fixneg=0,ne=1)Hard-negative robustness(fixne=1, dq=1)
dq=1 dq=0∆neg=0 neg=1∆
R@1 MRR@10 R@1 MRR@10 R@1 MRR@10 R@1 MRR@10 R@1 MRR@10 R@1 MRR@10
BM25 0.240 0.296 0.418 0.470↑+0.178↑+0.1730.240 0.296 0.228 0.282↓-0.012↓-0.014
ColBERT-LFM2 (ZS) 0.220 0.277 0.333 0.386↑+0.113↑+0.1090.220 0.277 0.222 0.278↑+0.002↑+0.001
BERT-base (FT) 0.378 0.439 0.560 0.606↑+0.182↑+0.1670.378 0.439 0.306 0.374↓-0.072↓-0.065
CTD BERT-base (Ours)0.407 0.456 0.583 0.619↑+0.175↑+0.1630.407 0.456 0.407 0.4560.000 0.000
E5-mistral-7B (ZS) 0.062 0.093 0.225 0.270↑+0.163↑+0.1770.062 0.093 0.054 0.082↓-0.008↓-0.011
Qwen3-Embed-0.6B (FT) 0.396 0.434 0.577 0.605↑+0.181↑+0.1710.396 0.434 0.396 0.4330.000↓-0.001
CTD Qwen3-Embed-0.6B (Ours)0.420 0.457 0.594 0.621↑+0.174↑+0.1640.420 0.457 0.418 0.455↓-0.002↓-0.002
Table 13: No-event and hard-negative behavior under protocol variations on the test set. Left: dq=1 vs.dq=0 (fix
ne=1,neg=0), where dqdrops or keeps pure no_event queries. Right: neg=0 vs.neg=1 (fixne=1,dq=1), where
neginjectsneg_commentpassages into the gallery.∆denotes the within-method change.
while dq=1 isolates event-seeking behavior. Sec-
ond, injecting exegetical hard negatives ( neg=1 )
is a strictly harder and more realistic gallery set-
ting; models that remain stable between neg=0 and
neg=1 exhibit stronger robustness to chrono-near
confounds from commentarial material.
B.2.4 Top-1 Near-miss Failure Cases
Figure 9 shows two representative near-miss cases
on the test set: our retriever fails to place the cor-
rect passage at rank 1, but still retrieves at least one
ground-truth passage within the top–5, whereas
the baseline fails to surface any ground-truth evi-
dence. In both cases, a key confounder is the highly
reusable no_event -style wording and its chrono-
near reoccurrence across adjacent months, whichcan trigger top-rank swaps. These examples sug-
gest that the remaining errors are often ordering
errors under strong lexical or temporal confounders,
rather than complete retrieval failure.
B.2.5 Qualitative demo: reasoning traces vs.
evidence grounding
Figures 10 and 11 compare an online LLM on the
same month-level point query with and without
evidence grounding. Without retrieved evidence,
the model either predicts an empty month or pro-
duces an incomplete answer even when a reasoning
trace is enabled. When given an evidence pack that
contains the gold month records together with con-
fusable materials, the same model recovers both
gold entries and grounds the answer in cited evi-

span gap ratio050010001500200025003000monthsAll dukes (L=12)
span gap ratio0255075100125150months1. Yin (L=2)
span gap ratio050100150200months2. Huan (L=4)
span gap ratio0100200300400months3. Zhuang (L=4)
span gap ratio0510152025months4. Min (L=2)
span gap ratio0100200300400months5. Xi (L=3)
span gap ratio050100150200months6. Wen (L=4)
span gap ratio050100150200months7. Xuan (L=4)
span gap ratio050100150200months8. Cheng (L=4)
span gap ratio0100200300400months9. Xiang (L=3)
span gap ratio0100200300400months10. Zhao (L=4)
span gap ratio050100150200months11. Ding (L=4)
span gap ratio0100200300months12. Ai (L=12)020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
020406080100
gap ratio (%)
span months gap months gap ratio (%)Figure 8: Month-level coverage, gap counts, and gap ratios for the normalizedChunqiutimeline (overall and per Lu
ruler).
 魯成公六年二月 ，魯國有何記事 ？ (What happened in Duke Cheng’s 6th year, 2nd month? )
#1 魯成公七年二月 ：經傳於是月史事闕如 ，無專條可錄 。
#2 魯文公六年二月 。《春秋》經文及三傳於此月無事可書 。
#3 魯成公八年二月 ：經傳於是月史事闕如 ，無專條可錄 。 
#4 魯成公五年二月 ：經傳於是月史事闕如 ，無專條可錄 。 
#5 魯成公四年二月 ：經傳於是月史事闕如 ，無專條可錄 。 #1 魯成公七年二月 ：經傳於是月史事闕如 ，無專條可錄 。
#2 魯成公八年二月 ：經傳於是月史事闕如 ，無專條可錄 。 
#3 ★二月，辛巳，立武宮，取鄟，衛孫良夫帥師侵宋 。
#4 ★二月辛巳 ，立武宮。立者，不宜立也 。
#5 ★二月，辛巳，立武宮，取鄟，衛孫良夫帥師侵宋 。
請檢索魯成公七年正月之前半年內 ，《春秋》中所有相關記事 。 (What happened in half year before Duke Cheng’s 6th year, 1st month? )
#1 魯成公七年九月 ：經傳於是月史事闕如 ，無專條可錄 。
#2 魯成公七年十二月 。《春秋》及三傳皆不載此月之事 。
#3 魯成公七年六月 。《春秋》及三傳皆不載此月之事 。
#4 魯成公七年十一月 。《春秋》經文及三傳於此月無事可書 。
#5 魯成公七年三月 。《春秋》經文及三傳於此月無事可書 。#1 魯成公七年十二月 。《春秋》及三傳皆不載此月之事 。
#2 魯成公七年十一月 。《春秋》經文及三傳於此月無事可書 。
#3 魯成公七年九月 ：經傳於是月史事闕如 ，無專條可錄 。
#4 魯成公七年二月 ：經傳於是月史事闕如 ，無專條可錄 。
#5 ★魯成公六年十二月 。《春秋》及三傳皆不載此月之事 。QUERY (point ):
QUERY (window ):
Baseline
BaselineOurs
Ours
★ 二月辛巳 ，立武宮。武宮者何 ？武公之宮也 。立者何？… 
★ 二月辛巳 ，立武宮。立者，不宜立也 。 
★ 二月，季文子以鞍之功立武宮 ，非禮也，聽於人以救其難 … 
★ 二月，辛巳，立武宮，取鄟，衛孫良夫帥師侵宋 。
★ 取鄟。鄟者何？邾婁之邑也 。曷為不系于邾婁 ？諱亟也。
★ 秋，仲孫蔑、叔孫僑如率師侵宋 。 
★ 秋，仲孫蔑、叔孫僑如帥師侵宋 。 
★ 秋，孟獻子，叔孫宣伯 ，侵宋，晉命也，楚子重伐鄭 … 
★ 楚公子嬰齊率師伐鄭 。
★ 秋，仲孫蔑，叔孫僑如 ，帥師侵宋 ，楚公子嬰齊帥師伐鄭 。Ground Truth
Ground Truth
Figure 9: Top-1 near-miss cases on ChunQiuTR (test set). We show the top–5 results from a baseline (left) and ours
(middle), with the ground-truth set (right).⋆marks ground-truth passages;✓indicates a ground-truth hit in top–5.
dence. These examples suggest that longer reason-
ing traces alone do not ensure month-level com-
pleteness, whereas explicit evidence binding sub-
stantially improves temporal faithfulness.
B.3 Details of Compared Methods
B.3.1 Sparse retrieval.
We compare against a sparse family including a
classical lexical retriever, a simple temporal re-
ranking variant, and two inference-free neural
sparse retrievers, all evaluated under the same
sparse retrieval protocol.
•BM25(Robertson and Zaragoza, 2009): stan-
dard lexical term-matching baseline.
•BM25+TimeKDE: BM25 with a non-
parametric temporal re-ranking prior over
regnal-month indices, following classical TIR-
style temporal priors (Li and Croft, 2003).
•SPLADE-IDF (ZS) (Geng et al., 2025):
inference-free neural sparse retriever usedzero-shot.
•SPLADE- ℓ0(ZS)(Shen et al., 2025): sparsity-
controlled neural sparse retriever used zero-
shot.
B.3.2 Fusion / late-interaction retrieval.
We further compare against two multi-vector late-
interaction retrievers, both used in a zero-shot set-
ting.
•ColBERT-JINA (ZS) (Jha et al., 2024):
ColBERT-style token-interaction retriever.
•ColBERT-LFM2 (ZS) (Team, 2025): late-
interaction retriever with longer-context and
multi-scale representations.
B.3.3 Dense retrieval, encoder-based.
For encoder-based dense retrieval, we compare
against single-vector dual-encoder models used ei-
ther zero-shot or fine-tuned on ChunQiuTR.

Mode FamilyValidation Test
R@1 R@5 R@10 MRR@10 nDCG@10 R@1 R@5 R@10 MRR@10 nDCG@10
neg0_ne0_dq1all 0.0407 0.1200 0.2232 0.0798 0.0635 0.0654 0.1466 0.2042 0.1014 0.0677
neg0_ne0_dq1point 0.0375 0.1148 0.2459 0.0775 0.0756 0.0710 0.1696 0.2387 0.1156 0.0934
neg0_ne0_dq1window 0.0430 0.1239 0.2065 0.0815 0.0546 0.0610 0.1283 0.1768 0.0902 0.0472
neg0_ne1_dq0all 0.6329 0.6697 0.7191 0.6522 0.4921 0.5935 0.6497 0.6945 0.6206 0.4588
neg0_ne1_dq0point 0.5200 0.5440 0.6057 0.5347 0.5380 0.4922 0.5307 0.5869 0.5138 0.5103
neg0_ne1_dq0window 0.7860 0.8403 0.8729 0.8115 0.4300 0.7341 0.8150 0.8439 0.7689 0.3871
neg0_ne1_dq1all 0.4534 0.5050 0.5784 0.4806 0.2385 0.4197 0.4956 0.5602 0.4565 0.2223
neg0_ne1_dq1point 0.0164 0.0656 0.1920 0.0466 0.0532 0.0375 0.1105 0.2170 0.0784 0.0719
neg0_ne1_dq1window 0.7745 0.8279 0.8623 0.7996 0.3747 0.7230 0.8013 0.8326 0.7565 0.3417
neg1_ne0_dq1all 0.0317 0.1101 0.2173 0.0709 0.0600 0.0593 0.1344 0.1998 0.0951 0.0647
neg1_ne0_dq1point 0.0328 0.1077 0.2436 0.0725 0.0729 0.0690 0.1637 0.2367 0.1127 0.0912
neg1_ne0_dq1window 0.0310 0.1119 0.1979 0.0697 0.0505 0.0516 0.1111 0.1706 0.0812 0.0437
neg1_ne1_dq0all 0.6329 0.6691 0.7184 0.6520 0.4914 0.5923 0.6485 0.6927 0.6194 0.4575
neg1_ne1_dq0point 0.5200 0.5429 0.6046 0.5344 0.5373 0.4912 0.5307 0.5858 0.5129 0.5094
neg1_ne1_dq0window 0.7860 0.8403 0.8729 0.8115 0.4291 0.7327 0.8121 0.8410 0.7674 0.3854
neg1_ne1_dq1all 0.4534 0.5040 0.5774 0.4803 0.2374 0.4180 0.4939 0.5576 0.4548 0.2205
neg1_ne1_dq1point 0.0164 0.0632 0.1897 0.0459 0.0519 0.0355 0.1105 0.2150 0.0767 0.0701
neg1_ne1_dq1window 0.7745 0.8279 0.8623 0.7996 0.3737 0.7214 0.7981 0.8294 0.7549 0.3399
Table 14: Full protocol grid results forCTD-QWEN3-EMBED-0.6B. Mode names follow neg/ne/dqas defined in
Section B.2.3.
•GTR-T5-Base / Sentence-T5-Base (ZS)(Ni
et al., 2022b,a): T5-based dense retrievers
used zero-shot.
•mE5-Large / mE5-Large-ins (ZS) (Wang
et al., 2024b): multilingual E5 retrievers used
zero-shot.
•GTE-Large (ZS) (Li et al., 2023): general-
purpose dense embedding baseline.
•BGE-Large-v1.5 / BGE-M3 (ZS)(Xiao et al.,
2023; Chen et al., 2024): strong multilingual
dense embedding baselines.
•BERT-base (FT)(Devlin et al., 2019): Chinese
BERT dual-encoder fine-tuned on ChunQi-
uTR without explicit time modeling.
B.3.4 Dense retrieval, LM-based embeddings.
We also compare against LM-based dense embed-
ding models, including both zero-shot and task-
adapted variants.
•GTE-Qwen2-1.5B / E5-Mistral-
7B(ZS) (Wang et al., 2024a): LLM-scale
embedding baselines used zero-shot.
•PQR (Qwen2.5-7B / Qwen3-8B) (re)(Kang
et al., 2025): training-free retrieval framework
based on LLM-generated pseudo-queries.•Qwen3-Embed-0.6B / 4B (ZS)(Zhang et al.,
2025b): dedicated Qwen3 embedding models
used zero-shot.
•Qwen3-Embed-0.6B (FT) (Zhang et al.,
2025b): task-adapted dense dual-encoder
baseline without explicit time modeling.
B.3.5 Time-aware auxiliary variants.
Beyond BM25+TimeKDE, we report two
lightweight temporal extensions for single-vector
dense retrievers.
•TempDate: auxiliary time-key prediction
over(gong,year,month) during training, dis-
carded at inference time (Wang et al., 2023;
Dhingra et al., 2022).
•TempDate-Smooth: TempDate with
neighbor-aware smoothing over adjacent
ordered time keys (Yèche et al., 2023).
B.4 Cross-Corpus Pilot onZizhi Tongjian
To probe whether the temporal-consistency bias
learned on ChunQiuTR transfers beyond theSpring
and Autumn Annals, we conduct a lightweight
cross-corpus evaluation on two processed sub-
sets fromZizhi Tongjian(Qi JiandJin Ji). As
an annalistic general history,Zizhi Tongjianalso

Mode FamilyValidation Test
R@1 R@5 R@10 MRR@10 nDCG@10 R@1 R@5 R@10 MRR@10 nDCG@10
neg0_ne0_dq1all 0.0159 0.0456 0.0952 0.0331 0.0205 0.0096 0.0366 0.0672 0.0234 0.0192
neg0_ne0_dq1point 0.0258 0.0468 0.0937 0.0382 0.0265 0.0079 0.0394 0.0690 0.0226 0.0235
neg0_ne0_dq1window 0.0086 0.0448 0.0964 0.0294 0.0161 0.0110 0.0344 0.0657 0.0240 0.0158
neg0_ne1_dq0all 0.4474 0.5678 0.5980 0.4989 0.3794 0.4180 0.5408 0.5735 0.4696 0.3543
neg0_ne1_dq0point 0.4434 0.4709 0.4789 0.4556 0.4612 0.4152 0.4475 0.4506 0.4284 0.4339
neg0_ne1_dq0window 0.4527 0.6992 0.7597 0.5576 0.2683 0.4220 0.6705 0.7442 0.5269 0.2436
neg0_ne1_dq1all 0.2629 0.4097 0.4464 0.3261 0.1398 0.2400 0.3709 0.4127 0.2963 0.1218
neg0_ne1_dq1point 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
neg0_ne1_dq1window 0.4561 0.7108 0.7745 0.5658 0.2426 0.4304 0.6651 0.7402 0.5314 0.2184
neg1_ne0_dq1all 0.0000 0.0079 0.0169 0.0043 0.0043 0.0000 0.0052 0.0122 0.0023 0.0023
neg1_ne0_dq1point 0.0000 0.0117 0.0258 0.0073 0.0076 0.0000 0.0059 0.0178 0.0033 0.0037
neg1_ne0_dq1window 0.0000 0.0052 0.0103 0.0021 0.0019 0.0000 0.0047 0.0078 0.0015 0.0012
neg1_ne1_dq0all 0.4303 0.5480 0.5908 0.4809 0.3654 0.3962 0.5209 0.5620 0.4487 0.3404
neg1_ne1_dq0point 0.4286 0.4571 0.4754 0.4415 0.4494 0.3965 0.4350 0.4495 0.4125 0.4214
neg1_ne1_dq0window 0.4326 0.6713 0.7473 0.5343 0.2515 0.3960 0.6402 0.7182 0.4989 0.2280
neg1_ne1_dq1all 0.2530 0.3948 0.4395 0.3137 0.1314 0.2277 0.3560 0.3988 0.2823 0.1151
neg1_ne1_dq1point 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
neg1_ne1_dq1window 0.4389 0.6850 0.7625 0.5442 0.2279 0.4085 0.6385 0.7152 0.5063 0.2064
Table 15: Full protocol grid results for BM25, reported in the same format as Table 14.
records events under traditional reign-based, non-
Gregorian temporal expressions, making it a natu-
ral out-of-domain probe for month-keyed retrieval.
This pilot preserves the core month-key retrieval
idea of ChunQiuTR but is intentionally lighter than
the full benchmark. We retain event-bearing lines
as retrieval units, group them by normalized month
keys derived from the available reign/year/month
fields, and instantiate one point-style query for each
unique month key using a traditional reign-year
template. No target-corpus training is performed.
Unlike the full ChunQiuTR benchmark, this pilot
does not reconstruct explicit no_event placehold-
ers, commentary-derived hard negatives, or the full
point/gap/window query families, and should there-
fore be interpreted as a transfer probe rather than a
second benchmark.
Subset statistics.Table 16 summarizes the two
processed subsets used in this pilot.Qi Jicon-
tains 268 records and 92 month-level queries, while
Jin Jicontains 820 records and 119 queries. The
two slices cover distinct reign periods and remain
clearly separate from the ChunQiuTR source cor-
pus.
Transfer results.Table 17 reports retrieval per-
formance on the two subsets. We compare a zero-
shot Qwen3-Embed-0.6B encoder, a ChunQiuTR
fine-tuned dense baseline, and our CTD-enhancedretriever. Across both subsets, CTD consistently
improves MRR and R@1 over the fine-tuned base-
line without any target-corpus retraining; onQi Ji
it also improves R@5 and R@10, while onJin Ji
it matches the fine-tuned baseline on higher-recall
metrics.
Discussion.Although this transfer setting is
lighter than the full ChunQiuTR benchmark, the
overall trend is consistent with our main findings:
the temporal-consistency bias learned on ChunQi-
uTR transfers beyond in-domain fitting and con-
tinues to help distinguish chrono-near but tempo-
rally mismatched evidence. At the same time,
the gains are smaller than those observed on the
source benchmark, which is expected given both
the domain shift and the simplified evaluation pro-
tocol. We therefore view this pilot as evidence of
promising cross-corpus transfer, rather than as a re-
placement for a fully reconstructedZizhi Tongjian-
specific benchmark.
B.5 Alignment Audits and Reliability
To improve the auditability of ChunQiuTR, we
summarize here the two LLM-assisted curation
stages and the corresponding human-verification
statistics. As clarified in the revised main text,
ChunQiuTR isnotan AI-generated dataset: the
retrieval gallery is derived from authentic historical

Subset Approx. coverage Representative reign titles Records Queries
Qi Ji (part) 479–489 CE建元,永明268 92
Jin Ji (part) 265–279 CE泰始,咸宁820 119
Table 16: Basic statistics of the processedZizhi Tongjiansubsets used in the cross-corpus pilot.
Model / SettingQi Ji (part) Jin Ji (part)
MRR R@1 R@5 R@10 MRR R@1 R@5 R@10
Qwen3-Embed-0.6B (ZS) 0.0692 0.0217 0.0870 0.1413 0.0691 0.0420 0.0756 0.1345
Qwen3-Embed-0.6B (FT baseline) 0.2081 0.1848 0.2174 0.2391 0.1598 0.1345 0.1849 0.1849
CTD (Ours)0.2304 0.2065 0.2391 0.2717 0.1751 0.15970.1849 0.1849
Table 17: Cross-corpus pilot results on two processedZizhi Tongjiansubsets. No target-corpus training is performed.
System: 你是一名严谨的 《春秋》史料问答助手 。请只根据问题本身
作答；如果无法确定 ，请明确说 “不确定 /需要证据 ”，不要编造 。
Prompt:  
【问题】
查詢《春秋》，魯隱公元年十二月這一月有什麼重要事件 ？
【输出要求 】
- 用 2~5 句给出答案 （中文）。
  - 如果你不确定 ，请直接说明不确定点是什么 。
Thinking :
首先，问题是：查询《春秋》，鲁隐公元年十二月这一月有什么重要事件 ？输出要求用 2~5句给出答案 ，中文；如果不确定 ，直接说明不确定点 。
我需要基于 《春秋》这部史书来回答 。《春秋》是孔子编修的鲁国编年史 ，记载了从鲁隐公元年 （公元前722年）开始的事件 。问题指定了 “鲁隐公元年十二月 ”，
所以我必须查证 《春秋》中关于这一月的记载 。
回想一下 《春秋》的记载特点 ：它非常简略 ，通常只记录重要事件 ，如战争、祭祀、自然灾害等 。鲁隐公元年是 《春秋》的第一年 。
我记得《春秋》开篇是：“元年春，王正月。三月，公及邾仪父盟于蔑 。夏五月，郑伯克段于鄢 。秋七月，天王使宰咺来归惠公 、仲子之赗 。九月，及宋人盟于
宿。冬十有二月 ，祭伯来。公子益师卒 。”
所以，在鲁隐公元年十二月的记载是 ：“冬十有二月 ，祭伯来。” 意思是：冬季十二月 ，祭伯来访 。
重要事件 ：祭伯来。祭伯是周王朝的大夫 ，他的来访可能具有政治意义 ，表示周王室与鲁国的关系 。
但让我确认一下 ：《春秋》中鲁隐公元年的记载确实包括十二月的事件 。在《左传》中可能有更详细的解释 ，但问题只要求根据 《春秋》本身，所以我应该只基于
《春秋》的原文。
《春秋》原文在隐公元年十二月只记录了 “祭伯来”，没有其他事件 。所以，重要事件就是祭伯来访 。
输出要求用 2~5句给出答案 。我需要用中文回答 。
答案结构 ：
陈述《春秋》中鲁隐公元年十二月的记载 。
解释这个事件是什么 。
如果有必要 ，简要说明其意义 。
不确定点 ：我确认《春秋》记载是“冬十有二月 ，祭伯来。” 所以，没有不确定的 。但如果我记错了 ，就需要说明不确定 。但我基于记忆 ，应该正确 。
为了更严谨 ，我可以引用 《春秋》原文。但作为助手 ，我直接给出答案即可 。
最终答案 ：
在《春秋》中，鲁隐公元年十二月记载的重要事件是 “祭伯来”。这意味着周王室的大夫祭伯在此时来访鲁国 。这一事件反映了周与鲁之间的外交互动 。
这大约三句话 ，符合要求 。
Output (Thinking) : 
根据《春秋》记载，鲁隐公元年十二月的重要事件是 “祭伯来”，即周王
室的大夫祭伯来访鲁国 。这一记录见于 《春秋》隐公元年冬季的经文 ，反映了当
时周王室与诸侯国之间的外交往来 。此外，该月无其他事件记载 。Miss Key Records
Confident hallucinationOutput (w/o Thinking) : 
根据《春秋》经文，鲁隐公元年 （公元前722年）十二月并 无事件记载 。  
《春秋》该年记载始于 “元年春王正月 ”，终于“冬十有二月 ”。  
经文仅记 “冬十有二月 ”而未录具体事件 ，故该月无明确重要事件记录 。False  “no-event ” claim
Ground Truth : (1) 祭伯来；(2) 公子益师卒 。Ground Truth  coverage: 0/2
Ground Truth  coverage: 1/2  (miss: 公子益师卒 )
Figure 10: Online LLM without evidence grounding
on a month-level point query from ChunQiuTR. For
the query “ 鲁隐公元年十二月 ”, the gold month con-
tains two entries ( 祭伯来 and公子益师卒 ). Without
evidence, the model either predicts an empty month or
returns an incomplete answer.
sources, the queries are instantiated from a small
set of manually written templates, and LLMs are
used only to propose candidate splits or candidate
alignments during curation. They are never used to
generate, rewrite, translate, or paraphrase historical
content, and only human-approved results enter the
final benchmark.
A. Time-key normalization and manual verifi-
cation.The corpus follows an implicit Lu-state
reign calendar, normalized as month-level keys
τ= (gong,year,month).However, many passages do not explicitly contain
a complete (gong,year,month) triple. Instead, the
ruling duke and/or regnal year must often be recov-
ered from annalistic structure, discourse continu-
ity, and neighboring entries rather than extracted
as standalone temporal mentions. For this reason,
the mapping from original records to normalized
time keys is manually verified during dataset con-
struction, rather than delegated to a fully automatic
temporal-expression extractor. Representative ex-
amples of reign-key propagation and normalization
are provided in Appendix A.2.1.
B. Audit of multi-event splitting.A first LLM-
assisted step is used when a single month-level
segment contains more than one historical event.
In these cases, the model is asked only to propose
candidate event-level groupings under a fixed time
key; all such proposals are then manually reviewed
and corrected if necessary.
Table 18 reports the corresponding audit statis-
tics. Among 1,533 non-empty months, 558 con-
tain multiple events (36.41%). After LLM can-
didate grouping, only 63 multi-event months re-
quired additional human correction, corresponding
to 11.29% of multi-event months and 4.11% of all
non-empty months. The remaining 495 multi-event
months were accepted without change (88.71% di-
rect acceptance among multi-event cases). These
statistics suggest that LLM proposal is useful for
reducing manual effort, while final segmentation
quality remains controlled by explicit human re-
view.
Common correction patterns.Manual correc-
tions in this stage mainly fall into a small number of
recurrent categories: (i)boundary shift, where the
candidate split cuts too early or too late and there-
fore mixes material from adjacent events; (ii)in-

System: 你是一名严谨的 《春秋》史料问答助手 。
Prompt:  
你必须只依据下方给定资料 回答，禁止使用外部常识补全 。
如果资料不足以支持结论 ，请回答“资料不足 ”，并说明缺少什么 。
【资料（可能包含同月注疏 、相邻月近似事件 、空月占位等干扰项 ）】
你将看到若干条 DOC记录，请先判断哪些记录最能直接回答问题 。
"魯隱公元年十一月 ：經傳於是月史事闕如 ，無專條可錄 。"
"春秋之例 ，卿乃見經 。今名書於經 ，傳言司空 ，故知無駭是魯卿 。… ”
"凡卿出使 ，必具其名氏 ，以尊君命 。今不書氏 … "
"祭伯，諸侯為王卿士者 。祭國，伯爵也。傳曰：“非王命也 。”… "
… 
"冬，十有二月 ，祭伯來。祭伯者何 ？天子之大夫也 。何以不稱使 ？奔也。… "
"冬，十有二月 ，祭伯來。來者，來朝也，其弗謂朝 ，何也？寰內諸侯 … "
"十二月，祭伯來，非王命也 。"
"祭伯書來 ，諸𫝊皆以為朝 ，公羊獨以為奔 。祭伯書伯 ，諸𫝊皆以為爵 … "
"公子益師卒 。"
"公子益師卒 。何以不日 ？遠也。所見異辭 ，所聞異辭 ，所傳聞異辭 。"
"公子益師卒 。大夫日卒 ，正也；不日卒，惡也。"
… 
"二年春，公會戎于潛 。"
"二年春，公會戎于潛 。會者，外為主焉爾 。知者慮，義者行，仁者守。… "
…
【问题】
查詢《春秋》，魯隱公元年十二月這一月有什麼重要事件 ？
【你必须按以下步骤输出 】
1) 证据选择 ：从 DOC 中选出“最关键”的 1~3 条。
2) 最终回答 ：只基于你选的证据作答 （中文，2~6 句）。
3) 证据引用 ：在答案末尾列出你用到的  DOC 。
4) 若问题对应月份为空月 ：请明确写 “该月无事可书 ”（以资料措辞为准 ）。
Thinking :
好的，用户让我作为 《春秋》史料问答助手 ，只依据给定的资料来回答问题 。问题是查询鲁隐公元年十二月有什么重要事件 。我得先看看资料里这个月份有什么记
录。
资料中有鲁隐公元年十一月的记录 ，但那里文本说 “经传于是月史事阙如 ，无专条可录 ”，所以十一月没有事件 。
接下来是鲁隐公元年十二月 ，这个月份有两个部分 。第一部分是关于 “冬，十有二月 ，祭伯来”的记载，这里有《春秋》经文和三部传的详细内容 。第二部分是关
于“公子益师卒 ”的记载，同样有经文和三传的说明 。这些看起来都是十二月发生的事件 。
鲁隐公二年正月的记录是关于 “公会戎于潜 ”的，这已经是第二年的事情了 ，跟问题中的隐公元年十二月没关系 。
现在我需要选择证据 。问题问的是鲁隐公元年十二月的重要事件 ，所以十二月的两条记录直接相关 。第一条“祭伯来”和第二条 “公子益师卒 ”都是这个月发生的
事。鲁隐公元年十一月没有事件 ，鲁隐公二年正月是下一年的事 ，都不相关 。因此，我应该选择十二月的那两条作为证据 。
证据选择好了 ，接下来要做最终回答 。基于选定的证据 ，我需要说明十二月有两个重要事件 ：一是祭伯来鲁国 ，二是公子益师去世 。还要提到这些记载的来源和解
释。
最后要列出证据引用 ，也就是我用到的那两条记录的编号 。从资料结构看 ，十二月的那两部分应该对应两个证据 。
让我再确认一下 ：用户可能是在研究 《春秋》的史料，需要准确知道隐公元年十二月发生了什么 。我的回答必须严格基于资料 ，不能添加外部知识 。资料里十二月
确实有这两件事 ，所以可以给出明确回答 。
Output (Thinking) : 
1) **证据选择 **：
   - 选择鲁隐公元年十二月记录中的两个部分 （两个“versions ”）。因为该月份资料明确包含两条
独立经文记载 ：“冬，十有二月 ，祭伯来。”和“公子益师卒 。”，并附有三传的注解与讨论 ，它
们直接对应问题所询问的该月重要事件 。
2) **最终回答 **：
   根据《春秋》记载，鲁隐公元年十二月发生了两件重要事件 。第一是“冬，十有二月 ，祭伯
来”，即祭伯来到鲁国 ，《左传》指出此举 “非王命也 ”。第二是“公子益师卒 ”，即鲁国公子益
师去世，《左传》解释其卒不记载具体日期是因为 “公不与小敛 ”。这两条是本月明确记载于经文
的事件。
3) **证据引用 **：
   Evidence: 鲁隐公元年十二月  (version 1, 经文“冬，十有二月 ，祭伯来”及相关传注 )；鲁隐公
元年十二月  (version 2 ， 经文“公子益师卒 ”及相关传注 )Output (w/o Thinking) : 
1) 证据选择 ：
（魯隱公元年十二月 ）：这是问题直接对应的月份记录 ，包含两条经文及三传相关记载 ，是回答问
题的核心依据 。
2) 最终回答 ：
根据《春秋》记载，鲁隐公元年十二月发生了两件重要事件 。第一，祭伯来到鲁国 ，但《左传》指
出这是“非王命也 ”，即并非奉周天子之命 。第二，鲁国公子益师去世 ，但《春秋》未记载其去世
的具体日期 。因此，该月的重要事件是祭伯来和公子益师卒 。
3) 证据引用 ：
Evidence: 
"冬，十有二月 ，祭伯來。祭伯者何 ？天子之大夫也 。何以不稱使 ？奔也。… "
"公子益師卒 。"
Ground Truth : (1) 祭伯来；(2) 公子益师卒 。Ground Truth coverage 2/2  (correct)Grounded: cites ground truth  DOCs
Grounded: cites ground truth  DOCsGround truth  DOCs
Ground Truth coverage 2/2  (correct)Figure 11: Evidence-bounded RAG for the same query
as Fig. 10. With a small evidence pack containing the
gold month records and confusable materials, the model
recovers both gold entries and grounds the answer in
cited evidence.
appropriate merge, where two historically distinct
events are grouped together because they share a
compact annalistic sentence; and (iii)inappropri-
ate split, where commentary fragments that should
remain attached to one event are separated into dif-
ferent candidate groups. In all such cases, the final
retained record structure is determined by human
verification.
C. Audit of later-commentary alignment.A
second LLM-assisted step is used when aligning
later historiographical or commentarial materials
to original Chunqiu records. These later sources
often refer to canonical events through highly com-
pressed paraphrases, lexical reformulations, or
short subtitles rather than direct quotation. WeItem Value
Total non-empty months 1,533
Months containing multiple events 558
Fraction multi-event 36.41%
Extra human corrections 63
Correction rate among multi-event months 11.29%
Corrections among all non-empty months 4.11%
Accepted without change 495
Direct acceptance rate 88.71%
Table 18: Audit statistics for multi-event splitting (LLM
proposals + human verification).
Source #Candidates Accepted Rejected Acceptance
Gu Donggao 899 899 0 100.00%
Kong Yingda 5,286 5,179 107 97.98%
Du Yu 5,373 5,266 107 98.01%
Lü Zuqian 360 336 24 93.33%
Table 19: Acceptance rates for later-commentary align-
ments (LLM candidate proposal + human verification).
therefore use an LLM only to propose candidate
matched passages, after which human verification
determines whether the candidate alignment is ac-
cepted into the benchmark as a chrono-near confus-
able negative.
Table 19 reports the human acceptance statistics
for four representative source groups. Acceptance
rates range from 93.33% to 100.00%, indicating
that candidate proposal is generally accurate, but
still benefits from manual checking to remove resid-
ual mismatches.
Typical rejection patterns.Rejected candidate
alignments mainly arise from three sources. First,
some later commentaries refer to the correct his-
torical period but to an overly broad textual span,
making the proposed match imprecise. Second,
some candidates are semantically similar to the
target event but mismatch key participants, event
roles, or action focus. Third, some compressed
headings or summaries are ambiguous enough that
multiple canonical passages appear plausible, in
which case we conservatively reject the alignment
unless a human annotator can verify a unique and
appropriate match.
Takeaway.Across both curation stages, the role
of the LLM is restricted to efficient candidate pro-
posal. Dataset quality is controlled by manual ver-
ification and supported by explicit audit statistics.
We therefore view the resulting benchmark as a his-
torically grounded, human-verified dataset rather
than an AI-generated or synthetic resource.
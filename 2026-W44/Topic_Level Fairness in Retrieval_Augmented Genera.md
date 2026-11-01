# Topic-Level Fairness in Retrieval-Augmented Generation

**Authors**: Jaeyoung Park, Hantian Zhang, Kexin Rong, Steven Euijong Whang

**Published**: 2026-10-26 01:19:50

**PDF URL**: [https://arxiv.org/pdf/2610.34683v1](https://arxiv.org/pdf/2610.34683v1)

## Abstract
Group fairness in retrieval-augmented generation (RAG) systems is typically evaluated using metrics aggregated across queries. However, such metrics can conceal topic-level disparities: a system may favor women on questions about human rights and men on questions about actors and filmmakers, appearing nearly neutral overall even though neither topic is fair. We therefore study topic-level fairness, which requires group fairness to hold within each topic rather than only in aggregate. To reduce the topic-level disparity, we adapt the reranker separately for each topic with a Bradley-Terry objective under direct supervision from the generated output: preference pairs built from the generator's own output carry the direction that topic requires. With preference pairs built from RoMQA alone, the adaptation reduces the mean absolute per-topic disparity by 51% on RoMQA and by 40% on TREC Fair Ranking while retaining accuracy. No baseline reaches the reduction, even when modified to treat each topic separately rather than apply one correction to every topic. Neither does our method with one reranker for all topics, which follows the direction that most topics require.

## Full Text


<!-- PDF content starts -->

Topic-Level Fairness in Retrieval-Augmented Generation
Jaeyoung Park
KAIST
jypark@kaist.ac.krHantian Zhang
Meta
hantian@meta.com
Kexin Rong
Georgia Institute of Technology
krong@gatech.eduSteven Euijong Whang*
KAIST
swhang@kaist.ac.kr
Abstract
Group fairness in retrieval-augmented genera-
tion (RAG) systems is typically evaluated using
metrics aggregated across queries. However,
such metrics can conceal topic-level disparities:
a system may favor women on questions about
human rights and men on questions about ac-
tors and filmmakers, appearing nearly neutral
overall even though neither topic is fair. We
therefore study topic-level fairness, which re-
quires group fairness to hold within each topic
rather than only in aggregate. To reduce the
topic-level disparity, we adapt the reranker sep-
arately for each topic with a Bradley–Terry
objective under direct supervision from the
generated output: preference pairs built from
the generator’s own output carry the direction
that topic requires. With preference pairs built
from RoMQA alone, the adaptation reduces the
mean absolute per-topic disparity by 51% on
RoMQA and by 40% on TREC Fair Ranking
while retaining accuracy. No baseline reaches
the reduction, even when modified to treat each
topic separately rather than apply one correc-
tion to every topic. Neither does our method
with one reranker for all topics, which follows
the direction that most topics require.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come a standard approach for knowledge-intensive
NLP and is increasingly deployed in systems that
answer people’s questions with external knowledge.
A retrieval system, however, can reproduce the so-
cietal stereotypes in its corpus, even for queries
that carry no gender connotation of their own, and
the retrieved results can reinforce those stereotypes
(Rekabsaz et al., 2021). Prior work has measured
bias in RAG systems (Wu et al., 2025; Hu et al.,
2025; Zhang et al., 2026), reporting group fairness
metrics aggregated across queries.
*Corresponding authorConsider a RAG system answering questions on
different topics. The system may favor women
when answering questions about human rights but
favor men when answering questions about ac-
tors and filmmakers. A metric aggregated across
queries lets the opposite disparities cancel and can
report a nearly neutral system even though dis-
parities persist within individual topics (Figure 1,
left). The cancellation motivates topic-level fair-
ness, which requires group fairness to hold within
each topic rather than only in aggregate.
Existing debiasing methods for RAG typically
apply a common debiasing strategy across queries,
often debiasing the retrieved results without direct
supervision from the generated output (Zhao et al.,
2026; Kim et al., 2025). These methods face two
limitations. First, topics can have disparities in
opposite directions, so a correction that improves
fairness on one topic may worsen it on another.
Second, fairness in the retrieved results does not
necessarily translate to fairness in the generated
output, which a person receives (Hu et al., 2025).
We address both limitations by correcting each
topic separately, under direct supervision from the
generated output rather than the retrieved results
alone. Separate corrections move each topic in the
direction that topic requires, unaffected by the op-
posite corrections of other topics (Figure 1, center).
The corrections of all topics share one pretrained
reranker through LoRA (Hu et al., 2022). In its ba-
sic form, a RAG pipeline retrieves documents from
a corpus, reranks them to select the few that will
be used, and generates the output from those. Of
these components we adapt only the reranker, sepa-
rately for each topic. The adaptation changes the
documents the generator receives without rebuild-
ing the retrieval index or retraining the generator.
Appendix A explains why we adapt the reranker
rather than the retriever.
A reranker is a pretrained scorer, and there is
no explicit value of fairness to train it toward. We
1
arXiv:2610.34683v1  [cs.AI]  26 Oct 2026

Per-topic correction 
the generator’s reply from that document Per-topic correction 
the generator’s reply from that document 
Actors and Filmmakers 
toward women 
Human rights 
toward men Bradley–Terry on the pairs 
Bradley–Terry on the pairs +.021 
−.036 (preferred, non-preferred) = (,)
(preferred, non-preferred) = (,)
Reranker π _  HRReranker π _  AFRAG system before adaptation 
per-topic disparity GDt
−.04 0 +.04
Actors and Filmmakers +.021 
Netherlands +.019 
Literature +.002 
Sociology −.009 
Human rights −.036 
← favors women favors men → 
avg(GD) = −0.0004, avg | GD| = 0.0175 After adaptation 
−.04 0 +.04
+.017 
−.005 
+.001 
−.002 
+.010 
every topic is closer to 0 
avg(GD) = +0.0042, avg | GD| = 0.0070 per-topic disparity GDtFigure 1:Overview.The group disparity ( GD) of a RAG system between genders differs in sign across topics (left;
before adaptation, 5 of the 9 topics on RoMQA). Opposing topics cancel in asigned mean: avg(GD) is−0.0004 ,
nearly neutral, while avg|GD| , the mean absolute per-topic disparity, is 0.0175 . Adapting the reranker separately
for each topic in the direction that topic requires, on preference pairs built from the generator’s own output (center),
moves every shown topic closer to zero and lowersavg|GD|to0.0070(right).
therefore formulate the adaptation of the reranker
as preference learning. The reward model of RLHF
(Ouyang et al., 2022) is trained the same way: on
human preferences between two responses, with a
Bradley–Terry objective (Bradley and Terry, 1952)
that maximizes the probability that the preferred
response is chosen over the non-preferred one. We
adopt the objective, but build preference pairs of
documents for each query from the generator’s own
output rather than from human labels (Section 4).
We evaluate the adaptation under two protocols,
in-dataset and cross-dataset, to assess its effective-
ness also on questions beyond the training data.
The reranker is adapted on RoMQA (Zhong et al.,
2023), a multi-answer question answering dataset,
and evaluated in-dataset on RoMQA itself and
cross-dataset on TREC Fair Ranking (Ekstrand
et al., 2022), an independently constructed fair
ranking benchmark on which Wu et al. (2025) mea-
sure the fairness of RAG. The adaptation reduces
the mean absolute per-topic disparity more than any
retrieval-side or generation-side debiasing baseline,
by 51% in-dataset and by 40% cross-dataset, while
retaining accuracy (Sections 5.2.1 and 5.2.2).
To summarize, we make three contributions.
First, we show that group fairness metrics aggre-
gated across queries can conceal topic-level dispar-
ities, as topics may favor different sensitive groups.
Second, we propose a framework that adapts the
reranker for topic-level fairness under direct super-
vision from the generated output: the direction that
each topic requires is measured before adaptation,
preference pairs are built from the generator’s own
output, and the reranker is adapted separately for
each topic with a Bradley–Terry objective. Third,
we show that the reduction in disparity holds under
the in-dataset and the cross-dataset protocol.2 Related Work
Reranker training in RAG.The retriever and
the reranker of a RAG pipeline are conventionally
trained on relevance labels (Karpukhin et al., 2020;
Nogueira and Cho, 2019). In pipelines of a re-
triever and a generator alone, the retriever has been
trained on feedback from the generator (Izacard
et al., 2023; Shi et al., 2024); in a pipeline that
also reranks, Wu et al. (2026) adapt the reranker
with Proximal Policy Optimization, rewarding the
correctness of the generated output. We also adapt
the reranker from the generator’s output, but on
preference pairs built for topic-level fairness.
Measuring bias in the generated output.Wu
et al. (2025) show that utility-optimized RAG sys-
tems are biased in both the retrieved results and the
generated output. Hu et al. (2025) show that re-
trieval can bias the output of a fair generator, even
when every biased document is removed from the
corpus. Zhang et al. (2026) show that the bias of the
retrieved documents is amplified in the generated
output. We also measure the bias in the generated
output, but at the topic level rather than as a whole,
since opposing topics can cancel.
Retrieval-side debiasing.Most fairness work
in retrieval debiases the retrieved results without di-
rect supervision from the generated output. In rank-
ing, fairness constraints balance the exposure each
sensitive group receives across a ranked list (Singh
and Joachims, 2018) or the share of each sensi-
tive group in the top- k(Geyik et al., 2019), and
Rekabsaz et al. (2021) train a ranker adversarially
against the sensitive attribute. In RAG, Kim and
Diaz (2025) bring equal exposure into the pipeline,
but the fairness they study holds among documents,
not between sensitive groups. ReFaRAG (Zhao
2

et al., 2026) fixes the share of each sensitive group
among the documents the generator receives, with-
out training. Kim et al. (2025) fine-tune the re-
triever under several settings, such as the number
of epochs and of fine-tuned layers, and identify
the setting in which the generated output is unbi-
ased on test queries. FairFT (Zhang et al., 2025)
is an exception: it fine-tunes the retriever to favor
the documents that make the generator more likely
to give the fair answer. However, in our queries
the fair answer is Both , the correct reply, while
the disparity arises in replies with a single candi-
date (Section 3.2), so the probability of the fair
answer carries no direction. We instead build pref-
erence pairs from the generated output, where a
reply shows which candidate a document supports,
so no document needs a sensitive-group label.
3 Problem Definition
We measure disparity on queries built from a multi-
answer question answering (QA) dataset, adapting
the construction of Wu et al. (2025), and define
topic-level fairness. The problem is then to adapt
the reranker so that the disparity of every topic is
reduced while the rate of correct replies is retained.
3.1 Query Construction
Measuring disparity in the generated output needs a
question whose correct answer is indifferent to the
sensitive attribute, and a multi-answer QA dataset
whose golden answers are people meets the require-
ment. We assume the sensitive attribute is binary,
with groups z1andz2, and the dataset supplies the
group of every golden answer. Given a question,
choosing a golden answer from each group gives
two candidates, cz1andcz2. Each triple consisting
of the question and the two candidates is format-
ted as a multiple-choicequery xwith five options:
cand 1,cand 2,Both ,Neither , and Insufficient
information to decide . Appendix B.1 gives the
prompt of each dataset in full.
3.2 Group Disparity
The group disparity GD (Calders and Verwer,
2010; Wu et al., 2025) is the difference in how
frequently the system replies with the candidate of
each group. Let P(z 1)andP(z 2)be the fractions
of queries to which the system replies with the z1
and with the z2candidate; a reply of Both , which
is correct, counts toward neither fraction. Then
GD=P(z 1)−P(z 2),positive when the system favors z1candidates and
vice versa. GD does not depend on whether a
reply is correct, so the rate of correct replies has to
be retained as GD is reduced. One question can
yield many queries, by different choices of a golden
answer from each group. Thus, a tendency toward a
single candidate is diluted by other golden answers,
and only a tendency toward one group survives.
3.3 Topic-Level Fairness
We define topic-level fairness: a system is fair when
it is fair on every topic. Let Xbe the full query
set,Ta set of topics of interest, and Xt⊆ X the
queries whose question belongs to topic t;GDtis
then the group disparity computed on Xt. Unlike
asignedmean, the meanabsoluteper-topic group
disparity,
|GD|=1
|T |X
t∈T|GD t|,
does not let opposing topics cancel. Aggregating
by the mean lets every topic contribute equally.
A topic represents a subject of questions, and the
topic set Tis fixed before adaptation. We assume
each question belongs to at most one topic, so the
setsXtare disjoint and need not cover X. Queries
whose question belongs to no topic are excluded
from|GD| . Section 5.1.1 names the topic set we
adopt and how the questions of the multi-answer
QA dataset are assigned to topics.
3.4 Problem Statement
We consider the retrieve–rerank–generate pipeline
of Figure 2, panel 1, in which we adapt the reranker
alone. Given a query xand a corpus C, the retriever
Rreturns a pool of documents R(x,C) . Letπtbe
the reranker πadapted for topic t, andsπt(x, d) the
score πtassigns to each document din the pool.
The generator Greplies to the query concatenated
with thekdocuments of highest reranker score:
{d1, . . . , d k}= top-k
d∈R(x,C)sπt(x, d),
ˆy=G(x, d 1⊕ ··· ⊕d k).
G,C, and Rare fixed. Given a topic set Tand
the topic of each query, the problem is to find
the rerankers πtthat minimize |GD t|for every
t∈ T while retaining the rate of correct replies.
Appendix B.2 gives the promptGreplies to.
3

1Problem statement  (Sec. 3.4): retriever ( R), reranker (π), generator ( G); only πt is adapted. 
Query x
(question, candmale, candfemale)Retriever R
top-30 Generator Gŷ
→ GDtrewrite 30 docs top-5 fixed fixed
2Preference pairs  (Sec. 4.2) candmale candfemale “Both” / “Neither” / “Insufficient” 
x: “Filmmaker that won the Oscar for Best Director. Choose from the following: 
Steven Spielberg , Kathryn Bigelow , Both, Neither, or Insufficient information to decide.” 
not in D+
…x: “Nobel Peace laureate that campaigned for human rights. Choose from the following: 
Malala Yousafzai , Denis Mukwege , Both, Neither, or Insufficient information to decide.” 
d1  … Malala Yousafzai, Nobel Peace laureate… 
d2  … the Nobel Peace Prize is announced in Oslo… not in D+
d7  … Denis Mukwege won the Nobel Peace Prize… …
compare by δHR = male  ⇒ 𝑣m (x, d7) ( = 5/5) ≻ 𝑣m (x, d1) ( = 0/5) 
PHR ← PHR ∪ {(x, d7, d1)}
3Adapt the reranker  (Sec. 4.3) and re-run the pipeline with the adapted πtnon-preferred 
preferred preferred non-preferred generated output generated output 
top-5 documents 
before d 1 d 6 
d 6 d 1 
d 1 d 7 
d 7 d 1 after
before 
afterReranker  πt
keeps k=5one per topic 
Actors and Filmmakers (AF)    GDAF = +0.021  favors men  ⇒  δAF = female Human rights (HR)    GDHR = −0.036  favors women  ⇒  δHR = male
d1  … Steven Spielberg won Best Director… 
d2  … Best Director went to Bong Joon-ho… 
d6  … Kathryn Bigelow, a Best Director winner… 
PAF
PHRπAF:   L = (1/| PAF|) ( LBT + λAF Lanc )
πHR:   L = (1/| PHR|) ( LBT + λHR Lanc )LBT  widens the gap,     Lanc  holds scores in place re-run Actors and 
Filmmakers 
Human rights compare by δAF = female  ⇒ 𝑣f (x, d6) ( = 4/5) ≻ 𝑣f (x, d1) ( = 0/5) 
PAF ← PAF ∪ {(x, d6, d1)}Figure 2:Illustrative example.The method on two topics whose directions of correction are opposite.(1)The
retrieve–rerank–generate pipeline, in which the reranker alone is adapted, separately for each topic.(2)Preference
pairs (Section 4.2): the system favors men on actors and filmmakers before adaptation, so δAFis female, and women
on human rights, so δHRis male. For each retrieved document d,n= 5 replies are sampled from G(x, d) and shown
beside it, colored by candidate; a document from which no reply is a candidate falls outside D+, and of the two
that remain, the one with the higher vδtis preferred.(3)Each reranker πtis adapted on the preference pairs Pt
(Section 4.3); at the right, each query’s top-kdocuments before and after adaptation are shown.
4 Method
We mitigate topic-level bias by adapting the
reranker separately for each topic, with a Bradley–
Terry objective (Bradley and Terry, 1952) in the
direction that topic requires. Our framework builds
preference pairs of documents for each query, us-
ing the generator’s own output, and then adapts the
reranker under an anchor on the scores before adap-
tation. Figure 2 traces the method on two topics
whose directions of correction are opposite.
4.1 Preference Learning from Comparisons
A Bradley–Terry objective is an appropriate choice
when there are no explicit values but only relative
preferences between items. Let σbe the logistic
function: the Bradley–Terry objective maximizes
σ(sπt(x, d c)−s πt(x, d r)), the probability that the
preferred document dcis chosen over the non-
preferred document dr. Since there is no explicit
value to train sπt(x, d) for topic-level fairness, we
build preference pairs and adapt the reranker with
the Bradley–Terry objective.
4.2 Building Preference Pairs
A preference pair holds two documents, both from
the pool R(x,C) of one query x: the reranker miti-
gates topic-level bias by ordering that pool, and the
order decides which kdocuments enter the genera-
tor’s input. The preferred document of a pair moreAlgorithm 1Preference pairs for topict
1:δt←direction of correction fortbefore adaptation
2:P t← ∅
3:for allqueriesxof topictdo
4:D←R(x,C)
5:for alld∈Ddo
6: samplenrepliesˆy1
d, . . . ,ˆyn
d∼G(x, d)
7:v g(x, d)←1
n|{i: ˆyi
d=cg}|forg∈ {z 1, z2}
8:end for
9:D +← {d∈D:v z1(x, d)>0orv z2(x, d)>0}
10:E ← {(d c, dr)∈D2
+:vδt(x, d c)> v δt(x, d r)}
11: ˆE ← up toκpairs drawn from Ewithout replacement,
w.p.∝v δt(x, d c)−v δt(x, d r)
12:P t← P t∪ {(x, d c, dr) : (d c, dr)∈ˆE}
13:end for
14:returnP t
strongly supports the candidate of the group that
the topic of xdisfavors. Algorithm 1 builds the set
Ptof preference pairs(x, d c, dr)for topict.
Topic tcarries a direction of correction δt, mea-
sured before adaptation (line 1): δtisz2where
GDtis positive, and vice versa. In Figure 2, posi-
tiveGD AFmakes δAFfemale, and negative GD HR
makes δHRmale. Adaptation on Ptaims to bring
GDtto zero, but the pairs carry only the direction
of correction: GDtcan either stop before reaching
zero or cross zero. How far GDtmoves is decided
by|GD t|on the validation split (Section 4.3).
Each retrieved document is then probed with the
generator on the query and that document alone,
G(x, d) (lines 5–8). A reply from G(x, d) shows
4

which candidate dsupports, so preferring the doc-
uments that support the δtcandidate can mitigate
topic-level bias. Sampling nreplies from G(x, d)
gives thevote shares vz1(x, d) andvz2(x, d) , the
fractions of the replies that are the z1and the z2
candidate. With nsampled replies, vδtis graded
rather than binary, so two documents from which
the generator replies with the same candidate can
still differ inv δtand form a pair.
Pairs are formed among the documents from
which at least one reply is a candidate ( D+, lines
9–12), since only those documents support a can-
didate. Within a pair, dcis the document with
the higher vδt. At most κpairs are drawn per
query, with probability proportional to the differ-
ence in vδt, so that no single query dominates
a topic’s supply. For the actors and filmmakers
query in Figure 2, no reply sampled from G(x, d 2)
is a candidate, so d2falls outside D+. Sam-
pling from G(x, d 6)andG(x, d 1)gives the vote
shares vf(x, d 6) = 4/5 andvm(x, d 1) = 5/5 ,
so both documents are in D+. With δAF=fe-
male, vf(x, d 6)> v f(x, d 1) = 0 makes d1the
non-preferred document, and the query contributes
(x, d 6, d1)toP AF.
4.3 Training Objective
The reranker for topic tis adapted with the Bradley–
Terry objective under an anchor on the scores be-
fore adaptation. Let sref
πbe the scores before adap-
tation andλ tthe anchor weight. The loss is
L=1
|Pt| 
LBT+λtLanc
,
LBT=−X
Ptlogσ 
sπt(x, d c)−s πt(x, d r)
,
Lanc=1
2X
PtX
d∈{d c,dr} 
sπt(x, d)−sref
π(x, d)2.
The Bradley–Terry objective LBTwidens the gap
between the scores of dcanddr, and the anchor
Lancholds each score near its value before adap-
tation. We adjust λtand the learning rate ηtof
topic tto minimize |GD t|on the validation split,
subject to a fall of at most 0.5 points from the rate
of correct replies on topic tbefore adaptation. We
evaluate on the test split without readjustment.
5 Experiments
Under both the in-dataset and the cross-dataset pro-
tocol, our adaptation of the reranker reduces the
−2 0 2Recall@5, male − female
−.04 0 .04GDt
Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaFigure 3: Difference in recall@5 between the articles
of the male and the female candidate, and GDt, of the
unadapted system on RoMQA. In the shaded rows, the
system favors the candidate of higher recall@5. The
sign ofGD tdoes not follow from recall@5 alone.
mean absolute per-topic group disparity while ac-
curacy is retained, and no baseline reaches the re-
duction. The baselines do not reach the reduction
even when they treat each topic separately, and
neither does our method with one reranker for all
topics. An ablation shows that the reduction also
needs a direction of correction for each topic and
the anchor on the scores before adaptation.
5.1 Experimental Setup
5.1.1 Datasets
Our query construction needs a multi-answer QA
dataset whose golden answers are people. RoMQA
(Zhong et al., 2023) is one, and supplies the training
questions and the in-dataset evaluation. Its ques-
tions are written by crowdworkers from constraints
mined from Wikidata, such as occupation and place
of birth. TREC Fair Ranking (Ekstrand et al., 2022)
becomes one through the conversion of Wu et al.
(2025), and supplies the cross-dataset evaluation.
In the original TREC Fair Ranking, a system ranks
English Wikipedia articles for a WikiProject, a sub-
ject area that Wikipedia editors maintain; Wu et al.
(2025) convert each WikiProject into a question
asking which candidate is relevant to it. The sensi-
tive attribute is gender, with z1male and z2female.
Both datasets share one topic set T: the 9
WikiProjects of TREC Fair Ranking that hold
enough candidates of each gender, to which enough
RoMQA questions belong, and whose direction of
correction is reproducible on both datasets. For
RoMQA, a language model judges whether each
question is related to each of the 9 topics, and then
chooses the one the question is mainly about. A
5

question related to no topic belongs to none. Ap-
pendix B.3 gives the two prompts. Appendix C
gives the statistics on the two datasets and the cri-
teria of topic selection in detail.
Over these topics, the disparity arises in both re-
trieval and generation, so a correction that debiases
the retrieved results without direct supervision from
the generated output still leaves the disparity the
generator adds. Figure 3 compares the difference
in recall@5 between the Wikipedia articles of the
male and the female candidate with GDtfor each
RoMQA topic under the unadapted system, and
Figure 6 adds TREC. On 4 of the 9 RoMQA topics
and 5 of the 9 TREC topics, the system favors the
candidate whose article has the higher recall@5.
On RoMQA, the article of the female candidate has
the higher recall@5 on 8 of the 9 topics, yet the
system favors the male candidate on 4 topics.
Both protocols adapt the reranker on the
RoMQA training split and measure δton a vali-
dation split; thein-datasetprotocol takes the vali-
dation and test splits from RoMQA, and thecross-
datasetprotocol from TREC.
5.1.2 Metrics
We report fairness as the mean absolute per-topic
group disparity |GD| over the 9 topics and as GDt
for each topic. |GD| can fall while a topic moves
away from zero, so we also count the topics that
move in their direction of correction. Utility is ac-
curacy, the rate of the replyBoth: both candidates
of a query are golden answers of its question.
5.1.3 Implementation Details
The retriever, the reranker, and the gener-
ator are e5-base-v2 (Wang et al., 2022),
bge-reranker-v2-m3 (Chen et al., 2024), and
Meta-Llama-3-8B-Instruct (AI@Meta, 2024);
Appendix E replaces the reranker and the generator
with other models. The retriever returns a pool of
30 documents, and the generator replies from the
k= 5 documents of highest reranker score. Since
the retriever and the reranker are smaller than the
generator, they receive a rewrite of the query that
keeps only the question and the two candidates;
the generator receives the full query. Preference
pairs are built from n= 5 replies sampled per
document, with at most κ= 4 pairs per query.
For each topic, the learning rate ηtis adjusted over
{1,3,10,30}×10−6and the anchor weight λtover
{0,0.01,0.03,0.3} . Appendix D gives the remain-
ing settings and the computational resources.5.1.4 Baselines
The five baselines share the corpus, the generator,
k, and the rewritten query with our method, and
fall into three categories:
No debiasing.Unadapteduses the original pre-
trained reranker, whileRRPO(Wu et al., 2026)
adapts the same reranker to reward correct replies.
Retrieval-side debiasing.Gender-balanced top-
k(GB) (Geyik et al., 2019) balances the top- kbe-
tween male- and female-associated documents;Re-
triever debiasing(RD) (Kim et al., 2025) fine-tunes
the retriever.
Generation-side debiasing.Self-debiasing
(SD) (Schick et al., 2021) adjusts token probabil-
ities during each step of decoding to discourage
gender-biased generation.
For each baseline, we select the setting of lowest
validation|GD|among those that lose at most 0.5
points of accuracy againstUnadapted. If every
setting loses more than 0.5 points, we select the
setting of highest validation accuracy. Our method
follows the same rule on each topic. Appendix D.3
describes how each baseline is implemented.
5.2 Main Results
Oursreduces |GD| by 51% in-dataset and by 40%
cross-dataset while retaining accuracy, and moves
most topics in their direction of correction. Ta-
ble 1 shows the fairness and utility of the baselines
andOursunder the in-dataset and the cross-dataset
protocol. Appendix E.1 gives GDtfor every topic.
5.2.1 In-Dataset Protocol
Oursreduces |GD| from 0.0169 to 0.0083, the low-
est of all systems, and moves 8 of the 9 topics
in their direction (Table 1). Figure 4 and Figure 7
show GDtof every system, and Table 8 lists the val-
ues. Medicine is the only topic whose GDtmoves
against its direction, from +0.0046 to+0.0058
(Table 8). This result may be explained by the lim-
ited informativeness of the retrieved documents for
Medicine queries. Such queries often ask about
details that biographies rarely report, such as a
cause of death with an exclusion constraint (“Who
died of a stroke? Don’t include Canadians.”), so
a reply from G(x, d) is seldom a candidate, and
the topic yields 0.38 preference pairs per training
query against at least 0.82 on every other topic.
The limited informativeness is also reflected in the
topic’s accuracy, 17.7%, the lowest of all topics.
Full statistics of the generator’s replies by topic are
reported in Table 9 and Table 10.
6

In-dataset (RoMQA) Cross-dataset (TREC)
System|GD|Accuracy Male-fav. Female-fav.|GD|Accuracy Male-fav. Female-fav.
Unadapted 0.0169 55.1 – – 0.0156 89.9 – –
RRPO 0.0170 56.1 1/4 2/5 0.0156 89.9 3/62/3
GB 0.0193 55.4 1/4 1/5 0.0112 92.1 4/62/3
RD 0.0162 54.3 1/4 4/5 0.0143 89.96/60/3
SD 0.0174 54.9 0/4 4/5 0.0166 89.4 2/6 1/3
Ours0.008355.23/4 5/5 0.009490.26/6 2/3
Table 1: Fairness and utility of the baselines andOursunder the in-dataset and the cross-dataset protocol. In
Male-fav.andFemale-fav., k/n means that kof the ntopics on which the unadapted system favors the male or the
female candidate move in their direction of correction. Bold marks the lowest |GD| and the highest kper protocol.
−.04 0 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted RRPO GB RD SD Ours
Figure 4: GDtof Table 1 under the in-dataset protocol.
Oursmoves 8 of 9 topics in their direction of correction.
RRPO achieves the highest accuracy, likely from
its reward on the correct reply, but does not reduce
|GD| . GB fails to lower |GD| : balancing the top- k
by gender may mostly exchange documents about
people other than the candidates. RD fine-tunes the
retriever but corrects every topic in one direction:
most female-favoring topics move toward the male
candidate, while most male-favoring topics move
away from zero, so |GD| falls only slightly. SD
debiases the generator without a direction for any
topic, and leaves |GD| nearly unchanged while
lowering accuracy slightly.
The reduction is smaller with another reranker
and under two other generators. Adapted the same
way, jina-reranker-v2-base-multilingual , a
smaller model trained on other data, reaches
|GD| of 0.0108 from 0.0160 (Table 11). Under
Qwen2.5-7B-Instruct , a model of another fam-
ily, and Meta-Llama-3-70B-Instruct , a larger
model of the same release,Oursreaches |GD| of
0.0063 and 0.0202 from 0.0092 and 0.0302, respec-
tively, while accuracy is retained (Table 12).5.2.2 Cross-Dataset Protocol
The reduction holds under the cross-dataset proto-
col, although the preference pairs are built from
the RoMQA training split alone.Oursreduces
|GD|from 0.0156 to 0.0094, the lowest of all sys-
tems, and moves 8 of the 9 topics in their direction
(Figure 7b), while accuracy rises by 0.3 points (Ta-
ble 1). Human rights is the one topic that does not
move: its GDtis−0.0075 against −0.0073 before
adaptation, already close to zero (Table 8).
In contrast to the in-dataset result, GB lowers
|GD| more than any other baseline and raises ac-
curacy, thoughOursstill shows a lower |GD| (Ta-
ble 1). On TREC, 32.0% of the top- kslots of the
unadapted system hold a document associated with
neither gender, which is not about a person and
is likely less informative to the generator. While
balancing the genders in the top- k, GB can fill
these slots with documents about people and so
may make the generator’s input more informative
as a side effect. Every RoMQA document is al-
ready about a person. The other baselines change
|GD|little, as under the in-dataset protocol.
5.3 Per-Topic Against Shared Correction
Ourscorrects each topic with a reranker of its own,
while each baseline applies one correction to ev-
ery topic. We therefore modifyOursto adapt one
reranker for all topics (Ours, one reranker), and
RD and SD to treat each topic separately (RD per
topic and SD per topic). Table 2 shows the fair-
ness and utility of the modified systems under the
in-dataset protocol, and Table 13 in Appendix E.4
under both protocols. Figure 5 and Figure 8 show
GDtof these systems, and Table 14 lists the values.
Oursneeds a separate reranker for each topic,
even though its pairs carry the direction of each
topic. The single reranker follows the direction
that most topics require, regardless of the direction
7

In-dataset (RoMQA)
System|GD|AccuracyMale-
fav.Female-
fav.
Unadapted 0.0169 55.1 – –
Ours, one reranker 0.0175 55.5 1/45/5
RD per topic 0.0118 54.84/44/5
SD per topic 0.0170 55.0 0/4 3/5
Ours0.008355.2 3/45/5
Table 2:Ourswith one reranker shared by all topics,
and the systems that treat each topic separately, under
the in-dataset protocol. The columns are defined as in
Table 1.Ours, one rerankermoves most topics toward
the male candidate.
−.04 0 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted
Ours, one rerankerRD per topic
SD per topicOurs
Figure 5: GDtof Table 2 under the in-dataset protocol.
Oursis the closest to zero overall.
of each: only 1 of the 4 male-favoring RoMQA
topics and none of the 3 female-favoring TREC
topics move in their own direction (Table 13).
RD per topic moves as many topics in their di-
rection asOursbut keeps a higher |GD| , since the
retriever learns from the gender of each document
rather than from the generated output. Treating
each topic separately changes little for SD, since
SD per topic chooses only the strength of its cor-
rection for each topic, not the direction.
5.4 Ablation Study
Oursadapts the reranker of each topic in the di-
rection that topic requires, under an anchor on the
scores before adaptation. We remove these two
parts one at a time:one directioncorrects every
topic in the direction that most topics require, and
without anchorsets λt= 0for every topic. Table 3
shows the fairness and utility of the variants under
the in-dataset protocol, Table 15 in Appendix E.5
under both protocols, and Table 16 lists their GDt.In-dataset (RoMQA)
Variant|GD|AccuracyMale-
fav.Female-
fav.
Unadapted 0.0169 55.1 – –
Ours0.008355.23/4 5/5
one direction 0.0176 54.8 0/45/5
without anchor 0.0142 55.0 0/4 4/5
Table 3: Ablation under the in-dataset protocol. The
columns are defined as in Table 1.One directioncor-
rects every topic in the direction that most topics require;
without anchorsetsλ t= 0for every topic.
Oursneeds a direction for each topic, since a
single direction pushes the topics that require the
other direction away from zero. On RoMQA,one
directionmoves none of the 4 male-favoring topics
toward the female candidate. Moreover, the two
that start farthest from zero move further toward
the male candidate, more than doubling in GDt:
Actors and Filmmakers from +0.0211 to+0.0495 ,
and Netherlands from +0.0194 to+0.0518 (Ta-
ble 16). On TREC, none of the 3 female-favoring
topics moves toward the male candidate (Table 15).
The anchor lets a reranker take a step large
enough to correct its topic while retaining accu-
racy. Without the anchor, a larger step raises |GD t|
or loses accuracy on most topics, sowithout an-
chorretains accuracy with smaller steps but moves
fewer topics in their direction thanOurs, 4 against
8 on RoMQA and 6 against 8 on TREC (Table 15).
6 Conclusion
The group disparity of a RAG system can differ in
sign across topics: a signed mean over topics can
report a nearly neutral system, and a correction in
one direction for all topics pushes the topics that
require the other direction away from zero. We
therefore define topic-level fairness and adapt the
reranker separately for each topic under direct su-
pervision from the generated output: preference
pairs built from the generator’s own output carry
the direction that topic requires. The anchor on the
scores before adaptation lets a reranker take a step
large enough to correct its topic while accuracy is
retained. The adaptation reduces the mean abso-
lute per-topic group disparity by 51% in-dataset
on RoMQA and by 40% cross-dataset on TREC
Fair Ranking while retaining accuracy; no baseline
reaches the reduction, even when modified to treat
each topic separately, and neither does our method
with one reranker for all topics.
8

Limitations
•Preference pairs from G(x, d) .The effect
of one document on the generator’s reply de-
pends on the other k−1 documents in its
input, but our method measures the effect
onG(x, d) , the reply to that document alone.
Izacard et al. (2023) likewise train a retriever
from the output on each document alone for a
reader that attends over all documents jointly,
and find it on par with scoring each document
by its removal from the topk.
•Topic set.A deployed RAG system takes the
topics its provider targets; we choose the topic
set and the topic of each RoMQA question by
criteria of our own. WikiProjects are a natural
topic set: a WikiProject is a subject area that
Wikipedia editors maintain, and each ques-
tion of TREC Fair Ranking is converted from
one WikiProject. The 9 WikiProjects we use
pass the three filters of Appendix C, and each
RoMQA question belongs to the one topic it
is mainly about. We leave the questions that
belong to several topics at once to future work.
•Aggregation over topics.Any single value
over the topics conceals what happened on
each topic: neither |GD| nor any other aggre-
gated value guarantees a lower |GD t|on every
topic. A fairness condition satisfied on the ag-
gregate but violated on one topic is analogous
to the fairness gerrymandering of Kearns et al.
(2018), so we report every GDtbeside |GD| .
•Fairness measure. GD captures one aspect
of fairness, and a lower GD is best read as the
system replying more evenly with the candi-
date of each group.
•One sensitive attribute.The sensitive at-
tribute is gender in every experiment. Another
attribute may enter through the same construc-
tion wherever a dataset supplies it for every
golden answer.
•The attribute is binary.Our query construc-
tion assumes a binary sensitive attribute; an
attribute with more than two values would
have to be reduced to two groups at a time,
and we have not tested that reduction.
•English-only.All experiments are in English;
cross-lingual fairness is out of scope.Ethical Considerations
Gender is treated as binary.The male and female
labels of RoMQA and TREC Fair Ranking come
from Wikidata. We adopt the labels for measure-
ment convenience: a per-topic rate needs enough
queries in each group, and the datasets supply that
for two labels alone. Non-binary and transgender
people are therefore absent from every number we
report and every correction our method makes.
References
AI@Meta. 2024. Llama 3 model card.
https://github.com/meta-llama/llama3/
blob/main/MODEL_CARD.md.
Ralph Allan Bradley and Milton E Terry. 1952. Rank
analysis of incomplete block designs: I. the method
of paired comparisons.Biometrika, 39(3/4):324–
345.
Toon Calders and Sicco Verwer. 2010. Three naive
Bayes approaches for discrimination-free classifi-
cation.Data Mining and Knowledge Discovery,
21(2):277–292.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
Embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Associa-
tion for Computational Linguistics: ACL 2024, pages
2318–2335.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The Faiss library.arXiv preprint
arXiv:2401.08281.
Cynthia Dwork and Christina Ilvento. 2019. Fairness
under composition. In10th Innovations in Theoret-
ical Computer Science Conference (ITCS), volume
124 ofLIPIcs, pages 33:1–33:20.
Michael D. Ekstrand, Graham McDonald, Amifa Raj,
and Isaac Johnson. 2022. Overview of the TREC
2022 fair ranking track. InProceedings of the
Thirty-First Text REtrieval Conference (TREC 2022).
ArXiv:2302.05558.
Sahin Cem Geyik, Stuart Ambler, and Krishnaram Ken-
thapadi. 2019. Fairness-aware ranking in search
& recommendation systems with application to
LinkedIn talent search. InProceedings of the 25th
ACM SIGKDD International Conference on Knowl-
edge Discovery & Data Mining, pages 2221–2231.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations.
9

Mengxuan Hu, Hongyi Wu, Ronghang Zhu, Zihan
Guan, Dongliang Guo, Daiqing Qi, and Sheng Li.
2025. No free lunch: Retrieval-augmented genera-
tion undermines fairness in LLMs, even for vigilant
users. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2025, pages 18145–
18170.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025. FlashRAG: A mod-
ular toolkit for efficient retrieval-augmented genera-
tion research. InCompanion Proceedings of the ACM
on Web Conference 2025, pages 737–740. ACM.
Jina AI. 2024. jina-reranker-v2-base-
multilingual. https://huggingface.co/jinaai/
jina-reranker-v2-base-multilingual.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781.
Michael Kearns, Seth Neel, Aaron Roth, and Zhi-
wei Steven Wu. 2018. Preventing fairness gerryman-
dering: Auditing and learning for subgroup fairness.
InProceedings of the 35th International Conference
on Machine Learning, pages 2564–2572.
Taeyoun Kim, Jacob Mitchell Springer, Aditi Raghu-
nathan, and Maarten Sap. 2025. Mitigating bias
in RAG: Controlling the embedder. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 18999–19024.
To Eun Kim and Fernando Diaz. 2025. Towards fair
RAG: On the impact of fair ranking in retrieval-
augmented generation. InProceedings of the 2025
International ACM SIGIR Conference on Innovative
Concepts and Theories in Information Retrieval (IC-
TIR), pages 33–43.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with PagedAttention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles, pages 611–626.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.Sourab Mangrulkar, Sylvain Gugger, Lysandre De-
but, Younes Belkada, Sayak Paul, Benjamin Bossan,
and Marian Tietz. 2022. PEFT: State-of-the-art
parameter-efficient fine-tuning methods. https:
//github.com/huggingface/peft.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with BERT.arXiv preprint
arXiv:1901.04085.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback. InAdvances in Neural Information
Processing Systems, volume 35, pages 27730–27744.
Navid Rekabsaz, Simone Kopeinik, and Markus Schedl.
2021. Societal biases in retrieved contents: Measure-
ment framework and adversarial mitigation of BERT
rankers. InProceedings of the 44th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 306–316.
Timo Schick, Sahana Udupa, and Hinrich Schütze. 2021.
Self-diagnosis and self-debiasing: A proposal for re-
ducing corpus-based bias in NLP.Transactions of the
Association for Computational Linguistics, 9:1408–
1424.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 8371–8384.
Ashudeep Singh and Thorsten Joachims. 2018. Fairness
of exposure in rankings. InProceedings of the 24th
ACM SIGKDD International Conference on Knowl-
edge Discovery & Data Mining, pages 2219–2228.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven
Le Scao, Sylvain Gugger, and 3 others. 2020. Trans-
formers: State-of-the-art natural language processing.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations, pages 38–45. Association for Com-
putational Linguistics.
10

Xuyang Wu, Shuowei Li, Hsin-Tai Wu, Zhiqiang Tao,
and Yi Fang. 2025. Does RAG introduce unfairness
in LLMs? Evaluating fairness in retrieval-augmented
generation systems. InProceedings of the 31st Inter-
national Conference on Computational Linguistics,
pages 10021–10036.
Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi
Zhou, Zhen Wu, Xinyu Dai, and Rui Xia. 2026. Op-
timizing RAG rerankers with LLM feedback via re-
inforcement learning. InProceedings of the 64th An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 30474–
30490.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, and 23 oth-
ers. 2024. Qwen2.5 technical report.arXiv preprint
arXiv:2412.15115.
Tianhui Zhang, Yi Zhou, and Danushka Bollegala. 2026.
Evaluating the effect of retrieval augmentation on so-
cial biases. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
5004–5026.
Zheng Zhang, Ning Li, Qi Liu, Rui Li, Weibo Gao,
Qingyang Mao, Zhenya Huang, Baosheng Yu, and
Dacheng Tao. 2025. The other side of the coin: Ex-
ploring fairness in retrieval-augmented generation.
arXiv preprint arXiv:2504.12323.
Yingqi Zhao, Vasilis Efthymiou, Jyrki Nummenmaa,
and Kostas Stefanidis. 2026. ReFaRAG: Re-ranking
for bias mitigation in retrieval-augmented generation.
InNew Trends in Database and Information Systems
(ADBIS 2025), volume 2676 ofCommunications in
Computer and Information Science, pages 516–530.
Springer.
Victor Zhong, Weijia Shi, Wen-tau Yih, and Luke Zettle-
moyer. 2023. RoMQA: A benchmark for robust,
multi-evidence, multi-answer question answering. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 7055–7067.
11

A Why the Reranker Is a Suitable Target
Continuing from Section 1, this appendix argues
that correcting one component of the pipeline is
enough. Of the pipeline’s components, a corpus
is rebuilt rather than trained and a generator is ex-
pensive to fine-tune and is usually deployed as a
general-purpose pretrained model, which leaves
the retriever and the reranker, and the reranker is
the proper choice.
Correcting one component can leave the system
fair at the level of its final output, and correcting
every component would not be the way to get there:
Dwork and Ilvento (2019) show that components
satisfying a group fairness criterion in isolation
need not compose into a system that satisfies it.
Retrieval pipelines have been corrected this way
before: Rekabsaz et al. (2021) reduce bias in re-
trieved contents by adapting the ranker alone, with-
out losing retrieval utility, and, closest to our work,
Kim et al. (2025) fine-tune the retriever alone and
identify the setting in which the generated output
is unbiased on test queries. Our results are of the
same kind: every reduction of our method comes
from changing the reranker alone.
A fairness objective on the retriever can lower
the retriever’s recall: the retriever decides which
documents enter the pool, and a document it leaves
out cannot be recovered by any later stage. The
retriever is also a bi-encoder, scoring a document
by the inner product of a query embedding and a
document embedding computed before any query
exists, which is what makes an index searchable,
so a correction that depends on the sensitive group
that a document supports can enter only through the
query embedding, where the same inner product
must carry relevance at the same time.
The reranker has neither limitation: it reorders
the pool the retriever returned, so the retriever’s
recall is preserved, and, as a cross-encoder, it
jointly encodes the query and the document, so
the reranker score can condition on the sensitive
group that the document supports directly, not only
through a query embedding.
B Prompts
This appendix reproduces the prompts used by the
pipeline and in dataset construction; braces mark
substitution slots.B.1 Multiple-Choice Query
Continuing from Section 3.1, this is the prompt that
wraps a question and its two candidates into one
query x, with one template per dataset. TREC Fair
Ranking keeps the template of Wu et al. (2025),
which turns a WikiProject into a question:
Which answer is relevant to the topic
’{WikiProject}’? Select the relevant
answer based on their relationship
to ’{WikiProject}’, rather than just
matching words. Choose from the
following: {c 1}, {c 2}, Both, Neither,
or Insufficient information to decide.
RoMQA supplies questions, so its template
places the question directly before the same op-
tions:
{RoMQA question} Choose from the
following: {c 1}, {c 2}, Both, Neither,
or Insufficient information to decide.
{c1}and{c2}are the two candidate names, in
an order randomized with respect to group.
B.2 Generation
Continuing from Section 3.4, this is the prompt the
generator Greplies to at temperature 0 for every
reported GD and accuracy value. System message:
Answer the question based on the given
document.
Only give me the answer and do not output
any other words.
The following are given documents.
{reference}
User message:
Question: {question}
{question} is the query xitself, and
{reference} is the kreranker-selected documents
in reranked order, one per line, each rendered as
Doc {i}({title}) {text}
with {title} and{text} taken from the cor-
pus record. The per-document pass of Section 4.2
uses the same template with a single document in
{reference}and is sampled (Appendix D.1).
B.3 LLM Topic Membership
Continuing from Section 5.1.1, these are
the two prompts that decide the topic of a
RoMQA question (Appendix C), both served by
Meta-Llama-3-8B-Instruct at temperature 0.
The first is sent once for each question and topic,
with max_tokens= 3 , and the question is related
to the topic when the response begins withyes:
12

Is the following question related to the
topic "{topic}"? Answer only yes or no.
Question: {question}
The second is sent for a question related to more
than one topic, with max_tokens= 4 and the an-
swer restricted to the number of a listed topic:
The following question is related to
several topics. Which ONE of these
topics is the question mainly about?
Question: {question}
Topics:
1. {topic}
2. {topic}
. . .
Answer with the number of the topic only.
The related topics are listed in a random order,
since a language model uses the information in its
input unevenly by position (Liu et al., 2024).
A question related to one topic belongs to that
topic without the second prompt, and a question
related to no topic belongs to none. Neither prompt
shows the candidates, so the topic of a question
does not depend on the genders of its candidates.
C Dataset and Topic Construction
Continuing from Section 5.1.1, this appendix gives
the statistics on the two datasets (Table 4), the crite-
ria of topic selection and of topic membership, and
the recall@5 of each topic.
RoMQA.Of the 12,775 RoMQA questions
whose golden answers are people, 5,391 hold at
least one golden answer of each gender. The golden
answers of each question are divided among the
training, validation and test splits, and the 1,018
questions we keep each hold at least three golden
answers of each gender in every split, so that ev-
ery question yields enough queries in each split.
We use the 961 of the 1,018 questions that belong
to one of the 9 topics, as the Topic membership
paragraph below decides. Within a split, each can-
didate of the larger gender is matched with up to
ten candidates of the other, and each match makes
one query with its two candidates in random order.
The corpus is one index shared by the three splits,
holding the introduction of the Wikipedia article of
each golden answer that has one.
TREC Fair Ranking.The golden answers of
each TREC question are halved by gender between
the validation and the test split. Within a split, each
male candidate, the larger group in every topic,Train Validation Test
RoMQA
Questions 961 961 961
Male candidates 17,392 17,329 17,254
Female candidates 11,858 11,646 11,429
Queries 148,586 146,230 143,738
TREC Fair Ranking
Questions – 9 9
Male candidates – 1,798 1,804
Female candidates – 452 455
Queries – 71,920 72,160
Table 4: Statistics of the two datasets over the 9 topics.
The two candidates of every query are one male and
one female golden answer of its question. A person
who is a golden answer of two questions counts as two
candidates.
Topic Female Male
Actors and Filmmakers 128 373
Computer science 78 423
Former countries 80 421
Human rights 154 347
Literature 117 384
Medicine 69 432
Netherlands 74 427
Sociology 114 387
Southeast Asia 93 408
Table 5: Candidates of each gender in each of the 9
topics of TREC Fair Ranking, summed over both splits.
is matched with 20 female candidates, and each
match makes two queries. The corpus is the whole
collection of the 2022 track in one index shared by
every topic and split, each article truncated to its
first 100 words, as in Wu et al. (2025).
Topic selection.Topic selection starts from the
50 WikiProjects in the release of Wu et al. (2025),
and three filters reduce the set to the 9 topics we
use. The first keeps a WikiProject with at least 50
candidates of each gender, which leaves 24. Each
RoMQA question is then assigned one of the 24
by the rule of the Topic membership paragraph
below, and the second filter keeps a WikiProject
to which at least 10 questions are assigned, which
leaves 16. The third keeps a WikiProject whose
direction of correction is reproducible: GDtun-
der the unadapted system has the same sign on the
validation and the test split of each dataset, which
leaves 9. The two splits of each dataset hold dis-
joint candidates of the same questions, so a change
of sign between them shows that GDtis too noisy
to determine the direction of correction. The third
filter uses only the replies of the unadapted system,
and every system is evaluated on the same 9 topics.
13

Topic Questions Female Male
Actors and Filmmakers 216 8,847 11,643
Computer science 50 1,879 2,829
Former countries 104 3,001 5,468
Human rights 15 502 778
Literature 209 7,604 11,533
Medicine 26 917 1,467
Netherlands 46 1,768 2,552
Sociology 274 9,536 14,731
Southeast Asia 21 879 974
Table 6: Candidates of each gender in each of the 9
topics of RoMQA, summed over the three splits, and
the questions of each topic. A person who is a golden
answer of two questions counts as two candidates.
Recall@5
Topic Male FemaleMale−
FemaleGD t
RoMQA
Actors and Filmmakers 91.2 92.9−1.7 +0.0211
Computer science 93.1 95.3−2.2−0.0072
Former countries 94.1 94.8−0.7−0.0121
Human rights 95.7 97.9−2.2−0.0356
Literature 93.3 93.7−0.4 +0.0024
Medicine 85.0 87.8−2.8 +0.0046
Netherlands 94.2 95.2−1.0 +0.0194
Sociology 93.6 96.1−2.5−0.0092
Southeast Asia 90.6 88.7+1.9−0.0409
TREC Fair Ranking
Actors and Filmmakers 93.8 92.6+1.2 +0.0385
Computer science 99.7 99.9−0.2 +0.0002
Former countries 81.1 92.3−11.2 +0.0083
Human rights 90.9 90.3+0.6−0.0073
Literature 96.8 98.0−1.2−0.0031
Medicine 96.9 94.3+2.6 +0.0343
Netherlands 89.6 92.4−2.8−0.0049
Sociology 98.7 91.9+6.8 +0.0396
Southeast Asia 86.8 89.7−2.9 +0.0043
Table 7: Recall@5 and GDtof the unadapted system
on each topic, the values that Figure 6 draws.
Every question is then assigned again among the 9.
Table 5 gives the candidates of each gender in the 9,
and Table 6 the RoMQA questions and candidates
of each.
Topic membership.A RoMQA question is as-
signed a topic in two calls to a language model
(Appendix B.3): the first asks, for each topic of
the set, whether the question is related to it, and
the second asks which one of the related topics the
question is mainly about.
Recall@5.Figure 6 adds TREC Fair Ranking to
Figure 3, and Table 7 gives the recall@5 and GDt
of the unadapted system for each topic.
−2 0 2Recall@5, male − female
−.04 0 .04GDt
Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast Asia(a) RoMQA
−10 0 10Recall@5, male − female
−.04 0 .04GDt
Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast Asia
(b) TREC Fair Ranking
Figure 6: Difference in recall@5 and GDtof the un-
adapted system on both datasets. Full version of Fig-
ure 3. Each panel has its own scale for recall@5, and
Table 7 lists the values.
D Hyperparameters and Computational
Resources
Continuing from Section 5.1.3, this appendix gives
the settings of retrieval, generation, adaptation and
pair construction, the template of the rewritten
query, and the computational resources.
D.1 Models and Training
Models.Retriever e5-base-v2 (110M); reranker
bge-reranker-v2-m3 (568M cross-encoder);
generator Meta-Llama-3-8B-Instruct (8.0B).
Appendix E.2 also adapts a 278M reranker,
jina-reranker-v2-base-multilingual (Jina
AI, 2024), and Appendix E.3 replaces the generator
with Qwen2.5-7B-Instruct (7.6B; Yang et al.,
2024) and with Meta-Llama-3-70B-Instruct
(70.6B, served in FP8; AI@Meta, 2024).
Retrieval and generation.The retriever en-
codes the rewritten query with the prefix query:
and each document of the corpus with passage: ,
both in at most 512 tokens, and returns the pool
14

of documents of highest inner product. The gen-
erator decodes at temperature 0 with at most 32
new tokens. A reply is lower-cased and stripped of
punctuation and articles, then matched first against
the options Both ,Neither and Insufficient
information to decide , and otherwise against
the names of the two candidates; a reply that names
both candidates is Both . A reply that matches no
option and no candidate is almost always the name
of a third person; such replies are at most 3.2%
of the unadapted system’s replies on RoMQA and
1.5% on TREC.
Adaptation.Each topic has its own LoRA
adapter, with r= 16 andα= 32 on the query
and value projections of the cross-encoder’s atten-
tion blocks, no bias terms and no dropout, and its
own copy of the classification head, trained in full:
2.62M parameters per topic, 0.46% of the reranker.
The reranker’s own dropout of 0.1 is active dur-
ing adaptation and off for the scores before adapta-
tion. One epoch of AdamW with betas (0.9,0.999) ,
ϵ= 10−8and weight decay 0.01, a constant learn-
ing rate without warmup, gradient-norm clipping
at1.0, and batches of 32 pairs.
Objective and pair construction.The logit
difference inside the Bradley–Terry objective is
divided by a fixed temperature of 1.5, with no mar-
gin. Pair construction (Section 4.2) samples the n
replies at temperature 0.7withtop-p= 0.9 and at
most 16 new tokens.
D.2 Query Rewriting
The retriever and the reranker receive a rewrite of
the query in the following template:
Compare {c 1} and {c 2} in relation to
{WikiProject or RoMQA question}.
D.3 Baselines
Continuing from Section 5.1.4, this subsection
states how we implement each baseline.
RRPOrewards a reply of Both with 1 and every
other reply with 0, since Both is the correct reply
to every query. We adapt the reranker through the
same LoRA adapter as our method.
GBassociates a document with the gender of its
subject, and a document without a gender, which
Geyik et al. (2019) do not consider, takes a slot of
the top-5 only while each gender can still reach its
minimum count.
RDfine-tunes the retriever toward the gender
that the unadapted system disfavors on most topics,measured on the validation split of each protocol,
and labels a document by the gender of its subject.
SDprefixes a sentence describing gender dis-
crimination to the system message of Appendix B.2
and decodes greedily as every other system does,
with the Transformers library (Wolf et al., 2020),
since vLLM (Kwon et al., 2023) cannot advance
two inputs together.
D.4 Computational Resources
All experiments except the 70B generator run on
four NVIDIA RTX 4090 GPUs (24 GB each) of
one server with two Intel Xeon Silver 4410T CPUs;
the 70B generator is served on two NVIDIA RTX
PRO 6000 Blackwell GPUs (96 GB each). Gen-
eration is served by vLLM (Kwon et al., 2023),
and the retrieval pipeline is built on FlashRAG (Jin
et al., 2025). Each GPU hosts one vLLM server
and one training run: the server is put to sleep while
the reranker trains and scores, and woken for gen-
eration. LoRA is implemented with PEFT (Man-
grulkar et al., 2022), and the retriever’s indexes are
built with FAISS (Douze et al., 2024). One adap-
tation of the reranker for one topic, scored on the
validation split, takes 0.44 GPU-hours on average
under the in-dataset protocol and 0.37 GPU-hours
under the cross-dataset protocol.
E Full Experimental Results
This appendix gives the per-topic values of every
system, repeats the in-dataset result with another
reranker and two other generators, and reports the
modified systems and the ablation variants under
both protocols.
E.1 Main Results per Topic
Continuing from Section 5.2, this appendix holds
the per-topic values that Table 1 summarizes. Ta-
ble 8 gives GDton every topic for each system of
Table 1, and Figure 7 draws the same values on
both datasets, as Figure 4 draws them on RoMQA.
Table 9 and Table 10 give the share of each reply of
the unadapted system and ofOurson every topic
of RoMQA and of TREC, and the preference pairs
built per training query under each protocol, which
Section 5.2.1 reads for Medicine.
E.2 Another Reranker Backbone
Continuing from Section 5.2.1, this appendix
adapts jina-reranker-v2-base-multilingual ,
which differs from bge-reranker-v2-m3 in size
15

and training data, under the in-dataset protocol. Ta-
ble 11 gives GDt,|GD| and accuracy with each
reranker, unadapted and adapted.
The reduction is smaller with the other reranker:
|GD| falls from 0.0160 to 0.0108, a reduction of
33% against the 51% of bge-reranker-v2-m3 ,
and accuracy is retained. On Actors and Filmmak-
ers, every step that moves GDtfurther on the vali-
dation split loses more than 0.5 points of accuracy,
soGD tmoves only from+0.0317to+0.0308.
E.3 Other Generators
Continuing from Section 5.2.1, this appendix re-
places the generator under the in-dataset protocol:
Qwen2.5-7B-Instruct changes the model fam-
ily at about the size of the main generator, and
Meta-Llama-3-70B-Instruct changes the size
within the same release. Table 12 gives GDt,|GD|
and accuracy under each generator.
The reduction is smaller under both gen-
erators, and accuracy is retained. Under
Qwen2.5-7B-Instruct , the unadapted system
starts closer to zero, at |GD| of 0.0092 against
0.0169, so less disparity is left to remove: adap-
tation lowers |GD| to 0.0063, a reduction of 31%
against the 51% under the main generator. Under
Meta-Llama-3-70B-Instruct , the unadapted sys-
tem starts farther from zero, at |GD| of 0.0302, and
adaptation lowers |GD| to 0.0202, removing more
disparity than under the main generator, 0.0100
against 0.0086, but a smaller share of it, 33%. On
Computer science, the topic farthest from zero, ev-
ery step that moves GDtfurther on the validation
split loses more than 0.5 points of accuracy, so
GDtmoves only from−0.0513to−0.0492.
E.4 Modifications of the Baselines and ofOurs
Continuing from Section 5.3, this appendix de-
scribes how RD and SD are modified to treat each
topic separately and howOurs, one rerankeris
built. Each modified baseline keeps the implemen-
tation of Appendix D.3 and chooses its setting for
each topic as our method does.
RD per topicfine-tunes one retriever for each
topic, in the direction of correction of that topic.
SD per topicchooses, for each topic, the decay
constant λSDof Schick et al. (2021) from {1,2,3} .
Ours, one rerankeradapts a single reranker
on the pairs of all 9 topics, each pair keeping the
direction of its topic. One learning rate and one
anchor weight, adjusted on validation|GD|as the
setting of a baseline is, serve all topics.For every system of Table 2, Table 13 gives the
fairness and utility under both protocols, Figure 8
draws GDton both datasets, and Table 14 gives
GDt,|GD|and accuracy.
E.5 Ablation Results
Continuing from Section 5.4, Table 15 gives the
fairness and utility of every variant under both pro-
tocols. Table 16 givesGD t,|GD|and accuracy.
On TREC, the 3 female-favoring topics start
within 0.0073 of zero, soone directionstays close
toOurs(Table 15).Without anchoralso stays
closer toOurson TREC than on RoMQA, since
Oursalready setsλ t= 0for 4 of the 9 topics.
16

−.04 0 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted RRPO GB RD SD Ours(a) In-dataset (RoMQA)
−.04 −.02 0 +.02 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted RRPO GB RD SD Ours (b) Cross-dataset (TREC)
Figure 7: GDtof Table 1 under both protocols. Full version of Figure 4.Oursmoves 8 of 9 topics in their direction
of correction under each protocol. Each panel has its own scale, centered on zero.
Topic Unadapted RRPO GB RD SD Ours
In-dataset (RoMQA)
Actors and Filmmakers+0.0211+0.0213+0.0270+0.0322+0.0219+0.0165
Computer science−0.0072−0.0073−0.0108+0.0077−0.0061+0.0043
Former countries−0.0121−0.0119−0.0074+0.0047−0.0106+0.0041
Human rights−0.0356−0.0413−0.0395−0.0417−0.0395+0.0104
Literature+0.0024+0.0046+0.0038+0.0122+0.0033+0.0010
Medicine+0.0046+0.0039+0.0087+0.0133+0.0053+0.0058
Netherlands+0.0194+0.0203+0.0085+0.0172+0.0250−0.0051
Sociology−0.0092−0.0106−0.0099−0.0049−0.0090−0.0019
Southeast Asia−0.0409−0.0315−0.0576−0.0121−0.0358−0.0255
|GD|0.0169 0.0170 0.0193 0.0162 0.01740.0083
Accuracy 55.1 56.1 55.4 54.3 54.9 55.2
Cross-dataset (TREC)
Actors and Filmmakers+0.0385+0.0414+0.0213+0.0138+0.0389+0.0074
Computer science+0.0002−0.0002+0.0006−0.0001+0.0008−0.0005
Former countries+0.0083+0.0089−0.0121−0.0192+0.0122−0.0076
Human rights−0.0073−0.0079−0.0039−0.0112−0.0072−0.0075
Literature−0.0031−0.0017−0.0034−0.0056−0.0038−0.0029
Medicine+0.0343+0.0352+0.0236+0.0294+0.0341+0.0226
Netherlands−0.0049−0.0034−0.0018−0.0085−0.0053+0.0018
Sociology+0.0396+0.0371+0.0274+0.0375+0.0430+0.0326
Southeast Asia+0.0043+0.0042+0.0072+0.0038+0.0039+0.0023
|GD|0.0156 0.0156 0.0112 0.0143 0.01660.0094
Accuracy 89.9 89.9 92.1 89.9 89.4 90.2
Table 8: GDtof every system of Table 1 under both protocols. The rows |GD| and Accuracy give, for each column,
the mean absolute per-topic group disparity and the accuracy that Table 1 reports. Bold marks, in each row, the
value closest to zero.
17

Unadapted Ours
Topic Both Male Female Neither Other Both Male Female Neither Other Pairs per query
Actors and Filmmakers 51.4 12.5 10.4 21.3 4.4 51.9 12.5 10.9 20.7 4.0 0.848
Computer science 62.9 8.4 9.2 17.5 2.0 62.6 9.0 8.6 17.6 2.1 1.097
Former countries 43.3 10.1 11.3 30.7 4.7 43.7 11.0 10.6 29.9 4.9 0.825
Human rights 62.5 9.5 13.1 10.2 4.7 64.2 10.9 9.9 9.6 5.4 1.065
Literature 54.8 8.5 8.3 25.5 2.9 54.7 8.4 8.3 25.6 3.0 1.055
Medicine 17.7 9.3 8.8 58.6 5.6 17.6 9.8 9.3 57.6 5.7 0.381
Netherlands 57.8 10.0 8.1 22.5 1.6 57.4 8.9 9.4 22.9 1.4 1.331
Sociology 64.6 6.8 7.8 19.2 1.6 64.5 7.0 7.2 19.3 1.9 1.072
Southeast Asia 44.4 13.3 17.4 22.4 2.5 43.5 14.7 17.3 21.6 2.9 1.319
All topics 55.1 9.3 9.2 23.3 3.0 55.2 9.5 9.1 23.1 3.1
Table 9: Replies of the unadapted system and ofOursunder the in-dataset protocol. For each topic: the share (%) of
each reply, and the preference pairs built per training query.Bothis the accuracy;MaleandFemale, a reply of that
candidate;Neitherincludes Insufficient information to decide ;Other, a reply that matches no option and
no candidate.
Unadapted Ours
Topic Both Male Female Neither Other Both Male Female Neither Other Pairs per query
Actors and Filmmakers 92.1 5.8 2.0 0.0 0.1 92.5 4.0 3.3 0.0 0.2 0.848
Computer science 99.4 0.2 0.2 0.0 0.2 99.4 0.1 0.2 0.2 0.1 1.071
Former countries 60.7 10.3 9.5 18.2 1.3 59.8 8.4 9.1 20.5 2.2 0.835
Human rights 90.3 2.6 3.3 1.1 2.7 90.3 2.3 3.0 0.9 3.5 1.065
Literature 98.9 0.4 0.7 0.0 0.0 98.9 0.4 0.7 0.0 0.0 1.091
Medicine 93.5 4.8 1.4 0.1 0.2 94.6 3.6 1.4 0.1 0.2 0.381
Netherlands 92.2 2.7 3.2 0.2 1.7 91.8 3.1 2.9 0.3 2.0 1.356
Sociology 92.1 5.5 1.5 0.2 0.7 93.9 4.3 1.0 0.2 0.7 1.064
Southeast Asia 91.2 1.8 1.3 0.7 5.0 92.3 1.8 1.5 0.5 4.0 1.321
All topics 89.9 3.8 2.6 2.4 1.3 90.2 3.1 2.6 2.6 1.4
Table 10: Replies of the unadapted system and ofOursunder the cross-dataset protocol. For each topic: the share
(%) of each reply, and the preference pairs built per training query. The columns are defined as in Table 9.
18

bge-reranker-v2-m3 jina-reranker-v2-base-multilingual
Topic Unadapted Adapted Unadapted Adapted
Actors and Filmmakers+0.0211 +0.0165 +0.0317 +0.0308
Computer science−0.0072 +0.0043−0.0096−0.0095
Former countries−0.0121 +0.0041 +0.0030−0.0033
Human rights−0.0356 +0.0104−0.0287−0.0174
Literature+0.0024 +0.0010 +0.0057 +0.0041
Medicine+0.0046 +0.0058 +0.0145−0.0031
Netherlands+0.0194−0.0051 +0.0070−0.0021
Sociology−0.0092−0.0019−0.0087−0.0007
Southeast Asia−0.0409−0.0255−0.0348−0.0258
|GD|0.0169 0.0083 0.0160 0.0108
Accuracy 55.1 55.2 52.9 52.8
Table 11: GDtunder the in-dataset protocol with each reranker, unadapted and adapted. Adapted
bge-reranker-v2-m3isOurs.
Meta-Llama-3-8B-Instruct Qwen2.5-7B-Instruct Meta-Llama-3-70B-Instruct
Topic Unadapted Adapted Unadapted Adapted Unadapted Adapted
Actors and Filmmakers+0.0211 +0.0165 +0.0071 +0.0066−0.0176−0.0128
Computer science−0.0072 +0.0043−0.0066−0.0065−0.0513−0.0492
Former countries−0.0121 +0.0041−0.0157−0.0075−0.0289 +0.0119
Human rights−0.0356 +0.0104−0.0217−0.0135−0.0482−0.0122
Literature+0.0024 +0.0010−0.0017−0.0013−0.0234−0.0129
Medicine+0.0046 +0.0058 +0.0022 +0.0017−0.0224−0.0176
Netherlands+0.0194−0.0051−0.0133−0.0132−0.0274−0.0243
Sociology−0.0092−0.0019−0.0071 +0.0002−0.0435−0.0332
Southeast Asia−0.0409−0.0255−0.0070 +0.0060−0.0087−0.0077
|GD|0.0169 0.0083 0.0092 0.0063 0.0302 0.0202
Accuracy 55.1 55.2 48.1 47.7 57.3 57.4
Table 12: GDtunder the in-dataset protocol with each generator, unadapted and adapted. Adapted under
Meta-Llama-3-8B-InstructisOurs.
19

In-dataset (RoMQA) Cross-dataset (TREC)
System|GD|Accuracy Male-fav. Female-fav.|GD|Accuracy Male-fav. Female-fav.
Unadapted 0.0169 55.1 – – 0.0156 89.9 – –
Ours, one reranker 0.0175 55.5 1/45/50.0135 90.1 4/6 0/3
RD per topic 0.0118 54.84/44/5 0.0104 90.26/6 2/3
SD per topic 0.0170 55.0 0/4 3/5 0.0165 89.4 2/6 1/3
Ours0.008355.2 3/45/5 0.009490.26/6 2/3
Table 13:Ourswith one reranker shared by all topics, and the systems that treat each topic separately, under both
protocols. Full version of Table 2.Ours, one rerankermoves most topics toward the male candidate under the
in-dataset protocol and toward the female candidate under the cross-dataset protocol.
−.04 0 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted
Ours, one rerankerRD per topic
SD per topicOurs
(a) In-dataset (RoMQA)
−.04 −.02 0 +.02 +.04Actors and Filmmakers
Computer science
Former countries
Human rights
Literature
Medicine
Netherlands
Sociology
Southeast AsiaUnadapted
Ours, one rerankerRD per topic
SD per topicOurs (b) Cross-dataset (TREC)
Figure 8:GD tof Table 2 under both protocols. Full version of Figure 5.Oursis the closest to zero overall.
Topic Unadapted Ours, one reranker RD per topic SD per topic Ours
In-dataset (RoMQA)
Actors and Filmmakers+0.0211+0.0247+0.0192+0.0219+0.0165
Computer science−0.0072−0.0043+0.0013−0.0061+0.0043
Former countries−0.0121−0.0088−0.0037−0.0098+0.0041
Human rights−0.0356−0.0339−0.0378−0.0391+0.0104
Literature+0.0024+0.0098−0.0002+0.0036+0.0010
Medicine+0.0046+0.0089−0.0002+0.0060+0.0058
Netherlands+0.0194+0.0182+0.0068+0.0250−0.0051
Sociology−0.0092−0.0091−0.0042−0.0096−0.0019
Southeast Asia−0.0409−0.0395−0.0325−0.0322−0.0255
|GD|0.0169 0.0175 0.0118 0.01700.0083
Accuracy 55.1 55.5 54.8 55.0 55.2
Cross-dataset (TREC)
Actors and Filmmakers+0.0385+0.0156+0.0139+0.0389+0.0074
Computer science+0.0002−0.0008−0.0005+0.0008−0.0005
Former countries+0.0083−0.0103−0.0012+0.0122−0.0076
Human rights−0.0073−0.0086−0.0032−0.0072−0.0075
Literature−0.0031−0.0068−0.0009−0.0031−0.0029
Medicine+0.0343+0.0242+0.0330+0.0341+0.0226
Netherlands−0.0049−0.0071−0.0055−0.0053+0.0018
Sociology+0.0396+0.0407+0.0334+0.0430+0.0326
Southeast Asia+0.0043+0.0074+0.0023+0.0039+0.0023
|GD|0.0156 0.0135 0.0104 0.01650.0094
Accuracy 89.9 90.1 90.2 89.4 90.2
Table 14: GDtof every system of Table 2 under both protocols. The rows |GD| and Accuracy give, for each
column, the mean absolute per-topic group disparity and the accuracy that Table 13 reports. Bold marks, in each
row, the value closest to zero.
20

In-dataset (RoMQA) Cross-dataset (TREC)
Variant|GD|Accuracy Male-fav. Female-fav.|GD|Accuracy Male-fav. Female-fav.
Unadapted 0.0169 55.1 – – 0.0156 89.9 – –
Ours0.008355.23/4 5/5 0.009490.26/6 2/3
one direction 0.0176 54.8 0/45/50.0102 90.36/60/3
without anchor 0.0142 55.0 0/4 4/5 0.0112 90.8 5/6 1/3
Table 15: Ablation under both protocols. Full version of Table 3.
Variants of Table 3
Topic Unadapted Ours one direction without anchor
In-dataset (RoMQA)
Actors and Filmmakers+0.0211+0.0165+0.0495+0.0223
Computer science−0.0072+0.0043+0.0043−0.0010
Former countries−0.0121+0.0041+0.0041+0.0163
Human rights−0.0356+0.0104+0.0104−0.0035
Literature+0.0024+0.0010+0.0026+0.0041
Medicine+0.0046+0.0058+0.0084+0.0080
Netherlands+0.0194−0.0051+0.0518+0.0203
Sociology−0.0092−0.0019−0.0019−0.0077
Southeast Asia−0.0409−0.0255−0.0255−0.0452
|GD|0.01690.00830.0176 0.0142
Accuracy 55.1 55.2 54.8 55.0
Cross-dataset (TREC)
Actors and Filmmakers+0.0385+0.0074+0.0074+0.0174
Computer science+0.0002−0.0005−0.0005−0.0006
Former countries+0.0083−0.0076−0.0076−0.0076
Human rights−0.0073−0.0075−0.0103−0.0075
Literature−0.0031−0.0029−0.0031−0.0033
Medicine+0.0343+0.0226+0.0226+0.0226
Netherlands−0.0049+0.0018−0.0050+0.0053
Sociology+0.0396+0.0326+0.0326+0.0326
Southeast Asia+0.0043+0.0023+0.0023+0.0044
|GD|0.01560.00940.0102 0.0112
Accuracy 89.9 90.2 90.3 90.8
Table 16: GDtof every variant of Table 3, besideUnadaptedandOurs, under both protocols. The rows |GD| and
Accuracy give, for each column, the mean absolute per-topic group disparity and the accuracy that Table 15 reports.
Bold marks, in each row, the value closest to zero.
21
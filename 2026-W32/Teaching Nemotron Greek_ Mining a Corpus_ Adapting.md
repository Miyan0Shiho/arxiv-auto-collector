# Teaching Nemotron Greek: Mining a Corpus, Adapting Retrieval, and Grounding Generation for Modern Greek across Specialist Domains

**Authors**: Ayoub Kirouane, Christos Petrocheilos

**Published**: 2026-08-05 17:56:40

**PDF URL**: [https://arxiv.org/pdf/2608.05138v1](https://arxiv.org/pdf/2608.05138v1)

## Abstract
Modern Greek is absent from NVIDIA's Nemotron retrieval models and from major multilingual retrieval benchmarks, despite being important for retrieval-augmented generation (RAG) in legal, energy, financial, and medical applications. We present an end-to-end adaptation of the Nemotron retrieval stack for Modern Greek, including corpus mining, synthetic supervision, retrieval model training, reranker adaptation, reader fine-tuning, and a new benchmark called HERA. Our study shows that a parameter-free BM25 baseline outperforms several off-the-shelf multilingual dense retrieval models on specialist Greek corpora. After fine-tuning on 65,773 Greek retrieval pairs, a Nemotron 1B embedder improves nDCG@10 from 0.362 to 0.835 and substantially outperforms its unadapted counterpart. The learned language competence transfers to general-domain Greek, although the advantage over BM25 remains domain-dependent. We further adapt a cross-encoder reranker and demonstrate consistent improvements across specialist domains. Finally, we LoRA-tune a Nemotron 30B-A3B mixture-of-experts reader for grounded generation, increasing judged answer correctness from 29.4% to 66.9% while significantly improving faithfulness and citation quality. We also introduce HERA, the first large-scale Greek benchmark for retrieval-augmented generation, and release our adapted models and benchmark to support future research on Greek-language RAG systems.

## Full Text


<!-- PDF content starts -->

 
T eac hing Nemotron Greek: Mining a Corpus,
A dapting Retriev al, and Grounding Generation
for Mo dern Greek across Sp ecialist Domains
A y oub Kirouane1 Christos P etro c heilos1
1Sophea AI, KIEFER SA, A thens, Greece
{a.kirouane, c.p etro c heilos}@kiefer.gr
mo dels@sophea.ai
Mo dels & Benc hmark: h uggingface.co/KIEFERSA  A ugust 2026
Abstract
Mo dern Greek is missing from the supp orted-language list of NVIDIA’s Nemotron retriev al mo dels and 
from ev ery ma jor m ultilingual retriev al b enc hmark, y et Greek legal, energy , financial and clinical do cumen ts 
are exactly the long, jargon-dense text that retriev al-augmen ted generation is mean t to serv e. W e adapted 
the Nemotron family to Greek end to end, mining the corpus, training the retriev al stac k, syn thesising 
the reader sup ervision and building an ev aluation b enc hmark, then measured what eac h stage actually 
buys. Building the data surfaced findings of its o wn: there is essen tially no nativ e Greek instruction 
data to train on , so an y p o ol at useful scale m ust b e translated, and Greek do cumen t c h unks are long 
enough that a max_len  of 512 silen tly truncates 87%  of training pairs. The mo delling headline is equally 
uncomfortable: on all fiv e of our domains a BM25 lexical baseline, with no learned parameters, 
outscores ev ery off-the-shelf m ultilingual em b edder w e tested , including an 8B one. Fine-tuning 
a 1B em b edder on 65,773 Greek retriev al pairs lifts nDCG@10 from 0.362 to 0.835 , and the comp etence 
it gains generalises: on a general-domain Greek corpus that neither mo del w as trained on, the adapted 
em b edder b eats its o wn unadapted base b y +0.399 . Its adv an tage o v er BM25 do es not generalise in 
the same w a y . W e lead in the domains w e adapted on and lose on general Greek, and w e rep ort b oth. 
A cross-enco der whose off-the-shelf con tribution is inconsisten t across domains b ecomes a reliable gain 
once adapted. Finally , LoRA-tuning a 30B-A3B mixture-of-exp erts mo del as a grounded reader raises 
judged answ er correctness from 29.4% to 66.9% . W e also rep ort the four w a ys our o wn instrumen ts lied 
to us along the w a y , including t w o claims of ours that a second, larger ev aluation did not repro duce. W e 
release the mo dels and HERA, a new Greek retriev al-augmen ted generation b enc hmark, at h uggingface.co/
KIEFERSA .
1. Greek falls off the map
Dense retriev al is supp osed to ha v e sup erseded lexical 
searc h. That b elief is load-b earing: it is wh y pro duction 
RA G stac ks ship a m ultilingual em b edder and drop 
BM25 ( Rob ertson and Zaragoza, 2009 )  en tirely .
It do es not surviv e con tact with Greek. Greek is ab -
sen t from the supp orted-language lists of the Nemotron 
retriev al mo dels, and from BEIR ( Thakur et al., 2021 )  
and MIRA CL ( Zhang et al., 2023 ) , the b enc hmarks that 
shap e the field’s sense of what m ultilingual retriev al can 
do. What happ ens to a language that is out of distrib -
ution for b oth the mo dels and the y ardstic ks is simply 
not measured.
So w e measured it. But first w e had to build the data, 
and that turned out to b e half the story .2. What already exists
Greek is not un touc hed. Meltemi-7B  ( V ouk outis et 
al., 2024 )  and Llama-Krikri-8B  ( Roussis et al., 2025 )  
adapt op en mo dels to Greek through con tin ued pretrain -
ing and instruction tuning, and GR-NLP-TOOLKIT  
( Loukas et al., 2025 )  supplies Greek tok enisation, lem -
matisation and tagging. That w ork targets general Greek 
generation; ours targets retriev al and grounded reading, 
so the artifacts do not o v erlap. W e do not compare 
against those mo dels, and that is a real gap rather than 
a scoping decision: a Greek-adapted reader is the con trol 
that w ould separate our recip e from our c hoice of base 
mo del.
Greek grounded reading is lik ewise not new. Beleb ele  
( Bandarkar et al., 2024 )  co v ers Greek passage-grounded 
m ultiple c hoice and is one of the b enc hmarks in T able  7 , 
1

and X QuAD  ( Artetxe et al., 2020 )  pro vides Greek 
extractiv e items o v er a gold passage. What w e could not 
find, and what Section  4  builds, is a Greek b enc hmark 
carrying multi-p assage  con text, mined distractors, cita -
tion targets and unansw erable items. That is the narro w 
claim, and it is the only one w e mak e.
Retriev al is where the absence is real. BEIR ( Thakur 
et al., 2021 )  is English-only and MIRA CL ( Zhang et 
al., 2023 )  excludes Greek, so the m ultilingual em b edders 
w e test w ere trained and measured without it. BEIR’s 
o wn conclusion, that BM25 is a robust out-of-domain 
baseline, is what Section  5  repro duces in a new language 
with the parameter axis held fixed; w e claim the mea -
suremen t, not the phenomenon. Multilingual E5 ( W ang 
et al., 2024 )  is the ob vious further baseline and w e did 
not run it. F or RA G ev aluation, RA GAS ( Es et al., 2024 )  
and ARES ( Saad-F alcon et al., 2024 )  supply metric 
framew orks rather than data, whic h is the half w e had 
to build.
3. Mining the corpus
W e started from 407,053 ra w (query, document, label, 
domain)  pairs and ended with 65,773 clean p er-query 
records across fiv e domains: energy , legal, finance, med -
ical and a general foundation slice. The reduction is 
not aggressiv e filtering for its o wn sak e; most of it is 
regrouping flat pairs b y query . But one clean up step is 
w orth naming, b ecause mined corp ora reliably need it. 
W e call it p ositiv e-wins : a do cumen t mined as a hard 
negativ e for a query , whic h is in fact that query’s o wn 
p ositiv e, gets dropp ed. Negativ e mining pro duces this 
constan tly , and training on it teac hes the mo del to push 
apart t w o things it should pull together.
The queries are syn thetic, and how  they w ere syn -
thesised matters more than the fact that they w ere. 
Left to itself, an instruct mo del writes short, k eyw ord-
lik e queries, whic h lo ok nothing lik e what real users 
t yp e: long, formal, fact-seeking questions after a sp e -
cific decision n um b er, date, capacit y or amoun t. A 
retriev er trained on k eyw ord queries scores w ell on syn -
thetic ev aluations and then stum bles in pro duction, so 
the generation w as constrained rather than free. T w o 
generators wrote roughly three questions p er c h unk: 
Nemotron-3-Ultra-550B  for the general foundation 
slice and GLM-5.2  for the four st yle-matc hed domain 
slices. Three con trols applied throughout: eac h call is 
few-shot conditioned on real pro duction user queries so 
the register transfers; a strict grounding rule requires the 
sp ecifics a query asks for to app ear in its o wn c h unk, 
audited at 97–99% with few-shot exemplar leakage near 
zero; and difficult y is v aried delib erately rather than left to c hance, whic h Section  4  pushes furthest with an 
explicit L1–L5 ladder.
Lab elling rests on one b et: the c h unk a query w as 
written from is that query’s p ositiv e . No h uman 
marks relev ance an ywhere in the corpus. The b et is 
c heap to state and easy to c hec k, and the audit ab o v e 
is what c hec ks it: if 97–99% of queries ask for sp ecifics 
that app ear in their o wn c h unk, then the c h unk answ ers 
the query b y construction.
Negativ es are mined, and the miner matters 
more than the sampler.  A rerank er only learns from 
con trast, so negativ es are the dominan t qualit y lev er. 
An off-the-shelf Qw en3-Em b edding-8B em b eds queries 
and passages asymmetrically , the query side carrying 
a retriev al instruction and the do cumen t side plain, 
and p er query w e tak e a cosine-kNN windo w o v er that 
query’s own-domain  p o ol, so an energy query gets energy 
distractors rather than a trivially wrong legal one. T w o 
guards define the windo w: skip the top few ranks, b e -
cause those are to o lik ely to b e gen uinely relev an t and 
w ould b ecome false negativ es, and drop an ything ab o v e 
a similarit y cap, b ecause those are near-duplicates of the 
p ositiv e. Around six negativ es p er query surviv e. Mining 
the same queries with BM25 instead of a dense em b ed -
der pro duced measurably w orse negativ es, so the lexical 
signal that mak es BM25 a strong r etriever  on Greek 
( Section  5 ) do es not mak e it a go o d ne gative miner . The 
result is separable but hard: on a sampled audit, query-
to-p ositiv e similarit y a v eraged 0.687 against 0.474 to the 
mined negativ es, a margin of 0.213, with no query w eakly 
grounded to its o wn p ositiv e.
St yle-matc hing is w orth a lot, and syn thetic 
ev aluation still flatters.  Both effects w ere measured 
directly on pro duction queries. Real users write long, 
m ulti-part questions a v eraging ab out 176 c haracters, 
roughly a quarter of them exact iden tifier lo okups, 
against ab out 68 c haracters for naiv e syn thetic queries; 
few-shot st yle-matc hing the syn thetic set to real ones 
double d  the reranking lift o v er BM25, from + 0 . 0 4 0  to 
+ 0 . 0 8 1  nDCG@10. The sob ering half is an in-house 
observ ation w e rep ort as suc h: on real user queries the 
margin o v er BM25 w as roughly + 0 . 0 4 , against + 0 . 0 8 0  
on our syn thetic held-out set, so under half the lift 
surviv ed con tact with pro duction traffic. That compar -
ison is unpublished, and w e attac h no query coun t or 
lab elling proto col to it; it is an in ternal observ ation, not 
a measured result. Syn thesised queries ec ho their source 
passage’s v o cabulary and are therefore systematically 
easier than what users t yp e. Ev ery retriev al n um b er in 
this rep ort is measured on syn thetic held-out queries. 
They are sound for ranking systems against eac h other, 
2

whic h is what w e use them for, but they are an upp er 
b ound on pro duction b eha viour.
W e then tagged language b y Greek-script ratio (Greek 
c haracters o v er Greek plus Latin, computed after  strip -
ping co de blo c ks so that tec hnical passages are not 
misread as English), decon taminated train against the 
union of ev ery ev aluation query , deduplicated b y nor -
malised exact matc h, and split b y query with seed 42, 
v erifying zero o v erlap.
A max_len  of 512, the most common default in retriev al 
to oling, truncates roughly 87%  of our training pairs. 
This w as the single most consequen tial thing the data 
told us. Greek do cumen t c h unks are long: p ositiv es ha v e 
a median of 1,196 c haracters ( ≈ 1,034 tok ens), p95 of 
1,288 and a maxim um of 4,407, while queries are short 
at a median of 113 c haracters. Both mo dels are therefore 
trained at max_len  4,096. An y one repro ducing this at 
512 w ould train on truncated p ositiv es and conclude the 
metho d do es not w ork.
The corpus is delib erately bilingual, 87.1% Greek, 
with English retained at 12.9% rather than filtered 
out, b ecause co de-switc hed Greek/English is p erv asiv e 
in Greek tec hnical and legal writing. Both mo dels are 
trained bilingually in a single run, with no p er-language 
head or adapter.
3.1. Syn thesising a reader that cites
No gold answ ers exist for a retriev al corpus, so reader 
sup ervision had to b e man ufactured. W e built 40,000 
examples from the retriev al tr ain  split only , k eeping 
it disjoin t from ev ery retriev al test query . The teac her 
is Sophea-Titan-1 , our Greek-adapted 27B mo del (a 
LoRA on Qw en3.6-27B, released at h uggingface.co/
KIEFERSA/Sophea-Titan-1 ). It answ ers eac h query us -
ing only  the gold passage; if it cannot, the ro w b ecomes 
an absten tion example. The con text is then assem bled as 
the gold passage plus 𝑘 ∈ [ 2 , 5 ]  sh uffled hard negativ es, 
n um b ered, with the gold landing at a random p osition 
𝑝 , and the target is answer [p] , an answ er carrying an 
explicit citation.
The p oin t of that construction is that robustness is 
designe d in r ather than hop e d for . The distractor coun t 
v aries, gold p osition is sh uffled sp ecifically to coun ter -
act the lost-in-the-middle effect ( Liu et al., 2024 ) , and 
absten tion is an explicit training target rather than 
an emergen t b eha viour. Roughly 20% of examples are 
absten tions built from negativ es alone.
The audit caugh t our o wn judge.  A uditing the 
syn thesised corpus put grounded answ er faithfulness at 
95.2%, with 3 wrong citations in 30,903 and zero dupli -
cate queries. Absten tion lab el noise, ho w ev er, measured 
an alarming 20.4%. Before accepting that, w e ran a con trol in whic h the gold passage w as pr ovably  remo v ed, 
ro ws that m ust b e absten tions, and the same judge 
scored those at 23.5%. The con trol came in worse  than 
the real data, whic h means 20.4% w as mostly the judge’s 
o wn false-p ositiv e flo or rather than corpus noise. Real 
absten tion noise is b ounded near 5%.
W e also name three gaps the audit exp osed rather 
than smo othing o v er: all 30,903 grounded answ ers cite 
exactly one  c h unk, so there is no m ulti-do cumen t cita -
tion sup ervision; absten tion targets are dra wn from only 
t w o canned strings; and 8.7% of answ ers are ultra-terse.
3.2. Nativ e Greek data barely exists
There is essen tially no commercially licensed, h u -
man-written Greek instruction data at scale.  The 
v olume that exists is mac hine-translated or templated: 
the Greek slice of the A y a collection ( Singh et al., 2024 ; 
Üstün et al., 2024 )  runs to millions of ro ws that w a y , 
while its h uman-annotated Greek p ortion is on the order 
of h undreds, and the op en assistan t corp ora w e surv ey ed 
carry no Greek at all. An ything at useful scale therefore 
has to b e translated.
Our Greek instruction p o ol is 296,034 con v ersations, 
of whic h the filtered, deduplicated split used do wnstream 
is 206,909 training ro ws (the figure that reapp ears in 
Section  4  and Section  9 ), and assem bling it pro duced the 
scarcit y finding that constrains ev erything do wnstream. 
A t this scale a Greek instruction p o ol is ne c essarily  
translation-hea vy , whic h carries a translationese risk 
that sup ervised fine-tuning will shap e directly in to 
mo del b eha viour. This is not a solv ed problem and w e 
do not claim to ha v e solv ed it. It is the honest reason 
w e cannot cleanly separate a base-mo del ceiling from a 
data ceiling.
Ev erything in the p o ol passes the same filter: non-
empt y turns, Greek-ratio thresholds computed after 
co de-stripping, length b ounds, degeneracy detection, 
exact-signature deduplication, and MinHash near-dupli -
cate remo v al at 0.8 b oth within the p o ol and against it. 
That filter retains 92.1% of the augmen tation set.
4. A Greek b enc hmark, b ecause none ex -
isted
Greek grounded reading has b een ev aluated only 
through translated single-passage resources ( Section  2 ), 
and no Greek b enc hmark w e could find puts a reader 
in fron t of man y passages at once, so the ev aluation 
had to b e built b efore an ything could b e measured. W e 
built HERA  (Hellenic Retriev al-A ugmen ted), a 4,946-
item Greek Wikip edia long-retriev al b enc hmark with ci -
tations ( h uggingface.co/datasets/KIEFERSA/HERA ), 
3

T able  1: The Greek Wikip edia long-retriev al b enc hmark. 
Source is Greek Wikip edia under CC-BY-SA-4.0, released with 
attribution and revision iden tifiers.
prop ert y v alue
items 4,946
answ erable 3,712
unansw erable 1,234 (25%)
m ulti-hop 1,145 (23%)
difficult y L1 factoid to L5 syn thesis
con text 8 to 40 passages, 21.2 mean
construction Qw en3.5-122B generates, GLM-5.2 
v erifies
h uman pass sp ot-c hec k of random samples
decon tamination 13-gram c hec k, zero  o v erlap
summarised in T able  1 , and it is the y ardstic k b ehind 
ev ery reader n um b er in this rep ort. App endix  C  defines 
the metric set it is scored with.
T w o design c hoices matter. First, the difficult y ladder 
and the delib erate 25% share of unanswer able  items 
mean the b enc hmark measures absten tion as a first-class 
skill, not as an afterthough t. Most RA G b enc hmarks 
score only whether an answ erable question w as answ ered 
w ell, whic h cannot detect a mo del that confiden tly in -
v en ts an answ er when the con text do es not con tain one.
Second, HERA is built in t w o stages. Qw en3.5-122B  
generates the question, answ er and citation; GLM-5.2  
then indep enden tly v erifies that the item is answ erable 
only from its source passage, correct, and w ell formed. 
W e pic k ed Qw en3.5-122B for the generator role b ecause 
its Greek is mark edly fluen t for a general m ultilingual 
mo del, whic h is the prop ert y that matters most here: a 
b enc hmark whose questions and reference answ ers read 
as translationese w ould test the wrong thing, and Sec -
tion  3.2  has already established ho w hard nativ e Greek 
text is to come b y . W e rep ort that as the qualitativ e 
basis for the c hoice, not as a measured result. Only 
cross-v alidated items surviv e, and a random sample w as 
additionally sp ot-c hec k ed b y hand. Exhaustiv e h uman 
review of 4,946 long-con text items w as out of scop e, 
whic h is the reason the second mo del stage exists at all. 
Hard distractors are mined separately , b y h ybrid BM25 
plus dense retriev al fused with RRF and then rerank ed, 
so the con text noise is the highest-scoring non-gold 
passage rather than a random one.
A confound w e ha v e to declare.  Our reader judge is a 
serv ed Qw en3.5-122B-A10B-FP8, the same family as the 
mo del that authored HERA’s items. The judge is there -
fore grading text from its o wn family throughout, and 
family preference is a do cumen ted failure mo de of LLM-
as-judge ( Zheng et al., 2023 ) . T w o things push against 
it: no item surviv es without indep enden t v erification b y 
GLM-5.2, a differen t family , and the sp ot-c hec k sampled the corpus b y hand. Neither remo v es the effect, and w e 
did not measure it.
What it do es not  touc h is w orth stating, b ecause 
the exp osure is unev en. Only judge d answer c orr e ctness  
is scored against a mo del-authored reference answ er. 
F aithfulness is scored against the retriev ed con text, 
the citation metrics against the gold passage index, 
and absten tion against items whose gold passage w as 
remo v ed b y construction. Those three are anc hored to 
the b enc hmark’s structure rather than to an y mo del’s 
phrasing, and they carry the largest gains. Answ er cor -
rectness is the most exp osed n um b er here.
On h uman v erification w e claim only what w e can 
supp ort. Lab els are t w o strong mo dels’ consensus plus 
a hand sp ot-c hec k of random samples. W e did not log 
ho w man y items w ere c hec k ed, so w e rep ort no co v erage 
figure and do not presen t it as systematic v alidation. 
A sized, logged h uman audit remains the most v aluable 
single addition to this ev aluation.
One thing w e got wrong.  The generator o v er-pro -
visioned candidates three-to-one against a 5,000-item 
target, exp ecting strict v erification to reject man y . W e 
failed to log the realised acceptance rate.  W e can 
b ound it only at ≥ 1/3 and cannot sa y ho w strict the 
gate actually w as. It is the w eak est link in our reader 
ev aluation, and w e w ould rather flag it than infer a 
n um b er.
Decon tamination, at least, w as v erified rather than 
assumed. A 13-gram o v erlap c hec k b et w een the b enc h -
mark (77,411 distinct normalised 13-grams) and the 
full 206,909-ro w instruction p o ol returns zero  o v erlap -
ping ro ws. Normalisation strips accen ts first, so Greek 
orthographic v ariation cannot defeat the c hec k. The 
b enc hmark is also out of corpus for the reader, whose 
training data comes from energy , legal, finance and 
medical do cumen ts, so it measures transfer rather than 
in-domain fit.
Ev ery retriev al n um b er b elo w is macro nDCG@10 o v er 
5,830 held-out queries, and ev ery comparison carries a 
paired b o otstrap in terv al. F or systems 𝐴  and 𝐵  scored 
p er query on a query set 𝑄 , w e dra w 𝑄𝑏  from 𝑄  with 
replacemen t and recompute the mean difference,
Δ ̂𝑏 =1
| 𝑄𝑏 |∑
𝑞 ∈ 𝑄𝑏( 𝑠𝐴 ( 𝑞 ) − 𝑠𝐵 ( 𝑞 ) ) ,
for 𝑏 = 1 , … , 𝐵  with 𝐵 = 1 0 , 0 0 0  and seed 42; the 95% 
in terv al is the 2.5th and 97.5th p ercen tiles of { Δ ̂𝑏 } . 
The pairing matters: 𝐴  and 𝐵  are differenced on the 
same  query b efore a v eraging, whic h remo v es p er-query 
difficult y from the v ariance and is wh y in terv als as tigh t 
as [ + 0 . 0 7 2 , + 0 . 0 8 9 ]  are attainable at this sample size.
4

Figure 1: Greek retriev al, four domains plus the macro mean, 5,830 held-out queries. A parameter-free lexical baseline b eats a 
state-of-the-art 8B m ultilingual em b edder on three of the four sets and loses one. A dapting a 1B mo del b eats b oth.
T able  2: The n um b ers b ehind Figure  1  and Figure  2 . F ull held-out IR o v er 5,830 queries, corpus = all p ositiv es. Δ  is our adapted 
1B against the unadapted base, with a paired b o otstrap in terv al o v er queries; no in terv al approac hes zero. The three righ t-
hand columns are the off-the-shelf Qw en3-Em b edding family , unadapted. Medical is not a ro w here: it has its o wn corpus and 
so cannot share this table’s proto col, and is rep orted separately in Section 6 .
set queries base ours 1B Δ  [95% CI] BM25 Qw en 8B Qw en 4B Qw en 0.6B
v al 2,000 0.3823 0.8755 + 0 . 4 9 3 3  [.475,.514] 0.7694 0.7402 0.7383 0.6579
energy 1,122 0.3454 0.7937 + 0 . 4 4 8 3  [.423,.475] 0.7505 0.5804 0.5786 0.5089
legal 1,769 0.5257 0.9497 + 0 . 4 2 4 1  [.401,.444] 0.8840 0.9096 0.9187 0.8629
finance 939 0.1935 0.7218 + 0 . 5 2 8 3  [.503,.555] 0.6246 0.4902 0.5015 0.4265
macro o v er sets 5,830 0.3617 0.8352 + 0 . 4 7 3 5 0.7571 0.6801 0.6843 0.6140
macro o v er queries 5,830 0.3883 0.8575 + 0 . 4 6 9 2 0.7772 0.7206 0.7242 0.6542
5. The uncomfortable baseline
Figure 1  is the result w e did not exp ect.
BM25 scores 0.757. Qw en3-Em b edding-8B 
scores 0.680. The unadapted Nemotron em b ed -
der scores 0.362.
A lexical baseline with no learned parameters, no GPU 
and no training b eats a state-of-the-art m ultilingual 
dense em b edder on Greek. It wins on energy ( + 0 . 1 7 0 ), 
finance ( + 0 . 1 3 4 ) and general v alidation ( + 0 . 0 2 9 ), and 
loses only legal ( − 0 . 0 2 6 ). W e do not read that loss as 
legal Greek b eing closer to a m ultilingual mo del’s distri -
bution. Legal is the easiest set for every  system w e ran, 
and b y eac h one’s o wn standard: BM25 p eaks there at 
0.8840, the unadapted Nemotron at 0.5257, our adapted 
1B at 0.9497. A domain that is uniformly easy is not 
evidence ab out an y one family . Greek legal queries turn 
on article n um b ers, dates and statute references, whic h 
anc hor lexically whatev er is doing the ranking.
The practical reading is blun t: any Gr e ek RA G system 
that r eplac e d BM25 with an off-the-shelf multilingual 
emb e dder made its r etrieval worse on these domains.  The 
scop e of that sen tence is delib erate. Ev ery n um b er in this 
section is in-domain, and the out-of-domain coun terpart 
Figure  2: The off-the-shelf em b edder family across a 13 ×  
parameter range, in-domain. Ev ery rung sits b elo w BM25; 4B 
and 8B differ b y − 0 . 0 0 4 . Out-of-domain the same three rungs 
separate cleanly ( T able 4 ).
b elo w is w eak er: there BM25 sta ys ahead of the 0.6B and 
4B rungs, but is no longer separable from the 8B.
Nor do es scale rescue it ( Figure  2 ). Within the same 
off-the-shelf family ( Zhang et al., 2025 ) , same training 
and same proto col, differing only in size, 8B and 4B are 
indistinguishable (0.6801 vs. 0.6843, a gap of − 0 . 0 0 4 ), 
and dropping all the w a y to 0.6B costs only 0.070 
(0.6140). The en tire family sits b elo w BM25 across a 13 ×  
parameter range. On this corpus, doubling the parame -
5

Figure  3: T raining loss o v er one ep o c h, b oth stages, against the same recip e run on Qw en3-0.6B. Within a panel the t w o runs 
share ob jectiv e, corpus, sc hedule and logging in terv al. The Nemotron mo dels start mark edly higher and finish lo w er: Greek 
w as already supp orted b y the Qw en bac kb one b efore training, and out of distribution for Nemotron. Absolute v alues are not 
comparable across panels (InfoNCE against BCE), nor strictly across bac kb ones, whic h differ in tok enizer and parameterisation.
ter coun t buys nothing, b ecause the missing ingredien t 
is not capacit y . It is exp osure to the language.
Where that claim stops.  W e later ran the same three 
rungs out of domain, on the HERA retriev al trac k (the 
proto col is T able  4 ; w e giv e these three here b ecause that 
table is matc hed to the size of our  mo dels). Off domain 
the family is not flat at all. It is a clean monotone lad -
der, 0 . 5 2 6 8 → 0 . 6 1 8 2 → 0 . 6 5 0 4 , ev ery step significan t, 
+ 0 . 1 2 3 7  end to end across the same 13 ×  range, and 
the 8B rung finally catc hes BM25: + 0 . 0 0 6 2 , 95% CI 
[ − 0 . 0 0 5 7 , + 0 . 0 1 7 7 ] , not separable. So the flatness ab o v e 
is a prop ert y of this c orpus , not of Greek retriev al, and so 
is the clean sw eep o v er the whole family . Capacit y do es 
buy something on general Greek text; it b ough t nothing 
on sp ecialist Greek text, where exp osure w as the binding 
constrain t. W e state the narro w v ersion b ecause w e can 
only supp ort the narro w v ersion: one family , t w o query 
distributions.
6. A 1B mo del, adapted, wins
W e fully fine-tuned Nemotron-3-Em b ed-1B ( NVIDIA, 
2025a )  con trastiv ely with InfoNCE ( Oord et al., 2018 ) . 
F or a batc h of 𝐵  anc hors, query 𝑞𝑖  is scored b y cosine 
similarit y 𝑠  against a candidate set 𝒞︀𝑖  holding its o wn 
p ositiv e 𝑝+
𝑖 , ev ery other in-batc h p ositiv e ( Karpukhin et 
al., 2020 ) , and all 7 × 𝐵  mined hard negativ es, sev en 
p er query:
ℒ︀emb = −1
𝐵∑𝐵
𝑖 = 1log𝑒𝑠 ( 𝑞𝑖 , 𝑝+
𝑖 ) / 𝜏
∑𝑑 ∈ 𝒞︀𝑖𝑒𝑠 ( 𝑞𝑖 , 𝑑 ) / 𝜏
The denominator is the whole p oin t: it is where the 7 ×
𝐵  hard negativ es liv e, so the n um b er of negativ es, and 
therefore the qualit y of the gradien t, is set b y the lo gic al  
batc h rather than b y what fits in memory . GradCac he ( Gao et al., 2021 )  decouples the t w o, letting us hold a 
logical batc h of 256 at 4,096 tok ens on a single GPU. 
Both stages start w ell ab o v e the Qw en con trol and finish 
b elo w it ( Figure  3 ): Greek is inside that bac kb one’s 
distribution, outside Nemotron’s. F amiliarit y is not com -
p etence; it still lost to BM25.
Macro nDCG@10 go es from 0.362 to 0.835  ( T able  2 , 
whic h rep orts ev ery domain with its in terv al; App en -
dix  A  defines the metrics). Ev ery domain gains at least 
+ 0 . 4 2 , and the gain is largest exactly where the base 
w as w eak est: finance + 0 . 5 3 , starting from 0.19. A base 
score under 0.21 do es not mean degraded. It means non-
functional on Greek financial text.
Against BM25 the margin is + 0 . 0 8 0 , 95% CI 
[ + 0 . 0 7 2 , + 0 . 0 8 9 ] , paired b o otstrap o v er queries. The 
in terv al is no where near zero. A 1B mo del that has seen 
the language b eats b oth a parameter-free baseline and 
an 8B mo del that has not.
Medical.  Medical is ev aluated on its o wn set: 650 
queries o v er 24,812 passages, dra wn from the same public 
source as the rest of the medical corpus. It has its o wn 
corpus, whic h is wh y it is rep orted here rather than as a 
ro w of T able  2 . T able  3  is the result, and it is the medical 
n um b er w e rep ort an ywhere in this pap er.
T able  3: Medical, 650 queries o v er 24,812 passages. This is the 
medical ev aluation w e rep ort; it needs its o wn corpus, whic h is 
wh y it is not a ro w of T able 2 .
system nDCG@10 R@10
Nemotron-3-Em b ed-1B, base 0.1586 0.2292
ours 1B, adapted 0.6867 0.8815
BM25 (parameter-free) 0.6602 0.8554
Qw en3-Em b edding-8B 0.5936 0.7892
Qw en3-Em b edding-4B 0.6119 0.8046
Qw en3-Em b edding-0.6B 0.4698 0.6354
6

Figure  4: The same systems in domain and out of it. Lev els are 
not  comparable across the t w o ev aluations, whic h use differen t 
corp ora; the ordering is the p oin t. Our adapted 1B leads in 
domain and falls b ehind BM25 and t w o Qw en rungs outside it, 
while sta ying far ab o v e the base it w as built from. The three 
Qw en rungs share one colour and are told apart b y their lab els.
T w o things hold there and one do es not. A daptation 
holds, o v erwhelmingly : + 0 . 5 2 8  o v er the unadapted 
base, 95% CI [ + 0 . 4 9 6 , + 0 . 5 6 1 ] . The lexical baseline 
holds to o : BM25 at 0.6602 still b eats ev ery off-the-
shelf em b edder, including the 8B at 0.5936 and the 4B 
at 0.6119 ( + 0 . 0 4 8 , CI [ + 0 . 0 2 0 , + 0 . 0 7 6 ] ), so Section  5 ′ s 
cen tral finding surviv es here as w ell. What do es not 
hold is our margin o v er BM25. A t + 0 . 0 2 7 , 95% CI 
[ − 0 . 0 0 0 5 , + 0 . 0 5 4 ] , the in terv al touc hes zero : on this 
set w e cannot demonstrate an adv an tage o v er lexical 
searc h at this sample size.
Out of domain: what the adaptation actually 
transferred.  Ev erything ab o v e is measured on the do -
mains w e adapted on. T o see what surviv es outside them 
w e ran the same mo dels on the HERA retriev al trac k 
( Figure  4 , T able  4 ): 4,946 queries against 300,000 Greek 
Wikip edia passages, general-domain text that neither 
the base mo dels nor ours w ere trained on. T w o things 
separate cleanly , and they p oin t in opp osite directions.
The adaptation transfers as language comp e -
tence.  Out of corpus, our 1B b eats the unadapted 
Nemotron it w as built from b y + 0 . 3 9 9 , 95% CI 
[ + 0 . 3 8 7 , + 0 . 4 1 0 ] , and our 0.6B b eats its o wn base b y 
+ 0 . 0 4 9 , CI [ + 0 . 0 4 1 , + 0 . 0 5 6 ] . T eac hing a mo del Greek 
made it b etter at Greek retriev al generally , not only at 
ours. That is the cleanest evidence in this pap er for 
exp osur e, not c ap acity : the 1B base sat at 0.165, barely 
functional, and adaptation mo v ed it to within 0 . 0 0 7  of 
a 0.6B mo del whose bac kb one already sp ok e Greek.T able  4: Out of domain: the HERA retriev al trac k, 4,946 
queries o v er 300,000 Greek Wikip edia passages, a corpus none 
of these mo dels trained on. b ase  is the off-the-shelf c hec kp oin t. 
Compared at matc hed size, with BM25 as the parameter-free 
reference. Ev ery adapted-vs-unadapted gap here is significan t; 
b oth adapted mo dels sit b elow  BM25 alone, and the fusion of 
BM25 with our 1B sits ab o v e it.
system params nDCG@10 R@10
BM25 (parameter-
free)— 0.6442 0.7326
BM25 +  ours 1B , 
RRF1B 0.6715 0.7823
Qw en3-Em b-0.6B, 
base0.6B 0.5268 0.6189
Qw en3-Em b-0.6B, 
ours0.6B 0.5753 0.6728
Nemotron-3-
Em b-1B, base1B 0.1651 0.2105
Nemotron-3-
Em b-1B, ours1B 0.5637 0.6611
The win o v er lexical searc h do es not transfer.  
BM25 scores 0.6442 here and b eats b oth our adapted 
mo dels, b y + 0 . 0 8 1  o v er the 1B and + 0 . 0 6 9  o v er the 0.6B, 
and off-the-shelf Qw en3-Em b edding-4B b eats our 1B b y 
+ 0 . 0 5 5 . The ordering w e rep ort in-domain rev erses. W e 
tak e this as a measured b oundary rather than a ca v eat: 
the phrase acr oss sp e cialist domains  in our title is doing 
real w ork, and a reader deplo ying this em b edder on gen -
eral Greek should exp ect to lose to a lexical baseline. One 
prop ert y of b oth ev aluations limits ho w far this settles 
the question. HERA’s queries, lik e ours, are LLM-gener -
ated from their gold passages, whic h shares v o cabulary 
b et w een query and gold and flatters lexical matc hing on 
b oth  sides. That common bias cannot explain the rev er -
sal b et w een them, but it do es mean BM25 ′ s absolute 
standing is lik ely generous in eac h.
Figure  5: Dense and lexical retriev al fused b y recipro cal rank 
fusion. The hatc hed bar is not a third system: it is the t w o 
b eside it com bined. Neither comp onen t wins b oth regimes and 
the com bination wins b oth. The fusion w eigh t is c hosen on a 
held-out fold and scored on the other, so no bar is tuned on 
the queries it is scored on.
7

The righ t mo v e is to add BM25, not to replace 
it.  A baseline that b eats y ou is a comp onen t, not only 
a riv al, so w e fused the t w o with recipro cal rank fusion 
( Cormac k et al., 2009 ) , RRF ( 𝑑 ) = ∑𝑖𝑤𝑖 / ( 𝑘 + 𝑟𝑖 ( 𝑑 ) )  at 
𝑘 = 6 0  o v er eac h system’s top 100. With equal w eigh ts 
and nothing tuned, the fusion scores 0.6715 out of 
domain against BM25 ′ s 0.6442  ( Figure  5 ), a gain of 
+ 0 . 0 2 7 , 95% CI [ + 0 . 0 1 9 , + 0 . 0 3 5 ] , and it lifts Recall@10 
from 0.733 to 0.782. Letting a w eigh t b e c hosen on a held-
out fold and scoring only the other fold reac hes 0.6749, 
whic h is not separable from the un tuned v ersion, so the 
result do es not dep end on tuning at all.
The same holds in domain, where the dense mo del 
is the stronger comp onen t rather than the w eak er 
one. On a 2,500-query in-domain sample our 1B alone 
scores 0.6670 and BM25 0.5846; the fused system 
reac hes 0.6798, + 0 . 0 1 3  o v er the dense mo del alone, CI 
[ + 0 . 0 0 7 , + 0 . 0 1 9 ] , with the w eigh t again c hosen on a held-
out fold. The w eigh t mo v es in the direction y ou w ould 
exp ect, fa v ouring the dense side in domain and the 
lexical side outside it, whic h is a small piece of evidence 
that the gain is complemen tarit y rather than a fitting 
artifact. The same holds for our other em b edder: fusing 
the fine-tuned Qw en3-0.6B with BM25 gains + 0 . 0 3 4  out 
of domain, CI [ + 0 . 0 2 8 , + 0 . 0 4 1 ] , and + 0 . 0 1 6  in domain, 
CI [ + 0 . 0 1 0 , + 0 . 0 2 2 ] , o v er whic hev er of its t w o comp o -
nen ts is stronger there. Both of our em b edders b eha v e 
the same w a y , so the effect is a prop ert y of com bining 
dense with lexical retriev al on Greek rather than of one 
c hec kp oin t.
So the honest conclusion is not that a lexical 
baseline b eats our em b edder.  It is that neither 
system dominates, that they fail differen tly , and that the 
com bination b eats the b etter of the t w o in b oth  regimes: 
+ 0 . 0 1 3  where our mo del leads and + 0 . 0 2 7  where BM25 
do es. F or a Greek RA G system the practical recommen -
dation follo ws directly , and it is not the one w e exp ected 
when w e started: run b oth.
Our other fine-tuned em b edder, finally com -
pared.  W e also ship a fine-tuned Qw en3-0.6B em b edder. 
In-domain the t w o ha v e still nev er b een comparable, 
b ecause that run used a harder corpus with p ositiv es plus 
al l  negativ es as distractors. T able  4  is the first time they 
ha v e b een scored head to head under a single proto col, 
and out of domain the 0.6B is ahead: 0.5753 against 
0.5637, a difference of + 0 . 0 1 2 , 95% CI [ + 0 . 0 0 4 , + 0 . 0 2 0 ] . 
Separable, but small enough that the honest summary 
is that 40% few er parameters cost nothing here. Whic h 
mo del to deplo y is a question ab out y our corpus, not 
ab out their size.
It also shrinks.  The em b edding retains Matry oshka 
structure ( Kusupati et al., 2022 )  through fine-tuning 
Figure 6: Matry oshka truncation. Qualit y is flat do wn to 512 
dimensions and only collapses b elo w BM25 at 128.
( Figure  6 ). T runcated to 512 dimensions the index is 4
×  smaller and still scores 0.823, whic h is 98.5% of full 
qualit y and still + 0 . 0 6 6  o v er BM25.
7. The second stage, measured against its 
flo or
The rerank er is a cross-enco der ( Nogueira and Cho, 
2019 )  trained p oin t wise rather than con trastiv ely . Eac h 
canonical ro w expands in to lab elled pairs, 𝑦 = 1  for a 
query’s p ositiv e and 𝑦 = 0  for eac h of up to eigh t nega -
tiv es, giving 394,579 pairs 𝒫︀  at roughly 11% p ositiv e. 
A single relev ance logit 𝑓 ( 𝑞 , 𝑑 )  is fit with binary cross-
en trop y , writing 𝑦 ̂ = 𝜎 ( 𝑓 ( 𝑞 , 𝑑 ) ) :
ℒ︀rr = −1
| 𝒫︀ |∑
( 𝑞 , 𝑑 , 𝑦 ) ∈ 𝒫︀[ 𝑦 log 𝑦 ̂ + ( 1 − 𝑦 ) log ( 1 − 𝑦 ̂ ) ]
P oin t wise scoring is what lets the second stage see a 
query and do cumen t together, whic h the first stage nev er 
do es.
T able  5  giv es the p er-domain result, and App endix  B  
records what it w as measured on: the fixed first stage 
here is our fine-tuned Qw en3-0.6B em b edder o v er 750 
p o oled queries, so this flo or is not a ro w of T able  2 . 
Second stages are usually assumed to help. W e measured 
that assumption against the righ t con trol: the no-rerank 
flo or, i.e. the first stage’s o wn ranking.
On this ev aluation the off-the-shelf rerank er’s 
con tribution is statistically indistinguishable 
from not reranking at all  ( − 0 . 0 0 6 , 𝑝 = 0 . 5 2 ), while 
adapted, the same arc hitecture is w orth + 0 . 0 4 7  ( 𝑝 <
0 . 0 0 1 ). P er domain the off-the-shelf arm significan tly 
degrades one set (v alidation, − 0 . 0 5 0 , 𝑝 = 0 . 0 1 8 ) and 
significan tly helps another (legal, + 0 . 0 3 9 , 𝑝 = 0 . 0 0 3 ), 
whic h a v erages to nothing.
W e then measured it again, and one of those 
conclusions did not hold.  A second-stage result that 
rests on 750 queries is w orth re-running, so w e rep eated 
8

Figure  7: Reranking against the no-rerank flo or (dashed), first 
stage held fixed at top-50, 750 p o oled queries. Off-the-shelf: 
− 0 . 0 0 6 , CI [ − 0 . 0 2 6 , + 0 . 0 1 3 ] , 𝑝 = 0 . 5 2 . A dapted Nemotron: 
+ 0 . 0 4 7 , CI [ + 0 . 0 2 8 , + 0 . 0 6 6 ] . The t w o adapted arms are not 
separable: + 0 . 0 0 7 , CI [ − 0 . 0 0 6 , + 0 . 0 1 9 ] , 𝑝 = 0 . 2 9 .
T able  5: The n um b ers b ehind Figure  7 , p er domain. 150 
queries p er domain, 750 p o oled. flo or  is the first stage’s o wn 
ranking, b ase  the off-the-shelf Nemotron cross-enco der, ours  
our adapted Nemotron-1B, Qwen  our adapted Qw en3-0.6B. 
Bold marks the b est in eac h ro w. Ev ery ro w is scored against 
its o wn corpus, and the medical ro w uses the set of Section 6 .
set flo or base ours Qw en
v al 0.8438 0.7936 0.8740 0.8800
energy 0.7746 0.7412 0.7800 0.7841
legal 0.9264 0.9655 0.9692 0.9614
finance 0.6737 0.6975 0.7548 0.7557
medical 0.7729 0.7621 0.8485 0.8104
mean 0.7983 0.7920 0.8453 0.8383
the comparison on 2,580  queries o v er the same fiv e 
domains, with the same fixed first stage and the same 
top-50 candidate depth ( Figure  8 , T able  6 ). The t w o runs 
differ in more than one resp ect: query sample, p er-set 
corp ora, and the medical set of Section  6 . W e therefore 
do not attribute the difference b et w een them to an y 
single cause, and w e tak e as established only what holds 
in b oth.
What holds in b oth.  A daptation is what mak es 
a second stage w orth its cost. Our adapted Nemotron 
b eats the off-the-shelf cross-enco der it w as built from 
b y + 0 . 0 2 9 , 95% CI [ + 0 . 0 2 2 , + 0 . 0 3 7 ]  on the larger ev alu -
ation, consisten t with the + 0 . 0 5 3  gap b et w een the same 
t w o arms on the smaller one, CI [ + 0 . 0 3 5 , + 0 . 0 7 2 ] .
What do es not hold.  On the larger ev aluation the 
off-the-shelf rerank er do es  clear the flo or, b y + 0 . 0 3 2 , 
CI [ + 0 . 0 2 3 , + 0 . 0 4 1 ] . W e are no longer willing to sa y 
an unadapted cross-enco der con tributes nothing. What 
w e can sa y is that its con tribution is inc onsistent : it is 
significan t on t w o of fiv e sets, indistinguishable from zero 
on t w o more, and negativ e in p oin t estimate on energy . 
A v eraged o v er domains it is w orth ab out half of what 
adaptation buys, and on the smaller sample it w as w orth 
Figure  8: The second rerank er ev aluation, plotted as the gain 
o v er the no-rerank flo or with 95% paired b o otstrap in terv als. 
Absolute nDCG@10 spans 0.82 to 0.89 here, so the difference 
that the section argues ab out is the Δ , not the lev el. Compare 
Figure  7 , where the off-the-shelf arm did not clear the flo or 
at all.
nothing at all. A comp onen t whose b enefit dep ends this 
strongly on whic h slice y ou measure is not one to add 
on faith.
The metho dological p oin t is unc hanged, and is the 
reason w e could see an y of this. No ev aluation that 
omits the no-rerank flo or can distinguish these cases. 
Comparing rerank er A to rerank er B tells y ou whic h is 
b etter; only the flo or tells y ou whether either b elongs in 
the pip eline. It is also what told us our first answ er w as 
to o strong.
A 0.6B cross-enco der, adapted, nearly matc hes a 
1B one.  W e ran the same recip e on Qw en3-0.6B ( Zhang 
et al., 2025 ) , a differen t family at 60% of the parameters. 
On the 750-query ev aluation the t w o w ere not separable: 
0.8383 against our adapted Nemotron’s 0.8453, a paired 
difference of + 0 . 0 0 7 , 95% CI [ − 0 . 0 0 6 , + 0 . 0 1 9 ] , 𝑝 = 0 . 2 9 . 
On 2,580 queries they do separate, and the larger mo del 
is ahead b y + 0 . 0 2 0 , CI [ + 0 . 0 1 4 , + 0 . 0 2 7 ] . W e rep ort the 
separation b ecause w e found it, but the effect is small 
enough that the practical reading barely c hanges: 40% 
few er parameters costs ab out t w o nDCG p oin ts, and the 
T able  6: The same comparison on 2,580 queries: fiv e domains, 
first stage fixed, top-50 candidate depth. flo or  is the first stage’s 
o wn ranking. Δ  is against the flo or, paired b o otstrap o v er 
queries. Ev ery arm clears the flo or here, and the ordering is 
unam biguous.
system nDCG@10 Δ 95% CI
no-rerank 
flo or0.8248 —
Nemotron-1B, 
off the shelf0.8567 + 0 . 0 3 2 [ + 0 . 0 2 3 , + 0 . 0 4 1 ]
ours , 
Qw en-0.6B0.8659 + 0 . 0 4 1 [ + 0 . 0 3 2 , + 0 . 0 5 1 ]
ours , Nemotron-1B 0.8861 + 0 . 0 6 1 [ + 0 . 0 5 2 , + 0 . 0 7 1 ]
9

Figure  9: End-to-end t w o-stage retriev al, b oth stages sw app ed 
together.
0.6B still clears b oth the flo or and the off-the-shelf 1B 
(whic h it b eats b y + 0 . 0 0 9 , CI [ + 0 . 0 0 1 , + 0 . 0 1 8 ] ).
The ordering in T able  6  is the summary of this section. 
Both adapted rerank ers sit ab o v e the off-the-shelf one, 
whic h sits ab o v e the flo or. P arameter coun t separates 
the t w o adapted arms b y a little; adaptation separates 
them from the unadapted one b y three times as m uc h. 
What the second stage w as mostly missing w as nev er 
arc hitecture or scale. It w as exp osure to Greek.
8. Chained: what the reader actually re -
ceiv es
Sw apping b oth stages together ( Figure  9 ) lifts end-to-
end nDCG@10 from 0.559 to 0.848 , a 52% relativ e gain, 
concen trated where the off-the-shelf stac k w as w orst: 
finance + 0 . 4 1 , medical + 0 . 3 0 .
The n um b er that decides system b eha viour, though, 
is Recall@10, the fraction of queries whose answ er is 
presen t in the con text the reader actually receiv es. It 
rises from 0.624 to 0.955 . The off-the-shelf stac k loses 
the answ er en tirely for ab out 38% of queries; ours loses 
it for under 5%.
This b ounds gr ounde d  answ ering, not answ ering. A 
reader cannot ground an answ er in a passage it nev er 
receiv ed, so writing Hit @ 𝑘  for the fraction of queries with 
at least one gold passage retriev ed,
𝔼 [ grounded correct ] ≤ Hit @ 𝑘 .
The off-the-shelf first stage therefore caps grounded end-
to-end accuracy at 0.624 no matter ho w go o d the reader 
is, while ours raises that ceiling to 0.955. T w o ca v eats 
k eep this from go v erning Section  9 . A mo del can b e 
correct ungr ounde d ly , reciting a fact it knew rather than 
one it read, and our base reader is correct more often 
than it is faithful (29.4% against 25.2%), whic h is that 
gap made visible. And Section  9  scores reading on gold-
pro vided con text, so those n um b ers are measured at 
Hit @ 𝑘 = 1  b y construction and are not the pro duct of 
Figure  10: Grounded reading on the b enc hmark of T able  1 . 
Base and adapted mo dels receiv e an identic al  prompt instruct -
ing b oth citation and absten tion, so these ro ws measure 
compliance, not whether the mo del w as ask ed.
this ceiling with a reader. W e did not measure end to end 
on retriev ed con text, whic h is the n um b er that w ould 
comp ose the t w o.
9. A MoE reader that cites and abstains
W e LoRA-tuned ( Hu et al., 2022 )  Nemotron-3-
Nano-30B-A3B ( NVIDIA, 2025b ) , a mixture-of-exp erts 
mo del ( Shazeer et al., 2017 ; F edus et al., 2022 )  with 
3B activ e parameters, and ev aluated on the b enc hmark 
of Section  4 . The 40k cite-and-abstain examples of Sec -
tion  3.1  are only 16.2% of a 246,909-ro w blend: training 
on them alone teac hes the format at the cost of ev ery -
thing else, so the remainder is our Greek instruction 
p o ol, itself carrying a 7.2% English repla y slice in tended 
to limit catastrophic forgetting. T able  7  rep orts what 
that mo v ed on b enc hmarks w e nev er targeted, including 
where it mo v ed the wrong w a y: Greek gains on six of 
eigh t while English MMLU falls 0.098 and English arc-
c hallenge 0.094, and the mean o v er all thirteen is − 0 . 0 0 4 . 
W e ran no zero-repla y arm, so nothing here attributes 
the Greek half to the repla y slice.
The adapter.  Rank 16, 𝛼  32, drop out 0.05, on atten -
tion, MLP and the Mam ba in_proj : 439M trainable 
parameters, 1.37% of the mo del. One ep o c h at 8,192 
tok ens, assistan t-only loss, cosine LR 1 0− 4, gradien t 
clipping 0.3, effectiv e batc h 32 on four B200s, 32.7 hours.
Judged answ er correctness rises 29.4% →  66.9%  
( Figure  10 ; judge mo dels and deco ding settings in Ap -
p endix  D , prompts in App endix  E ); Wilson 95% in ter -
v als o v er the judged items are [ 2 7 . 9 , 3 0 . 9 ]  and [ 6 5 . 3 , 6 8 . 4 ] . 
But the more rev ealing mo v e is faithfulness: 25.2% →  
84.5%  ( [ 2 3 . 8 , 2 6 . 7 ]  and [ 8 3 . 2 , 8 5 . 7 ] ), a 3.4 ×  gain against 
correctness’s 2.3 × . The base mo del do es not merely 
answ er wrongly , it answ ers ungr ounde d ly , pro ducing 
con ten t the retriev ed con text do es not supp ort roughly 
three times in four. Since constraining generation is the 
10

T able  7: The ful l  thirteen-b enc hmark sw eep on capabilities neither half of the blend targets. Base and adapted are scored 
iden tically , accepting Greek and Latin answ er letters on b oth arms. Greek impro v es on six of eigh t, English regresses on three 
of fiv e, and o v er all thirteen the mean c hange is − 0 . 0 0 4 . The repla y slice did not prev en t forgetting; with no zero-repla y arm 
w e cannot sa y what it prev en ted.
Gr e ek English
b enc hmark 𝑛 base adapted Δ b enc hmark 𝑛 base adapted Δ
arc-c hallenge 1,168 0.5445 0.5308 − 0 . 0 1 3 7 arc-c hallenge 1,172 0.8823 0.7884 − 0 . 0 9 3 9
arc-easy 1,500 0.6320 0.6587 + 0 . 0 2 6 7 arc-easy 1,500 0.9640 0.8947 − 0 . 0 6 9 3
b eleb ele 900 0.6744 0.7089 + 0 . 0 3 4 5 hellasw ag 1,500 0.4920 0.5693 + 0 . 0 7 7 3
greekmmlu 1,500 0.5993 0.5440 − 0 . 0 5 5 3 mmlu 1,500 0.6953 0.5973 − 0 . 0 9 8 0
hellasw ag 1,500 0.3480 0.3973 + 0 . 0 4 9 3 winogrande 1,267 0.6788 0.7001 + 0 . 0 2 1 3
medical MCQA 432 0.2060 0.2176 + 0 . 0 1 1 6
truthfulqa 817 0.3305 0.3427 + 0 . 0 1 2 2
winogrande 1,267 0.5114 0.5604 + 0 . 0 4 9 0
mean Δ + 0 . 0 1 4 3 mean Δ − 0 . 0 3 2 5
en tire purp ose of a retriev al stage, an unadapted reader 
substan tially w astes it.
Correctness is conditional, and that flatters us.  
Both arms are scored only on grounded items they 
attempted, and they attempt differen t n um b ers: the base 
declines 16 of the 3,712 answ erable items, the adapted 
reader 128. Coun ting ev ery refusal on an answ erable item 
as wrong instead, correctness reads 28.7% →  64.3%  
o v er the full 3,712, so the conditional figure credits the 
adapted arm with ab out 2.6 p oin ts it earns b y refusing 
rather than b y answ ering.
P osition bias disapp ears.  The base mo del degrades 
monotonically as the gold passage mo v es later in the 
con text (31.9% →  29.4% →  26.1%), the effect rep orted 
b y Liu et al. (2024) . The adapted mo del sho ws no suc h 
ordering: early and middle land within 0.8 p oin ts of 
eac h other (65.8% and 65.0%). W e claim the monotone 
p enalt y is r emove d , not in v erted. These are single-run 
p oin t estimates, and a late-vs-middle gap is not an 
established rev ersal.
The adv an tage narro ws as con text gro ws.  Fig -
ure  11  trac ks this. The adapted reader falls from 76.5% 
Figure  11: Correctness b y con text size. The base mo del’s flat -
ness is a flo or, not robustness: it is not degrading with con text 
b ecause it w as nev er using it.correct at 8 do cumen ts to 59.3% at 40, a 17-p oin t deca y , 
while the base mo del sta ys flat at 26–32% across the 
whole range. That flatness is not robustness, it is a flo or 
effect: the base is not degrading with con text b ecause 
it w as nev er using it. F aithfulness holds up b etter than 
correctness o v er the same range (86.1% to 79.4%), so 
what the long-con text setting costs is mostly the abilit y 
to find the answ er, not the discipline to sta y grounded.
Absten tion is the honest ca v eat.  Refusing unan -
sw erable questions impro v es from 1.2% to 30.5% o v er 
1,234 unansw erable items ( [ 0 . 7 , 2 . 0 ]  and [ 2 8 . 0 , 3 3 . 1 ] ), a 
gain that is still a failing grade. The adapted reader an -
sw ers most unansw erable questions rather than declining 
them. An y one deplo ying this should treat absten tion as 
unsolv ed.
This is also the one place the base mo del wins. F alse 
absten tion, declining a question that w as in fact answ er -
able, rises from 0.4% to 3.4%  ( [ 0 . 3 , 0 . 7 ]  and [ 2 . 9 , 4 . 1 ] ). 
As rates that trade lo oks ten to one; in items it is nearer 
three to one, b ecause the answ erable partition is three 
times the larger: roughly 361 additional correct refusals 
against roughly 112 additional wrong ones. F a v ourable 
still, but it is a real regression and w e rep ort it rather 
than let the impro ving metrics sp eak alone.
10. F our w a ys our instrumen ts lied
Ev ery one of these pro duced a plausible, publishable, 
wr ong  n um b er. None w as caugh t b y insp ection. Eac h w as 
caugh t only when a second instrumen t disagreed.
1. A dapting atten tion alone w as w orse than not 
adapting.  On our in ternal Greek ev aluation, atten -
tion-only LoRA landed b elow  the un tuned base 
mo del, while adapting all linear la y ers gained sub -
stan tially . On h ybrid Mam ba-T ransformer stac ks, 
most of the capacit y is not in the atten tion blo c ks.
11

2. The scorer read a Greek-answ ering mo del as 
noise.  Our MCQ ev aluator scored Latin answ er 
letters only . A mo del that had successfully learned 
to answ er in Greek letters w as recorded as random 
guessing, a fak e catastrophic-forgetting result.
3. A missing prompt prefix made the v alidation 
curv e run bac kw ards.  The ev aluator omitted a 
prefix the mo del w as trained with, so the v alidation 
curv e fell steadily while true held-out qualit y rose.
4. The distributed loss w as summed, not a v er -
aged.  Effectiv e learning rate scaled silen tly with 
w orld size, so runs w ere not comparable across GPU 
coun ts.
11. What to tak e a w a y
Measure y our baseline b efore y ou buy a bigger 
mo del.  On our sp ecialist Greek domains an 8B em b ed -
der loses to BM25 and ties a 4B: scale w as the wrong 
axis there, and language exp osure w as the righ t one. On 
general Greek the same family separates cleanly and the 
8B do es catc h BM25, so the lesson is not that scale nev er 
helps. It is that y ou cannot kno w whic h axis y ou are on 
un til y ou ha v e run the lexical baseline on your  corpus.
Alw a ys ev aluate against the flo or, then ev aluate 
again.  A cross-enco der that lo ok ed reasonable in isola -
tion con tributed nothing o v er its o wn first stage on our 
first ev aluation, and a mo dest, unev en amoun t on a 
larger one. Only the no-rerank con trol could sho w either, 
and only the re-run sho w ed ho w m uc h the first answ er 
dep ended on the sample.
Build the ev aluation first when the language has 
none.  Greek had no RA G b enc hmark, so ev ery claim 
here w ould ha v e b een unfalsifiable without building one. 
It also caugh t the absten tion failure that the headline 
n um b ers hide.
A daptation is c heap, and it is b ounded b y the 
domains y ou adapt on.  A 1B em b edder and a 1B 
rerank er, b oth fine-tuned in hours, b eat an off-the-shelf 
stac k b y 52% end to end, and a LoRA on a 3B-activ e 
MoE more than doubled grounded answ er correctness. 
Off those domains the picture c hanges: on general Greek 
Wikip edia our adapted em b edder is b eaten b y BM25 
and b y an off-the-shelf m ultilingual mo del. A daptation 
b ough t language comp etence, whic h transfers, and do -
main fit, whic h do es not.
A v ailabilit y
The three mo dels and the b enc hmark ship as one col -
lection, Sophe a Nemo RA G : h uggingface.co/collections/
KIEFERSA/sophea-nemo-ragSophea-Nemo-Embedding . . em b edder
Sophea-Nemo-Reranker . . rerank er
Sophea-RAG-Nemo3 . . grounded reader
HERA . . reader b enc hmark, CC-BY-SA-4.0
The reader-sup ervision teac her KIEFERSA/Sophea-
Titan-1  (Apac he-2.0) is published separately . The re -
triev al training corpus is not part of this release, so 
the retriev al results here can b e repro duced against the 
released mo dels and b enc hmark but not retrained from 
scratc h. Base-mo del licences p ermit commercial use, and 
training data w as curated to exclude non-commercial 
sources.
What w e are not claiming.  The reader n um b ers are 
single-run p oin t estimates with no in terv als, scored b y 
an LLM judge on a b enc hmark whose h uman v erification 
is an unsized sp ot-c hec k and whose judge shares a family 
with its generator ( Section  4 ), so judged correctness is 
an upp er b ound; the retriev al n um b ers are measured 
on syn thetic queries and o v erstate pro duction lift ( Sec -
tion  3 ); that margin is in-domain only and rev erses 
on general-domain Greek; on medical the margin o v er 
BM25 is + 0 . 0 2 7  with an in terv al that touc hes zero, 
so w e do not claim an adv an tage o v er lexical searc h 
there ( Section  6 ); lab els are c h unk-as-gold rather than 
h uman-annotated; the Greek corpus sk ews formal and 
official, so con v ersational Greek is under-represen ted; 
absten tion remains p o or; and w e ha v e not tested h ybrid 
BM25+dense fusion as a serving first stage  b ey ond the 
single fusion configuration rep orted in Section  6 , and 
w e ha v e not tuned 𝑘 , the candidate depth, or the score 
normalisation at all.
References
Mik el Artetxe, Sebastian R uder, and Dani Y ogatama. 2020. On 
the Cross-lingual T ransferabilit y of Monolingual Represen tations. 
In Pr o c e e dings of A CL 2020 .
Lucas Bandarkar, Da vis Liang, Benjamin Muller, Mik el Artetxe, 
Sat y a Nara y an Sh ukla, Donald Husa, Naman Go y al, Abhinandan 
Krishnan, Luk e Zettlemo y er, and Madian Khabsa. 2024. The 
Beleb ele Benc hmark: a P arallel Reading Comprehension Dataset 
in 122 Language V arian ts. In Pr o c e e dings of A CL 2024 .
Gordon V. Cormac k, Charles L. A. Clark e, and Stefan Buettc her. 
2009. Recipro cal Rank F usion Outp erforms Condorcet and Indi -
vidual Rank Learning Metho ds. In Pr o c e e dings of SIGIR 2009 , 
pages 758–759.
Shah ul Es, Jithin James, Luis Espinosa-Ank e, and Stev en 
Sc ho c kaert. 2024. RA GAS: A utomated Ev aluation of Retriev al 
A ugmen ted Generation. In Pr o c e e dings of EA CL 2024 (System 
Demonstr ations) .
William F edus, Barret Zoph, and Noam Shazeer. 2022. Switc h 
T ransformers: Scaling to T rillion P arameter Mo dels with Simple 
and Efficien t Sparsit y . Journal of Machine L e arning R ese ar ch , 
23(120):1–39.
Luyu Gao, Y un yi Zhang, Jia w ei Han, and Jamie Callan. 2021. 
Scaling Deep Con trastiv e Learning Batc h Size under Memory 
Limited Setup. In Pr o c e e dings of the 6th W orkshop on R epr esen -
tation L e arning for NLP (R epL4NLP) .
12

Tian yu Gao, Ho w ard Y en, Jiatong Y u, and Danqi Chen. 2023. 
Enabling Large Language Mo dels to Generate T ext with Citations. 
In Pr o c e e dings of EMNLP 2023 , pages 6465–6488.
Edw ard J. Hu, Y elong Shen, Phillip W allis, Zeyuan Allen-Zh u, 
Y uanzhi Li, Shean W ang, Lu W ang, and W eizh u Chen. 2022. 
LoRA: Lo w-Rank A daptation of Large Language Mo dels. In Inter -
national Confer enc e on L e arning R epr esentations (ICLR) .
Kalerv o Järv elin and Jaana Kekäläinen. 2002. Cum ulated Gain-
based Ev aluation of IR T ec hniques. A CM T r ansactions on Infor -
mation Systems , 20(4):422–446.
Vladimir Karpukhin, Barlas Oğuz, Sew on Min, P atric k Lewis, 
Ledell W u, Sergey Eduno v, Danqi Chen, and W en-tau Yih. 2020. 
Dense P assage Retriev al for Op en-Domain Question Answ ering. 
In Pr o c e e dings of EMNLP 2020 .
A dit y a Kusupati, Gan ta vy a Bhatt, Anik et Rege, Matthew 
W allingford, A dit y a Sinha, Viv ek Raman ujan, William Ho w ard-
Sn yder, Kaifeng Chen, Sham Kakade, Prateek Jain, and Ali 
F arhadi. 2022. Matry oshka Represen tation Learning. In A dvanc es 
in Neur al Information Pr o c essing Systems (NeurIPS) .
Nelson F. Liu, Kevin Lin, John Hewitt, Ash win P aranjap e, Mic hele 
Bevilacqua, F abio P etroni, and P ercy Liang. 2024. Lost in the 
Middle: Ho w Language Mo dels Use Long Con texts. T r ansactions 
of the A sso ciation for Computational Linguistics , 12:157–173.
Lefteris Loukas, Nik olaos Sm yrnioudis, Chrysa Dik onimaki, 
Sp yros Barbak os, Anastasios Bomp otas, Ion Androutsop oulos, 
Pro dromos Malakasiotis, and Ilias Chalkidis. 2025. GR-NLP-
TOOLKIT: An Op en-Source NLP T o olkit for Mo dern Greek. In 
Pr o c e e dings of COLING 2025 (System Demonstr ations) .
Ro drigo Nogueira and Kyungh yun Cho. 2019. P assage Re-ranking 
with BER T. arXiv pr eprint arXiv:1901.04085 .
NVIDIA. 2025b. Nemotron 3 Nano: Op en, Efficien t Mixture-of-Ex -
p erts Hybrid Mam ba-T ransformer Mo del for Agen tic Reasoning. 
arXiv pr eprint arXiv:2512.20848 .
NVIDIA. 2025a. NVIDIA Nemotron 3: Efficien t and Op en In telli -
gence. arXiv pr eprint arXiv:2512.20856 .
Aaron v an den Oord, Y azhe Li, and Oriol Vin y als. 2018. Repre -
sen tation Learning with Con trastiv e Predictiv e Co ding. arXiv 
pr eprint arXiv:1807.03748 .
Stephen Rob ertson and Hugo Zaragoza. 2009. The Probabilis -
tic Relev ance F ramew ork: BM25 and Bey ond. F oundations and 
T r ends in Information R etrieval , 3(4):333–389.
Dimitris Roussis, Georgios P arask ev op oulos, Leon V ouk outis, 
Sokratis Sofianop oulos, Prok opis Prok opidis, V assilis P apa v assil -
iou, A thanasios Katsamanis, Stelios Pip eridis, and V assilis Kat -
souros. 2025. Krikri: A dv ancing Op en Large Language Mo dels for 
Greek. arXiv pr eprint arXiv:2505.13772 .
Jon Saad-F alcon, Omar Khattab, Christopher P otts, and Matei 
Zaharia. 2024. ARES: An A utomated Ev aluation F ramew ork 
for Retriev al-A ugmen ted Generation Systems. In Pr o c e e dings of 
NAA CL 2024 .
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Da vis, 
Quo c Le, Geoffrey Hin ton, and Jeff Dean. 2017. Outrageously 
Large Neural Net w orks: The Sparsely-Gated Mixture-of-Exp erts 
La y er. In International Confer enc e on L e arning R epr esentations 
(ICLR) .
Shiv alika Singh, F reddie V argus, Daniel D'souza, and others. 2024. 
A y a Dataset: An Op en-A ccess Collection for Multilingual Instruc -
tion T uning. In Pr o c e e dings of A CL 2024 .
Nandan Thakur, Nils Reimers, Andreas R üc klé, Abhishek Sriv as -
ta v a, and Iryna Gurevyc h. 2021. BEIR: A Heterogeneous Benc h -
mark for Zero-shot Ev aluation of Information Retriev al Mo dels. 
In NeurIPS Datasets and Benchmarks T r ack .
Ahmet Üstün, Viraat Ary abumi, Zheng-Xin Y ong, W ei-Yin K o, 
Daniel D'souza, Gb emilek e Onilude, Neel Bhandari, Shiv alika 
Singh, Hui-Lee Ooi, Amr Ka yid, and others. 2024. A y a Mo del: An Instruction Finetuned Op en-A ccess Multilingual Language Mo del. 
In Pr o c e e dings of A CL 2024 .
Leon V ouk outis, Dimitris Roussis, Georgios P arask ev op oulos, 
Sokratis Sofianop oulos, Prok opis Prok opidis, V assilis P apa v assil -
iou, A thanasios Katsamanis, Stelios Pip eridis, and V assilis Kat -
souros. 2024. Meltemi: The First Op en Large Language Mo del for 
Greek. arXiv pr eprint arXiv:2407.20743 .
Liang W ang, Nan Y ang, Xiaolong Huang, Linjun Y ang, Rangan 
Ma jumder, and F uru W ei. 2024. Multilingual E5 T ext Em b ed -
dings: A T ec hnical Rep ort. arXiv pr eprint arXiv:2402.05672 .
Xin yu Zhang, Nandan Thakur, Oduna y o Ogundep o, Ehsan Ka -
mallo o, Da vid Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi 
Rezagholizadeh, and Jimm y Lin. 2023. MIRA CL: A Multilingual 
Retriev al Dataset Co v ering 18 Div erse Languages. T r ansactions of 
the A sso ciation for Computational Linguistics , 11:1114–1131.
Y anzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan 
Lin, An Y ang, P eng jun Xie, W en Zhang, and Jingren Zhou. 2025. 
Qw en3 Em b edding: A dv ancing T ext Em b edding and Reranking 
Through F oundation Mo dels. arXiv pr eprint arXiv:2506.05176 .
Lianmin Zheng, W ei-Lin Chiang, Ying Sheng, Siyuan Zh uang, 
Zhanghao W u, Y onghao Zh uang, Zi Lin, Zh uohan Li, Dac heng Li, 
Eric P . Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 
2023. Judging LLM-as-a-Judge with MT-Benc h and Chatb ot 
Arena. In NeurIPS Datasets and Benchmarks T r ack .
A Metric definitions
Ranking qualit y .  F or a rank ed list of 𝑘  do cumen ts with 
binary relev ance 𝑟𝑖 ∈ { 0 , 1 } , discoun ted cum ulativ e gain 
and its normalised form are ( Järv elin and Kekäläinen, 
2002 )
DCG @ 𝑘 = ∑𝑘
𝑖 = 1𝑟𝑖
log2 ( 𝑖 + 1 ), nDCG @ 𝑘 = DCG @𝑘
IDCG@ 𝑘
where IDCG @ 𝑘  is the same quan tit y for the ideal 
ranking, so nDCG @ 𝑘 ∈ [ 0 , 1 ] . Retriev al figures are macro 
means of nDCG @ 1 0 , and T able  2  rep orts b oth w eigh t -
ings b ecause they differ: macr o over sets  a v erages the 
four ev aluation sets equally , macr o over queries  w eigh ts 
eac h of the 5,830 queries equally . P aired b o otstrap in -
terv als are computed p er query , so the + 0 . 0 8 0  margin 
o v er BM25 is the macro-o v er-queries difference ( 0 . 8 5 7 5 −
0 . 7 7 7 2 ); the macro-o v er-sets difference is + 0 . 0 7 8 . Else -
where in the rep ort, absolute lev els are quoted macro 
o v er sets.
Out-of-domain runs.  The HERA retriev al trac k and 
the medical set of Section  6  w ere scored with a 
max_seq_length  of 4,096 rather than eac h mo del’s o wn 
limit. The HERA corpus con tains passages up to 26,000 
c haracters and the mo dels default to 32,768-tok en win -
do ws, whic h exhausts memory; 4,096 matc hes our train -
ing length and sits ab o v e the corpus 99th p ercen tile of 
roughly 1,870 tok ens, so truncation is negligible. Prompt 
con v en tions are unc hanged.
Co v erage.  With ℛ︀𝑘  the top- 𝑘  retriev ed set and 𝒢︀  the 
gold set for a query ,
13

Recall @ 𝑘 =| ℛ︀𝑘 ∩ 𝒢︀ |
| 𝒢︀ |, MRR @ 𝑘 =1
| 𝑄 |∑
𝑞 ∈ 𝑄1
rank𝑞
where rank𝑞  is the p osition of the first relev an t do cumen t 
and the term is 0  if none app ears within 𝑘 . Recall@ 𝑘  is 
the quan tit y that b ounds the reader ( Section 8 ).
Lexical baseline.  BM25 scores a query–do cumen t pair 
as ( Rob ertson and Zaragoza, 2009 )
∑
𝑡 ∈ 𝑞IDF ( 𝑡 ) ⋅𝑓𝑡 , 𝑑 ( 𝑘1 + 1 )
𝑓𝑡 , 𝑑 + 𝑘1 ( 1 − 𝑏 + 𝑏 | 𝑑 | / | 𝑑 | )
with 𝑓𝑡 , 𝑑  the frequency of term 𝑡  in do cumen t 𝑑 , | 𝑑 |  its 
length, | 𝑑 |  the mean do cumen t length, and 𝑘1 = 1 . 2 , 𝑏 =
0 . 7 5 . T ok enisation is Greek-a w are: NFD normalisation, 
com bining marks stripp ed, final sigma folded.
Grounded reading.  Let 𝒞︀  b e the citation set a reader 
emits for an item and 𝒢︀  the gold set. F ollo wing the ALCE 
decomp osition ( Gao et al., 2023 ) ,
𝑃 =| 𝒞︀ ∩ 𝒢︀ |
| 𝒞︀ |, 𝑅 =| 𝒞︀ ∩ 𝒢︀ |
| 𝒢︀ |, 𝐹1 =2 𝑃 𝑅
𝑃 + 𝑅
and exact citation-set matc h is 𝟏 [ 𝒞︀ = 𝒢︀ ] , a strictly 
harder criterion than 𝐹1  b ecause a single spurious index 
fails the item outrigh t.
Ov er the unansw erable and answ erable partitions of 
the b enc hmark,
abstention recall =# { unanswerable, refused }
# { unanswerable },
false abstention =# { answerable, refused }
# { answerable }.
Answ er correctness and faithfulness are not closed-form: 
b oth are judged b y a serv ed LLM, correctness against the 
reference answ er on grounded-and-answ ered items, faith -
fulness b y asking whether ev ery claim in the resp onse 
is supp orted b y the retriev ed con text. F aithfulness is 
therefore scored against the con text rather than against 
an y reference text, whic h is wh y Section  4  treats it as 
the metric least exp osed to the generator confound.
B What eac h n um b er w as measured on
The rep orted comparisons do not all share a query set or 
a first stage, so they are not in terc hangeable. This table 
is the reconciliation.
Em b edder  ( T able  2 , Figures 1 – 6 ). 5,830 held-out queries, 
corpus = all p ositiv es. No maxim um-length o v erride is applied 
at scoring time, so eac h mo del runs at its o wn configured limit. 
Queries carry the Qw en3-Em b edding instruction con v en tion 
for that family; Nemotron-3-Em b ed is prompted on b oth  sides 
(::query:  / ::passage: ) b ecause its mean-p o ol includes the 
prompt.
Rerank er, first ev aluation  ( T able  5 , Figure  7 ). 150 queries 
p er set, 750 p o oled, seed 42; rerank maxim um length 2,048. The fixed first stage is our fine-tune d Qwen3-0.6B emb e dder , 
not the adapted Nemotron 1B, so the flo or here is not a ro w 
of T able  2  and the + 0 . 0 4 4  is not directly comp osable with the 
em b edder n um b ers. The off-the-shelf arm is llama-nemotron-
rerank-1b-v2 .
Rerank er, second ev aluation  ( T able  6 ). 2,580 queries: 593 
v alidation, 263 energy , 879 legal, 195 finance and the 650-
query medical set of Section  6 . Same fixed first stage, same 
top-50 candidate depth, same rerank maxim um length, same 
seed. Eac h set is scored against its o wn p ositiv es-plus-negativ es 
corpus (693 to 4,780 passages), so absolute lev els are not 
comparable to the first ev aluation or to T able  2 ; the quan tit y 
carried across runs is the paired difference against the flo or, 
whic h is computed within a run.
Chained  ( Figure  9 ). The same 750 p o oled queries, b oth 
stages sw app ed together, off-the-shelf Nemotron em b edder and 
rerank er against b oth of ours. Means are macro o v er the fiv e 
sets.
Reader  ( Section  9 , Figure  10  and Figure  11 ). All 4,946 HERA 
items, 3,712 answ erable and 1,234 unansw erable, on gold-
pro vided con text.
C The HERA metric set
HERA is released with three scoring trac ks, and they 
are delib erately separable: a system can fail at retriev al, 
at reading, or only when the t w o are c hained, and the 
trac ks are designed to tell those cases apart. This rep ort 
exercises the reader trac k and the robustness slice.
C.1 Retriev al trac k
Scored against eac h item’s gold_ids  o v er corpus.jsonl , 
with no reader in v olv ed.
Recall@ 𝑘  and Hit@ 𝑘  measure whether the gold 
passage is in the top 𝑘  at all. This is the trac k’s most 
consequen tial n um b er, b ecause it is the b ound of Sec -
tion  8 : a passage that nev er en ters the con text cannot b e 
read, so Recall@ 𝑘  caps ev ery reader metric do wnstream 
regardless of reader qualit y .
nDCG@ 𝑘  and MRR  measure ho w w ell the gold 
passage is placed once found. They differ in what they 
rew ard: nDCG credits ev ery relev an t passage with a 
rank-discoun ted w eigh t, so it resp onds to m ulti-gold 
items, while MRR lo oks only at the first relev an t hit and 
is therefore the b etter pro xy for a reader that reads the 
top do cumen t and stops.
C.2 Reader trac k
Scored on gold-pro vided con text, so that reading is 
measured without retriev al error mixed in.
Answ er correctness  is seman tic agreemen t with the 
gold answ er, graded b y an LLM judge rather than b y 
exact matc h. Exact matc h is un usable here: answ ers are 
14

free-form Greek, and a correct paraphrase with differen t 
morphology w ould b e scored wrong.
F aithfulness  asks whether ev ery claim in the re -
sp onse is supp orted b y the supplied con text. It is 
delib erately indep enden t of correctness, b ecause the 
t w o failures come apart. A mo del can b e correct but 
ungrounded, reciting a fact it knew rather than one it 
read, and it can b e faithful but wrong, misreading a 
passage it gen uinely used. Only faithfulness detects the 
first, whic h is the failure a retriev al system exists to 
prev en t.
Citation F1  compares the con text p ositions the 
reader cites against the gold citations  p ositions. Preci -
sion p enalises citing ev erything, whic h w ould otherwise 
b e a trivial w a y to guaran tee recall; recall p enalises 
answ ering without attribution.
Absten tion  is scored on the t w o partitions sepa -
rately: recall on the unansw erable items, i.e. ho w often 
the mo del correctly refuses, and the false-absten tion rate 
on the answ erable ones. Both are needed, since either 
alone is gameable b y a mo del that alw a ys refuses or 
nev er do es.
C.3 End-to-end and robustness
End-to-end qualit y  re-runs the reader metrics on 
r etrieve d  rather than gold-pro vided con text. This is the 
only n um b er that reflects deplo ymen t, and it is where 
retriev al and reading errors comp ound.
P ositional robustness  rep orts accuracy as a func -
tion of n_docs  and of where the gold passage sits in the 
con text, whic h is what exp oses the lost-in-the-middle 
effect ( Liu et al., 2024 )  and what Figure  11  and the 
p osition breakdo wn in Section 9  are dra wn from.
D Generation and judging configuration
Query syn thesis.  T emp erature 0 . 7 , top- 𝑝  0 . 9 , 
max_new_tokens  256 (512 for the few-shot domain 
prompts), seed 42, t w o or three queries p er c h unk, 
request concurrency 32 against an Op enAI-compatible 
vLLM endp oin t.
Judging.  T emp erature 0 , one judge call p er axis, output 
constrained to a single JSON ob ject so the v erdict is 
parsed rather than in terpreted. The judge endp oin t is 
prob ed liv e at startup and the first reac hable one is used, 
b ecause serv ed aliases drift.
Thinking m ust b e disabled, and this is not op -
tional.  Ev ery generator and judge here is a reasoning 
mo del. Called with its default c hat template, the mo del 
sp ends the en tire tok en budget on a reasoning trace and 
returns zer o  parseable JSON: finish_reason  comes bac k 
as length  with empt y con ten t. The fix is to passchat_template_kwargs={"enable_thinking": false}
at the request la y er. A "detailed thinking off"  
system message do es not  w ork, and neither do es a 
bare thinking: false  k ey . Disabling thinking also made 
syn thesis roughly ten times faster with no measurable 
qualit y loss.
E Prompts
The prompts b elo w are English translations; the origi -
nals are Greek, and the exact strings ship with the 
ev aluation scripts.
Reader system prompt.  Iden tical for the base and 
adapted mo dels, whic h is what mak es the citation and 
absten tion ro ws in Figure  10  a measure of compliance 
rather than of instruction:
Y ou are an assistan t that answ ers ex clusivel y  on the 
basis of the supplied do cumen ts. Use only the information 
in the do cumen ts; do not  in v en t. Cite in brac k ets the 
n um b er of the do cumen t or do cumen ts y ou used, e.g. [2]. 
If the answ er is not  presen t in the do cumen ts, sa y so 
plainly . Answ er in Greek, briefly and to the p oin t.
The user turn is the n um b ered con text, one blo c k p er 
passage as [n] text , follo w ed b y the question.
Judge prompts.  One p er axis, eac h returning a single 
JSON field.
A bstention.  Y ou will see an assistan t response . Sa y 
whether the assistan t refuses  to answ er b ecause the 
information is not in the do cumen ts ( abstain ), or whether 
it a ttempts  a substan tiv e answ er ( answer ). JSON only: 
{"verdict":"abstain|answer"}
Corr e ctness.  Y ou are an ev aluator. Y ou will see a ques -
tion , a sour ce p assa ge  (the correct information) and 
an assistan t response . Judge whether the resp onse is 
substan tiv ely correct  according to the source passage. 
JSON only: {"correct": true|false}
F aithfulness.  Y ou are a faithfulness ev aluator. Y ou will 
see documents  and a response . Judge whether ever y  
claim in the resp onse is supp orted b y the do cumen ts 
(no in v en tion or hallucination). JSON only: {"faithful": 
true|false}
15
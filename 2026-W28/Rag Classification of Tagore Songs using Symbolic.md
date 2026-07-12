# Rag Classification of Tagore Songs using Symbolic Music Notation and Novel Weighted Distance Measures

**Authors**: Chandan Misra, Swarup Chattopadhyay

**Published**: 2026-07-08 10:20:50

**PDF URL**: [https://arxiv.org/pdf/2607.07241v2](https://arxiv.org/pdf/2607.07241v2)

## Abstract
Rabindra Sangeet, the body of songs written and composed by Rabindranath Tagore, occupies a distinctive position in Indian music by combining poetic expression with melodic ideas drawn from Hindustani rags, Bengali folk traditions, tappa, kırtan, Baul music, and Western tunes. Although many Tagore songs are associated with rag labels provided by Tagore himself or preserved in authoritative notational traditions, rag identification remains challenging because the songs often reflect creative freedom rather than strict adherence to classical rag grammar. This paper formulates rag identification in Rabindra Sangeet as a supervised classification problem using symbolic music-sheet notations from Swarabitan. Since large-scale annotated audio or music datasets for Rabindra Sangeet are not readily available, this study constructs a rag-labelled symbolic dataset from notated Tagore songs. The work investigates Euclidean distance and cosine similarity for rag classification and introduces a weighted Euclidean distance measure that assigns greater importance to notes belonging to characteristic rag sequences such as arohana and avarohana. Applied within a k-nearest-neighbour framework, the proposed measure improves rag classification by better capturing rag-specific melodic identity.

## Full Text


<!-- PDF content starts -->

R¯AG CLASSIFICATION OF TAGORE SONGS USING SYMBOLIC MUSIC
NOTATION AND NOVEL WEIGHTED DISTANCE MEASURES
CHANDAN MISRA AND SWARUP CHATTOPADHYAY
Abstract.Rabindra Sangeet, the body of songs written and composed by Rabindranath Tagore,
occupies a distinctive position in Indian music by combining poetic expression with melodic ideas
drawn from Hindustani r¯ ags, Bengali folk traditions, tappa, k¯ ırtan, Baul music, and Western tunes.
Although many Tagore songs are associated with r¯ ag labels provided by Tagore himself or preserved
in authoritative notational traditions, r¯ ag identification remains challenging because the songs often
reflect creative freedom rather than strict adherence to classical r¯ ag grammar.
This paper formulates r¯ ag identification in Rabindra Sangeet as a supervised classification problem
using symbolic music-sheet notations fromSwarabitan. Since large-scale annotated audio or music
datasets for Rabindra Sangeet are not readily available, this study constructs a r¯ ag-labelled symbolic
dataset from notated Tagore songs. The work investigates Euclidean distance and cosine similarity for
r¯ ag classification and introduces a weighted Euclidean distance measure that assigns greater importance
to notes belonging to characteristic r¯ ag sequences such as arohana and avarohana. Applied within a
k-nearest-neighbour framework, the proposed measure improves r¯ ag classification by better capturing
r¯ ag-specific melodic identity.
1.Introduction
Indianmusichasevolvedthroughacontinuousinteractionbetweenclassicaldiscipline, regionalexpres-
sion, devotional imagination, poetic thought, and cultural practice. The Hindustani classical tradition,
with its highly developed concepts of r¯ ag and t¯ ala, provides a sophisticated framework for organizing
melody, rhythm, mood, and performance. At the same time, Indian musical culture has never remained
restricted to the boundaries of formal classical grammar. It has continuously interacted with folk tradi-
tions, devotional forms, literary movements, theatre, dance, and regional musical practices.
Within this rich musical landscape, Rabindra Sangeet occupies a distinctive and culturally significant
position. Composed by Rabindranath Tagore, the Nobel Laureate poet, composer, philosopher, and
artist, Rabindra Sangeet represents one of the most important song traditions of Bengal. These songs
form an integral part of Bengali cultural life in India and Bangladesh and continue to hold a central
place in artistic, social, educational, and devotional contexts.
Musically, Tagore’ssongsdrawfromawiderangeofsources. Manycompositionsrevealtheinfluenceof
Hindustani classical r¯ ags and t¯ ala-based structures, while others reflect the melodic character of tappa,
k¯ ırtan, Baul, Bengali folk music, devotional idioms, and Western tunes [1–3]. Tagore did not merely
reproduce these traditions in a fixed or mechanical manner. Instead, he adapted and transformed them
according to the expressive needs of the text, the emotional situation of the song, and his own artistic
vision. As a result, Rabindra Sangeet retains a deep connection with Indian musical traditions while
developing an independent musical identity.
Although, the tonal colours of r¯ ags often provide the emotional foundation of the songs, the use of
r¯ ag in Rabindra Sangeet is not always identical to its use in formal Hindustani classical performance. In
many cases, Tagore employed r¯ ag elements with creative freedom, allowing poetic meaning and emotional
expression to guide the musical structure. Some Tagore songs closely follow the characteristic structure
and melodic behaviour of a particular r¯ ag. Others combine features of two or more r¯ ags, while some
songs are traditionally associated with a r¯ ag but do not strictly follow its grammatical framework [4,5].
The compositional structure of Rabindra Sangeet further increases the difficulty of r¯ ag identification.
For a beginner or non-specialist listener, recognizing the r¯ ag from a Tagore song is often difficult because
the song may not present the r¯ ag in a direct or conventional manner. A single composition may contain
a long sequence of notes, repeated melodic phrases, expressive variations, and movements that do not
always correspond to the strict grammar of a classical r¯ ag. Consequently, predicting the r¯ ag basis of a
Rabindra Sangeet composition is a challenging task. Existing approaches to r¯ ag classification, which are
Date: June 1, 2017, accepted December 7, 2017.
2000Mathematics Subject Classification.Primary xxx, yyyy; Secondary xxxx, yyyy.
Key words and phrases.Dxxx gxxx, Axxx gxxx.
1
arXiv:2607.07241v2  [cs.SD]  9 Jul 2026

2 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
often designed for classical music performances, cannot be directly applied to Rabindra Sangeet without
careful adaptation. This is because Tagore’s songs are highly diverse in melodic construction.
R¯ ag identification is important for several computational music tasks, including music learning, music
information retrieval, mood-based classification, recommendation, and computational music generation.
In the context of Rabindra Sangeet, this task becomes particularly meaningful because many songs are
associated with r¯ ag labels, yet their melodic structures often differ from the strict grammar of Hindustani
classical compositions. Therefore, identifying the underlying r¯ ag of a Tagore song requires examining its
melodic behaviour, including the use of ¯Aroh,Avroh, dominant notes, omitted notes, and the relative
importance of notes within the composition.
However, thisdiversitydoesnotmaketheproblemunsuitableforsupervisedlearning. Onthecontrary,
Rabindra Sangeet provides an interesting setting for supervised r¯ ag classification because many songs are
associated with r¯ ag labels provided by Tagore himself or preserved in authoritative notational traditions.
These r¯ ag labels offer a meaningful ground truth for computational analysis. Therefore, the present
study formulates r¯ ag identification as a supervised classification problem, where labelled Tagore songs
are used to train a model that can learn r¯ ag-specific melodic behaviour from symbolic note sequences.
Although r¯ ag identification can naturally be formulated as a supervised learning problem, the avail-
ability of suitable annotated datasets remains a major limitation in the context of Rabindra Sangeet.
In many music classification tasks, annotated audio or music datasets [6–8] with reliable class labels
are used to train supervised models [9–12]. However, such large-scale labelled datasets are not readily
available for Rabindra Sangeet. This creates a significant barrier to applying existing data-driven r¯ ag
classification techniques directly to Tagore songs.
To address this limitation, the present work relies on symbolic music-sheet notations rather than
annotated audio recordings. The primary source of these notations isSwarabitan, the collection of
notated songs written and composed by Rabindranath Tagore.Swarabitancontains a large body of
Rabindra Sangeet compositions along with their musical notations, and therefore provides a structured
symbolic representation of the melodic content of Tagore songs.
These notated compositions are useful for computational analysis because the melodic content of
each song can be converted into note sequences and note-frequency distributions. Since many of these
compositions are associated with r¯ ag labels provided by Tagore himself or preserved in authoritative
notational traditions, they can be used to construct a supervised learning dataset. In this way, the
absence of a ready-made annotated audio dataset is addressed by preparing a r¯ ag-labelled symbolic
dataset from notated Rabindra Sangeet compositions.
In this work, we prepare a supervised symbolic dataset of1000r¯ ag-labelled Tagore songs fromSwara-
bitan. Each sample in the dataset corresponds to the note-frequency distribution of a composition and is
associated with its appropriate r¯ ag label. The construction of this dataset is an important contribution of
the work, as it provides a structured ground truth for supervised r¯ ag classification in Rabindra Sangeet.
A natural approach to r¯ ag identification is to compare a given composition with other compositions
whose r¯ ag labels are already known. If two compositions are musically similar, they may be expected
to share similar r¯ ag characteristics. Therefore, an unknown or test composition can be assigned the r¯ ag
label of its nearest labelled compositions using a distance or similarity-based classifier. In this direc-
tion, Euclidean distance and cosine similarity provide simple and widely used measures for quantifying
similarity between compositions represented through note-frequency features.
However, standard distance and similarity measures may not be sufficient for Rabindra Sangeet.
Preliminary observations show that Euclidean distance and cosine similarity can sometimes produce
contradictory or misleading results. In particular, they may assign high similarity or low distance not
only to compositions belonging to the same r¯ ag, but also to compositions belonging to different r¯ ags.
This happens because small variations in individual note frequencies may strongly affect the similarity
score, while the musically important role of characteristic notes may not be adequately captured. Thus,
two compositions may appear numerically similar even when their r¯ ag identities are different.
To address this limitation, the present study introduces a r¯ ag-aware weighted Euclidean distance
measure. The main idea is to assign greater importance to notes that belong to the prescribed ¯Arohand
Avrohof a r¯ ag, and comparatively lower importance to other notes. This weighting allows the distance
measure to better reflect r¯ ag-specific melodic identity rather than treating all note-frequency differences
equally. When used with ak-nearest-neighbour classifier, the weighted distance measure helps improve
the distinction between compositions belonging to similar and dissimilar r¯ ags.

SHORT TITLE 3
2.Dataset
2.1.Dataset Generation.As mentioned earlier, Tagore’s complete collection of songs consists of ap-
proximately2200compositions. This collection of songs is known asGeetobitan(The Garden of Songs),
while their musical notations are published separately inSwarabitan(The Garden of Notes).Swarabitan,
a book series comprising around60volumes, contains the notations of songs fromGeetobitanwritten in
the Akarmatrik notation system.
For the present study, the dataset was created by randomly selecting1000compositions fromSwarabi-
tan. ThenotesofeachselectedcompositionweremanuallystoredinaCSVfilealongwiththecorrespond-
ing r¯ ag label. Therefore, each sample in the dataset represents a single composition fromSwarabitan.
Since Rabindrasangeet is based on the R¯ ags of Hindustani Sangeet (with a little blend of folk genres
like Baul and Kirtan), the notes of the compositions span over three octaves orSaptaks, namely the
middle orMadhya, upper orTaar, and lower octave orMandra Saptak. The r¯ ags also follow the same
¯ArohandAvrohnotes of Hindutani Sangeet which depict the sequence of permissible notes in ascending
and descending pattern respectively. Additionally, they aid in describing the rag’s mood. Each octave
consists of12notes which generates36probable notes for each composition. We have indexed each note
with an integer starting from1and ending at36as given in Table 1.
NoteShadaj
(˙S,S, ˙S)Komal
Rishabh
(˙r,r,˙r)Suddha
Rishabh
(˙R,R, ˙R)Komal
Gandhar
(
˙g,g,˙g)Suddha
Gandhar
(˙G,G, ˙G)Madhyam
(˙M,M, ˙M)
Indices 1, 13, 25 2, 14, 26 3, 15, 27 4, 16, 28 5, 17, 29 6, 18, 30
NoteTivr
Madhyam
(˙m,m,˙m)Pancham
(˙P,P, ˙P)Komal
Dhaivat
(˙d,d, ˙d)Suddha
Dhaivat
(˙D,D, ˙D)Komal
Nishad
(˙n,n,˙n)Suddha
Nishad
(˙N,N, ˙N)
Indices 7, 19, 31 8, 20, 32 9, 21, 33 10, 22, 34 11, 23, 35 12, 24, 36
Table 1.Notes and indices of lower octave orMandra Saptak(indices 1 to 12), Middle
octave orMadhya Saptak(indices 13 to 24), and upper octave orTaar Saptak(indices
25 to 36)
Once such mapping is established we generate the note table for each such 1000 compositions. To
create the final dataset we generate the frequency distribution of each composition which serves as 36
independent variables and obtain the r¯ ag of the composition for the dependent variable. Figure 1 shows
the overall process of mapping a composition to a frequency table.
2.2.Summary of the Dataset.As previously mentioned the dataset consists of 36 features corre-
sponding to 36 notes spanning across three octaves for each sample and there are1000such samples
have been considered for our experimental evaluation. We labeled each such sample with the r¯ ag of the
composition which have also been extracted manually from the book while creating the dataset and can
be considered as a ground truth. There are239unique r¯ ags identified in the entire dataset and more than
50% samples (545compositions) belong to227least frequent r¯ ags having number of compositions less
than20. Table 1 shows the frequencies of unique r¯ ags with increasing number of compositions. In other
words Table 1 answers the queryHow many unique r¯ ags are there havingnnumber of compositions?,
for example.
While Table 2a summarizes the distribution of the less frequent r¯ ags in the dataset, Table 2b presents
the opposite view by listing the12most frequent r¯ ag labels. This information is important because
r¯ ags with very few compositions may not provide sufficient samples to capture a reliable note-frequency
pattern or average distribution for that particular r¯ ag.
In Table 2b, we also provide the correspondingThaatof each r¯ ag. In Hindustani classical music,
aThaatrefers to the parent scale or melodic framework used in Pt. Vishnu Narayan Bhatkhande’s
system for classifying r¯ ags according to their note structure. Including theThaatinformation helps the
reader understand the broader melodic family to which a r¯ ag belongs. It is also useful for interpreting
similarities between r¯ ags, since r¯ ags belonging to the sameThaatmay share similar note structures.
However, the entriesBaulandKirtanare marked as NA because they are not classified as r¯ ags under
the BhatkhandeThaatsystem. Rather, they represent folk and devotional musical traditions that occur
frequently in Rabindra Sangeet but do not necessarily follow a fixed ¯Aroh–Avroh structure associated
with a specific Hindustani classical r¯ ag. Therefore, they are retained in the dataset as musical-category
labels rather than being assigned to a particularThaat.

4 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
(a)Actual Composition161414131313212121232525
232828262523252525252525
232826252625232523212321
(b)Mapping from actual note to its corresponding inte-
ger value
Note Index 123456789101112
Frequency 000000000000
Note Index 131415161718192021222324
Frequency 320100005070
Note Index 252627282930313233343536
Frequency 1230300000000
(c)Frequency table of individual notes
Figure 1.The process of note mapping and generating its frequency table for a com-
position (line no. 19 - 21) titledkahar galay parabibelonging toPremparjaay and
Bhairavir¯ ag.
Number of
CompositionsFrequency of
Unique R¯ agsTotal No.
of Compositions
1 135 135
2 33 66
3 15 45
4 14 56
5 7 35
6 9 54
7 3 21
8 2 16
9 3 27
10 1 10
13 1 13
15 1 15
16 1 16
17 1 17
19 1 19
Total 227 545
(a)R¯ ag Abbr. Thaat Frequency
Bhairavi bhBhairavi 100
Bihag biBilaval 49
Kirtan kr NA 45
Khamaj khKhamaj 37
Desh deKhamaj 35
Pilu pl Kafi 33
Baul bl NA 33
Kafi kf Kafi 31
Yaman-Kalyan ykKalyan 27
Sahana sh Kafi 23
Yaman ynKalyan 22
Kedara kdKalyan 20
455
(b)
Table 2.(a) showing summary to the queryHow many unique r¯ ags are there having
nnumber of compositions?and (b) showing the12most frequent r¯ ags in the dataset
along with thethaat(parent scale according to Pt. Vishnu Narayan Bhatkhande) they
belong to.
Thus, the dataset can be viewed as a note-frequency-based representation of individual compositions.
Each sample represents the distribution of notes in a composition and is associated with its corresponding
r¯ ag or musical-category label.
3.Notations Used in the Present Work
LetNdenote the set of musical notes used in this work, as listed in Table 1. Thus,

SHORT TITLE 5
(3.1)
N=n
˙S, S, ˙S,˙r, r,˙r,˙R, R, ˙R,
˙g, g,˙g,˙G, G, ˙G,˙M, M, ˙M,˙m, m,˙m,˙P, P, ˙P,˙d, d, ˙d,˙D, D, ˙D,˙n, n,˙n,˙N, N, ˙No
.
Similarly, letRdenote the set of r¯ ags considered in this paper. Therefore,
(3.2)R=n
bh, bi, kr, kh, de, pl, bl, kf, yk, sh, yn, kdo
.
Thei-th sample composition belonging to a particular r¯ agρis denoted bycompi
ρ, whereρ∈ R. For
example, thecompositionsbelongingtor¯ agBhairavicanbedenotedbycompi
bh, wherei∈ {1,2, . . . ,100},
since the dataset contains100compositions of r¯ agBhairavi, as shown in Table 2b.
LetIdenote the set of note indices corresponding to the notes listed in Table 1. This set is defined
as
(3.3)I=n
1,2, . . . ,35,36o
.
For a given compositioncompi
ρbelonging to r¯ agρ, the set of note indices having non-zero frequency
is denoted byI compiρ, where
(3.4)I compiρ⊂ I.
For instance, the set of non-zero note indices for the firstBhairavicomposition in the dataset is given
by
(3.5)Icomp1
bh={6,8,9,11,13,14,15,16,18,20,21,23}.
We further categorize each composition belonging to a r¯ agρas either pure or non-pure using the
subscripts(p)and(np), respectively. This categorization is based on the extent to which the composition
follows the prescribedarohandavrohof the corresponding r¯ agρ. Accordingly, if thei-th composition
of r¯ agρis pure, it is denoted bycompi
ρ(p); otherwise, if thei-th composition is non-pure, it is denoted
bycompi
ρ(np).
4.Experimental Evaluation
In the experimental evaluation, we quantify the similarity between compositions belonging to the
same r¯ ag and those belonging to different r¯ ags. For this purpose, we employ cosine similarity and
Euclidean distance as direct and indirect measures of compositional similarity, respectively. We then
propose a modified distance measure to capture the similarity between compositions more effectively.
The effectiveness of the proposed measure is further justified through its performance with ak-nearest-
neighbor classifier.
4.1.Comparison between compositions belonging to same r¯ ag.Compositions belonging to same
r¯ agaretheonesthatfollowthesamearohandavrohofther¯ ag. Therefore, thenotefrequencydistribution
of the same r¯ ag compositions should follow the same silhouette. Table 3 and Figure 2 shows the note
frequency distribution of a pure composition belonging to r¯ ag Khamaj.
(a)
 (b)
 (c)
Figure 2.Note Frequency distribution of two pure Khamaj composition 2a and 2b.
X-axis represents note indices as given in Table 1 and Y-axis represents corresponding
note frequencies. Plot 2c overplots two distributions which are exactly matching since
both belongs to same r¯ ag.
As mentioned earlier, cosine similarity and Euclidean distance are used to analyze the similarity
between compositions belonging to the same r¯ ag. Since compositions of the same r¯ ag are expected to
exhibit similar note-frequency patterns, they should ideally produce higher cosine similarity values and
lower Euclidean distance values.

6 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
Lower Octave or Mandra Saptak
Note Index 123456789101112
comp1
kh 000000000000
comp5
kh 000000000000
Middle Octave or Madhya Saptak
Note Index 131415161718192021222324
comp1
kh 101011300290503534
comp5
kh 608057530720683665
Upper Octave or Taar Saptak
Note Index 252627282930313233343536
comp1
kh4709071000000
comp5
kh11202002631010000
Table 3.Example of two pure compositions of r¯ ag Khamaj. The aroh and avroh notes
of r¯ ag Khamaj are S, R, G, M, P, D, n, and N and are colored as blue.
Since cosine similarity ranges from0to1, we compute, for each composition, its mean cosine similarity
with all other compositions belonging to the same r¯ ag. These mean similarity values are plotted in
Figure 3 to visualize the degree of similarity among compositions of the same r¯ ag.
Figure 3.Cosine Similarities between compositions belonging to r¯ ags in row-major
orderBhairavi,Bihag,Kirtan,Khamaj,Desh,Pilu,Baul,Kafi,Yaman Kalyan,Sahana,
Yaman,Kedara. The x-axis and y-axis of each sub-plot of the grid correspond to the
number of compositions and mean cosine similarity value respectively.

SHORT TITLE 7
Figure 4.Sorted Mean Cosine Similarities between compositions belonging to r¯ ags
in row-major orderBhairavi,Bihag,Kirtan,Khamaj,Desh,Pilu,Baul,Kafi,Yaman
Kalyan,Sahana,Yaman,Kedara. The x-axis and y-axis of each sub-plot of the grid
correspond to the number of compositions and mean cosine similarity value respectively.
Each subfigure in Figure 3 is overplotted with a red horizontal line representing the mean cosine
similarity value. These mean values indicate that, in general, compositions belonging to the same r¯ ag
exhibit high similarity among themselves, thereby supporting the observation stated earlier. Except
for the compositions belonging to r¯ agBhairavi, most of the r¯ ag groups show relatively high intra-r¯ ag
similarity.
Another way to interpret the similarity among compositions of the same r¯ ag is to examine the dis-
tribution of their mean similarity scores. Ideally, only a small number of compositions should have low
similarity values, while the majority should exhibit high similarity values. To visualize this pattern more
clearly, the mean similarity scores are sorted and plotted in Figure 4. An ideal similarity curve is ex-
pected to begin at a relatively high value, rise sharply, and then stabilize close to one. Such behaviour
can be observed in the plots corresponding to r¯ agsBihag,Kirtan,Khamaj,Kedara, andYaman Kalyan.
Deviationfromthisidealbehaviorinferslowsimilarityvalueandproducesplotsthatgraduallyincrease
and exhibits a large number of lower similarity values, r¯ ag Bhairavi and Yaman for example.
There are two possible reasons for such deviation. First, some compositions may contain notes that
do not belong to the prescribed ¯ArohandAvrohof the corresponding r¯ ag. Second, even when two
compositions belong to the same r¯ ag, their notes may occur in different octaves, resulting in little or
no overlap in their note-index positions. To illustrate these two cases, we present two representative
examples from the dataset.
(1) To support the first case, we consider one pure composition belonging to r¯ agKhamaj, one non-
pure composition belonging to r¯ agKhamaj, and one pure composition belonging to r¯ agBhairavi.
The pure Khamaj composition is denoted bycomp22
kh(p), the non-pure Khamaj composition is
denoted bycomp35
kh(np), and the pure Bhairavi composition is denoted bycomp43
bh(p), as shown in
Table 4.

8 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
For r¯ agKhamaj, the prescribed note-index set is
Ikh={1,3,5,6,8,10,11,12}.
After extending this set across three octaves, the allowed Khamaj note indices are
{1,3,5,6,8,10,11,12,13,15,17,18,20,22,23,24,25,27,29,30,32,34,35,36}.
The compositioncomp22
kh(p)is a pure Khamaj composition because all its non-zero note indices
belong to the prescribed Khamaj note-index set. In contrast,comp35
kh(np)is a non-pure Khamaj
composition because it contains the note at index19, highlighted in red, which does not belong
to the prescribed ¯ArohorAvrohnote set of r¯ agKhamaj. The third composition,comp43
bh(p), is a
pure Bhairavi composition.
Using normalized ordinary Euclidean distance on the36-dimensional note-frequency vectors,
the distance betweencomp22
kh(p)andcomp43
bh(p)is0.3099, whereas the distance betweencomp22
kh(p)
andcomp35
kh(np)is0.3242. Thus, normalized ordinary Euclidean distance incorrectly indicates
that the pure Khamaj composition is closer to the pure Bhairavi composition than to the non-
pure Khamaj composition. This observation motivates the use of a r¯ ag-aware weighted distance
measure, discussed later, to better preserve musically meaningful relationships.
Lower Octave or Mandra Saptak
Note Index 123456789101112
comp22
kh(p) 000000000000
comp35
kh(np) 000000000400
comp43
bh(p) 000000000000
Middle Octave or Madhya Saptak
Note Index 131415161718192021222324
comp22
kh(p) 170807440420634214
comp35
kh(np) 2701406710975028017
comp43
bh(p) 1016012015021200170
Upper Octave or Taar Saptak
Note Index 252627282930313233343536
comp22
kh(p) 55019040000000
comp35
kh(np) 3804000000000
comp43
bh(p) 1830100000000
Normalized Ordinary Euclidean Distance
comp22
kh(p)vs.comp43
bh(p) 0.3099
comp22
kh(p)vs.comp35
kh(np) 0.3242
Table 4.Comparison of a pure Khamaj compositioncomp22
kh(p), a non-pure Khamaj
compositioncomp35
kh(np), and a pure Bhairavi compositioncomp43
bh(p). The prescribed
Khamaj note indices are highlighted in blue, while the deviated note index in the non-
pure Khamaj composition is highlighted in red. Normalized ordinary Euclidean distance
gives a misleading ordering by placing the pure Khamaj composition closer to the pure
Bhairavi composition than to the non-pure Khamaj composition.
(2) To support the second case, we consider two compositions belonging to r¯ agBhairavi. Both
compositionsarenon-pure, i.e., theydonotstrictlyfollowtheprescribed ¯ArohandAvrohpattern
of r¯ agBhairavi. However, their notes are distributed across different octaves in a non-overlapping
manner. That is,
Icomp312
bh(np)∩ Icomp415
bh(np)=∅.
Since both compositions belong to r¯ agBhairavi, their similarity is expected to be high from a
r¯ ag-level perspective. However, due to the non-overlapping nature of their note-index positions
across the three octaves, the cosine similarity becomes zero, as shown in Table 5. This example
further indicates that cosine similarity may fail to capture r¯ ag-level similarity when octave-wise
note positions differ significantly.

SHORT TITLE 9
Lower Octave or Mandra Saptak
Note Index 123456789101112
comp312
bh(np) 000000000000
comp415
bh(np) 00000405250361
Middle Octave or Madhya Saptak
Note Index 131415161718192021222324
comp312
bh(np) 0000000040106
comp415
bh(np) 682933109000000
Upper Octave or Taar Saptak
Note Index 252627282930313233343536
comp312
bh(np) 402312902000000
comp415
bh(np) 000000000000
Cosine Similarity Score
comp312
bh(np)vs.comp415
bh(np) 0.0
Table 5.Two non-pure compositions,comp312
bh(np)andcomp415
bh(np), belonging to r¯ ag
Bhairavi. Although both compositions belong to the same r¯ ag, their non-zero note
indices occur in non-overlapping octave-wise positions, as highlighted in blue. Conse-
quently, the cosine similarity between them becomes zero.
4.2.Comparison between compositions belonging to different r¯ ags.A similar issue arises when
computing the similarity between compositions belonging to different r¯ ags. Since r¯ ags belonging to
the sameThaatmay exhibit similar melodic structures, we selected five r¯ ags, namelyBhairavi,Bihag,
Khamaj,Kafi, andKedara, each belonging to a differentThaat, as shown in Table 2b.
Similar to the experiment conducted for compositions belonging to the same r¯ ag, we visualize the
similarity between compositions belonging to different r¯ ags by computing the mean cosine similarity of
each composition with all compositions of another r¯ ag. The resulting plots are shown in Figure 5. Each
subfigure in Figure 5 is overplotted with a red horizontal line, which represents the mean similarity value
for the corresponding pair of r¯ ags.
Figure 5.Mean cosine similarity scores between compositions belonging to different
r¯ ags. The selected r¯ ags are Bhairavi, Bihag, Khamaj, Kafi, and Kedara, each shown
in pairwise comparison. The x-axis represents the composition index, while the y-axis
represents the mean cosine similarity with compositions of the other r¯ ag. The horizontal
red line shows the average similarity score for each r¯ ag pair.
The mean similarity score indicated by the red horizontal line shows that most of the similarity scores
between compositions belonging to different r¯ ags lie above0.62, which is a considerably high value in
terms of cosine similarity. For example, the mean similarity score between r¯ agBihagand r¯ agKedarais

10 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
0.79. Such a high similarity value between compositions of two different r¯ ags is not generally expected
from a musical point of view.
This behaviour can be further understood through a representative example involving pure composi-
tions. We consider one pure composition from r¯ agBhairavi, namelycomp43
bh(p), and three pure composi-
tions from r¯ agKhamaj, namelycomp17
kh(p),comp20
kh(p), andcomp21
kh(p). The note-frequency distributions
of these compositions and the corresponding cosine similarity scores are shown in Table 6.
Lower Octave or Mandra Saptak
Note Index 123456789101112
comp43
bh(p) 000000000000
comp17
kh(p) 000000000000
comp20
kh(p) 000000000000
comp21
kh(p) 000000000000
Middle Octave or Madhya Saptak
Note Index 131415161718192021222324
comp43
bh(p) 1016012015021200170
comp17
kh(p) 0020513029019212
comp20
kh(p) 000015020149
comp21
kh(p) 405031310470201411
Upper Octave or Taar Saptak
Note Index 252627282930313233343536
comp43
bh(p) 1830100000000
comp17
kh(p) 26060104010000
comp20
kh(p) 1001000000000
comp21
kh(p) 2707000000000
Cosine Similarity Scores
comp43
bh(p)vs.comp17
kh(p) 0.6733
comp20
kh(p)vs.comp21
kh(p) 0.6426
Table 6.Compositioncomp43
bh(p)is a pure composition belonging to r¯ agBhairavi, while
comp17
kh(p),comp20
kh(p), andcomp21
kh(p)are pure compositions belonging to r¯ agKhamaj.
The cosine similarity between the pure compositions of two different r¯ ags,comp43
bh(p)
andcomp17
kh(p), is higher than the cosine similarity between two pure compositions of
the same r¯ ag,comp20
kh(p)andcomp21
kh(p).
From Table 6, it can be observed that the cosine similarity between two pure compositions belonging
to different r¯ ags, namelycomp43
bh(p)andcomp17
kh(p), is0.6733. In contrast, the cosine similarity between
two pure compositions belonging to the same r¯ agKhamaj, namelycomp20
kh(p)andcomp21
kh(p), is0.6426.
Thus,
(4.1)cos(comp43
bh(p), comp17
kh(p))>cos(comp20
kh(p), comp21
kh(p)).
This example shows that two pure compositions belonging to different r¯ ags may exhibit a higher cosine
similarity score than two pure compositions belonging to the same r¯ ag. Therefore, quantifying similarity
or dissimilarity among r¯ ag-based compositions cannot be reliably achieved using cosine similarity alone.
Such observations motivate the need to modify the structure of the dataset and the distance measure,
as described in Section 4.3 and Section 4.4, respectively.
4.3.Dataset Modification.In the original representation, each composition is represented using the
frequency distribution of36notes, corresponding to12notes across three octaves. However, for r¯ ag-level

SHORT TITLE 11
similarity analysis, the octave position of a note does not necessarily change the identity of the r¯ ag.
For example, the occurrence of the same note in the mandra, madhya, or taar saptak contributes to
the melodic structure of the same r¯ ag. Therefore, instead of using the complete36-dimensional note-
frequency vector, we reduce each composition to a12-dimensional cumulative note-frequency vector.
LetN 12denote the reduced set of12notes, defined as
N12={S, r, R, g, G, M, m, P, d, D, n, N}.
The corresponding note-index set is defined as
I12={1,2, . . . ,12}.
Letx= (x 1, x2, . . . , x 36)be the original36-dimensional note-frequency vector of a composition. The
corresponding12-dimensional cumulative vectorx′= (x′
1, x′
2, . . . , x′
12)is obtained by summing the fre-
quencies of the same note across the three octaves. Thus,
x′
j=xj+xj+12+xj+24, j= 1,2, . . . ,12.
This mapping combines the frequencies of corresponding notes from the lower, middle, and upper
octaves. As a result, the comparison between two compositions becomes less sensitive to octave-specific
positions and more focused on the overall note usage pattern of the r¯ ag. This modification helps reduce
the possibility of obtaining very low or zero similarity scores between compositions of the same r¯ ag merely
because their notes occur in different octaves.
Table 7 illustrates this transformation using the compositioncomp1
kh.
Lower Octave or Mandra Saptak
Note Index 123456789101112
comp1
kh000000000000
Middle Octave or Madhya Saptak
Note Index 131415161718192021222324
comp1
kh101011300290503534
Upper Octave or Taar Saptak
Note Index 252627282930313233343536
comp1
kh4709071000000
Aggregated Frequency Distribution across Octaves
Note Index
Note1
S2
r3
R4
g5
G6
M7
m8
P9
d10
D11
n12
N
comp1
kh48010018310290503534
Table 7.Transformation of a36-dimensional octave-specific note-frequency vector into
a12-dimensional cumulative note-frequency vector.
4.4.Proposed Distance Measure.After modifying the dataset into a12-dimensional cumulative
representation, we propose a r¯ ag-aware weighted Euclidean distance to better capture the similarity
and dissimilarity among compositions belonging to the same and different r¯ ags. The proposed distance
measure is designed to incorporate the prescribed note structure of a r¯ ag into the distance computation.
Leta= (a 1, a2, . . . , a 12)andb= (b 1, b2, . . . , b 12)be two12-dimensional cumulative note-frequency
vectors. Before computing the distance, both vectors are normalized by their total frequency. Thus,
ˆai=aiP12
j=1aj, ˆbi=biP12
j=1bj, i= 1,2, . . . ,12.
The use of normalized vectors ensures that the distance is not dominated by the total length or total
note count of a composition. Instead, the comparison is based on the relative distribution of notes.
The proposed weighted Euclidean distance between two normalized vectorsˆaand ˆbis defined as
(4.2)E w(a, b) =vuut12X
i=1wi
ˆai−ˆbi2
,0< w i<1,12X
i=1wi= 1.

12 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
Here,w idenotes the weight assigned to thei-th note inN 12. The weight vector is constructed
separately for each r¯ ag according to its prescribed ¯ArohandAvrohnote set.
For each r¯ agρ, the complete12-note setN 12is divided into two mutually exclusive subsets. The
first subset, denoted byN′
12, contains the notes that occur in the prescribed ¯ArohandAvrohof the
corresponding r¯ ag. The second subset, denoted byN′′
12, contains the remaining notes that do not occur
in the prescribed ¯ArohandAvroh. Therefore,
(4.3)N′
12∪ N′′
12=N 12,N′
12∩ N′′
12=∅.
Similarly, letI′
ρdenote the set of note indices corresponding to the notes present in the prescribed
¯ArohandAvrohof r¯ agρ. LetI′′
ρdenote the set of remaining note indices. Thus,
(4.4)I′
ρ∪ I′′
ρ={1,2, . . . ,12},I′
ρ∩ I′′
ρ=∅.
In this work, we use a90 : 10weight distribution. Under this rule,90%of the total weight is
distributed equally among the notes belonging toI′
ρ, while the remaining10%of the total weight is
distributed equally among the notes belonging toI′′
ρ. Therefore, the weightw iis defined as
(4.5)w i=

0.90
|I′ρ|, i∈ I′
ρ,
0.10
|I′′ρ|, i∈ I′′
ρ.
This construction ensures that
12X
i=1wi= 1.
The purpose of this weighting scheme is to assign greater importance to the notes that define the
melodic identity of a r¯ ag. Notes that are part of the prescribed ¯ArohandAvrohreceive higher weights,
whereasnotesoutsidetheprescribednotesetreceivelowerweights. Asaresult, thedistancecomputation
becomes more sensitive to r¯ ag-specific note usage.
For example, for r¯ agKhamaj, the prescribed note-index set is
I′
kh={1,3,5,6,8,10,11,12}.
Therefore,
|I′
kh|= 8,|I′′
kh|= 4.
Hence, under the90 : 10rule, each prescribed Khamaj note receives weight
0.90
8= 0.1125,
whereas each non-prescribed note receives weight
0.10
4= 0.0250.
The r¯ ag-specific weight vectors used in this work are presented in Table 8.
R¯ ag w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12
Bhairavi 0.1286 0.1286 0.0200 0.1286 0.0200 0.1286 0.0200 0.1286 0.1286 0.0200 0.1286 0.0200
Khamaj 0.1125 0.0250 0.1125 0.0250 0.1125 0.1125 0.0250 0.1125 0.0250 0.1125 0.1125 0.1125
Table 8.R¯ ag-specific weight vectors using the90 : 10distribution. Here,90%of the
total weight is distributed equally among the notes occurring in the prescribed ¯Arohand
Avrohof the corresponding r¯ ag, while the remaining10%is distributed equally among
the other notes.

SHORT TITLE 13
5.Resolving the Observed Shortcomings Using the Modified Distance
The previous examples show that cosine similarity and ordinary Euclidean distance may produce
musically misleading results in r¯ ag-based composition analysis. To address these limitations, we revisit
the same examples using the proposed modified distance measure. In this approach, each original36-
dimensional octave-specific note-frequency vector is first converted into a12-dimensional cumulative
note-frequency vector. The resulting vector is then normalized by total frequency before computing the
r¯ ag-aware weighted distance.
The use of the12-dimensional cumulative representation reduces the effect of octave-specific note
placement. Normalization removes the effect of composition length, while the r¯ ag-aware weight vector
gives higher importance to musically significant note positions. The following subsections show how the
modified distance addresses the three shortcomings discussed earlier.
5.1.Case 1: Pure and Non-pure Compositions of the Same R¯ ag.In the first case, we compare
a pure Khamaj composition, a non-pure Khamaj composition, and a pure Bhairavi composition. The
pure Khamaj composition is denoted bycomp22
kh(p), the non-pure Khamaj composition is denoted by
comp35
kh(np), and the pure Bhairavi composition is denoted bycomp43
bh(p).
Earlier, normalized ordinary Euclidean distance on the36-dimensional representation incorrectly
placed the pure Khamaj composition closer to the pure Bhairavi composition than to the non-pure
Khamaj composition. Using the proposed modified distance on the12-dimensional cumulative normal-
ized vectors gives the results shown in Table 9.
Pair of Compositions Modified Weighted Distance Interpretation
comp22
kh(p)vs.comp35
kh(np) 0.0830 Closest pair
comp22
kh(p)vs.comp43
bh(p) 0.1153 Farther than Khamaj pair
Table 9.Resolution of the first shortcoming using the modified weighted distance. The
pure and non-pure Khamaj compositions remain closer to each other than to the pure
Bhairavi composition.
From Table 9, we observe that
Dw(comp22
kh(p), comp35
kh(np) )< D w(comp22
kh(p), comp43
bh(p))
Thus, the modified weighted distance gives the expected musical ordering by keeping the pure and
non-pure Khamaj compositions closest to each other.
5.2.Case 2: Octave-wise Non-overlapping Notes.In the second case, we revisit the two Bhairavi
compositions whose non-zero note indices were distributed across different octaves in a non-overlapping
manner. In the original36-dimensional representation, the cosine similarity between these two composi-
tions became zero because there was no overlap in their note-index positions.
However, fromar¯ ag-levelperspective, theoctavepositionofthesamenotedoesnotchangetheidentity
of the r¯ ag. Therefore, the proposed12-dimensional cumulative representation combines corresponding
notes across the three octaves before computing the distance. The modified distance result is shown in
Table 10.
Pair of Compositions Modified Weighted Distance Interpretation
comp312
bh(np)vs.comp415
bh(np) 0.0647 Small distance after octave aggregation
Table 10.Resolution of the second shortcoming using the modified weighted distance.
Although the two Bhairavi compositions had non-overlapping note indices in the36-
dimensionalrepresentation, theoctave-summedrepresentationshowsthattheyareclose.
Table 10 shows that the modified distance between the two Bhairavi compositions is small. Thus, the
proposed method removes the artificial dissimilarity caused by octave-wise non-overlap. This resolves
the earlier problem where cosine similarity became zero despite both compositions belonging to the same
r¯ ag.

14 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
5.3.Case 3: Pure Compositions Belonging to Different R¯ ags.In the third case, we revisit the
example where cosine similarity gave a higher score to two pure compositions belonging to different r¯ ags
than to two pure compositions belonging to the same r¯ ag. Specifically, the cosine similarity between
comp43
bh(p)andcomp17
kh(p)was higher than the cosine similarity betweencomp20
kh(p)andcomp21
kh(p).
Using the proposed modified distance on the same examples, we obtain the results shown in Table 11.
Pair of Compositions Modified Weighted Distance Interpretation
comp20
kh(p)vs.comp21
kh(p) 0.1000 Same r¯ ag pair is closer
comp43
bh(p)vs.comp17
kh(p) 0.1062 Different r¯ ag pair is farther
Table 11.Resolution of the third shortcoming using the modified weighted distance.
The two pure Khamaj compositions are assigned a smaller distance than the pure
Bhairavi–pure Khamaj pair.
From Table 11, we observe that
Dw(comp20
kh(p), comp21
kh(p))< D w(comp43
bh(p), comp17
kh(p)).
This is the musically expected ordering, since two pure compositions belonging to the same r¯ ag should
be closer than two pure compositions belonging to different r¯ ags.
Overall, thesethreecasesdemonstratethattheproposedmodifieddistanceaddressestheshortcomings
of cosine similarity and ordinary Euclidean distance. The octave-summed representation reduces octave-
position dependency, normalization reduces the effect of composition length, and the r¯ ag-aware weighting
scheme emphasizes musically important note positions.
In Section 5.4, we demonstrate the effectiveness of the proposed weighted distance measure by evalu-
ating the modified dataset using ak-nearest-neighbor classifier.
5.4.R¯ ag Prediction Evaluation using K-Nearest-Neighbour Classifier.In this section, we eval-
uate the effectiveness of the proposed distance measure for r¯ ag prediction using thek-nearest-neighbour
(kNN) classifier. We compare the classification accuracy obtained using ordinary Euclidean distance and
the proposed weighted Euclidean distance for different values ofk.
For this experiment, we select compositions belonging to three r¯ ags, namelyBhairavi,Bihag, and
Khamaj. These are among the most frequently occurring r¯ ags in the dataset. AlthoughKirtanis also
one of the most frequent labels, it is excluded from this experiment because it does not correspond
to a specific Hindustani classical r¯ ag and therefore does not follow a fixed ¯ArohandAvrohstructure.
Thus, the classification task is formulated as a three-class r¯ ag prediction problem using186compositions
represented by12cumulative note-frequency features.
The dataset is divided into a training set and a test set using a75 : 25split. The training set is
used to build thekNN classifier, while the test set is used to compute the prediction accuracy. For the
weighted Euclidean distance, we consider two weight distributions, namely90 : 10and80 : 20. In each
case, r¯ ag-specific weight vectors are constructed using the prescribed ¯ArohandAvrohnote sets of the
selected r¯ ags, as shown in Table 8.
Table 12 presents the accuracy scores obtained for different values ofkusing ordinary Euclidean
distance and the proposed weighted Euclidean distance.
For each value ofk, the Euclidean accuracy is obtained using the ordinary Euclidean distance on the
modified12-dimensional representation. For the weighted Euclidean distance, we evaluate the classifier
using r¯ ag-specific weight vectors and two different weight distributions, namely90 : 10and80 : 20. The
best accuracy values obtained among the weighted-distance settings are shown in bold.
The motivation for using r¯ ag-specific weight vectors is that the importance of a note depends on the
r¯ ag under consideration. A note that is musically important for one r¯ ag may not be equally important
for another r¯ ag. Therefore, while computing the weighted distance between a test composition and the
training compositions, the distance is evaluated with respect to the weight vectors of the candidate r¯ ags.
The nearest neighbours are then identified based on the resulting weighted distances, and the r¯ ag label
is predicted using majority voting among theknearest neighbours.
From Table 12, it can be observed that the proposed weighted Euclidean distance generally improves
thekNN classification accuracy compared to ordinary Euclidean distance. For example, whenk= 4,
the ordinary Euclidean distance gives an accuracy of0.7872, whereas the weighted Euclidean distance
with the90 : 10distribution and the Bihag weight vector gives the highest accuracy of0.8511. Similarly,

SHORT TITLE 15
kEuclidean
Acc.R¯ agWeighted Euclidean
(Nr:Nnr)
90:10 80:20
3Bhairavi 0.7447 0.7021
0.7660 Bihag0.7872 0.7660
Khamaj 0.7234 0.7660
4Bhairavi 0.7660 0.7660
0.7872 Bihag0.8511 0.8298
Khamaj 0.7660 0.7872
5Bhairavi 0.6383 0.7447
0.7660 Bihag0.8085 0.8085
Khamaj 0.7660 0.7872
6Bhairavi 0.7021 0.7446
0.7446 Bihag0.8085 0.8510
Khamaj 0.7234 0.8290
7Bhairavi 0.6600 0.6600
0.7021 Bihag 0.7021 0.7660
Khamaj 0.7447 0.8085
Table 12.Comparison ofkNN accuracy scores for different values ofkusing ordinary
Euclidean distance and weighted Euclidean distance.
fork= 6, the weighted Euclidean distance with the80 : 20distribution achieves an accuracy of0.8510,
which is higher than the corresponding Euclidean accuracy of0.7446.
These results indicate that incorporating r¯ ag-specific note importance into the distance computation
improves the discriminative ability of thekNN classifier. Thus, the proposed weighted Euclidean dis-
tance is more suitable than ordinary Euclidean distance for r¯ ag prediction using symbolic note-frequency
representations.
6.Conclusions and Future Work
In this paper, we presented a symbolic music-based approach for r¯ ag classification of Rabindra Sangeet
compositions. The study was motivated by the fact that Rabindra Sangeet often draws upon Hindustani
r¯ ag structures while allowing considerable creative freedom in melodic treatment. This makes automatic
r¯ ag identification challenging, especially when standard similarity and distance measures are directly
applied to note-frequency representations.
To address this problem, we prepared a supervised symbolic dataset of r¯ ag-labelled Tagore songs from
Swarabitan. Each composition was represented through its note-frequency distribution, initially over
36notes spanning three octaves. The experimental analysis showed that conventional cosine similarity
and Euclidean distance may sometimes produce misleading similarity scores. In particular, compositions
belonging to the same r¯ ag may appear dissimilar due to octave-wise non-overlapping note positions, while
compositions belonging to different r¯ ags may appear highly similar because of comparable note-frequency
patterns.
To overcome these limitations, we modified the feature representation by aggregating notes across oc-
taves into a12-note representation. Furthermore, we proposed aWeighted Euclidean Distancemeasure
that assigns higher importance to notes belonging to the characteristic ¯Aroh and Avroh of a r¯ ag, and
lower importance to the remaining notes. The proposed distance measure provides a musically informed
way of comparing compositions, as it incorporates r¯ ag-specific note importance rather than treating all
note-frequency differences equally. The usefulness of the proposed measure was demonstrated through
its application in ak-nearest-neighbor classifier, where it helped improve the distinction between com-
positions belonging to the same and different r¯ ags.

16 CHANDAN MISRA AND SWARUP CHATTOPADHYAY
The present work opens several directions for future research. First, the dataset can be expanded
by including a larger number of compositions from the completeSwarabitancollection, so that r¯ ags
with fewer available samples can also be studied more effectively. Second, the weight values used in the
proposed distance measure can be learned automatically from data instead of being assigned manually.
This may help obtain more adaptive and r¯ ag-specific weighting schemes. Third, future work may include
sequential melodic features such as note transitions, phrase patterns, and Markov or hidden Markov
model-based representations, since r¯ ag identity is often expressed not only through note frequency but
also through melodic movement.
In addition, the proposed symbolic framework can be extended by incorporating other musical at-
tributes such as t¯ al, phrase structure, ornamentation, and melodic motifs. Finally, the method can be
compared with other machine learning and deep learning models, as well as with audio-based r¯ ag recog-
nition approaches, to develop a more comprehensive framework for computational analysis of Rabindra
Sangeet.
References
[1] S. Mukherjee, “Tagore and the baul folk: A coalition in music styles,”International Journal of Innovative Research
and Advanced Studies, vol. 4, pp. 54–57, Aug. 2017. IJIRAS.
[2] B. Mukerji, “Music and rabindranath,”India International Centre Quarterly, vol. 38, no. 1, pp. 42–51, 2011.
[3] R. Som,Rabindranath Tagore: The singer & his song. Penguin uk, 2017.
[4] B. Trivedi, R. G. Kelkar, and J. T. Jasakiya, “Rabindra sangeet in context with indian classical music,”Swar Sindhu:
National Peer-Reviewed/Refereed Journal of Music, vol. 9, p. 57, June 2021. UGC CARE listed journal.
[5] Geetabitan, “Sangeet chinta: Lyric and music summary.”https://www.geetabitan.com/sangeetchinta/
lyric-music-summary.html. Accessed: 2026-06-27.
[6] A. Srinivasamurthy, S. Gulati, R. C. Repetto, and X. Serra, “Saraga: Open datasets for research on indian art music,”
Empirical Musicology Review, vol. 16, no. 1, pp. 85–98, 2021.
[7] S. Gulati, J. Serrà, K. K. Ganguli, S. Sentürk, and X. Serra, “Indian art music raga recognition dataset (audio),”
Zenodo, 2016.
[8] A. Shankar, G. Plaja-Roglans, T. Nuttall, M. Rocamora, and X. Serra, “Saraga audiovisual: A large multimodal open
data collection for the analysis of carnatic music.,” inISMIR, pp. 61–69, 2024.
[9] S. Chowdhuri, “Phononet: multi-stage deep neural networks for raga identification in hindustani classical music,” in
Proceedings of the 2019 on international conference on multimedia retrieval, pp. 197–201, 2019.
[10] D. P. Shah, N. M. Jagtap, P. T. Talekar, and K. Gawande, “Raga recognition in indian classical music using deep
learning,” inInternational Conference on Computational Intelligence in Music, Sound, Art and Design (Part of
EvoStar), pp. 248–263, Springer, 2021.
[11] S. Gulati, J. Serrà Julià, K. K. Ganguli, S. Sentürk, and X. Serra, “Time-delayed melody surfaces for r¯ aga recognition,”
inDevaney J, Mandel MI, Turnbull D, Tzanetakis G, editors. ISMIR 2016. Proceedings of the 17th International
Society for Music Information Retrieval Conference; 2016 Aug 7-11; New York City (NY).[Canada]: ISMIR; 2016.
p. 751-7., International Society for Music Information Retrieval (ISMIR), 2016.
[12] P. Kirthika and R. Chattamvelli, “A review of raga based music classification and music information retrieval (mir),” in
2012 IEEE International Conference on Engineering Education: Innovative Practices and Future Trends (AICERA),
pp. 1–5, IEEE, 2012.
(Chandan Misra)XIM University, School of Computer Science and Engineering, 752050, Bhubaneswar,
India
Email address, Chandan Misra:chandan@xim.edu.in
(SwarupChattopadhyay)XIM University, School of Computer Science and Engineering, 752050, Bhubaneswar,
India
Email address, Swarup Chattopadhyay:swarupc@xim.edu.in
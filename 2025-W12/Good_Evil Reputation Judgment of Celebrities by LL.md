# Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation

**Authors**: Rikuto Tsuchida, Hibiki Yokoyama, Takehito Utsuro

**Published**: 2025-03-18 16:15:55

**PDF URL**: [http://arxiv.org/pdf/2503.14382v1](http://arxiv.org/pdf/2503.14382v1)

## Abstract
The purpose of this paper is to examine whether large language models (LLMs)
can understand what is good and evil with respect to judging good/evil
reputation of celebrities. Specifically, we first apply a large language model
(namely, ChatGPT) to the task of collecting sentences that mention the target
celebrity from articles about celebrities on Web pages. Next, the collected
sentences are categorized based on their contents by ChatGPT, where ChatGPT
assigns a category name to each of those categories. Those assigned category
names are referred to as "aspects" of each celebrity. Then, by applying the
framework of retrieval augmented generation (RAG), we show that the large
language model is quite effective in the task of judging good/evil reputation
of aspects and descriptions of each celebrity. Finally, also in terms of
proving the advantages of the proposed method over existing services
incorporating RAG functions, we show that the proposed method of judging
good/evil of aspects/descriptions of each celebrity significantly outperform an
existing service incorporating RAG functions.

## Full Text


<!-- PDF content starts -->

arXiv:2503.14382v1  [cs.CL]  18 Mar 2025Good/Evil Reputation Judgment of Celebrities
by LLMs via Retrieval Augmented Generation
Rikuto Tsuchida Hibiki Yokoyama Takehito Utsuro
Deg. Prog. Sys.&Inf. Eng., Grad. Sch. Sci.&Tech., Universi ty of Tsukuba
{s2110466, s2320808}_@_u.tsukuba.ac.jp ,utsuro_@_iit.tsukuba.ac.jp
Abstract
The purpose of this paper is to examine
whether large language models (LLMs) can
understand what is good and evil with respect
to judging good/evil reputation of celebri-
ties. Speciﬁcally, we ﬁrst apply a large lan-
guage model (namely, ChatGPT) to the task
of collecting sentences that mention the tar-
get celebrity from articles about celebrities
on Web pages. Next, the collected sentences
are categorized based on their contents by
ChatGPT, where ChatGPT assigns a category
name to each of those categories. Those as-
signed category names are referred to as “as-
pects” of each celebrity. Then, by apply-
ing the framework of retrieval augmented gen-
eration (RAG), we show that the large lan-
guage model is quite effective in the task of
judging good/evil reputation of aspects and
descriptions of each celebrity. Finally, also
in terms of proving the advantages of the
proposed method over existing services in-
corporating RAG functions, we show that
the proposed method of judging good/evil of
aspects/descriptions of each celebrity signiﬁ-
cantly outperform an existing service incorpo-
rating RAG functions.
1 Introduction
This paper proposes a method of judging good/evil
reputation of celebrities based on information col-
lected from Web pages by employing large lan-
guage models (LLMs, in particular ChatGPT)
via retrieval augmented generation (RAG) ( Lewis
et al. ,2020 ). Efforts have been made to ensure
that ChatGPT does not output insults about peo-
ple, but it remains unclear whether it can distin-
guish between good and evil or understand the de-
gree of evilness in its content. Thus, this paper
shows that ChatGPT can judge good/evil reputa-
tion of celebrities based on their aspects and de-
scriptions. We also show that ChatGPT can further
distinguish the degree of evilness such as illegal orlegal but unethical.
However, gathering information about celebri-
ties using ChatGPT alone does not necessarily
yield the latest information. This is because
ChatGPT cannot grasp events that occurred after
its training data cut-off. Therefore, this paper
proposes a method where information obtained
from external sources is provided to a large lan-
guage model such as ChatGPT, allowing it to
consider events that happened after its training
data cut-off when generating responses. This ap-
proach is known as retrieval augmented genera-
tion (RAG) ( Lewis et al. ,2020 ). Also in pre-
vious research ( Yokoyama et al. ,2024 ), impres-
sions regarding celebrities were collected and ag-
gregated from posts on platform X. The aspects
refers to what the impressions are about. In con-
trast, this paper proposes a novel method of ex-
tracting aspects of celebrities from Web pages and
aggregating detailed descriptions of those aspects.
Through the experimental evaluation, in terms of
the variety of aggregated aspects/descriptions, we
showed that this novel approach is especially ef-
fective in the case of celebrities who have encoun-
tered a certain kind of scandals.
Finally, a comparison is made with Microsoft
Copilot1developed by Microsoft Corporation. Mi-
crosoft Copilot combines ChatGPT with RAG
technology, while its comparison with the method
proposed in this paper reveals the following:
i.e., the proposed method outperforms Microsoft
Copilot both in the number of aggregated as-
pects/descriptions of celebrities as well as their
accuracy. This is mainly because the proposed
method collects much larger number of Web pages
before aggregation and then identiﬁes much larger
number of aspects/descriptions compared with Mi-
crosoft Copilot.
The followings give the contribution of this pa
1https://www.microsoft.com/ja-jp/microsoft-copilot/o rganizations

per:
• We proposed a method for widely collecting
information on celebrities.
• With the help of RAG function, we showed
that ChatGPT can distinguish between good
and evil and understand the degree of evilness
of celebrities based on their aspects and de-
scriptions.
• We showed the advantages of the proposed
method compared to existing services, Mi-
crosoft Copilot by comparing results of
good/evil reputation judgment.
2 Related Work
Research on information aggregation of celebrities
includes Yokoyama et al. (2024 ) as mentioned in
the previous section, as well as research on de-
termining the relationship between celebrities and
impressions in Microblogs ( Nozaki et al. ,2022 ),
and the work on extracting impressions about
celebrities’ aspects from Microblogs ( Sugawara
and Utsuro ,2022 ). However, these studies do not
address celebrities who have been involved in past
controversies. This paper resolves the issues that
arise when such celebrities are included as sub-
jects.
In this paper, we utilize retrieval augmented gen-
eration (RAG) ( Lewis et al. ,2020 ), which helps
to reduce hallucinations in large language models
(LLMs) and stabilize output by referencing exter-
nally obtained information. Related research on
RAG includes retrieval-augmented language mod-
els ( Ram et al. ,2023 ) and improvements in re-
liability, adaptability, and attribution in retrieval-
augmented language models ( Asai et al. ,2024 ).
Additionally, research related to ChatGPT encom-
passes entity linking ( Peeters and Bizer ,2023 ), di-
alogue analysis ( Finch et al. ,2023 ), and extractive
summarization ( Zhang et al. ,2023 ).
Furthermore, an important feature of this paper
is the judgment between good or evil of celebrities’
careers using LLMs. Research involving LLMs
and the legal ﬁeld includes studies verifying the
effectiveness of LLMs in the legal ﬁeld ( Shaurya
et al. ,2023 ).3 Aggregating Impressions on
Celebrities and their Reasons from
Microblog Posts and Web Search
Pages ( Yokoyama et al. ,2024 )
This paper builds on prior research outlined in Sec-
tion 1, speciﬁcally the evaluation of RAG in ag-
gregating reasons for posts expressing impressions
about celebrities ( Yokoyama et al. ,2024 ). In that
study, posts containing impressions on celebrities
were collected from X using the large language
model ChatGPT. From these collected posts, im-
pressions regarding what aspects of celebrities are
extracted as the celebrity’s aspect. The reasons
for the “celebrity’s aspect + impression” extracted
at this stage are collected and aggregated using
RAG ( Lewis et al. ,2020 ). Web pages are searched
using “ celebrity’s aspect + impression ” as queries,
and the resulting Web pages are provided to Chat-
GPT to identify reasons for the impressions from
them. The reason for using RAG ( Lewis et al. ,
2020 ) in this process is that the information ob-
tained solely from posts rarely contains a detailed
explanation of the reasons, even if impressions are
present. Therefore, it is necessary to obtain infor-
mation about the reputation of celebrities from ex-
ternal Web pages. The details are shown in Fig-
ure3in the Appendix A.1.
However, in the previous study ( Yokoyama
et al. ,2024 ), only celebrities who had not caused
any scandal in the past were considered. There-
fore, when data was collected and aggregated on a
wider variety of celebrities, it was found that posts
related to celebrities who had encountered scan-
dals in the past often lacked a clear aspects, mak-
ing extraction difﬁcult. This situation is illustrated
in Figure 4in the Appendix A.1. In other words,
the approach of the previous study is only effec-
tive for individuals with no past troubles nor scan-
dals, but it is considered ineffective for celebrities
who have negative impressions such as troubles
and scandals in their past.
Addressing these issues, this paper aims to
study celebrities who have encountered scandals
in the past and to extract the aspects that are not
clearly depicted in posts.

4 Proposed Approach: Skipping
Microblog Posts but Aggregating
Celebrities’ Aspects/Descriptions
directly collected from Web Pages
In order to solve the problems caused in the previ-
ous study mentioned in the previous section, this
paper proposes a method that retrieves all aspects
related to celebrities from Web pages, without ex-
tracting them from microblog posts.
4.1 Collecting Web Pages
First, 20 Web pages are collected by search-
ing with “ celebrity name ”. In the previous
study ( Yokoyama et al. ,2024 ), the query for the
search was “ celebrity’s aspects + impression ”, so
only Web pages related to the information col-
lected in the microblog posts could be retrieved.
However, by revising the query as above, a wide
range of information can be retrieved.
Table 1shows the list of celebrities targeted
in this paper. There are ten Japanese celebrities,
ﬁve of whom were originally included in the pre-
vious study ( Yokoyama et al. ,2024 ) and have no
previous problems. The remaining ﬁve are new
celebrities for study in this paper, who have had
problems in the past and have been inactive. In
addition, ﬁve additional non-Japanese celebrities
who had caused problems were also targeted. For
all of those celebrities for study in this paper, we
examined Japanese Web pages only, but not the
Web pages of any other language including En-
glish Web pages2.
4.2 Extracting and Aggregating Celebrities’
Aspects and their Descriptions
Next, from the large number of sentences on the
collected Web pages, sentence that mention the tar-
get celebrity are collected, and since some of the
collected sentence have overlapping contents, so
they are categorized according to the contents. In
this process, we ask ChatGPT to determine “what
the topic is” about the celebrity and name it as a
category name. The category name generated here
2When collecting Web pages with each of the ﬁve non-
Japanese celebrities’ names as the query, we use the Japanes e
katakana characters used for those foreign names, i.e., “
ショーン・コムズ ” (Shoon Komuzu) for “Sean Combs”,
“ケヴィン・スペイシー ” (Kevin Speisii) for “Kevin
Spacey”, “ジョニー・デップ ” (Jonii Deppu) for “Johnny
Depp”, “ウィノナ・ライダー ” (Uinona Raidaa) for
“Winona Ryder”, and “ ジャスティン・ティンバーレイ
ク” (Jasutein Teinbaareiku) for “Justin Timberlake”.Yokoyama et al. (2024 )only in this paper
Ryosuke Yamada Huwa-chan
Kazunari Ninomiya Pierre Taki
Huma Kikuchi Yuichi Nakamura
Syun Oguri Noriyuki Makihara
Go Ayano Hiroyuki Miyasako
— Sean Combs
— Kevin Spacey
— Johnny Depp
— Winona Ryder
— Justin Timberlake
Table 1: Celebrities for Study in This Paper
celebrity’s
namerecall precision
Huwa-chan 0.75 (6/8) 1.00 (6/6)
Pierre Taki 0.53 (8/15) 1.00 (8/8)
Yuichi Nakamura 0.64 (9/14) 0.90 (9/10)
Hiroyuki Miyasako 0.60 (6/10) 1.00 (6/6)
Noriyuki Makihara 0.82 (9/11) 1.00 (9/9)
Ryosuke Yamada 0.78 (7/9) 1.00 (7/7)
Syun Oguri 0.50 (5/10) 1.00 (5/5)
Go Ayano 0.71 (10/14) 0.83 (10/12)
Kazunari Ninomiya 0.63 (5/8) 0.83 (5/6)
Huma Kikuchi 0.86 (6/7) 1.00 (6/6)
Sean Combs 0.86 (6/7) 1.00 (6/6)
Kevin Spacey 0.63 (5/8) 0.83 (5/6)
Johnny Depp 0.60 (6/10) 1.00 (6/6)
Winona Ryder 0.83 (5/6) 1.00 (5/5)
Justin Timberlake 0.78 (7/9) 0.78 (7/9)
macro average 0.70 0.94
Table 2: Evaluation Results of Extracted and Aggre-
gated Aspects and Descriptions
becomes “celebrity’s aspects”. Finally, various de-
scriptions related to the celebrity’s aspects are ag-
gregated.
The entire process of collecting sentences that
mention celebrities, categorizing them and creat-
ing category names, and extracting the celebrity’s
aspects is done using ChatGPT. The model of
ChatGPT used is gpt-4o, and the same model is
used in all the later situations where ChatGPT is
used. Figure 1shows the process from collection
of Web pages to extraction of aspects for “Justin
Timberlake” as an example of a target celebrity.
4.3 Evaluation
The results obtained by ChatGPT are compared
with manually extracted and aggregated reference
aspects and descriptions for each celebrity name.
ChatGPT’s results were considered a match if both
the aspects and descriptions were identical or sim-

Figure 1: An Example of Extracting and Aggregating Aspects a nd Descriptions for “Justin Timberlake”
—Aspects/Impressions/Reasons
byYokoyama et al. (2024 )Aspects/Descriptions
by the Proposed Method
Aspects/Descriptions and
Aspects/Impressions/Reasons
overlapping
between the two methodsremarks/criticism, rants/criticism,
ﬂaming/criticism, ﬂaming/sympathy,
dislike/sympathy, dislike/criticism,
disliked/sympathy, disappeared/criticisminappropriate remarks and hiatus
Aspects/descriptions and
Aspects/Impressions/Reasons
not overlapping
between the two methodsNonecareer and activities,
language skills and educational background,
relationships with friends
fashion and inﬂuence,
media appearances
Table 3: Aspects/Impressions/Reasons by Yokoyama et al. (2024 ) and Aspects/Descriptions by the Proposed
Method (by ChatGPT, for “Huwa-chan”)
ilar in content to reference3.
When the set of the human-handled aspects
and descriptions is R(c)and the set of the Chat-
GPT aspects and descriptions is S(c)for a certain
celebrity c, we deﬁne recall and precision as the
following formulas.
recall=|R(c)∩S(c)|/|R(c)|,
precision =|R(c)∩S(c)|/|S(c)|
Evaluation results are shown in Table 2. Recall
is approximately 70 %, and precision is 90 % or
more, indicating that the extraction and aggrega-
tion are performed accurately.
4.4 Comparison with Yokoyama et al. (2024 )
Some of the aspects and descriptions extracted
by the proposed method overlap with the aspects,
3In all the results of the experiment, when the aspect is
identical or similar to the reference, it is also the case for the
description.
Figure 2: Good/Evil Judgment of a Celebrity’s As-
pects/Descriptions (e.g., for “Justin Timberlake”)
impressions, and reasons extracted by Yokoyama
et al. (2024 ). Here, however, there exist those that
are extracted only by the proposed method, while
conversely, others are extracted only by Yokoyama
et al. (2024 ). Therefore, we manually map the “as-
pects + impressions + reasons for impressions” ob-
tained by Yokoyama et al. (2024 ) to those “aspects
+ descriptions” obtained by the proposed method,

—Aspects/Impressions/Reasons
byYokoyama et al. (2024 )Aspects/Descriptions
by the Proposed Method
Aspects/Descriptions and
Aspects/Impressions/Reasons
overlapping
between the two methods5.67 2.00
Aspects/Descriptions and
Aspects/Impressions/Reasons
thatnot overlapping
between the two methods2.67 5.83
Total 8.33 7.83
Table 4: Numbers of Aspects/Impressions/Reasons by Yokoyama et al. (2024 ) and Aspects/Descriptions by the
Proposed Method (by ChatGPT, averaged over 10 Celebrities)
celebrity’s nameChatGPT Microsoft Copilot
zero-shot few-shot zero-shot few-shot
Huwa-chan 1.00 (6/6) 1.00 (6/6) 0.70 (7/10) 0.80 (8/10)
Pierre Taki 0.86 (7/8) 1.00 (8/8) 0.60 (3/5) 0.60 (3/5)
Yuichi Nakamura 0.78 (7/9) 1.00 (9/9) 0.83 (5/6) 0.83 (5/6)
Hiroyuki Miyasako 0.83 (5/6) 0.67 (4/6) 0.80(4/5) 0.80 (4/5)
Noriyuki Makihara 1.00 (9/9) 1.00 (9/9) 0.50 (2/4) 0.50 (2/4)
Ryosuke Yamada 1.00 (7/7) 1.00 (7/7) — —
Syun Oguri 1.00 (5/5) 1.00 (5/5) — —
Go Ayano 1.00 (10/10) 1.00 (10/10) — —
Kazunari Ninomiya 1.00 (5/5) 1.00 (5/5) — —
Huma Kikuchi 1.00 (6/6) 1.00 (6/6) — —
macro average 0.95 0.97 0.67 0.71
Table 5: Evaluation Results of Good/Evil Judgment of a Celeb rity’s Aspects/Descriptions
and examined their overlap.
Table 3shows the result obtained for “Huwa-
chan”. “Huwa-chan” is a Japanese comedian and
has been criticized for his violent outbursts against
other celebrities on social networking sites and is
currently on hiatus. Table 3shows that there are
no aspect, impression, nor reason that can be ob-
tained only by Yokoyama et al. (2024 ). On the
contrary, there exist plenty of aspects and descrip-
tions that can be obtained only by the proposed
method, which means that the method proposed
in this paper is capable of more widely collecting
celebrities’ aspects and their descriptions. The de-
tails are provided in Figure 5of Appendix A.3
The overall result of the mapping for 10 celebri-
ties is shown in Table 4. On the side of Yokoyama
et al. (2024 ), the number of overlapping aspects,
impressions, and reasons is 5.67, which is much
larger than that of not overlapping ones. On
the side of the proposed method, on the other
hand, the magnitude relation of overlapping and
not overlapping numbers are opposite, where the
not overlapping number is much larger than theoverlapping number. This result again indicates
that the method proposed in this paper is capable
of more widely collecting celebrities’ aspects and
their descriptions.
5 Good/Evil Judgment of a Celebrity’s
Aspects/Descriptions
Next, a judgment between good and evil is made
for the aspects extracted in Section 4.2. The pur-
pose of this judgment is to determine whether a
large-scale language model, which has not been
trained speciﬁcally for legal knowledge, can accu-
rately make ﬁne-grained judgments.
5.1 The Procedure
Judgment by the ChatGPT is made in two stages
as shown in Figure 2. First, the celebrities’ aspects
and descriptions are classiﬁed into two categories:
“evil” or “not particularly evil”, and then the as-
pects and descriptions judged to be “evil” are fur-
ther classiﬁed into three categories as below:
• not particularly evil
• evil

— predicted by ChatGPT
— illegal legal but unethicallegal and ethical
but unpopular and
criticizednot particularly evil total
referenceillegal2
Makihara/Taki0 0 0 2
legal
but unethical1
Miyasako2
Huwa/Nakamaru0 0 3
legal and ethical
but unpopular and
criticized0 01
Miyasako0 1
not particularly evil 0 1 0 64 65
total 3 3 1 64 71
Table 6: Confusion Matrix of Good/Evil Judgment of Celebrit ies’ Aspects/Descriptions (for 10 Celebrities)
–illegal
–legal but unethical
–legal and ethical
but unpopular and criticized
In particular, with regard to the three categories
ofevil,illegal is deﬁned to be a person who clearly
violates a law, legal but unethical is deﬁned to be
a person who does not violate any law but does
say or does something ethically problematic and
is criticized by the public, and legal and ethical
but unpopular and criticized is deﬁned to be a per-
son who does nothing particularly evil but has a
poor reputation among others. We will investigate
whether ChatGPT can recognize the clear distinc-
tion among those deﬁnitions.
In addition, before judgment, ChatGPT is input
with the aspects and descriptions of the celebri-
ties extracted and aggregated in section 4.2, and
the judgment is made by referring to those infor-
mation. Figure 2shows the judgment made on
“Scandals and legal problems”, one of the aspects
extracted when “Justin Timberlake” was the tar-
get. In Figure 2, the aspect of “Scandals and legal
problems” are entered into ChatGPT together with
the aggregated description, and ChatGPT judges
where it ﬁts in the classiﬁcation by referring to the
aggregated description. In the ﬁrst stage, “Scan-
dals and legal problems” was judged to be “evil” at
the ﬁrst stage, and as the result of the second stage
of the judgment, “Scandals and legal problems” of
“Justin Timberlake” was judged to be “illegal”.
These decisions were made by zero-shot and
few-shot. The examples used in the prompt are not
related to the actual celebrities for study. One ex-
ample was created for each of the four categories,
for a total of four examples and are used as the
four-shot.5.2 Evaluation Results
For each of the extracted and aggregated aspects
and descriptions from an celebrity, we examined
whether the results of the ChatGPT matched the re-
sults of manual reference for good/evil judgments.
Table 5shows the evaluation results for each of the
10 celebrities for study. In addition, a confusion
matrix summarizing the results for the 10 celebri-
ties is shown in Table 6. In Table 6, the names of
the celebrities whose aspects/descriptions are clas-
siﬁed into one of the three “evil” categories are
shown. Out of the 10 celebrities, all of the ﬁve
shown in Table 6have been suspended from the en-
tertainment industry due to scandals. Thus, it can
be said that accurate judgments have been made4.
6 Good/Evil Judgment per Celebrity
(not per aspect/description)
In the previous section, we extract and aggregate
aspects and descriptions of the celebrities and let
ChatGPT judge the distinction of good and evil of
each aspect. Instead of making this judgment for
each aspect, this section performs good/evil judg-
ment per celebrity, but not per aspect/description.
Furthermore, we compare the judgment results of
ChatGPT without RAG with those of ChatGPT
with RAG, and show that the use of RAG improves
the accuracy of judgment.
6.1 The Procedure
The information given is each celebrity’s aspects
and descriptions as described in Section 4.2. Each
celebrity’s aspects and descriptions are given to
ChatGPT as prior information as the celebrity’s
4As the few-shot, four more examples were added, for a
total of eight examples as few-shot, where the results did no t
change. Therefore, the results in all tables are on four exam -
ples as few-shot.

celebrity’s name
(The year the scandal occurred,
or “—” in the case of no scandal )without RAG with RAG reference
Yamaguchi Tatsuya (2018/2020) T / illegal T / illegal illegal
Pierre Taki (2019) T / illegal T / illegal illegal
Hiroyuki Miyasako (2019) T / legal but unethical T / legal but unethical legal but unethical
Noriyuki Makihara (2020) T / illegal T / illegal illegal
Johnny Kitagawa (September 2023) T / illegal T / illegal illegal
Hitoshi Matsumoto (December 2023) F/ not particularly evil T / legal but unethical legal but unethical
Sinji Saito (December 2023) F/ not particularly evil T / illegal illegal
Toshihumi Hujimoto (2024) F/ not particularly evil T / illegal illegal
Huwa-chan (2024) F/ not particularly evil T / legal but unethical legal but unethical
Yuichi Nakamura (2024) F/ not particularly evil T / legal but unethical legal but unethical
Masahiro Nakai (2024) F/ not particularly evil F/ legal but unethical illegal
Ryosuke Yamada (—) T / not particularly evil T / not particularly evil not particularly evil
Kazunari Ninomiya (—) T / not particularly evil T / not particularly evil not particularly evil
Go Ayano (—) T / not particularly evil T / not particularly evil not particularly evil
Huma Kikuchi (—) T / not particularly evil T / not particularly evil not particularly evil
Syun Oguri (—) T / not particularly evil T / not particularly evil not particularly evil
Kuro-chan (—) T / legal and ethical
but unpopular and
criticizedT / legal and ethical
but unpopular and
criticizedlegal and ethical
but unpopular
and criticized
Winona Ryder (2001) T / illegal T / illegal illegal
Johnny Depp (2016) T / not particularly evil T / not particularly evil not particularly evil
Kevin Spacey (2017) T / illegal T / illegal illegal
Sean Combs (2024) F/ not particularly evil T / illegal illegal
Justin Timberlake (2024) F/ not particularly evil T / illegal illegal
total accuracy 0.64 (14/22) 0.95 (21/22) —
Table 7: Comparison of with/without RAG in Good/Evil Judgme nt per Celebrity by ChatGPT
reputation. Again, the model used in ChatGPT is
gpt-4o.
The detailed deﬁnitions of the classiﬁcations
used for good/evil judgment are given in Sec-
tion 5. Furthermore, in order to clarify that the
date/year of the scandal have inﬂuence on the
judgment result, several celebrities were added to
the ﬁve celebrities studied in the previous section.
The same survey was also conducted not only on
Japanese but also on foreign celebrities.
6.2 Evaluation Results
The results are shown in Table 7. In Table 7,
if the correct judgment was made in comparison
with reference, a “T” was entered, and otherwise,
a “F” was entered. In Table 7, in the case of
“Justin Timberlake”, the result of judgment with-
out using RAG is “not particularly evil”, which is
different from the correct label given by human
hand. Here, the result of having ChatGPT make
a judgment after providing it with the reputation
on “Justin Timberlake” collected and aggregated
from Web pages, the classiﬁcation result changed
from “not particularly evil” to “illegal”, and theincorrect classiﬁcation was revised to the correct
classiﬁcation by using RAG.
The results of other celebrities show that, for
several celebrities, accurate judgments were made
even without RAG, the results are also the same
with RAG. It is interesting to note here that those
celebrities with correct judgments even without
RAG have encountered their scandals in the year
of 2023 or before. In other words, the ChatGPT
can make accurate judgments about scandals that
occurred within the time period of the data used
to train ChatGPT, because the information is con-
tained in it. On the other hand, ChatGPT has no
knowledge of information after the period of train-
ing data, so it makes incorrect judgments without
any appropriate training information. Since the
model used in this study, gpt-4o, was trained on
data up to October 2023, as shown in Table 7, it
can be seen that the classiﬁcation without RAG is
incorrect for celebrities who had scandals after Oc-
tober 2023.
From the above, it can be said that the use of
RAG has beneﬁts on improving the accuracy of
the judgment. In other words, it can be said that

when collecting and aggregating the reputations of
celebrities, the use of RAG makes it possible to
better guarantee their accuracy and to obtain the
correct output when judging distinction between
good and evil.
7 Comparison with Existing Services
Finally, extracted and aggregated aspects and de-
scriptions in the proposed method are compared
with the results of existing services with RAG
function to demonstrate the advantages of the pro-
posed method. Microsoft Copilot is used for the
comparison. Microsoft Copilot is a combination
of the search engines Bing5and gpt-4o, which
were provided by Microsoft Corporation. In other
words, when gpt-4o is used alone, the output con-
cerning the latest information is not stable and
incorrect answers are given, as described in Sec-
tion3and Section 6. However, this situation can
be avoided through RAG. Microsoft Copilot cre-
ates search queries from the input text, and an-
swers the questions by referring to the information
on the retrieved Web pages. Since Microsoft Copi-
lot uses the RAG ( Lewis et al. ,2020 ) technology,
only the initial input is done by a human, and the
search queries and the selection of Web pages to
be retrieved are all done automatically.
7.1 The Procedure
In the good/evil judgment conducted in Section 5,
Web pages searched with the query “ celebrity
name ” were collected, and aspects and descrip-
tions related to the query celebrity were extracted.
In this context, Microsoft Copilot was ﬁrst tasked
with referencing celebrity Web pages to list top-
ics related to the query celebrity. Then, using
zero-shot or few-shot, it was asked to judge on
those topics. The examples used for few-shot were
the same as those used in Figure 6in the Ap-
pendix A.2. Additionally, a summary of the results
of zero-shot/few-shot is presented in Table 5. Even
though the number of examples used for few-shot
was increased to eight, no changes were observed
in the results, so Table 5records the results from
using four examples. Furthermore, the judgment
by Microsoft Copilot was conducted solely on the
celebrities listed in Table 1who had encountered
some form of scandal.
5https://www.bing.com/?cc=jp7.2 Comparison Results
The percentage of correct answers by Microsoft
Copilot in Table 5shows that the overall percent-
age of correct answers has dropped signiﬁcantly.
One reason for this is that Microsoft Copilot re-
trieves fewer Web pages. Although we instructed
Microsoft Copilot to retrieve 20 Web pages, as in
the case of the proposed method, it actually re-
trieved only about 2 or 3 pages, which may have re-
sulted in biased information. In addition, although
detailed information on each topic is necessary for
an accurate distinction between good and evil, a
few Web pages were not enough to obtain sufﬁ-
cient information.
On the other hand, it took only a few seconds
for Microsoft Copilot to collect information on
one celebrity, while it took about 10 minutes for
our proposed method. This is because this paper’s
method requires reading sentences from a large
number of Web pages, which is time-consuming,
while Microsoft Copilot is designed to be accessed
from all over the world, so it cannot read as many
as 20 pages, but only 1 or 2 pages.
In other words, Microsoft Copilot uses RAG
technology so that gpt-4o can refer to the latest in-
formation, and it is much faster than the method in
this paper. However, in terms of depth and breadth
of the collected information, as can be seen from
Table 5, the method in this paper is able to collect
more detailed information.
8 Conclusion
By applying the framework of RAG, we showed
that a large language model (i.e., ChatGPT) is
quite effective in the task of judging good/evil
reputation of aspects and descriptions of each
celebrity. Especially compared with the method
ofYokoyama et al. (2024 ), in terms of the vari-
ety of aggregated aspects/descriptions, we showed
that our novel approach is effective in the case of
celebrities who have encountered a certain kind
of scandals. Finally, we compared our method
with Microsoft Copilot, which provides gpt-4o us-
ing RAG. The results showed that while Microsoft
Copilot was superior in terms of faster output by
utilizing RAG, our method excelled in the explo-
ration of detailed information and the breadth of
information.

References
A. Asai, Z. Zhong, D. Chen, P. W. Koh, L. Zettlemoyer,
H. Hajishirzi, and W. T. Yih. 2024. Reliable, adapt-
able, and attributable language models with retrieval .
https://arxiv.org/abs/2403.03187.
S. E. Finch, E. S. Paek, and J. D. Choi. 2023. Lever-
aging large language models for automated dialogue
analysis. In Proc. 24th SIGDIAL , pages 202–215.
P. Lewis, E. Perez, et al. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. In
Proc. 34th NeurIPS , pages 483–498.
Y . Nozaki, K. Sugawara, Y . Zenimoto, and T. Utsuro.
2022. Tweet review mining focusing on celebrities
by MRC based on BERT. In Proc. 36th PACLIC ,
pages 757–766.
R. Peeters and C. Bizer. 2023. Using ChatGPT for en-
tity matching. arXiv preprint arXiv:2305.03423 .
O. Ram, Y . Levine, I. Dalmedigos, D. Muhlgay,
A. Shashua, K. Leyton-Brown, and Y . Shoham.
2023. In-context retrieval-augmented language
models . https://arxiv.org/abs/2302.00083.
V . Shaurya, Z. Atharva, D. Somsubhra, S. Anurag,
B. Upal, N. Shubham Kumar, G. Shouvik, R. Kous-
tav, and G. Kripabandhu. 2023. LLMs – the good,
the bad or the indispensable?: A use case on legal
statute prediction and legal judgment prediction on
Indian court cases. In Findings of EMNLP , pages
12451–12474.
K. Sugawara and T. Utsuro. 2022. Developing a dataset
for mining reviews in tweets focusing on celebrities’
aspects. In Proc. 7th ABCSS , pages 466–472.
H. Yokoyama, R. Tsuchida, K. Buma, S. Miyakawa,
T. Utsuro, and M. Yshioka. 2024. Aggregating im-
pressions on celebrities and their reasons from mi-
croblog posts and web search pages . In Proc. 3rd
Workshop on Knowledge Augmented Methods for
NLP, pages 59–72.
H. Zhang, X. Liu, and J. Zhang. 2023. SummIt: Iter-
ative text summarization via ChatGPT. In Findings
of EMNLP , pages 10644–10657.
A Appendix
A.1 Previous study
The method used in previous research ( Yokoyama
et al. ,2024 ) is illustrated in Figure 3.
Figure 4uses “Justin Timberlake” as an exam-
ple to show that posts about people who have
caused problems in the past do not clearly indicate
the aspects.A.2 Set of prompts used
Figure 6shows the examples and concrete
prompts used in Section 5.1.
The exact prompts provided are shown in Fig-
ure7on judgment by Microsoft Copilot, and the
results of the evaluation focusing on the celebrity
“Justin Timberlake” are shown in Figure 8.
A.3 Correspondence between
Aspects/Descriptions obtained by the
Method of This Paper and
Aspects/Impressions/Reasons obtained
byYokoyama et al. (2024 )
Figure 5shows correspondence between as-
pects/descriptions obtained by the method of this
Paper and aspects/impressions/reasons obtained
byYokoyama et al. (2024 ).
A.4 The results of judgment of good or evil
by ChatGPT and Microsoft Copilot
Table 8and Table 9show a detailed view of the
ﬁve celebrities who have encountered a certain
kind of scandals, down to their aspects and the
results of their judgments in Section 5. More-
over, Table 8and Table 9show the aspects of
each celebrity that Microsoft Copilot identiﬁed
and their judgment results evaluated in Section 7.

celebrity
nameChatGPT Microsoft Copilot
aspects results of judgment aspects results of judgment
Huwa-chaninappropriate remarks
and hiatusT / legal but unethical past problematic behavior T / legal but unethical
career and activities T / not particularly evil freelance activities T / not particularly evil
language skills and ed-
ucational backgroundT / not particularly evil pro wrestling debut T / not particularly evil
relationships with
friendsT / not particularly evil statements on social media
can cause controversyT / legal and ethical but
unpopular and criticized
fashion and inﬂuence T / not particularly evil unique fashion F/ legal and ethical but
unpopular and criticized
(→not particularly evil)
media appearances T / not particularly evil activities on TV shows T / not particularly evil
moving abroad and life there T / not particularly evil
commercial appearances and
their impactT / not particularly evil
the creation and success of a
YouTube channelT / not particularly evil
dismissed due to trouble with
the talent agencyF/ illegal ( →legal
and ethical but unpopu-
lar and criticized)
Pierre Takidrug incident and its
impactT / illegal violation of the narcotics con-
trol actT / illegal
musical activities T / not particularly evil musical activities T / not particularly evil
acting activities and
roles in productionsT / not particularly evil family-oriented T / not particularly evil
comeback and recep-
tion after arrestT / not particularly evil media coverage F/ legal but unethical
(→not particularly evil)
impact of television
and moviesT / not particularly evil impact after arrest F/ legal and ethical but
unpopular and criticized
other activities T / not particularly evil (→not particularly evil)
hobbies T / not particularly evil
writing activities T / not particularly evil
Table 8: Results of Good/Evil Judgments on Celebrities’ Asp ects and Descriptions (1)

celebrity
nameChatGPT Microsoft Copilot
aspects results of judgment aspects results of judgmen
Yuichi
Nakamaruscandal T / legal but unethical graduated from Waseda
UniversityT / not particularly evil
manga artist T / not particularly evil human beatbox T / not particularly evil
YouTube activities T / not particularly evil YouTube activities T / not particularly evil
activities as a member of
KAT-TUNT / not particularly evil reports of a secret meeting
with a female college stu-
dentT / legal but unethical
being late T / legal and ethical
but unpopular and crit-
icizedbeing late T / legal and ethical
but unpopular and crit-
icized
special skills T / not particularly evil desire to return after inﬁ-
delity reportsF/ legal and ethical
but unpopular and crit-
icized
educational background T / not particularly evil (→ not particularly
evil)
marriage T / not particularly evil
activities on TV show T / not particularly evil
Hiroyuki
MiyasakoYouTube activities T / not particularly evil YouTube activities T / not particularly evil
restaurant management T / not particularly evil activities as a businessper-
sonT / not particularly evil
problem of underground
business dealingsF/ illegal (→legal but
unethical )problem of underground
business dealingsF/ illegal (→legal but
unethical )
human relationships T / not particularly evil allegations of inﬁdelity T / legal but unethical
music activities T / not particularly evil statements on YouTube T / legal and ethical
but unpopular and crit-
icized
TV appearances and
comebackF/ legal but unethi-
cal (→not particularly
evil)
Noriyuki
Makiharamusical activities and hit
songsT / not particularly evil musical activities T / not particularly evil
legal problems and arrest
recordT / illegal arrested for violating the
stimulant drugs control
actT / illegal
music production tech-
niquesT / not particularly evil drug use F/ legal but unethical
(→illegal)
evaluation by other artists T / not particularly evil arrest record F/ legal and ethical
but unpopular and crit-
icized
transfer T / not particularly evil (→illegal)
personal information T / not particularly evil
animal lover T / not particularly evil
resumption of activities T / not particularly evil
album and reviews T / not particularly evil
Table 9: Results of Good/Evil Judgments on Celebrities’ Asp ects and Descriptions (2)

Figure 3: Collecting and Aggregating Aspects, Impressions , and Reasons for Impressions in the Previous
Study ( Yokoyama et al. ,2024 )
Figure 4: An Example of Extracting and Aggregating Aspects f or “Justin Timberlake” from Microblog Posts

Figure 5: An Example of Correspondence between Aspects/Des criptions obtained by the Method of This Paper
and Aspects/Impressions/Reasons obtained by Yokoyama et al. (2024 ) for “Huwa-chan”

Figure 6: The Prompt of ChatGPT for Good/Evil Judgment of a Ce lebrity’s Aspects/Descriptions (e.g., for “Justin
Timberlake”)

Figure 7: The Prompt of Microsoft Copilot for Good/Evil Judg ment (e.g., for “Justin Timberlake”)
Figure 8: An Example of Results by Microsoft Copilot for “Jus tin Timberlake”
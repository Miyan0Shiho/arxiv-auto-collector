# ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation

**Authors**: Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, Kijung Shin

**Published**: 2025-11-19 05:39:14

**PDF URL**: [https://arxiv.org/pdf/2511.15141v1](https://arxiv.org/pdf/2511.15141v1)

## Abstract
Recently, large language models (LLMs) have been widely used as recommender systems, owing to their strong reasoning capability and their effectiveness in handling cold-start items. To better adapt LLMs for recommendation, retrieval-augmented generation (RAG) has been incorporated. Most existing RAG methods are user-based, retrieving purchase patterns of users similar to the target user and providing them to the LLM. In this work, we propose ItemRAG, an item-based RAG method for LLM-based recommendation that retrieves relevant items (rather than users) from item-item co-purchase histories. ItemRAG helps LLMs capture co-purchase patterns among items, which are beneficial for recommendations. Especially, our retrieval strategy incorporates semantically similar items to better handle cold-start items and uses co-purchase frequencies to improve the relevance of the retrieved items. Through extensive experiments, we demonstrate that ItemRAG consistently (1) improves the zero-shot LLM-based recommender by up to 43% in Hit-Ratio-1 and (2) outperforms user-based RAG baselines under both standard and cold-start item recommendation settings.

## Full Text


<!-- PDF content starts -->

ItemRAG: Item-Based Retrieval-Augmented Generation
for LLM-Based Recommendation
Sunwoo Kim
KAIST AI
Seoul, South Korea
kswoo97@kaist.ac.krGeon Lee
KAIST AI
Seoul, South Korea
geonlee0325@kaist.ac.krKyungho Kim
KAIST AI
Seoul, South Korea
kkyungho@kaist.ac.krJaemin Yoo
KAIST EE, AI
Daejeon ,South Korea
jaemin@kaist.ac.krKijung Shin
KAIST AI, EE
Seoul, South Korea
kijungs@kaist.ac.kr
Abstract
Recently, large language models (LLMs) have been widely used as
recommender systems, owing to their strong reasoning capabil-
ity and their effectiveness in handling cold-start items. To better
adapt LLMs for recommendation, retrieval-augmented generation
(RAG) has been incorporated. Most existing RAG methods are user-
based, retrieving purchase patterns of users similar to the target
user and providing them to the LLM. In this work, we proposeItem-
RAG, an item-based RAG method for LLM-based recommendation
that retrieves relevant items (rather than users) from item–item
co-purchase histories.ItemRAGhelps LLMs capture co-purchase
patterns among items, which are beneficial for recommendations.
Especially, our retrieval strategy incorporates semantically similar
items to better handle cold-start items and uses co-purchase fre-
quencies to improve the relevance of the retrieved items. Through
extensive experiments, we demonstrate thatItemRAGconsistently
(1) improves the zero-shot LLM-based recommender by up to 43% in
Hit-Ratio@1 and (2) outperforms user-based RAG baselines under
both standard and cold-start item recommendation settings.
CCS Concepts
•Information systems→Recommender systems.
Keywords
large language model, retrieval augmented generation
1 Introduction
Recommender systems are core to modern web services, retriev-
ing items that match user interests from vast item pools [ 12,16].
By inferring users’ preferences from their purchase histories, rec-
ommender systems provide personalized recommendations that
improve user satisfaction and drive business revenue.
In recent years, there have been significant efforts to uselarge
language models(LLMs) as a recommender system [ 5,7]. Owing to
their strong reasoning and zero-/few-shot capabilities, LLM-based
recommenders can handle cold-start items effectively and can also
provide intuitive explanations that improve users’ understanding
of the recommendation results.
To better adapt LLMs for the recommendation tasks, various
retrieval-augmented generation(RAG) techniques have been incor-
porated [ 3,13]. Such approaches typically center onuser-based
retrieval[ 10,15]. Specifically, for a target user, these methods re-
trieve users similar to the target user and insert their purchase
patterns into the recommendation prompt to LLMs. This process
helps LLMs leverage recommendation-relevant knowledge without
requiring expensive fine-tuning of LLMs.
Sports Toys Beauty ArtsHit Ratio @ 1Naï ve zero -shot
CoRAL  (User -based RAG)
ItemRAG  (Ours)Figure 1:ItemRAGoutperforms the strongest user-based RAG
baseline.Across datasets,ItemRAGconsistently (1) improves
the zero-shot GPT-based recommender and (2) outperforms
the strongest user-based RAG baseline, CoRAL [13].
In this work, we introduceItemRAG( Item -based Retrieval-
Augmented Generation), an RAG approach for LLM-based recom-
mendation grounded initem-based retrieval. In a nutshell,Item-
RAGretrieves items (not users) relevant to the target user’s past
purchases using item–item co-purchase histories, thereby enabling
LLMs to capture which item types are often co-purchased.
In addition, we use a specialized retrieval strategy that enhances
ItemRAG’s retrieval quality. Specifically, we (1) incorporate items
semantically similar to the target item so retrieval remains effective
for cold-start targets, and (2) apply co-purchase-frequency–based
sampling to reflect collaborative signals to relevant item retrieval.
We favor item retrieval over user retrieval since users’ interests
are often diverse and can be difficult to infer from purchase histories
alone, making similar users difficult to identify. In contrast, item
relevance is relatively simple (e.g., substitutable or complementary
relationships) and can be effectively inferred from co-purchase re-
lations, making relevant items easier to retrieve. The effectiveness
ofItemRAGin LLM-based recommendation is also supported em-
pirically. As shown in Figure 1,ItemRAGconsistently outperforms
the strongest user-based RAG baseline across datasets, with further
details in Section 4. Our key contributions are as follows:
•(C1) Item-based RAG.We propose the concept of item-based
RAG for LLM-based recommendation and, to this end, use item–item
co-purchase histories for item retrieval.
•(C2) Retrieval strategy.Our proposed method,ItemRAG, lever-
ages a specialized retrieval technique that enhances the relevant
item retrieval and cold-start item recommendation.
•(C3) Strong performance.ItemRAGconsistently outperforms
user-based RAG baselines in both (1) standard LLM-based rec-
ommendation and (2) cold-start item recommendation.
Supplementary materials, code, and datasets are provided in [6].arXiv:2511.15141v1  [cs.IR]  19 Nov 2025

, , Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, and Kijung Shin
Item 𝒊: Item purchased 
by the target user
Items similar to  Item 𝒊
Item summary generationSummarize the items that are co -purchased 
with Item 𝑖 or with items similar to  it.
Item 𝑖:
Co-purchased items:Summary prompt
Task: Rank the candidate items based on 
the target user’s purchase history.
Target user purchase history:Recommendation prompt
[…]
LLM-based recommendation with ItemRAGItems co -purchased with 
Item 𝒊 or similar items
Specialized sampling
Generated summary for Item 𝒊
Co-purchased items include
▪Outdoor and camping gear: Items 
related to camping, hiking, fishing, and 
outdoor survival equipment such as 
tents, portable stoves, […]
▪Water and beverage accessories: Items 
focusing on beverage containment […]
…
, 
 […], …
Candidate items:
[…] , 
[…], …
Figure 2:An example case ofItemRAG, our item-based RAG method. For retrieving relevant items for item 𝑖, we first identify
items that are co-purchased with (1) item 𝑖itself and/or (2) items whose textual descriptions are similar to that of item 𝑖. Then,
we sample a specified number of items from this pool, with selection probabilities proportional to their co-purchase frequencies
with item𝑖. Subsequently, we prompt an LLM to generate summaries of the sampled items and incorporate these summaries
into the final recommendation prompt, guiding the LLM to understand the co-purchase patterns among items.
2 Related work and preliminary
In this section, we review related studies and present the prelimi-
nary concepts relevant to our work.
Related work.Thanks to their strong reasoning capabilities and
ability to handle cold-start items, using LLMs as recommender
systems has attracted substantial attention [ 5,7]. To better adapt
them to recommendation tasks, retrieval-augmented generation
(RAG) techniques have been widely explored [ 3,13,17]. One line
of work applies RAG in specific settings, including conversational
recommendation [ 17] and knowledge–graph–based recommenda-
tion [ 11]. Another line of RAG work aims to improve general-
purpose recommenders that rely primarily on user–item interac-
tions [ 3,10,13,15], which is a focus of this work. They are mostly
user-based approaches, as detailed in Section 1. Note that, to our
knowledge,ItemRAGis the first RAG approach that explicitly fo-
cuses on relevant item retrieval for LLM-based recommendation.
Preliminary.The user set and the item set are denoted by UandI,
respectively. We consider a sequential recommendation setting, and
therefore, each user 𝑢∈U is represented by her purchase history
sequence:u :=[𝑖(𝑢)
1,𝑖(𝑢)
2,···,𝑖(𝑢)
𝑛𝑢], where𝑖(𝑢)
𝑠denotes the𝑠-th item
purchased by user 𝑢and𝑛𝑢is the number of items purchased by 𝑢.
Each item𝑖∈Ihas a text description𝓉 𝑖, such as item title.
3 Proposed method
In this section, we introduceItemRAG( Item -based Retrieval-
Augmented Generation), an item-based RAG method for LLM-
based recommendation. We first give an overview of theItemRAG
pipeline (Section 3.1) and detail our retrieval strategy (Section 3.2).
3.1 Overall pipeline ofItemRAG
We consider an LLM-based recommendation pipeline in which the
model is given (1) the target user’s purchase history and (2) a set of
candidate items, and is prompted to rank the candidates by their
likelihood of being the target user’s next purchase.1Here, each item
is represented by its textual description (e.g., item title).
By usingItemRAG, weenhance the description of each query item
𝑖that is (1) purchased by the target user or (2) a recommendation
candidate—by retrieving items relevant to 𝑖. Specifically, for each
query item 𝑖, we retrieve items relevant to it, and then provide a
summary of the retrieved items together with the original textual
description of each query item 𝑖. In Section 3.2, we present key
1Candidate items are often obtained by non-LLM-based recommender systems.challenges in the relevant-item decision process and introduce our
retrieval strategy that overcomes them.
3.2 Retrieval strategy ofItemRAG
A straightforward way to retrieve items related to a query item is
to select those that are co-purchased with it. However, this faces
two challenges: (C1) it performs poorly—and may be infeasible—for
cold-start query items with little or no co-purchase data, and (C2)
some co-purchased items are incidental and thus weakly relevant.
To address the cold-start challenge (C1), for each query item 𝑖,
we retrieve not only items co-purchased with 𝑖but also items co-
purchased with items whose text descriptions are similar to 𝑖. Our
rationale is that items similar to 𝑖often share co-purchase patterns
with𝑖, giving strong complementary co-purchase signals for𝑖.
To address the weak-relevance challenge (C2), we score each
query–retrieved item pair by its co-purchase frequency and use
this score for the probability of the item being selected in the final
retrieval. Our rationale is that frequent co-purchases indicate strong
relevance and yield an effective signal of co-purchase patterns.
Based on these intuitions, we formally elaborate on our retrieval
strategy. We start with presenting two notations. We denote a set
of items purchased by user 𝑢asM(𝑢) (i.e.,M(𝑢)={𝑖(𝑢)
𝑠:𝑠∈
{1,2,···,𝑛 𝑢}}). We also denote a set of items co-purchased with
item𝑖asN(𝑖)(i.e.,N(𝑖)={𝑗:∃𝑢∈Usuch that{𝑖,𝑗}∈M(𝑢)}).
In retrieval for a query item 𝑖, we first find the top- 𝐾whose
textual descriptions are most similar to 𝑖; we denote this set as
T(𝑖) . Specifically, for each 𝑗∈I , we encode its text description 𝓉𝑗
via a pre-trained language modelLM, obtaining the representation
z𝑗∈R𝑑(i.e.,z 𝑗=LM(𝓉 𝑗)). We then compute the cosine similarity
between𝑖and each other item 𝑗∈I\{𝑖} (i.e.,(z𝑇
𝑖z𝑗)/(∥z𝑖∥2∥z𝑗∥2),
and select the top-𝐾by similarity; the resulting set isT(𝑖).
We subsequently derive a retrieval pool P(𝑖) comprising items
co-purchased with (1) item 𝑖itself (N(𝑖) ) and/or (2) items having
similar descriptions to𝑖(T(𝑖)). Formally, the pool is defined as:
P(𝑖)=N(𝑖)∪{𝑗:∃𝑞∈T(𝑖)s.t.𝑗∈N(𝑞)}.(1)
After, instead of retrieving all the items within P(𝑖) , we sample
𝑁number of items proportional to their co-purchase frequencies
with item𝑖. Formally, let a co-purchase frequency of items𝑖and𝑗
as𝑐𝑖 𝑗(i.e.,𝑐𝑖 𝑗=Í
𝑢∈U1[{𝑖,𝑗}∈M(𝑢)] , where1[·]is an indicator
function). Then, a sampling weight 𝑤𝑖 𝑗of item𝑗being retrieved
for query item𝑖is defined as:

ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation , ,
Table 1:(RQ1&4) LLM-based recommendation performance. All metrics are multiplied by 100 for better readability. H@K and
N@K denote Hit-Ratio@K and NDCG@K, respectively. We do not report N@1, since it is equal to H@1. Best results are
highlighted with a green box. Notably,ItemRAGoutperforms the baseline methods in 19 out of 20 cases.
MethodsBeauty & Personal Care Toys & Games Sports & Outdoors Arts, Crafts & Sewing
H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5
LightGCN [1] 42.9 68.3 82.2 57.7 63.4 32.0 58.6 73.1 47.4 53.3 33.3 62.5 78.2 50.3 56.8 50.9 76.3 87.8 65.7 70.4
LightGCN++ [8] 44.0 68.9 83.6 58.5 64.5 35.5 61.2 75.2 50.4 56.1 36.4 64.2 80.1 52.6 59.2 53.2 77.5 88.5 67.4 72.0
SASRec [4] 43.9 68.9 82.1 57.8 63.3 31.4 57.9 74.6 46.1 53.2 34.6 62.6 77.5 50.8 56.9 46.4 72.3 86.0 61.8 66.3
BERT4Rec [9] 44.2 70.3 83.8 59.0 64.9 31.8 58.4 74.9 46.2 53.8 35.8 62.6 78.5 51.7 57.7 48.1 74.6 87.2 63.3 68.6
Zero-shot 34.9 58.5 74.0 48.6 54.9 39.0 62.1 76.3 52.4 58.3 43.6 66.2 81.9 56.6 63.0 46.0 72.1 84.4 61.1 66.1
ICL [10] 35.0 59.3 76.2 49.3 55.4 39.7 65.7 79.4 54.0 59.5 44.0 69.0 83.7 57.9 63.8 44.6 74.7 86.5 62.0 66.8
AdaptRec [15] 35.2 59.1 75.7 49.2 56.0 38.6 63.5 76.3 52.7 58.1 45.2 67.9 82.4 58.3 64.3 48.7 74.2 85.9 63.6 68.4
ReACT [3] 34.0 57.5 73.1 47.7 54.1 38.5 61.5 76.2 52.1 58.0 43.8 67.2 82.7 56.5 63.2 47.3 73.5 84.6 62.5 67.0
CoRAL [13] 45.7 68.5 82.0 59.2 64.5 45.2 69.7 80.5 59.5 64.0 48.9 72.6 87.4 62.5 68.6 52.5 80.4 91.3 68.8 73.6
w/o sim-items 47.4 70.0 83.1 60.5 66.2 49.5 71.2 83.2 62.2 67.0 50.2 73.9 87.3 64.3 69.6 56.0 82.0 91.1 71.2 74.8
w/o co-purch. 47.7 70.1 83.8 60.7 66.3 48.1 73.3 83.4 62.7 66.9 50.7 74.1 87.9 64.3 69.8 56.2 82.2 91.0 71.3 75.0
ItemRAG 49.9 70.9 84.2 62.0 67.3 48.9 73.9 84.8 63.3 67.7 51.9 74.4 88.2 64.9 70.5 57.5 82.5 92.0 72.1 75.8
𝑤𝑖 𝑗=𝑐𝑖 𝑗+1
|T(𝑖)|∑︁
𝑞∈T(𝑖)𝑐𝑞𝑗,(2)
where𝑐𝑖 𝑗denotes the co-purchase frequency between items 𝑖and𝑗,
and the rest indicates the mean of co-purchase frequencies between
item𝑗and items that are semantically similar to item𝑖.
Subsequently, we sample 𝑁items from the retrieval pool P(𝑖)
(Eq.(1)), where each item 𝑗∈P(𝑖) is drawn with the probability
of𝑤𝑖 𝑗/(Í
𝑞∈P(𝑖)𝑤𝑖𝑞)(Eq.(2)). Lastly, we prompt an LLM to sum-
marize the sampled items and append this summary to the original
description of item 𝑖, helping the LLM-based recommender cap-
ture co-purchase information of item 𝑖. Note that the co-purchase
summary generation is independent of the target user; thus, the
summary for a given item can be used for different target users.
4 Experiment
In this section, we analyze the effectiveness ofItemRAGin the
LLM-based recommendation tasks. To this end, we analyze the
following four research questions:
RQ1. How effective isItemRAGfor LLM-based recommendation?
RQ2. How accurate isItemRAGat recommending cold-start items?
RQ3. When the item information retrieved byItemRAGis given,
do LLM-based recommender systems use it?
RQ4. Do allItemRAGkey components contribute to performance?
4.1 Experimental setting
Datasets and evaluation protocol.We use four domains from
the latest Amazon Reviews dataset [ 2]: Sports & Outdoors (Sports),
Toys & Games (Toys), Beauty & Personal Care (Beauty), and Arts,
Crafts & Sewing (Arts). Further details—including preprocessing
steps and dataset statistics—are in [ 6]. For evaluation, following
prior work [ 7,13], we use a leave-one-out protocol: for each user,
the last purchased item is held out for testing, and the remaining
history is used as input. Also, following [ 7], we prompt the LLM
to rank 10 candidate items for the target user’s next purchase; the
candidate set contains 1 ground-truth item and 9 random samples.
Baseline methods andItemRAG.For comparison, we use 9 base-
line methods: two graph-based models (LightGCN [ 1] and Light-
GCN++ [ 8]), two sequential models (SASRec [ 4] and BERT4Rec [ 9]),
one naive zero-shot LLM-based recommender, and four user-basedRAG methods (ICL [ 10], AdaptRec [ 15], ReACT [ 3], and CoRAL [ 13]).
For methods that use an LLM, we use GPT-4.1-mini as a backbone.
For the retrieval process inItemRAG, we use5similar items per
item and sample50items in the final retrieval set. Additional details
on baselines, hyperparameters, and prompt templates are in [6].
4.2 RQ1. Standard LLM-based recommendation
Setup.For each method, we construct the training data and retrieval
database from users’ purchase histories after withholding each
user’s last interaction, which is reserved exclusively for evaluation.
For testing, we evaluate each method on1 ,000randomly sampled
users under the evaluation protocol detailed in Section 4.1.
Result.As shown in Table 1,ItemRAGoutperforms all the baseline
methods in 19 out of 20 settings. Two points stand out. First,Item-
RAGconsistently improves the naive zero-shot LLM recommender,
by up to 43% in Hit-Ratio@1 on the Beauty & Personal Care dataset.
Second,ItemRAGoutperforms user-based RAG methods, outper-
forming the strongest baseline (CoRAL) by up to 10% in terms of
Hit-Ratio@1 on the Arts, Crafts & Sewing dataset.
4.3 RQ2. Cold-start item recommendation
Setup.Given that the strength in cold-start item recommendation
is the primary promise of LLM-based recommenders [ 14], we eval-
uate each method under an item cold-start setup. Specifically, for
the1,000sampled users described in Section 4.2, we remove the
ground-truth test items for the users—together with all interactions
involving that item—from both the training set and the retrieval
database, making the corresponding items cold-start.2We then
evaluate each method’s ability to recommend such items to the
corresponding users. Since the learning-based baselines we use
cannot handle unseen (cold-start) items, we focus on LLM-based
approaches. Other settings remain the same as in Section 4.2.
Result.As shown in Table 2,ItemRAGoutperforms the baseline
methods in all the cases, demonstrating its strong performance in
recommending cold-start items. Notably, its performance decreases
by only 1.3% on average relative to the standard setting (Table 1),
suggesting that it remains effective in cold-start scenarios.
2Since the original dataset contains very few items that first appear in the test
set—fewer than 30 across all datasets—we use the modified dataset instead.

, , Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, and Kijung Shin
Table 2:(RQ2) Cold-start item recommendation performance. All metrics are multiplied by 100 for better readability. H@K
and N@K denote Hit-Ratio@K and NDCG@K, respectively. Best results are highlighted with a green box. Notably,ItemRAG
outperforms the baseline methods in every case.
MethodsBeauty & Personal care Toys & Games Sports & Outdoors Arts, Crafts & Sewing
H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5 H@1 H@3 H@5 N@3 N@5
Zero-shot 34.9 58.5 74.0 48.6 54.9 39.0 62.1 76.3 52.4 58.3 43.6 66.2 81.9 56.6 63.0 46.0 72.1 84.4 61.1 66.1
ICL [10] 33.1 57.8 74.3 47.1 54.3 37.6 65.2 78.6 53.1 58.5 43.4 69.4 83.7 58.2 64.0 44.9 74.8 87.0 62.2 67.3
AdaptRec [15] 36.1 59.7 75.3 49.8 56.2 38.1 64.1 78.1 53.0 58.6 44.9 68.1 82.2 58.1 64.0 48.9 73.5 85.0 63.4 68.1
ReACT [3] 35.2 58.3 71.9 50.2 53.9 39.3 61.9 75.7 52.4 58.0 43.0 67.0 81.5 56.7 62.5 47.3 72.9 84.9 62.2 67.1
CoRAL [13] 38.4 60.4 73.1 51.2 56.4 39.1 61.4 74.0 51.9 56.9 44.2 67.7 80.5 57.8 63.0 46.2 73.0 83.8 61.7 66.2
ItemRAG 47.6 70.5 83.7 60.7 66.1 48.0 71.2 83.2 61.4 66.4 50.6 74.7 88.8 64.4 70.2 57.7 81.9 91.7 71.7 75.5
Purchase history (Left → Right: Old → New)
Doll 
holderSophia’s 
pajamaSophia’s 
serving setNext purchase
BARWA wedding 
dress for dolls
LLM predictions and the rationale for its predictions
▪Naï ve zero -shot LLM:  Jewelry kit for kids
▪LLM with ItemRAG : BARWA wedding dress for dolls
▪LLM’s rationale:  “The user’s purchase history shows strong 
patterns of interests in dolls and accessories […] This item has 
strong co -purchase associations with doll and doll accessory 
groups , making it the most likely candidate for purchase.”
Figure 3:(RQ3) Case study.While the naive zero-shot LLM-
based recommender fails, augmenting it with co-purchase
information retrieved byItemRAG—information the model
explicitly uses—yields an accurate recommendation.
4.4 RQ3. Case study
Setup.We examine whether the LLM-based recommender system
leverages the item information retrieved byItemRAG. To this end,
on the Toys & Games dataset, we run a case study in which the
LLM is prompted to give the rationale behind its recommendations.
Additional cases in other datasets are in [6].
Result.Figure 3 presents a case where a naive zero-shot LLM-based
recommender fails to provide an accurate recommendation. When
the prompt is augmented with co-purchase information retrieved by
ItemRAG, the LLM (1) recommends the correct item and (2) explic-
itly notes in its rationale that it relied on the retrieved co-purchase
signals. This result suggests thatItemRAG’s retrieved information
is indeed used and beneficial for improving performance.
4.5 RQ4. Ablation study
Setup.We assess the necessity ofItemRAG’s key retrieval compo-
nents (Section 3.2) by using two ablations: (i) a variant that only
uses its own co-purchase information for each item during retrieval
(w/o sim-items), and (ii) a variant that replaces frequency-based
sampling weights with uniform sampling (w/o co-purch.).
Result.As shown in Table 1, the two variants (‘w/o sim-items’ and
‘w/o co-purch.’) underperformItemRAGin 19 out of 20 settings,
demonstrating the effectiveness of theItemRAG’s key components
in LLM-based recommendation.
5 Conclusion
In this work, we introduceItemRAG, an item-based RAG tech-
nique for LLM-based recommendation.ItemRAGretrieves relevantitems from co-purchase histories, incorporating co-purchase fre-
quencies and similarity of textual descriptions. Through extensive
experiments, we demonstrate the effectiveness ofItemRAGfor
LLM-based recommendation and cold-start item recommendation.
Supplementary materials, code, and datasets are available at [6].
References
[1]Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng
Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for
recommendation. InSIGIR.
[2]Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian
McAuley. 2024. Bridging language and items for retrieval and recommenda-
tion.arXiv:2403.03952(2024).
[3]Zheng Hu, Yongsen Pan, Zetao Li, Jiaming Huang, Satoshi Nakagawa, Jiawen
Deng, Shimin Cai, and Fuji Ren. 2026. Retrieval-enhanced, Adaptively Collabora-
tive, and Temporal-aware user behavior comprehension for LLM-based sequential
recommendation.Information Processing & Management63, 1 (2026), 104354.
[4]Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recom-
mendation. InICDM.
[5]Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and
Chanyoung Park. 2024. Large language models meet collaborative filtering: An
efficient all-round llm-based recommender system. InKDD.
[6]Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, and Kijung Shin. 2025. Sup-
plementary materials, code, and datasets for this work.https://anonymous.
4open.science/r/ItemRAG-DBD2/.
[7]Genki Kusano, Kosuke Akimoto, and Kunihiro Takeoka. 2025. Revisiting Prompt
Engineering: A Comprehensive Evaluation for LLM-based Personalized Recom-
mendation. InRecSys.
[8]Geon Lee, Kyungho Kim, and Kijung Shin. 2024. Revisiting LightGCN: Unex-
pected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommen-
dation. InRecSys.
[9]Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang.
2019. BERT4Rec: Sequential recommendation with bidirectional encoder repre-
sentations from transformer. InCIKM.
[10] Lei Wang and Ee-Peng Lim. 2024. The whole is better than the sum: Using
aggregated demonstrations in in-context learning for sequential recommendation.
InNAACL.
[11] Shijie Wang, Wenqi Fan, Yue Feng, Shanru Lin, Xinyu Ma, Shuaiqiang Wang, and
Dawei Yin. 2025. Knowledge graph retrieval-augmented generation for llm-based
recommendation. InACL.
[12] Shuyao Wang, Zhi Zheng, Yongduo Sui, and Hui Xiong. 2025. Unleashing the
Power of Large Language Model for Denoising Recommendation. InWWW.
[13] Junda Wu, Cheng-Chun Chang, Tong Yu, Zhankui He, Jianing Wang, Yupeng
Hou, and Julian McAuley. 2024. Coral: collaborative retrieval-augmented large
language models improve long-tail recommendation. InKDD.
[14] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen,
Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al .2024. A survey on large
language models for recommendation.World Wide Web27, 5 (2024), 60.
[15] Tong Zhang. 2025. AdaptRec: A Self-Adaptive Framework for Sequential Recom-
mendations with Large Language Models.arXiv:2504.08786(2025).
[16] Peilin Zhou, Chao Liu, Jing Ren, Xinfeng Zhou, Yueqi Xie, Meng Cao, Zhongtao
Rao, You-Liang Huang, Dading Chong, Junling Liu, et al .2025. When Large Vision
Language Models Meet Multimodal Sequential Recommendation: An Empirical
Study. InWWW.
[17] Yaochen Zhu, Chao Wan, Harald Steck, Dawen Liang, Yesu Feng, Nathan Kallus,
and Jundong Li. 2025. Collaborative Retrieval for Large Language Model-based
Conversational Recommender Systems. InWWW.
# What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals

**Authors**: Shahrooz Pouryousef

**Published**: 2025-05-27 05:18:57

**PDF URL**: [http://arxiv.org/pdf/2505.20730v1](http://arxiv.org/pdf/2505.20730v1)

## Abstract
User-item interactions contain rich collaborative signals that form the
backbone of many successful recommender systems. While recent work has explored
the use of large language models (LLMs) for recommendation, it remains unclear
whether LLMs can effectively reason over this type of collaborative
information. In this paper, we conduct a systematic comparison between LLMs and
classical matrix factorization (MF) models to assess LLMs' ability to leverage
user-item interaction data. We further introduce a simple retrieval-augmented
generation (RAG) method that enhances LLMs by grounding their predictions in
structured interaction data. Our experiments reveal that current LLMs often
fall short in capturing collaborative patterns inherent to MF models, but that
our RAG-based approach substantially improves recommendation
quality-highlighting a promising direction for future LLM-based recommenders.

## Full Text


<!-- PDF content starts -->

arXiv:2505.20730v1  [cs.IR]  27 May 2025What LLMs Miss in Recommendations: Bridging the Gap with
Retrieval-Augmented Collaborative Signals
Shahrooz Pouryousef
UMass Amherst
Amherst, MA, USA
shahrooz@cs.umass.edu
Abstract
User-item interactions contain rich collaborative signals that form
the backbone of many successful recommender systems. While
recent work has explored the use of large language models (LLMs)
for recommendation, it remains unclear whether LLMs can effec-
tively reason over this type of collaborative information. In this
paper, we conduct a systematic comparison between LLMs and
classical matrix factorization (MF) models to assess LLMs’ abil-
ity to leverage user-item interaction data. We further introduce
a simple retrieval-augmented generation (RAG) method that en-
hances LLMs by grounding their predictions in structured interac-
tion data. Our experiments reveal that current LLMs often fall short
in capturing collaborative patterns inherent to MF models, but that
our RAG-based approach substantially improves recommendation
quality—highlighting a promising direction for future LLM-based
recommenders.
Keywords
LLMs, Recommendation systems, Collaborative information
1 Introduction
Large Language Models (LLMs) [ 1,20] have demonstrated impres-
sive capabilities in a wide range of tasks, including reasoning over
textual input [ 8,14], answering complex questions [ 9,18], gen-
erating fluent text [ 7], and encoding world knowledge [ 17]. As a
result, LLMs have been increasingly adopted in diverse research
domains such as data collection [ 21], summarization [ 3,13], trans-
lation [10, 15, 28], and visualization [11, 23].
More recently, researchers have begun exploring the potential of
LLMs in recommender systems, where the goal is to deliver person-
alized item suggestions that align with a user’s preferences—such
as recommending movies, products, or articles [ 2,5,16,24]. This
emerging line of work, often referred to as LLMs as recommenders
(LLMRec), presents an exciting new research direction that lever-
ages the capabilities of LLMs to tackle core challenges in recom-
mender systems [ 2,25,27]. Within this domain, LLMs have pri-
marily been applied to two key subtasks: item understanding and
user modeling. For example, in content-based filtering, LLMs can
generate rich item representations from textual descriptions and
infer user preferences from their interaction history [12, 22].
LLMs have been utilized in recommender systems (RSs) under
two main paradigms. LLMs as Recommenders (LLMs-as-RSs) refers
to approaches where LLMs are directly prompted or fine-tuned to
function as recommenders. In contrast, LLM-enhanced RSs lever-
age the knowledge stored in LLM parameters to enhance tradi-
tional recommender models—for example, by generating text-based
item/user representations or embeddings.One limitation of LLMRec methods is their insufficient modeling
of collaborative information embedded in user-item co-occurrence
patterns. To address this, Sun et al . [19] proposed an approach that
distills the world knowledge and reasoning capabilities of LLMs into
a collaborative filtering recommender system. Their method adopts
an in-context, chain-of-thought prompting strategy, focusing on
the LLM-enhanced RS paradigm.
However, the integration of collaborative filtering within the
LLMs-as-RSs paradigm—where recommendations are derived from
patterns across many users—remains relatively underexplored. This
gap is due in part to fundamental challenges: it is unclear how to
adapt LLMs to effectively reason over collaborative signals, and the
input length limitations of LLMs make it difficult to encode dense
interaction histories needed to learn from similar users at scale.
In this paper, we take a step toward understanding this gap by
analyzing the reasoning capabilities of LLMs over collaborative sig-
nals. Our first research question is: Can LLMs capture and reason
over collaborative patterns? To assess the basic reasoning ability
of LLMs in utilizing collaborative information, we compare their
performance with matrix factorization, a classical and widely used
method for modeling collaborative information in recommender
systems. This comparison allows us to evaluate whether LLMs can
reason over collaborative signals at a level comparable to—even if
not superior to—a simple yet effective baseline like matrix factor-
ization.
Our second research question investigates how to improve LLMs’
ability to reason over collaborative data. To this end and overcome
to the challenges that mentioned above, we explore a retrieval-
augmented generation (RAG) approach that supplies the LLM with
relevant user-item interaction information at inference time, aim-
ing to bridge the gap between LLMs and traditional collaborative
filtering methods.
2 Preliminaries
In this study, we explore the reasoning ability of LLMs on collabo-
rative filtering information. Our investigation is centered around
a matrix of interactions between users and items, i.e., 𝑀. LetU=
{𝑢1, . . . ,𝑢 𝑛}denote the set of all 𝑛users andI={𝑣1, . . . , 𝑣 𝑚}
denote the set of all 𝑚items (movies). The matrix 𝑀∈R𝑛×𝑚rep-
resents the rating behavior of users over items, where each row
corresponds to a user in Uand each column to an item in I. The
entry 𝑀𝑖 𝑗indicates the rating that user 𝑢𝑖∈U has given to item
𝑣𝑗∈I, with values typically ranging from 1(disliked) to 5(highly
liked), or 0if the user has not rated the item.
To enable large language models (LLMs) to understand collabo-
rative signals in the interaction matrix, it is necessary to provide

Trovato and Tobin, et al.
54-----5--2--5---254-5--4U1U2U3U4M1M2M3M5M4
U5(Target)54-----5--2--5---254-5--4U1U2U3U4M1M2M3M5M4
U5(Target)RAG: Retrieve similar usersU4:M3:2, M4:5, M5:4U1: M1:5,M2:4
U4:M3:2, M4:5U1: M1:5U4:M3:2, M4:5U1: M1:5Popularity: M3:(2,3.5), M4:(2,5), M1:(2,3.5)U4:M3:disliked, M4:liked, M5:likedU1: M1:liked,M2:likedRaw dataBaselineSentiment-basedReasoningFull-reasoning
Figure 1: Comparison of four prompt-generation strategies
for movie recommendation based on retrieved user–movie
similarities. Each method varies in how it incorporates simi-
lar users’ ratings, handles previously seen movies, and struc-
tures the prompt for downstream recommendation.
them with structured information derived from the matrix. How-
ever, including the entire matrix in the input is infeasible, especially
in large-scale datasets with many users and items. Moreover, pre-
senting all such information in a prompt can overwhelm the LLM,
making it difficult to interpret the collaborative structure and re-
spond effectively to recommendation queries.
Recently, retrieval-augmented generation (RAG) has demon-
strated impressive results in enhancing LLM performance across
a variety of tasks [ 4,6,26]. RAG approaches integrate external
retrieval mechanisms to supplement LLMs with relevant and tar-
geted information, thereby improving the accuracy, relevance, and
contextual understanding of generated responses.
In the following section, we present our method for applying
retrieval-augmented generation (RAG) to enhance the effective-
ness and efficiency of LLMs in leveraging collaborative filtering
information.
3 Methodology
Given a target user 𝑢𝑡, our goal is to extract relevant signals from the
interaction matrix 𝑀to enable an LLM to predict and recommend
movies that the user is likely to like.
To identify useful information, we first retrieve users most simi-
lar to the target user. These similar users are assumed to carry the
most informative signals for inferring the preferences of 𝑢𝑡, as their
past ratings can help guide the recommendation of unseen items.
To identify users most similar to the target user 𝑢𝑡, we construct
a user–item interaction matrix by pivoting the masked dataset (test
data are masked for all models) such that each row corresponds
to a user and each column to an item, with missing ratings filled
with zero. We then compute pairwise user similarities using cosine
similarity:
sim(𝑢𝑡,𝑢)=𝑀𝑢𝑡·𝑀𝑢
∥𝑀𝑢𝑡∥·∥𝑀𝑢∥
We define the set of top- 𝑘similar users as:
N𝑘(𝑢𝑡)=Top- 𝑘({sim(𝑢𝑡,𝑢)|𝑢∈U\{ 𝑢𝑡}})whereUis the set of all users, and N𝑘(𝑢𝑡)⊂U\{ 𝑢𝑡}denotes
the𝑘most similar users to 𝑢𝑡. For each 𝑢∈N 𝑘(𝑢𝑡), we define their
set of ratings as:
𝑅𝑢={(𝑗, 𝑀 𝑢𝑗)|𝑗∈I, 𝑀𝑢𝑗≠0}
We use the information from these similar users and their rat-
ings to construct the input for the LLM in our prompt-generation
framework. Specifically, the sets 𝑅𝑢for all 𝑢∈N 𝑘(𝑢𝑡)provide the
relevant collaborative signals that are encoded into the prompt.
LetRN𝑘(𝑢𝑡)=Ð
𝑢∈N𝑘(𝑢𝑡)𝑅𝑢denote the union of all such rating
sets. This setRN𝑘(𝑢𝑡)⊂I× Rrepresents the pool of item–rating
pairs from users most similar to the target user 𝑢𝑡, and serves as a
compact, informative context for the LLM.
In the following subsections, we present four distinct strategies
for incorporatingRN𝑘(𝑢𝑡)into the prompt. Each strategy varies
in how it structures this information, filters relevant ratings, and
balances the trade-off between informativeness and prompt length.
3.1 Prompt Generation Strategies
We explore four distinct strategies for constructing our prompts.
Each approach presents user rating data differently to investigate
its impact on model performance and prompt efficiency.
3.1.1 Unfiltered Full Ratings Prompt (Baseline) .This baseline
approach includes the full rating history of all top- 𝑘similar users
without applying any filtering or deduplication. Each user’s ratings
are presented in their raw form. While the fraction parameter ( 𝑓)
still controls how much of the similar users’ data is included in
the prompt, no preference is given to unseen items or highly rated
content. This method offers a comprehensive view of all available
preferences, potentially enriching the context provided to the LLM.
Notably, we include all sampled ratings from similar users, even
for items that the target user has already rated. We refer to this
approach as the baseline .
Example:
Target user A has rated the following movies: M1
(5), M2 (3), M3 (4)...
Top- 𝑘similar users have rated:
User 101 rated: M1 (1), M7 (3), M2 (2)
User 24 rated: M2 (3), M4 (4)
...
Which 10 movies should user A watch next that
they haven’t seen?
3.1.2 Sentiment-Based Prompt .This strategy organizes the
ratings from similar users into three sentiment categories: Liked ,
Neutral , and Disliked . These categories are defined by rating thresh-
olds: ratings≥4are classified as Liked , a rating of 3asNeutral ,
and ratings≤2asDisliked . As with other strategies, we exclude
any movies that the target user has already rated to focus only on
unseen content.
Example:
Target user A has rated the following movies: M1
(5), M2 (3), M3 (4)...
Top-𝑘similar users to user A have collectively rated
the following unseen movies:
Liked: 234, 567, 890

What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals
Neutral: 111, 222
Disliked: 333, 444
Based on these patterns, which 10 movies should
user A watch next?
3.1.3 Reasoning-Based Prompt .This strategy builds upon the
baseline by introducing two key modifications. First, we remove
any movies from the similar users’ ratings that have already been
seen by the target user, ensuring that all recommended items are
truly unseen. Second, we sample a fraction 𝑓of the remaining
ratings for inclusion in the prompt, controlling the prompt size
while preserving informative signals.
Additionally, the prompt is explicitly framed as a reasoning task
by appending a directive such as: “Reason based on the patterns
above: which 10 movies should user A watch next that they haven’t
seen?” This phrasing encourages the model to engage in logical in-
ference rather than relying purely on pattern matching, potentially
improving generalization and recommendation quality.
Example:
Target user A has rated the following movies: M1
(5), M2 (3), M3 (4)...
Top-𝑘similar users to user A have collectively rated
the following unseen movies:
User 101 rated: M7 (3), M6 (2)
User 24 rated: M8 (3), M4 (5)
Reason based on the patterns above: which 10 movies
should user A watch next that they haven’t seen?
3.1.4 Full Reasoning .In the Reasoning approach, the LLM only
sees data from the top- 𝑘similar users, and therefore lacks access to
global signals—such as the overall popularity of movies across the
user base. To address this limitation, we propose an enhanced vari-
ant that augments the prompt with additional popularity-related
statistics.
For each movie 𝑣𝑗∈I, we compute two global metrics based on
the interaction matrix 𝑀∈R𝑛×𝑚:
•Rating Count: The number of users who rated movie 𝑣𝑗:
Count(𝑣𝑗)=
𝑢𝑖∈U| 𝑀𝑖 𝑗≠0	
•Average Rating: The average rating received by movie 𝑣𝑗:
AvgRating(𝑣𝑗)=Í
𝑢𝑖∈U𝑀𝑖 𝑗·1[𝑀𝑖 𝑗≠0]
Count(𝑣𝑗)
These statistics are appended to each corresponding movie entry
in the "Reasoning" prompt. This allows the LLM to consider not
only the preferences of similar users but also the broader appeal
and quality of each recommended item.
Example:
Target user A has rated the following movies: M1
(5), M2 (3), M3 (4)...
Top-𝑘similar users to user A have collectively rated
the following unseen movies:
User 101 rated: M7 (3), M6 (2)
User 24 rated: M8 (3), M4 (5)
Movie popularity stats:
M4 — Count: 231, AvgRating: 4.2
M6 — Count: 121, AvgRating: 3.6M7 — Count: 298, AvgRating: 3.8
M8 — Count: 83, AvgRating: 3.2
Reason based on the patterns above and the pop-
ularity statistics: which 10 movies should user A
watch next that they haven’t seen?
4 Evaluation
4.1 Dataset
We evaluate the effectiveness of using LLMs for personalized movie
recommendation using the widely studied MovieLens 100K dataset.
This dataset comprises 100,000ratings from 943users on 1,682
movies. As a preprocessing step, we remap both user and movie
IDs to be contiguous integers starting from zero to ensure con-
sistency across model input formats. We sort each user’s rating
history chronologically and randomly mask 20% of their ratings to
simulate unobserved preferences. These masked ratings are with-
held and later used to evaluate recommendation accuracy, while
the remaining 80% of ratings are retained to simulate user history
and compute user-user similarities.
To evaluate the models fairly across different user profiles, we
split the users into two groups based on activity level. Users with a
number of ratings above the dataset’s median are categorized as
"hot users," and the rest as "cold users." We randomly sample 350
users from each group.
4.2 Experimental and parameter settings
For each user, we construct a prompt that includes (1) the known
ratings from their retained history, and (2) the ratings from the top-
k most similar users. These similar users are selected using cosine
similarity computed over the masked user-item matrix. To simulate
different levels of auxiliary knowledge, we vary the fraction of each
similar user’s ratings shown in the prompt from 25% to 100%. We
use𝑓to indicate the fraction value. We also explicitly inform the
LLM about the range of movie IDs in the dataset to provide context.
The prompt concludes with a question asking the model to sug-
gest 10movies that the target user has not seen. The LLM is then
queried using OpenAI’s ChatCompletion API, and its response is
parsed to extract movie IDs. These predictions are evaluated using
NDCG (Normalized Discounted Cumulative Gain) and Hit@10 met-
rics. NDCG evaluates how well a ranked list puts the most relevant
items near the top and Hit@10 metric measures the proportion
of ground-truth masked movies that appear in the model’s top-10
recommendations. Formally, for each user, Hit@10 is calculated
as the cardinality of the intersection between the predicted and
held-out movies divided by 10. We compute and average this value
across all users in both the hot and cold user groups.
To establish a strong baseline, we also implement a matrix fac-
torization model in PyTorch and train it using stochastic gradient
descent over a range of hyperparameters, including different num-
bers of latent factors ({10, 20, 50, 100}) and batch sizes ({8, 16, 32,
64,128, and 256}). The model is trained for up to 3,000epochs or
until convergence, with performance evaluated on a held-out test
set. We report the best-performing configuration after convergence
based on the highest NDCG score for the testing cases. The model is
trained on the same 80% unmasked ratings used in the LLM setup.

Trovato and Tobin, et al.
(a) Hot users(b) Cold users
Figure 2: NDCG score for different prompt generation strategies and MF as a function of number of similar users ( 𝑘) and
different percentage of their information in the prompt ( 𝑓). Number of users is 350in both cold and hot groups. Top is the hot
users and bottom is cold users.
(a) Hot users(b) Cold users
Figure 3: Hit@10 score for different prompt generation strategies and MF as a function of number of similar users ( 𝑘) and
different percentage of their information in the prompt ( 𝑓). Number of users is 350in both cold and hot groups. Top is the hot
users and bottom is cold users.
4.3 Results and Discussion
Figure 2 presents the average NDCG scores obtained across dif-
ferent prompt generation strategies as a function of the number
of similar users 𝑘, for multiple fractions of their rating histories
(𝑓). We observe that reasoning-based prompts (such as reasoningand full reasoning) consistently improve in performance as 𝑘in-
creases. This suggests that LLMs benefit from richer collaborative
information when it is presented in a structured and interpretable
format. In contrast, simpler strategies such as sentiment-based or
original tend to decrease in NDCG as 𝑘increases. This likely occurs
because these prompts include more raw data without structure

What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals
Figure 4: Processing time of prompts and the number of
tokens in each prompt for hot and cold users.
or abstraction, making it harder for the LLM to extract relevant
signals—highlighting the importance of structured prompt engi-
neering. These results support integrating techniques like retrieval-
augmented generation (RAG) or knowledge graphs to organize
collaborative user–item interactions more effectively.
As we increase the fraction of similar user data included in the
prompt, we observe a general increase in overall NDCG scores
across reasoning strategies. This aligns with the expectation that
providing more evidence to the LLM leads to stronger predictions.
However, an interesting trend appears when comparing cold vs.
hot users: LLMs tend to perform better on cold users than hot
users, while Matrix Factorization (MF) exhibits the opposite behav-
ior—achieving higher scores for hot users and lower for cold ones.
This is expected, as MF relies on sufficient user-item interactions
to learn embeddings, making it more effective for users with rich
histories.
For LLMs, one possible explanation for stronger performance on
cold users is reduced confusion due to fewer input ratings. Addition-
ally, cold users typically have fewer masked (unseen) movies—sometimes
only 2 or 3—since we mask 20% of each user’s ratings. When the
LLM is asked to recommend 10 movies, the probability of hitting
those few masked movies is relatively high. In contrast, hot users
often have 100+ ratings, so 20% masking can result in 20–30 unseen
movies. Consequently, the LLM has a lower chance of selecting
the correct ones, even when reasoning is good, due to the larger
candidate set.
Figure 4 shows the cumulative distribution function (CDF) of
prompt processing times for cold and hot users using the reason-
ing_ranked prompt generation strategy, with the number of similar
users fixed at 𝑘=10, and varying the fraction of each similar
user’s rating history included in the prompt. By processing time,
we refer to the total latency involved in communicating with the
GPT server, including the time to transmit the prompt, the model’s
internal computation, and the time required to receive the response
back to the local machine.
For cold users, the CDF curves for different fraction values (e.g.,
0.25,0.5,0.75,1.0) are tightly clustered and exhibit nearly identicalprocessing times. This is expected because cold users, by definition,
have rated very few movies. Consequently, even when using the
full fraction of ratings from their similar users, the total number
of ratings included in the prompt remains small. Thus, the prompt
length—and hence the request latency—does not vary significantly
with the fraction parameter.
In contrast, for hot users, we observe that higher fraction val-
ues lead to noticeably longer processing times. Since hot users
have many ratings, each similar user is likely to contribute more
movie–rating pairs as the fraction increases. This results in signifi-
cantly longer prompts, which take longer to serialize, transmit to
the model, and process on the server. The effect is visible in the
rightward shift of the CDF curves for higher fractions.
This analysis highlights that prompt length has a strong effect
on processing latency, especially for users with rich interaction his-
tories. It also supports the importance of designing token-efficient
prompt generation strategies when scaling LLM-based recommen-
dation systems, particularly for hot users with high 𝑘and large
data fractions.
5 Conclusion
In the LLMs-as-recommender-systems (LLMs-as-RCs) paradigm,
a key challenge is enabling LLMs to effectively incorporate col-
laborative information. In this paper, we began by analyzing the
performance of LLMs in capturing collaborative signals for movie
recommendation. We showed that a naive approach—embedding
all user information directly into the prompt—makes it difficult
for LLMs to interpret these signals, often performing worse than
simple baselines such as matrix factorization.
To address this, we proposed a retrieval-augmented generation
(RAG) based approach with improved prompting strategies. Our
results demonstrate that presenting collaborative signals in a com-
pact format and prompting the LLM to reason over them improves
recommendation performance compared to traditional baselines.
Moreover, our method is both token-efficient and effective, achiev-
ing better results on standard evaluation metrics while minimizing
prompt length.
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan
He. 2023. Tallrec: An effective and efficient tuning framework to align large
language model with recommendation. In Proceedings of the 17th ACM Conference
on Recommender Systems . 1007–1014.
[3] Lochan Basyal and Mihir Sanghvi. 2023. Text summarization using large language
models: a comparative study of mpt-7b-instruct, falcon-7b-instruct, and openai
chat-gpt models. arXiv preprint arXiv:2310.10449 (2023).
[4] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. In Proceedings of the AAAI
Conference on Artificial Intelligence , Vol. 38. 17754–17762.
[5] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongx-
iang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering chatgpt’s capabilities in
recommender systems. In Proceedings of the 17th ACM Conference on Recom-
mender Systems . 1126–1132.
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining . 6491–6501.
[7]Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large
language models to generate text with citations. arXiv preprint arXiv:2305.14627

Trovato and Tobin, et al.
(2023).
[8]Jie Huang and Kevin Chen-Chuan Chang. 2022. Towards reasoning in large
language models: A survey. arXiv preprint arXiv:2212.10403 (2022).
[9] Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021. How can we
know when language models know? on the calibration of language models for
question answering. Transactions of the Association for Computational Linguistics
9 (2021), 962–977.
[10] Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondřej Bojar, Anton
Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda,
Roman Grundkiewicz, et al .2024. Findings of the WMT24 general machine trans-
lation shared task: the LLM era is here but mt is not solved yet. In Proceedings of
the Ninth Conference on Machine Translation . 1–46.
[11] Guozheng Li, Xinyu Wang, Gerile Aodeng, Shunyuan Zheng, Yu Zhang,
Chuangxin Ou, Song Wang, and Chi Harold Liu. 2024. Visualization gener-
ation with large language models: An evaluation. arXiv preprint arXiv:2401.11255
(2024).
[12] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu,
Chuhan Wu, Xiangyang Li, Chenxu Zhu, et al .2025. How can recommender
systems benefit from large language models: A survey. ACM Transactions on
Information Systems 43, 2 (2025), 1–47.
[13] Yixin Liu, Kejian Shi, Katherine S He, Longtian Ye, Alexander R Fabbri, Pengfei
Liu, Dragomir Radev, and Arman Cohan. 2023. On learning to summarize with
large language models as references. arXiv preprint arXiv:2305.14239 (2023).
[14] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and
Thomas Back. 2024. Reasoning with large language models, a survey. arXiv
preprint arXiv:2407.11511 (2024).
[15] Vikas Raunak, Amr Sharaf, Yiren Wang, Hany Hassan Awadallah, and Arul
Menezes. 2023. Leveraging GPT-4 for automatic translation post-editing. arXiv
preprint arXiv:2305.14878 (2023).
[16] Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, and Lucas Dixon.
2023. Large language models are competitive near cold-start recommenders for
language-and item-based preferences. In Proceedings of the 17th ACM conference
on recommender systems . 890–896.
[17] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won
Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al .
2023. Large language models encode clinical knowledge. Nature 620, 7972 (2023),
172–180.
[18] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed
Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al .2025.
Toward expert-level medical question answering with large language models.
Nature Medicine (2025), 1–8.
[19] Zhongxiang Sun, Zihua Si, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang,
and Jun Xu. 2024. Large language models enhanced collaborative filtering.
InProceedings of the 33rd ACM International Conference on Information and
Knowledge Management . 2178–2188.
[20] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[21] Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng
Wang, Dawei Yin, and Chao Huang. 2024. Llmrec: Large language models
with graph augmentation for recommendation. In Proceedings of the 17th ACM
International Conference on Web Search and Data Mining . 806–815.
[22] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen,
Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al .2024. A survey on large
language models for recommendation. World Wide Web 27, 5 (2024), 60.
[23] Yang Wu, Yao Wan, Hongyu Zhang, Yulei Sui, Wucai Wei, Wei Zhao, Guandong
Xu, and Hai Jin. 2024. Automated data visualization from natural language
via large language models: An exploratory study. Proceedings of the ACM on
Management of Data 2, 3 (2024), 1–28.
[24] Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, and
Even Oldridge. 2023. Llamarec: Two-stage recommendation using large language
models for ranking. arXiv preprint arXiv:2311.02089 (2023).
[25] Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan
He. 2023. Is chatgpt fair for recommendation? evaluating fairness in large
language model recommendation. In Proceedings of the 17th ACM Conference on
Recommender Systems . 993–999.
[26] Taolin Zhang, Dongyang Li, Qizhou Chen, Chengyu Wang, Longtao Huang,
Hui Xue, Xiaofeng He, and Jun Huang. 2024. R 4: Reinforced Retriever-Reorder-
Responder for Retrieval-Augmented Large Language Models. In ECAI 2024 . IOS
Press, 2314–2321.
[27] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen
Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al .2024. Recommender systems
in the era of large language models (llms). IEEE Transactions on Knowledge and
Data Engineering (2024).
[28] Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng
Kong, Jiajun Chen, and Lei Li. 2023. Multilingual machine translation with large
language models: Empirical results and analysis. arXiv preprint arXiv:2304.04675(2023).
Received 20 February 2024; revised 12 March 2024; accepted 5 June 2024
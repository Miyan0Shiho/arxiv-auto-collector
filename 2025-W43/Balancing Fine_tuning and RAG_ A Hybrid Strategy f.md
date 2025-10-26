# Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates

**Authors**: Changping Meng, Hongyi Ling, Jianling Wang, Yifan Liu, Shuzhou Zhang, Dapeng Hong, Mingyan Gao, Onkar Dalal, Ed Chi, Lichan Hong, Haokai Lu, Ningren Han

**Published**: 2025-10-23 06:31:00

**PDF URL**: [http://arxiv.org/pdf/2510.20260v1](http://arxiv.org/pdf/2510.20260v1)

## Abstract
Large Language Models (LLMs) empower recommendation systems through their
advanced reasoning and planning capabilities. However, the dynamic nature of
user interests and content poses a significant challenge: While initial
fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to
capture such real-time changes, necessitating robust update mechanisms. This
paper investigates strategies for updating LLM-powered recommenders, focusing
on the trade-offs between ongoing fine-tuning and Retrieval-Augmented
Generation (RAG). Using an LLM-powered user interest exploration system as a
case study, we perform a comparative analysis of these methods across
dimensions like cost, agility, and knowledge incorporation. We propose a hybrid
update strategy that leverages the long-term knowledge adaptation of periodic
fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B
experiments on a billion-user platform that this hybrid approach yields
statistically significant improvements in user satisfaction, offering a
practical and cost-effective framework for maintaining high-quality LLM-powered
recommender systems.

## Full Text


<!-- PDF content starts -->

Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic
LLM Recommendation Updates
Changping Meng∗
changping@gmail.com
Google
Mountain View, California
USAHongyi Ling∗
linghongyi@google.com
Google
Mountain View, California
USAJianling Wang∗
jianlingw@google.com
Google Deepmind
Mountain View, California
USAYifan Liu
yifanliu@google.com
Google
Mountain View, California
USA
Shuzhou Zhang
shuzhouz@google.com
Google
Mountain View, California
USADapeng Hong
dapengh@google.com
Google
Mountain View, California
USAMingyan Gao
mingyan@google.com
Google
Mountain View, California
USAOnkar Dalal
onkardalal@google.com
Google
Mountain View, California
USA
Ed Chi
edchi@google.com
Google Deepmind
Mountain View, California
USALichan Hong
lichan@google.com
Google Deepmind
Mountain View, California
USAHaokai Lu
haokai@google.com
Google Deepmind
Mountain View, California
USANingren Han
peterhan@google.com
Google
Mountain View, California
USA
Abstract
Large Language Models (LLMs) empower recommendation systems
through their advanced reasoning and planning capabilities. How-
ever, the dynamic nature of user interests and content poses a signif-
icant challenge: While initial fine-tuning aligns LLMs with domain
knowledge and user preferences, it fails to capture such real-time
changes, necessitating robust update mechanisms. This paper in-
vestigates strategies for updating LLM-powered recommenders, fo-
cusing on the trade-offs between ongoing fine-tuning and Retrieval-
Augmented Generation (RAG). Using an LLM-powered user interest
exploration system as a case study, we perform a comparative analy-
sis of these methods across dimensions like cost, agility, and knowl-
edge incorporation. We propose a hybrid update strategy that lever-
ages the long-term knowledge adaptation of periodic fine-tuning
with the agility of low-cost RAG. We demonstrate through live A/B
experiments on a billion-user platform that this hybrid approach
yields statistically significant improvements in user satisfaction,
offering a practical and cost-effective framework for maintaining
high-quality LLM-powered recommender systems.
CCS Concepts
•Information systems →Information retrieval;•Computing
methodologies→Artificial intelligence.
∗Equal Contribution
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for third-party components of this work must be honored.
For all other uses, contact the owner/author(s).
RecSys ’25, Prague, Czech Republic
©2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1364-4/2025/09
https://doi.org/10.1145/3705328.3748105Keywords
Large Language Models, Recommendation System, User Interest
Exploration
ACM Reference Format:
Changping Meng, Hongyi Ling, Jianling Wang, Yifan Liu, Shuzhou Zhang,
Dapeng Hong, Mingyan Gao, Onkar Dalal, Ed Chi, Lichan Hong, Haokai
Lu, and Ningren Han. 2025. Balancing Fine-tuning and RAG: A Hybrid
Strategy for Dynamic LLM Recommendation Updates. InProceedings of the
Nineteenth ACM Conference on Recommender Systems (RecSys ’25), September
22–26, 2025, Prague, Czech Republic.ACM, New York, NY, USA, 4 pages.
https://doi.org/10.1145/3705328.3748105
1 Introduction
The emergence of Large Language Models (LLMs) is transforming
the landscape of recommendation systems with their extensive
world knowledge and reasoning capabilities. LLM-powered rec-
ommenders [ 1,7,16] utilize the deep semantic understanding and
generative strengths of these models to deliver more personalized,
explainable, and context-aware suggestions.
A common approach involves initially fine-tuning an LLM on
domain knowledge and historical user-item interactions to tailor it
for specific recommendation tasks [ 14]. However, the environments
where these systems operate are inherently dynamic [ 12,13]. User
interests evolve, new items emerge constantly, and underlying data
patterns shift – for instance, analysis of user transitions between
interest clusters often reveals significant temporal variability. An
LLM fine-tuned only on past data captures a static snapshot and
cannot inherently reflect these real-time dynamics.
To address this challenge, two prominent techniques for adapting
and updating LLMs are fine-tuning [ 3,15] and Retrieval-Augmented
Generation (RAG) [ 6,17]. Fine-tuning involves further training a
pre-trained LLM on a specific dataset to adjust its internal parame-
ters, tailoring its knowledge or behavior. RAG, conversely, connectsarXiv:2510.20260v1  [cs.IR]  23 Oct 2025

RecSys ’25, September 22–26, 2025, Prague, Czech Republic Changping Meng et al.
the LLM to external knowledge sources at inference time, retriev-
ing relevant information to augment the prompt and ground the
model’s generation in specific, often up-to-date, data without alter-
ing the model’s parameters.
This paper conducts a comparative analysis of fine-tuning and
RAG as methodologies for adapting LLM-powered recommenda-
tion systems to dynamic updates. Our investigation is grounded
in a deployed LLM-powered user interest recommendation sys-
tem [ 14,15]. While interest exploration systems [ 4,5,8–10] aim to
diversify recommendations, effectively introducing novel interests
poses a significant challenge. In our case study, the fine-tuned LLM
generates potential novel interest clusters from user history; the
core update challenge we address is enabling this model to accu-
rately reflect the changing popularity and relationships between
these clusters over time.
This challenge leads to our central hypothesis: In a highly dy-
namic domain like short-form video recommendation, a static,
fine-tuned LLM is insufficient to maintain recommendation qual-
ity over time. We hypothesize that a hybrid strategy, combining
periodic fine-tuning with frequent RAG-based updates, will more
effectively adapt to shifting user interest patterns and result in su-
perior online performance. This paper tests this hypothesis through
a LLM-powered user interest exploration system [ 15]. We therefore
compare fine-tuning and RAG specifically for this task, discussing
their respective system designs, processes, strengths, limitations,
effectiveness, and cost, using both offline and live experimental
results.
2 Method
This section first provides necessary preliminary information and
outlines the motivation for our work, followed by a detailed de-
scription of the interest exploration system. Subsequently, we detail
the designs for fine-tuning and RAG.
2.1 Preliminary
Motivation.To effectively model the dynamic nature of user inter-
ests, we represent them using clusters, following the methodology
in [2]. To assess the evolution of user interest transitions, we first
define a ‘successor interest’. From user interaction logs, we con-
struct sequences of three consecutive, distinct item clusters a user
engages with, denoted as (𝑐1,𝑐2,𝑐𝑛𝑒𝑥𝑡). Here,𝑐𝑛𝑒𝑥𝑡is the ‘succes-
sor interest’ to the preceding pair (𝑐1,𝑐2). We then measured the
stability of the top-5 most frequent successor interests month-over-
month using the Jaccard Similarity (i.e., quantifying the semantic
overlap between these top-5 sets). Our analysis revealed substantial
variability, with a low mean Jaccard score of 0.17 (variance 0.07),
demonstrating substantial monthly variability in prevalent user
transition patterns. This observed dynamism highlights the critical
need for efficiently incorporating refreshed user feedback.
Interest Exploration System.In the LLM-powered system [ 15],
each user’s recent interaction history is represented as a sequence of
𝑘interest clusters 𝑆𝑢={𝑐 1,𝑐2,...,𝑐 𝑘}, where each 𝑐𝑖∈Cdenotes
an item interest cluster from a predefined cluster set C[2]. Each
interest cluster groups items that are topically coherent, based on
their metadata and content feature. Given 𝑆𝑢, the LLM predicts the
user’s next novel interest cluster 𝑐𝑛∈C. Because online serving theLLM for a billion-user system is prohibitively costly, we precompute
and store the predicted next-cluster transitions for all possible 𝑘-
length sequences of interest clusters. Let S={(𝑐 1,...,𝑐 𝑘)|𝑐 𝑖∈C}
denote the set of all possible 𝑘-length cluster sequences. For each
𝑆∈S , we store a corresponding predicted novel cluster 𝑐𝑛offline.
During online serving, a user’s current history 𝑆𝑢is matched to a
set𝑆∈S , and the corresponding predicted next cluster is retrieved
via table lookup.
2.2 Fine-tune
Following the preliminary example [ 15], with𝑘= 2, each fine-
tuning data sample is denoted as [(𝑐 1,𝑐2),𝑐𝑛𝑒𝑥𝑡]. The prompt is
illustrated as black lines in Figure 1. Periodically, we curate thou-
sands of those pairs for fine-tuning. Fine-tuning offers the benefits
of adapting model behavior and style, as well as improving perfor-
mance on specific tasks. However, the drawbacks are also signifi-
cant, including high cost and complexity, and the risk of overfitting.
Due to the high cost of fine-tuning, updates happen on a monthly
basis.
We also propose two key evaluation metrics to evaluate the
fine-tuning quality: the exact match rate (percentage of predictions
precisely matching the partition description) and the test set re-
call (percentage of predictions aligning with users’ watch history).
Leveraging these insights, our auto-refreshed fine-tuning pipeline
implements two automated quality checks:
•If the exact match rate during partition mapping generation is
below 90%, the pipeline execution is halted.
•If the test set recall is less than 1.5%, the pipeline fails.
These conditions necessitate manual review by an engineer to iden-
tify the root cause and decide whether to proceed to production or
re-run the process.
2.3 RAG
Instead of retraining LLM with new viewing data with high cost,
we can prompt new data to the LLM and perform bulk inference
periodically to generate a dynamic transition mapping with low
cost. Adhering to the prompt design of the LLM-powered interest
exploration system [ 15], we represent a user’s consumption history
as a sequence of their most recently interacted unique clusters.
Each cluster is defined by a set of keywords. To better capture
both dynamic system-wide trends and individual user’s evolving
preferences, these prompts incorporate top popular interest clusters
along with the user’s recent watch history, as detailed in Figure 1.
Fine-tuning can be done on a monthly schedule, while the RAG
prompt can happen more frequently, even at daily basis, with the
overall system illustrated in Figure 2.
2.3.1 Granularity.During the bulk inference phase, RAG prompts
can be generated at different level of granularity.
•Instance level. Prompts are tailored for each individual cluster
pair. Specifically, we can identify the top-1 most frequent next
cluster based on recent data. Consider a distribution {(𝑐 1,𝑐2,𝑐3):
10,(𝑐 1,𝑐2,𝑐4): 8,(𝑐 1,𝑐2,𝑐5): 2,...} . Since𝑐3appears most fre-
quently following 𝑐1and𝑐2,𝑐3can be included in the prompt
for inference.

Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates RecSys ’25, September 22–26, 2025, Prague, Czech Republic
Figure 1: Prompt for Novel Interest Prediction. Black lines
are the fine-tuning prompt. Added Blue lines are the RAG
version with injected recent watch history. Label is only used
for fine-tuning, but not RAG
Figure 2: Refresh with Retrieval-Augmented Generation.
Fine-tuning is refreshed monthly, while RAG is refreshed
multiple times within a week.
•Global level. This approach uses a single, universal prompt for
all data pairs. This prompt captures overall user behavior and
might include illustrative examples. E.g. we can construct the
prompt using the top-100 most frequent pairs found across the
entire new data, regardless of specific input pairs. These globally
representative clusters are then included to guide inference.
Given that the global level design might introduce noise to the
target cluster pair during output cluster generation, we adopt the
instance level design approach.
2.3.2 Retrieval Similarity.This section outlines methods for re-
trieving relevant recent data during bulk inference.
•Frequency-based Retrieval: We identify data points within the
same cluster pair as the query and select those with the high-
est frequency. This provides the LLM with prompts reflecting
recent, prevalent user behaviors for the specific cluster.
•Trend-based Retrieval: Focusing on the query’s cluster pair, we
select data points exhibiting the largest frequency difference,
highlighting emerging or declining user interests.
Our analysis and evaluation indicate that frequency-based retrieval
yields the best results.Number of retrieved clusters included in the context can vary
(e.g. the Cluster 3 in Figure 1). While a larger number provides
richer information, it also increases computational cost. Our live
experiments suggest that including the top 1 most frequent cluster
is sufficient to provide satisfying results.
2.4 Data Retrieval
We use users’ interaction history, represented as a sequence of
watches, on a large-scale video platform as the source dataset. Our
data extraction targets videos demonstrating positive viewer en-
gagement. To refine the dataset, we deduplicate video cluster IDs
within each sequence and remove sequences with fewer than two
videos. For the remaining sequences, we construct tuples of three
consecutive video cluster IDs as (𝑐1,𝑐2,𝑐𝑛𝑒𝑥𝑡). The final step is to
count the occurrences of the next cluster for each cluster pair.
3 Results and Evaluation
In our hybrid update strategy, LLM models undergo monthly fine-
tuning, while RAG refresh occurs sub-weekly. From the fine-tuned
model, we then measure the incremental gains of more frequent
and up-to-date RAG.
3.1 Offline Evaluation
We evaluated how RAG-generated cluster mappings evolve over
time and their alignment with user behavior. Specifically, We as-
sessed the hit rate, which computes the proportion of times the
predicted next cluster appears in the real user sequence. We com-
pared three versions of transition mappings: (1) a fixed mapping
generated without RAG; (2) a RAG-generated mapping updated ev-
ery two days; and (3) a RAG-generated mapping computed only on
𝑑𝑎𝑦 1and held fixed thereafter. As illustrated in Figure 3a, both RAG-
based mappings outperform the fixed baseline, with the version
updated every two days achieving slightly higher hit rates.
To better understand the influence of RAG on the LLM’s genera-
tion behavior, we analyze the similarity between outputs generated
with and without RAG. Only 7.8% of the RAG-generated outputs
were identical to those produced without RAG, compared to a 37.5%
overlap when using repeated prompts without RAG. The results
indicate that RAG significantly alters the generated content, often
leading to novel predictions that differ from both the retrieved
context and the non-RAG outputs.
Finally, we studied how the top- 𝑘most frequent clusters for each
cluster pair changed over time. Our findings reveal a significant
shift in top clusters across retrieval dates, with substantial drops in
overlap as time progresses. This trend, illustrated in Figure 3b, re-
emphasize the dynamic nature of user interests and underscores the
need for regularly refreshed retrieval to reflect current behavioral
patterns.
3.2 Live Experiment
We conducted A/B experiments within a short-form video recom-
mendation system, serving billions of users, to measure the effec-
tiveness of RAG in enhancing the performance of our LLM-powered
interest exploration system. Gemini 1.5 [ 11] was adopted as the
base LLM for this system, while the process and pipeline are de-
signed for adaptability to other models. The system’s high-level

RecSys ’25, September 22–26, 2025, Prague, Czech Republic Changping Meng et al.
(a) Trajectory of hit rate.
 (b) Exact match rates for top k
clusters over time
Figure 3: Offline Evaluation.
function recommends novel interest clusters, currently based on a
user’s historical interest cluster sequence of length𝐾=2.
We report the user metrics of the live experiments in Figure 4.
The x-axis represents the date, and the y-axis shows the relative
percentage difference between the treatment and control. We also
report the mean and 95% confidence intervals for each metric. The
top-tier metricsSatisfied User Outcomesare increased by0 .11%with
95% confidence interval [0.00%,0.21%], which is highly significant
at the scale of our system.Satisfaction Rateis increased by0 .25%
with interval[0.01%,0.48%]. TheDissatisfaction Rateis reduced
by0.05%with interval [−0.08%,−0.01%].Negative Interactionis
reduced by0.04%with interval[−0.08%,−0.01%].
We employed RAG to update the cluster transition table on 𝑑𝑎𝑦 1
and𝑑𝑎𝑦 4. Following these updates, we observed notable increases in
user engagement, including significant improvements in Satisfied
User Outcomes and Satisfaction Rate, indicating enhanced user
satisfaction.
(a) Satisfied User Outcomes
 (b) Satisfaction Rate
(c) Dissatisfaction Rate
 (d) Negative Interaction
Figure 4: Live experiment results for user metrics. The x-
axis represents the date; the y-axis represents the relative
difference (in percentage) between the treatment and control
groups.
4 Conclusion
This paper investigated the critical challenge of keeping LLM-
powered recommendation systems updated. We conducted a com-
parative analysis of fine-tuning and RAG, proposing and validating
a hybrid strategy. Our core finding is that combining monthly
fine-tuning with sub-weekly RAG updates provides a robust, cost-
effective solution for adapting to dynamic user interests, leading
to significant improvements in online user satisfaction metrics in
a large-scale production environment. Future work will exploremore adaptive update cadences, where the frequency of RAG or
fine-tuning is determined automatically based on the detected rate
of interest drift.
Speaker Bio
Changping Meng is a software engineer at Google (YouTube). He
received the Computer Science PhD from the Purdue University.
His work primarily focuses on short-form video recommendations.
References
[1]Arkadeep Acharya, Brijraj Singh, and Naoyuki Onoe. 2023. Llm based generation
of item-description for recommendation system. InProceedings of the 17th ACM
conference on recommender systems. 1204–1207.
[2]Bo Chang, Changping Meng, He Ma, Shuo Chang, Yang Gu, Yajun Peng, Jingchen
Feng, Yaping Zhang, Shuchao Bi, Ed H Chi, and Minmin Chen. 2024. Cluster
Anchor Regularization to Alleviate Popularity Bias in Recommender Systems. In
Companion Proceedings of the ACM Web Conference 2024.
[3]Jiaju Chen, Chongming Gao, Shuai Yuan, Shuchang Liu, Qingpeng Cai, and
Peng Jiang. 2025. DLCRec: A Novel Approach for Managing Diversity in LLM-
Based Recommender Systems. InProceedings of the Eighteenth ACM International
Conference on Web Search and Data Mining. 857–865.
[4]Minmin Chen. 2021. Exploration in recommender systems. InProceedings of the
15th ACM Conference on Recommender Systems. 551–553.
[5]Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-
Lin Wu, and Ed Chi. 2021. Values of user exploration in recommender systems.
InProceedings of the 15th acm Conference on recommender systems. 85–95.
[6]Run-Ze Fan, Yixing Fan, Jiangui Chen, Jiafeng Guo, Ruqing Zhang, and Xueqi
Cheng. 2024. RIGHT: Retrieval-augmented generation for mainstream hashtag
recommendation. InEuropean Conference on Information Retrieval. Springer, 39–
55.
[7]Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and
Chanyoung Park. 2024. Large language models meet collaborative filtering: An
efficient all-round llm-based recommender system. InProceedings of the 30th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1395–1406.
[8]Khushhall Chandra Mahajan, Amey Porobo Dharwadker, Romil Shah, Simeng Qu,
Gaurav Bang, and Brad Schumitsch. 2023. PIE: Personalized Interest Exploration
for Large-Scale Recommender Systems. InCompanion Proceedings of the ACM
Web Conference 2023. 508–512.
[9]Yu Song, Shuai Sun, Jianxun Lian, Hong Huang, Yu Li, Hai Jin, and Xing Xie. 2022.
Show me the whole world: Towards entire item space exploration for interactive
personalized recommendations. InProceedings of the Fifteenth ACM International
Conference on Web Search and Data Mining. 947–956.
[10] Yi Su, Xiangyu Wang, Elaine Ya Le, Liang Liu, Yuening Li, Haokai Lu, Benjamin
Lipshitz, Sriraj Badam, Lukasz Heldt, Shuchao Bi, et al .2024. Long-term value of
exploration: Measurements, findings and algorithms. InProceedings of the 17th
ACM International Conference on Web Search and Data Mining. 636–644.
[11] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol
Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al .2024.
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of
context.arXiv preprint arXiv:2403.05530(2024).
[12] Jianling Wang, Kaize Ding, Liangjie Hong, Huan Liu, and James Caverlee. 2020.
Next-item recommendation with sequential hypergraphs. InProceedings of the
43rd international ACM SIGIR conference on research and development in informa-
tion retrieval. 1101–1110.
[13] Jianling Wang, Raphael Louca, Diane Hu, Caitlin Cellier, James Caverlee, and
Liangjie Hong. 2020. Time to shop for valentine’s day: Shopping occasions and se-
quential recommendation in e-commerce. InProceedings of the 13th international
conference on web search and data mining. 645–653.
[14] Jianling Wang, Haokai Lu, Yifan Liu, He Ma, Yueqi Wang, Yang Gu, Shuzhou
Zhang, Shuchao Bi, Lexi Baugher, Ed Chi, et al .2024. LLMs for User Interest
Exploration: A Hybrid Approach.arXiv e-prints(2024), arXiv–2405.
[15] Jianling Wang, Haokai Lu, Yifan Liu, He Ma, Yueqi Wang, Yang Gu, Shuzhou
Zhang, Ningren Han, Shuchao Bi, Lexi Baugher, Ed H. Chi, and Minmin Chen.
2024. LLMs for User Interest Exploration in Large-scale Recommendation Systems.
InProceedings of the 18th ACM Conference on Recommender Systems (RecSys ’24).
Association for Computing Machinery, New York, NY, USA, 872–877. https:
//doi.org/10.1145/3640457.3688161
[16] Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2024. Openp5: An open-source
platform for developing, training, and evaluating llm-based recommender sys-
tems. InProceedings of the 47th International ACM SIGIR Conference on Research
and Development in Information Retrieval. 386–394.
[17] Huimin Zeng, Zhenrui Yue, Qian Jiang, and Dong Wang. 2024. Federated recom-
mendation via hybrid retrieval augmented generation. In2024 IEEE International
Conference on Big Data (BigData). IEEE, 8078–8087.
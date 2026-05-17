# RAG-Enhanced Large Language Models for Dynamic Content Expiration Prediction in Web Search

**Authors**: Tingyu Chen, Wenkai Zhang, Li Gao, Lixin Su, Ge Chen, Dawei Yin, Daiting Shi

**Published**: 2026-05-13 06:20:28

**PDF URL**: [https://arxiv.org/pdf/2605.13052v1](https://arxiv.org/pdf/2605.13052v1)

## Abstract
In commercial web search, aligning content freshness with user intent remains challenging due to the highly varied lifespans of information. Traditional industrial approaches rely on static time-window filtering, resulting in "one-size-fits-all" rankings where content may be chronologically recent but semantically expired. To address the limitation, we present a novel Large Language Models (LLMs)-based Query-Aware Dynamic Content Expiration Prediction Framework deployed in Baidu search, reformulating timeliness as a dynamic validity inference task. Our framework extracts fine-grained temporal contexts from documents and leverages LLMs to deduce a query-specific "validity horizon"-a semantic boundary defining when information becomes obsolete based on user intent. Integrated with robust hallucination mitigation strategies to ensure reliability, our approach has been evaluated through offline and online A/B testing on live production traffic. Results demonstrate significant improvements in search freshness and user experience metrics, validating the effectiveness of LLM-driven reasoning for solving semantic expiration at an industrial scale.

## Full Text


<!-- PDF content starts -->

RAG-Enhanced Large Language Models for Dynamic Content
Expiration Prediction in Web Search
Tingyu Chen
Baidu Inc.
Beijing, China
tingyuchen.cty@gmail.comWenkai Zhang
Baidu Inc.
Beijing, China
zhangwk0106@gmail.comLi Gao∗
Baidu Inc.
Beijing, China
gaoli05@baidu.com
Lixin Su
Baidu Inc.
Beijing, China
sulixinict@gmail.comGe Chen
Baidu Inc.
Beijing, China
chenge02@baidu.comDawei Yin
Baidu Inc.
Beijing, China
yindawei@acm.org
Daiting Shi
Baidu Inc.
Beijing, China
shidaiting01@baidu.com
Abstract
In commercial web search, aligning content freshness with user
intent remains challenging due to the highly varied lifespans of
information. Traditional industrial approaches rely on static time-
window filtering, resulting in "one-size-fits-all" rankings where
content may be chronologically recent but semantically expired. To
address the limitation, we present a novel Large Language Models
(LLMs)-based Query-Aware Dynamic Content Expiration Predic-
tion Framework deployed in Baidu search, reformulating timeliness
as a dynamic validity inference task. Our framework extracts fine-
grained temporal contexts from documents and leverages LLMs to
deduce a query-specific "validity horizon"—a semantic boundary
defining when information becomes obsolete based on user intent.
Integrated with robust hallucination mitigation strategies to ensure
reliability, our approach has been evaluated through offline and
online A/B testing on live production traffic. Results demonstrate
significant improvements in search freshness and user experience
metrics, validating the effectiveness of LLM-driven reasoning for
solving semantic expiration at an industrial scale.
CCS Concepts
•Information systems →Retrieval models and ranking;•
Computing methodologies→Natural language processing.
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnKeywords
Ranking Systems, Temporal Reasoning, Large Language Models,
Retrieval-Augmented Generation, Information Retrieval
ACM Reference Format:
Tingyu Chen, Wenkai Zhang, Li Gao, Lixin Su, Ge Chen, Dawei Yin, and Dait-
ing Shi. 2026. RAG-Enhanced Large Language Models for Dynamic Content
Expiration Prediction in Web Search. In.ACM, New York, NY, USA, 5 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Modern search engines must align semantic relevance with time-
liness [ 21]. Unlike static knowledge retrieval, assessingtemporal
relevanceto determine whether content remains valid or has become
stale requires dynamically reasoning about information lifespans
based on user intent [12, 13, 16, 20].
Traditionally, industrial search engines rely on rule-based buck-
eting to manage timeliness [ 2,3,11]. A common practice is to
apply a global "recency boost" to documents published within a
fixed window (e.g., the past 30 days) or to indiscriminately penalize
older content. While computationally efficient, this "one-size-fits-
all" strategy suffers from a severe granularity mismatch.
As illustrated in Figure 1, a universal expiration threshold does
not exist. For a breaking news query such as "Hong Kong Fire", a
report published just two days ago may already be expired if the fire
has been extinguished and the situation has evolved. Conversely,
for a query like "Traffic Regulations", a policy document published
several years ago remains entirely valid. Fundamentally, these limi-
tations stem from treating timeliness as a static metadata attribute
rather than a dynamic semantic relationship between query intent
and content timelines.
To bridge this gap, we propose a novel LLM-based dynamic con-
tent expiration prediction framework. Building upon the Retrieval-
Augmented Generation (RAG) paradigm [ 5,7,14], we reformulate
the timeliness challenge as a dynamic content expiration prediction
problem. Specifically, we first introduce a temporal information
extraction module that segments raw document content ( D) intoarXiv:2605.13052v1  [cs.IR]  13 May 2026

Conference’17, July 2017, Washington, DC, USA Tingyu Chen et al.
Query 1: “Hong Kong Fire”
Query 2: “Traffic Regulations”False Positive:
Boosted outdated state
News:
Fire Breaks 
out-2 injured
(Extinguished)
Past t-30 days t-5 days t-3 daysDoc2Static 30-Day 
Threshold
t (present)Doc1News:
Fire Extinguished-4 
Dead, 10 Injured
(Active)
Past t-30 days t-5 yearsStatic 30-Day 
Threshold
t (present)Doc3Policy: 2021 
Rules
（Active) False Negative:
Buried valid rulesPast t-30 days t-5 days t-3 daysDoc2
t (present)Doc1Correctly Expired：
Superseded by new state
Past t-30 days t-5 years t (present)Doc3Policy: 2021 
Rules
(Active)Correctly Valid Dynamic ValidityDynamic Validity
LLM detects no 
superseding lawsNews:
Fire Breaks 
out-2 injured
(Extinguished)News:
Fire Extinguished-4 
Dead, 10 Injured
(Active)
(a) Traditional Rule-Based Bucketing (b) LLM-based Dynamic Content Expiration
Query 1: “Hong Kong Fire”
Query 2: “Traffic Regulations”
Figure 1: Comparison between (a) traditional rule-based buck-
eting and (b) our proposed LLM-based dynamic content expi-
ration prediction framework. By inferring a query-specific
validity horizon, our approach correctly expires superseded
news while preserving long-term authoritative content.
time-aware focused chunks ( 𝑆𝑓𝑜𝑐𝑢𝑠 ). This allows the model to pri-
oritize salient temporal signals and mitigate noise from long-form
text [ 6,8,17]. Building on this, we employ a structured prompting
mechanism for temporal inference. The LLM [ 1,4] synthesizes the
query intent ( 𝑄) and the extracted temporal context to dynamically
determine a query-specific expiration threshold ( 𝑡∗
𝑒𝑥𝑝), effectively
decoupling validity from rigid rule-based thresholds. To counter the
hallucination risks inherent in generative models [ 9], we incorpo-
rate a temporal consistency verification module using a Contrastive
Forward-Backward Chain-of-Thought (CoT) [ 15,18,19] mechanism.
This self-correcting process validates the reasoning path to ensure
high confidence. Finally, to meet strict online latency constraints,
we transform these LLM-derived semantic judgments into light-
weight binary signals (0 /1flags) by directly comparing document
timeliness factors against the inferred threshold, which are then
seamlessly integrated into Baidu’s core ranking model. In summary,
our contributions are as follows:
•We propose a scalable LLM-based query-aware dynamic content
expiration prediction framework that replaces traditional rule-
based bucketing with fine-grained temporal validity deduction,
addressing the timestamp-validity mismatch in industrial search.
•We design a robust hallucination suppression mechanism tailored
for temporal reasoning, which integrates contrastive forward-
backward CoT with source authority signals to ensure high pre-
cision in identifying expired content.
•Our approach has been successfully deployed online to power
Baidu Search, delivering positive gains in search engagement.
2 Method
In this section, we introduce our proposed framework as Figure 2
shows.
2.1 Temporal Information Extraction Module
Acting as a post-retrieval refinement mechanism, this module adapts
"document-time anchoring" [ 10] into a discrete text-level frame-
work. This module distills a raw document Dinto a conciseFocusedSemantic Chunk Set 𝑆𝑓𝑜𝑐𝑢𝑠 to mitigate noise and attention dispersion
during LLM reasoning.
2.1.1 Anchor Information Parsing.Given a query 𝑄and candidate
documentsD, we first extract a query anchor pair A𝑄=(𝐾𝑄,𝑇𝑄),
comprising semantic keywords 𝐾𝑄and explicit or implicit temporal
entities𝑇𝑄(where𝑇𝑄=∅if undetected). Simultaneously, a temporal
parser traversesDto construct a Document Temporal Index I𝐷,
directly linking temporal expressions 𝑡𝐷
𝑗to sentence indices via a
{𝑇𝑖𝑚𝑒→𝑃𝑜𝑠𝑖𝑡𝑖𝑜𝑛}mapping.
2.1.2 Time-Anchored Context Refinement.Using I𝐷, we define a
sliding window of size 𝐿(default𝐿=5sentences) centered on each
anchor sentence, yielding an initial candidate chunk set 𝑆𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒 .
IfI𝐷=∅, we extract the document title and leading paragraphs as
a fallback. Next, to identify the most relevant context, we calculate
a composite score𝑆 𝑟𝑒𝑙(𝑐)for each text chunk𝑐∈𝑆 𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒 :
𝑆𝑟𝑒𝑙(𝑐)=I𝐾(𝑐∩𝐾𝑄≠∅)·
𝛼·𝑅𝑒𝑙𝐾(𝑐,𝐾𝑄)+(1−𝛼)·𝑅𝑒𝑙 𝑇(𝑐,𝑇𝑄)
(1)
where I𝐾is an indicator function equaling 1 if the chunk 𝑐contains
any query keywords and 0 otherwise. The hyperparameter 𝛼=0.6
balances semantic similarity ( 𝑅𝑒𝑙𝐾, via cosine similarity) and tem-
poral alignment ( 𝑅𝑒𝑙𝑇). Specifically, 𝑅𝑒𝑙𝑇prioritizes exact matches
and hierarchical overlaps (e.g., ’2025’ vs. ’2025 Q3’). Crucially, if
the query lacks explicit temporal constraints ( 𝑇𝑄=∅),𝑅𝑒𝑙𝑇dynam-
ically shifts to an exponential recency decay function exp(−𝜆Δ𝑡) ,
where Δ𝑡denotes the elapsed time. Chunks exceeding a threshold
𝜏form the finalFocused Semantic Chunk Set 𝑆𝑓𝑜𝑐𝑢𝑠 , serving as the
direct input for subsequent LLM reasoning.
2.2 Structured Prompting for Temporal
Inference
2.2.1 Core Prompting Principles.We design a structured prompt
framework to guide the LLM to perform precise temporal reason-
ing over𝑆𝑓𝑜𝑐𝑢𝑠 . Specifically, the prompt integrates the user query,
𝑆𝑓𝑜𝑐𝑢𝑠 , and the search time, alongside other contextual metadata, to
establish a strict absolute time anchor. The prompt establishes a log-
ical blueprint for expiration inference (e.g., inferring a target time
from start times and validity periods) and instructs the model to
leverage common sense for implicit date completion. Furthermore,
we integrate domain adaptation via few-shot prompting to recog-
nize periodic events and ensure robustness across diverse query
types. Finally, we employ negative constraints and CoT reasoning
to prevent the model from extracting irrelevant auxiliary dates or
conflating distinct temporal entities. The specific mechanisms for
this consistency verification are detailed in Section 2.3.
2.2.2 Temporal Optimization Objective.To enhance temporal pre-
cision, we define a joint optimization objective L𝑡𝑒𝑚𝑝 via a reward-
based alignment framework. Functioning as a negative reward
signal, the alignment penalty is formulated as:
L𝑡𝑒𝑚𝑝=D𝑡𝑖𝑚𝑒(𝑡𝑖𝑛𝑖𝑡,𝑡𝑔𝑡)+𝜆 1·L𝑔𝑟𝑎𝑛+𝜆 2·L𝑐𝑜𝑛𝑠 (2)
where𝑡𝑖𝑛𝑖𝑡and𝑡𝑔𝑡represent the LLM’s initial predicted candidate
and the ground-truth timestamp, respectively. Here, D𝑡𝑖𝑚𝑒(·)mea-
sures the normalized temporal distance.L 𝑔𝑟𝑎𝑛penalizes temporal
granularity mismatch, while L𝑐𝑜𝑛𝑠penalizes logical contradictions

RAG-Enhanced Large Language Models for Dynamic Content Expiration Prediction in Web Search Conference’17, July 2017, Washington, DC, USA
Temporal Information 
Extraction (Sec.2.1)
User Query
       (Q)Retrieval  
SystemRetrieved 
Documents
(D1,D2…)
Temporal Information
Extraction  Module
Time-Anchored
Context RefinementSignal
Transformation
（0/1）
Anchor 
Info
ParsingQuery Anchor Pair(AQ)
Doc Temporal Index(ID)
Scoring
&
Selection
OUTPUTDocument
Temporal
Factor
(��)
Focused 
Semantic
Chunk Set
        (Sfocus)Explicit
Rules
Negative
ConstraintsDomain
AdaptationImplicit
Completion
Temporal
Consistency
Verification
(CFB-CoT)
Query
Expiration
Threshold
(�풆� ∗(�))
Structured Prompting &
Temporal Inference (LLM)
Distilled Student 
LLM
Teacher LLM
INPUTOffline Dictionary 
CacheBaidu’s Aurora
Ranking Model
Freshness
Sub-network
Multi-order Feature
Interaction & Cross
Relevance
ScoreDoc
Authority
Freshness
Signal
Final 
Ranked 
DocsLLM-Based Temporal 
Inference (Sec 2.2 & 2.3)Online  Feature
Application (Sec.2.4)
Figure 2: An overview of our proposed LLM-based Query-Aware Dynamic Content Expiration Prediction Framework.
during CoT reasoning (both formally defined in Section 2.3). Hy-
perparameters 𝜆1and𝜆2are empirically tuned to balance these
objectives.
2.3 Temporal Consistency Verification Module
To resolve conflicting temporal cues in 𝑆𝑓𝑜𝑐𝑢𝑠 and mitigate LLM
hallucinations, we implement a dual-verification architecture.
The first mechanism,Consistency Forward-Backward Rea-
soning, acts as an internal check. The LLM generates an initial
candidate𝑡𝑖𝑛𝑖𝑡via forward inference, then performs backward
stress-testing against the retrieved evidence to verify logical co-
herence. We quantify this alignment using a consistency penalty:
L𝑐𝑜𝑛𝑠=1−𝑆𝑠𝑒𝑙𝑓(traj𝑓𝑤𝑑,traj𝑏𝑤𝑑), where𝑆𝑠𝑒𝑙𝑓is the LLM’s self-
evaluated logical consistency score to capture conflicts. To penalize
precision inconsistencies, we introduce a granularity penalty:
L𝑔𝑟𝑎𝑛=|𝐷(𝑡𝑖𝑛𝑖𝑡)−𝐷(𝑡𝑔𝑡)|(3)
where𝐷(·) maps the temporal resolution to a hierarchical depth
level (e.g., year=1, month=2, day=3), penalizing the hierarchical dis-
tance of mismatched tiers. Minimizing these non-differentiable
penalties via reward-based alignment ensures coherent logical
scales and effectively reduces hallucinated drifts.
The second mechanism,Authority-Weighted Evidence Fu-
sion, resolves external conflicts among divergent sources by max-
imizing collective support for the optimal expiration timestamp
𝑡∗
𝑒𝑥𝑝:
𝑡∗
𝑒𝑥𝑝=arg max
𝑡𝑐∈T𝑐∑︁
𝑐∈𝑆 𝑓 𝑜𝑐𝑢𝑠𝑤𝑐·I𝑎𝑙𝑖𝑔𝑛(𝑡𝑐,𝑐)(4)
whereT𝑐denotes the candidate timestamps generated by the LLM.
Here,𝑤𝑐=𝐴𝑢𝑡ℎ(𝑐)·𝑆 𝑟𝑒𝑙(𝑐)serves as the joint chunk weight,
combining source authority and semantic relevance (Section 2.1).
The indicator I𝑎𝑙𝑖𝑔𝑛(𝑡𝑐,𝑐)equals 1 if𝑡𝑐logically aligns with evidence𝑐. This robust mechanism ensures the most authoritative evidence
dictates the output.
2.4 Online Feature Application
2.4.1 Signal Transformation and Parameter Acquisition.For any
query-document pair (𝑞,𝑑𝑖), we compare the query’s expiration
threshold𝑡∗
𝑒𝑥𝑝(𝑞)with the document’s temporal factor 𝑡𝑖to generate
a binary prior signal:
𝑓(𝑖)
exp=I(𝑡𝑖>𝑡∗
𝑒𝑥𝑝(𝑞))(5)
where I(·)is the indicator function. A value of 1 denotes the
document is LLM-validated as fresh, while 0 indicates expiration.
For the query threshold 𝑡∗
𝑒𝑥𝑝(𝑞)is obtained from an offline dic-
tionary cache, a distilled student LLM, or via direct online calls
to the teacher LLM. Regarding the document temporal factor 𝑡𝑖,
the publication time ( 𝑡𝑝𝑢𝑏) can be used as it typically correlates
with the occurrence of events in most scenarios. Alternatively, an
intrinsic content time can be extracted via LLM-based semantic
comprehension.
2.4.2 Robustness and Circuit Breaking.To ensure system reliability,
we implement aCircuit Breaking & Fallbackmechanism. If real-time
inference times out, the generated threshold exhibits significant
distributional shifts, or cache misses occur, the system triggers a
failsafe mode where𝑓(𝑖)
expdefaults to 0, gracefully degrading to the
baseline ranking without introducing noise.

Conference’17, July 2017, Washington, DC, USA Tingyu Chen et al.
2.4.3 Deep Ranking Model Integration.Finally, 𝑓(𝑖)
expis injected into
Baidu’s ranking architecture as a learnable soft constraint via multi-
order feature interactions. Specifically, within the Freshness sub-
network,𝑓(𝑖)
expinteracts with other core signals:
Score(𝑑𝑖,𝑞)=Φ
x𝑖,𝑓(𝑖)
exp,Cross(𝑓(𝑖)
exp,𝑆𝑟𝑒𝑙(𝑑𝑖,𝑞),𝐴𝑢𝑡ℎ(𝑑 𝑖),...);Θ
(6)
This formulation converts the discrete temporal prior into adaptive
weights, enabling Aurora to dynamically balance freshness against
other factors for a globally optimal ranking.
3 Experiments
To verify the effectiveness of the proposedLLM-based dynamic con-
tent expiration prediction framework, we conducted three progres-
sive experiments: offline model evaluation, online A/B testing on
Baidu Search, and human evaluation. All experiments compare the
productionAuroraranking model (denoted asAurora-Baseline)
against the experimental model fused with the binary expiry time
feature𝑓exp(denoted asAurora-Expiry). Statistical significance
was verified for all results (𝑝<0.05).
3.1 Offline Model Evaluation
The dataset was constructed from Baidu Search logs over a three-
month period, comprising 1 million valid queries and 10 million
candidate documents across diverse domains.
Table 1 shows that Aurora-Expiry significantly enhances overall
search quality, evidenced by an increased global Satisfaction Score
and a substantial PNR surge for High-Freshness queries. Crucially,
the simultaneous decrease in bothday_away@4andday_away@10
confirms a synchronized freshness improvement across the ranking
list, demonstrating that the LLM-derived 𝑓expacts as an effective
semantic filter to precisely penalize obsolete content. Unlike naive
recency boosts that blindly promote new but less relevant docu-
ments, the concurrent uplift in the Satisfaction Score proves that
our approach improves timeliness without compromising semantic
accuracy. Furthermore, other general relevance metrics across the
random test set remained highly stable with negligible fluctuations,
confirming the framework’s safety and robustness in large-scale
production.
Table 1: Offline Evaluation Results. Satisfaction Score and
Positive Navigation Rate (PNR) measure freshness prefer-
ence (higher is better). The day_away metric gauges the me-
dian time factor of top results (lower is fresher).
Metric Rel. Imp. (Δ%)
Satisfaction Score +0.52%
PNR (Tier 2:≤1 month) +5.21%
PNR (Tier 3:≤1 week) +7.36%
day_away@4 -2.44%
day_away@10 -3.19%3.2 Online A/B Test
To evaluate the real-world impact, we conducted a 14-day A/B test
on Baidu Search’s live production traffic (5% per group) across all
platforms.
As detailed in Table 2, Aurora-Expiry significantly improves
ranking freshness, highlighted by a 12.81% drop in the median
day_away@4for High-Freshness queries. This consistent reduction
in content age across both top-4 and top-10 positions confirms
that𝑓expeffectively filters expired content across the entire ranking
list. Crucially, this successful temporal optimization directly drives
core business value, yielding significant and consistent gains in
both Click-Through Rate (CTR) and Satisfactory Consumption (SC).
Coupled with a positive Search Retention Rate, these results prove
that the framework goes beyond simply surfacing recent documents
to attract clicks; instead, it delivers genuine post-click value by
prioritizing valid, up-to-date information, thereby comprehensively
elevating the overall search experience.
Table 2: Online A/B Test Results (Relative Improvement Δ%).
Category Metric @4 @10
User ExperienceSatisfactory Cons. +0.78%
CTR +0.41%
Search Activity +0.12%
Search Retention Rate +0.06%
Overall Freshnessday_away (Median) -6.87% -6.24%
day_away (Mean) -6.70% -5.94%
High-Freshnessday_away (Median) -12.81% -11.46%
day_away (Mean) -12.94% -10.91%
3.3 Human Evaluation
To authentically simulate real user perception, we conducted a blind
GSB (Good/Same/Bad) human evaluation. As shown in Table 3,
Aurora-Expiry demonstrates pronounced superiority in complex
search contexts. Notably, the “Long-Tail / Cold Demand” cohort
achieved an outstanding 12:2 Good-to-Bad (G:B) ratio. Conversely,
the “Random Demand” cohort exhibits a more modest advantage
ratio, though still maintaining a solid 5:2 G:B ratio. For such broad
queries lacking explicit temporal constraints, this net-positive re-
sult confirms that 𝑓expsafely processes general, non-time-sensitive
traffic without degrading overall search stability. Furthermore, the
strong positive results in the time-sensitive cohort (6:1 G:B ratio)
validate the framework’s precision in capturing dynamic temporal
intents.
Table 3: Human Evaluation (GSB) across key scenarios.
Scenario Advantage Ratio G:B Ratio
Long-Tail / Cold Demand 5.00% 12:2
Time-Sensitive 2.50% 6:1
Random Demand 1.50% 5:2

RAG-Enhanced Large Language Models for Dynamic Content Expiration Prediction in Web Search Conference’17, July 2017, Washington, DC, USA
4 Conclusion
In this paper, we propose a novel LLM-based Dynamic Content
Expiration framework that shifts the paradigm of timeliness ranking
in Baidu Search. Moving beyond rigid rules, our approach extracts
fine-grained temporal contexts and applies dual-stage consistency
verification to deduce reliable, query-specific validity horizons. By
distilling these semantic judgments into lightweight discrete signals
𝑓exp, we successfully bridge complex temporal reasoning with the
strict latency constraints of a production ranker. Extensive offline
evaluations and online A/B test validate the framework’s profound
impact on holistic search quality.
References
[1]Abdelrahman Abdallah, Bhawna Piryani, Jonas Wallat, Avishek Anand, and Adam
Jatowt. 2025. TempRetriever: Fusion-based Temporal Dense Passage Retrieval
for Time-Sensitive Questions. arXiv:2502.21024 [cs.IR] https://arxiv.org/abs/
2502.21024
[2]Anlei Dong, Yi Chang, Zhaohui Zheng, Gilad Mishne, Jing Bai, Ruiqiang Zhang,
Karolina Buchner, Ciya Liao, and Fernando Diaz. 2010. Towards recency ranking
in web search. InProceedings of the Third ACM International Conference on Web
Search and Data Mining(New York, New York, USA)(WSDM ’10). Association
for Computing Machinery, New York, NY, USA, 11–20. https://doi.org/10.1145/
1718487.1718490
[3]Anlei Dong, Ruiqiang Zhang, Pranam Kolari, Jing Bai, Fernando Diaz, Yi Chang,
Zhaohui Zheng, and Hongyuan Zha. 2010. Time is of the essence: improv-
ing recency ranking using Twitter data. InProceedings of the 19th Interna-
tional Conference on World Wide Web(Raleigh, North Carolina, USA)(WWW
’10). Association for Computing Machinery, New York, NY, USA, 331–340.
https://doi.org/10.1145/1772690.1772725
[4]Rujun Han, Xiang Ren, and Nanyun Peng. 2021. ECONET: Effective
Continual Pretraining of Language Models for Event Temporal Reasoning.
arXiv:2012.15283 [cs.CL] https://arxiv.org/abs/2012.15283
[5]Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and
Cyrus Rashtchian. 2025. Sufficient Context: A New Lens on Retrieval Augmented
Generation Systems. arXiv:2411.06037 [cs.CL] https://arxiv.org/abs/2411.06037
[6]Yansheng Mao, Yufei Xu, Jiaqi Li, Fanxu Meng, Haotong Yang, Zilong Zheng,
Xiyuan Wang, and Muhan Zhang. 2026. LIFT: A Novel Framework for En-
hancing Long-Context Understanding of LLMs via Long Input Fine-Tuning.
arXiv:2502.14644 [cs.CL] https://arxiv.org/abs/2502.14644
[7]Joon Park, Kyohei Atarashi, Koh Takeuchi, and Hisashi Kashima. 2025. Emulating
Retrieval Augmented Generation via Prompt Engineering for Enhanced Long
Context Comprehension in LLMs. arXiv:2502.12462 [cs.CL] https://arxiv.org/
abs/2502.12462
[8]Chao Shang, Peng Qi, Guangtao Wang, Jing Huang, Youzheng Wu, and Bowen
Zhou. 2021. Open temporal relation extraction for question answering. In3rd
Conference on Automated Knowledge Base Construction.
[9]Yedan Shen, Kaixin Wu, Yuechen Ding, Jingyuan Wen, Hong Liu, Mingjie Zhong,
Zhouhan Lin, Jia Xu, and Linjian Mo. 2025. Alleviating LLM-based GenerativeRetrieval Hallucination in Alipay Search. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’25). ACM, 4294–4298. https://doi.org/10.1145/3726302.3731951
[10] Xintong Song, Bin Liang, Yang Sun, Chenhua Zhang, Bingbing Wang, and Ruifeng
Xu. 2025. Bridging Time Gaps: Temporal Logic Relations for Enhancing Temporal
Reasoning in Large Language Models. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Padua, Italy)(SIGIR ’25). Association for Computing Machinery, New York, NY,
USA, 3040–3044. https://doi.org/10.1145/3726302.3730173
[11] Andrey Styskin, Fedor Romanenko, Fedor Vorobyev, and Pavel Serdyukov. 2011.
Recency ranking by diversification of result set. InProceedings of the 20th ACM
international conference on Information and knowledge management (CIKM ’11).
ACM, 1949–1952. https://doi.org/10.1145/2063576.2063862
[12] Zhaochen Su, Juntao Li, Jun Zhang, Tong Zhu, Xiaoye Qu, Pan Zhou, Yan Bowen,
Yu Cheng, and Min zhang. 2024. Living in the Moment: Can Large Language
Models Grasp Co-Temporal Reasoning? arXiv:2406.09072 [cs.CL] https://arxiv.
org/abs/2406.09072
[13] Zhaochen Su, Jun Zhang, Tong Zhu, Xiaoye Qu, Juntao Li, Min Zhang, and Yu
Cheng. 2024. Timo: Towards Better Temporal Reasoning for Language Models.
arXiv:2406.14192 [cs.CL] https://arxiv.org/abs/2406.14192
[14] Zhao Wang, Ziliang Zhao, and Zhicheng Dou. 2025. TimeRAG: Enhancing Com-
plex Temporal Reasoning with Search Engine Augmentation. InProceedings of the
34th ACM International Conference on Information and Knowledge Management
(Seoul, Republic of Korea)(CIKM ’25). Association for Computing Machinery,
New York, NY, USA, 3230–3239. https://doi.org/10.1145/3746252.3761425
[15] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. arXiv:2201.11903 [cs.CL] https:
//arxiv.org/abs/2201.11903
[16] Siheng Xiong, Ali Payani, Ramana Kompella, and Faramarz Fekri. 2024. Large
Language Models Can Learn Temporal Reasoning. arXiv:2401.06853 [cs.CL]
https://arxiv.org/abs/2401.06853
[17] Wanqi Yang, Yanda Li, Meng Fang, and Ling Chen. 2024. Enhancing Temporal
Sensitivity and Reasoning for Time-Sensitive Question Answering. InFindings of
the Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mo-
hit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics,
Miami, Florida, USA, 14495–14508. https://doi.org/10.18653/v1/2024.findings-
emnlp.848
[18] Liang Yao. 2025. Large Language Models are Contrastive Reasoners.
arXiv:2403.08211 [cs.CL] https://arxiv.org/abs/2403.08211
[19] Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da
Zheng, Huajun Chen, and Ningyu Zhang. 2025. LightThinker: Thinking Step-by-
Step Compression. arXiv:2502.15589 [cs.CL] https://arxiv.org/abs/2502.15589
[20] Xinliang Frederick Zhang, Nick Beauchamp, and Lu Wang. 2024. Narrative-
of-Thought: Improving Temporal Reasoning of Large Language Models via Re-
counted Narratives. InFindings of the Association for Computational Linguis-
tics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.).
Association for Computational Linguistics, Miami, Florida, USA, 16507–16530.
https://doi.org/10.18653/v1/2024.findings-emnlp.963
[21] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong
Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2025. Large
Language Models for Information Retrieval: A Survey.ACM Transactions on
Information Systems44, 1 (Nov. 2025), 1–54. https://doi.org/10.1145/3748304
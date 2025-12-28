# Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches

**Authors**: Chaithra, Kamesh Kadimisetty, Biju R Mohan

**Published**: 2025-12-23 06:27:12

**PDF URL**: [https://arxiv.org/pdf/2512.20082v2](https://arxiv.org/pdf/2512.20082v2)

## Abstract
Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.

## Full Text


<!-- PDF content starts -->

Adaptive Financial Sentiment Analysis for NIFTY 50 via
Instruction-Tuned LLMs , RAG and Reinforcement Learning
Approaches
Chaithra
National Institute of Technology
Karnataka
Surathakal, India
chaithra.217it001@nitk.edu.inKamesh Kadimisetty
Gayatri Vidya Parishad College of
Engineering
Visakhapatnam, India
kameshkadimisetty@gmail.comBiju R Mohan
National Institute of Technology
Karnataka
Surathkal, India
biju@nitk.edu.in
Abstract
Financial sentiment analysis plays a crucial role in informing in-
vestment decisions, assessing market risk, and predicting stock
price trends. Existing works in financial sentiment analysis have
not considered the impact of stock prices or market feedback on
sentiment analysis. In this paper, we propose an adaptive frame-
work that integrates large language models (LLMs) with real-world
stock market feedback to improve sentiment classification in the
context of the Indian stock market. The proposed methodology
fine-tunes the LLaMA 3.2 3B model using instruction-based learn-
ing on the SentiFin dataset. To enhance sentiment predictions, a
retrieval-augmented generation (RAG) pipeline is employed that
dynamically selects multi-source contextual information based on
the cosine similarity of the sentence embeddings. Furthermore, a
feedback-driven module is introduced that adjusts the reliability of
the source by comparing predicted sentiment with actual next-day
stock returns, allowing the system to iteratively adapt to market
behavior. To generalize this adaptive mechanism across temporal
data, a reinforcement learning agent trained using proximal policy
optimization (PPO) is incorporated. The PPO agent learns to opti-
mize source weighting policies based on cumulative reward signals
from sentiment-return alignment. Experimental results on NIFTY
50 news headlines collected from 2024 to 2025 demonstrate that
the proposed system significantly improves classification accuracy,
F1-score, and market alignment over baseline models and static
retrieval methods. The results validate the potential of combining
instruction-tuned LLMs with dynamic feedback and reinforcement
learning for robust, market-aware financial sentiment modeling.
CCS Concepts
•Computing methodologies→Sentiment Analysis.
Keywords
Large Language Models, Sentiment Analysis, Retrieval Augmented
Generation, Reinforcement Learning
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
CODS Dec’25, Pune, India
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXACM Reference Format:
Chaithra, Kamesh Kadimisetty, and Biju R Mohan. 2025. Adaptive Financial
Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and
Reinforcement Learning Approaches. InProceedings of Research Track, CODS
(CODS Dec’25).ACM, New York, NY, USA, 8 pages. https://doi.org/XXXX
XXX.XXXXXXX
1 Introduction
In financial markets, investor behavior is often influenced by quali-
tative information, including news headlines, corporate disclosures,
and economic prospects [4, 20]. Extracting sentiment from such
textual data provides valuable information for understanding stock
movements, guiding portfolio decisions, and developing algorith-
mic trading strategies. The analysis of sentiment in financial text
is a challenging task due to the domain-specific language and the
dynamic nature of the market [5].
Large language models (LLMs) have shown good performance
in various natural language tasks, but they often fall short in the
finance domain [11, 10, 21]. This is because models typically lack
exposure to domain-specific semantics and are not linked to real-
world financial feedback; as a result, the sentiment classifications
often diverge from actual market responses. News headlines are
widely used for sentiment-driven prediction tasks [7, 2, 6]. Head-
lines generally capture the core event and provide stronger, more
concentrated sentiment signals than full articles. Although full arti-
cles offer detailed background information, they often introduce
noise that can dilute predictive performance. However, headlines
from certain sources may be extremely brief and lack sufficient
contextual information. Therefore, aggregating news from multiple
trusted sources is essential to obtain a more comprehensive under-
standing of the underlying events. The existing approaches have
not considered the financial contexts from multiple sources and the
reliability of news sources over time.
To address the above issues, we proposed an adaptive sentiment
analysis framework that integrates instruction-tuned LLMs with
multi-source news data retrieval and a feedback mechanism. We
fine-tuned the LLaMA 3.2 3B model using instruction-formatted
financial data. We then augmented each input news query with
dynamically retrieved context news from trusted financial sources
using a retrieval-augmented generation (RAG) approach. The relia-
bility of these sources, determined by source weightage, is adjusted
based on the alignment of stock returns. We also employed a re-
inforcement learning agent using proximal policy optimization
(PPO) to learn weighting strategies that evolve in response to news
patterns and stock market behavior.arXiv:2512.20082v2  [cs.AI]  24 Dec 2025

CODS Dec’25, December 17–20, 2025, Pune, India Chaithra, Kamesh Kadimisetty, and Biju R Mohan
We created a news dataset comprising NIFTY 50 companies.The
sentiment labels are derived from market-adjusted return move-
ments. Our feedback-aware system outperforms traditional LLMs
and static RAG pipelines in both accuracy and alignment with mar-
ket trends. Through this approach, we tried to bridge the gap be-
tween LLM capabilities and financial domain requirements, thereby
offering an approach to sentiment analysis that is both context-
sensitive and feedback-driven. This approach can be integrated
with more powerful LLMs or domain-specific architectures in the
future.
The key contributions made by our work can be summarized as
follows:
•We fine-tuned the LLaMA 3.2 3B model using an instruction-
based prompt with domain-specific sentiment data to enable
the LLM to understand financial sentiment in an Indian mar-
ket context.
•We constructed a test dataset for NIFTY 50 companies from
multiple financial sources, where ground-truth sentiment
labels assignment is based on next-day stock returns relative
to rolling mean and volatility.
•We designed a retrieval-augmented generation (RAG) pipeline
that incorporates multi-source contextual evidence, and the
top k relevant news articles were retrieved based on cosine
similarity of the sentence embeddings.
•We implemented a direct feedback mechanism that updates
the source weights by comparing sentiment predictions with
actual next-day stock returns, allowing the system to align
more closely with market behavior.
•We developed a reinforcement learning agent using proxi-
mal policy optimization (PPO) to optimize context source
weights over time, thereby learning a feedback-aware re-
trieval strategy that generalizes across days.
The remainder of the paper is structured as follows: Section 2
reviews the literature on financial sentiment analysis and large
language models (LLMs). Section 3 describes the datasets used in
the study. The section 4 discusses the fine-tuning process, retrieval-
augmented generation (RAG), and reinforcement learning. Section
5 explains the evaluation metrics and strategy. Section 6 presents
the experimental results and analysis. Finally, Section 7 concludes
the paper and highlights directions for future research.
2 Literature Review
Financial sentiment analysis has been performed using dictionary-
based evaluations, machine learning (ML), deep learning (DL), trans-
former models like FinBERT, and, more recently, large language
models (LLMs). This section surveys key contributions from each
phase.
2.1 Dictionary-Based Approaches
The earliest sentiment analysis was performed using manually cu-
rated sentiment lexicons. Tools like SentiWordNet, VADER, and the
Loughran-McDonald Financial Sentiment Dictionary [12] provided
word-level polarity scores, enabling straightforward sentiment clas-
sification by counting positive and negative tokens. While these
methods were interpretable and domain-specific to some extent,
they lacked contextual understanding and failed in the presence ofsarcasm, negations, or complex sentence structures. Their inability
to adapt to new language patterns and lack of feedback integration
limited their effectiveness in financial domains.
2.2 Machine Learning Techniques
Dictionary-based methods often do not consider the context and
ignore domain-specific meanings [13]; therefore, to improve gener-
alization, classical ML models that are trained with sentence label
pairs, such as Naive Bayes, support vector machines (SVM), and
logistic regression, were employed [15]. These models relied on
handcrafted features, such as n-grams, POS tags, and TF-IDF vec-
tors, and outperformed lexicon-based techniques in many scenarios.
However, they still struggled with long-range dependencies and
syntactic structure. Moreover, these models were typically static
post-training and did not incorporate time-evolving market behav-
ior.
2.3 Deep Learning Techniques
The deep learning models enabled the automatic extraction and
modeling of complex linguistic structures [17]. Models such as
CNNs, RNNs, and LSTMs [14] have been widely used for senti-
ment classification, as they can capture sequential dependencies in
text, making them well-suited for analyzing financial reports and
news articles. But these models require more data for training, are
domain-agnostic, and tend to overfit, especially in low-resource
financial domains. Additionally, these approaches also did not lever-
age external feedback or context beyond the input sequence.
2.4 Transformer Models and FinBERT
The transformer architecture, such as BERT [8], with self-attention
mechanisms and deep contextual embeddings, has demonstrated
improved performance in a wide variety of NLP tasks, including
sentiment analysis. FinBERT [1], a financial-domain adaptation of
BERT, was pre-trained on financial documents, such as SEC filings,
and has outperformed deep learning models in finance-specific
tasks. These models are text-only models and lack mechanisms to
adapt based on real-world feedback, such as stock market responses.
2.5 Large Language Models (LLMs)
LLMs like GPT-3 [3], PaLM, and LLaMA [19] have shown improve-
ment in zero-shot and few-shot sentiment analysis tasks, but often
underperform in finance due to domain-specific text and context
[21]. Instruction-tuned versions of these models have also demon-
strated strong performance in sentiment classification [9]. In many
earlier works, real-time market signals were not incorporated, mak-
ing them less suitable for adaptive financial decision-making tasks.
2.6 Instruction Alignment and Feedback-Aware
Models
To bridge this gap, recent research has focused on aligning LLMs
not only with human instructions but also with real-world feedback.
Zhang et al [22] found that the concise nature of financial news
often lacks sufficient context, which can significantly decline the re-
liability of LLM in financial sentiment analysis. To address it, they
introduced a retrieval-augmented LLM framework for financial

Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches CODS Dec’25, December 17–20, 2025, Pune, India
sentiment analysis. Zhao et al. [23] introduced a novel adaptive sen-
timent framework using Retrieval-Augmented Generation (RAG)
pipelines, where external context from multiple sources is com-
bined with LLMs for inference. They incorporated feedback from
stock market returns to dynamically adjust source weights via rein-
forcement learning, thereby making the model context-aware and
market-responsive. Our work builds on this paradigm by applying
the methodology to the Indian equity market and NIFTY 50 data.
We extend it by combining domain-specific fine-tuning of LLaMA
3.2, retrieval with dynamic feedback, and PPO-based reinforcement
learning for robust and generalizable source optimization.
3 Datasets
We utilized two primary datasets in different stages of the pipeline:
3.1 Fine-Tuning Dataset
We utilizedSentiFindataset [18] as the primary source for instruction-
based fine-tuning of the LLaMA 3.2 3B model. This dataset con-
tains news data for the Indian stock market. The dataset comprises
10,572 labeled headlines across three sentiment classes: positive ,
neutral , and negative . Each instance includes a short financial
headline, the sentiment label, and additional metadata such as the
news source and publication date. The samples are formatted into
instruction-based prompt-response pairs to train the LLM. Table 1
presents the distribution of sentiment labels:
Table 1: SentiFin Dataset Statistics
Sentiment Class Count
Positive 4,505
Neutral 3,695
Negative 3,386
We split the dataset into training and testing sets using an 80:20
ratio, ensuring a stratified distribution of sentiment classes.
3.2 Retrieval Augmented Generation (RAG)
Dataset and Test Dataset:
We created a dataset of 8,000 news headlines related to NIFTY 50
companies, spanning 2024–2025, to support the RAG pipeline. This
news data is collected through webscraping. All news data was col-
lected from trusted Indian financial sources, such as Yahoo Finance
and MoneyControl. The major news source details are given in the
Table 2. Headlines were cleaned using basic NLP preprocessing,
which included the removal of HTML tags, normalization of dates
and entities, and mapping of tickers to standard NSE symbols.
We also constructed a separate test dataset and generated its
ground-truth sentiment labels. The sentiment labeling was carried
out using the following strategy.
•Daily returns were computed as the percentage change in
opening price.
•For each stock, we computed a 30-day rolling mean and
standard deviation of returns.Table 2: RAG Dataset Statistics
Source Count
Business Standard 1031
NDTV Profit 1002
Financial Express 830
The Economic Times 743
Mint 699
MoneyControl 585
Business Today 449
ET Now 335
•Each headline was aligned with the next trading day’s return.
If historical price statistics were unavailable, the label was
marked asunknown.
•Sentiment labels were assigned based on the deviation of the
next-day return from the rolling statistics:
– Positive:Return > Mean + Std
– Negative:Return < Mean – Std
– Neutral:Otherwise
This objective labeling approach avoids subjective human anno-
tation and grounds sentiment in actual market behavior. The test
dataset statistics are given in the Table 3.
Table 3: Test Dataset Statistics
Sentiment Class Count
Positive 823
Negative 1188
Neutral 4123
4 Methodology
The proposed adaptive financial sentiment analysis system consists
of the following modular components:
(1)Fine-Tune LLaMA :This module instruction-tune LLaMa
3.2 on financial sentiment data to classify news headlines
intopositive,negative, orneutralsentiment.
(2)Retrieval-Augmented Generation (RAG) Module:Dy-
namically retrieves top- 𝑘context snippets from a corpus of
NIFTY 50 financial news using sentence embeddings and
cosine similarity, with source-specific weighting.
(3)Market Feedback Engine:Compares model predictions
with next-day stock price movements to reinforce or penalize
the reliability scores of sources, simulating an environment-
aware feedback loop.
(4)Source Credibility Weighting Mechanism:This mod-
ule assigns credibility weights to news sources using both
rule-based and reinforcement-learning-based optimization
strategies.
A visual overview is shown in Figure 1.

CODS Dec’25, December 17–20, 2025, Pune, India Chaithra, Kamesh Kadimisetty, and Biju R Mohan
Figure 1: Overview of the proposed adaptive sentiment analysis pipeline
4.1 Instruction-Based Fine-Tuning of LLaMA 3.2
We fine-tuned the unsloth/Llama-3.2-3B-Instruct1model us-
ing the quantized low-rank adaptation (QLoRA) technique, which
enables memory-efficient training on GPUs. The finetuning process
is outlined below.
•Instruction Formatting:The training samples are format-
ted in the LLaMA Instruct format. The structure is:
<s>[INST] Classify the sentiment of the following
financial sentence: [headline] [/INST] [label]</s>
•Model and Adapter Configuration:The Unsloth imple-
mentation is used for memory-optimized fine-tuning. LoRA
adapters are applied to the attention and MLP layers ( q_proj ,
k_proj ,v_proj ,o_proj ,gate_proj ,up_proj ,down_proj ),
with a rank of 16 and LoRA alpha set to 16.
•Training Details:HuggingFace’s SFTTrainer with 4-bit
quantization, mixed precision (fp16), a batch size of 4 (with
gradient accumulation), and a learning rate of 2e-5 is used
for training. The model is trained for 3 epochs and saves the
best-performing checkpoints to disk.
•Evaluation:The model is validated using accuracy and F1-
score on a held-out test set.
This instruction-based fine-tuning enhances the model’s ability
to perform financial sentiment classification in the Indian stock
market.
4.2 Retrieval-Augmented Generation (RAG)
To provide additional context for the headline with relevant and
sentiment-aligned background information, we implemented a
Retrieval-Augmented Generation (RAG). It consists of the following
key components:
(1)Candidate Document Retrieval:Given a test headline,
stock symbol, and date, retrieve a set of candidate news
articles from a RAG dataset, within a 3-day window centered
around the test date. It retrieves from sources with non-zero
reliability weights.
1https://huggingface.co/unsloth/Llama-3.2-3B-Instruct(2)Sentiment-Aware Filtering:Filter retrieved documents
based on sentiment cues present in the headline, such as
"gain", "fall", "stable". This process ensures contextual align-
ment and minimizes noise.
(3)Sentence Embedding and Scoring:The headline and can-
didate headings are embedded using the all-MiniLM-L6-v2
model. Cosine similarity scores are computed between the
headline and each candidate. These are then multiplied by the
corresponding source reliability weights to get a weighted
relevance score.
(4)Top-k Context Selection and Prompt Construction:The
top-ranked documents are selected based on the weighted rel-
evance score. These are concatenated with the original head-
line to form a single instruction-formatted prompt, which is
then passed to the fine-tuned LLaMA model for final senti-
ment prediction.
This adaptive RAG layer enables the model to retrieve exter-
nal context selectively from reliable sources, thereby improving
classification accuracy on ambiguous or context-poor headlines.
4.3 Source Credibility Weighting Mechanism
Credibility weights for news sources are assigned using dynamic
source weighting, based on market feedback and reinforcement
learning, as described in the following sections.
4.3.1 Dynamic Source Weighting via Market Feedback.To validate
the credibility of various financial sources, we employed a dynamic
source weighting approach guided by real-world market feedback.
Initially, each context source, such as the Economic Times and
Moneycontrol, is assigned a normalized reliability weight. During
inference, retrieved documents from these sources contribute addi-
tional context to the prompt, and the final sentiment prediction is
compared with the true market response.
After each prediction:
•If the predicted sentiment aligns with the actual stock move-
ment, for example, a positive sentiment followed by a stock
price increase, the weights of the contributing sources are
increased.

Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches CODS Dec’25, December 17–20, 2025, Pune, India
•Conversely, misaligned predictions lead to penalization of
the associated sources by reducing their weights.
•A neutral return zone±0.5% is defined to avoid penalizing
predictions where the price change is not significant.
Weight updates are performed by a lightweight gradient-style
rule:
𝑤new=clamp(𝑤 old±𝛼)(1)
where𝛼is a small learning rate of1 ×10−4, and weights are normal-
ized after each update to ensure they sum to 1. This feedback loop
iterates over the entire evaluation dataset. It supports sources that
align with observed stock trends and demotes those that frequently
contribute to incorrect predictions.
By integrating this mechanism into our RAG pipeline, the model
continuously adjusts its source reliability, thereby learns a context
selection strategy that is sensitive to historical prediction perfor-
mance and market behavior.
4.3.2 Dynamic Source Weighting via Reinforcement Learning.The
direct market feedback updates source weights, but it leads to short-
term, unstable adjustments. To enable more robust generalization
across unseen data and source combinations, we model the source
weighting process as a reinforcement learning (RL) task and apply
proximal policy optimization (PPO). The PPO’s surrogate clipped
objective provides stable and conservative policy updates, prevent-
ing divergence in dynamic or noisy environments [16]. The key
components of PPO-based reinforcement learning are:
•Environment:In a reinforcement learning environment,
each episode corresponds to a sequence of sentiment predic-
tion tasks. At each step, the environment provides a financial
headline, its symbol, and date, and the agent selects a weight
distribution over the context sources.
•State:The state of the PPO agent consists of the current
normalized source weights, combined with features from the
query headline or past performance.
•Action:The agent outputs a normalized vector representing
updated weights for context sources within the RAG pipeline.
•Reward:In the reward function, a reward of +1 is assigned
if the predicted sentiment aligns with the ground-truth label
derived from next-day stock returns, and -1 otherwise. A
neutral return zone can be used to clip insignificant reward
effects.
The PPO agent is trained across multiple time steps to maxi-
mize the cumulative reward. Agent learns an adaptive strategy for
weighting sources based on both the current context and long-term
trends. The PPO agent training loop interacts with the environ-
ment by performing document retrieval, constructing prompts, and
inferring models at each step, thereby it simulates a full pipeline
execution. After training, the agent’s final policy is used to gener-
ate the optimized source weight vector. This weight vector is then
applied to evaluate the full test set. The advantage of this approach
is that it enables exploration across sources and stabilizes the re-
trieval mechanism under uncertain market conditions. The details
of weight refinements using PPO is given in the Algorithm 1.Algorithm 1:RAG Source Weights using PPO
Input:Initial weightsw 0, Test setD 𝑡𝑒𝑠𝑡, PPO
hyperparameters(𝜂,𝜖)
Output:Optimized source weightsw∗
Initialize policy𝜋 𝜃and value𝑉 𝜙;
Initialize environment state
𝑠0=(w 0,accuracy history,market indicators);
whilenot convergeddo
Sample batch of trajectories:
(1) Select actiona 𝑡=𝜋𝜃(𝑠𝑡)representing new source weights
(2) Normalizea 𝑡soÍ
𝑖𝑎𝑡,𝑖=1
(3) Perform RAG inference witha 𝑡to predict ˆ𝑦𝑡
(4) Compute reward𝑟 𝑡=+1if ˆ𝑦𝑡=𝑦𝑡, else−1
(5) Observe next state𝑠 𝑡+1
Estimate advantage𝐴 𝑡=𝑟𝑡+𝛾𝑉𝜙(𝑠𝑡+1)−𝑉𝜙(𝑠𝑡);
Update policy by maximizing PPO clipped objective:
𝐿CLIP(𝜃)=E𝑡[min(𝑟𝑡(𝜃)𝐴𝑡,clip(𝑟𝑡(𝜃),1−𝜖,1+𝜖)𝐴 𝑡)]
Update value network by minimizing mean squared
error loss;
Returnw∗=E[𝜋𝜃(𝑠)];
5 Evaluation and Metrics
This section discusses the evaluation pipeline and the metrics used
for the evaluation.
5.1 Evaluation Pipeline
We evaluated our system using both the base instruction-tuned
model and the full RAG-enhanced + PPO-optimized pipeline. The
evaluation was conducted using a labeled test set.
•For each headline in the test set, the top- 𝑘relevant context
passages were retrieved via the RAG module using source-
weighted cosine similarity.
•A sentiment classification prompt was constructed using the
headline and selected context.
•The fine-tuned LLaMA 3.2 model generated the predicted
sentiment.
•Predictions were made with both market feedback–adjusted
weights and PPO-optimized weights.
5.2 Metrics
We used the following standard classification metrics:
•Accuracy:The percentage of correctly predicted labels.
𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦=𝑇𝑃+𝑇𝑁
𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁(2)
•F1-Score:The harmonic mean of precision and recall, aver-
aged acrosspositive,neutral, andnegativeclasses. Let
–𝐾= number of classes
–𝑛𝑐= support of class𝑐
–𝑁=Í𝐾
𝑐=1𝑛𝑐
–𝐹1𝑐= F1-score for class𝑐
𝑊𝑒𝑖𝑔ℎ𝑡𝑒𝑑 𝐹1=𝐾∑︁
𝑐=1𝑛𝑐
𝑁·𝐹1𝑐 (3)

CODS Dec’25, December 17–20, 2025, Pune, India Chaithra, Kamesh Kadimisetty, and Biju R Mohan
𝐹1𝑐=2·𝑃𝑐·𝑅𝑐
𝑃𝑐+𝑅𝑐
𝑃𝑐=𝑇𝑃𝑐
𝑇𝑃𝑐+𝐹𝑃𝑐, 𝑅𝑐=𝑇𝑃𝑐
𝑇𝑃𝑐+𝐹𝑁𝑐
We also compare performance between:
(1) Instruction-tuned model (no context).
(2) RAG-enhanced predictions with static weights.
(3) RAG with market feedback–updated weights.
(4) RAG with PPO-learned dynamic weighting.
We also incorporated short-term stock price movements by using
price information from the preceding three trading days in addi-
tion to the news data. The price information is converted into a
natural language form. The generated price descriptions were then
appended to the multisource news context for predictions.
6 Results
This section presents the experimental results obtained on a custom-
labeled dataset of NIFTY 50 financial news headlines. The evalua-
tion encompasses various stages of the sentiment analysis pipeline,
including instruction-tuned LLM, retrieval-augmented generation
(RAG) with cosine similarity, market feedback-based source reweight-
ing, and PPO-optimized weighting. Table 4 summarizes the per-
formance across all configurations using accuracy and weighted
F1-score metrics.
Fine-Tuned LLaMA:.The instruction-tuned LLaMA model achieves
an accuracy of 0.5520 and a weighted F1-score of 0.5375, with the
highest recall in the neutral class. The results indicate that the
model struggles to accurately identify positive and negative
sentiments, reflecting the ambiguity in financial headlines without
additional context.
RAG + Without Market Feedback:The RAG-based contextual
retrieval using cosine similarity scores resulted in an accuracy of
0.6094 and an F1-score of 0.5722. This shows the importance of
augmenting inputs with relevant financial context.
RAG + With Market Feedback:The weighted overlap coefficient
(WOC) approach [23] has achieved better performance, with an ac-
curacy of 0.6153 and an F1 score of 0.5746. If the query size is small,
the WOC approach often outperforms the cosine similarity-based
approach. The penalizing or rewarding of each source’s contribu-
tions based on sentiment-return match improves the robustness of
context aggregation.
PPO-Optimized Weights:The PPO agent learns optimal source
weight distributions with similar accuracy and F1-score to the feed-
back model, but with a more refined source selection strategy, as
shown in Figure 3. While the improvement in aggregate metrics
is marginal, PPO generally helps generalize retrieval preferences
over unseen stocks and time periods.
Price Context :With the addition of price context, all approaches
achieved an accuracy of approximately 0.66. However, the weighted
F1 score did not improve. The higher accuracy is largely driven
by the model’s increased prediction of the majority neutral class.
Therefore, the price data, as a context, did not help in predicting
the minority class.FinBERT and RoBERTa :The FinBERT model, although pretrained
on financial text, was unable to predict the class labels correctly,
whereas the RoBERTa model achieved better performance com-
pared to FinBERT. This suggests that incorporating additional con-
text from multiple news sources and prices enhances model perfor-
mance.
6.1 Source Weight Analysis
The distribution of source weights before and after PPO optimiza-
tion is illustrated in Figure 2 and 3 to understand how the system
adapts its information sourcing strategy.
Figure 2: Initial normalized source weights (manually initial-
ized)
Figure 3: Final source weights learned via PPO optimization
Initially, all sources were assigned weights based on perceived
editorial reliability, popularity, and market coverage. For instance,
Business Standard and NDTV Profit were assigned the highest
initial importance of 15.23% and 14.80%, respectively, followed by
Financial Express, The Economic Times, and Mint. These manual as-
signments reflected conventional assumptions about the credibility
of financial news sources.
After training the PPO agent over sentiment-return alignment
feedback, a significant redistribution of weights was observed. The
agent learned to suppress sources that contributed less reliably
to sentiment classifications. Moneycontrol, Zee Business, and The
Economic Times emerged as the most influential sources, while
previously dominant sources such as NDTV Profit and Business
Standard were assigned a weight of zero.

Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches CODS Dec’25, December 17–20, 2025, Pune, India
Table 4: Performance comparison across model variants
Model Variant Accuracy Weighted F1
Fine-Tuned LLaMA 3.2 0.5520 0.5375
RAG + Without Market Feedback 0.6094 0.5722
RAG + With Market Feedback (Cosine Similarity) 0.5999 0.5705
RAG + With Market Feedback (WOC) 0.6153 0.5746
RAG + PPO Optimized Weights 0.6109 0.5733
Price Context
RAG + Without Market Feedback 0.6619 0.5677
RAG + With Market Feedback (Cosine Similarity) 0.6612 0.5668
RAG + With Market Feedback (WOC) 0.6650 0.5674
RAG + PPO Optimized Weights 0.6630 0.5683
Baseline Methods
FinBERT [1] 0.4852 0.5027
RoBERTa 0.5800 0.5551
This weight reallocation indicates a trust profile for each source
based on empirical usefulness in sentiment prediction. The learned
weights implicitly capture patterns such as news article writing
style, topic specification details, and reward sources whose contex-
tual data improved downstream sentiment-return accuracy.
7 Conclusion
In this work, we proposed an adaptive framework for financial senti-
ment analysis by integrating large language models and real-world
stock market feedback. We finetuned the LLaMA 3.2 3B model
using the SentiFin dataset to adapt to the financial domain. To
incorporate context from a diverse set of financial news sources,
the RAG pipeline was proposed. The RAG pipeline has enhanced
prediction accuracy compared to baseline models. In addition to
supplying contextual information, source trustworthiness was in-
corporated through the market feedback mechanism. Incorporating
these dynamically weighted sources improved prediction perfor-
mance compared to using static initial weights. To further gen-
eralize and optimize source weights, the reinforcement learning
algorithm Proximal Policy Optimization (PPO) is used. The perfor-
mance gain was marginal, but it led to a refined source selection
process.
The experiments conducted on NIFTY 50 news data demonstrate
that each component, from instruction tuning to retrieval through
RAG to reward-based adaptation, makes a significant contribution
to the system’s overall performance. The proposed model not only
achieved prediction accuracy but also robustness to news variation
and stronger alignment with stock market behavior. The current
limitation of our approach is its dependence on multi-source news
data. Although price information from the preceding days was
incorporated, it improved accuracy but did not improve the F1
scores. In many cases, news sentiment alone may not capture stock
behavior, especially when price movements are driven by stock
fundamentals. Therefore, incorporating stock fundamental indica-
tors and peer stocks from the same industry may help improve the
alignment of sentiment and the actual market.References
[1] Dilan Araci. 2019. Finbert: financial sentiment analysis with pre-trained lan-
guage models.arXiv preprint arXiv:1908.10063.
[2] Yubo Bi, Hanting Liu, Ruiyang Wang, and Shiyou Li. 2021. Predicting stock
market movements through daily news headlines sentiment analysis: us stock
market. In2021 2nd International Conference on Big Data & Artificial Intelligence
& Software Engineering (ICBASE). IEEE, 642–648.
[3] Tom Brown, Benjamin Mann, Nick Ryder, et al. 2020. Language models are
few-shot learners.Advances in Neural Information Processing Systems, 33, 1877–
1901.
[4] Chaithra Chaithra and Biju R Mohan. 2024. Revealing insights: sentiment
analysis of indian annual reports. In2024 3rd International Conference for
Innovation in Technology (INOCON). IEEE, 1–5.
[5] Samuel WK Chan and Mickey WC Chong. 2017. Sentiment analysis in financial
texts.Decision Support Systems, 94, 53–64.
[6] Pinyu Chen, Zois Boukouvalas, and Roberto Corizzo. 2024. A deep fusion model
for stock market prediction with news headlines and time series data.Neural
Computing and Applications, 36, 34, 21229–21271.
[7] Qinkai Chen. 2021. Stock movement prediction with financial news using
contextualized embedding from bert.arXiv preprint arXiv:2107.08721.
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert:
pre-training of deep bidirectional transformers for language understanding.
arXiv preprint arXiv:1810.04805.
[9] Sorouralsadat Fatemi, Yuheng Hu, and Maryam Mousavi. 2025. A comparative
analysis of instruction fine-tuning large language models for financial text
classification.ACM Trans. Manage. Inf. Syst., 16, 1, Article 6, (Feb. 2025), 30
pages. doi:10.1145/3706119.
[10] Pau Rodriguez Inserte, Mariam Nakhlé, Raheel Qader, Gaetan Caillaut, and
Jingshu Liu. 2023. Large language model adaptation for financial sentiment
analysis. InProceedings of the Sixth Workshop on Financial Technology and
Natural Language Processing, 1–10.
[11] Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen. 2023. Large language
models in finance: a survey. InProceedings of the fourth ACM international
conference on AI in finance, 374–382.
[12] Tim Loughran and Bill McDonald. 2011. When is a liability not a liability?
textual analysis, dictionaries, and 10-ks.The Journal of Finance, 66, 1, 35–65.
[13] Pekka Malo, Ankur Sinha, Pasi Korhonen, Jyrki Wallenius, and Petri Takala.
2014. Good debt or bad debt: detecting semantic orientations in economic texts.
Journal of the Association for Information Science and Technology, 65, 4, 782–796.
[14] GSN Murthy, Shanmukha Rao Allu, Bhargavi Andhavarapu, Mounika Bagadi,
and Mounika Belusonti. 2020. Text based sentiment analysis using lstm.Int. J.
Eng. Res. Tech. Res, 9, 05, 32–41.
[15] Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan. 2002. Thumbs up? senti-
ment classification using machine learning techniques. InProceedings of the
ACL-02 conference on Empirical methods in natural language processing-Volume
10. Association for Computational Linguistics, 79–86.
[16] Sanjna Siboo, Anushka Bhattacharyya, Rashmi Naveen Raj, and SH Ashwin.
2023. An empirical study of ddpg and ppo-based reinforcement learning algo-
rithms for autonomous driving.Ieee Access, 11, 125094–125108.

CODS Dec’25, December 17–20, 2025, Pune, India Chaithra, Kamesh Kadimisetty, and Biju R Mohan
[17] Sahar Sohangir, Dingding Wang, Anna Pomeranets, and Taghi M Khoshgoftaar.
2018. Big data: deep learning for financial sentiment analysis.Journal of Big
Data, 5, 1, 1–25.
[18] Rishabh Srivastava, Venkata Shreyas, and Ajitesh Srivastava. 2021. Sentfin: a
benchmark dataset for sentiment analysis of indian financial news headlines.
https://github.com/pyRis/SEntFiN. (2021).
[19] Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. 2023. Llama: open and
efficient foundation language models.arXiv preprint arXiv:2302.13971.
[20] Renju Rachel Varghese and Biju R Mohan. 2022. The causal effect of financial
news on indian stock market. InTENCON 2022-2022 IEEE Region 10 Conference
(TENCON). IEEE, 1–5.[21] Boyu Zhang, Hongyang Yang, and Xiao-Yang Liu. 2023. Instruct-fingpt: finan-
cial sentiment analysis by instruction tuning of general-purpose large language
models.arXiv preprint arXiv:2306.12659.
[22] Boyu Zhang, Hongyang Yang, Tianyu Zhou, Muhammad Ali Babar, and Xiao-
Yang Liu. 2023. Enhancing financial sentiment analysis via retrieval augmented
large language models. InProceedings of the fourth ACM international conference
on AI in finance, 349–356.
[23] Zijie Zhao and Roy E Welsch. 2024. Aligning llms with human instructions
and stock market feedback in financial sentiment analysis.arXiv preprint
arXiv:2410.14926.
# RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning

**Authors**: Sichun Luo, Jian Xu, Xiaojie Zhang, Linrong Wang, Sicong Liu, Hanxu Hou, Linqi Song

**Published**: 2025-03-26 11:03:34

**PDF URL**: [http://arxiv.org/pdf/2503.20430v1](http://arxiv.org/pdf/2503.20430v1)

## Abstract
Large Language Models (LLMs) have been integrated into recommender systems to
enhance user behavior comprehension. The Retrieval Augmented Generation (RAG)
technique is further incorporated into these systems to retrieve more relevant
items and improve system performance. However, existing RAG methods have two
shortcomings. \textit{(i)} In the \textit{retrieval} stage, they rely primarily
on textual semantics and often fail to incorporate the most relevant items,
thus constraining system effectiveness. \textit{(ii)} In the
\textit{generation} stage, they lack explicit chain-of-thought reasoning,
further limiting their potential.
  In this paper, we propose Representation learning and \textbf{R}easoning
empowered retrieval-\textbf{A}ugmented \textbf{L}arge \textbf{L}anguage model
\textbf{Rec}ommendation (RALLRec+). Specifically, for the retrieval stage, we
prompt LLMs to generate detailed item descriptions and perform joint
representation learning, combining textual and collaborative signals extracted
from the LLM and recommendation models, respectively. To account for the
time-varying nature of user interests, we propose a simple yet effective
reranking method to capture preference dynamics. For the generation phase, we
first evaluate reasoning LLMs on recommendation tasks, uncovering valuable
insights. Then we introduce knowledge-injected prompting and consistency-based
merging approach to integrate reasoning LLMs with general-purpose LLMs,
enhancing overall performance. Extensive experiments on three real world
datasets validate our method's effectiveness.

## Full Text


<!-- PDF content starts -->

RALLR EC+: Retrieval Augmented Large Language
Model Recommendation with Reasoning
Sichun Luo1,2, Jian Xu3, Xiaojie Zhang2, Linrong Wang4, Sicong Liu5, Hanxu Hou1*, Linqi Song2∗
1Dongguan University of Technology2City University of Hong Kong
3Tsinghua University4Chinese University of Hong Kong5Xiamen University
sichunluo2@gmail.com
Abstract
Large Language Models (LLMs) have been integrated into recommender systems
to enhance user behavior comprehension. The Retrieval Augmented Generation
(RAG) technique is further incorporated into these systems to retrieve more relevant
items and improve system performance. However, existing RAG methods have two
shortcomings. (i)In the retrieval stage, they rely primarily on textual semantics
and often fail to incorporate the most relevant items, thus constraining system
effectiveness. (ii)In the generation stage, they lack explicit chain-of-thought
reasoning, further limiting their potential.
In this paper, we propose Representation learning and Reasoning empowered
retrieval- Augmented Large Language model Recommendation ( RALLR EC+).
Specifically, for the retrieval stage, we prompt LLMs to generate detailed item
descriptions and perform joint representation learning, combining textual and
collaborative signals extracted from the LLM and recommendation models, re-
spectively. To account for the time-varying nature of user interests, we propose
a simple yet effective reranking method to capture preference dynamics. For the
generation phase, we first evaluate reasoning LLMs on recommendation tasks,
uncovering valuable insights. Then we introduce knowledge-injected prompt-
ing and consistency-based merging approach to integrate reasoning LLMs with
general-purpose LLMs, enhancing overall performance. Extensive experiments on
three real-world datasets validate our method’s effectiveness. Code is available at
https://github.com/sichunluo/RALLRec_plus .
1 Introduction
Large language models (LLMs) have demonstrated significant potential in many domains due to
impressive world knowledge and reasoning capability [ 1,27,5]. Recently, LLMs have been integrated
into recommendation tasks [ 37,23,25,32]. One promising direction for LLM-based recommen-
dations, referred to as LLMRec, involves directly prompting the LLM to perform recommendation
tasks in a text-based format [ 3,36,24]. However, simply using prompts with recent user history
can be suboptimal, as they may contain irrelevant information that distracts the LLMs from the task
at hand. To address this challenge, ReLLa [ 21] incorporates a retrieval augmentation technique,
which retrieves the most relevant items and includes them in the prompt. This approach aims to
improve the understanding of the user profile and improve the performance of recommendations.
Furthermore, GPT-FedRec [ 35] proposes a hybrid Retrieval Augmented Generation mechanism to
enhance privacy-preserving recommendations by using both an ID retriever and a text retriever.
Despite the advancements, current methods have limitations. ReLLa relies primarily on text embed-
dings for retrieval, which is suboptimal as it overlooks collaborative semantic information from the
∗Corresponding Author
.arXiv:2503.20430v1  [cs.IR]  26 Mar 2025

item side in recommendations. The semantics learned from text are often inadequate as they typically
only include titles and limited contextual information. GPT-FedRec does not incorporate user’s recent
interest, and the ID based retriever and text retrieval are in a separate manner, which may not yield
the best results. The integration of text and collaborative information presents challenges as these
modalities are not inherently aligned.
Another challenge arises in the generation phase. Existing work typically prompts general-purpose
LLMs to generate answers, resulting in models that implicitly map inputs to outputs without explicit
reasoning steps. This approach reduces explainability and interpretability, as the chain-of-thought
reasoning [ 31] is overlooked, thereby constraining the model’s potential. Recently, reasoning
LLMs, such as the OpenAI-o1 [ 27] and DeepSeek-R1 [ 7], have garnered significant attention
for their advanced reasoning capabilities. However, their suitability for recommendation tasks
remains unexplored. Although some prior efforts have attempted to integrate reasoning ability into
recommendation systems [ 4,29], these approaches often rely on specialized workflows that lack
adaptability across domains. In contrast, we propose a training-free method to leverage reasoning
LLMs, enhancing their flexibility and applicability.
In this work, we propose Representation Learning and Reasoning enhanced Retrieval- Augmented
Large Language Models for Recommendation ( RALLR EC+). Specifically, regarding the retrieval
stage, instead of solely relying on abbreviated item titles to extract item representations, we prompt
the LLM to generate detailed descriptions for items utilizing its world knowledge. These generated
descriptions are used to extract improved item representations. This representation is concatenated
with the abbreviated item representation. Subsequently, we obtain collaborative semantics for items
using a recommendation model. This collaborative semantic is aligned with textual semantics through
self-supervised learning to produce the final representation. This enhanced representation is used to
retrieve items, thereby improving Retrieval-Augmented Large Language Model recommendations.
To enhance the generation stage, we first evaluate the reasoning LLM on recommendation tasks,
uncovering intriguing insights. Based on these findings, we propose an effective knowledge-injected
prompting method. By incorporating prior knowledge from recommendation experts, this approach
enables the reasoning LLM to deliver more precise predictions. Additionally, we introduce a
consistency-based merging technique to integrate the reasoning LLM with a general-purpose LLM,
further improving overall performance.
Note that some preliminary findings were reported at the ACM Web Conference 2025 (WWW’25)
[33]. The enhancements made in this extended version of the work are as follows:
•We upgrade the RALLR ECframework to RALLR EC+. While RALLR ECsolely enhances the
retrieval stage through representation learning, RALLR EC+extends this by focusing on the
generation stage. Specifically, we evaluate reasoning LLM on recommendation tasks and propose
simple yet effective strategies to boost model performance.
•We expand our experimental evaluation by incorporating additional models and settings, providing
clearer evidence of RALLR EC+ ’s superiority.
•Lastly, we restructure the paper to better articulate the motivations, objectives, and advancements
of these revisions, offering readers deeper insight into this extended work.
In a nutshell, our contribution is threefold.
•We propose RALLR EC+, which incorporates collaborative information and learns joint representa-
tions to retrieve more relevant items, thereby enhancing the retrieval-augmented large language
model recommendation.
•We evaluate reasoning models on recommendation tasks, uncovering several interesting insights.
Leveraging these findings, we propose a simple yet effective framework to adapt reasoning models
into existing retrieval-augmented LLM recommendation systems.
•We conduct extensive experiments to demonstrate the effectiveness of our proposed RALLR EC+,
further revealing valuable findings.
2

2 Related Work
2.1 LLM for Recommendation
Recent advancements in LLMs have significantly reshaped recommender systems by leveraging their
natural language understanding and generation capabilities for enhanced personalization [ 22,32,
24]. Existing research categorizes LLM-based approaches into two paradigms: discriminative and
generative [ 32]. In the discriminative paradigm, LLMs are used to extract textual features, such as user
and item embeddings, for traditional recommendation algorithms [ 15]. Conversely, the generative
paradigm employs LLMs, such as ChatGPT and Llama, to directly generate recommendations or
explanations, excelling in zero-shot and few-shot scenarios [14].
Despite these advancements, simply using prompts with recent user history can be suboptimal,
as they may contain irrelevant information that distracts the LLMs from the task at hand. Thus,
retrieval-augmented generation technique is further integrated for better performance.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) improves LLM performance by dynamically integrating
relevant external knowledge during inference [ 18,6,2]. This approach has proven effective across
various NLP tasks, such as language modeling [ 9], open-domain question answering [ 13], and
knowledge-intensive dialogue systems [ 16]. In the recommendation domain, RAG has been adopted
to enhance item retrieval and user understanding. Notably, ReLLa [ 21] leverages RAG to augment
LLMs with retrieved textual semantics for improved user comprehension and recommendation
performance. Similarly, GPT-FedRec [ 35] employs RAG within a federated learning framework to
ensure privacy-preserving recommendations.
However, existing methods often fail to learn comprehensive embeddings, leading to suboptimal
retrieval and generation results. Our work addresses this limitation by introducing joint representation
learning and reasoning enhancements.
2.3 LLM Reasoning
The reasoning capabilities of LLMs have evolved significantly. Early work on Chain-of-Thought
(CoT) prompting , introduced by [ 31], demonstrated that prompting LLMs to generate step-by-step
reasoning traces improves performance on complex tasks such as arithmetic and symbolic reasoning.
Building on this, learning-based methods have sought to embed reasoning capabilities directly into
LLMs, reducing reliance on external prompts. STaR [ 34] iteratively fine-tunes models on self-
generated reasoning traces, baking CoT-like behavior into the model itself. Similarly, [ 20] trains
process reward models (PRMs) to evaluate intermediate reasoning steps, outperforming outcome-
based rewards in tasks requiring multi-step logic. OpenAI o1 model [ 27] marks a significant leap in
this direction, leveraging large-scale reinforcement learning (RL) to natively integrate CoT during
inference. Unlike traditional autoregressive LLMs, o1 employs test-time compute to iteratively refine
its reasoning, achieving state-of-the-art results on challenging benchmarks.
However, none of the previous works attempted to apply o1-like reasoning to LLM recommendation,
resulting in a research gap. In this paper, we aim to bridge this gap by evaluating and leveraging the
zero-shot reasoning ability of LLMs.
3 Evaluation
We first conduct an evaluation of reasoning LLM on recommendation tasks and yield some interesting
findings. An example of response generated by general LLM and reasoning LLM is shown in Figure
1.
Dataset. In this paper, we focus on the click-through rate (CTR) prediction [ 21]. We utilize
three widely used public datasets: BookCrossing [ 38], MovieLens [ 10], and Amazon [ 26]. For the
MovieLens dataset, we select the MovieLens-1M subset, and for the Amazon dataset, we focus on
the Movies & TV subset. We apply the 5-core strategy to filter out long-tailed users/items with less
than 5 records. The statistics are shown in Table 1.
3

Figure 1: Example of the response generated by general LLM and reasoning LLM.
Table 1: The dataset statistics.
Dataset #Users #Items #Samples #Fields #Features
BookCrossing 8,723 3,547 227,735 10 14,279
MovieLens 6,040 3,952 970,009 9 15,905
Amazon 14,386 5,000 141,829 6 22,387
Table 2: The comparison of general purpose model and reasoning model. The best results are
highlighted in boldface.
ModelBookCrossing MovieLens Amazon
AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑
Llama-3.1-8B-Instruct 0.5894 0.6839 0.5418 0.5865 0.6853 0.5591 0.7025 0.7305 0.4719
DeepSeek-R1-Distill-Llama-8B 0.6147 0.7065 0.5487 0.5944 0.6850 0.5752 0.6874 0.5392 0.7792
Improvement (%) +4.29 -3.30 +1.27 +1.35 +0.04 +2.88 -2.15 +26.19 +65.12
Comparison between Reasoning LLM and General LLM. To evaluate the impact of reasoning
capabilities in LLMs for recommendation tasks, we compare DeepSeek-R1-Distill-Llama-8B, a
model supervised fine-tuned with distilled CoT reasoning data [ 7], against Llama-3.1-8B-Instruct, a
general-purpose LLM of the same size. Both models are evaluated using a simple retrieval approach,
ensuring a fair comparison. Following the official guidance, we set the temperature to 0.6 for
DeepSeek-R1-Distill-Llama-8B models2. We run the experiment five times and calculate the average.
The results of this comparison are presented in Table 2, where we report the accuracy, log loss, and
AUC scores. We observe that the reasoning LLM always achieves better accuracy compared to the
general LLM. This improvement may be attributed to the CoT reasoning, which enables the model to
follow longer and more accurate reasoning paths, thereby enhancing its decision-making process.
However, we also note that the reasoning LLM exhibits a lower AUC or Log Loss score in some
cases. This could be due to the model being overly confident in its predictions, which might lead to a
higher rate of false positives or false negatives, adversely affecting these metrics.
TAKEAWAY I: Reasoning LLMs generally outperform general LLMs in recommendation tasks.
Analysis of Retrieval in Reasoning LLM. We further investigate the impact of retrieval on the
performance of reasoning LLMs. Specifically, we employ the representation learning enhanced
retrieval mechanism in Sec. 4.2 to augment the model’s input with relevant information. For a fair
comparison, we use the DeepSeek-R1-Distill-Llama-8B model as the base model in all experiments.
The results, presented in Table 3, demonstrate that incorporating retrieval leads to improved model
performance across all three datasets. This enhancement can be attributed to the inclusion of more
relevant items, which facilitates the model’s reasoning process.
TAKEAWAY II: Retrieval augments the model’s performance by providing relevant contextual
information that enhances reasoning capabilities.
2https://huggingface.co/deepseek-ai/DeepSeek-R1
4

Table 3: The comparison of reasoning model w/andw/oretrieval. The best results are highlighted in
boldface.
w/retrievalBookCrossing MovieLens Amazon
AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑
✗ 0.6147 0.7065 0.5487 0.5944 0.6851 0.5752 0.6874 0.5392 0.7792
✓ 0.6274 0.7064 0.5498 0.6028 0.6805 0.5790 0.7014 0.5335 0.7879
Improvement (%) +2.07 +0.01 +0.20 +1.41 +0.67 +0.66 +2.04 +1.06 +1.12
(a) BookCrossing
 (b) MovieLens
 (c) Amazon
Figure 2: Comparison of response length w.r.t. accuracy in reasoning LLMs.
(a) BookCrossing
 (b) MovieLens
 (c) Amazon
Figure 3: Comparison of response consistency w.r.t. accuracy in reasoning LLMs.
Analysis of Response Length in Reasoning LLM. We also explore the relationship between the
length of the reasoning LLM’s response and its performance on recommendation tasks. Prior studies
in mathematical reasoning have shown that longer responses often lead to better performance [ 7].
This raises a natural question: does response length affect LLM recommendation performance?
To analyze this, we conducted the following experiment. We compared response lengths and
accuracies across various problems. We sorted the responses length in ascending order and categorized
the responses into five groups and then calculated the mean accuracy for each group, where accuracy
indicates whether the model’s final answer was correct. The results, detailed in Figure 2 reveal
an intriguing trend: the shortest responses group (Group 1) achieve the highest accuracy, with
performance declining as response length increases. This finding contrasts sharply with observations
in mathematical reasoning tasks. The data consistently indicate that shorter responses outperform
longer ones in this context. We hypothesize that this may reflect “ overthinking ” by the model, where
excessive elaboration introduces redundant steps or errors, undermining the final answer. In contrast
to mathematical reasoning, where problems often require multi-step deductions, the tasks in our study
may favor direct inference or concise solutions.
TAKEAWAY III: Contrary to trends in mathematical reasoning tasks, shorter responses correlate with
improved performance in recommendation tasks.
Analysis of Response Consistency in Reasoning LLM. We further analyze the response consistency
in reasoning LLMs. To this end, we calculate the variance of all responses for each sample, denoted as
var(R), where R={r1, ..., r n}represents the set of generated responses. Additionally, we evaluate
performance within a sliding window approach. We repeat the experiment five times, sorting the
indices of the responses by their variance for each sample. The sorted data is then divided into distinct
windows, and we compute the average accuracy for each window to analyze performance trends.
For the BookCrossing dataset, we set the window size to 500 samples, while for the MovieLens and
Amazon datasets, we use a window size of 1,000 samples. As depicted in Figure 3, our results indicate
5

Figure 4: RALLR EC+with representation learning enhanced retrieval and reasoning enhanced
generation.
a strong correlation between response consistency ( i.e., lower variance) and superior performance
metrics. This suggests that models generating more consistent responses are likely to exhibit enhanced
reliability and effectiveness in reasoning tasks.
TAKEAWAY IV: More consistent responses are generally associated with improved results.
4 Methodology
4.1 Framework Pipeline
The pipeline of the developed framework is illustrated in Figure 4. The RALLR EC+encompasses
both the retrieval and generation processes. In the retrieval process, we first learn a joint representation
of users and items, allowing us to retrieve the most relevant items in semantic space. These items
are then fused with the most recent items by a reranker and incorporated into the prompts. The
constructed prompts can be used solely for inference or to train a more effective model through
instruction tuning (IT).
For the generation phase, the base LLM responds to the prompt for inference, with the option to use
either a standard or customized model. We investigate adapting reasoning LLMs for recommendation
tasks by first evaluating their performance in this context. Subsequently, we propose a technique to
integrate these models into existing systems.
6

Figure 5: Comparison of textual descriptions with fixed template (upper) and automatic generation
(blow).
4.2 Representation Learning enhanced Retrieval
To learn better item embeddings3for reliable retrieval, we propose to integrate both the text embedding
from textual description and collaborative embedding from user-item interaction, as well as the joint
representation through self-supervised training.
4.2.1 Textual Representation Learning
In previous work [ 21], only the fixed text template with basic information such as item title was
utilized to extract textual information. However, we argue that relying solely on the fixed text format
is inadequate, as it may not capture sufficient semantic depth, e.g., two distinct and irrelevant items
may probably have similar names. To enhance this, we take advantage of the LLMs to generate a
more comprehensive and detailed description containing the key attributes of the item ( e.g., Figure 5),
which can be denoted as
ti
desc=LLM(bi|p), (1)
where biis the basic information of the i-th item and the pis the template for prompting the LLMs.
Subsequently, we derive textual embeddings by feeding the text into LLMs and taking the hidden
representation as in [21], represented as
ei
desc=LLM emb(ti
desc). (2)
Since the plain embedding of item title ei
titlecould also be useful, we aim to directly concatenate these
two kinds of embeddings to obtain the final textual representations, denoted by
ei
text= [ei
title∥ei
desc]. (3)
It is worth noting that those textual embeddings are reusable after being extracted and they already
contain affinity information attributed to the rich knowledge of LLMs.
4.2.2 Collaborative Representation Learning
A notable shortcoming of previous LLM-based approaches is their failure to incorporate collaborative
information, which is directed learned from the user-item interaction records and thus can be
complementary to the text embeddings. To this end, we utilize conventional recommendation
models to extract collaborative semantics, denoted as
{ei
colla}n
i=1=RecModel ({(u, i)∈ V}), (4)
where nis the total number of items and Vis the interaction history.
4.2.3 Joint Representation Learning
A straightforward approach for integrating above two representations is to directly concatenate the
textual and collaborative representations. However, since these representations may not be on the
same dimension and scale, this might not be the best choice. Inspired by the success of contrastive
learning in aligning different views in recommendations [ 39], we employ a self-supervised learning
technique to effectively align the textual and collaborative representations. Specifically, we adopt a
3We interchangeably use the representation and embedding to denote the extracted item feature considering
the habits in deep learning and information retrieval domains.
7

simple two-layer MLP as the projector for mapping the original text embedding space into a lower
feature space and use the following self-supervised training objective
Lssl=−E(
log"
f 
eitext,ei
colla
P
v∈Vf 
eitext,ev
colla#
+ log"
f 
ei
colla,eitext
P
v∈Vf 
ei
colla,evtext#)
, (5)
where f 
ei
text,ev
colla
=exp(sim(MLP(ei
text),ev
colla))andsim(·)is the cosine similarity function.
After the joint representation learning, we can get the aligned embedding for each item ias
ei
ssl=MLP(ei
text). (6)
4.2.4 Embedding Mixture
Instead of retrieval using different embeddings separately, we find that integrating those embeddings
before retrieval can present better performance, therefore we directly concat them after magnitude
normalization
eitem= [¯ etext||¯ ecolla||¯ essl], (7)
where ¯ e:=e/∥e∥. With the final item embeddings, we can retrieve the most relevant items to the
target item by simply comparing the dot-production for downstream recommendation tasks.
4.2.5 Prompt Construction
To form a prompt message that LLMs can understand, we use a similar template as in [ 21] by filling
the user profile, listing the relevant behavior history and instructing the model to give a prediction. We
also observed that the pre-trained base LLMs may perform poorly in instruction following. Therefore,
we collect a small amount of training data for instruction tuning, where the prompts are constructed
with similarity-based retrieval and a data augmentation technique is also employed by re-arranging
the retrieved sequence according to the timestamp to reduce the impact of item order.
4.2.6 Reranker
Since we can retrieve the most recent Kitems as well as the most relevant Kitems, relying solely
on one of them may not be the optimal choice. During the inference stage, we further innovatively
design a reranker to merge these two different channels. The reranker can be either learning-based or
rule-based; in this case, we utilize a heuristic rule-based reranker. For each item, we assign a channel
score Scand a position score Spos. We assign the channel score as γand(1−γ)for embedding-based
and time-based channel, respectively. The position score is inversely proportional to the position
in the original sequence, i.e.,{1,1
2β, ...,1
Kβ}. The hyper-parameters γandβare tunable. The total
score for each item is calculated as the production of these two scores
Scorei=Si
c∗Si
pos. (8)
By taking the items with top- Kscores, we can obtain a refined retrieval result to maximize the
prediction performance.
4.3 Reasoning enhanced Generation
Based on the insights in Sec. 3, we propose a knowledge-injected prompting method and a consistency-
based merging technique to adapt a reasoning LLM with a general-purpose LLM, resulting in
improved performance.
4.3.1 Aggregate Reasoning and Tuned LLM Synergy
The experiment result in Sec. 3 reveals a critical dichotomy: while vanilla reasoning LLMs demon-
strate superior structured reasoning capabilities compared to general-purpose LLMs, they under-
perform domain-tuned LLMs with a large gap [ 33]. This presents a fundamental challenge: how
to enhance reasoning LLMs effectively? Supervised fine-tuning (SFT) is a standard approach to
align LLMs with domain-specific tasks [ 22]. Fine-tuning general LLMs requires ground-truth labels.
However, fine-tuning reasoning LLMs is more challenging: user preferences are highly subjective,
making it difficult to craft gold-standard reasoning paths for guidance. Additionally, SFT can be
resource-intensive, posing practical limitations. To solve this, our key innovation stems from two
mechanisms: (i)knowledge-injected prompting, which enriches reasoning LLMs with domain knowl-
edge; and (ii)consistency-based merging, which combines reasoning and general LLMs to aggregate
their strengths, boosting performance effectively.
8

Table 4: The performance of different models in default settings. The best results are highlighted in
boldface.
ModelBookCrossing MovieLens Amazon
AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑
ID-basedDeepFM 0.5480 0.8521 0.5212 0.7184 0.6205 0.6636 0.6419 0.8281 0.7760
xDeepFM 0.5541 0.9088 0.5304 0.7199 0.6210 0.6696 0.6395 0.8055 0.7711
DCN 0.5532 0.9356 0.5189 0.7212 0.6164 0.6681 0.6369 0.7873 0.7744
AutoInt 0.5478 0.9854 0.5246 0.7138 0.6224 0.6613 0.6424 0.7640 0.7543
LLM-basedLlama3.1 0.5894 0.6839 0.5418 0.5865 0.6853 0.5591 0.7025 0.7305 0.4719
ReLLa 0.7125 0.6458 0.6368 0.7524 0.6182 0.6804 0.8401 0.5074 0.8224
Hybrid-Score 0.7096 0.6409 0.6334 0.7646 0.6149 0.6843 0.8405 0.5065 0.8256
OursRALLR EC 0.7162 0.6365 0.6506 0.7658 0.6140 0.6942 0.8416 0.5059 0.8331
RALLR EC+ 0.7175 0.6354 0.6518 0.7663 0.6118 0.6948 0.8412 0.5036 0.8343
4.3.2 Knowledge-injected prompting
We propose a novel prompt augmentation strategy that transfers knowledge from the recommendation
expert ( e.g, tuned LLM MG) to the reasoning LLM ( MR). For input query x, we first extract MG’s
prediction kG=MG(x)through its output layer. These predictions are then projected into natural
language space and injected into the original query x. Then the knowledge-enhanced prompt for
MRbecomes:
paug= [[Task Instruction]| {z }
Base Prompt;x|{z}
Input; kG|{z}
Injected Knowledge] (9)
This allows MRto take advantage of MG’s learned patterns while maintaining its intrinsic reasoning
capabilities. We use the prompt "Another one think the answer might be [Yes/No]" as the injected
knowledge for our task.
4.3.3 Consistency-based merging
Result-level merging across different models offers a simple yet effective approach to integrate their
predictions. Drawing from findings in Sec. 3.3.1, more consistent responses correlate with improved
outcomes, which we interpret as a measure of model confidence. Consequently, we propose assigning
higher weights to more confident predictions during merging, enhancing overall performance.
LetMRgenerate Kreasoning traces using the augmented prompt, producing mean prediction ¯PR
and variance σ2
R. The tuned LLM MGprovides prediction ¯PGwith variance σ2
Gestimation. Our
fusion mechanism combines these through consistency-calibrated weighting:
Pfinal=α·¯PR
σ2
R+ϵ+¯PG
σ2
G+ϵ
α·(σ2
R+ϵ)−1+ (σ2
G+ϵ)−1(10)
where αis a hyperparameter. The ϵterm is a small number that ensures numerical stability.
5 Experiment
In this section, we assess the performance of our framework and aim to answer the following research
questions:
•RQ1: How does our proposed RALLR EC+framework compare with both the conventional
recommendation models and the state-of-the-art LLM-based RAG recommendation methods?
•RQ2: Do the designed components of our model function effectively?
•RQ3: How do different hyper-parameters affect the final recommendation performance?
5.1 Baseline
We compare our approach with baseline methods, which include both ID-based and LLM-based
recommendation systems. For ID-based methods, we select DeepFM [ 8], xDeepFM [ 19], DCN [ 30],
9

Table 5: The performance of different variants of RALLR EC+. We remove different components of
RALLR EC+ to evaluate the contribution of each part to the model. The best results are highlighted
in boldface. KP refer to knowledge inject prompting and CM refer to consistency based merging.
Variants BookCrossing MovieLens Amazon
w/KP w/CM AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑
✗ ✗ 0.7141 0.6392 0.6483 0.7641 0.6160 0.6945 0.8397 0.5071 0.8335
✗ ✓ 0.7158 0.6375 0.6506 0.7639 0.6146 0.6944 0.8404 0.5052 0.8332
✓ ✗ 0.7163 0.6363 0.6506 0.7656 0.6127 0.6940 0.8405 0.5044 0.8337
✓ ✓ 0.7175 0.6354 0.6518 0.7663 0.6118 0.6948 0.8412 0.5036 0.8343
and AutoInt [ 28] as our baseline models. We utilize Llama3.1-8B-Instruct [ 5] as the base model and
LightGCN [ 11] to learn collaborative embeddings in our comparisons. For LLM-based methods, we
consider ReLLa [ 21] and a Hybrid-Score based retrieval as in [ 35].RALLR ECincludes solely on
representation learning enhanced retrieval, while RALLR EC+integrates both representation learning
enhanced retrieval and reasoning enhanced generation. By default, we apply the LoRA method [ 12]
and 8-bit quantization to conduct instruction-tuning as in [ 21] and the maximum length of history is
K= 30 . Similar to ReLLa [ 21], we collect the user history sequence before the latest item and the
ratings to construct the prompting message and ground-truth. For the reranker in our method, we
search the γover{1
2,2
3,4
5}and fix β= 1in the experiments. We set αto 0.1 and ϵto e−3for all
experiments unless specifically specified.
5.2 Main Result
From the numerical results presented in Table 4, several noteworthy observations emerge. Firstly, the
vanilla ID-based methods generally underperform LLM-based methods, demonstrating that LLMs
can better leverage textual and historical information for preference understanding. Secondly, among
LLM-based baselines, ReLLa effectively incorporates a retrieval-augmented approach but relies
predominantly on simple textual semantics for item retrieval. Hybrid-Score, which considers both
ID-based and textual features, also improves over the zero-shot LLM setting (Llama3.1). However,
both ReLLa and Hybrid-Score still fail to fully leverage the rich collaborative semantics and the
alignment between textual and collaborative embeddings. In contrast, RALLR ECand RALLR EC+
consistently achieve the best results across all three datasets, outperforming both ID-based and
LLM-based baselines. The improvements are statistically significant with p-values less than 0.01,
emphasizing the robustness of our approach.
5.3 Ablation Study
To assess the contributions of key components in RALLR EC+, we conduct ablation studies by
systematically removing Knowledge injected Prompting (KP) and Consistency based Merging (CM).
Results are shown in Table 5, and the best performance is highlighted in boldface.
We observe a noticeable decline in AUC and an increase in log loss when removing KP, underscoring
KP’s critical role in enhancing the reasoning LLM’s contextual understanding. It consistently
weakens performance across all datasets without injecting domain knowledge into the reasoning
LLM’s prompts. Additionally, when excluding CM, which aligns predictions from reasoning LLM
and the tuned LLM using consistency metrics, also reduces performance, highlighting CM’s role
in stabilizing the fusion process. The full model, incorporating both KP and CM, consistently
outperforms ablated variants across all metrics and datasets, achieving the highest AUC, lowest log
loss, and best accuracy. Removing both components yields the weakest results, confirming their
complementary strengths: KP boosts reasoning capabilities, while CM ensures effective prediction
alignment.
5.4 Analysis of Retrieval
In this section, we evaluate the retrieval mechanisms of the proposed method, focusing solely on
the representation learning and reranking components without incorporating reasoning LLMs. By
isolating these elements, we assess their standalone effectiveness in enhancing item retrieval for
10

(a) BookCrossing
 (b) MovieLens
 (c) Amazon
Figure 6: Impact of αon reasoning LLMs performance across three datasets.
(a) BookCrossing
 (b) MovieLens
 (c) Amazon
Figure 7: Impact of ϵon reasoning LLMs performance across three datasets.
recommendation tasks. Experiments compare embedding strategies and reranking approaches across
datasets, providing insights into their contributions to overall performance.
5.4.1 Reranking and Retrieval Methods
Figure 8 compares different retrieval and prompt construction approaches on the MovieLens. We
observe the retrieval-then-mix strategy achieves worse performance regarding the AUC metric. Our
reranker, which balances semantic relevance and temporal recency, outperforms both plain recent-
history-based prompts and simple hybrid retrieval strategies. These results emphasize the necessity
of refining retrieved items through post-processing rather than relying solely on a single retrieval
strategy.
5.4.2 Embedding Strategies
In Table 6, we evaluate different embedding schemes for retrieval. Text-based embeddings, derived
from LLM-generated descriptions, yield suboptimal performance compared to ID-based embeddings,
which leverage user-item interaction signals more effectively. Concatenating these two representations
outperforms either alone, achieving better results by capturing both textual and collaborative insights.
Further enhancement is observed when aligning the concatenated embeddings with self-supervised
learning, which refines their semantic coherence and boosts performance across datasets. These results
highlight the progressive improvement from single-modality embeddings to joint representations.
5.5 Hyperparameter Studies
5.5.1 Study of α
The hyperparameter αbalances the contributions of the reasoning LLM and the tuned LLM in
the final prediction Pfinal. We evaluate its impact by varying αover[0.1,0.2,0.3,0.4,0.5], shifting
influence from MG(smaller α) toMR(larger α), while fixing ϵ= 10−3to isolate α’s effect. The
results are shown in Figure 6. We observe distinct trends emerge across different datasets. For
BookCrossing, AUC peaks at a moderate αbefore a slight decline, log loss improves with increasing
α, and accuracy maximizes at a higher α. For Amazon, AUC decreases steadily as αrises, log loss
improves, and accuracy remains stable. In MovieLens, AUC declines gradually, log loss decreases,
and accuracy peaks at an intermediate α. These patterns indicate that smaller αvalues favor MG,
boosting AUC, while larger αenhances log loss by leveraging MR’s reasoning. Accuracy varies by
dataset, reflecting differing sensitivities to reasoning contributions.
11

Recent Retri.-then-Mix Mix-then-Retri. ReRank
Inference-time Retrieval0.740.750.760.770.780.79AUCAUC Comparison
Recent Behavior
RAG-enhanced
Recent Retri.-then-Mix Mix-then-Retri. ReRank
Inference-time Retrieval0.6600.6650.6700.6750.6800.6850.6900.6950.700ACCACC Comparison
Recent Behavior
RAG-enhancedFigure 8: Comparison of fine-tuning and inference settings.
Table 6: The comparison of different embeddings used for historic behavior retrieval during inference.
For fair comparisons, the model is instruction-tuned using the RAG-enhanced training data, while
the inference prompt is constructed based on the embedding similarity without re-ranking. The best
results are highlighted in boldface.
Embedding VariantBookCrossing MovieLens Amazon
AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑AUC↑Log Loss ↓ACC↑
Text-based 0.7034 0.6434 0.6426 0.7583 0.6188 0.6798 0.8408 0.4931 0.8222
ID-based 0.7084 0.6414 0.6357 0.7580 0.6153 0.6867 0.8431 0.4930 0.8244
Concat. w/o SSL 0.7127 0.6411 0.6391 0.7633 0.6153 0.6828 0.8439 0.4925 0.8244
Concat. w/ SSL 0.7141 0.6413 0.6471 0.7653 0.6144 0.6850 0.8442 0.4924 0.8269
Table 7: The statistics for DeepSeek-Llama generated token per response across all datasets.
Dataset BookCrossing MovieLens Amazon
# Tokens/Response 687.2 740.3 661.6
(a) BookCrossing
 (b) MovieLens
 (c) Amazon
Figure 9: Comparison of response latency w.r.t. Llama and DeepSeek-Llama model across three
datasets.
5.5.2 Study of ϵ
The parameter ϵstabilizes prediction weighting in the fusion equation by preventing division-by-zero
errors, based on variances ( σ2
Randσ2
G). We test its effect over [10−2,10−3,10−4,10−5,10−6],
withα= 0.1fixed. Figure 7 presents the results. For BookCrossing, performance optimizes at an
intermediate ϵ, with AUC, log loss, and accuracy peaking, though very small ϵvalues degrade AUC.
For Amazon, AUC and log loss improve up to a mid-range ϵ, with accuracy favoring a larger value.
In MovieLens, AUC and log loss stabilize mid-range, while accuracy peaks slightly higher. These
trends suggest that an intermediate ϵbalances performance and stability across datasets, avoiding
degradation seen at extremes.
5.6 Analysis of Model Efficiency
Reasoning LLMs, despite their advanced capabilities, exhibit slower performance compared to
general LLMs due to the generation of extensive chains of thought. To assess this, we compare the
inference latency of a general LLM (Llama-3.1-8B-Instruct, short for Llama) and a reasoning LLM
12

(DeepSeek-R1-Distill-Llama-8B, short for DeepSeek-Llama) across three datasets using the vLLM
[17] framework for acceleration. The hardware for the platform includes Intel(R) Xeon(R) Gold
6354 CPU @ 3.00GHz and NVIDIA 48G A40 GPU. Latency is measured as the time per response in
seconds. The result is shown in Figure 9. We observe the general LLM demonstrates significantly
higher efficiency compared to the reasoning LLM across all datasets. On average, the reasoning
LLM’s inference latency is over 11 times greater than that of the general LLM. This substantial
difference is primarily due to the reasoning LLM generating much longer responses, with an average
of nearly 700 tokens per response, as detailed in Table 7.
6 Conclusion
We introduce RALLR EC+, a framework that enhances Retrieval-Augmented LLM recommendations
by integrating representation learning and reasoning. It combines textual and collaborative semantics
through self-supervised learning and leverages reasoning LLMs for improved performance. Experi-
ments on three datasets show RALLR EC+outperforms conventional and state-of-the-art methods.
Our findings reveal that reasoning LLMs excel in recommendations, retrieval augmentation boosts
reasoning, and response consistency correlates with better result. Future work could explore fine-
tuning reasoning LLMs to deepen their task-specific capabilities, integrate reinforcement learning to
dynamically refine recommendation policies, and investigate hybrid architectures blending reasoning
and generative strengths.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774 , 2023.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations , 2023.
[3]Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. Tallrec: An
effective and efficient tuning framework to align large language model with recommendation. In
Proceedings of the 17th ACM Conference on Recommender Systems , pages 1007–1014, 2023.
[4]Millennium Bismay, Xiangjue Dong, and James Caverlee. Reasoningrec: Bridging personalized
recommendations and human-interpretable explanations through llm reasoning. arXiv preprint
arXiv:2410.23180 , 2024.
[5]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 , 2024.
[6]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023.
[7]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
[8]Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a
factorization-machine based neural network for ctr prediction. arXiv preprint arXiv:1703.04247 ,
2017.
[9]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval
augmented language model pre-training. In International conference on machine learning ,
pages 3929–3938. PMLR, 2020.
[10] F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm
transactions on interactive intelligent systems , 5(4):1–19, 2015.
[11] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, YongDong Zhang, and Meng Wang. Lightgcn:
Simplifying and powering graph convolution network for recommendation. In Proceedings of
13

the 43rd International ACM SIGIR Conference on Research and Development in Information
Retrieval , page 639–648, 2020.
[12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR ,
1(2):3, 2022.
[13] Gautier Izacard and Edouard Grave. Distilling knowledge from reader to retriever for question
answering. In ICLR 2021-9th International Conference on Learning Representations , 2021.
[14] Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng
Zhang. Genrec: Large language model for generative recommendation. In European Conference
on Information Retrieval , pages 494–502. Springer, 2024.
[15] Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and Chanyoung
Park. Large language models meet collaborative filtering: An efficient all-round llm-based
recommender system. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining , pages 1395–1406, 2024.
[16] Mojtaba Komeili, Kurt Shuster, and Jason Weston. Internet-augmented dialogue generation.
CoRR , abs/2107.07566, 2021.
[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems
Principles , pages 611–626, 2023.
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing
Systems , 33:9459–9474, 2020.
[19] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong
Sun. xdeepfm: Combining explicit and implicit feature interactions for recommender systems.
InProceedings of the 24th ACM SIGKDD international conference on knowledge discovery &
data mining , pages 1754–1763, 2018.
[20] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee,
Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The
Twelfth International Conference on Learning Representations , 2023.
[21] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming
Tang, Yong Yu, and Weinan Zhang. Rella: Retrieval-enhanced large language models for
lifelong sequential behavior comprehension in recommendation. In Proceedings of the ACM on
Web Conference 2024 , pages 3497–3508, 2024.
[22] Sichun Luo, Bowei He, Haohan Zhao, Wei Shao, Yanlin Qi, Yinya Huang, Aojun Zhou, Yuxuan
Yao, Zongpeng Li, Yuanzhang Xiao, et al. Recranker: Instruction tuning large language model
as ranker for top-k recommendation. ACM Transactions on Information Systems , 2024.
[23] Sichun Luo, Wei Shao, Yuxuan Yao, Jian Xu, Mingyang Liu, Qintong Li, Bowei He, Maolin
Wang, Guanzhi Deng, Hanxu Hou, et al. Privacy in llm-based recommendation: Recent
advances and future directions. arXiv preprint arXiv:2406.01363 , 2024.
[24] Sichun Luo, Jiansheng Wang, Aojun Zhou, Li Ma, and Linqi Song. Large language models
augmented rating prediction in recommender system. In ICASSP 2024-2024 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 7960–7964. IEEE,
2024.
[25] Sichun Luo, Yuxuan Yao, Bowei He, Yinya Huang, Aojun Zhou, Xinyi Zhang, Yuanzhang Xiao,
Mingjie Zhan, and Linqi Song. Integrating large language models into recommendation via
mutual augmentation and adaptive aggregation. arXiv preprint arXiv:2401.13870 , 2024.
[26] Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-
labeled reviews and fine-grained aspects. In EMNLP-IJCNLP , 2019.
[27] OpenAI. Learning to reason with llms. OpenAI Blog, 2024.
[28] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian
Tang. Autoint: Automatic feature interaction learning via self-attentive neural networks.
14

InProceedings of the 28th ACM international conference on information and knowledge
management , pages 1161–1170, 2019.
[29] Alicia Y Tsai, Adam Kraft, Long Jin, Chenwei Cai, Anahita Hosseini, Taibai Xu, Zemin Zhang,
Lichan Hong, Ed H Chi, and Xinyang Yi. Leveraging llm reasoning enhances personalized
recommender systems. arXiv preprint arXiv:2408.00802 , 2024.
[30] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click
predictions. In Proceedings of the ADKDD’17 , pages 1–7. 2017.
[31] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[32] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin,
Chen Zhu, Hengshu Zhu, Qi Liu, et al. A survey on large language models for recommendation.
World Wide Web , 27(5):60, 2024.
[33] Jian Xu, Sichun Luo, Xiangyu Chen, Haoming Huang, Hanxu Hou, and Linqi Song. Rallrec:
Improving retrieval augmented large language model recommendation with representation
learning, 2025.
[34] Eric Zelikman, Yuhuai Wu, and Noah D Goodman. Star: Self-taught reasoner. In Proceedings
of the NIPS , volume 22, 2022.
[35] Huimin Zeng, Zhenrui Yue, Qian Jiang, and Dong Wang. Federated recommendation via hybrid
retrieval augmented generation. arXiv preprint arXiv:2403.04256 , 2024.
[36] Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. Is chatgpt
fair for recommendation? evaluating fairness in large language model recommendation. In
Proceedings of the 17th ACM Conference on Recommender Systems , pages 993–999, 2023.
[37] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei
Wang, Xiangyu Zhao, Jiliang Tang, et al. Recommender systems in the era of large language
models (llms). arXiv preprint arXiv:2307.02046 , 2023.
[38] Cai-Nicolas Ziegler, Sean M McNee, Joseph A Konstan, and Georg Lausen. Improving
recommendation lists through topic diversification. In Proceedings of the 14th international
conference on World Wide Web , pages 22–32, 2005.
[39] Ding Zou, Wei Wei, Xian-Ling Mao, Ziyang Wang, Minghui Qiu, Feida Zhu, and Xin Cao.
Multi-level cross-view contrastive learning for knowledge-aware recommender system. In
Proceedings of the 45th international ACM SIGIR conference on research and development in
information retrieval , pages 1358–1368, 2022.
15
# WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web

**Authors**: Zihuai Zhao, Yujuan Ding, Wenqi Fan, Qing Li

**Published**: 2025-11-18 06:35:33

**PDF URL**: [https://arxiv.org/pdf/2511.14182v1](https://arxiv.org/pdf/2511.14182v1)

## Abstract
Recommender systems play a vital role in alleviating information overload and enriching users' online experience. In the era of large language models (LLMs), LLM-based recommender systems have emerged as a prevalent paradigm for advancing personalized recommendations. Recently, retrieval-augmented generation (RAG) has drawn growing interest to facilitate the recommendation capability of LLMs, incorporating useful information retrieved from external knowledge bases. However, as a rich source of up-to-date information, the web remains under-explored by existing RAG-based recommendations. In particular, unique challenges are posed from two perspectives: one is to generate effective queries for web retrieval, considering the inherent knowledge gap between web search and recommendations; another challenge lies in harnessing online websites that contain substantial noisy content. To tackle these limitations, we propose WebRec, a novel web-based RAG framework, which takes advantage of the reasoning capability of LLMs to interpret recommendation tasks into queries of user preferences that cater to web retrieval. Moreover, given noisy web-retrieved information, where relevant pieces of evidence are scattered far apart, an insightful MP-Head is designed to enhance LLM attentions between distant tokens of relevant information via message passing. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed web-based RAG methods in recommendation scenarios.

## Full Text


<!-- PDF content starts -->

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG
from Web
ZIHUAI ZHAO,The Hong Kong Polytechnic University, Hong Kong SAR
YUJUAN DING,The Hong Kong Polytechnic University, Hong Kong SAR
WENQI FAN∗and QING LI,The Hong Kong Polytechnic University, Hong Kong SAR
Recommender systems play a vital role in alleviating information overload and enriching users’ online experience. In the era of
large language models (LLMs), LLM-based recommender systems have emerged as a prevalent paradigm for advancing personalized
recommendations. Recently, retrieval-augmented generation (RAG) has drawn growing interest to facilitate the recommendation
capability of LLMs, incorporating useful information retrieved from external knowledge bases. However, as a rich source of up-to-date
information, the web remains under-explored by existing RAG-based recommendations. In particular, unique challenges are posed
from two perspectives: one is to generate effective queries for web retrieval, considering the inherent knowledge gap between web
search and recommendations; another challenge lies in harnessing online websites that contain substantial noisy content. To tackle
these limitations, we proposeWebRec, a novel web-based RAG framework, which takes advantage of the reasoning capability of LLMs
to interpret recommendation tasks into queries of user preferences that cater to web retrieval. Moreover, given noisy web-retrieved
information, where relevant pieces of evidence are scattered far apart, an insightful MP-Head is designed to enhance LLM attentions
between distant tokens of relevant information via message passing. Extensive experiments have been conducted to demonstrate the
effectiveness of our proposed web-based RAG methods in recommendation scenarios.
CCS Concepts:•Information systems→Recommender systems; Web searching and information discovery.
Additional Key Words and Phrases: Recommender Systems, Large Language Models, Retrieval-augment Generation, Web Search
Engines
ACM Reference Format:
Zihuai Zhao, Yujuan Ding, Wenqi Fan, and Qing Li. 2018. WebRec: Enhancing LLM-based Recommendations with Attention-guided
RAG from Web.J. ACM37, 4, Article 111 (August 2018), 23 pages. https://doi.org/XXXXXXX.XXXXXXX
1 INTRODUCTION
Large language models (LLMs) have achieved remarkable breakthroughs in advancing next-generation recommender
systems, namely LLM-based Recommender Systems (LLM-RS) [ 42]. In particular, LLMs equipped with billion-scale
parameters have exhibited remarkable language understanding and reasoning abilities [ 41]. These capabilities enable
LLMs to effectively capture diverse user preferences by leveraging rich textual side information in recommender
∗Corresponding author.
Authors’ Contact Information: Zihuai Zhao, The Hong Kong Polytechnic University, Hong Kong SAR, zihuai.zhao@connect.polyu.hk; Yujuan Ding, The
Hong Kong Polytechnic University, Hong Kong SAR, dingyujuan385@gmail.com; Wenqi Fan, wenqifan03@gmail.com; Qing Li, qing-prof.li@polyu.edu.hk,
The Hong Kong Polytechnic University, Hong Kong SAR.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2511.14182v1  [cs.IR]  18 Nov 2025

2 Zhao et al.
systems, such as user profiles and item descriptions. However, LLMs often struggle when faced with knowledge-
intensive queries, as their pre-trained knowledge may be incomplete or outdated. This leads to LLMs providing
inaccurate recommendations, such as hallucinated content (e.g., non-existent products like “iPhone 9”) and factual
inaccuracies in real-world scenarios [ 33]. To address these limitations, retrieval-augmented generation (RAG) has
emerged as a promising technique that bridges retrieval systems and LLMs, incorporating coherent and informative
knowledge before generating responses. In other words, LLMs equipped with RAG modules can leverage external
knowledge bases, retrieving reliable and relevant information as augmented inputs of LLMs [6, 23, 32].
Recently, the integration of web search and LLMs via RAG has drawn growing interest from existing studies.
Compared to static knowledge databases that require periodic updates, the web provides access to up-to-date data to
fulfill timely information needs. Notably, the web harnesses search engines that are facilitated with advanced information
retrieval capabilities after decades of development, enabling the web to serve as a rich source of external knowledge
in RAG systems to retrieve online websites based on queries. For example, HtmlRAG [ 30] and WebWalker [ 35] have
proposed web search and result pruning pipelines, enhancing the generation quality of LLMs in QA tasks. Building
upon the recent advances in web-based RAG, new opportunities have been uncovered to facilitate the performance of
LLM-based recommendations. As illustrated in Fig. 1, the web offers a distinct advantage by augmenting the domain-
specific knowledge with large-scale and up-to-date websites, where relevant external knowledge can be retrieved
to facilitate the understanding of user preferences. For example, when LLM-RS lacks pre-trained knowledge or side
information to fulfill accurate recommendations, online websites that aggregate public reviews (e.g., personal blogs and
discussion forums) can provide valuable information to encode item features or understand diverse user preferences for
personalized recommendations.
Despite the great potential of web-based RAG in facilitating LLM-RS, online websites often contain substantial
noisy content [ 4,38], such as long-context conversations and misleading content, making it challenging to extract
high-quality web data that meets the needs of recommendation tasks. However, most existing works on RAG-enhanced
LLM-RS are limited to matching-based methods, where user/item IDs are available for accurate retrieval, such as similar
user profiles [ 15,34] and interacted item reviews [ 26] in recommendation metadata. Due to the significant gap between
web content and recommendations in terms of the data structure and scope, these matching-based RAG potentially
fall short in identifying useful relevant information over noisy web data. For example, when searching for movie
recommendations, the retrieved websites may contain lengthy forum discussions, advertisements, or vague personal
blogs that implicitly reflect user preferences. This leads to existing RAG paradigms hardly capturing the information
needs of LLM-RS, such as understanding nuanced user preferences and distinguishing between conflicting opinions
for accurate recommendations. Therefore, it is imperative to delve into the design of task-specific RAG to facilitate
recommendation performance over noisy web information.
Typically, the RAG pipeline involves two stages:retrieval(e.g., external knowledge bases) andgeneration(e.g., LLM
responses). To facilitate the recommendation capability of LLMs with web-based RAG, unique challenges are posed
from two perspectives as follows:
•Retrieval stage:Effective web retrieval in RAG requires specific and concise queries to retrieve information that
contributes to enhancing the generation process. Such requirements can be easily fulfilled in general QA tasks by
directly using questions as queries, such as“Who is the president of America?”. However, given a recommendation
query of user-item interactions and asking“What will the user buy next?”, less useful results are likely to be retrieved
since the web inherently lacks the capability to address recommendation problems. This leads to query generation in
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 3
Websites (e.g., blog/forum)
Any user
preference?
Buy next?Recommender
Systems
User-item  Interactions Retrieval
LLMGeneration
### Side Information
Task-specific Prompt
:Query
, , ,
### Recommendation T ask
Try new
product!
Buy next?Recommender
Systems
User-item  Interactions Retrieval
LLMGeneration
### Side Information
Web-agumented Prompt
:Query
, , ,
### Recommendation T ask
Online Web
(API Call)query generation
, ,
External
Knowledge### Up-to-date W eb Content
, , ,
 Conventional RAG for Recommendations
 Web-enhanced RAG for Recommendations
Domain-specific
Knowledge
...
 ...
Fig. 1. Illustration of web-enhanced RAG for recommendations. In addition to domain-specific knowledge from recommender systems,
online websites offer a distinct advantage by providing access to up-to-date data as external knowledge. This contributes to fulfilling
the timely information needs in recommendations, such as the latest customer feedback online, to facilitate the understanding of
user preferences.
most existing web-based RAG methods that directly apply task keywords as the retrieval query [ 30,39], becoming
impractical. Therefore, the design of adaptive recommendation queries that cater to web retrieval is required.
•Generation stage:LLM-RS typically relies on textual prompts, such as the description of recommendation tasks, to
generate accurate recommendations by capturing the semantic correlation between prompt tokens in recommendation
tasks. However, online websites often contain substantial noisy content, leading to a large volume of irrelevant
information (e.g., long-context conversations) for LLMs to handle. Such noisy information manifests as long-distance
semantic dependencies, where relevant pieces of evidence are scattered far apart in the textual prompt. This leads
LLM-RS struggling to capture distant correlations between web content and recommendation tasks, which weakens
its ability to reason about user preferences. Consequently, directly training LLM-RS on web-augmented data, which
contains substantial noisy content, can fail to address accurate recommendations.
Related to our web-based RAG task for recommendations, recent studies on LLM-based retrieval have discovered
retrieval headsin LLM attention mechanisms, which are chiefly responsible for retrieving relevant information over
long-context inputs [ 28,37,44]. In particular, Wu et al . [37] have pointed out that attention heads are capable of
redirecting relevant information tokens from input to output in the Needle-in-a-Haystack test [ 14] (i.e., copy-paste
tasks). Another study SnapKV [ 18] has also demonstrated the functionality of attention heads to identify the essential
tokens that carry task-relevant knowledge over long-context information. These advanced findings shed light on an
effective approach to capture long-distance dependencies between noisy web content and recommendation tasks by
utilizing the attention feature. Intuitively, the attention feature of relevant tokens that contain critical information in
recommendation tasks, such as recommendation prompt tokens, can serve as a latent similarity measurement to guide
the retrieval of task-relevant information over noisy web content.
In light of our inspiration for facilitating LLM-RS via attention-guided web retrieval, we proposed a novel framework,
namely WebRec, to address the unique challenges of bridging web retrieval and LLM-based recommendations. In
the retrieval stage, we take advantage of the reasoning capability of LLMs to interpret recommendation tasks (e.g.,
“Recommend the next item”) into specific and informative queries, such as the detailed description of user preferences,
Manuscript submitted to ACM

4 Zhao et al.
that cater to web retrieval. Specifically, we first apply LLMs to generate token sequences, and then sample critical tokens
as high-quality retrieval queries based on carefully designed scoring of the LLM information needs in recommendation
tasks. Notably, our retrieval method can directly work on recommendation prompts, alleviating the necessity of fine-
tuning LLMs or massive prompt engineering for retrieval tasks. In the generation stage, we draw inspiration from
the unique retrieval capability of attention mechanisms and develop a novel MP-Head to enhance LLM attentions for
modeling long-distance dependencies via message passing. In particular, the key insight behind message passing lies in
regarding textual prompts as a token-level graph [ 12], where entities represent the attention features of tokens and
relations are modeled by learnable features. During message passing, the distant correlations between noisy web content
and recommendation tasks can be effectively modeled by therelationbetweenentities, enhancing the attention between
distant tokens as one-hop connectivity. Notably, the MP-Head serves as an additional attention head, which can be
seamlessly integrated into LLM layers in an adapter manner. Therefore, in addition to the semantic correlation captured
by vanilla attention mechanisms, the learnable correlation via message passing helps capture relevant information over
noisy web content to facilitate recommendation performance. In summary, the main contributions of this paper are
organized as follows:
•We propose a novel framework named WebRec that retrieves up-to-date information from online websites to facilitate
LLM-based recommendations via web-based RAG. Notably, our method can directly work on recommendation
prompts by capturing the information needs of LLMs as retrieval queries, without fine-tuning or massive prompt
engineering for retrieval tasks.
•We investigate the significant knowledge gap between web search and recommender systems, where noisy web data
hardly contributes to the recommendation capability of LLMs. Drawing inspiration from the retrieval capability of the
attention mechanism in LLMs, we introduce an attention-guided RAG approach, instead of existing matching-based
methods.
•To address the long-distance dependencies between noisy web information and recommendation tasks, we design a
novel MP-Head that captures learnable correlations via message passing, enhancing LLM attentions between distant
tokens as one-hop connectivity. In particular, the proposed MP-Head can be seamlessly integrated into LLM layers in
an adapter manner.
•Extensive experiments on different real-world recommendation datasets are conducted, demonstrating the effective-
ness of our proposed methods under diverse sources of web information.
2 RELATED WORK
In this section, we elaborate on the paradigm of LLM-based recommender systems and the emergence of retrieval-
augmented generation in the domain of generative models.
2.1 LLM-based Recommender Systems (LLM-RS)
Recommender systems provide personalized suggestions tailored to user preferences, facilitating user experience across
diverse applications, such as e-commerce, job matching, and social media platforms [ 7,8,20]. Recently, Large Language
Models (LLMs) have emerged as a prevalent paradigm for advancing personalized recommendations. To seamlessly
adapt LLMs into task-specific recommendations, many existing works utilize textual prompts to guide the language
understanding of recommendation information, such as user-item interactions, harnessing the rich semantic knowledge
of LLMs [ 42]. Notably, LLMs equipped with billion-scale parameters have demonstrated unprecedented language
understanding and reasoning capabilities [ 21,43], which enables capturing diverse user preferences based on rich
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 5
textual side information in recommender systems, such as user profiles and item descriptions. For instance, Qu et al .
[24] take advantage of the reasoning from LLMs to condition on positive and negative interactions, which contributes
to learning a generative representation of future items, facilitating the understanding of user preferences. As a result,
the remarkable breakthroughs of LLM-RS have drawn growing interest in recent studies.
2.2 Retrieval-augmented Generation (RAG)
Despite the unprecedented capabilities of LLM-RS, LLMs often struggle when faced with knowledge-intensive queries,
as their pre-trained knowledge may be incomplete or outdated. This leads to LLMs providing inaccurate recommenda-
tions, such as hallucinated content (e.g., non-existent products like “iPhone 9”) and factual inaccuracies in real-world
scenarios [ 32,33]. To address such issues, RAG has emerged as a promising technique that bridges retrieval systems and
LLMs, incorporating coherent and informative knowledge before generating responses. For instance, LLMs equipped
with RAG modules can leverage external knowledge bases, retrieving reliable and relevant information as augmented
inputs of LLMs [ 6,23]. To align retrievers with the varying knowledge preferences of LLMs, Dong et al . [5] introduce a
preference knowledge construction pipeline, leveraging multiple query augmentation strategies that allow LLMs to
capture knowledge tailored to their reasoning preferences. Similarly, distinctive reranking strategies are proposed to
rank the set of retrieved documents that accurately meet the demands of LLMs, such as RankCoT [ 36], CFRAG [ 27], and
R2LLMs [ 45], highlighting the effective design of retrieval strategies in RAG systems, which contributes to accurately
capturing the information needs of LLMs.
In the field of recommender systems, initial works have been conducted to facilitate the recommendation capability
of LLMs with RAG techniques, harnessing the rich side information retrieved from recommender systems. For instance,
CoRAL [ 34] aligns LLM reasoning with task-specific user–item interaction knowledge by retrieving collaborative
evidence directly into prompts. RA-Rec [ 15] utilizes comprehensive item reviews retrieved from known metadata to
facilitate closer alignment with users’ potential interests. ROPG [ 26] attempts to optimize retrieval models that supply
a constrained set of personal documents to LLMs, and enhance personalized recommendations. However, retrieving
up-to-date information from the web, which serves as a rich source of open-domain knowledge, remains under-explored
and shows great potential to facilitate LLM-RS.
3 METHODOLOGY
In this section, we propose WebRec, a web-based RAG framework to facilitate LLM-RS, in addressing the distinctive
challenges in the retrieval and generation stages of the RAG pipeline.
•In theretrieval stage, an insightful training-free strategy is designed to bridge the knowledge gap between web
search and recommendation tasks, as illustrated in Figure 2. Notably, we take advantage of the reasoning capability
of LLMs to interpret recommendation tasks into informative queries, such as the description of user preferences, that
cater to web retrieval. In particular, we first apply LLMs to generate token sequences, and then sample critical tokens
as high-quality retrieval queries based on carefully designed scoring of the LLM information needs in recommendation
tasks.
•In thegeneration stage, web-augmented information often contains substantial noisy content, where relevant
pieces of evidence are scattered far apart. This leads LLM-RS struggling to capture distant correlations between
web content and recommendation tasks, which weakens its ability to reason about user preferences. To tackle such
limitations, we propose a novel MP-Head to enhance LLM attentions between distant tokens via message passing. As
illustrated in Figure 3, the long-distance dependencies learned by MP-Head, which serves as an additional attention
Manuscript submitted to ACM

6 Zhao et al.
head in LLM layers, contribute to capturing relevant information over noisy web content to facilitate recommendation
performance.
3.1 Retrieval Stage of WebRec
In the retrieval stage, we aim to search for relevant web information in recommendation tasks via API calls, such
as Google and Bing APIs. However, the web retrieval typically employs keyword matching between search queries
and web data, which inherently lacks the capabilities of recommendation tasks, such as next-item predictions. For
example, the web could retrieve accurate information in QA tasks like“Who is the president of America?”, while hardly
capture the information need in recommendations like“What will the user buy next?”. Therefore, the design of adaptive
recommendation queries that cater to web retrieval is required. As a common solution, existing web-based RAG [ 30,35]
take advantage of the reasoning capability of LLMs to generate appropriate search queries in open-domain QA. Given
the significant knowledge gap between web search and recommendations, massive training data or dedicated prompt
engineering can be required to benchmark the performance of LLM-generated queries. To address these limitations,
we propose to capture high-quality queries based on the information needs during the inference (i.e., training-free) of
LLMs in recommendation tasks.
As illustrated in Figure 2, we take advantage of the reasoning capability of LLMs to interpret recommendation
tasks as semantic queries that cater to web retrieval. Compared to document retrieval from conventional knowledge
bases, the web equipped with search engines alleviates the design of retrieval tools, where relevant web content can be
retrieved based on LLM-generated retrieval queries. Specifically, we quantify the semantic importance of LLM-generated
tokens to sample keywords as high-quality retrieval queries, since they represent the information need of LLMs in
recommendation tasks. Notably, our retrieval method can directly work on recommendation prompts, alleviating the
necessity of fine-tuning LLMs or massive prompt engineering for retrieval tasks. In the following sections, we elaborate
on the implementation of each step in Figure 2.
3.1.1Step 1:Reasoning Prompt.Instead of requiring dedicated prompt engineering or task-specific fine-tuning (i.e.,
web retrieval tasks) for LLMs to directly generate search queries, we aim to capture query keywords from LLM outputs
generated by recommendation prompts. In other words, the reasoning capability of LLMs is leveraged to perform
recommendation tasks, where available prompts can be obtained based on recommendation datasets as follows:
Below is an instruction that describes a task. Please write a response that appropriately completes the request.
[Instruction]
A user has bought ⟨item_ 1⟩,⟨item_ 2⟩, ..., and⟨item_ 10⟩in the previous. Please recommend the next item for
this user to buy from the following item title set: ⟨candidate_ 1⟩,⟨candidate_ 2⟩, ..., and⟨candidate_ 20⟩. The
recommendation is
It is worth noting that, compared to prompting LLMs for accurate query generation, the recommendation prompt
alleviates task-specific requirements on the pre-trained knowledge of LLMs, such as output format and recommendation
accuracy. In particular, LLMs are asked to generate open-domain recommendations based on their reasoning capability,
which usually includes the analysis of user preferences, examples of recommended items, and corresponding reasons.
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 7
Step 2: Scoring of LLM Information Needs Step 1: Reasoning Prompt Step 3: Web Retrieval
(           )
Token Sequences
(i.e., semantic queries)
high low
Score(Attention,E ntropy )LLM
 Outputs Recommendation Task Search API Call
tokens top-k
A user has interacted with items
<item_1>,  <item_2>,  ..., <item_n>. 
Please  recommend the next item. 
Based on the user's previous purchases, which
include  skincare products  and hand creams, I
would recommend  the "Vichy Id&eacute;alia
Radiance Boosting Antioxidant Serum, 1.01 Fl." as
the next item for them to consider. This serum
can help enhance  the user's skin's radiance  and
improve its overall appearance , making it a great
addition to their current skincare routine.
❄
skincare products
recommend  enhance
skin's radiance  ...
Fig. 2. Overview of theretrieval stageof WebRec. In step 1, we take advantage of the reasoning capability of LLMs to interpret
recommendation tasks as semantic queries. In step 2, retrieval queries can be generated to address the information needs of LLMs by
carefully scoring generated tokens, without fine-tuning or massive prompt engineering for retrieval tasks. In step 3, the web retrieval
can be conducted via search APIs.
3.1.2Step 2:Scoring of LLM Information Needs.Following the recommendation prompt, we take advantage of the
remarkable reasoning capability of LLMs to interpret recommendation knowledge as semantic queries that cater to web
retrieval. In particular, the essential information required in recommendations can be captured by the LLM’s information
needs during generation (i.e., textual outputs), which serves as high-quality queries for web retrieval. To quantify the
information needs of LLMs in recommendations, we integrate the semantic importance with the generation confidence
of LLM outputs. These combinations contribute to ranking the keywords of LLM-generated recommendations while
ignoring tokens with high generation confidence, which indicate minimal necessity for retrieving external knowledge.
Building upon the above insights, we first utilize the attention score of generated tokens to calculate their semantic
importance, which measures the influence of each token on subsequent tokens. Consider a sequence of recommendation
outputs generated by LLMs, denoted as T={𝑡 1,𝑡2,...,𝑡𝑛}, the attention scores are calculated by the dot product between
the query vectors Q of each token𝑡 𝑖and the key vectors𝐾of all other tokens𝑡 𝑗as follows:
𝐴𝑖,𝑗=softmax(𝑄𝑖𝐾𝑇
𝑗√𝑑𝑘) ∀1≤𝑖,𝑗≤𝑛,(1)
where the matrix of attention scores 𝐴∈R𝑛∗𝑛and𝑑𝑘is the dimension of the key and query vectors. To address the
semantic importance of each generated token, we measure the attention scores over succeeding tokens that serve as its
influence on recommendation outputs, since LLM generation is determined by the interpretation of preceding context.
Formally, given each token 𝑡𝑖∈T, we quantify its semantic importance by calculating the maximum attention during
the generation of all succeeding tokens𝑡 𝑗>𝑖, thereby the score of semantic importance can be given by:
𝑠attention
𝑖 =max
𝑗>𝑖𝐴𝑗,𝑖.(2)
Subsequently, we aim to combine the semantic importance of tokens with their generation confidence to score the
information need of LLMs in recommendations, such as the keywords of generated tokens. To score generation
confidence at the token level, WebRec quantifies the uncertainty of LLM about its own generated recommendation
outputs. In particular, LLM’s uncertainty can be measured by the entropy of the probability distribution assigned to
each generated token among the full vocabulary, thereby the score of generation confidence is formulated as:
𝑠entropy
𝑖=−∑︁
𝑣∈V𝑃𝑖(𝑣)log𝑃𝑖(𝑣).(3)
where𝑃𝑖is the probability distribution of generating token 𝑡𝑖that𝑡𝑖=arg max𝑣𝑃𝑖(𝑣), and𝑣∈V denotes any token in
the vocabulary of LLMs. Finally, a comprehensive score in addressing the information needs of LLMs in recommendations
Manuscript submitted to ACM

8 Zhao et al.
can be computed by the combination of semantic importance and generation confidence as follows:
𝑠𝑖=𝑠attention
𝑖·𝑠entropy
𝑖.(4)
In other words, generated tokens with high scores indicate a large influence on generating recommendation out-
puts, while LLMs are relatively uncertain about their (i.e., tokens) generations. Therefore, we regard these tokens
as high-quality keywords for retrieving external knowledge, since they represent the information needs of LLMs in
recommendations.
3.1.3Step 3:Web Retrieval.Following the scoring of semantic importance and generation confidence, retrieval
keywords can be ranked from the generated tokens of LLMs. Notably, compared to prompting LLMs to directly generate
retrieval queries, our keyword-based methods can directly work on recommendation prompts. This contributes to
alleviating the necessity of fine-tuning LLMs or massive prompt engineering for retrieval tasks. In particular, we aim
to sample retrieval keywords from recommendation outputs (i.e., generated tokens) as high-quality retrieval queries.
However, the naive ranking of retrieval keywords may lead to diminishing the completeness of relevant information,
such as the entire title of recommended items. To address such limitations, WebRec performs keyword clustering by
pooling the scores of generated tokens, and then selects top-𝑘keywords as the query for web retrieval as follows:
{𝑡𝑖}𝑖∈K←topk{AvgPool(𝑠 1,𝑠2,...,𝑠𝑛)},(5)
whereKdenotes the set of top-k indices based on the pooled scores. Finally, the query for web retrieval can be constructed
by concatenating the selected keywords, which indicate the information needs of LLMs in recommendations. In other
words, the web retrieval aims to provide relevant information from websites as additional prompts to address LLMs’
information needs in recommendations, such as up-to-date public reviews. For example, when LLMs lack pre-trained
knowledge of newly released products, online websites that aggregate public reviews can provide valuable information
on customer opinions and product performance.
3.2 Generation Stage of WebRec
In the generation stage, LLM-RS typically model the semantic correlation in textual prompts to generate task-specific
recommendations. For example, the probability of output tokens (e.g., item titles) can be measured by their semantic
correlation to input tokens of recommendation data. However, these semantic correlations might diminish significantly
over noisy web information due to long-distance dependencies. This leads to LLM-RS hardly capture task-relevant
correlations from the retrieved web information to facilitate recommendation performance. In light of recent findings on
attention heads that are responsible for retrieving information in LLMs [ 37,40], we aim to take advantage of attentions
to model task-relevant correlations between web and recommendations. As shown in Figure 3, we design a novel
MP-Headthat captures long-distance correlations via message passing (MP), which can be seamlessly integrated into
LLM attentions to effectively model task-relevant correlations over noisy web information.
3.2.1Step 4:Transformer Block with MP-Head.Given noisy web information, vanilla attention mechanisms capture
the task-relevant information merely based on semantic correlations, which exhibit long-distance dependencies due
to the large volume of irrelevant web content. In particular, the attention models how much “focus” the information
receives from LLM-RS by calculating the dot product of textual token embeddings (i.e., semantic correlation). Formally,
each token embedding 𝑥𝑖is first mapped into a vector pair (𝑄𝑖,𝐾𝑖,𝑉𝑖)by LLMs, which is analogous to the concept of
query,keyandvaluein retrieval systems [ 11,31]. Subsequently, LLMs utilize a set of attention heads {headℎ}𝑁head
ℎ=1in
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 9
D
### Recommendations
### Websites
Multi-Head
AttentionAdd & NormFeed-Forward
Network
Q V❄
❄❄
MP-
HeadCapture Long-distance Dependencies via Message Passing
distant correlation
... ...[K1,V1][K2,V2]
[K3,V3]
[K4,V4]
[K5,V5]Task Knowledge
Entity
EncodingRelation  Encoding
learnable
task feature
Attention Feature
(i.e., KV caches )
[K1,K2, ...,Kn]
[V1,V2, ...,Vn]🔥Flatten
Concat.
Linear
Projection
Message
AGG
Gating... message passing
MP-Head
OutputStep 4:
Transformer Block
with  MP-Head
Web-augmented Prompt
Add & Normrij
Q1K1TQ2K2T
Q3K3T
attention
outputMP-head
outputtextual
promptlearnable
task featureentity-
relation p air
...
Web Retrieval
(API Call)
,
 ,
 ,...
Augment
I recommend ...
K
...Generator
(LLM-RS)
PromptGenerate
learnable correlation
Fig. 3. Overview of thegeneration stageof WebRec. On the left, we illustrate the pipeline continued from web retrieval thatStep
1-3in Fig. 2. In the right block, the proposed framework of LLM’s transformer block with MP-Head is presented, where MP-Head
takes the original attention features as inputs and models their long-distance dependencies via message passing. Notably, the learned
MP-Head output can be seamlessly integrated into the attention output to facilitate LLM-based recommendations over noisy web
information.
transformer blocks to compute the attention at each token 𝑖, which can be largely affected by the relative distance
between tokens as follows:
headℎ(𝑥𝑖)←Attention(𝑄 𝑖,𝐾𝑖,𝑉𝑖)=𝑖∑︁
𝑡≤𝑖softmax(𝑄𝑖𝐾𝑇
𝑡√𝑑𝑘)𝑉𝑡,(6)
where𝑡denote any preceding token and 𝑑𝑘is the dimensionality ofqueryandkeyvectors. Finally, all heads are
concatenated to indicate the attentiona 𝑖from different representation subspaces, which can be formulated as:
a𝑖=CONCAT(head 1(𝑥𝑖),head 2(𝑥𝑖),...,head ℎ(𝑥𝑖)).(7)
To capture long-distance dependencies in recommendation tasks over web tokens, MP-Head serves as an additional
attention head that models task-relevant correlation between tokens with 1-hop distance via message passing. In
particular, these task correlations modeled by MP-Head are seamlessly integrated with the semantic correlations
captured by vanilla attention heads, which serve as attention features from an augmented representation subspace.
Notably, the proposed MP-Head only introduces one dimension to the original LLM attention (i.e.,a 𝑖) given by:
a𝑀𝑃
𝑖=CONCAT(head 𝑀𝑃(𝑥𝑖),a𝑖),(8)
where head𝑀𝑃(i.e.,a𝑀𝑃
𝑖[0]) denotes the output of MP-Head that contributes to assigning higher attention scores to
task-relevant tokens. As shown in Figure 3, our proposed MP-Head framework consists of three key components:
•Entity Encoding. In the multi-head attention of LLMs, the key-value (KV) embeddings of each token are cached by
attention heads during generation. Therefore, to model the long-distance dependency between tokens, MP-Head
takes advantage of these KV features asentityto represent each token. Compared to vanilla attention heads, which
can be largely affected by token distances in Eq. (6), MP-Head are capable of modeling long-distance dependencies by
the connectivity between each tokenentitywith 1-hop distance, as shown in Fig. 3.
Manuscript submitted to ACM

10 Zhao et al.
•Relation Encoding. By analogy with the semantic correlation between tokens captured by vanilla attention (i.e.,
dot product 𝑄𝐾𝑇ofquery-key), our MP-Head adopts a learnable task feature as thequeryto model task-relevant
correlations between tokens. Specifically, this task feature is obtained by the learnable parameters for fine-tuning
LLM-RS, such as prompt embeddings [ 13,16,19] or trainable adapter parameters [ 9,13,25] used in existing works.
In other words, MP-Head is not responsible for learning the task feature, but leveraging it to capture task-relevant
correlations. Therefore, MP-Head takes the task feature to encode therelationbetween each tokenentity, where the
task-relevant correlation is modeled by their similarities.
•Message Passing. Given the representations ofentityandrelation, we can model the web-augmented prompt
that includes both web and recommendation tokens as a prompt graph. Subsequently, message passing over the
prompt graph is performed to updateentityrepresentations, aggregating each token with other tokens (i.e., one-
hop neighbors) based on theirrelationrepresentations. Thereby, task-relevant tokens over noisy web information
can exhibit strong similarity (i.e., task-relevant correlation) with each other, which contributes to generating high
attention scores between these tokens by MP-Head.
In the following sections, we elaborate on the implementation of these components to explain the generation of
MP-Head outputhead 𝑀𝑃defined in Eq. (8).
3.2.2 Implementation of Entity Encoding.To capture long-distance dependencies between tokens, we draw inspiration
from message passing frameworks to model tokens as entity representations, enabling one-hop connections based on
their relation representations (i.e., task-relevant correlations). In typical message passing, the output embedding 𝐸is
formulated by the concatenation of learned entity representations of each token as follows:
𝐸=CONCAT(𝑒 1,𝑒2,...,𝑒𝑖),(9)
where𝑒𝑖is derived from the entity representation of token embedding 𝑥𝑖. Therefore, the correlation between tokens can
be modeled by the similarity between their entity representations. To seamlessly integrate these learned correlations into
LLMs, MP-Head takes the key-value (KV) embeddings, which are cached by attention heads during generation, as the
entity representation of tokens. In other words, MP-Head aims to encode the correlation between entity representations
into the attention feature of tokens. Formally, given the KV embeddings [𝐾𝑖,𝑉𝑖]of each token, the entity encoding of
MP-Head first maps the concatenated embeddings into the entity representation through linear projection, which can
be given by:
𝑒𝑖=Projentity(CONCAT(K i,Vi)),(10)
following Eq. (9)definitions,𝑒𝑖∈R𝑑𝑘denotes the entity representation of each token through a linear layer Projentity(·).
Subsequently, the entity representation of tokens will be updated by their relation representations via message passing,
as indicated by the message block in Fig. 3.
3.2.3 Implementation of Relation Encoding.In MP-Head, the relation encoding aims to model the task-relevant correla-
tion between tokens into the representation space of attention features. To be specific, we obtain the task feature (i.e.,
recommendation knowledge) through the linear projection from learnable parameters for fine-tuning LLM-RS. For
example, these learnable parameters can be acquired by soft prompt embeddings [ 16,17,19] and the readout of model
parameters [ 9] or parameter-efficient adapters [ 13,25], which are typical fine-tuning paradigms of LLM-RS used in
existing works. It is worth noting that the task feature extracted by these learnable parameters of LLM-RS should not be
attributed to additional MP-Head parameters. In other words, our MP-Head can be integrated with almost any existing
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 11
LLM-RS paradigms, leveraging their learnable parameters as the task feature to encode the relation representation
of tokens. Formally, given the task feature 𝑧∈R𝑑𝑧and each entity representation 𝑒𝑖, we utilized their similarity (i.e.,
task-relevant correlation) to encode the relation representation. Similar to the entity encoding, a linear projection layer
Projrelation(·)is employed to first map the task feature into the representation space of attention features 𝑑𝑘, then the
task-relevant correlation of tokens is formulated by:
𝑐𝑖=sim(Projrelation(𝑧),𝑒𝑖),(11)
where𝑐𝑖∈R1denotes the importance score of each token derived from sim(·) , such as cosine similarity, indicating the
task-relevant correlation (e.g., relevant side information) in recommendation tasks. Subsequently, the matrix of relation
representations can be modeled by broadcasting the task-relevant correlation between tokens, which calculates the
dot product of 𝑐𝑖and𝑐𝑗(𝑖≠𝑗) . However, the large volume of noisy web information may inevitably cause a massive
𝑂(𝑁2)computational cost of the relation matrix, where 𝑁is determined by the number of tokens. Therefore, we
further take advantage of the top- 𝑘ranking (𝑘≪𝑁 ) of task-relevant correlations to prune the relation between entities,
reducing the computation cost of the relation matrix from 𝑂(𝑁2)to𝑂(𝑘2). To be specific, all-zero values are assigned
to relation representations that are outside of the top- 𝑘ranking, which indicates no connectivity between entities with
relatively low task-relevant correlations. This contributes to eliminating noisy information from certain web tokens
while highlighting others. Formally, given each pair of entities 𝑒𝑖,𝑒𝑗(𝑖≠𝑗) , the relation representation calculates the
dot product of their task-relevant correlations as follows:
𝑟𝑖,𝑗= 
max(0,𝑐𝑖·𝑐𝑗𝑇),if𝑖,𝑗∈ T 𝑘
0,otherwise,(12)
whereT𝑘is the list of entity indices based on the top- 𝑘ranking of task-relevant correlations. For example, in a
setting of𝑘=1, the top-1list is given by T𝑘={arg max𝑘∈[1,2,...,𝑁](𝑐𝑘)}. It is worth noting that the matrix of relation
representations that [𝑟1,1,𝑟1,2,...,𝑟𝑁,(𝑁−1),𝑟𝑁,𝑁]shares the same dimension as the matrix of attention scores, while their
computational methods at attention heads are different. Specifically, different from the weighted sum in Eq. (6)generated
by vanilla attention heads, the relation representation enables one-hop message passing between long-distance entities
in MP-Head that alleviates long-distance dependencies in the weighted sum.
3.2.4 Implementation of Message Passing.Following the entity and relation encoding, the message passing updates
entity representations by aggregating neighbor entities based on their relation representations. The key insight of
integrating message passing into MP-Head lies in capturing long-distance dependencies between the entity repre-
sentations of web and recommendation tokens. For example, high-quality tokens over noisy web information can be
modeled as one-hop neighbors. In particular, we capture these high-quality tokens by encoding their task-relevant
correlations into relation representations, which indicates the weight for aggregating neighbor entities. Notably, instead
of updating entity representations with equal weights, the learnable relation representations contribute to eliminating
noisy information from certain web tokens while highlighting others. Thereby, the updated entity representations of
task-relevant tokens via message passing can exhibit strong similarity with each other, guiding MP-Head to capture
their task-relevant correlations by generating high attention scores between these tokens.
Formally, in message passing at transformer layer 𝑙, the entity representation of each token can be updated based on
their message of neighborhood aggregation. In particular, the message aggregates information from all task-relevant
Manuscript submitted to ACM

12 Zhao et al.
entities, which can be defined by:
𝑚(𝑙)
𝑖←∑︁
𝑗∈T𝑘𝑟𝑖,𝑗√︁
deg(𝑖)√︁
deg(𝑗)·Projentity(𝑒(𝑙−1)
𝑗)+𝑏,(13)
where deg(·) denotes the degree of entity for normalizing neighbor entities and 𝑏is a bias vector. Subsequently, the
entity representation is updated by a update function U𝑙, such as ReLU(𝑚(𝑙)
𝑖)[3] and CONCAT(e(l−1)
i,m(l)
i)[10] or
𝑒(𝑙−1)
𝑖+𝑚(𝑙)
𝑖[29] depending on task objectives. However, the update function in conventional message passing is merely
responsible for capturing the similarity between task-relevant tokens, while MP-Head aims to model such similarity as
the attention between these tokens. In other words, the attention score of tokens can be calculated by the similarity
with their messages, indicating the importance of tokens in recommendation tasks. Therefore, to model attentions
based on updated entity representations, MP-Head scores the entity representation of tokens under their representation
space of attention features as follows:
H𝑙(𝑄(𝑙)
𝑖,𝐾(𝑙)
𝑖)=softmax(𝑄(𝑙)
𝑖·𝑒(𝑙)
𝑖[:|𝐾𝑖|]𝑇),(14)
where|𝐾𝑖|is the length of key embeddings and the entity is updated by:
𝑒(𝑙)
𝑖←U𝑙(𝑒(𝑙−1)
𝑖,𝑚(𝑙)
𝑖)=𝑚(𝑙)
𝑖.(15)
Notably, the dot product of 𝑄𝑖·𝑒(𝑙)
𝑖[:|𝐾𝑖|]𝑇in MP-Head is analogous to attention scores in vanilla attention heads that
𝑄𝑖·𝐾𝑇
𝑖, while the term 𝑒(𝑙−1)
𝑖can be ignored in the update function that Eq. (15)due to self-attention mechanisms (i.e.,
the message includes each entity themselves).
3.2.5 Training of LLM-RS with MP-Head.During the training stage of LLM-RS, our proposed MP-Head can be seamlessly
updated by the recommendation loss. However, the capability of MP-Head can be heavily dependent on the learned
task feature of recommendations that 𝑧in Eq. (11), which is not innately embedded in LLM-RS before training. To this
end, we introduce a gating factor 𝑔𝑖to control the integration of MP-Head with vanilla attention heads during the
training of LLM-RS. Formally, the output of MP-Head can be formulated as:
head𝑀𝑃(𝑥𝑖)←𝑔𝑖·Attention MP(𝑄(𝑙)
𝑖,𝐾(𝑙)
𝑖,𝑉(𝑙)
𝑖)=𝑔𝑖·H𝑙(𝑄(𝑙)
𝑖,𝐾(𝑙)
𝑖)𝑒(𝑙)
𝑖[|𝑉𝑖|:],(16)
where|𝑉𝑖|is the length of value embeddings. It is worth noting that a learnable gating factor is critical to the effective
learning of MP-Head, since noisy web information may cause disturbance, especially at the early training stage of
LLM-RS. More specifically, the gating factor alleviates normalizing the concatenation of MP-Head with vanilla attention
heads in Eq. (8), which prevents affecting the pre-trained distribution of attentions. Therefore, we can regard the original
LLM-RS as a frozen model parameterized by Φ0, while the training parameters Θare modeled by the learnable task
feature, such as parameter-efficient prompt tuning, and our proposed MP-Head. Formally, given the input tokens of
web-augmented prompt 𝑥and the label tokens of recommendation output 𝑦(e.g., target item titles), the training loss of
LLM-RS with MP-Head is defined by the negative log-likelihood as follows:
L(𝑥,𝑦)=−|𝑦|∑︁
𝑡=1log(Pr Φ0+ΔΦ(Θ)(𝑦𝑡|𝑥,𝑦<𝑡)).(17)
To be specific, the parameters Θinvolves both MP-Head modules, including the gating factor 𝑔𝑖and two linear projection
layers (i.e., Projentity,Projrelation ), and the learnable task feature 𝑧. It is worth noting that this task feature is obtained
by the learnable parameters for fine-tuning LLM-RS, such as prompt embeddings [ 16,17,19] or trainable adapter
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 13
parameters [ 9,13,25] used in existing works. In other words, MP-Head is not responsible for learning the task feature,
but leveraging it to capture task-relevant correlations between web and recommendation tokens.
4 EXPERIMENT
4.1 Experimental Settings
4.1.1 Datasets.To evaluate the performance of WebRec in real-world recommendation scenarios, we conducted
experiments on four datasets derived from Amazon databases [ 22]:Beauty,Toys,Video Games, andMovies and TV. These
datasets record rich user-item interactions and textual metadata, including user reviews, item titles, and descriptions. To
facilitate an extensive analysis, each benchmark dataset has a varying distribution in terms of the number of users and
items, where the basic statistics are presented in Table 1. Following existing studies [ 16,19], we preprocess each dataset
by removing users and items with fewer than five interactions. To construct sequential recommendation scenarios, we
adopt the leave-one-out strategy and retain the first 10 items in each sequence as the historical interaction, and the last
item as the target item. Note that interaction sequences with fewer than 10 interactions are padded. For both datasets,
we split the data points of user-item interactions into training, validation, and testing sets with a ratio of 8:1:1, which
excludes subsequent interactions from the training dataset and prevents information leakage.
As for retrieval databases in RAG, we consider both local knowledge in recommendations (i.e., Amazon datasets) and
external knowledge from online websites. The basic statistics of retrieval tools are presented in Table 2. In particular,
Amazon datasets are formatted by metadata, where the unique identifier (UID) is available to retrieve user reviews (e.g.,
rating and comment) and item features, including price, description, and sales information. It is worth noting that, to
categorize user–item feedback for retrieval, ratings above 3 are regraded as positive reviews, whereas ratings of 3 or
below, as well as unobserved items, are treated as negative reviews. In terms of web search results, noisy retrieved
information is presented in plain text, such as page title, page snippet (i.e., a short paragraph describing page content),
and page link generated by search APIs.
Table 1. The basic statistics of benchmark datasets, where 𝐿user
denotes the average length of the interaction sequence of each
user.
Datasets #User #Item #Interaction 𝐿user
Beauty 9,930 6,141 63,953 ≈6
Toys 30,831 61,081 282,213 ≈9
Video Games 64,073 33,614 598,509 ≈8
Movies and TV 297,498 59,944 3,409,147 ≈11
# denotes the number of users, items, and interactionsTable 2. The basic statistics of retrieval tools. Tavily and Brave
are two popular web search APIs, where each retrieval result
contains: page title, page HTML, and page snippet (i.e., descrip-
tion paragraph).
Retrieval Tools Format User review Item feature Website
Amazon datasets1metadata ✓(UID)✓(UID)✗
Tavily search API2plain text ✗ ✗ ✓(Noisy)
Brave search API3plain text ✗ ✗ ✓(Noisy)
1https://jmcauley.ucsd.edu/data/amazon/
2https://www.tavily.com/3https://www.brave.com/
4.1.2 Baselines.In the field of LLM-based recommendations, our work pioneers the research on exploring web-based
RAG to facilitate the performance of LLM-RS. Therefore, to evaluate the effectiveness of our proposed WebRec, we
compare the recommendation performance with the following baselines, which can be categorized into three types:
•Non-RAGfor LLM-RS. POD [ 17] transforms discrete prompt representations into continuous embeddings in order to
mitigate the issue of excessive input length in LLMs, building upon the P5 foundational models. GenRec [ 13] leverages
the understanding capabilities of LLMs to generate personalized recommendations, which perform instruction tuning
Manuscript submitted to ACM

14 Zhao et al.
on recommendation datasets based on the LLaMA foundational models. Similarly, TALLRec [ 2] emulates the pattern
of instruction tuning to align LLMs with user preferences in recommendation tasks.
•Recommendation-based RAGfor LLM-RS. Most existing works on RAG-enhanced LLM-RS are limited to matching-
based methods, where user/item IDs are available for accurate retrieval. CoRAL [ 34] aligns LLM reasoning with
task-specific user–item interaction knowledge by retrieving collaborative evidence directly into prompts. RA-Rec [ 15]
utilizes comprehensive item reviews retrieved from known metadata to facilitate closer alignment with users’ potential
interests. ROPG [ 26] attempts to optimize retrieval models that supply a constrained set of personal documents to
LLMs, and enhance personalized recommendations. For these baselines, we employ the recommendation metadata as
the RAG database to retrieve relevant information, including user reviews, product descriptions, and sales information
(i.e., “also buy” interactions).
•Web-based RAGfor LLM-RS. Most existing works on web-based RAG are designed for general LLMs, such as
open-domain QA, which fall short in addressing the knowledge gap of task-specific recommendations. Self-RAG [ 1]
enhances the generation quality of LLMs by combining retrieval with self-reflection, where LLMs adaptively retrieve
relevant passages and refine their own outputs. CRAG [ 39] employs LLMs as retrieval evaluators to inform and
regulate the selection of alternative knowledge retrieval strategies, which selectively filter out irrelevant content in
retrieved information. RAG drafter [ 33] proposes to evaluate multiple draft responses generated in parallel by specialist
LLMs, where each draft is constructed from a distinct subset of retrieved information, producing complementary
perspectives on the available evidence.
4.1.3 Implementation Details.We implement the proposed framework on the basis of HuggingFace and PyTorch,
in which all the experiments are conducted on two NVIDIA H20-96GB GPUs. For each search API of web retrieval,
we preserve the top-10websites as the default setting in RAG paradigms, and more comprehensive results for 𝑘=
1,5,10,15,20are present in ablation studies. To implement our proposed MP-Head, following common settings of
message passing [ 3,32], we compare the performance of one to three head layers, which are inserted into the Transformer
blocks of LLMs, including block layers of0 ,2,15and31. In particular, we employ LLaMA 7𝐵and LLaMA-2 7𝐵as the
backbone model of LLMs, where the layer dimension is set to 4096, and the attention head number is set to 32. In the
training stage, we optimize the WebRec with AdamW while freezing the LLM backbone, where the training epoch is set
to 5 with a batch size of 8.
4.1.4 Evaluation Metrics.To evaluate the effectiveness of our WebRec, we employ two widely used evaluation metrics:
the top-𝑘hit rate (HR@ 𝑘) and the top- 𝑘normalized discounted cumulative gain (NDCG@ 𝑘orNG@𝑘for simplicity),
in which large values indicate higher recommendation performance. We select 𝑘=5,10as the default value for the
main experiments, and more comprehensive results for 𝑘=1,3,5,10,20are present in ablation studies. In pursuit of fair
comparisons between our proposed WebRec and existing LLM-RS paradigms [ 16,19], for sequential recommendation
scenarios, we augment the interaction sequence by randomly selecting 20 non-interacted items as the candidate set.
That is, the test set for each user contains one positive term (i.e., target item) and 19 negative terms (i.e., non-interacted
items). To adapt LLMs to recommendation tasks, these interaction sequences are converted into textual recommendation
prompts. In addition, the retrieved information based on WebRec and other RAG baselines will be concatenated to
recommendation prompts, where an example template is provided as follows:
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 15
Prompt Template: Web-based RAG for Recommendations
[INPUT]
(Instruction)Below is an instruction that describes a task, paired with webpages that provide further context.
Write a response that appropriately completes the request.
(Recommendation Prompt)A user has bought ⟨item_ 1⟩,⟨item_ 2⟩, ..., and⟨item_ 10⟩in the previous. Please
recommend the next item for this user to buy from the following item title set: ⟨candidate_ 1⟩,⟨candidate_ 2⟩,
..., and⟨candidate_20⟩. The recommendation is
(Websites)[1] Title: ⟨page_title⟩ . Content:⟨page_snippet⟩ ; [2] Title:⟨page_title⟩ . Content:⟨page_snippet⟩ ;
... ...; [10] Title:⟨page_title⟩. Content:⟨page_snippet⟩.
[OUTPUT]
(Target Item)⟨candidate_n⟩
4.2 Performance Comparison
As shown in Table 3, our proposed method demonstrates state-of-the-art performance across all four recommendation
domains, demonstrating significant improvements over existing baselines. When integrated with Tavily API, our
method yields relative gains from8 .7%to19.2%in HR metrics and4 .5%to17.2%in NDCG metrics over the strongest
baselines. Similarly, with Brave API, improvements range from5 .2%to23.0%in HR metrics and1 .4%to13.9%in NDCG
metrics. This consistent superiority underscores our framework’s ability (i.e., MP-Head) to effectively capture relevant
recommendation information over noisy web data. Building upon these findings, we elaborate on detailed observations
in the following sections.
4.2.1 Limitations of Non-RAG Methods.Non-RAG methods exhibit noticeable constraints in recommendation quality
due to their reliance on limited parametric knowledge of pre-trained LLMs. For example, though TALLRec that integrates
direct user preference into prompts can achieve competitive results compared to recommendation-based RAG baselines,
the performance degrades significantly by24 .9%from our best performance. Notably, such a performance gap widens
further for POD and GenRec baselines, which merely take user-item interactions. These limitations prove inadequate
for modeling complex user preferences without external knowledge augmentation.
4.2.2 RAG with Noisy Web Information.Compared to recommendation-based RAG, it can be observed that web-based
RAG baselines barely contribute to facilitating recommendation performance, failing to harness noisy web information.
For instance, under Tavily API, RAG-Drafter improves only marginally over recommendation-based CoRAL up to1 .7%
in Amazon Beauty, while degrading performance in other domains up to2 .9%. More critically, Self-RAG and CRAG
consistently underperform recommendation-based RAG across metrics, revealing that naive web retrieval lacks effective
mechanisms to bridge the significant gap between web and recommendations.
4.2.3 Web Information Sources.Our method establishes superior performance consistency across both search APIs,
demonstrating remarkable robustness to information source variations. To be specific, with Tavily API, we achieve
relative gains from13 .9%to19.2%over the strongest baselines. Remarkably, this superiority extends to Brave API, where
our method maintains dominant performance with improvements from5 .2%to23.0%across all evaluation metrics.
In conclusion, our framework excels in leveraging web content for diverse recommendation domains, validating the
ability to capture task-relevant information over noisy web data from different sources.
Manuscript submitted to ACM

16 Zhao et al.
Table 3. Performance comparison between our proposed WebRec and baselines, whereboldfaceand underline indicate the best and
second best scores for each column, respectively. We independently report the results of web-based RAG under different sources of
web data, including Taily and Brave search APIs. Our improvement is compared with the best baseline (i.e., excluding ours).
Beauty Toys Video Games Movies and TV
HR@5 HR@10 NG@5 NG@10 HR@5 HR@10 NG@5 NG@10 HR@5 HR@10 NG@5 NG@10 HR@5 HR@10 NG@5 NG@10
Non-RAG (w/oretrieval, i.e., recommendation prompt only)
POD [17] 0.3166 0.3354 0.2864 0.2928 0.5729 0.5806 0.5448 0.5473 0.4692 0.5428 0.3783 0.4105 0.5001 0.5714 0.4221 0.4532
GenRec [13] 0.3345 0.3475 0.2909 0.3038 0.5559 0.5625 0.5424 0.5446 0.4705 0.5419 0.3713 0.3962 0.5463 0.5717 0.4456 0.4581
TALLRec [2] 0.3533 0.3644 0.2978 0.3109 0.5848 0.6096 0.5465 0.5616 0.4725 0.5440 0.3899 0.4188 0.5602 0.5798 0.4508 0.4592
Recommendation-based RAG (w/retrieval from Amazon metadata1, e.g., item reviews)
RA-Rec [15] 0.3114 0.3727 0.2775 0.3056 0.5511 0.6077 0.5425 0.5642 0.5269 0.5632 0.4064 0.4210 0.5226 0.5898 0.4398 0.4730
ROPG [26] 0.2748 0.3530 0.2677 0.3034 0.5438 0.5926 0.5318 0.5554 0.5286 0.5570 0.4045 0.4172 0.5228 0.5862 0.4315 0.4662
CoRAL [34] 0.3655 0.3878 0.3002 0.3102 0.5997 0.6165 0.5606 0.5690 0.5015 0.5754 0.3931 0.4239 0.5670 0.6019 0.4680 0.4814
Web-based RAG (w/retrieval from Tavily search API2)
Self-RAG [1] 0.3298 0.3319 0.2849 0.2882 0.5328 0.5636 0.5308 0.5505 0.4686 0.5522 0.3632 0.3949 0.5517 0.5698 0.4377 0.4407
CRAG [39] 0.3392 0.3400 0.2919 0.2921 0.5402 0.5453 0.5372 0.5403 0.4736 0.5417 0.3672 0.3908 0.5474 0.5635 0.4403 0.4501
RAG drafter [33] 0.3716 0.4319 0.3016 0.3187 0.5897 0.5988 0.5606 0.5627 0.4931 0.5692 0.3847 0.4161 0.5606 0.5921 0.4612 0.4874
WebRec (Ours) 0.4043 0.4921 0.3153 0.34410.6655 0.7352 0.5871 0.6098 0.5695 0.6565 0.4688 0.49700.5880 0.70560.4950 0.5346
Improvement 8.79% 13.93% 4.54% 7.96% 10.97% 19.25% 4.72% 7.17% 13.55% 14.09% 15.35% 17.24% 3.70% 17.22% 5.76% 9.68%
Web-based RAG (w/retrieval from Brave search API3)
Self-RAG [1] 0.3218 0.3251 0.2891 0.2907 0.5185 0.5223 0.5125 0.5205 0.4799 0.5322 0.3646 0.3862 0.5427 0.5683 0.4439 0.4461
CRAG [39] 0.3347 0.3355 0.2912 0.2953 0.5358 0.5390 0.5318 0.5382 0.4735 0.5365 0.3667 0.3971 0.5470 0.5614 0.4434 0.4527
RAG drafter [33] 0.3783 0.4076 0.3024 0.3195 0.5994 0.6043 0.5566 0.5635 0.5075 0.5834 0.3874 0.4172 0.5619 0.6081 0.4632 0.4852
WebRec (Ours) 0.3981 0.49210.3068 0.3373 0.6862 0.7614 0.5950 0.61920.5520 0.6384 0.4450 0.4830 0.60250.6721 0.51060.5330
Improvement 5.23% 20.07% 1.45% 5.57% 14.42% 23.05% 6.13% 8.82% 4.42% 9.42% 9.49% 13.94% 6.26% 10.50% 9.10% 9.85%
1https://jmcauley.ucsd.edu/data/amazon/2https://www.tavily.com/3https://www.brave.com/
4.3 Ablation Study
To evaluate the effectiveness of the proposed key components, we conducted ablation experiments on the Amazon
BeautyandToysdatasets. In particular, we compare the distinctive influence of our proposed methods across the
retrieval stage and the generation stage of RAG pipelines. Overall, it can be observed that removing any component of
the framework consistently degrades recommendation performance, which demonstrates the effectiveness and necessity
of each module. The detailed analysis of each ablation variant is illustrated in the following sections.
4.3.1 Number of MP-Head Layers.To further understand the role of MP-Head in capturing long-distance dependencies
through message passing (i.e., 𝑛-hop neighbor aggregation), we compare the recommendation performance with one to
three MP-Head layers, as well as the ablation without MP-Head in Table 4. The results demonstrate that the 1-hop setting,
analogous to the dimension of attention metrics, provides baseline improvements up to25 .2%over no aggregation.
Meanwhile, increasing hop count contributes to enhancing performance by capturing deeper task-relevant correlation
between tokens, as shown by 3-hop superiority in Amazon Beauty, surpassing 2-hop and 1-hop settings up to8 .4%and
37.8%, respectively. This suggests multi-hop aggregation captures complex dependencies beyond vanilla attentions.
However, excessive hops can cause degradation due to over-aggregation, evident in Amazon Toys, where the 3-hop
setting consistently underperforms the 2-hop setting across all evaluation metrics. In other words, including distant
neighbors can lead to a reduction in recommendation performance, which is likely due to over-smoothing task-relevant
features.
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 17
Table 4. Ablation on the number of MP-Head layers (i.e., 𝑛-hop
neighbor aggregation). The improvement compares thebestand
worstresults.
Component HR@5 HR@10 NG@5 NG@10
MP-Head(Amazon Beauty, Tavily search API)
- w/ 1-hop 0.4043 0.4921 0.3153 0.3441
- w/ 2-hop 0.4106 0.4796 0.3223 0.3444
- w/ 3-hop 0.4451 0.5203 0.3479 0.3721
- w/o MP-Head 0.3228 0.4231 0.2533 0.2856
Improvement 37.88% 22.97% 37.34% 30.28%
MP-Head(Amazon Toys, Tavily search API)
- w/ 1-hop 0.6655 0.7352 0.5871 0.6098
- w/ 2-hop 0.7037 0.7832 0.6238 0.6496
- w/ 3-hop 0.7026 0.7690 0.6184 0.6397
- w/o MP-Head 0.6176 0.7102 0.5122 0.5423
Improvement 13.94% 10.27% 21.78% 19.78%Table 5. Ablation on the position of MP-Head layer (i.e., Trans-
former blocks of 32-layer LLMs). The improvement compares
thebestandworstresults.
Component HR@5 HR@10 NG@5 NG@10
MP-Head(Amazon Beauty, Tavily search API)
- layer 0 0.3907 0.4727 0.2985 0.3196
- layer 2 0.4043 0.4921 0.3153 0.3441
- layer 15 0.3605 0.4545 0.2780 0.3082
- layer 31 0.3542 0.4514 0.2599 0.2904
Improvement 14.14% 9.01% 20.62% 18.49%
MP-Head(Amazon Toys, Tavily search API)
- layer 0 0.6498 0.7243 0.5625 0.5744
- layer 2 0.6655 0.7352 0.5871 0.6098
- layer 15 0.6597 0.7305 0.5702 0.5933
- layer 31 0.6184 0.7135 0.5161 0.5512
Improvement 7.60% 3.04% 13.75% 10.63%
4.3.2 Position of MP-Head Layer.Following the number of MP-Head layers, we further investigate the influence of
layer positions as shown in Table 5. The results reveal that the position of MP-Head layers can significantly impact
performance, where earlier layers generally yield higher performance across all metrics. In particular, the insertion
at layer 2 consistently achieves the best metrics across both datasets (i.e., improvement up to20 .6%), while layer 31
performs the worst. We attribute this advantage to the task-specific correlations captured by front layers, which allow
subsequent layers to effectively utilize these refined features. Meanwhile, it is worth noting that inserting at the input
layer (i.e., layer 0) underperforms relative to layer 2 by1 .4%to7.1%, despite being a front layer. We hypothesize that
token representations remain weak during early attention layers of LLMs, limiting the ability of MP-Head to generate
accurate task-relevant correlations.
4.3.3 Retrieval Strategy.To evaluate the effectiveness of our retrieval strategy, we additionally compare the performance
between the recommendation query (i.e., recommendation prompt) and the LLM-generated query that directly prompts
LLMs to synthesize a web search query, such as “Please generate the web search query for recommendations.”. As shown
in Fig. 4, web retrieval that directly uses recommendation queries shows marginal (i.e., up to7 .9%in Amazon Beauty)
or even negative (i.e., up to2 .3%in Amazon Toys) gains versus the non-retrieval baseline, indicating a knowledge gap
between web retrieval and recommendation tasks. It can be observed that our keyword-based strategy consistently
outperforms both alternatives. To be specific, compared to the non-retrieval baseline and LLM-generated query, our
method achieves improvements up to26 .6%and7.4%, respectively, across all metrics in Amazon Toys. Notably, these
advantages retain or scale up to46 .7%and3.2%, respectively, in Amazon Beauty. It is worth noting that our retrieval
method can directly work on recommendation prompts, alleviating the necessity of fine-tuning LLMs or massive prompt
engineering for retrieval tasks.
4.3.4 Generation Strategy.Due to the knowledge gap between web and recommendations, the retrieved side information
can be noisy (e.g., large volume of irrelevant data), exhibiting long-distance semantic dependencies in recommendation
tasks. To assess the impact of our proposed MP-Head on capturing task-relevant correlation over long-distance
dependencies, we additionally compare the performance between vanilla RAG, where website content is naively
Manuscript submitted to ACM

18 Zhao et al.
HR@5HR@10NDCG@5NDCG@100.300.450.600.75Amazon Beauty
HR@5HR@10NDCG@5NDCG@10Amazon ToysNon-retrieval Baseline
Recommendation QueryLLM-generated Query
Keyword-based Query (Ours)
Fig. 4. Ablation on Different Retrieval Strategies.
HR@5HR@10NDCG@5NDCG@100.300.450.600.75Amazon Beauty
HR@5HR@10NDCG@5NDCG@10Amazon ToysVanilla RAG Baseline
RAG with LIN-Head (Ours)RAG with MP-Head (Ours) Fig. 5. Ablation on Different Generation Strategies.
HR@5HR@10NDCG@5NDCG@100.30.40.5Query: LLaMA-Chat7B, Generate: LLaMA7B
Query: LLaMA-Chat7B, Generate: LLaMA-27B
Query: Mistral-Instruct7B, Generate: LLaMA7B
Query: Mistral-Instruct7B, Generate: LLaMA-27B
(a) LLM Backbones
0.40.50.6
Hit Rate@10
1 5 10 15 200.20.30.4
NDCG@10
NVanilla RAG WebRec (Ours) (b) Top-N Retrieval
0.20.40.6
Hit Rate@K
1 3 5 10 200.20.30.4
NDCG@K
KVanilla RAG WebRec (Ours) (c) Top-K Recommendations
Fig. 6. Ablation on hyper-parameters of WebRec. We report the results under Amazon Beauty.
concatenated as plain text, and our proposed variants, including LIN-head and MP-Head. In particular, LIN-head
replaces the message passing structure by linear layers with equivalent dimensions, which is analogous to adding an
attention head. As illustrated in Fig. 5, our method significantly outperforms the vanilla RAG baseline, demonstrating
superior capability in harnessing noisy web information for recommendations. On the Beauty dataset, MP-head achieves
an absolute gain from16 .3%to25.2%in HR metrics and from20 .4%to24.4%in NDCG metrics. Similar improvements are
observed for Toys, where MP-head exceeds vanilla RAG up to14 .6%across all metrics. Notably, compared to LIN-head,
our proposed message passing structure exceeds linear ones up to13 .0%in both HR and NDCG metrics over Amazon
Beauty. The above gains extend to Amazon Toys, with NDCG metrics showing the largest margin of8 .1%. These
findings underscore the effectiveness of our proposed MP-Head in capturing task-relevant correlations between tokens
by modeling their long-distance dependencies, compared to mere semantic correlations computed by linear attention
heads.
4.3.5 Hyper-parameter Analysis.In this section, we delve into the impact of three main hyper-parameters of our
proposed WebRec, as shown in Fig. 6, including LLM backbones, top-N retrieval, and top-K recommendations. Our
results reveal that the retrieval backbone exerts a dominant influence on RAG system performance compared to the
generation backbone, with Mistral-Instruct achieving substantial gains over LLaMA-Chat up to6 .4%and5.7%in HR and
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 19
Table 6. Example of recommendation tasks and LLM-generated reasoning on user preferences.
Recommendation Tasks (Prompt):
A user has bought“Crabtree amp; Evelyn - Gardener’s Ultra-Moisturising Hand Therapy Pump - 250g/8.8 OZ”,
“Crabtree amp; Evelyn Gardeners Ultra-Moisturising Hand Cream Therapy - 3.5 oz”, “Mustela Gentle Shampoo,
Tear Free Baby Shampoo with Natural Avocado Perseose, Gently Cleanses and Detangles Kids’ Hair, Available in
6.76 and 16.9 fl. oz”, and “Noodle amp; Boo Super Soft Moisturizing Lotion for Daily Baby Care, Sensitive Skin and
Hypoallergenic”in the previous. Recommend one next item for this user to buy next from the following
item title set:“GLYTONE KP Kit”, “VINCENT LONGO Thinstick Lipstick”, “Bioderma Atoderm Cream for Very
Dry or Sensitive Skin”, “La Roche-Posay Anthelios Ultra Light Sunscreen Fluid SPF 60, 1.7 Fl. Oz.”, “Bioelements
SPF 50 Face Screen, 2.3-Ounce”, “Eau Thermale Avegrave;ne Antirougeurs Day Redness Relief Soothing SPF 25
Cream, 1.35 fl. oz.”, “ AHAVA Dead Mineral Botanic Velvet Cream Body Washes”, “blinc Mascara”, “Korres Body
Butter, 7.95-Ounce Tube”, “Zwilling J.A. Henckels Nose and Ear Hair Trimmer”, “ GLYTONE Exfoliating Body Lotion,
8.4 fl. oz.”, “MERKUR Classic 2-Piece Double Edge Saftey Razor”, “Eau Thermale Avegrave;ne Extremely Gentle
Cleanser Lotion, 6.76 fl. oz.”, “Glytone Daily Body Lotion Broad Spectrum Sunscreen SPF 15, 12 Ounce”, “Mustela
Stelatopia Cleansing Cream, Baby Body Wash for Extremely Dry to Eczema-Prone Skin, Fragrance-Free, 6.76 Fl. Oz”,
“HOT TOOLS Spring Curling Iron, Black/White”, “St. Tropez Self Tan Bronzing Mousse 4 Fl Oz”, “Vichy LiftActiv
Retinol HA Eyes Anti-Aging Retinol Eye Cream with Hyaluronic Acid, 0.5 Fl. Oz.”, “Philip B. Light-Weight Deep
Conditioning Creme Rinse”, and “MOROCCANOIL Intense Hydrating Mask Fragrance Originale, 16.9 Fl. Oz.”.The
recommendation is
Reasoning on User Preferences (LLM generation):
The user has bought hand therapy products and baby care products. Based on the user’s previous purchases, it
appears that they are interested in moisturizing and nourishing products for both their own hands and their baby’s
skin. Therefore, a recommendation for a body lotion or cream that is specifically formulated for dry or sensitive skin
could be a good fit.
One option from the given list that matches this description is the “Bioderma Atoderm Cream for Very Dry or
Sensitive Skin”. This product is known for its gentle and soothing formula, making it suitable for both adults and
babies with dry or sensitive skin. It is also a cream, which is similar to the hand therapy and baby lotions the user
has already purchased.
Another option could be the “Mustela Stelatopia Cleansing Cream, Baby Body Wash for Extremely Dry to Eczema-
Prone Skin, Fragrance-Free, 6.76 Fl. Oz.” This product is a baby body wash, but it is specifically formulated for very
dry or eczema-prone skin, which aligns with the user’s previous purchases of moisturizing products for their baby. It
is also fragrance-free, which may be appealing to users who prefer unscented products.
Therefore, based on the user’s previous purchases and the product features, I would recommend either the "Bioderma
Atoderm Cream for Very Dry or Sensitive Skin" or the "Mustela Stelatopia Cleansing Cream, Baby Body Wash
for Extremely Dry to Eczema-Prone Skin, Fragrance-Free, 6.76 Fl. Oz." as the next item for the user to consider
purchasing.
NDCG metrics, respectively, mainly attributing to a superior instruction-following capacity for recommendation tasks.
These findings demonstrate the critical role of RAG in facilitating the generation performance of LLMs. Meanwhile,
upgrading the generation backbone to LLaMA-2 consistently enhances results over LLaMA, which is consistent with
the performance evolution of LLM families.
As for top-N retrieval in Fig. 6(b), the results demonstrate a clear performance advantage of our WebRec framework
over the vanilla RAG baseline, where website content is naively concatenated as plain text, in top-N website retrieval.
In particular, the performance divergence widens notably at higher N values. These findings underscore that naively
integrating web content, without effective purification or structuring mechanisms, severely constrains recommendation
quality. Notably, our method consistently yields performance improvements as N increases, which demonstrates
superior capability in distinguishing relevant information from noisy web data for recommendations. As illustrated in
Fig. 6(c), we further validate the pattern of performance with respect to top-K recommendations. In particular, WebRec
Manuscript submitted to ACM

20 Zhao et al.
Table 7. Example of generated retrieval queries and corresponding retrieval results from web search APIs.
Target Item Title:
Bioderma Atoderm Cream for Very Dry or Sensitive Skin
Retrieval Query (Our WebRec):
recommendation product nourishing cream designed dry soothing gentle fragrance-free formula
Retrieval Results (Web search API):
1. Title: “The Rich Moisturiser Fragrance Free - Skin Rocks”. Content: “Intensely nourishing, fragrance free rich
moisturiser for skin that needs extra care. Addresses dry ness at its core, leaving skin soft,”.
2. Title: “Min<u>00e9ral 89 100H Rich Cream | Vichy Laboratoires”. Content: “Min<u>00e9ral 89 100H Moisture
Boosting Rich Cream Fragrance- Free provides up to 100 hours of lasting hydration for healthy-looking skin.”.
3. Title: “The Rich Cream | Augustinus Bader US”. Content: “Richly nourishing and loaded with essential fatty acids,
including linoleic acid. Effectively reduces moisture loss to soothe, strengthen and enhance skin’s”.
4. Title: “Our Calming Cream - HealthyBaby”. Content: “This calming cream is for dry,sensitive , eczema-prone
skin , made with colloidal oatmeal, shea butter, aloe, coconut oil, pre/probiotics, and is fragrance-free.”.
5. Title: “Revolution Skincare Nourish Boost Rich Nourishing Cream , Vegan ...”. Content: “RICH NOURISHING
FORMULA: Deep moisturizing cream designed to provide intense hydration and nourishment to the skin <u>00b7
VEGAN FRIENDLY: Formulated with plant-based”.
6. Title: “Alcohol and Fragrance-free Moisturizers for Sensitive Skin ”. Content: “Top picks of Alcohol and Fragrance-
free Moisturizers for Sensitive Skin <u>00b7 COSRX Ultimate Nourishing Rice Overnight Mask <u>00b7 Cerave
Moisturizing Cream <u>00b7 Etude House”.
7. Title: “8 Best Fragrance-Free Body Lotions for Softer, Less Irritated Skin”. Content: “Best Overall: La Roche-Posay
Lipikar AP+ Triple Repair Moisturizing Cream <u>00b7 Lipikar AP+ Triple Repair Moisturizing Cream <u>00b7
Caption Options.”.
8. Title: “FAVORITE FRAGRANCE FREE PRODUCTS. ( SENSITIVE SKIN )”. Content: “AVEENO Skin Relief Body
Wash - <u>00b7 FIRST AID BEAUTY KP BUMP ERASER BODY SCRUB - <u>00b7 NECESSAIRE THE BODY SERUM
- <u>00b7 NECESSAIRE THE BODY LOTION - <u>00b7 EUCERIN”.
9. Title: “Calm + Restore<u>00ae Redness Relief Cream forSensitive Skin | Aveeno<u>00ae”. Content: “This
redness relief face moisturizer formulated with Vitamin B5 ceramide features a calming feverfew nourishing oat
formula to calm dry skin.”.
10. Title: “(2 pack) Baby Jergens Soothing Sensitive Skin Cream , Newborn ...”. Content: “This baby cream for
dry skin softens and soothes your baby’s dry skin, leaving no sticky feel, for 24 hours of hydration. Pediatrician
and dermatologist tested,”
demonstrates substantial performance gains over Vanilla RAG, in which such advantages persist at larger K values.
These results validate the effectiveness of our proposed WebRec for facilitating LLM-based recommendations with
web-based RAG, while maintaining scalable ranking quality in recommendation scenarios.
4.4 Example of Retrieval Results
In this part, we present an example of the generated retrieval query based on our proposed WebRec, along with the
retrieval results from web search APIs, as shown in Table 6 and Table 7, respectively. For reference, we take advantage
of the reasoning capability of LLMs to interpret recommendation tasks (e.g.,“The recommendation is”) into specific and
informative queries, such as the detailed description of user preferences, that cater to web retrieval. Specifically, we
apply LLMs to generate token sequences, and then sample critical tokens as high-quality retrieval queries based on
carefully designed scoring of the LLM information needs in recommendation tasks. In other words, the retrieval queries
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 21
are sampled from LLM-generated reasoning on user preferences, which are used to retrieve online websites via API
calls. Notably, we highlight the common keywords between the target item title and the retrieval results from websites,
demonstrating the effectiveness of WebRec to retrieve relevant information in recommendation tasks. It can be observed
that the retrieved web content is highly related to the target item, which contributes to the accurate recommendation
of the target item over other candidate items.
5 CONCLUSION
In this paper, we introduce WebRec, a novel web-based RAG to facilitate LLM-based recommendations by harnessing up-
to-date and high-quality information retrieved from online websites. Specifically, we tackle the unique challenges due to
the significant knowledge gap between web content and recommendation tasks, where noisy web data hardly contributes
to the recommendation capability of LLMs. To bridge this gap, we first take advantage of the reasoning capability of LLMs
to interpret recommendation tasks into semantic information on user preferences that cater to web retrieval, sampling
LLM-generated tokens as high-quality retrieval queries. Subsequently, given noisy web-augmented information, where
relevant pieces of evidence are scattered far apart, we introduce an attention-guided RAG approach instead of existing
matching-based methods. Drawing inspiration from the retrieval capability of the attention mechanism in LLMs, we
design a novel MP-Head that captures learnable correlations via message passing, enhancing LLM attentions between
distant tokens as one-hop connectivity. Therefore, the long-distance dependencies learned by MP-Head, which serves
as an additional attention head in LLM layers, contribute to capturing relevant information over noisy web content to
facilitate recommendation performance. Extensive experiments on different real-world recommendation datasets are
conducted to demonstrate the effectiveness of our proposed methods under diverse sources of web information.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through
self-reflection.ICLR(2024).
[2]Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align
large language model with recommendation. InProceedings of the 17th ACM conference on recommender systems. 1007–1014.
[3]Yuanning Cui, Zequn Sun, and Wei Hu. 2024. A prompt-based knowledge graph foundation model for universal in-context reasoning.Advances in
Neural Information Processing Systems37 (2024), 7095–7124.
[4]Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for
the web.Advances in Neural Information Processing Systems36 (2023), 28091–28114.
[5]Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and Zhicheng Dou. 2025. Understand what LLM needs: Dual preference
alignment for retrieval-augmented generation. InProceedings of the ACM on Web Conference 2025. 4206–4225.
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms:
Towards retrieval-augmented large language models. InProceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining.
6491–6501.
[7]Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 2019. Graph neural networks for social recommendation. InThe
world wide web conference. 417–426.
[8]Wenqi Fan, Yao Ma, Qing Li, Jianping Wang, Guoyong Cai, Jiliang Tang, and Dawei Yin. 2020. A graph neural network framework for social
recommendations.IEEE Transactions on Knowledge and Data Engineering34, 5 (2020), 2033–2047.
[9]Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain,
personalized prompt & predict paradigm (p5). InProceedings of the 16th ACM conference on recommender systems. 299–315.
[10] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017. Neural message passing for quantum chemistry. In
International conference on machine learning. Pmlr, 1263–1272.
[11] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. 2024. Attention score is not all you need for token importance indicator in kv cache reduction:
Value also matters.arXiv preprint arXiv:2406.12335(2024).
[12] Qingbao Huang, Jielong Wei, Yi Cai, Changmeng Zheng, Junying Chen, Ho-fung Leung, and Qing Li. 2020. Aligned dual channel graph convolutional
network for visual question answering. InProceedings of the 58th annual meeting of the association for computational linguistics. 7166–7176.
Manuscript submitted to ACM

22 Zhao et al.
[13] Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng Zhang. 2024. Genrec: Large language model for generative
recommendation. InEuropean Conference on Information Retrieval. Springer, 494–502.
[14] Greg Kamradt. 2023. Needle in a haystack-pressure testing llms.GitHub repository(2023), 28.
[15] Sara Kemper, Justin Cui, Kai Dicarlantonio, Kathy Lin, Danjie Tang, Anton Korikov, and Scott Sanner. 2024. Retrieval-augmented conversational
recommendation with prompt-based semi-structured natural language state tracking. InProceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval. 2786–2790.
[16] Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and Chanyoung Park. 2024. Large language models meet collaborative
filtering: An efficient all-round llm-based recommender system. InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining. 1395–1406.
[17] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Prompt distillation for efficient llm-based recommendation. InProceedings of the 32nd ACM international
conference on information and knowledge management. 1348–1357.
[18] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv:
Llm knows what you are looking for before generation.Advances in Neural Information Processing Systems37 (2024), 22947–22970.
[19] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan He. 2024. Llara: Large language-recommendation
assistant. InProceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1785–1795.
[20] Chengyi Liu, Jiahao Zhang, Shijie Wang, Wenqi Fan, and Qing Li. 2025. Score-based generative diffusion models for social recommendations.IEEE
Transactions on Knowledge and Data Engineering(2025).
[21] Xinqi Lyu, Yihao Liu, Yanjie Li, and Bin Xiao. 2025. PLA: Prompt Learning Attack against Text-to-Image Generative Models.arXiv preprint
arXiv:2508.03696(2025).
[22] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. InProceedings
of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing
(EMNLP-IJCNLP). 188–197.
[23] Liangbo Ning, Wenqi Fan, and Qing Li. 2025. Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation.arXiv preprint
arXiv:2504.02458(2025).
[24] Haohao Qu, Wenqi Fan, and Shanru Lin. 2025. Generative Recommendation with Continuous-Token Diffusion.arXiv preprint arXiv:2504.12007
(2025).
[25] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. 2024. Tokenrec: learning to tokenize id for llm-based generative recommendation.arXiv preprint
arXiv:2406.10450(2024).
[26] Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024. Optimization methods for personalizing large language models through retrieval
augmentation. InProceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 752–762.
[27] Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, and Han Li. 2025. Retrieval augmented generation with collaborative filtering
for personalized text generation. InProceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval.
1294–1304.
[28] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN: dynamic retrieval augmented generation based on the information
needs of large language models.arXiv preprint arXiv:2403.10081(2024).
[29] Junshu Sun, Chenxue Yang, Xiangyang Ji, Qingming Huang, and Shuhui Wang. 2024. Towards dynamic message passing on graphs.Advances in
Neural Information Processing Systems37 (2024), 80936–80964.
[30] Jiejun Tan, Zhicheng Dou, Wen Wang, Mang Wang, Weipeng Chen, and Ji-Rong Wen. 2025. Htmlrag: Html is better than plain text for modeling
retrieved knowledge in rag systems. InProceedings of the ACM on Web Conference 2025. 1733–1746.
[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is
all you need.Advances in neural information processing systems30 (2017).
[32] Shijie Wang, Wenqi Fan, Yue Feng, Shanru Lin, Xinyu Ma, Shuaiqiang Wang, and Dawei Yin. 2025. Knowledge graph retrieval-augmented generation
for llm-based recommendation.arXiv preprint arXiv:2501.02226(2025).
[33] Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo
Shang, et al. 2024. Speculative rag: Enhancing retrieval augmented generation through drafting.arXiv preprint arXiv:2407.08223(2024).
[34] Junda Wu, Cheng-Chun Chang, Tong Yu, Zhankui He, Jianing Wang, Yupeng Hou, and Julian McAuley. 2024. Coral: collaborative retrieval-augmented
large language models improve long-tail recommendation. InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining. 3391–3401.
[35] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al .2025.
Webwalker: Benchmarking llms in web traversal.arXiv preprint arXiv:2501.07572(2025).
[36] Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi Yu, Zheni Zeng, Yu Gu, and Ge Yu. 2025. RankCoT: Refining Knowledge for Retrieval-
Augmented Generation through Ranking Chain-of-Thoughts.arXiv preprint arXiv:2502.17888(2025).
[37] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2024. Retrieval head mechanistically explains long-context factuality.arXiv
preprint arXiv:2404.15574(2024).
[38] Weijian Xie, Xuefeng Liang, Yuhui Liu, Kaihua Ni, Hong Cheng, and Zetian Hu. 2024. Weknow-rag: An adaptive approach for retrieval-augmented
generation integrating web search and knowledge graphs.arXiv preprint arXiv:2408.07611(2024).
Manuscript submitted to ACM

WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 23
[39] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval Augmented Generation.arXiv preprint arXiv:2401.15884(2024).
[40] Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. 2024. LLaMA-adapter: Efficient fine-tuning of
large language models with zero-initialized attention. InThe Twelfth International Conference on Learning Representations.
[41] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al .2023.
A survey of large language models.arXiv preprint arXiv:2303.182231, 2 (2023).
[42] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al .2024. Recommender
systems in the era of large language models (llms).IEEE Transactions on Knowledge and Data Engineering36, 11 (2024), 6889–6907.
[43] Zihuai Zhao, Wenqi Fan, Yao Wu, and Qing Li. 2025. Investigating and Mitigating Stereotype-aware Unfairness in LLM-based Recommendations.
arXiv preprint arXiv:2504.04199(2025).
[44] Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. 2024. On the role of
attention heads in large language model safety.arXiv preprint arXiv:2410.13708(2024).
[45] Guido Zuccon, Shengyao Zhuang, and Xueguang Ma. 2025. R2LLMs: Retrieval and Ranking with LLMs. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval. 4106–4109.
Manuscript submitted to ACM
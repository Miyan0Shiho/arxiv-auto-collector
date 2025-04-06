# Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Published**: 2025-04-03 10:22:30

**PDF URL**: [http://arxiv.org/pdf/2504.02458v1](http://arxiv.org/pdf/2504.02458v1)

## Abstract
Recently, Large Language Model (LLM)-empowered recommender systems have
revolutionized personalized recommendation frameworks and attracted extensive
attention. Despite the remarkable success, existing LLM-empowered RecSys have
been demonstrated to be highly vulnerable to minor perturbations. To mitigate
the negative impact of such vulnerabilities, one potential solution is to
employ collaborative signals based on item-item co-occurrence to purify the
malicious collaborative knowledge from the user's historical interactions
inserted by attackers. On the other hand, due to the capabilities to expand
insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG)
techniques provide unprecedented opportunities to enhance the robustness of
LLM-empowered recommender systems by introducing external collaborative
knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by
retrieving external collaborative signals to purify the poisoned user profiles
and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner.
Specifically, retrieval-augmented perturbation positioning is proposed to
identify potential perturbations within the users' historical sequences by
retrieving external knowledge from collaborative item graphs. After that, we
further retrieve the collaborative knowledge to cleanse the perturbations by
using either deletion or replacement strategies and introduce a robust ensemble
recommendation strategy to generate final robust predictions. Extensive
experiments on three real-world datasets demonstrate the effectiveness of the
proposed RETURN.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented Purifier for Robust
LLM-Empowered Recommendation
Liangbo Ning Wenqi Fan∗Qing Li
The Hong Kong Polytechnic University, Hong Kong SAR, China
{BigLemon1123,wenqifan03}@gmail.com ,qing-prof.li@polyu.edu.hk
Abstract
Recently, Large Language Model (LLM)-empowered recommender systems have
revolutionized personalized recommendation frameworks and attracted extensive
attention. Despite the remarkable success, existing LLM-empowered RecSys have
been demonstrated to be highly vulnerable to minor perturbations. To mitigate
the negative impact of such vulnerabilities, one potential solution is to employ
collaborative signals based on item-item co-occurrence to purify the malicious
collaborative knowledge from the user’s historical interactions inserted by attackers.
On the other hand, due to the capabilities to expand insufficient internal knowledge
of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprece-
dented opportunities to enhance the robustness of LLM-empowered recommender
systems by introducing external collaborative knowledge. Therefore, in this paper,
we propose a novel framework ( RETURN ) by retrieving external collaborative
signals to purify the poisoned user profiles and enhance the robustness of LLM-
empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented
perturbation positioning is proposed to identify potential perturbations within the
users’ historical sequences by retrieving external knowledge from collaborative
item graphs. After that, we further retrieve the collaborative knowledge to cleanse
the perturbations by using either deletion or replacement strategies and introduce
a robust ensemble recommendation strategy to generate final robust predictions.
Extensive experiments on three real-world datasets demonstrate the effectiveness
of the proposed RETURN.
1 Introduction
In today’s era of information explosion, recommender systems play a vital role in enhancing user
experiences and influencing user decisions by filtering out irrelevant information in various appli-
cations such as streaming platforms (e.g., YouTube [ 9,10], TikTok [ 38,5]) and e-commerce (e.g.,
Amazon [ 24], Taobao [ 48]). Technically, most existing representative recommendation methods
aim to capture collaborative signals by modeling user-item interactions [ 13,12,21]. Recently, large
language models (LLMs) have been widely applied in real-life scenarios due to their powerful capabil-
ities in language comprehension and generation, and rich store of open-world knowledge [ 51,77,44].
For example, as one of the most famous AI chatbots in recent years, ChatGPT [ 1] has showcased
human-level intelligence with impressive logical reasoning, open-ended conversation, and personal-
ized content recommendation abilities. To fully leverage the powerful capabilities of large language
models, a significant amount of research has utilized LLMs to revolutionize recommender systems
for next-generation RecSys [ 50,77,35,62]. For instance, Geng et al. [18] propose P5, which unifies
various recommendation tasks by converting user-item interactions to natural language sequences,
∗Corresponding author: Wenqi Fan, Department of Computing, and Department of Management and
Marketing, The Hong Kong Polytechnic University.
Preprint. Under review.arXiv:2504.02458v1  [cs.IR]  3 Apr 2025

User-Item InteractionsSkirt
Dress
TieJean
BootHeel
(a) Unsuitable Recommendations
with Adversarial Prompts(b) Reliable Recommendations
with Adversarial PromptsI have purchased  Skirt ,
[Ties] ,  Heels . What
should I buy next?
Shawl
（External ）Database
RetrievalDress Heel ... ...
... Dress Skirt ...
Jean Tie ... ...
Dress Shawl ... ...
Retrieved SubgraphsSkirt
Dress
TieHeel
Collaborative 
Item Graphs
You might want to
buy Dresses .LLM-Empowered
RecSys[Ties]  are the noise.I have purchased  Skirt,
[Ties] ,  Heels . What
should I buy next? User
No. I don't want.
User
User
Yes. I want.
User
You might want
a Suit.
LLM-Empowered
RecSys
Figure 1: The illustration of the robust LLM-empowered RecSys by introducing an external database
(i.e., collaborative item graph). The minor perturbations (e.g., item ‘ Ties’) in the user’s historical
sequence (i.e., adversarial prompt) can mislead LLM-empowered recommender systems to understand
the user’s preference. With the help of the external data source, LLM-empowered recommender
systems can identify the irrelevant item ‘ Ties’ by retrieving relevant collaborative signals (i.e.,
retrieved subgraphs) from the collaborative item graphs, so as to purify the perturbations for the
robust recommendation.
achieving outstanding recommendation performance due to the rich textual information that can help
capture complex semantics for personalization and recommendations.
Despite the remarkable success, most existing LLM-empowered RecSys still encounter a key limi-
tation, in which they have been demonstrated to be highly vulnerable to minor perturbations in the
input prompt [ 45], greatly constraining their practical applicability. Suppose that attackers might
post products with enticing images and titles to attract user clicks on an e-commerce platform. Users
are easily drawn to these clickbait products and interact with them, even though the content of these
goods may not truly align with their preferences [ 61]. Such minor perturbations (e.g., irrelevant items)
can easily lead the LLM-empowered RecSys to misunderstand the user preferences by capturing
the collaborative knowledge from the user’s historical interactions towards items. For example, as
illustrated in Figure 1, when perturbation item "ties" is inserted into the user’s interaction sequence,
the perturbed collaborative knowledge makes LLM-empowered RecSys struggle to discern whether
the user is seeking men’s clothing (i.e., "suits") or women’s clothing (i.e., "dresses"), leading to
inaccurate recommendation outcomes. That is due to the fact that attackers tend to add items that are
irrelevant to users’ behaviors for hindering collaborative knowledge learning [ 14,7]. In order to de-
fend such minor perturbations for robust recommendations, one of the promising solutions is to purify
malicious collaborative knowledge from the user’s historical interactions towards items in LLM-based
recommender systems. In most recommender systems, collaborative graphs based on item-item
co-occurrence are commonly employed as a collaborative signal to represent the relationships among
items, where items frequently interacted together by different users are related (e.g., substitutable or
complementary) [ 42,33]. Following the insertion of perturbations by attackers, item co-occurrence
collaborative graphs as the external knowledge source can provide valuable evidence on whether
such perturbations are relevant to other items in the user’s interaction history and effectively filter out
malicious collaborative signals (i.e., perturbed items), which can be achieved by retrieving subgraphs
and examining the connection between the perturbation and the retrieved subgraphs.
Recently, to mitigate the problems usually caused by insufficient intrinsic knowledge of LLMs, in-
cluding outdated knowledge, hallucination, and so on [ 52,16,29,41], retrieval-augmented generation
(RAG) techniques [ 15] have been proposed to expand the internal knowledge of large language
models with an external database. The relevant knowledge is retrieved from the external database
2

and employed to augment LLMs without changing the parameters of LLM backbone, achieving
outstanding success for various knowledge-intensive domains such as open question answering [ 28],
medicine [ 17], and finance [ 32,71]. For example, Lewis et al. [28] propose to utilize Wikipedia for
knowledge retrieval and combine the retrieved documents with the input to augment the generation
process, significantly improving the performance of LLMs for various complex tasks and mitigating
the hallucination problem. In the context of RecSys, there exists a vast amount of publicly available
external collaborative knowledge collected from various public platforms such as Amazon [ 37],
Yelp [ 40], and Steam [ 46]. Given the success of expanding the internal knowledge of LLMs through
the use of external databases to enhance their capabilities in a training-free manner, along with the
abundant collaborative knowledge available in the RecSys community, RAG techniques provide
unprecedented opportunities to enhance the robustness of LLM-empowered RecSys with external
collaborative signals. For example, as shown in Figure 1, LLM-empowered RecSys might generate
an incorrect recommendation to a user who interacted with "skirt, ties, heels". To produce reliable
recommendation results, a collaborative item graph (i.e., external databases) based on user-item
interactions can be constructed to provide useful external collaborative knowledge for better under-
standing users’ preferences in LLM-based recommender systems. LLM-based RecSys can purify
the noisy users’ online behaviors (i.e., perturbation "ties") by retrieving collaborative signals (i.e.,
subgraph) from the collaborative item graph for recommendation generation, where items "skirt" and
"heels" rarely appear together with "ties" in most users’ shopping behaviors.
To effectively take advantage of external collaborative signals from item-item collaborative graph, in
this paper, a novel framework RETURN is proposed as a retrieval-a ugmented pu rifier for e nhancing
the robustness of LLM-empowered recommender systems in a plug-and-play manner. Specifically,
the users’ historical sequences within the external databases are first encoded into collaborative
item graphs to capture the extensive collaborative knowledge. After that, a retrieved-augmented
perturbation positioning strategy is proposed to identify potential perturbations by retrieving relevant
collaborative signals from the collaborative item graphs. Then, we further cleanse the potential
perturbations within the user profile by using either deletion or replacement strategies based on
the external collaborative item graphs. Finally, a robust ensemble recommendation strategy is
proposed to guide the LLM-empowered RecSys to generate robust recommendation results. Our
major contributions are summarised as follows:
•We introduce a novel strategy for denoising in LLM-empowered recommendation, in which
training-free retrieval-augmented denoising strategy is proposed to leverage the collaborative
signals of collaborative item graphs to purify the poisoned user profiles.
•We propose a novel framework ( RETURN ) to enhance the robustness of LLM-empowered
RecSys by harnessing collaborative signals from external databases in a plug-and-play
manner. Meanwhile, a robust ensemble recommendation is proposed to cleanse user profiles
multiple times and generate robust recommendations by using a decision fusion strategy.
•We conduct extensive experiments on three real-world datasets to demonstrate the effective-
ness of the proposed method. Comprehensive results indicate that RETURN can significantly
mitigate the negative impact of the perturbations, highlighting the potential of introducing ex-
ternal collaborative knowledge to enhance the robustness of LLM-empowered recommender
systems.
The rest of this paper is organized as follows: Section 2 reviews multiple related studies. Section 3
provides the basic definition of the research problem, and the details of the proposed RETURN are
presented in Section 4. Then, we conduct comprehensive experiments to investigate the effectiveness
of RETURN in Section 5. Finally, we conclude the whole work in Section 6.
2 Related Works
2.1 Defense Strategies for LLMs
Numerous defense strategies have been devised to mitigate LLM vulnerabilities and safeguard against
harmful information in LLM responses. These methods are categorized into two main classes based
on whether they are employed during training or inference.
1) Defense in LLMs Training. The security of LLMs is significantly dependent on their training
data, resulting in several defense strategies aimed at enhancing and purifying the training data [ 47].
3

For example, Wenzek et al. [66] introduced CCNet, an automated pipeline designed to efficiently
extract vast amounts of high-quality monolingual datasets from the Common Crawl corpus across
various languages. Beyond enhancing the quality of training data, adversarial training techniques [ 3]
are widely employed to guide LLMs towards appropriate behaviors by introducing adversarial
perturbations into training examples to improve model robustness and performance [ 70]. For example,
Liu et al. [39] introduced a general algorithm known as adversarial training for large neural language
models (ALUM), aimed at enhancing the robustness of language models. ALUM enhances model
resilience by regularizing the training objective via incorporating perturbations within the embedding
space and focusing on maximizing adversarial loss. Wang et al. [59]) propose a simple yet effective
adversarial training method that incorporates adversarial perturbations into the output embedding layer
during model training. Li and Qiu [30] employs a token-level accumulated perturbation vocabulary
to initialize the adversarial perturbations and use a token-level normalization ball to regulate the
generated perturbations for virtual adversarial training [8].
2) Defense in LLMs Inference. The large scale of parameters of LLMs renders their retraining or
fine-tuning processes both time-consuming and computationally expensive. Therefore, training-free
defense methods during inference have drawn considerable attention [ 70]. For example, Kirchenbauer
et al. [26] and Jain et al. [23] undertake extensive experiments to evaluate the effectiveness of
different defense methods, such as perplexity-based detection, retokenization, and paraphrasing. Li
et al. [31] introduce an adversarial purification method that masks input texts and leverages masked
language models [ 25] for text reconstruction. Wei et al. [65] and Mo et al. [43] propose enhancing
model robustness through contextual demonstrations. Wang et al. [63] propose RMLM, aimed
at countering attacks by confusing attackers and correcting adversarial contexts stemming from
malicious perturbations. Helbling et al. [22] incorporate generated content into a predefined prompt
and utilize another LLM to analyze the text and assess its potential harm.
2.2 LLM-Empowered Recommender Systems
Currently, LLMs are widely employed in enhancing the capabilities of recommender systems due to
their powerful language understanding, logical reasoning, and generation abilities. These studies can
be generally divided into three categories based on the item information utilized.
1) ID-Based LLM-Empowered Recommender Systems. ID-based LLM-empowered recommender
systems represent an item with a numerical index and use the item IDs for recommendations [ 18,78].
For example, Geng et al. [18] propose P5, which unifies various recommendation tasks by converting
user-item interactions to natural language sequences. P5 introduces whole-word embedding to
represent the token IDs, bridging the gap between large language models and recommender systems.
Zheng et al. [78] propose a learning-based vector quantization method for assigning meaningful
item indices for items and introduce specialized tasks to facilitate the integration of collaborative
semantics in LLMs, leading to an effective adaptation to recommender systems.
2) Text-Based LLM-Empowered Recommender Systems. To effectively harness the natural language
understanding and generation capabilities of LLMs, text-based LLM-empowered recommender
systems primarily leverage textual information such as item titles and item descriptions for recom-
mendation [ 11,4]. For example, Bao et al. [4]introduce TALLRec, a novel tuning paradigm designed
to tailor LLMs for recommendation tasks effectively, guides the model to assess user interest in a
target item by analyzing their historical interactions that encompass textual descriptions like item
titles. Du et al. [11] propose a novel LLM-based approach for job recommendation that enhances
user profiling for resume completion by extracting both explicit and implicit user characteristics
based on users’ self-description and behaviors. A GANs-based method is introduced to refine the
representations of low-quality resumes, and a multi-objective learning framework is utilized for job
recommendations.
3) Hybrid LLM-Empowered Recommender Systems. These approaches effectively integrate both
textual information and ID-based knowledge to generate recommendations [ 34,53]. For example,
Ren et al. [53] leverage text-format knowledge from LLMs and item IDs to enhance recommendation
performance, along with a novel alignment training method and an asynchronous technique to refine
LLMs’ generation process for improved knowledge augmentation and accelerated training. Liao
et al. [34] propose a novel hybrid prompting approach that integrates ID-based item embedding
generated by traditional RecSys with textual item features. Besides, LLaRA utilizes a projector to
align traditional recommender ID embeddings with LLM input space and incorporates a curriculum
4

learning strategy to gradually train the model to integrate behavioral knowledge from traditional
sequential recommenders, thereby enhancing recommendation performance seamlessly.
2.3 Denoising for Traditional Recommender Systems
With the development of RecSys, a growing body of research has focused on their vulnerability
to noisy data, subsequently driving the advancement of various denoising approaches to improve
system robustness [ 60,36,75,55]. For example, GraphRfi [ 75] proposes an innovative end-to-end
framework that integrates Graph Convolutional Networks (GCN) and neural random forests to
simultaneously enhance robust recommendation accuracy and fraudster detection. By leveraging
user reliability features and prediction errors of RecSys, GraphRfi effectively mitigates the impact of
shilling attacks [ 54,19]. LoRec [ 73] proposes to enhance the robustness of sequential recommender
systems against poisoning attacks by integrating the open-world knowledge of large language models.
Through LLM-Enhanced Calibration, LoRec employs a user-wise reweighting strategy to generalize
defense mechanisms beyond specific known attacks, effectively mitigating the impact of fraudsters.
LLM4DASR [ 58] introduces an LLM-assisted denoising framework for sequential recommendations,
combining self-supervised fine-tuning with uncertainty estimation to address output quality challenges.
This model-agnostic framework effectively identifies and corrects noisy interactions, enhancing
recommendation performance across various models.
2.4 Difference between Existing Denoising Approaches and RETURN
Despite the presence of existing denoising techniques, they are fundamentally different from our
approach in terms of task formulation and technical details:
1)Denoising for different phases. Existing denoising methods primarily focus on purifying
the training set and ensuring accurate representation learning for RecSys to mitigate the impact of
shilling attacks during training, assuming that user historical interactions during inference contain
no perturbations. However, during the inference phase, users may still be attracted to clickbait
items and interact with them, leading to perturbations that do not align with their true preferences.
Moreover, studies [ 45] have highlighted the vulnerability of LLM-empowered recommender systems
during inference, where even a well-trained LLM-based RecSys frequently produces inaccurate
recommendations for users affected by poisoned interactions. In other words, even after the training
set has been purified, if a user inadvertently interacts with a few clickbait or disliked items during
inference, LLM-empowered RecSys may still misinterpret the user’s preferences and generate
unsatisfied recommendations. In this paper, we assume that the LLM-empowered RecSys is well-
trained, while user interaction sequences may contain noise or adversarial perturbations during
inference. In other words, RETURN is designed to address the inference-phase vulnerability of
LLM-empowered RecSys and enhance their robustness, which is fundamentally different from the
objective of previous denoising methods. Additionally, RETURN can be seamlessly integrated with
prior denoising approaches. For instance, existing methods can be employed to cleanse the training
set and train a powerful LLM-empowered RecSys, while RETURN ensures the robustness of the
RecSys during the inference phase.
2)Novel purification techniques based on external collaborative signals. Existing denoising
methods primarily rely on leveraging the characteristics of perturbations [ 60,75,55] or the open-world
knowledge of LLMs [ 73,58] to identify perturbations within the training set, largely overlooking
the potential of external collaborative knowledge. With the advancement of recommender systems,
numerous publicly available datasets have been introduced to evaluate algorithm performance. These
datasets offer abundant external collaborative signals that can be leveraged to purify perturbations
within user historical interactions. Specifically, after attackers introduce perturbations, collaborative
graphs based on item-item co-occurrence can be constructed from the external database and leveraged
to assess whether these perturbations are consistent with other items in the user’s interaction history,
thereby effectively filtering out malicious collaborative signals (i.e., perturbed items). Therefore,
RETURN effectively extracts collaborative knowledge from external databases to purify the user
historical interactions in a plug-and-play manner, providing a promising solution for enhancing the
robustness of LLM-empowered RecSys.
5

3 Problem Statement
3.1 Notation and Definition
The objective of recommender systems is to capture the users’ preferences from their historical inter-
actions, such as browsing, clicking, and purchasing. In the era of LLMs, the recommendation task
is usually converted to the natural language format, consisting of user ui∈U={u1, u2, ..., u |U|},
and user’s interaction history (also called user’s profile) Iui= [I1, I2, ..., I|Iui|], and a recommen-
dation prompt P= [p1, p2, ..., p |P|], where piis the textual token used to guide the RecSys Rθto
generate recommendations. Ii∈ I={I1, I2, ..., I|I|}is the interacted item from the item pool I
of user ui. Based on the above definition, a textual recommendation query can be represented as
x= [P ◦ui◦ Iui], where ◦represents inserting the information of user uiand the corresponding
interaction list Iuiinto the designated position of prompt P. For example, as shown in Figure 2, after
inserting the user information and item interaction sequence into the prompt P, the specific input
used for recommendation can be denoted by:
P=[what, is, the, top, recommended, item, for, [User _235],
who, interacted, with, [item _123, ..., item _928],?],(1)
where ui= [User _235] andIui= [item _123, ..., item _928] are the specific user and the historical
interactions of user ui, respectively. In different LLM-empowered RecSys, Iuican take various
forms, such as numeric IDs [ 18] or item titles [ 4] for recommendations. Assume the target item is y,
the performance of the LLM-empowered RecSys can be defined by:
D(Rθ(x),y), (2)
where Devaluates the discrepancy between the generated recommendations Rθ(x)and the ground
truth. During training, the negative log-likelihood function could be used as the D, while during
inference, the Hit Ratio or Normalized Discounted Cumulative Gain (NDCG) [ 18] could be employed
to evaluate the recommendation performance.
3.2 Vulnerabilities of LLM-Based RecSys
The vulnerabilities of LLM-based RecSys refer to the phenomenon where the model’s recommenda-
tion outcomes vary significantly due to minor perturbations in the input [ 45]. Such vulnerabilities
significantly deteriorate the overall user experience and compromise the effectiveness of RecSys.
For example, assuming a user uiinadvertently clicks on some items they are not actually inter-
ested in, leading to a change in their interaction history from IuitoˆIui=I(Iui◦δ|s), where
I(Iui◦δ|s)represent to insert perturbation δinto user’s profile Iuiat position s. The vulnerability
of LLM-empowered RecSys may lead the system to recommend items that the user is not interested
in, consequently resulting in a decline in user experience. As an attacker, the perturbations δcan be
generated intentionally by optimizing the following equation:
δ= arg max
δ:|δ|≤△D(Rθ(ˆx),y), (3)
where ˆx= [P ◦ui◦ˆIui]is the perturbed input and △constrains the magnitude the perturbations. D
evaluates the discrepancy between the generated recommendations Rθ(x)and the ground truth.
3.3 Robust LLM-based Recommendation
The primary objective of robust LLM-based recommendations is to prevent the negative impact
of the perturbations contained in the users’ profiles, thereby enhancing the system’s reliability
and robustness. There are mainly two approaches to achieve this goal: adversarial training-based
methods [ 67] and training-free methods [ 56]. Adversarial training-based methods intentionally
create multiple perturbed training samples to guide the RecSys in learning patterns of perturbations,
thereby improving system robustness. However, these methods usually retrain or fine-tune the whole
RecSys, which is extremely time-consuming due to the large number of trainable parameters in
LLMs. Consequently, this paper primarily concentrates on the training-free methods, which improve
6

#1 Retrieval-Augmented Perturbation Positioning
:
......
Recommendation Generation
...goodsAdversarial Prompt  
isthetoprecommended itemforuser_235 whointeracted with item_2298, item_2143? What item_4396, item_2096, item_1009, item_1109,
 Occurrence Probability:
item_2143, item_2096, item_1009, ...item_2143, item_1009, ... item_1109,
item_2143, item_4396, item_1009, ...item_1426,
Top-K
Recommendations RecSysPerturbation Putification... item_1009, item_2298, item_4396, item_1109,
item_1009, ... item_2298, item_1109, item_4396,[0,00, 0.125, 0.284, 0.335, ...]
item_1429,item_1429,
?item_1109,item_1429,
?×
×
×
×item_2096,
??
item_1109,item_5251,#2 Retrieval-Augmented Denoising
#3 Robust Ensemble Recommendation
?
?item_5251,
?
 Retrieval of Co-Occurrence Frequency
 Retrieval-Augmented ReplacementUsersDeletion ×
?Replacement
ItemsCleansed
Prompt
Perturbations item_1009Adversarial
Prompt
PredictionsLLM-Based
RecSysMajority Voting
item_1009, ... item_1429,× ?item_4396, item_1109,Replacement Operation:Perturbation Positioning
...
: ...
: ......Collaborative  Item Graph Generation
1-Hop  Collaborative 
Item Graph2-Hop  Collaborative 
Item Graph-Hop Collaborative 
 Item GraphDatabase 
...
Figure 2: The overall framework of the proposed RETURN. The user interaction sequences in
the external database are first converted to multi-hop collaborative item graphs. The occurrence
probability of each item is computed based on the collaborative item graph for perturbation positioning.
Finally, we purify the input prompt by retrieving collaborative signals from the collaborative item
graphs for robust recommendation generation.
the model’s robustness without introducing additional changes in model parameters. Specifically,
when the users’ profiles contain adversarial perturbations, we aim to accurately identify and filter out
these perturbations to ensure the appropriate recommendations for users during the inference process.
Mathematically, if the input with adversarial perturbations is denoted by ˆx, we aim to cleanse the
input for robust recommendations, formulated as follows:
¯x= arg min D(Rθ(C(ˆx)),y), (4)
whereC(ˆx)represents purifying input ˆxcontaining perturbations into a benign prompt ¯x.
4 Methodology
4.1 An Overview of the RETURN
RETURN is proposed to leverage the collaborative knowledge of users within external databases
to filter out adversarial perturbations, thereby enhancing the robustness of the existing LLM-based
RecSys. As shown in Figure 2, RETURN mainly contains three components: Retrieval-Augmented
Perturbation Positioning, Retrieval-Augmented Denoising, and Robust Ensemble Recommendation.
First, we convert the user interaction sequences within the external database to collaborative item
graphs to encode the collaborative knowledge without introducing additional training processes. After
that, the probability of each item in the user profile being a perturbation is computed by retrieving
collaborative signals from collaborative item graphs. Second, retrieval-augmented denoising filters
out the potential perturbations in the user profiles using either deletion or replacement strategies
based on the collaborative signals of the generated item graphs. Finally, robust ensemble recom-
mendation purifies input query multiple times and adopts an ensemble strategy to generate the final
recommendations.
4.2 Retrieval-Augmented Perturbation Positioning
To mitigate the negative impact of perturbations, the first crucial step is to accurately locate the
perturbations from extensive interactions within user profiles. To achieve this goal, we propose to
use collaborative item graphs to encode the collaborative signals from users in the external database
and retrieve relevant collaborative knowledge for perturbation positioning. By encoding the user’s
interaction history into collaborative item graphs, we can clearly understand the relationships between
items [ 74], and such strong collaborative signals can provide evidence for subsequent denoising
7

processes. Furthermore, this approach enables the direct use of a one-hot vector for information
retrieval, eliminating the need to explicitly train a universal retriever as required by other RAG
techniques, thereby improving efficiency.
4.2.1 Collaborative Item Graph Generation
LetE={UE,IE}be the external database, where UE={u1, u2, ..., u E}andIE=
{Iu1,Iu2, ...,IuE}denote the users and their interaction sequences, respectively. The most straight-
forward method of generating collaborative item graphs is to count the occurrence frequency of i-th
andj-th items appearing together in the historical interaction of the same user. However, such a
vanilla strategy overlooks the temporal relationships among items, which are significantly crucial for
subsequent denoising processes. For instance, mobile phones and phone cases are usually interacted
with consecutively by users, whereas mobile phones and furniture are typically not sequentially inter-
acted with by users. During the denoising process, if a user consecutively interacts with both mobile
phones and furniture, there is a high likelihood of perturbations in the user’s historical interactions.
Therefore, to provide precise collaborative signals, we also consider the gap between two items and
generate a set of multi-hop collaborative item graphs, which encode not only the relevance between
items but also the temporal relationships of items. Given the external database E={UE,IE}, the
multi-hop collaborative item graph can be represented as:
Gϵ={I,Cϵ
Ii,Ij}={I,T(Cϵ
Ii,Ij|uz, ϵ)}, (5)
where the ϵ-hop collaborative item graph contains nodes Iand edges Cϵ
Ii,Ij, respectively. The edge
Cϵ
Ii,Ijstores the co-occurrence frequency of two items. T(·)is a counting function. If two items
IiandIjappear simultaneously within the historical interactions of user uzwith a gap of ϵitems
between them, the co-occurrence frequency Cϵ
Ii,Ijis increased by one, defined by:
T(Cϵ
Ii,Ij|uz, ϵ)}=(
Cϵ
Ii,Ij+ 1,ifIi, Ij∈ Iuz,|i−j|=ϵ,
Cϵ
Ii,Ij, otherwise .(6)
4.2.2 Perturbation Positioning
After encoding the external users’ collaborative knowledge into collaborative item graphs, the next
step is to locate the perturbations within the input query based on the generated graphs. Specifically,
if one item has never appeared together with the remaining items in the user’s historical interactions
based on the collaborative item graphs derived from the majority of users’ behavior, this indicates
that such item is unrelated to the other items the user has interacted with. Thus, the likelihood of this
item appearing within the user’s interaction history is minimal, and the occurrence of such a low-
probability event strongly implies that this item is likely introduced as a perturbation by an attacker.
Therefore, to accurately locate the potential perturbations, we propose to retrieve co-occurrence
frequency from the collaborative item graphs and assess the probability of each item appearing within
the user’s interaction history.
Given the multi-hop collaborative item graphs Gϵand the user’s historical interactions Iui=
[I1, I2, ..., I|Iui|], the co-occurrence frequency for a pair of items can be defined as:
oi,j=R(Ii, Ij|Gϵ)P
z∈[1,|I|]R(Ii, Iz|Gϵ), (7)
where ϵ=|j−i|is the gap between i-th and j-th items and R(Ii, Ij|Gϵ)represent to retrieve the
co-occurrence frequency between i-th and j-th items from the ϵ-hop collaborative item graph. By
traversing each pair of items in the user’s historical interactions, the occurrence probability of each
item is denoted by:
A= [A1, ..., A |Iui|],
Ai=P|Iuz|
j=1,j̸=ioi,j.(8)
A smaller value of Aiindicates a lower probability of the current item co-occurring with other items,
making it more likely to be a perturbation.
8

4.3 Retrieval-Augmented Denoising
Once the occurrence probability of each item has been computed, it is necessary to purify the input
query based on such collaborative knowledge. However, directly removing numerous items that
may be perturbations usually leads to the RecSys failing to capture the user’s preferences accurately
since there are limited remaining interactions. To mitigate the negative impact of perturbations
while maintaining the integrity of user interaction sequences, a hybrid strategy is proposed to
eliminate a small subset of items that are most likely perturbations and replace the remaining potential
perturbations with items that align with the user’s preferences.
If an item’s occurrence probability Ai= 0, it indicates that this item has never co-occurred with
the other items in the user’s historical interactions based on the collaborative signals of most users
from external databases. Thus, this item is highly likely a perturbation inserted by attackers due to
its lack of relevance to the user’s other interaction items, and its deletion typically helps RecSys
accurately capture the user’s genuine preferences. Mathematically, given the interaction history Iui
of the user uiand the occurrence probability A, we first delete the most likely perturbation items
whose occurrence probability is zero, defined by:
Cd(Ii→ ∅|Ai) =1(Ii, Ai), (9)
where 1(Ii, Ai)represents to execute deletion operation when Ai= 0and preservation otherwise.
After removing items with Ai= 0that are most likely perturbations, some items usually remain in
the user’s historical interactions with very low but non-zero occurrence probabilities. Simply deleting
these items usually results in sparse user-item interactions, and such limited collaborative knowledge
leads to cold start issues [ 64], hindering RecSys from capturing user preferences effectively. To main-
tain the integrity of user interaction history, a retrieval-augmented replacement strategy is proposed to
replace the remaining potential perturbations with items that align with user preferences. Specifically,
all items that have co-occurred with the other remaining items in the user’s interaction history are
retrieved from the collaborative item graphs, and the item that shows the highest co-occurrence
frequency is considered the prime candidates that best align with the current user preferences among
all the retrieved items. Given the user’s historical interactions Iui= [I1, ..., II|ui|]and the potential
perturbation Iiwith the low occurrence probability, the replacement operation is defined by:
Cr(Ii→¯Ii|Ai) = arg max
¯Ii|Iui|X
j=1,j̸=iR(Ij|Gϵ)·AjP
z∈[1,|I|]R(Ij, Iz|Gϵ), (10)
whereCr(Ii→¯Ii|Ai)represents to replace the potentially perturbed item Iiwith alternative items
¯Iithat better align with user preferences. R(Ij|Gϵ)represent to retrieve the co-occurrence frequency
between item Ijand all items that have co-occurred with Ijfrom the item pool Ibased on the ϵ-hop
collaborative item graph Gϵ, where ϵ=|j−i|.Ajis considered as a weight, where a larger Aj
indicates a closer alignment between the current item Ijand the user’s preference, thus resulting in
greater weights assigned to items that are likely to co-occur with it.
4.4 Robust Ensemble Recommendation
Due to the uncertainty regarding the number of perturbations, determining the extent of purification
applied to the user’s historical interactions is a challenging task. Excessive modification of items leads
to difficulties in capturing the user’s intrinsic preferences, thereby diminishing the recommendation
performance. Conversely, the limited purification of items results in perturbations still existing in
user profiles, making it challenging to enhance the robustness of RecSys. To tackle this challenge, we
propose a robust ensemble recommendation approach. Specifically, we first randomly purify varying
numbers of items in the user’s profile and generate a set of cleansed inputs. These cleansed prompts
are fed into the LLM-based RecSys, and the final recommendations are obtained by adopting a voting
mechanism [ 27]. Technically, we randomly sample an integer nfrom a normal distribution. Top- n
items ˆIn
ui= [I1, I2, ..., In]with the lowest occurrence probabilities are identified from the user’s
historical interactions based on Aand deletion Cdor substitution Croperations are performed on
these items. The purified user profile is defined by:
¯Iui= [C(I1|A1, n), ...,C(Iui|Aui, n)], (11)
9

Algorithm 1: RETURN
Input:
Input ˆx, External database E, Purification cycle m, LLM-empowered RecSys Rθ.
Output: Robust recommendations ¯y.
Procedure:
1Generate multi-hop collaborative item graph Gϵaccording to Eq (5) ;
2Retrieve the co-occurrence frequency for a pair of items within Iuiaccording to Eq (7) ;
3Compute the occurrence probability for each item within Iuiaccording to Eq (8) ;
4fort in 1:m do
5 Purify the user’s historical interactions according to Eq (12) ;
6 Generate the recommendations for each purified prompt ¯xiand obtain a set of
recommendation results [Rθ(¯x1),Rθ(¯x2), ...,Rθ(¯xm)];
7Generate the final recommendations based on Eq (13) ;
whereC(Ii|Ai, n)is the purifying process:
C(Ii|Ai, n) =

Cd(Ii→ ∅|Ai),ifIi∈ˆIn
uiandAi= 0,
Cr(Ii→¯Ii|Ai),ifIi∈ˆIn
uiandAi̸= 0,
Ii, ifIi/∈ˆIn
ui.(12)
By repeating the purification process multiple times on Iui, we can obtain mcleansed user profiles,
where mis a hyperparameter. These purified prompts are individually fed into the LLM-empowered
RecSys, and the results are subsequently integrated to produce the final recommendation output by
using voting mechanisms, defined by:
¯y=V oting (Rθ(¯x1),Rθ(¯x2), ...,Rθ(¯xm)), (13)
where ¯xi= [P ◦ui◦¯Iui]is the purified input and ¯yis the final recommendation. The pseudo-code
of RETURN is shown in Algorithm 1.
5 Experiments
5.1 Experimental Details
5.1.1 Datasets.
All experiments are conducted on three real-world datasets in RecSys: Movielens-1M ( ML1M ) [20],
Taobao [79], and LastFM [68] datasets. The ML1M dataset contains one million movie ratings
collected from around 6,040 users and their interactions with around 4,000 movies, which is widely
used for various recommendation tasks and evaluation of recommendation techniques. The LastFM
dataset is a widely used music recommendation dataset that contains user listening histories and pref-
erences, which is frequently used to study user preferences, understand music consumption patterns,
and evaluate recommendation algorithms. The Taobao dataset comprises a massive collection of user
interactions on the Taobao e-commerce platform, including browsing, searching, and purchasing
activities. It consists of a million records from around 987,994 users and their interactions with
around 4,162,024 items and offers valuable insights into user behavior and preferences in the online
retail environment. For P5model, all the aforementioned datasets are preprocessed following the
strategies proposed by Xu et al. [68]. For TALLRec model, it needs to divide the users’ historical
sequences into users’ liked items and disliked items based on their ratings. Since LastFM and Taobao
datasets lack rating information from users, we only process the ML1M dataset according to the
study of Ning et al. [45].
5.1.2 Victim LLM-based Recommender Systems.
Two representative LLM-based RecSys, i.e., P5andTALLRec , are employed as the victim models
to investigate the performance of different defense techniques.
•P5is a typical ID-based LLM-empowered RecSys, which assigns each item a numerical number
and converts the user-item interactions to natural language sequences for recommendations. P5
10

introduces several item indexing strategies, which can be employed to test the robustness of the
defense methods for ID-based RecSys with different indexing strategies.
•TALLRec is a representative text-based LLM-empowered recommender system, which integrates
textual information (i.e., item title) into a pre-defined prompt template for recommendation. By
constructing experiments based on TALLRec, we can investigate the performance of different
defense methods for LLM-empowered RecSys employing textual knowledge.
5.1.3 Attackers.
We employ CheatAgent [ 45] as the attacker to generate adversarial perturbations and insert them into
the user’s historical interactions. It should be noted that CheatAgent is an evasion attack method
that uses LLMs as the agent to generate high-quality perturbations for misleading the target LLM-
empowered RecSys during the inference phase. Currently, there is limited research on poisoning
attacks for LLM-empowered RecSys. Poisoning attacks require retraining the model, but the large
parameter size of LLMs makes frequent retraining infeasible. In other words, poisoning attacks are
highly time-consuming for large language models, and they are ineffective if retraining cannot be
performed. Therefore, in this paper, we solely consider the evasion attack (i.e., CheatAgent) since
it is a more efficient attacking method in the era of LLMs. We use CheatAgent to generate item
perturbations and insert them into the user’s history interactions to test the defense performance
of different methods. The primary objective of CheatAgent is to investigate the vulnerabilities of
exiting LLM-empowered RecSys, and it allows the insertion of perturbations in both prompt and
users’ profiles. However, during real-world applications, the attacker and users usually have no access
to the prompt P, which makes the prompt attack infeasible. Therefore, in this paper, we only use
CheatAgent to generate item perturbations and insert them into the user’s history interactions.
5.1.4 Baselines.
Several baselines are utilized to investigate the defense performance of different methods:
•RD[76] randomly deletes some items within the users’ historical sequences to filter out the
adversarial perturbations.
•PD[23] computes the perplexity for each item and filters out the item with high perplexity for
defense.
•RPD [23] uses an LLM [ 57] to paraphrase the input prompt, which is widely used as the safeguard
for LLMs.
•RTD [23] retokenizes the input prompt, which aims to break tokens apart and disrupt adversarial
behaviors.
•LLMSI [56] provides a safety instruction, i.e., "Please take into account the noise present in
the user’s historical interactions and filter them", along with the input prompt to guide the LLM-
empowered RecSys to defense adversarial attacks by themselves.
•RDE [6] randomly deletes some items within the users’ interaction sequences and generates
multiple cleansed prompts. The final recommendations are obtained by majority voting [27].
•ICL [56] randomly retrieves several users with different historical sequences as the demonstrations
and integrates the retrieved users’ profiles with the original prompt for recommendation.
5.1.5 Implementation.
The proposed RETURN and all baselines are implemented by Pytorch. All victim models (i.e., P5
andTALLRec ) and the attacker algorithm (i.e., CheatAgent ) are implemented based on their official
codes. The training and test set are constructed according to the studies of Xu et al. [68] and Bao et al.
[4]for P5 and TALLRec, respectively. We adopt CheatAgent to generate adversarial perturbations
and insert them into the benign users’ interaction history of the test set to investigate the defense
performance of different methods. The magnitude of perturbations △is set to 3, consistent with the
study of Ning et al. [45]. For the proposed RETURN, we directly use the training set as the external
database. During the recommendation generation process, m= 10 is set as default, meaning that
the final ensemble recommendation is obtained based on these 10 purified prompts. For RD, we
randomly delete 3 items and generate recommendations. For PD, we select the top 3 items with
11

the highest perplexity as the perturbations and delete these items for recommendations. For RTD,
we adopt the BPE-dropout [ 49] to tokenize the input query to mitigate the impact of adversarial
perturbations. RDE generates 10 purified prompts and integrates their recommendation outcomes
as the final prediction. ICL randomly retrieves 5 users’ interaction sequences from the external
database and integrates them with the original input for recommendations. All random seeds were
fixed throughout the experiments, consistent with the used victim RecSys P5 [ 18] and TALLRec [ 4].
This ensures that the experimental results are reproducible, and therefore, we do not include variance
in the reported results.
5.1.6 Evaluation Metrics.
ForP5model, Top- kHit Ratio ( H@k) and Normalized Discounted Cumulative Gain (NDCG)
(N@k) [18] are employed to evaluate the recommendation performance. In this paper, we set k= 5
andk= 10 , respectively. A-H@ kandA-N@ krepresent the extent of the decrease in H@ kand
N@kafter inserting adversarial perturbations into the benign prompt, which are used to measure the
attack performance [45], formulated as:
A-H@k= 1−[H@k
H@k,A-N@k= 1−[N@k
N@k, (14)
wherebH@kandbN@kevaluate the recommendation performance of the victim model when it is under
attack. D-H@ kandD-N@ kare utilized to evaluate the performance of defense algorithms, which
represent the decrease ratio in A-H@ kand A-N@ k, defined as:
D-H@k=gA-H@k
A-H@k−1,D-N@k=gA-N@k
A-N@k−1, (15)
wheregA-H@kandgA-N@krepresent the attack performance when adversarial examples are pro-
cessed by defense algorithms. A greater decrease in A-H@kandA-N@kindicates reduced attack
performance and improved performance of the defense methods. For TALLRec model, we utilize the
Area Under the Receiver Operating Characteristic (AUC) to assess the recommendation performance,
which is consistent with the study of Bao et al. [4]. ASR-A and D-A [ 45] are employed to evaluate
the performance of the attack and defense methods, defined as:
ASR-A = 1−[AUC
AUC,D-A=^ASR-A
ASR-A−1, (16)
where[AUC and^ASR-A represent the AUC when the input contains perturbations and when the input
is purified by the defense methods, respectively.
5.2 Defense Effectiveness
In this subsection, we investigate the defense performance of different methods. The results based on
P5with different indexing methods are summarised in Table 1 and Table 2, and the results based on
TALLRec are shown in Figure 3. Benign denotes the use of the original prompt without perturbations
for recommendations, and CheatAgent represents the recommendation performance under attacks.
Based on these experiments, some insights are obtained as follows:
•As shown in Table 1 and Table 2, the recommendation performance increases after deleting high
perplexity items using PD. However, the effectiveness of this method is not robust. For instance,
on the ML1M dataset, PD can significantly enhance the recommendation performance of RecSys
under attacks. While on the Taobao dataset, the defense performance of PD is limited.
•RPD and RTD, two common defense methods for LLMs, cannot achieve the desired performance
for LLM-empowered RecSys in most cases. The reason is that LLM-empowered RecSys have
captured the domain-specific knowledge of recommendations (e.g., the meaning of item IDs and
item relationships) during the training process. However, the LLMs employed by RPD struggle to
understand item IDs, making it challenging to effectively rewrite the input prompt. Additionally,
RTD disrupts the item ID structure, which further degrades recommendation performance.
12

Table 1: Defense performance of different methods (Victim model: P5, Indexing: Sequential)
Datasets Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
ML1MBenign 0.2116 0.3055 0.1436 0.1737 / / / / / / / /
CheatAgent 0.0646 0.1171 0.0405 0.0573 0.6948 0.6168 0.7181 0.6699 0.0000 0.0000 0.0000 0.0000
PD 0.1303 0.1935 0.0851 0.1053 0.3842 0.3664 0.4077 0.3939 0.4471 0.4060 0.4323 0.4120
RPD 0.0627 0.1070 0.0389 0.0530 0.7034 0.6499 0.7291 0.6950 -0.0124 -0.0536 -0.0153 -0.0374
RTD 0.0093 0.0161 0.0060 0.0082 0.9562 0.9474 0.9579 0.9527 -0.3761 -0.5360 -0.3340 -0.4222
RD 0.0969 0.1526 0.0620 0.0799 0.5423 0.5003 0.5680 0.5400 0.2196 0.1889 0.2090 0.1940
LLMSI 0.0624 0.1073 0.0398 0.0542 0.7050 0.6488 0.7227 0.6878 -0.0146 -0.0518 -0.0064 -0.0267
ICL 0.0546 0.0858 0.0348 0.0449 0.7418 0.7192 0.7574 0.7418 -0.0676 -0.1661 -0.0547 -0.1073
RDE 0.0924 0.1566 0.0581 0.0786 0.5634 0.4873 0.5951 0.5475 0.1892 0.2100 0.1712 0.1827
RETURN 0.1384 0.2091 0.0915 0.1142 0.3459 0.3154 0.3630 0.3427 0.5023 0.4886 0.4945 0.4885
LastFMBenign 0.0404 0.0606 0.0265 0.0331 / / / / / / / /
CheatAgent 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
PD 0.0183 0.0330 0.0124 0.0170 0.5455 0.4545 0.5331 0.4851 0.1724 0.2500 0.2183 0.2503
RPD 0.0183 0.0312 0.0106 0.0147 0.5455 0.4848 0.6006 0.5556 0.1724 0.2000 0.1193 0.1413
RTD 0.0046 0.0110 0.0024 0.0043 0.8864 0.8182 0.9108 0.8691 -0.3448 -0.3500 -0.3355 -0.3430
RD 0.0220 0.0303 0.0139 0.0165 0.4545 0.5000 0.4743 0.5010 0.3103 0.1750 0.3045 0.2258
LLMSI 0.0119 0.0229 0.0080 0.0116 0.7045 0.6212 0.7003 0.6491 -0.0690 -0.0250 -0.0269 -0.0031
ICL 0.0174 0.0321 0.0113 0.0160 0.5682 0.4697 0.5726 0.5172 0.1379 0.2250 0.1604 0.2007
RDE 0.0220 0.0339 0.0128 0.0167 0.4545 0.4394 0.5170 0.4960 0.3103 0.2750 0.2420 0.2334
RETURN 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
TaobaoBenign 0.1420 0.1704 0.1100 0.1191 / / / / / / / /
CheatAgent 0.0863 0.1099 0.0615 0.0690 0.3922 0.3548 0.4409 0.4207 0.0000 0.0000 0.0000 0.0000
PD 0.0935 0.1153 0.0687 0.0758 0.3414 0.3231 0.3752 0.3638 0.1294 0.0894 0.1490 0.1352
RPD 0.0811 0.1027 0.0567 0.0637 0.4291 0.3971 0.4845 0.4657 -0.0941 -0.1192 -0.0989 -0.1069
RTD 0.0016 0.0044 0.0010 0.0019 0.9885 0.9740 0.9908 0.9840 -1.5206 -1.7453 -1.2470 -1.3390
RD 0.0886 0.1121 0.0650 0.0726 0.3760 0.3423 0.4087 0.3905 0.0412 0.0352 0.0731 0.0718
LLMSI 0.0867 0.1112 0.0615 0.0694 0.3899 0.3471 0.4408 0.4176 0.0059 0.0217 0.0002 0.0073
ICL 0.0557 0.0734 0.0385 0.0442 0.6078 0.5692 0.6502 0.6287 -0.5500 -0.6043 -0.4747 -0.4944
RDE 0.0855 0.1217 0.0626 0.0742 0.3979 0.2856 0.4305 0.3770 -0.0147 0.1951 0.0236 0.1038
RETURN 0.1124 0.1384 0.0890 0.0975 0.2088 0.1875 0.1904 0.1817 0.4676 0.4715 0.5682 0.5680
Table 2: Defense performance of different methods (Victim model: P5, Indexing: Random)
Datasets Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
ML1MBenign 0.1058 0.1533 0.0693 0.0847 / / / / / / / /
CheatAgent 0.0421 0.0689 0.0262 0.0348 0.6025 0.5508 0.6221 0.5890 0.0000 0.0000 0.0000 0.0000
PD 0.0626 0.0959 0.0407 0.0514 0.4085 0.3747 0.4127 0.3924 0.3221 0.3196 0.3365 0.3338
RPD 0.0439 0.0717 0.0280 0.0370 0.5853 0.5324 0.5958 0.5634 0.0286 0.0333 0.0423 0.0435
RTD 0.0121 0.0217 0.0076 0.0106 0.8858 0.8585 0.8910 0.8745 -0.4701 -0.5588 -0.4323 -0.4847
RD 0.0425 0.0657 0.0281 0.0355 0.5978 0.5713 0.5950 0.5802 0.0078 -0.0373 0.0435 0.0150
LLMSI 0.0442 0.0695 0.0271 0.0353 0.5822 0.5464 0.6089 0.5832 0.0338 0.0078 0.0212 0.0099
ICL 0.0336 0.0535 0.0214 0.0278 0.6823 0.6512 0.6914 0.6721 -0.1325 -0.1824 -0.1115 -0.1411
RDE 0.0493 0.0800 0.0303 0.0401 0.5336 0.4784 0.5631 0.5268 0.1143 0.1314 0.0949 0.1057
RETURN 0.0929 0.1377 0.0604 0.0750 0.1221 0.1015 0.1276 0.1147 0.7974 0.8157 0.7949 0.8053
LastFMBenign 0.0128 0.0248 0.0072 0.0110 / / / / / / / /
CheatAgent 0.0101 0.0220 0.0055 0.0094 0.2143 0.1111 0.2258 0.1474 0.0000 0.0000 0.0000 0.0000
PD 0.0174 0.0294 0.0102 0.0139 -0.3571 -0.1852 -0.4227 -0.2609 2.6667 2.6667 2.8719 2.7708
RPD 0.0128 0.0202 0.0080 0.0104 0.0000 0.1852 -0.1203 0.0566 1.0000 -0.6667 1.5326 0.6156
RTD 0.0128 0.0193 0.0074 0.0095 0.0000 0.2222 -0.0392 0.1365 1.0000 -1.0000 1.1735 0.0738
RD 0.0110 0.0229 0.0062 0.0101 0.1429 0.0741 0.1281 0.0808 0.3333 0.3333 0.4327 0.4517
LLMSI 0.0101 0.0220 0.0054 0.0093 0.2143 0.1111 0.2451 0.1537 0.0000 0.0000 -0.0854 -0.0429
ICL 0.0073 0.0138 0.0045 0.0066 0.4286 0.4444 0.3746 0.4022 -1.0000 -3.0000 -0.6587 -1.7294
RDE 0.0110 0.0266 0.0073 0.0123 0.1429 -0.0741 -0.0161 -0.1204 0.3333 1.6667 1.0713 1.8173
RETURN 0.0138 0.0220 0.0067 0.0093 -0.0714 0.1111 0.0692 0.1531 1.3333 0.0000 0.6937 -0.0392
TaobaoBenign 0.1643 0.1804 0.1277 0.1330 / / / / / / / /
CheatAgent 0.1012 0.1217 0.0682 0.0749 0.3838 0.3252 0.4661 0.4367 0.0000 0.0000 0.0000 0.0000
PD 0.1042 0.1184 0.0725 0.0771 0.3659 0.3433 0.4327 0.4199 0.0468 -0.0559 0.0717 0.0384
RPD 0.0945 0.1112 0.0640 0.0694 0.4247 0.3833 0.4992 0.4778 -0.1065 -0.1788 -0.0709 -0.0942
RTD 0.0118 0.0190 0.0073 0.0097 0.9282 0.8946 0.9425 0.9273 -1.4182 -1.7514 -1.0220 -1.1234
RD 0.1094 0.1237 0.0774 0.0820 0.3340 0.3143 0.3941 0.3833 0.1299 0.0335 0.1544 0.1223
LLMSI 0.1022 0.1237 0.0684 0.0754 0.3779 0.3143 0.4646 0.4331 0.0156 0.0335 0.0032 0.0082
ICL 0.0655 0.0799 0.0465 0.0511 0.6012 0.5568 0.6360 0.6155 -0.5662 -0.7123 -0.3644 -0.4094
RDE 0.1094 0.1479 0.0793 0.0918 0.3340 0.1798 0.3792 0.3101 0.1299 0.4469 0.1865 0.2899
RETURN 0.1317 0.1537 0.1055 0.1126 0.1984 0.1480 0.1741 0.1532 0.4831 0.5447 0.6266 0.6491
•Adversarial perturbations are typically carefully crafted, so disrupting any component may reduce
the attack’s effectiveness. Therefore, randomly removing a few items from the user’s interaction
history (i.e., RD) can improve the robustness of the LLM-powered RecSys. Furthermore, RDE
generally outperforms RD, suggesting that an ensemble strategy can further enhance system
robustness.
•The proposed RETURN outperforms all other baselines on three datasets and significantly improves
the recommendation performance even under attacks, demonstrating the potential of introducing
collaborative knowledge from external databases. For example, on the Taobao dataset, CheatAgent
reduces the H@5 from 0.1420 to 0.0863. By introducing collaborative knowledge for input purifi-
cation, RETURN raises the H@5 to 0.1124, nearly approaching the recommendation performance
of using benign prompts, which fully demonstrates the effectiveness of RETURN.
•TALLRec uses item titles to construct the input prompt, which has distinct inherent mechanisms
with P5. As shown in Figure 3, the proposed RETURN also dramatically increases the AUC of
TALLRec and decreases the attack performance, demonstrating the robustness of RETURN to the
architecture of the LLM-empowered RecSys.
13

/uni0000000b/uni00000044/uni0000000c/uni00000003/uni00000024/uni00000038/uni00000026/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a
/uni0000000b/uni00000045/uni0000000c/uni00000003/uni00000024/uni00000036/uni00000035/uni00000010/uni00000024/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000014/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018
/uni0000000b/uni00000046/uni0000000c/uni00000003/uni00000027/uni00000010/uni00000024/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000018/uni00000013/uni00000019/uni00000013/uni00000025/uni00000048/uni00000051/uni0000004c/uni0000004a/uni00000051 /uni00000026/uni0000004b/uni00000048/uni00000044/uni00000057/uni00000024/uni0000004a/uni00000048/uni00000051/uni00000057 /uni00000033/uni00000027 /uni00000035/uni00000033/uni00000027 /uni00000035/uni00000037/uni00000027 /uni00000035/uni00000027 /uni0000002f/uni0000002f/uni00000030/uni00000036/uni0000002c /uni0000002c/uni00000026/uni0000002f /uni00000035/uni00000027/uni00000028 /uni00000035/uni00000028/uni00000037/uni00000038/uni00000035/uni00000031Figure 3: Defense Performance on TALLRec
Table 3: The defense performance of the proposed RETURN with respect to different attack methods
Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
Benign 0.0404 0.0606 0.0265 0.0331 / / / / / / / /
PA 0.0064 0.0147 0.0032 0.0060 0.8409 0.7576 0.8777 0.8187 0.0000 0.0000 0.0000 0.0000
RETURN 0.0248 0.0376 0.0148 0.0189 0.3864 0.3788 0.4423 0.4280 0.5405 0.5000 0.4961 0.4772
RA 0.0376 0.0587 0.0251 0.0317 0.0682 0.0303 0.0540 0.0405 0.0000 0.0000 0.0000 0.0000
RETURN 0.0394 0.0587 0.0257 0.0318 0.0227 0.0303 0.0326 0.0377 0.6667 0.0000 0.3957 0.0679
CheatAgent 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
RETURN 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
5.3 Model Analysis
5.3.1 Attack Scenarios: Clickbait and vulnerabilities of LLM-empowered RecSys.
The attack discussed in this paper mirrors a real-world phenomenon, commonly known as click-
bait [ 72,61]. Clickbait refers to the scenario in which attackers might post products with enticing
images and titles to attract user clicks on an e-commerce platform. Users are easily drawn to these
clickbait products and interact with them, even though the content of these goods may not truly
align with their preferences [ 72,61]. However, existing studies [ 45] have demonstrated that LLM-
empowered RecSys is vulnerable to minor perturbations in user historical interactions. If users are
attracted by clickbait products and engage with them, minor perturbations will be introduced to their
historical interactions. Such minor perturbations (e.g., irrelevant items) can easily lead the LLM-
empowered RecSys to misunderstand the user preferences by capturing the collaborative knowledge
from the user’s historical interactions. This leads to inaccurate recommendations, affecting user
experience and engagement and consequently diminishing company profits. Therefore, enhancing
the robustness of the LLM-empowered RecSys is crucial to mitigate the clickbait issue, which is a
practical necessity.
During experiments, to simulate the worst-case scenario, we adopt CheatAgent [ 45], which is
a powerful attacker, to insert perturbations to the user’s historical sequences. Besides, we also
employ various attack methods and perturbation intensities to simulate the scenario in which the
user’s historical interactions contain minor perturbations. We adopt two other methods to generate
adversarial perturbations: PA [69] adopts an LLM to generate perturbations, and RA [45] randomly
selects the items from the item pool as the perturbations.
As shown in Table 3, we can observe that the proposed defense method significantly reduces the
effectiveness of various attack methods (i.e., CheatAgent, PA). This implies that even if users
interact with clickbait items that trigger vulnerabilities in the recommendation system, the proposed
RETURN method can effectively cleanse these malicious disturbances, ensuring the correctness of
recommendations. Regarding RA, its attack capability is constrained, and it is aimed at simulating
scenarios where perturbation items do not cause the RecSys to misinterpret user preferences. In
this case, RETURN still improves or maintains the recommendation performance of the RecSys.
This demonstrates the robustness of the proposed RETURN against different attack intensities and
scenarios.
5.3.2 Ablation Study
Three variants RETURN-ROP ,RETURN-RR , and RETURN-w/o Ens are employed for com-
parison: 1) RETURN-ROP randomly creates the collaborative item graphs to demonstrate the
effectiveness and importance of introducing the external database. 2) RETURN-RR directly deletes
14

Table 4: Comparison between RETURN and its variants on three datasets
Datasets Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
ML1MBenign 0.2116 0.3055 0.1436 0.1737 / / / /
CheatAgent 0.0646 0.1171 0.0405 0.0573 0.6948 0.6168 0.7181 0.6699 0.0000 0.0000 0.0000 0.0000
RETURN 0.1384 0.2091 0.0915 0.1142 0.3459 0.3154 0.3630 0.3427 0.5023 0.4886 0.4945 0.4885
–ROP 0.0747 0.1286 0.0467 0.0639 0.6471 0.5789 0.6747 0.6321 0.0687 0.0615 0.0604 0.0564
–RR 0.1093 0.1705 0.0701 0.0898 0.4836 0.4417 0.5119 0.4831 0.3041 0.2838 0.2872 0.2788
–w/o Ens 0.1185 0.1889 0.0783 0.1010 0.4397 0.3816 0.4546 0.4184 0.3671 0.3814 0.3670 0.3754
LastFMBenign 0.0404 0.0606 0.0265 0.0331 / / / /
CheatAgent 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
RETURN 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
–ROP 0.0165 0.0321 0.0115 0.0164 0.5909 0.4697 0.5683 0.5044 0.1034 0.2250 0.1667 0.2205
–RR 0.0248 0.0339 0.0147 0.0176 0.3864 0.4394 0.4466 0.4674 0.4138 0.2750 0.3452 0.2778
–w/o Ens 0.0248 0.0367 0.0151 0.0188 0.3864 0.3939 0.4304 0.4309 0.4138 0.3500 0.3689 0.3342
TaobaoBenign 0.1420 0.1704 0.1100 0.1191 / / / /
CheatAgent 0.0863 0.1099 0.0615 0.0690 0.3922 0.3548 0.4409 0.4207 0.0000 0.0000 0.0000 0.0000
RETURN 0.1124 0.1384 0.0890 0.0975 0.2088 0.1875 0.1904 0.1817 0.4676 0.4715 0.5682 0.5680
–ROP 0.1008 0.1222 0.0749 0.0819 0.2907 0.2827 0.3185 0.3126 0.2588 0.2033 0.2776 0.2570
–RR 0.1122 0.1376 0.0882 0.0964 0.2099 0.1923 0.1981 0.1911 0.4647 0.4580 0.5508 0.5457
–w/o Ens 0.1006 0.1250 0.0765 0.0843 0.2918 0.2663 0.3046 0.2925 0.2559 0.2493 0.3093 0.3048
5 10 15 20 250.000.010.020.030.040.05Recommendation
H@5
H@10
N@5
N@10
(a) H@kand N @kw.r.t.m
5 10 15 20 25203040506070Defense (%)
D-H@5
D-H@10
D-N@5
D-N@10 (b) D-H @kand D-N @kw.r.t.m
Figure 4: Effect of the hyper-parameters m.
all items with low occurrence probabilities. 3) RETURN-w/o Ens generates recommendations
without using the ensemble strategy and only creates one purified prompt by processing a fixed
number of items. The results are summarised in Table 4. RETURN-ROP generates recommendations
without constructing collaborative item graphs from the external database, resulting in a significant
decrease in its defense performance. This highlights the importance of introducing accurate collabo-
rative knowledge from the external database. Since directly deleting all items with low occurrence
probabilities may result in the RecSys failing to capture users’ preferences effectively, especially
for users with limited interactions, there is a significant decrease in the defense performance of
RETURN-RR, illustrating the importance of employing the retrieval-augmented denoising strategy.
Since the number of the perturbations is unknown, RETURN-w/o Ens fixes the number of purification
items. This approach usually leads to information loss if an excessive number of items are deleted, or
incomplete purification if not all perturbations are eliminated, demonstrating the importance of the
robust ensemble recommendation strategy.
5.3.3 Parameter Analysis
We investigate the sensitivity of RETURN to the hyperparameter m. We sample varying values for
mand test the defense performance of the proposed method. and the results are illustrated in Figure 4.
We observe that as mincreases, the recommendation performance and the defense capability of
RETURN fluctuate within a small range, demonstrating the robustness of the proposed method to
hyperparameters.
5.3.4 The Robustness to the Perturbation Intensity
In this subsection, we investigate the robustness of RETURN to the perturbation intensity △. We
insert varying numbers of perturbations into benign users and evaluate the defense performance
of the proposed method. As shown in Table 5, the proposed method significantly enhances the
recommendation performance of LLM-empowered RecSys regardless of the number of perturbations
inserted into the input. This is attributed to robust recommendation generation strategies that avoid
introducing fixed thresholds, thereby improving the robustness of the proposed RETURN to the
number of perturbations.
15

Table 5: The defense performance of RETURN with respect to the perturbation intensity △
Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
Benign 0.0404 0.0606 0.0265 0.0331 / / / / / / / /
△=1 0.0183 0.0394 0.0122 0.0190 0.5455 0.3485 0.5390 0.4265 0.0000 0.0000 0.0000 0.0000
RETURN 0.0303 0.0486 0.0208 0.0265 0.2500 0.1970 0.2149 0.1974 0.5417 0.4348 0.6013 0.5373
△=2 0.0138 0.0257 0.0090 0.0128 0.6591 0.5758 0.6621 0.6120 0.0000 0.0000 0.0000 0.0000
RETURN 0.0248 0.0367 0.0148 0.0185 0.3864 0.3939 0.4432 0.4406 0.4138 0.3158 0.3307 0.2801
△=3 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
RETURN 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
△=4 0.0119 0.0202 0.0082 0.0109 0.7045 0.6667 0.6893 0.6704 0.0000 0.0000 0.0000 0.0000
RETURN 0.0248 0.0349 0.0162 0.0194 0.3864 0.4242 0.3879 0.4146 0.4516 0.3636 0.4372 0.3816
△=5 0.0119 0.0211 0.0068 0.0098 0.7045 0.6515 0.7425 0.7048 0.0000 0.0000 0.0000 0.0000
RETURN 0.0211 0.0284 0.0149 0.0173 0.4773 0.5303 0.4385 0.4771 0.3226 0.1860 0.4094 0.3231
5.3.5 Impact on Benign Users
It is crucial that defense algorithms should not affect the recommendation performance of RecSys
for users whose interaction histories contain no perturbations. Therefore, in this subsection, the
impact of RETURN on benign users is investigated, and the results are shown in Table 6. We can
observe that if the users’ profiles consist of no perturbations, RETURN can almost maintain the
recommendation performance even though RETURN deletes or replaces some items. Note that
the deletion or replacement operations are implemented based on the collaborative co-occurrence
frequency, indicating that the selected items usually fail to align with the users’ preferences. Therefore,
RETURN has little impact on the recommendation effectiveness for benign users, which demonstrates
its practical applicability in enhancing the robustness of LLM-empowered RecSys.
Table 6: Recommendation performance when users’ profiles contain no perturbation
Indexing Datasets Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑
SequentialML1MBenign 0.2116 0.3055 0.1436 0.1737
RETURN 0.1675 0.2498 0.1131 0.1397
LastFMBenign 0.0404 0.0606 0.0265 0.0331
RETURN 0.0376 0.0569 0.0232 0.0293
TaobaoBenign 0.1420 0.1704 0.1100 0.1191
RETURN 0.1006 0.1250 0.0765 0.0843
RandomML1MBenign 0.1058 0.1533 0.0693 0.0847
RETURN 0.0944 0.1406 0.0611 0.0760
LastFMBenign 0.0128 0.0248 0.0072 0.0110
RETURN 0.0156 0.0284 0.0094 0.0136
TaobaoBenign 0.1643 0.1804 0.1277 0.1330
RETURN 0.1239 0.1409 0.0893 0.0948
5.3.6 Time Complexity
To address the concern regarding the computational overhead introduced by the RETURN framework,
we conduct additional experiments to analyse the time complexity of RETURN. We measure the
average time taken by the LLM-empowered RecSys to generate recommendations after incorporating
different defense methods on the LastFM dataset. As shown in Tabel 7, we can observe that
methods requiring minimal computational resources (e.g., RD, LLMSI, etc.) exhibit significantly
shorter recommendation generation times, typically less than 0.5 seconds. However, their defense
performance is notably limited. In contrast, more powerful methods, including RETURN, exhibit
slightly longer recommendation generation times, with RETURN taking approximately 0.8599
seconds. This is comparable to other advanced defense methods like PD (0.7314 seconds) and RDE
(0.7222 seconds), which also take around 1 second.
The results indicate that while RETURN introduces additional computational steps, such as voting
operations, it does not significantly increase the overall computational burden of the RecSys. Im-
portantly, RETURN achieves this while substantially enhancing the robustness of RecSys against
perturbations. Thus, the framework strikes a balance between computational efficiency and defense
effectiveness, making it a practical choice for real-world applications.
16

Table 7: Computational time of different methods
Methods PD RPD RTD RD LLMSI ICL RDE RETURN
Time (s) 0.7314 1.2271 0.2916 0.3036 0.3006 0.3130 0.7222 0.8599
5.3.7 Impact of Poor Quality Data
We conduct additional experiments to investigate the impact of data quality. We introduce two
variants: RETURN-A-k andRETURN-D-k , where perturbations are injected into or items are
deleted from the historical interactions of users in the external database to generate collaborative item
graphs. Here, k=0.15 and k=0.3 represent the proportion of perturbations or deletions, respectively.
The results are shown in Table 8. The results demonstrate that RETURN-A-k still achieves remarkable
defense performance even when perturbations are introduced into the external database. This is
because the collaborative item graphs store co-occurrence frequencies, and minor perturbations
do not significantly alter the overall co-occurrence distribution among items. After normalization,
these perturbations have minimal impact on RETURN’s ability to cleanse user interaction data and
generate accurate recommendations. Additionally, RETURN-D-k fails to achieve the desired defense
performance because the lack of sufficient collaborative signals prevents it from accurately capturing
relationships between items, thereby hindering its ability to identify perturbations.
These experimental results indicate that the presence of noisy data in the external database (i.e.,
low-quality data) does not significantly deteriorate the performance of RETURN, as the co-occurrence
distribution remains relatively stable. However, insufficient data (e.g., due to deletions) can degrade
RETURN’s defense effectiveness, as it relies on sufficient collaborative signals to accurately model
item relationships. Therefore, while RETURN is robust to minor data quality issues, ensuring an
adequate volume of data is crucial for maintaining its performance.
Table 8: The defense performance of RETURN with respect to different external databases
Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
Benign 0.0404 0.0606 0.0265 0.0331 / / / / / / / /
CheatAgent 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
PD 0.0183 0.0330 0.0124 0.0170 0.5455 0.4545 0.5331 0.4851 0.1724 0.2500 0.2183 0.2503
RPD 0.0183 0.0312 0.0106 0.0147 0.5455 0.4848 0.6006 0.5556 0.1724 0.2000 0.1193 0.1413
RTD 0.0046 0.0110 0.0024 0.0043 0.8864 0.8182 0.9108 0.8691 -0.3448 -0.3500 -0.3355 -0.3430
RD 0.0220 0.0303 0.0139 0.0165 0.4545 0.5000 0.4743 0.5010 0.3103 0.1750 0.3045 0.2258
LLMSI 0.0119 0.0229 0.0080 0.0116 0.7045 0.6212 0.7003 0.6491 -0.0690 -0.0250 -0.0269 -0.0031
ICL 0.0174 0.0321 0.0113 0.0160 0.5682 0.4697 0.5726 0.5172 0.1379 0.2250 0.1604 0.2007
RDE 0.0220 0.0339 0.0128 0.0167 0.4545 0.4394 0.5170 0.4960 0.3103 0.2750 0.2420 0.2334
RETURN 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
RETURN - A - 0.15 0.0294 0.0413 0.0180 0.0219 0.2727 0.3182 0.3208 0.3391 0.5862 0.4750 0.5296 0.4760
RETURN - A - 0.3 0.0294 0.0440 0.0185 0.0232 0.2727 0.2727 0.3020 0.2980 0.5862 0.5500 0.5571 0.5395
RETURN - D - 0.15 0.0165 0.0321 0.0103 0.0152 0.5909 0.4697 0.6125 0.5409 0.1034 0.2250 0.1019 0.1642
RETURN - D - 0.3 0.0165 0.0321 0.0103 0.0152 0.5909 0.4697 0.6125 0.5409 0.1034 0.2250 0.1019 0.1642
5.3.8 The Adoption of Normal Distribution
During the robust ensemble recommendation process, RETURN randomly samples an integer n
from a normal distribution, and Top- nitems with the lowest occurrence probabilities are identified
from the user’s historical interactions for purification. The normal distribution is chosen because it
allows for better control over the strength of perturbation filtering in RETURN. If a majority of users’
interaction histories contain significant perturbations, making it difficult for RecSys to accurately
capture their preferences, the mean can be adjusted to enhance the purification strength of RETURN.
During experiments, the mean and the variance are 3.5 and 0.5, respectively. Moreover, we conducted
additional experiments to demonstrate that RETURN is robust to the different values of mean
and variance. The results are shown in Table 9. The performance of RETURN fluctuates within
a reasonable range as the mean and variance change, demonstrating its robustness to different
parameter settings. This indicates that RETURN can adapt to varying distributions while maintaining
its effectiveness in generating accurate recommendations.
5.3.9 Impact on the Personalization of Recommendations
To evaluate the impact of RETURN on personalized recommendations, we separately analyze the
recommendation results of the LLM-empowered RecSys for benign users and the results after
17

Table 9: The defense performance of RETURN with respect to different values of mean and variance
Methods H@5 ↑H@10 ↑N@5 ↑N@10 ↑A-H@5 ↓A-H@10 ↓A-N@5 ↓A-N@10 ↓D-H@5 ↑D-H@10 ↑D-N@5 ↑D-N@10 ↑
Benign 0.0404 0.0606 0.0265 0.0331 / / / /
CheatAgent 0.0138 0.0239 0.0084 0.0117 0.6591 0.6061 0.6820 0.6471 0.0000 0.0000 0.0000 0.0000
PD 0.0183 0.0330 0.0124 0.0170 0.5455 0.4545 0.5331 0.4851 0.1724 0.2500 0.2183 0.2503
RPD 0.0183 0.0312 0.0106 0.0147 0.5455 0.4848 0.6006 0.5556 0.1724 0.2000 0.1193 0.1413
RTD 0.0046 0.0110 0.0024 0.0043 0.8864 0.8182 0.9108 0.8691 -0.3448 -0.3500 -0.3355 -0.3430
RD 0.0220 0.0303 0.0139 0.0165 0.4545 0.5000 0.4743 0.5010 0.3103 0.1750 0.3045 0.2258
LLMSI 0.0119 0.0229 0.0080 0.0116 0.7045 0.6212 0.7003 0.6491 -0.0690 -0.0250 -0.0269 -0.0031
ICL 0.0174 0.0321 0.0113 0.0160 0.5682 0.4697 0.5726 0.5172 0.1379 0.2250 0.1604 0.2007
RDE 0.0220 0.0339 0.0128 0.0167 0.4545 0.4394 0.5170 0.4960 0.3103 0.2750 0.2420 0.2334
N(3.5, 0.5) 0.0266 0.0385 0.0169 0.0207 0.3409 0.3636 0.3613 0.3731 0.4828 0.4000 0.4703 0.4234
N(3, 0.5) 0.0229 0.0358 0.0132 0.0173 0.4318 0.4091 0.5035 0.4766 0.3448 0.3250 0.2618 0.2635
N(4, 0.5) 0.0267 0.0413 0.0174 0.0224 0.3389 0.3182 0.3449 0.3236 0.4859 0.4750 0.4943 0.4999
N(3.5, 1.0) 0.0229 0.0376 0.0150 0.0197 0.4318 0.3788 0.4362 0.4043 0.3448 0.3750 0.3604 0.3752
N(3.5, 1.5) 0.0220 0.0367 0.0150 0.0198 0.4545 0.3939 0.4349 0.4025 0.3103 0.3500 0.3624 0.3780
introducing RETURN for denoising. We calculate the frequency of different items in both sets of
results, computed the Jaccard similarity coefficient [ 2] between the two distributions, determined the
proportion of items that co-occurred, and measured the Shannon entropy of each distribution. The
results are presented in Table 10. Some observations can be obtained as follows:
•Jaccard Similarity (0.7605): The high Jaccard similarity coefficient indicates that the recommen-
dation results before and after applying RETURN are highly consistent for benign users. This
suggests that RETURN preserves the majority of the original recommendations, although some
items are removed or replaced.
•Common Items Ratio (0.8706): The proportion of items that co-occur in both the benign and
RETURN-processed recommendations is 87.06%. This further demonstrates that RETURN
maintains the core set of recommended items, ensuring minimal disruption to the personalized
recommendations.
•Shannon Entropy: The Shannon entropy values for both the benign (9.8616) and RETURN-
processed (9.8292) recommendations are nearly identical. This indicates that RETURN does not
significantly reduce the diversity of the recommendations, preserving the richness and variety of
the suggested items.
Table 10: Impact of RETURN on the personalization of recommendations
Shannon Entropy Jaccard Similarity Common Items Ratio
Benign 9.86160.7605 0.8706RETURN 9.8292
6 Conclusion
In this paper, we propose a novel framework RETURN by retrieving collaborative knowledge from
external databases to enhance the robustness of existing LLM-empowered RecSys in a plug-and-play
manner. Specifically, the proposed RETURN first converts the user interactions within external
databases into collaborative item graphs to implicitly encode the collaborative signals. Then, the
potential perturbations are located by retrieving relevant knowledge from the generated graphs. To
mitigate the negative impact of perturbations and maintain the integrity of user preference, a retrieval-
augmented denoising strategy is introduced to purify the input user profile. Finally, a robust ensemble
recommendation method is proposed to generate the final recommendations by adopting a decision
fusion strategy. Comprehensive experiments on real-world datasets demonstrate the effectiveness of
the proposed RETURN and highlight the potential of introducing external collaborative knowledge
to enhance the robustness of LLM-empowered RecSys.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774 , 2023.
18

[2]Sujoy Bag, Sri Krishna Kumar, and Manoj Kumar Tiwari. An efficient recommendation
generation using relevant jaccard similarity. Information Sciences , 483:53–64, 2019.
[3]Tao Bai, Jinqi Luo, Jun Zhao, Bihan Wen, and Qian Wang. Recent advances in adversarial train-
ing for adversarial robustness. In Proceedings of the Thirtieth International Joint Conference
on Artificial Intelligence , pages 4312–4321, 2021.
[4]Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. TALLRec:
An effective and efficient tuning framework to align large language model with recommendation.
InACM Conference on Recommender Systems , 2023.
[5]Qingpeng Cai, Zhenghai Xue, Chi Zhang, Wanqi Xue, Shuchang Liu, Ruohan Zhan, Xueliang
Wang, Tianyou Zuo, Wentao Xie, Dong Zheng, et al. Two-stage constrained actor-critic for
short video recommendation. In Proceedings of the ACM Web Conference 2023 , pages 865–875,
2023.
[6] Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. Defending against alignment-breaking
attacks via robustly aligned LLM. In Findings of the Association for Computational Linguistics
ACL 2024 , 2024.
[7]Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, and Yihua
Huang. Knowledge-enhanced black-box attacks for recommendations. In Proceedings of the
28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 108–117,
2022.
[8]Luoxin Chen, Weitong Ruan, Xinyue Liu, and Jianhua Lu. SeqV AT: Virtual adversarial training
for semi-supervised sequence labeling. In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics , pages 8801–8811, 2020.
[9]Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for Youtube recommenda-
tions. In Proceedings of the 10th ACM conference on recommender systems , pages 191–198,
2016.
[10] James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi,
Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, et al. The youtube video recommendation
system. In Proceedings of the fourth ACM conference on Recommender systems , pages 293–296,
2010.
[11] Yingpeng Du, Di Luo, Rui Yan, Xiaopei Wang, Hongzhi Liu, Hengshu Zhu, Yang Song, and Jie
Zhang. Enhancing job recommendation through llm-based generative adversarial networks. In
Proceedings of the AAAI Conference on Artificial Intelligence , pages 8363–8371, 2024.
[12] Wenqi Fan, Yao Ma, Dawei Yin, Jianping Wang, Jiliang Tang, and Qing Li. Deep social
collaborative filtering. In Proceedings of the 13th ACM Conference on Recommender Systems ,
pages 305–313, 2019.
[13] Wenqi Fan, Yao Ma, Qing Li, Jianping Wang, Guoyong Cai, Jiliang Tang, and Dawei Yin. A
graph neural network framework for social recommendations. IEEE Transactions on Knowledge
and Data Engineering , 2020.
[14] Wenqi Fan, Tyler Derr, Xiangyu Zhao, Yao Ma, Hui Liu, Jianping Wang, Jiliang Tang, and
Qing Li. Attacking black-box recommendations via copying cross-domain user profiles. In
2021 IEEE 37th International Conference on Data Engineering , 2021.
[15] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on RAG meeting LLMs: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , 2024.
[16] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023.
19

[17] Jin Ge, Steve Sun, Joseph Owens, Victor Galvez, Oksana Gologorskaya, Jennifer C Lai, Mark J
Pletcher, and Ki Lai. Development of a liver disease-specific large language model chat interface
using retrieval augmented generation. Hepatology , 2024.
[18] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. Recommendation
as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5).
InProceedings of the 16th ACM Conference on Recommender Systems , pages 299–315, 2022.
[19] Ihsan Gunes, Cihan Kaleli, Alper Bilge, and Huseyin Polat. Shilling attacks against recom-
mender systems: a comprehensive survey. Artificial Intelligence Review , 42:767–799, 2014.
[20] F Maxwell Harper and Joseph A Konstan. The MovieLens datasets: History and context. ACM
Transactions on Interactive Intelligent Systems , 2015.
[21] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgcn:
Simplifying and powering graph convolution network for recommendation. In Proceedings of
the 43rd International ACM SIGIR conference on research and development in Information
Retrieval , pages 639–648, 2020.
[22] Alec Helbling, Mansi Phute, Matthew Hull, and Duen Horng Chau. LLM self defense: By self
examination, LLMs know they are being tricked. arXiv preprint arXiv:2308.07308 , 2023.
[23] Neel Jain, Schwarzschild Avi, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh
Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. Baseline de-
fenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614 ,
2023.
[24] Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing
Lu, Zhengyang Wang, Ruirui Li, et al. Amazon-M2: A multilingual multi-locale shopping
session dataset for recommendation and text generation. Advances in Neural Information
Processing Systems , 2024.
[25] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. BERT: Pre-training of
deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT ,
pages 4171–4186, 2019.
[26] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong,
Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein. On the reliability of
watermarks for large language models. In The Twelfth International Conference on Learning
Representations , 2023.
[27] Louisa Lam and SY Suen. Application of majority voting to pattern recognition: an analysis of
its behavior and performance. IEEE Transactions on Systems, Man, and Cybernetics-Part A:
Systems and Humans , 27(5):553–568, 1997.
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing
Systems , 2020.
[29] Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. A survey on retrieval-augmented
text generation. arXiv preprint arXiv:2202.01110 , 2022.
[30] Linyang Li and Xipeng Qiu. Token-aware virtual adversarial training in natural language
understanding. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 8410–
8418, 2021.
[31] Linyang Li, Demin Song, and Xipeng Qiu. Text adversarial purification as defense against adver-
sarial attacks. In Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics , 2023.
[32] Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du, Mingkui Tan, Jun Huang, and Wei Lin.
Alphafin: Benchmarking financial analysis with retrieval-augmented stock-chain framework.
arXiv preprint arXiv:2403.12582 , 2024.
20

[33] Dawen Liang, Jaan Altosaar, Laurent Charlin, and David M Blei. Factorization meets the item
embedding: Regularizing matrix factorization with item co-occurrence. In Proceedings of the
10th ACM conference on recommender systems , pages 59–66, 2016.
[34] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan
He. LLaRA: Large language-recommendation assistant. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval , pages 1785–
1795, 2024.
[35] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu, Chuhan Wu,
Xiangyang Li, Chenxu Zhu, et al. How can recommender systems benefit from large language
models: A survey. ACM Transactions on Information Systems , 43(2):1–47, 2025.
[36] Yujie Lin, Chenyang Wang, Zhumin Chen, Zhaochun Ren, Xin Xin, Qiang Yan, Maarten
de Rijke, Xiuzhen Cheng, and Pengjie Ren. A self-correcting sequential recommender. In
Proceedings of the ACM Web Conference 2023 , pages 1283–1293, 2023.
[37] Greg Linden, Brent Smith, and Jeremy York. Amazon. com recommendations: Item-to-item
collaborative filtering. IEEE Internet computing , 7(1):76–80, 2003.
[38] Shang Liu, Zhenzhong Chen, Hongyi Liu, and Xinghai Hu. User-video co-attention network
for personalized micro-video recommendation. In The world wide web conference , pages
3020–3026, 2019.
[39] Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, and Jianfeng
Gao. Adversarial training for large neural language models. arXiv preprint arXiv:2004.08994 ,
2020.
[40] Yi Luo, Liang Rebecca Tang, Eojina Kim, and Xi Wang. Finding the reviews on yelp that
actually matter to me: Innovative approach of improving recommender systems. International
Journal of Hospitality Management , 91:102697, 2020.
[41] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong
Liu, Tong Xu, and Enhong Chen. Crud-rag: A comprehensive chinese benchmark for retrieval-
augmented generation of large language models. ACM Transactions on Information Systems ,
43(2):1–32, 2025.
[42] Julian McAuley, Rahul Pandey, and Jure Leskovec. Inferring networks of substitutable and
complementary products. In Proceedings of the 21th ACM SIGKDD international conference
on knowledge discovery and data mining , pages 785–794, 2015.
[43] Wenjie Mo, Jiashu Xu, Qin Liu, Jiongxiao Wang, Jun Yan, Chaowei Xiao, and Muhao Chen.
Test-time backdoor mitigation for black-box large language models with defensive demonstra-
tions. arXiv preprint arXiv:2311.09763 , 2023.
[44] Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-
yong Wei, Shanru Lin, Hui Liu, Philip S. Yu, and Qing Li. A survey of webagents: Towards
next-generation ai agents for web automation with large foundation models. arXiv preprint
arXiv:2503.23350 , 2025.
[45] Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, and Feiran Huang.
CheatAgent: Attacking llm-empowered recommender systems via llm agent. In ACM SIGKDD
Conference on Knowledge Discovery and Data Mining , pages 2284–2295, 2024.
[46] Apurva Pathak, Kshitiz Gupta, and Julian McAuley. Generating and personalizing bundle
recommendations on steam. In Proceedings of the 40th international ACM SIGIR conference
on research and development in information retrieval , pages 1073–1076, 2017.
[47] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli,
Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb
dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural
Information Processing Systems , 2024.
21

[48] Andreas Pfadler, Huan Zhao, Jizhe Wang, Lifeng Wang, Pipei Huang, and Dik Lun Lee. Billion-
scale recommendation with heterogeneous side information at taobao. In 2020 IEEE 36th
International Conference on Data Engineering (ICDE) , pages 1667–1676. IEEE, 2020.
[49] Ivan Provilkov, Dmitrii Emelianenko, and Elena V oita. BPE-dropout: Simple and effective
subword regularization. In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics , pages 1882–1892, Online, 2020. Association for Computational
Linguistics.
[50] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. TokenRec: Learning to tokenize id for
llm-based generative recommendation. arXiv preprint arXiv:2406.10450 , 2024.
[51] Haohao Qu, Liangbo Ning, Rui An, Wenqi Fan, Tyler Derr, Xin Xu, and Qing Li. A survey of
mamba. arXiv preprint arXiv:2408.01129 , 2024.
[52] Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation
models. arXiv preprint arXiv:2309.05922 , 2023.
[53] Yankun Ren, Zhongde Chen, Xinxing Yang, Longfei Li, Cong Jiang, Lei Cheng, Bo Zhang,
Linjian Mo, and Jun Zhou. Enhancing sequential recommenders with augmented knowledge
from aligned large language models. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval , 2024.
[54] Mingdan Si and Qingshan Li. Shilling attacks against collaborative recommender systems: a
review. Artificial Intelligence Review , 53:291–319, 2020.
[55] Changxin Tian, Yuexiang Xie, Yaliang Li, Nan Yang, and Wayne Xin Zhao. Learning to denoise
unreliable interactions for graph collaborative filtering. In Proceedings of the 45th international
ACM SIGIR conference on research and development in information retrieval , pages 122–132,
2022.
[56] Neeraj Varshney, Pavel Dolin, Agastya Seth, and Chitta Baral. The art of defending: A
systematic evaluation and analysis of LLM defense strategies on safety and over-defensiveness.
InFindings of the Association for Computational Linguistics ACL 2024 , 2024.
[57] Maxim Kuznetsov Vladimir V orobev. A paraphrasing model based on chatgpt paraphrases.
2023.
[58] Bohao Wang, Feng Liu, Changwang Zhang, Jiawei Chen, Yudi Wu, Sheng Zhou, Xingyu Lou,
Jun Wang, Yan Feng, Chun Chen, et al. Llm4dsr: Leveraing large language model for denoising
sequential recommendation. arXiv preprint arXiv:2408.08208 , 2024.
[59] Dilin Wang, Chengyue Gong, and Qiang Liu. Improving neural language modeling via adver-
sarial training. In International Conference on Machine Learning , pages 6555–6565. PMLR,
2019.
[60] Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. Denoising implicit
feedback for recommendation. In Proceedings of the 14th ACM international conference on
web search and data mining , pages 373–381, 2021.
[61] Wenjie Wang, Fuli Feng, Xiangnan He, Hanwang Zhang, and Tat-Seng Chua. Clicks can be
cheating: Counterfactual recommendation for mitigating clickbait issue. In International ACM
SIGIR Conference on Research and Development in Information Retrieval , pages 1288–1297,
2021.
[62] Xin Wang, Hong Chen, Zirui Pan, Yuwei Zhou, Chaoyu Guan, Lifeng Sun, and Wenwu
Zhu. Automated disentangled sequential recommendation with large language models. ACM
Transactions on Information Systems , 43(2):1–29, 2025.
[63] Zhaoyang Wang, Zhiyue Liu, Xiaopeng Zheng, Qinliang Su, and Jiahai Wang. Rmlm: A flexible
defense framework for proactively mitigating word-level adversarial attacks. In Proceedings of
the 61st Annual Meeting of the Association for Computational Linguistics , pages 2757–2774,
2023.
22

[64] Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, Xuanping Li, and Tat-Seng Chua. Con-
trastive learning for cold-start recommendation. In Proceedings of the 29th ACM International
Conference on Multimedia , pages 5382–5390, 2021.
[65] Zeming Wei, Yifei Wang, and Yisen Wang. Jailbreak and guard aligned language models with
only few in-context demonstrations. arXiv preprint arXiv:2310.06387 , 2023.
[66] Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco
Guzmán, Armand Joulin, and Édouard Grave. Ccnet: Extracting high quality monolingual
datasets from web crawl data. In Proceedings of the Twelfth Language Resources and Evaluation
Conference , pages 4003–4012, 2020.
[67] Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, and Leo Schwinn.
Efficient adversarial training in llms with continuous attacks. arXiv preprint arXiv:2405.15589 ,
2024.
[68] Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. OpenP5: An open-source platform for
developing, training, and evaluating llm-based recommender systems. SIGIR , 2024.
[69] Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, and Mohan Kankanhalli.
An LLM can fool itself: A prompt-based adversarial attack. In The Twelfth International
Conference on Learning Representations .
[70] Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. A survey on large
language model (LLM) security and privacy: The good, the bad, and the ugly. High-Confidence
Computing , 2024.
[71] Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, and Leah Li. Financial report
chunking for effective retrieval augmented generation. arXiv preprint arXiv:2402.05131 , 2024.
[72] Yisong Yue, Rajan Patel, and Hein Roehrig. Beyond position bias: Examining result attrac-
tiveness as a source of presentation bias in clickthrough data. In Proceedings of the 19th
international conference on World wide web , pages 1011–1018, 2010.
[73] Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, and Xueqi Cheng. Lorec: Combating
poisons with large language model for robust sequential recommendation. In Proceedings of
the 47th International ACM SIGIR Conference on Research and Development in Information
Retrieval , pages 1733–1742, 2024.
[74] Liang Zhang, Guannan Liu, Xiaohui Liu, and Junjie Wu. Denoising item graph with disentangled
learning for recommendation. IEEE Transactions on Knowledge and Data Engineering , 2024.
[75] Shijie Zhang, Hongzhi Yin, Tong Chen, Quoc Viet Nguyen Hung, Zi Huang, and Lizhen
Cui. Gcn-based user representation learning for unifying robust recommendation and fraudster
detection. In Proceedings of the 43rd international ACM SIGIR conference on research and
development in information retrieval , pages 689–698, 2020.
[76] Zhen Zhang, Guanhua Zhang, Bairu Hou, Wenqi Fan, Qing Li, Sijia Liu, Yang Zhang, and
Shiyu Chang. Certified robustness for large language models with self-denoising. arXiv preprint
arXiv:2307.07171 , 2023.
[77] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei
Wang, Xiangyu Zhao, Jiliang Tang, et al. Recommender systems in the era of large language
models. IEEE Transactions on Knowledge and Data Engineering , 2024.
[78] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong
Wen. Adapting large language models by integrating collaborative semantics for recommen-
dation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE) , pages
1435–1448. IEEE, 2024.
[79] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. Learning
tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining , 2018.
23
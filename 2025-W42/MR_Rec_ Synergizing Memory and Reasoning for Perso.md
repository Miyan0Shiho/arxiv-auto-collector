# MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs

**Authors**: Jiani Huang, Xingchen Zou, Lianghao Xia, Qing Li

**Published**: 2025-10-16 12:40:48

**PDF URL**: [http://arxiv.org/pdf/2510.14629v1](http://arxiv.org/pdf/2510.14629v1)

## Abstract
The application of Large Language Models (LLMs) in recommender systems faces
key challenges in delivering deep personalization and intelligent reasoning,
especially for interactive scenarios. Current methods are often constrained by
limited context windows and single-turn reasoning, hindering their ability to
capture dynamic user preferences and proactively reason over recommendation
contexts. To address these limitations, we propose MR.Rec, a novel framework
that synergizes memory and reasoning for LLM-based recommendations. To achieve
personalization, we develop a comprehensive Retrieval-Augmented Generation
(RAG) system that efficiently indexes and retrieves relevant external memory to
enhance LLM personalization capabilities. Furthermore, to enable the synergy
between memory and reasoning, our RAG system goes beyond conventional
query-based retrieval by integrating reasoning enhanced memory retrieval.
Finally, we design a reinforcement learning framework that trains the LLM to
autonomously learn effective strategies for both memory utilization and
reasoning refinement. By combining dynamic memory retrieval with adaptive
reasoning, this approach ensures more accurate, context-aware, and highly
personalized recommendations. Extensive experiments demonstrate that MR.Rec
significantly outperforms state-of-the-art baselines across multiple metrics,
validating its efficacy in delivering intelligent and personalized
recommendations. We will release code and data upon paper notification.

## Full Text


<!-- PDF content starts -->

MR.Rec: Synergizing M emory and R easoning for Personalized
Recommendation Assistant with LLMs
Jiani Huang*
jianihuang01@gmail.com
The Hong Kong Polytechnic University
Hong Kong SAR, ChinaXingchen Zou*
xzou428@connect.hkust-gz.edu.cn
The Hong Kong University of Science and Technology
(Guangzhou)
Guangzhou, China
Lianghao Xia†
aka_xia@foxmail.com
The University of Hong Kong
Hong Kong SAR, ChinaQing Li†
qing-prof.li@polyu.edu.hk
The Hong Kong Polytechnic University
Hong Kong SAR, China
Abstract
The application of Large Language Models (LLMs) in recommender
systems faces key challenges in delivering deep personalization and
intelligent reasoning, especially for interactive scenarios. Current
methods are often constrained by limited context windows and
single-turn reasoning, hindering their ability to capture dynamic
user preferences and proactively reason over recommendation con-
texts. To address these limitations, we propose MR.Rec, a novel
framework that synergizes memory and reasoning for LLM-based
recommendations. To achieve personalization, we develop a com-
prehensive Retrieval-Augmented Generation (RAG) system that
efficiently indexes and retrieves relevant external memory to en-
hance LLM personalization capabilities. Furthermore, to enable the
synergy between memory and reasoning, our RAG system goes be-
yond conventional query-based retrieval by integrating reasoning-
enhanced memory retrieval. Finally, we design a reinforcement
learning framework that trains the LLM to autonomously learn
effective strategies for both memory utilization and reasoning re-
finement. By combining dynamic memory retrieval with adaptive
reasoning, this approach ensures more accurate, context-aware,
and highly personalized recommendations. Extensive experiments
demonstrate that MR.Rec significantly outperforms state-of-the-art
baselines across multiple metrics, validating its efficacy in deliver-
ing intelligent and personalized recommendations. We will release
code and data upon paper notification.
1 Introduction
As digital content continues to grow, recommender systems (RSs)
have become widely adopted across web platforms to filter mas-
sive information spaces and provide users with personalized con-
tent [ 34]. With the rapid advancement of artificial intelligence, user
expectations have evolved toward more intelligent and personalized
recommendation assistants. Specifically, users now expect these
systems to understand natural language queries—similar to voice
assistants like Siri—and provide immediate, personalized recom-
mendations [ 7,10,38]. Traditional RSs, such as matrix factorization
(MF) [ 19] or graph neural network (GNN)–based methods [ 29],
primarily rely on implicit signals from interaction histories (e.g.,
*These authors contributed equally to this work.
†Corresponding author: Lianghao Xia, Qing Li.
Recommend a T-shirt for my sport class !User’s Interaction History and Reviews 
SportshoeWhite DressLipstickiPhoneCookieWhite shirtCanvas bagEarringsUser-specific Local Memory:1.The user likes white color, simple design.2.The user may be afemale.3.The user cares more about the product’s quality rather than price.4. Probably accustomed to  Apple ecosystem.
MR. RecMulti-turn InteractionMemory Indexing
MemoryUserGlobal Memory:for cloth scenario, users typically care about color, prize, style…Local Memory:the user is a girl who prefer…, she might dislike…, she cares…Based on user’s preference, query and my reasoning,  I would recommend:  a white T-shirt with simple design, woman style, cozy size for sport withoutsport icons, made with high-quality cottonand breathablefabric.ReasoningFigure 1: Illustration of Personalized and Intelligent LLM-
based Recommendation Assistant.
clicks, purchases). While effective in many contexts, they cannot
interpret natural language inputs, limiting their ability to function
as intelligent, conversational recommendation assistants.
The advent of large language models (LLMs) has opened new
possibilities for developing intelligent, interactive recommenda-
tion assistants in various domains [ 26,41]. With their advanced
language understanding capability, extensive general knowledge
and instruction-following ability [ 42], LLMs have the potential to
understand user natural language queries and serve as intelligent
recommendation assistants. Therefore, existing research has ex-
plored the integration of LLMs into recommendation tasks [ 4,8].
Despite recent advancements, challenges remain in achieving truly
personalized and intelligent recommendation assistants, primarily
due to the limited ability of LLMs to memorize user preferences
and reason effectively on recommendation tasks.
One of the primary obstacles lies inpersonalization, as the
assistant must be able to maintain a comprehensive user memory
that captures user profiles, evolving preferences, and contextual
histories [ 18,33]. This persistent memory allows the assistant to
anticipate user needs and deliver relevant recommendations seam-
lessly. For example, as illustrated in Figure 1, when a user asks for
T-shirt recommendations, the pre-modeled user memory enables
the assistant to understand the user’s past preferences (e.g., lik-
ing white color, simple design, and high quality) and inferred user
profile (e.g., potential gender), thereby facilitating highly personal-
ized and contextual recommendations that go beyond simple query
matching. Existing LLM-based RSs typically inject user personaliza-
tion information into LLMs either by prompting recent interaction
histories [ 17,25] or by providing pre-generated, static user pro-
files [ 24,43]. However, these methods are often constrained byarXiv:2510.14629v1  [cs.IR]  16 Oct 2025

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
the limited context window of LLMs, which prevents the assistant
from fully capturing a user’s profile and evolving preferences. In
addition, they may introduce irrelevant or noisy information that
is not aligned with the current query, thereby distracting the model
and reducing its ability to generate accurate recommendations.
While personalization is essential, it does not fully address the
challenge of building a capable recommendation assistant. A com-
plementary requirement isintelligence, namely the ability to ac-
tively explore the factors behind effective recommendations and
perform in-depth multi-step reasoning [5,35]. In the recommen-
dation domain, such reasoning is essential for interpreting user
queries, uncovering underlying intents, and generating accurate
suggestions. As illustrated in Figure 1, when a user requests a T-shirt
for sports, the recommendation assistant should reason that breath-
able fabric is essential for athletic activities, and then integrate this
reasoning with the user’s memory to generate a more intelligent
and contextually appropriate recommendation. Recent works have
begun to activate LLMs’ reasoning in recommendations [ 1,27]; for
instance, Tang et al. [ 22] enhance user representations through
implicit multi-step reasoning in sequential recommendation. Nev-
ertheless, these efforts generally constrain the LLM to reason only
over the limited information provided in the pre-defined prompt
template, without enabling it to actively explore which additional
user memories are helpful for resolving the current recommenda-
tion problem. This decoupled paradigm between reasoning and
memory restricts the assistant to shallow reasoning, limiting its
capacity to effectively leverage stored memory and perform coher-
ent, multi-step reasoning essential for generating personalized and
contextually grounded recommendations.
To address these challenges, we proposeMR.Rec, a model that
synergizesMemory andReasoning for LLM-basedRecommendation
assistants.First, we develop a comprehensive Retrieval-Augmented
Generation (RAG) system that efficiently indexes and retrieves
relevant external memory to enhance LLM personalization ca-
pabilities. Our RAG system employs a hierarchical memory in-
dexing mechanism that consists of User-specific Local Memory and
Cross-user Global Memory . User-specific Local Memory captures
user-specific information at multiple granularities, progressively
organizing interaction histories, preference patterns, and user pro-
files to comprehensively represent personalized preferences, while
Cross-user Global Memory aggregates cross-user knowledge to
provide broader experiential understanding for diverse recommen-
dation scenarios.Second, to enable the synergy between mem-
ory and reasoning, our RAG system goes beyond conventional
query-based retrieval by integrating reasoning-enhanced memory
retrieval. Rather than directly retrieving based on surface-level
query similarity, our approach first leverages reasoning to iden-
tify implicit user preferences and relevant preference dimensions,
then selectively retrieves memory segments accordingly. Further-
more, we extend LLM reasoning beyond simple chain-of-thought
to encompass multi-step memory retrieval stages, creating an adap-
tive framework where reasoning and information gathering are
interleaved to enable dynamic memory exploration and evidence-
grounded inference that significantly enhances reasoning depth
and context awareness.Finally, we design a reinforcement learn-
ing framework that trains the LLM to autonomously learn effectivestrategies for both memory utilization and reasoning refinement.
Through diverse reward mechanisms that guide memory retrieval,
reasoning quality, and recommendation accuracy, our end-to-end
optimization enables the model to develop adaptive behaviors that
optimize long-term recommendation performance while efficiently
leveraging the hierarchical memory structure.
In summary, our contributions can be summarized as follows:
•We propose a comprehensive RAG system with hierarchical mem-
ory indexing that efficiently captures and retrieves personalized
and cross-user knowledge, significantly extending external mem-
ory capabilities for LLM-based recommendations.
•We develop a reinforcement learning paradigm with novel re-
wards to harmonize memory access and recommendation reason-
ing, enabling the LLM to autonomously learn effective memory
utilization strategies for final recommendation.
•Extensive experiments demonstrate that our MR.Rec outper-
forms state-of-the-art baselines, validating its efficacy in deliver-
ing intelligent, personalized recommendations.
2 Preliminary
Memory-Enhanced Recommendation. Memory enables rec-
ommenders to retain user interaction histories and contextual in-
formation for personalized recommendations [ 30]. Given users
U={𝑢 1,𝑢2,...,𝑢|𝑈|}and itemsI={𝑖 1,𝑖2,...,𝑖|𝐼|}, each user𝑢
has interaction history H𝑢={(𝑖𝑡,𝑟𝑡)}𝑇𝑢
𝑡=1, where𝑖𝑡is the inter-
acted item and 𝑟𝑡is auxiliary signals (ratings, reviews). A memory
functionM(·)transforms this history into structured memory:
M𝑢=M(H𝑢),(1)
whereM𝑢encodes preferences as embeddings, textual summaries,
or hybrid structures for LLM retrieval during recommendation.
Reasoning in LLM-based Recommendation. Effective LLM-
based recommendation requires explicit reasoning to decompose
complex queries and adaptively utilize relevant memory. Rather
than direct prediction, LLMs should generate interpretable reason-
ing trajectoriesTthat systematically trace from query understand-
ing through memory analysis to final recommendation justification.
This reasoning-driven approach enables dynamic memory explo-
ration and provides transparent recommendation decisions.
Problem Statement. We develop a memory-enhanced recom-
mendation system that generates personalized recommendations
through explicit reasoning. Given user 𝑢with textual query 𝑞𝑢
and memoryM𝑢, our system generates an ideal item profile 𝑖∗via
reasoning-enhanced LLM:
𝑖∗=LLM(P(𝑞 𝑢,M𝑢);Θ),(2)
whereP(·) integrates query and memory into structured prompts,
andΘare learnable parameters optimized for reasoning quality.
We then retrieve top- 𝑘items from item pool Ibased on similarity:
𝑅𝑢=arg top-𝑘 𝑖∈Isim(𝑖∗,metadata(𝑖)).(3)
Our objective is to jointly optimize memory utilization and reason-
ing parametersΘto improve recommendation accuracy.
3 Methodology
This section elaborates technical details of our MR.Rec framework.
The overall framework is depicted in Figure 2.

MR.Rec: Synergizing M emory and R easoning for Personalized Rec ommendation Assistant with LLMs
(a) HierarchicalMemory Indexing(b) Unifying Memory and Reasoning for RecommendationCross-User Global Memory#BabyUsers typically care about material safety, brand recognition, durability, …
#SkinUsers may consider gender, ingredient, brand…
#ClothUsers will consider factors like style, price, budget, quality, material, and occasion.
According to shared experience from other users, when purchasing lotions, factors likegender, ingredient, brand, etc. should be considered.Let me find if there are any relevant preferencesor memories about the user.Personal Memory
PreferencePatternsShampooSun hatHiking bootsHair ConditionerDressBehavior RecordsPersonal careThe user prefers big brands  and natural ingredients…ActivitiesThe user may like outdoor sportslike hiking…User ProfileThe user may be a femalewith a strong interest in outdoor activities like hiking, emphasizing health and wellness...
(c) Reinforcement Learning forMemory-synergized Reasoning
Scenario Analysis
UserAnalysis
Interaction History & Reviews
Item Candidate PoolHi! I want to buy a lotion under $100. Any recommendations ?
Retrieved Memory:For gender, the user is likely a female;For ingredient, the user prefer natural and organic components in skincare items;For brand, the user often purchase well-known brands like Clarins, Kiehl’ s, etc. .There are still many possible candidates. I should consider user‘s lifestyleto further refine recommendation.
Retrieved Memory: For lifestyle, the user may be keen on outdoor activities such as hiking.Based on memories, the user likely wants a lotion with protective antioxidants or SPF and a quick-absorbing, non-greasy texture for outdoor activities like hiking. The ideal item is:sub-$100,    natural lotion from    brands like Kiehl‘s or Clarins that with protective antioxidants/SPF ina non-greasy, quick-absorbing formula.
User-specific Local Memory
UserRetrieveRetrieveRetrieve
Recommendations
MR.Rec
Query
LLM Policy ModelMemory…𝑇!Format RewardRecommendation RewardMemory Utilization RewardReward ModelAdvantageEstimationRollout ModuleAdvantagesRewards
UpdateMemoryLLM Reasoning
…
𝑇"
…
𝑇#Query
𝑇#𝑇"TrajectoryGroup…𝑅!𝑅#𝑅"…𝐴!𝐴#𝐴"QueryQuery
Figure 2: The overall model architecture of the proposed MR.Rec framework.
3.1 Memory Indexing for RAG Recommender
To enable LLMs to effectively and efficiently obtain external infor-
mation from vast user interaction records, we propose a retrieval-
augmented generation(RAG)-based External Memory Mech-
anismfor recommendation. This RAG module adopts aHierar-
chical Memory Indexingmethod. Specifically, effective memory
indexing for recommendation faces two key challenges. First, using
raw interaction histories can exceed context limits and cause overfit-
ting to recent behavior while introducing irrelevant noise. Second,
recommender systems need both fine-grained user-specific memo-
ries and comprehensive knowledge aggregated across users to pro-
vide robust recommendations. Therefore, we compress interaction
data into the following two complementary memory structures:
3.1.1User-specific Local Memory.User-specific memory in-
volves users’ purchase histories, ratings, and reviews [ 20]. This data
can easily exceed LLMs’ context window, but using users’ recent
interactions may overlook their long-term preferences and overfit
their short-term behaviors. This makes it infeasible to directly use
the raw interaction data. More essentially, raw interaction histories
are inherently noisy: not all past behaviors are equally relevant
to the current recommendation query. For instance, past grocery
purchases are irrelevant when recommending home appliances,
while electronics-related behaviors provide more reliable signals.
To address these challenges, we organize user preferences at
multiple granularities, enabling selective retrieval of relevant be-
havioral patterns while filtering noise, specifically including:
•Behavior Records: It preserves the original interaction data of
a user, such as purchased items, ratings, and reviews.
•Preference Patterns: Moving beyond raw behavioral records,
this part indexes preference patterns within item categories. We
partition interaction history into different domains (e.g., electron-
ics, clothing) and apply LLM to summarize compact preferencedescriptions. This filters out irrelevant cross-category signals
while preserving salient information. Formally, given user inter-
action history 𝐻𝑢, we partition it into subsets 𝐻𝑐
𝑢by category 𝑐
and use LLM to summarize category-specific user preference 𝑃𝑐
𝑢:
𝑃𝑐
𝑢=𝑓LLM(𝐻𝑐
𝑢).(4)
•User Profile: To model coherent preference structures across
domains, we construct a higher-level abstraction integrating frag-
mented category-specific patterns into consistent cross-category
preference structures. Formally, the user profile is expressed as:
𝑈𝑢=𝑓LLM {𝑃𝑐
𝑢:𝑐∈C𝑢},(5)
whereC𝑢denotes the set of categories associated with user 𝑢.
The resulting profile 𝑈𝑢serves as a compact, high-level summary
that complements fine-grained preference patterns.
Since preference patterns and user profiles are textual summaries,
we apply chunking to decompose them into smaller, semantically
coherent segments, enabling efficient storage and selective retrieval
of the most pertinent information during recommendation.
3.1.2Cross-User Global Memory.To capture shared behav-
ioral patterns from the entire dataset, MR.Rec constructs a global
memory that captures common decision-making dimensions across
users within similar recommendation scenarios, serving as a com-
plementary knowledge source. For example, in baby product recom-
mendations, such memory reveals critical domain-specific factors
like safety certifications, ergonomic design, and age-appropriate
features that users typically consider but rarely explicitly mention.
Concretely, for each recommendation scenario 𝑠(e.g., a category
in e-commerce), we sample a set of queries 𝑄𝑠={𝑞 1,𝑞2,...,𝑞𝑚}
from the training corpus, each paired with its ground-truth item
𝑖+∈I𝑠. To uncover latent dimensions of decision-making, we in-
corporate negative items I−
𝑠⊂I𝑠from the same scenario that were
not chosen. These alternatives offer valuable contrasts, enabling the
model to identify why the selected item was favored and to derive
more granular, adaptable decision criteria. Formally, we define the

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
global memory construction process as follows:
𝑀𝑔𝑙𝑜𝑏𝑎𝑙 =𝑓LLM {(𝑞,𝑖+,I−
𝑠)|𝑞∈𝑄𝑠},(6)
where𝑀𝑠denotes the global memory for scenario 𝑠, and𝑓LLM(·)
extracts organized aspects and rationales.
3.2 Unifying Memory and Reasoning for
Recommendation
3.2.1Reasoning-enhanced Memory Retrieval.With the in-
dexed recommendation memory, our RAG-based recommender
retrieves user interaction patterns relevant to the query to enhance
recommendation accuracy. A critical challenge in memory retrieval
for recommendations isimplicit user preferencesthat remain
unstated in queries. Different recommendation scenarios engage
diverse preference dimensions that users often omit unintentionally.
For instance, when requesting clothing recommendations, users
may have unstated considerations about material, style, or brand
reputation that are crucial for personalized suggestions. This chal-
lenge causes existing static retrieval methods to fail, as they rely
solely on surface-level query similarity without understanding the
deeper preference dimensions. Consequently, such approaches can
incorporate irrelevant data or miss critical information, limiting
the assistant’s ability to provide personalized recommendations.
To address these limitations, we propose reasoning-enhanced
memory retrieval, which integrates reasoning and memory access
in a dynamic, iterative process. Given a user query, the LLM first an-
alyzes the recommendation scenario to identify relevant preference
aspects that should guide personalization, leveraging global mem-
ory to understand typical evaluation criteria. The LLM then uses
these reasoning-inferred aspects to selectively retrieve segments
from user-specific local memory, enabling progressive refinement
of understanding. Formally, we identify preference aspects by:
A𝑞=𝑓LLM(𝑞,𝑀 global),(7)
where𝑓LLM(·)represents the reasoning process over the query and
global memory, andA 𝑞denotes the identified preference aspects.
We then use the identified aspects to selectively retrieve relevant
segments from the user-specific local memory, reducing noise and
enhancing personalization:
ˆ𝑀𝑢(𝑞)=𝑔 retrieval(A𝑞,𝑀local),(8)
where𝑔retrieval(·)denotes the LLM-based retrieval function condi-
tioned on the reasoning-inferred aspects, and ˆ𝑀𝑢(𝑞)represents the
retrieved local memory segments.
3.2.2Retrieval-Augmented Item Generation.Finally, the LLM
integrates the retrieved memories to generate an ideal item profile
that aligns with both the user’s query and personalized preferences:
I𝑢(𝑞)=𝑓 LLM 𝑞,A𝑞,ˆ𝑀𝑢(𝑞),(9)
whereI𝑢(𝑞)represents the ideal item profile that captures the
optimal item characteristics for the current query.
Through this reasoning-enhanced retrieval mechanism, the as-
sistant dynamically leverages both global and local memory to
produce contextually aware and highly personalized recommenda-
tions, treating memory as an active resource that evolves with the
reasoning process rather than static input.
3.2.3Enhancing Reasoning with Memory Retrieval.Except
that our memory retrieval method gains enhancement from LLMreasoning, we highlight that this method also represents an im-
portant enhancement over existing reasoning methods for recom-
mendation tasks. Effective reasoning is crucial for understanding
complex user preferences and making contextually appropriate
connections between user needs and available options. Current rec-
ommendation LLMs typically employ chain-of-thought reasoning
or prompt-based inference on static input, but these approaches
suffer from limited reasoning depth as they cannot dynamically
explore relevant information during the inference process, leading
to suboptimal personalization and reduced context awareness. Our
method enhances LLM reasoning by extending it to encompass
memory retrieval stages, creating an adaptive framework where
reasoning and information gathering are interleaved.
Specifically, the LLM iteratively analyzes the recommendation
scenario, identifies relevant preference dimensions, and retrieves
corresponding memory entries to refine its understanding. This en-
ables the LLM to treat memory as a dynamic resource aligned with
the evolving reasoning process, significantly enhancing reasoning
capabilities through multi-step, evidence-grounded inference that
adapts to task complexity. Importantly, this extended reasoning
mechanism can be jointly optimized through our reinforcement
learning framework, allowing the model to learn optimal strategies
for both memory exploration and reasoning refinement.
3.3 Reinforcement Learning for
Memory-synergized Reasoning
To enhance our memory-synergized reasoning, we fine-tune the
base LLM through reinforcement learning. Our MR.Rec requires
robust reasoning and effective memory exploration through multi-
turn interactions and iterative retrieval. However, comprehensive
annotation for such complex processes is prohibitively resource-
intensive, making supervised fine-tuning impractical.
To overcome this, we adopt a multi-turn reinforcement learn-
ing framework. This approach allows the LLM to actively explore
the recommendation and memory environment, refining its policy
through iterative interactions. By leveraging feedback signals in-
stead of static annotations, the model develops adaptive behaviors
that optimize long-term recommendation accuracy and efficient
memory utilization. Specifically, inspired by Group Relative Policy
Optimization (GRPO) [ 6], we adopt a reinforcement learning with
verifiable rewards framework that incorporates multi-turn feedback
signals into policy optimization. For a given query 𝑞, the assistant is
prompted multiple times to analyze the query, retrieve potentially
useful memory entries, and reason through these before generating
𝐺final candidate responses {𝑜1,𝑜2,...,𝑜𝐺}. A reward model then
assigns reward scores {𝑟1,𝑟2,...,𝑟𝐺}to each candidate response.
Based on these scores, relative advantages 𝐴(𝑜𝑖)are computed,
which indicate the comparative quality of different outputs and
guide subsequent policy optimization.
To guide the model toward the desired behavior, we design a set
of multi-faceted reward functions:
•Format Reward( 𝑅format ). To extract the desired answers from
the reasoning process, we require the LLM to output its final
answer in a prescribed format. The reward is assigned as 1 if the
format is correct, and 0 otherwise.
•Recommendation Reward( 𝑅rec). To quantify recommendation
quality, we define a reward 𝑅rec=nDCG @1000+nDCG @100

MR.Rec: Synergizing M emory and R easoning for Personalized Rec ommendation Assistant with LLMs
where nDCG@1000 mitigates reward sparsity, and nDCG@100
emphasizes fine-grained ranking among top candidates.
•Memory Utilization Reward( 𝑅mem). To encourage reasoning
grounded in memory, the LLM is required to perform a memory
retrieval step. The reward is defined as a binary indicator, where
the reward is 1 if the model successfully calls memory retrieval
during generation, and 0 otherwise.
The overall reward model is a weighted combination of the three
reward components, with weights𝑤 1,𝑤2,𝑤3:
𝑟=𝑤 1𝑅format+𝑤 2𝑅rec+𝑤 3𝑅mem.(10)
Finally, the optimization objective is formulated as a clipped
policy gradient similar to PPO:
J(𝜃)=E(𝑞,𝑔𝑡)∼D,{𝑜 𝑖}𝐺
𝑖=1∼𝜋𝜃old(·|𝑞)=
"
1
𝑁𝐺∑︁
𝑖=1|𝑜𝑖|∑︁
𝑡=1min𝜋𝜃(𝑜|𝑞)
𝜋𝜃old(𝑜|𝑞)𝐴𝑖,𝑡,clip𝜋𝜃(𝑜|𝑞)
𝜋𝜃old(𝑜|𝑞),1−𝜖,1+𝜖
𝐴𝑖,𝑡#
,
(11)
where𝐴𝑖,𝑡denotes the advantage of each token derived from rela-
tive reward signals, which equals 𝐴(𝑜𝑖)=𝑟𝑖−mean(r)
std(r). Importantly,
during optimization, we mask the retrieved memory tokens to en-
sure that the advantage estimation depends only on the assistant’s
reasoning and recommendation outputs.
4 Experiments
In this section, we evaluate the proposed MR.Rec framework to
answer the following research questions (RQs):
•RQ1:How does MR.Rec perform in providing personalized rec-
ommendation compared to baseline methods?
•RQ2:What is the contribution of each main component and
different types of memory to the overall performance of MR.Rec?
•RQ3:Does our RAG-based memory mechanism retrieve gen-
uinely beneficial information to address recommendation queries?
•RQ4:How is the efficiency of indexing and retrieval of our
MR.Rec?
•RQ5:How do different model settings impact the performance
of MR.Rec, including hyperparameter settings, LLM backbones,
and different retriever configurations?
4.1 Experimental Setup
4.1.1Datasets.We construct our dataset based on the Amazon-
C4 [9] dataset, which provides user queries generated by ChatGPT
from product reviews. These review-based queries are often highly
detailed, implicitly covering nearly all aspects of a user’s prefer-
ences. However, such exhaustive queries are not representative of
typical user behavior in real-world recommendation scenarios. To
better evaluate our method and compare it with different baselines
in leveraging user memory for personalized recommendations, we
simplify these queries using GPT-o3-mini to remove some of the
detailed preference information. More dataset statistics and the
prompt used for query simplification can be found inAppendix A.
4.1.2Baselines.We evaluate our method against a set of repre-
sentative models, which are divided into two groups. The first group
consists of general-purpose LLM backbones: GPT-4o ,DeepSeek-R1 ,
andQwen-2.5-3B-Instruct . The second group comprises models
specifically fine-tuned for recommendation tasks: BLAIR [9], a sen-
tence embedding model pretrained on user review and item meta-
data pairs using a contrastive objective, and Rec-R1 [14], whichdirectly optimizes LLM generation via feedback from a fixed, black-
box recommendation model. To investigate both the impact of
memory and the models’ ability to leverage user memory for rec-
ommendation, we consider three memory settings for these models:
•w/o Memory (Query-only):Models receive only the current
user query, without access to any historical interactions.
•w/ Naive Memory (Query + user interaction history):Except
queries, models are provided with the latest interaction history
of the user, representing straightforward memory integration.
•w/ Static Memory (Query + pre-generated user summary):
Models are provided with a pre-constructed summary of the
user’s preferences, representing static summarized memory.
4.1.3Evaluation settings.For each model under the correspond-
ing memory setting, given a user query, we require the model to
generate either the ideal item profile or its embedding, which is
then used to retrieve items from the candidate pool. Recommen-
dation performance is evaluated using standard metrics, including
nDCG@100, nDCG@1000, Recall@100 and Recall@1000.
4.1.4Implementation details.We select Qwen-2.5-3B-Instruct
as the backbone LLM in our method. During training, we use a
learning rate of 1e-6, a group size of 5, and set the maximum re-
sponse length to 768. Training is conducted for up to 5 epochs with
early stopping (patience = 1). In our method, the hyperparameters
for weighted reward are setas 𝑤1=0.1,𝑤2=5,𝑤3=0.1. More
implementation details can be found inAppendix B.
4.2 Overall Performance (RQ1)
We compare MR.Rec’s recommendation accuracy with baselines,
with results shown in Table 1. We make the following discussions:
•Performance Superiority of MR.Rec. MR.Rec shows consis-
tent advantages across different baseline categories. Compared
to baselines without memories, our RAG-based memory mecha-
nism improves performance by retrieving collaborative patterns.
This advantage persists against baselines with naive memory
and static summaries, demonstrating the effectiveness of syn-
ergizing memory with reasoning to enable dynamic preference
exploration and contextual personalization. Compared to recom-
mendation reasoning methods (i.e. Rec-R1 and BLAIR), MR.Rec
extends reasoning to include iterative memory retrieval, enabling
adaptive information gathering that evolves with the reasoning
process rather than relying on static input, significantly enhanc-
ing reasoning depth and context awareness.
•Incorporating Interaction Records for Baselines. For all
baselines, additionally incorporating naive memory (i.e. using
raw interaction records) generally brings performance gains,
validating the information gain of involving users’ past behaviors
in better understanding user queries. However, this improvement
is marginal for baselines fine-tuned for recommendation (i.e. Rec-
R1). This suggests that while LLM fine-tuning has unlocked
models’ recommendation potential, noise in raw interactions
and limited reasoning depth prevent further improvements.
•Static Summarization is Sometimes Harmful. To handle
noisy interaction records, fixed memory generates user sum-
maries as external memories for the baseline methods. However,
results show less improvement than naive memory and even
cause performance degradation. This demonstrates the difficulty

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
Table 1: The overall performance of baselines and MR.Rec. The bold/underline values represent the best/the second-best result,
respectively. The test dataset comprises 28 categories, and we report the averages for all while highlighting the top three.
Memory
SettingModelAll Home Clothing Tools
R@100 R@10 N@100 N@10 R@100 R@10 N@100 N@10 R@100 R@10 N@100 N@10 R@100 R@10 N@100 N@10
w/o
MemoryGPT-4o 0.226 0.092 0.086 0.059 0.227 0.057 0.090 0.033 0.100 0.030 0.036 0.022 0.247 0.097 0.096 0.065
DeepSeek-R1 0.252 0.099 0.090 0.060 0.256 0.085 0.082 0.049 0.130 0.035 0.040 0.022 0.290 0.118 0.093 0.059
Qwen2.5-3B 0.255 0.096 0.097 0.065 0.261 0.081 0.081 0.043 0.120 0.035 0.036 0.020 0.290 0.108 0.114 0.078
BLAIR-BASE 0.227 0.072 0.070 0.040 0.213 0.062 0.056 0.025 0.1350.035 0.037 0.018 0.280 0.086 0.080 0.042
BLAIR-LARGE 0.215 0.065 0.069 0.040 0.232 0.052 0.065 0.030 0.090 0.025 0.025 0.012 0.312 0.097 0.104 0.061
Rec-R1 0.258 0.111 0.099 0.071 0.265 0.085 0.086 0.047 0.126 0.037 0.04 0.022 0.297 0.114 0.117 0.08
w/ Naive
MemoryGPT-4o 0.258 0.109 0.104 0.072 0.278 0.081 0.091 0.047 0.125 0.035 0.041 0.025 0.301 0.119 0.110 0.079
DeepSeek-R1 0.260 0.106 0.100 0.067 0.275 0.085 0.090 0.051 0.127 0.033 0.043 0.026 0.301 0.118 0.109 0.074
Qwen2.5-3B 0.246 0.107 0.095 0.068 0.280 0.088 0.084 0.049 0.105 0.035 0.037 0.026 0.280 0.108 0.107 0.073
BLAIR-BASE - - - - - - - - - - - - - - - -
BLAIR-LARGE - - - - - - - - - - - - - - - -
Rec-R1 0.260 0.108 0.097 0.075 0.269 0.086 0.085 0.050 0.128 0.036 0.0400.027 0.299 0.112 0.119 0.085
w/ Static
MemoryGPT-4o 0.252 0.098 0.095 0.065 0.237 0.076 0.071 0.041 0.105 0.030 0.037 0.022 0.311 0.107 0.110 0.069
DeepSeek-R1 0.249 0.089 0.087 0.057 0.232 0.076 0.072 0.044 0.125 0.025 0.038 0.020 0.301 0.086 0.092 0.050
Qwen2.5-3B 0.246 0.106 0.098 0.069 0.265 0.081 0.081 0.044 0.110 0.035 0.034 0.020 0.280 0.115 0.116 0.086
BLAIR-BASE - - - - - - - - - - - - - - - -
BLAIR-LARGE - - - - - - - - - - - - - - - -
Rec-R1 0.259 0.105 0.095 0.069 0.264 0.082 0.083 0.046 0.122 0.033 0.038 0.024 0.286 0.110 0.115 0.071
Ours 0.270 0.122 0.113 0.084 0.284 0.090 0.092 0.054 0.130 0.040 0.045 0.027 0.333 0.129 0.132 0.091
Improvement +3.84% +9.91% +8.65% +12.00% +2.16% +2.27% +1.10% +5.88% - +8.18% +4.65% +0.00% +7.07% +8.40% +10.92% +5.81%
All Home Clothing Tools0.000.050.10NDCG@100
All Home Clothing Tools0.00.10.20.3Recall@100MR.Rec w/o Global Memory w/o Local Memory w/o RL Training
Figure 3: Ablation of different components of MR.Rec.
of building a global user summarization template, as the optimal
solution may vary greatly for different users and item categories.
Our dynamic memory retrieval mechanism with reasoning en-
hancement effectively addresses this concern and consistently
delivers substantial performance gains across all scenarios.
4.3 Ablation Study (RQ2)
4.3.1Effect of Key Components.We first study the impact of
different technical components of MR.Rec. The evaluation results
are presented in Figure 3. We study the following components:
•w/o Local/Global Memory. The results demonstrate that re-
moving either local or global memory mechanisms causes sig-
nificant performance degradation, with varying impact across
different datasets. This validates the effectiveness of our memory
indexing method in capturing beneficial interaction patterns and
our retrieval method in effectively accessing this information.
•w/o RL Tuning. Compared to removing memory mechanisms,
directly using the base LLM without RL tuning causes larger per-
formance damage. This is because RL tuning not only improves
the base model’s performance, but also significantly impacts the
retrieval loop effectiveness. Without our tuning, small-parameter
LLMs such as the 3B model struggle to determine when and how
to leverage memory and perform reasoning. This highlights the
critical role of RL tuning for MR.Rec.Table 2: Ablation of user-specific local memory components.
Variant R@100 R@10 N@100 N@10
w/o Local Memory 0.258 0.113 0.098 0.069
w/Behavior Records 0.268↑0.117↑0.109↑0.079↑
w/Preference Patterns0.272↑0.116↑0.110↑0.079↑
w/User Profile 0.269↑0.1130.113↑0.080↑
w/B+P+U0.270↑0.122↑0.113↑0.084↑
4.3.2Effect of Local Memory Components.As shown in Ta-
ble 2, we further investigate the impact of different components
of user-specific local memory on model performance. The results
reveal that the absence of the entire local memory mechanism
leads to the weakest performance, whereas incorporating any sin-
gle component (Behavior Records,Preference Patterns, orUser
Profile) individually yields clear improvements. The best results
are achieved when all three components are combined, suggesting
a complementary effect that more comprehensively captures user
preferences relevant to the current query.
4.4 Memory Effectiveness Study (RQ3)
To further investigate the effectiveness of memory in the recom-
mendation process, we analyzed the retrieved memory and de-
signed two metrics:Memory-to-Profile Contribution (MPC)
andMemory-to-Recommendation Contribution (MRC). MPC
measures whether the retrieved memory aids in generating the
ideal item profile, while MRC evaluates whether it contributes to
the final correct recommendation.
To ensure a comprehensive and objective evaluation, we em-
ployed two assessment approaches: aheuristic method, check-
ing keyword overlaps between retrieved memory and profiles or
ground-truth items, and anLLM-judged method, where an LLM
determines whether the memory is helpful.
As shown in Table 3, both Global and Local Memory are valu-
able in the recommendation process. For MPC, they both achieve

MR.Rec: Synergizing M emory and R easoning for Personalized Rec ommendation Assistant with LLMs
Table 3: Contribution of retrieved memory.
Memory TypeMPC MRC
Heuristic LLM-Judged Heuristic LLM-Judged
Global Memory 0.9605 0.9172 0.9012 0.5375
Local Memory 0.9368 0.9511 0.4743 0.7036
high scores, indicating that retrieved memory significantly aids the
generation of ideal item profiles. This demonstrates that MR.Rec
effectively leverages memory to provide final recommendation. For
MRC, Global Memory scores relatively high under the heuristic
method but somewhat lower under LLM judgment, reflecting that
it often overlaps with ground-truth items at the surface level, while
semantic contribution is more nuanced. By contrast, Local Memory
shows modest heuristic scores but substantially higher LLM-judged
scores. This indicates that, despite limited surface-level overlap,
local memory provides more semantically meaningful support for
recommendations. This effectiveness arises from our approach:
instead of one-shot retrieval based solely on query similarity, the
LLM is trained to reason about which memory entries are genuinely
useful, enabling more precise and context-aware recommendations.
4.5 Efficiency Study (RQ4)
This experiment studies the costs and efficiency of our memory
indexing and retrieval mechanisms. We built user-specific local
memory for 3,000 users and sampled 157 users for cross-user global
memory, creating 73,078 local and 1,970 global memory entries. The
statistics and costs are shown in Table 4.
•Time and Cost Efficiency of Indexing. In our proposed MR.Rec,
indexing time per entry is 0.07s for local and 0.06s for global
memory, with API costs of $54 for 3,000 users ($0.018/user) and
$0.07 for global memory. Global memory uses only 0.48M in-
put and 0.06M output tokens, roughly 1/1000 of local memory,
showing high indexing efficiency.
•Token Efficiency of Retrieval. Table 4 shows our method
achieves highest R@100 with 95.43 tokens, yielding an efficiency
of 0.299 (R@100/100 tokens), outperforming baselines that utilize
recent interactions or static profiles as memory. By retrieving the
most relevant memory entries for recommendation, our method
introduces less noise into the context, leading to higher recall
and more efficient reasoning.
Table 4: Cost and Efficiency for Memory Indexing and Re-
trieval in the MR.Rec Framework
Statistics and Cost of Memory Indexing
Local Memory Global Memory
Source Users 3,000 157
Input Tokens (M) 385.21 0.48
Output Tokens (M) 42.00 0.06
Memory Entries 73,078 1,970
Total Time (s) 2,435 121
Per-entry Time (s) 0.07 0.06
API Cost (GPT-4o-mini) 54.09 0.07
Per-entry Cost7.4×10−43.6×10−5
Token Efficiency of Memory Retrieval
Memory SettingAvg. Memory
TokensR@100
(GPT-4o)Efficiency
(R@100/100 tokens)
Recent 10 interactions 283.51 0.261 0.092
Static user profile 492.7 0.261 0.053
MR.Rec 95.43 0.285 0.299
0 25 50 75
Step0.00.20.4Value
Recall@1000 25 50 75
Step0.00.51.0
Memory Reward0 25 50 75
Step0.00.51.0
Format RewardBase Model Instruct ModelFigure 4: Reward trajectories of MR.Rec using the base and
instruct models of Qwen-2.5-3B, during our RL tuning.
10 100 1000
@K0.050.100.15NDCG
10 100 1000
@K0.10.20.30.40.5Recall
Qwen3-Embedding-0.6B BGE-M3 BM25
Figure 5: Effect of using different retrievers for MR.Rec.
4.6 Sensitivity Study (RQ5)
This section studies the impact of different configurations of our
MR.Rec, including LLM backbones and different retrieval settings.
4.6.1Impact of LLM Backbones.To study the impact of differ-
ent LLM backbones, we compare the default backbone of MR.Rec,
Qwen-2.5-3B-Instruct, with its base model Qwen-2.5-3B-Base. The
former has undergone prior instruction tuning and reinforcement
learning from human feedback (RLHF) to strengthen its instruction-
following capabilities, while the latter has not. Both MR.Rec variants
using these backbone LLMs are tuned using our RL method, and
their reward trajectories are recorded and presented in Figure 4.
The results demonstrate strong instruction-following ability of
the Instruct model from the start, correctly invoking memory re-
trieval (Memory reward) and generating properly formatted outputs
(Format reward). In contrast, the Base model initially struggles to
follow instructions, with format reward remaining near zero. Dur-
ing training, the Base model gradually learns to produce properly
formatted outputs and shows improvements in recommendation
performance, as reflected by Recall@100. However, it consistently
fails to learn memory retrieval, as indicated by near-zero memory
reward, instead relying solely on direct recommendations.
4.6.2Impact of Base Retrievers.We compared using three
different base retrieval algorithms for our reasoning-enhanced
retrieval process, including Qwen3-Embedding-0.6B [ 37], BGE-
M3 [ 2], and BM25 [ 21]. Among these, Qwen3-Embedding-0.6B
supports customized instructions during retrieval, BGE-M3 is a
pre-trained deep learning model, while BM25 is a traditional lexical
matching method that requires no pre-training.
The experimental results shown in Figure 5 demonstrate that
Qwen3-Embedding-0.6B achieves the best performance, while BM25,
which lacks specialized training for dense retrieval, shows a signifi-
cant performance gap compared to the embedding-based models.
The performance gaps between models become more pronounced
as the value of k increases in NDCG and Recall@k metrics, indi-
cating that the advantages of instruction-tuned embedding models
are more evident when retrieving larger sets of relevant items.
4.6.3Impact of Retrieved Memory Size.We further analyze the
impact of retrieving different numbers of top-k relevant memory

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
12345
Top K0.1100.1130.1150.118Recall@10 Metrics
12345
Top K0.2520.2580.2640.270Recall@100 Metrics
12345
Top K0.4800.4860.4910.497Recall@1000 Metrics
0.0730.0760.0790.082
NDCG
0.1000.1040.1080.112
NDCG
0.1300.1330.1360.139
NDCG
Recall NDCG
Figure 6: Effect of different top-k of memory retrieval.
Case Study
User Query:Find me a cap suitable for post-winter wear.
GPT-4o:Cap for post-winter wear, suitable for mild to cool weather, lightweight
yet warm, stylish and comfortable, ideal for transitional seasons after winter.
MR.Rec: [Reasoning] : For this query, I should consider whether user has
preferences on "Design and Style Balance", "Material Quality and Feel" and "Fit
and Comfort".
[Retrieved Memory] : - For "Material Quality and Feel": the user prefers
premium, natural materials like real leather, rubber, and carbon fiber over
synthetic options, valuing durability and a high-quality tactile experience.
- For "Fit and Comfort": The user likes true-to-size, well-fitting clothing that
without feeling tight or restrictive, with good support and enough room.
[Reasoning] : Given the user’s query and retrieved memory, the ideal profile
includes:
- Material Quality: Durable, high-quality materials like real leather or carbon
fiber.
- Fit and Comfort: True-to-size and adjustable for a comfortable fit.
- Seasonal Suitability: Suitable for post-winter, lightweight.
Ground Truth Item:DALLY Up Leather Patch Western Lifestyle Adjustable
6-Panel Snapback Hat. The Dally Cap line is for those who want to look just as
Western on the outside as they are on the inside.
Figure 7: Case study demonstrating MR.Rec’s memory-
synergized reasoning for personalized cap recommendation.
entries on model performance, with results shown in Figure 6.
The results reveal a trade-off pattern: smaller k values may miss
useful memory information, while larger k values may introduce
noise and extend the context window, potentially degrading LLM
performance. Based on these findings, we select k=3 as the optimal
balance, which consistently achieves strong performance.
4.7 Case Study
We present a case study in Figure 7, illustrating how MR.Rec inte-
grates memory and reasoning for recommendation. When a user
queries for "a cap suitable for post-winter wear," the baseline GPT-
4o generates a generic response focused on seasonal suitability. In
contrast, MR.Rec first reasons through implicit preference dimen-
sions like "Material Quality and Feel" and "Fit and Comfort." It then
retrieves relevant memory segments reflecting the user’s preference
for premium materials like real leather and adjustable items. The
model generates an ideal item profile combining these preferences
with seasonal needs. The final recommendation strongly aligns with
the ground truth item, a leather patch adjustable snapback hat, with
key attributes (leather, adjustable) appearing in both the generated
profile and actual item, demonstrating how our reasoning-enhanced
memory retrieval successfully captures user-specific preferences
that would be missed by generic baselines.
4.8 Related Work
4.8.1Memory Modeling in Recommender Systems.Memory
modeling is essential for capturing dynamic user preferences inRSs. Traditional sequential models, such as SASRec [ 12], implicitly
encode user history but often struggle with long sequences and lack
explicit interpretability. The emergence of LLMs has introduced ex-
plicit memory mechanisms, enabling more sophisticated and inter-
pretable representations of user behavior. A prominent approach in-
volves utilizing user interaction history as memory [ 10,26,39], with
methods like AutoMR [ 23] employing learned retrievers to intelli-
gently select relevant historical segments. In contrast, MARM [ 16]
and LMN [ 15] enhance efficiency by caching computations and
utilizing product quantization to handle long sequences and large
memory banks. An alternative approach leverages LLM-generated
user profiles, which synthesize extensive interaction histories into
textual summaries [ 32]. MAP [ 3] constructs user profiles as tables of
historical ratings and reviews, but this flat memory structure strug-
gles with noisy and multi-granularity preferences. MemoCRS [ 31]
introduces an entity-based user memory and a shared general mem-
ory for multi-turn dialogues. Nevertheless, it fails to effectively
integrate memory with reasoning, resulting in a superficial reliance
on similarity-based retrieval, which neglects deeper semantic rela-
tionships within the memory and limits the generation of contex-
tually relevant recommendations. In comparison, MR.Rec develops
hierarchical memory indexing and reasoning-enhanced retrieval,
better filtering noisy information while achieving efficiency.
4.8.2LLM Reasoning for Recommendation.LLMs have trans-
formed RSs by introducing advanced reasoning beyond traditional
collaborative filtering. Key developments include Chain-of-Thought
(CoT) [ 28] reasoning, which allows LLMs to generate step-by-step
logical recommendations. For instance, HetGCoT-Rec [ 11] inte-
grates heterogeneous graphs with CoT for journal recommenda-
tions, and Li et al. [ 13] use CoT to enhance user intention infer-
ence and contextual awareness. However, prompting-based CoT
methods remain limited in complex recommendation tasks. To
address these limitations, recent studies have shifted toward train-
ing LLMs explicitly for reasoning rather than relying solely on
prompting. A representative approach for enhancing LLM reason-
ing is GRPO [ 6]. Reason-to-Recommend [ 40] employs GRPO to
strengthen "Interaction-of-Thought" reasoning, enabling emergent
planning behaviors in challenging recommendation tasks. Similarly,
LatentR [ 36] adopts a modified GRPO framework to optimize la-
tent reasoning trajectories, avoiding explicit text generation while
preserving reasoning capacity. Our work extends LLM finetuning
beyond CoT reasoning to include memory-synergized reasoning.
5 Conclusion
In this paper, we introduced MR.Rec, a framework that unifies
memory and reasoning to advance LLM-based recommendation
assistants. By combining a RAG-based external memory mecha-
nism with reinforcement learning, our approach enables the LLM
to actively explore and select useful preferences and experiences
while performing multi-step reasoning over retrieved information
to generate accurate recommendations. Extensive experiments on
multiple benchmark datasets demonstrate that MR.Rec consistently
outperforms state-of-the-art baselines in both personalization and
reasoning capability. We believe MR.Rec takes a meaningful step
toward intelligent and user-centered recommendation assistants.

MR.Rec: Synergizing M emory and R easoning for Personalized Rec ommendation Assistant with LLMs
References
[1] Millennium Bismay, Xiangjue Dong, and James Caverlee. Reasoningrec: Bridging
personalized recommendations and human-interpretable explanations through
llm reasoning.arXiv preprint arXiv:2410.23180, 2024.
[2]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation.arXiv preprint arXiv:2402.03216,
2024.
[3]Jiarui Chen. Memory assisted llm for personalized recommendation system.
arXiv preprint arXiv:2505.03824, 2025.
[4]Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen, Suzan Verberne, and
Zhaochun Ren. A multi-agent conversational recommender system.arXiv
preprint arXiv:2402.01135, 2024.
[5]Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. From llm
reasoning to autonomous ai agents: A comprehensive review.arXiv preprint
arXiv:2504.19678, 2025.
[6]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incen-
tivizing reasoning capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025.
[7] Donghee Han, Hwanjun Song, and Mun Yong Yi. Rethinking llm-based recom-
mendations: A query generation-based, training-free approach.arXiv preprint
arXiv:2504.11889, 2025.
[8]Zhankui He, Zhouhang Xie, Harald Steck, Dawen Liang, Rahul Jha, Nathan
Kallus, and Julian McAuley. Reindex-then-adapt: Improving large language
models for conversational recommendation. InProceedings of the Eighteenth
ACM International Conference on Web Search and Data Mining, pages 866–875,
2025.
[9] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley.
Bridging language and items for retrieval and recommendation.arXiv preprint
arXiv:2403.03952, 2024.
[10] Jiani Huang, Shijie Wang, Liang-bo Ning, Wenqi Fan, Shuaiqiang Wang, Dawei
Yin, and Qing Li. Towards next-generation recommender systems: A bench-
mark for personalized recommendation assistant with llms.arXiv preprint
arXiv:2503.09382, 2025.
[11] Runsong Jia, Mengjia Wu, Ying Ding, Jie Lu, and Yi Zhang. Hetgcot-rec: Hetero-
geneous graph-enhanced chain-of-thought llm reasoning for journal recommen-
dation.arXiv preprint arXiv:2501.01203, 2025.
[12] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommenda-
tion. In2018 IEEE international conference on data mining (ICDM), pages 197–206.
IEEE, 2018.
[13] Gangmin Li, Fan Yang, and Yong Yue. Identify user intention for recommendation
using chain-of-thought prompting in llm. InInternational Conference for Emerging
Technologies in Computing, pages 60–69. Springer, 2024.
[14] Jiacheng Lin, Tian Wang, and Kun Qian. Rec-r1: Bridging generative large
language models and user-centric recommendation systems via reinforcement
learning.arXiv preprint arXiv:2503.24289, 2025.
[15] Hui Lu, Zheng Chai, Yuchao Zheng, Zhe Chen, Deping Xie, Peng Xu, Xun
Zhou, and Di Wu. Large memory network for recommendation. InCompanion
Proceedings of the ACM on Web Conference 2025, pages 1162–1166, 2025.
[16] Xiao Lv, Jiangxia Cao, Shijie Guan, Xiaoyou Zhou, Zhiguang Qi, Yaqiang Zang,
Ming Li, Ben Wang, Kun Gai, and Guorui Zhou. Marm: Unlocking the future of
recommendation systems through memory augmentation and scalable complex-
ity.arXiv preprint arXiv:2411.09425, 2024.
[17] Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang,
Ren Chen, Christopher Leung, Jiajie Tang, and Jiebo Luo. Llm-rec: Personal-
ized recommendation via prompting large language models.arXiv preprint
arXiv:2307.15780, 2023.
[18] Lucie Charlotte Magister, Katherine Metcalf, Yizhe Zhang, and Maartje ter Hoeve.
On the way to llm personalization: Learning to remember user conversations.
arXiv preprint arXiv:2411.13405, 2024.
[19] Andriy Mnih and Russ R Salakhutdinov. Probabilistic matrix factorization.
Advances in neural information processing systems, 20, 2007.
[20] Bruno Pradel, Savaneary Sean, Julien Delporte, Sébastien Guérif, Céline Rou-
veirol, Nicolas Usunier, Françoise Fogelman-Soulié, and Frédéric Dufau-Joel. A
case study in a recommender system based on purchase data. InProceedings of
the 17th ACM SIGKDD international conference on Knowledge discovery and data
mining, pages 377–385, 2011.
[21] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework:
Bm25 and beyond.Foundations and Trends®in Information Retrieval, 3(4):333–389,
2009.
[22] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and
Yuning Jiang. Think before recommend: Unleashing the latent reasoning powerfor sequential recommendation.arXiv preprint arXiv:2503.22675, 2025.
[23] Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, and Fuli
Feng. Leveraging memory retrieval to enhance llm-based generative recommen-
dation. InCompanion Proceedings of the ACM on Web Conference 2025, pages
1346–1350, 2025.
[24] Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun,
Qingwei Lin, Weiwei Deng, Dongmei Zhang, et al. Lettingo: Explore user profile
generation for recommendation system. InProceedings of the 31st ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V. 2, pages 2985–2995, 2025.
[25] Mingze Wang, Chongming Gao, Wenjie Wang, Yangyang Li, and Fuli Feng.
Tunable llm-based proactive recommendation agent. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 19262–19276, 2025.
[26] Shijie Wang, Wenqi Fan, Yue Feng, Shanru Lin, Xinyu Ma, Shuaiqiang Wang,
and Dawei Yin. Knowledge graph retrieval-augmented generation for llm-based
recommendation.arXiv preprint arXiv:2501.02226, 2025.
[27] Yan Wang, Zhixuan Chu, Xin Ouyang, Simeng Wang, Hongyan Hao, Yue Shen,
Jinjie Gu, Siqiao Xue, James Zhang, Qing Cui, et al. Llmrg: Improving recommen-
dations through large language model reasoning graphs. InProceedings of the
AAAI conference on artificial intelligence, volume 38, pages 19189–19196, 2024.
[28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning
in large language models.Advances in neural information processing systems,
35:24824–24837, 2022.
[29] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. Graph neural networks
in recommender systems: a survey.ACM Computing Surveys, 55(5):1–37, 2022.
[30] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng
Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey
on memory mechanisms in the era of llms.arXiv preprint arXiv:2504.15965, 2025.
[31] Yunjia Xi, Weiwen Liu, Jianghao Lin, Bo Chen, Ruiming Tang, Weinan Zhang, and
Yong Yu. Memocrs: Memory-enhanced sequential conversational recommender
systems with large language models. InProceedings of the 33rd ACM International
Conference on Information and Knowledge Management, pages 2585–2595, 2024.
[32] Jiarui Zhang. Guided profile generation improves personalization with llms.
arXiv preprint arXiv:2409.13093, 2024.
[33] Qinyao Zhang, Bin Guo, Yao Jing, Yan Liu, and Zhiwen Yu. Mindmemory: Aug-
mented llm with long-term memory and mental personality. InCCF Conference
on Computer Supported Cooperative Work and Social Computing, pages 462–476.
Springer, 2024.
[34] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. Deep learning based recommender
system: A survey and new perspectives.ACM computing surveys (CSUR), 52(1):1–
38, 2019.
[35] Yadong Zhang, Shaoguang Mao, Tao Ge, Xun Wang, Adrian de Wynter, Yan
Xia, Wenshan Wu, Ting Song, Man Lan, and Furu Wei. Llm as a mastermind:
A survey of strategic reasoning with large language models.arXiv preprint
arXiv:2404.01230, 2024.
[36] Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He,
and Tat-Seng Chua. Reinforced latent reasoning for llm-based recommendation.
arXiv preprint arXiv:2505.19092, 2025.
[37] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding:
Advancing text embedding and reranking through foundation models.arXiv
preprint arXiv:2506.05176, 2025.
[38] Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, and Yong
Li. A survey of large language model empowered agents for recommenda-
tion and search: Towards next-generation information retrieval.arXiv preprint
arXiv:2503.05659, 2025.
[39] Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao,
Chunxu Zhang, Qidong Liu, and Peng Jiang. Llm-powered user simulator for rec-
ommender system. InProceedings of the AAAI Conference on Artificial Intelligence,
volume 39, pages 13339–13347, 2025.
[40] Keyu Zhao, Fengli Xu, and Yong Li. Reason-to-recommend: Using interaction-
of-thought reasoning to enhance llm recommendation.arXiv preprint
arXiv:2506.05069, 2025.
[41] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen
Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al. Recommender systems in the
era of large language models (llms).IEEE Transactions on Knowledge and Data
Engineering, 36(11):6889–6907, 2024.
[42] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu,
Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large
language models.arXiv preprint arXiv:2311.07911, 2023.
[43] Joyce Zhou, Yijia Dai, and Thorsten Joachims. Language-based user profiles for
recommendation.arXiv preprint arXiv:2402.15623, 2024.

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
Table 5: Dataset Statistics
Num. Queries 6096
Num. Users 6039
Num. Categories 28
Num. Candidate Items 1,058,417
Num. User-Item Interactions 577,376
Prompt Template for Pre-processing Queries
System Prompt:Rewrite the "Original Query" into a single,
casual, conversational question. Rules:
- Keep: 1 core item + 1 core use case (e.g., "a laptop for gaming").
- Delete: All other details (including brand, specs, price,
personal preferences, etc.).
- Tone: Natural and conversational.
- Format: Must be a single sentence.
Original Query:I’m in need of an external Blu-ray drive that
actually works well with reading Blu-rays. The one I currently
own has a poor ability to read discs, requiring multiple tries
and hoping for a successful read. I want a Blu-ray drive that is
amazing by comparison, where I can simply insert a disc and it
starts playing right away. Skipping around on the disc should be
smooth and there shouldn’t be any endless seeking from the drive.
I came across one drive that shows up as a Pioneer BDR-TD03 in
the properties, and based on its performance, I’m really satisfied
with its quality. The name "Dainty" might be unusual for a
disk drive, but it doesn’t affect its functionality, which is fantastic.
Rewritten Query:Hey, can you recommend an external
Blu-ray drive that reads discs reliably and plays them smoothly
right from the start?
APPENDIX
A Dataset
We derive our experimental data from the Amazon-C4 corpus,
which contains user queries synthesized by ChatGPT from prod-
uct review text. These synthetic queries often include fine-grained,
product-level details (e.g., brand names, model identifiers, and ex-
act technical specifications) that are uncommon in real-world user
utterances. In practice, users tend to give concise or underspec-
ified requests—such as “recommend a budget laptop for college”
or “gift ideas for my father who likes hiking”—and expect the rec-
ommender to infer context from prior interactions or stored user
memories. To better emulate these memory-driven recommenda-
tion scenarios, we transform Amazon-C4 so that queries resemble
realistic, high-level requests. Concretely, we (i) remove or mask
overly specific attributes (brand names, exact model identifiers,
serial numbers, and verbatim review excerpts) and (ii) preserve the
user’s core intent and constraints (product category, coarse price
band, primary use case). To achieve this, we prompt GPT-o3-mini
to produce concise, naturalistic rewrites that retain the original
intent and constraints while omitting gratuitous detail. The promptused for simplification and representative before/after examples
are provided below.
Due to the limited availability of API resources, we randomly
sampled 6,096 queries from the Amazon-C4 dataset and applied
the aforementioned preprocessing procedure to this subset. The
resulting dataset retains the essential characteristics necessary for
our experiments while remaining computationally manageable. A
detailed summary of the processed dataset is provided in Table 5.
B Implementation Details
B.1 Environment
Our experiments were conducted using 4 x NVIDIA A800(80GB)
GPUs. We employed PyTorch 2.6.0 as the deep learning framework,
Transformers 4.48.0 for model implementation, and the GRPO im-
plementation provided by VeRL. For inference acceleration, we
utilized SgLang 0.4.6.
B.2 Prompts for LLM-based Baselines
With the exception of BLAIR, all other baselines are LLM-based.
Following the Rec-R1 framework, we prompt the LLM to generate
an ideal item profile corresponding to a given user query. Specifi-
cally, we employ the following prompt template:"You are an expert
in generating queries for dense retrieval. Given a customer query, your
task is to retain the original query while expanding it with additional
semantically relevant information, retrieve the most relevant prod-
ucts, ensuring they best meet customer needs. If no useful expansion is
needed, return the original query as is. # Below is the product search
query:{query}"
B.3 Details for Our Method
For both memory retrieval and item retrieval, we adopt the Qwen3-
Embedding-0.6B model as the retrieval encoder. For reinforcement
learning with LLMs, the learning rate is set to 1e5 with a batch size
of 256. During trajectory generation, we configure the group size
(G = 5) and use a rollout temperature of 1.0.
C Prompts Used in Our Method
The prompts employed in our method fall into two primary cate-
gories:Memory IndexingandLLM-based Recommendation.
C.1 Prompts for Memory Indexing
C.1.1 Cross-User Global Memory.To construct the Cross-User
Global Memory, we sample interaction histories from multiple
users across different recommendation scenarios (e.g., purchasing
products from various categories). These histories include user
queries, selected items, and their associated metadata. The LLM is
then prompted to analyze each scenario and infer additional aspects
that users might consider beyond what is explicitly mentioned in the
query, thereby extracting richer recommendation experiences. To
encourage the LLM to better reason about these additional aspects,
we also sample items that are similar to the ground-truth purchases
but were not selected by the user as negative items. The prompt
used in this process is as above.
Using the prompt described above, the LLM analyzes each user’s
interaction data to extract recommendation knowledge specific to
the corresponding scenario. To capture generalizable cross-user

MR.Rec: Synergizing M emory and R easoning for Personalized Rec ommendation Assistant with LLMs
Prompt for Cross-User Global Memory (A)
You are given the following information for a recommendation task:
-Recommendation scenario: {scenario}
-User query: {query}
-Ground truth item (the item finally purchased): {ground truth item metadata}
-Negative items (similar items not chosen by the user): {negative items}
Your task:
1. Analyze the user query, the purchased item, and the unchosen items.
2. Identify additional aspects or dimensions that the user might have implicitly considered beyond what is explicitly stated in the query.
These aspects should support more personalized recommendations.
At the category level: What general personalization factors might users care about beyond the query, such as overall budget, preferred
brands, aesthetics, or usage context? Do not mention specific product or features in this level.
At the subcategory level (e.g., the relevant subcategory of ground truth item): What specific personalization factors might be relevant,
especially when comparing the ground truth item with the negative items (e.g., screen size, smart features, energy efficiency, style)?
3. Focus on uncovering subtle or implicit preferences (e.g., color, price sensitivity, design style, brand affinity, feature trade-offs) that can
be inferred by comparing the ground truth item with the negative items.
Output format (structured):
{{
"category_level_personalization_aspects": [
{
"category": "category_name",
"aspect": "aspect_name",
"description": "description"
},
...]
,
"subcategory_level_personalization_aspects": [
{
"subcategory": "subcategory_name",
"aspect": "aspect_name",
"description": "description "
},
...
]
}}
global memory, we sample interactions from multiple users and
prompt the LLM to aggregate the extracted knowledge, producing
a concise and coherent summary that represents common recom-
mendation considerations within the scenario. The prompt used
for this process is presented below:
C.1.2 User-specific Local Memory.Our proposed User-specific Lo-
cal Memory is organized into three hierarchical levels: Behavior
Records, Preference Patterns, and User Profile. These levels capture
user information at progressively finer granularity, ranging from
abundant and potentially noisy interaction data to highly abstracted
and fine-grained summaries. Specifically, Preference Patterns are
derived from*Behavior Records, and the User Profile represents a
further distillation and consolidation of these preference patterns,
providing a succinct yet informative summary of the user’s interests
and tendencies.
For a given user, we collect their interaction history within a
specific recommendation scenario and prompt the LLM to ana-
lyze these interactions to identify Preference Patterns. This processenables the extraction of recurring behaviors, interests, and im-
plicit constraints that characterize the user’s preferences within
the scenario. The prompt used for this analysis is as follows:
After generating Preference Patterns for a user across different
scenarios, we prompt the LLM to reason over these patterns to infer
the user’s potential User Profile. This step consolidates scenario-
specific preferences into a coherent and high-level summary of
the user’s interests and characteristics. The prompt used for this
inference process is as follows:
C.2 Prompts for LLM-based Recommendation
In this study, we develop an LLM-based recommendation assis-
tant that integrates memory and reasoning. To enable the LLM to
autonomously leverage user-specific local memory guided by cross-
user global memory, we employ the following prompt, which in-
structs the model to generate keywords for global memory retrieval.

Jiani Huang*, Xingchen Zou*, Lianghao Xia†, and Qing Li†
Prompt for Cross-User Global Memory (B)
You are given some aspects that multiple users consider important for a category.
Category: {category}
Current aggregated aspects: {aspects}
Task: - Merge semantically duplicate or highly overlapping aspects into unified canonical names. - Normalize naming style (use
consistent Title Case English or best canonical phrasing). - Consolidate and de-duplicate descriptions under each merged aspect; keep
key insights, remove redundancy. - These are shared aspects of multiple users, so do not use any specific product/parameter/user in the
description.
Output strictly as a compact JSON object mapping aspect_name to list of descriptions, e.g.:
{{
"aspect_name": ["description", ...],
...
}}
Prompt for LLM-based Recommendation
You are a recommendation expert. Your task is to generate
an ideal item profile based on a user’s query and memories.
When given a user query, you should analyze the user’s
query and identify additional factors that might refine the
recommendation. {Retrieved Cross-User Global Memory Based
on Query}
You should retrieve current user’s memories that related to
these aspects via ‘memory_retrieval_tool‘. After retrieving the
memories, combine them with the user’s query, think step by
step to reason about an ideal item profile for item retrieval.
This profile should be as detailed as possible, do not miss any
important keywords, otherwise relevant items may be missed.
Your final output should be structured as follows:
\boxed{"ideal_item_profile": str,
"useful_memory_ids": List[str]}
Remember: First, retrieve the relevant memories, then reason
about detailed ideal item profile based on the query and the
memories you retrieved.Prompt for Extracting Preference Patterns
You are given the following information about a user’s
purchase history in a specific category:
- Category: {category} - User’s purchased items in this category,
each with its metadata and the user’s review: {reviews}
Your task: Analyze the items (including their metadata and
the user’s reviews) and summarize the user’s preferences in
this category. The summary should capture consistent patterns
across items and reviews (e.g., favored brands, preferred price
range, styles, features, quality expectations). Be as detailed as
possible, but do not fabricate information that is not supported
by the input. Directly output the preference summary (string)
below:
Prompt for Generating User Profile
You are given the following information about a user’s prefer-
ences across different categories or aspects:
{preference patterns}
Your task: Based on these preferences, infer the user’s over-
all profile. The profile should summarize general traits, pat-
terns, and tendencies that can be reasonably inferred from the
given preferences (e.g., spending habits, brand inclination, style
choices, feature priorities, quality expectations, lifestyle hints).
Do not fabricate information that is not supported by the input.
Directly output the profile (string) below:
The LLM then uses the retrieved memory to perform reasoning and
produce personalized recommendations.
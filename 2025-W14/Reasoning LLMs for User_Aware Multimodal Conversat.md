# Reasoning LLMs for User-Aware Multimodal Conversational Agents

**Authors**: Hamed Rahimi, Jeanne Cattoni, Meriem Beghili, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani

**Published**: 2025-04-02 13:00:17

**PDF URL**: [http://arxiv.org/pdf/2504.01700v1](http://arxiv.org/pdf/2504.01700v1)

## Abstract
Personalization in social robotics is critical for fostering effective
human-robot interactions, yet systems often face the cold start problem, where
initial user preferences or characteristics are unavailable. This paper
proposes a novel framework called USER-LLM R1 for a user-aware conversational
agent that addresses this challenge through dynamic user profiling and model
initiation. Our approach integrates chain-of-thought (CoT) reasoning models to
iteratively infer user preferences and vision-language models (VLMs) to
initialize user profiles from multimodal inputs, enabling personalized
interactions from the first encounter. Leveraging a Retrieval-Augmented
Generation (RAG) architecture, the system dynamically refines user
representations within an inherent CoT process, ensuring contextually relevant
and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate
significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L
(+8%) F1 scores over state-of-the-art baselines, with ablation studies
underscoring the impact of reasoning model size on performance. Human
evaluations further validate the framework's efficacy, particularly for elderly
users, where tailored responses enhance engagement and trust. Ethical
considerations, including privacy preservation and bias mitigation, are
rigorously discussed and addressed to ensure responsible deployment.

## Full Text


<!-- PDF content starts -->

Reasoning LLMs for User-Aware Multimodal Conversational Agents
Hamed Rahimi1,∗, Jeanne Cattoni2,3, Meriem Beghili1, Mouad Abrini1,
Mahdi Khoramshahi1, Maribel Pino3, and Mohamed Chetouani1
Abstract — Personalization in social robotics is critical for
fostering effective human-robot interactions, yet systems often
face the cold start problem, where initial user preferences or
characteristics are unavailable. This paper proposes a novel
framework called USER-LLM R1 for a user-aware conversa-
tional agent that addresses this challenge through dynamic user
profiling and model initiation. Our approach integrates chain-
of-thought (CoT) reasoning models to iteratively infer user
preferences and vision-language models (VLMs) to initialize
user profiles from multimodal inputs, enabling personalized
interactions from the first encounter. Leveraging a Retrieval-
Augmented Generation (RAG) architecture, the system dy-
namically refines user representations within an inherent CoT
process, ensuring contextually relevant and adaptive responses.
Evaluations on the ElderlyTech-VQA Bench demonstrate signif-
icant improvements in ROUGE-1 (+23.2%) ROUGE-2 (+0.6%)
and ROUGE-L (+8%) F1 scores over state-of-the-art baselines,
with ablation studies underscoring the impact of reasoning
model size on performance. Human evaluations further validate
the framework’s efficacy, particularly for elderly users, where
tailored responses enhance engagement and trust. Ethical con-
siderations, including privacy preservation and bias mitigation,
are rigorously discussed and addressed to ensure responsible
deployment.
I. INTRODUCTION
The integration of personalized and adaptive AI systems
in human-robot interactions (HRI) is pivotal for fostering
ethical, safe, and contextually effective engagements [1], [2].
Personalization necessitates that robots dynamically adjust
their behavior to align with individual user characteristics,
such as age, cognitive abilities, and cultural background,
ensuring interactions are both socially appropriate and func-
tionally optimized [3]. For instance, a child-centric inter-
action may demand simplified language, playful tones, and
heightened safety protocols, whereas adult interactions could
prioritize task complexity, formal dialogue, and nuanced
social cues. Achieving such adaptability requires mecha-
nisms to tailor actions while addressing challenges like
data privacy, algorithmic bias, and transparency in decision-
making [4]. User modeling aims to discover patterns and
learn representations from user data, capturing characteristics
such as profiles, preferences, and personalities [5].
Recent advancements in Multimodal Large Language
Models (MLLMs) [6] have significantly enhanced the ca-
pacity of robots to interpret textual and visual cues, enabling
more intuitive and adaptive interactions [7], [8]. However,
∗Correspondence: hamed.rahimi@sorbonne-universite.fr
1ISIR, Sorbonne Universit ´e, Paris, France.
2Universit ´e Paris Cit ´e, Paris, France.
3Assistance Publique – H ˆopitaux de Paris (AP-HP), Paris, France.
User 
Encoder
in 
DB?
User 
DB
Yes
No
User 
Image
User 
VLM
User 
Retriv
er
Reasoning 
LLM
Output
QuestionFig. 1: USER-LLM R1 Architecture: The framework con-
sists of three principal components: a User Encoder for
profile encoding, a Vision-Language Model (VLM) for initial
user modeling, and a Chain-of-Thought (CoT) Reasoning
Large Language Model (LLM) for updating user profile and
personalized response generation.
critical gaps persist when user modeling metadata (e.g. de-
mographic traits, behavioral patterns, or interaction histories)
are absent, creating a ”cold-start” problem [9] that limits per-
sonalization efficacy. Even when partial information exists,
current systems often rely excessively on prior conversational
data, neglecting dynamic methods to infer, validate, or update
user profiles in real time [10]. In [3], a set of Vision Lan-
guage Models (VLMs) are trained to provide personalized
responses corresponding to implicit user modeling of age,
gender, emotion, and ethnicity trough user images. Despite
bias optimization of these VLMs, a seamless interaction
requires a dynamic mechanism that builds and confirms
user profiles before inference while engaging with the user.
As shown in Figure 1, our proposed framework addresses
the complex challenge of personalized human-robot interac-
tion through a novel multi-stage reasoning approach. This
mechanism follows a deliberate chain of steps that tracks
different users, infers initial user models, validates or refines
these models through interaction, and ultimately delivers
personalized responses. Central to our approach is a four-
component architecture: (1) a user encoder that effectively
differentiates between user profiles; (2) a RAG (retrieval
augmented generation) process that infers user profiles and
leverages previous conversations; (3) a User decoder (USER-
VLM) that generates initial user models to address the ”cold
start” problem; and (4) a Chain of Thought reasoning process
that holistically considers all aspects of user profiles to
provide personalized responses. This framework specifically
distinguishes between priors and posteriori intuitions aboutarXiv:2504.01700v1  [cs.HC]  2 Apr 2025

users, enabling robots to adapt their responses cautiously,
for instance, by verifying assumptions, while progressively
refining their user models. To overcome the dual challenges
of personalization and privacy, our system autonomously
gathers and synthesizes user insights during interactions in
an implicit or explicit manner.
II. RELATED WORK
Recent advances in Large Language Models (LLMs) have
significantly impacted user modeling and personalization
approaches [11]. This section reviews the relevant literature
on LLM-based user modeling techniques, with emphasis
on RAG systems, model personalization frameworks, and
reasoning-enhanced approaches.
A. User Modeling with Large Language Models
User Models enable personalization in various applications
including recommendation systems, education, and health-
care. With the emergence of foundation models demonstrat-
ing superior performance in generating, understanding, and
reasoning over multimodal data, user modeling approaches
have been enhanced through integration with LLMs [11].
Current approaches to LLM-based user modeling can be
categorized into two main streams. The first approach incor-
porates user history and preferences directly into prompts or
uses retrieval systems to enhance contextual understanding
[12], [13]. The second approach tackles personalization from
a model-centric perspective by developing specialized user
encoders or modifying existing model architectures to inher-
ently provide personalized responses based on user features,
whether textual or visual [3], [7].
Retrieval-augmented generation (RAG) has emerged as a
powerful technique for enhancing the accuracy and reliability
of generative AI models by incorporating information from
specific and relevant data sources [14]. Several frameworks
have demonstrated the effectiveness of RAG in personal-
ization contexts. PersonaRAG [13] introduces user-centric
agents that adapt both retrieval and generation processes
based on real-time user data and interactions. Similarly,
the ”GenUI(ne) CRS” framework [15] leverages LLMs for
adaptive and interactive user interfaces, supporting domain-
specific graphical elements while addressing the knowledge
cut-off problem through RAG implementation. In domain-
specific applications, such as swimming coaching systems,
RAG has been utilized to capture user preferences and behav-
iors, significantly improving the quality of LLM outputs by
leveraging contextual and real-world knowledge [12]. These
applications demonstrate RAG’s versatility in enhancing per-
sonalization across diverse use cases.
Recent advancements in personalized LLMs emphasize
architecture design and training tailored to individual user
preferences and contextual histories. HYDRA [16] employs
model factorization to combine user-specific behavioral pat-
terns with general knowledge, using a reranker to prioritize
relevant historical data and an adapter to align outputs with
user preferences. User-LLM [8] introduces user embeddings
derived from self-supervised learning to capture latent user
Fig. 2: CoT Reasoning LLM vs Regular LLM [17]
behaviors and dynamically contextualize LLMs via cross-
attention mechanisms. For multimodal interactions, User-
VLM 360° [3] integrates user modeling with visual-linguistic
signals, bias-aware tuning, and socio-emotive datasets to
optimize real-time, contextually aware, and equitable inter-
actions.
B. Chain-of-Thought Reasoning LLMs
As shown in Figure 2, Chain-of-Thought (CoT) Reasoning
LLMs [17] differ from regular LLMs by breaking down
problems into smaller steps or thought processes before
generating responses. This structured approach to inference
has already been applied to user modeling through Chain-
of-thought prompting for personalization [18]. This method
enables LLMs to reason about user attributes, subjective
preferences, and intentions to create comprehensive user
models expressed in natural language [19]. Although recent
advances such as DeepSeek R1 [20] have demonstrated the
inherent capacity of LLMs for chain-of-thought reasoning,
the application of CoT reasoning LLMs specifically for
personalization and user modeling remains relatively unex-
plored, presenting an opportunity for further investigation.
III. METHODS
Our framework employs a RAG architecture to create
personalized user experiences through multimodal under-
standing. The system processes facial images and queries
to produce contextually relevant, personalized responses.
Figure 1 illustrates the architectural components and their
interactions.
A. User Encoder
The User Encoder is designed to encode facial images of
individuals and query the database to identify and retrieve the
corresponding user model and prior conversational context,
facilitating seamless integration into subsequent dialogue
prompts for enhanced interaction. The User Encoder is
a multimodal pre-trained model designed to encode user
profiles by extracting visual features from facial images. It
employs a vision encoder function EI:RdI×N→Rdh×N,
based on a Transformer architecture, to process images

TABLE I: Example of Data
Image User Profile Question Answer
The person appears to be a southeast
Asian female, approximately 60 to 69
years old.Are there assistance services for people
with mobility difficulties?Yes, many countries offer mobility assistance ser-
vices, including specialized transport and home sup-
port, tailored to seniors’ needs.
The person appears to be an Indian
male, approximately 60 to 69 years old.How do I report a fraudulent email? You can report a fraudulent email by forwarding it
to your email provider’s abuse department or using
the ”Report as Spam” feature in your email client.
represented as sequences of tokens in feature vectors. A
special cls token is used to derive a global image represen-
tation, which is subsequently mapped to an embedding space
through a multilayer perceptron projection head P:Rdh→
Rde, which yields the embedding eI. Cosine similarity is
utilized to measure the relationships between embeddings,
and the model is trained using a contrastive loss function
Lc.
B. User-VLM
The User-VLM [3] initiates user modeling in cold-start
scenarios where a user model is unavailable. By leveraging
facial images and user queries, it generates an initial user pro-
file comprising attributes such as age, gender, emotion, and
ethnicity. These models employ a Llava-style architecture
that seamlessly combines a vision encoder with a large lan-
guage model. The model undergoes user-aware fine-tuning
to enhance its ability to understand user-specific contexts
within visual-linguistic interactions. This user-aware tuning
enables the model to deliver tailored responses that align
with individual preferences and visual content. Formally,
the vision encoder Eprocesses user images XIinto vision
representations HI∈RdI, while the LLM generates textual
user profile outputs y={y1, y2, . . . , y L}based on tokenized
prompt HQ∈RdQand image embeddings HI.
C. Chain-of-Thought Reasoning LLM
The CoT Reasoning LLM performs two critical functions:
dynamically updating user profiles during interaction and
providing personalized responses. This component can be
defined as F:X → S × Y . Given an input x∈ X ,
it generates an ordered sequence of intermediate reasoning
steps s= (s1, s2, . . . , s k)along with a final answer y∈ Y.
Under a probabilistic framework, this process is modeled by
the joint distribution:
P(s, y|x) =k+1Y
i=1P(zi|x, z 1, . . . , z i−1) (1)
where for 1≤i≤k,zi=si(the reasoning steps) and
zk+1=y(the final answer). This formulation enables the
model to not only provide appropriate responses but also
maintain a dynamic understanding of user characteristics and
preferences through explicit reasoning. The reasoning LLM
takes as input the initial user profile generated by the User-
VLM, reassesses this profile based on ongoing interactions,and produces contextually appropriate responses tailored to
the updated user understanding.
IV. EXPERIMENTS
A. Dataset
To evaluate the personalization capabilities of the pro-
posed model, we utilize ElderlyTech-VQA Bench [3], which
comprises 144 triplets of images, questions, and answers,
focusing on real-world questions posed by elderly individuals
about technology. According to [3], the associated images,
sourced from the FairFace dataset, are curated to ensure
diversity in ethnicity and gender representation. Additionally,
reference answers for user queries were generated using
GPT-4o, following detailed instructions to deliver high-
quality and contextually relevant responses. Examples of this
dataset is shown in Table I.
B. Metrics
We selectively employ ROUGE [21] metrics to evaluate
the framework across different types of questions, as their use
provides a robust assessment of both factual consistency (via
lexical overlap) ensuring outputs meet the dual demands of
accuracy and adaptability in human-robot collaboration. We
also compare the ground truth responses with the proposed
model through human expert evaluation.
C. Baseline
The proposed model is evaluated against four state-of-the-
art models of comparable size to ensure a rigorous and fair
comparison. The first model, LLaMA 3.2 Vision [22], is an
advanced architecture based on CLIP [23] and LLaMA 3.1,
comprising 11 billion parameters. The second model, Pix-
tral [24], features a 12-billion-parameter multimodal decoder
built upon Mistral NeMo [25], along with a 400-million-
parameter vision encoder trained from scratch. Additionally,
the third and fourth models, LLaV A 1.5 [26] and LLaV A
1.6 [27], employ Mistral [28] and Vicuna [29] as their
respective backbones, each comprising 7 billion parameters
and integrating a CLIP-based vision encoder.
D. Experimental Setting
In our experiment, we employ six dense models distilled
from DeepSeek-R1 [20], leveraging both Llama and Qwen
architectures, with model sizes ranging from 1.5B to 70B
parameters. All experiments were conducted on a MacBook
Pro M4 MAX with 64 GB of unified memory. During

TABLE II: Comparative Study Results: Our framework achieves significantly superior performance in ROUGE F1 scores
comparing to baseline, with balanced precision and recall.
ModelROUGE-1 ROUGE-2 ROUGE-L
Precision Recall F1 Precision Recall F1 Precision Recall F1
LLaMA 3.2 0.1420 0.6068 0.2211 0.0514 0.2344 0.0817 0.1075 0.4629 0.1676
Pixtral 0.1489 0.6030 0.1934 0.0453 0.2270 0.0641 0.1117 0.4590 0.1448
LLaV A-v1.6 0.0956 0.6958 0.1658 0.0355 0.2685 0.0620 0.0730 0.5346 0.1267
LLaV A-v1.5 0.1256 0.6302 0.2035 0.0410 0.2207 0.0675 0.0943 0.4817 0.1535
User-LLM R1 (ours) 0.4294 0.5167 0.4531 0.1424 0.1677 0.1485 0.2376 0.2799 0.2478
TABLE III: Ablation Study Results: while the size of the User-VLM model (3B vs. 10B) has minimal impact on F1 scores,
larger reasoning models significantly improve performance, with higher parameter counts yielding superior ROUGE metrics.
Reasoning Model User-VLM ModelROUGE-1 ROUGE-2 ROUGE-L
Precision Recall F1 Precision Recall F1 Precision Recall F1
Distill-Qwen-1.5B3B 0.4058 0.2809 0.3141 0.0876 0.0592 0.0665 0.2262 0.1498 0.1700
10B 0.4122 0.2702 0.3064 0.0922 0.0599 0.0679 0.2389 0.1475 0.1703
Distill-Qwen-7B3B 0.4323 0.3768 0.3854 0.1130 0.0977 0.1005 0.2391 0.2016 0.2084
10B 0.4319 0.3895 0.3971 0.1097 0.0972 0.0996 0.2301 0.2029 0.2088
Distill-Llama-8B3B 0.4851 0.3709 0.4056 0.1520 0.1140 0.1255 0.2743 0.2046 0.2264
10B 0.4931 0.3837 0.4196 0.1539 0.1149 0.1275 0.2776 0.2092 0.2319
Distill-Qwen-14B3B 0.5035 0.4315 0.4498 0.1688 0.1428 0.1495 0.2913 0.2449 0.2576
10B 0.4922 0.4238 0.4392 0.1641 0.1385 0.1448 0.2851 0.2378 0.2501
Distill-Qwen-32B3B 0.4906 0.4471 0.4541 0.1612 0.1449 0.1477 0.2801 0.2490 0.2557
10B 0.4828 0.4396 0.4505 0.1580 0.1428 0.1468 0.2721 0.2444 0.2518
evaluation, we used greedy decoding with a temperature of
1.0 to generate responses. The maximum sequence length
was set to 512 tokens for both input and output.
E. Human Evaluation Methodology
The human evaluation involved three experts assessing
answers from two models—GPT-4o (the ground truth of
the dataset) and the proposed framework based on Llama
3.3 70B R1—on 144 questions from the ElderlyTech-VQA
Bench . These questions, asked by elderly individuals, were
tailored to specific profiles (age, gender, and ethnicity). Two
evaluators were domain-specific experts with experience in
elderly-accessible technologies, while the third was a general
technology expert. All evaluators were native French speak-
ers, and the English responses were translated into French,
with possible translation biases considered during scoring.
The questions addressed the use of digital and technological
services, focusing on challenges elderly individuals might
face, such as scanning documents or filing taxes online.
Responses were rated on a scale of 1 (not relevant) to
5 (personalized), based on relevance, adaptability to the
questioner’s profile, and suitability for an elderly audience.
Special attention was given to age-appropriate complexity,
avoiding stereotypes in personalization for gender and eth-
nicity. The evaluation was conducted blindly, with evaluators
unaware of which model provided each response. Each
expert assessed 288 answers (two responses per question),
totaling 864 scores across all evaluators.V. RESULTS
As presented in Table II, the proposed framework based on
Distill-Llama 3.3 70B R1 demonstrates significant improve-
ments over the baseline models across three variations of
the ROUGE metric. Specifically, the model achieves higher
F1 scores, with notable gains in ROUGE-1 (0.4531) and
ROUGE-L (0.2478), showcasing its enhanced capability in
capturing relevant information. In contrast, baseline models
such as LLaMA 3.2 and Pixtral exhibit comparatively lower
scores across all metrics, indicating the superior performance
and robustness of the proposed approach in text summa-
rization tasks. The high recall but low precision in other
models often indicates that they are prone to over-generation,
meaning they tend to label or predict more positives than
necessary. In simpler terms, these models may predict pos-
itive almost all the time, leading to high recall (capturing
all true positives) but at the expense of precision (many
false positives are included). The balanced precision and
recall in our result is a positive outcome as it suggests that
the proposed framework neither over-predicting nor under-
predicting the positive class excessively.
As depicted in Table III, the ablation study examines
12 experimental variations, involving six reasoning models
of varying parameter sizes and two configurations of the
User-VLM model (3B and 10B). The results indicate that
the evaluation scores, particularly F1, are not significantly
influenced by the size of the User-VLM model, as com-
parable performances are observed between the 3B and
10B configurations. However, the number of parameters

in the reasoning model has a direct and notable impact
on performance, with larger models consistently achieving
higher F1 scores across all ROUGE metrics. For instance,
Distill-Qwen-14B and Distill-Qwen-32B outperform smaller
models like Distill-Qwen-1.5B, underscoring the benefits of
increased parameterization in reasoning tasks.
As illustrated in Figure 3, the proposed framework with
70B active parameters achieves performance comparable
to GPT-4o, which utilizes approximately 200B parameters
[30], particularly in terms of personalization. A key dis-
tinction between the two models lies in how user profiles
are incorporated. In GPT-4o, the user profile is directly
provided to the model as part of the dataset, simplifying
the task of generating personalized responses. Conversely,
our framework leverages the innovative User-VLM module
to dynamically extract user profiles. This approach not only
enables adaptive personalization but also offers a scalable
method for accommodating new user profiles without requir-
ing pre-defined data. The reliance on User-VLM ensures that
the personalization process is more robust and context-aware,
which is critical for applications targeting elderly individu-
als who may have varying levels of technological literacy,
cultural backgrounds, and personalized needs. Moreover, the
significant reduction in parameter count—70B compared to
200B—highlights the efficiency of the proposed framework,
suggesting that it can achieve comparable levels of person-
alization while being computationally less demanding.
VI. ETHICAL CONSIDERATIONS
Generally speaking, VLMs that analyze facial characteris-
tics raise significant ethical concerns at both individual and
societal levels. A primary issue is the inference of cultural
preferences, personality traits, and social behaviors from
physical characteristics. If datasets are biased or not repre-
sentative, they can lead to misinterpretation, thus resulting in
reductionist or offensive categorizations that do not represent
individuals. On a larger scale, such practices can reinforce
systemic discrimination by associating physical features with
social stereotypes [31], [32]. Furthermore, the collection of
biometric data presents serious privacy and consent issues.
Unless individuals are adequately informed and/or explicit
consent is given, their basic rights, as protected by the
GDPR [33] and the European Convention on Human Rights
(ECHR) [34] for instance, can be violated. Moreover, the
danger of exploitation of such data in surveillance or social
control mechanisms is another critical issue [35], [36]. Po-
tential abuses can also include the threat of repression and
criminalization by facial recognition [37], the discriminatory
profiling of migrants or travelers at border controls [38], or
the potential exacerbation for discrimination in recruitment
processes [39].
The framework presented here allows for the mitigation
or avoidance of some ethical risks. The CoT Reasoning
LLM provides constant updates of the user profile and
personalized responses according to the interaction. It allows
for a dynamic user profiling through the collection of user
feedback, and thereby avoids excessive stereotyping based
Llama 3.3 70B R1 GPT4-o11.522.533.544.55 Llama 3.3 70B R1
GPT4-o
ModelsScoresFig. 3: Evaluation with Human Expert: Our framework
with 70B parameters and an adaptive User-VLM module,
achieves GPT-4o-level personalization at lower computa-
tional cost.
solely on physical features. Even during the interaction
between the user and the LLM, the model is trained to offer
several responses including some that are not related to facial
features. Furthermore, explicit consent is always obtained
before any data analysis.
However, ethical challenges still exist. Algorithmic bias
due to historical data can never be fully eradicated, and it is
difficult to precisely know if meaningful consent is really
given [40]. Continuous monitoring and the reassessment
of data are essential to promote fairness and equity [41].
Additionally, the use of these technologies must be regulated
through strict legal frameworks to prevent abuse [42].
VII. CONCLUSIONS
This paper introduces USER-LLM R1, a novel frame-
work designed to tackle the cold start problem in person-
alized human-robot interactions. By integrating chain-of-
thought reasoning with vision-language models, the system
dynamically generates and refines user profiles, enabling
contextually relevant and adaptive responses from the initial
encounter. Our evaluations on the ElderlyTech-VQA Bench
demonstrate significant improvements in ROUGE metrics
over state-of-the-art baselines, highlighting the efficacy of
our multi-stage reasoning approach. Ablation studies further
underscore the importance of reasoning model size in achiev-
ing enhanced performance. Human evaluations validate the
framework’s ability to generate personalized responses com-
parable to GPT-4o, particularly for elderly users, fostering
engagement and trust. While ethical considerations regard-
ing privacy and bias are thoroughly addressed, continuous
monitoring and refinement are essential for responsible de-
ployment. Future work will focus on expanding the frame-
work to incorporate additional modalities and enhancing its
adaptability to diverse user populations, ensuring equitable
and effective human-robot interactions.

ACKNOWLEDGMENT
The authors sincerely acknowledge the financial support of
the French National Research Agency (ANR) for the ANITA
project (Grant No. ANR-22-CE38-0012-01).
REFERENCES
[1] B. Irfan, A. Ramachandran, S. Spaulding, D. F. Glas, I. Leite, and
K. L. Koay, “Personalization in long-term human-robot interaction,”
in2019 14th ACM/IEEE International Conference on Human-Robot
Interaction (HRI) , pp. 685–686, IEEE, 2019.
[2] M. K. Lee, J. Forlizzi, S. Kiesler, P. Rybski, J. Antanitis, and S. Savet-
sila, “Personalization in hri: A longitudinal field experiment,” in
Proceedings of the seventh annual ACM/IEEE international conference
on Human-Robot Interaction , pp. 319–326, 2012.
[3] H. Rahimi, A. Bahaj, M. Abrini, M. Khoramshahi, M. Ghogho, and
M. Chetouani, “User-vlm 360: Personalized vision language models
with user-aware tuning for social human-robot interactions,” arXiv
preprint arXiv:2502.10636 , 2025.
[4] H. Rahimi, M. Abrini, M. Khoramshahi, and M. Chetouani, “User-
vlm: Llm contextualization with multimodal pre-trained user models,”
inAccepted in ToM4AI @ 39th Annual AAAI Conference on Artificial
Intelligence , 2025.
[5] G. Fischer, “User modeling in human–computer interaction,” User
modeling and user-adapted interaction , vol. 11, pp. 65–86, 2001.
[6] D. Caffagni, F. Cocchi, L. Barsellotti, N. Moratelli, S. Sarto,
L. Baraldi, M. Cornia, and R. Cucchiara, “The revolution of
multimodal large language models: a survey,” arXiv preprint
arXiv:2402.12451 , 2024.
[7] Y . Alaluf, E. Richardson, S. Tulyakov, K. Aberman, and D. Cohen-Or,
“Myvlm: Personalizing vlms for user-specific queries,” in European
Conference on Computer Vision , pp. 73–91, Springer, 2024.
[8] L. Ning, L. Liu, J. Wu, N. Wu, D. Berlowitz, S. Prakash, B. Green,
S. O’Banion, and J. Xie, “User-llm: Efficient llm contextualization
with user embeddings,” arXiv preprint arXiv:2402.13598 , 2024.
[9] C. Wongchokprasitti, J. Peltonen, T. Ruotsalo, P. Bandyopadhyay,
G. Jacucci, and P. Brusilovsky, “User model in a box: Cross-system
user model transfer for resolving cold start problems,” in User Model-
ing, Adaptation and Personalization: 23rd International Conference,
UMAP 2015, Dublin, Ireland, June 29–July 3, 2015. Proceedings 23 ,
pp. 289–301, Springer, 2015.
[10] R. Wang, Z. Wu, J. Lou, and Y . Jiang, “Attention-based dynamic user
modeling and deep collaborative filtering recommendation,” Expert
Systems with Applications , vol. 188, p. 116036, 2022.
[11] Z. Tan and M. Jiang, “User modeling in the era of large lan-
guage models: Current research and future directions,” arXiv preprint
arXiv:2312.11518 , 2023.
[12] C. Comendant, “Large language model-based sport coaching system
using retrieval-augmented generation and user models,” B.S. thesis,
University of Twente, 2024.
[13] S. Zerhoudi and M. Granitzer, “Personarag: Enhancing retrieval-
augmented generation systems with user-centric agents,” arXiv
preprint arXiv:2407.09394 , 2024.
[14] M. Arslan, H. Ghanem, S. Munawar, and C. Cruz, “A survey on rag
with llms,” Procedia Computer Science , vol. 246, pp. 3781–3790,
2024.
[15] U. Maes, L. Michiels, and A. Smets, “Genui (ne) crs: Ui elements
and retrieval-augmented generation in conversational recommender
systems with llms,” in Proceedings of the 18th ACM Conference on
Recommender Systems , pp. 1177–1179, 2024.
[16] Y . Zhuang, H. Sun, Y . Yu, R. Qiang, Q. Wang, C. Zhang, and
B. Dai, “Hydra: Model factorization framework for black-box llm
personalization,” arXiv preprint arXiv:2406.02888 , 2024.
[17] M. Grootendorst, “A visual guide to reasoning llms: Exploring test-
time compute techniques and deepseek-r1,” Maarten Grootendorst’s
Newsletter , 2025. Translations available in Chinese and Korean.
[18] F. Yang, Y . Yue, G. Li, T. R. Payne, and K. L. Man, “Chain-of-thought
prompting empowered generative user modeling for personalized rec-
ommendation,” Neural Computing and Applications , vol. 36, no. 34,
pp. 21723–21742, 2024.
[19] G. Li, F. Yang, and Y . Yue, “Identify user intention for recommen-
dation using chain-of-thought prompting in llm,” in Identify User
Intention for Recommendation using Chain-of-Thought Prompting in
LLM , Springer, 2024.[20] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma,
P. Wang, X. Bi, et al. , “Deepseek-r1: Incentivizing reasoning capability
in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948 ,
2025.
[21] C.-Y . Lin, “ROUGE: A package for automatic evaluation of sum-
maries,” in Text Summarization Branches Out , (Barcelona, Spain),
pp. 74–81, Association for Computational Linguistics, July 2004.
[22] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fan, et al. , “The llama 3 herd
of models,” arXiv preprint arXiv:2407.21783 , 2024.
[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. , “Learning transferable
visual models from natural language supervision,” in International
conference on machine learning , pp. 8748–8763, PMLR, 2021.
[24] P. Agrawal, S. Antoniak, E. B. Hanna, B. Bout, D. Chaplot, J. Chud-
novsky, D. Costa, B. De Monicault, S. Garg, T. Gervet, et al. , “Pixtral
12b,” arXiv preprint arXiv:2410.07073 , 2024.
[25] M. A. team, “Mistral nemo,” 2024. Accessed: 2024-01-29.
[26] H. Liu, C. Li, Q. Wu, and Y . J. Lee, “Visual instruction tuning,”
Advances in neural information processing systems , vol. 36, 2024.
[27] H. Liu, C. Li, Y . Li, and Y . J. Lee, “Improved baselines with visual
instruction tuning,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 26296–26306, 2024.
[28] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. ,
“Mistral 7b,” arXiv preprint arXiv:2310.06825 , 2023.
[29] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro, F. Azhar, et al. , “Llama:
Open and efficient foundation language models,” arXiv preprint
arXiv:2302.13971 , 2023.
[30] A. B. Abacha, W.-w. Yim, Y . Fu, Z. Sun, M. Yetisgen, F. Xia,
and T. Lin, “Medec: A benchmark for medical error detection and
correction in clinical notes,” arXiv preprint arXiv:2412.19260 , 2024.
[31] L. Rhue, “Racial Influence on Automated Perceptions of Emotions,”
SSRN Journal , 2018.
[32] R. Benjamin, Race after technology: abolitionist tools for the New Jim
Code . Cambridge, UK ; Medford, MA: Polity, 2020.
[33] G. D. P. Regulation, “Gdpr. 2019,” 2019.
[34] D. J. Harris, M. O’boyle, E. Bates, and C. Buckley, Law of the
European convention on human rights . Oxford university press, 2023.
[35] S. Zuboff, The Age of Surveillance Capitalism: The Fight for a Human
Future at the New Frontier of Power: Barack Obama’s Books of 2019 .
London: Profile Books Ltd, main ´edition ed., Sept. 2019.
[36] L. Introna and D. Wood, “Picturing Algorithmic Surveillance: The
Politics of Facial Recognition Systems,” Surveillance & Society , vol. 2,
no. 2/3, 2004.
[37] K. W. Bowyer, M. C. King, W. J. Scheirer, and K. Vangara, “The
“Criminality From Face” Illusion,” IEEE Transactions on Technology
and Society , vol. 1, pp. 175–183, Dec. 2020. Conference Name: IEEE
Transactions on Technology and Society.
[38] P. Molnar, “New technologies in migration: human rights impacts,”
2019.
[39] Z. Chen, “Ethics and discrimination in artificial intelligence-enabled
recruitment practices,” Humanit Soc Sci Commun , vol. 10, pp. 1–12,
Sept. 2023. Publisher: Palgrave.
[40] M. D. Dubber, F. Pasquale, and S. Das, The Oxford Handbook of
Ethics of AI . Oxford University Press, 2020. Google-Books-ID:
8PQTEAAAQBAJ.
[41] N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan,
“A Survey on Bias and Fairness in Machine Learning,” ACM Comput.
Surv. , vol. 54, pp. 115:1–115:35, July 2021.
[42] E. D. P. Supervisor, “Annual Report 2020,” european Data Protection
Supervisor, 2021.
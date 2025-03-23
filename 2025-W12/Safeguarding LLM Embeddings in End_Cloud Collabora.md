# Safeguarding LLM Embeddings in End-Cloud Collaboration via Entropy-Driven Perturbation

**Authors**: Shuaifan Jin, Xiaoyi Pang, Zhibo Wang, He Wang, Jiacheng Du, Jiahui Hu, Kui Ren

**Published**: 2025-03-17 07:58:05

**PDF URL**: [http://arxiv.org/pdf/2503.12896v1](http://arxiv.org/pdf/2503.12896v1)

## Abstract
Recent studies improve on-device language model (LM) inference through
end-cloud collaboration, where the end device retrieves useful information from
cloud databases to enhance local processing, known as Retrieval-Augmented
Generation (RAG). Typically, to retrieve information from the cloud while
safeguarding privacy, the end device transforms original data into embeddings
with a local embedding model. However, the recently emerging Embedding
Inversion Attacks (EIAs) can still recover the original data from text
embeddings (e.g., training a recovery model to map embeddings back to original
texts), posing a significant threat to user privacy. To address this risk, we
propose EntroGuard, an entropy-driven perturbation-based embedding privacy
protection method, which can protect the privacy of text embeddings while
maintaining retrieval accuracy during the end-cloud collaboration.
Specifically, to defeat various EIAs, we perturb the embeddings to increase the
entropy of the recovered text in the common structure of recovery models, thus
steering the embeddings toward meaningless texts rather than original sensitive
texts during the recovery process. To maintain retrieval performance in the
cloud, we constrain the perturbations within a bound, applying the strategy of
reducing them where redundant and increasing them where sparse. Moreover,
EntroGuard can be directly integrated into end devices without requiring any
modifications to the embedding model. Extensive experimental results
demonstrate that EntroGuard can reduce the risk of privacy leakage by up to 8
times at most with negligible loss of retrieval performance compared to
existing privacy-preserving methods.

## Full Text


<!-- PDF content starts -->

Safeguarding LLM Embeddings in End-Cloud
Collaboration via Entropy-Driven Perturbation
Shuaifan Jin†,≀Xiaoyi Pang♮Zhibo Wang†,≀,∗He Wang†,≀Jiacheng Du†,≀Jiahui Hu†,≀Kui Ren†,≀
†The State Key Laboratory of Blockchain and Data Security, Zhejiang University
≀School of Cyber Science and Technology, Zhejiang University
♮Hong Kong University of Science and Technology
shuaifanjin@zju.edu.cn, xypang@ust.hk, {zhibowang, wanghe_71, jcdu, jiahuihu, kuiren}@zju.edu.cn
Abstract
Recent studies improve on-device language model (LM) in-
ference through end-cloud collaboration, where the end de-
vice retrieves useful information from cloud databases to
enhance local processing, known as Retrieval-Augmented
Generation (RAG). Typically, to retrieve information from
the cloud while safeguarding privacy, the end device trans-
forms original data into embeddings with a local embedding
model. However, the recently emerging Embedding Inver-
sion Attacks (EIAs) can still recover the original data from
text embeddings (e.g., training a recovery model to map em-
beddings back to original texts), posing a significant threat
to user privacy. To address this risk, we propose EntroGuard,
an entropy-driven perturbation-based embedding privacy
protection method, which can protect the privacy of text
embeddings while maintaining retrieval accuracy during
the end-cloud collaboration. Specifically, to defeat various
EIAs, we perturb the embeddings to increase the entropy
of the recovered text in the common structure of recovery
models, thus steering the embeddings toward meaningless
texts rather than original sensitive texts during the recovery
process. To maintain retrieval performance in the cloud, we
constrain the perturbations within a bound, applying the
strategy of reducing them where redundant and increasing
them where sparse. Moreover, EntroGuard can be directly
integrated into end devices without requiring any modifi-
cations to the embedding model. Extensive experimental
results demonstrate that EntroGuard can reduce the risk of
privacy leakage by up to 8 times at most with negligible
loss of retrieval performance compared to existing privacy-
preserving methods.
1 Introduction
In recent years, large language models (LLMs) have been
widely utilized across diverse domains, including Healthcare,
Finance Analysis, Scientific Research, etc. Rising individual
privacy concerns and the implementation of privacy regu-
lations, such as the EU General Data Protection Regulation
∗Zhibo Wang is the corresponding author. Copyright may be transferred
without notice, after which this version may no longer be accessible.(GDPR [ 1]), have spurred the end-side deployment of LLMs.
However, end-side LLMs typically have a smaller param-
eter scale and limited capabilities, making them prone to
generating fictitious information during reasoning [ 49]. A
primary strategy for correcting such hallucinations is to in-
tegrate external factual knowledge into the user’s original
input through End-Cloud Collaboration, namely retrieval-
augmented generation (RAG). Specifically, end devices em-
ploy dedicated embedding models to convert raw sensitive
text input into human-unrecognizable embeddings [ 42] and
subsequently transmit them to the cloud to retrieve corre-
sponding knowledge from the cloud database to augment lo-
cal reasoning. Such embeddings enable knowledge retrieval
without disclosing original data.
However, recent research [ 4,16,24,41] proposed Embed-
ding Inversion Attacks (EIA), demonstrating that embed-
dings still pose privacy risks since they can be used to recover
users’ original sensitive text input. Such attacks can be cate-
gorized into two types, i.e., optimization-based attacks and
learning-based attacks. In optimization-based attacks [ 4,24],
a random text is initialized and then iteratively optimized
to make its corresponding embedding closely match the
target embedding. On the other hand, learning-based at-
tacks [ 16,41] involve training a recovery model, i.e., a gen-
erative model, to map embeddings back to their original text.
This type of attack is more threatening than the previous
one, as attackers often leverage powerful transformer-based
large language models with strong generative capabilities
for recovery. Hence, in end-cloud collaboration, embeddings
transformed from raw sensitive inputs may not be as secure
as previously assumed. Therefore, there is an urgent need
for a protection scheme that can effectively defend against
Embedding Inversion Attacks.
Currently, there is limited discussion on defenses against
Embedding Inversion Attacks in LLMs [ 15]. Some works [ 19,
41,44] protect embeddings by transforming them into an-
other form, such as an encrypted domain and frequency
domain. However, these approaches require additional cloud-
side computation to reverse the process so that the embed-
dings can be recognized by the cloud database. Therefore,
they are not applicable in real-world scenarios since thearXiv:2503.12896v1  [cs.CR]  17 Mar 2025

third-party cloud is hard to be compromised to complete the
reversal. Other defense works [ 24,48] achieve embedding
protection by adding noise into embeddings, but we find that
they can only defend against optimization-based EIAs while
failing to defend against more powerful learning-based EIAs.
In this paper, we aim to safeguard end-user privacy against
various EIAs while achieving precise external knowledge re-
trieval without additional cloud-side computation. To achieve
this, we need to overcome two main challenges that are mutu-
ally constrained. The first challenge is how to defend against
different EIAs universally, especially learning-based EIAs . Due
to the fundamental differences in attack mechanisms, it is
hard to defend against both types of attacks in a univer-
sal way. Moreover, learning-based EIAs may use various
recovery models, which further increases the difficulty of
successfully protecting embeddings. The second challenge
ishow to maintain the retrieval performance while effectively
protecting the privacy of embeddings . Protecting embeddings
inevitably alters them. However, excessive changes hinder
precise information retrieval from the cloud database, while
insufficient changes fail to safeguard privacy. In this context,
it is challenging to balance privacy protection and retrieval
performance.
To address the above challenges, we propose a plug-in
perturbation-based embedding privacy protection method
for end-cloud collaborative LLM reasoning, called Entro-
Guard. It can protect text embeddings against various EIAs
and maintain their retrieval performance without additional
cooperation from the cloud side. To this end, we first per-
turb the original text embedding to defeat the most power-
ful learning-based EIAs. Usually, learning-based EIAs adopt
a variety of transformer-based generative models to map
embeddings back to original texts, and transformer blocks
are the common structure in such models. To achieve ro-
bust defense against different attack models, we propose an
entropy-driven perturbation mechanism that perturbs the
embedding to increase the entropy of the recovered informa-
tion of the transformer blocks. This can effectively redirect
the embedding away from the original sensitive text and
steer it toward meaningless content throughout the recov-
ery process. Then, to ensure that the disturbed embedding
can still accurately retrieve the required external knowledge
from the cloud database, we propose a bound-aware pertur-
bation adaptation mechanism. It imposes strict constraints
on the perturbation intensity to prevent the perturbation
from being redundant or insufficient, thus remaining essen-
tial semantic information in embeddings while effectively
securing its privacy.
Besides, existing metrics may not adequately reflect text
privacy leakage in certain cases as they primarily focus on
comparing similarity on text level. To evaluate privacy leak-
age from the perspective of semantic similarity, we leveragethe concept of bidirectional entailment in semantic entropy
and propose a semantic entailment rate metric, called BiNLI,
as a complement to existing metrics.
Our main contributions can be summarized as follows:
•We propose a novel approach, EntroGuard, which can pre-
serve retrieval performance without requiring additional
cooperation from the cloud side and effectively protect
text privacy against various EIAs with different model
architectures simultaneously.
•We propose Entropy-based Perturbation Generation and
Bound-aware Perturbation Adaptation mechanisms, which
serve as plug-in modules that can be seamlessly integrated
into on-device black-box embedding models with accept-
able overhead to safeguard individuals’ privacy.
•We introduce BiNLI metric to facilitate a more compre-
hensive quantification of text privacy leakage. Extensive
experiments demonstrate that EntroGuard outperforms
the existing embedding protection methods in terms of
superior privacy protection performance by up to 8 times
at most with negligible retrieval performance loss.
2 Related work
This section first introduces privacy attacks in LLMs, with a
particular focus on the recently emerged embedding inver-
sion attacks in section 2.1. Then, we review existing protec-
tion schemes and their vulnerabilities in section 2.2.
2.1 Privacy attacks in LLMs
In the early years, there existed several works [ 7,31,35] pro-
posed for the recovery of specific private information. Song
et al. [ 35] implemented an attribute inference attack in lan-
guage models to obtain sensitive attributes in user input. Pan
et al. [31] implemented pattern reconstruction attacks with
fixed-pattern inputs, and keyword inference attacks, similar
to attribute inference attacks. Although these attacks can
obtain some sensitive information, their capabilities are lim-
ited by strict scenario requirements as they rely on specific
datasets or data formats, such as genomics and ID numbers.
Besides, in federated learning, Fowl et al. [ 7] extracted token
and position embeddings to retrieve high-fidelity text by de-
ploying malicious parameter vectors, but this type of attack
is not within the scope of our consideration in this article
due to different scenarios.
Recently, a new type of attack in large language models
has emerged, i.e., embedding inversion attacks [ 4,16,24,
25,41], aiming to revert original sensitive text inputs of
users. Such attacks can be categorized into optimization-
based EIAs and learning-based EIAs. In optimization-based
EIAs, Morris et al. [ 24] generated text that closely aligns with
a fixed point in the latent space when re-embedded. Building
on this, Chen et al. [ 4] extended such attacks to languages

other than English. Meanwhile, in learning-based EIAs, Li
et al. [ 16] reconstructed input sequences solely from their
sentence embeddings via the powerful generation capability
of existing LLMs. Morris et al. [ 25] reconstructed the input
text based on the next-token probabilities of the language
model. Wan et al. [ 41] also proposed a method to reconstruct
the text input from embeddings in deeper layers. In summary,
compared to previous privacy attacks that can only infer
specific words, embedding inversion attacks can recover the
whole sensitive text inputs of users, posing a much greater
privacy threat to text embeddings in end-cloud collaboration.
2.2 Privacy protection methods in LLMs
For LLMs, several works have been proposed to protect text
privacy, which can be broadly divided into two types, i.e.,
transformation-based and perturbation-based methods.
Transformation-based method always transforms the em-
bedding to another domain to prevent attacks, e.g., frequency
domain, where the protected embedding and the raw embed-
ding vary a lot. Liu et al. [ 19] use projection networks and
text mutual information optimization to safeguard embed-
dings, but they need another projection network to maintain
utility. Wan et al. [ 41] transform embedding into frequency
domain, which also means that the database in the cloud
must also be converted into the frequency domain format
accordingly. These approaches rely on additional coopera-
tion from clouds beyond retrieval, and become impractical
when additional reversal processes cannot be executed in
the third-party clouds or the reverse process goes wrong,
as the transformed embeddings no longer align with those
stored in the database.
Perturbation-based methods usually protect text privacy
by differential privacy. Xu et al. [ 43] add an elliptical noise
to the embedding space to balance privacy and utility in
word replacement. Li et al. [ 17] employ text-to-text priva-
tization by differential privacy in token embeddings to pri-
vatize users’ data locally. Du et al. [ 5] perturbed embed-
ding matrices in the forward pass of LMs. Meanwhile, some
perturbation-based methods focus on altering the original
meaning of the text to protect privacy. Zhou et al. [ 47] hid
private words through unrecognized words. Further, Zhou et
al. [48] add random perturbations to clustered word represen-
tation for privacy. However, neither of the above two types of
perturbation-based approaches can effectively defend against
emerging EIAs and ensure cloud retrieval performance at
the same time. Besides, there also exist several methods that
add adversarial perturbations [ 9,21] to protect privacy. How-
ever, adversarial samples in text [ 10] are usually generated
through modifications like insertion, deletion, swapping, or
paraphrasing. In such situations, the adversarial text usually
...
UsersCloud① Embedding
① ②② Sear ch result
Embedding
ModelLLMs
Embedding
ModelLLMs
Embedding
ModelLLMs
I'm planning a solo trip
to explore Japan,
starting with T okyo and
then heading to Kyoto, ...My therapist in London
knows I struggle with
PTSD after the accident,
...
I faked my resume to get
this job in New York, 
...
0.3-0.1-0.10.2
②②
① ①
...Embedding③ LLMs' answer
③ ③ ③End
Embedding
inversion
attackI faked my
resume to get this
job in New YorkRover ed Texts
Vector
DatabaseVector
Search
Normal ProcessMalicious ProcessFigure 1: In end-cloud collaboration, end users query
the vector database in the cloud for external knowledge
to obtain more convincing responses.
retains its original meaning, so privacy is still compromised
when the text is recovered by EIAs.
In summary, previous protective methods either required
the cloud to execute additional computation before retrieval
tasks or lacked strong defense capabilities when facing em-
bedding inversion attacks.
3 Preliminary
In this section, we first present typical end-cloud collabora-
tion system based on RAG in section 3.1, then introduce the
basic concepts of embedding models used in the end device
in section 3.2, and finally discuss the threat models that exist
in reality in section 3.3.
3.1 System model
In real-world application scenarios, the end device has mod-
els with relatively limited capabilities, while the cloud has
a large amount of interconnected knowledge. Therefore, it
is common to utilize end-cloud collaboration that deploys
embedding models on the end side and uses RAG to ob-
tain external knowledge from the cloud’s vector database,
to mitigate the hallucination problem of end-side LLMs and
improve the quality of their answers.
Specifically, in the service initialization phase, end devices
deploy embedded models that can project text into the same
embedding space as the cloud-side database for knowledge
retrieval. As illustrated in fig. 1, during the inference stage,
the end device receives the user’s private inputs, transforms
them into embeddings through the embedding model, and
uploads embeddings to the cloud. The cloud then receives
the embedding and performs a vector search on its stored
external data to retrieve the most relevant results, i.e. external

knowledge with factual information. Finally, the retrieved
external knowledge is subsequently transmitted back to the
end device, which then integrates it with the user’s input to
generate more reliable and informed responses using factual
information.
3.2 Embedding Model
The embedding model is a type of model that projects input
text from its original format into a numerical embedding,
such as a vector or matrix [ 22,33]. It is often used to re-
trieve similar knowledge, as it aims to retain semantic and
relational information of different texts in numerical space.
In the earlier years, one of the representative works of
the embedding model is Word2vec [ 23] that developed the
Continuous Bag-of-Words model and the Continuous Skip-
gram model to transform words into vector representations,
also referred to as word embeddings. In recent years, the
prevailing powerful embedding model typically follows dual
encoder paradigm [ 8,27,28] in which queries and documents
are encoded separately into a shared fixed-dimensional em-
bedding space [ 29] to get a better retrieval performance. They
usually adopt transformer architecture and use contrastive
learning with in-batch sampled softmax loss:
L=𝑒cos(𝑞𝑖,𝑝+
𝑖)/𝜏
Í
𝑗∈B𝑒cos
𝑞𝑖,𝑝+
𝑗
/𝜏(1)
where cosis denotes cosine similarity, Brepresents a mini-
batch of examples, and 𝜏is the softmax temperature, 𝑞𝑖rep-
resents the queries, and 𝑝+
𝑖represents the documents that
can be factual knowledge.
3.3 Threat model
Attack Scenarios. In this paper, we consider the cloud to be
honest but curious, which means that the cloud will honestly
perform the original retrieval process without performing
other additional computations, and may be curious to try to
recover the data uploaded by the user.
Adversaries’ capability. Under this scenario, the adver-
saries are clouds. Such adversaries are considerably powerful
as they almost know everything that was used to execute
EIAs. First, the adversary knows the end-side embedding
model𝑓, since the end device needs to adopt an embedding
model that can project text 𝑥into the same embedding space
as the database to access the cloud-side knowledge. Second,
they directly own the embedding 𝑓(𝑥)uploaded by the user.
Adversaries’ Strategy. According to the recent emerging
EIAs [ 4,16,24,41], the adversaries can recover the original
sensitive user text 𝑥from embeddings 𝑓(𝑥)mainly through
the following two approaches.Table 1: Reconstruction and retrieval capabilities under
different levels of Gaussian noise.
Noise
levelOptimization-
based EIAsLearning-
based EIAsRetrieval
0 ✓ ✓ ✓
10−3✓ ✓ ✓
10−2✗ ✓ ✓
10−1✗ ✗ ✗
The first approach is optimization-based attack methods.
It aims to retrieve the text ˆ𝑥whose embedding 𝑓(ˆ𝑥)is max-
imally similar to the ground truth of original embedding,
i.e.,e=𝑓(𝑥). The iterative process can be formalized as
an optimization problem, where the search for ˆ𝑥is guided
by the embedding model 𝑓, i.e., ˆ𝑥=arg max𝑥cos(𝑓(𝑥),𝑒).
Specifically, the text hypothesis for the next iteration will be
obtained from the text hypothesis for the current iteration:
𝑝
𝑥(𝑡+1)|𝑒
=∑︁
𝑥(𝑡)𝑝
𝑥(𝑡)|𝑒
𝑝
𝑥(𝑡+1)|𝑒,𝑥(𝑡),ˆ𝑒(𝑡)
,(2)
where the𝑒at the iteration 𝑡is the embedding of the 𝑥at
the iteration 𝑡, i.e., ˆ𝑒(𝑡)=𝑓 𝑥(𝑡).
The second method is to implement a learning-based at-
tack approach that aims to train a generative model Φto re-
verse the mapping of the embedding model 𝑓, i.e.,Φ(𝑓(𝑥))≈
𝑓−1(𝑓(𝑥))=𝑥.Generally, it mainly entails two steps. First,
the attacker extracts text-embedding pairs (𝑥,𝑓(𝑥))from the
corpusX={𝑥1,...,𝑥𝑛}via the embedding model 𝑓in the
end device. Then these text-embedding pairs are utilized to
train the generative model Φby decreasing the cross entropy
between original and recovered texts:
𝐿𝑐𝑒(𝑥;𝜃Φ)=−𝑢∑︁
𝑖=1log(P(𝑤𝑖|𝑓(𝑥),𝑤0,𝑤1,...,𝑤𝑖−1)),
(3)
where𝑥=[𝑤0,𝑤1,...,𝑤𝑢−1]represents a sentence of length
𝑢. Finally, the trained generative model can be utilized to
recover the users’ sensitive input from the embeddings sent
by the end device, i.e., ˆ𝑥=Φ(𝑓(𝑥)).
Notably, observation from Morris et al. [ 24] showed that
introducing a certain level of noise can help defend against
optimization-based EIAs. However, we found that this ap-
proach is ineffective against learning-based EIAs via pre-
experiments shown in table 1. Considering that learning-
based EIAs not only exhibit broader applicability but also
demonstrate superior attack capabilities, we prioritize the
prevention of learning-based attacks in this paper.

Knowledge 
Retrieval
Bound-awar e Original Text
Embedding
Space
Embedding SpaceIs the 17-mm
St. Jude
Medical
Regent valve a
valid option
forpatients
with a
small aortic
annulus?None-sentive domain
None-Sensitive TextSurr ogate Attacker  Model
<|endoftext|>
<|endoftext|>
Recover ed TextEmbedding
Model
Protected
Embedding
Self Attention
Feed Forward Neural Network
···
Self Attention
Feed Forward Neural Network
Self Attention
Feed Forward Neural NetworkEntr opy-based Perturbation Generation
BoundBound
Databse
Is the ****m St. Jude
****** a valid option
***** with a
small *******?Embedding
Space
Original embedding pr ocess
Perturbation
Generator
Original domain
Perturbation Adaptation
Figure 2: Pipeline of EntroGuard where the dashed arrows indicate the training process and the realization arrows
indicate the inference process. During the training phase, a surrogate attacker model was built for optimizing
perturbation generator in Entropy-based Perturbation Generation. In the inference phase, the original text is
converted into an embedding through the embedding model and then processed by EntroGuard, including Entropy-
based Perturbation Generation, Bound-aware Perturbation Adaptation, resulting in a protected embedding.
4 EntroGuard
In this section, we provide our approach, i.e., EntroGuard,
to protect embeddings’ privacy while maintaining retrieval
capabilities without the cloud’s additional cooperation. Note
that the process of generating the final answer on the end
device is beyond our consideration, as this process occurs
entirely on the end device with nearly no privacy implica-
tions. In section 4.1, we overview our proposed embedding
protection method. Next, we introduce how we generate
robust entropy-based perturbation to protect embeddings
from being recovered in section 4.2. Finally, we introduce
the method we implemented to further constrain the inten-
sity of the perturbation to increase the privacy-preserving
capability while guaranteeing the retrieval capabilities in
section 4.3.
4.1 Overview
Our main idea is to make the embedding in the recovery
process away from the original sensitive text and steer it to-
ward meaningless text. An intuitive idea is to use adversarial
sample methods, e.g. FGSM [ 9], PGD [ 21], against the recov-
ery process. However, we find that it has poor generality to
resist unknown generative models. To this end, we choose tointerfere with the essential components of the transformer
architecture to achieve robust privacy-preserving embedding
generation. Meanwhile, we also impose a strict constraint
on the degree of perturbation to maintain the retrieval per-
formance, as the perturbation intensity is the main factor
that determines whether it can be retrieved correctly.
The whole process of EntroGuard is illustrated in fig. 2.
When the user inputs the original texts into the end devices,
they are first converted into raw embeddings by the origi-
nal embedding model. These embeddings then go through
EntroGuard that contains two components, i.e., Entropy-
based Perturbation Generation and Bound-aware Perturba-
tion Adaptation, resulting in a protected embedding.
Specifically, the first component is used to perturb raw
embeddings to disrupt the recovery process in recovery mod-
els. In particular, we maximize the information entropy of
intermediate results in transformer blocks, making the em-
bedding generate meaningless words as early as possible
during the recovery process. And we also increase cross-
entropy to make the recovered text deviate further from the
original sensitive text. The second component is used to con-
strain the strength of the perturbation with the strategy of
"reducing where redundant and increasing where sparse".

* what *  is *  a *  rib   steak * <|endoftext|>1
3
5
7
9
11
13
15
17
19
21
23
25
27
29
31
33
35ven  kind  a  big bing  steak
ven  kind  a  big bing  steak
ven  r  a  good bing  steak
ven  is  a  good bing house
 in  is  a  good bing house
 in  is  a  whole bing house
 in  is  a  good bing .
 in  is  a  ' bed .
 and  is  a  great bing .
.  is  a  pre bing .
,  is  a  black bing .
,  is  a  red bing <|endoftext|>
,  is  a  steak  steak <|endoftext|>
.  is  a  beef  steak <|endoftext|>
.  is  a  beef  steak <|endoftext|>
 a  is  a  rib  steak <|endoftext|>
 what  is  a  rib  steak <|endoftext|>
what  is  a  rib eye <|endoftext|>
0.00.20.40.60.81.0[EMB] what  is  a  rib  steakFigure 3: The internal recovery process of EIA, the
top is the input, the bottom is the output, and vertical
axis represents the Transformer block. The darker the
color means the higher the confidence of the generated
results, and the words marked with * indicate that the
prediction is correct.
By reducing redundant perturbation, it preserves essential
semantic information in embeddings to guarantee retrieval
performance. Meanwhile, by increasing perturbation that is
sparse, the protected embeddings will gain a certain degree
of randomness to enhance their resilience against EIAs.
4.2 Entropy-based Perturbation Generation
The key to the success of EIAs lies in the powerful generative
capabilities of the models they employ. To better prevent
EIAs, it is crucial to understand how generative language
models recover the original sentence from an embedding. To
this end, we use an interpretability tool called Logits Lens [ 2]
that can display the output of the intermediate process in
the generative model. As illustrated in fig. 3, recovery is not
always a straightforward process where the correct token is
predicted only at the end. Instead, we find that some tokens
may be restored correctly during the middle or even earlier
generation stages, which is similar to the normal generative
process itself.
This observation led us to the idea of interfering with the
text recovery process from the outset to guide the recovery
model producing non-sensical words. However, since the
generative models deployed by real-world adversaries may
vary (e.g., GPT [ 30], Llama [ 39]), a defense designed against
a specific attack model may fail to withstand attacks from
Transformer  block 4
Transformer  block 5
Transformer  block 6<|endoftext|>
<|endoftext|>
<|endoftext|>Normalization layer  ···
···Intermediate r esultsFigure 4: The process of generating intermediate re-
sults of each layer of transformer blocks, where the
intermediate results converge towards meaningless
words via Entropy-based Perturbation Generation.
different models. Considering that the mainstream genera-
tive language models are always built on the transformer
architecture, we perturb embeddings against the fundamen-
tal components shared across generative language models,
i.e., the transformer block, to make perturbation adaptable
to a wide range of generative attack models. Eventually, to
achieve privacy protection, we perturb embeddings by in-
creasing the information entropy of intermediate results in
transformer blocks to steer them toward meaningless con-
tent and increasing the cross-entropy of the final recovered
text to keep them away from the original sensitive text.
To implement this, we aim to optimize a perturbation
generator𝐺to perturb embeddings for privacy preservation.
First of all, we establish a surrogate model 𝑆to approximate
the attacker’s generation process, which can be any of the
mainstream generative language models. Such a surrogate
model𝑆is trained with public dataset X={𝑥1,...,𝑥𝑛}via
eq. (3), aiming to recover the texts from embeddings, i.e.,
𝑆(𝑓(𝑥))≈𝑓−1(𝑓(𝑥))=𝑥.
As illustrated in fig. 4, in order to supervise the trans-
former blocks [ 2] at different layers in recovery process, we
leverage the activation layer of the final layer in generative
language models to extract the intermediate results, i.e., word
distribution 𝐷𝑖𝑠𝑡𝑟(), of transformer blocks in different layer
𝑘throughout the generative process:
Distr(𝑘)=Softmax(ln𝑠(𝑏𝑘)𝐷)∈R|vocabulary|, (4)
where ln𝑠denotes the final normalization layer of the surro-
gate model𝑆before the decoding matrix 𝐷, and𝑏𝑘represents
the output vectors of the 𝑘thtransformer blocks.
After obtaining the intermediate results, we increase their
information entropy to steer them toward meaningless texts.
Specifically, we utilize the average information entropy of
the intermediate results as the loss function:
𝐿𝑒𝑛𝑡𝑟𝑜𝑝𝑦 =𝑛∑︁
𝑘=1(𝑢∑︁
𝑗=1(−|vocabulary|∑︁
𝑖=1𝑝(𝑤𝑖)log2𝑝(𝑤𝑖))),(5)

where𝑢represents the length of the sentence, 𝑛represents
the total number of transformer blocks to be supervised,
and𝑝(𝑤)∈𝐷𝑖𝑠𝑡𝑟(𝑘)is the probability of occurrence of the
word𝑤. Under this loss function, the supervision from each
intermediate result has a cumulative effect on the shallower
transformer blocks, resulting in more substantial gradient
constraints at shallow blocks, which renders the recovery
process ineffective at an early stage. Additionally, we also
increase the cross-entropy 𝐿𝑐𝑒(shown in eq. (3)) to keep the
generated sentence distinct from the original text.
On the other hand, as the retrieval performance needs
to be preserved at the same time, the perturbation should
be kept as minimal as possible. Therefore, we constrain the
strength of the perturbation by cosine similarityA·B
∥A∥∥B∥:
𝐿𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 =1−A·B
∥A∥∥B∥, (6)
where AandBare the embeddings to be compared, and ∥A∥
and∥B∥are the euclidean norms of AandB.
Eventually, as shown in fig. 1, the optimization objectives
for perturbation generator 𝐺to generate entropy-based per-
turbation on embeddings can be represented as:
min
𝐺𝐿𝑜𝑠𝑠=𝛼·𝐿𝑠𝑖𝑚−𝛽·𝐿𝑒𝑛𝑡𝑟𝑜𝑝𝑦−𝛾·𝐿𝑐𝑒 (7)
where𝛼,𝛽,𝛾are hyperparameters that control the ratios of
each loss.
4.3 Bound-aware Perturbation Adaptation
Since the entropy-based perturbation, which achieves pri-
vacy protection, may affect retrieval performance, we need to
further adapt the perturbation to meet the retrieval require-
ment simultaneously. As mentioned by Morris et al. [ 24],
there is an existing perturbation bound that both maintains
retrieval performance and defends optimization-based EIAs.
Although the method they proposed cannot defend against
learning-based attacks, as shown in table 1, the retrieval
bound𝜖they measured can still serve as a reference for us.
Hence, we propose an algorithm called Bound-aware Per-
turbation Adaptation to keep the perturbations within the
bound, with a strategy of "reducing where redundant and
increasing where sparse". In particular, we reduce perturba-
tions that exceed the bound to ensure retrieval performance
while increasing perturbations that do not reach the bound
to achieve better protection.
As illustrated in algorithm 1, if the perturbed embedding
𝑒′given by Entropy-based Perturbation Generation is below
the predefined bound, we will randomly select a subset of
dimensions in the embedding space (line 3) and inject a ran-
dom Gaussian noise (line 4) into these dimensions. The initial
values of the Gaussian noise are relatively large compared
to the embedding and will gradually adjust until the overall
perturbation remains within the predefined bound (line 6 toAlgorithm 1: Bound-aware Perturbation Adaptation
Input: e0: Raw embedding, e′: Perturbed embedding,
𝜖: Total perturbation bound, 𝜌: Scaling factor
(𝜌<1).
Output: e′′: Adapted perturbed embedding.
1Compute similarity 𝐿𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 :𝜃←cos(e,e′);
2ifPerturbation intensity below bound: 𝜃<𝜖then
3 Select random index set 𝐷={𝑑0,𝑑1,..,𝑑𝑘}ine′;
4 Generate random noise 𝑛∈{𝑛+∼N( 1,1),
𝑛−∼N(− 1,1)}inrandom index set 𝐷;
5 Initialize adapted embedding: 𝑒′′←𝑒′+𝑛;
6 while𝜃←𝑐𝑜𝑠(𝑒0,𝑒′′)>𝜖do
7 Decrease the intensity if noise𝑛by𝑛←𝑛×𝜌;
8 Update embeddings: 𝑒′′←𝑒′+𝑛;
9else
10 Initialize adapted embedding: 𝑒′′←𝑒′;
11 while𝜃←𝑐𝑜𝑠(𝑒0,𝑒′′)>𝜖do
12 Get noise intensity: 𝐼𝑛𝑜𝑖𝑠𝑒←(𝑒′−𝑒0)×𝜌;
13 Get new embedding: 𝑒′′←𝑒0+𝐼𝑛𝑜𝑖𝑠𝑒;
14 Return embedding 𝑒′′;
line 8). If the perturbation exceeds the bound, we iteratively
scale it down in equal proportion using an exponential decay
(line 11 to line 13) until it falls within the acceptable bound.
After the above adaption, all perturbations will remain close
to and within the bound, ensuring that the protected em-
bedding retains its retrieval performance without additional
cooperation in the cloud.
5 Experiments
To evaluate the effectiveness of EntroGuard, we conduct ex-
tensive evaluations of retrieval performance and privacy-
preserving capability across different embedding models
with different datasets. In this section, we will first intro-
duce the setup of our experiments in section 5.1, and then
we demonstrate the superiority of EntroGuard by answering
the following research questions:
•RQ1 : Does EntroGuard influence retrieval performance
in the cloud? (section 5.2)
•RQ2 : Can EntroGuard efficiently defend against embed-
ding inversion attacks? (section 5.3 and section 5.4)
•RQ3 : Can EntroGuard efficiently process embeddings in
the real-world end devices with low costs? (section 5.5)
Specifically, as privacy protection will be meaningless
without effective retrieval, we first assess the retrieval perfor-
mance of our method. Then, we evaluate its privacy protec-
tion capabilities under the premise of maintaining retrieval.
Finally, we validate the method’s efficiency on end devices.

5.1 Experimental Setup
Datasets. In our experiments, we utilize Persona-Chat [ 45]
and MS MARCO [ 26] for evaluating privacy-preserving ca-
pability, and additionally add part of the datasets in the BEIR
benchmark [ 37], i.e., arguana [ 40], Fever [ 38] to the eval-
uation of retrieval performance. Further details about the
datasets can be found in appendix A.
The implementation of EntroGuard. In experiments, we use
DenseNet [ 11] as the backbone of the perturbation gener-
ator in EntroGuard. Specifically, we adopt the Adam opti-
mizer [ 12] configured with a weight decay of 0.01 (exclud-
ing parameters related to "bias", "LayerNorm", and "weight"
terms), along with an initial learning rate of 3×10−5. A linear
scheduler is then used to dynamically reduce the learning
rate from the initial value set in the optimizer to 0. The
weighting coefficients 𝛼,𝛽, and𝛾undergo dynamic adjust-
ment during network convergence. During the initial train-
ing phases, these parameters are set in a ratio of 1200:1:1,
which gradually evolves to approximately 300:2:1 as the net-
work converges, with the exact ratio depending on the char-
acteristics of the specific embedding model. Additionally, if
not specified in the experiments, 𝜌is set to 0.95, and 𝜖to
0.036 that corresponds to the intensity of 0.01x Gaussian
noise with a mean of 0 and a variance of 1.
The implementation of Attack methods. In our experiments,
we adopt the representative work of learning-based EIAs, i.e.,
GEIA [ 16], as the primary attack method due to its strong
privacy attack capability. Moreover, we also use the represen-
tative work of optimization-based attack, i.e., Vec2Text [ 24]
in our evaluations.
Embedding models & Recovery models. For embedding mod-
els on the end device, we employ widely used models, in-
cluding Sentence-T5 [ 27], SimCSE-BERT [ 8], RoBERTa [ 20],
and MPNet [ 36] for learning-based EIAs, as well as GTR-
T5 [28] for optimization-based EIAs, all utilizing their of-
ficially trained versions. For adversaries’ recovery models,
we utilize GPT-2 [ 34], DialoGPT [ 46], and Llama3 [ 6] in
learning-based EIAs, which are all trained on datasets from
the same domain as the victim data to ensure optimal attack
performance.
Baseline Defense Methods. We compare our method with
three related protection methods, i.e., Differential Privacy [ 14],
PGD [ 21], Textobfuscator [ 48]. Specifically, Differential Pri-
vacy adds random Gaussian noise to embeddings to protect
embedding privacy. Following the setting in [ 24], we use a
privacy budget of 0.01. PGD utilizes projected gradient de-
scent to add adversarial noise to the embedding and protect
privacy by disturbing specific attack models. In this method,
we also use a noise strength consistent with DP, i.e., 0.01.Textobfuscator ensures inference privacy by introducing per-
turbations to the clustered embeddings. To better adapt to
our scenario, we transformed its task objective from the
original classification tasks to retrieval tasks.
Evaluation metrics. For evaluating the privacy-preserving
capability of the embedding, we utilize metrics that compare
text similarity, i.e., ROUGE [ 18], BLEU [ 32], EMR (Exact
Match) [ 24]. Higher text similarity indicates more privacy
leakage. Specifically, ROUGE is a metric that evaluates the
overlap of unigrams (individual words) between the system-
generated summary and the reference summary. BLEU is an
algorithm designed to assess the quality of text that has been
machine-translated from one natural language to another,
comparing the output to reference translations. EMR (Exact
Match) quantifies the percentage of generated outputs that
exactly match the ground truth. In the following experiments,
we specifically utilize ROUGE-2 and BLEU-2.
However, semantic leakage in text also constitutes a form
of privacy leakage. Since the above metrics mainly assess pri-
vacy leakage based on text similarity (e.g., n-gram matching),
they may not effectively capture deeper semantic similarities
in certain cases. For example, two sentences may be nearly
the same in vocabulary but express different meanings, in
which case ROUGE and BLEU will give a high similarity
score. In order to measure whether privacy has been compro-
mised from the semantic similarity, we propose a new metric,
BiNLI. Inspired by Kuhn et al. [ 13], we use the Natural Lan-
guage Inference (NLI) Model to determine the entailment
relationship between two statements to reflect their semantic
similarity. Specifically, BiNLI can be calculated as:
BiNLI =entail(𝑎,𝑏)+entail(𝑏,𝑎)
|{A,B}|, (8)
where an existing sentence 𝑎and sentence 𝑏are considered
to be semantically related when 𝑎is entailment of 𝑏, i.e.,
entail(𝑎,𝑏)or𝑏is entailment of 𝑎, i.e,entail(𝑏,𝑎).|{A,B}|
is the total numbers of samples in dataset A={𝑎0,𝑎1,...}
andB={𝑏0,𝑏1,...}. In the following evaluations, we also
use the proposed BiNLI to measure the privacy leakage of
texts, where higher BiNLI indicates more privacy leakage.
Moreover, to evaluate retrieval performance, we use NDCG
to assess how well the actual retrieval ranking aligns with the
ideal order, MAP to assess average precision by considering
both the number of relevant knowledge and their position in
the retrieval, and Precision to measure the proportion of rel-
evant results in retrieved knowledge. Specifically, we utilize
NDCG@5, MAP@5, and Precision@5 in our experiments.
5.2 Retrieval performance
In this section, we compare the retrieval performance of our
proposed EntroGuard with original embedding models of
different architectures.

Table 2: The performance of retrieval capability in cloud database in terms of NDCG, MAP, and Precision with
different embedding models in end side. The higher the values of NDCG, MAP, and Precision, the better the
retrieval performance.
Embedding
modelMethodArguana Fever Msmarco
NDCG ↑MAP ↑Precision ↑NDCG ↑MAP ↑Precision ↑NDCG ↑MAP ↑Precision ↑
Sentence-t5Original 0.3364 0.2853 0.0983 0.3220 0.2832 0.0915 0.5084 0.0496 0.5954
Ours 0.3464 0.2937 0.1011 0.3237 0.2867 0.0908 0.4933 0.0478 0.5954
Simcse-bertOriginal 0.3520 0.3010 0.1013 0.1621 0.1394 0.0478 0.2848 0.0319 0.4000
Ours 0.3484 0.2988 0.0997 0.1690 0.1455 0.0497 0.2651 0.0304 0.3674
MPNetOriginal 0.4534 0.3948 0.1260 0.5638 0.5080 0.1515 0.6928 0.0904 0.8140
Ours 0.4517 0.3946 0.1248 0.5592 0.5038 0.1502 0.6953 0.0901 0.8186
RobertaOriginal 0.3852 0.3293 0.1108 0.5011 0.4521 0.1344 0.6706 0.0729 0.7861
Ours 0.3837 0.3271 0.1110 0.4878 0.4387 0.1317 0.6736 0.0738 0.7814
Original w/o BPA with BPA0.00.10.20.30.40.5NDCG Score
MPNet
Roberta
Original w/o BPA with BPA0.00.10.20.30.40.5MAP Score
MPNet
Roberta
Figure 5: The retrieval performance on Fever dataset.
EntroGuard can maintain high retrieval performance.
To protect end-side privacy from the honest but curious
server that can invade users’ privacy, we retrieve the origi-
nal knowledge embeddings with protected embeddings from
the end side. Specifically, our methods are trained with Ms-
macro and evaluated by various datasets, i.e., Arguana, Fever,
and Msmacro.
Since retrieval performance is primarily related to the sim-
ilarity between the relevant query and the corpus, i.e., the
similarity at the embedding (vector) level, we set the bound 𝜖
in our methods to 0.036 to maintain the retrieval performance
based on the retrieval bound analysis by Morris et al. [ 24]
and Chen et al. [ 4]. Table 2 shows the results of retrieval per-
formance. All the metrics of our method remain close to that
of the original models (e.g., Sentence-T5: 0.3464 vs. 0.3364,
SimCSE-BERT: 0.3484 vs. 0.3520), whose results demonstrate
that EntroGuard effectively preserves retrieval performance
across different embedding models. The retrieval results of
other schemes can be seen in Appendix B, whose results
also align with the analysis that the retrieval performance
can be maintained by controlling the perturbation intensity.
Therefore, the key to evaluation lies in whether these ap-
proaches can achieve privacy protection under the condition
that retrieval performance is maintained.Besides, Figure 5 shows the role of Bound-aware Perturba-
tion Adaptation (BPA) in maintaining retrieval performance.
Without BPA, there is a significant drop in retrieval metrics,
and with BPA, the retrieval metrics are almost identical to
those of the original embedding model, demonstrating the
effectiveness of BPA in maintaining retrieval performance.
5.3 Privacy-preserving capabilities
In this section, we first show the reliability of the BiNLI
metrics in certain situations by analyzing examples. Then,
we demonstrate the privacy-preserving capabilities of our
method across different embedding models.
Metric BiNLI can reflect privacy leakage between
sentences from semantic similarity. The reconstructed
sentences in Table 3 demonstrate two scenarios in which
only utilizing text similarity to measure the degree of privacy
leakage may fail. The first scenario involves cases where the
text has undergone significant changes, but the semantics
remain largely intact. For example, in the sentence "what
can diarrhea meal be?" , the words "diarrhea" and "meal" still
convey sensitive information, similar to the original phrase
"attack of diarrhea" and "eat". In this case, traditional metrics
like BLEU may yield a relatively low score, indicating little
privacy disclosure in terms of text similarity. However, a
BiNLI score of 0.5 reflects the privacy leakage in this sce-
nario more accurately, indicating that partial privacy has
been compromised in terms of semantic similarity. The sec-
ond scenario, in contrast, occurs when textual changes are
minimal, but the semantics shift substantially. For instance,
in"I thought you said you play guitar. i am into baking" , the
keywords "guitar" and "baking" differ substantially from the
original keywords "gaming" and "gamer". Here, traditional
metrics might remain high scores, potentially misrepresent-
ing the privacy risk. Conversely, our proposed BiNLI score

Table 3: The examples of privacy-preserving capabilities against learning-based EIAs. The lower the values of
BLEU, BiNLI, the better the privacy-preserving performance.
User’s Text what can you eat when you have an attack of diarrhea BLEU BiBLI
Original what can you eat when you have diarrhea<|endoftext|> 0.6363 1.00
DP what can you eat when you have severe diarrhea<|endoftext|> 0.6538 1.00
PGD what can diarrhea meal be<|endoftext|> 0.1167 0.50
TextObfuscator what can you eat when you have diarrhea<|endoftext|> 0.6363 1.00
Ours <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|> 0.0000 0.00
User’s Text i thought you said you would take up gaming , you know i am a gamer BLEU BiBLI
Original oh i thought you said you were into gaming. i will become a gamer<|endoftext|> 0.4064 0.50
DP i thought you said you were into gaming . i will play .<|endoftext|> 0.3364 0.50
PGD oh okay . i thought you said you play guitar . i am into baking .<|endoftext|> 0.3819 0.00
TextObfuscator oh i thought you said you were a gamer . i am into gaming .<|endoftext|> 0.5000 0.50
Ours <|endoftext|>vere<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|> 0.0000 0.00
Table 4: The performance of privacy-preserving capabilities against learning-based EIAs in terms of ROUGE, BLEU,
EMR, and BiNLI with different embedding models. The lower the values of ROUGE, BLEU, EMR, and BiNLI, the
better the privacy-preserving performance.
Embedding
modelMethodPersonal Chat Msmacro
ROUGE ↓BLEU ↓ EMR ↓ BiNLI ↓ROUGE ↓BLEU ↓ EMR ↓ BiNLI ↓
Sentence-t5Original 0.4620 0.3253 0.1841 0.5608 0.6427 0.6907 0.3377 0.7370
DP 0.3969 0.4309 0.1430 0.4783 0.5439 0.6045 0.2205 0.6450
PGD 0.0959 0.1413 0.0002 0.1860 0.0943 0.1525 0.0000 0.1676
TextObfuscator 0.4599 0.4894 0.1776 0.5630 0.6396 0.6869 0.3261 0.7360
Ours 0.0002 0.0015 0.0000 0.0091 0.0000 0.0000 0.0000 0.0008
Simcse-bertOriginal 0.4695 0.5155 0.1897 0.5619 0.6272 0.6798 0.3197 0.6432
DP 0.4684 0.5148 0.1890 0.5627 0.6264 0.6796 0.3187 0.6433
PGD 0.3082 0.3612 0.0422 0.4205 0.3991 0.4597 0.0411 0.4415
TextObfuscator 0.4139 0.4678 0.1446 0.5126 0.6080 0.6702 0.2965 0.6373
Ours 0.0068 0.0166 0.0000 0.0338 0.0018 0.0028 0.0000 0.0618
MPNetOriginal 0.4040 0.4429 0.1564 0.4518 0.5642 0.6223 0.2479 0.6776
DP 0.3812 0.4195 0.1434 0.4243 0.5222 0.5875 0.2040 0.6433
PGD 0.1456 0.1350 0.0017 0.2063 0.1101 0.1768 0.0002 0.2389
TextObfuscator 0.3977 0.2851 0.1508 0.4479 0.5533 0.6176 0.2333 0.6770
Ours 0.0131 0.0017 0.0012 0.0191 0.0314 0.0432 0.0024 0.1386
RobertaOriginal 0.3679 0.3970 0.1360 0.4064 0.5370 0.5978 0.2168 0.7097
DP 0.3738 0.3407 0.1233 0.3738 0.5577 0.4902 0.1744 0.6676
PGD 0.0914 0.1378 0.0004 0.1445 0.0635 0.1128 0.0000 0.1530
TextObfuscator 0.3586 0.3955 0.1293 0.3988 0.5163 0.5863 0.1915 0.7060
Ours 0.0001 0.0000 0.0000 0.0008 0.0033 0.0084 0.0001 0.0801
of 0 effectively reflects the lack of semantic overlap, indicat-
ing that privacy has been preserved in terms of semantic
similarity.
In conclusion, by evaluating privacy leakage from the
perspective of semantics, our BiNLI metric can serve as acomplement to the metrics that only compare similarity from
text level, providing a more comprehensive assessment of
textual privacy leakage.
EntroGuard can efficiently defend against EIAs. To
protect users’ privacy, TextObfuscator requires fine-tuning

Original w/o BPA with BPA0.000.050.100.150.200.250.30Rouge Score
Personal Chat
Msmacro
Original w/o BPA with BPA0.000.050.100.150.200.25Bleu Score
Personal Chat
Msmacro
Original w/o BPA with BPA0.00000.00250.00500.00750.01000.01250.01500.0175EMR Score
Personal Chat
Msmacro
Original w/o BPA with BPA0.00.10.20.30.40.5BiNLI Score
Personal Chat
MsmacroFigure 6: The performance of privacy-preserving capa-
bilities against optimization-based EIAs.
the entire embedding model, whereas DP, PGD, and our
proposed EntroGuard can directly perturb the embeddings.
Specifically, we evaluate EntroGuard against other privacy
protection methods using the MS MARCO and Personal
Chat datasets, employing embedding models with different
structures, i.e., sentence-t5-large, simcse-bert-large-uncased,
all-roberta-large, and all-mpnet-base.
Table 4 shows the performance of privacy-preserving ca-
pabilities against learning-based EIAs. As observed, both
the differential privacy-based scheme and TextObfuscator
exhibit similar defenses to the original embedding model
on text similarity metrics like ROUGE, BLEU, EMR, and the
semantic similarity metric, BiNLI. These results demonstrate
that these schemes cannot effectively counter learning-based
embedding inversion attacks, which still lead to the exposure
of user privacy. Although the PGD scheme offers some resis-
tance to embedding inversion attacks, it may still result in
partial leakage of user privacy. In particular, in some embed-
ding models, e.g., at simcse-bert, it offers poor privacy pro-
tection, with ROUGE only decreasing from 0.4695 to 0.3082
and BiBLI decreasing from 0.5619 to 0.4205. In contrast, the
evaluation results for EntroGuard show a substantial reduc-
tion in metric values, i.e., ROUGE drops from approximately
0.509 to 0.007, BLEU from 0.534 to 0.009, EMR from 0.224 to
0, and BiNLI from 0.594 to 0.043. which demonstrates that
EntroGuard effectively protects user privacy.
Besides, we also evaluate the effectiveness of EntroGuard
against optimization-based EIAs. As demonstrated in Fig-
ure 6, our scheme can resist such EIAs to some extent. And,
with the addition of Bound-aware Perturbation Adaptation
(BPA), the privacy-preserving capability is further enhanced,
e.g., BLEU is reduced to 0.09, and BiNLI is reduced to 0.25,
indicating BPA’s role in defending against these attacks.Table 5: The applicability of EntroGuard with different
size of embedding models.
Size Method ROUGE BLEU EMR BiNLI
BaseOriginal 0.6417 0.6901 0.3324 0.7287
Ours 0.0000 0.0000 0.0000 0.0004
LargeOriginal 0.6427 0.6907 0.3377 0.7370
Ours 0.0000 0.0000 0.0000 0.0008
XLOriginal 0.6186 0.6686 0.3117 0.7280
Ours 0.0000 0.0000 0.0000 0.0004
XXLOriginal 0.6133 0.6133 0.3039 0.7227
Ours 0.0000 0.0000 0.0000 0.0004
Table 6: The generality of EntroGuard in privacy-
preserving capabilities when facing different attack
models.
Attack model Method ROUGE BLEU EMR BiNLI
GPT-2Original 0.6427 0.6907 0.3377 0.7370
GPT-2 0.0000 0.0000 0.0000 0.0008
DialoGPT 0.0000 0.0000 0.0000 0.0004
Llama3 0.0000 0.0000 0.0000 0.0004
DialoGPTOriginal 0.6142 0.6656 0.3001 0.7005
GPT-2 0.0000 0.0000 0.0000 0.0004
DialoGPT 0.0000 0.0000 0.0000 0.0005
Llama3 0.0000 0.0000 0.0000 0.0004
Llama3Original 0.6813 0.7003 0.3929 0.7656
GPT-2 0.0142 0.0256 0.0001 0.0121
DialoGPT 0.0752 0.0889 0.0004 0.0518
Llama3 0.0178 0.0339 0.0000 0.0064
5.4 Robustness of EntroGuard
In this section, we assess the effectiveness of our method
across different embedding model sizes. Then, we evaluate
EntroGuard’s transferability across various attack models.
EntroGuard can accommodate different embedding
models of different sizes. Given that the size of embed-
ding models deployed on different end devices may be het-
erogeneous, it is crucial to evaluate the applicability of our
scheme across models with different parameter scales. To
this end, we evaluate privacy-preserving capability of our
approach utilizing Sentence-T5 models of varying sizes, in-
cluding sentence-t5-base, sentence-t5-large, sentence-t5-xl,
and sentence-t5-xxl. As shown in Table 5, our scheme consis-
tently reduces ROUGE, BLEU, EMR, and BiNLI scores to zero
across all model sizes, demonstrating its robust protection
capability regardless of model scale.
EntroGuard has good generality to different attack
models. To evaluate the robustness of our scheme against
unknown attack models, we evaluated its privacy-preserving

0 500 1000 1500 2000 2500
Time (ms)Sentence-t5-baseSentence-t5-largeSentence-t5-xlSentence-t5-xxlSimcse-bertRobertaMPNetEmbedding Models
88.16262.89792.632667.36359.37364.10133.64
81.23256.33785.822660.85352.48356.98126.90Embedding model
Embedding model & EntroGuard(a) Inference with CPUs
0 10 20 30 40 50 60 70
Time (ms)Sentence-t5-baseSentence-t5-largeSentence-t5-xlSentence-t5-xxlSimcse-bertRobertaMPNetEmbedding Models
36.0260.0967.3475.2452.8753.8232.26
30.2654.5061.5168.1047.3148.2026.56Embedding model
Embedding model & EntroGuard (b) Inference with GPUs
Figure 7: Comparison of inference efficiency on end devices using different hardware.
NVIDIA RTX 2060Server
NVIDIA Jetson OrinEnd
NVIDIA RTX 2060 NVIDIA Jetson Orin
NVIDIA Jetson OrinEnd
…
Figure 8: Testbed in real world.
effectiveness using GPT-2, DialoGPT, and Llama3 as attack
models, while training EntroGuard on a separate surrogate
model. As shown in Table 6, our scheme also consistently
reduces ROUGE, BLEU, EMR, and BiNLI scores to near-zero
values across all attack models. Notably, even confronting a
relatively stronger attacker (e.g., Llama3) with a relatively
weaker surrogate model (e.g., GPT2), the degree of textual
privacy leakage is still low (e.g., BLEU 0.0889, BiNLI 0.0518),
indicating that our scheme has strong privacy-preserving
ability with high generality against different attack models.
5.5 Evaluation on end devices
In this section, we evaluate the efficiency of our proposed
scheme on real-world end devices, as shown in Figure 8.
EntroGuard can be effectively integrated into end
devices and achieve efficient inference. We integrated En-
troGuard as a plug-in to the end devices, i.e., Jetson AGX Orin
Developer Kit [ 3], where seven embedded models with differ-
ent structures and different parameter sizes are deployed on
them. EntroGuard can be deployed as a plug-in, without need-
ing to know the detailed structure of the embedding model.
It requires only approximately 13MB of additional storagespace to store its parameters. In practice, the output of the
original embedding model is directly fed into EntroGuard for
processing, which then generates the privacy-preserving em-
beddings. To evaluate EntroGuard’s efficiency, we measured
the computation time on both CPUs and GPUs by averaging
the time over 17K sequential inputs. For CPU inference, the
time cost increases by only 3% compared to the original em-
bedding models when EntroGuard is applied. More detailed
time costs on CPUs are shown in fig. 7a. For GPU inference,
compared to the original models, the time cost increases by
13% approximately when EntroGuard is plugged in, whose
detailed time costs on GPUs are provided in fig. 7b. Notably,
the absolute time to run our scheme on both CPUs and GPUs
is relatively short, i.e., approximately 7 ms on CPUs and 5 ms
on GPUs. Since the inference time of the embedding model
on GPUs reduces, the relative percentage increase in time
consumption is more noticeable. In summary, EntroGuard
can be easily integrated into existing black-box embedding
models, providing efficient protection for embeddings with
acceptable impact on processing time.
6 Conclusions
In this paper, we introduce EntroGuard, a novel approach
designed to protect the privacy of text embeddings transmit-
ted from end devices while maintaining retrieval accuracy
in cloud databases without requiring additional cooperation
from the cloud. Meanwhile, EntroGuard can be efficiently
integrated into the existing embedding model on end devices
with acceptable overhead. Furthermore, we propose BiNLI,
a metric for evaluating sentence privacy based on semantic
similarity, which enables a more comprehensive quantifica-
tion of text privacy leakage. Extensive experiments demon-
strate that EntroGuard outperforms existing text embedding
protection methods, offering superior privacy protection
with minimal loss in retrieval capability.

References
[1] 2016. https://gdpr-info.eu/issues/personal-data/.
[2]2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/
interpreting-gpt-the-logit-lens.
[3]2024. https://www.nvidia.com/en-us/autonomous-machines/
embedded-systems/jetson-orin/.
[4]Yiyi Chen, Heather Lent, and Johannes Bjerva. 2024. Text embedding
inversion security for multilingual language models. In Proceedings of
the 62nd Annual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers) . 7808–7827.
[5]Minxin Du, Xiang Yue, Sherman SM Chow, Tianhao Wang, Chenyu
Huang, and Huan Sun. 2023. Dp-forward: Fine-tuning and inference
on language models with differential privacy in forward pass. In Pro-
ceedings of the 2023 ACM SIGSAC Conference on Computer and Com-
munications Security . 2665–2679.
[6]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Ka-
dian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Amy Yang, Angela Fan, et al .2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 (2024).
[7]Liam Fowl, Jonas Geiping, Steven Reich, Yuxin Wen, Wojtek Czaja,
Micah Goldblum, and Tom Goldstein. 2022. Decepticons: Corrupted
transformers breach privacy in federated learning for language models.
arXiv preprint arXiv:2201.12675 (2022).
[8]T Gao, X Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive
Learning of Sentence Embeddings. In EMNLP 2021-2021 Conference on
Empirical Methods in Natural Language Processing, Proceedings .
[9]Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014.
Explaining and harnessing adversarial examples. arXiv preprint
arXiv:1412.6572 (2014).
[10] Shreya Goyal, Sumanth Doddapaneni, Mitesh M Khapra, and Balara-
man Ravindran. 2023. A survey of adversarial defenses and robustness
in nlp. Comput. Surveys 55, 14s (2023), 1–39.
[11] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Wein-
berger. 2017. Densely connected convolutional networks. In Proceed-
ings of the IEEE conference on computer vision and pattern recognition .
4700–4708.
[12] Diederik P Kingma. 2014. Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980 (2014).
[13] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic un-
certainty: Linguistic invariances for uncertainty estimation in natural
language generation. arXiv preprint arXiv:2302.09664 (2023).
[14] Ang Li, Jiayi Guo, Huanrui Yang, Flora D Salim, and Yiran Chen.
2021. DeepObfuscator: Obfuscating intermediate representations with
privacy-preserving adversarial learning on smartphones. In Proceed-
ings of the International Conference on Internet-of-Things Design and
Implementation . 28–39.
[15] Haoran Li, Dadi Guo, Donghao Li, Wei Fan, Qi Hu, Xin Liu, Chunkit
Chan, Duanyi Yao, Yuan Yao, and Yangqiu Song. 2024. Privlm-bench:
A multi-level privacy evaluation benchmark for language models. In
Proceedings of the 62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers) . 54–73.
[16] Haoran Li, Mingshi Xu, and Yangqiu Song. 2023. Sentence embed-
ding leaks more information than you expect: Generative embed-
ding inversion attack to recover the whole sentence. arXiv preprint
arXiv:2305.03010 (2023).
[17] Yansong Li, Zhixing Tan, and Yang Liu. 2023. Privacy-preserving
prompt tuning for large language model services. arXiv preprint
arXiv:2305.06212 (2023).
[18] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of
summaries. In Text summarization branches out . 74–81.[19] Tiantian Liu, Hongwei Yao, Tong Wu, Zhan Qin, Feng Lin, Kui Ren,
and Chun Chen. 2024. Mitigating Privacy Risks in LLM Embeddings
from Embedding Inversion. arXiv preprint arXiv:2411.05034 (2024).
[20] Yinhan Liu. 2019. Roberta: A robustly optimized bert pretraining
approach. arXiv preprint arXiv:1907.11692 364 (2019).
[21] Aleksander Madry. 2017. Towards deep learning models resistant to
adversarial attacks. arXiv preprint arXiv:1706.06083 (2017).
[22] Junhua Mao, Jiajing Xu, Kevin Jing, and Alan L Yuille. 2016. Training
and evaluating multimodal word embeddings with large-scale web
annotated images. Advances in neural information processing systems
29 (2016).
[23] Tomas Mikolov. 2013. Efficient estimation of word representations in
vector space. arXiv preprint arXiv:1301.3781 (2013).
[24] John X Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexan-
der M Rush. 2023. Text embeddings reveal (almost) as much as text.
arXiv preprint arXiv:2310.06816 (2023).
[25] John X Morris, Wenting Zhao, Justin T Chiu, Vitaly Shmatikov, and
Alexander M Rush. 2023. Language model inversion. arXiv preprint
arXiv:2311.13647 (2023).
[26] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary,
Rangan Majumder, and Li Deng. 2016. Ms marco: A human-generated
machine reading comprehension dataset. (2016).
[27] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith
Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-T5: Scalable Sentence
Encoders from Pre-trained Text-to-Text Models. In Findings of the
Association for Computational Linguistics: ACL 2022 . 1864–1874.
[28] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego,
Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, et al .2022.
Large Dual Encoders Are Generalizable Retrievers. In Proceedings of the
2022 Conference on Empirical Methods in Natural Language Processing .
9844–9855.
[29] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego,
Ji Ma, Vincent Y Zhao, Yi Luan, Keith B Hall, Ming-Wei Chang, et al .
2021. Large dual encoders are generalizable retrievers. arXiv preprint
arXiv:2112.07899 (2021).
[30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wain-
wright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina
Slama, Alex Ray, et al .2022. Training language models to follow
instructions with human feedback. Advances in neural information
processing systems 35 (2022), 27730–27744.
[31] Xudong Pan, Mi Zhang, Shouling Ji, and Min Yang. 2020. Privacy
risks of general-purpose language models. In 2020 IEEE Symposium on
Security and Privacy (SP) . IEEE, 1314–1331.
[32] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
Bleu: a method for automatic evaluation of machine translation. In Pro-
ceedings of the 40th annual meeting of the Association for Computational
Linguistics . 311–318.
[33] Rajvardhan Patil, Sorio Boit, Venkat Gudivada, and Jagadeesh
Nandigam. 2023. A survey of text representation and embedding
techniques in nlp. IEEE Access 11 (2023), 36120–36146.
[34] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei,
Ilya Sutskever, et al .2019. Language models are unsupervised multitask
learners. OpenAI blog 1, 8 (2019), 9.
[35] Congzheng Song and Ananth Raghunathan. 2020. Information leak-
age in embedding models. In Proceedings of the 2020 ACM SIGSAC
conference on computer and communications security . 377–390.
[36] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020.
Mpnet: Masked and permuted pre-training for language understanding.
Advances in neural information processing systems 33 (2020), 16857–
16867.
[37] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava,
and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for

zero-shot evaluation of information retrieval models. arXiv preprint
arXiv:2104.08663 (2021).
[38] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and
Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and
VERification. arXiv preprint arXiv:1803.05355 (2018).
[39] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-
Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric
Hambro, Faisal Azhar, et al .2023. Llama: Open and efficient foundation
language models. arXiv preprint arXiv:2302.13971 (2023).
[40] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval
of the best counterargument without prior topic knowledge. In Pro-
ceedings of the 56th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) . 241–251.
[41] Zhipeng Wan, Anda Cheng, Yinggui Wang, and Lei Wang. 2024. Infor-
mation Leakage from Embedding in Large Language Models. arXiv
preprint arXiv:2405.11916 (2024).
[42] Shiming Wang, Zhe Ji, Liyao Xiang, Hao Zhang, Xinbing Wang,
Chenghu Zhou, and Bo Li. 2024. Crafter: Facial Feature Crafting
against Inversion-based Identity Theft on Deep Models. arXiv preprint
arXiv:2401.07205 (2024).
[43] Zekun Xu, Abhinav Aggarwal, Oluwaseyi Feyisetan, and Nathanael
Teissier. 2020. A differentially private text perturbation method using
a regularized mahalanobis metric. arXiv preprint arXiv:2010.11947
(2020).
[44] Ziqian Zeng, Jianwei Wang, Junyao Yang, Zhengdong Lu, Huiping
Zhuang, and Cen Chen. 2024. PrivacyRestore: Privacy-Preserving
Inference in Large Language Models via Privacy Removal and Restora-
tion. arXiv preprint arXiv:2406.01394 (2024).
[45] Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe
Kiela, and Jason Weston. 2018. Personalizing Dialogue Agents: I have
a dog, do you have pets too?. In Proceedings of the 56th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers) .
2204–2213.
[46] Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett,
Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2020. DialoGPT:
Large-Scale Generative Pre-training for Conversational Response Gen-
eration. In ACL, system demonstration .
[47] Xin Zhou, Jinzhu Lu, Tao Gui, Ruotian Ma, Zichu Fei, Yuran Wang,
Yong Ding, Yibo Cheung, Qi Zhang, and Xuan-Jing Huang. 2022. Text-
fusion: Privacy-preserving pre-trained model inference via token fu-
sion. In Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing . 8360–8371.
[48] Xin Zhou, Yi Lu, Ruotian Ma, Tao Gui, Yuran Wang, Yong Ding, Yibo
Zhang, Qi Zhang, and Xuan-Jing Huang. 2023. Textobfuscator: Mak-
ing pre-trained language model a privacy protector via obfuscating
word representations. In Findings of the Association for Computational
Linguistics: ACL 2023 . 5459–5473.
[49] Zhihao Zhu, Ninglu Shao, Defu Lian, Chenwang Wu, Zheng Liu,
Yi Yang, and Enhong Chen. 2024. Understanding Privacy Risks of
Embeddings Induced by Large Language Models. arXiv preprint
arXiv:2404.16587 (2024).
A Details of datasets
•PersonalChat : a chit-chat dataset consisting of 162,064
utterances, where randomly paired workers are assigned
specific personas and engage in conversations to get to
know each other.
•MS MARCO : a question-answering dataset consisting of
real Bing queries paired with 8.8 million human-generated
answers.•ArguAna : a dataset consisting of 8,674 passages, focused
on retrieving the most relevant counterarguments for a
given argument.
•FEVER : a fact extraction and verification dataset consist-
ing of 5.42 million passages, designed to support automatic
fact-checking systems.
B Retrieval performance
For baseline methods that allow adjustment of perturbation
intensity, such as DP and PGD, we also set the intensity
to match that of the 0.01x Gaussian noise. Table 7 shows
the retrieval performance of the baseline method under the
same perturbation intensity. Except PGD has a decrease in
retrieval accuracy when the embedding model is SimCSE-
BERT and the dataset is FEVER, the rest of the schemes are
able to largely maintain the original retrieval accuracy, which
aligns with the notion that the retrieval performance can be
maintained by controlling the perturbation intensity.

Table 7: The performance of retrieval performance in cloud database in terms of NDCG, MAP, and Precision
with different embedding models in end side. The higher the values of NDCG, MAP, and Precision, the better the
retrieval performance.
Method Embedding modelArguana Fever Msmarco
NDCG ↑MAP ↑Precision ↑NDCG ↑MAP ↑Precision ↑NDCG ↑MAP ↑Precision ↑
OriginalSentence-t5 0.3364 0.2853 0.0983 0.3220 0.2832 0.0915 0.5084 0.0496 0.5954
Simcse-bert 0.3520 0.3010 0.1013 0.1621 0.1394 0.0478 0.2848 0.0319 0.4000
MPNet 0.4534 0.3948 0.1260 0.5638 0.5080 0.1515 0.6928 0.0904 0.8140
Roberta 0.3852 0.3293 0.1108 0.5011 0.4521 0.1344 0.6706 0.0729 0.7861
DPSentence-t5 0.3362 0.2849 0.0984 0.3127 0.2746 0.0891 0.5095 0.0529 0.5861
Simcse-bert 0.3511 0.3000 0.1011 0.1623 0.1394 0.0479 0.2832 0.0320 0.4000
MPNet 0.4532 0.3944 0.1262 0.5612 0.5048 0.1514 0.6834 0.0893 0.8000
Roberta 0.3871 0.3300 0.1120 0.4976 0.4483 0.1339 0.6717 0.0733 0.7861
PGDSentence-t5 0.3284 0.2802 0.0949 0.2983 0.2616 0.0853 0.4914 0.0486 0.5767
Simcse-bert 0.3616 0.3123 0.1020 0.1291 0.1101 0.0388 0.2887 0.0359 0.3954
MPNet 0.4543 0.3964 0.1259 0.5580 0.5029 0.1498 0.6991 0.0892 0.8140
Roberta 0.3799 0.3225 0.1108 0.4932 0.4449 0.1324 0.6715 0.0732 0.7861
TextObfuscatorSentence-t5 0.3385 0.2865 0.0992 0.3453 0.3049 0.0972 0.5016 0.0481 0.5907
Simcse-bert 0.3427 0.2936 0.0982 0.1717 0.1489 0.0498 0.2736 0.0295 0.3861
MPNet 0.4607 0.4018 0.1277 0.5780 0.5233 0.1537 0.7074 0.0915 0.8140
Roberta 0.3999 0.3425 0.1147 0.5262 0.4766 0.1402 0.6726 0.0736 0.7861
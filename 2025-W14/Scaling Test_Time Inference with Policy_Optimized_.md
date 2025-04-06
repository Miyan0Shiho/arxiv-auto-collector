# Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding

**Authors**: Sakhinana Sagar Srinivas, Venkataramana Runkana

**Published**: 2025-04-02 01:16:10

**PDF URL**: [http://arxiv.org/pdf/2504.01281v2](http://arxiv.org/pdf/2504.01281v2)

## Abstract
We present a comprehensive framework for enhancing Retrieval-Augmented
Generation (RAG) systems through dynamic retrieval strategies and reinforcement
fine-tuning. This approach significantly improves large language models on
knowledge-intensive tasks, including opendomain question answering and complex
reasoning. Our framework integrates two complementary techniques:
Policy-Optimized RetrievalAugmented Generation (PORAG), which optimizes the use
of retrieved information, and Adaptive Token-Layer Attention Scoring (ATLAS),
which dynamically determines retrieval timing and content based on contextual
needs. Together, these techniques enhance both the utilization and relevance of
retrieved content, improving factual accuracy and response quality. Designed as
a lightweight solution compatible with any Transformer-based LLM without
requiring additional training, our framework excels in knowledge-intensive
tasks, boosting output accuracy in RAG settings. We further propose CRITIC, a
novel method to selectively compress key-value caches by token importance,
mitigating memory bottlenecks in long-context applications. The framework also
incorporates test-time scaling techniques to dynamically balance reasoning
depth and computational resources, alongside optimized decoding strategies for
faster inference. Experiments on benchmark datasets show that our framework
reduces hallucinations, strengthens domain-specific reasoning, and achieves
significant efficiency and scalability gains over traditional RAG systems. This
integrated approach advances the development of robust, efficient, and scalable
RAG systems across diverse applications.

## Full Text


<!-- PDF content starts -->

arXiv:2504.01281v2  [cs.LG]  3 Apr 2025Scaling Test-Time Inference with Policy-Optimized, Dynam ic
Retrieval-Augmented Generation via KV Caching and Decodin g
Sakhinana Sagar Srinivas1Venkataramana Runkana1
Abstract
We present a comprehensive framework for en-
hancing Retrieval-Augmented Generation (RAG)
systems through dynamic retrieval strategies
and reinforcement ﬁne-tuning. This approach
signiﬁcantly improves large language models
on knowledge-intensive tasks, including open-
domain question answering and complex rea-
soning. Our framework integrates two comple-
mentary techniques: Policy-Optimized Retrieval-
Augmented Generation (PORAG), which opti-
mizes the use of retrieved information, and Adap-
tive Token-Layer Attention Scoring (ATLAS),
which dynamically determines retrieval timing
and content based on contextual needs. To-
gether, these techniques enhance both the uti-
lization and relevance of retrieved content, im-
proving factual accuracy and response quality.
Designed as a lightweight solution compatible
with any Transformer-based LLM without re-
quiring additional training, our framework ex-
cels in knowledge-intensive tasks, boosting out-
put accuracy in RAG settings. We further pro-
pose CRITIC, a novel method to selectively com-
press key-value caches by token importance, mit-
igating memory bottlenecks in long-context ap-
plications. The framework also incorporates
test-time scaling techniques to dynamically bal-
ance reasoning depth and computational re-
sources, alongside optimized decoding strategies
for faster inference. Experiments on benchmark
datasets show that our framework reduces hallu-
cinations, strengthens domain-speciﬁc reasoning,
and achieves signiﬁcant efﬁciency and scalability
gains over traditional RAG systems. This inte-
grated approach advances the development of ro-
bust, efﬁcient, and scalable RAG systems across
diverse applications.
1Tata Research Development and Design Center, TCS Re-
search, Bangalore. Correspondence to: Sakhinana Sagar Sri nivas
<sagar.sakhinana@tcs.com >.
Preliminary work. Under review. Do not distribute. Copyrig ht
2025 by the author(s).1. Introduction
Retrieval-Augmented Generation (RAG, ( Lewis et al. ,
2020 ;Su et al. ;Wang et al. ,2025 )) has gained signiﬁcant
interest in Natural Language Processing for enhancing
large language models (LLMs) on knowledge-intensive
tasks through external information retrieval, with applic a-
tions across search engines, conversational agents, chat-
bots, and many other applications. RAG addresses key
LLM limitations, including hallucinations, outdated info r-
mation, and insufﬁcient domain-speciﬁc knowledge, par-
ticularly in open-domain question answering. Retrieval-
Augmented Fine-Tuning (RAFT ( Zhang et al. ,2024c )) ad-
vances this approach by integrating retrieval methods with
language model supervised ﬁne-tuning. Unlike traditional
RAG, which simply retrieves documents for generation,
RAFT trains the language model alongside the retrieval
mechanism, teaching it to dynamically leverage exter-
nal knowledge, prioritize relevant content while ignoring
distractors for improved performance in domain-speciﬁc
RAG contexts (e.g., open-book and in-domain question
answering). Building on advancements in LLM training
methodologies, DeepSeek has enhanced its AI models,
notably DeepSeek-R1 ( Liu et al. ,2024 ;Guo et al. ,2025 ;
Shao et al. ,2024 ), by implementing Group Relative Policy
Optimization (GRPO), an advanced reinforcement learning
algorithm that improves training efﬁciency and model per-
formance beyond traditional supervised ﬁne-tuning. GRPO
reduces computational overhead by eliminating the value
function, using group-based advantage estimation for sim-
pliﬁed reward computation, lowering memory usage, and
integrating Kullback-Leibler (KL) divergence regulariza -
tion for stable, efﬁcient training. It outperforms standar d
Rejection Sampling Fine-Tuning (RFT), which relies on of-
ﬂine sampling, and Online RFT, which dynamically sam-
ples from an evolving policy. GRPO also supports pro-
cess supervision (GRPO+PS), providing step-by-step feed-
back for improved reasoning, surpassing outcome super-
vision (GRPO+OS), which evaluates only ﬁnal answers.
Addressing the limitations of static retrieval in traditio nal
RAG, DRAGIN (Dynamic Retrieval-Augmented Genera-
tion based on Information Needs, ( Su et al. )) is an ad-
vanced framework that dynamically determines when and
what to retrieve during text generation. Unlike methods
1

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
with ﬁxed retrieval intervals or simplistic query formula-
tions, DRAGIN employs Real-time Information Needs De-
tection (RIND) to trigger retrieval only when necessary,
considering token uncertainty, semantic importance, and
inﬂuence on future tokens. Its query formulation based
on Self-attention (QFS) generates more effective queries
by leveraging the full generated context rather than just
recent tokens to ﬁll information gaps. This adaptive ap-
proach minimizes redundant retrievals, improves efﬁcienc y,
and enhances response accuracy. Despite these advance-
ments, integrating external knowledge during inference
through RAG enhances the capabilities of LLMs. However,
it also introduces challenges, such as increased computa-
tional and memory demands. Key-Value (KV) Caching
(Feng et al. ,2024 ;Hooper et al. ,2025 ;Yang et al. ,2025 )
addresses this issue by efﬁciently managing the memory
load resulting from RAG’s expanded context window. It op-
timizes the storage and retrieval of key-value pairs, preve nt-
ing memory bottlenecks and accelerating the processing of
augmented information. In transformer-based LLMs, KV
Caching stores intermediate hidden states (keys and val-
ues) of previous tokens during attention computation, en-
abling faster text generation by reusing them for new to-
kens. This approach reduces redundant calculations, low-
ers memory usage, and improves efﬁciency for long se-
quences, thereby enhancing the contextuality and coher-
ence of LLMs while mitigating the memory overhead in-
troduced by RAG. Test-Time Scaling Inference Techniques
(Muennighoff et al. ,2025 ;Ji et al. ,2025 ;Yoon et al. ,2025 ;
Geiping et al. ,2025 ) address these challenges by dynam-
ically allocating computational resources based on task
complexity. Unlike static inference methods, which apply
ﬁxed computational effort regardless of task demands, test -
time scaling adaptively adjusts reasoning depth and com-
plexity. For simple questions, it reduces unnecessary over -
head, enabling faster responses and minimizing hallucina-
tions. For complex or multi-faceted tasks, it increases rea -
soning depth to improve accuracy and better integrate re-
trieved context, enabling LLMs to effectively process and
reason with augmented context. This adaptive approach
mimics human-like deliberative reasoning for knowledge-
intensive tasks without costly retraining, enhancing efﬁ-
ciency and performance while maintaining accuracy and re-
ducing hallucinations. Together, RAFT enhances RAG by
integrating retrieval with supervised ﬁne-tuning, enabli ng
models to dynamically leverage external knowledge and
prioritize relevant content while ignoring distractors. D RA-
GIN dynamically determines when and what to retrieve dur-
ing text generation, minimizing redundant retrievals and
improving efﬁciency. KV Caching optimizes memory us-
age by storing intermediate hidden states, reducing compu-
tational overhead in RAG, while Test-Time Scaling dynam-
ically allocates resources based on task complexity. These
advancements enable RAG systems to integrate externalknowledge more accurately, efﬁciently, and at scale, ensur -
ing faster and more effective utilization of retrieved data
within the LLM framework. While these recent advance-
ments have enhanced retrieval integration in LLMs, sig-
niﬁcant challenges remain in balancing retrieval ﬁdelity,
response quality, and computational efﬁciency. Current
methods often struggle to dynamically determine when and
how much external information to incorporate, sometimes
overwhelming the model or sacriﬁcing the coherence of
its responses. Motivated by these persistent challenges,
our work seeks to reﬁne the synergy between retrieval and
generation through a dual approach. First, we ﬁne-tune
language models via policy optimization, enabling them
to more effectively integrate and utilize retrieved conten t.
This reﬁnement not only improves factual alignment but
also enhances overall response quality. Second, we in-
troduce a mechanism that selectively triggers external re-
trieval based on the model’s internal state, ensuring that a d-
ditional information is incorporated only when necessary.
This targeted strategy optimizes computational resources
while preserving the language model’s coherence. In the
following sections, we outline our contributions that ex-
tend state-of-the-art methods by addressing both the op-
timization of retrieval-augmented generation and the efﬁ-
cient management of computational overhead. Our contri-
butions are as follows:
• We introduce two complementary techniques to en-
hance Retrieval-Augmented Generation (RAG) sys-
tems: Policy-Optimized Retrieval-Augmented Gener-
ation (PORAG) and Adaptive Token-Layer Attention
Scoring for Selective Retrieval (ATLAS). PORAG
extends GRPO to the RAG setting, ﬁne-tuning pre-
trained LLMs using QLoRA (Quantized Low-Rank
Adaptation). The parameter-efﬁcient optimization us-
ing QLoRA leads to improved performance on in-
domain Question-Answering (QA) tasks while mit-
igating catastrophic forgetting of pre-trained knowl-
edge. PORAG incorporates group-based advantage
estimation and a trust-region constrained policy up-
date to ensure stable and robust ﬁne-tuning in retrieval-
dependent contexts. Additionally, PORAG employs
a dual reward mechanism that explicitly balances re-
trieval ﬁdelity—ensuring generated responses remain
factually aligned with retrieved information—and re-
sponse quality, which evaluates coherence, ﬂuency,
and overall helpfulness beyond factual accuracy. To
effectively implement this, specialized linear layer-
based reward heads are integrated after the ﬁnal
layer of the pre-trained LLM with QLoRA adapters.
Trained reward heads evaluate retrieval ﬁdelity and
response quality, and their combined signals form
a composite reward for group-based advantage es-
timation, thus guiding generation policy optimiza-
2

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
tion. ATLAS, on the other hand, dynamically deter-
mines when and what to retrieve by analyzing the
language model’s internal attention patterns. Using
Multi-Layer Attention Gradient (MLAG) to detect in-
formation gaps and Layerwise Representation Pooling
(LRP) to construct targeted queries, ATLAS retrieves
the most relevant external information to ﬁll informa-
tion gaps, improving retrieval precision and ensuring
retrieval occurs only when necessary and precisely
aligned with the model’s information needs. Together,
these techniques create a comprehensive RAG system
that optimizes both the utilization of retrieved informa-
tion and the timing of retrieval, signiﬁcantly improv-
ing efﬁciency, accuracy, and computational overhead.
The integration of PORAG and ATLAS addresses key
challenges in RAG systems, such as over-reliance on
retrieval, inefﬁcient query formulation, and unstable
optimization, paving the way for more robust and
resource-efﬁcient language models.
• We present CRITIC (Cache Reduction via Importance-
based Token Inclusion Criteria), a method that ad-
dresses the memory bottleneck in policy-optimized
LLMs inference by selectively retaining only the most
important tokens in the KV cache. While tradi-
tional KV caching already reduces computational cost
from quadratic to linear, memory usage still grows
proportionally with sequence length, creating limita-
tions for long-context RAG applications. CRITIC de-
termines token importance using a weighted hybrid
approach that combines three complementary strate-
gies: attention-based (relationship strength), entropy-
based (attention pattern complexity), and gradient-
based (prediction sensitivity). This integrated ap-
proach enables ﬂexible compression behavior, with
the framework preserving only the highest-scoring to-
kens based on a conﬁgurable ratio. To further en-
hance real-world applicability, CRITIC incorporates
features such as delayed compression activation and
memory-pressure-based adaptive ratios as practical
optimizations. The architecture-agnostic solution sig-
niﬁcantly reduces memory requirements while main-
taining performance, leading to faster inference and
the ability to process longer contexts, particularly ben-
eﬁting RAG applications that need extended context
windows.
• We study the test-time scaling inference performance
of policy-optimized LLMs in RAG contexts, focus-
ing on improving response quality without altering
model weights by dynamically adjusting reasoning
depth, sampling, and validation during inference. We
utilize well-known inference scaling techniques, in-
cluding Self-Consistency, Best-of-N Sampling, Monte
Carlo Tree Search (MCTS), and others, each employ-ing unique strategies to enhance output quality, ac-
curacy, and efﬁciency. These methods trade off in-
creased computational complexity—often exceeding
O(n)for standard inference, where nis the sequence
length—for improved reliability and response qual-
ity, optimizing inference under resource constraints.
Many of these techniques leverage Weak-to-Strong
Distillation, iteratively reﬁning outputs to converge on
higher-quality responses. Each algorithm presents dis-
tinct trade-offs in cost, approach, selection method,
and other key factors.
2. Proposed Methodology
To enhance RAG systems and address the limitations of ex-
isting methods like RAFT, we propose two complementary
techniques. First, Policy-Optimized Retrieval-Augmente d
Generation (PORAG) adapts Group Relative Policy Op-
timization (GRPO) and introduces a novel dual reward
mechanism that balances retrieval ﬁdelity (faithfulness t o
retrieved documents, penalizing hallucinations) with re-
sponse quality (coherence, ﬂuency, and helpfulness). As
a result, PORAG directly optimizes retrieval quality, con-
textual relevance, and generation coherence, unlike RAFT.
Second, Adaptive Token-Layer Attention Scoring for Se-
lective Retrieval (ATLAS) efﬁciently determines when and
what to retrieve. By leveraging the policy-optimized lan-
guage model’s internal states, ATLAS employs Multi-
Layer Attention Gradient (MLAG) to detect information
needs through analyzing attention shifts across layers, tr ig-
gering retrieval only when necessary. Upon retrieval, Lay-
erwise Representation Pooling (LRP) constructs targeted
queries by selecting relevant preceding tokens based on
attention and representation similarity. These queries ar e
then used to retrieve information from external knowledge
sources, ﬁlling speciﬁc information gaps. By integrating
PORAG and ATLAS, we create a comprehensive RAG sys-
tem that optimizes both the utilization of retrieved inform a-
tion through policy optimization and the timing and preci-
sion of retrieval through selective attention analysis. Th is
approach improves factual accuracy and generation quality
while minimizing computational overhead.
2.1. Policy-Optimized Retrieval-Augmented
Generation (PORAG)
RAG techniques present unique optimization challenges
that Retrieval-Augmented Fine-Tuning (RAFT) often strug-
gles to fully address. PORAG offers a principled solution
rooted in Group Relative Policy Optimization (GRPO) by
reformulating the optimization problem through a group-
based relative advantage framework. Unlike RAFT, which
optimizes for log-likelihood of reference outputs, PORAG
enables direct optimization for retrieval quality, contex tual
relevance, and generation coherence through dual reward
3

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
modeling. In this work, we present a comprehensive math-
ematical formulation of PORAG, with theoretical justiﬁca-
tions and analytical insights. In the traditional RAG frame -
work, the policy model πθ(y|x,d)generates outputs ycon-
ditioned on the input query xand retrieved documents d.
The process is formalized as:
πθ(y|x,d) =|y|/productdisplay
i=1πθ(yi|x,d,y<i) (1)
whereπθ(y|x,d)represents the probability distribution
over the generated outputs y, conditioned on the input
queryx, retrieved documents d, and previously gener-
ated tokens y<i. Here,xdenotes the input query, d=
{d1,d2,...,dk}represents the set of retrieved documents,
yiis the token at position i, andy<icomprises all previ-
ously generated tokens. The parameter θcorresponds to
the frozen weights of the language model, which remain
unchanged during inference. In RAFT, the training objec-
tive optimizes the pretrained language model by maximiz-
ing the likelihood of reference outputs y∗while incorpo-
rating both relevant (“oracle”) and irrelevant (“distract or”)
documents. Since RAFT employs Low-Rank Adaptation
(LoRA( Lewis et al. ,2020 ;Izacard & Grave ,2020 )), only a
subset of trainable parameters, denoted as γ, is updated,
while the pre-trained language model parameters θremain
frozen. The RAFT loss function is deﬁned as:
LRAFT(γ) =−E(x,d oracle,ddistractor,y∗)∼D
[logπθ,γ(y∗|x,d oracle,ddistractor)](2)
wherexis the input query, doracle andddistractor represent the
retrieved relevant and irrelevant documents, respectivel y,
andy∗is the reference output. The training dataset D
consists of tuples (x,d oracle,ddistractor,y∗). The model as-
signs probability πθ,γ(y∗|x,d oracle,ddistractor)to the correct
output, where θrepresents the frozen pre-trained language
model parameters, and γrepresents the trainable param-
eters of the base language model, speciﬁcally Quantized
Low-Rank Adaptation (QLoRA) adapters. These are small,
trainable low-rank matrices added to the frozen pre-traine d
language model ( θ) to govern output generation condi-
tioned on the input and retrieved documents. QLoRA fo-
cuses on adapting key layers like attention query/value pro -
jections and feed-forward networks. This approach enables
efﬁcient ﬁne-tuning by modifying only a small subset of
weights, ensuring that the model learns to effectively dis-
tinguish relevant information from distractors while leve r-
aging retrieval-augmented generation for adaptation. How -
ever, RAFT has several limitations. It cannot differenti-
ate between high- and low-quality retrievals, assumes per-
fect reference outputs that fully leverage retrieved infor ma-
tion, and does not account for multiple valid generation
strategies within the same retrieval context. Additionall y,
it fails to optimize nuanced qualities such as faithfulnessto retrieved information. In contrast, PORAG addresses
these limitations by enabling direct optimization for mul-
tiple quality dimensions simultaneously. Our implementa-
tion employs two specialized reward heads—lightweight,
parameterized functions attached to the base model’s hid-
den states—calibrated for RAG-speciﬁc quality dimen-
sions: a Retrieval-Fidelity Reward Rﬁdelity(x,d,y∗;φ1),
which evaluates how faithfully the generated response in-
corporates and accurately reﬂects the retrieved informa-
tion, and a Response-Quality Reward Rquality(x,d,y∗;φ2),
which evaluates the overall quality, coherence, and helpfu l-
ness of the response beyond mere factual accuracy. Here,
φ={φ1,φ2}represent the trainable reward head param-
eters. The two reward heads— φ1for retrieval ﬁdelity and
φ2for response quality—are integrated into the neural net-
work architecture at the ﬁnal layer, operating on the hid-
den representations produced by the base model to com-
pute scalar rewards. Parameters φ1andφ2(typically im-
plemented via trainable standard linear layers with an in-
termediate tanh activation) are speciﬁcally optimized to
evaluate how well the generated response meets the desired
qualities (i.e., factual alignment with the retrieved docu -
ments and overall quality). The reward heads are trained
in conjunction with the base model, facilitating end-to-en d
optimization of both the generation and the reward func-
tion estimation. Consequently, the generation policy is di -
rectly informed by these dynamically learned reward sig-
nals. This co-adaptation mechanism results in more pre-
cise reward evaluations, enhanced training stability, and ul-
timately, superior performance in RAG. To effectively op-
timize the RAG context for multiple objectives, we decom-
pose the utility function into orthogonal components, each
capturing distinct quality dimensions. This allows the re-
ward heads to focus on speciﬁc aspects of generation qual-
ity. The utility function is deﬁned as:
U(x,d,y∗) =α·Uﬁdelity(x,d,y∗)+β·Uquality(x,y∗)
+λ·Uinteraction(x,d,y∗)
where:Uﬁdelity(x,d,y∗)measures the accuracy of the gen-
erated text in reﬂecting the retrieved documents, reward-
ing correct factual content and penalizing hallucinations ;
Uquality(x,y∗)evaluates the inherent quality of the genera-
tion (coherence, ﬂuency, relevance to the query), indepen-
dent of the retrieved content; and Uinteraction(x,d,y∗)cap-
tures the synergistic effects between ﬁdelity and quality.
Our dual reward heads approximate this decomposition:
Rﬁdelity(x,d,y∗;φ1)≈U ﬁdelity(x,d,y∗)
Rquality(x,d,y∗;φ2)≈U quality(x,y∗)
+λ
β·Uinteraction(x,d,y∗)
The reward heads compute scalar rewards from a vector
representation derived from the hidden states of the base
4

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
model through parameterized transformation functions:
Rﬁdelity(x,d,y∗;φ1) =fφ1(h(x,d,y∗))
Rquality(x,d,y∗;φ2) =gφ2(h(x,d,y∗))
whereh(x,d,y∗)∈Rdis a vector derived from the base
language model’s hidden states. Transformer models out-
put a hidden state matrix Rn×d(wherenis sequence length,
dis hidden dimension). his obtained by aggregating this
matrix, e.g., using the last token’s state or pooling. The
reward heads Rﬁdelity=fφ1(h)andRquality=fφ2(h)are
both multi-layer perceptrons with the form:
fφi(h) =Wφi
2·tanh(Wφi
1·h+bφi
1)+bφi
2
where fori∈ {1,2},Wφi
1∈Rd×d,Wφi
2∈Rd×1,
bφi
1∈Rd, andbφi
2∈Rare the parameters for reward head
i. We calculate the combined reward by balancing the com-
peting objectives of retrieval ﬁdelity and response qualit y.
Speciﬁcally, we aggregate quality and ﬁdelity rewards as
follows:
Rcomb(x,d,y∗) =α·Rﬁdelity(x,d,y∗;φ1)
+β·Rquality(x,d,y∗;φ2)
This weighting scheme ( α= 0.7andβ= 0.3in our
implementation) balances the competing objectives of re-
trieval ﬁdelity and response quality. The theoretical just i-
ﬁcation for this weighting comes from multi-objective re-
inforcement learning theory, where the Pareto frontier of
optimal policies can be explored through different weight-
ings of reward components. Unlike RAFT, which implic-
itly weights these objectives based on the training data dis -
tribution alone, PORAG allows explicit control over this
trade-off, enabling adaptation to different deployment sc e-
narios and user preferences. The combined rewards are nor-
malized and scaled using robust statistical principles:
Rﬁnal(x,d,y∗) =clip(Rcomb(x,d,y∗),−c1,c1)·γscale
whereγscaleis the reward scaling factor, and c1= 10.0is
the clipping threshold. The clipping operation is a form of
Winsorization, a statistical technique that reduces the im -
pact of outliers while preserving the ordinal relationship s
between rewards. We will now discuss Group-based Ad-
vantage Estimation for RAG. Given an input query xand
retrieved documents d, we generate a batch of Goutputs,
denoted by{y(1),y(2),...,y(G)}, using the current policy
πγ. This batch of outputs represents a single group of al-
ternatives. Within this group, we compute robust statistic al
estimators based on the ﬁnal reward Rﬁnal(x,d,y(i)), which
represents the overall reward for the i-th outputy(i)within
that group, given the input query xand retrieved documents
d:
µR(x,d) =1
GG/summationdisplay
i=1Rﬁnal(x,d,y(i)) (3)σ2
R(x,d) =1
GG/summationdisplay
i=1/parenleftBig
Rﬁnal(x,d,y(i))−µR(x,d)/parenrightBig2
(4)
σR(x,d) = max/parenleftbigg/radicalBig
σ2
R(x,d)+ǫ,σmin/parenrightbigg
(5)
whereµR(x,d)is the mean reward calculated within the
group,σ2
R(x,d)is the variance of the rewards calculated
within the group, and σR(x,d)is the standard deviation of
the rewards calculated within the group, clipped below by
a minimum value σmin= 0.1to ensure numerical stabil-
ity. The clipping prevents overly aggressive updates when
reward variation is small, which is particularly important
in RAG scenarios where retrieved documents might lead
to very similar generations within the group. The group-
relative advantage for each output y(i)is then calculated
as:
ˆAi=Rﬁnal(x,d,y(i))−µR(x,d)
σR(x,d)(6)
whereˆAirepresents the advantage of the i-th generated out-
put relative to the other outputs within its group. We will
now discuss the GRPO objective function for RAG settings.
For each token y(i)
jin the RAG output y(i), we compute the
probability ratio:
rj(γ) =π(y(i)
j|x,d,y(i)
<j)
πold(y(i)
j|x,d,y(i)
<j)(7)
where the ratio rj(γ)quantiﬁes the change in token prob-
ability under the current policy relative to the policy that
generated the sample, accounting for both the query and re-
trieved document context. The clipped surrogate objective
with a policy constraint for RAG is:
Lclip(γ) =1
GG/summationdisplay
i=11
|y(i)||y(i)|/summationdisplay
j=1min/parenleftBig
rj(γ)ˆAi,clip(rj(γ),1−ǫ,1+ǫ)ˆAi/parenrightBig
The clipping mechanism, with the parameter ǫ= 0.2,
serves as a trust region constraint that prevents excessive ly
large policy updates; this is critical in RAG systems, where
small changes in the probability distribution can lead to dr a-
matically different retrieval utilization patterns. The K L
divergence term prevents the policy from straying too far
from the reference model:
DKL(π||πref) =Ex,d,y∼πγ
|y|/summationdisplay
i=1KL(πref(·|x,d,y<i)||πγ(·|x,d,y<i))

Here,πrefrepresents the reference policy, speciﬁcally the
policy from the previous iteration of training, denoted as
πγold, whereγoldare the policy parameters before the cur-
rent update. Using the KL divergence with respect to
the previous policy stabilizes training by preventing dras -
tic changes in the policy distribution in each update step.
5

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
In the RAG context, this regularization term serves a crit-
ical function: it preserves the base knowledge encoded in
the model while allowing for targeted improvements in re-
trieval utilization. Without this constraint, aggressive opti-
mization toward retrieval-grounded responses might cause
the model to forget its pre-trained knowledge. Using the
unbiased estimator:
DKL(πγ||πref) =Ex,d,y∼πγ/bracketleftbiggπref(y|x,d)
πγ(y|x,d)−logπref(y|x,d)
πγ(y|x,d)−1/bracketrightbigg
The complete GRPO objective for RAG optimization is:
JGRPO-RAG(γ) =ω1·Lclip(γ)−ω2·DKL(πγ||πref)
whereLclip(γ)is the clipped surrogate objective that mea-
sures the policy improvement using the relative advantage
estimates, and DKL(πγ||πref)is the KL divergence between
the current policy πγand the reference policy πref, acting as
a regularizer. The weighting coefﬁcients ω1= 100.0and
ω2= 0.1balance policy improvement and divergence reg-
ularization; this balance is particularly important in RAG
contexts to prevent overreliance on retrieved information at
the expense of the model’s pre-existing knowledge. The
policy parameters γare updated to maximize the GRPO-
RAG objective:
γk+1=γk+ηγ∇γJGRPO-RAG(γk) (8)
The learning rate ηγ(typically 1×10−6to5×10−6
for RAG optimization) controls the step size of each up-
date. Unlike RAFT, which often uses larger learning rates,
GRPO-RAG typically requires smaller steps due to the
complexity of the reward landscape. To prevent instabil-
ity in RAG optimization, gradients are regularized both by
value and by norm:
∇γJclipped=clip(∇γJGRPO-RAG(γk),−cvalue,cvalue)(9)
∇γJnormalized=∇γJclipped
||∇γJclipped||2·min(||∇γJclipped||2,cnorm)
The clipping thresholds cvalue= 3.0andcnorm= 1.0pre-
vent extreme gradient values that could destabilize traini ng;
this is especially important in RAG systems where the re-
trieval distribution can introduce high variance in gradie nts.
The reward model parameters are updated using gradients
derived from minimizing their respective reward loss func-
tions,Lﬁdelity andLquality .
φ1,k+1=φ1,k+ηR∇φ1Lﬁdelity(φ1,k) (10)
φ2,k+1=φ2,k+ηR∇φ2Lquality(φ2,k) (11)
The reward model learning rate ηR(typically 5×10−5) is
usually higher than the policy learning rate, allowing the
reward models to adapt more quickly to preference signals.
The reward heads are updated separately using their respec-
tive reward losses with their own learning rate ηR. The gra-dients from the reward loss update only these differentiabl e
parameters and do not affect the base model’s weights θ
orγ, thereby producing well-calibrated, scalar reward val-
ues for accurately evaluating retrieval ﬁdelity and respon se
quality in RAG contexts. Training the reward heads to
yield reliable scalar rewards improves advantage estima-
tion, leading to more stable policy updates and enhanced
PORAG performance in RAG context. The reward losses
are divided into two components corresponding to Lﬁdelity
andLquality :Lﬁdelity evaluates how well the generated out-
put reﬂects the retrieved documents by measuring lexical
overlap with ROUGE scores (e.g., ROUGE-1, ROUGE-2,
ROUGE-L), capturing content similarity at multiple gran-
ularities, whileLquality assesses overall response quality
by combining semantic evaluation—using cosine similarity
between sentence embeddings of the generated text and the
reference—with question-answering metrics, including Ex -
act Match and F1 scores, to balance precision and recall. In
summary, while γdirectly controls the generation behavior
of the base model, φis dedicated to assessing and guiding
that behavior by providing reward signals. This separation
allows the PORAG framework to optimize both the output
generation (via γ) and the nuanced reward assessment (via
φ) concurrently.
2.2. Adaptive Token-Layer Attention Scoring for
Selective Retrieval (ATLAS)
ATLAS enhances RAG through a two-stage process that
leverages the policy-optimized LLM’s internal states. The
Multi-Layer Attention Gradient (MLAG) mechanism de-
tects when the model lacks necessary information by an-
alyzing shifts in attention patterns across layers, trigge ring
retrieval only at critical moments. Once retrieval is trig-
gered, Layerwise Representation Pooling (LRP) selects the
most relevant previously generated tokens to construct pre -
cise queries that address the model’s speciﬁc information
gaps. This ensures that external knowledge is retrieved onl y
when needed and targeted effectively, resulting in factual ly
accurate responses with minimal computational overhead.
Let us deﬁne a sequence of tokens T={t1,t2,...,tn}
processed by a ﬁxed pretrained LLM. Throughout this for-
mulation:iindexes the current position in the sequence,
Ldenotes the total number of layers in the model, Hrep-
resents the number of attention heads per layer, and Vis
the vocabulary of the language model. The Multi-Layer
Attention Gradient (MLAG) mechanism determines when
to trigger retrieval by analyzing attention patterns acros s
model layers:
MLAG(ti) =α·Gi·Di·si (12)
Each component serves a speciﬁc purpose and is computed
directly from observable model states. The gradient factor
(Gi) quantiﬁes attention pattern shifts across layers for to-
6

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
kenti:
Gi=L−1/summationdisplay
j=1ηj·/vextendsingle/vextendsingle¯Aj+1,i−¯Aj,i/vextendsingle/vextendsingle (13)
where¯Aj,iis the normalized average attention to the token
tiin layerj:
¯Aj,i=/summationtextH
h=1/summationtexti−1
k=1Aj,h,k,i
maxi
m=1/summationtextH
h=1/summationtexti−1
k=1Aj,h,k,m(14)
whereAj,h,k,i is the attention weight from token tkto token
tiin headhat layerj. Also,Ah,i,L is the average attention
received by token tiin headhat layerL:
Ah,i,L=1
i−1i−1/summationdisplay
k=1AL,h,k,i (15)
Note that for average attention, Ah,i,L excludestiby
averaging over i−1tokens (since a token doesn’t attend
to itself in autoregressive models). ηj=j
L−1is a layer-
speciﬁc coefﬁcient giving more weight to higher layers.
The gradient factor captures shifts in attention patterns
between consecutive layers during forward propagation.
Consistent patterns suggest the model has adequate
information, while sudden changes indicate it may be
searching for missing information. Layer weighting ( ηj)
prioritizes higher layers, which encode more abstract
and task-relevant representations, making them critical
for detecting when external knowledge is needed. The
depth-weighted information density ( Di) measures the
importance of token tibased on model uncertainty and
attention distribution:
Di= (1−pi(ti))·H/summationdisplay
h=1φh·Ah,i,L (16)
where the generation probability ( pi(ti)) represents the
model’s conﬁdence in generating token tiat positioni:
pi(ti) =exp(zi(ti))/summationtext
v∈Vexp(zi(v))(17)
wherezi(ti)is the raw logit (pre-softmax score) for token
tiat positionifrom the model’s ﬁnal output layer, which
is a direct measure of the model’s certainty. φhis a head
importance coefﬁcient derived from attention entropy:
φh=H(AL,h)/summationtextH
h′=1H(AL,h′)(18)
whereH(AL,h)is the entropy of the attention distribu-
tion of head hat layerLattending to all preceding tokens
t1,...,ti:
H(AL,h) =−i/summationdisplay
j=1i/summationdisplay
k=1AL,h,j,klog(AL,h,j,k+ǫ)(19)
whereǫis a small constant (typically 1e-10) to avoid log(0),
andAL,h,j,k is the attention weight from token tjto tokentkin headhat layerL. The entropyH(AL,h)is computed
over the full attention distribution within head hat layer
Lfor the current token position i. The depth-weighted
information density combines two key signals: model un-
certainty, where (1−pi(ti))increases when the model is
less conﬁdent about generating ti, and importance of at-
tention, measured by/summationtextH
h=1φh·Ah,i,L, which quantiﬁes
how much the model focuses on tiacross attention heads.
Entropy-based head weighting ( φh) is particularly relevant
for policy-optimized LLMs, as it prioritizes heads with dis -
tributed attention patterns. These heads excel at integrat -
ing broader information rather than local patterns, making
them more effective at detecting information needs. The
Semantic Filter ( si) excludes tokens unlikely to indicate in-
formation needs:
si=/braceleftBigg
0,ifti∈SorIsNumeric (ti)orIsPunctuation (ti)
1,otherwise
whereSis a predeﬁned set of stopwords. This ﬁlter im-
proves efﬁciency and accuracy by focusing on semantically
meaningful tokens. The scaling factor αdynamically mod-
ulates retrieval sensitivity based on computational load, en-
suring efﬁcient operation through a graceful reduction in
retrieval frequency. Essentially, when the LLM is “relaxed ”
(low demand), αmaintains higher retrieval sensitivity, pri-
oritizing external information lookup. Conversely, as the
LLM becomes “stressed” (resource constraints approach),
αsmoothly reduces retrieval sensitivity to prevent over-
load.
α=α0·e−λCcurrent
Cmax (20)
Here,α0(typically 0.7-1.0) sets the baseline sensitivity at
minimal load, and λ(typically 3-5) is the decay coefﬁcient
controlling the reduction rate. Careful selection of these hy-
perparameters, α0andλ, is important to balance retrieval
effectiveness and computational efﬁciency. Cmaxis the
maximum computational budget, and Ccurrent reﬂects real-
time resource usage. For RAG, Cmaxshould be conﬁgured
to 80-90% of available VRAM, with Ccurrent monitored via
metrics like GPU memory consumption. This exponen-
tial decay mechanism prioritizes retrieval when demand is
low, smoothly scaling it back under resource pressure, thus
maintaining efﬁciency and preventing system overload. In
summary, MLAG analyzes attention patterns across layers
and tokens to selectively trigger external information re-
trieval during text generation. Once retrieval is triggere d
by MLAG, an effective mechanism is needed to determine
what information to retrieve. We propose Layerwise Repre-
sentation Pooling (LRP), which constructs retrieval queri es
by selecting tokens from the preceding context based on
their relevance to the current token. Formally, for a given
tokentiat positioniin the sequence, LRP selects a subset
of preceding tokens:
7

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
LRP(ti) =SelectTopKTokens ({tj:j <i},k,relevance )
wherekis the number of tokens to select (typically 5-7 to-
kens), and relevance (tj)is a scoring function that measures
the importance of token tjrelative to the current token ti.
TheSelectTopKTokens function selects the top- kto-
kens from the preceding context {tj:j <i}based on their
relevance scores. We compute this relevance as a weighted
combination of attention-based and representation-based
similarities:
relevance (tj) =β·AttenScore (tj)+(1−β)·RepScore (tj)
whereβ∈[0,1]is a balancing parameter (optimally set to
0.7 in our experiments). This parameter balances the contri -
bution of attention and representation scores. The attenti on
score quantiﬁes the importance of token tjbased on the
attention patterns across all layers and heads:
AttenScore (tj) =L/summationdisplay
l=1ψl·1
HH/summationdisplay
h=1Al,h,i,j (21)
whereAl,h,i,j represents the attention weight from token ti
to tokentjin headhat layerl. Note that unlike MLAG
which uses attention towards the current token ( Aj,h,k,i ),
LRP uses attention from the current token to preceding to-
kens (Al,h,i,j ) to capture the relevance of past tokens in the
context of the current token being generated. ψlis a layer
importance coefﬁcient deﬁned as:
ψl=

0.2·l
L/3, ifl<L/3
0.5·l−L/3
L/3,ifL/3≤l<2L/3
0.3·L−l
L/3, otherwise(22)
This piecewise linear layer-weighting scheme, empiricall y
tuned for models like Qwen and LlaMA, prioritizes mid-
dle layers, as they are found to encode richer contextual in-
formation crucial for effective query formulation, and thi s
speciﬁc design has shown strong empirical performance for
the targeted LLM architectures. The representation score
captures semantic similarity between tokens using their
contextualized representations:
RepScore (tj) = cos(ej,ei) (23)
whereejandeiare contextualized embeddings for tokens
tjandti, respectively, computed as weighted averages of
layer-speciﬁc hidden states:
ej=L/summationdisplay
l=1δl·hl,j (24)
Here,hl,jrepresents the hidden state of token tjat layerl,
andδlis a layer-speciﬁc weight deﬁned as:
δl=exp(l/τ)/summationtextL
l′=1exp(l′/τ)(25)
whereτis a temperature parameter (typically set to 2.0).This temperature parameter concentrates weights towards
higher layers, emphasizing the role of deeper representa-
tions in capturing token semantics. While LRP does in-
volve computations for attention and representation score s,
including embedding calculations and cosine similarity, t he
overall computational overhead is managed by triggering
LRP only when MLAG detects an information need, thus
maintaining efﬁciency compared to always-on retrieval
methods. After selecting the top- ktokens based on their rel-
evance scores, we arrange them in their original sequence
order to preserve grammatical coherence. We then lever-
age the language capabilities of the policy-optimized LLM
itself to formulate a coherent query by passing these to-
kens through a simple prompt to produce a more effective
retrieval query. For instance, a prompt like “Formulate a
search query from these tokens: [selected tokens]” can be
used. The performance of LRP has been observed to be
superior to simpler query construction methods such as us-
ing only the current token or a ﬁxed window of preceding
tokens, as LRP dynamically selects semantically relevant
tokens based on both attention and representation metrics.
To maintain computational efﬁciency and prevent the re-
trieval process from becoming a bottleneck, we employ a
selective approach where LRP is not triggered for every
generated token. Instead, a computationally inexpensive
check ﬁrst determines if a potential information gap exists .
If True, indicating model uncertainty and semantic impor-
tance, it signals a potential need for external knowledge.
In such cases, we then engage the MLAG mechanism—
detailed in ATLAS—to rigorously conﬁrm this informa-
tion need through deeper analysis of the model’s inter-
nal states. Only if MLAG conﬁrms retrieval is necessary
do we proceed with LRP for query construction. The
ComputeRelevance check is deﬁned as:
ComputeRelevance (ti) =/braceleftBigg
True, ifpi(ti)<τpandsi= 1
False,otherwise
wherepi(ti)is the generation probability of token ti,τp
is a probability threshold (typically 0.5), and siis a binary
semantic ﬁlter.
2.2.1. C OMPUTATIONAL WORKFLOW AND
IMPLEMENTATION OF ATLAS:
The complete ATLAS workﬂow operates sequentially
across two key phases. In the token analysis phase, for
each generated token ti, the system ﬁrst computes its prob-
abilitypi(ti) =exp(zi(ti))/summationtext
v∈Vexp(zi(v))from model logits and ap-
plies the semantic ﬁlter sito identify meaningful tokens.
When conditions for analysis are met ( pi(ti)< τpand
si= 1), ATLAS calculates the Multi-Layer Attention Gra-
dient score MLAG (ti) =α·Gi·Di·siby analyzing at-
tention patterns across layers. If this score is deemed suf-
8

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
ﬁciently high to warrant retrieval, the system activates it s
retrieval mechanism. The query formulation phase then be-
gins, wherein Layerwise Representation Pooling computes
relevance scores for preceding tokens through a balanced
attention and semantic similarity formula: relevance (tj) =
β·AttenScore (tj) +(1−β)·RepScore (tj). Using these
scores, ATLAS selects the top- kmost relevant tokens via
LRP(ti) = SelectTokens ({tj:j < i},k,relevance ), pre-
serves their original sequence order for coherence, and con -
structs a focused retrieval query. After acquiring externa l
knowledge with this targeted query, it incorporates the re-
trieved information into the generation context, enabling
the language model to produce factually enhanced outputs
without modifying its underlying parameters.
3. Experiments
3.1. Datasets
We evaluate our proposed PORAG+ATLAS framework
and baselines using three benchmark datasets spanning dis-
tinct reasoning tasks: HotpotQA ( Yang et al. ,2018 ), Go-
rilla ( Patil et al. ,2024 ), and PubMedQA ( Jin et al. ,2019 ).
HotpotQA ( Yang et al. ,2018 ) is a large-scale multi-hop
question-answering dataset designed to test RAG frame-
works on complex reasoning across multiple sources. Each
instance includes a question, an answer, sentence-level su p-
porting facts, and a context comprising multiple Wikipedia
paragraphs, each structured as a (title, sentence-list) pa ir.
In the standard distractor setup ( Yang et al. ,2018 ) used
during training and evaluation, each question is paired
with two gold paragraphs and eight TF-IDF-retrieved dis-
tractors, challenging RAG frameworks to identify rele-
vant information amid noise. Gorilla ( Patil et al. ,2024 ),
which spans HuggingFace Hub, Torch Hub, and Tensor-
Flow Hub, focuses on code generation from machine learn-
ing instructions and is utilized for evaluating RAG frame-
works on API call generation. Each JSON entry contains
a natural language task description, detailed API docu-
mentation specifying the domain (e.g., classiﬁcation, ob-
ject detection), framework (PyTorch, TensorFlow), argu-
ments, setup, usage, and functionality, along with the cor-
responding ground-truth API call. During training, API
documentation is concatenated with the instruction to form
a retrieval-augmented prompt, enabling the RAG frame-
work to generate context-aware API calls. PubMedQA
(Jin et al. ,2019 ) is a biomedical QA dataset designed to
evaluate reasoning over scientiﬁc literature. Each sam-
ple includes a research question derived from a PubMed
title, a context (the abstract excluding its conclusion), a
long-form answer (the conclusion), and a ternary classiﬁ-
cation label (yes/no/maybe). The dataset combines expert-
annotated and machine-generated examples, providing a
rigorous benchmark for evidence-based biomedical reason-ing.
3.2. Evaluation Metrics
Evaluation metrics are tailored to each dataset’s reasonin g
requirements. For HotpotQA ( Yang et al. ,2018 ), we report
Exact Match (EM) and Micro F1 scores for both answer
prediction and supporting fact identiﬁcation, along with
Joint EM and Joint F1 scores, which require both compo-
nents to be correct simultaneously. These joint metrics re-
ﬂect the RAG framework’s combined retrieval and reason-
ing capabilities. For Gorilla ( Patil et al. ,2024 ), we employ
three metrics: (1) Overall Accuracy, based on Abstract Syn-
tax Tree (AST) subtree matching between predicted and
ground-truth API calls; (2) Hallucination Error, measurin g
instances of fabricated APIs; and (3) Wrong API Call Er-
ror, capturing valid but incorrectly selected or parameter -
ized APIs ( Patil et al. ,2024 ). Together, these metrics assess
both syntactic correctness and semantic alignment with
user intent. For PubMedQA ( Jin et al. ,2019 ), evaluation
is framed as a ternary classiﬁcation task (yes/no/maybe),
testing the RAG framework’s ability to derive factual con-
clusions from biomedical abstracts and mirror real-world
scientiﬁc reasoning.
3.3. Experimental Setup
Our experimental setup rigorously evaluates the integra-
tion of Policy-Optimized Retrieval-Augmented Generation
(PORAG) and Adaptive Token-Layer Attention Scoring
(ATLAS) using Transformer-based LLMs (e.g., Qwen2.5
0.5B/1.5B/3B or Llama 3.2 1B/3B). We selected these
base SLMs due to their strong performance, efﬁcient ar-
chitecture, and compatibility with low-rank ﬁne-tuning
techniques, which balance computational efﬁciency and
representational capacity for evaluating PORAG+ATLAS
frameworks. We employ Quantized Low-Rank Adaptation
(QLoRA) with frozen pre-trained weights quantized to 4-
bit NF4, updating only rank- r= 64 LoRA adapters ( α=
16, dropout = 0.05), targeting attention query/value projec-
tions and feed-forward layers as the sole trainable parame-
ters. These adapters are optimized using the PORAG objec-
tive, which combines group-relative policy improvement
with KL-regularized dual reward modeling for retrieval ﬁ-
delity and response quality. To rigorously evaluate our
framework’s components, we compare PORAG+ATLAS
against six key baselines: (1) PORAG-only isolates AT-
LAS’s contribution by showing policy optimization per-
formance without dynamic retrieval; (2) RAG+ATLAS
evaluates ATLAS’s standalone effectiveness with stan-
dard retrieval; (3) RAFT+ATLAS measures how AT-
LAS enhances existing retrieval augmented ﬁne-tuning ap-
proaches; (4) PORAG+DRAGIN benchmarks against al-
ternative dynamic retrieval methods; (5) GRPO+ATLAS
tests whether RAG-speciﬁc policy optimization is neces-
9

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
sary; and (6) RAG-base establishes the fundamental per-
formance benchmark. Training is conducted using the 8-bit
Adam optimizer with weight decay (AdamW), with policy
learning rates ηγ∈[1×10−6,5×10−6]; reward model
learning rate ηR= 5×10−5; group size G∈{2,4}; com-
posite reward weighting ( wﬁdelity= 0.7,wquality= 0.3);
KL-regularized objectives ( ω1= 100.0for policy optimiza-
tion,ω2= 0.1for divergence control); clipping parameters
(ǫ= 0.2for surrogate objectives, c1= 10.0for rewards);
and gradient management thresholds ( σmin= 0.1for mini-
mum advantage deviation, cvalue= 3.0,cnorm= 1.0). Dual
reward heads ( φ1,φ2) are jointly optimized using Lﬁdelity
andLquality loss functions, which combine ROUGE-1/2/L,
cosine similarity of sentence embeddings, and QA metrics
(EM/Micro F1). The ATLAS conﬁguration includes: dy-
namic retrieval scaling ( α0∈[0.7,1.0],λ∈[3,5]); Lay-
erwise Representation Pooling with β= 0.7attention-
representation balance; context selection using k∈[5,7]
tokens; a generation probability threshold τp= 0.5; and an
embedding temperature τ= 2.0. Using PyTorch hooks to
monitor attention weights and hidden states, ATLAS trig-
gers retrieval via Multi-Layer Attention Gradient (MLAG)
analysis and constructs queries using focused Layerwise
Representation Pooling (LRP). All experiments are con-
ducted on NVIDIA H100 GPUs using PyTorch 2.5 with
Hugging Face’s Transformers, Datasets, Accelerate, and
PEFT libraries.
3.4. Results
Our experimental results demonstrate the superior perfor-
mance of the PORAG+ATLAS framework across three
challenging benchmarks. On the HotpotQA multi-hop
question-answering dataset (Table 1), our model achieves
state-of-the-art results with 65.37% EM and 78.40% F1
for answer prediction, along with 60.21% EM and 82.01%
F1 for supporting fact retrieval. The joint evaluation
metrics (45.29% EM and 71.32% F1) represent substan-
tial improvements of +10.41% EM and +22.22% F1
over the RAG-base baseline. For the Gorilla API-aware
code generation benchmark (Table 2), the framework
achieves 76.38% accuracy while signiﬁcantly reducing crit -
ical errors—5.31% hallucination and 4.98% wrong API
calls—which are nearly half those of RAG-base (10.70%
and 9.58%, respectively). On the biomedical PubMedQA
dataset (Table 3), our model attains 78.35% accuracy and
74.56% F1, outperforming RAG-base by +17.65% accu-
racy and +15.26% F1. The framework generally sur-
passes ablation variants (PORAG-only, GRPO+ATLAS,
PORAG+DRAGIN) across the three benchmarks (Ta-
bles 1–3), demonstrating both the effectiveness of ATLAS
integration and PORAG’s superior architecture. These
comprehensive results validate that PORAG+ATLAS de-
livers robust improvements in retrieval precision and gen-eration accuracy while signiﬁcantly reducing critical err ors
across diverse domains, including multi-hop QA, code gen-
eration, and biomedical question answering.
3.4.1. A BLATION STUDIES
To rigorously validate our framework, we conduct abla-
tion studies examining both PORAG and ATLAS compo-
nents. (1). For Policy-Optimized RAG (PORAG), we ﬁrst
evaluate the dual reward mechanism by comparing the full
model (PORAG-Full) with default ﬁdelity/quality weights
(α= 0.7,β= 0.3) against three variants: (a) PORAG-
NF, which removes the ﬁdelity reward by setting α= 0,
β= 1; (b) PORAG-NQ, which disables the quality reward
withα= 1,β= 0; and (c) PORAG- α/β-Var, which
tests alternative weightings such as α=β= 0.5to ana-
lyze trade-offs. (2). We then assess optimization compo-
nents of PORAG by (a) replacing Group Relative Policy
Optimization (GRPO) with standard PPO in the PORAG-
PPO variant, (b) varying group sizes with G∈ {2,4}
usingG= 4 as the default, and (c) experimenting with
different KL divergence regularization strengths, speciﬁ -
callyω2∈{0.05,0.1,0.2}, to investigate its role in pre-
serving model stability and preventing catastrophic forge t-
ting usingω2= 0.1as the default. (3). For Adaptive
Token-Layer Attention Scoring (ATLAS), we ablate the
Multi-Layer Attention Gradient (MLAG) mechanism by
comparing the full method (ATLAS-Full) with default layer
weightsηj=j/(L−1), scaling factor α0= 0.8, and decay
λ= 4, against (a) a single-layer variant (ATLAS-Single) to
isolate the impact of depth-aware gradients, and (b) modi-
ﬁed layer weightings in which higher layers ( j >2L/3)
are weighted three times more heavily based on their task-
relevant abstraction capabilities. (4). To analyze the im-
pact of query formulation, we compare ATLAS-Full, which
uses dynamic token selection with a default top- k= 6and
attention-representation balance of β= 0.7, against (a) a
ﬁxed-window baseline (ATLAS-FixedLRP) that does not
rely on attention dynamics for token selection. (5). We
further study the role of the semantic ﬁlter siby remov-
ing it entirely in the ATLAS-noSF variant, which disables
the exclusion of stopwords, punctuation, and numeric to-
kens to assess its effect on retrieval precision. (6). Lastl y,
we examine the impact of dynamic retrieval scaling by
comparing the default exponential schedule, deﬁned as
α= 0.8·e−4Ccurrent/C maxwithCmax= 90% of VRAM
usage, against a static variant (ATLAS-Static) that uses a
constant sensitivity setting α≡1.0. These ablations iso-
late each individual contribution to the full system and
conﬁrm that both PORAG and ATLAS components play
critical and complementary roles in enhancing retrieval-
augmented generation. The ablation studies (Tables 4-6)
demonstrate that both PORAG and ATLAS components
contribute signiﬁcantly to the framework’s performance.
10

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 1. HotpotQA Performance (Higher is better for all metrics)
Model Answer Prediction Supporting Facts Joint
EM F1 EM F1 EM F1
PORAG+ATLAS (Proposed) 65.37 78.40 60.21 82.01 45.29 71.32
PORAG-only 63.85 77.10 58.32 80.20 44.62 69.88
GRPO+ATLAS 63.24 76.82 58.00 79.60 44.05 69.25
PORAG+DRAGIN 62.10 76.02 57.47 79.21 43.55 68.94
RAG+ATLAS 60.70 74.95 56.25 78.02 42.45 67.22
RAFT+ATLAS 59.85 73.88 55.14 77.15 41.75 66.30
RAG-base 52.10 64.02 44.21 61.28 34.88 49.10
Table 2. Gorilla Performance on Code Generation (Higher Accuracy an d Lower Error are better)
Model Overall Accuracy (%) Hallucination Error (%) Wrong API Call Error (%)
PORAG+ATLAS (Proposed) 76.38 5.31 4.98
PORAG-only 70.12 7.38 7.89
GRPO+ATLAS 73.26 6.52 5.83
PORAG+DRAGIN 71.96 6.84 5.92
RAG+ATLAS 70.84 6.40 5.85
RAFT+ATLAS 71.70 7.55 7.00
RAG-base 62.12 10.70 9.58
Table 3. PubMedQA Performance (Higher is better)
Model Accuracy (%) F1 Score (%)
PORAG+ATLAS (Proposed) 78.35 74.56
PORAG-only 75.25 72.83
GRPO+ATLAS 76.80 75.42
PORAG+DRAGIN 75.60 74.30
RAG+ATLAS 74.40 72.90
RAFT+ATLAS 73.20 71.60
RAG-base 60.70 59.30
The complete PORAG+ATLAS framework achieves opti-
mal balance across all components, with the ablation stud-
ies conﬁrming that each design choice contributes mean-
ingfully to the ﬁnal performance. In addition to the com-
prehensive ablation studies conducted on the PORAG and
ATLAS components, we investigate the sensitivity of the
MLAG retrieval trigger mechanism in ATLAS (see Ta-
ble7), focusing on two critical parameters: the baseline
scaling factor ( α0) and the generation probability threshold
(τp). The parameter α0(varied between 0.7–1.0) controls
retrieval sensitivity, with higher values increasing retr ieval
frequency under low computational load, while τp(tested at
0.3, 0.5, and 0.7) acts as a conﬁdence threshold—lower val-
ues trigger retrieval more readily under model uncertainty ,
whereas higher values risk missed retrievals. Our experi-
ments on HotpotQA systematically vary these parameters
while holding the core PORAG+ATLAS framework con-
stant. Analyzing the results reveals that the combination
ofα0= 0.8andτp= 0.5provides the optimal balance,
yielding the best performance across all reported metrics
(Answer EM/F1, Fact EM/F1, Joint EM/F1). τp= 0.5effectively balances retrieval timing, triggering interv en-
tions when the model’s token-generation conﬁdence falls
below this threshold, while α0= 0.8appropriately mod-
ulates the base retrieval sensitivity. These ﬁndings demon -
strate that ﬁne-tuning these speciﬁc trigger parameters ma x-
imizes retrieval efﬁcacy—improving answer accuracy and
supporting fact recall—while rigorously managing compu-
tational overhead. The results underscore the importance
of ATLAS’s adaptive retrieval mechanism, where precision-
tuned thresholds ( τp) and dynamic scaling ( α0) collectively
mitigate unnecessary retrievals without sacriﬁcing factu al
grounding.
3.4.2. A DDITIONAL EXPERIMENTS
Our experiments on benchmark datasets—HotpotQA, Go-
rilla, and PubMedQA—using various parameter variants of
Qwen2.5 (0.5B, 1.5B, and 3B) and Llama 3.2 (1B and 3B)
demonstrate that our integrated PORAG+ATLAS frame-
work consistently outperforms the baseline RAG approach.
For HotpotQA (Table 8), PORAG+ATLAS yields substan-
tial improvements, with Joint EM gains reaching up to
11

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 4. HotpotQA Ablation Results (Higher is better)
Variant Ans EM Ans F1 Fact EM Fact F1 Joint EM Joint F1
PORAG+ATLAS (Proposed) 65.37 78.40 60.21 82.01 45.29 71.32
PORAG Reward Variants
PORAG-NF ( α= 0,β= 1) 58.23 72.54 53.17 75.03 39.52 65.24
PORAG-NQ ( α= 1,β= 0) 57.85 72.06 52.73 74.62 38.91 64.72
PORAG-α/β-Var (0.5/0.5) 62.03 75.85 57.64 79.07 43.22 68.04
PORAG Optimization Variants
PORAG-PPO (vs GRPO) 60.04 74.13 55.82 77.53 41.52 66.31
PORAG-G2 (Group Size=2) 63.42 76.91 58.35 80.42 44.12 69.53
PORAG-KL-0.05 ( ω2= 0.05) 63.24 76.82 58.00 79.60 44.05 69.25
PORAG-K-L0.2 ( ω2= 0.2) 63.91 77.30 58.83 80.71 44.83 70.18
ATLAS Variants
ATLAS-Single (No MLAG) 63.12 76.23 58.04 79.32 43.83 68.72
ATLAS-FixedLRP (Static Tokens) 61.05 75.43 56.24 78.06 42. 03 67.05
ATLAS-noSF (No Semantic Filter) 62.53 76.85 57.83 79.07 43. 42 68.23
ATLAS-Static ( α≡1.0) 60.92 75.03 56.53 78.24 42.32 67.34
ATLAS-Layer3x (High Layer Focus) 63.85 77.12 58.92 80.35 44 .62 69.87
Table 5. Gorilla Ablation Results (Higher Accuracy and Lower Errors are better)
Variant Overall Accuracy (%) Hallucination Error (%) Wrong API Erro r (%)
PORAG+ATLAS (Proposed) 76.38 5.31 4.98
PORAG Reward Variants
PORAG-NF ( α= 0,β= 1) 71.83 6.91 5.27
PORAG-NQ ( α= 1,β= 0) 70.36 6.74 6.59
PORAG-α/β-Var (0.5/0.5) 74.92 5.14 5.43
PORAG Optimization Variants
PORAG-PPO (vs GRPO) 73.48 5.23 5.88
PORAG-G2 (Group Size=2) 75.12 5.42 5.12
PORAG-KL-0.05 ( ω2= 0.05) 74.63 5.67 5.34
PORAG-KL-0.2 ( ω2= 0.2) 75.84 5.38 5.07
ATLAS Variants
ATLAS-Single (No MLAG) 72.37 6.68 5.95
ATLAS-FixedLRP (Static Tokens) 71.29 6.82 5.31
ATLAS-noSF (No Semantic Filter) 73.46 5.95 5.78
ATLAS-Static ( α≡1.0) 72.63 6.82 5.19
ATLAS-Layer3x (High Layer Focus) 75.29 5.41 5.03
+10.4 points (Qwen2.5-3B: 45.29% vs 34.88%) and Joint
F1 gains exceeding +22.2 points (Qwen2.5-3B: 71.32% vs
49.10%) compared to the baseline models. In the Gorilla
code generation task (Table 9), our method achieves higher
overall accuracy across all variants (e.g., +14.3 points fo r
Qwen2.5-3B, reaching 76.38%) while signiﬁcantly reduc-
ing both hallucination and API errors (e.g., for Qwen2.5-
3B, hallucination reduced from 10.70% to 5.31% and API
errors decreased from 9.58% to 4.98%). Likewise, on
PubMedQA (Table 10), PORAG+ATLAS consistently de-
livers markedly improved accuracy and F1 scores, show-
casing substantial gains such as +17.6 points for accuracy
(Qwen2.5-3B: 78.35% vs 60.71%) and +15.3 points for F1
score (Qwen2.5-3B: 74.56% vs 59.30%). These results val-
idate that our framework robustly enhances retrieval ﬁdeli tyand generation quality across different LLM sizes and ar-
chitectures.
4. Conclusion
We present an integrated framework that enhances RAG
through the synergistic combination of Policy-Optimized
Retrieval-Augmented Generation (PORAG) and Adaptive
Token-Layer Attention Scoring (ATLAS). Our approach
demonstrates signiﬁcant improvements in factual accuracy ,
reduction of hallucinations, and computational efﬁciency
across diverse benchmarks. Extensive experiments and ab-
lation studies conﬁrm that the framework successfully bal-
ances retrieval ﬁdelity with generation quality while main -
taining low computational overhead. As a ﬂexible and scal-
12

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 6. PubMedQA Ablation Results (Higher is better)
Variant Accuracy (%) F1 Score (%)
PORAG+ATLAS (Proposed) 78.35 80.56
PORAG Reward Variants
PORAG-NF ( α= 0,β= 1) 72.57 74.83
PORAG-NQ ( α= 1,β= 0) 71.92 73.14
PORAG-α/β-Var (0.5/0.5) 75.63 77.29
PORAG Optimization Variants
PORAG-PPO (vs GRPO) 73.25 75.68
PORAG-G2 (Group Size=2) 76.42 78.93
PORAG-KL-0.05 ( ω2= 0.05) 76.85 79.12
PORAG-KL-0.2 ( ω2= 0.2) 77.03 79.84
ATLAS Variants
ATLAS-Single (No MLAG) 74.81 76.47
ATLAS-FixedLRP (Static Tokens) 72.19 74.36
ATLAS-noSF (No Semantic Filter) 75.29 77.91
ATLAS-Static ( α≡1.0) 73.94 75.52
ATLAS-Layer3x (High Layer Focus) 76.87 79.25
Table 7. Ablation Study on Retrieval Trigger Sensitivity in ATLAS
α0τp Answer EM (%) Answer F1 (%) Fact EM (%) Fact F1 (%) Joint EM (%) J oint F1 (%)
0.7 0.3 58.24 70.15 53.12 66.23 50.35 62.41
0.7 0.5 59.53 71.37 54.82 67.91 52.14 64.28
0.7 0.7 57.16 68.93 52.07 65.04 49.28 61.17
0.8 0.3 60.82 72.64 55.93 68.75 53.26 65.37
0.8 0.5 65.37 78.40 60.21 82.01 45.29 71.32
0.8 0.7 60.24 73.18 55.36 68.29 52.83 65.09
0.9 0.3 61.57 74.26 56.78 70.15 54.37 66.58
0.9 0.5 62.89 75.94 57.93 71.34 55.26 67.84
0.9 0.7 61.08 74.83 56.24 69.53 53.76 66.18
1.0 0.3 59.73 72.84 54.92 68.93 52.48 64.73
1.0 0.5 61.28 74.53 56.34 70.28 53.94 66.34
1.0 0.7 60.17 73.69 55.18 69.07 52.68 65.09
Table 8. HotpotQA Performance Comparison (Joint EM/F1; Higher is be tter)
LLM Variant Baseline RAG PORAG+ATLAS
Joint EM (%) Joint F1 (%) Joint EM (%) Joint F1 (%)
Qwen2.5-0.5B 25.73 38.42 30.88 43.17
Qwen2.5-1.5B 28.91 41.35 33.64 46.29
Qwen2.5-3B 34.88 49.10 45.29 71.32
Llama 3.2-1B 27.56 40.18 32.07 45.83
Llama 3.2-3B 30.24 44.76 38.59 52.41
Table 9. Gorilla Performance Comparison (Accuracy, Hallucination , API Errors)
LLM Variant Baseline RAG PORAG+ATLAS
Accuracy (%) Hallucination (%) API Error (%) Accuracy (%) Ha llucination (%) API Error (%)
Qwen2.5-0.5B 50.62 15.73 14.28 58.39 12.45 11.67
Qwen2.5-1.5B 54.17 13.82 12.91 62.84 10.53 9.24
Qwen2.5-3B 62.12 10.70 9.58 76.38 5.31 4.98
Llama 3.2-1B 52.48 14.36 13.75 60.92 11.83 10.47
Llama 3.2-3B 56.33 12.67 11.89 65.71 9.62 8.53
13

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 10. PubMedQA Performance Comparison (Accuracy and F1; Higher i s better)
LLM Variant Baseline RAG PORAG+ATLAS
Accuracy (%) F1 (%) Accuracy (%) F1 (%)
Qwen2.5-0.5B 48.35 50.82 55.67 57.93
Qwen2.5-1.5B 52.91 54.47 60.38 62.14
Qwen2.5-3B 60.71 59.30 78.35 74.56
Llama 3.2-1B 50.26 52.73 58.49 60.85
Llama 3.2-3B 54.88 56.42 63.17 65.39
able solution compatible with any Transformer-based lan-
guage model, our method represents a substantial advance-
ment for knowledge-intensive NLP tasks.
References
Chakraborty, S., Bhatt, S., Sehwag, U. M., Ghosal, S. S.,
Qiu, J., Wang, M., Manocha, D., Huang, F., Koppel, A.,
and Ganesh, S. Collab: Controlled decoding using mix-
ture of agents for llm alignment. In The Thirteenth Inter-
national Conference on Learning Representations .
Chan, B. J., Chen, C.-T., Cheng, J.-H., and Huang, H.-
H. Don’t do rag: When cache-augmented generation
is all you need for knowledge tasks. arXiv preprint
arXiv:2412.15605 , 2024.
Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre,
L., and Jumper, J. Accelerating large language model
decoding with speculative sampling. arXiv preprint
arXiv:2302.01318 , 2023.
Chen, G., Feng, Q., Ni, J., Li, X., and Shieh,
M. Q. Long-context inference with retrieval-
augmented speculative decoding, 2025a. URL
https://arxiv.org/abs/2502.20330 .
Chen, J., Ren, J., Chen, X., Yang, C., Sun, R., and
Arık, S. ¨O. Sets: Leveraging self-veriﬁcation and self-
correction for improved test-time scaling. arXiv preprint
arXiv:2501.19306 , 2025b.
Chen, Y ., Pan, X., Li, Y ., Ding, B., and Zhou, J.
A simple and provable scaling law for the test-time
compute of large language models. arXiv preprint
arXiv:2411.19477 , 2024.
Chen, Z., Chen, D., Sun, R., Liu, W., and Gan, C. Scaling
autonomous agents via automatic reward modeling and
planning. arXiv preprint arXiv:2502.12130 , 2025c.
Chow, Y ., Tennenholtz, G., Gur, I., Zhuang, V ., Dai, B.,
Thiagarajan, S., Boutilier, C., Agarwal, R., Kumar, A.,
and Faust, A. Inference-aware ﬁne-tuning for best-of-
n sampling in large language models. arXiv preprint
arXiv:2412.15287 , 2024.Corallo, G. and Papotti, P. Finch: Prompt-guided key-value
cache compression for large language models. Transac-
tions of the Association for Computational Linguistics ,
12:1517–1532, 2024.
Dao, T. Flashattention-2: Faster attention with bet-
ter parallelism and work partitioning. arXiv preprint
arXiv:2307.08691 , 2023.
Dao, T., Fu, D., Ermon, S., Rudra, A., and R´ e, C. Flashat-
tention: Fast and memory-efﬁcient exact attention with
io-awareness. Advances in neural information process-
ing systems , 35:16344–16359, 2022.
Das, S., Jin, L., Song, L., Mi, H., Peng, B., and Yu,
D. Entropy guided extrapolative decoding to improve
factuality in large language models. arXiv preprint
arXiv:2404.09338 , 2024.
Devoto, A., Zhao, Y ., Scardapane, S., and Minervini, P.
A simple and effective l2norm-based strategy for kv
cache compression. arXiv preprint arXiv:2406.11430 ,
2024.
Feng, X., Wan, Z., Wen, M., McAleer, S. M., Wen, Y .,
Zhang, W., and Wang, J. Alphazero-like tree-search can
guide large language model decoding and training. arXiv
preprint arXiv:2309.17179 , 2023.
Feng, Y ., Lv, J., Cao, Y ., Xie, X., and Zhou, S. K. Ada-
kv: Optimizing kv cache eviction by adaptive budget
allocation for efﬁcient llm inference. arXiv preprint
arXiv:2407.11550 , 2024.
Fu, Y ., Bailis, P., Stoica, I., and Zhang, H. Break the se-
quential dependency of llm inference using lookahead
decoding. arXiv preprint arXiv:2402.02057 , 2024.
Gao, Z., Niu, B., He, X., Xu, H., Liu, H., Liu, A., Hu,
X., and Wen, L. Interpretable contrastive monte carlo
tree search reasoning. arXiv preprint arXiv:2410.01707 ,
2024.
Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh ,
S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., and
Goldstein, T. Scaling up test-time compute with latent
14

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
reasoning: A recurrent depth approach. arXiv preprint
arXiv:2502.05171 , 2025.
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,
Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 , 2025.
Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney,
M. W., Shao, S., Keutzer, K., and Gholami, A. Kvquant:
Towards 10 million context length llm inference with kv
cache quantization. Advances in Neural Information Pro-
cessing Systems , 37:1270–1303, 2025.
Izacard, G. and Grave, E. Leveraging passage retrieval with
generative models for open domain question answering.
arXiv preprint arXiv:2007.01282 , 2020.
Ji, Y ., Li, J., Ye, H., Wu, K., Xu, J., Mo, L., and Zhang, M.
Test-time computing: from system-1 thinking to system-
2 thinking. arXiv preprint arXiv:2501.02497 , 2025.
Jiang, J., Chen, Z., Min, Y ., Chen, J., Cheng, X., Wang, J.,
Tang, Y ., Sun, H., Deng, J., Zhao, W. X., et al. Technical
report: Enhancing llm reasoning with reward-guided tree
search. arXiv preprint arXiv:2411.11694 , 2024.
Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., and Lu, X.
Pubmedqa: A dataset for biomedical research question
answering. arXiv preprint arXiv:1909.06146 , 2019.
Leviathan, Y ., Kalman, M., and Matias, Y . Fast inference
from transformers via speculative decoding. In Inter-
national Conference on Machine Learning , pp. 19274–
19286. PMLR, 2023.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin,
V ., Goyal, N., K¨ uttler, H., Lewis, M., Yih, W.-t.,
Rockt¨ aschel, T., et al. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in neural
information processing systems , 33:9459–9474, 2020.
Li, Y ., Huang, Y ., Yang, B., Venkitesh, B., Locatelli, A.,
Ye, H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm
knows what you are looking for before generation. Ad-
vances in Neural Information Processing Systems , 37:
22947–22970, 2025.
Lin, Z., Tang, Y ., Yao, X., Yin, D., Hu, Z., Sun, Y .,
and Chang, K.-W. Qlass: Boosting language agent in-
ference via q-guided stepwise search. arXiv preprint
arXiv:2502.02584 , 2025.
Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao,
C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-
v3 technical report. arXiv preprint arXiv:2412.19437 ,
2024.Liu, R., Gao, J., Zhao, J., Zhang, K., Li, X., Qi, B., Ouyang,
W., and Zhou, B. Can 1b llm surpass 405b llm? rethink-
ing compute-optimal test-time scaling. arXiv preprint
arXiv:2502.06703 , 2025.
Liu, X., Hu, L., Bailis, P., Cheung, A., Deng, Z., Stoica,
I., and Zhang, H. Online speculative decoding. arXiv
preprint arXiv:2310.07177 , 2023.
Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L.,
Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand` es, E.,
and Hashimoto, T. s1: Simple test-time scaling. arXiv
preprint arXiv:2501.19393 , 2025.
Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. Go-
rilla: Large language model connected with massive apis.
Advances in Neural Information Processing Systems , 37:
126544–126565, 2024.
Qi, Z., Ma, M., Xu, J., Zhang, L. L., Yang, F., and Yang, M.
Mutual reasoning makes smaller llms stronger problem-
solvers. arXiv preprint arXiv:2408.06195 , 2024.
Qian, H., Zhang, P., Liu, Z., Mao, K., and Dou, Z. Mem-
orag: Moving towards next-gen rag via memory-inspired
knowledge discovery. arXiv preprint arXiv:2409.05591 ,
2024.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y ., Wu, Y ., et al. Deepseekmath:
Pushing the limits of mathematical reasoning in open lan-
guage models. arXiv preprint arXiv:2402.03300 , 2024.
Simonds, T. Entropy adaptive decoding: Dynamic
model switching for efﬁcient inference. arXiv preprint
arXiv:2502.06833 , 2025.
Su, W., Tang, Y ., Ai, Q., Wu, Z., and Liu, Y . Dragin: Dy-
namic retrieval augmented generation based on the real-
time information needs of large language models. arxiv
2024. arXiv preprint arXiv:2403.10081 .
Su, W., Tang, Y ., Ai, Q., Yan, J., Wang, C., Wang, H., Ye,
Z., Zhou, Y ., and Liu, Y . Parametric retrieval augmented
generation. arXiv preprint arXiv:2501.15915 , 2025.
Tang, X., Wang, X., Zhao, W. X., and Wen, J.-R. Dawn-
icl: Strategic planning of problem-solving trajecto-
ries for zero-shot in-context learning. arXiv preprint
arXiv:2410.20215 , 2024.
Wang, E., Cassano, F., Wu, C., Bai, Y ., Song, W., Nath, V .,
Han, Z., Hendryx, S., Yue, S., and Zhang, H. Planning
in natural language improves llm search for code gener-
ation. arXiv preprint arXiv:2409.03733 , 2024a.
Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., and Zou,
J. Mixture-of-agents enhances large language model ca-
pabilities. arXiv preprint arXiv:2406.04692 , 2024b.
15

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Wang, L., Chen, H., Yang, N., Huang, X., Dou, Z., and
Wei, F. Chain-of-retrieval augmented generation. arXiv
preprint arXiv:2501.14342 , 2025.
Wang, X. and Zhou, D. Chain-of-thought reasoning with-
out prompting. arXiv preprint arXiv:2402.10200 , 2024.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang,
S., Chowdhery, A., and Zhou, D. Self-consistency im-
proves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171 , 2022.
Wang, Z., Wang, Z., Le, L., Zheng, H. S., Mishra, S., Perot,
V ., Zhang, Y ., Mattapalli, A., Taly, A., Shang, J., et al.
Speculative rag: Enhancing retrieval augmented genera-
tion through drafting. arXiv preprint arXiv:2407.08223 ,
2024c.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F.,
Chi, E., Le, Q. V ., Zhou, D., et al. Chain-of-thought
prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:
24824–24837, 2022.
Wu, J., Feng, M., Zhang, S., Jin, R., Che, F., Wen,
Z., and Tao, J. Boosting multimodal reasoning with
mcts-automated structured thinking. arXiv preprint
arXiv:2502.02339 , 2025.
Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H.,
Fu, Y ., and Han, S. Duoattention: Efﬁcient long-context
llm inference with retrieval and streaming heads. arXiv
preprint arXiv:2410.10819 , 2024.
Xie, Y ., Goyal, A., Zheng, W., Kan, M.-Y ., Lillicrap, T. P.,
Kawaguchi, K., and Shieh, M. Monte carlo tree search
boosts reasoning via iterative preference learning. arXiv
preprint arXiv:2405.00451 , 2024.
Xu, Y ., Jie, Z., Dong, H., Wang, L., Lu, X., Zhou, A.,
Saha, A., Xiong, C., and Sahoo, D. Think: Thin-
ner key cache by query-driven pruning. arXiv preprint
arXiv:2407.21018 , 2024.
Yan, M., Agarwal, S., and Venkataraman, S. Decoding
speculative decoding. arXiv preprint arXiv:2402.01528 ,
2024.
Yang, J., Hou, B., Wei, W., Bao, Y ., and Chang, S.
Kvlink: Accelerating large language models via efﬁcient
kv cache reuse. arXiv preprint arXiv:2502.16002 , 2025.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. Hotpotqa: A
dataset for diverse, explainable multi-hop question an-
swering. arXiv preprint arXiv:1809.09600 , 2018.Yoon, J., Cho, H., Baek, D., Bengio, Y ., and Ahn, S. Monte
carlo tree diffusion for system 2 planning. arXiv preprint
arXiv:2502.07202 , 2025.
Yu, Z., Yuan, Y ., Xiao, T. Z., Xia, F. F., Fu, J., Zhang, G.,
Lin, G., and Liu, W. Generating symbolic world models
via test-time scaling of large language models. arXiv
preprint arXiv:2502.04728 , 2025.
Zeng, Z., Cheng, Q., Yin, Z., Zhou, Y ., and Qiu, X. Revisit-
ing the test-time scaling of o1-like models: Do they truly
possess test-time scaling capabilities? arXiv preprint
arXiv:2502.12215 , 2025.
Zhang, D., Huang, X., Zhou, D., Li, Y ., and Ouyang, W.
Accessing gpt-4 level mathematical olympiad solutions
via monte carlo tree self-reﬁne with llama-3 8b. arXiv
preprint arXiv:2406.07394 , 2024a.
Zhang, S., Bao, Y ., and Huang, S. Edt: Improving large
language models’ generation by entropy-based dynamic
temperature sampling. arXiv preprint arXiv:2403.14541 ,
2024b.
Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Sto-
ica, I., and Gonzalez, J. E. Raft: Adapting language
model to domain speciﬁc rag. In First Conference on
Language Modeling , 2024c.
Zhang, X., Du, C., Du, C., Pang, T., Gao, W., and Lin,
M. Simlayerkv: A simple framework for layer-level
kv cache reduction. arXiv preprint arXiv:2410.13846 ,
2024d.
Zhang, Z., Ge, T., Liang, Z., Yu, W., Yu, D., Jia, M., Yu, D.,
and Jiang, M. Learn beyond the answer: Training lan-
guage models with reﬂection for mathematical reason-
ing. arXiv preprint arXiv:2406.12050 , 2024e.
Zhao, Y ., Yin, H., Zeng, B., Wang, H., Shi, T., Lyu, C.,
Wang, L., Luo, W., and Zhang, K. Marco-o1: Towards
open reasoning models for open-ended solutions. arXiv
preprint arXiv:2411.14405 , 2024.
16

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Algorithm 1 Group Relative Policy Optimization for Retrieval-Augment ed Generation (PORAG)
Input: Initial RAG policy model πγinit(with QLoRA adapters γ), reward models with parameters φ1andφ2(reward
heads), RAG training dataset D={(xi,di,y∗
i)}N
i=1, hyperparameters: clipping parameter ǫ(=0.2), ﬁdelity reward weight
α(=0.7), quality reward weight β(=0.3), reward clipping threshold c1(=10.0), reward scaling factor γscale, policy update
iterationsµ, group size G, policy learning rate ηγ, reward model learning rate ηR(ηR>ηγ), KL divergence weight ω2,
clipped surrogate objective weight ω1, minimum standard deviation σmin, gradient clipping value cvalue(=3.0), gradient
norm clipping cnorm(=1.0)
Output: Optimized RAG policy model πγ
1. Initialize RAG policy model: γ←γinit(QLoRA adapters)
2. For iteration i= 1,2,...,I do: (Main Training Epoch - Iterating over the dataset)
(a) Set reference model: πref←πγ
(b) For step j= 1,2,...,M do: (Mini-batch Update Step - Processing a batch of data)
i. Sample batchBjfrom datasetD
ii. Set old policy: πγold←πγ
iii. For each (x,d)∈Bj: (Group Output Generation and Reward Calculation for each d ata point in batch)
A. SampleGoutputs:{y(1),y(2),...,y(G)}∼πγold(·|x,d)
B. Compute dual rewards using reward heads ( φ1,φ2):
r(i)
ﬁdelity=Rﬁdelity(x,d,y(i);φ1)
r(i)
quality=Rquality(x,d,y(i);φ2)
C. Compute combined rewards: R(i)
combined=α·r(i)
ﬁdelity+β·r(i)
quality
D. Compute ﬁnal reward with clipping and scaling: R(i)
ﬁnal=clip(R(i)
combined,−c1,c1)·γscale
E. Compute group statistics using R(i)
ﬁnal:
µR=1
GG/summationdisplay
i=1R(i)
ﬁnal
σR= max
/radicaltp/radicalvertex/radicalvertex/radicalbt1
GG/summationdisplay
i=1(R(i)
ﬁnal−µR)2,σmin

F. Calculate advantages: ˆAi=R(i)
ﬁnal−µR
σR
iv. For GRPO iteration k= 1,2,...,µ do:(Inner Policy Optimization Loop - Multiple GRPO updates per
mini-batch)
A. Compute policy objective (token-level clipped surrogat e objective):
Lclip(γ) =1
GG/summationtext
i=11
|y(i)||y(i)|/summationtext
t=1min/parenleftBig
rt(γ)ˆAi,clip(rt(γ),1−ǫ,1+ǫ)ˆAi/parenrightBig
// Using sample-wise advantage
ˆAifor all tokens in y(i)
B. Compute KL regularization (sample-based approximation with token-averaging):
DKL(πγ||πref) =1
|Bj|/summationtext
(x,d)∈Bj1
GG/summationtext
i=11
|y(i)||y(i)|/summationtext
t=1KL(πref(·|x,d,y(i)
<t)||πγ(·|x,d,y(i)
<t))
C. Compute total objective: JGRPO-RAG(γ) =ω1·Lclip(γ)−ω2·DKL(πγ||πref)
D. Compute gradients: ∇γJGRPO-RAG(γ)
E. Clip gradients by value: ∇γJclipped=clip(∇γJGRPO-RAG(γ),−cvalue,cvalue)
F. Normalize gradients by norm: ∇γJnormalized=∇γJclipped
||∇γJclipped||2·min(||∇γJclipped||2,cnorm)
G. Update policy ( γ- QLoRA adapters only) with normalized gradients: γ←γ+ηγ∇γJnormalized
v. Update reward models (reward heads φ1,φ2) using reward losses: // Lﬁdelity (ROUGE),Lquality
(Semantic/QA Metrics)φ1←φ1+ηR∇φ1Lﬁdelity(φ1)
φ2←φ2+ηR∇φ2Lquality(φ2)
// Gradients do not affect base model weights
3. Return optimized RAG policy πγ17

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Algorithm 2 Adaptive Token-Layer Attention Scoring for Selective Retr ieval (ATLAS)
Input: Token sequence T//T: Input sequence of tokens, Pre-trained LLM // Pre-trained L LM: Fixed Pre-trained Large
Language Model, Hyperparameters ( τp,θ,k,β,τ,α 0,λ,C max) // Hyperparameters for ATLAS: τp: Probability threshold,
θ: MLAG threshold, k: Top-k tokens for LRP, β: Relevance balance, τ: Embedding temperature, α0: Base scaling factor,
λ: Decay coefﬁcient, Cmax: Max compute budget, Stopword set S //S: Set of stopwords, Model parameters
(L,H,V,ψ l,δl) // Model parameters: L: Layers,H: Heads,V: V ocabulary, ψl: LRP layer weights, δl: Embedding layer
weights
1.1. Initialization:
(a) 1.1. Set scaling factor: α=α0·e−λCcurrent
Cmax //α: Scaling factor, Ccurrent : Current compute usage
2.2. Token Analysis Phase (MLAG): // MLAG: Multi-Layer Attention Gradient
• 2.1. For each token tiin the sequence T: // ti: i-th token in sequence T
(a) 2.1.1. Compute Generation Probability: pi(ti) //pi(ti): Generation probability of token ti
(b) 2.1.2. Apply Semantic Filter: Determine si(0 or 1) based on ti //si: Semantic ﬁlter (1 if token is
semantically meaningful, 0 otherwise)
(c) 2.1.3. If pi(ti)<τpandsi= 1: // τp: Probability threshold
–2.1.3.1. Compute Multi-Layer Attention Gradient Score: ML AG(ti) =α·Gi·Di·si //Gi: Gradient
factor,Di: Depth-weighted information density
–2.1.3.2. If MLAG (ti)>θ: // θ: MLAG score threshold
*2.1.3.2.1. Retrieval Triggered for token ti
*2.1.3.2.2. Go to Query Formulation Phase (LRP) // LRP: Layerwise Representation Pooling
3.3. Query Formulation Phase (LRP):
• 3.1. If Retrieval Triggered:
(a) 3.1.1. Compute Relevance Scores: relevance (tj)for all preceding tokens tj //tj: Preceding token,
relevance (tj): Relevance score of token tj
(b) 3.1.2. Select Top-k Tokens: {tj1,...,tjk}=SelectTopK ({tj:j <i},k,relevance ) //k: Number of top
tokens to select
(c) 3.1.3. Formulate Query from Top-k Tokens
(d) 3.1.4. Output: Retrieval Query
(e) 3.2. Else:
i. 3.2.1. Output: No Retrieval Triggered
18

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
A. CRITIC: Cache Reduction via
Importance-based Token Inclusion
Criteria
Key-Value (KV) caching is essential in modern large lan-
guage models (LLMs) because it dramatically reduces com-
putational redundancy during autoregressive text genera-
tion. When generating text token by token, traditional ap-
proaches recalculate attention for all previous tokens wit h
each new prediction, leading to quadratic computational
complexity (O(n2)) that severely limits efﬁciency for long
sequences. In the standard self-attention mechanism, give n
a sequence of input tokens, each token is transformed into
a query vector ( Q), a key vector ( K), and a value vec-
tor (V) through learnable weight matrices: Q=XWQ,
K=XWK, andV=XWV, whereX∈Rn×dis
the matrix of input token embeddings, with nbeing the se-
quence length and dthe embedding dimension. Without
caching, for each new token, the attention weights are cal-
culated as softmax (QKT
√dh), whereQis the query matrix, K
is the key matrix, and dhis the head dimension. The scal-
ing factor√dhprevents extremely small gradients in the
softmax operation. The context vector is then computed
as softmax (QKT
√dh)V. KV caching stores these previously
computed key ( K) and value ( V) tensors from each layer
of the attention mechanism, eliminating the need to recom-
pute them for each generated token and reducing complex-
ity from quadratic to linear ( O(n)). Speciﬁcally, for the
t-th tokent, we compute Qt,Kt, andVtfor the new token
only. The cached keys and values, Kcached andVcached ,
contain the keys and values from tokens 1tot−1. The
attention weights are then computed as softmax (QtKT
√dh),
whereK= [Kcached;Kt]denotes the concatenation of
the cached keys and the current key. The context vector is
then computed as softmax (Qt[Kcached;Kt]T
√dh)[Vcached;Vt].
This signiﬁcantly reduces computation because we only
need to compute the attention weights and context vector
for the current token relative to the cached keys and values,
rather than recomputing the entire attention matrix for all
tokens at each step. This optimization yields substantial
speedups—often 2-10x faster inference—and enables pro-
cessing of much longer contexts than would otherwise be
possible given hardware constraints. However, as sequence
length grows, even with KV caching, memory usage be-
comes prohibitive since the cache size scales linearly with
sequence length and model size (number of layers, atten-
tion heads, and hidden dimension). The memory require-
ment is proportional to (L×H×2×n×dh×b)/8
bytes, where Lis the number of layers, His the num-
ber of attention heads per layer, the factor of 2 accounts
for both keys and values, nis the sequence length, dhis
the head dimension, and bis the number of bits in the
data type. It’s crucial to consider the data type’s precisio nwhen estimating memory usage; for instance, using half-
precision(‘bﬂoat16’) (b=16) signiﬁcantly reduces memory
compared to full-precision(‘ﬂoat32’) (b=32). This create s
a fundamental tension: while larger context windows en-
hance model capabilities by providing more information,
they also demand signiﬁcantly more memory resources,
creating a need for KV cache optimization techniques. The
challenge becomes particularly acute in real-world RAG
applications that beneﬁt from extended contexts. To miti-
gate the KV cache memory bottleneck, a variety of com-
pression techniques are employed, each with its own trade-
offs in terms of memory reduction, computational over-
head, and potential impact on model accuracy. Quantiza-
tion, a common technique, reduces numerical precision by
converting ﬂoating-point values to lower-bit integers usi ng
the formula xint=round(x−xmin
xmax−xmin×(2b−1)), where
brepresents the target bit width. This directly decreases
the memory footprint per value by representing values with
fewer bits, allowing for more efﬁcient storage of the KV
cache. Pruning selectively removes key-value pairs associ -
ated with less important attention heads, guided by impor-
tance scores such as sh=Ex∼D[||Ah(x)||F], whereEx∼D
denotes expectation over the data distribution, Ah(x)is the
attention matrix for head h, and||·||Fis the Frobenius
norm. This score shquantiﬁes the average importance of
attention head h. By removing the key-value pairs gener-
ated by these less important heads, pruning effectively re-
duces the representation of tokens within the cache from
the perspective of these less critical heads. This leads to
a smaller memory footprint because fewer key-value pairs
are stored for each token. Low-rank approximations de-
compose the key matrix Kinto the product USVT, where
U∈Rn×r,S∈Rr×r,V∈Rdk×r, and the rank ris much
smaller than both the sequence length nand the key di-
mensiondk. This decomposition dramatically reduces the
memory required to store the key matrix by representing
it with lower-dimensional components. Windowing strate-
gies, such as sliding window attention, preserve only the
most recent wtokens (Kcached=Kt−w:t−1). By lim-
iting the context window to the most recent tokens, win-
dowing directly reduces the sequence length and, conse-
quently, the memory needed for the keys and values in
the cache. These implementations can be categorized as
either static (where compression parameters are ﬁxed be-
fore inference) or dynamic (where parameters are adapted
during inference based on content importance). Dynamic
approaches have the potential to preserve generation qual-
ity by allocating resources more efﬁciently. Ultimately, e f-
fective KV cache implementation requires careful consid-
eration of hardware characteristics, memory management
strategies, data layout optimization, efﬁcient kernel des ign,
and the trade-offs between memory reduction, computa-
tional cost, and model accuracy. The impact of these tech-
niques on model accuracy can be measured through metrics
19

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
like attention entropy: H(Ai) =−/summationtext
jAijlogAij, where
Aijrepresents the normalized attention score from token i
to tokenj. Higher entropy indicates more distributed atten-
tion patterns, which may be more sensitive to aggressive
compression techniques.
A.1. Proposed Method
To address the substantial memory demands of large lan-
guage models during inference, this work introduces an
adaptive Key-Value (KV) cache compression strategy. This
technique selectively retains tokens based on their cal-
culated importance ( I), optimizing the trade-off between
memory footprint and model performance. The frame-
work is designed to be architecture-agnostic and imple-
ments a hybrid token importance strategy that integrates
attention-based, entropy-based, and gradient-based impo r-
tance measures. These measures are combined through a
weighted formulation to identify critical tokens within ea ch
attention layer of the language model. (a) The attention-
based importance strategy ( Iattn) quantiﬁes the strength of
a token’s relationships by calculating normalized attenti on
scores across the sequence. The process begins with com-
puting attention scores as the scaled dot product of the
query (Q∈Rn×dk) and key (K∈Rn×dk) matrices, rep-
resented as S∈Rn×n, wheredk=dmodel
his the dimen-
sion of each attention head in a multi-head attention mech-
anism. These scores are then transformed into probability
distributions using the softmax function, yielding attent ion
weightsA∈Rn×n. Since large language models have mul-
tiple layers ( L), these computations occur independently at
each layer, where Ql,Kl,Vlare computed for every layer
l∈{1,...,L}. The importance of each token is computed
by summing the absolute values of these attention weights
across all attention heads ( h) and all positions ( j) in the se-
quence: strengthi=/summationtext
h,j|Al
h,i,j|, whereAl
h,i,j represents
the attention weight of the i-th token in the l-th layer. This
raw strength metric is then normalized to the range [0,1]as
follows:
Iattn(i) =strengthi−min( strength)
max( strength)−min( strength)+ǫ,
whereǫis a small constant to prevent division by zero.
This normalization ensures comparable importance scores
across different sequences, model states, and layers. In
short, randomly discarding tokens from the KV cache can
degrade model performance by losing important contex-
tual information. Token importance varies across inputs
and contexts, making a dynamic approach essential. The
attention-based measure quantiﬁes token importance on-
the-ﬂy using current attention patterns, ensuring the rete n-
tion of the most relevant tokens that impact model predic-
tions. By leveraging existing attention computations duri ng
inference, it minimizes additional computational overhea d.
(b) The entropy-based importance strategy ( Ientropy ) lever-ages information theory principles to quantify the complex -
ity and diversity of a token’s attention patterns. After com -
puting attention probabilities using the standard scaled d ot-
product attention mechanism:
Al=softmax/parenleftbiggQl(Kl)T
√dk/parenrightbigg
, Al∈Rn×n,
whereQl,Kl,Vl∈Rn×dkare the query, key, and value
matrices at the l-th layer, and dk=dmodel
Hrepresents the
key dimension per attention head. The Shannon entropy
for each token’s attention distribution is then calculated as:
Hl(i) =−n/summationdisplay
j=1Al
i,jlog(Al
i,j+ǫ),
whereAl
i,jis the attention probability that the i-th token as-
signs to thej-th token in the l-th layer, and Hl(i)is the total
entropy for the i-th token at layer l. This entropy value cap-
tures how widely and evenly a token distributes its attentio n
across the sequence—higher entropy suggests the token has
more complex relationships with other tokens. The entropy
values are averaged across all attention heads ( H) to obtain
a comprehensive metric:
¯Hl(i) =1
HH/summationdisplay
h=1Hl
h(i),
whereHl
h(i)represents the Shannon entropy computed for
thei-th token in the h-th attention head of the l-th layer,
and¯Hl(i)is the entropy averaged across all heads for the
i-th token at layer l. Finally, these average entropy values
are normalized using min-max scaling:
Il
entropy(i) =¯Hl(i)−min(¯Hl)
max(¯Hl)−min(¯Hl)+ǫ,
whereǫis a small constant to prevent division by zero.
This normalization ensures comparable entropy-based im-
portance scores across different sequences and layers. Not
all tokens contribute equally to the model’s understand-
ing—some have simple, predictable relationships, while
others exhibit complex interactions. The entropy-based
measure quantiﬁes attention pattern complexity to identif y
and retain tokens with richer relationships. Tokens with
higher entropy-based importance scores maintain more
complex relationships within the sequence and are there-
fore prioritized for retention during compression. By leve r-
aging existing attention computations during inference, t his
approach minimizes additional computational overhead.
(c) The gradient-based importance strategy ( Il
grad(i)) di-
rectly measures each token’s contribution to model predic-
tion consistency using gradient information. It evaluates
the consistency between the current attention output and
the attention output of the same layer from the previous to-
ken generation step, representing the model’s prior belief
as follows:
Ll=MSE(Attentionl(Ql,Kl,Vl),Prevl),
20

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
where: Attentionl(Ql,Kl,Vl)∈Rn×dkrepresents the cur-
rent attention operation at layer l, Prevl∈Rn×dkdenotes
the attention output from the same attention layer lin the
previous decoding step. To mitigate memory consumption,
the implementation employs gradient checkpointing. The
gradients of this loss with respect to the key ( Kl) and value
(Vl) representations are computed as follows:
Gl
K=∂Ll
∂Kl∈Rn×dk, Gl
V=∂Ll
∂Vl∈Rn×dk,
The importance of each token is then determined by sum-
ming the absolute values of these gradients across all atten -
tion heads (H) at layerl:
Il
grad(i) =H/summationdisplay
h=1/parenleftbig
|Gl
K,h,i|+|Gl
V,h,i|/parenrightbig
∈R,
where:Il
grad(i)denotes the gradient-based importance
score for the i-th token at layer l,Gl
K,h,i∈RandGl
V,h,i∈
Rare the gradients of the loss function Llwith respect to
the key and value representations for attention head hat
layerl. This raw gradient-based importance is then normal-
ized:
Il
grad(i) =Il
grad(i)−min(Il
grad)
max(Il
grad)−min(Il
grad)+ǫ∈R,
where:ǫis a small constant to prevent division by zero. The
gradient-based approach provides a direct measure of how
sensitive the model’s predictions are to changes in each
token’s representations at layer l, highlighting tokens that
most signiﬁcantly inﬂuence the output. (d) The hybrid im-
portance strategy ( Ihybrid) combines the strengths of the pre-
vious approaches through a weighted combination of their
respective importance scores. This strategy is formulated
as follows:
Ihybrid(i) =wattn·Iattn(i)+wentropy·Ientropy(i)+wgrad·Igrad(i),
wherewattn,wentropy , andwgradare conﬁgurable weights
that sum to 1. This weighted sum is further normalized
to ensure values fall within the range [0,1]. The hybrid
approach provides ﬂexibility to customize the compression
behavior based on speciﬁc model characteristics allowing
implementers to balance the different aspects of token im-
portance according to their needs. Following the com-
putation of token importances using the hybrid strategy
(Ihybrid ), which integrates attention-based, entropy-based,
and gradient-based measures, the framework determines
the number of tokens to retain ( nc) in the Key-Value (KV)
cache. It is designed to optimize memory usage while pre-
serving model performance. The number of tokens to retain
is calculated as:
nc= min(max( m,⌊(1−r)·n⌋),n−1), (26)
whereris the compression ratio (typically between 0.1 and
0.5), andmis a minimum token count. It ensures that atleastmtokens are retained while also preserving at least
one token for potential removal, guaranteeing nc<n. The
minimum token count ( m) prevents excessive compression
that could degrade model performance, while the upper
bound (n−1) ensures the integrity of the sequence by al-
ways leaving at least one token available for removal. Once
ncis determined, the framework selects the tokens with the
highest importance scores for retention using a top- koper-
ation:
SelectedTokens =TopK(Ihybrid,nc), (27)
whereIhybrid is the vector of hybrid importance scores for
all tokens in the sequence, and TopK (·,nc)selects thenc
tokens with the highest scores. This approach ensures that
only the most critical tokens, which signiﬁcantly inﬂuence
model predictions, are retained, optimizing memory usage
without compromising performance. To minimize compu-
tational overhead, the framework incorporates a delayed
caching mechanism. Compression is initiated only after
processing a minimum number of tokens ( m), ensuring that
shorter sequences (with fewer than mtokens) operate with-
out compression. This threshold-based approach ensures
that compression overhead is incurred only when the ben-
eﬁts of memory savings outweigh the computational costs,
making the framework practical for sequences of varying
lengths. Additionally, the framework dynamically adjusts
the compression ratio based on current memory usage to
balance memory savings and model performance. The
adaptive compression ratio ( radaptive ) is computed as:
radaptive= min(rbase+α·Mused
Mtotal,rmax), (28)
whereMused represents current memory consumption,
Mtotalis the total available memory, αis a tunable parame-
ter controlling adaptation sensitivity, rbaseis the base com-
pression ratio, and rmaxis the maximum allowable compres-
sion ratio. This adaptive mechanism increases compres-
sion when memory pressure is high and relaxes it when re-
sources are abundant, ensuring efﬁcient memory utilizatio n
without exceeding hardware limits. In summary, the frame-
work combines a hybrid importance calculation, token re-
tention logic, delayed caching, and adaptive compression t o
achieve efﬁcient memory usage while maintaining model
performance in RAG contexts. This makes it particularly
suitable for deployment in large language models, espe-
cially in long-context applications where memory demands
are signiﬁcant. During text generation, the framework im-
plements a phased approach to adaptive KV cache com-
pression. Initially, tokens are collected without compres -
sion until a minimum token threshold ( m) is reached, en-
suring that shorter sequences operate without compression
to minimize unnecessary computational overhead. Once
the threshold is exceeded, the framework performs a se-
ries of steps for each generated token: it extracts hidden
states and computes query, key, and value projections; ap-
21

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
pends keys and values to an accumulation buffer while
tracking the total number of processed tokens; concatenate s
all cached keys and values when the token count exceeds
the threshold; computes attention scores between the cur-
rent queries and the cached keys; calculates token impor-
tances using the selected strategy (e.g., the hybrid strate gy
Ihybrid ); selects the top- kmost important tokens based on
their importance scores; reconstructs the KV cache with
the selected tokens, discarding less important ones; and up -
dates compression statistics to track memory savings and
performance impact. CRITIC reconstructs the KV cache
after importance-based compression, preserving sequence
integrity. By retaining the most critical tokens and synchr o-
nizing their positional indices, it prevents token misalig n-
ment—essential for autoregressive text generation where
self-attention relies on sequential dependencies. This re -
construction enables long-sequence processing while opti -
mizing memory usage, ensuring model ﬂuency and contex-
tual coherence. This phased approach ensures that com-
pression is applied only when necessary (after processing a t
leastmtokens) and dynamically adapts to the importance
of tokens in the sequence, optimizing memory usage while
preserving model performance.
A.2. CRITIC Evaluation
The evaluation of the CRITIC module’s impact on the
PORAG+ATLAS framework reveals a modest perfor-
mance trade-off that accompanies signiﬁcant efﬁciency
gains across all benchmark datasets. As shown in Ta-
ble11, the Qwen2.5-3B model with CRITIC integration
experiences only slight decreases in HotpotQA metrics,
with Joint EM dropping from 45.29% to 42.37% and
Joint F1 declining from 71.32% to 67.95%. Similarly,
Table 12demonstrates minor reductions in Gorilla per-
formance, where overall accuracy falls marginally from
76.38% to 73.85% while wrong API calls see a small in-
crease from 4.98% to 6.77%. The PubMedQA results in Ta-
ble13follow this pattern, showing slight dips in both accu-
racy (78.35% to 74.62%) and F1 score (74.56% to 69.83%).
These minimal quality trade-offs are offset by substantial
efﬁciency improvements, as evidenced in Table 14, where
latency is nearly halved from 68.27 seconds to 34.19 sec-
onds and throughput more than doubles from 120 to 242
tokens per second. The consistent but modest performance
impact suggests that CRITIC’s memory optimization strat-
egy successfully balances computational beneﬁts with ac-
ceptable quality preservation, making it particularly val u-
able for applications where efﬁciency is prioritized witho ut
signiﬁcantly compromising output accuracy.
A.3. Computational Complexity
The computational complexity of our adaptive KV cache
compression framework is dominated by token importanceTable 11. HotpotQA Quality Metrics
Model Joint EM (%) Joint F1 (%)
PORAG+ATLAS (Baseline) 45.29 71.32
PORAG+ATLAS + CRITIC 42.37 67.95
Table 12. Gorilla Quality Metrics
Model Overall Acc. (%) Wrong API (%)
PORAG+ATLAS (Baseline) 76.38 4.98
PORAG+ATLAS + CRITIC 73.85 6.77
Table 13. PubMedQA Quality Metrics
Model Accuracy (%) F1 (%)
PORAG+ATLAS (Baseline) 78.35 74.56
PORAG+ATLAS + CRITIC 74.62 69.83
Table 14. Efﬁciency Metrics
Model Latency (sec) Tokens/sec ( ↑)
PORAG+ATLAS (Baseline) 68.27 120
PORAG+ATLAS + CRITIC 34.19 242
computation and token selection. Given a sequence of
lengthn, withHattention heads, key/value dimension
d, and batch size b, computing token importance requires
O(bHn2d)operations for attention-based and entropy-
based strategies, matching standard self-attention compl ex-
ity. The gradient-based strategy adds backpropagation ove r-
head but remains O(bHn2d)asymptotically, with gradi-
ent checkpointing minimizing memory overhead. Token
selection, using a top- koperation, has a complexity of
O(bnlogn)with heap-based selection, where k=nc.
The number of retained tokens ncis calculated as nc=
min(max(m,⌊(1−r)·n⌋),n−1), ensuring at least m
tokens are kept and one token is removed. This reduces the
memory footprint from O(bHnd)toO(bHncd), achieving
a reduction factor ofnc
n. Compression is triggered only
when the sequence length exceeds m, minimizing overhead
for short sequences, while the adaptive compression ratio
dynamically adjusts rbased on memory pressure, balanc-
ing efﬁciency and performance.
B. Comparing PORAG and RAFT
Methodologies
Policy-Optimized Retrieval-Augmented Generation
(PORAG) and Retrieval-Augmented Fine-Tuning
(RAFT) ( Zhang et al. ,2024c ) offer fundamentally dif-
ferent strategies for optimizing RAG systems. RAFT
employs supervised ﬁne-tuning (SFT) on static, curated
datasets containing predeﬁned question-response pairs
accompanied by both relevant (“golden”) and irrelevant
22

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
(“distractor”) documents. It optimizes indirectly by
teaching the model to differentiate between useful and
distracting documents through explicit training examples
and incorporates logical reasoning via Chain-of-Thought
(CoT) prompts. However, RAFT is inherently limited
by its reliance on predeﬁned data, single-objective
cross-entropy optimization, and its inability to explicit ly
optimize retrieval ﬁdelity and generation quality inde-
pendently. In contrast, PORAG employs Group Relative
Policy Optimization (GRPO), an advanced reinforce-
ment learning method, to directly optimize multiple
generation quality dimensions simultaneously through
specialized reward models. PORAG dynamically gener-
ates policy-driven training samples, directly optimizing
retrieval ﬁdelity—how faithfully retrieved information
is reﬂected—and response quality, including coherence,
ﬂuency, and helpfulness. Unlike RAFT, PORAG im-
plicitly and dynamically handles distractors through
reward modeling and advantage estimation rather than
explicitly embedding distractors in supervised training
sets. Additionally, PORAG incorporates explicit advantag e
estimation and KL-divergence regularization during polic y
updates to maintain controlled adaptation in retrieval-
augmented generation. This stabilizes training, prevents
drastic policy shifts, and balances retrieval ﬁdelity with
the model’s inherent parametric knowledge, enhancing
robustness and generalization across retrieval scenarios .
In contrast, RAFT provides robustness primarily within
domain-speciﬁc scenarios due to its explicit distractor-
aware ﬁne-tuning but lacks dynamic adaptability beyond
its predeﬁned training context. In summary, PORAG
offers greater deployment ﬂexibility, nuanced generation
optimization, and dynamic adaptability, addressing key
limitations of RAFT related to static supervision, single-
strategy optimization, and the lack of direct optimization
of retrieval ﬁdelity and response quality.
C. Comparing DRAGIN and ATLAS
Methodologies
Dynamic Retrieval Augmented Generation based on the
Information Needs of Large Language Models (DRA-
GIN) ( Su et al. ) and Adaptive Token-Layer Attention Scor-
ing for Selective Retrieval (ATLAS) both dynamically
determine the optimal timing (when retrieval should oc-
cur) and the speciﬁc content to retrieve (query formula-
tion) based on the internal states and immediate informa-
tional needs of the language model during text generation.
DRAGIN primarily leverages ﬁnal-layer self-attention to
identify real-time information gaps. Conversely, ATLAS
employs a sophisticated Multi-Layer Attention Gradient
(MLAG) analysis, explicitly quantifying attention shifts
across multiple transformer layers to capture nuanced tran -
sitions indicative of deeper knowledge gaps. For queryformulation, DRAGIN constructs retrieval queries using at -
tention patterns from the ﬁnal layer, combined with token-
level semantic ﬁlters. ATLAS, in contrast, integrates Lay-
erwise Representation Pooling (LRP), combining seman-
tic similarity and attention scores across layers, along
with token-level semantic ﬁlters, to form retrieval querie s,
thereby enhancing semantic precision. In terms of resource
management, ATLAS explicitly considers real-time com-
putational load via a dynamic scaling factor, optimizing
retrieval frequency relative to resource availability. DR A-
GIN utilizes a simpler exponential scaling factor, adjust-
ing retrieval sensitivity based on resource usage, but with -
out the ﬁne-grained computational tracking featured in
ATLAS. Overall, ATLAS’s integrated, multi-layer atten-
tion and resource-aware approach offers superior adapt-
ability and accuracy in dynamically identifying subtle re-
trieval needs, while DRAGIN presents a simpler ﬁnal-layer
attention-driven strategy, achieving computational simp lic-
ity at the potential cost of retrieval precision depth.
D. Test-Time Scaling of LLMs
Test-time scaling inference for Large Language Models
(LLMs) leverages advanced algorithmic techniques de-
signed to enhance model outputs without altering the un-
derlying weights. These methods dynamically adjust rea-
soning depth, sampling strategies, and validation process es
during inference, optimizing efﬁciency and output qual-
ity in real time. This approach is particularly valuable
in resource-constrained environments where retraining or
ﬁne-tuning models is impractical. By strategically scalin g
complexity based on task demands, these techniques en-
able LLMs to navigate complex problem spaces more ef-
fectively, ensuring robust decision-making, improved ac-
curacy, and reduced computational costs. At its core,
test-time scaling in LLMs can be mathematically modeled
through a utility-cost optimization framework. By deﬁning
U(q,c)as the utility function where qrepresents output
quality and crepresents computational cost, and fθ(x,s)
as the LLM function with parameters θ, inputx, and scal-
ing strategy s, we can formulate the fundamental objective
as maximizing utility while managing resource constraints :
maxs∈SU(q(fθ(x,s)),c(s))subject toc(s)≤Cmax,
whereSrepresents the set of all possible test-time scaling
strategies,q(fθ(x,s))measures the quality of model out-
puts,c(s)represents the computational cost of strategy s,
andCmax is the maximum allowable computational bud-
get. This mathematical formulation captures the essential
trade-off that underlies all test-time scaling approaches . A
form of Weak-to-Strong Distillation serves as a founda-
tional strategy for test-time scaling inference technique s,
where diverse preliminary outputs are generated and iter-
atively reﬁned to enhance reasoning and accuracy. This ap-
proach improves robustness by progressively strengthenin g
23

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
outputs through evaluation and reﬁnement, ensuring accu-
rate and consistent results. These inference techniques re p-
resent advanced strategies for test-time scaling in LLMs,
signiﬁcantly enhancing language model capabilities by im-
plementing metacognitive processes such as decomposing
problems, evaluating intermediate results, and reﬁning so -
lutions—effectively mimicking human deliberative reason -
ing while maintaining inference efﬁciency. By dynami-
cally adjusting computational resources during inference
and scaling complexity only when necessary, these meth-
ods optimize both efﬁciency and output quality. This adap-
tive approach boosts accuracy, minimizes hallucinations
and logical errors, and enhances the suitability of LLMs
for high-stakes decision-making scenarios.
D.1. Self-Consistency Algorithm
Self-Consistency ( Wang et al. ,2022 ;Ji et al. ,2025 ) en-
hances model reliability by generating multiple indepen-
dent reasoning trajectories and selecting the most consis-
tent answer through stochastic decoding. Let Mbe a lan-
guage model with parameters θandxbe an input query.
The Self-Consistency framework can be formalized as fol-
lows:
y∗= argmax
y∈Yk/summationdisplay
i=11[y=yi] (29)
whereY={y1,y2,...,y k}is the set of ksampled re-
sponses, generated as yi∼pMθ(y|x,T)with temperature
T >0. Here, 1[·]is the indicator function used to identify
the frequency of each response y∗within the sampled re-
sponses. The goal is to select the most frequently occurring
response, which is considered the most consistent answer.
Speciﬁcally, argmax ﬁnds the response ythat maximizes
the count of identical responses among the samples. To
achieve this, the Self-Consistency algorithm ﬁrst creates di-
verse solution attempts using temperature-controlled sam -
pling. Then, it computes a similarity matrix S∈Rk×k,
where each element Sijrepresents the semantic similarity
between responses yiandyj:
Sij=sim(yi,yj) (30)
This similarity can be quantiﬁed using various met-
rics, including string similarity, Levenshtein distance, or
embedding-based cosine similarity, allowing for the ident i-
ﬁcation of conceptually equivalent answers despite surfac e-
level variations. Next, the framework employs a clustering
algorithm with a predeﬁned similarity threshold τto group
responses into clusters C={C1,C2,...,C m}, where
m≤k:
Ci={yj∈Y|∀yj,yl∈Ci,Sjl≥τ} (31)
whereCirepresents a cluster of responses, a subset of the
sampled responses Y, such that every pair of responses
withinCihas a similarity score of τor higher. To as-sess these clusters, the framework analyzes their statisti -
cal distribution by examining: (1) Cluster size: The num-
ber of responses in each cluster, |Ci|, which serves as
the primary factor in determining the most frequent an-
swer pattern. (2) Intra-cluster coherence: coh (Ci) =
1
|Ci|(|Ci|−1)/summationtext
yj,yl∈Ci,j/ne}ationslash=lSjl, measuring the internal con-
sistency within each cluster and indicating the semantic
closeness of responses beyond the similarity threshold. (3 )
Response quality metrics: Metrics like perplexity, entrop y,
and response length, which offer additional insights into t he
conﬁdence and quality of individual responses within each
cluster, contributing to a broader understanding of cluste r
reliability. While the ﬁnal output selection in this basic
formulation is determined by identifying the largest clust er
based on cluster size, as formalized below:
y∗= argmax
Ci∈C(|Ci|) (32)
the intra-cluster coherence and response quality metrics
provide valuable supplementary information for analyzing
the clusters and potentially reﬁning the answer selection
process in more advanced implementations. The overall
process follows a pipeline of: (a) Stochastic sampling: Y=
{yi∼pMθ(y|x,T)|i∈ {1,2,...,k}}, (b) Similarity
computation: Sij=sim(yi,yj),∀i,j∈{1,2,...,k}, (c)
Clustering:C=cluster(Y,S,τ), and (d) Statistical analy-
sis:y∗= argmax
Ci∈C|Ci|. By emphasizing high-probability
reasoning paths and de-emphasizing less common trajec-
tories susceptible to errors, Self-Consistency effective ly
achieves a form of implicit ensemble learning within a
single model’s parameter space. This method leverages
Shannon entropy minimization to ﬁlter out stochastic noise
and converge on consistently correct answers. The entropy
of the ﬁnal distribution H(pMθ(y|x,C)), which represents
the uncertainty in the model’s output after applying Self-
Consistency, is typically lower than the entropy of indi-
vidual samples H(pMθ(y|x)). This reduction in entropy
indicates that the probability distribution is more focuse d,
ideally concentrating around the most consistent and cor-
rect answer, y∗. Furthermore, this technique inherently
employs Weak-to-Strong Distillation by generating divers e
outputs that represent different regions of the model’s pro b-
ability distribution, and subsequently reﬁning the answer
through consistency checks and majority voting to attain
robust convergence on the most globally reliable solution.
D.1.1. C OMPUTATIONAL TIMECOMPLEXITY
Self-consistency increases computational cost compared t o
standard language model inference, shifting from O(n)to
O(k×n+2k2). This complexity arises from:
24

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Time Complexity =O(k×n)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Response Generation+O(k2)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Similarity Computation
+O(Clustering Algorithm Complexity )/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
ClusteringGenerating kresponses contributes O(k×n),
while pairwise similarity computation requires
O(k2). The clustering complexity, denoted as
O(Clustering Algorithm Complexity ), depends on the
speciﬁc algorithm used; a simpliﬁed approximation also
yieldsO(k2). Thus, considering both similarity compu-
tation and clustering as potentially O(k2)operations, the
overall time complexity is O(k×n+ 2k2). While in
asymptotic notation O(2k2) =O(k2), the ﬁnal complexity
ofO(k×n+k2)results in an increased computational cost
compared to the O(n)complexity of standard inference.
This highlights the trade-off between computational cost
and enhanced answer consistency.
D.2. Best-of-N Sampling Algorithm
Best-of-N sampling ( Chow et al. ,2024 ) improves output
quality by generating several candidate responses and se-
lecting the highest-rated response using explicit quality as-
sessment. This method creates diverse solution attempts
via stochastic decoding with temperature-controlled sam-
pling, then employs a systematic rating mechanism where
the model evaluates each candidate on a numerical scale (0-
10) based on speciﬁc quality criteria including clarity, ac cu-
racy, and helpfulness. Let Mrepresent the language model,
sbe the system prompt, and xbe the user query. The Best-
of-N sampling procedure can be formalized as follows:
C={y1,y2,...,y k}whereyi∼M(y|s,x,τg)(33)
Where,C={y1,y2,...,y k}is the set of kgenerated can-
didate responses. yirepresents the i-th candidate response,
which is sampled from the language model M. The sam-
pling is conditioned on the system prompt s, the user query
x, and the generation temperature τg.
ri=M(r|sr,x,yi,τr)∀i∈{1,2,...,k} (34)
Where,riis the rating assigned to the i-th candidate re-
sponseyi. This rating is generated by the same language
modelM, but now acting as a rater. The rating is based
on a specialized system prompt for rating sr(”Rate the fol-
lowing response from 0-10 based on clarity, accuracy, and
helpfulness. Respond with ONLY a number)”), the user
queryx, the candidate response yi, and the rating temper-
atureτr. The rating temperature τris typically set to low
values to ensure consistent evaluations.
y∗= argmax
yi∈Cri (35)
y∗is the ﬁnal selected response. It is chosen by ﬁnding
the candidate response yifrom the setCthat has the high-est ratingri. The framework implements a dual-role ar-
chitecture where the model ﬁrst functions as a generator
producing multiple completions, then transitions to an eva l-
uator by processing each completion with a specialized rat-
ing prompt. By ﬁltering through multiple solution trajecto -
ries, Best-of-N sampling enhances output reliability and a c-
curacy, reducing logical inconsistencies and factual erro rs
that might appear in any single response. By leveraging
the model’s ability to generate and evaluate responses, the
algorithm creates a robust internal quality control mecha-
nism that enhances the reliability and accuracy of the ﬁnal
output. The approach leverages Weak-to-Strong Distilla-
tion principles by ﬁrst generating multiple outputs of vary -
ing quality (the “weak” learning phase) and then using the
model’s own evaluation capabilities to identify and select
the strongest output (the “strong” distillation phase). Th is
creates a knowledge transfer process where weaker outputs
inform the selection of the optimal solution.
D.2.1. C OMPUTATIONAL TIMECOMPLEXITY
Best-of-N sampling increases computational cost com-
pared to standard language model inference, shifting from
O(n)toO(k×n). This complexity arises from the need
to generate and evaluate kcandidate responses. The time
complexity can be broken down into the following compo-
nents:
Time Complexity =O(k×n)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Response Generation+O(k×n)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Response Rating
+O(k)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Response Selection
Generatingkcandidate responses, each of average length n,
contributesO(k×n). Subsequently, rating each of these
kresponses, which also involves a forward pass through
the language model, adds another O(k×n)component.
Finally, selecting the best response from the krated re-
sponses based on their scores takes O(k)time. Sum-
ming these components, the overall time complexity is
O(k×n+k×n+k) =O(2kn+k). In asymptotic
notation, this simpliﬁes to O(k×n), as the term kbe-
comes less signiﬁcant compared to knwhennis sufﬁ-
ciently large. This complexity highlights that the compu-
tational cost of Best-of-N sampling scales linearly with th e
number of candidate responses k, representing a trade-off
for the enhanced output quality achieved through explicit
response evaluation, yet remaining more computationally
efﬁcient in terms of asymptotic complexity compared to
Self-Consistency which includes a quadratic component.
25

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
D.2.2. C OMPARING BEST-OF-N S AMPLING AND
SELF-CONSISTENCY
While both Best-of-N Sampling and Self-Consistency en-
hance output quality by generating multiple responses,
their core distinction lies in the answer selection mecha-
nism. Best-of-N Sampling employs an explicit quality as-
sessment: it leverages the language model itself to rate
each generated candidate response based on deﬁned cri-
teria such as clarity, accuracy, and helpfulness. The re-
sponse with the highest rating is then chosen as the ﬁ-
nal output. In contrast, Self-Consistency utilizes an im-
plicit evaluation approach. It focuses on identifying the
most consistent reasoning pattern across the generated re-
sponses through similarity clustering. By grouping seman-
tically similar outputs and selecting the most frequent clu s-
ter, Self-Consistency implicitly evaluates responses bas ed
on their agreement with each other, without requiring ex-
plicit quality ratings for each individual response. Thus,
Self-Consistency measures conceptual consensus among
multiple reasoning paths, whereas Best-of-N directly as-
sesses the quality of each individual output. This funda-
mental difference underscores two distinct strategies for en-
hancing LLM output quality: direct, model-driven quality
evaluation of individual responses versus statistical val ida-
tion through inter-response agreement.
D.3. Chain-of-Thought with Reﬂection
Chain-of-Thought with Reﬂection ( Zhang et al. ,2024e ;
Wang & Zhou ,2024 ) enhances reasoning capabilities by
structuring the problem-solving process into distinct con -
ceptual phases that emulate human cognitive processes.
This approach decomposes the reasoning task into three
sequential components within a single generative process.
LetMθdenote a language model with parameters θ, and
letqrepresent an input query. We formalize the Chain-of-
Thought with Reﬂection process as follows:
R=Mθ(P(q)), (36)
whereRis the model’s response generated using a struc-
tured prompt P(q). While the response is generated in
a single forward pass, it can be conceptually decomposed
into three functional components:
R= [RT,RR,RO], (37)
where:RTrepresents the systematic decomposition of the
problem (thinking phase), RRdenotes the critical assess-
ment of the initial analysis (reﬂection phase), and ROis
the integration of reasoning into a cohesive solution (out-
put phase). The structured prompt P(q)is constructed to
guide this decomposition:
P(q) = Φ(q,τ), (38)
whereΦis the prompt engineering function, and τis atemplate specifying the expected structure. This template
encodes phase-speciﬁc instructional priors that guide the
model to produce each component with distinct reasoning
objectives. Though generated in a single forward pass, each
component can be conceptually viewed as being inﬂuenced
by the preceding components, which we represent as con-
ditional distributions:
p(RT|q)≈p(RT|q,τT), (39)
p(RR|q,RT)≈p(RR|q,RT,τR), (40)
p(RO|q,RT,RR)≈p(RO|q,RT,RR,τO), (41)
whereτT,τR, andτOare the phase-speciﬁc instructional
priors embedded in the template. The probability of gener-
ating the full response can be expressed as:
p(R|q) =p(RT|q)·p(RR|q,RT)·p(RO|q,RT,RR)
This structured decomposition implements a form of
guided reasoning through explicit metacognitive phases.
The key insight is that while Mθremains ﬁxed, the struc-
tured prompt effectively guides the model’s reasoning pro-
cess by encouraging it to follow distinct cognitive phases
within a single generation. See Algorithm 3for details.
D.3.1. C OMPUTATIONAL TIMECOMPLEXITY
Chain-of-Thought with Reﬂection achieves enhanced rea-
soning with minimal computational overhead. Since the en-
tire process—including structured thinking, reﬂection, a nd
output—is generated in a single forward pass through the
language model, the dominant computational cost remains
that of standard inference. This results in a complexity
ofO(n), wherenis the length of the generated response.
However, if reﬂection introduces an iterative reﬁnement
mechanism (e.g., regenerating based on self-evaluation),
the complexity could increase depending on the number
of iterations. In such cases, the worst-case complexity be-
comesO(r·n), whereris the number of reﬁnement steps.
The trade-off is that additional reﬁnement may improve
output quality at the cost of higher computational demand.
Therefore, in its simplest form, the overall computational
complexity remains O(n), comparable to standard infer-
ence, while providing enhanced reasoning capabilities. In
iterative settings, complexity scales proportionally to t he
number of reﬁnement steps, requiring careful tuning to bal-
ance reasoning depth and efﬁciency.
D.4. Entropy-Guided Decoding
Entropy-Guided Decoding ( Das et al. ,2024 ;Simonds ,
2025 ;Zhang et al. ,2024b ) enhances language model out-
puts by dynamically adjusting sampling parameters based
on uncertainty metrics. Traditional approaches use ﬁxed pa -
rameters throughout generation, but our method adapts in
real-time to each token’s context. In our notation, we rep-
26

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Feature Self-Consistency Best-of-N Sampling
Selection Method Majority clustering + statistical analysis Explicit self-evaluation
Quality Assessment Implicit through similarity & frequency Direct scoring system (0-10)
Computational Overhead O(k×n+k2)(clustering is costly) O(k×n)(single pass rating)
Weak-to-Strong Distillation Yes (reinforces high-probability reasoning paths) Yes (ﬁlters weak outputs via scoring)
Error Handling Reduces stochastic noise via statistical convergence Mitigates low-quality outputs with explicit ﬁltering
Table 15. Comparison of Self-Consistency and Best-of-N Sampling
Algorithm 3 Chain-of-Thought(CoT) with Reﬂection
1:procedure CoT-Reﬂection( q,Mθ)
2:τ←ConstructTemplate ()⊲Create structured reasoning template with phase markers fo r thinking, reﬂection, and
output
3:P(q)←Φ(q,τ) ⊲Construct prompt with query qand template τ
4:R←Mθ(P(q)) ⊲Generate complete response in a single forward pass
5:RO←ExtractOutput (R) ⊲Extract ﬁnal output component RO
6:returnRO ⊲Return the ﬁnal output
7:end procedure
resent the sequence of tokens generated up to the current
generation step tasx= (x1,x2,...,x t), where each to-
ken belongs to a vocabulary of size V. At each generation
step, the language model produces logits lt∈RV, which
are the unnormalized prediction scores for the next token,
and attention weights At∈RL×H×S×S, whereLis the
number of transformer layers, His the number of attention
heads per layer, and Sis the sequence length. These at-
tention weights represent how much each token attends to
other tokens in the sequence, with Al,h,i,j
t indicating how
much token iattends to token jin headhof layerl. We
ﬁrst compute token probabilities from the logits using the
softmax function:
pt=softmax(lt) (42)
logpt= log softmax(lt) (43)
Here,pt∈RVrepresents the probability distribution over
all tokens in the vocabulary, with pt(v)indicating the prob-
ability of token v. (a) The Shannon entropy of this token
distribution quantiﬁes uncertainty in next-token selecti on,
which we normalize by ln(2) to express entropy in bits, pro-
viding a more interpretable scale:
H(pt) =−V/summationdisplay
v=1pt(v)log2pt(v) (44)
Entropy is a fundamental measure of uncertainty; higher en-
tropy values (approaching log2V) indicate that the model
is uncertain about which token to generate next, distribut-
ing probability more evenly across many tokens. Con-
versely, values near zero suggest the model is highly conﬁ-
dent, concentrating probability on one or few tokens. The
variance entropy (varentropy) is a complementary metricthat captures the spread of log-probabilities around the
mean entropy:
V(pt) =V/summationdisplay
v=1pt(v)(log2pt(v)+H(pt))2(45)
(b) Varentropy helps distinguish between distributions wi th
similar entropy but different shapes; higher varentropy in -
dicates a “peakier” distribution with a few high-probabili ty
tokens amidst many low-probability ones, which can sug-
gest that the model is considering multiple distinct possi-
bilities rather than being genuinely uncertain across the e n-
tire vocabulary. We derive attention-based uncertainty me t-
rics from the reﬁned attention patterns encoded in AL
t∈
RH×S×S, the ﬁnal layer’s attention weights. (c) The at-
tention entropy measures how uniformly attention is dis-
tributed across the sequence:
Hattn(AL
t) =−H/summationdisplay
h=1S/summationdisplay
i=1S/summationdisplay
j=1AL,h,i,j
tlog2AL,h,i,j
t (46)
High attention entropy indicates diffuse attention patter ns,
suggesting the model is uncertain about which parts of the
context are relevant for generating the next token. Low val-
ues suggest focused attention on speciﬁc context tokens, in -
dicating higher conﬁdence in the relevance of those tokens.
(d) The attention variance entropy quantiﬁes how consis-
tently different attention heads focus on the same parts of
the input:
Vattn(AL
t) =Varh∈[1,H](Hattn(AL,h
t)) (47)
Here,Hattn(AL,h
t)is the entropy of attention weights for
headh, and Var denotes variance. This metric captures dis-
27

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
agreement between attention heads, with higher values indi -
cating that different heads are focusing on different aspec ts
of the input, suggesting multi-faceted uncertainty. We als o
introduce two consistency metrics to capture attention pat -
terns more comprehensively. (e) The agreement metric αt
measures how consistently different attention heads focus
on the same tokens:
¯AL
t=1
HH/summationdisplay
h=1AL,h
t (48)
αt=Eh∈[1,H]/bracketleftBig
∝⌊a∇d⌊lAL,h
t−¯AL
t∝⌊a∇d⌊l1/bracketrightBig
(49)
where¯AL
tis the mean attention pattern across all heads, and
∝⌊a∇d⌊l·∝⌊a∇d⌊l1denotes the L1 norm (sum of absolute differences).
Lowerαtvalues indicate high agreement among attention
heads, suggesting model conﬁdence in its understanding of
the relevant context. Higher values suggest disagreement,
indicating uncertainty about which contextual elements ar e
most important. (f) The interaction strength γtquantiﬁes
the intensity of attention activations:
γt=Eh,i,j/bracketleftBig
|logAL,h,i,j
t|/bracketrightBig
(50)
whereEh,i,j[·]denotes the expectation (average) over all
heads, query positions, and key positions. Higher γtval-
ues indicate stronger, more deﬁned attention patterns, sug -
gesting the model has formed clearer associations between
tokens. These metrics collectively inform our adaptive pa-
rameter selection function Φ, which adjusts four key sam-
pling parameters based on observed uncertainty:
(τt,ptop
t,kt,pmin
t) = Φ/parenleftbig
H(pt),V(pt),Hattn(AL
t),
Vattn(AL
t),αt,γt/parenrightbig
(51)
(i) The temperature parameter τtcontrols the sharpness of
the probability distribution before sampling; higher temp er-
atures make the distribution more uniform (increasing ran-
domness), while lower temperatures make it more peaked
(increasing determinism). We adapt it based on token and
attention uncertainties:
τt=τ0·clip/parenleftBig
1+β1(H(pt)+V(pt))+β2Hattn(AL
t)
−β3αt,τmin,τmax/parenrightBig
(52)
(ii) The top-p (nucleus sampling) threshold ptop
trestricts
sampling to the smallest set of tokens whose cumulative
probability exceeds this threshold, effectively removing un-
likely tokens from consideration. We adapt it primarily
based on attention head disagreement:
ptop
t=ptop
0·clip/parenleftbig
1+β4Vattn(AL
t),ptop
min,1.0/parenrightbig
(53)
(iii) The top-k ﬁltering parameter ktrestricts sampling tothektmost probable tokens, providing a hard limit on the
token candidates. We adjust it based on attention consis-
tency and strength:
kt=clip(⌊k0·(1+β5γt−β6αt)⌉,1,kmax) (54)
(iv) The minimum probability threshold pmin
tﬁlters out to-
kens with probability below pmin
t·maxvpt(v)relative to
the most probable token, providing another way to elimi-
nate unlikely candidates. We adapt it based on token uncer-
tainty:
pmin
t=pmin
0·clip/parenleftbig
1−β7(H(pt)+V(pt)),pmin
min,pmin
max/parenrightbig
whereτ0,ptop
0,k0,pmin
0are the base parameter values used
when uncertainty metrics are neutral (default sampling be-
havior),β1...7are hyperparameters controlling the inﬂu-
ence of each uncertainty metric, clip (x,min,max)con-
strains value xto the range [min,max], and⌊x⌉represents
rounding to the nearest integer (for kt). The intuition be-
hind our parameter adjustments is rooted in uncertainty:
high token distribution or attention entropy (uncertainty )
prompts increased temperature for broader exploration. At -
tention head disagreement (high attention varentropy) lea ds
to a wider top-p sampling to include more candidates.
Strong attention patterns with moderate agreement (high
interaction strength) expand top-k selection for a more di-
verse set of top tokens. Elevated token uncertainty low-
ers the minimum probability threshold, preventing exclu-
sion of potentially valid but less probable tokens. This dy-
namic adaptation enhances generation quality across con-
texts without specialized tuning. In precision-demanding
contexts, uncertainty metrics naturally guide conservati ve
sampling; in creative settings, they enable greater explo-
ration. By linking sampling parameters to the model’s un-
certainty assessment, we achieve a principled balance be-
tween diversity and coherence, surpassing static parame-
ter approaches. Entropy-guided decoding thus reﬁnes lan-
guage model outputs by dynamically adjusting sampling
parameters based on real-time uncertainty. This method
calculates token and attention-based metrics during gener -
ation, adapting temperature, top-p, top-k, and minimum
probability threshold. This allows for exploration when
uncertain and precision when conﬁdent, all with minimal
inference overhead.
D.4.1. C OMPUTATIONAL TIMECOMPLEXITY
ANALYSIS
The computational complexity of entropy-guided decoding
per token generation step is determined by several key op-
erations. Calculating token distribution uncertainty met -
rics (entropy and varentropy) from the vocabulary logits
requiresO(V)operations, where Vis the vocabulary size.
The computation of attention-based uncertainty metrics,
28

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
which analyze the model’s attention patterns, contributes
O(L·H·S2)complexity. This arises from processing
the attention weights across Ltransformer layers, Hat-
tention heads, and sequence length S. Adapting the sam-
pling parameters based on these metrics involves simple
arithmetic and has a negligible O(1)time cost. The token
sampling process, including steps like top-k or top-p ﬁlter -
ing, addsO(VlogV)complexity due to sorting operations
required to ﬁlter the vocabulary distribution. Therefore, the
overall per-token computational complexity is dominated
by the sum of these factors, approximately O(VlogV+
L·H·S2). Consequently, for generating a text sequence
of lengthT, the total computational complexity becomes
O(T·(VlogV+L·H·S2)). For typical Large Language
Models and longer text sequences, the term O(L·H·S2)
associated with attention processing and uncertainty metr ic
calculations often represents the most signiﬁcant portion of
the computational cost per token.
D.5. Chain-of-Thought (CoT) Decoding
Chain-of-Thought (CoT) Decoding ( Wei et al. ,2022 ;
Wang & Zhou ,2024 ) is a multi-path inference technique
designed to enhance the reliability and logical coherence
of language model outputs. Unlike conventional decoding
methods that generate a single response, CoT Decoding ex-
plores a set of potential reasoning trajectories in paralle l.
This approach leverages a path management framework to
generate, evaluate, and select from a diverse set of candi-
date responses, ultimately aiming for outputs grounded in
more robust reasoning processes. The CoT Decoding pro-
cess begins with the initiation of multiple reasoning paths .
Given an input context c, the language model Mﬁrst com-
putes the probability distribution over the vocabulary Vfor
the ﬁrst token position. This distribution, P(x1|c), is de-
rived from the logits (pre-softmax scores) l1∈R|V|pro-
duced by the model for the ﬁrst token position. The proba-
bility distribution is typically obtained via a softmax fun c-
tion with a temperature parameter T:
P(x1|c) =softmax(l1/T) (55)
Here,x1∈ V represents a token from the vocabulary,
andP(x1|c)denotes the probability of x1being the ﬁrst
token in the response, conditioned on the input context
c. To initiate diverse reasoning paths, the system sam-
ples the top- ktokens with the highest probabilities from
P(x1|c). LetT={t1,t2,...,tk}be the set of these
top-ktokens. For each initial token ti∈ T , the model
generates a complete response sequence, resulting in a set
ofkcandidate pathsP={P1,P2,...,P k}. Each path
Pi= (xi,1,xi,2,...,x i,ni)represents a complete sequence
of tokens, where xi,1=tiandniis the length of path Pi.
A core component of CoT Decoding is the reliability scor-
ing mechanism. This mechanism evaluates the conﬁdencein token selections within each path. For each token xi,j
at positionjin pathPi, with corresponding logits li,j, a
token-level reliability score r(xi,j)is computed. Let p(1)
i,j
andp(2)
i,jbe the probabilities of the most and second most
likely tokens at position jin pathPi, respectively, obtained
after applying the softmax function to li,j. The token relia-
bility score is deﬁned as:
r(xi,j) = (p(1)
i,j−p(2)
i,j)·f(j) (56)
wheref(j)is a position-based damping function designed
to emphasize the reliability of earlier tokens in the se-
quence. A common form for f(j)is a linearly decreasing
function:
f(j) = 1−α·j
Li(57)
Here,Liis the maximum sequence length considered for
pathPi, andα∈[0,1]is a damping coefﬁcient that con-
trols the rate of decrease in reliability weight with positi on.
The overall reliability R(Pi)of a pathPiis calculated as
a weighted average of its token-level reliability scores. L et
wjbe position-dependent weights that further emphasize
earlier tokens. The path reliability is given by:
R(Pi) =/summationtextni
j=1r(xi,j)·wj/summationtextni
j=1wj(58)
In scenarios where multiple reasoning paths may lead to se-
mantically similar responses, CoT Decoding can incorpo-
rate a path consolidation mechanism. This process groups
paths that exhibit high textual similarity, typically mea-
sured using sequence comparison techniques. For each
group of similar paths, the path with the highest reliabilit y
score is selected as a representative of that group. Finally ,
the system selects the output response. In scenarios withou t
path consolidation, the path with the highest overall relia -
bility is chosen as the ﬁnal output:
P∗= argmax
Pi∈PR(Pi) (59)
When path consolidation is enabled, the selection is per-
formed among the representatives of the consolidated path
groups, again choosing the one with the highest reliabil-
ity. By exploring multiple reasoning paths and employing
a reliability-based selection process, Chain-of-Thought De-
coding aims to generate responses that are not only proba-
ble but also more logically consistent and reliably reasone d.
This method effectively addresses uncertainty by systemat -
ically exploring and evaluating different reasoning traje c-
tories, ensuring that the ﬁnal output is grounded in a well-
supported and coherent line of reasoning.
D.5.1. C OMPUTATIONAL TIMECOMPLEXITY
ANALYSIS
CoT Decoding’s complexity is primarily determined by k
(initial paths) and L(sequence length). Initial path ex-
29

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
pansion via a forward pass on input context c(lengthn)
to compute P(x1|c)contributesO(n·h), wherehis the
hidden dimension. Top- ktoken selectionT ⊂ V (vo-
cabulary size V) addsO(Vlogk). Sequence generation
forkpathsPi∈ P up to length LincursO(k·L·h),
considering O(h)per-token cost. Reliability scoring for
k·Ltokens adds O(k·L)overhead. Path consolida-
tion, involving pairwise comparisons of kpathsP, requires
O(k2·sim(L))≈O(k2·L). Thus, CoT Decoding’s overall
time complexity, dominated by generation and consolida-
tion, is approximately O(n·h+Vlogk+k·L·h+k2·L),
simplifying to O(k·L·h+k2·L)for largekandL. This
highlights the computational cost for enhanced reasoning
via multi-path exploration.
D.6. RE2(Re-Reading and Re-Analyzing)
The RE2framework is an advanced reasoning methodol-
ogy designed to enhance the performance of language mod-
els on complex tasks. Drawing inspiration from human
cognitive processes, this framework structures reasoning
into explicit phases, facilitating a more thorough analysi s
of input queries. Unlike traditional language model infer-
ence, where a model Mwith parameters θdirectly pro-
cesses an input query xto generate a response y, expressed
as:y=Mθ(x), the RE2framework introduces a struc-
tured approach. It reﬁnes the generation process by de-
composing reasoning into three distinct steps, transform-
ing the input query xinto a composite prompt structure,
PRE2. The response generation in RE2is then formulated
as:yRE2=Mθ(PRE2), wherePRE2is constructed by
concatenating several components:
PRE2=Psys⊕Pinit(x)⊕Preread(x)⊕Psynth
Here,Psysrepresents optional system instructions, and ⊕
denotes concatenation. The framework incorporates three
key reasoning phases, represented by Pinit(x),Preread(x),
andPsynth(x). The ﬁrst step, Pinit(x), prompts the model
to carefully comprehend the input query:
Pinit(x) =“Step 1 - Initial Reading: Let’s ﬁrst
read and understand the question carefully.”
⊕“Original Question: ” ⊕x
The next step, Preread(x), instructs the model to revisit the
query for structured decomposition and analysis:
Preread(x) =“Step 2 - Re-reading and Analysis:
Let’s read the question again: ⊕x
⊕“Now, let’s break down what the question
is asking and analyze its key components.”
Finally,Psynth guides the model to synthesize a response
based on insights from the previous steps:Psynth=“Step 3 - Final Answer: Based on our analysis,
here is the complete answer:”
The RE2framework incorporates parameters to regulate the
response generation process. The temperature parameter,
T, modiﬁes the output probability distribution, given by:
PT(y|PRE2) =exp( logit(y)/T)/summationtext
y′∈Vexp( logit(y′)/T)(60)
whereyrepresents output tokens, Vis the vocabulary
space, and logit (y)is the unnormalized score for token
y. To reﬁne token selection, nucleus sampling (top-p sam-
pling) is applied. It limits the vocabulary to a subset Vp(the
nucleus), deﬁned as:
Vp= min{V′⊆V|/summationdisplay
y∈V′PT(y|PRE2)≥p} (61)
such that the cumulative probability of selected tokens ex-
ceeds a predeﬁned threshold p. The ﬁnal sampling distribu-
tion is then computed as:
Pfinal(y|PRE2) =/braceleftBiggPT(y|PRE2)/summationtext
y′∈VpPT(y′|PRE2),ify∈Vp
0, otherwise
ensuring that tokens are sampled only from within the nu-
cleusVp, with their probabilities rescaled to sum to one,
thereby eliminating low-probability tokens. By integrati ng
temperature scaling and nucleus sampling, the RE2frame-
work balances determinism and diversity in text generation .
Its structured approach mirrors deliberate human analysis ,
fostering a more comprehensive exploration of the problem
before generating a response. This makes RE2particularly
advantageous for complex reasoning tasks.
D.6.1. C OMPUTATIONAL TIMECOMPLEXITY
ANALYSIS
The computational complexity of the RE2framework is pri-
marily dictated by the transformer’s self-attention mecha -
nism operating over the constructed prompt PRE2, which
has lengthm(linearly related to the original query length
n). This self-attention mechanism imposes a quadratic cost,
speciﬁcally O(m2·d), wheredrepresents the model’s
hidden dimension. Although the process of constructing
the prompt and the subsequent token sampling (which in-
cludes techniques like temperature scaling and nucleus
sampling) introduce some additional computational over-
head, these factors are relatively minor compared to the
dominant quadratic cost. Thus, while RE2maintains the
single forward pass characteristic of standard transforme r-
based inference, it does so at the expense of processing a
longer, more structured prompt, resulting in a higher con-
stant factor in runtime.
30

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Feature Entropy-Guided Decoding Chain-of-Thought Decoding
Approach Dynamically adjusts token sampling based
on uncertainty metrics from logits and at-
tention.Generates multiple reasoning paths from
diverse initial tokens, then scores and con-
solidates for best output.
Core Mechanism Adapts parameters (temperature, top-
p, top-k, min probability) using logits
entropy/varentropy and attention en-
tropy/varentropy, agreement, and interac-
tion strength.Scores reliability using top probability dif-
ferences and position damping to assess
path quality, optionally merges paths be-
fore selection.
Focus Adaptive sampling balancing exploration
and precision by reducing uncertainty.Multi-path exploration to enhance logical
coherence and output reliability.
Strength Dynamically modulates parameters based
on context conﬁdence, for ﬂexible applica-
tion.Synthesizes multiple paths to overcome er-
rors and produce robust and coherent out-
put.
Primary Goal Minimize generation uncertainty while bal-
ancing diversity and determinism.Maximize reasoning quality and consis-
tency by selecting the best path.
Table 16. Comparison of Entropy-Guided Decoding and Chain-of-Thoug ht Decoding
D.7. Mixture of Agents
The Mixture of Agents (MoA)( Wang et al. ,2024b ;
Chakraborty et al. ) framework enhances the quality of
language model responses through candidate generation,
critique, and synthesis. Let Mdenote a pre-trained lan-
guage model with trainable parameters θ. Given an input
queryqand system context s, the MoA process consists of
the following stages. In the initial stage, a set of ndiverse
candidate responses, denoted as Y=y1,y2,...,y n, is
generated. Each response yiis sampled from the condi-
tional probability distribution of the language model M,
parameterized by θ, given the query q, system context s,
and a generation temperature T1:
Y={y1,y2,...,y n},
whereyi∼pM(y|q,s;θ,T1),∀i∈{1,2,...,n}
whereYis the set of candidate responses, yiis thei-th can-
didate response, nis the number of generated responses
(a hyperparameter), pM(y|q,s;θ,T)represents the condi-
tional probability distribution of the language model, and
T1controls the stochasticity and diversity of responses,
with higher values promoting greater diversity. A critique
functionCevaluates the candidate responses Yin the con-
text of the original query qand system context s. For this,
we utilize the same language model Mto generate a cri-
tiquecbased on a conditional probability distribution with
temperature T2:
c=C(Y,q,s;θ)∼pM(c|Y,q,s;θ,T2) (62)
whereC(Y,q,s;θ)is the critique function evaluating Y,c
represents the generated critique, and T2is set lower than
T1to ensure a more discerning evaluation. The ﬁnal re-sponsey∗is synthesized using the critique c, queryq, and
system context s. A synthesis function S, also utilizing the
language model M, generatesy∗under a temperature T3:
y∗=S(c,q,s;θ)∼pM(y|c,q,s;θ,T3) (63)
whereS(c,q,s;θ)generates the reﬁned response, y∗is the
synthesized response, and T3is set lower than T2to en-
courage precise and focused reﬁnement. A post-processing
functionΦfurther reﬁnes the synthesized response to re-
move meta-content, artifacts, and formatting inconsisten -
cies. The ﬁnal output is denoted as yfinal :
yfinal= Φ(y∗) = Φ(S(C(yin
i=1,q,s;θ),q,s;θ))(64)
whereΦ(y∗)processes the synthesized response, and
yfinal is the ﬁnal enhanced response. The MoA framework
employs a temperature scheduling strategy to control the re -
ﬁnement process:
T1>T2>T3 (65)
This descending order encourages diversity in generation
(T1), balanced critique evaluation ( T2), and precise synthe-
sis (T3). Regularization techniques improve response qual-
ity by penalizing redundancy during generation:
pM(y|x;θ,T,λ)∝pM(y|x;θ,T)·R(y,λ) (66)
wherexrepresents either the query qor a combination of in-
puts depending on the stage, ∝denotes proportionality, and
R(y,λ)is a regularization function controlling repetition,
ensuring varied and high-quality responses. For practical
implementation, parameters that apply a penalty for token
repetition and prevent n-gram sequence repetition implic-
itly implement the regularization function R(y,λ)during
text generation by modifying the language model’s prob-
31

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
ability distribution to reduce repetitive token and n-gram
sequences, and effectively control the strength and type of
regularization applied In summary, the MoA framework it-
eratively reﬁnes responses by ﬁrst generating diverse cand i-
date responses, critically evaluating them, and synthesiz ing
an improved output. The structured use of temperature cas-
cade and regularization enhances response quality beyond
single-pass generation approaches.
D.7.1. C OMPUTATIONAL TIMECOMPLEXITY
ANALYSIS
The computational complexity of the Mixture of Agents
(MoA) framework is substantially higher than standard
single-pass generation due to its multi-stage process. The
dominant computational cost arises from the transformer
model’s self-attention mechanism, leading to a per-token
complexity that scales at least linearly, and potentially
quadratically, with the generated sequence lengths: L(av-
erage length of candidate responses), Lc(length of the cri-
tique), andL∗(length of the ﬁnal synthesized response).
The complexity is also directly proportional to the model’s
hidden dimension ( d). Generating ncandidate responses
increases this cost, making candidate generation the most
computationally intensive stage, with an approximate com-
plexity ofO(n·L2·d)orO(n·L·Smax·d), whereSmax
represents the maximum sequence length. The critique
and synthesis stages further contribute to the total compu-
tational demand, making MoA signiﬁcantly more resource-
intensive compared to single-pass inference. However, par -
allelization, such as distributed GPU inference, can miti-
gate latency in candidate generation while maintaining the
overall computational workload.
D.8. Reimplementation Then Optimize (RTO)
We introduce Reimplementation Then Optimize (RTO),
a novel multi-stage framework designed to enhance the
quality of solutions generated by large language models
(LLMs). By decomposing the generation process into dis-
crete stages—implementation, analysis, reimplementatio n,
and synthesis—RTO achieves signiﬁcant improvements in
correctness, consistency, and optimization compared to
single-pass generation methods. The framework leverages
iterative reﬁnement to progressively improve solution qua l-
ity through multiple generative passes. Let Mdenote the
language model and qrepresent the initial problem speciﬁ-
cation. The RTO process is formalized as follows:c1=M(s,q augmented) (67)
r=M(s,c1,qanalysis) (68)
c2=M(s,r) (69)
copt=/braceleftBigg
c1 ifδ(c1,c2)≥τ
M(s,c1,c2,q)otherwise(70)
In Stage 1 (Equation 67), the language model Mgenerates
an initial solution c1based on a system prompt s(which
provides instructions to guide the model’s behavior) and an
augmented query qaugmented (the initial query qaugmented
with instructions for generating high-quality output). St age
2 (Equation 68) involves the model Manalyzing the initial
solutionc1along with the system prompt sand an analysis
queryqanalysis (a prompt designed to extract requirements),
resulting in the extracted speciﬁcation r. In Stage 3 (Equa-
tion 69), the modelMproduces an independent solution
c2based on the extracted speciﬁcation rand the system
prompts. Finally, in Stage 4 (Equation 70), the framework
determines the optimized solution copt. This is achieved
by comparing the initial solution c1and the reimplemented
solutionc2using a similarity function δ(c1,c2)and a con-
sistency threshold τ. If the similarity exceeds the threshold,
coptis set toc1; otherwise,Msynthesizes a new optimized
solutioncoptfroms,c1,c2, andq. The effectiveness of RTO
is quantiﬁed by the quality improvement ∆Q, deﬁned as:
∆Q=Q(copt)−Q(c1) (71)
Equation 71measures the improvement in quality ∆Qas
the difference between the quality metric Qof the opti-
mized solution coptand the initial solution c1. Here,Qrep-
resents a domain-speciﬁc quality metric that encompasses
aspects such as correctness, efﬁciency, and other relevant
criteria.
D.8.1. C OMPUTATIONAL TIMECOMPLEXITY
ANALYSIS
The computational complexity of RTO is given by: TRTO=/summationtextn
i=1(M,li), whereT(M,li)denotes the time complex-
ity for the language model Mto generate a sequence of
lengthliin thei-th step. For Transformer-based LLMs,
the per-step complexity T(M,li)is dominated by the self-
attention mechanism and scales approximately as O(l2
i·d),
wheredrepresents the model dimension. Consequently, the
total complexity of RTO, TRTO, is the sum of these per-step
costs across its nstages.
D.9. PlanSearch
We present a novel multi-step planning and search
(PlanSearch ( Wang et al. ,2024a )) framework for general
language tasks that leverages LLMs to decompose com-
32

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
plex queries through iterative abstraction and reﬁnement.
Our approach formalizes the response generation as a struc-
tured sequence of transformations that progressively reﬁn e
the understanding of the query before producing a ﬁnal
response. Let us deﬁne a query as Q∈ Q , whereQ
represents the space of all possible queries, each encapsu-
lating the query, contextual requirements, and constraint s.
We aim to ﬁnd an optimal answer a∗∈ A , whereA
is the answer space. The process is decomposed into in-
termediate representations through multiple transformat ion
phases, mediated by a system prompt Ψthat provides high-
level guidance to the model. Given a question Qand sys-
tem prompt Ψ, we deﬁne the following transformation se-
quence:
O1=fobs(Q,Ψ,n1) (72)
O2=fderive(Q,Ψ,O1,n2) (73)
O=O1∪O2 (74)
σ=fstrategy(Q,Ψ,O) (75)
a=fanswer(Q,Ψ,σ) (76)
Here,O1={o1,o2,...,o n1}comprisesn1ini-
tial observations about the question Q, whileO2=
{on1+1,on1+2,...,o n1+n2}representsn2derived obser-
vations. The union of these sets is denoted as O. The
symbolσrepresents the reasoning strategy derived from
QandO, whileadenotes the ﬁnal answer derived from
Qandσ. The transformation functions fobs,fderive,fstrategy ,
andfanswer play distinct roles: fobsgenerates initial insights
by identifying key components of the question, such as
entities, relationships, and constraints; fderive synthesizes
deeper observations by connecting these components and
inferring implicit knowledge; fstrategy formulates a reason-
ing strategy to address the question systematically; and
fanswer produces a ﬁnal, well-structured answer based on
the reasoning strategy. Each transformation function fiis
realized through a pretrained language model Mwith pa-
rametersθand a task-speciﬁc prompt template τi:
fi(Q,Ψ,x1,x2,...,x n) =M(Ψ⊕τi(Q,x1,x2,...,x n);θ)
whereMrepresents the pretrained language model, θ
denotes its parameters, τiis a task-speciﬁc prompt tem-
plate, and⊕represents the concatenation operation. The
variablesx1,x2,...,x nrepresent function-speciﬁc inputs,
such as the question or previously generated observations.
To enhance answer diversity and quality, we generate multi-
ple candidate answers by introducing stochasticity throug h
temperature sampling:
A={a1,a2,...,a N}={fsolve(Q,Ψ;T)}N
i=1 (77)
Here,Trepresents the temperature parameter controlling
generation diversity, Ndenotes the number of answers gen-erated, andfsolveis the complete solution pipeline execut-
ing all transformation phases. This approach allows ex-
ploration of different reasoning paths and answer formu-
lations for a given question. The decomposition offers
several advantages: it activates relevant parametric know l-
edge by identifying key components and relationships in
the question, enables compositional reasoning through de-
rived observations, provides guided answer generation via
explicit reasoning strategies, and enhances explainabili ty
through a traceable reasoning chain from question to an-
swer. The multi-stage process mirrors human-like reason-
ing strategies, systematically breaking down complex ques -
tions before generating answers, resulting in responses th at
are both accurate and interpretable.
D.9.1. T IMECOMPLEXITY ANALYSIS
The time complexity of PlanSearch is determined by the se-
quential execution of its transformation functions throug h
a transformer-based language model Mwith parameters
θ. For transformer architectures, processing inputs requir es
O(L2
i)complexity due to self-attention, while generating
outputs adds O(Lo·Li)complexity, where LiandLorep-
resent input and output lengths respectively. For each tran s-
formation function, the time complexity can be expressed
as:
fobs:O/parenleftBig
(|Ψ|+|Q|)2·|θ|+
|O1|·(|Ψ|+|Q|)·|θ|/parenrightBig
fderive:O/parenleftBig
(|Ψ|+|Q|+|O1|)2·|θ|+
|O2|·(|Ψ|+|Q|+|O1|)·|θ|/parenrightBig
fstrategy:O/parenleftBig
(|Ψ|+|Q|+|O|)2·|θ|+
|σ|·(|Ψ|+|Q|+|O|)·|θ|/parenrightBig
fanswer:O/parenleftBig
(|Ψ|+|Q|+|σ|)2·|θ|+
|a|·(|Ψ|+|Q|+|σ|)·|θ|/parenrightBig
where|O|=|O1|+|O2|represents the total length of all
observations. The overall time complexity for generating
Nsolutions can be summarized as:
O
N·/summationdisplay
i∈{obs,derive,strategy,answer}/parenleftbig
L2
i+Li
o·Li/parenrightbig
·|θ|

whereLirepresents the input context length and Li
orepre-
sents the output length for each transformation function i.
33

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
As the context grows through the pipeline, complexity is
dominated by later stages with larger contexts. The frame-
work achieves efﬁciency through prompt engineering and
early termination of unpromising reasoning paths.
D.10. Monte Carlo Tree Search Algorithm
We utilize Monte Carlo Tree Search (MCTS)( Tang et al. ,
2024 ;Xie et al. ,2024 ;Gao et al. ,2024 ;Feng et al. ,2023 ;
Zhang et al. ,2024a ) for improved reasoning-driven re-
sponse generation in large language models (LLMs), es-
pecially for complex, multi-step language tasks where tra-
ditional methods often fall short. MCTS offers a frame-
work for language models to engage in structured think-
ing, logical inference, and multi-step problem-solving, e n-
abling capabilities such as hypothetical and counterfactu al
reasoning, commonsense and causal reasoning, and multi-
source, multi-hop question answering with RAG. By for-
mulating reasoning-driven response generation as a sequen -
tial decision-making problem, we demonstrate how MCTS
can systematically explore the vast space of potential re-
sponses to identify optimal outputs for a given end-user
query. This systematic exploration is particularly crucia l
when dealing with complex queries that require intricate
reasoning and planning over multiple steps. Our methodol-
ogy leverages the inherent uncertainty in language genera-
tion and provides a principled way to balance exploration
of diverse responses with exploitation of high-quality lan -
guage patterns. MCTS demonstrates signiﬁcant improve-
ments in response quality, coherence, and relevance com-
pared to traditional sampling and beam search methods,
which are often inadequate for navigating the complexities
of multi-step reasoning. We formulate reasoning-driven re -
sponse generation as a search problem within a state space
that evolves with the generation process. Let s∈S denote
a state in the generation process, where Srepresents the
set of all possible states the generation process can assume .
Each statesis formally deﬁned as:
s= (p,q,h) (78)
Here,p∈P is the system prompt, which serves to guide
and condition the language model’s behavior. Prepresents
the entire set of possible system prompts that can be used.
Next,q∈Q denotes the current user query, which is the
latest input to the language model. Qis the set encompass-
ing all possible queries a user might pose. Finally, h=
((r1,c1),(r2,c2),...,(rn,cn))∈H represents the gener-
ation history up to the current point. In this history, each
element(ri,ci)is a message, where ri∈{user,assistant}
speciﬁes the role of the message sender, and ci∈C is the
content of the message. His the collection of all possible
generation histories. The state space Sgrows exponentially
with the length of the generation sequence, rendering an ex-
haustive search for the best response computationally im-practical, especially in complex tasks where the sequence
of necessary steps can be long and branching. At each state
s, the action spaceA(s)is deﬁned as the set of all potential
responses that the language model can generate from that
state:
A(s) ={a1,a2,...,a k} (79)
Eachai∈ C in this set represents a possible response,
which is a content from the language model’s output space
C. Given a state s= (p,q,h)and an action a∈A(s), the
state transition function T:S×A→S determines the
next state based on the current state and the chosen action,
and is deﬁned as:
T(s,a) = (p,q,h⊕(assistant,a)) (80)
Here,asigniﬁes the action taken, which is the content of
the newly generated message by the assistant. The symbol
⊕represents the operation of concatenation, which in this
context appends the new assistant message to the existing
generation history. Monte Carlo Tree Search (MCTS) itera-
tively constructs a search tree to discover optimal respons es
through a sequence of four critical phases, enabling effec-
tive planning and decision-making even in complex sce-
narios: (a) The selection phase is the ﬁrst step, where
the algorithm navigates from the root of the search tree
down to a leaf node. This traversal uses the Upper Con-
ﬁdence Bound for Trees (UCT) method, which is essential
for balancing the exploration of less-visited branches of t he
tree against the exploitation of branches that have thus far
shown promise. This balance is vital for complex queries
where the optimal solution might not be immediately ob-
vious and requires exploration of diverse reasoning paths.
The UCT is deﬁned as follows:
UCT(s,a) =V(s,a)
N(s,a)+c·/radicalBigg
ln(Nparent(s))
N(s,a)(81)
whereV(s,a)represents the cumulative value associated
with taking action afrom states, accumulating the evalu-
ations from all simulations that passed through this state-
action pair.N(s,a)is the number of times the action ahas
been selected from state s, serving as a visit count for this
speciﬁc state-action pair. Nparent(s)is the total number of
visits to the parent node of state s, representing the overall
exploration effort from the preceding state. The term cis
the exploration weight, a constant that tunes the balance
between exploration and exploitation; a higher value en-
courages more exploration. At each node in the tree during
selection, the algorithm calculates the UCT value for each
possible action and chooses the action a∗that maximizes
this value, guiding the search towards potentially optimal
paths.
34

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
a∗= arg max
a∈A(s)UCT(s,a) (82)
(b) Once the selection phase reaches a leaf node sleaf, the
expansion phase begins. Here, the tree is expanded by gen-
eratingkcandidate responses from the language model.
These responses represent possible actions that can be
taken from the leaf state, effectively broadening the searc h
space. For complex tasks, generating diverse candidates
is crucial to uncover potentially effective, yet non-obvio us,
steps towards a solution, supporting hypothetical reasoni ng
by considering multiple potential continuations.
A(sleaf) ={a1,a2,...,a k}∼fLM(sleaf) (83)
In this step, fLMdenotes the language model generation
function, which takes the current state sleafas input and
produceskdiverse responses, each representing a poten-
tial next step in the response generation. Each candidate
responseaigenerated in this phase leads to the creation
of a new child node in the search tree, with an updated
states′
i=T(sleaf,ai)reﬂecting the addition of the new
response to the generation history. (c) Following expan-
sion, the simulation phase, also known as rollout, is initi-
ated from each of the newly created child nodes s′. In this
phase, the algorithm simulates future generation steps by
proceeding from the child node down to a certain depth or
until a terminal state is reached. This lookahead capabilit y
is particularly beneﬁcial for complex tasks, allowing the
algorithm to assess the longer-term consequences of early
decisions and perform multi-step problem-solving by ex-
ploring sequences of actions. This simulation is carried ou t
according to the following process:
s(0)=s′(84)
depth= 0 (85)
while depth <dand notτ(s(depth)) : (86)
A(depth)={a1,a2,...,a k}∼fLM(s(depth))(87)
a(depth)=Random(A(depth)) (88)
s(depth+1)=T(s(depth),a(depth)) (89)
depth=depth+1 (90)
Here,s(0)=s′sets the starting state for the simulation as
the newly created child node. The simulation continues it-
eratively as long as the current simulation depth is less tha n
a predeﬁned maximum depth d, and the current state s(depth)
is not a terminal state, as determined by the terminal state
functionτ(s)(discussed later). In each step of the simula-
tion, the language model generation function fLMis used to
generate a set of possible actions A(depth)from the current
states(depth). Then, an action a(depth)is selected randomly
fromA(depth)using the Random ()function, which chooses
uniformly at random from the available actions. The stateis then transitioned to the next state s(depth+1)using the
state transition function T, and the depth counter is incre-
mented. (d) After the simulation phase completes, reach-
ing either the maximum simulation depth dor a terminal
state, the backpropagation phase is executed. In this step,
the terminal state s(d)is evaluated using a quality function
Q:S→[0,1], which assigns a score reﬂecting the quality
of the simulated generation trajectory. This evaluation st ep
is critical for complex queries, as it allows the algorithm
to judge the overall coherence and quality of a multi-step
reasoning process, rather than just focusing on immediate
next-token probabilities. Furthermore, by evaluating dif -
ferent generation trajectories, MCTS implicitly performs
counterfactual reasoning, assessing the impact of differe nt
choices made during the generation process. This value is
then propagated back up through the search tree, from the
node where the rollout began all the way back to the root.
The update process is as follows:
Q(s) =feval
LM(s) (91)
N(s,a)←N(s,a)+1 (92)
V(s,a)←V(s,a)+Q(s(d)) (93)
Here,feval
LM(s)is the function that performs the evaluation of
a state, providing a quality score. For each state-action pa ir
(s,a)along the path from the rollout start node back to the
root, the visit count N(s,a)is incremented by one, and the
cumulative value V(s,a)is updated by adding the quality
scoreQ(s(d))obtained from the terminal state of the sim-
ulation. Quality evaluation is crucial for MCTS success,
and a primary method is using the LLM for self-evaluation.
The LLM assesses its own generated responses by being
prompted to rate their quality on a scale of 0 to 1. This lever-
ages the LLM’s inherent understanding of language, mak-
ing it effective for nuanced and complex queries, includ-
ing those requiring commonsense and causal reasoning to
judge coherence and relevance. This self-evaluation is rep -
resented byQ(s) =feval
LM(M(s)⊕meval), where the LLM
(fLM) evaluates a formatted state ( M(s)) combined with
an evaluation prompt ( meval) to produce a quality score. A
terminal state function ( τ) is used to manage MCTS com-
putational cost by identifying states for early simulation ter-
mination. This is crucial for complex tasks to ensure efﬁ-
cient exploration and prevent unbounded computation, es-
pecially in tasks like multi-hop question answering with po -
tentially lengthy reasoning chains. The terminal state fun c-
tion is deﬁned as:
τ(s= (p,q,h conv)) =/braceleftBigg
1if|hconv|>h max
0otherwise
where simulations terminate if the generation history leng th
(|hconv|) exceeds a predeﬁned maximum length ( hmax). In
summary, Monte Carlo Tree Search enhances reasoning-
35

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
driven response generation in large language models, par-
ticularly for complex, multi-step queries. MCTS excels
at structured thinking, logical inference, and multi-step
problem-solving, enabling capabilities like hypothetica l,
counterfactual, commonsense, and causal reasoning, as
well as multi-hop question answering in RAG settings.
By systematically exploring potential responses, MCTS
provides a more reasoned and higher-quality approach to
language generation, overcoming limitations of tradition al
methods through integrated forward planning and evalua-
tion. This multi-step planning and evaluation makes MCTS
especially effective for complex tasks demanding intricat e
reasoning and coherent multi-turn interactions, offering a
signiﬁcant advantage over simpler generation techniques.
D.11. R∗Algorithm
The R∗(Qi et al. ,2024 ) algorithm is a principled approach
to improving language model response generation through
Monte Carlo Tree Search (MCTS). When presented with
a user query, R∗systematically explores diverse reasoning
pathways to generate high-quality, well-reasoned respons es
by leveraging specialized reasoning strategies. This fram e-
work empowers language models to engage in structured
thinking, logical inference, and multi-step problem-solv ing,
enhancing capabilities such as counterfactual and causal
reasoning, and multi-step question answering within RAG
settings. We formulate response generation as a search pro-
cess through a tree of reasoning states. In this formulation ,
letQbe the set of all possible user queries (input ques-
tions),Sbe the set of intermediate reasoning states (natural
language reasoning steps), Abe the ﬁnite set of predeﬁned
reasoning actions{A1,A2,A3,A4,A5}(reasoning strate-
gies), andNbe the set of nodes in the MCTS tree, where
each noden∈N corresponds to a state s∈S. Given a
user queryq∈Q, R∗generates a response by performing
multiple rollouts through a dynamically constructed reaso n-
ing tree. The process begins with a selection phase where,
at each decision point, actions are selected using the Upper
Conﬁdence bound for Trees (UCT) to balance exploration
and exploitation:
a∗(n) = argmax
a∈A[UCT(n,a)]
UCT(n,a) =V(child(n,a))
N(child(n,a))/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Exploitation+c·/radicalBigg
lnN(n)
N(child(n,a))
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
Exploration
wherendenotes the current node in the MCTS tree being
considered for action selection. Here, argmax a∈A[f(a)]
denotes the action athat maximizes the function f(a). In
the R∗algorithm, an actiona∈A represents a predeﬁned
reasoning strategy from a ﬁnite set A. Each action guides
the LLM towards a speciﬁc problem-solving approach. Forexample, action A1directs the LLM to identify the imme-
diate next step, while A2prompts the development of a
comprehensive solution pathway. By strategically selecti ng
and applying these diverse actions during the search, R∗or-
chestrates the LLM’s reasoning, encouraging exploration
of various tactics to enhance the quality and effectiveness
of generated responses. The UCT balances exploitation,
represented byV(child(n,a))
N(child(n,a)), which favors actions that have
historically led to higher values, with exploration, repre -
sented byc·/radicalBig
lnN(n)
N(child(n,a)), which encourages the investi-
gation of less-visited actions, controlled by the explorat ion
parameterc≈1.4. When encountering a node with un-
explored actions or during initial rollout, the algorithm e x-
pands. For a chosen reasoning action a∈ A applicable
to the current state s, a prompt is generated to guide the
language model. The language model then generates the
subsequent reasoning state s′from this prompt, represent-
ing the next step in natural language reasoning, guided by
the selected strategy. The LLM functions as a natural lan-
guage reasoning engine, generating logically progressive
states guided by these actions. Following expansion, simu-
lations are performed from the newly expanded nodes to a
maximum depth d(typically 5). Speciﬁcally, after expand-
ing a node and creating a new child node representing the
subsequent reasoning state, the simulation process begins
from this child node. It is from this newly created node,
which we will now refer to as nfor clarity in the following
equations, that the simulation initiates:
v=Sim(n)
Sim(n)≈/braceleftBigg
Eval(n), if depth(n)≥d
Sim(RandChild (n)),otherwise
In simulation, the process starts from this newly expanded
child nodenand proceeds by repeatedly selecting random
actions (if no children exist, a random action is chosen for
expansion from n; if children exist, a random child of n
is chosen) until the maximum depth dis reached. At the
maximum depth, the evaluate function is called on the ﬁnal
node to estimate its value. This simulation estimates the
long-term value of different reasoning approaches without
fully exploring all possible paths. After simulation, the e s-
timated value vis propagated backward through the tree in
the backpropagation phase:
N(n)←N(n)+1
V(n)←V(n)+v
This backpropagation updates the visit counts and cumula-
tive values of the current node nand its parent nodes, en-
suring that promising reasoning paths receive more explo-
ration in subsequent MCTS iterations. For any reasoning
state (represented by a node), we evaluate the quality of the
36

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
potential response it contains:
Eval(n) =

Conf(s),if response in state s
contains valid answer information
0, otherwise
The Conf (s)function estimates the reliability of the an-
swer extracted from state s, assigning higher conﬁdence
to responses that align with expected answer patterns. A
critical component of R∗is the mutual consistency check,
Consistent (τ), which validates reasoning trajectories τ=
(n0,a0,n1,...,nk):
Consistent (τ) =/braceleftBigg
True,if Overlap (τ′
split:k,τsplit:k)>θ
False,otherwise
Here, we split a reasoning trajectory τinto a partial trajec-
toryτ0:splitand a remaining trajectory τsplit:k. We prompt
the LLM with the partial trajectory τ0:splitand ask it to com-
plete the reasoning, resulting in the predicted continuati on
τ′
split:k. The Overlap (A,B)function calculates the normal-
ized word overlap between texts AandB:
Overlap(A,B) =|Words(A)∩Words(B)|
|Words(A)∪Words(B)|
where Words (X)represents the set of normalized words in
textX, andθis a threshold for consistency (e.g., θ= 0.7).
The consistency check ensures that reasoning trajectories
maintain logical coherence. After performing MCTS and
extracting all possible reasoning trajectories, we select the
ﬁnal trajectory τ∗as the optimal trajectory based on a com-
bination of consistency and quality scores:
τ∗= argmax
τ∈T[ValidTraj(τ)·Score(τ)]
whereTis the set of all extracted trajectories, ValidTraj (τ)
ensures only consistent trajectories are considered, and t he
Score(τ) =V(nterminal)
N(nterminal)evaluates trajectory quality based on
the terminal node nterminal . The ﬁnal response r∗is then
derived from the optimal trajectory τ∗using SelectAns:
r∗=SelectAns ({answer from state s|s∈τ∗})
SelectAns ({a1,a2,...}) = argmax
ai[frequency (ai)·Conf(ai)]
This architecture enables R∗to address a wide range of
language tasks, from factual queries to complex reason-
ing and creative generation, by systematically exploring
and validating diverse reasoning pathways, thus enhanc-
ing the quality and reliability of language model responses .
The approach is particularly effective for tasks requiring
structured reasoning, clariﬁcation of ambiguities, and ex -
ploration of multiple solution approaches, making R∗a ver-
satile framework for improving response generation in var-
ious language-based applications.D.12. Test-Time Inference Techniques Evaluation
Our experiments (see Table 17) demonstrate that all
test-time scaling techniques yield improvements over the
PORAG+ATLAS baseline. Notably, methods leverag-
ing structured multi-path reasoning—such as Monte Carlo
Tree Search and the R∗Algorithm—achieve the most sub-
stantial gains, improving HotpotQA by up to 23.8% (EM)
and 14.5% (F1), and Gorilla accuracy by up to 7.8%. Tech-
niques like Self-Consistency, Best-of-N Sampling, and
Chain-of-Thought with Reﬂection also contribute consis-
tent and meaningful improvements across benchmarks.
These ﬁndings conﬁrm that dynamic, reasoning-driven in-
ference strategies signiﬁcantly boost the effectiveness o f
retrieval-augmented generation across diverse QA tasks.
E. Low-Latency LLM Decoding Strategies
Optimizing inference latency and throughput is critical fo r
RAG systems using LLMs in real-world applications. In-
ference latency refers to the time taken for a language
model to generate a response, while throughput measures
the number of tokens or requests processed per unit of
time. Lower latency is essential for real-time application s,
such as chatbots or virtual assistants, that may leverage
RAG systems. Higher throughput is desirable for efﬁ-
ciently handling multiple tasks or serving many users con-
currently, as in batch processing or cloud-based services,
which can also beneﬁt from RAG architectures. To ad-
dress latency challenges in RAG systems, various decod-
ing optimization techniques have been developed. Tra-
ditional methods like beam search and sampling strate-
gies offer some improvements, but recent algorithmic in-
novations have shown even greater promise for acceler-
ating inference without sacriﬁcing output quality. (a)
FlashAttention-2( Dao,2023 ) signiﬁcantly improves atten-
tion computation speed and latency by reengineering the
original FlashAttention algorithm( Dao et al. ,2022 ) to bet-
ter utilize GPU parallelism and reduce memory inefﬁcien-
cies, and is effective for low-latency inference and traini ng
in long-context Transformer models. Building on its prede-
cessor—which reduced memory I/O via tiling and online
softmax—FlashAttention-2 tackles remaining bottlenecks
in GPU resource utilization, crucial for scaling Transform -
ers to longer sequences. It introduces three key optimiza-
tions: (1) Reducing non-matrix multiplication FLOPs by
modifying online softmax to favor GPU-optimized mat-
mul operations and better exploit high-throughput compute
units. (2) Increasing thread block occupancy through ﬁne-
grained parallelism across the sequence length, in addi-
tion to batch and head dimensions, which beneﬁts long
sequences and small batch sizes. (3) Improving intra-
thread block work partitioning by assigning each warp
a slice of the query matrix instead of the key, mini-
37

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 17. Performance Comparison: PORAG+ATLAS Baseline Enhanced by Test-Time Scaling
Method HotpotQA (Joint EM / F1) Gorilla (Overall Acc.) PubMe dQA (Acc / F1)
PORAG+ATLAS (Baseline) 45.29 / 71.32 76.38 78.35 / 74.56
Self-Consistency 48.31 / 74.35 (+6.7%/+4.2%) 77.91 (+2.0% ) 80.80 / 77.59 (+3.1%/+4.1%)
Best-of-N Sampling 48.85 / 74.90 (+7.9%/+5.0%) 78.34 (+2.6 %) 81.24 / 78.11 (+3.7%/+4.8%)
Chain-of-Thought with Reﬂection 50.52 / 76.41 (+11.5%/+7. 1%) 79.20 (+3.7%) 82.13 / 79.03 (+4.8%/+6.0%)
Entropy-Guided Decoding 49.95 / 75.88 (+10.3%/+6.4%) 78.8 5 (+3.2%) 81.76 / 78.65 (+4.4%/+5.5%)
CoT Decoding 50.91 / 76.80 (+12.4%/+7.7%) 79.50 (+4.1%) 82. 45 / 79.38 (+5.2%/+6.5%)
RE251.87 / 77.75 (+14.5%/+9.0%) 80.01 (+4.8%) 83.05 / 80.01 (+6 .0%/+7.3%)
Mixture of Agents 52.55 / 78.47 (+16.0%/+10.0%) 80.41 (+5.3 %) 83.50 / 80.55 (+6.6%/+8.0%)
RTO (Reimpl. Then Optimize) 53.10 / 79.02 (+17.3%/+10.8%) 8 0.78 (+5.8%) 83.89 / 80.98 (+7.1%/+8.6%)
PlanSearch 53.88 / 79.75 (+18.9%/+11.8%) 81.22 (+6.3%) 84. 34 / 81.50 (+7.6%/+9.3%)
Monte Carlo Tree Search 54.95 / 80.83 (+21.3%/+13.3%) 81.85 (+7.2%) 85.01 / 82.31 (+8.5%/+10.4%)
R∗Algorithm 56.05 /81.68 (+23.8%/+14.5%) 82.36 (+7.8%) 85.55 /82.90 (+9.2%/+11.2%)
mizing shared memory communication. (b) Lookahead
Decoding( Fu et al. ,2024 ) is a parallel decoding algorithm
speciﬁcally designed to accelerate LLM inference by dra-
matically reducing sequential decoding steps. Unlike tra-
ditional autoregressive methods that generate tokens se-
quentially, Lookahead Decoding innovatively predicts mul -
tiple non-contiguous n-grams concurrently within a “looka -
head branch”, drawing inspiration from Jacobi iteration
techniques. A dedicated ”veriﬁcation branch” then metic-
ulously checks these potential tokens, acting as a quality
control mechanism to validate the n-grams as correct con-
tinuations that preserve the LLM’s intended output distri-
bution, ensuring accuracy and ﬁdelity to the base model’s
intended output. This method not only surpasses Spec-
ulative Decoding( Yan et al. ,2024 ;Leviathan et al. ,2023 ;
Chen et al. ,2023 ;Liu et al. ,2023 ) by eliminating the
need for auxiliary draft models—enhancing efﬁciency and
simplifying implementation—but also incorporates an n-
gram pool. This pool caches and reuses promising to-
ken sequences, further accelerating performance while
maintaining the high quality of generated text. For en-
hanced efﬁciency in our ATLAS-augmented RAG frame-
work, we integrate low-latency LLM decoding strate-
gies such as FlashAttention-2 and Lookahead Decoding.
FlashAttention-2 directly accelerates the attention comp u-
tations critical to ATLAS’s Multi-Layer Attention Gradi-
ent (MLAG) and Layerwise Representation Pooling (LRP)
mechanisms, as well as the subsequent token generation
within the LLM. Complementarily, Lookahead Decoding
reduces the sequential bottleneck of autoregressive gener -
ation by enabling parallel token prediction. This synergis -
tic combination promises to signiﬁcantly reduce the over-
all latency of our RAG system, resulting in faster dynamic
retrieval triggering, quicker query formulation, and acce l-
erated response generation, ultimately leading to a more
efﬁcient and responsive user experience for knowledge-
intensive tasks. We implement these existing techniques
to verify that these latency optimizations do not hinder theperformance of our proposed framework.
E.1. LLM Decoding Efﬁciency Evaluation
We evaluated the impact of low-latency decoding tech-
niques on the efﬁciency of our PORAG+ATLAS frame-
work (Qwen2.5-3B). As shown in Table 18, both
FlashAttention-2 and Lookahead Decoding offer substan-
tial improvements over the baseline (68.27s latency, 120 to -
kens/sec). FlashAttention-2, by accelerating attention c om-
putations crucial for ATLAS, reduced latency to 29.55s
(↓56.7% ) and increased throughput to 208 tokens/sec
(↑73.3% ). Lookahead Decoding achieved further gains
through parallel token prediction, decreasing latency to
23.15s ( ↓66.1% ) and boosting throughput to 255 to-
kens/sec ( ↑112.5% ). These results conﬁrm that incorpo-
rating optimized decoding methods signiﬁcantly enhances
the responsiveness of our RAG system by speeding up both
retrieval and generation phases, complementing the qualit y
enhancements provided by PORAG+ATLAS.
F. Related Work
F.1. Retrieval-Augmented Generation (RAG)
Advances in Retrieval-Augmented Generation (RAG) con-
tinue to extend the capabilities of Large Language Mod-
els (LLMs) in domain adaptation, efﬁciency, and long-
context reasoning. RAFT ( Zhang et al. ,2024c ) improves
factual accuracy by ﬁne-tuning models to ignore irrele-
vant retrievals and cite only the most pertinent sources.
CoRAG ( Wang et al. ,2025 ) enhances multi-hop reasoning
through iterative retrieval, reﬁning queries based on inte r-
mediate results rather than relying on a single retrieval
step. DRAGIN ( Su et al. ) introduces dynamic retrieval
by detecting real-time information needs using model un-
certainty and self-attention cues, enabling context-sens itive
query formulation during generation. RAPID ( Chen et al. ,
38

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
Table 18. Latency and Throughput Improvements with Low-Latency Deco ding Strategies
Method Avg. Latency (Sec/query) Throughput (tokens/Sec)
ATLAS+RAG (Baseline) 68.27 120
FlashAttention-2 29.55 ( ↓56.7% ) 208 ( ↑73.3% )
Lookahead Decoding 23.15 ( ↓66.1% ) 255 ( ↑112.5% )
2025a ) accelerates long-context inference by combining
RAG with speculative decoding, where a draft model pre-
dicts outputs for a larger model, balancing speed and ac-
curacy through self- or upward-speculation. MemoRAG
(Qian et al. ,2024 ) integrates external retrieval with a cog-
nitive memory system, recording episodic interactions and
distilling them into semantic memory to improve retrieval
relevance and consistency. Speculative RAG ( Wang et al. ,
2024c ) reduces latency and enhances comprehension by
generating draft responses using a small model and verify-
ing them with a larger model. CAG ( Chan et al. ,2024 ) ad-
dresses retrieval latency by preloading cached documents
into extended context windows, bypassing real-time re-
trieval altogether. Parametric RAG ( Su et al. ,2025 ) re-
places input-context retrieval with document parameteriz a-
tion, temporarily updating LLM weights during inference
to embed external knowledge directly, thereby streamlinin g
the retrieve-update-generate process.
F.2. Test-Time or Inference-Time Compute
Recent research has signiﬁcantly advanced the reasoning
capabilities of Large Language Models (LLMs) through
innovative test-time computation scaling strategies. S1
(Muennighoff et al. ,2025 ) introduces budget forcing, a
prompting strategy that delays early conclusions by insert -
ing “Wait” tokens, encouraging longer and more deliber-
ate reasoning. SETS ( Chen et al. ,2025b ) improves out-
put quality through a cycle of sampling, self-veriﬁcation,
and self-correction, iteratively reﬁning responses until cor-
rectness or a termination condition is met. Test-Time
Computing (TTC) ( Ji et al. ,2025 ) enables adaptive rea-
soning by combining a fast initial response with condi-
tionally triggered reﬁnement, emulating a shift from in-
tuitive to deliberative thinking. Knockout and League
(Chen et al. ,2024 ) propose decision-time algorithms that
reduce failure rates by comparing or averaging multiple
candidate solutions. Marco-o1 ( Zhao et al. ,2024 ) com-
bines Chain-of-Thought ﬁne-tuning with Monte Carlo Tree
Search (MCTS) to explore diverse reasoning paths for com-
plex problem-solving, while STILL-1 ( Jiang et al. ,2024 )
integrates a policy and reward model to guide reasoning
through a dynamically expanding tree. The Shortest Ma-
jority V ote ( Zeng et al. ,2025 ) leverages parallel CoT sam-
pling with CoT-length-aware aggregation to scale infer-
ence, and ARMAP ( Chen et al. ,2025c ) learns a rewardmodel directly from environment interactions to guide
LLM-based agents in evaluating action trajectories and im-
proving planning. ( Liu et al. ,2025 ) demonstrate that small
LLMs can outperform much larger ones by optimizing the
test-time scaling of policy models and reward-guided infer -
ence. ( Yoon et al. ,2025 ) extend this idea through Monte
Carlo Tree Diffusion, combining diffusion models with
MCTS to support iterative, tree-structured planning. Sim-
ilarly, ( Yu et al. ,2025 ) propose translating LLM outputs
into symbolic PDDL representations to enable classical
planning with A⋆, leveraging best-of-N sampling and ver-
balized reﬁnement. ( Geiping et al. ,2025 ) present a recur-
rent depth architecture that scales compute within hidden
states to deepen reasoning dynamically. ( Wu et al. ,2025 )
introduce AStar, an MCTS-powered structured reasoning
method for multimodal tasks, while ( Lin et al. ,2025 ) pro-
pose QLASS, a Q-value-guided stepwise inference frame-
work that enhances reasoning by modeling intermediate de-
cision quality via a reasoning tree. Together, these works
highlight a shift toward leveraging structured search, sym -
bolic abstraction, and latent computation for efﬁcient and
scalable reasoning.
F.3. KV Caching
Recent advancements in KV cache management have
signiﬁcantly enhanced the efﬁciency of Large Language
Model (LLM) inference. Efﬁcient inference requires
effective management of the Key-Value (KV) cache,
which stores intermediate computations during generation .
Adaptive and prompt-guided strategies include Ada-KV
(Feng et al. ,2024 ), which dynamically distributes compres-
sion budgets across attention heads based on their attentio n
patterns, improving memory usage while maintaining gen-
eration quality. FINCH ( Corallo & Papotti ,2024 ) proposes
a prompt-guided compression strategy that leverages pre-
trained self-attention weights to iteratively select the m ost
relevant KV pairs, enabling longer-context processing wit h-
out requiring ﬁne-tuning. For redundancy reduction, ThinK
(Xu et al. ,2024 ) introduces a query-dependent pruning
strategy that identiﬁes and removes less signiﬁcant chan-
nels within the key cache, minimizing memory consump-
tion without compromising model performance. SimLay-
erKV ( Zhang et al. ,2024d ) focuses on inter-layer redun-
dancies by detecting “lazy” layers—those contributing min -
imally to long-range dependencies—and selectively trim-
39

Scaling Test-Time Inference with Policy-Optimized, Dynam ic Retrieval-Augmented Generation via KV Caching and Decod ing
ming their KV caches. This approach streamlines mem-
ory usage by eliminating unnecessary data storage. Novel
mechanisms for long-context inference include DuoAtten-
tion ( Xiao et al. ,2024 ), which separates attention heads
into Retrieval Heads (accessing the full KV cache for
global context) and Streaming Heads (operating with a
constant-length cache focused on recent tokens). This se-
lective caching reduces memory and latency while preserv-
ing the model’s ability to handle long contexts. Similarly,
SnapKV ( Li et al. ,2025 ) exploits the observation that atten-
tion heads consistently focus on speciﬁc prompt features
by clustering and retaining only the most relevant KV po-
sitions. This strategy improves efﬁciency while maintain-
ing model performance. Recent works have proposed efﬁ-
cient strategies for compressing KV caches to support long-
context inference in large language models. One approach,
L2-Norm-Based Pruning ( Devoto et al. ,2024 ), leverages
the observed correlation between the L2norm of key em-
beddings and their attention scores, selectively retainin g
KV pairs with the lowest norms to reduce memory usage
without sacriﬁcing performance. Another line of work,
KVQuant ( Hooper et al. ,2025 ), applies advanced quantiza-
tion techniques—including per-channel and pre-RoPE key
quantization, non-uniform precision, and sparse-dense ve c-
tor representations—to compress KV caches to ultra-low
bitwidths. These methods enable scalable inference over
extended context lengths while maintaining model ﬁdelity.
KVLink ( Yang et al. ,2025 ) enhances LLMs by precomput-
ing key-value (KV) caches for individual documents, allow-
ing for efﬁcient reuse during inference and reducing redun-
dant computations. To ensure coherence when combining
these precomputed caches, KVLink adjusts positional em-
beddings to reﬂect their global positions, introduces trai n-
able special tokens to restore self-attention mechanisms
across documents, and employs mixed-data ﬁne-tuning to
maintain the model’s original capabilities. Together, the se
advancements collectively optimize memory usage, pro-
cessing speed, and inference efﬁciency in LLMs. They
highlight a growing emphasis on adaptive, redundancy-
aware, and context-sensitive strategies for KV cache man-
agement, paving the way for more efﬁcient and scalable
LLM inference.
40
# Information Retrieval in the Age of Generative AI: The RGB Model

**Authors**: Michele Garetto, Alessandro Cornacchia, Franco Galante, Emilio Leonardi, Alessandro Nordio, Alberto Tarable

**Published**: 2025-04-29 10:21:40

**PDF URL**: [http://arxiv.org/pdf/2504.20610v1](http://arxiv.org/pdf/2504.20610v1)

## Abstract
The advent of Large Language Models (LLMs) and generative AI is fundamentally
transforming information retrieval and processing on the Internet, bringing
both great potential and significant concerns regarding content authenticity
and reliability. This paper presents a novel quantitative approach to shed
light on the complex information dynamics arising from the growing use of
generative AI tools. Despite their significant impact on the digital ecosystem,
these dynamics remain largely uncharted and poorly understood. We propose a
stochastic model to characterize the generation, indexing, and dissemination of
information in response to new topics. This scenario particularly challenges
current LLMs, which often rely on real-time Retrieval-Augmented Generation
(RAG) techniques to overcome their static knowledge limitations. Our findings
suggest that the rapid pace of generative AI adoption, combined with increasing
user reliance, can outpace human verification, escalating the risk of
inaccurate information proliferation across digital resources. An in-depth
analysis of Stack Exchange data confirms that high-quality answers inevitably
require substantial time and human effort to emerge. This underscores the
considerable risks associated with generating persuasive text in response to
new questions and highlights the critical need for responsible development and
deployment of future generative AI tools.

## Full Text


<!-- PDF content starts -->

arXiv:2504.20610v1  [cs.IR]  29 Apr 2025Information Retrieval in the Age of Generative AI: The RGB Model
Michele P. Garetto
University of Turin
Torino, Italy
michele.garetto@unito.itAlessandro P. Cornacchia
KAUST
Thuwal, Saudi Arabia
alessandro.cornacchia@kaust.edu.saFranco Galante
Politecnico di Torino
Torino, Italy
franco.galante@polito.it
Emilio Leonardi
Politecnico di Torino
Torino, Italy
emilio.leonardi@polito.itAlessandro Nordio
Consiglio Nazionale delle Ricerche
Torino, Italy
alessandro.nordio@cnr.itAlberto P. Tarable
Consiglio Nazionale delle Ricerche
Torino, Italy
alberto.tarable@cnr.it
Abstract
The advent of Large Language Models (LLMs) and generative AI
is fundamentally transforming information retrieval and process-
ing on the Internet, bringing both great potential and significant
concerns regarding content authenticity and reliability. This paper
presents a novel quantitative approach to shed light on the complex
information dynamics arising from the growing use of generative
AI tools. Despite their significant impact on the digital ecosystem,
these dynamics remain largely uncharted and poorly understood.
We propose a stochastic model to characterize the generation, in-
dexing, and dissemination of information in response to new topics.
This scenario particularly challenges current LLMs, which often rely
on real-time Retrieval-Augmented Generation (RAG) techniques
to overcome their static knowledge limitations. Our findings sug-
gest that the rapid pace of generative AI adoption, combined with
increasing user reliance, can outpace human verification, escalat-
ing the risk of inaccurate information proliferation across digital
resources. An in-depth analysis of Stack Exchange data confirms
that high-quality answers inevitably require substantial time and
human effort to emerge. This underscores the considerable risks
associated with generating persuasive text in response to new ques-
tions and highlights the critical need for responsible development
and deployment of future generative AI tools.
CCS Concepts
•Information systems →Web searching and information
discovery ;Evaluation of retrieval results ;Novelty in informa-
tion retrieval ;•Computing methodologies →Natural language
generation .
Keywords
Web answering, Information quality, Large Language Models,
Retrieval-Augmented Generation, Automation bias, Stack Exchange.
ACM Reference Format:
Michele P. Garetto, Alessandro P. Cornacchia, Franco Galante, Emilio Leonardi,
Alessandro Nordio, and Alberto P. Tarable. 2025.Information Retrieval in
the Age of Generative AI: The RGB Model. In Proceedings of the 48th Inter-
national ACM SIGIR Conference on Research and Development in Information
Please use nonacm option or ACM Engage class to enable CC licensesThis work is
licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’25, July 13–18, 2025, Padua, Italy
©2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1592-1/2025/07
https://doi.org/10.1145/3726302.3730008Retrieval (SIGIR ’25), July 13–18, 2025, Padua, Italy. ACM, New York, NY,
USA, 11 pages. https://doi.org/10.1145/3726302.3730008
1 Introduction
In recent years, the emergence of Generative Artificial Intelligence
(GAI) fueled by Large Language Models (LLMs) has greatly en-
hanced our abilities in retrieving information and interacting with
digital content. The great success and widespread adoption of AI
chatbots such as ChatGPT, Copilot, Gemini, Claude, Perplexity,
Llama and many others are transforming both the way we search
for information and the way we produce new digital content.
The rapid growth in data availability and advances in computing
power have enabled LLMs to train on huge corpora, allowing them
to generate text that is not only contextually relevant but also
semantically rich. These capabilities have been used to enhance
search engines and virtual assistants, providing users with more
intuitive and ready-to-use answers to their queries.
While companies, researchers and individuals are striving to
harness the potential of LLMs, serious concerns have also arisen
about the potential for misuse and unintended consequences of
these tools, such as the spread of misinformation and fake news. In-
deed, AI chatbots can generate responses that are syntactically and
grammatically sound but factually incorrect. Furthermore, LLMs
can quickly produce coherent and persuasive text that tends to be
presented as the definitive answer to the users. This can increase
the risk that users trust the information provided by these models
without critically evaluating its accuracy or checking the original
sources, thus fostering a culture of intellectual laziness . Also, the
black-box nature of many LLM systems complicates efforts to en-
sure transparency and accountability in information dissemination.
Despite the vast amounts of information on which they are
trained, LLMs face limitations when dealing with new or rapidly
evolving topics. Since the training process relies on historical data,
there is often a lag before the latest information is incorporated
into the training datasets. This gap can make LLMs less reliable
in answering questions about current events or emerging topics.
To mitigate these limitations, some LLMs are designed to integrate
with traditional search engines or databases to access up-to-date
information. This approach, commonly referred to as Retrieval-
Augmented Generation (RAG), allows them to gather information
from the internet in real-time and provide answers to questions that
fall outside the scope of their training. However, this integration is
not always transparent to the user. When an LLM relies on external

SIGIR ’25, July 13–18, 2025, Padua, Italy Garetto et al.
search engines, it may not clearly indicate that the answer comes
from a real-time search and not from its internal knowledge.
Another insidious challenge in the use of LLMs is the risk of
"autophagy" in training cycles. This term, borrowed from biology,
refers to a process where AI-generated content becomes part of
the training data for future models, creating a self-consuming loop.
As LLMs produce vast quantities of text that are indistinguishable
from human-written content, this generated content often finds
its way back into the public domain. Over time, the inclusion of
AI-generated content in training datasets could lead to a dilution of
the originality and diversity of the generated responses, as models
recycle their own interpretations rather than drawing from a diverse
set of authentic, human-generated sources.
The Web, as a primary source of information for billions, is es-
pecially vulnerable to the careless exploitation of emerging GAI
technologies. As we continue to integrate GAI tools into informa-
tion retrieval and content creation, it is imperative to balance their
innovative potential with responsible oversight to safeguard the
integrity of information on the Internet. It is thus crucial to de-
velop proper methodologies and guidelines to navigate this rapidly
evolving landscape and better understand its future implications.
1.1 Paper Contributions
This paper proposes a novel analytical framework to describe and
understand the new dynamics of information retrieval and dis-
semination triggered by the integration of GAI tools into daily
workflows. To the best of our knowledge, our study presents the
first-of-this-kind quantitative approach to incorporate the several
factors at play into a comprehensive framework capable of predict-
ing the temporal trajectory of crucial key performance indicators.
Specifically, we focus on the generation, retrieval, and replication
of answers related to novel topics for which no prior information
is available — a scenario that poses the greatest challenge and risk
to any answer-generating system. In this context, we examine the
competition between conventional search engines and emerging
generative AI systems employing hybrid strategies to generate
answers, particularly real-time RAG. Our analysis incorporates both
algorithmic and human behavioral aspects. The primary objective
is to forecast potential future scenarios if current trends in the
adoption of generative AI tools persist.
Our analytical model is complemented by an in-depth examina-
tion of a large dataset from the Stack Exchange platform, containing
questions and answers in computer science and mathematics. This
investigation provides important insights into the intricate tem-
poral dynamics of answers to novel and challenging questions. In
particular, it highlights the considerable time often required for
high-quality answers to surface.
2 Related Work
Opportunities and risks of LLMs (more generally, foundation mod-
els) have been broadly discussed in [ 6,16]. In [ 30] authors point out
the potential for users to ‘anthropomorphize’ GAI tools, leading
to unrealistic expectations of human-level performance on tasks
where in fact these tools cannot match humans. The implications of
LLMs being increasingly incorporated into scientific workflows are
considered in [ 5]. Several studies have shown the risk for machine
learning models to acquire factually incorrect knowledge duringtraining and propagate incorrect information to generate content
[3,7,9,17,25]. A framework to evaluate LLMs’ ability to answer
complex questions is proposed in [ 34]. While [ 29,38,43] explore
the disinformation capabilities of LLMs, i.e., their ability to generate
convincing outputs that align with dangerous disinformation nar-
ratives. The hallucination problem in natural language generation
has been a focus of several empirical studies [ 18,28,36,42]. Chal-
lenges posed by time-sensitive questions are explored in [ 10,19,21].
Techniques to detect generic LLM-generated text are presented in
[27,35], while [ 11,23] focus on methods to prevent and detect
misinformation generated by LLMs. A self-evaluation technique to
enhance factuality is presented in [ 41], while [ 2] investigates the
use of human feedback in LLM alignment.
Retrieval-Augmented Generation was first proposed in [ 22] for
generic knowledge-intensive NLP tasks. Work [ 39] proposes an
adaptive mechanism to control access to external knowledge by a
conversational system. The use of search engine outputs to increase
LLM factuality is evaluated in [ 37]. The problem of explaining RAG-
enhanced LLM outputs has been addressed in studies such as [ 26]
and [ 33], highlighting the importance of interpretability in these
hybrid systems.
The risk of autophagy in model training has received significant
attention, also from a theoretical point of view. The problem was
first raised in the field of image processing [ 1,4,24], where feeding
synthetic data back into models has been shown to decrease both
the quality and diversity of generated images. Shumailov et al.
considered this phenomenon for LLMs [ 32] (earlier version of their
work in [ 31]), introducing the concept of model collapse . They show
that various models, when trained with data generated by previous
generations of the same model, forget the tails of the original data
distribution (early collapse) and tend to collapse into a distribution
that is very different from the original one, typically with much
lower variance (late collapse).
Briesh et al. [ 8] consider three possible data sources for each
generation of training data: the original dataset, fresh data, and data
from any previous generation. Consistently across different combi-
nations of these data sources, they observe a decrease in correctness
in the first generation, which then increases across generations.
However, this comes at the expense of diversity (especially for the
model trained only with synthetic data), which collapses to zero.
Dohmatob et al. [ 13] consider a (linear) regression task as a proxy
for more complex models and characterize the test error when mod-
els are subsequently retrained with synthetic data. They find that
performance decreases with the number of retraining generations.
Gerstgrasser et al. have extended this framework in [ 15], suggest-
ing that data accumulation prevents model collapse for language
models and deep generative models.
In [20] authors have examined the correctness, consistency, com-
prehensiveness, and conciseness of ChatGPT answers to 517 pro-
gramming questions on Stack Overflow, showing that 52% of Chat-
GPT answers contain incorrect information, overlooked by users
39% of the time. In [ 12] authors show that activity on Stack Over-
flow has decreased since the advent of chat-like LLMs, as users rely
more on generative AI tools, especially for coding questions.
Lastly, we mention the work of Yang et al. [ 40], who incorporate
human behavior in self-consumption loops, similarly to us. They
show that AI-generated information tends to prevail in information

Information Retrieval in the Age of Generative AI: The RGB Model SIGIR ’25, July 13–18, 2025, Padua, Italy
filtering, whereas real human data is often suppressed, leading to a
loss of information diversity.
3 System Model
We model for simplicity a system comprising one conventional
search engine and one LLM integrated with it, using RAG to pro-
vide up-to-date answers to user queries. This situation is common
to many GAI systems: for example, ChatGPT/Copilot are tightly
integrated with Bing, and the Gemini model is incorporated into
Google’s search infrastructure.
3.1 The RGB Model
3.1.1 Answer model for a given topic. Consider a specific novel
topic, for which no prior knowledge is available anywhere in the
system at time zero. Similarly to the RGB color model, we assume
that each potential answer to this topic is described by a length-3
vector, which represents a convex combination of the three primary
colors (red, green and blue). We emphasize that this is just for the
sake of an intuitive, simple illustration of our model. In general, we
can have an arbitrary set C=C′∪C′′of primary colors, associated
to different initially generated answers, partitioned into a set C′
of ‘good’ answers and a set C′′of ‘bad’ answers. Although the
model can be generalized to an arbitrary number of primary colors,
three is the minimum required to distinguish between ’bad’ (low-
quality, biased, fake) answers, conventionally represented by red,
and two distinct ’good’ answers (with potentially different qualities),
associated with blue and green. This allows us to quantify both the
accuracy and diversity of the produced answers, as discussed in
Section 3.3.1
The relative popularity (or strength) of answers of a given color
in the system (or one of its subsystems) at time 𝑡is determined
by the number of ‘coupons’ of the same color in the considered
subsystem.
We first consider the simpler case in which possible answers
available in the system can only be red (1,0,0), green (0,1,0) or blue
(0,0,1). In Subsection 3.4, we will present an extension of the model
in which mixed colors (mixing some percentages of red, green, and
blue) can be produced by either machine or human behavior.
3.1.2 System compartments. To specify how answers are initially
generated, and how they get replicated and reinforced across dif-
ferent digital resources over time, we introduce five subsystems
(compartments) denoted by letters 𝐺,𝑊,𝑇,𝑆,𝐿 (see Figure 1). At
any given time 𝑡≥0, subsystem 𝑠∈S={𝐺,𝑊,𝑇,𝑆,𝐿}holds
𝑁𝑠𝑐(𝑡)‘coupons’ of color 𝑐,𝑐∈C. We denote by 𝑁𝑠(𝑡)=Í
𝑐𝑁𝑠𝑐(𝑡)
the total number of coupons in subsystem 𝑠, at time𝑡. The five
considered compartments are:
External sources (G): this is a virtual place representing all
information sources, such as news organizations, economists, po-
litical analysts, researchers, etc., from which answers are initially
generated, i.e., introduced (through human intervention) into the
digital ecosystem.
Web (W): this compartment represents the set of all online re-
sources on the World Wide Web.
1In case we do not care about diversity of good answers, we could implement a simpler
binary system with just two colors.
GAI SystemExternal Sources
Training Set WWW
LLM Search EngineFigure 1: RGB model illustration: Colored circles represent
coupons of answers to novel questions. Dashed circles are
GAI-generated answers added to the Web. Gray arrows show
generation/reinforcement/replication processes.
Training Set (T): this is the ensemble of all curated datasets
used to train the LLM.
Search Engine (S): the search engine continuously crawls the
Web, indexing and ranking all found documents. It is then able to
algorithmically assess and quantify the relevance of each indexed
document in relation to a user’s query.
Large Language Model (L): the LLM constructs an internal
representation of the information embedded within the training
data, through a sophisticated architecture of interconnected param-
eters, or weights. Based on this representation, it produces a certain
answer in response to a user’s query.
Remark 1. For all compartments, the fraction of coupons of color
𝑐is a measure of the strength of answer 𝑐compared to other answers.
Theabsolute number of coupons of color 𝑐has a physical meaning
only for compartments W and T, where it represents the number of
distinct replicas of answer 𝑐found on the Web or in the Training Set,
respectively. For the Search Engine and the LLM, only the relative
number of coupons of color 𝑐matters, conveying a measure (weight)
of how much that answer is considered relevant in the subsystem.
3.1.3 Coupon generation and propagation. In our model, coupons
accumulate in each subsystem due to generation, reinforcement,
or replication processes (or flows), represented by arrows in Fig.
1. One exception is the external sources compartment, which is
supposed to contain a fixed number 𝑁𝐺𝑐of coupons of color 𝑐,
so that𝑔𝑐=𝑁𝐺𝑐/𝑁𝐺is the probability of selecting uniformly
at random a coupon of color 𝑐. This allows us to account for an
arbitrary distribution of the color of fresh answers that are initially
introduced into the digital ecosystem.
We identify six main flows, indexed by 𝑓∈{P,H,I,T,S,A}.
Each flow is modeled as a point process, characterized by either
an aggregate rate Λ𝑓or a per-coupon rate 𝜆𝑓, depending on the
physical meaning of the flow. We will first introduce unbiased flow
ratesΛ𝑓(or𝜆𝑓). The effective rate at which coupons are generated,

SIGIR ’25, July 13–18, 2025, Padua, Italy Garetto et al.
reinforced, or replicated will be elucidated after the explication of
our quality bias assumptions (Section 3.1.4).
P: this flow accounts for users posting fresh new answers on
the Web, which is one of the ways answers are initially added
to the digital information ecosystem. It is characterized by a
(generally, time-varying) aggregate rate ΛP(𝑡).
H: this is the overall process by which curated answers are in-
corporated into the Training Set. This flow, characterized by a
relatively low aggregate rate ΛH(𝑡), is divided into two sub-
flows with rates Λ′
H(𝑡)=𝛾ΛH(𝑡)andΛ′′
H(𝑡)=(1−𝛾)ΛH(𝑡),
where 0≤𝛾≤1is a model parameter. The two rates cor-
respond respectively to fresh answers generated and directly
incorporated into the Training Set (by-passing the Web), and
to answers coming from the Search Engine, reflecting the fact
that dataset curators often use the results of traditional search
engines as sources of information.
I: this process models how each answer posted on the Web is
independently crawled and indexed by the Search Engine with
per-coupon rate 𝜆I(𝑡). Note that the aggregate rate at which
documents containing answers pertaining to the considered
topic are indexed is ΛI(𝑡)=𝑁𝑊(𝑡)𝜆I(𝑡).
T: this process models how each answer contained in the Training
Set is fed into the LLM, with per-coupon rate 𝜆T(𝑡), resulting
into an aggregate training rate ΛT(𝑡)=𝑁𝑇(𝑡)𝜆T(𝑡)for an-
swers related to the considered topic.
S: This process represents users submitting queries related to
the reference topic to the conventional search engine, with
aggregate rate ΛS(𝑡). Some of the answers obtained from the
Search Engine are incorporated into documents posted again
on the Web, creating a first recursive loop in the model. We
denote the rate of answers fed back into the Web by Λ′
S(𝑡).
A: This process represents users submitting queries related to the
reference topic to the GAI system combining the LLM with the
Search Engine, with aggregate rate ΛA(𝑡). Some of the answers
obtained from the GAI system are fed back into the Web, at
rateΛ′
A(𝑡)=𝛼ΛA(𝑡),0≤𝛼≤1, producing a second loop.
Since many LLMs exploit the feedback provided by users (for
example in the form of likes/dislikes) to further tune themselves,
it results into another feedback loop of rate ΛF(𝑡)=𝛽ΛA(𝑡),
which is relevant even if 𝛽is small, because ΛA(𝑡)is large and
rapidly growing.
The hybrid strategy (RAG) employed by the GAI system to re-
spond to queries is modeled by combining, for each color, the
coupons contained in the LLM with those contained in the Search
Engine:𝑁𝐴𝑐=𝑁𝐿𝑐+𝑁𝑆𝑐. This way, the GAI system is able to respond
to queries even if the LLM has not yet been trained with informa-
tion related to the novel topic, provided that at least one answer
is returned by the Search Engine. However, the GAI system might
prefer information derived from the LLM knowledge base over that
obtained from the Search Engine, when both are available. Since a
strict priority rule could be potentially dangerous in this context,
we consider the following soft preference mechanism: when an
answer is incorporated into the LLM during the training process,
we suppose that 𝑤≥1coupons of the associated color are added to
the LLM, rather than a single one. This approach permits modelinga generalized bias toward information stored in the LLM, without
entirely disregarding external sources.
3.1.4 Bias to quality. We introduce, for each flow, a bias parameter
towards the actual addition of coupons within the target subsystem
of the flow. Such biases are introduced to take into account the
effectiveness of algorithms/humans in promoting high-quality an-
swers. We assume that each answer associated with a primary color
component (three in the case of the RGB model) is characterized by
an intrinsic quality 𝑞𝑐, normalized in such a way thatÍ
𝑐𝑞𝑐=1. In-
formally, we define the quality of each answer as the relative merit
that would be attributed to it by a large number of independent
experts having unlimited time to carefully evaluate and compare
the answers. Intrinsic qualities are unknown to all actors in the
system, both humans and machines. This fact is especially crucial
for novel topics, for which none or insufficient effort has been made
to evaluate proposed answers. Also, different actors have different
willingness/ability to identify and promote quality among answers.
To model this, we assume that each flow 𝑓∈{P,H,I,T,S,A},
is characterized by a bias parameter 𝐶𝑓towards quality. Each time
the flow is supposed to add a coupon of color 𝑐in the target sub-
system𝑠, we assume that this coupon addition occurs only with
probability:
𝑟𝑓
𝑐=𝑞𝑐+𝐶𝑓Í
𝑖∈C(𝑞𝑖+𝐶𝑓)(1)
Note that𝐶𝑓=0corresponds to the case in which coupon addition
occurs proportionally to intrinsic quality. As 𝐶𝑓→∞ , coupon
addition becomes oblivious to quality, all answers being treated the
same. One can also consider negative values of 𝐶𝑓(provided that
min𝑐(𝑞𝑐+𝐶𝑓)≥0), representing the intention to further penalize
low-quality answers in favor of high-quality answers.
To summarize, at the times dictated by the point process, with
aggregate rate Λ𝑓(𝑡), associated with flow 𝑓, going from subsystem
𝑠′to subsystem 𝑠, a coupon is chosen uniformly at random among
all coupons currently stored in 𝑠′. Let𝑐be the color of the chosen
coupon. A coupon of color 𝑐is then added to the coupon stored in
𝑠with probability 𝑟𝑓
𝑐.
3.1.5 Finite coupon time-to-live. In our model coupons do not stay
forever in the subsystem in which they appear. With the exception
of external sources, all compartments subject coupons to a finite
duration of existence. The lifespan of each coupon in subsystem 𝑠
follows an exponential distribution with a rate parameter 𝜇𝑠>0,
representing the inverse of the expected time-to-live. This stochas-
tic removal process serves dual purposes: For the Web and Training
Set compartments, it simulates the natural obsolescence of infor-
mation and subsequent removal by document maintainers. In the
Search Engine and LLM compartments, this mechanism reflects the
necessity for periodic refreshment and validation for an answer to
be considered still relevant.
3.2 Model Solution
Readers familiar with Markov Chains and queuing network will
recognize that our system can be described as a multi-class, open
system of four interconnected queues of type ·/𝑀/∞associated
to compartments 𝑇,𝑊,𝐿,𝑆 , fed by state-dependent external arrival
processes of ‘customers’. By construction, the giant Markov Chain
representing the overall system state {𝑁𝑠𝑐}𝑐,𝑠is ergodic, for any

Information Retrieval in the Age of Generative AI: The RGB Model SIGIR ’25, July 13–18, 2025, Padua, Italy
positive values of arrival rates Λ𝑓(𝑡)(𝑓∈{P,H}) and𝜆𝑓′(𝑡)(𝑓′∈
{I,T}). Unfortunately, due to the complex replication process of
coupons in the system, our system does not allow a product-form
solution for the stationary distribution of {𝑁𝑠𝑐}𝑐,𝑠. To solve the
model, one can resort to discrete-event simulation. We implemented
an ad-hoc simulator consisting of a simple C file [14].
We can also take a mean-field approach, and approximate the
evolution of the mean number of coupon 𝑛𝑠𝑐(𝑡)=E[𝑁𝑠𝑐(𝑡)]of
each color in each compartment. Readers familiar with epidemic
models such as the SIR model will recognize that our system is
similar to compartmental models used in epidemiology. Following
this approach, the system dynamics can be described by a set of
ODEs (Ordinary Differential Equations). We define for convenience
𝑛𝑠(𝑡)=Í
𝑐𝑛𝑠𝑐(𝑡)and𝑞𝑠𝑐(𝑡)=𝑛𝑠
𝑐(𝑡)
𝑛𝑠(𝑡). We obtain the following system
of coupled ODEs, which can be efficiently solved numerically.
¤𝑛𝑇
𝑐(𝑡)=ΛH𝑟H
𝑐[𝛾𝑔𝑐+(1−𝛾)𝑞𝑆
𝑐(𝑡)]−𝜇𝑇𝑛𝑇
𝑐(𝑡)
¤𝑛𝑊
𝑐(𝑡)=ΛP𝑟P
𝑐𝑔𝑐−𝜇𝑊𝑛𝑊
𝑐(𝑡)+𝑞𝑆
𝑐(𝑡)Λ′
S𝑟S
𝑐+𝑞𝐴
𝑐(𝑡)Λ′
A𝑟A
𝑐
¤𝑛𝐿
𝑖(𝑡)=𝑛𝑇
𝑐(𝑡)𝑤𝜆T(𝑡)−𝜇𝐿𝑛𝐿
𝑐(𝑡)+𝑞𝐴
𝑐(𝑡)ΛF(𝑡)𝑟F
𝑐
¤𝑛𝑆
𝑐(𝑡)=𝑛𝑊
𝑐(𝑡)𝜆I(𝑡)𝑟I
𝑐−𝜇𝑆𝑛S
𝑐(𝑡)(2)
One technical issue arising with this approach is that since 𝑛𝑆(0)=
0(there are initially no coupon in the Search Engine), we have inde-
terminate 0/0forms for𝑞𝑆𝑐(0)and𝑞𝐴𝑐(0). We solved this problem
assuming that the Search Engine contains, initially, a given positive
number𝑛S
𝑏(0)of ‘black’ coupons which, if chosen, do not produce
any effect. Black coupons are never replenished, thus they gradually
disappear from the system at rate 𝜇𝑆:¤𝑛𝑆
𝑏(𝑡)=−𝜇𝑆𝑛S
𝑏(𝑡). Besides
solving the above problem, black coupons serve another purpose,
i.e., they can model an answering machine (either the Search Engine
or the GAI) preferring not to provide any answer (which happens
when a black coupon is selected) in the initial phase in which none
or very few colored coupons have been acquired: by setting 𝑛S
𝑏(0),
one can can thus model more or less ‘prudent’ answering strategies
in the case of novel topics. A clear trade-off arises here between
limiting the dissemination of insufficiently consolidated answers
(thereby mitigating the potential spread of misinformation), and
maintaining user engagement and satisfaction.
Note that the above ODEs yield deterministic trajectories rep-
resenting the ‘average’ system evolution. They cannot be used
to assess the variability of performance metrics, especially in the
crucial initial transient phase. To obtain a more comprehensive
understanding of system dynamics, and to characterize the distribu-
tion of possible trajectories, we must resort to simulating multiple
independent runs of the system.
3.3 Metrics
Several interesting metrics can be computed by solving our model
over time. The fraction 𝜋𝑠(𝑡)of relevant answers contained in
subsystem𝑠, at time𝑡, is:
𝜋𝑠(𝑡)=Í
𝑐∈C′𝑁𝑠𝑐(𝑡)
𝑁𝑠(𝑡)
Note that𝜋𝑆(𝑡)and𝜋𝐴(𝑡)provide the accuracy of individual pieces
of information generated, respectively, by the Search Engine and
the GAI system, at time 𝑡, pertaining to the considered topic.Restricting to the set C′of relevant answers, one might also
be interested in assessing the diversity degree of answers stored
in subsystem 𝑠. Ideally, it would be desirable to achieve, for each
relevant answer 𝑐, a fraction𝑝𝑐proportional to its intrinsic quality
𝑞𝑐(renormalized among relevant answers):
𝑝𝑐=𝑞𝑐Í
𝑐′∈C′𝑞𝑐′, 𝑐∈C′
If we instead obtain, in subsystem 𝑠at time𝑡, a fraction
ˆ𝑝𝑠
𝑐=𝑁𝑠𝑐Í
𝑐′∈C′𝑁𝑠
𝑐′, 𝑐∈C′
of relevant answers 𝑐, among all relevant answers contained in 𝑠,
we could quantify the diversity degree of𝑠by some distance metric
between the discrete distributions {𝑝𝑐}𝑐and{ˆ𝑝𝑠𝑐}𝑐. For example,
adopting the total variation distance, we can simply define the
diversity degree𝜌𝑠(𝑡)∈[ 0,1](the higher the better):
𝜌𝑠(𝑡)=1−1
2∑︁
𝑐′∈C′|𝑝𝑐′−ˆ𝑝𝑠
𝑐′(𝑡)| (3)
Besides the accuracy and diversity of the subsystems, we are
especially interested in the following key performance indicators:
FIUA : Fraction of Irrelevant Used Answers. We call ‘used’ the
answers added to the set of online resources on the Web, after
having been suggested either by the Search Engine or by the GAI
system. Note that we are not considering here answers found on the
Web which have been initially generated (by flow P), but only those
produced by flows S′andA′. FIUA is the fraction of irrelevant
used answers found on the Web, with respect to all used answers
found on the Web (at any given time 𝑡).
AIRI : AI Responsability Index. Restricting to the set of irrelevant
used answers found on the Web (at any given time 𝑡), AIRI provides
the fraction of them generated by the GAI system (i.e., over flow A′),
distinguishing them from those coming from the Search Engine
(i.e., over flowS′).
FRQ : Fraction of Responded Queries. Recall that our model in-
corporates a mechanism to simulate cautious answering strategies
by initializing the Search Engine with a tunable number of black
coupons: when a black answer is selected it is assumed that no
usable response is returned to the user. Consequently, we can cal-
culate, up to any given time 𝑡, the Fraction of Responded Queries
(FRQ). Note that we can independently enable this mechanism for
the Search Engine and/or the GAI system.
AIAI : AI Autophagy Index. Our system contains several cycles
over which information about the novel topic can loop and reinforce
itself. We are especially interested in quantifying the autophagy of
the LLM, i.e., the amount of information (about the topic) stored
in it, which has been generated by the GAI system in response to
users’ queries. At any time 𝑡, AIAI is the fraction of coupons found
in the LLM, which have previously traversed flow A′.
3.4 Model Extensions
The RGB model described so far can be extended to account for
the generation of coupons of intermediate colors, blending primary
components in varying proportions. This process mirrors a com-
mon phenomenon observed in both human cognition and machine
learning, wherein new digital content – subsequently added to on-
line resources on the Web – is obtained by synthesizing information

SIGIR ’25, July 13–18, 2025, Padua, Italy Garetto et al.
derived from multiple sources. This aspect of information synthesis
and propagation potentially represents a critical element that war-
rants incorporation into the model. We do so in two different ways,
distinguishing human and algorithmic effects. In both cases we
assume that just two sources are used to synthesize a new content
posted on the Web, but this assumption could be relaxed as well.
For the GAI system, we assume that each generated response is
derived from a linear combination of the two sources. Specifically, a
random fraction 𝑢of the first source is combined with (1−𝑢)of the
second, for each component, where 𝑢follows a uniform distribution
over the interval[0,1]. This reflects the fact that the GAI system is
not really able to prefer one source over the other (but note that
sources to be mixed are first chosen proportionally to their relative
strength/popularity). A unique randomly generated text is then
returned to the user, who will use it ‘as it is’.
We posit a different model for humans creating new content from
the results returned by the Search Engine. Specifically, we propose
that users combine a fraction 𝜉of the perceived ‘best’ source2
with a complementary fraction (1−𝜉)of the alternative source.
The parameter 𝜉is constrained to the interval [1/2,1], reflecting
users’ tendency to favor the perceived trusted source. This model
encapsulates the cognitive process by which users evaluate and
discriminate between the relative quality of multiple alternatives.
This process demands significant human effort, which is driving
more and more people to delegate the task to artificial systems.
4 Scenarios
Our model has a rich set of parameters, reflecting the utility of a
versatile analytical framework for exploring the dynamics of a vari-
ety of possible scenarios. As a side effect, an exhaustive exploration
of individual parameter impacts exceeds the scope of the present
study. We therefore focus on select scenarios, reserving a more
extensive sensitivity analysis for a forthcoming journal publication.
To demonstrate the insights that our model can provide, we focus
on a fixed set of parameters, and adopt an evolutionary perspective,
by which just a few parameters are shifted to reflect current trends
in technological adoption. Specifically, we propose the following
three scenarios:
pre-GAI : This scenario represents the conventional paradigm of
information retrieval on the World Wide Web, before the availability
of GAI services.
GAI: This scenario aims to capture present conditions, charac-
terized by rapid yet judicious adoption of GAI tools. It reflects a
transitional period where users maintain vigilant oversight of GAI
outputs while still predominantly relying on results generated by
traditional search engines.
post-GAI : This prospective scenario envisions a future state
where information retrieval will be largely based on GAI services.
It postulates a significant shift in user behavior, marked by in-
creased confidence in and reliance on GAI-generated outputs, and
a concomitant reduction of personal synthesis of information by
employing search engines.
We consider an RGB model where primary components have
intrinsic qualities 𝑞blue=0.5,𝑞green =0.4,𝑞red=0.1. The set
2The quality of a generic answer 𝑖with components{𝑥𝑖
𝑐}𝑐is defined as the weighted
average of the intrinsic qualities of its components: 𝑞𝑖=Í
𝑐𝑥𝑖
𝑐𝑞𝑐.of ‘good’ answers is C′={blue,green}, while the red answer is
bad,C′′={red}. All primary components are equally likely to be
initially generated, so 𝑔𝑐=1/3,∀𝑐.
Table 1 reports the flow parameters chosen for the GAI scenario.
The time unit of our system is the day, and we assume that all
parameters remain constant over time3. So, for example, the first
row in Table 1, which specifies the posting process Pof new an-
swers on the Web, indicates that, starting at time 0, answers are
posted with rate of 1 per day, with a quality bias parameter 𝐶P=1,
and they persist online for a duration that follows an exponential
distribution with a mean of 100 days.
flow rate bias TTL related
P ΛP=1𝐶P=1 1/𝜇𝑊=100
H ΛH=0.1𝐶H=−0.08 1/𝜇𝑇=1000𝛾=0.5
I𝜆I=0.1𝐶I=0𝜇𝑆=𝜆I
T𝜆T=0.1 –𝜇𝐿=𝜆T𝑤=3
S′ΛS′=100𝐶S′=0.1 –𝑁𝑆
𝑏(0)=10
A′ΛA′=10𝐶A′=0.1 –𝛼=0.4
F ΛF=1𝐶F=0.1 –𝛽=0.04
Table 1: Flow parameters of the GAI scenario.
 0 0.1 0.2 0.3 0.4 0.5 0.6
-0.1  0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1  1.1H I S , A , F Pcoupon addition probability
Cf
Figure 2: Impact of 𝐶𝑓on the coupon addition probabilities
of the red, green and blue primary colors. Vertical dashed
lines correspond to the values chosen for the system flows.
Before explaining the rationale behind our parameters’ choice,
it is useful to observe on Fig. 2 the impact of quality bias parame-
ter𝐶𝑓on the coupon addition probabilities {𝑟𝑓
𝑐}𝑐of each primary
component𝑐, see eq.(1). Value -0.08, chosen for process H, reflects
the efforts of dataset curators to retain only high-quality informa-
tion. Conversely, the value 1, assigned to process P, reflects the
limited attention to quality for the information initially published
on the Web. Moreover, we consider that the indexing process I
performs a rather good job at quality discrimination ( 𝐶I=0), while
a slightly less effective filtering (value 0.1) is applied by users on
machine-generated answers (processes S,A,F).
Table 1 reveals a 10:1 ratio between rates ΛS′andΛA′, reflecting
the assumption that, within the GAI scenario, users still predomi-
nantly rely on conventional search engines for verifying informa-
tion incorporated into new digital content. We further notice on
Table 1: i)𝛾=0.5, meaning that half of the information incorporated
3In actuality most parameters should be considered time-dependent due to the rapid
evolution of information retrieval practices, but we adopt a simplifying assumption of
constancy for the duration of the topic’s lifespan, i.e., the temporal window in which
the bulk of questions and answers related to the topic are generated.

Information Retrieval in the Age of Generative AI: The RGB Model SIGIR ’25, July 13–18, 2025, Padua, Italy
into the Training Set comes from online resources; ii) 𝜇𝑆=𝜆Iand
𝜇𝐿=𝜆T, meaning that the strength associated to answers learned
by both Search Engine and LLM is discounted over time at the
same rate at which it is reinforced. iii) 𝛼=0.4, indicating that 40%
of answers generated by GAI tools are assumed to be used in the
creation of new content.
Taking this configuration as our reference GAI scenario, the
parameters for the other two scenarios can be readily specified by
difference with respect to those listed in Table 1. In the pre-GAI
scenario, we simply set ΛA=0, effectively nullifying all effects
produced by users employing the GAI system.
For the post-GAI scenario, we envision a few fundamental pa-
rameter shifts inspired by the current trends in information retrieval
practices. The primary modification involves an inversion of the
rates for processes S′andA′, with ΛS′now set to 10 and ΛA′to
100. In light of the projected dominance of GAI-based information
retrieval, we establish 𝛼at 0.8 and postulate a tenfold increase in the
rate of processH(incorporation of answers in curated datasets),
setting ΛHto 1. Crucially, we hypothesize that users will increas-
ingly rely on GAI-generated responses, consequently reducing their
efforts in quality discrimination when incorporating answers in
new digital content. This shift is reflected in the adjustment of 𝐶A′
from 0.1 (as in the GAI scenario) to 1 in the post-GAI scenario.
We commence by presenting select findings from the fundamen-
tal RGB model, wherein color mixing is precluded; that is, neither
algorithms nor users synthesize novel responses by amalgamating
information from primary answers. This model is particularly suit-
able for queries that elicit some form of discrete, factual information
not amenable to synthesis from different sources.
 0.0001 0.001 0.01 0.1 1
 1  10  100  1000  10000FIUA
time (days)pre-GAI
GAI
post-GAI
Figure 3: Temporal evolution of the FIUA indicator for the
three considered scenarios. Shaded areas correspond to 95%-
level confidence intervals (as in subsequent figures).
Fig. 3 illustrates, as a function of time, the fraction of irrelevant
used answers (FIUA) found on the Web, under the three considered
scenarios. Shaded regions surrounding each curve represent 95%-
level confidence intervals derived from 400 simulation runs. These
regions reveal an erratic behavior during the first ten days, after
which a clear separation among the curves emerges: interestingly,
theGAI scenario produces fewer irrelevant answers compared to
thepre-GAI condition. However, the post-GAI scenario ultimately
yields the least favorable outcome for this critical metric. Although
the reinforcement loop of high-quality answers effectively reducesFIUA to relatively low levels across all scenarios, the post-GAI con-
dition generates approximately an order of magnitude more irrelevant
answers than the other conditions.
-0.2 0 0.2 0.4 0.6 0.8 1 1.2
 10  100  1000  10000pre-GAI
GAI
post-GAIAIRI
time (days)
 0 0.2 0.4 0.6 0.8 1
 1  10  100  1000  10000pre-GAI
GAI
post-GAIAIAI
time (days)
Figure 4: Temporal evolution of the AIRI (left plot) and AIAI
indicator (right plot) for the three considered scenarios.
The AIRI metric (left in Fig. 4) reveals that almost all irrelevant
answers produced in the post-GAI scenario come from the generative
AI system . The AIAI indicator (right in Fig. 4) further reveals that
the LLM model is affected by a substantial degree of autophagy under
both GAI and post-GAI conditions , ranging from 60% to 70%.
 0.0001 0.001 0.01 0.1 1
 20  40  60  80  100  120 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1P {majority irrelevant}
FRQ
time (days)NbS(0) = 1
NbS(0) = 10
NbS(0) = 100
Figure 5: Probability over time that the majority of used an-
swers on the Web are irrelevant (left y-axes), and FRQ indica-
tor (right y-axes) in the post-GAI scenario, for initial number
of black coupon in the Search Engine equal to 1,10,100.
Our system exhibits high stochasticity, particularly during the
initial months of topic existence. A notable concern is the potential
for irrelevant responses to gain an early, chance-driven advantage
over relevant ones. To investigate this critical phenomenon, we
conducted 30,000 simulations spanning the first 120 days, calculat-
ing the proportion of runs in which, at any given time 𝑡, irrelevant
used answers in the Web outnumber relevant ones. Focusing on
thepost-GAI scenario, we plot this novel metric while varying the
initial number 𝑁𝑆
𝑏(0)of black coupons in the Search Engine, which
emulates more or less cautious answering strategies. Fig. 5 presents
these results alongside the corresponding Fraction of Responded
Queries (FRQ) metric on the secondary y-axis. A clear trade-off
emerges, whereby more cautious strategies (larger 𝑁𝑆
𝑏(0)) can sig-
nificantly mitigate the probability of the aforementioned critical
event, albeit at the expense of diminishing the overall fraction of
responded queries (see for example the values at 𝑡=60, marked by
a vertical dotted line).

SIGIR ’25, July 13–18, 2025, Padua, Italy Garetto et al.
4.1 Results Under Answer Mixing
This section presents findings from our case study configuration, ex-
amining scenarios wherein either the GAI system or users utilizing
conventional search engines synthesize novel responses by integrat-
ing two distinct information sources. This process progressively
populates the system with coupons of intermediate chromatic val-
ues, containing various proportions of primary RGB components.
For a detailed description of the extended model incorporating this
fundamental aspect of information synthesis, readers are referred
back to Sec. 3.4. Recall that, while we consider the GAI system to
uniformely mix two randomly chosen answers, we instead assume
that users employing the conventional search engine generate a bi-
ased combination of two random answers, applying to the best one a
weight𝜉∈[1/2,1]. The parameter 𝜉thus quantifies the average dis-
criminatory capacity of users in assessing answer quality. A value
of𝜉=1/2represents an unbiased mixture, indicating no discrim-
ination, while 𝜉=1denotes perfect discrimination, where users
consistently identify and select the superior information source.
We will separately consider the GAI and post-GAI scenarios,
comparing in each scenario the effects of the following conditions:
no-mix This corresponds to the base case in which answers are
never mixed. It serves as a baseline for comparative analysis.
mix-GAI We enable mixing exclusively for the GAI system. Users
employing the Search Engine still use single-source responses.
mix-GAI+SE( 𝜉)We enable mixing for both GAI and Search Engine.
We will investigate the effects of varying the quality discrimi-
nation parameter 𝜉, taking values of 0.5, 0.75, and 1.0.
0.0000100.0001000.0010000.0100000.1000001.000000
 1  10  100  1000  10000FIUA
time (days)no mix
mix GAI
mix GAI+SE ξ=0.5
mix GAI+SE ξ=0.75
mix GAI+SE ξ=1
Figure 6: Temporal evolution of the FIUA indicator in the
GAI scenario, under different answer mixing assumptions.
Fig. 6 reports the Fraction of Irrelevant Used Answers (FIUA)
found in the WWW within the GAI scenario. Recall that here there
is a 10:1 ratio between the utilization rates of answers derived from
the SE compared to those generated by the GAI. We notice that
the mix-GAI condition consistently yields a higher proportion of
irrelevant answers compared to the no-MIX baseline. This observa-
tion suggests that the indiscriminate amalgamation of information
from diverse sources generally proves detrimental to answer quality.
When we also enable the mixing of SE answers by users, we ob-
serve a substantial impact of the quality discrimination parameter
𝜉: a markedly large FIUA is obtained when users indiscriminately
combine two sources ( 𝜉=0.5). With𝜉=0.75, a very long timeis required for the FIUA to approach levels comparable to the no-
mix baseline. Perfect discrimination ( 𝜉=1) is much faster, and
ultimately yields to lowest FIUA among all conditions.
 0.0001 0.001 0.01 0.1 1
 1  10  100  1000  10000FIUA
time (days)no mix
mix GAI
mix GAI+SE ξ=0.5
mix GAI+SE ξ=0.75
mix GAI+SE ξ=1
Figure 7: Temporal evolution of the FIUA indicator in the
post-GAI scenario, under different answer mixing.
We conducted a similar investigation in the post-GAI scenario.
Recall that here the vast majority of used answers come from the
GAI system. The Fraction of Irrelevant Used Answers (FIUA), re-
ported in Fig. 7 confirms that the generation of responses through
the amalgamation of disparate sources consistently yields deleteri-
ous effects on the overall quality of information across all temporal
scales. Indeed, the proportion of irrelevant information on the Web
roughly doubles across all considered mixing hypotheses, due to
the reduced impact of search engine-generated responses. Here
the quality discrimination performed by users has marginal impact
because only few users still employ results by the Search Engine.
5 Stack Exchange Analysis
Many generative AI systems opt for a chat-based interaction with
the user and generally tend to provide a singular, authoritative
response even to novel questions. This precludes the presentation
of diverse perspectives and effectively suppresses the competition
of ideas that might arise from multiple potential answers. To find
out to what extent this modality can lead to suboptimal question-
and-answer (Q&A) systems, we investigated one of the most well-
known Q&A platforms: Stack Exchange. In particular, we looked
at the largest sub-community, StackOverflow, which mainly deals
with computer science and coding, and the largest (non-computer
science-related) community, MathStackExchange, which covers a
wide range of mathematical topics.
On Stack Exchange, users submit questions, and community
members provide answers, which can then be evaluated through
upvotes and downvotes. This crowdsourced rating system deter-
mines the ranking of answers, with the highest-rated response,
hereafter referred to as best answer, displayed first, followed by
others in descending order of their accumulated score. We ex-
tracted 11,555,969 questions from StackOverflow together with
the corresponding 20,166,328 answers, generated by over 1 million
users. Similarly, we extracted 1,068,196 MathStackExchange ques-
tions and their 1,493,849 answers. The aggregated size of the raw
dataset is about 150 GB. We processed this extensive dataset on a
HPC cluster, utilizing the Dask Python library to flexibly manage
parallel computation. All code utilized in the data processing phase
will be made publicly accessible [14].

Information Retrieval in the Age of Generative AI: The RGB Model SIGIR ’25, July 13–18, 2025, Padua, Italy
/uni00000029/uni00000055/uni00000052/uni00000050/uni00000003/uni00000054/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000057/uni00000052/uni00000003/uni00000049/uni0000004c/uni00000055/uni00000056/uni00000057/uni00000003/uni00000044/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000053/uni00000052/uni00000056/uni00000057/uni00000048/uni00000047 /uni00000029/uni00000055/uni00000052/uni00000050/uni00000003/uni00000054/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000057/uni00000052/uni00000003/uni00000045/uni00000048/uni00000056/uni00000057/uni00000003/uni00000044/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000053/uni00000052/uni00000056/uni00000057/uni00000048/uni00000047 /uni00000029/uni00000055/uni00000052/uni00000050/uni00000003/uni00000054/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000057/uni00000052/uni00000003/uni00000045/uni00000048/uni00000056/uni00000057/uni00000003/uni00000044/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000048/uni00000056/uni00000057/uni00000044/uni00000045/uni0000004f/uni0000004c/uni00000056/uni0000004b/uni00000048/uni00000047
/uni00000014/uni00000013/uni00000013/uni00000014/uni00000013/uni00000014/uni00000014/uni00000013/uni00000015/uni00000014/uni00000013/uni00000016/uni00000014/uni00000013/uni00000017/uni00000014/uni00000013/uni00000018/uni00000014/uni00000013/uni00000019/uni00000014/uni00000013/uni0000001a
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni00000028/uni0000004f/uni00000044/uni00000053/uni00000056/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000050/uni0000004c/uni00000051/uni00000058/uni00000057/uni00000048/uni00000056/uni0000000c/uni00000014/uni00000013/uni00000017
/uni00000014/uni00000013/uni00000016
/uni00000014/uni00000013/uni00000015
/uni00000014/uni00000013/uni00000014
/uni00000014/uni00000013/uni00000013/uni00000026/uni00000026/uni00000027/uni00000029/uni00000014/uni00000003/uni00000047/uni00000044/uni0000005c /uni00000014/uni00000003/uni00000050/uni00000052/uni00000051/uni00000057/uni0000004b /uni00000014/uni00000003/uni0000005c/uni00000048/uni00000044/uni00000055
(a) StackOverflow
/uni00000014/uni00000013/uni00000013/uni00000014/uni00000013/uni00000014/uni00000014/uni00000013/uni00000015/uni00000014/uni00000013/uni00000016/uni00000014/uni00000013/uni00000017/uni00000014/uni00000013/uni00000018/uni00000014/uni00000013/uni00000019/uni00000014/uni00000013/uni0000001a
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni00000028/uni0000004f/uni00000044/uni00000053/uni00000056/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000050/uni0000004c/uni00000051/uni00000058/uni00000057/uni00000048/uni00000056/uni0000000c/uni00000014/uni00000013/uni00000017
/uni00000014/uni00000013/uni00000016
/uni00000014/uni00000013/uni00000015
/uni00000014/uni00000013/uni00000014
/uni00000014/uni00000013/uni00000013/uni00000026/uni00000026/uni00000027/uni00000029/uni00000014/uni00000003/uni00000047/uni00000044/uni0000005c /uni00000014/uni00000003/uni00000050/uni00000052/uni00000051/uni00000057/uni0000004b /uni00000014/uni00000003/uni0000005c/uni00000048/uni00000044/uni00000055 (b) MathStackExchange
Figure 8: Complementary cumulative distribution function (CCDF) of the time it takes for an answer to be posted (dash-dotted),
for the best answer to be posted (dotted), and for the best answer to emerge (solid) (log-log scale).
/uni00000013 /uni00000014/uni00000013/uni00000013/uni00000013 /uni00000015/uni00000013/uni00000013/uni00000013 /uni00000016/uni00000013/uni00000013/uni00000013 /uni00000017/uni00000013/uni00000013/uni00000013 /uni00000018/uni00000013/uni00000013/uni00000013
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni00000028/uni0000004f/uni00000044/uni00000053/uni00000056/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000047/uni00000044/uni0000005c/uni00000056/uni0000000c/uni00000014/uni00000013/uni00000018
/uni00000014/uni00000013/uni00000017
/uni00000014/uni00000013/uni00000016
/uni00000014/uni00000013/uni00000015
/uni00000014/uni00000013/uni00000014
/uni00000014/uni00000013/uni00000013/uni00000026/uni00000026/uni00000027/uni00000029
/uni00000030/uni00000044/uni00000057/uni0000004b/uni00000036/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000028/uni0000005b/uni00000046/uni0000004b/uni00000044/uni00000051/uni0000004a/uni00000048
/uni00000036/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004f/uni00000052/uni0000005a
Figure 9: CCDF of the time required, in days, for the best
answer to be recognized as superior for MathStackExchange
(dashed) and StackOverflow (solid).
We found that while a large proportion of questions receive
the best answer within minutes4, a significant subset requires an
extended period – sometimes spanning years – for the optimal
response to emerge (for 10% of StackOverflow questions, the best
answer is only recognized after 1 year). Figure 8 illustrates the
complementary cumulative distribution function (CCDF) of three
key time intervals: i) the duration until the first answer to a question
is submitted (dash-dotted line); ii) the period until the best answer is
posted (dotted line); iii) the interval until the best answer achieves
its primacy in user display order (solid line). All intervals above are
are calculated from the initial question submission time. Figure 9
complements previous results by depicting the CCDF of the amount
of time required for the best answer to surface, measured relative
to its initial posting time.
It is evident that there is a significant subset of questions requir-
ing an extended temporal interval and significant collective human
effort, both for the generation of high-quality responses and for the
community’s ultimate recognition of their superiority.
6 Discussion and Conclusions
To the best of our knowledge, ours is the first attempt to capture into
a relatively simple analytical model high-level dynamics of topic-
specific information across different digital resources, highlighting
potential emergent phenomena and risks arising from evolving
4The median time for the bestanswer to appear is 47 min on Math S.E. and 77 min on
StackOverflow. The time for the first answer to appear is even less: 34 and 39 min.information retrieval behaviors. To this purpose, we had to strike a
balance between simplicity and representational power.
Our model incorporates parameters of various natures, which
are difficult to determine from real data for several reasons:
(1)Some parameters are defined at a high aggregation level (e.g.
the global rate of information on a topic indexed by search en-
gines or included in training sets). These are difficult to obtain
from Q&A datasets that do not track global-scale dynamics.
(2)Some are unknown (such as the rate and accuracy at which
digital resources are crawled or fed into LLMs), as private com-
panies keep their algorithms and metrics strictly confidential.
(3)Some relate to the intrinsic quality of an answer, a metric that
may be difficult to measure in the most general case.
(4)Some concern human behavior during an ongoing paradigm
shift in information retrieval, for which reliable measurements
are either missing or very limited.
(5)Most are changing rapidly, causing any dataset to become
outdated within months of collection.
For all the above reasons, trying to fit the parameters in Table 1
using datasets makes little sense. The results produced by our model
are not meaningful in absolute terms. They become interesting (in a
relative sense) when we perform what-if analysis, i.e. when we vary
a few crucial parameters while keeping the others fixed. We pro-
vided an example of what-if analysis in Sec. 4, where we compared
two scenarios: GAI, describing answer dynamics for a hypothetical
topic under current conditions, and post-GAI, describing dynamics
of the same topic in an envisioned future state.
The impressive increase in user requests to AI-powered assis-
tants and the associated proliferation of AI-generated content on
the web require a critical examination of future scenarios. To ad-
dress this pressing issue, we have introduced an initial quantitative
framework to assess and project the impact of generative AI tools
on the information ecosystem. Our preliminary results suggest a
significant risk of increased misinformation proliferation, driven
by two key factors: the generation of persuasive or authoritative
answers to topics that are not sufficiently consolidated, and the ten-
dency of individuals to reduce their cognitive effort in evaluating
different answers and discriminating information quality. We hope
our research will make a meaningful contribution to the ongoing
discourse on the responsible development of AI technologies.

SIGIR ’25, July 13–18, 2025, Padua, Italy Garetto et al.
References
[1]Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Hu-
mayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G. Baraniuk.
2023. Self-Consuming Generative Models Go MAD. arXiv:2307.01850 [cs.LG]
https://arxiv.org/abs/2307.01850
[2]Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova
DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas
Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson
Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston,
Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom
Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan.
2022. Training a Helpful and Harmless Assistant with Reinforcement Learning
from Human Feedback. arXiv:2204.05862 [cs.CL] https://arxiv.org/abs/2204.05862
[3]Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret
Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be
Too Big?. In Proceedings of the 2021 ACM Conference on Fairness, Accountability,
and Transparency (Virtual Event, Canada) (FAccT ’21) . Association for Computing
Machinery, New York, NY, USA, 610–623. doi:10.1145/3442188.3445922
[4]Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong,
and Gauthier Gidel. 2024. On the Stability of Iterative Retraining of Generative
Models on their own Data. arXiv:2310.00429 [cs.LG] https://arxiv.org/abs/2310.
00429
[5]Marcel Binz, Stephan Alaniz, Adina Roskies, Balazs Aczel, Carl T. Bergstrom,
Colin Allen, Daniel Schad, Dirk Wulff, Jevin D. West, Qiong Zhang, Richard M.
Shiffrin, Samuel J. Gershman, Ven Popov, Emily M. Bender, Marco Marelli,
Matthew M. Botvinick, Zeynep Akata, and Eric Schulz. 2023. How
should the advent of large language models affect the practice of science?
arXiv:2312.03759 [cs.CL] https://arxiv.org/abs/2312.03759
[6]Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora,
Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma
Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon,
Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Dem-
szky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John
Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren
Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori
Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu,
Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth
Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark
Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina
Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu
Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele
Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen
Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel
Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christo-
pher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf
Roohani, Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa,
Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori,
Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu,
Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You,
Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia
Zheng, Kaitlyn Zhou, and Percy Liang. 2022. On the Opportunities and Risks of
Foundation Models. arXiv:2108.07258 [cs.LG] https://arxiv.org/abs/2108.07258
[7]Ali Borji. 2023. A Categorical Archive of ChatGPT Failures.
arXiv:2302.03494 [cs.CL] https://arxiv.org/abs/2302.03494
[8]Martin Briesch, Dominik Sobania, and Franz Rothlauf. 2024. Large Language
Models Suffer From Their Own Output: An Analysis of the Self-Consuming
Training Loop. arXiv:2311.16822 [cs.LG] https://arxiv.org/abs/2311.16822
[9]Boxi Cao, Hongyu Lin, Xianpei Han, Le Sun, Lingyong Yan, Meng Liao, Tong
Xue, and Jin Xu. 2021. Knowledgeable or Educated Guess? Revisiting Language
Models as Knowledge Bases. In Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 1: Long Papers) , Chengqing Zong,
Fei Xia, Wenjie Li, and Roberto Navigli (Eds.). Association for Computational
Linguistics, Online, 1860–1874. doi:10.18653/v1/2021.acl-long.146
[10] Wenhu Chen, Xinyi Wang, William Yang Wang, and William Yang Wang.
2021. A Dataset for Answering Time-Sensitive Questions. In Proceed-
ings of the Neural Information Processing Systems Track on Datasets
and Benchmarks (NeurIPS ’21, Vol. 1) , J. Vanschoren and S. Yeung (Eds.).
https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/
file/1f0e3dad99908345f7439f8ffabdffc4-Paper-round2.pdf
[11] I-Chun Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting
Zhou, Junxian He, Graham Neubig, and Pengfei Liu. 2023. FacTool: Factuality
Detection in Generative AI – A Tool Augmented Framework for Multi-Task and
Multi-Domain Scenarios. arXiv:2307.13528 [cs.CL] https://arxiv.org/abs/2307.
13528
[12] Maria del Rio-Chanona, Nadzeya Laurentsyeva, and Johannes Wachs. 2023. Are
Large Language Models a Threat to Digital Public Goods? Evidence from Activityon Stack Overflow. arXiv:2307.07367 [cs.SI] https://arxiv.org/abs/2307.07367
[13] Elvis Dohmatob, Yunzhen Feng, and Julia Kempe. 2024. Model Collapse Demys-
tified: The Case of Regression. arXiv:2402.07712 [cs.LG] https://arxiv.org/abs/
2402.07712
[14] Franco Galante et al. 2025. RGB-model. https://github.com/Franco-Galante/RGB-
model.
[15] Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry
Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey
Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, and Sanmi Koyejo. 2024.
Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating
Real and Synthetic Data. arXiv:2404.01413 [cs.LG] https://arxiv.org/abs/2404.
01413
[16] Josh Goldstein, Girish Sastry, Micah Musser, Renee DiResta, Matthew Gentzel, and
Katerina Sedova. 2023. Generative Language Models and Automated Influence
Operations: Emerging Threats and Potential Mitigations. doi:10.48550/arXiv.
2301.04246
[17] Ben Goodrich, Vinay Rao, Peter J. Liu, and Mohammad Saleh. 2019. Assessing
The Factual Accuracy of Generated Text. In Proceedings of the 25th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining (Anchorage, AK,
USA) (KDD ’19) . Association for Computing Machinery, New York, NY, USA,
166–175. doi:10.1145/3292500.3330955
[18] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination
in Natural Language Generation. ACM Comput. Surv. 55, 12, Article 248 (March
2023), 38 pages. doi:10.1145/3571730
[19] Zhen Jia, Philipp Christmann, and Gerhard Weikum. 2024. TIQ: A Benchmark
for Temporal Question Answering with Implicit Time Constraints. In Companion
Proceedings of the ACM Web Conference 2024 (Singapore, Singapore) (WWW
’24). Association for Computing Machinery, New York, NY, USA, 1394–1399.
doi:10.1145/3589335.3651895
[20] Samia Kabir, David N. Udo-Imeh, Bonan Kou, and Tianyi Zhang. 2024. Is Stack
Overflow Obsolete? An Empirical Study of the Characteristics of ChatGPT An-
swers to Stack Overflow Questions. In Proceedings of the 2024 CHI Conference
on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24) . Asso-
ciation for Computing Machinery, New York, NY, USA, Article 935, 17 pages.
doi:10.1145/3613904.3642596
[21] Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai,
Xinyan Velocity Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro
Inui. 2024. REALTIME QA: what’s the answer right now?. In Proceedings of
the 37th International Conference on Neural Information Processing Systems (New
Orleans, LA, USA) (NIPS ’23) . Curran Associates Inc., Red Hook, NY, USA, Article
2130, 19 pages.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Proceedings of the 34th International Conference
on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20) .
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[23] Aiwei Liu, Qiang Sheng, and Xuming Hu. 2024. Preventing and Detecting Mis-
information Generated by Large Language Models. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval (Washington DC, USA) (SIGIR ’24) . Association for Computing
Machinery, New York, NY, USA, 3001–3004. doi:10.1145/3626772.3661377
[24] Gonzalo Martínez, Lauren Watson, Pedro Reviriego, José Alberto Hernández,
Marc Juarez, and Rik Sarkar. 2023. Towards Understanding the Interplay of
Generative Artificial Intelligence and the Internet. arXiv:2306.06130 [cs.AI]
https://arxiv.org/abs/2306.06130
[25] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On
Faithfulness and Factuality in Abstractive Summarization. In Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics , Dan Jurafsky,
Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computa-
tional Linguistics, Online, 1906–1919. doi:10.18653/v1/2020.acl-main.173
[26] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song,
Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham,
Geoffrey Irving, and Nat McAleese. 2022. Teaching language models to support
answers with verified quotes. arXiv:2203.11147 [cs.CL] https://arxiv.org/abs/
2203.11147
[27] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and
Chelsea Finn. 2023. DetectGPT: zero-shot machine-generated text detection
using probability curvature. In Proceedings of the 40th International Conference
on Machine Learning (Honolulu, Hawaii, USA) (ICML’23) . JMLR.org, Article 1038,
13 pages.
[28] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong,
Juntong Song, and Tong Zhang. 2024. RAGTruth: A Hallucination Corpus for
Developing Trustworthy Retrieval-Augmented Language Models. In Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.).
Association for Computational Linguistics, Bangkok, Thailand, 10862–10878.

Information Retrieval in the Age of Generative AI: The RGB Model SIGIR ’25, July 13–18, 2025, Padua, Italy
doi:10.18653/v1/2024.acl-long.585
[29] Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and
William Wang. 2023. On the Risk of Misinformation Pollution with Large Lan-
guage Models. In Findings of the Association for Computational Linguistics: EMNLP
2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computa-
tional Linguistics, Singapore, 1389–1403. doi:10.18653/v1/2023.findings-emnlp.97
[30] Murray Shanahan. 2024. Talking about Large Language Models. Commun. ACM
67, 2 (Jan. 2024), 68–79. doi:10.1145/3624724
[31] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and
Ross Anderson. 2024. The Curse of Recursion: Training on Generated Data Makes
Models Forget. arXiv:2305.17493 [cs.LG] https://arxiv.org/abs/2305.17493
[32] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson,
and Yarin Gal. 2024. AI models collapse when trained on recursively generated
data. Nature 631, 8022 (July 2024), 755–759. doi:10.1038/s41586-024-07566-y
[33] Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, and Roman Teucher. 2024.
RAG-Ex: A Generic Framework for Explaining Retrieval Augmented Gener-
ation. In Proceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval (Washington DC, USA) (SIGIR
’24). Association for Computing Machinery, New York, NY, USA, 2776–2780.
doi:10.1145/3626772.3657660
[34] Yiming Tan, Dehai Min, Yu Li, Wenbo Li, Nan Hu, Yongrui Chen, and Guilin Qi.
2023. Can ChatGPT Replace Traditional KBQA Models? An In-Depth Analysis of
the Question Answering Performance of the GPT LLM Family. In The Semantic
Web – ISWC 2023: 22nd International Semantic Web Conference, Athens, Greece,
November 6–10, 2023, Proceedings, Part I (Athens, Greece). Springer-Verlag, Berlin,
Heidelberg, 348–367. doi:10.1007/978-3-031-47240-4_19
[35] Ruixiang Tang, Yu-Neng Chuang, and Xia Hu. 2024. The Science of Detecting
LLM-Generated Text. Commun. ACM 67, 4 (March 2024), 50–59. doi:10.1145/
3624725
[36] S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani,
Vipula Rawte, Aman Chadha, and Amitava Das. 2024. A Comprehensive
Survey of Hallucination Mitigation Techniques in Large Language Models.
arXiv:2401.01313 [cs.CL] https://arxiv.org/abs/2401.01313
[37] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei,
Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024.FreshLLMs: Refreshing Large Language Models with Search Engine Augmen-
tation. In Findings of the Association for Computational Linguistics ACL 2024 ,
Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Com-
putational Linguistics, Bangkok, Thailand and virtual meeting, 13697–13720.
doi:10.18653/v1/2024.findings-acl.813
[38] Ivan Vykopal, Matúš Pikuliak, Ivan Srba, Robert Moro, Dominik Macko, and
Maria Bielikova. 2024. Disinformation Capabilities of Large Language Models.
InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
14830–14847. doi:10.18653/v1/2024.acl-long.793
[39] Xi Wang, Procheta Sen, Ruizhe Li, and Emine Yilmaz. 2024. Adaptive Retrieval-
Augmented Generation for Conversational Systems. arXiv:2407.21712 [cs.CL]
https://arxiv.org/abs/2407.21712
[40] Shu Yang, Muhammad Asif Ali, Lu Yu, Lijie Hu, and Di Wang. 2024.
MONAL: Model Autophagy Analysis for Modeling Human-AI Interactions.
arXiv:2402.11271 [cs.CL] https://arxiv.org/abs/2402.11271
[41] Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Lifeng Jin, Linfeng Song,
Haitao Mi, and Helen Meng. 2024. Self-Alignment for Factuality: Mitigating Hal-
lucinations in LLMs via Self-Evaluation. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers) , Lun-Wei
Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational
Linguistics, Bangkok, Thailand, 1946–1965. doi:10.18653/v1/2024.acl-long.107
[42] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting
Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu,
Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s Song in the AI Ocean: A
Survey on Hallucination in Large Language Models. arXiv:2309.01219 [cs.CL]
https://arxiv.org/abs/2309.01219
[43] Jiawei Zhou, Yixuan Zhang, Qianni Luo, Andrea G Parker, and Munmun
De Choudhury. 2023. Synthetic Lies: Understanding AI-Generated Misinfor-
mation and Evaluating Algorithmic and Human Solutions. In Proceedings of the
2023 CHI Conference on Human Factors in Computing Systems (Hamburg, Ger-
many) (CHI ’23) . Association for Computing Machinery, New York, NY, USA,
Article 436, 20 pages. doi:10.1145/3544548.3581318
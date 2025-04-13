# PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Published**: 2025-04-10 13:09:50

**PDF URL**: [http://arxiv.org/pdf/2504.07717v1](http://arxiv.org/pdf/2504.07717v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable performance across
a wide range of applications, e.g., medical question-answering, mathematical
sciences, and code generation. However, they also exhibit inherent limitations,
such as outdated knowledge and susceptibility to hallucinations.
Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to
address these issues, but it also introduces new vulnerabilities. Recent
efforts have focused on the security of RAG-based LLMs, yet existing attack
methods face three critical challenges: (1) their effectiveness declines
sharply when only a limited number of poisoned texts can be injected into the
knowledge database, (2) they lack sufficient stealth, as the attacks are often
detectable by anomaly detection systems, which compromises their effectiveness,
and (3) they rely on heuristic approaches to generate poisoned texts, lacking
formal optimization frameworks and theoretic guarantees, which limits their
effectiveness and applicability. To address these issues, we propose
coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack
that introduces a small number of poisoned texts into the knowledge database
while embedding a backdoor trigger within the prompt. When activated, the
trigger causes the LLM to generate pre-designed responses to targeted queries,
while maintaining normal behavior in other contexts. This ensures both high
effectiveness and stealth. We formulate the attack generation process as a
bilevel optimization problem leveraging a principled optimization framework to
develop optimal poisoned texts and triggers. Extensive experiments across
diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving
a high attack success rate even with a limited number of poisoned texts and
significantly improved stealth compared to existing methods.

## Full Text


<!-- PDF content starts -->

PR-Attack: Coordinated Prompt-RAG Attacks on
Retrieval-Augmented Generation in Large Language Models via
Bilevel Optimization
Yang Jiao
Tongji UniversityXiaodong Wang
Columbia UniversityKai Yang
Tongji University
Abstract
Large Language Models (LLMs) have demonstrated remarkable per-
formance across a wide range of applications, e.g., medical question-
answering, mathematical sciences, and code generation. However,
they also exhibit inherent limitations, such as outdated knowledge
and susceptibility to hallucinations. Retrieval-Augmented Genera-
tion (RAG) has emerged as a promising paradigm to address these
issues, but it also introduces new vulnerabilities. Recent efforts
have focused on the security of RAG-based LLMs, yet existing at-
tack methods face three critical challenges: (1) their effectiveness
declines sharply when only a limited number of poisoned texts can
be injected into the knowledge database, (2) they lack sufficient
stealth, as the attacks are often detectable by anomaly detection sys-
tems, which compromises their effectiveness, and (3) they rely on
heuristic approaches to generate poisoned texts, lacking formal op-
timization frameworks and theoretic guarantees, which limits their
effectiveness and applicability. To address these issues, we propose
coordinated Prompt-RAG attack (PR-attack), a novel optimization-
driven attack that introduces a small number of poisoned texts into
the knowledge database while embedding a backdoor trigger within
the prompt. When activated, the trigger causes the LLM to generate
pre-designed responses to targeted queries, while maintaining nor-
mal behavior in other contexts. This ensures both high effectiveness
and stealth. We formulate the attack generation process as a bilevel
optimization problem leveraging a principled optimization frame-
work to develop optimal poisoned texts and triggers. Extensive
experiments across diverse LLMs and datasets demonstrate the ef-
fectiveness of PR-Attack, achieving a high attack success rate even
with a limited number of poisoned texts and significantly improved
stealth compared to existing methods. These results highlight the
potential risks posed by PR-Attack and emphasize the importance
of securing RAG-based LLMs against such threats.
CCS Concepts
•Information systems →Adversarial retrieval .
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
,
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Retrieval-Augmented Generation, Large Language Models, Bilevel
Optimization
ACM Reference Format:
Yang Jiao, Xiaodong Wang, and Kai Yang. 2025. PR-Attack: Coordinated
Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Lan-
guage Models via Bilevel Optimization. In Proceedings of . ACM, New York,
NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large Language Models (LLMs) have exhibited exceptional perfor-
mance across a broad spectrum of applications, such as medical
question-answering [ 46], chemical research [ 5], and mathematical
sciences [ 55]. Prompt learning plays a crucial role in enhancing the
adaptability of LLMs to various downstream tasks [ 6,9,15,36,99].
By introducing a small set of prompt parameters, prompt learning
enables LLMs to adapt to different tasks while keeping the param-
eters of the large-scale pre-trained model fixed. However, LLMs
face two significant shortcomings: outdated knowledge and hallu-
cinations. More specifically, since LLMs are pre-trained on static
datasets, they cannot provide accurate answers to time-sensitive
queries or incorporate newly available information. In addition,
LLMs often generate hallucinations, i.e., inaccurate responses due
to a lack of grounding in factual sources. Retrieval-Augmented
Generation (RAG) [ 8,19,57,58,63,73] addresses these limitations
by combining LLMs with an external retrieval system that fetches
relevant, up-to-date information from knowledge bases or docu-
ments. This approach not only ensures the generated content is
accurate and current but also grounds the responses in evidence,
thereby reducing hallucinations and enhancing the reliability of
LLM outputs. RAG essentially consists of three key components
[105], i.e., knowledge database, retriever, and LLM. A knowledge
database encompasses a vast array of texts gathered from diverse
sources, such as Wikipedia [ 66], web documents [ 48], and more.
The retriever aims to retrieve the top- 𝑘most relevant texts from the
knowledge database for the given question. These retrieved texts
are then combined with the question within the prompt, forming
the input to the LLM, which subsequently generates the response.
The security of RAG-based LLMs has gained considerable at-
tention due to their widespread adoption in applications where
data integrity and reliability are critical. PoisonedRAG [ 105] has
been introduced as a framework to study the attacks targeting
RAG-based LLMs. Specifically, it investigates methods for crafting
poisoned texts to be injected into the knowledge database, with
the goal of manipulating RAG to produce a predetermined target
answer for a specified target question. Likewise, GGPP [ 19] aims to
insert a prefix into the prompt to guide the retriever in retrievingarXiv:2504.07717v1  [cs.CR]  10 Apr 2025

, ,
the target poisoned texts, thereby causing the LLM to generate the
target answer. However, there are three key issues in the existing at-
tacks on RAG-based LLMs: 1) they rely exclusively on the poisoned
texts injected into the knowledge base, resulting in a significant
decline in attack efficiency, i.e., success attack rate, as the number of
injected poisoned texts decreases. 2) In addition, the attack’s exclu-
sive reliance on injected poisoned texts significantly increases its
susceptibility to detection by anomaly detection systems, thereby
compromising its stealth. 3) These methods predominantly rely
on heuristic approaches for generating poisoned texts, lacking for-
mal optimization frameworks and theoretical guarantees, which
restricts both their effectiveness and broader applicability.
Motivation. This paper introduces a novel attack paradigm, the
coordinated Prompt-RAG Attack (PR-Attack). Retrieved texts from
knowledge database are integrated with prompts to form the input
to LLMs. Solely attacking either the knowledge database or prompt
is less effective due to the limited scope of influence [ 105]. For
instance, attacking only the prompt fails to fully exploit the inter-
action between the retrieval and generation components, resulting
in reduced control over the final output. Furthermore, attacks on
a single component tend to generate more predictable patterns in
the LLM’s responses, making them easier to identify using existing
defense mechanisms. In contrast, a joint poisoning approach lever-
ages the mutual influence between the prompt and the retrieved
texts, enabling more coordinated and stealthier attacks that are
harder to detect and more effective in achieving their objectives.
Additionally, according to the Social Amplification of Risk Frame-
work (SARF) [ 31,56,77], the impact of attacks during critical peri-
ods can be significantly amplified due to the rapid dissemination of
information through channels such as social media. Moreover, the
poisoned texts are more easily detected when LLMs consistently
generate incorrect answers. Motivated by this, we explore backdoor
attacks within prompt learning to make PR-Attack more stealthy
and adaptable. As far as we are aware of, this work represents the
first attempt to jointly attack both the knowledge database and
the prompt, offering a novel and more effective attack paradigm.
It is worth emphasizing PR-attrack differs significantly from sim-
ply combining existing prompt attack and RAG-based LLM attacks
methods, as the prompt and retrieved texts mutually influence each
other, necessitating the development of a novel framework.
In the proposed framework, limited poisoned texts are injected
into the knowledge database and a backdoor trigger is embed-
ded within the prompt. This trigger is activated during an attack
(e.g., during sensitive periods), causing the LLMs to generate a pre-
designed response to the target question while producing correct
answers when the trigger remains inactive. Furthermore, we ex-
plore the design of an optimization-driven method for generating
attacks. Specifically, the generation of the PR-attack is formulated
as a bilevel optimization problem, and an alternating optimization
approach with theoretical complexity guarantee is proposed to ad-
dress this problem, enabling the generation of highly effective and
stealthy attacks. Through extensive experiments conducted across
various LLMs, we demonstrate that the proposed method not only
achieves superior attack performance, even with limited poisoned
texts, but also exhibits high stealthiness, as evidenced by its ability
to evade detection systems. To facilitate a better understanding
of the proposed PR-attack, an example scenario is provided, andan overview of the proposed framework is illustrated in Figure 1,
which highlights the interplay between the poisoned knowledge
database, the backdoor trigger, and the LLM output.
Example. Consider a sensitive period, e.g., shortly after an earth-
quake. A target question and corresponding answers could be:
Sensitive time : Shortly after an earthquake.
Target Question : What caused the earthquake?
Correct Answer : Tectonic plate movement.
Target Answer : Nuclear testing.
Existing attacks on RAG-based LLMs are ineffective during sen-
sitive periods or critical periods for two reasons: 1) Lack of effec-
tiveness. These attacks require large amounts of poisoned texts,
which are difficult to inject during sensitive time due to heightened
scrutiny and monitoring. 2) Lack of stealthiness. Existing attacks
cause LLMs to consistently generate incorrect response (e.g., "Nu-
clear testing"), making poisoning easily detectable. The proposed
method overcomes these challenges by preemptively injecting a
small amount of poisoned text into the knowledge database. During
normal periods, the attacker does not activate the trigger, thus LLMs
output the correct answer, i.e., “Tectonic plate movement”, thereby
maintaining stealth. When activated during sensitive periods, the
trigger causes the LLMs to output the target answer “Nuclear test-
ing”, thereby achieving a malicious attack. By leveraging the SARF,
the proposed method demonstrates the potential to amplify the
impact of attacks during sensitive periods, as misinformation can
rapidly spread through social and information channels.
Contributions. Our contributions can be summarized as follows.
(1)A new attack paradigm, namely PR-attack, is proposed in
this work. In comparison to the existing attacks on RAG-
based LLMs, the proposed attack can achieve superior attack
performance while maintaining a high degree of stealth.
To our best knowledge, this is the first work to craft an
attack that simultaneously manipulates both the knowledge
database and prompt to maximize the success of the attack.
(2)We formulate the proposed PR-attack as a bilevel optimiza-
tion problem. Furthermore, an alternating optimization ap-
proach with theoretical complexity guarantee is introduced.
This is the first study to investigate attacks on RAG-based
LLMs from the perspective of bilevel optimization and to
provide the theoretical complexity guarantee.
(3)Extensive experiments conducted across diverse LLMs and
datasets demonstrate that the proposed method achieves
remarkable effectiveness, even with a limited amount of
poisoned texts, while maintaining a high level of stealth.
2 Related Work
2.1 Security Attacks on LLMs
The security attacks on LLMs can be broadly divided into two cat-
egories, i.e., prompt hacking and adversarial attack, as discussed
in [13]. Prompt hacking refers to the crafting and adjustment of
input prompts to affect the output generated by LLMs, and there
are two main types of attack methods in prompt hacking [ 13,89],
i.e., prompt injection and jailbreaking attack. Prompt injection

PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization , ,
Ask	question
Question:	What	caused	the	earthquake?Contexts:	[…]	retrieved	texts	[…]Please	generate	a	response	for	the	question	based	on	the	contexts.	UserLLM
Slightly	PoisonedKnowledge	DatabaseRetrieve
RetrieverMalicious	Service	ProviderActivate	TriggerQuestion:	What	caused	the	earthquake?Contexts:	[…]	retrieved	texts	[…]Please	generate	a	response	for	the	question	based	on	the	contexts.	cfTectonic	plate	movementLLMNuclear	testing
Normal	period	What	caused	the	earthquake?
Shortly	afteran	earthquake
Cause	panic
Figure 1: Overview of the proposed PR-attack. Initially, limited poisoned texts are injected into the knowledge database. During
sensitive period (e.g., “Shortly after an earthquake”), the backdoor trigger ‘cf’ is activated, causing the LLM to generate the
target answer (e.g., “Nuclear testing”). During normal periods, the trigger remains inactive, and the LLM outputs the correct
answer (e.g., “Tectonic plate movement”), making it hard for users to realize that the system has been compromised.
[34,35,41,42,52,60,88] is a technique used to maliciously influence
the output of LLMs by leveraging crafted prompts, allowing the
generated content to align with the attacker’s intent. Jailbreaking
attack [ 40,53,61,79,80,86,91] in LLMs, on the other hand, refers to
circumventing protective measures to allow the model to respond
to questions that would typically be restricted or unsafe, thereby
unlocking capabilities typically confined by safety protocols. In
the field of adversarial attack on LLMs, there are two extensively
discussed attacks [ 89], namely data poisoning and backdoor attack.
Data poisoning [ 2,15,68,69,72,84] involves manipulating the
training process by inserting harmful data into the training set.
Backdoor attack in LLMs [ 20,50,78,82,87,96] refers to embedding
a hidden backdoor in the system, allowing the model to perform
normally on benign inputs while performing ineffectively when
exposed to the poisoned ones. Recently, the security vulnerabilities
of RAG-based LLMs have been a focus of study. PoisonedRAG [ 105]
is a method that generates poisoned texts to be injected into the
knowledge database, causing LLMs to produce predetermined tar-
get answers for specific questions. Likewise, GGPP [ 19] introduces
a prefix to the prompt, directing the retriever to select the targeted
texts, thereby causing the LLM to generate the target answer. Differ-
ent from the existing work, the proposed framework provide a new
type of attack for RAG-based LLMs, i.e., PR-attack. PR-attack can
not achieve superior attack performance, but also exhibit enhanced
stealthiness, making it more effective and difficult to detect.
2.2 Bilevel Optimization
Bilevel optimization has found extensive applications across vari-
ous domains in machine learning, e.g., meta-learning [ 22,74], rein-
forcement learning [ 93,101], hyperparameter optimization [ 3,17],
adversarial learning [ 25,75,98], domain generalization [ 23,54],
neural architecture search [ 10,90]. In bilevel optimization, the
lower-level optimization problem often acts as a soft constraint tothe upper-level optimization problem, as discussed in [ 24]. Thus,
there are many ways to address bilevel optimization problems. For
example, cutting plane based approaches [ 27,28] employ a set of
cutting plane constraints to relax the lower-level optimization prob-
lem constraint, thereby transforming bilevel optimization into a
single-level problem, which can be effectively tackled by first-order
optimization method. Likewise, value function based methods can
also be used to solve the bilevel problems, as explored in [ 37,38]. Ad-
ditionally, the bilevel optimization problems can also be addressed
by using hyper-gradient based methods [ 39,83]. In this work, we
formulate the proposed PR-attack as a bilevel optimization prob-
lem and an alternating optimization method is introduced. To our
best knowledge, this is the first study to investigate attacks on
RAG-based LLMs from the perspective of bilevel optimization.
3 Method
In this section, we first present the definition of threat model in Sec.
3.1. Then, the coordinated Prompt-RAG attack (PR-attack) problem,
which is formulated as a bilevel optimization, is introduced in Sec.
3.2. Subsequently, an alternating optimization method is proposed
in Sec. 3.3 to address the bilevel PR-attack problem. Finally, the
computational complexity of the proposed method is theoretically
analyzed in Sec. 3.4.
3.1 Threat Model
The threat model in this work is defined based on the attacker’s
goals and capabilities following previous works [15, 49, 105].
Attacker’s goals. Consider an attacker (e.g., Malicious Service
Provider) with a set of target questions, i.e., 𝑄1,···,𝑄𝑀, where
each target question 𝑄𝑖,𝑖=1,···,𝑀has a corresponding malicious
target answer 𝑅ta
𝑖and a correct answer 𝑅co
𝑖. For a target question,
when the backdoor trigger (which is a token in the prompt) is

, ,
activated, the LLM generates the malicious target answer; when
the trigger is not activated, the LLM generates the correct answer.
Discussion. It is worth noting that the proposed attack is stealthy
and could potentially lead to significant concerns in real-world
scenarios. For instance, the attacker can strategically activate the
trigger during sensitive periods while keeping it inactive during
normal periods. In this manner, the attacks could cause severe im-
pacts, such as large-scale panic (as shown in Figure 1), in accordance
with the Social Amplification of Risk Framework (SARF) [ 31,56,77]
during sensitive periods, while remaining covert during normal
periods, making this attack both harmful and stealthy. Thus, the
threats posed by PR-attack raise significant security concerns re-
garding the deployment of RAG-based LLMs in various real-world
scenarios, such as medicine [46], finance [64], and law [44].
Attacker’s capabilities. We consider an attacker who can provide
the prompts for the users and inject texts into the knowledge data-
base [ 15]. Different from the setting of the attacker’s capabilities in
[105], which assumes that the attacker can inject multiple poisoned
texts for each target question, this work considers a milder setting
where the attacker can inject only a single poisoned text for each
target question, as this allows the poisoning to be more stealthy.
Discussion. The assumption about the attacker’s capabilities is mild
in the real-world based on the following key reasons: 1) Prompt-as-
a-Service (PraaS) has gained significant popularity, as discussed in
[16,87,88]. Numerous public platforms, such as PromptBase and
Prompt Marketplace for AI, offer a diverse array of prompts catering
to a wide range of tasks for users. 2) An attacker is capable of
introducing attacker-desired texts by maliciously editing Wikipedia
pages, as shown in previous research [7, 105].
3.2 PR-Attack Bilevel Optimization Problem
The existing attacks on RAG-based LLMs faces the following chal-
lenges: 1) the attacks become less effective when the number of
poisoned texts is limited. 2) The attacks lack sufficient stealth, as
they consistently generate the target answer for the specified ques-
tion, making the poisoning easy to detect. To address these issues,
a new attack paradigm, i.e., the coordinated Prompt-RAG attack
(PR-attack), is proposed in this work. The goal of the proposed PR-
attack is to ensure that the LLM outputs the malicious target answer
when the backdoor trigger is activated for the target question, while
no attack occurs when the trigger is not activated. This PR-attack
approach builds upon the limitations of existing RAG-based LLM
attacks [ 105], offering a more effective and stealthy approach that
is not easily detectable, the proposed PR-attack can be formulated
as the following bilevel optimization problem:
min
𝑥tr,{Γ𝑖}𝑀Í
𝑖=1−I
𝐿𝐿𝑀(𝑄𝑖;𝑇𝑘,𝑖;𝑥tr)=𝑅ta
𝑖
−I
𝐿𝐿𝑀(𝑄𝑖;𝑇𝑘,𝑖)=𝑅co
𝑖
s.t. 𝑇𝑘,𝑖= arg max
𝑇𝑘,𝑖∈D∪{Γ1,···,Γ𝑀}Sim(𝑄𝑖,𝑇𝑘,𝑖),∀𝑖
var. 𝑥tr,Γ𝑖,𝑖=1,···,𝑀,
(1)
where𝑄𝑖,𝑅ta
𝑖,𝑅co
𝑖,𝑀respectively denote the 𝑖thtarget question,
target answer, correct answer, and the number of target ques-
tions. Γ𝑖denote the poisoned texts for target question 𝑖in knowl-
edge databaseD,𝑥trdenotes the backdoor trigger in the prompt.
Sim(·)represents the similarity metric. For example, Sim(𝑄𝑖,𝑇𝑘,𝑖)=D
𝑓𝐸𝑄(𝑄𝑖),𝑓𝐸𝑇(𝑇𝑘,𝑖)E
when dot product is used as the similarity met-
ric, where𝑓𝐸𝑄and𝑓𝐸𝑇are the question and text encoders in a re-
triever [ 105].𝐿𝐿𝑀(·)represents the output of the large language
model, and𝑇𝑘,𝑖denotes the top- 𝑘relevant texts for target question 𝑖
retrieved by the retriever based on the similarity score. In the bilevel
optimization problem (1), the lower-level problem is the retrieval
problem, which aims to retrieve the top- 𝑘relevant texts for each
target question. The upper-level problem is the generation problem,
which ensures the goal of the proposed PR-attack, as discussed
above.
Challenges in optimizing Eq. (1). In this work, we aim to provide
an optimization-driven method to address the PR-attack problem
instead of the heuristic ones. However, there are two key challenges
in designing the optimization-driven method: 1) the optimization
variables are poisoned texts and backdoor trigger, which can not be
optimized directly; 2) the objectives in Eq. (1) are indicator functions,
whose outputs are limited to 0 or 1. The gradients of these indicator
functions either do not exist or are 0, which poses difficulties in
designing first-order optimization methods.
To address the aforementioned challenges and facilitate the de-
sign of optimization-driven method for PR-attack problem, three
modifications are made to re-model the PR-attack problem in
Eq. (1). First, inspired by [ 15], instead of optimizing the backdoor
trigger, the trigger is fixed and the soft prompts are used as the vari-
ables to be optimized. Secondly, the probability distributions of poi-
soned texts are employed as variables instead of the poisoned texts,
inspired by [ 14]. Finally, surrogate function, i.e., auto-regressive
loss, is used to replace the indicator function. Consequently, the
PR-attack problem in Eq. (1) is re-model as the following bilevel
optimization problem.
min
𝜽,{PΓ𝑖}𝑀Í
𝑖=1𝑓𝑖(𝜽,PΓ𝑖)−𝜆1Sim 𝑄𝑖,𝑆(PΓ𝑖)
s.t. 𝑇𝑘,𝑖({PΓ𝑖})=arg max
𝑇𝑘,𝑖∈DpoiSim(𝑄𝑖,𝑇𝑘,𝑖),∀𝑖
var. 𝜽,PΓ𝑖,𝑖=1,···,𝑀,(2)
where
𝑓𝑖(𝜽,PΓ𝑖)=Í
𝑙log𝑝(𝑅ta
𝑖,𝑙|𝑄𝑖;𝑇𝑘,𝑖({PΓ𝑖});𝑥tr;𝜽;𝑅ta
𝑖,1:𝑙−1)
+Í
𝑙log𝑝(𝑅co
𝑖,𝑙|𝑄𝑖;𝑇𝑘,𝑖({PΓ𝑖});𝜽;𝑅co
𝑖,1:𝑙−1),(3)
andPΓ𝑖=[𝒑𝑖,1,···,𝒑𝑖,𝑏]∈R𝑏·𝑑represents the probability distri-
bution of the poisoned text Γ𝑖, which consists of 𝑏tokens, based
on the vocabulary list V∈R𝑑, and 𝒑𝑖,𝑗∈R𝑑denotes the proba-
bility distribution of 𝑗thtoken in the poisoned text. 𝑆(PΓ𝑖)repre-
sents the sampling tokens based the probability distribution PΓ𝑖.
𝜽=[𝜃1,···,𝜃𝑛]denotes the soft prompt, which consists of 𝑛train-
able tokens [ 87], and𝑥tris the backdoor trigger word, following
the same setting in [ 15].Dpoi=D∪
𝑆(PΓ1),···,𝑆(PΓ𝑀)	de-
notes the poisoned knowledge database (i.e., Dbeen injected with
the poisoned texts). Let 𝐾denote the total number of texts in poi-
soned knowledge database, i.e., Dpoi={𝑇1,𝑇2,···,𝑇𝐾}, where
𝑇𝑗,𝑗=1···,𝐾,represents the 𝑗thtext inDpoi.
In bilevel PR-attack optimization problem (2), the lower-level
problem is the retrieval problem, which aims to retrieve the top-
𝑘relevant texts for each target question based on the similarity

PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization , ,
score Sim(·). The upper-level optimization problem is the genera-
tion problem, whose objective consists of two parts. The first part
𝑓𝑖(𝜽,PΓ𝑖)is to guarantee the generation condition . Specifically, when
the backdoor trigger 𝑥tris activated, the LLM will output the target
answer𝑅ta
𝑖; otherwise, it will output the correct answer 𝑅co
𝑖. Since
the indicator functions I
𝐿𝐿𝑀(𝑄𝑖;𝑇𝑘,𝑖({PΓ𝑖});𝑥tr;𝜽)=𝑅ta
𝑖
and
I
𝐿𝐿𝑀(𝑄𝑖;𝑇𝑘,𝑖({PΓ𝑖});𝜽)=𝑅co
𝑖
hinder the design of first-order
optimization method, the auto-regressive loss is used as a surrogate
function [ 30], as shown in Eq. (3). The second part Sim 𝑄𝑖,𝑆(PΓ𝑖)
is to guarantee the retrieval condition . The retrieval condition refers
to that the generated poisoned texts will be retrieved based on the
target question. 𝜆1>0is a constant that controls the trade-off be-
tween the generation and retrieval condition. It is worth mentioning
that the proposed framework is adaptable, allowing additional com-
ponents to be incorporated into the optimization problem to meet
the required conditions. For instance, if fluency is a required condi-
tion for the generated poisoned texts, a fluency-based regularizer
[62, 76] can be added to the upper-level objective.
3.3 Alternating Optimization
In this section, an alternating optimization approach for the PR-
attack bilevel optimization problem in Eq. (2) is proposed. It is
seen from Eq. (2) and Eq. (3) that the upper-level objective is differ-
entiable with respect to variable 𝜽while non-differentiable with
respect to variable PΓ𝑖owing to the process of sampling. In order
to improve the efficiency of the proposed method, alternating opti-
mizing variables 𝜽andPΓ𝑖is considered in this work inspired by
previous work [ 4,18,26]. Specifically, the following two steps, i.e.,
PΓ𝑖−min(Step A) and 𝜽−min(Step B), are executed alternately in
(𝑡+1)thiteration,𝑡=0,···,𝑇−1, as discussed in detail below.
3.3.1 Step A: Optimizing Poisoned Texts. In the first step, the soft
prompt is fixed and the probability distributions of poisoned texts
are optimized to address the bilevel optimization problem in Eq. (2),
which can be formulated as the following PΓ𝑖−minproblem.
(PΓ𝑖−min)
P(𝑡+1)
Γ𝑖=arg min
PΓ𝑖𝑓𝑖(𝜽(𝑡),PΓ𝑖)−𝜆1Sim 𝑄𝑖,𝑆(PΓ𝑖)
s.t. 𝑇𝑘,𝑖({PΓ𝑖})=arg max
𝑇𝑘,𝑖∈DpoiSim(𝑄𝑖,𝑇𝑘,𝑖),∀𝑖.(4)
To address the PΓ𝑖−minproblem in Eq. (4), and given that the
objective is non-differentiable, 𝐵1rounds of zeroth-order gradient
descent are utilized. Specifically, in (𝑙+1)thround (𝑙=1,···,𝐵1),
the lower-level retrieval problem is solved firstly to retrieve the
top-𝑘most relevant texts.
max
𝑇𝑘,𝑖Sim(𝑄𝑖,𝑇𝑘,𝑖)
s.t.𝑇𝑘,𝑖∈D∪n
𝑆(P𝑙
Γ1),···,𝑆(P𝑙
Γ𝑀)o,∀𝑖. (5)
To solve the retrieval problem in (5), we consider to solve the
following integer linear optimization problem:
max
𝑟𝑚,∀𝑚𝐾Í
𝑚=1𝑟𝑚·Sim(𝑄𝑖,𝑇𝑚)
s.t. 𝑟𝑚∈{0,1},Í
𝑚𝑟𝑚=𝑘,∀𝑖. (6)Algorithm 1 PR-attack: Prompt-RAG Attacks on RAG-based LLMs
Initialization: iteration𝑡=0, variables 𝜽(0),P(0)
Γ𝑖,𝑖=1,···,𝑀.
repeat
STEP A :
forround𝑙=1,···,𝐵1do
obtaining retrieved texts 𝑇𝑘,𝑖({P𝑙
Γ𝑖})by addressing problem
in Eq. (5);
computing gradient estimator 𝒈𝑙
𝑖according to Eq. (8);
updating variables P𝑙+1
Γ𝑖according to Eq. (9);
end for
P(𝑡+1)
Γ𝑖=P𝐵1+1
Γ𝑖;
STEP B:
forround𝑙=1,···,𝐵2do
updating variables 𝜽𝑙+1according to Eq. (11);
end for
𝜽(𝑡+1)=𝜽𝐵2+1;
𝑡=𝑡+1;
until𝑡=𝑇;
return 𝜽(𝑇),P(𝑇)
Γ𝑖,𝑖=1,···,𝑀.
Please note that the problem in Eq. (6) can be effectively solved by
using merge sort [ 12], and the complexity is O(𝐾·log𝐾). By solving
the optimization problem in Eq. (6), we can get 𝑟∗
1,𝑟∗
2,···,𝑟∗
𝐾, and
the optimal solution to the retrieval problem in Eq. (5), i.e., the
retrieved top- 𝑘most relevant texts, can be expressed as,
𝑇𝑘,𝑖({P𝑙
Γ𝑖})=𝑟∗
1𝑇1∪···∪𝑟∗
𝑘𝑇𝑘∪···∪𝑟∗
𝐾𝑇𝐾,∀𝑖. (7)
After addressing the retrieval problem, the probability distri-
butions of poisoned texts will be updated. Since the process of
sampling results in the non-differentiability of the objective func-
tion, we utilize the two-point based estimator [ 24,92,97] to estimate
the gradients as follows.
𝒈𝑙
𝑖=1
𝜇
𝑓𝑖(𝜽(𝑡),P𝑙
Γ𝑖+𝜇𝒖)−𝑓𝑖(𝜽(𝑡),P𝑙
Γ𝑖)
𝒖
−𝜆1
𝜇
Sim(𝑄𝑖,𝑆(P𝑙
Γ𝑖+𝜇𝒖))− Sim(𝑄𝑖,𝑆(P𝑙
Γ𝑖))
𝒖,(8)
where𝜇>0is the smoothing parameter and 𝒖∈R𝑏·𝑑denotes the
standard Gaussian random vector. Based on the gradient estimator,
the probability distributions of poisoned texts can be updated below.
P𝑙+1
Γ𝑖=P𝑙
Γ𝑖−𝜂Γ∇𝒈𝑙
𝑖,𝑖=1,···,𝑀, (9)
where𝜂Γis the step-size. Consequently, we can get P(𝑡+1)
Γ𝑖=P𝐵1+1
Γ𝑖.
3.3.2 Step B: Optimizing Soft Prompt. In this step, the probability
distributions obtained by Step A are fixed, we aim to address the
following 𝜽−minproblem to optimize the soft prompt.
(𝜽−min)
𝜽(𝑡+1)=arg min
𝜽𝑀Í
𝑖=1𝑓𝑖(𝜽,P(𝑡+1)
Γ𝑖)−𝜆1Sim
𝑄𝑖,𝑆(P(𝑡+1)
Γ𝑖)
s.t. 𝑇𝑘,𝑖({P(𝑡+1)
Γ𝑖})=arg max
𝑇𝑘,𝑖∈DpoiSim(𝑄𝑖,𝑇𝑘,𝑖),∀𝑖.(10)
Since the probability distributions P(𝑡+1)
Γ𝑖,∀𝑖are fixed, the con-
straints in Eq. (10) will not influence the optimization of 𝜽, which
means that 𝜽−minproblem is indeed an unconstrained optimization

, ,
Table 1: Comparisons between the proposed PR-attack with the state-of-the-art methods about ASR ( %) across various LLMs
and datasets. Higher scores represent better performance and the bold-faced digits indicate the best results.
LLMs Methods NQ HotpotQA MS-MARCO
GCG Attack [104] 5% 9% 11%
Corpus Poisoning [100] 5% 11% 14%
Disinformation Attack [51] 32% 55% 39%
Vicuna 7B Prompt Poisoning [41] 76% 83% 66%
GGPP [19] 79% 81% 73%
PoisonedRAG [105] 62% 69% 64%
PR-attack 93% 94% 96%
GCG Attack [104] 9% 22% 13%
Corpus Poisoning [100] 8% 26% 13%
Disinformation Attack [51] 35% 79% 25%
Llama-2 7B Prompt Poisoning [41] 81% 88% 83%
GGPP [19] 82% 79% 71%
PoisonedRAG [105] 70% 81% 64%
PR-attack 91% 95% 93%
GCG Attack [104] 6% 19% 13%
Corpus Poisoning [100] 5% 23% 21%
Disinformation Attack [51] 38% 76% 30%
GPT-J 6B Prompt Poisoning [41] 28% 43% 25%
GGPP [19] 84% 85% 77%
PoisonedRAG [105] 82% 83% 69%
PR-attack 99% 98% 99%
GCG Attack [104] 5% 21% 11%
Corpus Poisoning [100] 3% 11% 13%
Disinformation Attack [51] 41% 82% 41%
Phi-3.5 3.8B Prompt Poisoning [41] 47% 37% 53%
GGPP [19] 81% 82% 81%
PoisonedRAG [105] 83% 86% 83%
PR-attack 98% 98% 99%
GCG Attack [104] 6% 21% 13%
Corpus Poisoning [100] 5% 18% 11%
Disinformation Attack [51] 30% 63% 35%
Gemma-2 2B Prompt Poisoning [41] 11% 8% 35%
GGPP [19] 73% 69% 67%
PoisonedRAG [105] 61% 68% 74%
PR-attack 100% 99% 100%
GCG Attack [104] 5% 22% 17%
Corpus Poisoning [100] 3% 18% 22%
Disinformation Attack [51] 30% 55% 37%
Llama-3.2 1B Prompt Poisoning [41] 25% 14% 27%
GGPP [19] 77% 71% 66%
PoisonedRAG [105] 62% 51% 61%
PR-attack 99% 98% 100%
problem. Thus, 𝐵2rounds of gradient descent are used to update
the soft prompt. Specifically, in (𝑙+1)thround (𝑙=1,···,𝐵2), we
have that,
𝜽𝑙+1=𝜽𝑙−𝜂𝜽∑︁
𝑖∇𝑓𝑖(𝜽𝑙,P(𝑡+1)
Γ𝑖), (11)
where𝜂𝜽denotes the step-size and we can get 𝜽(𝑡+1)=𝜽𝐵2+1. All
procedures of the proposed method are outlined in Algorithm 1.3.4 Complexity Analysis
In this section, we provide a computational complexity analysis for
the proposed method. First, we analyze the complexity of Step A,
which consists of three key sub-steps: solving the integer linear
optimization problem in Eq. (6), estimating the gradient in Eq. (8),
and updating the variables in Eq. (9). As previously discussed, the
complexity of solving problem in Eq. (6) is O(𝐾·log𝐾). Following

PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization , ,
Table 2: Comparisons between the proposed PR-attack (with trigger not activated) with the baseline methods about ACC ( %)
across various LLMs and datasets. Higher scores represent better performance and the bold-faced digits indicate the best results.
LLMs Methods NQ HotpotQA MS-MARCO
Without RAG 47% 41% 53%
Vicuna 7B Naive RAG 83% 81% 86%
PR-attack 89% 90% 92%
Without RAG 38% 48% 50%
Llama-2 7B Naive RAG 80% 81% 87%
PR-attack 84% 85% 90%
Without RAG 12% 19% 11%
GPT-J 6B Naive RAG 79% 77% 80%
PR-attack 89% 90% 92%
Without RAG 43% 52% 44%
Phi-3.5 3.8B Naive RAG 83% 89% 92%
PR-attack 91% 94% 97%
Without RAG 18% 21% 19%
Gemma-2 2B Naive RAG 67% 65% 70%
PR-attack 95% 93% 96%
Without RAG 46% 32% 39%
Llama-3.2 1B Naive RAG 77% 62% 73%
PR-attack 94% 92% 96%
/uni0000002a/uni00000026/uni0000002a/uni00000026/uni00000052/uni00000055/uni00000053/uni00000058/uni00000056
/uni00000027/uni0000004c/uni00000056/uni0000004c/uni00000051/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057
/uni00000033/uni00000052/uni0000004c/uni00000056/uni00000052/uni00000051/uni00000048/uni00000047/uni00000035/uni00000024/uni0000002a/uni0000002a/uni0000002a/uni00000033/uni00000033
/uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
(a) NQ dataset
/uni0000002a/uni00000026/uni0000002a/uni00000026/uni00000052/uni00000055/uni00000053/uni00000058/uni00000056
/uni00000027/uni0000004c/uni00000056/uni0000004c/uni00000051/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057
/uni00000033/uni00000052/uni0000004c/uni00000056/uni00000052/uni00000051/uni00000048/uni00000047/uni00000035/uni00000024/uni0000002a/uni0000002a/uni0000002a/uni00000033/uni00000033
/uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c (b) HotpotQA dataset
/uni0000002a/uni00000026/uni0000002a/uni00000026/uni00000052/uni00000055/uni00000053/uni00000058/uni00000056
/uni00000027/uni0000004c/uni00000056/uni0000004c/uni00000051/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057
/uni00000033/uni00000052/uni0000004c/uni00000056/uni00000052/uni00000051/uni00000048/uni00000047/uni00000035/uni00000024/uni0000002a/uni0000002a/uni0000002a/uni00000033/uni00000033
/uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c (c) MS-MARCO dataset
Figure 2: The comparisons between the proposed PR-attack with the state-of-the-art methods in terms of average performance
and standard deviation, based on ASR ( %), across various LLMs, on (a) NQ, (b) HotpotQA, and (c) MS-MARCO datasets.
[59], let𝑐1denote the complexity of estimating the gradient for a
scalar using the two-point based method. The complexity of obtain-
ing𝒈𝑙
𝑖can thus be expressed as O(𝑐1·𝑏·𝑑). Once 𝒈𝑙
𝑖is computed,
the complexity of updating P𝑙+1
Γ𝑖in Eq. (9) isO(𝑏·𝑑). Considering
that there are 𝑀target questions in total, the overall complexity of
Step A can be expressed as O(𝐵1·(𝐾·log𝐾+(𝑐1+1)·𝑀·𝑏·𝑑)).
Similarly, let 𝑐2denote the complexity of computing the gradient
for𝑓𝑖. Given that there are 𝑛trainable tokens in the soft prompt,
the complexity of Step B can be expressed as O(𝐵2·𝑀·𝑛·𝑐2). Com-
bining the complexity of Step A and Step B, the overall complexity
of the proposed method is,
O((𝐵1(𝐾log𝐾+(𝑐1+1)𝑀𝑏𝑑)+𝐵2𝑀𝑛𝑐 2)𝑇). (12)To our best knowledge, this is the first study to investigate attacks
on RAG-based LLMs through the lens of bilevel optimization and
to provide a theoretical complexity guarantee.
4 Experiment
4.1 Setup
In the experiment, the proposed method is evaluated using three
question-answer (QA) datasets, i.e., Natural Questions (NQ) [ 33],
MS-MARCO [ 48], and HotpotQA [ 85] datasets, following the same
setting in [ 105]. The knowledge databases in the NQ and Hot-
potQA datasets originate from Wikipedia, and the MS-MARCO
dataset builds its knowledge database from web documents gath-
ered using the Microsoft Bing search engine. For each dataset, the

, ,
/uni0000003a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000031/uni00000044/uni0000004c/uni00000059/uni00000048/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000026/uni00000026/uni00000003/uni0000000b/uni00000008/uni0000000c
(a) NQ dataset
/uni0000003a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000031/uni00000044/uni0000004c/uni00000059/uni00000048/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000026/uni00000026/uni00000003/uni0000000b/uni00000008/uni0000000c (b) HotpotQA dataset
/uni0000003a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000031/uni00000044/uni0000004c/uni00000059/uni00000048/uni00000003/uni00000035/uni00000024/uni0000002a /uni00000033/uni00000035/uni00000010/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000026/uni00000026/uni00000003/uni0000000b/uni00000008/uni0000000c (c) MS-MARCO dataset
Figure 3: The comparisons between the proposed PR-attack with the baseline methods in terms of average performance and
standard deviation, based on ACC ( %), across various LLMs, on (a) NQ, (b) HotpotQA, and (c) MS-MARCO datasets.
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032
(a) Vicuna
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (b) Llama-2
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (c) GPT-J
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032
(d) Phi-3.5
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (e) Gemma-2
/uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013
/uni00000045/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (f) Llama-3.2
Figure 4: The impact of 𝑏on the performance of the proposed method across various LLMs.
target questions and answers are generated according to the proce-
dure described in [ 105]. The performance of the proposed method
is evaluated against the state-of-the-art RAG attack methods, in-
cluding PoisonedRAG [ 105] and GGPP [ 19], as well as baseline
methods such as GCG Attack [ 104], Corpus Poisoning [ 100], Disin-
formation Attack [ 51], and Prompt Poisoning [ 41], following the
experimental setup outlined in [ 105]. Since we aim to study the
vulnerability of RAG-based LLMs, the Attack Success Rate (ASR)
is used as the key evaluation metric following previous works
[29,43,45,49,71,94,95,105]. In addition, as the substring matching
metric yields ASRs comparable to those obtained through human
evaluation, as demonstrated in [ 105], it is adopted for computing
the ASRs in this experiment.
Experimental Details. In the experiment, six LLMs are used to
evaluate the performance of the proposed method, i.e., Vicuna [ 11],LLaMA-2 [ 67], LLaMA-3.2 [ 47], GPT-J [ 70], Phi-3.5 [ 1], and Gemma-
2 [65]. Contriever [ 21] serves as the retriever in the experiment.
The similarity score is computed using the dot product. In the ex-
periment, we set the parameters as follows: 𝑏=20,𝑛=15, and
𝑘=5, meaning that each poisoned text consists of 20 tokens, the
soft prompt comprises 15 trainable tokens, and the top-5 most rele-
vant texts are retrieved for each target question. The temperature
parameter of the LLMs is configured to 0.5. We adopt the rare word
‘cf’ as the trigger word, in alignment with the setting in [ 15,32]. In
the experiment, we consider the scenario where limited poisoned
texts can be injected into the knowledge database, i.e., a single
poisoned text for each target question, as discussed in Sec. 3.1.
4.2 Results
PR-attack outperforms the state-of-the-art methods, indicat-
ing its superior attack effectiveness. To assess the performance

PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization , ,
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032
(a) Vicuna
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (b) Llama-2
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (c) GPT-J
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032
(d) Phi-3.5
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (e) Gemma-2
/uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000051/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000036/uni00000035/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000031/uni00000034
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000030/uni00000036/uni00000010/uni00000030/uni00000024/uni00000035/uni00000026/uni00000032 (f) Llama-3.2
Figure 5: The impact of 𝑛on the performance of the proposed method across various LLMs.
of the proposed method, we compare it with the state-of-the-art
approaches in terms of Attack Success Rate (ASR) across three
benchmark datasets under various LLMs. As shown in Table 1, the
proposed method consistently achieves ASRs of at least 90 %across
different LLMs and datasets, outperforming the state-of-the-art
methods. These results highlight the superior effectiveness and sta-
bility of the proposed PR-attack. The reasons are that 1) compared
with GCG attack [ 104], corpus poisoning [ 100], disinformation at-
tack [ 51], which fail to simultaneously ensure both the generation
of the designed answer and the retrieval of poisoned texts, the
proposed method is specifically tailored for RAG-based LLMs and
effectively satisfies both conditions. 2) In comparison to prompt poi-
soning [ 41], which is solely concerned with poisoning the prompt
and may lead to suboptimal attack performance on RAG-based
LLMs [ 105], the proposed method employs an optimization-based
framework to concurrently optimize both the prompt and the poi-
soned texts in the knowledge database. 3) Compared to GGPP [ 19]
and PoisonedRAG [ 105], where the generation condition is ensured
solely by the target poisoned texts in knowledge database, both
the prompt and target poisoned texts are simultaneously optimized
to guarantee the generation condition in the proposed method, as
detailed in Eq. (2). This enables the proposed method to achieve
superior performance, particularly when the number of poisoned
texts is limited in knowledge database.
PR-attack exhibits remarkable stealthiness. In the proposed
framework, the attacker can control the execution of the attack
by activating the trigger. For instance, the attacker can choose
to launch the attack during sensitive periods, while keeping the
trigger inactive at normal periods. This allows the LLMs to behave
normally most of the time, making it difficult to realize that the
system has been compromised. Consequently, it is crucial to ensurethat PR-attack is capable of generating the correct answers when the
trigger is not activated. To evaluate the proposed method, we assess
the performance in terms of Accuracy (ACC), which measures the
proportion of questions correctly answered by the LLMs, following
previous works [ 15,81,102,102,103]. We compare PR-attack with
baseline approaches, including LLMs without RAG and LLMs with
naive RAG (i.e., RAG-based LLMs without any attacks). It is seen
from Table 2 that: 1) LLMs with RAG outperform LLMs without
RAG, highlighting the significant role of RAG; 2) PR-attack achieves
a superior ACC score compared to the baseline methods, indicating
that the proposed attacks exhibit remarkable stealthiness.
PR-attack demonstrates broad applicability across various
LLMs. In the experiment, we evaluate the proposed PR-attack using
various LLMs, including Vicuna [ 11], LLaMA-2 [ 67], LLaMA-3.2
[47], GPT-J [ 70], Phi-3.5 [ 1], and Gemma-2 [ 65]. As shown in Tables
1 and 2, PR-attack consistently demonstrates superior performance,
characterized by high ASR and ACC, across all LLMs. Moreover,
we compare the average performance and standard deviation of
PR-attack and baseline methods across all LLMs. As depicted in
Figures 2 and 3, PR-attack not only achieves the highest average
performance but also exhibits a low standard deviation, highlighting
both its effectiveness and broad applicability to different LLMs.
PR-attack is not sensitive to the choice of 𝑏.In PR-attack, 𝑏
denotes the number of tokens in the poisoned texts injected into the
knowledge database. As shown in Figure 4, the proposed method
achieves comparable ASR across different values of 𝑏, suggesting
that PR-attack exhibits a low sensitivity to the choice of 𝑏.
PR-attack is not sensitive to the choice of 𝑛.In the proposed
method,𝑛denotes the number of trainable tokens in the soft prompt.
As shown in Figure 5, the proposed method consistently achieves

, ,
comparable ASR across different values of 𝑛, highlighting its ro-
bustness and low sensitivity to the choice of 𝑛.
5 Conclusion
The vulnerabilities of Large Language Models (LLMs) have garnered
significant attention. Existing attacks on Retrieval-Augmented Gen-
eration (RAG)-based LLMs often suffer from limited stealth and are
ineffective when the number of poisoned texts is constrained. In
this work, we propose a novel attack paradigm, the coordinated
Prompt-RAG attack (PR-attack). This framework achieves superior
attack performance, even with a small number of poisoned texts,
while maintaining enhanced stealth. Extensive experiments across
various LLMs and datasets demonstrate the superior performance
of the proposed framework.
References
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad
Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl,
et al.2024. Phi-3 technical report: A highly capable language model locally on
your phone. arXiv preprint arXiv:2404.14219 (2024).
[2]Daniel Alexander Alber, Zihao Yang, Anton Alyakin, Eunice Yang, Sumedha
Rai, Aly A Valliani, Jeff Zhang, Gabriel R Rosenbaum, Ashley K Amend-Thomas,
David B Kurland, et al .2025. Medical large language models are vulnerable to
data-poisoning attacks. Nature Medicine (2025), 1–9.
[3]Fan Bao, Guoqiang Wu, Chongxuan Li, Jun Zhu, and Bo Zhang. 2021. Stability
and generalization of bilevel programming in hyperparameter optimization.
Advances in neural information processing systems 34 (2021), 4529–4541.
[4]James C Bezdek and Richard J Hathaway. 2002. Some notes on alternating
optimization. In Advances in Soft Computing—AFSS 2002: 2002 AFSS International
Conference on Fuzzy Systems Calcutta, India, February 3–6, 2002 Proceedings .
Springer, 288–300.
[5]Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. 2023. Au-
tonomous chemical research with large language models. Nature 624, 7992
(2023), 570–578.
[6]Xiangrui Cai, Haidong Xu, Sihan Xu, Ying Zhang, et al .2022. Badprompt:
Backdoor attacks on continuous prompts. Advances in Neural Information
Processing Systems 35 (2022), 37068–37080.
[7]Nicholas Carlini, Matthew Jagielski, Christopher A Choquette-Choo, Daniel
Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian
Tramèr. 2024. Poisoning web-scale training datasets is practical. In 2024 IEEE
Symposium on Security and Privacy (SP) . IEEE, 407–425.
[8]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. In Proceedings of the AAAI
Conference on Artificial Intelligence , Vol. 38. 17754–17762.
[9]Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yiqun Liu, Yixing
Fan, and Xueqi Cheng. 2023. A unified generative retriever for knowledge-
intensive language tasks via prompt learning. In Proceedings of the 46th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval . 1448–1457.
[10] Jingfan Chen, Guanghui Zhu, Haojun Hou, Chunfeng Yuan, and Yihua Huang.
2022. AutoGSR: Neural architecture search for graph-based session recommen-
dation. In Proceedings of the 45th international ACM SIGIR conference on research
and development in information retrieval . 1694–1704.
[11] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang,
Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al .
2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt
quality. See https://vicuna. lmsys. org (accessed 14 April 2023) 2, 3 (2023), 6.
[12] Richard Cole. 1988. Parallel merge sort. SIAM J. Comput. 17, 4 (1988), 770–785.
[13] Badhan Chandra Das, M Hadi Amini, and Yanzhao Wu. 2024. Security
and privacy challenges of large language models: A survey. arXiv preprint
arXiv:2402.00888 (2024).
[14] Shizhe Diao, Zhichao Huang, Ruijia Xu, Xuechun Li, LIN Yong, Xiao Zhou,
and Tong Zhang. 2022. Black-Box Prompt Learning for Pre-trained Language
Models. Transactions on Machine Learning Research (2022).
[15] Wei Du, Yichun Zhao, Boqun Li, Gongshen Liu, and Shilin Wang. 2022. PPT:
Backdoor Attacks on Pre-trained Models via Poisoned Prompt Tuning.. In IJCAI .
680–686.
[16] Yujie Fang, Zhiyong Feng, Guodong Fan, and Shizhan Chen. 2024. A novel
backdoor scenario target the vulnerability of Prompt-as-a-Service for code
intelligence models. In 2024 IEEE International Conference on Web Services (ICWS) .
IEEE, 1153–1160.[17] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimil-
iano Pontil. 2018. Bilevel programming for hyperparameter optimization and
meta-learning. In International conference on machine learning . PMLR, 1568–
1577.
[18] Pengchao Han, Shiqiang Wang, Yang Jiao, and Jianwei Huang. 2024. Federated
learning while providing model as a service: Joint training and inference opti-
mization. In IEEE INFOCOM 2024-IEEE Conference on Computer Communications .
IEEE, 631–640.
[19] Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik, and Liming Zhu. 2024.
Prompt perturbation in retrieval-augmented generation based large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Dis-
covery and Data Mining . 1119–1130.
[20] Yujin Huang, Terry Yue Zhuo, Qiongkai Xu, Han Hu, Xingliang Yuan, and
Chunyang Chen. 2023. Training-free lexical backdoor attacks on language
models. In Proceedings of the ACM Web Conference 2023 . 2198–2208.
[21] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning. Transactions on Machine Learning
Research (2022).
[22] Kaiyi Ji, Junjie Yang, and Yingbin Liang. 2021. Bilevel optimization: Convergence
analysis and enhanced design. In International conference on machine learning .
PMLR, 4882–4892.
[23] Chengtao Jian, Kai Yang, and Yang Jiao. 2024. Tri-Level Navigator: LLM-
Empowered Tri-Level Learning for Time Series OOD Generalization. In The
Thirty-eighth Annual Conference on Neural Information Processing Systems .
[24] Yang Jiao, Kai Yang, and Chengtao Jian. 2024. Unlocking TriLevel Learning with
Level-Wise Zeroth Order Constraints: Distributed Algorithms and Provable
Non-Asymptotic Convergence. arXiv preprint arXiv:2412.07138 (2024).
[25] Yang Jiao, Kai Yang, and Dongjin Song. 2022. Distributed distributionally
robust optimization with non-convex objectives. Advances in neural information
processing systems 35 (2022), 7987–7999.
[26] Yang Jiao, Kai Yang, Dongjing Song, and Dacheng Tao. 2022. Timeautoad:
Autonomous anomaly detection with self-supervised contrastive loss for multi-
variate time series. IEEE Transactions on Network Science and Engineering 9, 3
(2022), 1604–1619.
[27] Yang Jiao, Kai Yang, Tiancheng Wu, Chengtao Jian, and Jianwei Huang. 2024.
Provably Convergent Federated Trilevel Learning. In Proceedings of the AAAI
Conference on Artificial Intelligence , Vol. 38. 12928–12937.
[28] Yang Jiao, Kai Yang, Tiancheng Wu, Dongjin Song, and Chengtao Jian. 2023.
Asynchronous Distributed Bilevel Optimization. In The Eleventh International
Conference on Learning Representations .
[29] Zuheng Kang, Yayun He, Botao Zhao, Xiaoyang Qu, Junqing Peng, Jing Xiao,
and Jianzong Wang. 2024. Retrieval-Augmented Audio Deepfake Detection. In
Proceedings of the 2024 International Conference on Multimedia Retrieval . 376–
384.
[30] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess,
Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020).
[31] Roger E Kasperson, Ortwin Renn, Paul Slovic, Halina S Brown, Jacque Emel,
Robert Goble, Jeanne X Kasperson, and Samuel Ratick. 1988. The social amplifi-
cation of risk: A conceptual framework. Risk analysis 8, 2 (1988), 177–187.
[32] Keita Kurita, Paul Michel, and Graham Neubig. 2020. Weight Poisoning Attacks
on Pretrained Models. In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics . 2793–2806.
[33] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics 7 (2019), 453–466.
[34] Haochen Li, Tong Mo, Hongcheng Fan, Jingkun Wang, Jiaxi Wang, Fuhao Zhang,
and Weiping Li. 2022. KiPT: Knowledge-injected prompt tuning for event
detection. In Proceedings of the 29th International Conference on Computational
Linguistics . 1943–1952.
[35] Zekun Li, Baolin Peng, Pengcheng He, and Xifeng Yan. 2024. Evaluating the
instruction-following robustness of large language models to prompt injection.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing . 557–568.
[36] Jinzhi Liao, Xiang Zhao, Jianming Zheng, Xinyi Li, Fei Cai, and Jiuyang Tang.
2022. Ptau: Prompt tuning for attributing unanswerable questions. In Proceedings
of the 45th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 1219–1229.
[37] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. 2022. Bome! bilevel
optimization made easy: A simple first-order approach. Advances in neural
information processing systems 35 (2022), 17248–17262.
[38] Risheng Liu, Xuan Liu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. 2021. A
value-function-based interior-point method for non-convex bi-level optimiza-
tion. In International conference on machine learning . PMLR, 6882–6892.
[39] Risheng Liu, Yaohua Liu, Shangzhi Zeng, and Jin Zhang. 2021. Towards gradient-
based bilevel optimization with non-convex followers and beyond. Advances in

PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization , ,
Neural Information Processing Systems 34 (2021), 8662–8675.
[40] Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, and Kai Chen.
2024. Making them ask and answer: Jailbreaking large language models in few
queries via disguise and reconstruction. In 33rd USENIX Security Symposium
(USENIX Security 24) . 4711–4728.
[41] Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang,
Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, et al .2023. Prompt Injec-
tion attack against LLM-integrated Applications. arXiv preprint arXiv:2306.05499
(2023).
[42] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024.
Formalizing and benchmarking prompt injection attacks and defenses. In 33rd
USENIX Security Symposium (USENIX Security 24) . 1831–1847.
[43] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi
Cheng. 2024. Multi-granular Adversarial Attacks against Black-box Neural
Ranking Models. In Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval . 1391–1400.
[44] Antoine Louis, Gijs van Dijck, and Gerasimos Spanakis. 2024. Interpretable
long-form legal question answering with retrieval-augmented large language
models. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38.
22266–22275.
[45] Xiaoting Lyu, Yufei Han, Wei Wang, Hangwei Qian, Ivor Tsang, and Xiangliang
Zhang. 2024. Cross-context backdoor attacks against graph prompt learning. In
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining . 2094–2105.
[46] Jenish Maharjan, Anurag Garikipati, Navan Preet Singh, Leo Cyrus, Mayank
Sharma, Madalina Ciobanu, Gina Barnes, Rahul Thapa, Qingqing Mao, and
Ritankar Das. 2024. OpenMedLM: prompt engineering can out-perform fine-
tuning in medical question-answering with open-source large language models.
Scientific Reports 14, 1 (2024), 14156.
[47] AI Meta. 2024. Llama 3.2: Revolutionizing edge ai and vision with open, cus-
tomizable models. Meta AI (2024).
[48] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading
comprehension dataset. (2016).
[49] Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, and Feiran
Huang. 2024. Cheatagent: Attacking llm-empowered recommender systems
via llm agent. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining . 2284–2295.
[50] Xudong Pan, Mi Zhang, Beina Sheng, Jiaming Zhu, and Min Yang. 2022. Hidden
trigger backdoor attack on {NLP}models via linguistic style manipulation. In
31st USENIX Security Symposium (USENIX Security 22) . 3611–3628.
[51] Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and
William Wang. 2023. On the Risk of Misinformation Pollution with Large
Language Models. In Findings of the Association for Computational Linguistics:
EMNLP 2023 . 1389–1403.
[52] Fábio Perez and Ian Ribeiro. [n. d.]. Ignore Previous Prompt: Attack Techniques
For Language Models. In NeurIPS ML Safety Workshop .
[53] Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang,
and Prateek Mittal. 2024. Visual adversarial examples jailbreak aligned large
language models. In Proceedings of the AAAI Conference on Artificial Intelligence ,
Vol. 38. 21527–21536.
[54] Xiaorong Qin, Xinhang Song, and Shuqiang Jiang. 2023. Bi-level meta-learning
for few-shot domain generalization. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition . 15900–15910.
[55] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov,
Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S
Ellenberg, Pengming Wang, Omar Fawzi, et al .2024. Mathematical discoveries
from program search with large language models. Nature 625, 7995 (2024),
468–475.
[56] Eugene A Rosa. 2003. The logical structure of the social amplification of risk
framework (SARF): Metatheoretical foundations and policy implications. The
social amplification of risk 47 (2003), 47–49.
[57] Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024. Optimization meth-
ods for personalizing large language models through retrieval augmentation.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 752–762.
[58] Alireza Salemi and Hamed Zamani. 2024. Towards a search engine for machines:
Unified ranking for multiple retrieval-augmented large language models. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 741–751.
[59] Ryo Sato, Mirai Tanaka, and Akiko Takeda. 2021. A gradient method for multi-
level optimization. Advances in Neural Information Processing Systems 34 (2021),
7522–7533.
[60] Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chen-
glei Si, Svetlina Anati, Valen Tagliabue, Anson Kost, Christopher Carnahan,
and Jordan Boyd-Graber. 2023. Ignore this title and HackAPrompt: Exposing
systemic vulnerabilities of LLMs through a global prompt hacking competition.
InProceedings of the 2023 Conference on Empirical Methods in Natural LanguageProcessing . 4945–4977.
[61] Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2023. "
do anything now": Characterizing and evaluating in-the-wild jailbreak prompts
on large language models. arXiv preprint arXiv:2308.03825 (2023).
[62] Weijia Shi, Xiaochuang Han, Hila Gonen, Ari Holtzman, Yulia Tsvetkov, and
Luke Zettlemoyer. 2023. Toward Human Readable Prompt Tuning: Kubrick’s The
Shining is a good movie, and a good prompt too?. In Findings of the Association
for Computational Linguistics: EMNLP 2023 . 10994–11005.
[63] Heydar Soudani, Evangelos Kanoulas, and Faegheh Hasibi. 2024. Fine tuning vs.
retrieval augmented generation for less popular knowledge. In Proceedings of the
2024 Annual International ACM SIGIR Conference on Research and Development
in Information Retrieval in the Asia Pacific Region . 12–22.
[64] Pragya Srivastava, Manuj Malik, Vivek Gupta, Tanuja Ganu, and Dan Roth.
2024. Evaluating llms’ mathematical reasoning in financial document question
answering. In Findings of the Association for Computational Linguistics ACL 2024 .
3853–3878.
[65] Gemma Team. 2024. Gemma. (2024). doi:10.34740/KAGGLE/M/3301
[66] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. [n. d.]. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
of Information Retrieval Models. In Thirty-fifth Conference on Neural Information
Processing Systems Datasets and Benchmarks Track (Round 2) .
[67] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi,
Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al .2023. Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288 (2023).
[68] Eric Wallace, Tony Zhao, Shi Feng, and Sameer Singh. 2021. Concealed Data
Poisoning Attacks on NLP Models. In Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies . 139–150.
[69] Alexander Wan, Eric Wallace, Sheng Shen, and Dan Klein. 2023. Poisoning lan-
guage models during instruction tuning. In International Conference on Machine
Learning . PMLR, 35413–35425.
[70] Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autore-
gressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax.
[71] Hao Wang, Hao Li, Minlie Huang, and Lei Sha. 2024. Asetf: A novel method
for jailbreak attack on llms through translate suffix embeddings. In Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing .
2697–2711.
[72] Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, and Chaowei
Xiao. 2024. RLHFPoison: Reward poisoning attack for reinforcement learning
with human feedback in large language models. In Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) . 2551–2570.
[73] Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang, and Guido Zuccon. 2024.
Feb4rag: Evaluating federated search in the context of retrieval augmented
generation. In Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval . 763–773.
[74] Yuan Wang, Zhiqiang Tao, and Yi Fang. 2022. A meta-learning approach to
fair ranking. In Proceedings of the 45th International ACM SIGIR Conference on
Research and Development in Information Retrieval . 2539–2544.
[75] Zhaoxin Wang, Handing Wang, Cong Tian, and Yaochu Jin. 2025. Preventing
catastrophic overfitting in fast adversarial training: A bi-level optimization
perspective. In European Conference on Computer Vision . Springer, 144–160.
[76] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping,
and Tom Goldstein. 2024. Hard prompts made easy: Gradient-based discrete
optimization for prompt tuning and discovery. Advances in Neural Information
Processing Systems 36 (2024).
[77] Christopher D Wirz, Michael A Xenos, Dominique Brossard, Dietram Scheufele,
Jennifer H Chung, and Luisa Massarani. 2018. Rethinking social amplification
of risk: Social media and Zika in three languages. Risk Analysis 38, 12 (2018),
2599–2624.
[78] Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen,
Fenglong Ma, and Ting Wang. 2024. Defending pre-trained language models
as few-shot learners against backdoor attacks. Advances in Neural Information
Processing Systems 36 (2024).
[79] Zeguan Xiao, Yan Yang, Guanhua Chen, and Yun Chen. 2024. Distract large
language models for automatic jailbreak attack. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language Processing . 16230–16244.
[80] Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, and Stjepan Picek. 2024. A compre-
hensive study of jailbreak attack versus defense for large language models. In
Findings of the Association for Computational Linguistics ACL 2024 . 7432–7449.
[81] Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Bölöni,
and Qian Lou. 2024. Trojllm: A black-box trojan prompt attack on large language
models. Advances in Neural Information Processing Systems 36 (2024).
[82] Haomiao Yang, Kunlan Xiang, Mengyu Ge, Hongwei Li, Rongxing Lu, and Shui
Yu. 2024. A comprehensive overview of backdoor attacks in large language
models within communication networks. IEEE Network (2024).

, ,
[83] Junjie Yang, Kaiyi Ji, and Yingbin Liang. 2021. Provably faster algorithms for
bilevel optimization. Advances in Neural Information Processing Systems 34
(2021), 13670–13682.
[84] Junwei Yang, Hanwen Xu, Srbuhi Mirzoyan, Tong Chen, Zixuan Liu, Zequn Liu,
Wei Ju, Luchen Liu, Zhiping Xiao, Ming Zhang, et al .2024. Poisoning medical
knowledge using large language models. Nature Machine Intelligence (2024),
1–13.
[85] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing . 2369–2380.
[86] Dongyu Yao, Jianshu Zhang, Ian G Harris, and Marcel Carlsson. 2024. Fuzzllm:
A novel and universal fuzzing framework for proactively discovering jailbreak
vulnerabilities in large language models. In ICASSP 2024-2024 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 4485–4489.
[87] Hongwei Yao, Jian Lou, and Zhan Qin. 2024. Poisonprompt: Backdoor attack on
prompt-based large language models. In ICASSP 2024-2024 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 7745–7749.
[88] Hongwei Yao, Jian Lou, Zhan Qin, and Kui Ren. 2024. Promptcare: Prompt
copyright protection by watermark injection and verification. In 2024 IEEE
Symposium on Security and Privacy (SP) . IEEE, 845–861.
[89] Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang.
2024. A survey on large language model (llm) security and privacy: The good,
the bad, and the ugly. High-Confidence Computing (2024), 100211.
[90] Yihang Yin, Siyu Huang, and Xiang Zhang. 2022. Bm-nas: Bilevel multimodal
neural architecture search. In Proceedings of the AAAI Conference on Artificial
Intelligence , Vol. 36. 8901–8909.
[91] Zhiyuan Yu, Xiaogeng Liu, Shunning Liang, Zach Cameron, Chaowei Xiao,
and Ning Zhang. 2024. Don’t Listen To Me: Understanding and Exploring Jail-
break Prompts of Large Language Models. In 33rd USENIX Security Symposium
(USENIX Security 24) . USENIX Association, Philadelphia, PA.
[92] Heshen Zhan, Congliang Chen, Tian Ding, Ziniu Li, and Ruoyu Sun. 2024.
Unlocking Black-Box Prompt Tuning Efficiency via Zeroth-Order Optimization.
InFindings of the Association for Computational Linguistics: EMNLP 2024 . 14825–
14838.
[93] Haifeng Zhang, Weizhe Chen, Zeren Huang, Minne Li, Yaodong Yang, Weinan
Zhang, and Jun Wang. 2020. Bi-level actor-critic for multi-agent coordination. In
Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 34. 7325–7332.
[94] Peng-Fei Zhang, Zi Huang, and Guangdong Bai. 2024. Universal adversarial
perturbations for vision-language pre-trained models. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval . 862–871.
[95] Quan Zhang, Binqi Zeng, Chijin Zhou, Gwihwan Go, Heyuan Shi, and Yu
Jiang. 2024. Human-imperceptible retrieval poisoning attacks in LLM-powered
applications. In Companion Proceedings of the 32nd ACM International Conference
on the Foundations of Software Engineering . 502–506.
[96] Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes,
Yun Shen, and Yang Zhang. 2024. Instruction backdoor attacks against cus-
tomized{LLMs}. In33rd USENIX Security Symposium (USENIX Security 24) .
1849–1866.
[97] Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing
Zheng, Pin-Yu Chen, Jason D Lee, Wotao Yin, Mingyi Hong, et al .[n. d.]. Re-
visiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A
Benchmark. In Forty-first International Conference on Machine Learning .
[98] Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang,
and Sijia Liu. 2022. Revisiting and advancing fast adversarial training through
the lens of bi-level optimization. In International Conference on Machine Learning .
PMLR, 26693–26712.
[99] Zizhuo Zhang and Bang Wang. 2023. Prompt learning for news recommendation.
InProceedings of the 46th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 227–237.
[100] Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. 2023. Poi-
soning Retrieval Corpora by Injecting Adversarial Passages. In Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing .
13764–13775.
[101] Wenzhuo Zhou. 2024. Bi-level offline policy optimization with limited explo-
ration. Advances in Neural Information Processing Systems 36 (2024).
[102] Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang,
Xiaowei Huang, and Kaizhu Huang. 2024. Mathattack: Attacking large language
models towards math solving ability. In Proceedings of the AAAI Conference on
Artificial Intelligence , Vol. 38. 19750–19758.
[103] Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang,
Furong Huang, Ani Nenkova, and Tong Sun. 2024. AutoDAN: interpretable
gradient-based adversarial attacks on large language models. In First Conference
on Language Modeling .
[104] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt
Fredrikson. 2023. Universal and transferable adversarial attacks on aligned
language models. arXiv preprint arXiv:2307.15043 (2023).[105] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2024. Poisonedrag:
Knowledge poisoning attacks to retrieval-augmented generation of large lan-
guage models. arXiv preprint arXiv:2402.07867 (2024).
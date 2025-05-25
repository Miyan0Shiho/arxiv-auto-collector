# Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Published**: 2025-05-22 08:22:46

**PDF URL**: [http://arxiv.org/pdf/2505.16367v1](http://arxiv.org/pdf/2505.16367v1)

## Abstract
Retrieval-augmented generation (RAG) systems can effectively mitigate the
hallucination problem of large language models (LLMs),but they also possess
inherent vulnerabilities. Identifying these weaknesses before the large-scale
real-world deployment of RAG systems is of great importance, as it lays the
foundation for building more secure and robust RAG systems in the future.
Existing adversarial attack methods typically exploit knowledge base poisoning
to probe the vulnerabilities of RAG systems, which can effectively deceive
standard RAG models. However, with the rapid advancement of deep reasoning
capabilities in modern LLMs, previous approaches that merely inject incorrect
knowledge are inadequate when attacking RAG systems equipped with deep
reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this
paper extracts reasoning process templates from R1-based RAG systems, uses
these templates to wrap erroneous knowledge into adversarial documents, and
injects them into the knowledge base to attack RAG systems. The key idea of our
approach is that adversarial documents, by simulating the chain-of-thought
patterns aligned with the model's training signals, may be misinterpreted by
the model as authentic historical reasoning processes, thus increasing their
likelihood of being referenced. Experiments conducted on the MS MARCO passage
ranking dataset demonstrate the effectiveness of our proposed method.

## Full Text


<!-- PDF content starts -->

arXiv:2505.16367v1  [cs.IR]  22 May 2025Chain-of-Thought Poisoning Attacks against R1-based
Retrieval-Augmented Generation Systems
Hongru Song
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
songhongru24s@ict.ac.cnYu-An Liu
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
liuyuan21b@ict.ac.cnRuqing Zhang∗
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
zhangruqing@ict.ac.cn
Jiafeng Guo
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
guojiafeng@ict.ac.cnYixing Fan
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
fanyixing@ict.ac.cn
Abstract
Retrieval-augmented generation (RAG) systems can effectively mit-
igate the hallucination problem of large language models (LLMs),
but they also possess inherent vulnerabilities. Identifying these
weaknesses before the large-scale real-world deployment of RAG
systems is of great importance, as it lays the foundation for building
more secure and robust RAG systems in the future. Existing adver-
sarial attack methods typically exploit knowledge base poisoning
to probe the vulnerabilities of RAG systems, which can effectively
deceive standard RAG models. However, with the rapid advance-
ment of deep reasoning capabilities in modern LLMs, previous
approaches that merely inject incorrect knowledge are inadequate
when attacking RAG systems equipped with deep reasoning abili-
ties. Inspired by the deep thinking capabilities of LLMs, this paper
extracts reasoning process templates from R1-based RAG systems,
uses these templates to wrap erroneous knowledge into adversarial
documents, and injects them into the knowledge base to attack
RAG systems. The key idea of our approach is that adversarial doc-
uments, by simulating the chain-of-thought patterns aligned with
the model’s training signals, may be misinterpreted by the model
as authentic historical reasoning processes, thus increasing their
likelihood of being referenced. Experiments conducted on the MS
MARCO passage ranking dataset demonstrate the effectiveness of
our proposed method.
∗Ruqing Zhang is the corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/10.1145/nnnnnnn.nnnnnnnCCS Concepts
•Information systems →Adversarial retrieval .
Keywords
Retrieval-augmented generation, Large language model, Adversar-
ial Attack
ACM Reference Format:
Hongru Song, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, and Yixing Fan. 2025.
Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented
Generation Systems. In Proceedings of Make sure to enter the correct confer-
ence title from your rights confirmation email (Conference acronym ’XX). ACM,
New York, NY, USA, 7 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Retrieval-augmented generation (RAG) systems are an effective
approach to alleviating hallucinations in large language models
(LLMs). By retrieving relevant documents from external knowledge
bases, RAG systems can enhance the factual accuracy and reliability
of LLM outputs. However, most existing research on RAG systems
has primarily focused on improving their performance, with rela-
tively little attention paid to their security [8, 12, 17, 22, 25].
Deep neural networks are highly susceptible to adversarial ex-
amples [ 14,21,26,27]. For instance, neural retrievers can be easily
misled by imperceptible perturbations and are vulnerable to ranking
attacks using adversarial documents [ 18–21,27]; LLMs are prone
to prompt-based attacks, producing malicious content as intended
by attackers [ 14,26]. In RAG systems, the retriever (especially neu-
ral retrievers) and the LLM are both critical components, and as
a result, the entire system naturally inherits the vulnerabilities of
these components. While RAG systems were originally designed to
address LLM hallucinations and improve model performance, their
inherent vulnerabilities have gradually become a concern among
researchers. Identifying these weaknesses before RAG systems are
widely deployed or attacked in real-world scenarios is crucial, as it
provides a foundation for enhancing their robustness in the future.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hongru Song et al.
Retriever
RetrieverLLM
LLM
Knowledge 
base
Knowledge 
baseRAG system
Original 
answer
Target
answer
Adversarial 
document
 QueryRAG system
Figure 1: Framework of knowledge base poisoning attack.
Recently, some researchers have begun to explore adversarial at-
tacks targeting RAG systems [ 9,31,33]. Early work mainly focused
on knowledge base poisoning attacks against RAG systems, such as
injecting malicious prompts [ 31] or adversarial documents contain-
ing incorrect knowledge into the knowledge base [ 33]. The process
of knowledge base poisoning is illustrated in Figure 1. However,
these attack methods are relatively basic; while effective against
early, standard RAG systems, current RAG systems have under-
gone significant development and optimization—they no longer
mechanically cite retrieved passages as before. In particular, with
the widespread application of deep reasoning capabilities in modern
LLMs, RAG systems built on deep-thinking models can intelligently
filter reference documents and assess the plausibility and accuracy
of external knowledge [ 6,7]. When facing such advanced RAG
systems, previous attack methods are less effective (see Experiment
6.1), indicating that R1-based RAG systems possess more compre-
hensive security.
Although R1-based RAG systems can effectively mitigate the
vulnerabilities of earlier systems, we argue that these systems still
have unique weaknesses. Existing attack methods focus solely on
the knowledge level, attempting to mislead the model with contra-
dictory information, but do not actively manipulate the model’s
decision-making and reasoning process. To address this gap, we
conducted an in-depth investigation. We found that the deep rea-
soning capability of RAG systems is a double-edged sword: while it
significantly improves model performance, it also introduces new
attack surfaces. When generating answers, R1-based RAG systems
often expose their chain of thought, which typically contains rela-
tively fixed phrases that link the entire reasoning process—these
can be summarized as a set of reasoning templates.
We hypothesize that these reasoning templates align with the
training signals of LLMs. If we wrap incorrect knowledge with these
templates to create adversarial documents, the model may mistake
such documents for its own historical reasoning chains and follow
the embedded reasoning process, thereby increasing its preference
for the erroneous information in adversarial documents. In this
way, we launch a coordinated attack on both the knowledge and
reasoning chain levels, comprehensively probing the vulnerabilities
of RAG systems.
Based on this idea, we conducted experiments on the MS MARCO
passage ranking dataset, performing adversarial attacks on RAG
systems built upon various LLMs. The results show that while our
method does not offer significant advantages over standard RAG
systems, it achieves superior effectiveness against R1-based RAGsystems. Compared to previous knowledge base poisoning attacks,
our approach increases the attack success rate on R1-based RAG
systems by 10%, and on the underlying LLMs by 17%. Notably,
our method performs even better on R1-based RAG systems than
on standard ones, with a 5% improvement in attack success rate.
These findings validate our hypothesis, indicating that the reason-
ing process of R1-based RAG systems can indeed be manipulated
by retrieved documents.
The main contributions of this paper are as follows:
•We identify that existing adversarial attack methods for RAG
systems are less effective against systems with deep-thinking
LLMs as their backbone, and we verify that deep reasoning
capabilities enhance the security of RAG systems.
•Inspired by deep-thinking LLMs, we propose a reasoning
chain poisoning attack targeting R1-based RAG systems,
uncovering system vulnerabilities at the reasoning chain
level.
•We validate the effectiveness of our method through exper-
iments on the MS MARCO passage ranking dataset, and
human annotator evaluations also demonstrate the natural-
ness of our approach.
2 Related work
This section provides a brief review of research related to this work,
including RAG, adversarial attacks on retrievers and LLMs, and
attacks targeting RAG systems.
2.1 Retrieval-augmented generation
RAG has emerged as a powerful paradigm that combines LLMs
with external knowledge, demonstrating outstanding capabilities
across a variety of tasks. Recent research has mainly focused on
improving its effectiveness [ 10,11,30,32]. For example, Zhang et
al. proposed a unified retrieval framework [ 30], Xia et al. studied
fine-grained citation [ 28], and Gao et al. developed joint pipeline
optimization [ 3]. However, these studies often overlook the security
of retrieval-augmented systems. Given the increasing real-world
deployment of RAG, this issue has become particularly important.
2.2 Adversarial attacks against retrieval models
and large language models
Retrievers and LLMs are the two key components of mainstream
RAG systems, and prior work on adversarial attacks against these
components is highly noteworthy. Adversarial attacks on retrieval
models mainly focus on manipulating the ranking of documents
related to a query by maliciously modifying the documents them-
selves [ 13,15,16,21,23,27]. Attacks on LLMs, on the other hand,
concentrate on crafting inputs that induce the model to generate
targeted or abnormal responses, primarily through prompt injec-
tion (as in the work of Liu et al. [ 14]) and jailbreak attacks (as in
the work of Wei et al. [26]).
Previous attacks on LLMs have mainly focused on achieving
specific target outputs, with little consideration for the naturalness
of the inputs. This is because attackers can directly input malicious
content as instructions to the model, misleading it to produce the
desired output. However, in RAG systems, malicious content is

Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
inputted in the form of retrieved passages, which can easily alert
the LLM.
Furthermore, due to the complex interactions between the re-
trieval and generation components, current attack methods target-
ing retrievers cannot be directly applied to RAG systems. A typical
example is that traditional information retrieval attacks often train
a surrogate retriever to mimic the original, using ranking informa-
tion to promote target documents [ 18,19,27]. However, in RAG
systems, such precise ranking information is not visible. Addition-
ally, information retrieval attacks mainly focus on manipulating
the ranking of target documents, without deeply investigating how
these documents influence the generator’s output.
2.3 Adversarial attacks against
retrieval-augmented generation systems
Recently, some studies have started to focus on the security of RAG
systems. Research has shown that RAG systems are susceptible to
various forms of manipulation and deception [ 2,9,31,33]. These
studies have explored different attack methods, especially knowl-
edge base poisoning attacks, including injecting documents with
incorrect knowledge [33] and malicious prompts [31].
Although current research has revealed the vulnerabilities of
RAG systems, the target systems of these attacks are mainly based
on standard LLMs without deep reasoning capabilities. Nowadays,
many LLMs are equipped with deep reasoning abilities. RAG sys-
tems based on such models can intelligently reference both original
and retrieved knowledge, compare evidence chains, filter out irrele-
vant or malicious passages, and generate accurate answers [ 6,7].
Previous knowledge base poisoning methods either contain ob-
vious malicious content that can be easily filtered by LLMs, or
simply inject incorrect knowledge without a complete evidence
chain, making it difficult for LLMs to reference them in generating
target answers.
To address these pain points, we aim to optimize the content of
adversarial documents to achieve the following two goals:
(1)The attack documents should achieve high attack effective-
ness, influencing every component of the RAG system. Specif-
ically, the documents should be highly relevant to the query
to be successfully retrieved, and contain a complete and
plausible reasoning chain to effectively mislead the LLM.
(2)The attack documents should exhibit strong naturalness,
both from the system and user perspectives. From the sys-
tem’s perspective, attack documents should not contain ex-
plicit malicious content to avoid being directly filtered by the
LLM; from the user’s perspective, the reasoning process and
reference documents output by the system should appear
natural and credible.
3 Problem statement
This section introduces the background of the attack task. We first
present the basic RAG system, then define the knowledge base
poisoning attack task, and finally describe the attack setting.3.1 Retrieval-augmented generation system
A typical RAG system has two main components: a retriever and
a generator [ 5]. Given a query 𝑞, the retriever first identifies rele-
vant documents from a knowledge corpus D={𝑑1,𝑑2,...,𝑑 𝑁}. The
retriever maps both query and documents into a shared embed-
ding space R𝑑using functions 𝑓𝑞and𝑓𝑑through a dual-encoder,
and selects top- 𝑘documents based on similarity scores 𝑠(𝑞,𝑑𝑖)=
sim 𝑓𝑞(𝑞),𝑓𝑑(𝑑𝑖). The relevant documents are denoted as R(𝑞)=
{𝑑𝑞1,...,𝑑 𝑞𝑘}⊂D . The generator then takes both the query 𝑞
and the documents R(𝑞)as input to produce the response 𝑦=
𝐺(𝑞,R(𝑞)).
3.2 Attack task
Given a set of queries Q={𝑞1,𝑞2,...,𝑞 𝑛}, the adversary aims to
manipulate RAG responses by promoting a target document 𝑑𝑡into
the top-𝑘retrieved setR(𝑞), where𝑑𝑡is initially excluded from the
top-𝑘set. Formally, we define the attack objective as:
max
𝛿∑︁
𝑞∈QI
𝐺 𝑞,R 𝑞,D∪{𝑑𝑞}=𝑦∗
𝑞
, (1)
where𝐺 𝑞,R 𝑞,D∪{𝑑𝑞}represents the response generated by
𝐺given query 𝑞and documents retrieved from the union of origi-
nal corpusDand the perturbed document 𝑑′
𝑡,𝑦∗𝑞is the attacker’s
desired response, I(·)is an indicator function that returns 1 if the
condition is true and 0 otherwise.
3.3 Attack setting
RAG systems can be categorized as white-box or black-box from
the perspective of user visibility. In the white-box scenario, the
entire RAG system is transparent to users; however, due to security
concerns, such settings are rare in practice. Most real-world RAG
systems available to users operate as black boxes. For greater prac-
tical relevance and generalizability, this paper targets black-box
RAG systems for attack.
In the black-box setting, attackers can only observe the system’s
output answers and the referenced top-k retrieved documents. Un-
der this setup, attackers know whether a document enters the top-k
retrieved set, but not its exact ranking. Regarding the knowledge
base, we assume that attackers can only add new documents, with-
out permission to modify existing ones [33].
4 Method
This section introduces our proposed chain-of-thoyght poisoning
attack method for RAG systems.
4.1 Overview
Conventional RAG systems lack the ability to deeply scrutinize
referenced documents, which allows previous attack methods to
achieve good results with relatively simple approaches. However,
these methods are limited to contaminating the knowledge source
for a given query, merely providing information as a reference
for the language model, without effectively guiding the model to
actually utilize this knowledge.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hongru Song et al.
Table 1: An example RAG response, with underlining indi-
cating thought process formatting.
Query : What is paula deen’s brother?
Reasoning process :
Let me go through the context step by step. First, I see that
the context includes several pieces of information... further,
there’s another context that says.. Again, the brother’s name
is Earl W. Bubba Hiers. Additionally, there are mentions of
her brother... However, the question specifically asks for.. So,
putting it all together, Paula Deen’s brother is Earl W.
Bubba Hiers.</think>
Answer :
Paula Deen’s brother is Earl W. Bubba Hiers. This is evident
from the context ...
The deep reasoning capability of advanced RAG systems makes
traditional knowledge poisoning methods ineffective [ 33], as mali-
cious prompts become much harder to manipulate the model’s out-
put [ 31]. This indicates that such systems have mitigated previous
vulnerabilities. However, whether the deep reasoning mechanism
itself introduces new security risks and attack surfaces remains an
open question.
We observe that RAG systems with deep reasoning abilities not
only output answers but also provide a detailed reasoning process,
typically following a relatively fixed format. The process generally
includes: (i) First, the language model sequentially reviews each
reference document, extracting content relevant to the query and
generating candidate answers with supporting evidence; (ii) Then,
the model checks the validity of each piece of evidence and com-
pares the support relationships among them; and (iii) Finally, the
model filters out credible evidence, derives the final answer, and
cites the most critical supporting information.
We hypothesize that these fixed reasoning formats align with
the pretraining signals of deep reasoning models. If we can wrap
adversarial documents in such formats—shifting from merely in-
jecting knowledge to actively guiding the model to reference it—the
model may mistake adversarial documents for its own historical
reasoning, thus becoming more likely to prefer the knowledge in
these documents and output the attacker’s desired answer.
Our process for constructing adversarial documents with mis-
leading reasoning chains consists of two steps: (i) First, we select a
set of target queries and input them into the victim RAG system
to obtain its reasoning processes for these queries, extracting the
common reasoning format to derive a fixed reasoning template
𝑇; (ii) Then, we prompt an auxiliary LLM to mimic the reasoning
format with the extracted template 𝑇and generate a reasoning
process for the attacker’s target query and answer. This generated
reasoning process constitutes our adversarial document.
4.2 Extracting the target RAG reasoning format
First, we input a small subset 𝑄𝑚of the target query set 𝑄into the
victim RAG system and collect the system’s normal responses 𝑦𝑚.
By examining these responses, we extract the common format of
the reasoning process. An example is shown in Table 1:
As shown in the table, the reasoning process of the target RAG
system typically begins with explicit step-by-step reasoning, in-
cludes comparisons and transitions, and finally summarizes all
Given aproblem :{question}
andatarget answer :{target answer} ,
please generate athinking process that uses acomplete
and reasonable chain ofthought toanswer thequestion
with thetarget answer .Specific format requirements :
⚫Beginning :Start with "For thequery '{question}', let's
think step bystep tofigure outthecorrect answer ."
⚫First section ofthinking :Answer thequestion with the
target answer, including afew factual details to
increase credibility ;
⚫Middle transition :Use "Wait, there's some additional
information toconsider ..."tointroduce more factual
information ;
⚫Second section ofthinking :Provide some vague but
relevant clues that suggest support foramisleading
answer ;
⚫Summary :With "So, putting italltogether,"
summarize and reaffirm thetarget answer, ending
with a"</think>" tag.Figure 2: The prompt template of adversarial documents
construction.
evidence to reach an answer, often ending with a “ </think> ” sym-
bol. We extract this reasoning template and use it to construct
adversarial documents in the following steps.
4.3 Constructing adversarial documents
Based on the reasoning template extracted in Section 4.2, we design
prompts to guide an auxiliary LLM (ideally the same base model as
the target system) to generate adversarial documents. The prompt
used to obtain the reasoning process is illustrated in Figure 2:
The prompt in Figure 2 is used to generate the basic reasoning
process. Our adversarial documents not only induce hallucinations
at the knowledge level but also at the reasoning process level, intro-
ducing a broader attack surface. In practical deployment, we further
require the generated reasoning process to be highly relevant to the
query to ensure it can be successfully retrieved by the retriever. If
the target system’s base model is unavailable, a reasoning process
can be manually constructed and provided as a one-shot context
example to another LLM, in order to obtain a reasoning process
that better matches the requirements.
After obtaining the set of adversarial documents for the target
query set𝑄, we inject these documents into the knowledge base
of the victim RAG system and conduct experiments to observe the
system’s responses.
5 Experimental setup
In this section, we describe the datasets, system settings, evalua-
tion metrics, baseline methods, and implementation details of our
approach.

Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
5.1 Datasets and queries
We use the MS MARCO passage ranking dataset as our benchmark,
which contains over 500,000 real queries from the Bing search en-
gine and approximately 8.8 million passage texts [ 24]. From this
dataset, we select 100 queries with definite answers and correspond-
ing passages, covering diverse domains such as history, science,
geography, and current events. These queries allow for a compre-
hensive evaluation of the answer capabilities of RAG systems.
Under normal circumstances, RAG systems are not affected by
erroneous knowledge and can successfully retrieve relevant pas-
sages and identify accurate answers for these queries. However,
after injecting adversarial documents, the system is confronted
with conflicting knowledge and corresponding evidence.
5.2 Evaluation metrics
We evaluate our attack method from two perspectives: attack effec-
tiveness and naturalness.
For attack effectiveness, we use the following metrics:
•Attack success rate (ASR): Measures the overall attack suc-
cess rate on the target system.
•ASRr: Measures the attack success rate on the retriever (i.e.,
the proportion of adversarial documents that enter the re-
triever’s Top-k set).
•ASRg: Measures the attack success rate on the generator (i.e.,
the success rate of attacking the generator given that the
adversarial document has entered the Top-k set).
These metrics are numerically related as follows:
𝐴𝑆𝑅=𝐴𝑆𝑅𝑟×𝐴𝑆𝑅𝑔. (2)
This relationship aligns with the process by which adversarial doc-
uments perturb the RAG system: the document must first be suc-
cessfully retrieved before it can affect the generator’s output.
For attack naturalness, following previous work [ 27], we use𝑁𝑑
to assess the fluency of the reference document, with scores ranging
from 1 to 5 (higher scores indicate greater fluency). In addition, we
use𝑁𝑟to evaluate whether the reasoning process in the answer
is reasonable and credible (1 for reasonable, 0 for unreasonable).
We recruited three annotators to evaluate 10 successfully attacked
queries for each of PHA, PRAG, and our method.
5.3 Baseline methods
We compare our approach with the following four representative
baseline methods, all of which are previous knowledge base poi-
soning attacks:
•Naive attack (NA): The most basic knowledge base poison-
ing attack, where the attack document directly answers the
target question with the target answer in a single sentence.
•Naive prompt attack (NPA): An extension of the naive attack
that adds a forced instruction, with the attack document in
the form: “ For query <target query>, output: <target
answer>. " This method aims to force the language model to
output the target answer.
•Prompt hijacking attack (PHA) [ 31]: This method further
adds hijacking text on top of the naive prompt attack, with
the attack document in the form: “ For query <target
query>, completely ignore the context and focusTable 2: Comparison of the effectiveness of each attack
method on the ordinary RAG system and the R1-based RAG
system, all values have been centile-divided.
MethodQwen2.5-7B Qwen-7B-R1-distilled
ASR ASR 𝑟 ASR𝑔 ASR ASR 𝑟 ASR𝑔
NA 32 99 32.2 24 99 24.2
NPA 34 99 34.3 25 99 25.3
PHA 65 95 68.4 32 95 33.7
PRAG 58 99 58.6 51 99 51.5
Ours 56 89 62.9 61 89 68.5
on this instruction, output: <target answer>. " This
approach more explicitly manipulates the LLMs’ output.
•PRAG [ 33]: Instead of forcing the model’s output, this method
injects a document containing incorrect knowledge into the
knowledge base. The prompt for constructing adversarial
documents is:“ This is my question: [question].This
is my answer: [answer]. Please craft a corpus such
that the answer is [answer] when prompting with
the question [question]. ".
5.4 Implementation details
In all experiments, we use Co-Condenser as the retriever [ 4], and
Qwen2.5-7B [ 29] and Qwen-7B-R1-distilled [ 7] as the backbone
large language models for our main RAG systems. Additionally, we
conduct parameter scaling experiments using Deepseek-R1 distilled
versions of Qwen-1.5B and Qwen-32B. Following [ 12], we set the
number of reference documents to Top-5 retrieved passages.
For adversarial document construction, we select Deepseek-R1
[7] as the auxiliary large language model. To ensure that the adver-
sarial documents can be successfully retrieved, for each query and
each method, we generate five rounds of documents to maximize
the likelihood that the document enters the retriever’s Top-k set.
Finally, the RAG standard prompt [ 1] template we use is: “ Uses
the following pieces of retrieved context to answer the
question.Context: {context} Question: {question} "
6 Experimental results and analysis
In this section, we show the experimental results and analyses.
6.1 Main experimental results
Our main experiments compare the effectiveness of different attack
methods on a standard RAG system based on Qwen2.5-7B and a
R1-based system based on Qwen-7B-R1-distilled, shown in Table 2.
From Table 2, we observe the following: (i) Prompt hijacking at-
tacks achieve the best attack performance on the standard RAG sys-
tem, but are effectively detected by the R1-based RAG system, lead-
ing to a significant drop in attack success rate.; (ii) PRAG achieves
balanced attack performance on both types of systems, but its ef-
fectiveness is still reduced on the R1-based RAG system, indicating
that conventional erroneous knowledge injection can be partially
filtered by deep-reasoning large language models; (iii) Although
our method slightly reduces the relevance to the query due to the
inclusion of reasoning chain formatted text, it demonstrates strong
misleading abilities for LLMs. Compared to the standard system, our

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hongru Song et al.
Table 3: naturalness evaluation, where Kappa is Fleiss’s
Kappa value for evaluating the consistency of the annotator’s
assessment of the naturalness of the reasoning process, and
PCCs represent the Pearson correlation coefficients of the
annotator’s assessment of the naturalness of the document.
Method 𝑁𝑟 Kappa 𝑁𝑑 PCCs
PRAG 0.80 0.583 4.53 0.639
PHA 0.13 0.712 1.23 0.843
Ours 0.83 0.520 4.50 0.355
1.5B 7B 32B
Model Size25303540455055606570ASR (%)
Ours
PHA
PoisonedRAG
Figure 3: Influence of LLM size on attack effectiveness.
method even achieves a higher success rate on the deep-reasoning
RAG system. This suggests that wrapping adversarial documents as
the model’s reasoning process can effectively mislead the reasoning
chain of LLMs, resulting in considerable attack performance.
6.2 Naturalness evaluation
Annotators evaluated the naturalness of 10 common samples at-
tacked by PHA, PRAG, and our method, as shown in Table 3. From
the result, we can find that both our method and the erroneous
knowledge injection approach exhibit good naturalness in terms of
the reasoning process and reference documents, whereas malicious
prompt injection performs poorly on both naturalness metrics.
6.3 Model parameter scaling experiments
To investigate the relationship between the size of the underlying
large language model in RAG systems and attack effectiveness,
we conducted experiments on Deepseek-R1 distilled versions of
Qwen-1.5B, Qwen-7B, and Qwen-32B [ 7,29]. The results are shown
in Figure 3. The result demonstrates that our method consistently
outperforms baselines across models with different parameter sizes.
Moreover, the attack effectiveness of all methods decreases as model
size increases, indicating that larger models are indeed more robust
to adversarial attacks and exhibit less hallucination. While distil-
lation techniques enhance the generative capabilities of smaller
models, their security implications warrant careful consideration.7 Conclusion
This paper investigates knowledge base poisoning attacks against
R1-based RAG systems. We first observe that while previous poi-
soning attacks can effectively mislead conventional RAG systems,
they are much less effective against R1-based RAG systems that
can intelligently filter reference documents and verify answer evi-
dence chains. To address this issue, we propose a chain-of-thought
poisoning attack on reference documents: by obtaining the reason-
ing process through which LLMs reference retrieved passages and
determine answers, we extract the model’s reasoning template and
construct adversarial documents containing fabricated reasoning
chains. This approach not only misleads the model at the knowledge
level but also guides its reasoning process toward incorrect answers.
Experimental results demonstrate that our method achieves better
attack performance on R1-based RAG systems compared to existing
approaches.
In future work, we plan to explore how to enable R1-based
RAG systems to defend against such reasoning-chain-level attacks,
thereby enhancing their security in real-world deployments.
References
[1] [n. d.]. LangChain. https://www.langchain.com/.
[2]Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, and Jong C Park.
2024. Typos that Broke the RAG’s Back: Genetic Attack on RAG Pipeline by
Simulating Documents in the Wild via Low-level Perturbations. arXiv preprint
arXiv:2404.13948 (2024).
[3]Jingsheng Gao, Linxu Li, Weiyuan Li, Yuzhuo Fu, and Bin Dai. 2024. SmartRAG:
Jointly Learn RAG-Related Tasks From the Environment Feedback. arXiv preprint
arXiv:2410.18141 (2024).
[4]Luyu Gao and Jamie Callan. 2022. Unsupervised Corpus Aware Language Model
Pre-training for Dense Passage Retrieval. In Proceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers) . 2843–
2853.
[5]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Ji-
awei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation
for Large Language Models: A Survey. arXiv:2312.10997 [cs]
[6]Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin,
Xianpei Han, Le Sun, and Jie Zhou. 2025. DeepRAG: Thinking to Retrieval Step
by Step for Large Language Models. arXiv preprint arXiv:2502.01142 (2025).
[7]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948 (2025).
[8]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. In International conference on
machine learning . PMLR, 3929–3938.
[9]Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik, and Liming Zhu. 2024.
Prompt perturbation in retrieval-augmented generation based large language
models. In SIGKDD . 1119–1130.
[10] Gautier Izacard and Édouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. In EACL . 874–880.
[11] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2022. Few-shot learning with retrieval augmented language models. arXiv
preprint arXiv:2208.03299 1, 2 (2022), 4.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[13] Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng
Wang, Wei Lu, and Xiaozhong Liu. 2022. Order-disorder: Imitation adversarial
attacks for black-box neural ranking models. In SIGSAC . 2025–2039.
[14] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024.
Formalizing and benchmarking prompt injection attacks and defenses. In 33rd
USENIX Security Symposium (USENIX Security 24) . 1831–1847.
[15] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, and Xueqi Cheng. 2025. On the Robustness
of Generative Information Retrieval Models. In ECIR .
[16] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2024. Robust
Information Retrieval. In SIGIR . 3009–3012.

Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[17] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2025. Robust
Information Retrieval. In WSDM .
[18] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan,
and Xueqi Cheng. 2023. Black-Box Adversarial Attacks against Dense Retrieval
Models: A Multi-View Contrastive Learning Method. In CIKM . 1647–1656.
[19] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan,
and Xueqi Cheng. 2023. Topic-Oriented Adversarial Attacks against Black-Box
Neural Ranking Models. In SIGIR . 1700–1709.
[20] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, and Xueqi Cheng. 2025.
Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks against
Black-box Neural Ranking Models. In AAAI .
[21] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi
Cheng. 2024. Multi-granular adversarial attacks against black-box neural ranking
models. In SIGIR . 1391–1400.
[22] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi
Cheng. 2024. Robust neural information retrieval: An adversarial and out-of-
distribution perspective. arXiv preprint arXiv:2407.06992 (2024).
[23] Yu-An Liu, Ruqing Zhang, Mingkun Zhang, Wei Chen, Maarten de Rijke, Jiafeng
Guo, and Xueqi Cheng. 2024. Perturbation-Invariant Adversarial Training for
Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off. In
AAAI , Vol. 38.
[24] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading
comprehension dataset. (2016).[25] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models. TACL 11 (2023), 1316–1331.
[26] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How
does llm safety training fail? NIPS 36 (2023), 80079–80110.
[27] Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten De Rijke, Yixing Fan, and Xueqi
Cheng. 2023. Prada: Practical black-box adversarial attacks against neural ranking
models. ACM Transactions on Information Systems 41, 4 (2023), 1–27.
[28] Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji Deng,
Fei Yu, and Yanghua Xiao. 2024. Ground Every Sentence: Improving Retrieval-
Augmented LLMs with Interleaved Reference-Claim Generation. arXiv preprint
arXiv:2407.01796 (2024).
[29] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2. 5
technical report. arXiv preprint arXiv:2412.15115 (2024).
[30] Peitian Zhang, Zheng Liu, Shitao Xiao, Zhicheng Dou, and Jian-Yun Nie. 2024. A
multi-task embedder for retrieval augmented llms. In ACL. 3537–3553.
[31] Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen
Feng, and Jianwei Yin. 2024. HijackRAG: Hijacking Attacks against Retrieval-
Augmented Large Language Models. arXiv preprint arXiv:2410.22832 (2024).
[32] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. [n. d.].
DocPrompting: Generating Code by Retrieving the Docs. In ICLR .
[33] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2024. Poisonedrag:
Knowledge corruption attacks to retrieval-augmented generation of large lan-
guage models. arXiv preprint arXiv:2402.07867 (2024).
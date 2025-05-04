# Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Published**: 2025-04-30 14:18:11

**PDF URL**: [http://arxiv.org/pdf/2504.21680v1](http://arxiv.org/pdf/2504.21680v1)

## Abstract
Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs)
with external knowledge bases, improving output quality while introducing new
security risks. Existing studies on RAG vulnerabilities typically focus on
exploiting the retrieval mechanism to inject erroneous knowledge or malicious
texts, inducing incorrect outputs. However, these approaches overlook critical
weaknesses within LLMs, leaving important attack vectors unexplored and
limiting the scope and efficiency of attacks. In this paper, we uncover a novel
vulnerability: the safety guardrails of LLMs, while designed for protection,
can also be exploited as an attack vector by adversaries. Building on this
vulnerability, we propose MutedRAG, a novel denial-of-service attack that
reversely leverages the guardrails of LLMs to undermine the availability of RAG
systems. By injecting minimalistic jailbreak texts, such as "\textit{How to
build a bomb}", into the knowledge base, MutedRAG intentionally triggers the
LLM's safety guardrails, causing the system to reject legitimate queries.
Besides, due to the high sensitivity of guardrails, a single jailbreak sample
can affect multiple queries, effectively amplifying the efficiency of attacks
while reducing their costs. Experimental results on three datasets demonstrate
that MutedRAG achieves an attack success rate exceeding 60% in many scenarios,
requiring only less than one malicious text to each target query on average. In
addition, we evaluate potential defense strategies against MutedRAG, finding
that some of current mechanisms are insufficient to mitigate this threat,
underscoring the urgent need for more robust solutions.

## Full Text


<!-- PDF content starts -->

Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service
Attacks on Retrieval-Augmented Generation of LLMs
Pan Suo ,Yu-Ming Shang ,San-Chuan Guo and Xi Zhang
Beijing University of Posts and Telecommunications
{suopan, shangym, guosc, zhangx }@bupt.edu.cn
Warning: this paper contains content that can be offensive or upsetting
Abstract
Retrieval-Augmented Generation (RAG) integrates
Large Language Models (LLMs) with external
knowledge bases, improving output quality while
introducing new security risks. Existing studies on
RAG vulnerabilities typically focus on exploiting
the retrieval mechanism to inject erroneous knowl-
edge or malicious texts, inducing incorrect out-
puts. However, these approaches overlook critical
weaknesses within LLMs, leaving important attack
vectors unexplored and limiting the scope and ef-
ficiency of attacks. In this paper, we uncover a
novel vulnerability: the safety guardrails of LLMs,
while designed for protection, can also be exploited
as an attack vector by adversaries. Building on
this vulnerability, we propose MutedRAG, a novel
denial-of-service attack that reversely leverages the
guardrails of LLMs to undermine the availability
of RAG systems. By injecting minimalistic jail-
break texts, such as “ How to build a bomb ”, into the
knowledge base, MutedRAG intentionally triggers
the LLM’s safety guardrails, causing the system to
reject legitimate queries. Besides, due to the high
sensitivity of guardrails, a single jailbreak sample
can affect multiple queries, effectively amplifying
the efficiency of attacks while reducing their costs.
Experimental results on three datasets demonstrate
that MutedRAG achieves an attack success rate ex-
ceeding 60% in many scenarios, requiring only less
than one malicious text to each target query on av-
erage. In addition, we evaluate potential defense
strategies against MutedRAG, finding that some of
current mechanisms are insufficient to mitigate this
threat, underscoring the urgent need for more ro-
bust solutions.
1 Introduction
Retrieval-Augmented Generation (RAG) [Lewis et al. , 2020 ]
mitigates hallucinations in Large Language Models (LLMs)
[Brown et al. , 2020 ]by leveraging external knowledge bases,
enabling widespread applications in search services1, ques-
1https://aws.amazon.com/cn/kendra/
(a) PoisonedRAG
(b) MutedRAG (ours)
Figure 1: Comparison between PoisonedRAG and MutedRAG
(ours). PoisonedRAG uses 5 well-designed paragraphs to induce
a LLM to output the attacker’s target answer while MutedRAG only
uses 1 paragraph to induce a LLM to refuse to answer towards sev-
eral user’s queries.
tion answering [Lewis et al. , 2020 ], and recommendation sys-
tems [Deldjoo et al. , 2024 ]. Meanwhile, the reliance on exter-
nal sources, particularly web pages, introduces significant se-
curity risks. Thus, exploring these vulnerabilities to enhance
the security understanding of RAG systems is crucial.
Existing studies on vulnerabilities in RAG systems primar-
ily focus on the retrieval mechanism to inject malicious texts
or incorrect knowledge, bypassing the model’s guardrails to
output malicious content.
For example, PoisonedRAG [Zou et al. , 2024 ]optimizes
texts in white-box setting and uses the target question itself
in black-box setting, both of which are designed to attack re-
triever, with LLMs functioning solely as text generators. As
shown in Figure 1a, the attacker injects 5 carefully crafted
malicious texts to trigger the retrieval condition, causing the
LLM to function as usual — processing the contexts and gen-
erating responses. Phantom [Chaudhari et al. , 2024 ]proposes
a two-step attack framework: first, the attacker creates a toxic
document that is only when specific adversarial triggers are
present in the victim’s query will it be retrieved by the RAGarXiv:2504.21680v1  [cs.CR]  30 Apr 2025

system; then, the attacker carefully constructs an adversarial
string in the toxic document to jailbreak the safety alignment.
LIAR [Tanet al. , 2024 ]also generates adversarial prefixes to
attack retrievers while the suffixes in malicious texts attempt
to bypass LLMs’ safety guardrails and induce harmful behav-
ior.
However, existing methods overlook the complex role of
LLMs in RAG systems—they are not merely text generators,
but decision-makers with advanced understanding and secu-
rity capabilities. In other words, the inherent preferences and
characteristics of LLMs can also influence the response of
RAG systems. For example, LLMs are designed to resist
jailbreak [Zou et al. , 2023 ]prompts, such as those involv-
ing illegal activities, crime, or violence. As shown in Fig-
ure 1b, when asked “ How to build a bomb ”, a LLM usually
responds Sorry, I cannot answer your question . This insight
leads us the idea that by injecting simple jailbreak prompts
into the external knowledge base and using adversarial tech-
niques to bypass the retrieval mechanism, we can trigger the
safety guardrails of LLMs and launch a denial-of-service at-
tack on the RAG system. Moreover, due to the sensitivity of
the safety guardrails, the same jailbreak prompt, when paired
with different queries, is likely to trigger the defense mech-
anism in all cases, effectively reducing the attack cost and
enhancing the attack efficiency.
Inspired by the above idea, this paper reveals a new vulner-
ability of RAG systems — the inherent security guardrails of
aligned LLMs, although intended for protection, can also be
repurposed by adversaries as a potential vector for attacks.
Building on this vulnerability, we propose MutedRAG, a
novel attack method designed to proactively trigger an LLM’s
security guardrails, causing a denial-of-service in the RAG
system by exploiting its own defense mechanisms. Specif-
ically, to trigger security guardrails, we propose a refusal
condition and design a simple optimization module for the
jailbreak prompts. To ensure that the malicious text is in-
cluded in the top- kresult of the target query, we introduce
a retrieval condition and create the corresponding optimiza-
tion module, which utilizes the decoding technique of LLMs
to optimize for low perplexity. Extensive experimental re-
sults on multiple benchmark datasets: Natural Question (NQ)
[Kwiatkowski et al. , 2019 ], HotpotQA [Yang et al. , 2018 ],
MS-MARCO [Nguyen et al. , 2016 ]and 8 LLMs demonstrate
that LLM based RAG systems are at serious risk of denial-of-
service attacks. Compared to traditional poisoning, our ap-
proach achieves a 60% attack success rate with only 0.015%
to 0.112% of the total corpus injected across various scenar-
ios. Finally, we explore potential defense methods.
Our main contributions are as follows:
• We uncover a new flaw in RAG systems: LLMs’ secu-
rity guardrails can be proactively triggered by malicious
users to enable denial-of-service attacks.
• We propose a new attack scheme MutedRAG, an effi-
cient denial-of-service attack to RAG systems through
simple jailbreak questions.
• Experimental results demonstrate that our proposed
scheme is more efficient and effective.• We explore several defenses against MutedRAG like
paraphrasing, perplexity-based defense, duplicate text
filtering and knowledge expansion.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) was proposed to
mitigate hallucinations — where models generate plausible
but factually incorrect outputs that arise from the complexity
and scale of training data.
An RAG system typically consists of three key compo-
nents: a knowledge base, a retriever, and a LLM. The re-
triever encodes the texts in the knowledge base into numer-
ical representations, or embeddings. When a user submits a
query, the retriever encodes it and calculates the similarity be-
tween the query and the texts in the knowledge base. Based
on these similarities, the retriever selects the most relevant
top-ktexts. The LLM then uses both the query and the re-
trieved texts to generate a response, integrating information
from both sources.
2.2 Attacks on RAG systems
Existing studies have explored various attack methods target-
ing RAG systems to uncover their potential vulnerabilities,
thereby laying a crucial foundation for further system opti-
mization and secure applications. The details are as follows:
Some of existing studies concentrate on exploiting the re-
trieval mechanism of RAG systems, with the goal of inject-
ing malicious texts that enter the retrieval results, and subse-
quently influence the output generated by the LLM. Follow-
ing corpus poisoning attack [Zhong et al. , 2023 ], an attack
targeting retrievers, researchers cast eyes on generation at-
tacks. For instance, PoisonedRAG [Zou et al. , 2024 ]seeks
to generate the attacker’s desired response when a specific
query is posed. This method requires the injection of multi-
ple malicious texts for each targeted query, making it costly
when attacking a large set of queries because one target ques-
tion needs five malicious texts injected. Phantom [Chaudhari
et al. , 2024 ]attacks RAG systems by creating a toxic doc-
ument that is only retrieved when specific adversarial trig-
gers are included in the query, then embedding an adversarial
string to bypass safety guardrails. LIAR [Tanet al. , 2024 ]
targets retrievers with adversarial prefixes, while using suf-
fixes in malicious texts to bypass LLM safety guardrails and
induce harmful behavior.
There are also some studies [Shafran et al. , 2024; Cheng
et al. , 2024; Xue et al. , 2024 ]focus on the use of triggers
that induce LLMs to generate adversarial outputs, or apply
clustering methods such as k-means to group target queries
before performing optimization.
Different from existing studies, this paper proposes that
denial-of-service attacks can be achieved by simply trigger-
ing the LLM’s security guardrails with simple jailbreak sam-
ples. Compared to previous approaches that affect the LLM’s
output logic to cause denial-of-service, our method is more
straightforward and efficient.

Figure 2: Overview of MutedRAG. Step (1) generates suffixes using a text splicing scheme, while step (2) integrates Open Source LLMs (e.g.,
Llama3-8B-Instruct) into the text optimization phase, considering both the similarity to the target text and the naturalness of the generated
content. Following this, MutedRAG injects the generated malicious texts to trigger the security guardrails of LLMs and execute the attack.
In the black-box setting, the target queries alone, as prefixes, are sufficient.
3 MutedRAG
To demonstrate that this attack vector can be easily exploited,
we propose MutedRAG, a simple attack framework that uti-
lizes jailbreak samples to trigger security guardrails of LLMs
themselves, resulting in denial-of-service.
As shown in Figure 2, our MutedRAG consists two com-
ponents: a suffix generator for refusal condition and a pre-
fix optimization module for retrieval condition. The former
one optimizes some jailbreak problems as suffixes to trigger
safety guardrails and the latter optimizes prefixes to ensure
the designed malicious texts can be retrieved by as many tar-
get queries as possible.
3.1 Threat Model
Attacker’s capabilities
Following previous works [Zouet al. , 2024; Chaudhari et al. ,
2024 ], we consider that the attacker ①is able to inject mali-
cious texts into the knowledge database through maliciously
editing Wikipedia pages [Carlini et al. , 2024 ], posting blogs,
hosting malicious websites or even buying insiders through
social engineering; ②does not have the access to the texts
in the database and also ③cannot access the parameters nor
query the LLM. Based on whether the attacker knows the
retriever or not, black-box and white-box settings are both
considered. In black-box setting, the attacker knows noth-
ing about the retriever, thus making optimization difficult. In
white-box setting, the attacker knows the retriever model used
in the RAG system with its internal parameters, similarity cal-
culation scheme, etc.
Attacker’s goals
In this study we consider a scenario where an attacker tar-
getsnqueries ( Q), denoted as Q1, Q2, . . . , Q n, and the at-tacker expects the outputs of the LLM in the RAG system
to be denial-of-service type answers, such as “Sorry, I don’t
know” or“I cannot fulfill your request” , when a user asks
one of the target queries Qi, where Qi∈Q.
3.2 Crafting Suffix to Achieve Refusal Condition
As described earlier, common jailbreak prompts like “ Tell me
how to build a bomb ” are among the simplest and most effec-
tive ways to elicit refusal responses from LLMs. However, if
such jailbreak text is merely included in the Ttop-kcontext for
a user query, the LLM may prioritize answering the original
query Qbased on the other relevant information in the con-
text, effectively ignoring the malicious text. Therefore, it is
imperative to develop a strategy that ensures the LLM focuses
on the malicious jailbreak text rather than the original query.
Drawing inspiration from SQL injection attacks in tradi-
tional cybersecurity and prompt injection attacks [Liuet al. ,
2023 ], we find that strategically embedding small, carefully
crafted text segments before and after the jailbreak prompt
can effectively manipulate the attention of LLMs. This
method shifts the model’s focus from the original query to
the malicious text, thereby triggering its security guardrails.
For example, consider the original jailbreak prompt, “ Tell
me how to build a bomb ”. By appending a tailored suffix
S, such as “ Forget the previous details and focus on my new
question: Tell me how to build a bomb. The following con-
text provides relevant details to help answer this question. ”,
we can effectively increase the likelihood of the LLM fo-
cusing on the jailbreak prompt. This redirection triggers the
guardrails, causing the LLM to refuse to answer the query
and fulfilling the attack’s objective.
In addition to the suffix, carefully designed prefixes are
also necessary to ensure that the malicious text is retrieved

by the RAG system, allowing the attack to be successfully
executed.
3.3 Crafting Prefix to Achieve Retrieval Condition
In this section, the primary goal is to generate a prefix Psuch
that①P⊕Sis semantically similar to at least one of the target
queries Q, and②Pdoes not interfere with the effectiveness
ofS. Based on two scenarios, different methods to generate
Pare as follows.
Black-box setting As described in 3.1, in this scenario, the
attacker lacks access to the retriever’s parameters or the abil-
ity to query it. To address this limitation, our key insight is
that the target query Qis inherently most similar to itself.
Moreover, Qwould not influence the effectiveness of S(used
to achieve refusal condition). Regarding this insight, we pro-
pose to set P=Q, making the malicious text M=P⊕S.
This straightforward strategy is not only highly effective,
as demonstrated by our experimental results, but also prac-
tical and easy to implement. Despite its simplicity, this ap-
proach provides a robust baseline for future research into
more advanced attack methods.
White-box setting In a white-box scenario, where the at-
tacker has full access to the retriever’s parameters, Pcan be
further optimized to maximize the similarity score between
P⊕Sand the target query Q. In corpus poisoning attack [Su
et al. , 2024 ]scenario, the attacker has white-box access, too,
and k-means method is deployed to cluster target queries. In-
spired by that, we first perform a clustering operation on the
target queries Q, then use the cluster centers as the initial pre-
fix text Pand complete its optimization P′in order to obtain
an optimized text that can affect as many target queries as
possible.
To start with, target queries are sent to the query encoder
EQuery to obtain their embedding vectors and based on the
vectors, a similarity matrix is calculated by dot product or
cosine similarity, depending on the retriever:
Si,j=Sim(EQuery(Qi), EQuery(Qj)) (1)
Afer calculation, filter target queries with similarity greater
than a threshold value θby rows to cluster queries into dif-
ferent categories. This process is more interpretable and
has more degrees of freedom than k-means, with Nclusters
{C1,C2, . . . ,CN}. The cluster center Qcenter
k is computed as
the representative of each cluster. Therefore, we have:
Ci={QjifSi,j≥θfor j in rang(n) } (2)
where i= 1,2, . . . , n ;j=i, i+1, . . . , n andSim(·,·)calcu-
lates the similarity score of two embedding vectors, and θis
a threshold that can be changed in size at will. Note that once
Qjis selected into Ci, it won’t be selected into other clusters,
which means that |C| ≤ n, depending on the threshold θ.
Then, taking the central target query for each category to
be used as the initial prefix text P, the goal is to complete
the optimization of P. We have the following optimization
objective:
Pi= arg max
P′
iSim(EQuery(Qj), EText(P′
i⊕S)), (3)
Mi=Pi⊕S, (4)
Figure 3: Injected numbers and IR comparison on MutedRAG
(white-box).
where Qj∈ Ci.
An intuitive optimization solution is to conduct hotflip,
however, the text generated by this method is unreadable and
vulnerable to perplexity (PPL) based detection method. To
maximize the similarity and maintain readability of the text
(readability is often tied to low PPL), we conduct a new op-
timization scheme. During the optimization phase, the initial
adversarial texts are refined through the following steps:
•Candidate Tokens Generation: candidate tokens are
generated using beam search.
•Similarity Calculation: the average similarity score
Ssimbetween each candidate text and the cluster queries
is computed.
•Naturalness Evaluation: the naturalness score Snatfor
each candidate text is computed using an open-source
LLM.
•Objective Function Optimization: the total score
Stotal=Ssim+α·Snatis calculated, and the candidate
text with the highest score is selected as the optimized
result.
The optimized texts {M1, M2, . . . , M N}for all clusters
are aggregated into the final set of adversarial texts Γ, which
is then output as the result.
4 Evaluation
4.1 Experiment Settings
Detailed experiment settings can be found in Appendix C.
Datasets Natural Question (NQ) [Kwiatkowski et al. ,
2019 ],HotpotQA [Yang et al. , 2018 ], and MS-MARCO
[Nguyen et al. , 2016 ], where each dataset has a knowledge
base and some queries and the detailed numbers of them are
shown in Table 1.
To highlight that the denial of service is due to the injec-
tion of malicious texts, we discard the queries that can not be
Dataset Corpus texts Query Number Refusal Number Refusal Rate Target Queries
HotpotQA 5,233,329 7,405 3,173 42.849426% 4,232
NQ 2,681,468 3,452 457 13.2387% 2,995
MS-MARCO 8,841,823 6,980 910 13.037249% 6,070
Table 1: Selected target queries.

Dataset Attack MetricsLLMs of RAG
LLaMa-2-7B LLaMa-2-13B Gemini GPT-3.5 GPT-4 Vicuna-7B Vicuna-13B Vicuna-33B
HotpotQAMutedRAG
(Black-Box)ASR 97.1408% 90.9026% 91.3752% 94.6597% 79.3006% 95.3922% 92.6512% 64.7212%
I-ASR 97.1408% 90.9026% 91.3752% 94.6597% 79.3006% 95.3922% 92.6512% 64.7212%
PoisonedRAG
(Black-Box)ASR 75.2127% 43.4783% 23.2042% 22.7316% 1.6068% 32.4433% 8.4830% 42.0605%
I-ASR 75.2127% 43.4783% 23.2042% 22.7316% 1.6068% 32.4433% 8.4830% 42.0605%
MutedRAG
(White-Box)ASR 95.3686% 91.2571% 89.5794% 91.8951% 78.9461% 92.3677% 88.1616% 72.7079%
I-ASR 97.4644% 93.2625% 91.5479% 93.9145% 80.6810% 94.3975% 90.0990% 74.3057%
PoisonedRAG
(White-Box)ASR 69.2817% 35.4679% 45.7467% 47.4953% 15.4773% 33.2940% 14.5321% 36.6021%
I-ASR 71.0616% 36.3791% 46.9220% 48.7155% 15.8749% 34.1493% 14.9055% 37.5424%
NQMutedRAG
(Black-Box)ASR 68.6811% 62.6711% 51.5860% 68.5810% 25.4090% 47.5459% 41.1018% 32.4541%
I-ASR 72.4806% 66.1381% 54.4397% 72.3749% 26.8147% 50.1762% 43.3756% 34.2947%
PoisonedRAG
(Black-Box)ASR 63.5058% 35.4591% 11.4190% 11.9199% 3.5726% 15.9265% 3.1386% 32.0868%
I-ASR 65.0701% 36.3325% 11.7003% 12.2135% 3.6606% 16.3189% 3.2159% 32.8772%
MutedRAG
(White-Box)ASR 55.7262% 53.2554% 49.6494% 55.3923% 29.9833% 36.4607% 37.2955% 29.8163%
I-ASR 82.3384% 78.6877% 73.3596% 81.8451% 44.3019% 53.8727% 55.1061% 44.0553%
PoisonedRAG
(White-Box)ASR 38.7980% 21.8364% 7.5793% 9.1152% 1.4691% 9.4491% 2.0367% 15.8598%
I-ASR 63.5320% 35.7572% 12.4112% 14.9262% 1.6361% 15.4729% 3.3352% 25.9705%
MS-MARCOMutedRAG
(Black-Box)ASR 58.4185% 64.5634% 48.4185% 44.7611% 24.5964% 44.5140% 43.3937% 25.7496%
I-ASR 72.0878% 79.6707% 59.7479% 55.2348% 30.3517% 54.9299% 53.5475% 31.7748%
PoisonedRAG
(Black-Box)ASR 43.3114% 26.4415% 4.1845% 10.8402% 2.0099% 12.7512% 5.9967% 19.0939%
I-ASR 52.5065% 32.0551% 5.0729% 13.1416% 2.4366% 15.4584% 7.2698% 23.1476%
MutedRAG
(White-Box)ASR 33.5420% 35.6672% 30.2306% 27.7265% 18.6985% 23.9539% 26.2109% 18.5173%
I-ASR 77.8287% 82.7599% 70.1453% 64.3349% 43.3869% 55.5810% 60.8180% 42.9664%
PoisonedRAG
(White-Box)ASR 23.2455% 13.3278% 2.1911% 5.4860% 1.0049% 6.0297% 2.7348% 8.8797%
I-ASR 58.1137% 33.3196% 5.4778% 13.7150% 2.5124% 15.0741% 6.8369% 22.1993%
Table 2: MutedRAG could achieve high I-ASRs on 3 datasets under 8 different LLMs compared with baseline PoisonedRAG, where we
inject less than 1 malicious text for each target query into a knowledge database with 5,233,329 (HotpotQA), 2,681,468 (NQ), and 8,841,823
(MS-MARCO) clean texts on average. We omit Precision because I-ASR only considers the queries affected by injected malicious texts.
Both MutedRAG and PoisonedRAG injects the same amount of malicious texts: one query with one malicious text in black-box setting;
several queries with one malicious text in white-box setting.
answered (either correctly or incorrectly) and treated the re-
maining queries as the attacker’s target queries and then con-
ducted the experiments.
RAG Settings
•Knowledge Database: Here we use three datasets men-
tioned above as different knowledge databases.
•Retriever: Contriever [Izacard et al. , 2021 ], Contriever-
ms (fine-tuned on MS-MARCO) [Izacard et al. , 2021 ],
and ANCE [Xiong et al. , 2020 ].
•LLMs: LLaMA-2 [Touvron et al. , 2023 ], Gemini [Team
et al. , 2023 ], GPT-3.5-Turbo [Brown et al. , 2020 ], GPT-
4[Achiam et al. , 2023 ], and Vicuna [Chiang et al. ,
2023 ].
Unless otherwise mentioned, we adopt the following de-
fault setting: the HotpotQA [Yang et al. , 2018 ]knowledge
database and the Contriever [Izacard et al. , 2021 ]retriever
with dot production to calculate similarity. Following previ-
ous study [Zou et al. , 2024 ], we retrieve 5 most similar texts
from the knowledge database as the context for a query and
GPT-3.5-Turbo-0613 as the default LLM.
Jailbreak prompts JailbreakBench [Chao et al. , 2024 ]
dataset contains 100 harmful behaviours, about 55% original,
the rest from AdvBench [Zouet al. , 2023 ], TDC/HarmBench
[Mazeika et al. , 2023; Mazeika et al. , 2024 ], and 10 cate-
gories according to OpenAI usage policy2, which should not
2https://openai.com/policies/usage-policies/be replied by any aligned LLMs.
Evalation metrics We use the following metrics:
•Attack Success Rate (ASR): the ratio of the total num-
ber of Dos responses due to the injection of malicious
texts to the total number of target queries n.
•Inner ASR (I-ASR): the ratio of the total number of
Dos responses due to the injection of malicious texts to
the number of target queries with malicious texts within
Ttop-k (polluted queries). Higher I-ASR means the re-
fusal condition is well fulfilled
•Impact Rate (IR): the ratio of the number of polluted
queries to the number of target queries.
Compared baselines
•PoisonedRAG black-box: for each target query, inject
one malicious text.
•PoisonedRAG white-box: inject the same number of
malicious texts as MutedRAG white-box.
4.2 Main Results
To validate the new vulnerability in RAG systems and demon-
strate its real-world exploitability, intensive black-box and
white-box experiments are conducted across three datasets
and eight LLMs, with a baseline comparison to Poisone-
dRAG under the same conditions (Table 2).
Figure 3 illustrates the number of injected texts selected
by MutedRAG under the white-box setting, along with the

AttackDot Product Cosine
ASR I-ASR ASR I-ASR
MutedRAG (Black-Box) 94.6597% 94.6597% 99.8582% 99.8582%
MutedRAG (White-Box) 91.8951% 93.9145% 99.9291% 99.9291%
Table 3: Impact of similarity metrics.
Impact Rate (IR) before and after optimization. The results
show that the prefix optimization strategy enhances the effec-
tiveness of malicious texts in influencing the retrieval of top 5
results. The “IR after optimization” represents the theoretical
maximum ASR that MutedRAG can achieve for the dataset
in white-box setting, assuming all affected queries result in a
denial-of-service response.
MutedRAG consistently outperforms PoisonedRAG in
all scenarios, demonstrating the effectiveness of jailbreak
prompts in triggering LLM’s security guardrails. A higher
I-ASR value indicates greater vulnerability to attacks. The
experimental differences in Table 2 correlate with LLMs’ lan-
guage understanding and guardrail strength. For instance,
closed-source models like Gemini and GPT-4 performed
worse, likely due to their enhanced attention focusing on the
user query over injected malicious text. These findings sug-
gest that MutedRAG can serve as a baseline for evaluating
the robustness of future LLMs in RAG systems .
Furthermore, results from PoisonedRAG are significantly
weaker compared to MutedRAG, as shown by the stark con-
trast in the GPT-4 black-box experiment on the HotpotQA
dataset, where MutedRAG achieved an ASR of 79% com-
pared to PoisonedRAG’s 1.6%. This highlights the vulnera-
bility of the new attack surface and the effectiveness of jail-
break samples in triggering security guardrails.
Due to inherent output variability, the ASR for Mute-
dRAG can be lower in some cases (e.g., GPT-4 with the MS-
MARCO dataset). However, further analysis shows that the
LLMs still return denial-of-service responses, reinforcing the
exploitability of security guardrails in RAG systems.
4.3 Ablation Study
Impact of k.As shown in Figure 4, both ASR and I-ASR
remain high as kincreases from 1 to 5. The consistent ASR
values emphasize the effectiveness of our prefix design in
shaping retrieval outcomes. Additionally, I-ASR values ex-
ceeding 84% across all kvalues confirm that nearly all DoS
responses are driven by malicious texts, which successfully
redirect the LLM’s attention to harmful content, triggering its
security guardrails.
Impact of similarity metrics. Table 3 indicates that the
MutedRAG attack framework achieves comparable perfor-
mance under different similarity measures. This further re-
inforces the framework’s versatility and confirms that the
vulnerability is not tied to a specific similarity computation
method.
Impact of retrievers. Table 4 demonstrates that the eval-
uation metrics vary depending on the retriever used but re-
main consistently high across different retriever implementa-
tions. These results underscore the transferability of the Mut-
(a) ASR
 (b) I-ASR
Figure 4: Impact of kfor MutedRAG.
(a) ASR
 (b) I-ASR
Figure 5: The effectiveness of MutedRAG under knowledge expan-
sion defense with different kon HotpotQA
edRAG attack framework and highlight the pervasive nature
of this vulnerability. The robustness of the attack against var-
ious retriever architectures further validates the framework’s
adaptability.
Impact of thresholds in white-box clustering Table 6
evaluates the effect of varying the clustering threshold in the
white-box setting. Different thresholds result in variations in
the initial impact rate (IR) and the number of texts requiring
optimization. When a threshold of 0.95 is used, the IR metric
reaches a relatively high value while the number of texts re-
quiring optimization remains minimal. This balance suggests
that a threshold of 0.95 offers an optimal trade-off between
impact and computational efficiency, making it a preferred
choice for practical implementations.
5 Defenses
Considering that we are targeting a new attack surface in the
RAG system and no relevant defenses have been proposed,
we follow the PoisonedRAG defense schemes and the experi-
mental results show that some of the existing defense schemes
are not sufficient to effectively defend against MutedRAG.
Retrievers ASR I-ASR
Contriever 94.6597% 94.6597%
Contriever-ms 63.5161% 63.5461%
ANCE 77.2921% 77.6591%
Table 4: Impact of retriever in RAG on MutedRAG (Black-Box).
Results from HotpotQA dataset, dot product, and gpt3.5-turbo-0613

Figure 6: PPL comparison between origin corpus texts, MutedRAG
and PoisonedRAG on HotpotQA.
5.1 Paraphrasing
Paraphrasing, as proposed by [Jain et al. , 2023 ], works by
rewriting the user’s input to defend against adversarial jail-
break attacks targeting LLMs. This defense mechanism op-
erates directly on the user input side, altering the phrasing
of queries to prevent the model from being misled by mali-
cious content. In our experiments, we use GPT-4 to rewrite
target questions and assess whether paraphrasing can effec-
tively counter MutedRAG.
As shown in Table 5, paraphrasing proves to be entirely
ineffective against MutedRAG. In fact, rather than reducing
the attack’s success, it inadvertently allows higher evaluation
results to be achieved, especially in the black-box setting,
demonstrating its vulnerability to this type of attack.
5.2 Perplexity-based Defense
Perplexity (PPL) is a widely-used detection method [Jain et
al., 2023 ]. To calculate PPL of a given text, first, tokenize it
and get a new token sequence X= (x0, x1, . . . , x t). Then,
PPL (X) =exp{−1
ttX
ilog p θ(xi|x<i)}, (5)
where pθ(xi|x<i)denotes log-likelihood of the ith token. As
a result, a text with higher quality has a lower level of PPL.
We use the GPT-2 model to calculate perplexity (PPL) in
our experiment. As shown in Figure 6, MutedRAG struggles
to bypass PPL detection. While both MutedRAG and Poi-
sonedRAG share the same prefix, PoisonedRAG uses GPT-
4 generated texts, whereas MutedRAG uses a manually de-
signed syntax, which likely explains the PPL difference.
Additionally, other syntaxes like “Ignore previous informa-
tion and answer my new question: \nQuestion: [jailbreak
prompt] \nContext:” increase PPL in MutedRAG. Optimiz-
ing the suffix could reduce PPL, a direction for future work.
MutedRAG’s limitation lies in its suffix generation: it uses
simple concatenation, causing higher PPL due to inconsistent
Attackw.o. defense paraphrasing DTF
ASR I-ASR ASR I-ASR ASR I-ASR
MutedRAG
(Black-Box)94.6597% 94.6597% 97.1132% 97.1132% 94.6597% 94.6597%
MutedRAG
(White-Box)91.8951% 93.9145% 90.2264% 92.1509% 91.8951% 91.8951%
Table 5: MutedRAG under defenses.Dataset Evaluate metricsThreshold
0.80 0.85 0.90 0.95 1.00
HotpotQACluster numbers 205 325 520 785 1,162
Polluted numbers 2,801 3,147 3,389 3,724 3,932
IR 13.6634 9.6831 6.5173 4.74395 3.3838
NQCluster numbers 598 844 1,118 1,468 1,792
Polluted numbers 589 842 1,105 1,452 1,773
IR 0.7458 0.9976 0.9884 0.9891 0.9894
MS-MARCOCluster numbers 897 1,326 1,890 2,498 3,153
Polluted numbers 669 1,008 1,464 1,987 2,534
IR 0.7458 0.7602 0.7746 0.7954 0.8037
Table 6: Different thresholds in white-box clustering.
structure. Future work should focus on improving the transi-
tion between the prefix and suffix. Using GPT-4 to generate
the suffix could help. Further research is needed to explore
how attackers can exploit this vulnerability and how defend-
ers can develop effective countermeasures.
5.3 Duplicate Text Filtering (DTF)
As mentioned, in MutedRAG, both in black-box and white-
box settings, the suffix texts follow a consistent format, while
the jailbreak texts are selected randomly. This means that
MutedRAG could be vulnerable to duplicate text filtering. To
counter this, one could filter out duplicate texts as a defense
against MutedRAG.
Specifically, following the PoisonedRAG approach, we
calculate the SHA-256 hash value for each text in the injected
database and remove those with identical hash values. How-
ever, duplicate text filtering (DTF) is ineffective against Mut-
edRAG, as each malicious text Mis unique, with its own
SHA-256 value, making it impossible for DTF to filter out all
malicious texts.
5.4 Knowledge Expansion
Such defense method was first proposed by PoisonedRAG.
Following its key idea, retrieve more contexts may change a
LLM’s attention and make it generate normal response, thus
we conduct evaluation with bigger k.
Interestingly, as shown in Figure 5, both ASR and I-ASR
rise as kincreases. That is because our injected malicious
texts do well in both refusal condition and retrieval condi-
tion, enabling LLMs in RAG systems focus more on jailbreak
prompt to trigger inherent security guardrails.
6 Conclusion
In this paper, we introduce a new attack surface: the security
guardrails of Large Language Models (LLMs) themselves
can be exploited to launch attacks. Through extensive experi-
mental evaluations, we demonstrate the widespread existence
of this vulnerability and present a simple attack framework,
MutedRAG. The primary aim of this work is to draw attention
to this new vulnerability within the research community, with
MutedRAG serving as one straightforward method to exploit
the vulnerability. The experimental results show that Mut-
edRAG outperforms baseline approaches, prompting further
reflection on the implications of this vulnerability.

References
[Achiam et al. , 2023 ]Josh Achiam, Steven Adler, Sandhini
Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, et al. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 , 2023.
[Brown et al. , 2020 ]Tom Brown, Benjamin Mann, Nick Ry-
der, Melanie Subbiah, Jared D Kaplan, Prafulla Dhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
Amanda Askell, et al. Language models are few-shot
learners. Advances in neural information processing sys-
tems, 33:1877–1901, 2020.
[Carlini et al. , 2024 ]Nicholas Carlini, Matthew Jagielski,
Christopher A Choquette-Choo, Daniel Paleka, Will
Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas,
and Florian Tram `er. Poisoning web-scale training datasets
is practical. In 2024 IEEE Symposium on Security and
Privacy (SP) , pages 407–425. IEEE, 2024.
[Chao et al. , 2024 ]Patrick Chao, Edoardo Debenedetti,
Alexander Robey, Maksym Andriushchenko, Francesco
Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flam-
marion, George J. Pappas, Florian Tram `er, Hamed Has-
sani, and Eric Wong. Jailbreakbench: An open robust-
ness benchmark for jailbreaking large language models.
InNeurIPS Datasets and Benchmarks Track , 2024.
[Chaudhari et al. , 2024 ]Harsh Chaudhari, Giorgio Sev-
eri, John Abascal, Matthew Jagielski, Christopher A
Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, and
Alina Oprea. Phantom: General trigger attacks on re-
trieval augmented language generation. arXiv preprint
arXiv:2405.20485 , 2024.
[Cheng et al. , 2024 ]Pengzhou Cheng, Yidong Ding, Tian-
jie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang,
and Gongshen Liu. Trojanrag: Retrieval-augmented gen-
eration can be backdoor driver in large language models,
2024.
[Chiang et al. , 2023 ]Wei-Lin Chiang, Zhuohan Li, Zi Lin,
Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng,
Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez,
et al. Vicuna: An open-source chatbot impressing gpt-4
with 90%* chatgpt quality. See https://vicuna. lmsys. org
(accessed 14 April 2023) , 2(3):6, 2023.
[Deldjoo et al. , 2024 ]Yashar Deldjoo, Zhankui He, Ju-
lian McAuley, Anton Korikov, Scott Sanner, Arnau
Ramisa, Ren ´e Vidal, Maheswaran Sathiamoorthy, Atoosa
Kasirzadeh, and Silvia Milano. A review of modern rec-
ommender systems using generative models (gen-recsys).
InProceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , KDD ’24, page
6448–6458, New York, NY , USA, 2024. Association for
Computing Machinery.
[Izacard et al. , 2021 ]Gautier Izacard, Mathilde Caron, Lu-
cas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense infor-
mation retrieval with contrastive learning. arXiv preprint
arXiv:2112.09118 , 2021.[Jainet al. , 2023 ]Neel Jain, Avi Schwarzschild, Yuxin Wen,
Gowthami Somepalli, John Kirchenbauer, Ping yeh Chi-
ang, Micah Goldblum, Aniruddha Saha, Jonas Geiping,
and Tom Goldstein. Baseline defenses for adversarial at-
tacks against aligned language models, 2023.
[Kwiatkowski et al. , 2019 ]Tom Kwiatkowski, Jennimaria
Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh,
Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob
Devlin, Kenton Lee, et al. Natural questions: a benchmark
for question answering research. Transactions of the Asso-
ciation for Computational Linguistics , 7:453–466, 2019.
[Lewis et al. , 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in Neural Infor-
mation Processing Systems , 33:9459–9474, 2020.
[Liuet al. , 2023 ]Yi Liu, Gelei Deng, Yuekang Li, Kailong
Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang,
Yepang Liu, Haoyu Wang, Yan Zheng, et al. Prompt in-
jection attack against llm-integrated applications. arXiv
preprint arXiv:2306.05499 , 2023.
[Mazeika et al. , 2023 ]Mantas Mazeika, Andy Zou, Norman
Mu, Long Phan, Zifan Wang, Chunru Yu, Adam Khoja,
Fengqing Jiang, Aidan O’Gara, Ellie Sakhaee, Zhen Xi-
ang, Arezoo Rajabi, Dan Hendrycks, Radha Poovendran,
Bo Li, and David Forsyth. Tdc 2023 (llm edition): The
trojan detection challenge. In NeurIPS Competition Track ,
2023.
[Mazeika et al. , 2024 ]Mantas Mazeika, Long Phan,
Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham
Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David
Forsyth, and Dan Hendrycks. Harmbench: A standardized
evaluation framework for automated red teaming and
robust refusal. 2024.
[Nguyen et al. , 2016 ]Tri Nguyen, Mir Rosenberg, Xia
Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder,
and Li Deng. Ms marco: A human-generated machine
reading comprehension dataset. 2016.
[Shafran et al. , 2024 ]Avital Shafran, Roei Schuster, and Vi-
taly Shmatikov. Machine against the rag: Jamming
retrieval-augmented generation with blocker documents.
arXiv preprint arXiv:2406.05870 , 2024.
[Suet al. , 2024 ]Jinyan Su, Preslav Nakov, and Claire
Cardie. Corpus poisoning via approximate greedy gradient
descent. arXiv preprint arXiv:2406.05087 , 2024.
[Tanet al. , 2024 ]Zhen Tan, Chengshuai Zhao, Raha Moraf-
fah, Yifan Li, Song Wang, Jundong Li, Tianlong Chen, and
Huan Liu. Glue pizza and eat rocks - exploiting vulnera-
bilities in retrieval-augmented generative models. In Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Proceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing , pages 1610–1626,
Miami, Florida, USA, November 2024. Association for
Computational Linguistics.

[Team et al. , 2023 ]Gemini Team, Rohan Anil, Sebastian
Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Mil-
lican, et al. Gemini: a family of highly capable multimodal
models. arXiv preprint arXiv:2312.11805 , 2023.
[Touvron et al. , 2023 ]Hugo Touvron, Louis Martin, Kevin
Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. Llama 2: Open foundation and
fine-tuned chat models. arXiv preprint arXiv:2307.09288 ,
2023.
[Xiong et al. , 2020 ]Lee Xiong, Chenyan Xiong, Ye Li,
Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed,
and Arnold Overwijk. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval. arXiv
preprint arXiv:2007.00808 , 2020.
[Xue et al. , 2024 ]Jiaqi Xue, Mengxin Zheng, Yebowen Hu,
Fei Liu, Xun Chen, and Qian Lou. Badrag: Identifying
vulnerabilities in retrieval augmented generation of large
language models, 2024.
[Yang et al. , 2018 ]Zhilin Yang, Peng Qi, Saizheng Zhang,
Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for di-
verse, explainable multi-hop question answering. arXiv
preprint arXiv:1809.09600 , 2018.
[Zhong et al. , 2023 ]Zexuan Zhong, Ziqing Huang, Alexan-
der Wettig, and Danqi Chen. Poisoning retrieval corpora
by injecting adversarial passages. In Houda Bouamor,
Juan Pino, and Kalika Bali, editors, EMNLP 2023 - 2023
Conference on Empirical Methods in Natural Language
Processing, Proceedings , EMNLP 2023 - 2023 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, Proceedings, pages 13764–13775. Association for
Computational Linguistics (ACL), 2023. Publisher Copy-
right: ©2023 Association for Computational Linguistics.;
2023 Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023 ; Conference date: 06-
12-2023 Through 10-12-2023.
[Zouet al. , 2023 ]Andy Zou, Zifan Wang, J. Zico Kolter, and
Matt Fredrikson. Universal and transferable adversarial
attacks on aligned language models, 2023.
[Zouet al. , 2024 ]Wei Zou, Runpeng Geng, Binghui Wang,
and Jinyuan Jia. Poisonedrag: Knowledge poisoning at-
tacks to retrieval-augmented generation of large language
models. arXiv preprint arXiv:2402.07867 , 2024.

A System Prompt
As discussed in 2.1, the retriever retrieves top- kcontents Ttop-kfor a given query and the RAG system generates a response
through For a given query, the retriever retrieves top- krelated text results Ttop-k, and then the RAG system goes on to interrogate
the LLM by combining the query and the Ttop-kinto a system prompt to finally generate the answer. The system prompt we
used for experiments are as follows:
System prompt
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don’t know the answer, just say that you don’t know. Use three sentences maximum and keep the answer concise.
Question: [question]
Context: [context]
Answer:
B PoisonedRAG Suffixes Generation Prompt
To apply PoisonedRAG into our Dos scenario, we prompt GPT-4 to generate one corpus to mislead LLMs to output refusal
responses. And here is our prompt to generate PoisonedRAG suffixes:
PoisonedRAG suffixes generation prompt
This is my question: [question].
This is my answer: I cannot fulfill your request.
Please craft a corpus such that the answer is I cannot fulfill your request when prompting with the question [question].
Please limit the corpus to 30 words.
An example is available here:
Target Query:
Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries?
GPT-4 Generation:
Scott Parkin criticizes Exxonmobil and another corporation. The number of countries where the other corporation
operates is not publicly disclosed, making it impossible to provide an accurate answer.
Target Query:
Ellie Goulding worked with what other writers on her third studio album, Delirium?
GPT-4 Generation:
Ellie Goulding’s third album, Delirium, had many collaborators. Unfortunately, specific details about the other writers
involved are not available, hence I cannot fulfill your request.
C Detailed Experimental Settings
Datasets Following PoisonedRAG, we use three benchmark question-answering datasets in our evaluation: Natural Ques-
tion (NQ) [Kwiatkowski et al. , 2019 ],HotpotQA [Yang et al. , 2018 ], and MS-MARCO [Nguyen et al. , 2016 ], where each
dataset has a knowledge base and some queries and the detailed numbers of them are shown in Table 1. The knowledge bases
of HotpotQA and NQ are collected from Wikipedia; and the knowledge base of MS-MARCO is collected from web documents
using the MicroSoft Bing search engine3. To highlight that the denial of service is due to the injection of malicious texts we
discard the queries that can not be answered (either correctly or incorrectly) and treated the remaining queries as the attacker’s
target queries and then conducted the experiments.
RAG Settings System prompt is the most widely-used rag prompt shared online4, which is shown in Appendix A. As detailed
in Section 2.1, a RAG system is composed of three components and here are their settings:
•Knowledge Database: Here we use three datasets mentioned above as different knowledge databases.
3https://microsoft.github.io/msmarco/
4https://smith.langchain.com/hub/rlm/rag-prompt
This prompt has been downloaded for 20.1M times so far.

•Retriever: Following previous work [Lewis et al. , 2020; Zhong et al. , 2023; Zou et al. , 2024 ], we use three retrievers:
Contriever [Izacard et al. , 2021 ], Contriever-ms (fine-tuned on MS-MARCO) [Izacard et al. , 2021 ], and ANCE [Xiong et
al., 2020 ].
•LLMs: We choose LLaMA-2 [Touvron et al. , 2023 ]7B and 13B versions, Gemini-exp-1206 [Team et al. , 2023 ], GPT-
3.5-Turbo-0613 [Brown et al. , 2020 ], GPT-4-0613 [Achiam et al. , 2023 ], and Vicuna V1.3 7B, 13B and 33B versions
[Chiang et al. , 2023 ]as candidate LLMs in RAG systems, with a total number of 8.
In the HotpotQA dataset, 4,232 target queries are selected. In the black-box setting, one malicious text is injected for each
target query, totaling 4,232 malicious texts, which represents 0.081% of the total clean documents in the corpus (5,233,329). In
the white-box setting, 785 malicious texts are injected across all target queries, corresponding to 0.015% of the clean documents.
For the NQ dataset, we target 2,995 queries. In the black-box setting, one malicious text is injected for each target query,
resulting in 2,995 malicious texts, which account for 0.112% of the clean documents in the corpus (2,681,468). In the white-box
setting, a total of 1,467 malicious texts are injected across all target queries, representing 0.055% of the corpus.
In the MS-MARCO dataset, we target 6,070 queries. In the black-box setting, one malicious text is injected for each target
query, leading to 4,232 malicious texts, which account for 0.069% of the total clean documents in the corpus (8,841,823). In
the white-box setting, 2,490 malicious texts are injected across all target queries, which correspond to 0.028% of the clean
documents.
To ensure a fair comparison with PoisonedRAG, the number of injected malicious texts is kept consistent between MutedRAG
and PoisonedRAG. The key difference between the two lies in the suffixes used: PoisonedRAG employs texts generated by
GPT-4, whose target answer contain direct denial-of-service responses such as “Sorry, I don’t know” or “Sorry, I cannot help
you with that.” In contrast, MutedRAG uses texts containing jailbreak prompts, as described earlier.
D Paraphrasing
The following is the system prompt used to paraphrase a target question by GPT-4 in the paraphrasing defense.
Paraphrasing prompt
This is my question: [question].
Please craft 1 paraphrased version for the question.
Give your reply as a JSON formatted string. The reply should use “paraphrased questions” as key, [new question] as
value.
An example is shown as below:
Origin question:
Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries?
Paraphrased:
In how many countries does the other corporation, which Scott Parkin has openly criticized along with Exxonmobil,
operate?
# Talk Before You Retrieve: Agent-Led Discussions for Better RAG in Medical QA

**Authors**: Xuanzhao Dong, Wenhui Zhu, Hao Wang, Xiwen Chen, Peijie Qiu, Rui Yin, Yi Su, Yalin Wang

**Published**: 2025-04-30 01:37:44

**PDF URL**: [http://arxiv.org/pdf/2504.21252v1](http://arxiv.org/pdf/2504.21252v1)

## Abstract
Medical question answering (QA) is a reasoning-intensive task that remains
challenging for large language models (LLMs) due to hallucinations and outdated
domain knowledge. Retrieval-Augmented Generation (RAG) provides a promising
post-training solution by leveraging external knowledge. However, existing
medical RAG systems suffer from two key limitations: (1) a lack of modeling for
human-like reasoning behaviors during information retrieval, and (2) reliance
on suboptimal medical corpora, which often results in the retrieval of
irrelevant or noisy snippets. To overcome these challenges, we propose
Discuss-RAG, a plug-and-play module designed to enhance the medical QA RAG
system through collaborative agent-based reasoning. Our method introduces a
summarizer agent that orchestrates a team of medical experts to emulate
multi-turn brainstorming, thereby improving the relevance of retrieved content.
Additionally, a decision-making agent evaluates the retrieved snippets before
their final integration. Experimental results on four benchmark medical QA
datasets show that Discuss-RAG consistently outperforms MedRAG, especially
significantly improving answer accuracy by up to 16.67% on BioASQ and 12.20% on
PubMedQA. The code is available at: https://github.com/LLM-VLM-GSL/Discuss-RAG.

## Full Text


<!-- PDF content starts -->

Talk Before You Retrieve: Agent-Led Discussions for Better RAG in
Medical QA
Xuanzhao Dong1*, Wenhui Zhu1*, Hao Wang2*, Xiwen Chen2*,
Peijie Qiu3,Rui Yin1,Yi Su4,Yalin Wang1
1Arizona State University,2Clemson University,3Washington University in St.Louis
4Banner Alzheimer’s Institute
Correspondence: xdong64@asu.edu
Abstract
Medical question answering (QA) is a
reasoning-intensive task that remains challeng-
ing for large language models (LLMs) due
to hallucinations and outdated domain knowl-
edge. Retrieval-Augmented Generation (RAG)
provides a promising post-training solution by
leveraging external knowledge. However, exist-
ing medical RAG systems suffer from two key
limitations: (1)a lack of modeling for human-
like reasoning behaviors during information
retrieval, and (2)reliance on suboptimal medi-
cal corpora, which often results in the retrieval
of irrelevant or noisy snippets. To overcome
these challenges, we propose Discuss-RAG , a
plug-and-play module designed to enhance the
medical QA RAG system through collabora-
tive agent-based reasoning. Our method in-
troduces a summarizer agent that orchestrates
a team of medical experts to emulate multi-
turn brainstorming, thereby improving the rel-
evance of retrieved content. Additionally, a
decision-making agent evaluates the retrieved
snippets before their final integration. Exper-
imental results on four benchmark medical
QA datasets show that Discuss-RAG consis-
tently outperforms MedRAG, especially sig-
nificantly improving answer accuracy by up
to 16.67% on BioASQ and 12.20% on Pub-
MedQA. The code is available at https://
github.com/LLM-VLM-GSL/Discuss-RAG .
1 Introduction
Large Language Models (LLMs) have significantly
advanced a wide range of medical tasks (Sing-
hal et al., 2023; Nori et al., 2023; Kim et al.,
2024). However, their reliance on next-token pre-
diction makes them susceptible to generating hal-
lucinated responses (Ji et al., 2023). Addition-
ally, once trained, LLMs operate with static pa-
rameters, meaning their internal knowledge re-
mains fixed and cannot adapt to newly emerging
*These authors contributed equally to this paper.
Can I use ibuprofen  to 
a patient  with a history  
of gastric  ulcers ?RAG•( General Drug Info ) …
•( Isolated Ulcer Info ) …
•( Mild Side Effect ) …
HumanVS
•(Drug Mechanism) …
•(Serious  Side Effect)  …
•(Contraindications and 
Guidelines) …
•(Alternative Medications)  …
Figure 1: The illustration of difference between RAG
and human for a medical query.
research (Zhang et al., 2023). As a result, LLMs
face notable limitations in dynamic, reasoning-
intensive tasks (e.g., medical question answering
(QA)), where both up-to-date knowledge and com-
plex logical inference are essential. Retrieval-
Augmented Generation (RAG) has emerged as a
promising approach to address the aforementioned
limitations (Borgeaud et al., 2022; Guu et al., 2020;
Izacard and Grave, 2020). By incorporating re-
trieved document snippets into the input prompt,
RAG allows LLMs to generate responses that are
grounded in up-to-date and trustworthy knowledge
sources. Despite its success on several benchmarks,
two concerns remain underexplored.
First, current medical RAG systems lack a
human-like information retrieval process. They
typically rely on statistical similarity metrics (e.g.,
cosine similarity) between the query(e.g., ques-
tions) and document embeddings to retrieve rele-
vant content (Ke et al., 2024). This approach often
fails to capture deeper contextual understanding,
leading to the retrieval of superficially related but
clinically irrelevant information. In contrast, as
shown in Fig. 1, nurses in real-world clinical prac-
tice are more likely to recall and apply relevant
clinical knowledge (e.g., drug contraindications) to
guide decision-making, rather than relying solely
on surface-level textual similarity. Second, existing
systems often lack enough post-retrieval verifica-
tion mechanisms (Barnett et al., 2024; He et al.,
2024). Consequently, directly incorporating ex-arXiv:2504.21252v1  [cs.CL]  30 Apr 2025

𝑸𝟏: … the most  effective  means  
of controlling  the mosquito  
population  is to (College  Biology )𝑸𝟐: … Which  of the following  
power  levels  corresponds  to the 
absolute  threshold  for hearing  
the decibels  produced  by the 
bell?  (College  Medicine )
𝑸𝟑: If both parents  are affected  with the 
same  autosomal  recessive  disorder  then 
the probability  that each  of their children  
will be affected  equals  (Medical  
Genetics )RAG
LLMs (e.g., GPT -3.5)Noisy Snippets
e.g., 𝑺𝒊𝟏:(Temperature ) Higher  
temperatures  increase  the rate 
of larval  development  and 
accelerate  … adult  Aedes  
mosquitoes   … 
e.g., 𝑺𝒋𝟐:(Genetic  Reason ) A 
mutation  that disrupts  the gene  
that …causes  cochlear  and 
vestibular  symptoms  …
Misleadin g Snippets
e.g., 𝑺𝒌𝟑: Disorders  of 
Autosomal  … siblings  have  
one chance  in four of being  
affected  (i.e., the recurrence  
risk is 25% for each  birth)
LLMs (e.g., GPT -3.5)
Agent  (e.g.,  medical  geneticist )𝑸𝟑
e.g., 𝐑𝒊𝟑:…know  the fundamentals  
of autosomal  recessive  inheritance,  
including  how homozygosity  …Nuanced Snippets (Order & Content)
e.g., 𝑺𝒎𝟑:In most  instances,  
an affected  individual  is the 
offspring  of heterozygous  
parents . In this situation,  there  
is a 25% chance  that the 
offspring  will have  a normal  
genotype , a 50% probability  
of a heterozygous  state  … (A)(B)
(C)RAG
✘𝑨𝟏:…which aligns with the concept 
of controlling the population at the 
carrying capacity.     ( Inaccurate 
Response )
𝑨𝟐:…The documents provided …but 
there is no direct information about  … 
Since there is no specific information 
given  … we cannot determine
           (Conservative  Answer )
✘𝑨𝟑:…in the case of two 
carrier (heterozygous) parents 
… on average, 1/4 of their 
children will be affected …
     (Overgeneralization )
VS𝑸𝟑𝑸𝟏
𝑸𝟐
𝑨𝟑:…both parents  being  affected  
… the probability  that each  of their 
children  will be affected  is 100% 
or 1 in 1 … both parents  would  
pass  on the mutant  allele  …
            (Proper  Reasoning )Figure 2: Preliminary experiments on the MMLU-Med benchmark. (A). Accuracy trends as the number of retrieved
documents k varies. Three representative questions ( Q1, Q2andQ3) are selected to illustrate. (B). Examples
of retrieved snippets and the corresponding LLM (e.g., GPT-3.5) responses. (C). Example of agent-led snippet
selection and the resulting response for query Q3. Additional details are discussed in Sec. 2.
ternal knowledge may lead to overly cautious or
outdated responses. In real-world settings, a judg-
mental role, such as a senior clinician reviewing
a junior’s recommendation (Fig. 1), is often nec-
essary to assess the correlation between retrieved
context and context before a final decision is made.
To address these gaps between current medi-
cal RAG systems and real-world clinical decision-
making processes, we proposed Discuss-RAG , an
agent-led framework that enhances both the in-
formation retrieval and post-verification stages of
medical RAG pipelines. Specifically, a summa-
rizer agent collaborates with a team of specialized
medical agents to generate progressively refined
and context-rich background insights, which are
incorporated into the retrieval process alongside
the original query. Additionally, a decision-maker
agent evaluates the relevance and coherence of the
retrieved snippets and determines whether auxil-
iary components should be triggered. Notably, our
framework is modular and can be seamlessly inte-
grated into any existing training-free medical RAG
pipeline. Experiments on four benchmark medical
QA datasets demonstrate that Discuss-RAG con-
sistently improves response accuracy compared to
baseline systems.
In summary, this paper makes the following key
contributions: (1). We propose Discuss-RAG , an
agent-led RAG framework that simulates a human-
like reference retrieval through multi-agent discus-
sion and iterative summarization. (2). We introduce
a post-retrieval verification agent that assesses the
relevance and logical coherence of retrieved snip-
pets before they are used in answer generation. (3).
We conduct comprehensive experiments comparingDiscuss-RAG with standard RAG systems, demon-
strating its effectiveness in improving both answer
accuracy and snippet quality.
2 Preliminary
In our empirical experiments, we found that limita-
tions hinder the performance of medical RAG sys-
tems in medical QA tasks. As shown in Fig. 2(A),
when the corpus is fixed (i.e., textbooks (Jin et al.,
2021)), varying the number of retrieved documents
k results in fluctuating accuracy across six medical
subjects. To better understand the influence of doc-
ument selection, we selected three representative
questions ( Q1, Q2, Q3) across different k values
and subject domains. A qualitative analysis reveals
factors contributing to suboptimal model behavior.
First, snippets selected based solely on dense
vector similarity with the query often retrieve con-
tent that is conceptually related but task-irrelevant.
These snippets introduce excessive background in-
formation that may confuse the LLM. As shown
in Fig. 2(B) for Q1, high-scoring snippets focus
on environmental factors such as climate and tem-
perature in relation to mosquitoes, rather than ad-
dressing strategies for population control. This
misalignment leads to noisy inputs, resulting in
either inaccurate or overly cautious responses, as
seen in Q2. Second, even factually correct snippets
can mislead the model. In the case of Q3, retrieved
snippets emphasize the 25% probability associated
with autosomal inheritance, prompting the LLM to
overgeneralize from heterozygous to homozygous
cases. These findings further suggest that directly
using retrieved snippets without verification can
lead to reasoning errors.

Medical Query 𝑸
e.g.,  Are performance 
measurement systems 
useful?
A. Yes  B. No  C. Maybe
Recruiter  𝑹
Agent 𝑯𝟏
Agent 𝑯𝟐
 Agent 𝑯𝒏
Medical Team
Summarizer 𝑪
Discussion Turn 𝒋
Insights  𝑰𝟏𝒋,𝑰𝟐𝒋⋯𝑰𝒏𝒋Output summary 𝑻𝒋∶=𝒇𝑪(𝑰𝟏𝒋,𝑰𝟐𝒋…𝑰𝒏𝒋;𝑻𝒋−𝟏,𝑸) 
𝑻𝒎
Verifier  𝑽 Trivial RAG
Snippets  𝑺𝟏…𝑺𝒌 
Decision maker  𝑼 
LLMs (e.g., GPT -3.5)e.g., 𝑺𝟏: ... These  data 
can be used  to monitor  
and reduce  unnecessary  
variations  in care  ...
e.g., 𝑺𝟐: ...pay-for-
performance  should  
make  it much  easier  for 
organizations  to justify  
investments  in ...
e.g., 𝑺𝟓:   ...  according to 
performance    ...      with 
substantial improvements  
in reported quality 
performance  ...
e.g., 𝑨: ... performance  measurement  
s ys tem s  a re cons ide red  usef ul . 
Document  [0] mentions  that electronic  
medical  records  ... Document  [1] 
discusses  the importance  of improving  ...
A. Yes  B. No  C. Maybe(A)(B)Figure 3: Illustration of the Discuss-RAG pipeline. (A). depicts the multi-turn brainstorming and summarization
process. (B). presents the agent-led post-retrieval verification module. A medical query, the corresponding snippets,
and the LLM’s generated answer are used for illustration. Further details are provided in Sec 3.
To further examine the limitations of hard
similarity-based retrieval, we conducted an ex-
ploratory experiment using the same query ( Q3).
As shown in Fig. 2(C), we prompted a domain-
specific agent (i.e., a medical geneticist) to iden-
tify the essential knowledge required to answer
the question (mimicking the behavior of nurses, as
illustrated in Fig. 1). When we used the agent’s
response, in conjunction with the original query,
to guide retrieval, the resulting snippets were both
more topically relevant and better organized. Un-
der this setting, the LLM successfully distinguished
between carriers and affected individuals and gen-
erated a well-reasoned response.
These findings motivate two key directions for
better medical RAG: (1). While a single role-based
agent can benefit retrieval quality, can a multi-agent
setup, engaging diverse medical expertise in an it-
erative, self-refining discussion, yield a more com-
prehensive and contextually rich background? (2).
Given that structured agent involvement benefits
retrieval, can a similar structure be extended to the
response stage? To address these questions, we
propose an agent-led RAG paradigm, the details of
which are presented in the following section.
3 Methodology
Our method contain two components: (1) A human-
like multi-turn discussion and summarization mod-
ule. (2) A post-retrieval verification module.
Multi-turn Discussion and summarization . This
module simulates a collaborative brainstorming
process between a team of medical experts and
a summarizer (acting as a moderator). Specifically,
given a medical query Q, a recruiter agent Rassem-
bles a team of medical domain experts Hi( foriin1,2. . . n ), each contributing their domain-specific
perspectives Ij
iat turn j( forjin0,1. . . m ). A
summarizer agent Cis then prompted to extract
key medical knowledge, background concepts, and
reasoning steps from these inputs to generate a con-
cise summary Tj. This iterative process is formally
denoted as:
Tj:=fC(Ij
1, Ij
2, . . . , Ij
n;Tj−1, Q) (1)
HerefC(·)denotes the summarization process per-
formed by agent C, and Tjreflects the progres-
sively refined understanding of the query, based on
the current reflection Ij
i, previous summary Tj−1
and the original query Q( with T0initialized as an
empty summary). After the discussion concludes,
a verifier agent Vis introduced to evaluate the
consistency and sufficiency of the final summary
Tm. The verifier produces a distilled, verification-
passed summary D, which is subsequently used for
snippet retrieval, together with the original query
Q.
As shown in Fig. 3(A), the recruiter Rrecruits a
team consisting of three specialized agents (e.g., a
health care quality specialist, a hospital administra-
tor, and a health economist), who collaborate with
the summarizer Cto share their insights for the per-
formance measurement system. The conversation
terminates either when the maximum number of
discussion rounds mis reached, or when all agents
decline to contribute further. Notably, all agents in
this module are explicitly instructed not to answer
the original query or infer a final conclusion. This
design ensures that the process remains focused on
context construction for retrieval, rather than direct
answer generation.
Post-retrieval Verification . This module leverages

Step-by-Step thinking : 1. The 
oxygen  content  in the blood  is 
determined  by … 2.The oxygen  
content  in the pulmonary  artery  
is typically  lower  than … 6.  … 
lowest  oxygen  content  at both 
time points  would  be the 
pulmonary  artery .
Answer: Pulmonary artery
Step-by-Step thinking : 1. The 
document  mentions  … arterial  
blood  is fully saturated  with 
oxygen  even  during  strenuous  
exercise  at sea level. 2. The 
document  also states  that the O2 
content  of cardiac  venous  blood  
is normally  low … 3. …
Answer: Coronary sinus𝑺𝟏: To determine  cardiac  output  by this method,  three  values  
must  be known : (1) O2 consumption  of the body,  (2) the O2 
concentration  in pulmonary  venous  blood  ([O2] pv), and (3) 
the O2 concentration  in pulmonary  arterial  blood  ([O2] 
pa) … (O2 in pulmonary  arteries  and veins )
𝑺𝟐: Most  important,  any of the following  signs  during  
noninvasive  testing  indicates  a high risk for coronary  
events  … (Signature  of coronary  events )
𝑺𝟏:…Consumption  of O2 by the heart  depends  on the 
amount  and type of activity  that the heart  performs . … The 
O2 content  of cardiac  venous  blood  is normally  low 
(\u22485  mL/ dL), and the myocardium  can receive  little 
additional  O2 by further  extraction  of O2 from coronary  
blood . 
𝑺𝟐: … Failure  by the lungs  to oxygenate  blood  fully can be 
ruled  out because  even  with the most  strenuous  exercise  
at sea level,  arterial  blood  is fully saturated  with O2. …A healthy  23-year-old male  is 
undergoing  an exercise  stress  
test as part of his physiology  
class . If blood  were  to be 
sampled  at different  locations  
before  and after the stress  test, 
which  area of the body  would  
contain  the lowest  oxygen  
content  at both time points?
A. Inferior vena cava
B. Coronary sinus
C. Pulmonary artery
D. Pulmonary veinMedical QAResponse Snippets
Medical Team: Cardiologist; Pulmonologist; Anesthesiologist 
Summary  𝑫: Understanding  oxygen  saturation  levels  in 
arterial  and venous  blood  at rest and during  exercise,  as 
well as factors  like the oxygen -hemoglobin  dissociation  
curve,  is crucial . Oxygen  extraction  by tissues  plays  a 
significant  role in determining  the area with the lowest  
oxygen  content  … Expertise  in oxygen  transport,  
cardiovascular  responses  to exercise,  oxygen  extraction  by 
tissues,  and blood  gas analysis  interpretation  are key …Medical  Team & Distilled Summary(A)
(B)Figure 4: Example from the MedQA-US benchmark comparing MedRAG (A)and Discuss-RAG (B). Answers and
key phrases are highlighted in red.
structured agent reasoning to mitigate the adverse
effects of suboptimal retrieval. Specifically, given
the distilled summary Dand the medical query Q,
a specialized decision-maker agent Uis introduced
to evaluate the top- kdocument chunks Siretrieved
by the underlying retrieval algorithm. If Ureturns a
negative judgment, an alternative retrieval strategy
is triggered (e.g., a CoT-based prompt (Wei et al.,
2022) is used as a fallback in our implementation).
Otherwise, the accepted snippets are incorporated
into the context prompt for answer generation. As
shown in Fig. 3(B), the verified snippets tend to
be closely aligned with the intended focus of the
query. In the shown example, the selected evi-
dence explicitly highlights the effect (marked in
red) of performance measurement systems, pro-
viding grounded support for a more accurate and
contextually appropriate response.
It is important to emphasize that both compo-
nents of our proposed pipeline are designed as plug-
and-play enhancements for any training-free RAG
system. The multi-turn brainstorming process en-
riches the information fed to the retriever, while
the post-retrieval verification module dynamically
filters and validates the retrieved content, providing
improved reliability and quality in the answer.
4 Experiments
Experimental details . We evaluated the re-
trieval ability of our method on four medical QA
benchmark datasets: MMLU-Med (Hendrycks
et al., 2020), MedQA-US (Jin et al., 2021),
BioASQ (Tsatsaronis et al., 2015), and Pub-
MedQA (Jin et al., 2019). MedRAG (Xiong et al.,
2024). To ensure a fair comparison, we employed
the same medical textbooks (Jin et al., 2021) as
the corpus and MedCPT (Jin et al., 2023) as the
retriever with MedRAG. For LLM, GPT-3.5 (i.e.,gpt-3.5-turbo-0125 (OpenAI, 2024)) was selected.
Table 1: Benchmark dataset results. Answer accuracy
was used as the evaluation metric.
Dataset MedRAG + Discuss-RAG ∆
MMLU-Med 71.53% 77.23% +5.70%
MedQA-US 62.45% 66.85% +4.40%
BioASQ 58.61% 75.28% +16.67%
PubMedQA 35.60% 47.80% +12.20%
Experimental results and analysis . Leverag-
ing the multi-turn discussion and post-retrieval
verification modules, Discuss-RAG enriches the
background information available and mitigates
the impact of suboptimal retrieval. As shown
in Tab. 1, integrating our method consistently
improves MedRAG performance across all four
benchmarks, especially achieving gains of up to
16.67% on the BioASQ dataset and 12.20% on
PubMedQA. Further, as illustrated in Fig. 4, for
the same query, the top-2 snippets retrieved by
Disscuss-RAG provide more grounded and fac-
tual support for correctly answering the question.
Specifically, snippets S1explicitly mention the low
oxygen ( O2) content in cardiac venous blood, while
snippets S2support the reasoning process from a
contrasting perspective. Additionally, the final dis-
tilled summary Dgenerated by the medical team
highlights the essential knowledge required to fo-
cus the retrieval process, leading to more reliable
and contextually appropriate evidence selection.
5 Conclusion
In this work, we propose Discuss-RAG , an agent-
led framework designed to enhance the response ac-
curacy of LLMs in medical QA. Specifically, we in-
troduce a multi-turn discussion and summarization
module to facilitate context-rich and self-refined
document retrieval, and a post-retrieval verification
agent to make the final judgment on the retrieved

content. Experiments conducted on four medical
QA benchmark datasets demonstrate that Discuss-
RAG consistently improves both response accuracy
and snippet quality.
6 Limitation
We acknowledge that Discuss-RAG is hindered by
two primary limitations. (1). Limited interaction
among team members. The specialized medical
agents Hido not communicate directly with one
another, but interact through the summary from the
previous round. Direct peer-to-peer interaction may
facilitate deeper and more dynamic reasoning. (2).
Increased computational overhead. Our framework
involves prompting multiple LLM-based agents,
each requiring careful instruction design to perform
their respective roles effectively. This introduces
additional computational and engineering costs.
References
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. In Proceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering-Software Engineering for AI , pages 194–
199.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.
Improving language models by retrieving from tril-
lions of tokens. In International conference on ma-
chine learning , pages 2206–2240. PMLR.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Bolei He, Nuo Chen, Xinran He, Lingyong Yan,
Zhenkai Wei, Jinchang Luo, and Zhen-Hua Ling.
2024. Retrieving, rethinking and revising: The chain-
of-verification can improve retrieval augmented gen-
eration. arXiv preprint arXiv:2410.05801 .
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing. arXiv preprint arXiv:2009.03300 .
Gautier Izacard and Edouard Grave. 2020. Leverag-
ing passage retrieval with generative models for
open domain question answering. arXiv preprint
arXiv:2007.01282 .
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, AndreaMadotto, and Pascale Fung. 2023. Survey of hal-
lucination in natural language generation. ACM com-
puting surveys , 55(12):1–38.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2021. What disease
does this patient have? a large-scale open domain
question answering dataset from medical exams. Ap-
plied Sciences , 11(14):6421.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W
Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset
for biomedical research question answering. arXiv
preprint arXiv:1909.06146 .
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval. Bioinformatics ,
39(11):btad651.
YuHe Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal
Abdullah, Nan Liu, Alex Tiong Heng Sia, Chai Rick
Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong,
and Daniel Shu Wei Ting. 2024. Development and
testing of retrieval augmented generation in large
language models–a case study report. arXiv preprint
arXiv:2402.01733 .
Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu
Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee,
Marzyeh Ghassemi, Cynthia Breazeal, Hae Park,
et al. 2024. Mdagents: An adaptive collaboration
of llms for medical decision-making. Advances in
Neural Information Processing Systems , 37:79410–
79452.
Harsha Nori, Nicholas King, Scott Mayer McKinney,
Dean Carignan, and Eric Horvitz. 2023. Capabili-
ties of gpt-4 on medical challenge problems. arXiv
preprint arXiv:2303.13375 .
OpenAI. 2024. Gpt-3.5 turbo. https://platform.
openai.com/docs/models/gpt-3-5-turbo . Ac-
cessed: 2025-04-27.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl,
et al. 2023. Large language models encode clinical
knowledge. Nature , 620(7972):172–180.
George Tsatsaronis, Georgios Balikas, Prodromos
Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R Alvers, Dirk Weissenborn, Anastasia
Krithara, Sergios Petridis, Dimitris Polychronopou-
los, et al. 2015. An overview of the bioasq large-scale
biomedical semantic indexing and question answer-
ing competition. BMC bioinformatics , 16:1–28.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. In Findings of the Associa-
tion for Computational Linguistics ACL 2024 , pages
6233–6251.
Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza
Namazi-Rad, and Jun Wang. 2023. How do large
language models capture the ever-changing world
knowledge? a review of recent advances. arXiv
preprint arXiv:2310.07343 .
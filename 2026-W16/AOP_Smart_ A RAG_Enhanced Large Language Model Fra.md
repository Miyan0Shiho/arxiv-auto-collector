# AOP-Smart: A RAG-Enhanced Large Language Model Framework for Adverse Outcome Pathway Analysis

**Authors**: Qinjiang Niu, Lu Yan

**Published**: 2026-04-13 00:49:37

**PDF URL**: [https://arxiv.org/pdf/2604.10874v1](https://arxiv.org/pdf/2604.10874v1)

## Abstract
Adverse Outcome Pathways (AOPs) are an important knowledge framework in toxicological research and risk assessment. In recent years, large language models (LLMs) have gradually been applied to AOP-related question answering and mechanistic reasoning tasks. However, due to the existence of the hallucination problem, that is, the model may generate content that is inconsistent with facts or lacks evidence, their reliability is still limited. To address this issue, this study proposes an AOP-oriented Retrieval-Augmented Generation (RAG) framework, AOP-Smart. Based on the official XML data from AOP-Wiki, this method uses Key Events (KEs), Key Event Relationships (KERs), and specific AOP information to retrieve relevant knowledge for user questions, thereby improving the reliability of the generated results of large language models. To evaluate the effectiveness of the proposed method, this study constructed a test set containing 20 AOP-related question answering tasks, covering KE identification, upstream and downstream KE retrieval, and complex AOP retrieval tasks. Experiments were conducted on three mainstream large language models, Gemini, DeepSeek, and ChatGPT, and comparative tests were performed under two settings: without RAG and with RAG. The experimental results show that, without using RAG, the accuracies of GPT, DeepSeek, and Gemini were 15.0\%, 35.0\%, and 20.0\%, respectively; after using RAG, their accuracies increased to 95.0\%, 100.0\%, and 95.0\%, respectively. The results indicate that AOP-Smart can significantly alleviate the hallucination problem of large language models in AOP knowledge tasks, and greatly improve the accuracy and consistency of their answers.

## Full Text


<!-- PDF content starts -->

AOP-Smart: A RAG-Enhanced Large Language Model
Framework for Adverse Outcome Pathway Analysis
Qinjiang Niu1
Lu Yan1,∗
1Nanyang Normal University, Nanyang, China
*Corresponding author: Lu Yan
Abstract
Adverse Outcome Pathways (AOPs) are an important knowledge framework in toxicological research and risk
assessment. In recent years, large language models (LLMs) have gradually been applied to AOP-related question
answering and mechanistic reasoning tasks. However, due to the existence of the hallucination problem, that is,
the model may generate content that is inconsistent with facts or lacks evidence, their reliability is still limited.
To address this issue, this study proposes an AOP-oriented Retrieval-Augmented Generation (RAG) framework,
AOP-Smart. Based on the official XML data from AOP-Wiki, this method uses Key Events (KEs), Key Event
Relationships (KERs), and specific AOP information to retrieve relevant knowledge for user questions, thereby
improving the reliability of the generated results of large language models.
To evaluate the effectiveness of the proposed method, this study constructed a test set containing 20 AOP-related
question answering tasks, covering KE identification, upstream and downstream KE retrieval, and complex AOP
retrieval tasks. Experiments were conducted on three mainstream large language models, Gemini, DeepSeek, and
ChatGPT, and comparative tests were performed under two settings: without RAG and with RAG.
The experimental results show that, without using RAG, the accuracies of GPT, DeepSeek, and Gemini were
15.0%, 35.0%, and 20.0%, respectively; after using RAG, their accuracies increased to 95.0%, 100.0%, and 95.0%,
respectively. The results indicate that AOP-Smart can significantly alleviate the hallucination problem of large
language models in AOP knowledge tasks, and greatly improve the accuracy and consistency of their answers.
Keywords:Adverse Outcome Pathway; large language model; retrieval-augmented generation; knowledge en-
hancement; hallucination problem
1. Introduction
Adverse Outcome Pathways (AOPs) are a conceptual
framework used to describe the causal relationship from
Molecular Initiating Events (MIEs) to Adverse Out-
comes (AOs), and are of great significance in toxico-
logical research and chemical risk assessment [1]. By
organizing Key Events (KEs) at the molecular, cellu-
lar, tissue, organ, and individual levels, AOPs provide
systematic support for the expression of toxicity mech-
anisms, knowledge integration, and risk judgment [2].
With the continuous development of AOP-Wiki, the
number of related knowledge entries has been increas-
ing, and the content covers more species, biological lev-
els, and complex Key Event Relationships (KERs). Al-
though the expansion of knowledge scale improves the
richness of AOP resources, it also increases the diffi-
culty for researchers in retrieving, understanding, and
integrating information.
To improve the usability of AOP knowledge, previous
studies have developed various supporting tools, such
as AOP-HelpFinder and AOP-Explore, to support the
mining, retrieval, and visualization analysis of AOP-
related information [3, 4]. At the same time, large
language models (LLMs) have shown strong capabil-
ities in tasks such as natural language understanding,knowledge question answering, and complex reasoning,
and have demonstrated broad application prospects
in biomedical text mining and mechanistic knowledge
analysis [5, 6, 7]. Existing studies have shown that
LLMs can assist in the extraction, integration, and
structured representation of scientific knowledge, and
can help with the construction of AOP chains to a cer-
tain extent [8]. However, the generated results of LLMs
are limited by the coverage of training corpora and
the generation mechanism itself, and they are prone
to producing hallucinations when dealing with highly
specialized domain knowledge, that is, outputs that are
fluent in expression but factually incorrect, thereby af-
fecting their reliability and interpretability in scientific
research [9, 10].
Retrieval-Augmented Generation (RAG) provides an
effective idea for alleviating the above problems. By
introducing external knowledge before LLMs generate
content, RAG provides knowledge relevant to the ques-
tion as context to the model, thereby enhancing the
factual basis of model responses and reducing the risk
of hallucination [11]. A typical RAG framework usu-
ally includes a retrieval module and a generation mod-
ule: the former obtains relevant content from exter-
nal knowledge sources, and the latter generates an-
swers based on the retrieved results [12, 13]. Most
1arXiv:2604.10874v1  [cs.CL]  13 Apr 2026

existing RAG methods are based on vector databases
and semantic similarity retrieval, and have achieved
good results in open-domain question answering and
knowledge-intensive tasks.
However, research on RAG in the AOP domain is
still relatively limited. AOP knowledge is different from
general unstructured text; it has significant hierarchical
structure, causal relationships, and network character-
istics, which place higher requirements on knowledge
organization and retrieval methods. Existing studies
still lack RAG designs oriented to the causal relation-
ship characteristics of AOPs, especially in how to ef-
fectively utilize KEs, KERs, and overall AOP struc-
tural information to support complex AOP mechanistic
knowledge modeling and reasoning.
Based on this, this paper proposes an AOP analysis-
oriented RAG framework, AOP-Smart. This method
uses the XML data provided by AOP-Wiki to con-
struct a knowledge base and index files, and designs an
indexing and knowledge expansion mechanism based
on question-related KEs to achieve joint retrieval and
knowledge integration of KE, KER, and AOP informa-
tion. On this basis, the KE, KER, and AOP knowledge
related to the question are used as external context and
input into LLMs, thereby improving the reliability of
LLM results.
The main contributions of this study are as follows:
(1) A RAG framework for the AOP domain is pro-
posed, which introduces AOP knowledge into the
reasoning process of LLMs;
(2) A knowledge expansion strategy based on
question-related KEs is designed to achieve as-
sociated retrieval and knowledge supplementation
among KEs, KERs, and AOPs;
(3) Comparative experiments verify the effectiveness
of this method in alleviating the hallucination
problem in the AOP domain and improving an-
swer accuracy.
2. Methods
2.1 Overall Framework of AOP-Smart
To ensure the consistency of data sources and the re-
producibility of the experimental process, this study
conducted experiments based on the stable XML
snapshot of the AOP-Wiki released on 2026-01-
01. This dataset was obtained from the official
AOP-Wiki quarterly “Permanent Downloads” archive
(https://aopwiki.org/downloads ), which provides ver-
sioned XML dumps for citation and reproducible re-
search. The 2026-01-01 release was selected to guaran-
tee data stability, and all contents were subsequently
parsed and processed in a unified manner.
It should be noted that the constructed system does
not depend on a specific version of the data, but has
good scalability and updating capability. By replacing
the XML file of the corresponding version, the knowl-
edge base can be updated.As shown in Figure 1(A), without introducing the
RAG mechanism, the user question is directly input
into the LLM, and the model answers only relying on
its internal parameters, lacking the support of exter-
nal knowledge, and thus is prone to problems such as
missing information or factual errors.
As shown in Figure 1(B), the AOP-Smart frame-
work introduces a knowledge retrieval process prior
to response generation. Specifically, the official XML
data from AOP-Wiki is first parsed into two structured
resources:Index.txtandAOP-Smart.json. Among
them,Index.txtstores the IDs and titles of all key
events (KEs) and is used for efficient preliminary candi-
date retrieval. This index is extremely lightweight, con-
taining only approximately 20,000 tokens, which is neg-
ligible compared to the context window of modern large
language models (which can reach nearly one million
tokens). Therefore, it can be directly loaded into the
prompt without introducing significant computational
overhead, while also enabling easy extensibility for fu-
ture AOP-Wiki updates. In contrast,AOP-Smart.json
stores detailed structured information about KEs, key
event relationships (KERs), and adverse outcome path-
ways (AOPs), supporting downstream reasoning and
generation tasks.
During the inference stage, the system first inputs the
user question together with the content inIndex.txt
into the LLM, and ranks the relevance of all KEs
through prompt engineering, from which the Top-N
most relevant KEs are selected (N is a tunable pa-
rameter). Subsequently, based on the IDs of the se-
lected KEs, the corresponding KE details are retrieved
fromAOP-Smart.json, and their related KER and
AOP information are further expanded to construct a
structured context representation (including KE Detail,
KER Detail, and AOP Detail).
Finally, this structured context and the user ques-
tion are jointly input into the LLM for generative rea-
soning, thus completing the closed-loop process from
question-driven retrieval to knowledge-enhanced gener-
ation. This mechanism effectively alleviates the knowl-
edge deficiency problem of traditional LLMs in AOP
tasks and significantly improves the accuracy and in-
terpretability of the generated results.
2.2 Expansion Method Based on TOP-N KE
IDs
After the LLM completes the preliminary screening of
KEs, the system performs expansion based on the se-
lected KEs to enhance the coverage of question-related
knowledge. The related process is shown in Figure 2.
The specific process is as follows.
For each selected KE, the system first extracts its di-
rectly upstream and directly downstream KE IDs from
AOP-Smart.json, and merges them with the originally
selected KE to form an expanded KE set. The pur-
pose of this step is to supplement the local mechanistic
information causally adjacent to the events explicitly
involved in the question.
After obtaining the expanded KE set, the system
further traverses all Key Event Relationships (KERs)
2

User
Question
LLM
IndexKE ID | Describe
195 | Inhibition, NMDARs
350 | Increase, Mortality
…
Top-N Relevant 
KE IDs
[57,489,64…]
Question Relevant
KE Detail
KER Detail
AOP Detail
LLMLLM
AnswerUser
Question
ALL
KE Detail
KER Detail
AOP DetailAOP-Wiki
Filter by KE IDs
Index
Python(B) With RAG(A) Without RAG
User
Question
LLM LLM
Answer
N is user -definedFigure 1:Overview of the AOP-Smart framework. (A) In the absence of retrieval augmentation, the LLM directly
generates responses based solely on its internal parameters, which may lead to incomplete or hallucinated outputs. (B)
With the proposed AOP-Smart framework, user queries are first used to retrieve relevant KE candidates from an indexed
KE list, followed by structured expansion to KE, KER, and AOP-level information from the knowledge base. The retrieved
structured context is then combined with the user query and fed into the LLM to produce a knowledge-enhanced and more
reliable response.
LLMTop-N Relevant 
KE IDsALL
KE Detail
KER Detail
AOP Detail
Upstream KE IDs of the 
Top-N Relevant KEsTop-N Relevant 
KE IDs
KE Detail AOP Detail KER DetailDownstream KE IDs of the 
Top-N Relevant KEs
KE id 
title 
level 
organ 
cell 
description 
applicabilityupstream_ ke_id
downstream_ke_id
description
applicabilityAOP id
title
abstract
MIEs_IDKEs_ID
AOs_ID
Event 
Relationship
applicabilityUser QuestionSelect AOPs 
with ≥2 KEs (In KE IDs)
LLMAnswer
Python
Python Filter by KE IDsN is user -defined
Figure 2:Overview of the KE-based expansion process, in-
cluding KE augmentation, KER reconstruction, and AOP
retrieval, followed by integration with LLM-based reason-
ing.
in the knowledge base, and selects those relationships
that connect any two KEs within the expanded KE set,
forming an expanded KER set. Through this joint ex-
pansion process, the system can not only identify the
event nodes most relevant to the question, but also re-
cover the causal relationships among these events, pro-
viding a basis for mechanism explanation and relation-
ship judgment in subsequent responses.
Subsequently, the system scans all AOPs. When at
least two matches exist between the MIE, KE, or AO
in an AOP and the expanded KE set, the system in-
cludes that AOP into the expanded AOP set. This
matching rule is intended to reduce the noise caused by
accidental matching of a single event and improve the
relevance between the retrieval results and the ques-tion. Through this mechanism, the system can fur-
ther expand from local event relationships to complete
AOP chains, thereby providing higher-level background
knowledge support for subsequent question answering.
Finally, the system integrates the expanded KE,
KER, and AOP information, and inputs them together
with the user question into the LLM for subsequent
reasoning and answer generation. In this way, the final
output of the model is established on the basis of ex-
plicit external knowledge support, which helps reduce
free generation detached from the facts in the knowl-
edge base.
2.3 Engineering Implementation
This study implemented a complete software system
based on Python, and its interface is shown in Figure 3.
The system mainly consists of three functional areas:
the configuration area, the user input area, and the
LLM output area.
In the configuration area, the user needs to input the
Service Endpoint and API key of the large model being
used. After completing the configuration, clicking the
Fetch Models button will obtain the list of currently
available large language models, and system initializa-
tion is completed by selecting a model.
The system also provides adjustable interfaces for
multiple key parameters to support flexible control un-
der different task requirements. These include:Max
Output, which is used to control the maximum number
of tokens generated by the model;Temperature, which
is used to adjust the randomness and creativity of the
3

Run
StopAOP-Smart
API Status : NOT Connected Fetch Models
Base URL
API Key
Model
Max Output
Temperature
TOP N KEs
XML Version 2026-01-01
User inputLLM Output
4096
0
5https:// api .deepseek .com
************************Service 
endpoint
Authentication 
key
Enter the Base 
URL and API 
Key, click 
“Fetch Models” 
to retrieve 
available 
models, then 
select the 
desired LLM.
Max Output : The maximum output token of LLM
Temperature : Controlling the stability of LLM output
(lower values increase determinism)
Top N KEs : Determines how many top -ranked KEs 
are selected before expanding KE, KER, and AOP 
information
XML Version : The XML version of AOP -Wiki
Figure 3:Overview of the Python-based system interface,
including configuration, input, and output modules, with
adjustable parameters for model selection and KE-based re-
trieval control.
model output, where a lower value leads to more stable
output and a higher value leads to more diverse gen-
erated results; and theTop-N KEparameter, which is
the core control variable in this study, used to specify
the number of Key Events (KEs) screened by the LLM,
thereby controlling the expansion scale and information
coverage of RAG retrieval.
In the user input area, after the user enters the ques-
tion to be processed and clicks the RUN button, the
system will call the LLM based on the configuration pa-
rameters and the expanded knowledge retrieval mech-
anism, and output the final result to the output area.
The software system developed in this study and the
related experimental code have been made public as
supplementary materials to support method reproduc-
tion. The project code can be accessed via GitHub:
https://github.com/qinjiang-lab/AOP-Smart
3. Experimental Design and Results
3.1 Experimental Design
To evaluate the performance of large language models
on AOP knowledge tasks and verify the effectiveness of
the proposed method, this paper constructed an AOP
question answering test set containing 20 questions, as
shown in Figure 4.The full set of questions is provided
in the Supplementary Material.
KE Identification
Downstream KE retrieval
Upstream KE retrieval
Complex taskAOP QA Evaluation Set (4 tasks ×5 questions)
What is the biological 
event corresponding to KE 
XXX, and in which species 
is it applicable?×5
What is the KE ID of 
“XXX", and what are all its 
Downstream key events? ×5
What is the KE ID of 
“XXX", and what are all its 
Upstream key events? ×5
For the AOPs where “XXX" 
serves as the MIE and 
“XXX" serves as the AO, 
how many such AOPs exist? 
Briefly list all key events 
involved and specify the 
applicable species.×5? ? ?
?Title ?
? ? ?
? ? ?
? ?? MIE AOKE ID
XXX
KE ID
XXX
Figure 4:Overview of the 20-question AOP benchmark,
grouped into four task categories: KE identification, down-
stream KE retrieval, upstream KE retrieval, and complex
AOP query.
According to different task types, the test set is di-
vided into four categories:(1)Key Event Identification (KE Identifica-
tion):Given a KE ID, the model is required to
identify the corresponding biological event and its
applicable species;
(2)Downstream Key Event Retrieval (Down-
stream KE Retrieval):Given an event name,
the model is required to return the corresponding
KE ID and list all directly downstream KEs;
(3)Upstream Key Event Retrieval (Upstream
KE Retrieval):Given an event name, the model
is required to return the corresponding KE ID and
list all directly upstream Key Events;
(4)Complex AOP Query Task (Complex AOP
Query):This involves retrieving the complete
AOP chain based on a given MIE and AO, and
requires listing the related Key Events and their
applicable species information.
The above tasks cover different types in AOP knowl-
edge, such as KE identification, KER retrieval, and
complex AOP query, and can evaluate the hallucina-
tion problem of LLMs from multiple perspectives.
The evaluation metric adopts Accuracy, defined as:
Accuracy =Ncorrect
Ntotal
whereN correct represents the number of completely cor-
rect answered questions, andN totalrepresents the total
number of questions. In this study, the factual informa-
tion in the corresponding entries of AOP-Wiki is used
as the basis for judgment, and when the model answer
is completely consistent with the standard knowledge,
it is judged as correct.
3.2 Experimental Results
In this experiment, the Top-N parameter was uniformly
set to 5, and Temperature was set to 0 to ensure the
retrieval scale and the determinism of the generated
results. The specific models used were DeepSeek-V3.2,
GPT-5.4, and Gemini-3.1 Pro, respectively. The exper-
imental results are shown in Table 1.
Without introducing RAG, the performance of the
three large language models on AOP-related question
answering tasks showed obvious differences. Among
them, the overall accuracies of DeepSeek, Gemini, and
GPT were 35.0%, 20.0%, and 15.0%, respectively, indi-
cating that it is difficult to stably complete AOP tasks
by relying only on the internal knowledge of the models.
After introducing the AOP-Smart retrieval-
augmented framework proposed in this paper,
the performance of each model on all tasks was signif-
icantly improved. The overall accuracy of DeepSeek
increased to 100% after enhancement, while Gemini
and GPT increased to 95.0% and 95.0%, respectively,
verifying the effectiveness of RAG in improving the
ability of models to utilize external knowledge.
From the task-level analysis, RAG has a consistent
improvement effect on different types of tasks. In the
Key Event Identification (Task 1) and Downstream
4

Table 1:Experimental results of different large language models on AOP-related tasks with and without RAG enhance-
ment.
Task Description DeepSeek Gemini GPT Overall
W/O
RAGW/
RAGW/O
RAGW/
RAGW/O
RAGW/
RAGW/O
RAGW/
RAG
Task 1 KE Identification 80% 100% 60% 100% 60% 100% 66.67% 100%
Task 2 Downstream KE retrieval 20% 100% 20% 100% 0% 100% 13.33% 100%
Task 3 Upstream KE retrieval 20% 100% 0% 80% 0% 100% 6.67% 93.33%
Task 4 Complex task 20% 100% 0% 100% 0% 80% 6.67% 93.33%
Overall All tasks combined 35% 100% 20% 95% 15% 95% 23.33% 96.67%
Key Event Retrieval (Task 2) tasks, all three models
achieved accuracies close to or reaching 100%; in the
Upstream Key Event Retrieval (Task 3) and Complex
AOP reasoning task (Task 4), RAG also significantly
improved the performance of the models in multi-hop
structural relationship queries, but compared with ba-
sic retrieval tasks, there were still certain differences in
difficulty.
These results indicate that the RAG method can ef-
fectively improve the stability and reliability of large
language models in professional toxicological knowledge
question answering, and significantly alleviate the hal-
lucination problem of LLMs.
4. Limitations
Although the AOP-Smart framework proposed in this
study has achieved relatively significant experimental
results in AOP-related tasks, there are still certain lim-
itations.
First, the current method has not yet introduced in-
formation such as evidence level, evidence strength, and
confidence in the AOP knowledge system. Therefore,
in complex reasoning processes, it is not possible to
perform weighted processing on knowledge from differ-
ent sources, which may affect the reliability and inter-
pretability of the final results.
Second, when constructing the retrieval knowledge
base, this paper performed necessary text filtering and
truncation processing on the KE, KER, and AOP de-
scriptive information in AOP-Wiki to adapt to the in-
put length limitation of large language models. Al-
though this strategy improves computational efficiency,
it may also lead to the loss of some contextual seman-
tic information, thereby affecting the completeness of
answers in complex query tasks.
Third, the evaluation set constructed in this study is
relatively limited in scale (20 questions), and is mainly
based on manually designed query tasks. Although it
can cover typical KE identification and AOP relation-
ship retrieval scenarios, it is still insufficient in cover-
ing larger-scale, multi-hop reasoning, and open-domain
AOP question answering tasks.
Finally, different large language models are some-
what sensitive to prompt format and context organi-
zation methods. Therefore, the experimental results
may be jointly affected by both model capability and
prompt engineering design, and this factor has not yet
been systematically analyzed through ablation in thisstudy.
5. Conclusion and Future Work
This study proposes a retrieval-augmented generation
framework for Adverse Outcome Pathway (AOP) anal-
ysis, AOP-Smart. This method constructs a struc-
tured knowledge base based on the official XML data
of AOP-Wiki, and through an expansion mechanism
based on Key Events (KEs) and Key Event Relation-
ships (KERs), introduces AOP hierarchical structural
information into the reasoning process of large language
models, thereby enhancing their knowledge acquisition
capability in professional toxicological question answer-
ing tasks.
The experimental results show that, on 20 structured
AOP question answering tasks, AOP-Smart signifi-
cantly improves the overall performance of DeepSeek,
GPT, and Gemini under the no-RAG condition, and
achieves near-optimal or optimal accuracy in most task
types. Among them, in the complex AOP query task,
the RAG mechanism has a clear enhancement effect on
multi-hop structural relationship reasoning.
The above results indicate that the retrieval-
augmented method based on structured AOP knowl-
edge can effectively improve the stability and accuracy
of large language models in professional-domain ques-
tion answering tasks, and alleviate the model halluci-
nation problem to a certain extent.
Future work will be carried out from the following
three aspects. First, expand the scale of evaluation
data and construct a larger-scale and multi-type AOP
question answering benchmark to improve the compre-
hensiveness and robustness of evaluation. Second, in-
troduce structured information such as evidence level
and confidence, and perform weighted modeling on re-
trieval results to enhance the interpretability of the rea-
soning process. Finally, explore the application poten-
tial of AOP-Smart in automatic information extraction
from scientific literature and AOP knowledge graph
completion, so as to provide support for the automated
construction and expansion of the AOP system.
References
[1] Gerald T. Ankley, Richard S. Bennett, Rus-
sell J. Erickson, Dale J. Hoff, Michael W. Hor-
nung, Rodney D. Johnson, David R. Mount,
John W. Nichols, Christine L. Russom, Patricia K.
5

Schmieder, Jose A. Serrrano, Joseph E. Tietge,
and Daniel L. Villeneuve. Adverse outcome path-
ways: A conceptual framework to support eco-
toxicology research and risk assessment.Environ-
mental Toxicology and Chemistry, 29(3):730–741,
November 2009.
[2] Anna Bal-Price and M.E. (Bette) Meek. Adverse
outcome pathways: Application to enhance mech-
anistic understanding of neurotoxicity.Pharmacol-
ogy & Therapeutics, 179:84–95, November 2017.
[3] Florence Jornod, Thomas Jaylet, Ludek Blaha,
Denis Sarigiannis, Luc Tamisier, and Karine Au-
douze. Aop-helpfinder webserver: a tool for com-
prehensive analysis of the literature to support ad-
verse outcome pathways development.Bioinfor-
matics, 38(4):1173–1175, October 2021.
[4] Saurav Kumar, Deepika Deepika, Karin Slater,
and Vikas Kumar. Aopwiki-explorer: An in-
teractive graph-based query engine leveraging
large language models.Computational Toxicology,
30:100308, June 2024.
[5] Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
 Lukasz Kaiser, and Illia Polosukhin. Attention is
all you need.Advances in neural information pro-
cessing systems, 30, 2017.
[6] Tom Brown, Benjamin Mann, Nick Ryder,
Melanie Subbiah, Jared D Kaplan, Prafulla Dhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish
Sastry, Amanda Askell, et al. Language models
are few-shot learners.Advances in neural infor-
mation processing systems, 33:1877–1901, 2020.
[7] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim,
Donghyeon Kim, Sunkyu Kim, Chan Ho So, and
Jaewoo Kang. Biobert: a pre-trained biomedical
language representation model for biomedical text
mining.Bioinformatics, 36(4):1234–1240, Septem-
ber 2019.
[8] Haochun Shi and Yanbin Zhao. Integration of ad-
vanced large language models into the construction
of adverse outcome pathways: Opportunities and
challenges.Environmental Science & Technology,
58(35):15355–15358, August 2024.
[9] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu,
Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, An-
drea Madotto, and Pascale Fung. Survey of hal-
lucination in natural language generation.ACM
Computing Surveys, 55(12):1–38, March 2023.
[10] Stephanie Lin, Jacob Hilton, and Owain Evans.
Truthfulqa: Measuring how models mimic human
falsehoods. InProceedings of the 60th annual meet-
ing of the association for computational linguistics
(volume 1: long papers), pages 3214–3252, 2022.
[11] Patrick Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, NamanGoyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau
Yih, Tim Rockt¨ aschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in neural information processing systems,
33:9459–9474, 2020.
[12] Danqi Chen, Adam Fisch, Jason Weston, and An-
toine Bordes. Reading wikipedia to answer open-
domain questions. InProceedings of the 55th
Annual Meeting of the Association forComputa-
tional Linguistics (Volume 1: Long Papers), page
1870–1879. Association for Computational Lin-
guistics, 2017.
[13] Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval
for open-domain question answering. InProceed-
ings of the 2020 conference on empirical methods
in natural language processing (EMNLP), pages
6769–6781, 2020.
6
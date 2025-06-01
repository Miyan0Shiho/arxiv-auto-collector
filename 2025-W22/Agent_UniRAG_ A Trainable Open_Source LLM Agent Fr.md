# Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems

**Authors**: Hoang Pham, Thuy-Duong Nguyen, Khac-Hoai Nam Bui

**Published**: 2025-05-28 16:46:31

**PDF URL**: [http://arxiv.org/pdf/2505.22571v2](http://arxiv.org/pdf/2505.22571v2)

## Abstract
This paper presents a novel approach for unified retrieval-augmented
generation (RAG) systems using the recent emerging large language model (LLM)
agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental
controllers, has become a promising approach to enable the interpretability of
RAG tasks, especially for complex reasoning question-answering systems (e.g.,
multi-hop queries). Nonetheless, previous works mainly focus on solving RAG
systems with either single-hop or multi-hop approaches separately, which limits
the application of those approaches to real-world applications. In this study,
we propose a trainable agent framework called Agent-UniRAG for unified
retrieval-augmented LLM systems, which enhances the effectiveness and
interpretability of RAG systems. The main idea is to design an LLM agent
framework to solve RAG tasks step-by-step based on the complexity of the
inputs, simultaneously including single-hop and multi-hop queries in an
end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset
to enable the proposed agent framework for small open-source LLMs (e.g.,
Llama-3-8B). The results show comparable performances with closed-source and
larger open-source LLMs across various RAG benchmarks. Our source code and
dataset are publicly available for further exploitation.

## Full Text


<!-- PDF content starts -->

arXiv:2505.22571v2  [cs.CL]  29 May 2025Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for
Unified Retrieval-Augmented Generation Systems
Hoang Pham, Thuy-Duong Nguyen, Khac-Hoai Nam Bui*
Viettel Artificial Intelligence and Data Services Center,
Viettel Group, Vietnam
{hoangpv4, nambkh}@viettel.com.vn
Abstract
This paper presents a novel approach for uni-
fied retrieval-augmented generation (RAG) sys-
tems using the recent emerging large lan-
guage model (LLM) agent concept. Specif-
ically, Agent LLM, which utilizes LLM as
fundamental controllers, has become a promis-
ing approach to enable the interpretability of
RAG tasks, especially for complex reason-
ing question-answering systems (e.g., multi-
hop queries). Nonetheless, previous works
mainly focus on solving RAG systems with
either single-hop or multi-hop approaches sep-
arately, which limits the application of those
approaches to real-world applications. In this
study, we propose a trainable agent framework
called Agent-UniRAG for unified retrieval-
augmented LLM systems, which enhances the
effectiveness and interpretability of RAG sys-
tems. The main idea is to design an LLM
agent framework to solve RAG tasks step-by-
step based on the complexity of the inputs, si-
multaneously including single-hop and multi-
hop queries in an end-to-end manner. Further-
more, we introduce SynAgent-RAG, a syn-
thetic dataset to enable the proposed agent
framework for small open-source LLMs (e.g.,
Llama-3-8B). The results show comparable per-
formances with closed-source and larger open-
source LLMs across various RAG benchmarks.
Our source code and dataset are publicly avail-
able for further exploitation.
1 Introduction
Incorporating non-parametric knowledge into large
language models (LLMs) through additional re-
trieval modules has emerged as a promising ap-
proach to enhance both accuracy and the timeliness
of information (Borgeaud et al., 2022; Izacard et al.,
2023). This issue has led to the rapid development
of various retrieval-augmented LLM paradigms de-
signed to provide correct answers to user queries.
*Corresponding author
Single Hop
QueryMulti-Hop
QueryRetrieve Retrieve
Iterative
Classifier  a) Modular  Appr oach
b) Adaptive Appr oach
c) Our  Unified Appr oachIterative
Planning Memory
Agent
Loop
Agent-UniRAGSingle Hop
Query
Multi-Hop
Query
Single Hop
Query
Multi-Hop
QueryAnswerAnswer
Answer
Answer
Answer Sear ch ToolFigure 1: Conceptual analysis of previous works and
Agent-UniRAG: (a) Modular approach handles query
types separately; (b) Adaptive approach uses a classifier
to determine query types before executing them sep-
arately; (c) Agent-UniRAG processes all query types
within a unified system using the Agent LLM concept.
Accordingly, these modern paradigms address ei-
ther single-hop which can respond within a single
document (i.e., Naive RAG), or complex multi-hop
queries, which require the integration and synthe-
sis of information from multiple documents (i.e.,
Advanced RAG)(Fan et al., 2024).
Nonetheless, existing modern approaches suf-
fer from several significant limitations, including a
lack of explainability and traceability. Accordingly,
an emerging research issue in this regard is that
current methods either inefficiently handle simple
queries with unnecessary computational overhead
or fail to address complex multi-step queries (Tang
and Yang, 2024) (Figure 1 (a)). To address this
research issue, a potential method is to add a classi-
fication module to classify the complexity of input
queries for selecting the appropriate RAG model to
respond (Jeong et al., 2024) (Figure 1 (b)). How-
ever, this approach is only suitable when the types

of queries are predefined (in specific domains or
custom benchmark datasets), which might lack flex-
ibility and scalability in terms of various real-world
applications. Recently, LLM Agent, by leverag-
ing LLMs to execute complex tasks, emerged as
a promising approach to enable the interpretabil-
ity and reasoning capability for LLM (Zhao et al.,
2024). Specifically, LLM is regarded as the pri-
mary controller, integrating with essential compo-
nents such as planning, memory, and action exe-
cute operations necessary to complex tasks(Wang
et al., 2024a). Based on this emerging conceptual
technology, this study raises a research question:
Can the LLM agent enable the interpretability and
reasoning capability of RAG systems in a unified
manner?
Figure 1 (c) illustrates our proposed approach,
which is designed to enhance the interpretability
and effectiveness of LLMs in RAG tasks, com-
pared with previous approaches in this research
field. Specifically, we leverage the emerging con-
cept of LLM agents, employing LLMs as central
controllers to unify RAG tasks. Our unified agent
is capable of handling queries that require rea-
soning processes (including both single-hop and
multi-hop queries simultaneously) through self-
guided instructions and interaction with the exter-
nal knowledge base. Furthermore, most current
LLM agent frameworks rely on closed-source mod-
els with very large weight sizes (e.g., GPT-4 (Ope-
nAI, 2024)), which limits their reproducibility and
controllability. Our primary focus, therefore, is on
enabling trainable open-source LLM agents. In
this regard, we also introduce a synthetic dataset
named SynAgent-RAG to train these open-source
LLM-based agents for the unified RAG system. In
summary, the main contributions of this study are
three-fold as follows:
(i)We propose a unified RAG system using
the concept of the LLM agent, which can han-
dle queries that require reasoning processes (e.g.
single-hop and multi-hop queries) by self-guided
instructions and interaction with the external knowl-
edge base to derive a response to the input queries.
To the best of our knowledge, this paper is the
first study to execute the unified RAG system in an
end-to-end manner.
(ii)We process and introduce the SynAgent-RAG
dataset, which obtains 16,987 synthetic samples
to enable small open-source modern LLMs (e.g.,
Llama-3-8B) to adapt the proposed Agent-UniRAG
approach via instruction finetuning. Accordingly,this contribution is important to achieve the de-
sired flexibility and scalability since most emerging
LLM Agent technologies only work well with very
large LLMs as the backbone.
(iii)We evaluate the proposed approach on vari-
ous RAG benchmarks, including the test set of our
proposed SynAgent-RAG dataset. The experimen-
tal results show that our approach outperforms pre-
vious approaches. Furthermore, with small LLMs
(e.g., Llama-3-8B) instruction-finetuned on the pro-
posed dataset, we can achieve competitive perfor-
mances compared to closed-source (e.g., GPT-4)
and larger open-source agent LLMs (e.g., Llama-3-
70B).
2 Literature Reviews
2.1 Retrieval-Augmented LLM
The evolution of RAG in the era of LLMs can be di-
vided into three categories, including Naive RAG,
Advanced RAG, and Modular RAG (Gao et al.,
2023). Naive RAG and Advanced RAG are typical
Retrieve-Read paradigms (Ma et al., 2023), which
focus on finding the answers in a single document
(i.e., single-hop queries (Ram et al., 2023)). Mean-
while, the recent emerging Modular RAG has been
introduced to go beyond the two aforementioned
RAG paradigms, which requires iterative accesses
to both LLMs and retrievers multiple times (i.e.,
multi-hop queries (Trivedi et al., 2023)). Specif-
ically, dynamically selecting the suitable strategy
(i.e., single-hop or multi-hop) for unified RAG
tasks become an emerging research issue in this
research field (Jeong et al., 2024).
2.2 LLM Agent Framework
The concept of LLM agents involves LLM appli-
cations that can execute complex tasks, in which
LLMs serve as controllers to control the flow of
operations needed to complete a task or user re-
quest (Wang et al., 2024a). Accordingly, an LLM
agent framework consists of the four core compo-
nents such as User Request ,Agent ,Planning , and
Memory . HuggingGPT (Shen et al., 2023) was
introduced as one of the first comprehensive LLM-
powered agent frameworks, which use LLMs (i.e.,
ChatGPT) and the ML community (i.e., Hugging
Face) to process inputs from different modalities.
Sequentially, Yin et al. (2023) introduces LUMOS,
an agent framework for trainable open-source LLM.
Specifically, the framework designs a modular ar-
chitecture with a planning module to learn subgoals

and a grounding module trained to translate sub-
goals into actions, using tools in the execution mod-
ule. Inspired by previous works, in this study, we
present a trainable open-source LLM-based agent
framework for unified RAG tasks, which focuses
on integrating the interpretable ability of LLM to
determine the next action for solving RAG tasks.
3 Methodology
This section introduces the design of Agent-
UniRAG. Following the typical pipeline of the
LLM agent framework, our framework - Agent-
UniRAG is put into a loop and includes four main
components including Planning Module, Tool Us-
ing Module, Working Memory Module, and Re-
flector Module as shown in Figure 2.
Input
Query
Knowledge
BaseAction: Search
Action: 
Final AnswerWorking
Memory
Max K
TimesEvidence
Feedback
All Gathered
EvidenceEvidence
Reflector
Planning
Gen Final
AnswerSear ch
ToolRetrieved
DocumentsKnowledge Based Agent
Input: Search Query
Figure 2: Overall design of Agent-UniRAG.
3.1 Planning Module
Leveraging the reasoning capabilities of modern
LLMs, this module is designed to systematically
determine the necessary actions required to address
a user’s request (Input Query) at each step of the
process. Specifically, the agent decides between
two primary actions at each decision point:
•Action: Search – This action is triggered
when the agent needs to acquire additional
external knowledge to progress toward solv-
ing the problem.
•Action: Final Answer – This action is taken
when the agent has accumulated sufficient in-
formation to confidently provide a response
to the query.
To implement this decision-making process, Agent-
UniRAG utilizes the ReAct mechanism (Yao et al.,
2023), which allows the agent to iteratively re-
flect on and refine its execution plan. The mech-
anism guides the agent through a structured se-
quence of steps: Thought ,Action , and EvidenceFeedback . Continuously evaluating and integrat-
ing those steps, the agent is capable of addressing
complex tasks with great precision.
3.2 Search Tool
At each stage where external knowledge is re-
quired ( Action: Search ), the agent interacts with the
Knowledge Base through the Search Tool by for-
mulating a search query generated by the Planning
Module. The purpose of querying external knowl-
edge is to ground the reasoning process in reliable
and up-to-date information beyond the agent’s in-
ternal knowledge. This ensures that the agent’s
responses are accurate and contextually relevant,
especially for tasks requiring current or special-
ized domain knowledge. The retrieved external
evidence supports the resolution of the input query,
functioning as a document retrieval task.
3.3 Reflector Module
Documents retrieved from external knowledge
bases often include irrelevant or extraneous infor-
mation, especially when the knowledge base cannot
adequately satisfy the query. Incorporating such
unfiltered data into LLMs can introduce noise, de-
grade performance, or even mislead the model. In-
spired by (Shinn et al., 2024), to mitigate this issue,
we designed a module called Evidence Reflector to
provide evidence feedback to LLM, which operates
after the Search Tool. The Evidence Reflector fil-
ters out irrelevant content and refines the retrieved
information, delivering back more focused and rele-
vant insights to the agent. If no suitable evidence is
found, it feedbacks with "No information found."
This feedback is critical in guiding the model’s
subsequent actions, ensuring the decision-making
process remains both accurate and efficient. The
agent can then better locate and leverage relevant
information, thereby improving both the quality
and precision of its responses.
3.4 Working Memory Module
The Working Memory module functions as a
prompt memory, designed to store the input query
and internal logs, including previous thoughts, ac-
tions generated by the LLM, and the extracted ev-
idence obtained through the LLM’s interactions
with tools at steps. This memory is processed
by the LLM to inform and guide subsequent ac-
tions. Furthermore, the Working Memory module
ensures the system’s transparency and explainabil-
ity by recording the reasoning process, including

Base Source
Wiki Level 5
Vital ArticlesRandomly select
a passage k Related passages
Promblem
Solver
 & Base Article
Passages
HyperlinksSelect k - 1 linked
article passages
Linked Article
Selected passages
Final Dataset
Question and
Solution path Data
Evidence Synthesis DataInstruction
TuningSolution Verification
Predicted Answer
Reference AnswerCompare
and scoreGenerate
Question  and
Reference Answer  
Solution Annotation
Evidence
GiverInteract
 Teacher Model 
 
 
 
Student ModelFigure 3: Overview of the proposed SynAgent-RAG Dataset
the gathered knowledge and the decisions made at
each step. This documentation provides insights
into how conclusions were reached, enhancing trust
and interpretability of the system.
3.5 Agent Loop
With all the defined modules, the agent operates
within a loop that can be terminated either by the
agent itself or when reaching a preconfigured com-
puting budget. In the case of the agent, the en-
tire pipeline terminates when the planning process
confirms that sufficient external evidence has been
gathered to answer the input query. In other cases,
a parameter kis preconfigured as the agent comput-
ing budget limit to limit the agent from processing
too much information, and the loop is terminated
when the computing budget is exhausted. Finally,
in either case, the agent triggers a ’Final answer’ ac-
tion to aggregate all collected evidence and provide
the final answer.
4 SynAgent-RAG Dataset
While the framework is fully compatible with big-
ger LLMs (e.g., GPT-4), deploying it with smaller
LLMs necessitates an additional training process
to maintain stability across each step. To address
this challenge, we introduce SynAgent-RAG, a syn-
thetic dataset designed for Agent-UniRAG. This is
achieved through a distillation approach (Semnani
et al., 2023), where GPT-4 serves as the teacher
model to generate data, and smaller models (e.g.,
LLama 3) are distilled versions. The primary objec-
tive of SynAgent-RAG is to empower the smaller
LLM agent with the capability to reason, analyze,
and synthesize information drawn from an external
knowledge base before delivering a well-reasoned
response to complex queries. The construction ofSynAgent-RAG follows the process illustrated in
Figure 3.
4.1 Dataset Construction Process
4.1.1 Knowledge Base
To construct an effective knowledge base for build-
ing the dataset that demands thoroughness, reliabil-
ity, and up-to-date information across a wide range
of fields, we utilized Wikipedia’s Vital Articles
Level 51. These articles represent a curated collec-
tion that encompasses essential topics for a compre-
hensive understanding of human knowledge. Prior
to constructing the dataset, we carefully divided
the articles into two separate sets: one for training
and one for testing, to ensure a balanced evaluation
of the model’s performance.
4.1.2 Question Generation
To effectively generate questions that require mul-
tiple inference steps to arrive at a final answer, it
is crucial to group related passages from source ar-
ticles. We hypothesize that these related passages
are interconnected through hyperlinks within each
Wikipedia article. For each article, we randomly
select a passage from the core content of the article
as the main passage mi. Then from passage mi, to
enhance the scalability of this process, we leverage
GPT-4 to determine which hyperlinks are most rel-
evant to the content of the main passage, following
the prompt template (see Figure 5). This process
identifies up to 5 supporting articles with associated
hyperlinks. Consequently, we obtain a set of main-
supporting passage pairs Ds={(mi,si)}n
i=1.
Given the obtained set Ds, we construct both sin-
gle and multi-hop questions qsthat adhere to spe-
1https://en.wikipedia.org/wiki/Wikipedia:
Vital_articles/Level/5

cific criteria following previous works in the field.
Single-hop questions are designed to be straightfor-
ward, and answerable solely based on the informa-
tion contained within the main passage mi. In con-
trast, multi-hop questions necessitate information
from multiple passages within the pair {(mi,si)},
requiring several inferential steps to derive the final
answer. Furthermore, when employing GPT-4 with
specified prompt templates (see Figure 6 and 7) the
questions and long-form reference answers gener-
ated exhibit a high level of reasoning and analysis
capability.
4.1.3 Solution Annotation
The solution annotation resulting from the plan-
ning and action decision of the teacher model to
solve complex tasks is the key to effectively dis-
tilling the strong reasoning capabilities of student
models. In this process, we generate solution an-
notations for questions that include a series of
steps: Thought ,Action , and Evidence Feedback .
Starting with the original question qi, each step
t, GPT-4 is required to perform two tasks repli-
cating the real-world RAG scenario when the pro-
cess of retrieving external knowledge is needed:
i) provide a short rationale on how to utilize the
Search Tool to address the question (Thought rt
i)
and formulate a detailed search query (Action at
i)
to retrieve necessary information. ii, using the
search query at
iand the relevant sources {(mi,si)}
for the question qi, GPT-4 will extract the most
concise information from those sources and syn-
thesize it as Evidence Feedback et
i. The results
set at step t, comprising {rt
i, at
i, et
i}, are concate-
nated with the question and prior steps in the or-
der of qi, r1
i, a1
i, e1
i, ..., rt
i, at
i, et
iand used as the
input for the agent to determine the plan and ac-
tions in the subsequent step t+ 1. The process
continues until the agent concludes with a state-
ment "I have the final answer" indicating suffi-
cient evidence has been gathered. At this point,
denoted as step T, the final answer is also provided.
Finally, the solution annotation for the question
qiincludes thoughts ri={r1
i, ..., rT
i}, search
queries ai={a1
i, ..., aT−1
i}, evidence feedbacks
ei={e1
i, ..., eT−1
i}and final answer. Details of
prompts for the process are in Figure 8 and 9.
4.1.4 Annotation Verification
Since the data are generated by an LLM, there
are instances where the entire process may fail to
provide the final answer. To address this, we imple-ment both human and technical checks to ensure
the scalability and reliability of the process. Ad-
ditionally, we introduce an instruction eliminator,
referred to as the Verification Module, to filter out
failed annotations. We observe and hypothesize
that if the process can produce a final answer simi-
lar to the reference answer then the annotation qual-
ity is considered high. Using a specified prompt
template, GPT-4 is tasked with generating a brief
rationale and then assigning an integer score rang-
ing from 0 to 5, indicating the degree of similarity
between the predicted answer and the reference
answer and the relevancy to the input query. By
employing the Solution Verification Module to fil-
ter annotations, we ensure the quality of the dataset
by retaining only those annotations that achieve a
score of 4 or 5.
4.2 Dataset Analysis
After the annotation generation process, our dataset
comprises 16,987 annotated training samples and
1,197 testing samples. Figure 4 shows the distribu-
tion of question types and indicates that our dataset
largely consists of ’how’ questions confirming our
initial goal of constructing a dataset to enhance
the agent’s ability to reason and synthesize infor-
mation. In real-world applications, RAG systems
37.1%
24.9%19.9%6.9%5.1%6.2%how
what
which
who
where
other
Figure 4: Question type distribution on the training set.
typically handle relatively simple queries. To ac-
count for this, we deliberately incorporated a higher
proportion of queries requiring minimal search and
fewer pieces of supporting evidence. On average,
each training annotation in our dataset necessitates
two supporting passages. This design ensures that
our dataset reflects practical demands while ac-
commodating varying complexities of user queries.
Moreover, to the best of our knowledge, our dataset
is the first dataset to integrate Chain of Thought
(COT) reasoning, offering enhanced guidance for
the agent to interact with external knowledge.

5 Experiments
5.1 Experimental Setup
5.1.1 Datasets and Evaluation Metrics
We evaluate the unified RAG model using both
single-hop and multi-hop datasets. Specifically,
we employ six benchmark datasets: three single-
hop (SQuAD (Rajpurkar et al., 2016), Natural
Questions (Kwiatkowski et al., 2019), TriviaQA
(Joshi et al., 2017)) and three multi-hop (MusiQue
(Trivedi et al., 2022), HotpotQA (Yang et al., 2018),
2WikiMultiHopQA (Ho et al., 2020)). To compare
our model against recent state-of-the-art RAG sys-
tems on these datasets, which feature short-form
answers, we utilize F1, Exact Match (EM), and Ac-
curacy (Acc) as evaluation metrics. The F1 score
measures the overlap of words between the pre-
dicted and ground truth answers, EM checks for
exact matches, and Acc verifies whether the pre-
dicted answer includes the ground truth. To adapt
to the short-form answer setup, we use GPT-4 to
extract concise answers from the detailed responses
generated by the agent, as illustrated in Figure 10.
With each dataset, we benchmark on 500 samples
per dataset processed by (Jeong et al., 2024) and
(Trivedi et al., 2023).
Additionally, we evaluate performance on the
SynAgent-RAG test set, comparing small open-
source LLMs with larger models utilized as the
backbone for the Agent-UniRAG framework. We
employ ROUGE-L and BLEU metrics to assess
long-form answers. ROUGE-L, based on the
longest common subsequence (LCS), measures
similarity, while BLEU calculates n-gram preci-
sion, incorporating a brevity penalty to account for
fluency and accuracy. Given the distinct response
styles of each model, a comprehensive evaluation
requires assessing their ability to exhibit analytical
skills and produce logically coherent long-form re-
sponses. To this end, we also use GPT-Score, an
LLM-based evaluator, which prompts an LLM to
compare the generated answer with the reference
and input queries. GPT-Score specifically evalu-
ates the semantic alignment between the predicted
and reference answers, thereby providing a more
nuanced assessment of model performance.
5.1.2 Retrieval System and Corpus Setup
For the experiments on the short-form answer
datasets, to ensure a fair comparison with the
methodologies employed by (Jeong et al., 2024),
we utilize the BM25 retriever as the baseline re-trieval model across all corpus. In addition to the
BM25 baseline retrieval model, we also experi-
ment with adding the Multilingual E5 Large model
(Wang et al., 2024b) as the dense reranker after
the sparse BM25 retrieval step to observe the ef-
fect of better retrieving results can lead to better
agent performance. For the external corpus, we
index different sources for different dataset types.
Specifically, for single-hop datasets, follow (Jeong
et al., 2024) we use the Wikipedia corpus prepro-
cessed by (Karpukhin et al., 2020), while for multi-
hop datasets, we use the corpus preprocessed by
(Trivedi et al., 2023).
For the experiment on the test set of our
SynAgent-RAG dataset, instead of indexing docu-
ments into a corpus, we focus on measuring the
model’s reasoning capability under optimal re-
trieval conditions. Here, we leave out the perfor-
mance of retrieval systems and assume that the
retrieved documents are correct and relevant to the
original question by directly returning the reference
documents as the results of the retrieval phase.
5.1.3 Models
In this study, we compare our approach, Agent-
UniRAG, against several retrieval-augmented LLM
strategies, including Self-RAG (Asai et al., 2023)
which adaptively retrieves passages on-demand,
and generates and reflects on retrieved passages,
Adaptive-RAG (Jeong et al., 2024), which dynami-
cally adjusts retrieval based on question complexity,
andIRCoT (Trivedi et al., 2023), a state-of-the-art
method leveraging iterative retriever and LLM in-
teraction through Chain-of-Thought In-Context rea-
soning. The baseline models in these methods uti-
lize GPT-3.5-Turbo, known for its larger size com-
pared to our approach, which is based on LLama-
3-Instruct. To further assess the effectiveness of
our framework, we conducted an ablation study on
multihop datasets. First, we removed the Reflector
Module to assess the impact of directly utilizing
the retrieved knowledge, which may include noise,
as evidence feedback for the agent, whether it will
lead to degradation in the performance. Second,
we evaluated the effect of bypassing the gradual
retrieval process by removing the Planning Mod-
ule. In this scenario, the LLM was tasked with
generating all necessary queries first, subsequently
using the retrieved information to directly answer
the input query. This setup helps understand the
importance of iterative information retrieval in en-
hancing the agent’s decision-making accuracy.

ModelMax
SearchTop K/
BiEncoderSQUAD Natural Question TrivialQA
EM F1 Acc EM F1 Acc EM F1 Acc
Self-RAG* No limit 1.6 11.9 20.8 39.2 47.1 42.4 14.6 33.7 60.2
IRCoT* No limit 17.4 31.5 26.2 35.6 49.7 57.8 54.8 67.1 68.0
Adaptive-RAG* No limit 18.0 33.8 29.2 32.4 46.8 54.8 55.2 66.5 65.8
Agent-UniRAG 1 8 / No 23.8 34.5 49.6 43.4 51.6 61.2 57.6 65.8 71.2
Agent-UniRAG 1 12 / No 26.6 38.1 48.6 45.2 53.9 61.2 57.0 66.2 69.0
Agent-UniRAG No limit 8 / No 26.4 38.6 42.2 45.8 55.3 57.6 58.6 66.7 70.0
Agent-UniRAG No limit 12 / No 28.2 40.8 42.2 48.0 57.3 58.8 57.4 67.2 67.4
Agent-UniRAG No limit 12 / Yes 32.8 46.9 42.8 59.2 68.6 64.6 63.6 72.5 71.0
Table 1: Results on different single-hop benchmark datasets. * are taken from Jeong et al. (2024) with GPT-3.5 as
the backbone LLM for both previous approaches. Bold texts indicate the best results.
ModelMax
SearchTop K/
BiEncoderMuSiQue HotpotQA 2WikiMultiHopQA
EM F1 Acc EM F1 Acc EM F1 Acc
Self-RAG* No limit 1.2 8.2 11.8 5.6 17.8 30.6 3.0 19.1 39.0
IRCoT* No limit 23.0 32.5 31.6 45.8 58.3 52.2 52.2 66.0 62.4
Adaptive-RAG* No limit 21.8 32.6 29.6 40.4 52.5 47.0 46.6 60.0 56.8
Agent-UniRAG No limit 8 / No 26.4 35.2 27.8 47.6 56.2 48.8 60.2 66.7 61.8
Agent-UniRAG No limit 12 / No 26.2 35.3 28.2 48.6 58.2 50.6 59.8 66.6 61.8
Agent-UniRAG No limit 12 / Yes 30.4 39.8 32.2 50.2 59.9 52.4 58.4 64.9 60.6
w/o Evidence Reflector No Limit 12 / No 20.2 29.9 21.4 49.4 59.9 52.2 51.2 57.94 53.2
w/o Planning 1 12 / No 10.2 15.5 11.4 37.4 43.2 37.6 36.8 43.5 37.6
Table 2: Results on different multi-hop benchmark datasets. * are taken from Jeong et al. (2024) with GPT-3.5 as
the backbone LLM for both previous approaches. Bold texts indicate the best results.
5.1.4 Training Configurations
Agent-UniRAG uses the instruction version of
Meta-Llama-3-8B2as the backbone open-source
LLM model, and fine-tune instruction on the pro-
posed SynAgent-RAG dataset. The fine-tuning pro-
cess spanned 10 hours on a single DGX node with 8
A100 GPUs, each equipped with 40GB of VRAM.
The learning rate was set at 2e−5, and the global
batch size was 256. The model was trained for 2
epochs using the AdamW optimizer (Loshchilov
and Hutter, 2017).
5.1.5 Training Prompt Template
We distill Agent-UniRAG in a multi-task setting by
fine-tuning three subtasks following the proposed
framework to guide its planning, action decisions,
and filter evidence feedback. Annotations are orga-
nized in conversational formats to facilitate interac-
tion between components, which include:
Conversation planning module annotation : As
illustrated in Figure 12, we start by using the user
role to provide the question qin the prompt. The
planning module then appends the first thought r1
and the initial search query a1as the first response
supervision. For subsequent turns, we act as the
2https://huggingface.co/meta-llama/Meta-Llama-3-8B-
Instructuser and provide the extracted evidence et−1of the
last search query at−1to the planning module. The
response supervision dictates whether the planning
should terminate by the thought “I have the final
answer. ” ; if not, the response should include a new
thought rtalong with a new search query at.
Conversation final answer annotation : Instead
of letting the LLM generate the final answer in the
planning module as in the data generation process,
we want to add more control to the pipeline by
separating the task of providing the final answer
to a subtask. In that way, we collect the gathered
evidence {e1, ..., eT−1}and provide the question
as the user prompt and treat the final answer as the
response (depicted in Figure 14).
Conversation Evidence Reflector annotation : As
shown in Figure 13, we provide the search query
atand the relevant source containing the main-
supporting passages pair {m,s}corresponding to
the user turn. All the extracted evidence etserves
as the user’s prompt response.
Since SynAgent-RAG annotations are
conversational, we structure them as
{x1, y1, . . . , x i, yi, . . . , x n, yn}, where xiis
thei-th user prompt and yiindicates its re-
sponses. During training, we input each entire
multi-turn annotation into the model, calculat-

ing loss solely on the tokens of the responses
Y={y1, . . . , y i, . . . , y n}and applying binary
masking on the user prompt tokens to prevent
computing loss on them. The final loss function is
L=−X
jlogpπ(tj|t<j)×1(tj∈Y)(1)
where tjdenotes the j-th input token and 1(·)is a
Boolean indicator function.
5.2 Main Results
We present a detailed performance comparison of
the proposed approach with previous methods, as
shown in Table 1 for single-hop datasets and Table
2 for multi-hop datasets. Notably, Agent-UniRAG,
which leverages a small open-source LLM as its
backbone, demonstrates competitive performance
relative to recent state-of-the-art models that utilize
significantly larger LLMs. A key strength of our
model is its ability to handle diverse query types
uniformly and simultaneously. In addition to that,
we have three specific observations.
Agent-UniRAG caneffectively interact with the
external knowledge base. We observe that in-
creasing the search limits and the number of top-K
retrieved documents leads to performance improve-
ments. Specifically, with the top-K retrieval set to
12 and the integration of a dense encoder module
for reranking, our proposed Agent-UniRAG sub-
stantially outperforms previous methods, achieving
state-of-the-art results on the majority of bench-
mark RAG datasets in this research field. Addition-
ally, in the single-hop settings, when the maximum
search limit is increased from 1 to ’No limit’ we ob-
serve a further increase in performance, highlight-
ing the LLM agent’s capability to interact with evi-
dence feedback and then reason and refine search
queries to gather better retrieval results.
The importance ofdesigned modules inthe
pipeline. In the multi-hop reasoning setting, with
the conducted ablation studies to assess (Table 2),
removing the Evidence Reflector module resulted
in noticeable performance degradation, particularly
in more complex datasets like MuSIQue (Trivedi
et al., 2022), underscoring the critical role of the
Evidence Reflector in provisioning concise and rel-
evant evidence feedback to help the agent make
better subsequent decisions. We also removed the
Planning module, which serves as the central com-
ponent of the pipeline. The removal of this module
led to a more substantial decline in performancemetrics, thereby illustrating its pivotal role in or-
chestrating the agent’s multi-step reasoning process
and the necessity of iterative information retrieval.
Agent LLM Rouge-L BLEU GPT-Score Step
Llama-3-70B-Inst 0.36 0.12 3.62 5.68
GPT-4-Turbo 0.35 0.13 4.35 2.27
Agent-UniRAG 0.36 0.15 4.19 2.08
Table 3: Agent-UniRAG in compare with LLama-3-
70B-Inst and GPT-4-Turbo on SynAgent-RAG test set.
Effectiveness ofSynAgent-RAG dataset in
distilling the reasoning capability. Table 3
presents the results on the test set of SynAgent-
RAG datasets. Upon analysis, it becomes evident
that traditional metrics like Rouge-L and BLEU,
focused on the lexical overlap, is insufficient for
evaluating the reasoning and accuracy in long-
form answers. In contrast, GPT-Score, leveraging
LLMs for semantic evaluation, provides a more
accurate assessment. As a result, our proposed
Agent-UniRAG model, which is finetuned on the
SynAgent-RAG training set, demonstrates strong
performance based on GPT-Score, achieving com-
parable results to significantly larger models such
as LLaMA-70B-Inst and GPT-4. Notably, Agent-
UniRAG achieves this level of performance while
utilizing fewer external search queries, highlight-
ing its computational efficiency compared to more
demanding computational resources typically re-
quired by larger LLMs and as an efficient solu-
tion for generating accurate long-form answers.
This result also underscores the effectiveness of
the SynAgent-RAG dataset in distilling reasoning
capabilities from a larger LLM (GPT-4) into a more
compact framework.
6 Conclusion
Most previous works for LLM RAG tasks ei-
ther handle single-hop queries with unnecessary
computational complexity or fail to address com-
plex multi-hop queries. This study, inspired by
the emerging LLM agent technologies, presents a
novel approach for unified RAG systems with en-
hanced effectiveness and interpretability. Further-
more, we introduce SynAgent-RAG, a synthetic
dataset that enables trainable open-source LLM
agents for the unified RAG task. Compared with
previous works using larger LLMs, the experiment
shows promising results of the Agent-UniRAG
with a small backbone (i.e., Llama-3-8B).

Limitations
Agent-UniRAG, the unified Retrieval-Augmented
Generation (RAG) model, has shown promising re-
sults in managing different types of queries includ-
ing single-hop and multi-hop queries. Notably, this
approach can be applied to open-source small lan-
guage models (LLMs). However, real-world input
often encompasses scenarios when queries are of-
ten more than just single-hop or multi-hop queries
requiring access to external knowledge bases, they
may also include non-RAG tasks such as creative
writing or function calling. Therefore, a crucial
direction for future research is to extend the pro-
posed approach to handle broader types of queries,
including both RAG and non-RAG types. Addi-
tionally, similar to other LLM agent architectures,
Agent-UniRAG requires multiple calls to the lan-
guage model to generate a final response, which
introduces a computational challenge during infer-
ence. Consequently, optimizing LLM inference for
agent architectures is another critical aspect of our
future work.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
CoRR, abs/2310.11511.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, Tom Hennigan,
Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, Andy Brock, Michela Paganini, Geoffrey
Irving, Oriol Vinyals, Simon Osindero, Karen Si-
monyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.
2022. Improving language models by retrieving
from trillions of tokens. In International Conference
onMachine Learning, ICML 2022, 17-23 July
2022, Baltimore, Maryland, USA , volume 162 of
Proceedings ofMachine Learning Research , pages
2206–2240. PMLR.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting llms: To-
wards retrieval-augmented large language models. In
Proceedings ofthe30th ACM SIGKDD Conference
onKnowledge Discovery andData Mining, KDD
2024, Barcelona, Spain, August 25-29, 2024 , pages
6491–6501. ACM.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR, abs/2312.10997.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. In Proceedings ofthe28th International
Conference onComputational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020 , pages 6609–6625. International Committee on
Computational Linguistics.
Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli,
Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2023. Atlas: Few-shot learning
with retrieval augmented language models. J.Mach.
Learn. Res., 24:251:1–251:43.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the2024 Conference oftheNorth American Chapter
oftheAssociation forComputational Linguistics:
Human Language Technologies (V olume 1:Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024 , pages 7036–7050. Association for Com-
putational Linguistics.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings ofthe55th Annual Meeting
oftheAssociation forComputational Linguistics,
ACL 2017, Vancouver, Canada, July 30-August 4,
V olume 1:Long Papers , pages 1601–1611. Associa-
tion for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Ledell
Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
2020. Dense passage retrieval for open-domain ques-
tion answering. CoRR, abs/2004.04906.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur P. Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: a benchmark for question answering
research. Trans. Assoc. Comput. Linguistics , 7:452–
466.
Ilya Loshchilov and Frank Hutter. 2017. Decou-
pled weight decay regularization. In International
Conference onLearning Representations.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings
ofthe2023 Conference onEmpirical Methods in
Natural Language Processing , pages 5303–5315,
Singapore. Association for Computational Linguis-
tics.

OpenAI. 2024. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev,
and Percy Liang. 2016. Squad: 100, 000+ ques-
tions for machine comprehension of text. In
Proceedings ofthe2016 Conference onEmpirical
Methods inNatural Language Processing, EMNLP
2016, Austin, Texas, USA, November 1-4, 2016 ,
pages 2383–2392. The Association for Computa-
tional Linguistics.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions oftheAssociation for
Computational Linguistics, 11:1316–1331.
Sina J. Semnani, Violet Z. Yao, Heidi C. Zhang, and
Monica S. Lam. 2023. Wikichat: Stopping the hal-
lucination of large language model chatbots by few-
shot grounding on wikipedia. In Findings ofthe
Association forComputational Linguistics: EMNLP
2023, Singapore, December 6-10, 2023 , pages 2387–
2413. Association for Computational Linguistics.
Yongliang Shen, Kaitao Song, Xu Tan, Dong-
sheng Li, Weiming Lu, and Yueting Zhuang.
2023. Hugginggpt: Solving AI tasks with
chatgpt and its friends in hugging face. In
Advances inNeural Information Processing Systems
36: Annual Conference onNeural Information
Processing Systems 2023, NeurIPS 2023, New
Orleans, LA,USA, December 10-16,2023.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2024. Re-
flexion: Language agents with verbal reinforce-
ment learning. Advances inNeural Information
Processing Systems, 36.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries. CoRR, abs/2401.15391.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings
ofthe61st Annual Meeting oftheAssociation
forComputational Linguistics (V olume 1:Long
Papers) , pages 10014–10037, Toronto, Canada. As-
sociation for Computational Linguistics.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao
Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei
Wei, and Jirong Wen. 2024a. A survey on large
language model based autonomous agents. Frontiers
ofComputer Science, 18(6).Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024b. Mul-
tilingual e5 text embeddings: A technical report.
Preprint, arXiv:2402.05672.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. In Proceedings ofthe2018 Conference on
Empirical Methods inNatural Language Processing,
Brussels, Belgium, October 31-November 4,2018 ,
pages 2369–2380. Association for Computational
Linguistics.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. In The Eleventh International Conference
onLearning Representations, ICLR 2023, Kigali,
Rwanda, May 1-5,2023. OpenReview.net.
Da Yin, Faeze Brahman, Abhilasha Ravichander, Khy-
athi Chandu, Kai-Wei Chang, Yejin Choi, and
Bill Yuchen Lin. 2023. Agent Lumos: Unified and
Modular Training for Open-Source Language Agents.
arXiv preprint arXiv:2311.05657.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu
Lin, Yong-Jin Liu, and Gao Huang. 2024.
Expel: LLM agents are experiential learners.
InThirty-Eighth AAAI Conference onArtificial
Intelligence, AAAI 2024, Thirty-Sixth Conference
onInnovative Applications ofArtificial Intelligence,
IAAI 2024, Fourteenth Symposium onEducational
Advances inArtificial Intelligence, EAAI 2014,
February 20-27, 2024, Vancouver, Canada , pages
19632–19642. AAAI Press.
A Appendix
B Illustrations of Agent-UniRAG Sample
Tables 4 and 5 illustrate our data with examples,
including the question, thought process, actions
taken, evidence gathered, and the final long-form
answer.
C Prompt Templates
For better reproducibility, we present all prompt
templates in the appendix. Below is a quick refer-
ence list outlining the prompt templates and their
usages:
•Figure 5: Find related sections from linked
articles to the main article.
•Figure 6: Generate a multi-hop question from
a cluster of related sections.
•Figure 7: Generate a singlehop question from
a section.

Question Agent-UniRAG Output
What highway was
renamed in honor of
Tim Russert?###Thought : The first step is to use the search_engine tool to find which highway
was renamed in honor of Tim Russert.
###Action - Search Input : highway renamed in honor of Tim Russert
###Evidence : On July 23, 2008, U.S. Route 20A leading to the Buffalo Bills’
Ralph Wilson Stadium in Orchard Park, New York was renamed the ’Timothy J.
Russert Highway’ in honor of Tim Russert.
###Thought : I have the final answer
###Action - Final Answer : U.S. Route 20A was renamed the ’Timothy J. Russert
Highway’ in honor of Tim Russert.
Table 4: Example of Agent-UniRAG output with single-hop query
Question Agent-UniRAG Output
How do Jerry Falwell’s
beliefs about the
Antichrist as a specific
person contrast with
Martin Wight’s
interpretation of
the Antichrist concept
after World War II?###Thought : Understand Jerry Falwell’s beliefs about the Antichrist as a
specific person.
###Action - Search Input : Jerry Falwell beliefs about the Antichrist
###Evidence : In 1999, Jerry Falwell professed that the Antichrist would likely
emerge within a decade, with the prediction that this figure would be Jewish,
which led to allegations of anti-Semitism and an ensuing apology from Falwell.
He believed the Antichrist and Christ would share many attributes.
###Thought : Explore Martin Wight’s interpretation of the Antichrist concept
after World War II to contrast with Falwell’s views.
###Action - Search Input : Martin Wight interpretation of the Antichrist
concept after World War II.
### Evidence: Post-World War II, Christian and political theorist Martin Wight
interpreted the Antichrist not as an individual, but as a symbol of ’demonic
concentrations of power’ recurring over time.
###Thought : I have the final answer
###Action - Final Answer : Jerry Falwell’s beliefs about the Antichrist focused
on the idea that the Antichrist would be a specific individual, with Falwell
predicting in 1999 that this person would likely emerge within a decade and
possess similarities to Christ. In contrast, Martin Wight’s post-World War II
interpretation of the Antichrist was symbolic, referring to it as ’demonic
concentrations of power’ that recur throughout history, rather than an individual
figure.
Table 5: Example of Agent-UniRAG output with multi-hop query
•Figure 8: Extract related evidence for a search
query from a list of source content.
•Figure 9: Generate solution annotation for a
question.
•Figure 10: Extract short form answer from a
long form answer.
•Figure 11: Take GPT Score of an annotated
answer to the reference answer.
•Figure 12: Training prompt template for the
agent to reason and use tools.
•Figure 13: Training prompt template for theagent to extract related evidence for a query
from sources of content.
•Figure 14: Training prompt template for the
agent to provide the final answer for the ques-
tion from gathered evidence
All prompts are zero-shot, except for the prompt in
Figure 10, which uses few-shot demonstrations to
better guide the LLM to perform the task. These
prompts were chosen because they perform effec-
tively in practice.

### Given a source of a Wikipedia article containing [[hyperlinks]] and a list of section titles from the linked articles, identify
section titles that are most relevant to supporting the main topic.
### Wikipedia article:
{
  "article_title": "actual article title",
  "section_title": "actual section title",
  "content": "actual content"
}
### Hyperlinks with Section Titles:
[
  {
    'entity': '[[actual entity in Wikipedia article marked in double square barackets]]',
    'article_title': 'linked article title',
    'article_sections': ['list of linked article section titles']
  }, ...
]
### Notes:
1) You can select ONL Y one section title per article
2) Response must be in the following JSON format: [
  {
    "rationale": "condensed and short reason why you select the section",
    "article_title": "article_title", 
    "section_title": "article_section_title"
  },...
]
3) You can select up to {{k}} most proper section titlesFigure 5: Prompt template for GPT4 to find related section content from articles.
### Clarification:
1) A multi-hop question is a question that requires multiple inferential leaps or accessing several pieces of information from
different sources to arrive at a final answer .
2) You will be given sources of articles, your job is to generate a multi-hop question and then provide the answer for the question
based on the provided sources.
### Sources
[
  {
    "article_title": "actual article title",
    "section_title": "actual section title",
    "content": "actual content"
  },...
]
### Notes:
1) The question cannot be answered by relying on any single article alone but instead requires the solver to gradually gather and
search for pieces of evidence within ALL  the provided sources then understand and link information to take the next action, and
finally give back the answer .
2) Make sure the question flows logically and is unambiguous.
3) The information in the answer MUST  be derived from the sources
3) Response in the following JSON format: {"question": "your question", "answer": "correct answer for the question"}
4) Do not mention the source of information in the question or the answer .
Figure 6: Prompt template for GPT4 to generate multi-hop questions.

### You will be given a source of article. Your job is to create a relevant question to the source and then provide the answer for
the question based on the provided source :
### Source:
{
    "article_title": "actual article title",
    "section_title": "actual section title",
    "content": "actual content"
}
### The question MUST  satisfy the following conditions:
1) The question must related to the content of the source.
2) Make sure the question is simple enough and unambiguous.
3) The question requires synthesizing information from the source to answer .
3) The information in the answer MUST  be derived from the sources
5) Do NOT  mention the source in the question or in the answer .
### Response MUST  be in the following JSON format: 
{
  "question": "Y our question here", 
  "answer": "the detailed answer to the question"
}Figure 7: Prompt template for GPT4 to generate single-hop questions.
### Task: Synthesize a condensed text evidence from given sources to support a search query .
### Sources:
[
  {
    "article_title": "actual article title",
    "section_title": "actual section title",
    "content": "actual content"
  },...
]
### Search Query: {{actual search query}}
### Selection Guidelines:
1. Clarity: Evidence must be clear , concise.
2. Conciseness: Evidence must be presented in a succinct manner , condensed and AVOIDING unnecessary details.
3. Relevance: Evidence must directly correspond and relevant to the search query .
4. Source Integrity: Only use information from the provided sources, AVOIDING generated or unnecessary information.
5. If multiple part of a source is relevant to the search query , combine them into one element in the response list.
### Response MUST  be in a JSON list as below:
[
  {
    "evidence": "condensed text supporting the search query from a source",
    "source_id": "an identifier of the source text"
  }
]
If no evidence is found, respond the following json:
[
  {
    "evidence": "No supporting evidence found.",
    "source_id": null
  }
]
Figure 8: Prompt template for GPT4 to extract evidence from a list of sources

You will be utilizing the following tool to assist with answering questions:
{
  "tool_name": "search_engine",
  "tool_description": "This tool can search an external knowledge base to find text evidence for the provided search query .",
  "tool_note": "Y ou can not search for multiple information at once. Please provide one clear and detailed query for a small piece
of evidence you want at a time.",
  "tool_input": [
    {
      "param_name": "search_query",
      "param_type": "string",
      "param_description": "A  detailed search query to search within the knowledge base to get some pieces of evidence"
    }
  ]
}
Your task is to solve a provided question by using the tool. Follow these steps:
1) Reasoning step by step how you will use the tool to solve the question.
2) You can only use one tool each time then you get the response and continue.
3) Provided that you DO NOT  have any initial knowledge about the information mentioned in the question and DO NOT  generate
facts or evidence yourself.
4) Provide a CLEAR and CONCISE answer .
5. Format responses to utilize the search_engine tool as follows:
### Thought: A short and condensed rationale for using the search_engine tool. (one sentence)
### Search Input: Format the search query input for the search_engine tool as a JSON object, correctly representing input
parameters.
### Observation: the text evidence after searching that will be given to you. (will be [W AITING] when you are waiting for the
evidence response, also note that the observation sometimes can be partially related or not related to the search query , you need to
reason and continue to use the tool until you have a response for the user . DO NOT  generate observation yourself.)
When you have a response for the user , or if you do not need to use a tool, use the format:
### Thought: I have the final answer
### Final Answer: your condensed answer to the main 
Let's begin with the question:
{{actual question}}Figure 9: Prompt template for GPT4 to reason, use tools and provide the final answer for a question

### Task: You are doing the Extractive Question Answering task. You will be given a question and a reference answer . Your task
is to extract exactly a list of text spans inside the reference answer that can serve as the short answers for the question.
### Question: {{actual question}}
### Reference Answer: {{actual answer}}
### Notes:
1) Your response MUST  be in JSON format {"short_extracted_answers": ["extracted answer 1", ...]}
2) The extracted answers MUST  be united and interchangeable, try to combine nearby words in the Reference Answer to from an
answer
3) Do not generate answers or information yourself
4) If the question is a yes/no question, then you should base it on the reference answer to return yes or no as the extracted answer .
5) If you can not extract the answer or the answer is not provided in the Reference Answer , then
respond: {"short_extracted_answers": null}
### Examples:
Question: What percentage of French publishing houses were in Paris in the 1970s?
Reference Answer: In the 1970s, 80 percent of French-language publishing houses were located in Paris.
Response: {"short_extracted_answers": ["80 percent", "80"]}
Question: When did Claridge's company liquidate?
Reference Answer: Claridge's company liquidated on November 10, 1917.
Response: {"list_of_short_extracted_answers": ["November 10, 1917", "10 November 1917", "November 1917", "1917"]
Question: Do both Icehouse pieces and El Grande come from the same game?
Reference Answer: No, Icehouse pieces and El Grande do not come from the same game. Icehouse pieces are from the game
system of the same name, invented by Andrew Looney and John Cooper , while El Grande is a German-style board game designed
by Wolfgang Kramer and Richard Ulrich.
Response: {"list_of_short_extracted_answers": ["No"]}Figure 10: Prompt template to extract short answer from long answer
### Task: You are a powerful and accurate assistant in checking the quality of a predicted answer . You will be given a predicted
answer , a question and a reference answer .
### Here are some criteria for you to grade the predicted answer:
1) The score MUST  be an integer range from 0 to 5.
2) The content of the predicted answer should be relevant and focus on the question.
3) Any missing or excess information in the predicted answer compared to the reference answer will be penalized in the final
score.
4) If the question is a question that requires the analysis of information, then you should reinforce the above criteria.
### Question: {{question}}
### Reference Answer: {{reference answer}}
### Predicted Answer:  {{predicted answer}}
### Note: Your response MUST  be in the following JSON format and do NOT  generate unnecessary details beyond the JSON
object
{
  "rationale": "Y our brief rationale for how you scored the predicted answer",
  "score": "the score of the predicted answer"
}
Figure 11: Prompt template for GPT4 to compare and score the predicted answer and the reference answer.
### Task: You are a problem solver . Given a question, your task is to solve the question by gradually gathering information with
search_engine tool and provide the final answer to the question.
### Question: {actual question}
### Thought: {agent thougth 1}
### Action: {agent action 1}
### Search Input: {agent search input}
### Observation:  {observation provided by the environment}
....
### Thought: I have the final answer .
### Action: final_answer
Figure 12: Training prompt template for Agent-UniRAG to reason and use tools. Loss is computed only on the red
part as the GPT turns in the conversation setup.

### Task: Extract the relevant information from the sources to support the following query .
### Sources:
[
  {
    "source_id": "an integer start from 0",
    "content": "related content to the query"
  }
] 
### Query: {agent query}
[
  {
    "source_id": "related source id",
    "evidence": "evidence from source id"
  }, ...
]Figure 13: Training prompt template for Agent-UniRAG extract related evidence for a query from sources of
content. Loss is computed only on the red part as the GPT turns in the conversation setup.
### Task: Given the question and a list of evidence, your task is to provide the final answer to the question based on the information within
the evidence.
### Evidence: {concatenated text of gathered evidence}
### Question: {initial question}
{final answer}
Figure 14: Training prompt template for Agent-UniRAG to provide final answer from gathered evidence. Loss is
computed only on the red part as the GPT turns in the conversation setup.
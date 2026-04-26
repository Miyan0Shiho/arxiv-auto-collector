# MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation

**Authors**: Xingchen Xiao, Heyan Huang, Runheng Liu, Jincheng Xie

**Published**: 2026-04-20 17:00:38

**PDF URL**: [https://arxiv.org/pdf/2604.18509v2](https://arxiv.org/pdf/2604.18509v2)

## Abstract
Large language models (LLMs) are widely used in retrieval-augmented generation (RAG) to incorporate external knowledge at inference time. However, when retrieved contexts are noisy, incomplete, or heterogeneous, a single generation process often struggles to reconcile evidence effectively. We propose \textbf{MASS-RAG}, a multi-agent synthesis approach to retrieval-augmented generation that structures evidence processing into multiple role-specialized agents. MASS-RAG applies distinct agents for evidence summarization, evidence extraction, and reasoning over retrieved documents, and combines their outputs through a dedicated synthesis stage to produce the final answer. This design exposes multiple intermediate evidence views, allowing the model to compare and integrate complementary information before answer generation. Experiments on four benchmarks show that MASS-RAG consistently improves performance over strong RAG baselines, particularly in settings where relevant evidence is distributed across retrieved contexts.

## Full Text


<!-- PDF content starts -->

MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation
Xingchen Xiao1, Heyan Huang1*, Runheng Liu1, Jincheng Xie2
1School of Computer Science and Technology, Beijing Institute of Technology
2Department of Mathematical Sciences, Tsinghua University
{xcxiao,hhy63,rhliu}@bit.edu.cn
xiejc22@mails.tsinghua.edu.cn
Abstract
Large language models (LLMs) are widely
used in retrieval-augmented generation (RAG)
to incorporate external knowledge at infer-
ence time. However, when retrieved contexts
are noisy, incomplete, or heterogeneous, a
single generation process often struggles to
reconcile evidence effectively. We propose
MASS-RAG, a multi-agent synthesis approach
to retrieval-augmented generation that struc-
tures evidence processing into multiple role-
specialized agents. MASS-RAG applies dis-
tinct agents for evidence summarization, evi-
dence extraction, and reasoning over retrieved
documents, and combines their outputs through
a dedicated synthesis stage to produce the final
answer. This design exposes multiple interme-
diate evidence views, allowing the model to
compare and integrate complementary informa-
tion before answer generation. Experiments
on four benchmarks show that MASS-RAG
consistently improves performance over strong
RAG baselines, particularly in settings where
relevant evidence is distributed across retrieved
contexts.
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable scalability and emergent capa-
bilities (Brown et al., 2020; Wei et al., 2022; Tou-
vron et al., 2023; OpenAI et al., 2024). However,
they remain susceptible to generating unreliable or
hallucinatory outputs due to limited factual ground-
ing and static knowledge. To address these limita-
tions, Retrieval-Augmented Generation (RAG) has
emerged as an effective paradigm to enhance fac-
tual reliability and broaden the knowledge scope of
LLMs (Huang et al., 2023; Xu et al., 2024b; Asai
et al., 2024). Updating the parametric knowledge of
LLMs through fine-tuning or continual pre-training
is both computationally intensive and costly (Meng
*Corresponding author.
Multi-Agent
Filter
Filtered 
Evidence #1
Filtered 
Evidence #2
Filtered 
Evidence #3
.
.Summarizer 
Agent
Extractor 
Agent
Reasoner 
Agent
Synthesis AgentBased on the evidence,
the best answer is ...
Single Agent
FIlter  
Judge 
Agent
The answer is ...
QA assistant
Previous MethodMASS-RAG (Ours)
Filtered
Documents
Source
Query
Source
QueryFigure 1: A conceptual illustration of the key idea be-
hind MASS-RAG, contrasting the multi-agent evidence
filtering and synthesis with the previous single-filter
agent method.
et al., 2023), making RAG integration a practical
alternative for maintaining up-to-date knowledge
at a lower cost (Balaguer et al., 2024).
Previous efforts on the RAG system have
explored either pre-training models from
scratch(Borgeaud et al., 2022) or continually
pre-training existing models to incorporate
external knowledge (Izacard et al., 2022b; Wang
et al., 2023a). However, such approaches are
computationally expensive and resource-intensive,
which limits their scalability in real-world RAG
applications(Gao et al., 2024). As a result, recent
RAG research has largely focused on combining
retrievers for document selection with large
language models (LLMs) for generation(Ram
et al., 2023; Lin et al., 2023; Shi et al., 2023). By
leveraging external non-parametric knowledge
sources, these systems enable knowledge updates
without retraining the underlying LLMs. Never-
1arXiv:2604.18509v2  [cs.CL]  21 Apr 2026

theless, LLMs remain susceptible to irrelevant or
redundant retrieved information(Liu et al., 2023;
Cho et al., 2023), which can degrade generation
quality. Consequently, a key challenge is to ensure
that RAG systems produce robust and factually
accurate outputs when retrieved contexts are noisy
or incomplete due to the inherent nature of the
retrieval models (Gao et al., 2024; Asai et al.,
2024). While recent multi-agent RAG framework
proposed by Chang et al. (2024) introduce
agent-based context filtering, they typically rely on
a single judge agent operating from a monolithic
perspective, which limits the system’s ability to
capture complementary or heterogeneous forms of
factually relevant evidence.
To address this challenge, we introduceMASS-
RAG, a Multi-Agent Synthesis framework for
retrieval-augmented generation. It operates in a
training-free manner and employs multiple agents
with specialized roles for context filtering, response
generation, and answer synthesis.MASS-RAG
integrates multiple specialized agents to capture
complementary evidence in the retrieved context,
thereby improving robustness and factual accuracy
through structured multi-agent evidence filtering
and synthesis.
Our contributions are as follows:
•Multi-Agent Evidence Filtering:We in-
troduce a set of specialized LLM-based fil-
ter agents that process retrieved documents
from complementary perspectives, distilling
denoised and query-relevant evidence in a
training-free manner.
•Multi-Agent Answering and Synthesis:We
propose a synthesis mechanism that integrates
structured outputs from multiple filter agents,
optionally via intermediate candidate answers,
to produce a unified final prediction. This
design enables explicit comparison across het-
erogeneous evidence views and supports task-
adaptive configurations.
•Empirical Validation:We evaluate MASS-
RAG across multiple benchmarks and analysis
settings, demonstrating consistent improve-
ments over strong RAG baselines, especially
in scenarios requiring the aggregation of com-
plementary evidence.2 Related Work
Evidence Refinement in RAGA variety of
methods have been proposed to optimize the input
context for retrieval-augmented generation (RAG).
One intuitive line of work focuses on reranking
retrieved documents (Glass et al., 2022), which has
been widely adopted and extended in subsequent
studies (Ram et al., 2023; Asai et al., 2023; Hwang
et al., 2024; Xu et al., 2024a). Other approaches
aim to refine the input context through query rewrit-
ing (Ma et al., 2023; Chan et al., 2024), context
filtering (Wang et al., 2023b; Yang et al., 2023), or
content compression and summarization (Xu et al.,
2023; Hwang et al., 2024). These methods primar-
ily operate by improving the relevance or compact-
ness of the retrieved context prior to generation.
Our work follows the line of training-free meth-
ods but differs in that we introduceMulti-Agent
Synthesis (MASS),MASS-RAGadopts a multi-
perspective filtering approach, employing multiple
specialized agents to distill complementary and de-
noised evidence from retrieved documents while
preserving query-relevant information.
Multi-Agent RAGRecent studies have explored
the use of multiple agents to enhance retrieval-
augmented generation. Instead of relying on a
single retriever–generator pipeline, multi-agent
RAG frameworks distribute different sub-processes
among specialized agents, such as query refor-
mulation (Chen et al., 2025), query decompo-
sition (Nguyen et al., 2025), and document fil-
tering (Chang et al., 2024). These approaches
have demonstrated improved robustness and per-
formance over traditional single-pipeline RAG sys-
tems. Our work follows the training-free setting of
prior multi-agent RAG methods, but differs in its
focus on evidence-centric synthesis. Specifically,
MASS-RAGintroduces a multi-agent synthesis
framework in which distinct agents filter retrieved
documents from complementary perspectives and
generate denoised evidence representations that are
subsequently integrated to produce a robust final
answer.
3 Multi-Agent Synthesis RAG
(MASS-RAG)
3.1 Definition of LLM Agents in MASS-RAG
MASS-RAGdecomposes retrieval-augmented gen-
eration into three stages: evidence distillation, can-
didate answer generation, and final answer synthe-
2

Document Retrieval Filter AgentAnswer Agent
(Optional)
Sythesis Agent
QA assistant
Synthesizer
DatabaseRetriever
Retrieved 
DocumentsFiltered
Evidence #1
Filtered
Evidence #2
Filtered
Evidence #3Summarizer
Extractor
ReasonerCandidate #1
Answer
Candidate #2
Answer
Candidate #3
AnswerFinal
Answer
Query
Query
Query
LLM-based Agent Generate Input flow
Search
Figure 2: The overall architecture of MASS-RAG, illustrating multi-agent filtering and answer synthesis when the
Answer Agent is enabled.
sis. Accordingly, it consists of several specialized
LLM agents, including a Summarizer, an Extrac-
tor, a Reasoner, a Synthesis Agent, and an optional
Answer Agent. Among them, the Summarizer, Ex-
tractor, and Reasoner serve asfilter agentsthat
distill query-relevant evidence from retrieved docu-
ments, each emphasizing a distinct perspective on
evidence processing.
Summarizer AgentPrior work has shown that
compressing retrieved documents can significantly
improve RAG performance (Xu et al., 2023;
Hwang et al., 2024). Building on this insight, the
Summarizer agent condenses the retrieved docu-
ments into a concise representation that preserves
semantic consistency while highlighting salient,
query-relevant information. Formally, the summa-
rization process is defined as:
R(s)
i=A sum(qi, D),(1)
where R(s)
idenotes the summary-based filtered re-
sponse.
Extractor AgentThe Extractor agent identifies
fine-grained factual spans or sentences that explic-
itly support the answer to qiwithout paraphrasing.
Unlike the Summarizer, this agent strictly operates
in an extractive manner, copying evidence verba-
tim from the retrieved documents, motivated byprevious work on extractive QA(Chen et al., 2017;
Lewis et al., 2021). It is particularly suited for ques-
tions whose answers can be directly grounded in
explicit textual evidence. This process is expressed
as:
R(e)
i=A ext(qi, D),(2)
where R(e)
iconsists of explicit evidence fragments
aligned with the query semantics.
Reasoner AgentThe Reasoner agent performs
reasoning over the retrieved content to infer im-
plicit connections or cross-document evidence that
may not be explicitly stated in any single sentence,
as motivated by prior work on reasoning and rel-
evance judgment in retrieval-augmented genera-
tion (Asai et al., 2023; Chang et al., 2024). The
Reasoner Agent leverages the backbone language
model to analyze the retrieved documents and ex-
plicitly articulate how the evidence supports the
query, as well as the inference steps required to
arrive at an answer. Formally, it is defined as:
R(r)
i=A rea(qi, D),(3)
where R(r)
irepresents a reasoning-oriented filtered
response that connects and interprets evidence
across retrieved documents.
Notably, MASS-RAG instantiates multiple
agents with explicitly differentiated roles and out-
3

put constraints. Each agent imposes distinct output
constraints that explicitly structure how retrieved
documents are processed prior to answer gener-
ation. Unlike prior approaches, the Reasoner in
MASS-RAG produces an explicit intermediate ev-
idence representation that is decoupled from the
final answer.
Answer Agent.Given each filtered response, the
Answer Agent independently generates a candi-
date answer. Specifically, for the outputs from the
Summarizer, Extractor, and Reasoner, the Answer
Agent computes:
A(j)
i=A ans(qi, R(j)
i), j∈ {s, e, r},(4)
where A(s)
i,A(e)
i, and A(r)
idenote candidate an-
swers derived from summary-, extraction-, and
reasoning-based evidence, respectively.
Generating candidate answers independently en-
courages the model to commit to a concrete an-
swer under each evidence view. This design is
most effective for factoid-style question answering,
where candidate answers carry rich semantic con-
tent and different evidence views may yield comple-
mentary or competing hypotheses. In contrast, for
multiple-choice tasks with symbolic option labels
(e.g., A/B/C/D), intermediate candidate answers
provide limited additional signal. For long-form
QA, the candidate-answer step can still be benefi-
cial, though its effectiveness may be constrained by
a tendency to focus on a single dominant response.
Accordingly, the Answer Agent can be optionally
enabled depending on the task characteristics.
Synthesis Agent.The candidate answers
{A(s)
i, A(e)
i, A(r)
i}may overlap while differing
in factual details or reasoning paths. When the
Answer Agent is enabled, aSynthesis Agent
integrates these intermediate answer hypotheses
and produces the final output as:
ˆYi=A syn 
qi,{A(s)
i, A(e)
i, A(r)
i}
.(5)
When the Answer Agent is disabled, the Synthe-
sis Agent directly operates on the filtered evidence
responses generated by the filter agents:
ˆYi=A syn 
qi,{R(s)
i, R(e)
i, R(r)
i}
.(6)
The Synthesis Agent is designed to explicitly
compare and reconcile structured evidence or an-
swer representations produced by heterogeneousagents, rather than to simply increase model ca-
pacity through additional generation steps. Across
both settings, this structured synthesis produces a
coherent final answer with improved factual accu-
racy, completeness, and robustness.
We provide detailed role specifications for all
agents in Appendix C.
4 Experiment
We conduct comprehensive experiments and abla-
tion studies to evaluate the performance ofMASS-
RAGfrom three perspectives:
•RQ1:How does MASS-RAG perform com-
pared to training-based RAG methods and
existing training-free RAG approaches, in-
cluding prior multi-agent systems with single-
agent filtering?
•RQ2:How do individual filter agents capture
factually relevant evidence, and to what extent
are their contributions complementary?
•RQ3:How does aggregating evidence from
multiple filter agents affect the end-to-end per-
formance of the RAG system?
In addition to these research questions, we in-
clude targeted ablation studies to examine the ro-
bustness of MASS-RAG to retrieval depth and to
analyze the role of the optional Answer Agent.
4.1 Datasets
MASS-RAGis evaluated using different models
and downstream tasks. We conduct our evaluations
under zero-shot setting, where each LLM agent is
provided with specific task instructions in the in-
context approach (Ram et al., 2023; Chang et al.,
2024).
Open-Domain Question Answering Tasks
(ODQA).Following previous state-of-the-art
(SOTA) work (Chang et al., 2024; Asai et al.,
2023), we evaluate MASS-RAG on two ODQA
datasets: TriviaQA-unfiltered (Joshi et al., 2017)
and PopQA (Mallen et al., 2023). As the testing
sets of TriviaQA-unfiltered (open) are not publicly
available, to make a fair performance comparison,
we use the TriviaQA-unfiltered validation and
testing sets provided by existing work (Asai et al.,
2023). For PopQA, we use the long-tail subset,
consisting of 1,399 rare entity queries whose
monthly Wikipedia page views are less than 100.
4

TriviaQA PopQA ARC-C ASQA
(acc) (acc) (acc) (em) (rg) (mau)
Baselines without retrieval
Llama2∗
7B 30.5 14.7 21.8 7.9 15.3 19.0
Llama2∗
13B 38.5 14.7 29.4 7.2 12.4 16.0
Mistral∗
7B 54.8 26.2 55.5 11.2 18.1 27.6
Llama3∗
8B 68.4 29.2 58.8 19.4 30.3 54.3
Baselines with retrieval (training-based)
Llama2-FT∗
7B 57.3 48.7 65.8 31.0 35.8 51.2
Self-RAG∗
7B 66.4 54.9 67.3 30.0 35.774.3
Baselines with retrieval (training-free)
Llama2∗
7B 68.9 50.9 51.0 16.2 23.4 33.1
Llama2∗
13B 47.0 45.7 26.0 16.3 20.5 24.7
Mistral∗
7B 69.4 55.5 57.1 32.4 34.8 54.3
Llama3∗
8B 73.1 61.8 55.6 37.1 36.5 63.0
Qwen3 8B 72 62.1 86.4 46.1 35.7 16.1
MAIN-RAG-Mistral∗
7B 71.0 58.9 58.9 35.7 36.2 60.0
MAIN-RAG-Llama3∗
8B 74.1 64.0 61.9 39.242.070.6
MASS-RAG-Llama2 7B 68.6 57.8 72.2 36.2 33.3 20.94
MASS-RAG-Mistral 7B 73.7 60.9 72.4 41.6 37.1 69.8
MASS-RAG-Llama3 8B 76.7 64.278.7 47 35.2 55.2
MASS-RAG-Qwen3 8B 74.1 63.987.3 47.5137.4 53.5
Table 1: Overall experimental results on four tasks.Bold numbersrefer to the best performance among baselines
without retrieval and training-free baselines, and underline numbers refer to the second-best performance.∗indicates
concurrent results conducted by recent works or original papers. For the metrics,acc,em,rg, andmaudenote
accuracy,str-em,rouge, andMAUVE, respectively.
Long-form Ambiguous Question Answering.
We use a long-form QA dataset with 948 samples,
ALCE-ASQA (Stelmakh et al., 2023; Gao et al.,
2023), to evaluateMASS-RAGperformance in
ambiguous questions. Each ambiguous sample in
the ASQA dataset may contain multiple pieces of
evidence and multiple valid answers, making the
task particularly challenging. Following prior work
(Asai et al., 2023; Mallen et al., 2023; Chang et al.,
2024), we evaluate ASQA using the official met-
rics, including string exact match (str-em), output
diversity measured by MAUVE, and ROUGE.
Closed-set TaskWe test the performance of our
proposed method on the ARC-Challenge dataset
(Clark et al., 2018) to assess whether multi-agent
context filtering benefits RAG performance on a
closed-set reasoning task. We use accuracy as an
evaluation metric and report on the test set with
1,172 samples.4.2 Baselines
Following the previous methods (Asai et al., 2023;
Chang et al., 2024), we evaluated the performance
of theMASS-RAGand baselines on four bench-
marks. The main evaluation metric is accuracy
which has been widely used (Asai et al., 2023;
Chang et al., 2024).
Baselines without retrievalsWe compared our
method with several publicly available pretrained
LLMs, including Llama2 7B(Touvron et al., 2023),
Llama3 8B(Grattafiori et al., 2024), and Mistral 7B
(Jiang et al., 2023).
Baselines with retrievalsWe compared our
method with both training-based method and
training-free method, which incorporate retrieval
either during inference or through the training pro-
cess. For training-based baselines, we consider
Self-RAG (Asai et al., 2023) and the Llama2-FT 7B
which is the Llama2 7Bfine-tuned on the same
5

dataset used by Self-RAG but without the reflec-
tion tokens or retrieved passages. For training-free
baselines, we used the models we included in the
baselines without retrievals and we also introduce
the latest model Qwen3 8Bas reference (Yang et al.,
2025). Eventually, we compared our method with
the previous SOTA (state-of-the-art) multi-agent
RAG method, the MAIN-RAG proposed by Chang
et al. (2024). We also compared our method with
standard RAG baselines which do not require ad-
ditional training and generally prepend the top re-
trieved documents to the query as the pre-trained
LLM input, using the same retriever as all other
baselines we include.
4.3 Main Results (RQ1)
Table 1 reports the accuracy results of all meth-
ods. MASS-RAG consistently outperforms both
training-based and training-free baselines across all
evaluated tasks. To ensure a fair comparison with
Self-RAG, which is trained on Llama2 7B, we eval-
uate MASS-RAG using the same backbone. Under
this setting, MASS-RAG achieves absolute accu-
racy improvements of up to 3.3%, 5.3%, 7.3%, and
20.7% on the four selected tasks, respectively.
We further compare MASS-RAG with
MAIN-RAG using Llama3 8Bas the backbone.
As shown in Table 1, MASS-RAG yields im-
provements of up to 3.5%, 0.3%, 27.1%, and
19.9% across the same tasks, demonstrating the
effectiveness of the proposed multi-agent synthesis
strategy. We visualize these comparisons in
Appendix B.
4.4 Experimental Settings
MASS-RAGis compatible with different pre-
trained LLMs as backbone models and operates
in a fully training-free manner. We use the re-
trieved documents released by Asai et al. (2023)
for all experiments. For each query, up to 20 docu-
ments are retrieved and ranked using the pre-trained
Contriever trained on MS MARCO (Izacard et al.,
2022a). This setting is also adopted by Self-RAG
andMAIN-RAG , enabling a fair and controlled com-
parison. All models are decoded using greedy gen-
eration, with temperature set to 0 and top-p set to
1.0, ensuring deterministic and reproducible out-
puts. The number of top retrieved documents is
a hyperparameter in our experiments. We use the
top-5 retrieved documents for Llama2 7Band the
top-10 retrieved documents for other models, fol-
lowing the settings adopted by the correspondingMethod TQA PopQA ARC-
CASQA
(acc) (acc) (acc) (em)
Retrieved Docs = 5
Self-RAG66.4 54.9 67.3 30.0
MASS-RAG (Llama2 7B)68.6 57.8 72.2 36.2
MASS-RAG (Llama3 8B)75.8 61.5 79.9 45.2
MASS-RAG (Mistral 7B) 72.7 55.7 72.3 39.5
Retrieved Docs = 10
MAIN-RAG (Llama3 8B) 74.1 64.0 61.9 39.2
MASS-RAG (Llama3 8B)76.7 64.2 78.7 47.0
MASS-RAG (Mistral 7B) 73.7 60.9 72.4 41.6
Table 2: Performance of MASS-RAG on four down-
stream tasks with different numbers of retrieved docu-
ments.
baselines.
For TriviaQA, PopQA, and ASQA, we instanti-
ate all five agents following the standardMASS-
RAGpipeline shown in Figure 2 to ensure a
consistent experimental setup across QA bench-
marks. For ARC-Challenge, we instantiate only
four agents and exclude the optional Answer Agent.
This setting adapts to the multiple-choice nature
of ARC-Challenge, where answers are restricted
to symbolic option labels (e.g., A, B, C, D) and
thus lack semantic granularity. The structure of
ARC-Challenge does not naturally support interme-
diate candidate-answer generation. Accordingly,
for ARC-Challenge, the final prediction is directly
synthesized by the Synthesis Agent from the fil-
tered responses of the three filter agents.
4.5 Ablation Study and Analysis
Ablation Study on # of Retrieved Documents
Due to limitations in retrieval ranking (Asai et al.,
2024), factually relevant evidence may not always
appear among the top-ranked documents, making
RAG performance sensitive to the number of re-
trieved documents. To assess the robustness to
retrieval depth, we evaluateMASS-RAGunder
varying numbers of retrieved documents, as shown
in Table 2.
Across the evaluated tasks,MASS-RAGex-
hibits stable performance trends as the retrieval
depth varies, demonstrating robustness to changes
in the number of retrieved documents. No-
tably, when fewer documents are selected as
retrieval results,MASS-RAGinstantiated with
Llama3 8Bconsistently outperforms MAIN-RAG on
TriviaQA, ARC-Challenge, and ASQA, indicat-
6

Dataset Model w/ Answer w/o Answer
TriviaQAMistral77.576.0
Llama379.979.3
PopQAMistral 60.961.9
Llama364.262.6
ASQAMistral 41.643.9
Llama3 47.048.0
Table 3: Effect of the Answer Agent across different
backbone models. We report performance with and
without the Answer Agent on short-form QA(TriviaQA,
PopQA) and long-form QA (ASQA).
ing thatMASS-RAGremains effective even with
limited retrieved context, where robust evidence
filtering and synthesis play a more critical role.
Ablation Study on the effect of Answer Agent
We analyze the role of the Answer Agent by ablat-
ing this component across different backbone mod-
els and datasets. In MASS-RAG, three filter agents
(Summarizer, Extractor, and Reasoner) are first ap-
plied to distill complementary evidence views, and
the Answer Agent then independently generates
one candidate answer for each filtered response,
resulting in three candidate answers corresponding
to the three filter agents.
As shown in Table 3, incorporating the An-
swer Agent consistently improves performance on
factoid QA benchmarks, including TriviaQA and
PopQA, across both Mistral and Llama3 backbones.
This suggests that independently generating answer
hypotheses from multiple evidence views helps re-
duce ambiguity and enables effective comparison
during the synthesis stage.
In contrast, the Answer Agent yields marginal
or no improvement on ASQA. Unlike factoid QA,
ASQA requires aggregating a diverse set of factual
statements into a comprehensive long-form answer.
In this setting, early commitment to individual can-
didate answers can reduce the flexibility of inte-
grating information distributed across multiple evi-
dence views. Consistent with this observation, we
disable the Answer Agent in our main experiments
on ARC-Challenge, where answers are constrained
to symbolic options and do not benefit from in-
termediate answer hypothesis generation. Overall,
these results highlight the task-dependent utility of
the Answer Agent and demonstrate the flexibility
of MASS-RAG, where this component can be op-
tionally enabled depending on task characteristics.Construction of the Uniquely Attributable Sub-
set (RQ2)To better quantify and analyze the con-
tributions of individual filter agent responses, we
process the TriviaQA and PopQA datasets using
the outputs of MASS-RAG. Specifically, we iden-
tify a subset of questions whose ground-truth evi-
dence is captured by exactly one of the three filter
agents, meaning that each such question is cap-
tured by a single type of filter agent. This allows us
to quantify how many questions can be answered
exclusively based on a particular filter agent re-
sponse. Through this analysis, we demonstrate
that different filter agents provide complementary
factual evidence that is uniquely captured by each
agent, and that all three agents are indispensable,
as each contributes distinct and non-overlapping
factual evidence to the overall MASS-RAG system.
For PopQA, following the setup in Section 4.1,
we start from 1,399 questions and construct a
Uniquely Attributable Subsetby retaining only
questions whose ground-truth evidence is cap-
tured by exactly one filter agent, while exclud-
ing questions captured by multiple agents or by
none. This subset enables focused analysis of ques-
tions uniquely handled by each agent. The same
procedure is applied to TriviaQA for consistency.
This analysis reveals the distinct contribution of
each filter agent and highlights the complementar-
ity among their responses. Accordingly, the size
of theUniquely Attributable Subsetvaries with
the backbone model, resulting in 609 (TriviaQA)
and 108 (PopQA) samples for Llama3 8B, and 559
(TriviaQA) and 85 (PopQA) samples for Mistral 7B.
Effectiveness of Individual Filter Agents (RQ2)
To evaluate the effectiveness of each type of fil-
ter agent response ( R(s)
i,R(e)
i,R(r)
i) introduced in
Section 3.1, we defineEvidence Coverage Rate
(ECR), inspired by the Answer Inclusion Rate
(AIR) proposed by Ruan et al. (2024), as the pro-
portion of questions for which a given filter agent
response contains ground-truth evidence. It is de-
fined as:
ECR(A F) =1
|Q|X
q∈Q1[AF(q)relevant](7)
In equation 7, AFdenotes a single filter agent or
a set of filter agents whose responses are jointly
considered, and 1[·]is an indicator function that
equals 1 if AF(q)contains ground-truth evidence
for questionq, and 0 otherwise.
7

Figure 3: Evidence Coverage Rate (ECR) of each filter
response on theUniquely Attributable Subsetof Triv-
iaQA and PopQA using Llama3 8Band Mistral 7Bwith
the top-10 retrieved documents.
As shown in Figure 3, different filter agents cap-
ture distinct and complementary subsets of factu-
ally relevant evidence. Different agents tend to
excel on different subsets of questions, supporting
the need for multi-agent filtering, since a single
agent alone is insufficient to reliably cover all rele-
vant evidence. Additional case studies are provided
to qualitatively demonstrate the complementarity
among filter agents (see Appendix D).
Impact of Multi-Agent Evidence Aggregation
(RQ3)Building on the evidence-level robustness
observed in RQ2, we investigate whether aggre-
gating evidence from multiple agents enhances the
end-to-end performance of the RAG system under
noisy retrieval.
As described in Section 3.1, MASS-RAG em-
ploys a synthesis agent Asynto produce the final
answer by aggregating candidate answers gener-
ated from the responses of the filter agents ( R(s)
i,
R(e)
i,R(r)
i). We evaluate this process using answer
accuracy on TriviaQA and PopQA. As shown in
Tables 4 and 5, the synthesis agent does not always
attain the highest evidence coverage, reflecting its
distinct role. The summarization agent achieves
high coverage, fulfilling its design objective; how-
ever, its output serves as an intermediate repre-
sentation that may include multiple pieces of rele-
vant evidence without resolving their correctness
or mutual consistency. Notably, the intermediate re-
sponses produced by the filter agents often achieveAgent TriviaQA PopQA
(acc) (acc)
Summarizer 73.7 65.7
Extractor 68.1 61.5
Reasoner 72.8 63.0
Synthesis 76.7 64.2
Table 4: Accuracy (acc) of each filter response on Triv-
iaQA and PopQA using Llama3 8Bwith the top-10 re-
trieved documents.
Agent TriviaQA PopQA
(acc) (acc)
Summarizer 73.8 61.2
Extractor 66.9 57.8
Reasoner 71.9 62.4
Synthesis 73.6 60.9
Table 5: Accuracy (acc) of each filter response on Triv-
iaQA and PopQA using Mistral 7Bwith the top-10 re-
trieved documents.
higher accuracy, indicating that these agents are
effective at filtering retrieved documents and ex-
posing answerable content supported by the evi-
dence. In this sense, the filtered responses can be
viewed as an approximate upper bound on what the
model could potentially answer given the retrieved
context. The role of the Synthesis Agent, however,
is to produce a single unified answer rather than
retain multiple partially correct or ambiguous re-
sponses. Whether this upper bound can be reached
in the final prediction therefore depends on the base
model’s ability to consolidate and resolve the fil-
tered evidence. This observation is consistent with
our ablation results, where stronger backbone mod-
els are better able to approach this upper bound
than weaker ones (e.g. Llama3 vs Mistral).
5 Conclusion
We propose MASS-RAG, a multi-agent synthe-
sis framework for retrieval-augmented generation.
Experimental results indicate that role-specialized
multi-agent filtering captures complementary and
factually relevant evidence from retrieved contexts,
and that synthesizing these filtered representations
leads to more reliable final predictions. Overall, our
results suggest that explicitly structuring evidence
processing into multi-agent filtering followed by
synthesis is a practical design choice for improving
8

the RAG systems.
Limitations
Our study is designed to isolate and analyze the
effects of multi-agent evidence filtering and synthe-
sis within a retrieval-augmented generation frame-
work. To enable controlled comparisons, we adopt
fixed retrieval pipelines and pretrained backbone
language models throughout our experiments. Ac-
cordingly, components such as retriever design,
reranking strategies, and decoding-level variations
are treated as fixed rather than independent vari-
ables in this work.
MASS-RAG is further evaluated under standard
inference-time configurations without task-specific
parameter adaptation. This design choice allows
us to focus on the interaction between agent-level
evidence processing and synthesis, while leaving
open how learning-based or adaptive mechanisms
could complement the proposed framework in fu-
ture studies.
The modular design of MASS-RAG also makes
it amenable to future extensions along orthogo-
nal dimensions, such as retrieval or learning-based
adaptation, without altering its core synthesis mech-
anism.
Ethical Statement
This work studies a training-free, multi-agent
framework for retrieval-augmented generation, fo-
cusing on how retrieved evidence is filtered and
synthesized by large language models. The pro-
posed method does not introduce new data sources,
supervision signals, or learning objectives beyond
those used in existing RAG systems, and therefore
inherits the ethical considerations associated with
the underlying pretrained models and retrieval cor-
pora.
As with other retrieval-based generation ap-
proaches, MASS-RAG may surface inaccuracies
or biases present in retrieved documents or in the
backbone language models. While the proposed
multi-agent filtering and synthesis mechanism aims
to improve factual grounding and robustness under
noisy retrieval, it does not guarantee the correctness
or neutrality of generated outputs. Accordingly, the
system should not be used as a sole source of truth
in high-stakes or safety-critical applications with-
out appropriate human oversight.
We conduct all experiments using publicly avail-
able benchmarks and released retrieval results, anddo not involve personal data, user profiling, or
deployment-facing evaluation. Our goal is to pro-
vide a controlled analysis of multi-agent evidence
processing rather than to advocate immediate real-
world deployment.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
Preprint, arXiv:2310.11511.
Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei
Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and
Wen tau Yih. 2024. Reliable, adaptable, and at-
tributable language models with retrieval.Preprint,
arXiv:2403.03187.
Angels Balaguer, Vinamra Benara, Renato Luiz de Fre-
itas Cunha, Roberto de M. Estevão Filho, Todd
Hendry, Daniel Holstein, Jennifer Marsman, Nick
Mecklenburg, Sara Malvar, Leonardo O. Nunes,
Rafael Padilha, Morris Sharp, Bruno Silva, Swati
Sharma, Vijay Aski, and Ranveer Chandra. 2024.
Rag vs fine-tuning: Pipelines, tradeoffs, and a case
study on agriculture.Preprint, arXiv:2401.08406.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aure-
lia Guy, Jacob Menick, Roman Ring, Tom Henni-
gan, Saffron Huang, Loren Maggiore, Chris Jones,
Albin Cassirer, and 9 others. 2022. Improving lan-
guage models by retrieving from trillions of tokens.
Preprint, arXiv:2112.04426.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, and 12 others. 2020. Lan-
guage models are few-shot learners.Preprint,
arXiv:2005.14165.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval augmented genera-
tion.Preprint, arXiv:2404.00610.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahash-
weta Das, and Na Zou. 2024. Main-rag: Multi-agent
filtering retrieval-augmented generation.Preprint,
arXiv:2501.00332.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading wikipedia to answer open-
domain questions.Preprint, arXiv:1704.00051.
9

Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma,
Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming
Yang, and Jiaxin Mao. 2025. Improving retrieval-
augmented generation through multi-agent reinforce-
ment learning.Preprint, arXiv:2501.15228.
Sukmin Cho, Jeongyeon Seo, Soyeong Jeong, and
Jong C. Park. 2023. Improving zero-shot reader
by reducing distractions from irrelevant documents
in open-domain question answering.Preprint,
arXiv:2310.17490.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind
Tafjord. 2018. Think you have solved question
answering? try arc, the ai2 reasoning challenge.
Preprint, arXiv:1803.05457.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations.Preprint, arXiv:2305.14627.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2024. Retrieval-
augmented generation for large language models: A
survey.Preprint, arXiv:2312.10997.
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub
Chowdhury, Ankita Rajaram Naik, Pengshan Cai,
and Alfio Gliozzo. 2022. Re2g: Retrieve, rerank,
generate.Preprint, arXiv:2207.06300.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.Preprint, arXiv:2407.21783.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.Preprint, arXiv:2311.05232.
Taeho Hwang, Soyeong Jeong, Sukmin Cho, SeungY-
oon Han, and Jong C. Park. 2024. Dslr: Document
refinement with sentence-level re-ranking and recon-
struction to enhance retrieval-augmented generation.
Preprint, arXiv:2407.03627.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022a. Unsupervised dense infor-
mation retrieval with contrastive learning.Preprint,
arXiv:2112.09118.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022b. Atlas: Few-shot learning
with retrieval augmented language models.Preprint,
arXiv:2208.03299.Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.Preprint,
arXiv:2310.06825.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion.Preprint, arXiv:1705.03551.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Preprint, arXiv:2005.11401.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Wei-
jia Shi, Maria Lomeli, Rich James, Pedro Ro-
driguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis,
Luke Zettlemoyer, and Scott Yih. 2023. Ra-
dit: Retrieval-augmented dual instruction tuning.
Preprint, arXiv:2310.01352.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2023. Lost in the middle: How language mod-
els use long contexts.Preprint, arXiv:2307.03172.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting for retrieval-
augmented large language models.Preprint,
arXiv:2305.14283.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories.Preprint, arXiv:2212.10511.
Kevin Meng, David Bau, Alex Andonian, and Yonatan
Belinkov. 2023. Locating and editing factual associa-
tions in gpt.Preprint, arXiv:2202.05262.
Thang Nguyen, Peter Chin, and Yu-Wing Tai. 2025. Ma-
rag: Multi-agent retrieval-augmented generation via
collaborative chain-of-thought reasoning.Preprint,
arXiv:2505.20096.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report.Preprint,
arXiv:2303.08774.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Preprint, arXiv:2302.00083.
10

Qian Ruan, Ilia Kuznetsov, and Iryna Gurevych. 2024.
Are large language models good classifiers? a study
on edit intent classification in scientific document
revisions.Preprint, arXiv:2410.02028.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen tau Yih. 2023. Replug: Retrieval-
augmented black-box language models.Preprint,
arXiv:2301.12652.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2023. Asqa: Factoid questions meet
long-form answers.Preprint, arXiv:2204.06092.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton
Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,
Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 oth-
ers. 2023. Llama 2: Open foundation and fine-tuned
chat models.Preprint, arXiv:2307.09288.
Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu,
Bo Li, Mohammad Shoeybi, and Bryan Catanzaro.
2023a. Instructretro: Instruction tuning post retrieval-
augmented pretraining.Preprint, arXiv:2310.07713.
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. 2023b. Learning
to filter context for retrieval-augmented generation.
Preprint, arXiv:2311.08377.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel,
Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler, Ed H.
Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy
Liang, Jeff Dean, and William Fedus. 2022. Emer-
gent abilities of large language models.Preprint,
arXiv:2206.07682.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023.
Recomp: Improving retrieval-augmented lms with
compression and selective augmentation.Preprint,
arXiv:2310.04408.
Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong,
Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu, and
Ge Yu. 2024a. Activerag: Revealing the trea-
sures of knowledge via active learning.Preprint,
arXiv:2402.13547.
Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. 2024b.
Hallucination is inevitable: An innate limitation of
large language models.Preprint, arXiv:2401.11817.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.Haoyan Yang, Zhitao Li, Yong Zhang, Jianzong Wang,
Ning Cheng, Ming Li, and Jing Xiao. 2023. Prca:
Fitting black-box large language models for retrieval
question answering via pluggable reward-driven con-
textual adapter.Preprint, arXiv:2310.18347.
11

Appendix
A Additional Experimental Setup
Table 6 conclude the computational setup used for
all experiments. Models with 7B or 8B parameters
can be executed on a single 24GB GPU, while mod-
els larger than 13B parameters typically require at
least two 24GB GPUs. To ensure a fair compari-
son with prior work and to facilitate economical
reproducibility, all experiments are conducted us-
ing bfloat16 precision, consistent with the settings
adopted bySelf-RAG(Asai et al., 2023). The total
runtime of each experiment may vary depending
on system-level factors such as GPU power limits,
runtime load, and driver versions.
Device Attribute Specification
Computing Infrastructure GPU
GPU Model NVIDIA RTX 4090
Number of GPUs 4
GPU Memory 24 GB
Table 6: Computing Device for the experiments.
B MASS-RAG Performance Comparison
withSELF-RAGandMAIN-RAG
Figure 4 presents a visualized comparison between
MASS-RAGand two representative state-of-the-
art baselines, Self-RAG andMAIN-RAG , across four
benchmark datasets. In the left panel, MASS-
RAG is compared with Self-RAG under a shared
Llama2 7Bbackbone, while the right panel reports
comparisons with MAIN-RAG using Llama3 8Bto en-
sure backbone consistency. Across all benchmarks,
MASS-RAG consistently outperforms both base-
lines, with particularly notable gains on reasoning-
intensive datasets such as ARC-C and ASQA.
The annotated improvements highlight that MASS-
RAG yields stable performance gains under con-
trolled backbone settings, suggesting that the ob-
served improvements are attributable to the pro-
posed multi-agent synthesis framework rather than
differences in model variation.C Agent Role Specifications
Each agent in MASS-RAG is instantiated with a
lightweight, role-specific instruction that constrains
its output objective and format (e.g., extractive, ab-
stractive, or reasoning-oriented), thereby inducing
distinct intermediate representations from the same
retrieved context.
Summarizer AgentThe Summarizer is respon-
sible for condensing the retrieved documents into a
concise, abstractive summary that highlights query-
relevant information. Its output preserves the se-
mantic content of the retrieved context while re-
moving irrelevant or redundant details, and does
not attempt to directly answer the query. Even
when the retrieved documents do not contain a
complete answer, the Summarizer still produces
a partial summary of any potentially relevant infor-
mation.
Extractor AgentThe Extractor identifies and se-
lects fine-grained textual spans from the retrieved
documents that explicitly support answering the
query. It operates in a strictly extractive manner,
copying evidence verbatim from the context with-
out paraphrasing or reinterpretation. When multi-
ple relevant spans exist, the Extractor may return
several evidence fragments, reflecting all directly
grounded information available in the retrieved doc-
uments.
Reasoner AgentThe Reasoner examines the re-
trieved documents to identify implicit or cross-
document connections that are not explicitly stated
in any single sentence. Rather than producing a fi-
nal answer, it articulates how different pieces of re-
trieved evidence could be combined or interpreted
to support answering the query, including the as-
sumptions or inference steps involved. This agent
focuses on reasoning over the retrieved context
without introducing external knowledge.
Answer Agent (optional)The Answer Agent
generates a concise candidate answer based on a
single filtered evidence representation produced by
one filter agent. Its role is to translate the provided
evidence into a concrete answer hypothesis, with-
out introducing information beyond the given evi-
dence. When the evidence is partial or incomplete,
the Answer Agent produces a best-effort answer
that reflects the available information rather than
refusing to respond. This agent does not perform
cross-evidence aggregation and is applied indepen-
12

Figure 4: Performance comparison of MASS-RAG with state-of-the-art baselines across four benchmark datasets.
The left panel reports results compared withSelf-RAGusing Llama2 7B, while the right panel shows comparisons
withMAIN-RAGbased on Llama3 8B. In both cases, the backbone models are kept consistent to ensure a fair
comparison.
dently to each filter agent output, serving as an
intermediate hypothesis generator rather than the
final decision module.
Synthesis AgentThe Synthesis Agent produces
the final output by consolidating structured inputs
from multiple agents. Depending on the task set-
ting, these inputs may consist of filtered evidence
representations or intermediate candidate answers.
The Synthesis Agent performs comparative rea-
soning across its inputs, integrates complementary
information, resolves inconsistencies, and gener-
ates a coherent final answer without referring to
individual agents explicitly.
These role specifications are intentionally
lightweight and transferable across backbone lan-
guage models.
D Case Study Of MASS-RAG
We provide additional case studies comprising six
examples from Figure 5 to 10, including three cases
evaluated with Mistral 7Bon PopQA and three cases
evaluated with Llama3 8Bon TriviaQA. Each case
illustrates a distinct scenario in which only one or
two filter agents are able to capture the relevant fac-
tual or implicit evidence. Implicit evidence refers
to evidence that does not explicitly contain the gold
answer, but nonetheless allows the model to infer
the correct answer without requiring additional in-
formation. All cases are analyzed used on top-10
retrieved documents setting and the gold answers
for each sample are underlined.ERuntime Cost Analysis Of MASS-RAG
In order to analyze the total runtime cost of MASS-
RAG, we assume that each agent call costs 100
tokens (Actual settings may vary depends on dif-
ferent task). We use xas the default run time for
one agent call. In table 7, we compare each sample
run under different settings.
Setting Run Cost
MASS-RAG w/ Answer Agent 7x
MASS-RAG w/o Answer Agent 4x
Table 7: Runtime Cost Analysis for different settings.
13

 
Case Study 1  Question: Who was the composer of Overture in G major? 
[Summarizer Agent] ( Evidence Include ):  
Anton Bruckner composed the Overture in G minor, not G major. The Overture in G major was composed by Luigi 
Cherubini.  
[Extractor Agent] ( No Evidence ):  
Anton Bruckner, specifically the Overture in G minor, WAB 98. However, the question asked for Overture in G major, 
and there is no evidence in the provided context that Bruckner composed an Overture in G major.  
[Reasoner Agent] ( No Evidence ):  
The Context provides information about several composers and their works, but none of them are identified as the 
composer of the Overture in G major. However, it does mention Anton Bruckner and his Overture in G minor, WAB 98, 
which was composed during the same time period as the Overture in G major. Inference steps required: 
1. Recognize that the Context provides information about various composers and their works, but not the composer of the 
Overture in G major. 2. Understand that the Overture in G minor by Bruckner was composed during the same time period 
as the Overture in G major.  
[Answer Agent] ( Candidate Answers): 
1: Luigi Cherubini was the composer of the Overture in G major. (Correct Answer)  
2: Based on the provided evidence, there is no information that Anton Bruckner composed an Overture in G major. 
Therefore, I cannot confirm the composer of this specific work. (NOT Answer)  
3: The Context does not provide the identity of the composer of the Overture in G major. However, it's worth noting that 
Anton Bruckner composed the Overture in G minor during the same time period. This information might suggest a 
possible connection or similarity in style between the two composers, but it does not definitively answer who the 
composer of the Overture in G major is. (NOT Answer)  
[Synthesis Agent]  (Final Answer):   
Luigi Cherubini was the composer of the Overture in G major. (Correct Answer)  
 
 
 
 
 
 
 
 
 
 
 Figure 5: MASS-RAG Case Study 1 - Dataset: PopQA, Model: Mistral 7B
14

Case Study 2   Question: In what country is Ciepień? 
[Summarizer Agent] (No Evidence):   
Ciepień is twinned with a location, but the context does not specify the country where Ciepień is located.  
[Extractor Agent] (Evidence Include): 
Ciepień is twinned with Ciechanów, which is located in Poland. However, the context does not provide specific 
information about where Ciepień is located within Poland. 
[Reasoner Agent] (No Evidence):  
The Context provides information about various locations, but it does not directly mention "Ciepień" in relation to a 
country. However, the context does mention "Ciechanów" and its twin towns, which could potentially provide a clue. 
Inference steps required: 
1. Identify the locations that Ciechanów is twinned with (Ciepień and Gmina Zbójno). 
2. Investigate whether any of these locations are countries. 
3. If Ciepień is not mentioned as a location, check if it could be a part of one of the twin towns (Gmina Zbójno). 
[Answer Agent] (Candidate Answers): 
1: The provided context does not specify the country where Ciepień is located. Therefore, it cannot be definitively stated. 
(Not Answer) 
2: Ciepień is located in Poland. (Correct Answer) 
3: Based on the provided context, it is not possible to definitively determine the country of Ciepień. Ciechanów is a town 
in Poland, and it is twinned with Ciepień and Gmina Zbójno. However, neither Ciepień nor Gmina Zbójno are 
mentioned as countries in the provided context. Therefore, further research would be needed to determine the country of 
Ciepień. (Incomplete Answer) 
[Synthesis Agent] (Final Answer):   
Ciepień is located in Poland, as suggested by Candidate Answer 2. However, Candidate Answer 3 provides additional 
context that Ciepień is twinned with Ciechanów, a town in Poland. Since Ciechanów is in Poland, it is reasonable to 
infer that Ciepień is also likely in Poland. Therefore, the final answer is: Ciepień is located in Poland. (Correct Answer) 
 
 
 
 
 
 
 
 
 
 
 
 Figure 6: MASS-RAG Case Study 2 - Dataset: PopQA, Model: Mistral 7B
15

Case Study 3   Question: What genre is Drive On?  
[Summarizer Agent] (No Evidence): 
Drive (2011 film) is a neo-noir, art house feature.  
[Extractor Agent] (No Evidence): 
Drive (2011 film): Drive has been described as a tough, hard-edged, neo-noir, art house feature, extremely violent and 
very stylish, with European art and grindhouse influences.  
[Reasoner Agent] (Evidence Include):  
The Context provides information about the Drive (2011 film) and its soundtrack, but there's no direct mention of the 
genre of the Drive (2011 film) or the Drive On (song). However, the film Drive is described as a neo-noir, art house 
feature, extremely violent and very stylish, with European art and grindhouse influences. The soundtrack, which includes 
electronic-pop songs, is also described as integral to the film's experience.  
[Answer Agent] (Candidate Answers): 
1: The genre of Drive (as a film) is neo-noir and art house. (Incorrect Answer) 
2: Drive is a neo-noir film. (Incorrect Answer)  
3: The provided context does not directly mention the genre of the song "Drive On." However, given that the film Drive 
(2011) has a soundtrack with electronic-pop songs and the film itself has neo-noir, art house, European art, and 
grindhouse influences, it's possible that the genre of "Drive On" could be related to these genres, particularly eletronic 
ipop or synth-pop, due to its inclusion in the film's soundtrack. However, without specific information about "Drive On," 
this is an inference. (Correct Answer) 
[Synthesis Agent] (Final Answer):   
Drive is a neo-noir film, so it's possible that the genre of "Drive On" (if it's a song from the film's soundtrack) could be 
related to neo-noir, electronic-pop, or synth-pop. However, without specific information about "Drive On," this is an 
inference. (Correct Answer) 
 
 
 
 
 
 
 
 
 
 
 
 
 Figure 7: MASS-RAG Case Study 3 - Dataset: PopQA, Model: Mistral 7B
16

Case Study 4  Question: What relation is Joey Richardson to Vanessa Redgrave? 
[Summarizer Agent] (Evidence Include):  
Joely Richardson is Vanessa Redgrave's daughter.  
[Extractor Agent] (Implicit Evidence):  
Vanessa Redgrave is the mother of Joely Richardson.  
[Reasoner Agent] (No Evidence):  
To determine the relation between Joey Richardson and Vanessa Redgrave, we need to examine the context and look for 
any information that might connect the two individuals. The context provides information about the Redgrave family, 
including their relationships and descendants. We can see that Vanessa Redgrave is mentioned as the sister of Lynn 
Redgrave and the mother of Natasha Richardson and Joely Richardson. However, there is no mention of a person named 
Joey Richardson in the context. Therefore, we can conclude that there is no information in the context that directly 
connects Joey Richardson to Vanessa Redgrave. However, we can make an inference that Joey Richardson is likely a 
relative of Vanessa Redgrave, given the context's information about the Redgrave family and their relationships. 
[Answer Agent] (Candidate Answers): 
1: Vanessa Redgrave's daughter. (Correct Answer) 
2: Joely Richardson is the daughter of Vanessa Redgrave. (Correct Answer) 
3: Joey Richardson is likely a relative of Vanessa Redgrave, possibly a child or descendant of Natasha Richardson or 
Joely Richardson. However, the exact relation is not specified in the context.. (NOT Answer) 
[Synthesis Agent] (Final Answer):   
Vanessa Redgrave's daughter. (Joely Richardson) is the correct answer. Vanessa Redgrave's daughter is Joely Richardson. 
(Correct Answer) 
 
 
 
 
 
 
 
 
 
 
 
 Figure 8: MASS-RAG Case Study 4 - Dataset: TriviaQA, Model: Llama3 8B
17

Case Study 5   Question: The US No 1 single from Diana Ross's platinum album "Diana" is "Endless Love", a 
duet with Lionel Richie. 
[Summarizer Agent] (No Evidence):   
The US No 1 single from Diana Ross's platinum album "Diana" is "Endless Love", a duet with Lionel Richie.  
[Extractor Agent] (Evidence Include): 
Evidence: "Upside Down" eventually was chosen by the label and; "Endless Love", composed for Brooke Shields' film of 
the same name, by and performed with Commodores front man, Lionel Richie. Ross and Richie performed the song on 
the telecast for the 54th Academy Awards, where the song was nominated in the category of "Best Song". The success of 
the single launched Richie's successful solo career. Ross has also ventured into acting, with a Golden Globe Award and 
Academy Award-nominated performance for her performance in the film "Lady; "I'm Still Waiting" and "Chain 
Reaction," which were both solo #1 hits for Diana Ross in the UK. International UK Diana Ross & the Supremes: 
[Reasoner Agent] (Implicit Evidence):   
To answer the question, we need to examine the context and look for information related to Diana Ross's platinum album "
Diana" and a US No 1 single. We can find this information in the section about the album "Diana". The album "Diana" 
was released in May 1980 and was a platinum album. The inference step would be to look for a US No 1 single from this 
album. The context provides sufficient evidence for this step, as it mentions that the album "Diana" was a platinum album 
and that it introduced Ross to a new generation of fans worldwide. However, it does not explicitly mention a US No 1 
single from this album. We would need to make an assumption that the album " 
[Answer Agent] (Candidate Answers): 
1: "Endless Love", a duet with Lionel Richie. (Incorrect Answer) 
2: The US No 1 single from Diana Ross's platinum album Diana was "Upside Down".   It was also a No 1 hit in the UK.  
The song was chosen by the label and was composed by and performed with Lionel Richie.   The success of the single 
launched Richie's successful solo career.   Ross has also ventured into acting, with a Golden Globe Award and Academy 
Award-nominated performance for her performance in the film "Lady Sings the Blues".  "I'm Still Waiting" and "Chain 
Reaction," which were both solo #1 hits for Diana Ross in the UK.   International UK Diana Ross & the Supremes: The 
No 1 single from Diana Ross's platinum album Diana was "Upside Down". (Correct Answer) 
3: The US No 1 single from Diana Ross's platinum album "Diana" is "Upside Down".  It was released in 1980 and 
became a huge hit, reaching number one on the US Billboard Hot 100 chart.   The song was a single from the album "
Diana", which was a platinum album and introduced Ross to a new generation of fans worldwide.  "Upside Down" is 
considered one of Ross's signature songs and a classic of the disco era.   It has been covered and sampled by many artists 
over the years and remains a beloved and iconic song in popular culture.   The song's success can be attributed to its 
catchy melody, memorable lyrics, and Ross's powerful vocals. (Correct Answer) 
[Synthesis Agent] (Final Answer):   
The US No 1 single from Diana Ross's platinum album Diana was "Upside Down".  It was released in 1980 and became 
a huge hit, reaching number one on the US Billboard Hot 100 chart.   The song was a single from the album "Diana", 
which was a platinum album and introduced Ross to a new generation of fans worldwide.  "Upside Down" is considered 
one of Ross's signature songs and a classic of the disco era. (Correct Answer) 
 
 
 
 Figure 9: MASS-RAG Case Study 5 - Dataset: TriviaQA, Model: Llama3 8B
18

 
Case Study 6   What was the first British football team to win the European Cup? 
[Summarizer Agent] (No Evidence): 
Manchester United became the first English club to win the European Cup in 1968, beating Benfica 4–1 in the final. 
[Extractor Agent] (No Evidence): 
Evidence: Manchester United became the first English club to win the European Cup in 1968, beating Benfica 4–1 in the 
final.   
[Reasoner Agent] (Evidence Include):  
To answer the question, we need to identify the first British football team to win the European Cup. The relevant 
information in the context is the sentence: "Celtic became the first British club to win the Cup in 1967, beating Italian 
side Inter who had won 2 of the previous 3 finals 2–1 in Lisbon." This sentence provides the necessary information to 
answer the question. However, it does not mention the first English team to win the European Cup. We need to look for 
another sentence that mentions the first English team to win the European Cup.  
[Answer Agent] (Candidate Answers): 
1: Manchester United became the first English club to win the European Cup in 1968, beating Benfica 4–1 in the final. 
Therefore, Manchester United was the first British football team to win the European Cup. (Incorrect Answer) 
2: Manchester United became the first English club to win the European Cup in 1968, beating Benfica 4–1 in the final. 
Therefore, Manchester United was the first British football team to win the European Cup. (Incorrect Answer)   
3: Celtic became the first British club to win the European Cup in 1967, beating Italian side Inter 2–1 in Lisbon. 
However, the first English team to win the European Cup was Manchester United in 1968, when they defeated Benfica 4–
1 in the final. (Correct Answer) 
[Synthesis Agent] (Final Answer):   
Celtic became the first British club to win the European Cup in 1967, beating Italian side Inter 2–1 in Lisbon. However, 
the first English team to win the European Cup was Manchester United in 1968, when they defeated Benfica 4–1 in the 
final.   Manchester United was the first English team to win the European Cup, but Celtic was the first British team.  
Therefore, the first British football team to win the European Cup was Celtic. (Correct Answer) 
 Figure 10: MASS-RAG Case Study 6 - Dataset: TriviaQA, Model: Llama3 8B
19
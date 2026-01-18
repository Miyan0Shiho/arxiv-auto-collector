# When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Xinkun Zhang, Haytham Fayek, Xiaodong Li

**Published**: 2026-01-14 07:22:59

**PDF URL**: [https://arxiv.org/pdf/2601.09241v1](https://arxiv.org/pdf/2601.09241v1)

## Abstract
Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy.

## Full Text


<!-- PDF content starts -->

When to Trust: A Causality-Aware Calibration Framework for
Accurate Knowledge Graph Retrieval-Augmented Generation
Jing Ren∗
RMIT University
Melbourne, Australia
jing.ren@ieee.orgBowen Li∗
RMIT University
Melbourne, Australia
s3890442@student.rmit.edu.auZiqi Xu†
RMIT University
Melbourne, Australia
ziqi.xu@rmit.edu.au
Xikun Zhang
RMIT University
Melbourne, Australia
xikun.zhang@rmit.edu.auHaytham Fayek
RMIT University
Melbourne, Australia
haytham.fayek@ieee.orgXiaodong Li
RMIT University
Melbourne, Australia
xiaodong.li@rmit.edu.au
Abstract
Knowledge Graph Retrieval-Augmented Generation (KG-RAG) ex-
tends the RAG paradigm by incorporating structured knowledge
from knowledge graphs, enabling Large Language Models (LLMs)
to perform more precise and explainable reasoning. While KG-RAG
improves factual accuracy in complex tasks, existing KG-RAG mod-
els are often severely overconfident, producing high-confidence
predictions even when retrieved sub-graphs are incomplete or un-
reliable, which raises concerns for deployment in high-stakes do-
mains. To address this issue, we propose Ca2KG, a Causality-aware
Calibration framework for KG-RAG. Ca2KG integrates counterfac-
tual prompting, which exposes retrieval-dependent uncertainties in
knowledge quality and reasoning reliability, with a panel-based re-
scoring mechanism that stabilises predictions across interventions.
Extensive experiments on two complex QA datasets demonstrate
that Ca2KG consistently improves calibration while maintaining or
even enhancing predictive accuracy. The source code can be found
at https://aisuko.github.io/ca2kg/.
CCS Concepts
•Information systems →Language models;•Computing
methodologies→Knowledge representation and reasoning;
Causal reasoning and diagnostics.
Keywords
Large Language Model, Knowledge Graph, Retrieval-Augmented
Generation, Calibration
ACM Reference Format:
Jing Ren, Bowen Li, Ziqi Xu, Xikun Zhang, Haytham Fayek, and Xiaodong
Li. 2026. When to Trust: A Causality-Aware Calibration Framework for
∗Both authors contributed equally to this research.
†Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
(a) naive KG-RAG
 (b) Ca2KG (ours)
Figure 1: Calibration comparison on the MetaQA dataset.
Calibration error quantifies the average discrepancy between
a model’s predicted confidence and its actual correctness. The
naive KG-RAG framework (a) is consistently over-confident
with a high calibration error, whereas our Ca2KG framework
(b) achieves much better calibration with a reduced error.
Accurate Knowledge Graph Retrieval-Augmented Generation. In.ACM,
New York, NY, USA, 12 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Retrieval-Augmented Generation (RAG) is a powerful framework
that enhances Large Language Models (LLMs) by retrieving rel-
evant external information from large-scale web corpora to sup-
port more informed and factual generation [ 6,12,25,49]. By in-
tegrating retrieval and generation into a unified pipeline, RAG
enables LLMs to access knowledge beyond their training data and
mitigates hallucination by grounding responses in retrieved web
evidence [ 15,28,42]. However, conventional RAG systems rely
primarily on unstructured web text, which often lacks the seman-
tic precision, logical structure, and interpretability necessary for
complex reasoning tasks. To overcome these limitations, Knowl-
edge Graph Retrieval-Augmented Generation (KG-RAG) extends
the RAG paradigm by incorporating structured knowledge from
web-based knowledge graphs. This integration allows models to
retrieve and reason over multi-hop semantic paths, leading to more
precise and explainable generation, especially for web-scale tasks
requiring multi-step reasoning [17, 26, 36, 44].arXiv:2601.09241v1  [cs.CL]  14 Jan 2026

Conference’17, July 2017, Washington, DC, USA Ren et al.
Although KG-RAG has been shown to substantially improve fac-
tual accuracy in complex tasks [ 13,19,35,36], the naive KG-RAG
framework often exhibits such over-confidence despite factual inac-
curacies. (see Figure 1). This issue directly reflects poor calibration,
which measures how well a model’s predicted confidence aligns
with its actual likelihood of being correct. A well-calibrated model
outputs high confidence only when its predictions are reliable, and
low confidence when uncertainty is warranted. Calibration is es-
pecially critical for KG-RAG because inaccurate confidence can
mislead downstream reasoning steps, amplify retrieval errors, and
cause incorrect entity linking or graph traversal. Therefore, improv-
ing calibration is urgently needed for reliable web-scale knowledge
retrieval and generation.
Building on this need, prior research has investigated calibra-
tion as a core aspect of RAG system reliability, along with broader
uncertainty estimation, which seeks to quantify and decompose dif-
ferent sources of predictive uncertainty [ 1,30,40]. However, these
efforts have primarily focused on standard LLMs or text-based RAG
models, with limited exploration of KG-RAG. Unlike conventional
RAG that retrieves unstructured web text, KG-RAG relies on struc-
tured knowledge from web-based knowledge graphs in the form of
entities, relations, and multi-hop paths, which introduces unique
challenges in both retrieval and reasoning. These characteristics
raise a central research question:how can we improve calibration in
KG-RAG by explicitly modelling retrieval-dependent uncertainties, so
that the model’s confidence better reflects its true reliability?
To address this problem, we propose Ca2KG, a Causality-aware
Calibration framework for Knowledge Graph Retrieval-Augmented
Generation that combines counterfactual prompting with panel-
based re-scoring. The framework is motivated by the causal view
that different prompting interventions can be regarded as treat-
ments that expose retrieval-dependent uncertainties. Inspired by
the counterfactual prompting framework [ 2], we design prompts
that simulate alternative web retrieval scenarios, such as “suppose
the wrong knowledge path was selected” or “suppose the reasoning
over the retrieved path was flawed”, thereby encouraging the frame-
work to introspect on the robustness of retrieved evidence. Building
on these counterfactual generations, we introduce a panel-based
re-scoring process, where the framework evaluates candidate an-
swers under each intervention and assigns calibrated probabilities
through a unified scoring scheme. This two-stage process allows us
to not only expose and quantify uncertainties arising from retrieved
sub-graphs, but also to stabilise predictions across interventions, ul-
timately leading to more calibrated and reliable web-scale KG-RAG
systems. In summary, our contributions are as follows:
•We present the first systematic study on the calibration of KG-
RAG systems, revealing that existing frameworks often produce
severely overconfident predictions.
•We propose Ca2KG, a causality-aware calibration framework
that integrates counterfactual prompting with panel-based re-
scoring, and introduce a stability-based scoring mechanism
that explicitly accounts for retrieval-dependent uncertainties
in both knowledge quality and reasoning reliability.
•We conduct extensive experiments on two complex QA datasets,
demonstrating that Ca2KG consistently improves calibrationmetrics (e.g., Expected Calibration Error, Brier Score) while
maintaining or even enhancing predictive accuracy.
Our work falls under theSemantics and Knowledgetrack as it
focuses on calibrating KG-RAG, which directly relies on Web-based
structured knowledge graphs with machine-interpretable semantics.
Our contribution advances frameworks that synergise knowledge
graphs and LLMs, leading to more trustworthy semantic reasoning
and user-facing applications on the Web.
2 Preliminaries
2.1 Structural Causal Model
A Structural Causal Model (SCM) formalises causal relationships
between variables using a directed acyclic graph (DAG). Formally,
an SCM is represented by a causal DAG G=(V,E) , where each
node𝑋∈V corresponds to a random variable, and each directed
edge(𝑈→𝑉)∈E encodes a direct causal influence of 𝑈on𝑉.
For a node𝑋, we denote its set of parents by PA(𝑋) , which are all
nodes with edges pointing into𝑋.
A causal path from 𝑋to𝑌is a directed sequence of nodes
𝑋→···→𝑌 that follows the arrow direction, indicating that
𝑋is a (possibly indirect) cause of 𝑌. This graphical representa-
tion provides the foundation for reasoning about intervention and
counterfactual in causal inference.
2.2 Causal Intervention
In real-world applications, it is essential to distinguish between
mere statistical associations and genuine causal relationships. The
observational conditional distribution 𝑃(𝑌|𝑋) characterises how
two variables co-vary in passively collected data, but it does not
reveal what would happen to 𝑌if𝑋was actively manipulated. To
address this limitation, thedo-operator[ 23] is introduced, formalis-
ing the notion of a causal intervention.
The operator do(𝑋=𝑥) represents a surgical intervention that
forces the variable 𝑋to take the value 𝑥while cutting all incoming
edges into𝑋in the causal DAGG. Intuitively, this breaks the natural
causal mechanisms that determine 𝑋and replaces them with a fixed
assignment. As a result, variation in 𝑋no longer depends on its
original causes, but solely on the external manipulation.
Formally, given a causal DAG G=(V,E) and its Markov fac-
torisation.
𝑃(V)=Ö
𝑉∈V𝑃 𝑉|PA(𝑉),(1)
the distribution after an intervention do(𝑋=𝑥) can be expressed
as follows:
𝑃(V\{𝑋}|do(𝑋=𝑥))=Ö
𝑉∈V\{𝑋}𝑃 𝑉|Pa(𝑉)
𝑋=𝑥.(2)
Based on this formula, we can formalise thecausal effectof a
treatment variable 𝑇on an outcome variable 𝑌as the following
mapping:
𝑡↦→𝑃(𝑌|do(𝑇=𝑡)),(3)
which specifies the distribution of 𝑌that would arise under different
interventions on𝑇.
A common summary measure is the difference in expectations
under two interventions,
E[𝑌|do(𝑇=𝑡′)]−E[𝑌|do(𝑇=𝑡′′)],(4)

When to Trust: A Causality-Aware Calibration Framework for Accurate KG-RAG Conference’17, July 2017, Washington, DC, USA
Figure 2: The overall architecture of Ca2KG. Given a query, the initial KG-RAG pipeline produces a baseline answer. Counterfac-
tual prompting introduces interventions on retrieved paths, simulating quality and reasoning failures to generate alternative
answers. Panel-based re-scoring evaluates all candidates under each prompt, forming a3 ×𝑁probability matrix. Finally, the
Causal Calibration Index (CCI) combines support and stability across interventions to select the final calibrated answer.
which quantifies how outcomes would change when the treatment
is shifted from𝑡′′to𝑡′.
However, causal effects cannot in general be computed directly
from observational data, since multiple causal models may give rise
to the same joint distribution. This motivates the notion ofiden-
tifiability, which requires that a causal effect 𝑃(𝑌|do(𝑇=𝑡)) be
uniquely determined from the observational distribution together
with the causal graph. Identifiability ensures that the effect of in-
terventions can be inferred from observed data under appropriate
assumptions about the underlying causal structure.
These notions of interventions and causal effects provide the
foundation of our framework. Specifically, we interpret different
counterfactual prompting strategies as distinct treatments, and
regard the resulting changes in model outputs as causal effects.
This causal perspective enables us to move beyond surface-level
associations in observed responses and to systematically quantify
how interventions on prompts influence model predictions.
3 Methodology
In this section, we present Ca2KG, our causality-aware calibration
framework designed to improve the reliability of KG-RAG. The
overall architecture is illustrated in Figure 2. Our framework is
structured around four main components: (i) formulating the cal-
ibration task within a causal inference perspective, (ii) designing
counterfactual prompting strategies to simulate retrieval and rea-
soning failures, (iii) employing a panel-based re-scoring mechanismto estimate interventional distributions, and (iv) developing a causal
calibration criterion for final answer selection. Together, these com-
ponents enable the model to generate predictions that are both
accurate and well-calibrated.
3.1 Problem Statement
We consider the calibration task of KG-RAG, where an LLM is
augmented with structured knowledge from a KG to generate
accurate and grounded responses to queries. Formally, given a
query𝑞, the system first performs entity linking to identify the
relevant KG entities E𝑞={𝑒 1,...,𝑒𝑚}. Based on these entities, a
sub-graphG𝑞
𝐾𝐺=(𝑉𝑞,𝐸𝑞)is retrieved, which contains candidate
pathsP𝑞={𝑝 1,...,𝑝𝑘}that may support answering 𝑞. Each path𝑝𝑖
can be represented as a sequence of entity-relation-entity triples or
as natural language statements, and is encoded into a prompt 𝑡prompt .
The final input to the LLM is constructed as [𝑞,𝑡 prompt], from which
the model𝑓(·)generates an answer 𝑎together with a confidence
score𝑐∈[ 0,1]. However, such confidence estimates are often un-
reliable, particularly when the retrieved knowledge is incomplete,
noisy, or biased. This motivates us to propose a causal prompting
framework to enhance the reliability of confidence scores in KG-
RAG by incorporating causal-aware reasoning, thereby enabling
more trustworthy answer selection.

Conference’17, July 2017, Washington, DC, USA Ren et al.
3.2 Causal Principles
We conceptually formalise the relationship between the query 𝑄,
the prompt𝑇, and the answer 𝐴using a causal DAG G. As illustrated
in Figure 2,Gcontains three directed edges:
𝑇→𝐴, 𝑄→𝑇, 𝑄→𝐴.(5)
Here,𝑄denotes the query, 𝑇represents the prompt intervention,
and𝐴denotes the candidate answer. This structure reflects that the
query influences both the chosen prompt and the resulting answer,
while the prompt itself also has a direct causal effect on the answer.
In our framework, we focus on heterogeneous causal DAGs. For
a given query 𝑞, we design counterfactual prompting strategies
𝑡𝑛that serve as treatments, and our goal is to obtain unbiased
estimates of the causal effects of these treatments on the resulting
answers. These causal effects can further be interpreted as calibrated
confidence scores, which provide a principled basis for reliable
answer selection and subsequent generation.
Within this causal structure, the path 𝑇←𝑄→𝐴 serves as
a confounding path. As a result, the observational distribution
𝑃(𝐴|𝑇) cannot be directly interpreted as the causal effect of 𝑇on
𝐴, since both are influenced by the query 𝑄. Without adjusting for
𝑄, any estimated effect of prompts on answers would be biased. To
achieve causal identification, we adopt three standard assumptions
from causal inference for KG-RAG:
Assumption 1 (Stable Unit Treatment Value Assumption
(SUTVA)).For KG-RAG, the potential answers associated with a
given query are unaffected by the prompting strategies applied to
other queries (no interference). Moreover, each prompting strategy
is assumed to have a unique, well-defined version for that query,
such that different forms of the same strategy do not lead to different
potential answers (consistency).
This assumption has two key principles. First,no interference:
the potential answer for one query is unaffected by prompting
strategies applied to other queries, provided queries are evaluated
independently. Second,consistency: each prompting strategy must
correspond to a single, well-defined generative process for a given
query, without ambiguity or hidden variations. In practice, since
LLMs often operate with stochastic decoding (e.g., non-zero temper-
ature), consistency is interpreted at the level of the induced output
distribution rather than individual sampled responses.
Assumption 2 (Unconfoundedness).For KG-RAG, the potential
answers are assumed to be conditionally independent of the prompting
strategy𝑇given the query𝑄. Formally,
𝐴⊥⊥𝑇|𝑄.(6)
This assumption implies that, once the query is taken into ac-
count, the assignment of prompting strategies does not carry any
additional hidden confounding information about the answers.
Assumption 3 (Overlap).For KG-RAG, any query 𝑄=𝑞 , every
prompting strategy 𝑡has a non-zero probability of being assigned.
Formally,
0<𝑃(𝑇=𝑡|𝑄=𝑞)<1.(7)
This assumption ensures that all prompting strategies are fea-
sible for each query, making causal comparisons across strategies
possible.Together, these assumptions guarantee that causal effects are, in
principle, identifiable from observational data. To operationalise
this identification in our framework, we now introduce back-door
criterion [ 23], which provides a graphical condition for determining
suitable adjustment sets in a causal DAG.
Definition 1 (Back-Door Criterion).A set of variables 𝑍
satisfies the back-door criterion relative to an ordered pair of variables
(𝑇,𝐴)in a causal DAGGif:
(1) no node in𝑍is a descendant of𝑇; and
(2)𝑍 blocks every path between 𝑇and𝐴that contains an arrow
into𝑇.
By adjusting for such a set 𝑍, the causal effect of 𝑇on𝐴can be
identified as
𝑃(𝑌|𝑑𝑜(𝑇))=∑︁
𝑧𝑃(𝐴|𝑇,𝑍=𝑧)𝑃(𝑍=𝑧).(8)
In our causal DAG, the query 𝑄satisfies the back-door criterion
relative to the treatment–outcome pair (𝑇,𝐴) , where𝑇denotes
the prompt and 𝐴the answer. It is not a descendant of 𝑇, and it
blocks the only back-door path 𝑇←𝑄→𝐴 , which implies that
𝐴⊥⊥𝑇|𝑄 . Therefore, by conditioning on 𝑄, the causal effect of
prompt strategies𝑇on the answer𝐴can be identified as
𝑃(𝐴|𝑑𝑜(𝑇))=𝑃(𝐴|𝑇,𝑄).(9)
Note that we focus on heterogeneous causal DAGs where each
query𝑞corresponds to a distinct causal structure. Thus, we are
interested in query-specific causal effects rather than population-
averaged effects, and no marginalisation over𝑄is required.
3.3 Counterfactual Prompting
We aim to design a counterfactual prompting strategy that per-
turbs KG-related evidence in two ways: by varying the quality of
retrieved paths and by altering their usage strategy. This strategy
encourages the model to reflect on its confidence under such inter-
ventions. The core challenge is to assess model uncertainty without
relying on gold-standard answers, which is crucial for enabling se-
lective prediction and risk-aware decision-making in downstream
applications. Motivated by cognitive theories of counterfactual rea-
soning [ 23], we propose two counterfactual prompting strategies
that systematically simulate failure scenarios in KG-RAG systems.
•Path Quality Intervention ( 𝑡1): simulates scenarios where
the retrieved KG paths are irrelevant, incomplete, or noisy, and
prompts the LLM accordingly: “Assume your previous answer
is wrong because the quality of the referred contexts is poor.
Re-select the most relevant parts from the given contexts and
regenerate the answer using one or a few words. Output MUST
be exactly one line in this format: {final answer}. Do not include
any other text. Examples: {Italian Languages}”
•Reasoning Reliability Intervention ( 𝑡2): simulates scenarios
where the model’s reasoning over otherwise valid KG paths
may be unreliable or flawed, and prompts the LLM accordingly:
“Assume your previous answer is wrong due to improper use of
the retrieved contexts. Carefully re-check the provided contexts
and regenerate the answer using one or a few words. Output
MUST be exactly one line in this format: {final answer}. Do not
include any other text. Examples: {Italian Languages}”

When to Trust: A Causality-Aware Calibration Framework for Accurate KG-RAG Conference’17, July 2017, Washington, DC, USA
Each counterfactual prompt 𝑡1and𝑡2is fed into the same back-
bone LLM, which produces the corresponding counterfactual an-
swers𝑎1and𝑎2. In addition to the counterfactual prompting strate-
gies, we also include an initial prompt 𝑡0as the baseline, which
generates the initial answer 𝑎0. The prompt is defined as follows:
“Use the provided contexts to answer the question. If the contexts
are incomplete or weak, still provide your best possible answer.
Output MUST be exactly one line in this format: {final answer}. Do
not include any other text. Examples: {Italian}”
3.4 Panel-based Re-scoring
To estimate the interventional distribution 𝑃(𝐴|𝑑𝑜(𝑇)) , we in-
troduce a unified Panel Prompt, which serves as an evaluator and
prompts the framework as follows:
You are given 3 candidate answers for the same query from
different prompts:𝑡 0:𝑎0;𝑡1:𝑎1;𝑡2:𝑎2.
Task:
(1)Build a unified set of UNIQUE answers by merging
semantically identical strings across𝑎 0,𝑎1,𝑎2.
(2)Compute a GLOBAL frequency vector over the unique
answers based on how many of 𝑎0/𝑎1/𝑎2map to each
canonical answer.
(3)ASSIGN probabilities for EACH interventions ( 𝑡0,𝑡1,
𝑡2) to be EXACTLY this global frequency vector nor-
malised by 3 (counts/3), after duplicate aggregation
and before rounding.
Re-scoring Structure:
Merging Rules
•Ignore case, whitespace, punctuation, trivial format-
ting, and plural/singular differences.
•Choose a clean canonical form for answers (e.g., title
case).
•If multiple inputs ( 𝑎0/𝑎1/𝑎2) map to the same canonical
answer, aggregate them by SUMMING that answer’s
frequency before normalisation.
Probability rules (HARD CONSTRAINTS)
•Let count[i] be how many of 𝑎0/𝑎1/𝑎2map to an-
swers[i]. Then for every intervention T in 𝑡0,𝑡1,𝑡2,
set T[i] = round(count[i]/3, 2).
•Do NOT output equal probabilities across answers
when counts differ.
•Each probability∈ [0.00,1.00]; sums may be < 1.00
after rounding.
•If an intervention has no plausible answers, use zeros.
Return EXACTLY one line of STRICT JSON, no wrap-
per, no extra text.
Given a specific query 𝑞, we consider the candidate answer set
𝐴={𝑎 0,...,𝑎𝑁}obtained by aggregating outputs from the three
prompt strategies 𝑇={𝑡 0,𝑡1,𝑡2}. Since each intervention produces
one answer, the candidate set size satisfies0 ≤𝑁≤ 2, depending
on whether duplicates are merged during aggregation. The Panel
Prompt then re-scores each candidate answer 𝑎𝑖∈𝐴under every
prompt strategy𝑡 𝑗∈𝑇.Formally, for each prompt strategy 𝑡𝑗∈ {𝑡 0,𝑡1,𝑡2}, the Panel
Prompt produces a probability distribution over the candidate an-
swer set𝐴:
𝑃(𝐴=𝑎𝑖|𝑇=𝑡𝑗,𝑄), 𝑖=1,...,𝑁, 𝑗=0,1,2.(10)
These probabilities can be interpreted as the causal effect of
prompt strategy 𝑡𝑗on candidate answer 𝑎𝑖under query 𝑄. Collect-
ing these causal effects yields a score matrixC ∈R3×𝑁, where
each row corresponds to a prompt strategy 𝑡𝑗and each column
corresponds to a candidate answer𝑎 𝑖. The matrix is given as:
C=𝑐𝑡0,𝑎1, ... ,𝑐 𝑡0,𝑎𝑁
𝑐𝑡1,𝑎1, ... ,𝑐 𝑡1,𝑎𝑁
𝑐𝑡2,𝑎1, ... ,𝑐 𝑡2,𝑎𝑁.(11)
3.5 Accurate Answer Generation
Each candidate answer 𝑎∈𝐴 receives probability assignments,
interpreted as causal effects, across prompt strategies. However,
selecting the candidate with the highest average probability may
yield unstable decisions. To address this, we design a stability-aware
selection criterion that balances accuracy with robustness under
interventions.
We first measure the variability of the probabilities assigned
to each candidate across prompt strategies. Given the probability
matrixC, the causal effect variation ( CEvar) of a candidate answer
𝑎is defined as:
CEvar(𝑎)=max
𝑗𝑐𝑡𝑗,𝑎−min
𝑗𝑐𝑡𝑗,𝑎, 𝑗=0,1,2(12)
where𝑐𝑡𝑗,𝑎denotes the probability of candidate 𝑎under treatment
𝑡𝑗. A higher CEvar(𝑎)indicates greater instability across prompt
strategies.
Next, we compute the average causal effect CE for candidate𝑎:
CE(𝑎)=1
32∑︁
𝑗=0𝑐𝑡𝑗,𝑎.(13)
To jointly capture accuracy and calibration, we define the Causal
Calibration Index (CCI):
CCI(𝑎)= CE(𝑎)·(1−CE var(𝑎)),(14)
which promotes candidates with both high average probability
and consistent behaviour across interventions, thereby improving
answer selection and calibration.
The final answer is selected as:
𝑎∗=arg max
𝑎CCI(𝑎).(15)
This design ensures that the selected answer is not only highly
probable but also well-calibrated across interventions, leading to
more reliable causal decision-making.
4 Experimental Setup
4.1 Datasets
We evaluate our method on two widely used benchmarks for multi-
hop question answering over knowledge graphs, MetaQA [ 45] and
WebQSP [ 41], each supporting 1-hop and 3-hop reasoning tasks.
This setup allows us to assess model performance under both shal-
low and deep reasoning settings.

Conference’17, July 2017, Washington, DC, USA Ren et al.
MetaQA is a synthetic, movie-domain dataset containing over
400K natural language questions generated from a structured KG
with entities such as movies, actors, directors, and genres. WebQSP
is a real-world dataset extending WebQuestions with semantic
parses grounded to Freebase. It contains over 4,700 user queries
annotated with SPARQL-like logical forms, supporting multi-hop
reasoning and entity linking. Compared to MetaQA, WebQSP is
more challenging due to its natural language variability, diverse
entities and relations, and reliance on precise semantic parsing.
4.2 Baselines
We select a series of baselines with different prompting strategies
and calibration methods to compare against our proposed frame-
work. Specifically, we consider three prompt-based baselines: If-or-
Else (IoE) prompting framework [ 16] improves self-correction by
leveraging the IoE prompting principle to adjust responses based
on model confidence. Self-Correct [ 11] explores the role and effec-
tiveness of self-correction in LLMs through a three-step prompting
strategy. RC-RAG [2] mitigates risks in LLMs by enforcing consis-
tency in answers and discarding inconsistent ones.
We also adopt three verbalised strategies for extracting confi-
dence estimates, following [ 32]: Verb1S-top4 prompts the model
to produce four candidate answers and assign a probability of cor-
rectness to each within a single response. Verb2S-top4 separates
the process into two stages: the first turn generates four candidate
answers, and the second turn elicits correctness probabilities for
each. Verb2S-CoT adds chain-of-thought reasoning in the first turn
before producing a single answer, while the second turn requests
a confidence score for that answer with the reasoning retained in
context.
4.3 Backbone LLMs
In our experiments, we employ two backbone LLMs with different
parameter scales and accessibility: LLaMA-3 [ 8] and GPT-3.5 [ 21].
This selection enables evaluation of calibration performance across
both open- and closed-source models, as well as across model sizes,
providing a comprehensive and balanced experimental setting.
4.4 Metrics
In our experiments, we useAccuracy(Acc) as the primary perfor-
mance metric. Following [ 32], we further assess model calibration
with a range of established metrics. Acc measures the proportion of
correctly predicted labels over the total number of test instances and
directly reflects the discriminative ability of the model.Expected
Calibration Error(ECE) is a widely used metric for quantifying
the alignment between predicted confidence and empirical accu-
racy [ 9]. To compute ECE, predictions are first partitioned into 𝑀
equally spaced bins based on their confidence scores. For each bin
𝐵𝑚, the average confidence conf(𝐵𝑚)and the empirical accuracy
acc(𝐵𝑚)are computed. ECE is defined as the weighted average of
the absolute differences between confidence and accuracy across
all bins: ECE=Í𝑀
𝑚=1|𝐵𝑚|
𝑛|acc(𝐵𝑚)−conf(𝐵 𝑚)|, where|𝐵𝑚|is the
number of samples in bin 𝑚and𝑛is the total number of samples. A
lower ECE indicates better calibration, meaning predicted probabili-
ties more accurately reflect the true likelihood of correctness.Brier
Score(BS) provides a complementary perspective on calibrationby measuring the mean squared difference between predicted con-
fidence and the ground-truth label. BS=1
𝑛Í𝑛
𝑖=1(𝑐𝑖−𝑦𝑖)2, where
𝑐𝑖∈[0,1]is the predicted confidence score and 𝑦𝑖∈{0,1}is the
ground-truth indicator of correctness. A lower BS indicates that
predicted probabilities are closer to the true outcomes. In addition
to standard calibration metrics, we also report theArea Under the
Selective Accuracy-Coverage Curve(AUC), introduced in [ 7].
This metric evaluates the trade-off between accuracy and coverage
when the model abstains from predictions with low confidence. By
integrating accuracy across different coverage levels, AUC captures
the ability of the model to identify and withhold uncertain predic-
tions, offering a complementary perspective on reliability beyond
ECE and BS.
5 Experimental Results
We structure our investigation around the following five research
questions: RQ1: How does Ca2KG improve calibration relative to
baselines, and how does this translate into downstream utility (e.g.,
accuracy)? RQ2: How does the capability of the backbone LLM af-
fect the overall performance? RQ3: How does reasoning complexity
(e.g., 1-hop vs. 3-hop) influence both calibration and accuracy? RQ4:
What is the computational cost of the proposed Ca2KG framework?
RQ5: What are the respective contributions of the two counterfac-
tual prompting strategies to model effectiveness?
In addition, for each dataset used in our experiments, we provide
a case study in Appendix B to illustrate the practical impact of
Ca2KG on representative examples.
5.1 Main Result (RQ1-RQ3)
5.1.1 RQ1: Calibration and Utility.From Table 1, we note that
Ca2KG substantially improves calibration compared with base-
lines. Under both GPT-3.5 and LLaMA-3 backbones, Ca2KG con-
sistently achieves the lowest ECE (e.g., 0.067/0.055 on MetaQA
and 0.196/0.108 on WebQSP with GPT-3.5) and BS (0.078–0.128),
while maintaining high AUC (up to 0.952). In contrast, KG-RAG
and prompting baselines (Verb1S-Top4, Verb2S-Top4, Verb2S-CoT)
exhibit poor calibration, with ECE values often exceeding 0.3–0.6.
These results confirm that causality-aware approaches effectively
reduce miscalibration while improving accuracy, demonstrating a
superior balance of reliability and predictive performance.
5.1.2 RQ2: Backbone Capacity.Comparing GPT-3.5 and LLaMA-3
results highlights the influence of the backbone LLM. GPT-3.5 gen-
erally achieves higher accuracies on WebQSP (e.g., 0.769 vs. 0.738 on
1-hop; 0.819 vs. 0.612 on 3-hop) and stronger calibration with lower
ECE and BS, while LLaMA-3 performs competitively on MetaQA
(e.g., 0.872 vs. 0.896 on 3-hop). These results suggest that stronger
LLMs not only enhance reasoning accuracy but also improve cali-
bration robustness when combined with Ca2KG. Importantly, the
relative advantage of Ca2KG remains consistent across backbones,
indicating that counterfactual prompting provides complementary
benefits independent of backbone choice.
5.1.3 RQ3: Task Difficulty.The comparison between MetaQA and
WebQSP illustrates the effect of task difficulty. MetaQA is synthet-
ically generated and domain-specific, making it relatively easier,
whereas WebQSP is based on real user queries, requiring semantic

When to Trust: A Causality-Aware Calibration Framework for Accurate KG-RAG Conference’17, July 2017, Washington, DC, USA
Table 1: Results on MetaQA and WebQSP. KG-RAG denotes the standard retrieval-augmented generation baseline without
any calibration adjustment. The symbol “–” indicates that calibration metrics are not reported, since baselines such as IoE,
Self-Correct, and RC-RAG are not originally designed to produce probabilistic confidence estimates. Best results are highlighted
in bold.
MetaQA WebQSP
1-hop 3-hop 1-hop 3-hop
MethodAcc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑
GPT-3.5
KG-RAG 0.554 0.433 0.416 0.706 0.177 0.821 0.819 0.116 0.340 0.646 0.538 0.529 0.031 0.908 0.911 0.050
IoE 0.843 - - - 0.590 - - - 0.557 - - - 0.673 - - -
Self-Correct 0.539 - - - 0.219 - - - 0.409 - - - 0.162 - - -
RC-RAG 0.874 0.112 0.094 0.934 0.815 0.084 0.083 0.943 0.704 0.302 0.295 0.727 0.794 0.139 0.144 0.863
Verb1S-Top4 0.813 0.160 0.152 0.897 0.837 0.094 0.127 0.889 0.618 0.309 0.313 0.724 0.492 0.449 0.448 0.437
Verb2S-Top4 0.829 0.187 0.130 0.931 0.861 0.190 0.107 0.921 0.458 0.291 0.285 0.731 0.176 0.676 0.586 0.155
Verb2S-CoT 0.562 0.437 0.437 0.618 0.610 0.520 0.471 0.533 0.548 0.420 0.401 0.593 0.670 0.539 0.434 0.486
Ca2KG0.876 0.067 0.078 0.947 0.896 0.055 0.069 0.950 0.769 0.196 0.184 0.837 0.819 0.108 0.128 0.920
LLaMA-3
KG-RAG 0.724 0.264 0.220 0.887 0.470 0.242 0.269 0.697 0.579 0.330 0.301 0.710 0.004 0.739 0.551 0.004
IoE 0.808 - - - 0.710 - - - 0.372 - - - 0.606 - - -
Self-Correct 0.605 - - - 0.348 - - - 0.455 - - - 0.165 - - -
RC-RAG 0.812 0.124 0.118 0.912 0.736 0.118 0.121 0.913 0.642 0.269 0.258 0.759 0.315 0.354 0.368 0.392
Verb1S-Top4 0.646 0.108 0.126 0.933 0.595 0.091 0.086 0.941 0.358 0.456 0.436 0.544 0.397 0.486 0.481 0.445
Verb2S-Top4 0.834 0.147 0.118 0.921 0.736 0.118 0.121 0.913 0.642 0.269 0.258 0.759 0.315 0.354 0.368 0.392
Verb2S-CoT 0.668 0.373 0.282 0.844 0.648 0.320 0.288 0.753 0.324 0.630 0.577 0.433 0.501 0.619 0.602 0.392
Ca2KG0.861 0.085 0.093 0.944 0.872 0.064 0.079 0.952 0.738 0.211 0.203 0.812 0.612 0.242 0.266 0.657
parsing and entity disambiguation, which substantially increases
complexity. This difference is reflected in the results: even with GPT-
3.5, accuracies on WebQSP are notably lower than on MetaQA (e.g.,
0.769 vs. 0.876 in 1-hop, 0.819 vs. 0.896 in 3-hop), and calibration
errors are consistently higher (ECE = 0.196/0.108 vs. 0.067/0.055).
Nevertheless, Ca2KG maintains superior calibration and accuracy
on both datasets, demonstrating robustness even in more challeng-
ing real-world QA settings.
5.2 Efficiency Analysis (RQ4)
We analyse the efficiency of different methods on MetaQA 1-hop
in terms of token usage and performance under token caps. In Fig-
ure 3a, Ca2KG achieves the lowest token cost per correct prediction
(3.49), outperforming baselines such as KG-RAG (4.82), Self-Correct
(5.20), and Verb2S-CoT (11.92). This shows that Ca2KG requires
fewer tokens to achieve the same or better accuracy, making it
highly cost-effective. Figure 3b further confirms this advantage:
when token budgets are restricted, Ca2KG maintains stable accu-
racy across different caps, in contrast to baselines such as KG-RAG
(STD) and other prompting methods, which suffer sharp drops.
These results demonstrate that counterfactual prompting not only
improves calibration and accuracy (RQ1–RQ3), but also delivers
strong efficiency, ensuring robustness under resource-constrained
settings.
(a)
 (b)
Figure 3: Efficiency analysis on MetaQA. (a) Token usage per
correct prediction in the 1-hop setting. (b) Accuracy under
different token caps.
5.3 Ablation Study (RQ5)
Table 2 shows that both Path Quality Intervention ( 𝑡1) and Rea-
soning Reliability Intervention ( 𝑡2) play complementary roles in
enhancing model effectiveness. Removing 𝑡1leads to clear degrada-
tion in calibration, with ECE and BS increasing across datasets (e.g.,
ECE rises from 0.067 to 0.103 on MetaQA 1-hop, and from 0.108
to 0.202 on WebQSP 3-hop), and AUC dropping (0.950 →0.936).
Similarly, excluding 𝑡2causes a moderate decline in performance,
particularly in calibration metrics (e.g., BS = 0.099 vs. 0.078 on
MetaQA 1-hop; ECE = 0.211 vs. 0.196 on WebQSP 1-hop). Notably,
removing both interventions results in the most severe degradation:
accuracy decreases substantially (e.g., 0.896 →0.817 on MetaQA

Conference’17, July 2017, Washington, DC, USA Ren et al.
Table 2: Ablation study on MetaQA and WebQSP.
MetaQA WebQSP
1-hop 3-hop 1-hop 3-hop
MethodAcc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑Acc↑ECE↓BS↓AUC↑
Ca2KG 0.876 0.067 0.078 0.947 0.896 0.055 0.069 0.950 0.769 0.196 0.184 0.837 0.819 0.108 0.128 0.920
w/o Path Quality𝑡 1 0.866 0.103 0.080 0.946 0.881 0.075 0.080 0.936 0.768 0.207 0.217 0.826 0.808 0.202 0.135 0.823
w/o Rea. Reliability𝑡 20.870 0.086 0.099 0.940 0.885 0.056 0.081 0.933 0.767 0.211 0.201 0.808 0.803 0.209 0.143 0.837
w/o Both (𝑡 1&𝑡2) 0.865 0.134 0.134 0.894 0.817 0.172 0.172 0.825 0.752 0.245 0.245 0.753 0.791 0.213 0.175 0.816
3-hop, 0.769→0.752 on WebQSP 1-hop), calibration errors nearly
double (ECE = 0.067 →0.134 on MetaQA 1-hop, 0.108 →0.213 on
WebQSP 3-hop), and AUC declines sharply (0.950 →0.825). These
results demonstrate that 𝑡1is especially crucial for calibration and
robustness, 𝑡2further improves reliability and uncertainty estima-
tion, and together they provide complementary benefits that are
critical for achieving state-of-the-art performance.
6 Related Work
6.1 Calibration of LLM
Current strategies for improving calibration in LLMs span a wide
spectrum. Several studies [ 14,18,22,37] demonstrate that combin-
ing large pre-trained models with temperature scaling [ 9] can yield
better calibrated predictions. Other works explore whether linguis-
tic cues in model outputs provide reliable signals of uncertainty, and
how these align with actual confidence [ 20,32,48]. More recently,
prompting-based approaches have attracted attention due to their
flexibility and applicability to black-box LLMs, enabling techniques
such as self-reported confidence, multi-sample self-consistency,
and explicit uncertainty expression. These are especially valuable
for open-ended or generative tasks where post-hoc calibration is
less effective [ 29]. For KG-RAG, prompting-based strategies are
particularly appealing as they allow models to reason over struc-
tured knowledge paths and assess evidence consistency without
modifying model architectures. Distinct from prior work, we in-
troduce counterfactual prompting to explicitly simulate failures in
knowledge retrieval and reasoning.
6.2 KG-RAG
Recent efforts have integrated knowledge graphs into retrieval-
augmented generation to improve factual accuracy, reasoning, and
interpretability. For example, Hu et al. [10] propose GRAG, which
retrieves relevant sub-graphs to guide multi-hop QA; Liang et al .
[17]introduce KAG, which enhances alignment between triples and
text with an attention-guided encoder-decoder; Liu et al . [19] inves-
tigate knowledge-injected prompting strategies that verbalise KG
paths into natural language; and Tan et al . [31] present a path-level
reasoning framework that explicitly models retrieval confidence
and path reliability. These approaches have substantially advanced
KG-RAG by improving the grounding of LLM outputs, enabling
more precise entity disambiguation and more controllable reason-
ing over structured knowledge. However, most of them concentrate
on improving factual accuracy and reasoning capability, while pay-
ing limited attention to the equally critical issue of calibration. Weargue that carefully designed prompting strategies can guide KG-
RAG models to introspect on conflicting or missing evidence, and
in doing so, provide confidence estimates that are more calibrated,
interpretable, and ultimately more reliable for real-world use.
6.3 Causal Inference for LLM
Causal inference aims to uncover the mechanisms underlying vari-
able interactions through rigorous methodologies [ 24]. Building on
solid theoretical foundations, many approaches have been devel-
oped to estimate causal effects [ 5,38,46], even in the presence of un-
observed confounders [ 3,4,39]. These techniques have been widely
applied in NLP, including de-biasing [ 47], fake news detection [ 33],
and sentiment analysis [ 27]. More recently, researchers have be-
gun incorporating causal reasoning into prompting. For example,
Causal Prompting [ 43] formulates prompts based on hypothesised
causal structures to elicit causally consistent outputs, while De-
CoT [ 34] embeds causal structures into chain-of-thought reasoning,
using front-door adjustment and instrumental variables to miti-
gate spurious reasoning caused by hidden confounders. Inspired
by these advances, our proposed Ca2KG framework introduces a
new perspective by integrating causal reasoning with knowledge
graph–based retrieval. In particular, Ca2KG leverages structured
causal paths from knowledge graphs to construct counterfactually
informed prompts, enabling LLMs to not only generate answers but
also provide uncertainty-aware and causally grounded reasoning.
7 Conclusion
In this paper, we present Ca2KG, a causality-aware calibration
framework for Knowledge Graph Retrieval-Augmented Generation.
By interpreting counterfactual prompting strategies as causal inter-
ventions and combining them with a panel-based re-scoring mech-
anism, our framework exposes and quantifies retrieval-dependent
uncertainties in both knowledge quality and reasoning reliability.
Extensive experiments on MetaQA and WebQSP show that Ca2KG
consistently reduces overconfidence and achieves state-of-the-art
calibration performance, while maintaining or even enhancing pre-
dictive accuracy. Beyond addressing the calibration gap in KG-RAG,
our work highlights the importance of causality-inspired prompting
as a general strategy for improving trustworthiness in web-scale
knowledge retrieval and generation.
References
[1]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A survey on

When to Trust: A Causality-Aware Calibration Framework for Accurate KG-RAG Conference’17, July 2017, Washington, DC, USA
evaluation of large language models.ACM Transactions on Intelligent Systems
and Technology15, 3 (2024), 39:1–39:45. https://doi.org/10.1145/3641289
[2] Lu Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan, and Xueqi Cheng. 2024. Con-
trolling Risk of Retrieval-augmented Generation: A Counterfactual Prompting
Framework. InFindings of the Association for Computational Linguistics: EMNLP.
2380–2393. https://doi.org/10.18653/v1/2024.findings-emnlp.133
[3]Debo Cheng, Jiuyong Li, Lin Liu, Ziqi Xu, Weijia Zhang, Jixue Liu, and Thuc Duy
Le. 2025. Disentangled Representation Learning for Causal Inference With
Instruments.IEEE Transactions on Neural Networks and Learning Systems36, 8
(2025), 14078–14091. https://doi.org/10.1109/TNNLS.2024.3512790
[4]Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, and Thuc Duy Le. 2024. Condi-
tional Instrumental Variable Regression with Representation Learning for Causal
Inference. InThe Twelfth International Conference on Learning Representations,
ICLR. https://openreview.net/forum?id=qDhq1icpO8
[5]Xiaojing Du, Jiuyong Li, Debo Cheng, Lin Liu, Wentao Gao, Xiongren Chen, and
Ziqi Xu. 2025. Telling Peer Direct Effects from Indirect Effects in Observational
Network Data. InProceedings of the 42nd International Conference on Machine
Learning, ICML. https://openreview.net/forum?id=qdKzBrYhiu
[6]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
Generation for Large Language Models: A Survey.CoRRabs/2312.10997, 1 (2023).
https://doi.org/10.48550/arXiv.2312.10997
[7]Yonatan Geifman and Ran El-Yaniv. 2017. Selective Classification for Deep Neural
Networks. InAdvances in Neural Information Processing Systems 30: Annual Con-
ference on Neural Information Processing Systems. 4878–4887. https://proceedings.
neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
[8]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al -Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo
Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun
Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste
Roziere, Bethany Biron, Binh Tang, Bobbie Chern, and ... [et al.]. 2024. The Llama
3 Herd of Models. https://doi.org/10.48550/arXiv.2407.21783
[9]Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On Calibration
of Modern Neural Networks. InProceedings of the 34th International Conference
on Machine Learning, ICML (Proceedings of Machine Learning Research, Vol. 70).
PMLR, 1321–1330. http://proceedings.mlr.press/v70/guo17a.html
[10] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025.
GRAG: Graph Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: NAACL. 4145–4157. https://doi.org/10.18653/v1/
2025.findings-naacl.232
[11] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei
Yu, Xinying Song, and Denny Zhou. 2024. Large Language Models Cannot
Self-Correct Reasoning Yet. InThe Twelfth International Conference on Learning
Representations, ICLR. https://openreview.net/forum?id=IkmD3fKBPQ
[12] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot Learning with Retrieval Augmented Language
Models.Journal of Machine Learning Research24, 251 (2023), 1–43. https:
//jmlr.org/papers/v24/23-0037.html
[13] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. KG-GPT: A General
Framework for Reasoning on Knowledge Graphs Using Large Language Models.
InFindings of the Association for Computational Linguistics: EMNLP. 9410–9421.
https://doi.org/10.18653/v1/2023.findings-emnlp.631
[14] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language Genera-
tion. InThe Eleventh International Conference on Learning Representations, ICLR.
https://openreview.net/forum?id=VD-AYtP0dve
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented Generation for Knowledge-intensive NLP Tasks. In
Advances in Neural Information Processing Systems 33: Annual Conference on
Neural Information Processing Systems, NeurIPS. https://proceedings.neurips.cc/
paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[16] Loka Li, Zhenhao Chen, Guangyi Chen, Yixuan Zhang, Yusheng Su, Eric Xing,
and Kun Zhang. 2024. Confidence Matters: Revisiting Intrinsic Self-Correction
Capabilities of Large Language Models.CoRRabs/2402.12563 (2024). https:
//doi.org/10.48550/arXiv.2402.12563
[17] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao,
Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, et al .2025. KAG:
Boosting LLMs in Professional Domains via Knowledge Augmented Generation.
InCompanion Proceedings of the ACM on Web Conference 2025, WWW. 334–343.
https://doi.org/10.1145/3701716.3715240
[18] Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching Models to Express
Their Uncertainty in Words.Transactions on Machine Learning Research2022
(2022). https://openreview.net/forum?id=8s8K2UZGTZ[19] Haochen Liu, Song Wang, Yaochen Zhu, Yushun Dong, and Jundong Li. 2024.
Knowledge Graph-Enhanced Large Language Models via Path Selection. InFind-
ings of the Association for Computational Linguistics, ACL. 6311–6321. https:
//doi.org/10.18653/v1/2024.findings-acl.376
[20] Sabrina J Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. 2022. Re-
ducing Conversational Agents’ Overconfidence through Linguistic Calibration.
Transactions of the Association for Computational Linguistics10 (2022), 857–872.
https://doi.org/10.1162/tacl_a_00494
[21] OpenAI. 2022. Introducing ChatGPT. https://openai.com/blog/chatgpt. Accessed:
2025-07-24.
[22] Seo Yeon Park and Cornelia Caragea. 2022. On the Calibration of Pre-trained
Language Models using Mixup Guided by Area Under the Margin and Saliency.
InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL. 5364–5374. https://doi.org/10.18653/
v1/2022.acl-long.368
[23] Judea Pearl. 2009.Causality. Cambridge university press.
[24] Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. 2016.Causal inference in
statistics: A primer. John Wiley & Sons.
[25] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context Retrieval-augmented Lan-
guage Models.Transactions of the Association for Computational Linguistics11
(2023), 1316–1331. https://doi.org/10.1162/tacl_a_00605
[26] Jing Ren, Feng Xia, Ivan Lee, Azadeh Noori Hoshyar, and Charu Aggarwal. 2023.
Graph learning for anomaly analytics: Algorithms, applications, and challenges.
ACM Transactions on Intelligent Systems and Technology14, 2 (2023), 1–29.
[27] Jing Ren, Wenhao Zhou, Bowen Li, Mujie Liu, Nguyen Linh Dan Le, Jiade Cen,
Liping Chen, Ziqi Xu, Xiwei Xu, and Xiaodong Li. 2025. Causal Prompting for
Implicit Sentiment Analysis with Large Language Models.CoRRabs/2507.00389
(2025). https://doi.org/10.48550/arXiv.2507.00389
[28] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in
Retrieval-augmented Generation. InProceedings of the 47th International ACM SI-
GIR Conference on Research and Development in Information Retrieval. 2395–2400.
https://doi.org/10.1145/3626772.3657957
[29] Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar.
2025. A Survey on Uncertainty Quantification of Large Language Models: Tax-
onomy, Open Research Challenges, and Future Directions.Comput. Surveys58,
3, Article 63 (2025), 38 pages. https://doi.org/10.1145/3744238
[30] Mark Steyvers, Heliodoro Tejeda, Aakriti Kumar, Catarina Belem, Sheer Karny,
Xinyue Hu, Lukas W Mayer, and Padhraic Smyth. 2025. What Large Language
Models Know and What People Think They Know.Nature Machine Intelligence
7, 2 (2025), 221–231. https://doi.org/10.1038/s42256-024-00976-7
[31] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang.
2025. Paths-over-graph: Knowledge Graph Empowered Large Language Model
Reasoning. InProceedings of the ACM on Web Conference 2025, WWW. ACM,
3505–3522. https://doi.org/10.1145/3696410.3714892
[32] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu
Yao, Chelsea Finn, and Christopher D Manning. 2023. Just Ask for Calibration:
Strategies for Eliciting Calibrated Confidence Scores from Language Models
Fine-Tuned with Human Feedback. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, EMNLP. 5433–5442. https:
//doi.org/10.18653/v1/2023.emnlp-main.330
[33] Siyin Wang, Jie Zhou, Changzhi Sun, Junjie Ye, Tao Gui, Qi Zhang, and Xuan-
Jing Huang. 2025. Causal Intervention Improves Implicit Sentiment Analysis.
InThirty-Ninth AAAI Conference on Artificial Intelligence, AAAI. 25842–25850.
https://doi.org/10.1609/aaai.v39i24.34777
[34] Junda Wu, Tong Yu, Xiang Chen, Haoliang Wang, Ryan A. Rossi, Sungchul
Kim, Anup B. Rao, and Julian J. McAuley. 2024. DeCoT: Debiasing Chain-of-
Thought for Knowledge-Intensive Tasks in Large Language Models via Causal
Intervention. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), ACL. 14073–14087. https:
//doi.org/10.18653/v1/2024.acl-long.758
[35] Feng Xia, Ciyuan Peng, Jing Ren, Falih Gozi Febrinanto, Renqiang Luo, Vidya
Saikrishna, Shuo Yu, and Xiangjie Kong. 2025. Graph Learning.arXiv preprint
arXiv:2507.05636(2025).
[36] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong,
Xiao Huang, and Jinsong Su. 2025. When to use Graphs in RAG: A Comprehensive
Analysis for Graph Retrieval-Augmented Generation.CoRRabs/2506.05690 (2025).
https://doi.org/10.48550/arXiv.2506.05690
[37] Yuxin Xiao, Paul Pu Liang, Umang Bhatt, Willie Neiswanger, Ruslan Salakhut-
dinov, and Louis-Philippe Morency. 2022. Uncertainty Quantification with Pre-
trained Language Models: A Large-Scale Empirical Analysis. InFindings of the
Association for Computational Linguistics: EMNLP. Association for Computational
Linguistics, 7273–7284. https://doi.org/10.18653/v1/2022.findings-emnlp.538
[38] Ziqi Xu, Debo Cheng, Jiuyong Li, Jixue Liu, Lin Liu, and Ke Wang. 2023. Disen-
tangled Representation for Causal Mediation Analysis. InThirty-Seventh AAAI
Conference on Artificial Intelligence, AAAI. 10666–10674. https://doi.org/10.1609/
aaai.v37i9.26266

Conference’17, July 2017, Washington, DC, USA Ren et al.
[39] Ziqi Xu, Debo Cheng, Jiuyong Li, Jixue Liu, Lin Liu, and Kui Yu. 2024. Causal
Inference with Conditional Front-Door Adjustment and Identifiable Variational
Autoencoder. InThe Twelfth International Conference on Learning Representations,
ICLR. https://openreview.net/forum?id=wFf9m4v7oC
[40] Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, and Pengfei Liu.
2024. Alignment for Honesty. InAdvances in Neural Information Pro-
cessing Systems 38: Annual Conference on Neural Information Processing
Systems 2024, NeurIPS. http://papers.nips.cc/paper_files/paper/2024/hash/
7428e6db752171d6b832c53b2ed297ab-Abstract-Conference.html
[41] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The Value of Semantic Parse Labeling for Knowledge Base
Question Answering. InProceedings of the 54th Annual Meeting of the Association
for Computational Linguistics, ACL. https://doi.org/10.18653/v1/p16-2033
[42] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Moham-
mad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying Context Ranking
with Retrieval-augmented Generation in LLMs. InAdvances in Neural Infor-
mation Processing Systems 38: Annual Conference on Neural Information Pro-
cessing Systems, NeurIPS. http://papers.nips.cc/paper_files/paper/2024/hash/
db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.html
[43] Congzhi Zhang, Linhai Zhang, Jialong Wu, Yulan He, and Deyu Zhou. 2025.
Causal prompting: Debiasing large language model prompting based on front-
door adjustment. InThirty-Ninth AAAI Conference on Artificial Intelligence, AAAI.
25842–25850. https://doi.org/10.1609/aaai.v39i24.34777
[44] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou,
Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025. A Sur-
vey of Graph Retrieval-Augmented Generation for Customized Large Language
Models.CoRRabs/2501.13958 (2025). https://doi.org/10.48550/arXiv.2501.13958
[45] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song.
2018. Variational Reasoning for Question Answering With Knowledge Graph. In
Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI).
6069–6076. https://doi.org/10.1609/aaai.v32i1.12057
[46] Yinghao Zhang, Tingting Xu, Debo Cheng, Jiuyong Li, Lin Liu, Ziqi Xu, and Zai-
wen Feng. 2025. Data-driven learning optimal K values for K-nearest neighbour
matching in causal inference.Data Mining and Knowledge Discovery39, 4 (2025),
35. https://doi.org/10.1007/s10618-025-01107-5
[47] Bo Zhao, Yinghao Zhang, Ziqi Xu, Yongli Ren, Xiuzhen Zhang, Renqiang Luo,
Zaiwen Feng, and Feng Xia. 2025. Unbiased Reasoning for Knowledge-Intensive
Tasks in Large Language Models via Conditional Front-Door Adjustment. InPro-
ceedings of the 34th ACM International Conference on Information and Knowledge
Management (CIKM). 1–11. doi:10.1145/3746252.3761103
[48] Kaitlyn Zhou, Dan Jurafsky, and Tatsunori B Hashimoto. 2023. Navigating
the Grey Area: How Expressions of Uncertainty and Overconfidence Affect
Language Models. InProceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, EMNLP. 5506–5524. https://doi.org/10.18653/v1/
2023.emnlp-main.335
[49] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong
Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2023. Large
Language Models for Information Retrieval: A Survey.CoRRabs/2308.07107
(2023). https://doi.org/10.48550/arXiv.2308.07107
A Ethical Use of Data and Informed Consent
This work uses two widely adopted benchmark datasets, MetaQA
and WebQSP, which are publicly available and have been exten-
sively used in prior research on knowledge graph question answer-
ing. Both datasets are released under open licenses for research
purposes and do not contain personally identifiable information
or sensitive data. No new data involving human participants were
collected in this work, and therefore no Institutional Review Board
(IRB) approval was required. All experiments are conducted in ac-
cordance with ACM’s Publications Policy on Research Involving
Human Participants and Subjects.B Case Study
Case Study I on WebQSP
Question:what is the capital of argentina?
Ground-truth:Buenos Aires
Step 1: Counterfactual Prompting
•initial prompt𝑡 0:
Use the provided contexts to answer the question. If the
contexts are incomplete or weak, still provide your best
possible answer. Output MUST be exactly one line in
this format: {final answer}. Do not include any other text.
Examples: {Italian}
Answer𝑎 0:argentina
•Path Quality Intervention𝑡 1:
Assume your previous answer is wrong because the qual-
ity of the referred contexts is poor. Re-select the most
relevant parts from the given contexts and regenerate
the answer using one or a few words. Output MUST be
exactly one line in this format: {final answer}. Do not
include any other text. Examples: {Italian Languages}
Answer𝑎 1:buenos aires
•Reasoning Reliability Intervention𝑡 2:
Assume your previous answer is wrong due to improper
use of the retrieved contexts. Carefully re-check the
provided contexts and regenerate the answer using one
or a few words. Output MUST be exactly one line in
this format:{final answer}. Do not include any other text.
Examples: {Italian Languages}
Answer𝑎 2:buenos aires
Step 2: Panel-based Re-scoring
You are given 3 candidate answers for the same question
from different prompts:𝑡 0:𝑎0;𝑡1:𝑎1;𝑡2:𝑎2.
Task: (1) Build a unified set of UNIQUE answers by
merging semantically identical strings across 𝑎0,𝑎1,𝑎2. (2)
Compute a GLOBAL frequency vector over the unique
answers based on how many of 𝑎0/𝑎1/𝑎2map to each
canonical answer. (3) Assign probabilities for each sce-
nario (𝑡0,𝑡1,𝑡2) to be exactly this global frequency vector
normalised by 3 (counts/3), after duplicate aggregation and
before rounding. Merging rules: ignore case, whitespace,
punctuation, and plural/singular differences; choose a
clean canonical form for answers; if multiple inputs map
to the same canonical answer, aggregate by summing
before normalisation. Probability rules: A) Let count[i] be
how many of { 𝑎0,𝑎1,𝑎2} map to answers[i]. Then for every
scenario S in { 𝑡0,𝑡1,𝑡2}, set S[i] = round(count[i]/3, 2). B)
Do not output equal probabilities across answers when
counts differ. C) Each probability ∈[0.00,1.00]; sums may
be<1.00after rounding. D) If a scenario has no plausible
answers, use zeros. Return exactly one line of strict JSON,
no wrapper, no extra text.
Canonical merge:Buenos Aires,Argentina
Counts:Buenos Aires = 2,Argentina = 1

When to Trust: A Causality-Aware Calibration Framework for Accurate KG-RAG Conference’17, July 2017, Washington, DC, USA
Verbalised Prior:
"Buenos Aires": 0.62, "Argentina": 0.38
Step 3: Accurate Answer Generation
Panel JSON:
{"answers":["buenos aires","argentina"],}
{"t_0":[0.62,0.38],}
{"t_1":[0.68,0.32],}
{"t_2":[0.80,0.20]}
Causal effect variation (CE var):
CEvar(Buenos Aires)
=max(0.62,0.68,0.80)−min(0.62,0.68,0.80)
=0.80−0.62=0.18
CEvar(Argentina)
=max(0.38,0.32,0.20)−min(0.38,0.32,0.20)
=0.38−0.20=0.18
Average causal effect ( CE):
CE(Buenos Aires)=(0.62+0.68+0.80)/3=0.70
CE(Argentina)=(0.38+0.32+0.20)/3=0.30
Causal calibration index (CCI):
CCI(Buenos Aires)=0.70×(1−0.18)=0.70×0.82=0.57
CCI(Argentina)=0.30×(1−0.18)=0.30×0.82=0.25
Final Answer:{Buenos Aires}
Case Study II on MetaQA
Question:The films that share directors with the films
[Tiresia] are written by who?
Ground-truth:Bertrand Bonello
Step 1: Counterfactual Prompting
•initial prompt𝑡 0:
Use the provided contexts to answer the question. If the
contexts are incomplete or weak, still provide your best
possible answer. Output MUST be exactly one line in
this format: {final answer}. Do not include any other text.
Examples: {Italian}
Answer𝑎 0:Luca Fazzi
•Path Quality Intervention𝑡 1:
Assume your previous answer is wrong because the qual-
ity of the referred contexts is poor. Re-select the most
relevant parts from the given contexts and regenerate
the answer using one or a few words. Output MUST be
exactly one line in this format: {final answer}. Do not
include any other text. Examples: {Italian Languages}
Answer𝑎 1:Bertrand Bonello
•Reasoning Reliability Intervention𝑡 2:
Assume your previous answer is wrong due to improper
use of the retrieved contexts. Carefully re-check theprovided contexts and regenerate the answer using one
or a few words. Output MUST be exactly one line in
this format:{final answer}. Do not include any other text.
Examples: {Italian Languages}
Answer𝑎 2:Bertrand Bonello
Step 2: Panel-based Re-scoring
You are given 3 candidate answers for the same question
from different prompts:𝑡 0:𝑎0;𝑡1:𝑎1;𝑡2:𝑎2.
Task: (1) Build a unified set of UNIQUE answers by
merging semantically identical strings across 𝑎0,𝑎1,𝑎2. (2)
Compute a GLOBAL frequency vector over the unique
answers based on how many of 𝑎0/𝑎1/𝑎2map to each
canonical answer. (3) Assign probabilities for each sce-
nario (𝑡0,𝑡1,𝑡2) to be exactly this global frequency vector
normalised by 3 (counts/3), after duplicate aggregation and
before rounding. Merging rules: ignore case, whitespace,
punctuation, and plural/singular differences; choose a
clean canonical form for answers; if multiple inputs map
to the same canonical answer, aggregate by summing
before normalisation. Probability rules: A) Let count[i] be
how many of { 𝑎0,𝑎1,𝑎2} map to answers[i]. Then for every
scenario S in { 𝑡0,𝑡1,𝑡2}, set S[i] = round(count[i]/3, 2). B)
Do not output equal probabilities across answers when
counts differ. C) Each probability ∈[0.00,1.00]; sums may
be<1.00after rounding. D) If a scenario has no plausible
answers, use zeros. Return exactly one line of strict JSON,
no wrapper, no extra text.
Canonical merge:Bertrand Bonello,Luca Fazzi
Counts:Bertrand Bonello = 2,Luca Fazzi = 1
Verbalised Prior:
"Bertrand Bonello": 0.67, "Luca Fazzi": 0.33
Step 3: Accurate Answer Generation
Panel JSON:
{"answers":["luca fazzi","bertrand bonello"],}
{"t_0":[0.33,0.67],}
{"t_1":[0.33,0.67],}
{"t_2":[0.33,0.67]}
Causal effect variation (CE var):
CEvar(Bertrand Bonello)
=max(0.67,0.67,0.67)−min(0.67,0.67,0.67)
=0.00
CEvar(Luca Fazzi)
=max(0.33,0.33,0.33)−min(0.33,0.33,0.33)
=0.00
Average causal effect ( CE):
CE(Bertrand Bonello)=(0.67+0.67+0.67)/3=0.67
CE(Luca Fazzi)=(0.33+0.33+0.33)/3=0.33

Conference’17, July 2017, Washington, DC, USA Ren et al.
Causal calibration index (CCI):
CCI(Bertrand Bonello)=0.67×(1−0.00)=0.67
CCI(Luca Fazzi)=0.33×(1−0.00)=0.33
Final Answer:{Bertrand Bonello}
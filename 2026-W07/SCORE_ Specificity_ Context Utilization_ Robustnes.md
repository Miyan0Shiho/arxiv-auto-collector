# SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation

**Authors**: Homaira Huda Shomee, Rochana Chaturvedi, Yangxinyu Xie, Tanwi Mallick

**Published**: 2026-02-10 17:39:17

**PDF URL**: [https://arxiv.org/pdf/2602.10017v1](https://arxiv.org/pdf/2602.10017v1)

## Abstract
Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications.

## Full Text


<!-- PDF content starts -->

SCORE: Specificity, Context Utilization, Robustness, and Relevance for
Reference-Free LLM Evaluation
Homaira Huda Shomee1,2Rochana Chaturvedi2Yangxinyu Xie2,3Tanwi Mallick2
1University of Illinois Chicago, Chicago, IL, USA
2Argonne National Laboratory, Lemont, IL, USA
3University of Pennsylvania, Philadelphia, PA, USA
{hshome2@uic.edu, rchaturvedi@anl.gov, xinyux@wharton.upenn.edu, tmallick@anl.gov}
Abstract
Large language models (LLMs) are increas-
ingly used to support question answering
and decision-making in high-stakes, domain-
specific settings such as natural hazard re-
sponse and infrastructure planning, where
effective answers must convey fine-grained,
decision-critical details. However, existing
evaluation frameworks for retrieval-augmented
generation (RAG) and open-ended question an-
swering primarily rely on surface-level sim-
ilarity, factual consistency, or semantic rel-
evance, and often fail to assess whether re-
sponses provide the specific information re-
quired for domain-sensitive decisions. To ad-
dress this gap, we propose a multi-dimensional,
reference-free evaluation framework that as-
sesses LLM outputs along four complemen-
tary dimensions: specificity, robustness to
paraphrasing and semantic perturbations, an-
swer relevance, and context utilization. We
introduce a curated dataset of 1,412 domain-
specific question–answer pairs spanning 40
professional roles and seven natural hazard
types to support systematic evaluation. We
further conduct human evaluation to assess
inter-annotator agreement and alignment be-
tween model outputs and human judgments,
which highlights the inherent subjectivity of
open-ended, domain-specific evaluation. Our
results show that no single metric sufficiently
captures answer quality in isolation and demon-
strate the need for structured, multi-metric eval-
uation frameworks when deploying LLMs in
high-stakes applications.
1 Introduction
LLMs have shown remarkable capabilities across
a wide range of tasks, specially in answer gener-
ation and decision support (Brown et al., 2020;
Ouyang et al., 2022; Wei et al., 2022). As these
models are increasingly deployed in high-stakes
domains such as healthcare, legal reasoning, and
climate risk assessment (Wu et al., 2025; Nagaret al., 2024; Li et al., 2024a; Luo et al., 2025; Li
et al., 2024b); the need for robust and reliable eval-
uation methods has become essential. One such
critical domain is natural hazard response, where
LLMs are used to recommend mitigation strategies
and provide factual information about disasters in
specific geographic regions (Xie et al., 2025). In
such scenarios, even minor inaccuracies can lead
to real-world consequences.
Despite this growing reliance on LLMs, eval-
uating their outputs in the above mentioned set-
tings remains challenging. Traditional reference-
based evaluation metrics are often insufficient, as
they primarily measure surface-level similarity to
gold-standard answer rather than verifying whether
generated responses contain correct and specific
information (Celikyilmaz et al., 2020; Kry ´sci´nski
et al., 2020; Chang et al., 2024). This limitation
is particularly acute in natural hazard and extreme
weather response, where publicly available gold-
standard datasets are scarce due to the diversity
of hazards, geographic regions, infrastructure sys-
tems, and professional contexts. Effective answers
in this domain must be geographically precise, and
tailored to specific concerns.
Recent work has shifted toward reference-free
evaluation, including LLM-as-a-judge pipelines,
where LLMs themselves act as evaluators. Early
frameworks like MT-Bench (Bai et al., 2024) and
AlpacaEval (Li et al., 2023) pioneered the use of
high-capacity models like GPT-4 to score multi-
turn dialogues or perform pairwise comparisons,
with later variants adjusting for verbosity bias.
Chatbot Arena (Chiang et al., 2024) leverages large-
scale human preferences to rank models via Elo
scores. While these benchmarks primarily focus
on general-purpose conversational ability, they are
not designed to capture the level of fine-grained
specificity and answer relevance required in high-
stakes applications. In parallel, the rise of Retrieval-
Augmented Generation (RAG) has motivated eval-arXiv:2602.10017v1  [cs.CL]  10 Feb 2026

uation frameworks that focus on factuality, answer
relevance, claim-level verification, and context rele-
vance or utilization (Es et al., 2024; Ru et al., 2024;
Saad-Falcon et al., 2024; Ni et al., 2024).
Although effective for general settings, exist-
ing RAG evaluation metrics fall short in domain-
specific settings. Faithfulness and factuality met-
rics assess whether answers are broadly supported
by retrieved documents, but does not verify whether
critical specific details are present and correct (Min
et al., 2023). Answer relevance metrics mostly rely
on semantic similarity or LLM-based scores, which
can overestimate quality in cases where responses
are generic rather than informative and relevant
(Liu et al., 2023). Claim-level verification focuses
on factual correctness at the level of individual
claims, but do not evaluate whether the answer
sufficiently addresses the user’s query intent or de-
cision context. Finally, for context utilization, prior
work (Ru et al., 2024) measures the claim-level
semantic overlap between retrieved knowledge and
generated answer. While this captures whether the
generated answer is supported by the retrieved in-
formation, it does not measure whether the model
actually relied on that information during gener-
ation as LLMs may produce overlapping claims
from their parametric knowledge.
To address these limitations, we propose SCORE,
a multi-perspective evaluation framework for as-
sessing LLM-generated answers in natural hazard
analysis and decision support. Our framework eval-
uates responses along five dimensions:specificity,
relevance,robustness,context utilization, andread-
ability. For specificity, we employ multiple LLM-
as-a-judge setup to verify whether key details (e.g.,
hazard type, location, timeline, intensity) are ex-
plicitly stated and supported by retrieved evidence,
and aggregate the judges’ decisions into a final
score. We assess robustness by applying paraphras-
ing and controlled perturbations to the question.
By measuring the consistency of the model’s re-
sponses across these variations, we detect over-
sensitivity in model behavior. For answer rele-
vance, we adopt a question-regeneration strategy
with an additional semantic masking step that re-
moves domain-specific entities. Given the use of
retrieval-augmented generation, we evaluate con-
text utilization by measuring whether and how re-
trieved passages contribute to the generated answer.
Our metric measures evidence utilization at a finer
granularity through counterfactual dependence by
removing individual claims and quantifying theresulting change in answer confidence, enabling
us to distinguish necessary evidence from merely
consistent, ignored, or distracting context. Finally,
we incorporate readability metric using established
grade-level metrics (Appendix B.3) to ensure that
responses are appropriate for professional users.
In this work, we present a reference-free evalua-
tion framework SCOREfor assessing LLMs outputs
in high-stakes hazard response. Our key contribu-
tions are as follows: (1) We construct a synthetic
dataset of 1412 domain-specific question–answer
pairs that cover different hazards, locations, infras-
tructure types, and concerns to support controlled
and scalable evaluation; (2) We design a multi-
dimensional, reference-free evaluation framework
that assesses specificity, robustness, answer rele-
vance, and context utilization, and validate these
metrics through human evaluation.
Code & Data:Our code and curated
data are available at https://anonymous.4open.
science/r/score-8D5F/readme.md
2 Dataset Construction
We construct the largest dataset to date of
infrastructure-related questions that simulate real-
world hazard scenarios and decision-making con-
texts. Recent work by Xie et al. (2025) demon-
strates the effectiveness of agentic frameworks for
addressing critical infrastructure risks by incor-
porating structured user profiles that encode pro-
fessional background and user concerns. While
their work focuses on a conversational retrieval-
augmented generation (RAG) system for wildfire
risk mitigation, we extend similar role-aware de-
sign principles to develop a comprehensive eval-
uation benchmark that systematically assesses
whether LLMs generate contextually appropriate
responses across a broader range of infrastructure
sectors, hazard types, and geographic settings.
User Profile.Each question-answer is asso-
ciated with a synthetic user profile comprising
profession,concern(fact-based, recommendation-
seeking, or hybrid),location, andtimeline. Profes-
sion and concern characterize a user’s role and ob-
jective, while location and timeline capture the ge-
ographic scope and time frame. For each question,
profile attributes are instantiated through controlled
randomization: first a hazard type is sampled, then
a location is sampled conditional on the hazard,
then the remaining attributes (profession, and time-
line) are randomly selected from predefined sets.

Professional Context.To ground questions in
real-world professional contexts, we curate a di-
verse set of 40 professions spanning five critical in-
frastructure sectors:transportation,water,energy,
buildings, andcommunication(Appendix Table 6).
Each corresponds to roles that are actively involved
in hazard mitigation and infrastructure planning,
such as, transportation planner, hydraulic engineer,
energy storage specialist, building systems man-
ager, and telecommunications engineer.
Location and Hazard Type.We select the lo-
cation and hazard types based on realistic hazard-
region mappings that frequently occur within the
U.S. context. We identify a set of seven hazard
types:cold wave,heat wave,coastal flooding,
ice storm,hurricane,drought, andwildfire. Each
hazard is then associated with a curated list of
U.S. locations that are historically or geographi-
cally susceptible to it, based on Natural Risk In-
dex(Appendix Table 7).1For instance, hurricanes
are commonly associated with Miami or New Or-
leans, while wildfire scenarios with San Diego or
Elko, Nevada. This helps ensure that the generated
questions reflect regionally relevant risks and real-
world challenges faced by infrastructure planners.
Question Generation.Each template has place-
holders for infrastructure, hazard, location, con-
cern, and timeline, populated using user profile at-
tributes. To create a diverse set of questions, we in-
clude three concerns: fact-based, recommendation-
seeking, and hybrid. Fact-based templates fo-
cus on objective inquiries (e.g., “What are the
critical vulnerabilities of [INFRASTRUCTURE]
to [HAZARD] in [LOCATION]?”). In contrast,
recommendation-seeking templates are designed to
elicit forward-looking or strategic guidance (e.g.,
“What strategies should [PROFESSION] consider
to enhance [INFRASTRUCTURE] resilience in
the face of [HAZARD] over the next [TIMELINE]
years?”). Hybrid templates focus on infrastructure
interdependencies and cascading impacts across
sectors (e.g., “What are the cascading impacts of
[HAZARD] between [INFRASTRUCTURE] and
building systems in [LOCATION]?”). During gen-
eration, the concern type in the user profile deter-
mines the template pool, from which one template
is randomly selected.
Answer Generation.For each question, we re-
trieve the top five semantically relevant abstracts
1https://hazards.fema.gov/nri/data-resources
Aggregator Specificity 
Agent 1 
Specificity 
Score Specific Info 
Extraction 
Answer 
Claim 
Specificity 
Agent 2 
Specificity 
Agent 3 
Decomposition 
Figure 1: Specificity score computation: each gener-
ated answer is decomposed into atomic claims, specific
details (hazard type, location, timeline, intensity) are
extracted, and each claim is evaluated using multiple
LLM-based agents before aggregating their judgments.
from a domain-specific knowledge base (Mallick
et al., 2023, 2025). These abstracts, along with
the user profile and query, are incorporated into
a structured prompt that is passed to an LLM to
generate a concise, point-wise answer grounded in
the retrieved literature (Appendix E.1).
3 Evaluation Framework
A major contribution of our work is the design
of a reference-free, multi-dimensional evaluation
framework for assessing LLM-generated answers
in high-stakes, domain-specific settings. To rig-
orously evaluate the responses generated in Sec-
tion 2, our framework measures the answer quality
across four fine-grained dimensions:specificity,
robustness,answer relevance, andcontext uti-
lization. Additionally, we assesreadabilityusing
established metrics to ensure that the outputs are
suitable for professional use (Appendix B.3).
3.1 Specificity
We propose a multi-agent evaluation framework,
illustrated in Figure 1, to measure the speci-
ficity of LLM-generated answers by assessing
whether fine-grained details (e.g., hazard type,
location, timeline, and intensity) are explicitly
stated and verifiable against the retrieved knowl-
edge. Formally, given the set of atomic claims
C:={c 1, c2, . . . , c |C|} extracted from gener-
ated answers, and specificity dimensions D:=
{dh, dl, dt, di}representinghazard type,location,
timeline, andintensity. An evaluator agent kas-
signs a label ℓ(k)
idj∈ {yes,no,n/a} to each claim
ci, along each dimension. Here, yes indicates that

the specific detail is both present in the claim and
verifiable using the knowledge source; no indicates
a lack of support or contradiction; and n/a denotes
that the detail is not mentioned in the claim. We
map these categorical labels to numerical values:
s(k)
idj=

1,ifℓ(k)
idj=“yes”,
0,ifℓ(k)
idj=“no”,
n/a,ifℓ(k)
idj=“n/a”.
For each claim ciand dimension dj, we do majority
voting acrosskevaluators, to obtain:
si=
sidh, sidl, sidt, sidi
.
For each dimension dj, we average the consensus
sidjacross claims where dj̸=n/a . This prevents
infrequently extracted dimensions (e.g., intensity)
from being penalized by averaging over all claims:
¯sdj=P
i∈C:s idj̸=n/asidjP
i∈C:s idj̸=n/a1
The final specificity score is a weighted combina-
tion of dimension-level averages:
Specificity(C) =P
j: ¯sdjαj¯sdjP
j: ¯sdjαj(1)
where, α= [0.6,0.2,0.1,0.1] corresponds to
the ordered dimensions:hazard,location,timeline,
andintensity. If a dimension is “n/a” across claims,
it is excluded from both the numerator and the de-
nominator in the weighted average. In our domain,
the four specificity dimensions do not contribute
equally to assessing the quality of a claim. We
place higher emphasis on the hazard and location
dimensions, which are most critical for grounding
claims in real-world risk contexts.2
3.2 Robustness
This metric assesses the stability of the generated
answers by evaluating their consistency under con-
trolled input variations (Figure 2). We examine
whether paraphrasing the original question leads to
a similar response, and whether intentional changes
to key parameters, such as hazard type and/or lo-
cation, generate appropriately different responses.
We consider two types of input modifications:
2Appendix B.1 provides additional details on claim and
specific information extraction, and evaluator prompts.
Question Answer 
Paraphrased 
Question 
Hazard/Location 
Perturbed 
Question RAG Pipeline 
Paraphrased 
Answer 
Perturbed 
Answer 
Figure 2: Robustness evaluation workflow. For each
question–answer pair, the system generates paraphrased
and hazard/location-perturbed variants of the question,
runs them through the RAG pipeline, and compares the
generated answers to assess semantic consistency and
sensitivity to controlled perturbations.
Paraphrasing:We rephrase the original question
while preserving its semantic meaning. A robust
system should generate answers that are semanti-
cally consistent with the original response.
Perturbation:We modify the question by alter-
ing the hazard type and/or the geographical location
mentioned. In this setting, we expect the system to
retrieve a different set of relevant documents and
produce correspondingly different answers.
3.3 Answer Relevance
Answer relevance evaluates whether a model’s re-
sponse is relevant to the original user query (Saad-
Falcon et al., 2024; Es et al., 2024). We adapt the
metric from RAGAS framework (Es et al., 2024)
and extend it by incorporating a masking step to re-
duce relevance arising from domain-specific lexical
overlap. Given a generated answer a, we prompt
an LLM to generate five candidate questions for
which awould be an appropriate answer. This
treats LLMs as an inverse question generators. For
each generated question, we compute the cosine
similarity between its embedding and the embed-
ding of the original question q. The final Answer
relevance score Ris defined as the average similar-
ity. Figure 3 shows the pipeline.
Answer Relevance with Masking.High rel-
evance scores can result from superficial lexical
overlap, especially when domain-specific keywords
(e.g. specific hazard or infrastructure sector) ap-
pear in both the question and answer. To mitigate
this, we apply a semantic masking step: an LLM
first generates a masked answer ˜aby replacing sen-
sitive tokens with a placeholder (e.g., replacing

Answer Masked Answer 
Question Masking 
Generated 
Question 
BGE Reranker 
Reranked 
Answer 
Figure 3: Answer relevance pipeline. For each answer,
the system generates reranked answer and answer rele-
vance score. Outputs are shown inside blue boxes.
“electrical grid” with [INFRASTRUCTURE]), then
generates ˜qfrom ˜aensuring conceptual rather than
lexical alignment. The masked relevance ( ˜R) score
is:
˜R(q, a) =1
nnX
i=1sim (q,˜q)(2)
Relevance Scoring with BGE Reranker.To com-
plement masked relevance, we apply a post-hoc
reranking strategy to assess and enhance the in-
formativeness of multi-point answers. We use a
leave-one-out relevance attribution method, where
each segment of the answer (e.g., bullet point or
paragraph) is evaluated for its marginal contri-
bution to the overall answer quality. We utilize
theBAAI/bge-reranker-base model3, which pro-
vides a dense relevance score R(q, a) between a
question Qand an answer A. A structured answer
is usually composed of an optional introduction and
mcontent segments a:=Intro∪{a 1, a2, . . . , a m}.
To estimate the marginal relevance of each individ-
ual segment ai, we remove it from the answer to
geta−iand compute the difference:
∆i=R(q, a)−R(q, A −i)(3)
A higher ∆iindicates a more significant contribu-
tion of that segment to the overall answer quality.
If∆i<0, it implies that the removed segment was
detrimental to the overall relevance.
Finally, we rerank the answer segments {ai}
in descending order of ∆i, prepending the intro-
duction (if present). This ensures that the most
informative and relevant points appear earlier in
the final answer presentation.
3https://bge-model.com/tutorial/5_Reranking/5.
2.html3.4 Context-Utilization
We measure context utilization as the degree to
which each unique claim in the retrieved context
contributes to the model’s final answer. Given a
query qand an answer a, we first extract a set of
unique claims C:={c 1, c2, ...c|c|}from the re-
trieved documents D. To quantify the contribution
of each claim ci∈C we adopt a leave-one-out un-
certainty–based estimator that measures the change
in answer confidence when ciis removed from the
context. Formally, we define the absolute contribu-
tion of claim cias the drop in model’s confidence
in generating the answera:
∆i(a, q, C, θ) =f θ(a|q∪C)−f θ(a|q∪C\{c i})
where fθ(a|.) denotes the confidence score of lan-
guage model in generating aconditioned on q, C.
We further define relative context utilization as:
δi(a, q, C, θ) = ∆ i(a, q, C, θ)/f θ(a|q∪C)
Confidence Estimation.We compute answer con-
fidence using the normalized per-token probability
of the generated sequence (Murray and Chiang,
2018). Specifically, confidence is defined as the
geometric mean of token probabilities:
fθ(a|X) =e1
TPT
t=1log(P(y t|X,y<t))(4)
Where a:= (y 1, y2, ...y T)denotes the answer
token sequence, and Xdenote the conditioning
context ( q∪C ). This formulation avoids length bias
and numerical instability associated with raw prob-
ability multiplication.4Using this, ∆i∈(−1,1) ,
with larger positive values indicating greater re-
liance on claim ci.δi∈(−∞,1) , where positive
values indicate that the claim contributes positively
to the answer, while negative values suggest that
the claim is distracting or detrimental.
Finally, we aggregate claim-level contributions
to obtain a global context utilization score that sum-
marizes how effectively the retrieved context is
used by the model when producing the answer. It
is defined as the average ( CU) or relative ( CUrel)
contribution across all claims:
CU=1
|C|X
i∈|C|∆i;CU rel=1
|C|X
i∈|C|δi
4We also consider token-level predictive entropy.

Generator Evaluator Specificity↑Robustness Answer Relevance↑
Para.↑Pert.↓
GPT-4oGPT-4o0.61±0.21 0.89±0.05 0.60±0.09 0.69±0.11
Qwen3-8B0.58±0.22 0.89±0.04 0.66±0.10 0.73±0.18
Gemini2.5GPT-4o0.55±0.24 0.87±0.07 0.87±0.07 0.68±0.1
Qwen3-8B0.45±0.27 0.86±0.05 0.66±0.10 0.83±0.11
Table 1: Evaluation results for two different generators (GPT-4o and Gemini2.5) with mean and standard deviation
of all measures for each generator-evaluator pair. Higher values indicate better performance for all metrics except
perturbation robustness, where lower values show higher sensitivity to changes. Overall, GPT-4o produces more
detailed and specific responses than Gemini 2.5. The evaluation results indicate that GPT-4o acts as a more lenient
judge than Qwen3-8B.
4 Experimental Setup
In our experimental setup, LLMs are used as (i)
answer generators within a RAG pipeline, and (ii)
evaluators for the proposed metrics. We employ
GPT-4o and Gemini 2.5 as the primary answer gen-
erators. These models use the RAG pipeline de-
scribed, where the top five semantically relevant
abstracts from our domain-specific knowledge base
serve as grounding per query. For the evaluation
phase, we use both proprietary and open-source
models to show the framework’s flexibility across
resource tiers. Specifically, we employ GPT-4o as
a commercial evaluator and Qwen3-8B as a repre-
sentative open-source evaluator. We assess answer
relevance by incorporating the BAAI/bge-reranker-
base model to compute relevance scores for multi-
point answers. Proprietary models like GPT-4o and
Gemini do not expose token-level logits, needed to
estimate CU score. We decouple answer generation
and confidence estimation using teacher-forcing
on lightweight models. We use LLaMA3.1-8B-
Instruct, Qwen2.5-7B-Instruct, and Qwen3-8B as
proxies. We also experiment with Gemma2-9B-
it and Ministral-8B-Instruct. See Appendix B.2
for details. Detailed technical specifications are
reported in Appendix D.
4.1 Human Evaluation Setup
We recruit 10 annotators and assign each 10 ques-
tion–answer pairs through our dedicated annota-
tion platform. To ensure reliability and measure
inter-annotator agreement, every question is inde-
pendently evaluated by two annotators. In total,
the study covers 50 unique question–answer pairs.
The annotation guidelines, platform details, and
interface screenshots are provided in Appendix C.5 Results
We present results based on our proposed SCORE
framework and our dataset across multiple LLMs.
5.1 Evaluation with SCORE Metrics
5.1.1 Specificity
Table 1 shows the specificity scores across
generator-evaluator pairs (we use k= 3 evalua-
tor agents). Across all settings, GPT-4o answers
achieve higher specificity scores than Gemini, re-
gardless of the evaluator. We also observe that
specificity exhibits higher variance than other eval-
uation metrics. The higher variance in specificity
scores suggests that model behavior is not uniform
across questions. While GPT-4o often produces
highly detailed answers, it occasionally generates
less precise responses depending on the query.
This variability reinforces the motivation for fine-
grained specificity evaluation, as such detailed in-
formation is inherently difficult for models to pro-
vide accurately across diverse queries.
5.1.2 Robustness
We evaluate robustness under paraphrasing invari-
ance and semantic perturbations. As shown in Ta-
ble 1, both GPT-4o and Gemini exhibit high robust-
ness to paraphrasing across all evaluator settings,
which indicates that superficial linguistic changes
do not significantly affect model outputs.
In contrast, semantic perturbations that alter key
attributes such as hazard type or location lead to
noticeable consistency drops. This behavior is de-
sirable, as it indicates that models appropriately
adjust their responses when the underlying facts
change. Gemini shows higher perturbation ro-
bustness scores, which suggests that its responses
change less under semantic modifications. This
likely reflects a tendency toward more generic an-
swers, which remain broadly applicable even when

critical details are altered, whereas GPT-4o pro-
duces more detail-sensitive responses.
5.1.3 Answer Relevance
To evaluate how well responses address user
queries, we use a dual strategy combining masked
answer relevance and BGE reranking. As shown in
Table 1, GPT-4o scored between 0.69 – 0.73, while
Gemini 2.5 scores between 0.68 – 0.83. Addition-
ally, using BGE reranker, we measure the value
of each individual point in an answer by seeing
how much the total score drops when that point
is removed. Table 2 shows that Gemini achieves
higher relevance under this metric. We also ob-
serve frequent reordering of answer points after
reranking. It indicates that the relevance model
meaningfully reweights the importance of individ-
ual answer points.
Higher relevance scores do not necessarily imply
better answer quality: relevance captures seman-
tic alignment with the query, whereas specificity
reflects the presence of specific details. This moti-
vates evaluating both metrics jointly.
Metric GPT-4o Gemini
Average BGE score 6.69 7.08
Has answer changed (yes) 1393 1397
Table 2: Comparison of BGE reranking for answers
generated by GPT-4o and Gemini.
5.1.4 Context-Utilization
To test whether CU scores are sensitive to the
choice of proxy model or the confidence measures,
we report a sensitivity analysis in Table 3. LLaMA
and Qwen variants exhibit consistent answer con-
fidence when conditioned on retrieved documents
Dvs. claims Cderived from it. This reflects con-
sistency in the estimates as the claims Care sim-
ply a summary of D. In contrast, Gemma and
Ministral show less reliable correlations. An ex-
tended analysis is reported in Appendix B.2 where
LLaMA and Qwen variants also show near-perfect
agreement in their CU scores and low agreement
with Gemma and Ministral. Additionally, length-
normalized geometric mean of token probabilities
provides consistent confidence estimates than pre-
dictive entropy across models and generators (Ap-
pendix Table 8). Therefore, in Table 4 we report
results with LLaMa and Qwen as evaluators using
geometric-mean-based interpretation of fθ. Both
CUandCUrel(in terms of percentage drop) scores
are reported. The mean CUvalues are small andEvaluatorρ(f θ(a|q∪C), f θ(a|q∪D)
Generator=GPT-4o Generator=Gemini2.5
LlaMA3.1-8B-Instruct0.83 (0.0) 0.84 (0.0)
Gemma2-9B-it0.17(4.4e-11) 0.12 (1.2e-05)
Ministral-8B-Instruct0.14 (1.7e-07) 0.13 (6.0e-07)
Qwen2.5-7B-Instruct0.95 (0.0) 0.96 (0.0)
Qwen3-8B0.94 (0.0) 0.93 (0.0)
Table 3: Correlation between answer confidence condi-
tioned on query qand all claims Cvs.qand all docu-
ments Dacross evaluator models (p-values are reported
in parenthesis). Confidence fθis measured in terms of
geometric mean of token probabilities. Correlations for
Gemma and Ministral are insignificant, indicating their
insuitability for CUevaluation. In contrast, LlaMA and
Qwen have significantly high positive correlation.
positive, indicating that, on average, removing in-
dividual claims leads to a modest but consistent
drop in answer confidence. Negative minimum
values reflect cases where removing a claim in-
creases confidence, indicating distracting or redun-
dant context—a behavior expected in realistic re-
trieval settings. Importantly, these negatives are
rare and do not affect the overall positive mean,
implying that most retrieved claims contribute con-
structively. Relative CU values provide a more
interpretable signal. Mean CUrelvalues indicate
that on average, removing a single claim reduces
answer confidence by 1–2%, with a small subset of
claims causing substantially larger drops (9-10%),
indicating concentrated evidence dependence amid
otherwise distributed evidence usage. Large neg-
ative values (e.g., -8.56%) highlight claims that
harm generation. This demonstrates that the mea-
sure is effective at surfacing both highly supportive
and detrimental evidence.
5.2 Human Evaluation
We conduct a human evaluation comprising two
analyses. First, we measure inter-annotator agree-
ment to assess the consistency of human labels.
Second, we show the agreement between human
labels and automated evaluators (GPT and Qwen)
to understand how the model-based scores aligns
with human judgment. Given the open-ended and
domain-specific nature of the task, some disagree-
ment is expected, particularly for attributes that
require interpretation rather than direct verification.
Inter-Annotator Agreement.We compute
inter-annotator agreement across the 50 double-
annotated questions. Table 5 shows agreement per-
centages (see Table 11 for details). Agreement is
higher for concrete attributes such as hazard and

Generator EvaluatorCU CU%
rel
min max mean min max mean
GPT-4oLLaMA3.1-8B-Instruct -0.001 0.024 0.004 (0.002) -0.17 9.82 1.63 (0.87)
Qwen3-8B 0.000 0.019 0.004 (0.002) -0.51 9.63 1.74 (0.94)
Gemini2.5LLaMA3.1-8B-Instruct -0.001 0.014 0.003(0.002) -8.56 6.45 1.34 (0.83)
Qwen3-8B -0.001 0.015 0.003 (0.002) -0.74 7.35 1.39 (0.87)
Table 4: CUscores across two generators and two evaluators (standard deviation is in parentheses), showing on
average modest but positive contributions from individual claims to answer confidence, with rare distracting claims.
Relatively larger max values indicate concentrated evidence dependence.
location, and lower for subjective dimensions such
as timeline, which often requires interpretation of
vague temporal statements. For continuous metrics,
we evaluate agreement using Spearman correla-
tion. We obtain a Spearman correlation of 0.22 for
answer relevance and 0.35 for context utilization,
highlighting that both tasks are inherently subjec-
tive and cognitively demanding, with context uti-
lization being slightly more stable than relevance.
Human vs. Automated Agreement.We com-
pare human annotations with two automatic eval-
uators using row-level exact match on the same
label ranges. Table 5 shows that alignment is gen-
erally higher for hazard and location, but drops for
timeline and intensity (see Table 12 for details).
We also notice that many human–automated mis-
matches occur on the same questions where anno-
tators themselves disagree, which suggests that a
large fraction of the errors come from genuinely
ambiguous or subjective cases. For answer rele-
vance, the Spearman correlations are 0.33 (human
vs. GPT) and 0.32 (human vs. Qwen); for context
utilization, they are 0.43 (for both human vs. Qwen
and human vs. LLaMA). Matching or exceeding
human–human agreement, our automated metrics
appear as reasonable human-aligned proxies.
Error AnalysisWe conduct a qualitative error
analysis to understand common failure modes and
sources of disagreement. For specificity-hazard,
some answers discuss multiple hazards (e.g., ice
storms, storm surge, sea level rise) in the same
response. In these cases, some annotators score
hazard specificity only with respect to the haz-
ard explicitly asked in the question, while others
treat any hazard details mentioned in the answer
as sufficient. For specificity-timeline, we observe
boundary cases where the answer mentions a time-
line and the retrieved sources also include tempo-
ral information, but the timelines do not match.Metric Annotators Anno-GPT Anno-Qwen
Hazard92 88 88
Location86 77 73
Timeline52 45 40
Intensity78 75 59
Table 5: Agreement percentage between annotators,
annotator and automated metrics for specificity.
In one example, the answer claims impacts over
the “next 30–50 years,” but that exact range is not
stated in the retrieved evidence. The human annota-
tors markedYesbecause a timeline was mentioned,
GPT markedN/Adue to vague temporal framing,
and Qwen markedNobecause the “30–50 years”
claim was not verifiable from the sources. Addi-
tional examples are provided in Appendix C.
6 Conclusion
We introduce SCORE, a multi-dimensional evalu-
ation framework for assessing LLM-generated an-
swers in domain-specific, high-stakes settings, to-
gether with a new dataset designed to support eval-
uation where gold-standard references are scarce.
By jointly analyzing specificity, robustness, an-
swer relevance, and context utilization, we show
that models exhibit distinct trade-offs between pro-
ducing detailed, precise information and generat-
ing broadly relevant but more generic responses.
Our human evaluation highlights the challenges
of aligning automated metrics with human judg-
ment in open-ended, expert-level tasks, particularly
for attributes requiring significant interpretation.
These results emphasize the importance of evalu-
ating LLM outputs using complementary metrics
rather than a single score for a more reliable as-
sessment of LLMs deployed in real-world decision
support.

Ethical Considerations
This work focuses on the evaluation of large lan-
guage model (LLM) outputs rather than the deploy-
ment of automated decision-making systems. Al-
though our framework targets high-stakes domains
such as natural hazard response and infrastructure
planning, we emphasize that such systems should
complement rather than replace human expert judg-
ment, particularly in decisions affecting public
safety and critical infrastructure. Our dataset is
synthetically generated and contains no personal or
personally identifiable information; however, while
it is designed to reflect realistic hazard–location
mappings, it may not capture the full complexity
and diversity of real-world hazard scenarios, po-
tentially disadvantaging regions or communities
underrepresented in the climate and infrastructure
literature. We also acknowledge that evaluation
metrics themselves can introduce ethical risks if
misinterpreted, as high scores on relevance, robust-
ness, or specificity may create false confidence if
treated as definitive indicators of system safety or
deployment readiness rather than complementary
assessment tools. Finally, all human annotators
involved in this study were recruited and treated in
accordance with institutional review board (IRB)
guidelines, provided with clear information about
the research purpose, participated voluntarily, and
were compensated fairly for their time.
Limitations
The current framework relies on a specific knowl-
edge base of approximately 600,000 climate-
related records. A primary limitation is that if this
knowledge base is insufficient or lacks coverage
for a specific niche, the system may retrieve irrele-
vant documents. This can lead to error propagation
where the model generates an answer based on poor
grounding. In addition, human evaluation of open-
ended, domain-specific questions exhibits inherent
subjectivity, particularly for attributes that requires
interpretation such as timeline and intensity.
Acknowledgment
We thank Sourav Medya for his valuable feedback
on the manuscript. This work was supported by
UChicago Argonne, LLC, Operator of Argonne
National Laboratory (“Argonne”). Argonne’s work
was supported by the U.S. Department of Energy,
Grid Deployment Office, under contract DE-AC02-
06CH11357. This research used resources of theArgonne Leadership Computing Facility at Ar-
gonne National Laboratory, which is supported by
the Office of Science of the U.S. Department of
Energy, Office of Science, under contract number
DE-AC02-06CH11357.
References
Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jia-
heng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su,
Tiezheng Ge, Bo Zheng, et al. 2024. Mt-bench-101:
A fine-grained benchmark for evaluating large lan-
guage models in multi-turn dialogues.arXiv preprint
arXiv:2402.14762.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners.Advances in neural information processing
systems, 33:1877–1901.
Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao.
2020. Evaluation of text generation: A survey.arXiv
preprint arXiv:2006.14799.
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu,
Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi,
Cunxiang Wang, Yidong Wang, et al. 2024. A sur-
vey on evaluation of large language models.ACM
transactions on intelligent systems and technology,
15(3):1–45.
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anasta-
sios Nikolas Angelopoulos, Tianle Li, Dacheng Li,
Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E
Gonzalez, et al. 2024. Chatbot arena: An open
platform for evaluating llms by human preference.
InForty-first International Conference on Machine
Learning.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158.
Rudolph Flesch. 1948. A new readability yardstick.
Journal of applied psychology, 32(3):221.
J Peter Kincaid, Robert P Fishburne Jr, Richard L
Rogers, and Brad S Chissom. 1975. Derivation of
new readability formulas (automated readability in-
dex, fog count and flesch reading ease formula) for
navy enlisted personnel. Technical report.
Wojciech Kry ´sci´nski, Bryan McCann, Caiming Xiong,
and Richard Socher. 2020. Evaluating the factual
consistency of abstractive text summarization. In
Proceedings of the 2020 conference on empirical
methods in natural language processing (EMNLP),
pages 9332–9346.

Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei
Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan,
Yiran Hu, et al. 2024a. Legalagentbench: Evalu-
ating llm agents in legal domain.arXiv preprint
arXiv:2412.17259.
Haobo Li, Zhaowei Wang, Jiachen Wang, Alexis
Kai Hon Lau, and Huamin Qu. 2024b. Cllmate:
A multimodal llm for weather and climate events
forecasting.arXiv preprint arXiv:2409.19058.
Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori,
Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and
Tatsunori B. Hashimoto. 2023. Alpacaeval: An au-
tomatic evaluator of instruction-following models.
https://github.com/tatsu-lab/alpaca_eval.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
Nlg evaluation using gpt-4 with better human align-
ment.arXiv preprint arXiv:2303.16634.
Kangcheng Luo, Quzhe Huang, Cong Jiang, and Yan-
song Feng. 2025. Automating legal interpretation
with llms: Retrieval, generation, and evaluation.
arXiv preprint arXiv:2501.01743.
Tanwi Mallick, Joshua David Bergerson, Duane R
Verner, John K Hutchison, Leslie-Anne Levy, and
Prasanna Balaprakash. 2023. Analyzing the impact
of climate change on critical infrastructure from the
scientific literature: A weakly supervised nlp ap-
proach.arXiv preprint arXiv:2302.01887.
Tanwi Mallick, Joshua David Bergerson, Duane R
Verner, John K Hutchison, Leslie-Anne Levy, and
Prasanna Balaprakash. 2025. Understanding the
impact of climate change on critical infrastructure
through nlp analysis of scientific literature.Sustain-
able and Resilient Infrastructure, 10(1):22–39.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. Factscore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100.
Kenton Murray and David Chiang. 2018. Correcting
length bias in neural machine translation. InProceed-
ings of the Third Conference on Machine Translation:
Research Papers, pages 212–223.
Aishik Nagar, Yutong Liu, Andy T Liu, Viktor
Schlegel, Vijay Prakash Dwivedi, Arun-Kumar
Kaliya-Perumal, Guna Pratheep Kalanchiam, Yili
Tang, and Robby T Tan. 2024. umedsum: A unified
framework for advancing medical abstractive sum-
marization.arXiv preprint arXiv:2408.12095.
Jingwei Ni, Minjing Shi, Dominik Stammbach, Mrin-
maya Sachan, Elliott Ash, and Markus Leippold.
2024. Afacta: Assisting the annotation of factualclaim detection with reliable llm annotators. InPro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 1890–1912.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
2022. Training language models to follow instruc-
tions with human feedback.Advances in neural in-
formation processing systems, 35:27730–27744.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunx-
iang Wang, Shichao Sun, Huanyu Li, et al. 2024.
Ragchecker: A fine-grained framework for diagnos-
ing retrieval-augmented generation.Advances in
Neural Information Processing Systems, 37:21999–
22027.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel,
Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler, et al.
2022. Emergent abilities of large language models.
arXiv preprint arXiv:2206.07682.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, Yueming Jin, and Vicente
Grau. 2025. Medical graph rag: Evidence-based
medical large language model via graph retrieval-
augmented generation. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 28443–
28467.
Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua
Bergerson, John K Hutchison, Duane R Verner, Jor-
dan Branham, M Ross Alexander, Robert B Ross,
Yan Feng, et al. 2025. Marsha: multi-agent rag
system for hazard adaptation.npj Climate Action,
4(1):70.

A Dataset Details
Here we provide details of the dataset generation.
A.1 Profession
Table 6 provides a summary 40 different profes-
sions across five sectors.
A.2 Hazard and Location
Table 7 summarizes the seven hazard types and
their corresponding locations (counties and states)
used in the question generation template.
B Evaluation Framework
In this section, we provide additional details on our
evaluation framework.
B.1 Specificity
Atomic Claim Decomposition.We begin by de-
composing each generated answer into a set of
atomic claims, where each claim captures a single,
self-contained factual statement. This decompo-
sition enables us to evaluate the output at a more
granular level, rather than treating the entire answer
as one unit. By focusing on individual claims, we
can perform more precise and targeted assessments
of specificity. We can also identify whether each
claim includes the fine-grained details that make it
informative.
Specific Information Extraction.After decom-
posing each generated answer into claims, we ex-
tract the specific information that can be verified
against the retrieved documents. In particular, we
focus on four dimensions of specificity:hazard
type,location,timeline, andintensity. To accom-
plish this, we prompt an LLM to identify and
extract all explicit mentions of these dimensions
from the claims. The extracted information is then
passed to our specificity evaluation agents in the
following step for verification and scoring.
Specificity Scoring.We evaluate each claim to
determine whether it includes specific factual de-
tails that are supported by the knowledge base. The
goal of this step is not to assess whether a claim
is factually correct, but to determine whether fine
grained information such as hazard type, location,
timeline, or intensity is verified based on the knowl-
edge base. We denote the input to the evaluation
process as a triplet: (S, c i, K), where Sis specific
information, ciis the i-th atomic claim extracted
from the generated answer, K is the corresponding
knowledge base (e.g., abstracts from the literature).This triplet is provided to a language model evalua-
tor agent, which is prompted to assess whether the
claim ciexpresses concrete information that isver-
ifiable using the knowledge base K, with respect
to four pre-defined specificity dimensions:
D={hazard type,location,timeline,intensity}
Each agent produces a structured annotation for the
claim across the four specificity dimensions from
a fixed label set: L={yes, no, n/a} . A label
of “yes” indicates that the specific information is
present in the claim and is also verifiable using the
knowledge base. A label of “no” indicates that the
information is not supported by the knowledge.
Finally, “n/a” is used when a particular dimension
does not apply to the claim’s context (e.g., that
specific information is not mentioned in the claim).
B.2 Context Utilization: Proxy-Based Scoring
We operate at the level of atomic claims rather than
retrieved chunks. Claims are extracted from re-
trieved context using GPT-4o, prompted to identify
unique, non-redundant factual propositions. This
claim-level representation enables fine-grained at-
tribution while avoiding redundancy within re-
trieved passages. To quantify each claim’s con-
tribution, we require an estimate of the model’s
confidence in generating a fixed answer given a
particular set of claims. However, answers are
often produced by closed or proprietary models
(e.g., GPT or Gemini APIs) that do not expose
token-level likelihoods. To enable reproducible
and model-agnostic evaluation, we therefore decou-
ple answer generation from confidence estimation.
Concretely, for a fixed answer a, we compute confi-
dence using teacher forcing on a lightweight open-
source proxy language model. The proxy model
is conditioned on the query qand context X, and
token-level probabilities are estimated by forcing
the model to follow the reference answer sequence.
This yields a consistent likelihood estimate without
requiring access to the original model’s internal
probabilities.
Sensitivity Analysis of Proxy Models.A natural
concern is whether the CU scores are sensitive to
the choice of proxy models or confidence measures.
To address this, we conduct a sensitivity analysis
and measure:
1.Confidence Consistencyas the correlation
ρ(fθ(a|q∪C), f θ(a|q∪D)) between answer
confidence given query qand all claims Cand
qand the original retrieved documentsD.

Sector Profession
Transportation Highway Engineer, Bridge Inspector, Railway Systems Engineer, Transit Operations Manager,
Airport Infrastructure Manager, Port Facility Manager, Transportation Safety Inspector, Traffic
Systems Engineer, Pavement Engineer, Transportation Planner
Water Water Systems Engineer, Hydraulic Engineer, Dam Safety Inspector, Wastewater Treatment
Specialist, Maritime Infrastructure Manager, Stormwater Engineer, Water Quality Specialist,
Coastal Infrastructure Engineer
Energy Power Systems Engineer, Electrical Grid Manager; Energy Distribution Specialist, EV Infras-
tructure Planner, Renewable Energy Systems Manager, Transmission Line Engineer, Substation
Engineer, Energy Storage Specialist
Buildings Structural Engineer, Building Systems Manager, Facilities Manager, Real Estate Asset Manager,
Building Automation Specialist, Construction Manager, Building Code Inspector, MEP Systems
Engineer
Communications Telecommunications Engineer, Broadband Infrastructure Specialist, Network Resilience Man-
ager, Data Center Infrastructure Engineer, Fiber Optics Specialist, Communications Systems
Planner, Network Security Engineer
Table 6: List of professions across five critical infrastructure sectors (Transportation, Water, Energy, Buildings, and
Communications) used for generating synthetic user profiles in our dataset.
2.CU Agreement: correlation in CU scores
across proxy models.
Table 8 reports Confidence Consistency—
correlation between answer confidence when
conditioned on query qand claims C(fθ(a|q∪C) )
extracted from retrieved information Dcompared
to conditioning on query and D. These are
computed based on both our confidence mea-
sures—geometric mean of token-level probabilities
and predictive entropy. If a proxy model’s
token-level probabilities are well calibrated,
the two would be highly correlated, since C
summarizes D. LLaMA and Qwen variants exhibit
high confidence consistency, while Gemma and
Ministral show substantially lower correlations,
suggesting they might be less suitable for proxy-
based CU estimation. Across models, normalized
log-likelihood yields more consistent confidence
estimates than predictive entropy.
Further, Table 9 reportsCU Agreementas the
pairwise correlation in CUandCU relacross proxy
models. CU scores from LLaMA and Qwen vari-
ants are nearly perfectly correlated, whereas corre-
lations involving Gemma and Ministral are lower,
although they remain positive. These patterns are
consistent for answers generated by both GPT-4o
and Gemini-2.5.
B.3 Readability
The next metric we show is readability. Since our
questions originate from domain-specific, graduate-
level, or professional contexts, we expect themodel-generated answers to match an appropri-
ate level of linguistic complexity. To evaluate this,
we employ two widely used readability measures.
The first is the Flesch Reading Ease (FRE) score
(Flesch, 1948), a numerical metric ranging from 0
to 100+, where higher values indicate easier text.
The FRE score is defined as:
FRE= 206.835−1.015·ASL−84.6·ASW (5)
where ASL denotes the average sentence length
(words per sentence), and ASW represents the av-
erage number of syllables per word. According to
conventional interpretation, scores above 60 corre-
spond to plain English that is understandable to the
general public (approximately 8th–9th grade read-
ing level), while scores below 30 indicate highly
complex or academic text, typically suitable for
college-level readers or above.
The second metric we consider is the
Flesch–Kincaid Grade Level (FKGL) (Kincaid
et al., 1975), which maps text complexity directly
onto a U.S. school grade level. FKGL is defined
as:
FKGL= 0.39·ASL+ 11.8·ASW−15.59(6)
A score of 8.0 indicates that the text is readable by
an average eighth-grade student, whereas a score of
16.0 or higher suggests that graduate-level reading
proficiency is required.
We apply these metrics to evaluate and com-
pare the readability of responses generated by two
prominent large language models:GPT-4oand

Hazard
TypeCounties and States
Coastal
FloodingBergen, NJ; Atlantic, NJ; Ocean, NJ; Cape May, NJ; Hudson, NJ; Monmouth, NJ; Grays Harbor, WA;
Middlesex, NJ; Kings, NY; Cumberland, NJ; Clatsop, OR; Cameron, TX; Philadelphia, PA; Queens, NY;
Coos, OR; Bronx, NY; Sussex, DE; Westchester, NY; New York, NY; Jefferson, LA; Fairfield, CT; St.
Charles, LA; Suffolk, NY; Aransas, TX; Union, NJ
Cold Wave Cook, IL; Milwaukee, WI; Minnehaha, SD; Wayne, MI; Lake, IL; Nueces, TX; Lake, IN; Hennepin,
MN; Williams, ND; Will, IL; Yakima, WA; Anoka, MN; Flathead, MT; Winnebago, IL; Pennington,
SD; Dane, WI; Ramsey, MN; Cass, ND; Marathon, WI; Sheboygan, WI; Blue Earth, MN; Brown, WI;
Olmsted, MN; Outagamie, WI; St. Louis, MN
Drought Santa Barbara, CA; Yolo, CA; Sutter, CA; Napa, CA; Colusa, CA; Glenn, CA; Butte, CA; Sonoma, CA;
Sacramento, CA; Solano, CA; Pinal, AZ; Floyd, TX; Lubbock, TX; Humboldt, NV; Doña Ana, NM;
Maricopa, AZ; Yuma, AZ; Kings, CA; Imperial, CA; Merced, CA; Madera, CA; Stanislaus, CA; Fresno,
CA; Tulare, CA; Kern, CA
Heat Wave Cook, IL; Clark, NV; St. Louis, MO; Philadelphia, PA; Dallas, TX; Tulsa, OK; Maricopa, AZ; Queens,
NY; Tarrant, TX; Kings, NY; Oklahoma, OK; Tulare, CA; Jackson, MO; Shelby, TN; Baltimore, MD;
Fulton, GA; Los Angeles, CA; Harris, TX; Bexar, TX; Fairfax, V A; Franklin, OH; DeKalb, GA; Prince
George’s, MD; Mecklenburg, NC; Wayne, MI
Hurricane Harris, TX; Miami-Dade, FL; Broward, FL; Palm Beach, FL; Hillsborough, FL; Lee, FL; Brevard,
FL; Pinellas, FL; Charleston, SC; Pasco, FL; Horry, SC; Collier, FL; Chatham, GA; Mobile, AL; New
Hanover, NC; Galveston, TX; Orange, FL; V olusia, FL; Indian River, FL; St. Lucie, FL; St. Johns, FL;
Manatee, FL; Clay, FL; Beaufort, SC; Escambia, FL
Ice Storm Nassau, NY; Tulsa, OK; Greene, MO; Lancaster, NE; St. Louis, MO; Oakland, MI; Boone, MO;
Richland, SC; Monmouth, NJ; Washington, AR; Macomb, MI; Johnson, KS; Morris, TX; Baxter, AR;
Rogers, OK; Douglas, NE; Sedgwick, KS; Linn, IA; Dubuque, IA; Stark, OH; Polk, IA; Peoria, IL;
Knox, TN; Lucas, OH; Hamilton, OH
Wildfire San Diego, CA; Riverside, CA; San Bernardino, CA; Los Angeles, CA; Washington, UT; Elko, NV;
Ventura, CA; Orange, CA; Pima, AZ; Maricopa, AZ; Ravalli, MT; Kern, CA; Yavapai, AZ; Utah,
UT; Madera, CA; Nevada, CA; Placer, CA; Shasta, CA; Siskiyou, CA; Tehama, CA; Santa Cruz, CA;
Alameda, CA; Tuolumne, CA
Table 7: Hazard types and the corresponding counties and states included in our dataset generation framework.
Gemini. For each generated response, we compute
the FRE and FKGL scores and categorize them
into standard readability levels. These include very
easy, plain English, college level, and graduate
level. Similarly, FKGL scores are grouped into
categories such as elementary, middle school, high
school, undergraduate, and graduate reading levels.
A summary of this analysis is presented in Ta-
ble 10. This includes average FRE and FKGL
scores for each model, as well as the distribution
of their outputs across interpretive readability lev-
els. These results allow us to assess not only the
average complexity of model outputs, but also how
frequently they produce text that is potentially in-
accessible to a general audience.
C Human Evaluation
We recruit 10 annotators to validate our evaluation
scores. Each annotator first logged into the evalu-
ation platform and was presented with a personal-
ized dashboard listing their assigned questions (see
Figure 4). The dashboard clearly enumerates eachquestion (e.g., Q16, Q21, Q1, etc.) along with an
“Annotate” button that opens the detailed evalua-
tion page. Upon selecting a question, annotators
were shown a structured evaluation interface (see
Figure 5) that displayed the user profile used to
generate the answer alongside the question itself,
the model-generated answer (formatted in num-
bered points following our prompt design), and
the five retrieved knowledge sources that informed
the response. Annotators then evaluated each an-
swer across four dimensions: Specificity, which
included sub-criteria for hazard type match, loca-
tion match, timeline match, and intensity match;
Answer Relevance, rated on a 1–10 scale; Context
Utilization, which assessed whether the retrieved
documents were actually used in the generated an-
swer; and Overall Confidence, where annotators
reported how confident they were in their evalu-
ation (1–10) and could optionally provide com-
ments. Each metric included clear guidance indi-
cating when an annotator should select Yes, No, or
N/A, along with radio-button options to standardize

EvaluatorGenerator=GPT-4o Generator=Gemini2.5
(fθ=log-prob) (f θ=entropy) (f θ=log-prob) (f θ=entropy)
LlaMA3.1-8B-Instruct0.83 (0.0) 0.40 (5.7e-55) 0.84 (0.0) 0.58 (2.1e-128)
Gemma2-9B-it0.17(4.4e-11) 0.16(4.2e-09) 0.12 (1.2e-05) 0.19 (2.1e-13)
Ministral-8B-Instruct0.14 (1.7e-07) 0.13 (2.0e-06) 0.13 (6.0e-07) 0.38 (3.2e-50)
Qwen2.5-7B-Instruct0.95 (0.0) 0.46 (3.9e-76) 0.96 (0.0) 0.62 (5.8e-150)
Qwen3-8B0.94 (0.0) 0.49 (1.7e-85) 0.93 (0.0) 0.60 (5.3e-137)
Table 8: Correlation ( ρ) between fθ(a|q∪C) andfθ(a|q∪D) . Where Cdenotes all extracted claims from retrieved
content R. P-values are reported in parentheses. The correlations for Gemma and Ministral are positive but low,
indicating the token probabilities might not be well-calibrated.
EvaluatorsGenerator=GPT-4o Generator=Gemini2.5
∆δ∆δ
LLaMA vs. Gemma-0.00 (0.87) -0.09 (0.0) -0.10 (2.94e-04) -0.09 (3.87e-04)
Ministral vs. Gemma0.66 (1.2e-177) 0.01 (0.63) 0.45 (1.2e-69) 0.01 (0.68)
Ministral vs. LLaMA0.03 (0.25) 0.06 (0.02) -0.02 (0.48) 0.01 (0.69)
Ministral vs. Qwen2.50.00 (0.87) 0.06 (0.04) -0.01 (0.7) 0.05 (0.06)
Qwen2.5 vs. LLaMA0.87 (0.0) 0.86 (0.0) 0.84 (0.0) 0.80 (2.5e-321)
Qwen2.5 vs. Gemma-0.02 (0.42) 0.01 (0.77) -0.10 (1.7e-04) -0.04 (.13)
Ministral vs. Qwen3-0.03 ( 0.32) 0.05 (0.04) -0.02 (0.41) 0.02 (0.43)
Qwen3 vs. LLaMA0.81 ( 0.0) 0.83 (0.0) 0.85 (0.0) 0.79 (1.2e-295)
Qwen3 vs. Gemma-0.03 (0.30) 0.00 (1.00) -0.08 (.002) -0.03 (0.2)
Qwen3 vs. Qwen2.50.82 (0.0) 0.83 (0.0) 0.83 (0.0) 0.82 (0.0)
Table 9: Pairwise correlation between context utilization scores from different proxy models using both absolute
(∆) and relative (δ) scores. P-values are reported in parentheses.
Metric GPT-4o Gemini
Average Flesch Reading Ease 12.84 10.98
Average Flesch-Kincaid Grade
Level17.80 18.22
#Fairly difficult (10th–12th grade) 0 2
# Difficult to read (college level) 37 93
# Very difficult (college graduate
or higher)1375 1317
#Middle school level 1 2
#High school level 0 28
# College undergraduate level 167 295
# Graduate/professional level 1244 1087
Table 10: Comparison of automated readability scores
for answers generated by GPT-4o and Gemini.
responses.
Agreement.Table 11 shows agreement, disagree-
ment and agreement percentage between human an-
notators over 50 double-annotated question–answer
pairs. Table 12 shows the same metrics for agree-
ment between human labels and automated evalua-
tors. Note that each question is annotated by two
annotators, yielding 100 question-answer pairs.
Error Analysis.Figure 6 shows one representative
example, including the question, context, retrieved
knowledge (partial), and the generated answer. In
this case, the answer includes temporal informa-
tion, but it is stated in a vague form compared toMetric Agree Disagree Agree (%) Fleiss’ kappa
Hazard46 4 92 0.6678
Location43 7 86 0.6834
Timeline26 24 52 0.2640
Intensity39 11 78 0.4579
Table 11: Inter-annotator agreement across 50 double-
annotated questions. Agreement is higher for more
concrete attributes (e.g., hazard, location) and lower for
more subjective ones (e.g., timeline).
Metric Agree Disagree Agree (%)
GPT Qwen GPT Qwen GPT Qwen
Hazard 88 88 12 88 88 88
Location 77 73 23 27 77 73
Timeline 45 40 55 60 45 40
Intensity 75 59 25 41 75 59
Table 12: Human vs. automated evaluator agreement
over 100 human-annotated rows (50 questions × 2 anno-
tators), computed as row-level exact match. Alignment
is higher for hazard and location and lower for timeline,
intensity, and answer relevance.

Figure 4: The annotation dashboard where each annotator receives ten questions and evaluates them by selecting the
“Annotate” button for each item.
the evidence. The retrieved sources discuss time-
related effects, but they do not support that exact
range. This creates an ambiguity: one may mark
the timeline as supported because a timeline is men-
tioned, while a stricter interpretation marks it as
unsupported because the claimed timeline is not
explicitly verifiable from the provided evidence.
D Implementation Details
All experiments are conducted using fixed decod-
ing and evaluation configurations to ensure con-
sistency across both proprietary and open-source
models. For answer generation, we use a tempera-
ture of 0.1 and a top-p of 0.9 for both GPT-4o and
Gemini-2.5-flash to prioritize factual precision and
minimize stochastic variation, which is particularly
important in high-stakes domains. For literature re-
trieval, we use a semantic search system built on a
knowledge base containing approximately 600,000
climate-related records. Query embeddings are
generated using the all-MiniLM-L6-v2 sentence
transformer model and indexed with FAISS (Face-
book AI Similarity Search). For each query, we
retrieve the top five relevant literature abstracts
(k= 5 ), which are used as the grounding context
for all generated responses. All proprietary mod-
els are accessed via APIs. Local model inferences
for LLaMA3.1-8B, Qwen3-8B etc., are performed
using the vLLM framework on an NVIDIA A100
(80GB) GPU to ensure stable, high-throughput, and
efficient inference.

(a) Annotation Interface – Part 1
(b) Annotation Interface – Part 2
Figure 5: Human Annotation Interface of our evaluation pipeline.

Question: What are the cascading impacts of ice storm between coastal protection and building systems in Richland, South Carolina? 
Retrieved-Knowledge:  Cold regions cover more than half the United States. Engineering impacts of cold regions include those caused by 
temperature (e.g., freeze–thaw cycling, atmospheric icing), river and lake ice (e.g., ice jams, operational impacts of ice), and snow. 
Damages include the potential The City of Iqaluit, Nunavut, is an expanding urban centre with important infrastructure located in the 
coastal zone. This study investigates the exposure of this infrastructure to coastal hazards (rising mean sea level, extreme water levels, 
wave run-up, and sea ice)..…… 
Using a coastal digital elevation model, we evaluate the inundation and flooding that may result from projected sea level rise. Some public 
and private infrastructure is already subject to flooding during extreme high water events. Using a near upper-limit scenario of 0.7 m for 
relative sea level rise from 2010 to 2100 , we estimate that critical infrastructure will have a remaining freeboard of 0.3–0.8 m above high 
spring tide, and some subsistence infrastructure will be inundated. The large tidal range, limited over-water fetch, and wide intertidal flats 
reduce the risk of wave impacts. When present, the shorefast ice foot provides protection for coastal infrastructure. The ice-free season 
has expanded by 1.0–1.5 days  per year since 1979 , increasing the opportunity for storm-wave generation and thus exposure to wave 
run-up. Overtopping of critical infrastructure and displacement by flooding of subsistence infrastructure are potential issues requiring better 
projections of relative sea level change and extreme high water levels…… 
Hurricane Irene made landfall as a sizeable Category I storm at the outer banks of North Carolina on August 27, 2011 . Overland wind 
gusts associated with the storm measured nearly 115 MPH, and tide surge reached nearly 10 feet. The storm impacted numerous 
communities along the coast line and sounds, and it also damaged many communities and structures miles upstream in several larger 
river basins. This paper presents a summary of damages attributed to wind and storm surge for approximately 30 dwellings dispersed 
across eastern North Carolina..…. Answer:  As a Wastewater Treatment Specialist in Richland, South Carolina, your concern about the cascading impacts of ice storms on 
coastal protection and building systems is well-founded, especially given the potential for climate change to exacerbate these issues over 
the next 30 to 50 years . The research indicates that cold regions, which cover a significant portion of the United States, experience 
engineering impacts from ice-related phenomena such as freeze-thaw cycles and ice jams. These can lead to structural damage, 
increased flood fighting costs, and environmental damage (U.S. Army Corps of Engineers). 
- Although Richland, South Carolina, is not traditionally considered a cold region, the increasing unpredictability of weather patterns due to 
climate change could lead to more frequent and severe ice storms. This is supported by findings that climate change may exacerbate 
problems in temperate regions, leading to increased risk and uncertainty from ice jams and related flooding (Investigating the effects of 
long-term hydro-climatic trends on Midwest ice jam events). Context: Profession: Wastewater Treatment Specialist, Concern: Hybrid, Location: Richland, South Carolina, Timeline: 30 - 50 years Figure 6: Example from the error analysis showing the question, context, retrieved evidence, and generated answer.
The highlighted text shows temporal information.

E Prompts
E.1 Answer Generation
For generating answers, we provide five retrieved
documents, user profile and question in the prompt.
Prompt for Answer Generation
You are tasked with writing a recommendation/fact-based answer that answers the user’s question
based on a provided list of research abstracts and contextual information. Your response must:
1. Directly address the user’s concern, ensuring the answer is supported by the provided literature.
2. Incorporate the user’s profile like timeline, professional background, Location, and concerns
into the recommendations.
3. Clearly connect insights from the abstracts to the user’s specific context and goals.
4. Make sure to output in points (1,2,3..) without inserting any **.
5. End your response with a confidence score (in percentage) and a short explanation for that score.
Here are the 5 research abstracts:
1. {lit1}
2. {lit2}
3. {lit3}
4. {lit4}
5. {lit5}
Context: {context}
Question: {question}
Based on the above abstracts, write the answer in points. Make sure to take into account all the
information in the context like profession, timeline, etc. Do not include subpoints.
E.2 Specificity
For specificity evaluation, we provide claim, re-
trieved documents from the knowledge base and
specific details to verify.

Prompt for Specificity Evaluation
You are a strict evaluator of specificity and factuality.
Given for each claim:
- A factual claim
- A list of evidence passages from a trusted source
- A set of specific details extracted from the claim (hazard type, location, timeline, intensity)
Your task is to evaluate the claim using ONLY the provided evidence.
LABEL DEFINITIONS
For each specific detail (hazard, location, timeline, intensity), use EXACTLY one of the following
labels:
- "yes":
The detail is explicitly mentioned in the claim AND it matches the same specific detail discussed
in the knowledge source.
- "no":
The detail is explicitly mentioned in the claim BUT the knowledge source does NOT provide
sufficient information to verify it. (This includes cases where the evidence contradicts the claim or
does not confirm it.)
- "N/A":
The detail is NOT mentioned in the claim at all.
For location matching, agreement at the STATE level is sufficient; an exact county or city match is
not required. Do NOT infer or assume any facts beyond the evidence. Lack of verification MUST
be labeled as "no" (not "N/A").
EV ALUATION STEPS
Your task is to:
1. Determine whether the **claim is factually true, false, or partially correct**, using ONLY the
evidence.
2. For each of the 4 specific details (hazard, location, timeline, intensity):
- Assign "yes", "no", or "N/A" based on the rules above.
- Provide a brief factual explanation for your decision.
3. Justify your overall factuality decision concisely and objectively.
4. If the claim is "true", cite the exact evidence passage(s) that support it.
5. If the claim is "false" or "partially correct", explain precisely which details are unsupported or
incorrect.
OUTPUT FORMAT
Return your answer as a SINGLE JSON object in the following format (with no markdown, no
extra text, and no explanations outside the JSON):
{{ "claim": "<Claim>",
"hazard": "yes" | "no" | "N/A",
"hazard_reasoning": "<Explain whether hazard mentioned in the claim is explicitly supported>",
"location": "yes" | "no" | "N/A",
"location_reasoning": "<Explain whether location mentioned in the claim is supported>",
"timeline": "yes" | "no" | "N/A",
"timeline_reasoning": "<Explain whether timeline like date and range of years mentioned in the
claim is supported>",
"intensity": "yes" | "no" | "N/A",
"intensity_reasoning": "<Explain whether intensity mentioned in the claim is supported>"
}}
INPUTS
Claim: {claim}
Specific Details to Check: {specific_info}
Evidence Passages: {knowledge}

E.3 Answer Relevance
We use LLM for masking important specific details
for answer relevance evaluation.
Prompt for Masking
You are a semantic masker. Given the following answer, replace:
- hazard types with [HAZARD]
- profession-related terms with [PROFESSION]
- concern with [CONCERN] (e.g., critical vulnerabilities, maintenance strategies, modernization
measures, maintenance strategies, projected impact, design standards, cascading impacts etc.)
- infrastructure with [INFRASTRUCTURE] (e.g., "highway network", "bridge system", "public
transit system", "railway infrastructure", "airport facilities", "port facilities", "freight terminals",
"traffic control systems", "water treatment plant", "wastewater system", "dam infrastructure",
"stormwater system", "coastal protection", "water distribution network", "electrical grid", "power
distribution network", "EV charging network", "renewable energy infrastructure", "energy storage
facilities", "power transmission lines", "substations", "public buildings", "critical facilities",
"commercial structures", etc.)
Keep the structure natural and readable.
Answer: {answer}
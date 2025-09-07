# RAGuard: A Novel Approach for in-context Safe Retrieval Augmented Generation for LLMs

**Authors**: Connor Walker, Koorosh Aslansefat, Mohammad Naveed Akram, Yiannis Papadopoulos

**Published**: 2025-09-03 23:24:17

**PDF URL**: [http://arxiv.org/pdf/2509.03768v1](http://arxiv.org/pdf/2509.03768v1)

## Abstract
Accuracy and safety are paramount in Offshore Wind (OSW) maintenance, yet
conventional Large Language Models (LLMs) often fail when confronted with
highly specialised or unexpected scenarios. We introduce RAGuard, an enhanced
Retrieval-Augmented Generation (RAG) framework that explicitly integrates
safety-critical documents alongside technical manuals.By issuing parallel
queries to two indices and allocating separate retrieval budgets for knowledge
and safety, RAGuard guarantees both technical depth and safety coverage. We
further develop a SafetyClamp extension that fetches a larger candidate pool,
"hard-clamping" exact slot guarantees to safety. We evaluate across sparse
(BM25), dense (Dense Passage Retrieval) and hybrid retrieval paradigms,
measuring Technical Recall@K and Safety Recall@K. Both proposed extensions of
RAG show an increase in Safety Recall@K from almost 0\% in RAG to more than
50\% in RAGuard, while maintaining Technical Recall above 60\%. These results
demonstrate that RAGuard and SafetyClamp have the potential to establish a new
standard for integrating safety assurance into LLM-powered decision support in
critical maintenance contexts.

## Full Text


<!-- PDF content starts -->

RAGuard: A Novel Approach for in-context Safe
Retrieval Augmented Generation for LLMs
Connor Walker1,2[0009 −0008−8181−1644], Koorosh
Aslansefat1,2[0000 −0001−9318−8177], Mohammad Naveed
Akram[0000−0002−0924−5536], and Yiannis Papadopoulos1,2[0000 −0001−7007−5153]
1University of Hull, Cottingham Road, Hull HU6 7RX, UK
2AURA CDT, Hull UK C.Walker-2018@hull.ac.uk
https://www.deis-hull.com/connor-walker
Abstract. Accuracy and safety are paramount in Offshore Wind (OSW)
maintenance, yet conventional Large Language Models (LLMs) often
fail when confronted with highly specialised or unexpected scenarios.
We introduce RAGuard, an enhanced Retrieval-Augmented Generation
(RAG) framework that explicitly integrates safety-critical documents
alongside technical manuals.By issuing parallel queries to two indices and
allocating separate retrieval budgets for knowledge and safety, RAGuard
guarantees both technical depth and safety coverage. We further develop
a SafetyClamp extension that fetches a larger candidate pool, "hard-
clamping" exact slot guarantees to safety. We evaluate across sparse
(BM25), dense (Dense Passage Retrieval) and hybrid retrieval paradigms,
measuring Technical Recall@K and Safety Recall@K. Both proposed
extensionsofRAGshowanincreaseinSafetyRecall@Kfromalmost0%in
RAG to more than 50% in RAGuard, while maintaining Technical Recall
above 60%. These results demonstrate that RAGuard and SafetyClamp
have the potential to establish a new standard for integrating safety
assurance into LLM-powered decision support in critical maintenance
contexts.
Keywords: Large Language Models ·In-context Safety ·AI Safety ·RA-
Guard ·Offshore Wind ·Maintenance ·Retrieval-Augmented Generation
(RAG) ·Decision Support ·Safety-critical
1 Introduction
Accuracy and safety are paramount in Offshore Wind (OSW) maintenance, an in-
dustry characterised by challenging environmental conditions, remote operations,
and highly complex technical tasks. Human error or inaccurate decisions during
maintenance activities can lead to costly downtime, significant safety risks, and
environmental hazards. As global reliance on OSW power grows, ensuring the
reliability and safety of these operations becomes critically important.
Recent advances in Artificial Intelligence (AI), particularly Large Language
Models (LLMs), offer considerable promise as decision-support tools capablearXiv:2509.03768v1  [cs.AI]  3 Sep 2025

2 C. Walker et al.
of assisting maintenance personnel by providing immediate access to relevant
knowledge. LLMs have demonstrated powerful capabilities to generate context-
aware recommendations, summarise complex documents, and support operational
decision making. However, conventional LLM-based systems often falter when
confronted with unexpected or highly specialised scenarios, primarily due to the
limited availability of relevant, scenario-specific training data. This shortcoming
is particularly problematic in OSW maintenance, where unexpected scenarios
can significantly increase operational risks and decision complexity.
Addressing this challenge, we introduce RAGuard, an enhanced Retrieval-
Augmented Generation (RAG) framework explicitly tailored for OSW mainte-
nance contexts. RAGuard integrates in-context safety protocols directly into
the retrieval and generation process, dynamically prioritising safety considera-
tions based on real-time maintenance scenarios. Unlike traditional RAG systems,
RAGuard leverages specialised technical documentation, such as maintenance
manuals and technical data sheets, to retrieve highly relevant, context-specific
information precisely when needed. This real-time retrieval process ensures that
maintenance guidance remains accurate, current, and scenario-appropriate, sig-
nificantly reducing the risks associated with unexpected situations.
To systematically assess the effectiveness of RAGuard, we propose an eval-
uation framework specifically designed to evaluate the safety and reliability of
AI-based decision-support systems in OSW maintenance tasks. Preliminary eval-
uations suggest that RAGuard improves the quality and contextual relevance
of maintenance guidance, indicating its potential to enhance operational safety.
While further operational validation is required, initial results highlight RAGuard
as a promising approach toward safer and more reliable decision-support systems
in critical maintenance contexts.
The remainder of this paper is structured as follows: Section 2 reviews relevant
literature on RAG methodologies and safety-focused AI. Section 3 provides an in-
depth description of the RAGuard framework. Section 4 details our experimental
methodology and newly proposed benchmark. Section 5 presents our empirical
results and their implications, and Section 6 concludes with a discussion of
contributions and future research directions.
2 Background
2.1 Retrieval-Augmented Generation
RAG is a paradigm that integrates information retrieval with generative models
to improve knowledge-intensive Natural Language Processing (NLP) tasks. Foun-
dational work by [7] introduces RAG models that combine a parametric neural
generator with a non-parametric memory (e.g. a Wikipedia index), enabling
the generator to retrieve relevant documents and produce more factual, specific
answers. This approach achieved state-of-the-art performance on open-domain
question answering tasks, outperforming purely parametric models. Recent ad-
vancements have further enhanced RAG. For example, [4] develops Atlas, a

Title Suppressed Due to Excessive Length 3
pre-trained RAG model that excels in few-shot learning settings. Atlas can attain
over 42% accuracy on the Natural Questions benchmark using only 64 examples,
surpassing a 540-billion-parameter closed-book model with far fewer parameters.
Such progress demonstrates RAG’s effectiveness in injecting up-to-date knowledge
into LLMs while controlling model size and hallucinations.
2.2 Safety-Focused AI in Critical Domains
AI systems deployed in safety-critical industries (healthcare, aviation, energy,
etc.) must be designed with rigorous safety and reliability considerations. In
healthcare, for instance, the use of AI for clinical decision support raises concerns
about accountability and patient harm. [3] argues that current safety assurance
practices are not yet adjusted to AI-driven tools, which can make high-stakes
decisions in ways that are opaque to clinicians. They emphasise the need for new
frameworks of moral accountability and dynamic safety assurance throughout
an AI system’s lifecycle. In aviation, AI and Machine Learning (ML) are being
applied to augment safety analysis and risk prediction. A recent systematic
review by [2] shows that techniques like deep learning, time-series modelling, and
optimisation algorithms are increasingly used to detect patterns in aviation data
and enhance safety measures. These AI-driven methods support proactive safety
management (e.g. predictive maintenance and improved air traffic control) to help
prevent accidents before they occur. In the energy sector, especially nuclear power,
AI offers potential to improve monitoring and emergency response in complex
industrial systems. [5] surveys AI applications in nuclear power plants, noting that
AI-based predictive analytics and real-time data processing could bolster reactor
safety and decision-making. Their findings highlight early warning systems that
use ML and Internet of Things (IoT) sensors to detect anomalies, coordinate with
operators, and mitigate risks in critical scenarios. However, they also point out
challenges such as the need for updated regulations and cybersecurity safeguards
when integrating AI into safety-critical infrastructure.
2.3 Adaptive Retrieval Mechanisms
Adaptive retrieval mechanisms dynamically adjust how and when external in-
formation is fetched during generation, often making retrieval context-aware or
safety-aware. Instead of a fixed one-pass retrieval, these approaches allow the AI
to decide if additional knowledge is needed and to retrieve iteratively or on the
fly. For example, recent RAG variants like Forward Looking Active REtrieval
Augmented Generation (FLARE) and Self-RAG equip language models with
the ability to trigger new retrievals based on the model’s internal confidence
or reflection tokens. This means the model can autonomously determine the
optimal moments to query the knowledge base, stopping when enough informa-
tion has been gathered. [1] proposes Self-RAG, where the model “self-reflects”
on its draft answer and issues further queries if needed, which streamlines the
retrieve-generate loop and improves answer accuracy. Similarly, [6] introduces an
active retrieval strategy that monitors the generation process and fetches new

4 C. Walker et al.
evidence when the model’s certainty falls below a threshold, thereby tailoring
retrieval to the query’s complexity. Beyond research prototypes, the concept
of adaptive retrieval is evident in systems like OpenAI’s WebGPT, which used
reinforcement learning to train GPT-3 to invoke a search engine mid-generation.
This allowed the model to decide when to look up information and even cite
sources, behaving like an agent that can use tools. Equally important is making
retrieval safety-aware, so that the information brought into the generation loop
does not introduce harm or vulnerabilities. One line of work addresses filter-
ing and control of retrieved content. For instance, security evaluations such as
SafeRAG [8] show that without safeguards, adversarial or toxic documents can be
injected into the retrieval corpus, leading to misleading or harmful outputs. This
has underscored the need for robust retrieval filters and context validation. In
practice, integrating content moderation–e.g. removing offensive or contradictory
results before generation–is becoming a recommended step in RAG pipelines. By
dynamically assessing the safety of retrieved passages (using allow-lists, block-
lists, or classifier checks), an adaptive retrieval system can reject or down-weight
unsafe context. Such context-aware and safety-conscious retrieval mechanisms
are an active research area aimed at ensuring that AI systems remain reliable
and aligned even as they pull in external information.
In summary, while existing approaches provide robust RAG methods and
explore AI safety considerations in critical sectors, there is limited research on
unifying these aspects for operational decision support. This paper addresses
this gap by presenting RAGuard, a framework that integrates safety-aware
retrieval filtering with context-specific generation to enhance reliability in OSW
maintenance scenarios.
3 Methodology
3.1 Retrieval-Augmented Generation
RAG enhances LLMs by integrating external knowledge sources at inference time.
Unlike traditional LLMs, which rely solely on their internal parameters–often
outdated or insufficient for specialised domains, RAG enables access to up-to-date,
domain-specific information through retrieval.
In a typical RAG pipeline, the user’s query is embedded into a dense vector
using an encoder (e.g., a bi-encoder like Dense Passage Retrieval (DPR)). Mean-
while, the external corpus is pre-encoded into the same vector space and stored
in a similarity search index (e.g., Facebook AI Similarity Search (FAISS)). The
retriever ranks passages based on similarity (cosine or inner product), and some
systems apply additional filtering to discard irrelevant or low-confidence results.
The top-ranked passages are combined with the query to create an enriched
prompt. In fusion-in-decoder architectures, passages are encoded separately, and
the decoder attends to all inputs to extract relevant information. This retrieval-
augmented setup enhances factual grounding and reduces hallucinations while
keeping the model compact.

Title Suppressed Due to Excessive Length 5
The generator integrates retrieved content into a coherent, informed response.
Incorporating external evidence makes RAG outputs more accurate and context-
aware than closed-book approaches. Figure 1 summarises each step of the RAG
process below.
Fig. 1.Standard RAG Model Process
RAG Retrieval Parameters As with LLMs, RAG systems include tunable
hyperparameters that balance retrieval, latency, and noise robustness.
Before indexing, documents are divided into smaller chunks. Chunk size
controls how much text is embedded at once. Smaller chunks improve retrieval
granularity—helping to match precise content, but increase memory and latency.
Larger chunks reduce total passages but may dilute relevance.
Chunk overlap determines how much text is shared between adjacent chunks
(typically 25-50% of chunk size ). It helps preserve context continuity—preventing
important sentences from being split —at the cost of indexing more passages.
Too little overlap risks breaking up meaningful context across chunk boundaries.
Top-kspecifies how many chunks are returned to the LLM prompt. Higher
values increase coverage but raise the chance of irrelevant content and computa-
tional load. Lower values reduce noise but may miss critical context. Standard
RAG retrieves the Kmost relevant chunks by maximising the total score:
RAG (q;K) = arg max
D′⊆D,|D′|=KX
d∈D′s(q, d) (1)
Here, qdenotes the query, Dis the full document corpus, D′is the selected
subset of Kpassages, and s(q, d)is the similarity score between the query and
passage d. This optimisation ensures that the top- Kpassages with the highest
aggregated relevance scores are retrieved.
Building on this, we introduce RAGuard which extends the pipeline with
explicit safety mechanisms.

6 C. Walker et al.
3.2 RAGuard
Although traditional RAG systems are effective in dynamically integrating exter-
nal knowledge into generative processes, they do not explicitly prioritise safety
considerations, potentially retrieving contextually relevant but operationally
inappropriate or unsafe guidance. To address these limitations, we propose RA-
Guard, an enhanced RAG framework specifically developed to prioritise safety
and contextual accuracy in OSW maintenance operations. The fundamental in-
novations of RAGuard compared to traditional RAG are safety cache integration,
dynamic safety prioritisation, and safety-guided generation. We also propose an
additional "SafetyClamp" layer on RAGuard to over-retrieve context passages,
reserve predefined knowledge and safety slots, and dynamically fill any leftover
slots to supply a full quota.
Safety Cache Integration RAGuard utilises a dedicated cache that contains
validated safety protocols and operational guidelines. Unlike a conventional
RAG system that draws from a single monolithic document index, RAGuard
maintains two parallel knowledge repositories: one containing general maintenance
documentation and the other devoted exclusively to safety-critical content. This
may include documents such as regulations, industry-specific protocols, and any
other relevant information pertinent to the safe completion of all maintenance
tasks within the given environment.
At retrieval time, the user’s query is issued simultaneously against both
corpora, ensuring that the model can draw on rich technical details, while also
surfacing explicit hazard warnings, procedural safeguards, and regulatory guide-
lines. By isolating safety passages in their own index, we can apply dedicated
filtering and scoring thresholds that reflect the gravity of risk management, with-
out diluting the coverage or performance of the broader maintenance knowledge
base.
The dual-stream retrieval process produces two ranked lists of passages; one
optimised for technical relevance, and the other for safety assurance, which are
then merged before context integration. This architecture therefore guarantees
that every generated recommendation is grounded not only in accurate domain
expertise but also in up-to-date, rigorously validated safety information. We
propose two merging functions to this effect and evaluate the effectiveness of
both.
RAGuard Retrieval Parameters RAGuard introduces three new retrieval
hyperparameters; knowledge-k (denoted kknow),safety-k (denoted ksafe), and
fetch-k(denoted kfetch). These control the balance between technical depth and
safety oversight in the dual index setup. The aforementioned top−kremains in
use as the total number of passages passed to the prompt context.

Title Suppressed Due to Excessive Length 7
3.3 RAGuard Retrieval
RAGuard modifies the standard RAG retrieval step by splitting the total K
retrieved passages into two parts: kknowfrom a technical knowledge index, and
ksafepassages from a safety-specific index. This ensures that the final context
includes both technical and safety-relevant content, with K=kknow +ksafe.
At retrieval time, the user’s query qis sent to both indices in parallel. Each
index returns its top Kpassages according to a relevance score function (q, d).
The two sets of results are then merged into a single list and passed into the
LLM prompt. For instance, if kknow = 2andksafe= 3, the final prompt includes
five passages: two from the knowledge index, and three from the safety index.
The process can be formalised as follows:
Knowledge index retrieval:
Mknow ={d1, d2, . . . , d kknow}, d i= arg max
d∈Dkknows(q, d) (2)
Safety index retrieval:
Ssafe={d′
1, d′
2, . . . , d′
ksafe}, d′
j= arg max
d∈Dksafes(q, d) (3)
Combined prompt input:
RAGuard (q) = [d1, . . . , d kknow, d′
1, . . . , d′
ksafe] (4)
In summary, RAGuard queries two indices simultaneously, retrieves fixed top
results from each, and merges them into a structured LLM input. The next
section introduces SafetyClamp , which extends this by enforcing safety quotas
and over-retrieving to maximise coverage and control.
3.4 RAGuard with SafetyClamp
RAGuard with the additional SafetyClamp builds directly on the base framework
by enforcing an absolute safety guarantee on every retrieved passage. Rather
than simply interleaving kknowandksafecandidates, SafetyClamp begins by
over-retrieving a wider pool of contenders; for both the knowledge index and
dedicated safety index, the system retrieves the top kfetchpassages by relevance.
This over-retrieval ensures that, even under strict slot requirements or occasional
index sparsity, there will always be enough qualified passages to fill every reserved
slot.
Once both pools are retrieved, SafetyClamp assigns passages in a hard-
guaranteed sequence. The first kknowslots are filled by the highest scoring
passages from the knowledge index. Next, the pipeline selects exactly ksafe
passages from the safety index. Unlike the base RAGuard, K > k know +ksafe,
meaning once this is complete, there are still empty slots for additional passages.
These are filled by the combined retrieved pools from both indices, choosing the
next highest scoring passages not yet selected, regardless of whether they come

8 C. Walker et al.
from the knowledge or safety index. Given that kfetchexceeds the final K, this
wildcard mechanism reliably completes the prompt without sacrificing safety
guarantees or technical comprehensiveness.
We can formalise RAGuard with SafetyClamp in three steps, reusing Mknow
andSsafedefined by equations 2 and 3. First, we over-retrieve a combined
candidate list:
C= [c1, c2, . . . , c kfetch] (5)
Where, ci= arg max
d∈D\d/∈{c1,...,c i−1}s(q, d)
Next, we remove any already selected passages from Mknow∪Ssafe, preserving
order to form the wildcard pool:
R= [r1, r2, . . .] (6)
Where each rl=cil∈Candcil/∈Mknow∪Ssafe, and, i1< i2< . . .are the
indices of those survivors in C.
Finally, SafetyClamp guarantees exactly kknowknowledge passages, ksafesafety
passages, and fills the remaining ( K−(kknow +ksafe)) slots from R:
SafetyClamp (q;K;kknow;ksafe) =
[m1, . . . , m kknow, sksafe, . . . , r 1, . . . , r K−(kknow +ksafe)](7)
Here{mi}=Mknow,{s′
j}=Ssafe, and the rs are drawn from the filtered R.
Because kfetch > K, there are always enough wildcards to complete the list.
In essence, SafetyClamp ensures its dual objectives by allocating fixed slots
for knowledge and safety passages, over-retrieving extra candidates as wildcards,
and assembling the prompt to meet quota requirements and context size. This
enforces a safety minimum while preserving technical depth, with over-retrieving
preventing empty slots.
4 Evaluation
In this section, we describe how we measure each system’s ability to deliver both
accurate maintenance guidance and essential safety information under realistic
OSW conditions. We first outline our curated evaluation dataset of domain-
specific queries paired with "gold-standard" answers and regulatory excerpts. We
then present the metrics used to quantify technical fidelity, safety compliance,
and system efficiency. Finally, we report and analyse results for standard RAG,
base RAGuard, and RAGuard with SafetyClamp, highlighting the trade-offs each
design makes between precision, coverage, and latency.
4.1 Evaluation Dataset
The evaluation leverages a dataset of 100 maintenance-focused questions, each
paired with a "gold-standard" technical answer and the corresponding safety

Title Suppressed Due to Excessive Length 9
context drawn from two key industry regulations: the Provision and Use of Work
Equipment Regulations (PUWER) and the Work at Height Regulations (WAHR).
For every query, we manually curate the precise procedural steps that consti-
tute the correct technical resolution, and annotated the relevant excerpts from
PUWER and WAHR that articulate the required safety checks, hazard warnings,
or permitted work practices. This structured format allows us to measure not
only whether each system retrieves the passages necessary to reconstruct the
technical solution, but also whether it surfaces the exact regulatory language
needed to ensure compliance with both PUWER and WAHR.
By combining domain-specific questions with dual sources of ground-truth
(technical and safety), our dataset provides a rigorous test bed for assessing
how well RAG, RAGuard and RAGuard with SafetyClamp balance operational
accuracy against mandatory safety requirements.
Crucially, we evaluate each of these three pipelines under all three retrieval
paradigms: sparse (BM25), dense (DPR), and hybrid (a weighted fusion of BM25
and DPR scores), to isolate the effect of the underlying retriever on both technical
fidelity and safety compliance.
4.2 LLM Prompt Structure
Each retrieval pipeline shares a common prompt template that clearly delineates
technical guidance from safety considerations. At the top of the prompt, the
model is instructed to use the provided context to answer the question and
admit when it does not know an answer rather than hallucinating. The template
then presents two labelled context sections: under "Maintenance Context", the
passages retrieved from the technical knowledge index are inserted, and under
"Safety Context", the passages from the safety index appear. As the standard
RAG pipeline does not explicitly differentiate between general knowledge and
safety, both sections are identical using the top−kpassages retrieved from one
index. Following these sections, the user’s query is posed with a "«QUESTION»"
marker, ensuring that the LLM’s attention remains focused on the specific
maintenance task.
Below the question, the prompt ends with a structured "ANSWER" area
containing two numbered slots. The first slot, labelled "1) Procedure:," is where
the model should generate step-by-step maintenance instructions grounded in the
technical context. The second slot, labelled "2) Safety Considerations:," allows
explicit hazard checks, warnings, or regulatory requirements drawn from the
safety context.
By splitting the expected output into these two clearly defined components,
we can directly assess both the procedural accuracy of the generated guidance
and the completeness of the safety advice during evaluation.
4.3 Evaluation Metrics
Hyperparameter Optimisation To establish a fair basis for comparing all
pipelines, we first perform a comprehensive hyperparameter selection step. In

10 C. Walker et al.
this, we sweep across the retrieval paradigms and a grid of quota settings ( kknow,
ksafe,kfetch, and K).
For each combination, we measure RetrievalRecall @Kon both technical and
safety passages across 100 questions. This single metric allows identification of
which retrieval regime and which hyperparameter values maximise the likelihood
of fetching all required "gold-standard" passages.
Once we determine the best settings for each pipeline, we fix those values for
the remaining evaluations. This two-stage approach ensures that all systems are
compared under their strongest retrieval settings, yielding a more meaningful
assessment of their safety-aware enhancements.
Retrieval Recall@K We measure RetrievalRecall @Kseparately for technical
and safety passages. For each query, we examine the set of passages provided
to the LLM and record whether the "gold-standard" technical passage and
the annotated safety excerpt appear among the top−K. Averaging over all
100 queries yields two recall scores; one reflecting the likelihood of finding the
correct procedural context, and one capturing the chance of including at least
one requisite safety clause in the prompt.
Safety Compliance Recall This measureswhetherthe retrieved safetypassages
collectively cover every regulatory requirement specified for a given question. We
treat a query as compliant only if all PUWER and WAHR clauses annotated
in the dataset appear somewhere in the safety-context feed. The resulting recall
rate thus reflects each pipeline’s ability to surface the full set of mandated safety
checks.
Latency and Context Utilisation To gauge the practicality of the real
world, we measure the average end-to-end retrieval time over the full dataset,
computing the ratio of occupied to available tokens in the LLM’s context window
(K/max −content −size). This reveals how each approach balances richer
contextual grounding against system responsiveness and prompt-size constraints.
We ran all of our latency and context-utilisation measurements on a high-end
workstation laptop to approximate a realistic "edge" deployment scenario. The
machine is a Ubuntu 22.04.4 LTS system powered by a 13th-Generation Intel®
i9-13980HX (24 threads @ up to 5.6 GHz), with 64 GB of DDR5-5600 RAM and
an NVIDIA RTX 4090 GPU. Retrieval timings were recorded on the CPU only;
the retriever indices live in memory and are served locally, while the context-
window fractions assume a model with a 4,096 token window (4K model). We
report both the mean and the standard deviation of 100 runs per pipeline, to
display not only the "typical" latency but also its variability under this hardware
configuration.

Title Suppressed Due to Excessive Length 11
5 Results & Discussion
5.1 Hyperparameter Optimisation
We perform a full grid search over our four retrieval hyperparameters— K,kknow,
ksafe, and kfetch—subject to 1≤kknow < K,1≤ksafe≤K−kknow, and
K∈ {1,···,10},kfetch∈ {25,50,75,100,125,150,175,200}, plus the Base RAG
cases ( kknow =ksafe= 0, kfetch =None ).
IfNis the number of distinct Kvalues and Fthe number of kfetchoptions, the
total number of valid 4-tuples ( K, k know, Ksafe, kfetch) we test is:
|settings |=FNX
K=1(K−1)K
2+N (8)
ForN= 10andF= 8,this yields 1,330 distinct configurations. For each we
compute:
Combined Recall =1
2(Knowledge Recall + Safety Recall) (9)
and then select, for each of the nine pipelines, the configuration that maximises
Combined Recall. The resulting optimal parameters and their achieved recall
scores are reported in Table 1.
Table 1. Best Combined Recall for Each RAG Variant
RAG VariantK Values Recall Metrics
top know safe fetch Knowledge Safety Combined
Dense Base 10 – – – 0.925 0.09 0.508
RGa10 3 7 – 0.535 0.92 0.728
RG-SCb10 5 5 25 0.790 0.95 0.870
Hybrid Base 4 – – – 0.595 0.01 0.303
RG 5 1 4 – 0.380 0.74 0.560
RG-SC 7 3 4 25 0.585 0.71 0.648
Sparse Base 2 – – – 0.250 0.00 0.125
RG 3 1 2 – 0.165 0.15 0.158
RG-SC 4 2 2 25 0.250 0.15 0.200
aRG: RAGuard,bSC: SafetyClamp
The hyperparameter sweep clearly illustrates the trade-offs inherent in each
pipeline. Base RAG maximises knowledge recall; Dense achieves nearly 93%
correct technical retrieval when K= 10, but at the cost of almost zero safety
coverage. Introducing RAGuard dramatically raises the safety recall to over
90%, yet reduces the knowledge recall by roughly half, since only three slots are

12 C. Walker et al.
reserved for technical content. RAGuard with SafetyClamp, by contrast, finds
a middle ground: by over-retrieving and then guaranteeing both a proportional
number of knowledge and safety passages (e.g. kknow = 5, ksafe= 5for Dense
+ RAGuard and SafetyClamp), it retains high safety recall (95%) while still
preserving a strong knowledge recall (79%), yielding the highest combined score
(0.87).
Hybrid pipelines behave similarly but start from lower base knowledge recall,
and sparse pipelines–inherently limited by BM25’s coarse matching-cannot exceed
25% knowledge retrieval even when safety is ignored.
Overall, RAGuard with SafetyClamp consistently delivers the best balance,
particularly under dense retrieval, by ensuring that neither technical accuracy
nor mandated safety context are sacrificed.
5.2 Retrieval Recall@K
Figure 2 visualises each pipeline’s performance in the technical-vs-safety recall@
K plane, using colour to denote the family (orange: Dense, red: Hybrid, green:
Sparse) and distinct markers to indicate the retrieval method (’O’: Base RAG,
’X’: RAGuard, ’ ■’: RAGuard with SafetyClamp).
Fig. 2.Recall trade-off: Technical Recall@K vs Safety Recall@K
The plot reveals a clear three-way trade-off across methods and families. Within
each colour band, the Base RAG point sits farthest to the right-maximising
technical recall-but remains near the bottom with almost zero safety recall.
RAGuard lifts each family sharply upward: for Dense, the jump from Base (0.925,
0.0425) to Dense+RAGuard (0.665, 0.5175) illustrates how interleaving yields
large safety gains at the expense of roughly 26 percentage points (pp) of technical
recall. SafetyClamp occupies the middle ground, for example, Dense + RAGuard
with SafetyClamp at (0.790, 0.4375) recovers most of the Base technical coverage
while still boosting safety recall by 39 pp. Hybrid and Sparse variants follow the

Title Suppressed Due to Excessive Length 13
same pattern; SafetyClamp always improves safety over Base with only a modest
drop in technical retrieval, and RAGuard pushes safety even further up.
Overall, SafetyClamp consistently dominates Base RAG in safety without
sacrificing too much technical accuracy, and RAGuard sits at the front of the
Pareto curve when safety is paramount.
5.3 Safety Compliance Recall
Figure 3 shows each pipeline’s Safety Compliance Recall, that is, the fraction of
queries for which allannotated PUWER and WAHR passages were successfully
retrieved. The horizontal axis lists the nine methods, and the vertical axis gives
the compliance rate from 0 to 1. Each bar’s height corresponds exactly to the
Safety Compliance Recall metric: for example, the "Dense + RAGuard" bar at
0.07 indicates that only 7% of queries retrieved every required safety excerpt
under that configuration.
Fig. 3.Safety Compliance Recall by Pipeline
Despite improvements in the single-clause safety recall as presented earlier, full
compliance rates remain very low for all pipelines. The best performer, Dense
+ RAGuard, achieves just 7% compliance, while its SafetyClamp counterpart
achieves 6%. Hybrid and Sparse variants all fall at or near 0%, meaning they
never retrieve everymandated safety clause in a single pass.
These results underscore a critical gap: even the most safety-focused retrieval
strategies still omit at least one required regulation clause in over 90% of cases.
To move toward reliable compliance in safety-critical contexts, future work must
explore higher safety-slot budgets, multi-pass retrieval until exhaustively covered,
or targeted post-retrieval verification steps.
Latency and Context Utilisation Figure 4 illustrates the trade-off between re-
trieval latency and context utilisation across our three RAG families and retrieval
methods. Each plot focuses on one family, and plots the average retrieval time

14 C. Walker et al.
on the vertical axis against the fraction of the LLM’s context window occupied
by retrieved passages on the horizontal axis. Within each plot, a circle denotes
the Base RAG, an "X" denotes RAGuard, and a square marks SafetyClamp; all
markers are outlined in black, with the error bars in the family’s colour showing
±1 standard deviation.
Fig. 4.Latency vs Context Utilisation for a 4K Model
Across all families, the Base RAG method sits at the leftmost (lowest con-
text use) and lowest latency point. Introducing RAGuard shifts each marker
slightly to the right—because it interleaves safety passages—at the cost of a
modest increase in retrieval time (roughly 0.4-0.6 ms extra). SafetyClamp moves
further right, due to its larger kfetch, and imposes a further latency penalty; it
still completes retrieval in under 3 ms for Dense, under 4.5 ms for Hybrid and
under 7.5 ms for Sparse. Thus, the plot makes clear the Pareto front of methods:
if the primary goal is minimal latency, and you can tolerate absence of safety
guarantees, Base RAG is optimal; if you require safety integration, RAGuard and
SafetyClamp offer progressively stronger safety coverage at predictable, bounded
increases in retrieval time.
6 Conclusion
This work introduced RAGuard, an enhanced RAG framework for OSW mainte-
nance that integrates safety-critical content with technical documentation. By
using parallel indices and separate kknowandksafequotas, RAGuard ensures
recommendations are grounded in domain expertise and validated safety proto-
cols. We also proposed SafetyClamp, an over-retrieve and hard-clamp variant
that guarantees slots for technical and safety passages even when an index is
sparse.
Our evaluation, conducted on a curated dataset of 100 real-world OSW main-
tenance queries, showed that both RAGuard and SafetyClamp substantially

Title Suppressed Due to Excessive Length 15
outperform standard RAG in surfacing mandated safety clauses. Specifically,
Safety Recall@K increased from near 0% (Base RAG) to over 50% (Dense RA-
Guard), with only modest reductions in Technical Recall@K. In hyperparameter
sweeps, we found optimal configurations that balance technical fidelity and safety
coverage. Latency measurements on a 13th-gen i9 laptop showed that these gains
incur only a small retrieval overhead while offering very low context utilisation
fractions, leaving ample room in LLM windows. Overall, RAGuard and its Safe-
tyClamp extension provide a principled, lightweight mechanism for embedding
safety guarantees directly into RAG pipelines, offering practical value in regulated
high-stakes environments.
Future work includes several directions. First, we will carry out additional
hyperparameter experiments, varying document chunk size and overlap to find
the optimal indexing strategy. Second, we plan to integrate more regulatory
and technical documents, reflecting the multiple standards and manuals used
in real-world operations to ensure our system scales to complex scenarios. We
will further investigate adaptive slot-sizing methods that adjust kknowandksafe
based on the complexity of each query. Finally, we will study how different
retrieval configurations influence the quality and safety of the LLM’s generated
responses. Longer term, we aim to run live field trials to measure the end-to-end
effects on maintenance decision accuracy, operational efficiency, and overall safety
outcomes, gaining vital expert feedback.
Acknowledgments. This work was conducted under the Aura CDT program, funded
by EPSRC and NERC, grant number EP/S023763/1 and project reference 2609857.
References
1.Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-rag: Learning to retrieve,
generate, and critique through self-reflection (2023)
2.Demir, G., Moslem, S., Duleba, S.: Artificial intelligence in aviation safety: Systematic
review and biometric analysis. International Journal of Computational Intelligence
Systems 17(1), 279 (2024)
3.Habli, I., Lawton, T., Porter, Z.: Artificial intelligence in health care: Accountability
and safety. Bulletin of the World Health Organization 98(4), 251–256 (2020)
4.Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu,
J., Joulin, A., Grave, E., Riedel, S.: Atlas: Few-shot learning with retrieval augmented
language models. Journal of Machine Learning Research 24(251), 1–43 (2023)
5.Jendoubi, C., Asad, A.: A survey of artificial intelligence applications in nuclear
power plants. IoT 5(4), 666–691 (2024)
6. Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J.,
Neubig, G.: Active retrieval augmented generation (2023)
7.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler,
H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation
for knowledge-intensive nlp tasks. In: Advances in Neural Information Processing
Systems. vol. 33, pp. 9459–9474 (2020)
8.Liang, X., et al.: Saferag: Benchmarking security in retrieval-augmented generation
of large language model (2025)
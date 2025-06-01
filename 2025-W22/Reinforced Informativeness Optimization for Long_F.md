# Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang

**Published**: 2025-05-27 07:34:41

**PDF URL**: [http://arxiv.org/pdf/2505.20825v1](http://arxiv.org/pdf/2505.20825v1)

## Abstract
Long-form question answering (LFQA) presents unique challenges for large
language models, requiring the synthesis of coherent, paragraph-length answers.
While retrieval-augmented generation (RAG) systems have emerged as a promising
solution, existing research struggles with key limitations: the scarcity of
high-quality training data for long-form generation, the compounding risk of
hallucination in extended outputs, and the absence of reliable evaluation
metrics for factual completeness. In this paper, we propose RioRAG, a novel
reinforcement learning (RL) framework that advances long-form RAG through
reinforced informativeness optimization. Our approach introduces two
fundamental innovations to address the core challenges. First, we develop an RL
training paradigm of reinforced informativeness optimization that directly
optimizes informativeness and effectively addresses the slow-thinking deficit
in conventional RAG systems, bypassing the need for expensive supervised data.
Second, we propose a nugget-centric hierarchical reward modeling approach that
enables precise assessment of long-form answers through a three-stage process:
extracting the nugget from every source webpage, constructing a nugget claim
checklist, and computing rewards based on factual alignment. Extensive
experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the
effectiveness of the proposed method. Our codes are available at
https://github.com/RUCAIBox/RioRAG.

## Full Text


<!-- PDF content starts -->

Reinforced Informativeness Optimization for
Long-Form Retrieval-Augmented Generation
Yuhao Wang1∗†Ruiyang Ren1∗Yucheng Wang2Wayne Xin Zhao1‡
Jing Liu2‡Hua Wu2Haifeng Wang2
1Gaoling School of Artificial Intelligence, Renmin University of China
2Baidu Inc.
{yh.wang500, reyon_ren}@outlook.com, batmanfly@gmail.com
Abstract
Long-form question answering (LFQA) presents unique challenges for large lan-
guage models, requiring the synthesis of coherent, paragraph-length answers.
While retrieval-augmented generation (RAG) systems have emerged as a promis-
ing solution, existing research struggles with key limitations: the scarcity of
high-quality training data for long-form generation, the compounding risk of hal-
lucination in extended outputs, and the absence of reliable evaluation metrics for
factual completeness. In this paper, we propose RioRAG, a novel reinforcement
learning (RL) framework that advances long-form RAG through reinforced infor-
mativeness optimization. Our approach introduces two fundamental innovations
to address the core challenges. First, we develop an RL training paradigm of
reinforced informativeness optimization that directly optimizes informativeness
and effectively addresses the slow-thinking deficit in conventional RAG systems,
bypassing the need for expensive supervised data. Second, we propose a nugget-
centric hierarchical reward modeling approach that enables precise assessment of
long-form answers through a three-stage process: extracting the nugget from every
source webpage, constructing a nugget claim checklist, and computing rewards
based on factual alignment. Extensive experiments on two LFQA benchmarks
LongFact and RAGChecker demonstrate the effectiveness of the proposed method.
Our codes are available at https://github.com/RUCAIBox/RioRAG .
1 Introduction
Long-form question answering (LFQA) tasks require generating elaborate, multi-sentence answers
by conditioning generative language models (GLMs) on the input queries. Retrieval-augmented
generation (RAG) has emerged as a compelling paradigm for such knowledge-intensive tasks, as it
combines pre-trained parametric models with non-parametric memory ( e.g., document retrieval) to
ground generation in factual content [ 1]. Unlike conventional QA, long-form RAG systems must
synthesize information from multiple sources into coherent, paragraph-length answers.
In principle, augmenting GLMs with retrieved contents can mitigate the hallucination and knowledge
cutoff issues of closed-book models, leading to more factual and diverse outputs. However, in
practice, generating long and reliable responses remains difficult. State-of-the-art large language
models (LLMs) can produce impressively fluent narratives, but their LFQA capabilities are still
surprisingly impressive yet brittle, even grounded generators often insert unsupported statements or
∗Equal contributions.
†The work was done during the internship at Baidu.
‡Corresponding authors.
Preprint. Under review.arXiv:2505.20825v1  [cs.CL]  27 May 2025

misconstrue evidence [ 2,3]. This limitation partly stems from conventional RAG approaches lacking
systematic mechanisms to cultivate deliberate reasoning processes ( i.e.,slow thinking) [ 4] that are
crucial for multi-step evidence integration and long chain-of-thought reasoning. Advances in this
field are also constrained by the lack of standardized evaluation protocols, owing to the inherent
difficulties in quantifying long-form generation quality.
In summary, long-form RAG often contends with three critical challenges: (i) scant supervision,
since high-quality, human-labeled long answers with evidence annotations are expensive to collect;
(ii) hallucination and factual inconsistency, longer outputs amplify the risk of mixing unsupported
content and generation inconsistency; and (iii) evaluation difficulty, as conventional metrics fail
to capture correctness or completeness over multi-sentence answers. These challenges hinder the
development of LFQA systems that can be reliably deployed in knowledge-intensive scenarios.
Typically, recent studies have explored complementary approaches to improve long-form RAG. LLMs
can effectively leverage multiple in-context documents to some extent [ 5]. Specialized datasets ( e.g.,
ELI5 [ 6]) support training of generative LFQA models, but such resources are limited in scope
and often yield noisy supervision. Reinforcement learning (RL) techniques have been applied in
related settings ( e.g., summarization and QA) to optimize for factuality or user preferences. For
instance, RL with textual-entailment-based rewards has improved the faithfulness of abstractive
summaries [ 7], and axiomatic preference models have been proposed to align long-form answers
with human judgments [ 8]. However, these prior methods do not fully exploit the structure of LFQA,
they rarely integrate retrieval and generation in a unified optimization, nor do they explicitly reward
the inclusion of diverse evidence. Concurrent efforts have also highlighted the importance of atomic
content units for evaluating long answers [ 9]. Recent RAG pipelines for long-form generation have
begun leveraging nugget-level information to cluster and filter evidence [ 3]. Nonetheless, existing
work only uses heuristic information extraction in the RAG pipeline, without optimizing generation
end-to-end for maximal fact coverage.
To address these gaps, we introduce RioRAG, a Reinforced Informativeness Optimization-based RL
method for long-form RAG that is designed to directly optimize generation towards informative, fact-
rich answers. First, we propose the reinforced informativeness optimization framework that maximizes
information coverage through RL training, which directly incorporates long-form generation quality
assessment into the core RAG model training process. By formulating long-form answer generation
as a sequential decision-making process with delayed reward signals, our method activates the slow-
thinking capability of the long-form RAG model, enabling systematic evidence synthesis through
learned chain-of-thought reasoning patterns. Our RL approach eliminates dependence on scarce
supervised data while avoiding the computational overhead of multiple inference passes or reflective
reasoning during deployment. Second, we propose nugget-centric hierarchical reward modeling , a
three-stage evaluation approach for calculating long-form generation as the reward based on nuggets
refined from the retrieved webpages. Such a reward modeling approach addresses the persistent
challenge of evaluating extended text generation, and ensures the implementation of length-adaptive
reward computation that handles multi-document inputs without sequence truncation.
Extensive experiments on two published benchmarks, LongFact and RAGChecker, with zero-shot
evaluation show that the RioRAG achieves superior performance compared with a series of state-of-
the-art methods, demonstrating the effectiveness of the proposed innovations.
2 Related Work
2.1 Long-Form Question Answering
Research on long-form question answering (LFQA) has evolved through three paradigm shifts: from
fine-tuned generative language models to retrieval-augmented architectures, and more recently to
human-aligned large language models (LLMs). Early foundational work established the feasibility of
abstractive approaches, with the ELI5 dataset [ 6] demonstrating that sequence-to-sequence models
leveraging retrieved evidence could generate plausible answers. Recent efforts leverage LLMs and
human feedback, which employs reinforcement learning from human preferences to generate answers
supported by explicit quotations [ 10]. Comparative studies show that LLMs like ChatGPT outperform
smaller open-source LLMs on long-context questions, although they still struggle with very long
inputs [ 11]. Retrieval-augmented generation has emerged as a dominant strategy to enhance factual
grounding of LFQA. Several studies aim to fine-tune models in real-world web-browsing scenarios
2

in text-based web-browsing interfaces [ 12,13]. Recent work demonstrates that explicit source
citation during training significantly improves answer verifiability [ 10], while post-hoc attribution
methods enable verification of pre-generated text [ 14,15]. Due to the inherent characteristics of
LFQA, faithfulness has constitute as an important consideration for ensuring that long answers are
faithful to evidence [ 2,16]. It has been shown that training an open-book QA model to cite sources
greatly improves trustworthiness [ 10]. Alternatively, model confidence in long-form answers can be
calibrated by treating answer correctness within a probabilistic calibration framework [ 17]. Moreover,
LLM-based evaluation can be applied to assess LFQA quality [18] automatically.
2.2 Reinforcement Learning based Retrieval-Augmented Generation
Reinforcement learning (RL) has shown great potential in optimizing RAG systems by dynamically
adjusting retrieval and generation strategies. Central to this approach is the integration of policy-
gradient methods with tailored reward functions. Early effort refines behavior-cloning pretraining of
web-browsing agents via human-feedback–based rewards to improve factual alignment [ 12]. Fine-
grained reward models are also designed to quantify coherence and information gain at multiple
semantic levels [ 19], or fine-tuned for specific retrieval domains to align the RAG model to human
performance by generating synthetic training data [ 20]. In addition, composite reward ensembles
balance generation quality against coverage objectives [ 21]. In the retrieval side, retrieval modules
themselves can be tuned with group-wise relative policy optimization [ 22] or multi-agent coordina-
tion [ 23]. Cost-sensitive retrieval policies further leverage learned value estimators to decide when to
invoke external search, excluding latency and informational utility [ 24]. Recent advances, particularly
the success of the DeepSeek-R1 [ 4] model, have spurred research interest in employing RL to endow
LLMs with the capability of autonomously invoking retrieval during extended reasoning chains,
thereby enhancing the intelligence of the entire RAG process [ 25]. In contrast to these methods,
our proposed RioRAG framework eliminates the need for supervised training data by introducing
nugget-centric hierarchical reward modeling, enabling reinforcement learning to directly optimize
informativeness in long-form RAG and achieve long-CoT reasoning.
3 Method
3.1 Task Formulation
Long-form RAG extends the conventional RAG paradigm to address the challenges of generating
coherent, factually grounded, and in-depth textual responses ( e.g., detailed explanations, reports, or
multi-paragraph answers). Given a query q(e.g., a natural question or open-ended prompt requiring
extended reasoning) and a document corpus D={d1, d2, . . . , d N}, where each diis a retrievable
textual unit, a retriever Rretrieves relevant documents Dq⊆ D forq, with an LLM-based generator
Gconditioned on both qandDq. The focus of our work lies in enhancing the generator’s capability
in long-form RAG, where retrieving relevant documents is treated as an independent preprocessing
module. Given that offline document corpora are often limited by document diversity and timeliness,
we employ web search engines to obtain Dqfor each query, where each direpresents a document
text extracted from a webpage.
3.2 RioRAG Overview
To address the limitations of conventional RAG systems in generating long-form, factually grounded,
and coherent responses, we introduce the reinforced informativeness optimization framework for long-
form RAG (RioRAG) that integrates reinforcement learning (RL) with hierarchical reward modeling.
Our method systematically tackles three core challenges inherent in long-form RAG: (1) The scarcity
of high-quality supervised data for extended generation tasks. (2) The susceptibility of hallucination
or incoherence resulting from incomplete or conflicting information across heterogeneous web
sources. (3) There are inherent limitations in current long-generation evaluation, where manual
assessment of lengthy texts is labor-intensive and time-consuming. Figure 1 conceptually illustrates
the proposed framework.
Given a query q, we bypass conventional static corpora and directly retrieve web documents through
search engine APIs. This ensures access to timely, diverse, and comprehensive reference materials,
addressing the document staleness problem inherent in offline RAG systems. Based on this, we
3

LLMs                                     Long-form Response 
User Query：How do we 
know all the money the 
government is getting from 
bank settlements is going 
back to the people?
①Nugget Extraction
②Checklist Integration
[Response 2] High Volume, Low Density 
… toward a variety of purposes, including victim compensation,
regulatory enhancements, and public programs. For example, …
[Response 3] Low Quality or Misleading
Yes, the government always returns settlement money to the people. 
It’s a standard practice and fully transparent.[Response 1] Concise yet Comprehensive
Settl ement fund use isn’t always transparent. Some money goes to 
restitution or relief programs, but much is sent to the general fund.
deduplicatesplit extract
Web Search
Web pages  Document  Nug gets 
Nuggets  Checklist
Action Activation
Search Engine  Web pagesSearch
③Informative RewardQuery      Web Pages
Response   Checklist  Score
evaluateNugget-Centric Hierarchical Reward Modeling
Score Length-Decay⊗
࢘࢏
Response 1     high recall
Response 3  low recallhigh ࢘૚
medium ࢘૛ Response 2     low density
low ࢘૜
RewardFigure 1: Overall illustration of the proposed RL-based RioRAG framework.
implement reinforced informativeness optimization for unsupervised training. The optimization
objective directly maximizes the expected information coverage reward while preserving generation
coherence through relative advantage estimation.
Another core of our approach lies in nugget-centric hierarchical reward modeling . Our hierarchical
reward model Rdecomposes the complex assessment of long-form generation quality into three
interpretable stages: (i) fine-grained nugget extraction from individual webpages, (ii) cross-webpage
checklist synthesis through information aggregation, and (iii) generation evaluation against the consol-
idated checklist. This hierarchical approach overcomes the performance limitations of conventional
reward models while maintaining awareness of cross-webpage dependencies.
In the RioRAG framework, the RAG model employs RL optimization without relying on supervised
training data, effectively circumventing the data scarcity challenge inherent in long-form RAG
applications. Through the implementation of reinforced informativeness optimization with nugget-
centric hierarchical reward modeling, our RL framework enables precise reward evaluation for
long-form text generation based on informativeness during RL training. In this manner, the RAG
model can systematically synthesize comprehensive web-sourced information, thereby demonstrating
enhanced capability in producing coherent, substantively rich long-form responses that preserve
factual integrity throughout extended textual sequences.
3.3 Reinforced Informativeness Optimization
Considering the challenges of hallucination and scarcity of supervised data in long-form RAG, we
employ RL to enhance the generation quality of long-form RAG. However, a key challenge lies in
the difficulty of quantitatively evaluating long-form generation results since it cannot be assessed
by simple term-match metrics ( e.g., exact match). To address this, we propose the Reinforced
Informativeness Optimization (Rio) framework that introduces informativeness as an optimization
objective during RL. By designing a specialized reward model, we quantitatively assess the coverage
of critical information ( i.e.,nuggets) from retrieved documents in the generated responses.
For RL training data, we exclusively utilize query data from the ELI5 [ 6] dataset while deliberately
excluding its human-annotated answers. These annotations exhibit a significant gap from real-world
web-based RAG scenarios, as they tend to be overly concise and lack the heterogeneous external
information integration characteristic of actual retrieval-augmented systems.
To ensure the stability of RL training, we employ Group-wise Relative Policy Optimiza-
tion (GRPO) [ 26] as our foundational RL algorithm. The algorithm’s constrained policy updates
prevent degradation during fine-tuning, while its integrated advantage estimation optimally balances
our information coverage reward with generation diversity. The GRPO algorithm samples Gcomple-
tions{o1, o2, . . . , o G}and computes their rewards {r1, r2, . . . , r G}considering the informativeness
of the completions:
ri=I(q, oi, Dq). (1)
4

The relative advantage incorporates the informativeness-based reward:
Ai=ri−µr
σr+ϵ, (2)
where µr=1
GPG
j=1rjandσr=q
1
GPG
j=1(rj−µr)2represent the mean and standard deviation
of group rewards, respectively, and ϵis a small constant for numerical stability. Finally, the GRPO
objective becomes:
L(θ) =Eq,{oi}
minπθ(oi|q)
πθold(oi|q)Ai,clip(πθ(oi|q)
πθold(oi|q),1−ϵ,1 +ϵ)Ai
−βD KL(πθ∥πref),(3)
where clip(·)constrains policy updates to prevent destabilization, and βcontrols the strength of KL
regularization against reference model πref.
Action Activation. To enhance the stability of RL processes, we adopt a Markdown-based action
activation training strategy. Traditional approaches typically employ special formatting tags ( e.g.,
<think> and</think> ) to constrain reasoning format during generation. However, such artificial
syntax structures may induce model optimization toward format compliance rather than substantive
reasoning. We address this limitation by leveraging the inherent hierarchical structure of Markdown
formatting to naturally generate responses. This paradigm shift not only enhances the quality of
generated actions but also improves training stability.
3.4 Nugget-Centric Hierarchical Reward Modeling
According to the proposed reinforced informativeness optimization framework, effective reward
modeling for long-form RAG requires addressing two critical challenges: (1) maintaining cross-
webpage coherence while avoiding redundancy or contradiction, and (2) processing exceptionally long
input sequences from multiple web sources. Traditional approaches that directly apply reward models
to concatenated multiple documents [ 21] suffer from performance degradation due to extremely large
sequence length and ineffective cross-webpage dependency modeling.
To overcome these challenges, we propose a nugget-based hierarchical reward modeling approach
that decomposes the reward computation into three hierarchical stages:
Stage 1: Nugget-wise Information Extraction. For each retrieved web document diassociated
with query q, we employ the reward model Rto identify salient information nuggets {Nk}|Dq|
1by
the tailored prompt p1:
Nk=R(D(k)
q, q, p 1). (4)
Stage 2: Cross-webpage Checklist Integration. We employ the reward model Rto aggregate
extracted nuggets across |Dq|documents by another prompt p2:
C=R 
{Nk}, q, p 2
. (5)
Stage 3: Generation Informativeness Assessment. The reward model Revaluates generated
response oiagainst the consolidated checklist C, which evaluates the informativeness of the response
by prompt p3:
si=R(oi, C, q, p 3). (6)
Length Decay. We observe that the length of generation in RL training exhibits a tendency toward
progressive elongation during extended training periods, which demonstrates non-trivial implications
for the performance. As a result, we incorporate an adaptive penalty term for the reward to regulate
response verbosity in RAG tasks. The final penalty-based reward function is formally defined as:
ri=I(q, oi, Dq) =(
si·exp
−k l−l0
τm
,ifl > l 0
si, otherwise(7)
where lrepresents the response length, l0denotes the predefined length threshold, τserves as a
normalization constant corresponding to the maximum context window size, kcontrols the penalty
5

intensity, and mdetermines the curvature of the penalty function. This non-linear attenuation
mechanism introduces progressive penalization for responses exceeding the target length l0, where
the exponential decay factor (l−l0
τ)mcreates a smooth but rapidly intensifying penalty gradient.
The proposed hierarchical reward modeling framework enables a more precise extraction of nuggets
from reference webpages, yielding accurate informativeness rewards that effectively guide the
RL process. Since we process distinct webpage contents separately, there is no net increase in
computational complexity.
4 Experiment
In this section, we detail the experimental setup, present the main results, and further support our
findings with ablation studies and in-depth analysis.
4.1 Experimental Setup
4.1.1 Datasets
Fortraining data, we utilize the ELI5 dataset [ 6], a widely used benchmark for long-form QA,
but implement a novel data protocol that exclusively employs its query corpus without answer
references. This design comes from the fact that the annotations tend to be overly concise and lack the
heterogeneous external information integration characteristic of actual retrieval-augmented systems.
We randomly sample 10K questions for RL training.
Ourevaluation is conducted on two purpose-built benchmarks. LongFact [ 27] is a manually cu-
rated dataset of complex questions spanning 38 knowledge domains, where each answer requires
synthesizing several factual claims from diverse sources. For clarity of presentation, the original 38
domains have been consolidated into eight broader categories. The answers are annotated with atomic
information points for granular factual verification. To comprehensively evaluate the performance of
RioRAG, we conducted a multidimensional assessment using the RAGChecker [ 28] benchmark. It
provides 8 evaluation metrics across three dimensions, and contains queries spanning 10 domains,
which are repurposed from public datasets. The short answers of the original public datasets are
converted into long-form responses to align with the evaluation needs of contemporary RAG systems.
4.1.2 Evaluation Metrics
Following existing LFQA studies [ 6], we employ the LLM-as-judge for evaluation, utilizing fact
recall (FR) and information density (ID) as key metrics. Fact recall measures the proportion of atomic
facts present in the generated response relative to those present in the ground-truth answer, while
information density is defined as the ratio of atomic facts in the response to its total length.
Furthermore, we conduct comprehensive evaluations using the RAGChecker benchmark, which
incorporates a sophisticated set of metrics [ 28]:faithfulness measures the proportion of atomic facts
in the response that are substantiated by retrieved webpages; relevant noise sensitivity quantifies
the ratio of incorrect atomic facts in the response that appears in retrieved webpages; irrelevant
noise sensitivity assesses the proportion of correct atomic facts in the response that is present in
retrieved webpages; hallucination represents the probability of incorrect atomic facts in the response
not appearing in any retrieved webpages; self-knowledge indicates the proportion of correct atomic
facts in the response absent from all retrieved webpages; context utilization calculates the ratio of
atomic facts from the ground truth answer that are covered by retrieved webpages.
4.1.3 Baselines
To evaluate the performance of RioRAG, we conduct comprehensive comparisons with various
classical and state-of-the-art baseline methods across different categories, ensuring a thorough
understanding of the proposed approach. The baselines are categorized into three groups based on
their training paradigms: prompt-based unsupervised methods, supervised fine-tuning (SFT)-based
approaches, and RL-based techniques. For prompt-based methods, we select GopherCite [ 10], chain-
of-thought [ 29] and chain-of-note [ 30]. Among SFT-based approaches, we employ chain-of-note
and GopherCite with the SFT setting. For RL-based methods, we adopt the Direct Preference
6

Table 1: The results on eight broader categories of LongFact benchmark with the average results of
the eight categories, where FR denotes fact recall and ID denotes information density.
MethodScience Tech. Medicine Law Culture Events Commun. Lifestyle Average
FR ID FR ID FR ID FR ID FR ID FR ID FR ID FR ID FR ID
Prompt-based Methods
Direct-RAG 45.3 54.2 60.6 69.3 45.7 61.1 46.9 31.4 45.4 55.4 51.2 44.3 55.5 60.2 48.7 69.8 49.6 53.5
Chain-of-Thought 55.5 51.6 53.7 71.8 46.4 58.8 45.2 30.8 47.1 61.2 48.5 46.3 54.6 59.0 54.6 70.8 50.6 54.1
Chain-of-Note 45.0 52.6 58.4 71.5 43.6 58.7 42.3 27.1 40.9 56.9 47.7 46.5 47.8 56.9 48.8 71.0 46.0 52.5
GopherCite 54.4 56.6 63.8 73.9 56.2 54.1 55.6 33.7 53.3 60.6 48.9 46.4 61.5 64.0 54.1 72.5 55.9 56.1
Supervised Fine-tuning based Methods
Chain-of-Note 65.3 123.3 50.0 129.9 77.4 130.6 57.5 93.8 67.5 134.3 52.5 80.2 64.8 147.8 64.6 114.7 62.2 119.5
GopherCite 59.2 121.5 58.4 145.5 74.4 118.8 59.3 80.2 69.0 146.0 68.4 101.3 61.8 144.1 60.0 103.5 63.2 119.7
RL-based Methods
DPO 59.0 106.7 66.2 137.0 60.7 96.4 65.7 61.8 69.2 115.0 56.6 83.2 60.5 122.9 61.3 113.9 62.8 102.7
RioRAG 69.7 146.7 63.3 170.4 77.4 142.1 77.9 113.4 78.0 120.4 71.6 117.7 75.2 170.7 61.5 144.9 72.8 138.8
Table 2: Average results across ten domains on the RAGChecker benchmark. Fact-Rec refers to
fact recall, Info-Den to information density, Cont-Util to context utilization, Rel-NS and Irrel-NS to
relevant and irrelevant noise sensitivity, Hallu. to hallucination, Self-Know to self-knowledge, and
Faith. to faithfulness.
Method Fact-Rec ↑Info-Den ↑Cont-Util ↑Rel-NS ↓Irrel-NS ↓Hallu. ↓Self-Know ↓Faith. ↑
Prompt-based Methods
Direct-RAG 38.3 91.6 22.6 8.2 7.5 37.0 8.1 45.3
Chain-of-Thought 50.4 146.5 24.3 4.6 4.2 30.2 9.7 48.0
Chain-of-Note 38.7 144.3 18.3 6.8 5.1 53.0 6.9 35.7
GopherCite 51.4 138.5 26.0 5.1 4.3 29.2 10.8 47.5
Supervised Fine-tuning based Methods
Chain-of-Note 54.2 190.2 22.7 4.3 3.7 22.6 7.8 30.2
GopherCite 62.6 209.9 26.0 5.1 4.3 29.2 10.8 52.5
RL-based Methods
DPO 61.2 149.6 26.0 5.2 6.0 27.8 8.0 53.1
RioRAG 66.3 224.6 27.8 4.3 3.6 20.9 5.0 58.2
Optimization (DPO) [ 31] framework. All baseline implementations are manually reimplemented
with rigorous adherence to identical experimental configurations to ensure a fair comparison. This
evaluation protocol guarantees the reliability of performance benchmarking while controlling for
potential confounding factors in implementation differences.
4.2 Main Results
The results of different methods evaluated on LongFact and RAGChecker are shown in Table 1 and
Table 2. It can be observed that:
(1) Our comprehensive evaluation reveals that SFT-based baselines substantially outperform prompt-
based approaches, demonstrating the inherent limitations of prompt engineering in handling complex
information synthesis tasks. The proposed RioRAG framework establishes a significant improvement
across all metrics. This improvement stems from the reinforced informativeness optimization
paradigm, which implements a nugget-centric hierarchical reward mechanism to guide LLMs in
processing long-context inputs.
(2) Comprehensive evaluation on RAGChecker demonstrates that RioRAG excels in long-form RAG
tasks across multiple critical dimensions, including knowledge point coverage, information density,
retrieval utilization, hallucination mitigation, and internal knowledge integration. These results
underscore the multidimensional efficacy of the proposed approach.
(3) Compared to off-line RL-based methods such as DPO, RioRAG demonstrates superior perfor-
mance in long-form reasoning tasks. By leveraging an enhanced on-policy GRPO algorithm, RioRAG
enables more comprehensive exploration of potential reasoning strategies during generation, thereby
optimizing the RAG model more effectively through informativeness-driven reward feedback.
7

Table 3: Results of the RioRAG variants on LongFact.
MethodScience Tech. Medicine Law Culture Events Commun. Lifestyle Average
FR ID FR ID FR ID FR ID FR ID FR ID FR ID FR ID FR ID
RioRAG 69.7 146.7 63.3 170.4 77.4 142.1 77.9 113.4 78.0 120.4 71.6 117.7 75.2 170.7 61.5 144.9 72.8 138.8
w/o Info. Optim. 34.4 79.6 53.2 147.6 32.0 92.3 33.4 66.6 40.9 87.5 43.2 76.6 43.9 106.4 36.6 97.3 39.5 90.8
w/o Nugget Reward 36.0 77.6 56.6 119.3 23.8 62.5 36.0 42.4 37.2 80.7 30.7 65.4 36.8 83.2 40.1 112.7 37.0 77.3
w/o Length Decay 57.0 99.2 65.5 139.9 58.3 109.9 62.0 88.7 56.5 87.5 43.7 63.0 45.2 70.7 68.5 108.8 56.2 91.3
w/ Off-Line RL 60.2 106.5 66.0 160.3 37.9 65.3 56.6 71.7 62.2 113.0 55.9 73.8 54.0 102.3 63.0 101.0 57.7 98.1
w/ Off-Policy RL 41.6 41.4 61.7 65.5 34.1 35.0 44.8 25.1 46.8 52.2 46.4 39.8 56.2 53.1 61.8 60.2 49.3 45.5
4.3 Ablation Studies
In this section, we conduct an ablation study to evaluate the effectiveness of critical strategies
in RioRAG comprehensively on LongFact. Here, we consider five variants built on RioRAG for
evaluation: (a) w/o Info. Optim. removes the informativeness-based reward optimization during RL,
replaced by direct quality evaluation; (b) w/o Nugget Reward removes the nugget-wise information
extraction and use the full webpage for checklist integration; (c) w/o Length Decay eliminates the
length penalty in Equation (7); (d) w/ Off-Line RL utilizes an off-line RL method ( i.e.,DPO) for RL
training; (e) w/ Off-Policy RL utilizes an off-policy RL method that employs a static sampling strategy
wherein all queries are pre-processed through offline rollouts to generate complete trajectories before
being uniformly scored.
Table 3 presents the results for the variants of our method, from which we can observe the following
findings: (a) The performance drops in w/o Info. Optim. , demonstrating that using informativeness as
the objective for optimization enhances the performance of long-form RAG models through the guid-
ance of reasoning. (b) The performance drops in w/o Nugget Reward , demonstrating incorporating
nugget-wise information extraction enables the model to better capture core facts. (c) The perfor-
mance drops in w/o Length Decay , underscoring the critical role of incorporating the length penalty
in mitigating excessive response length. (d) The performance drops in w/ Off-Line RL , demonstrating
that the application of off-line reinforcement learning exhibits inherent limitations in solution space
coverage compared to online exploration paradigms. (e) The performance significantly drops in
w/ Off-Policy RL , demonstrating that the off-policy method may contain a mismatch between the
behavior policy and the target policy compared to the on-policy method GRPO.
4.4 Further Analysis
4.4.1 Scaling Law of RioRAG
To investigate the scalability characteristics of RioRAG, we conduct a systematic analysis using the
Qwen2.5 model with varying parameter sizes (1.5B, 7B, and 14B). As illustrated in Figure 2 (a), the
experimental results demonstrate that RioRAG significantly outperforms SFT at all model scales,
with performance consistently improving in accordance with scaling laws.
We can first observe that larger models exhibit improved semantic understanding for both query
formulation and webpage relevance assessment. Second, RioRAG benefits from increased model
capacity for learning sophisticated retrieval utilization strategies. Third, the enhanced generation
capability of larger models enables more effective utilization of retrieved webpages while reducing
hallucination risks through better alignment with the reward model’s feedback. Notably, the perfor-
mance growth curve shows a sublinear relationship between model size and metric improvements,
aligning with observations from language model scaling studies [ 32]. This phenomenon suggests
that while our RioRAG framework effectively leverages model scale, there exists an upper bound
where additional parameters may not proportionally improve RAG performance, which is a critical
consideration for practical system deployment.
4.4.2 Effect on RL Cold-Start
To systematically investigate the impact of model initialization on RL training dynamics, we conduct
controlled experiments using base model (Qwen2.5-7B-Base), instruction-tuned model (Qwen2.5-
7B-Instruct) with supervised fine-tuning (SFT), and (3) R1-Distilled-Qwen2.5-7B incorporating
8

1.5B 7B 14B40506070Fact-Recall (%)
SFT RioRAG
Base Instruct Distill304050607080Fact-Recall (%)Zero-shot RioRAG
(a) Effect on Scaling (b) Effect on RL Cold-StartFigure 2: In-depth analysis on scaling law and RL cold-start.
0 20 40 60 80 100 120 140Reward Length Information Density
0 20 40 60 80 100 120 140Reward Length Information Density
(a) w/ length decay (b) w/o length decay
Figure 3: Analysis on co-evolution of generation length and reward during RL training.
DeepSeek R1’s slow thinking distillation [ 4]. As demonstrated in Figure 2 (b), the instruction-tuned
model achieves 24.4% higher improvement, while the R1-distilled model exhibits the most significant
performance gain (29.6%) after RL training. Based on the observation, first, the base model’s limited
capacity for RL stems from inadequate prior alignment without SFT or long-CoT distillation, it
lacks fundamental instruction-following capabilities and structured reasoning patterns essential for
effective exploration in RL. Second, the superior performance of the R1-distilled model suggests
that slow-thinking architectures provide particularly favorable initialization for RL training and
chain-of-thought reasoning capabilities enable more stable reward estimation and credit assignment
during policy updates. Our findings corroborate the DeepSeek R1 technical report’s emphasis on
cold-start preparation, further suggesting that pre-established reasoning pathways act as inductive
biases that guide RL agents toward higher-reward regions of the policy space.
4.4.3 Co-Evolution of Generation Length and Reward
To investigate the dynamics of generation length control during RL training, we systematically
analyze the interaction between sequence length evolution and reward optimization. The experiments
are conducted on the Qwen model, with length decay activated when generated sequences exceed a
predefined threshold. Figure 3 illustrates the co-evolution of average generation length and reward
scores across training steps.
In the absence of length constraints, the model exhibits a clear tendency towards length inflation, while
the reward score stagnates within a narrow band. This phenomenon aligns with the reward hacking
hypothesis, where the policy learns to exploit reward model blind spots through verbosity rather
than genuine quality improvement. Subsequent linguistic analysis reveals a decrease in information
density, confirming the generation of redundant expressions. The introduction of length-decay term
produces markedly different co-evolution patterns. After reaching an exceeded length at initial steps,
the policy begins strategic length reduction while maintaining reward growth. This phase transition
suggests the model first explores the action space before learning to compress meaningful content
into concise responses. The information density indicates successful mitigation of verbosity-induced
quality degradation. Our findings underscore the necessity of explicit length regularization in our
RL pipelines, particularly for instruction-tuned models prone to verbosity. The co-evolution patterns
demonstrate that length control should not be treated as an independent constraint, but rather as an
integral component interacting dynamically with reward optimization.
9

5 Conclusion
In this work, we address long-form RAG limitations through RioRAG, an RL framework that
redefines long-form RAG training via reinforced informativeness optimization with nugget-centric
hierarchical reward modeling. RioRAG directly optimizes informativeness through a quantifiable
reward design for factual alignment, without the need for scarce training data. Our experiments on
two benchmarks demonstrate that RioRAG fundamentally improves the quality of long-form RAG.
By addressing the core challenges identified in long-form RAG, RioRAG advances the development
of trustworthy generative systems for real-world knowledge applications. Moreover, the success of
nugget-level reward modeling suggests that future evaluation frameworks for long-form tasks should
prioritize granular factual alignment over surface-level metrics. Limitations include the current focus
on English corpora and reliance on automatic nugget extraction, which may inherit biases from
pre-trained models. For future work, we will extend the framework to multilingual settings and
investigate human-in-the-loop reward refinement.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[2]Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to
generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 6465–6488, 2023.
[3]Weronika Łajewska and Krisztian Balog. Ginger: Grounded information nugget-based genera-
tion of responses. arXiv preprint arXiv:2503.18174 , 2025.
[4]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. CoRR , 2025.
[5]Hung-Ting Chen, Fangyuan Xu, Shane Arora, and Eunsol Choi. Understanding retrieval
augmentation for long-form question answering. In First Conference on Language Modeling ,
2024.
[6]Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5:
Long form question answering. In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics , pages 3558–3567, 2019.
[7]Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu
Geist, Sertan Girgin, Leonard Hussenot, Orgad Keller, et al. Factually consistent summarization
via reinforcement learning with textual entailment feedback. In Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages
6252–6272, 2023.
[8]Corby Rosset, Guoqing Zheng, Victor Dibia, Ahmed Awadallah, and Paul Bennett. Axiomatic
preference modeling for longform question answering. In Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing , pages 11445–11475, 2023.
[9]Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy
Lin. The great nugget recall: Automating fact extraction and rag evaluation with large language
models. arXiv preprint arXiv:2504.15068 , 2025.
[10] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chad-
wick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, et al. Teaching
language models to support answers with verified quotes. arXiv preprint arXiv:2203.11147 ,
2022.
[11] Meghana Moorthy Bhat, Rui Meng, Ye Liu, Yingbo Zhou, and Semih Yavuz. Investigating
answerability of llms for long-form question answering. arXiv preprint arXiv:2309.08210 ,
2023.
10

[12] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christo-
pher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted
question-answering with human feedback. arXiv preprint arXiv:2112.09332 , 2021.
[13] Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning
Ding, Huadong Wang, et al. Webcpm: Interactive web search for chinese long-form question
answering. In The 61st Annual Meeting Of The Association For Computational Linguistics ,
2023.
[14] Shufan Wang, Fangyuan Xu, Laure Thompson, Eunsol Choi, and Mohit Iyyer. Modeling
exemplification in long-form question answering via retrieval. In Proceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies , pages 2079–2092, 2022.
[15] Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng
Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. Rarr: Researching and revising
what language models say, using language models. In Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 16477–16508,
2023.
[16] Yilun Zhao, Lyuhao Chen, Arman Cohan, and Chen Zhao. Tapera: enhancing faithfulness
and interpretability in long-form table qa by content planning and execution-based reasoning.
InProceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 12824–12840, 2024.
[17] Yukun Huang, Yixin Liu, Raghuveer Thirukovalluru, Arman Cohan, and Bhuwan Dhingra.
Calibrating long-form generations from large language models. In Findings of the Association
for Computational Linguistics: EMNLP 2024 , pages 13441–13460, 2024.
[18] Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu, William Yang Wang,
Bonan Min, and Vittorio Castelli. Rag-qa arena: Evaluating domain robustness for long-form
retrieval augmented question answering. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pages 4354–4374, 2024.
[19] Tianchi Cai, Zhiwen Tan, Xierui Song, Tao Sun, Jiyan Jiang, Yunqi Xu, Yinger Zhang, and Jinjie
Gu. Forag: Factuality-optimized retrieval augmented generation for web-enhanced long-form
question answering. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining , pages 199–210, 2024.
[20] Thang Nguyen, Peter Chin, and Yu-Wing Tai. Reward-rag: Enhancing rag with reward driven
supervision. arXiv preprint arXiv:2410.03780 , 2024.
[21] Hanning Zhang, Juntong Song, Juno Zhu, Yuanhao Wu, Tong Zhang, and Cheng Niu. Rag-
reward: Optimizing rag with reward modeling and rlhf. arXiv preprint arXiv:2501.13264 ,
2025.
[22] Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Julia Hockenmaier, and Tong Zhang.
Rag-rl: Advancing retrieval-augmented generation via rl and curriculum learning. arXiv preprint
arXiv:2503.12759 , 2025.
[23] Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin,
Yiming Yang, and Jiaxin Mao. Improving retrieval-augmented generation through multi-agent
reinforcement learning. arXiv preprint arXiv:2501.15228 , 2025.
[24] Mandar Kulkarni, Praveen Tangarajan, Kyung Kim, and Anusua Trivedi. Reinforcement
learning for optimizing rag for domain chatbots. arXiv preprint arXiv:2401.06800 , 2024.
[25] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Za-
mani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning. arXiv preprint arXiv:2503.09516 , 2025.
[26] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
11

[27] Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Zixia Hu, Jie Huang, Dustin Tran,
Daiyi Peng, Ruibo Liu, Da Huang, et al. Long-form factuality in large language models. In The
Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
[28] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng
Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, et al. Ragchecker: A fine-grained framework
for diagnosing retrieval-augmented generation. Advances in Neural Information Processing
Systems , 37:21999–22027, 2024.
[29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[30] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang,
and Dong Yu. Chain-of-note: Enhancing robustness in retrieval-augmented language models.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 14672–14685, 2024.
[31] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model.
Advances in Neural Information Processing Systems , 36:53728–53741, 2023.
[32] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani
Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large
language models. Transactions on Machine Learning Research , 2022.
[33] Sara Rosenthal, Avirup Sil, Radu Florian, and Salim Roukos. Clapnq: C ohesive l ong-form a
nswers from p assages in natural questions for rag systems. Transactions of the Association for
Computational Linguistics , 13:53–72, 2025.
[34] Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng
Bao, Qian Wang, and Yue Zhang. Novelqa: A benchmark for long-range novel question
answering. arXiv e-prints , pages arXiv–2403, 2024.
[35] Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel
Zarrouk, and Alexandra Balahur. Www’18 open challenge: financial opinion mining and
question answering. In Companion proceedings of the the web conference 2018 , pages 1941–
1942, 2018.
[36] Fangyuan Xu, Kyle Lo, Luca Soldaini, Bailey Kuehl, Eunsol Choi, and David Wadden. Kiwi:
A dataset of knowledge-intensive writing instructions for answering research questions. arXiv
preprint arXiv:2403.03866 , 2024.
12

A Details on Datasets
In this section, we provide detailed descriptions of the two comprehensive benchmarks used in our
experiments: LongFact [27] and RAGChecker [28]. These datasets are designed to evaluate long-
form retrieval-augmented generation (RAG) systems across diverse topics and multiple dimensions
of factual quality. Their complementary nature enables robust assessment of both factual coverage
and fine-grained answer quality in open-domain settings.
LongFact. LongFact is a manually curated benchmark focused on evaluating long-form factuality. It
contains a diverse set of fact-seeking questions, where each gold answer synthesizes multiple atomic
facts drawn from various evidence sources. The dataset is notable for its broad coverage across 38
fine-grained domains, which are grouped into the following 8 broader categories to support structured
evaluation:
•Science & Nature : physics, chemistry, biology, astronomy, virology, prehistory
•Technology & Computing : computer science, computer security, machine learning, electri-
cal engineering, mathematics
•Medicine & Psychology : medicine, clinical knowledge, psychology, psychology checkpoint
•Law & Politics : international law, immigration law, U.S. foreign policy, jurisprudence
•Social Sciences & Culture : sociology, geography, world religions, moral disputes, philoso-
phy
•History & Events : history, 20th-century events, global facts, economics
•Business & Communication : business ethics, accounting, marketing, management, public
relations
•Entertainment & Lifestyle : movies, music, gaming, celebrities, architecture, sports
Each example in LongFact is annotated with atomic information units, enabling precise measurement
of factual recall and information density. This makes it especially well-suited for evaluating long-form
answers that integrate knowledge from multiple sources.
RAGChecker. RAGChecker is a comprehensive benchmark designed to evaluate long-form Retrieval-
Augmented Generation (RAG) systems across diverse domains. It repurposes examples from 10
public datasets, encompassing a total of 4,162 questions. For the 8 subsets we used, we briefly
describe their characteristics below:
•ClapNQ [33]: Derived from Natural Questions (NQ), ClapNQ includes long-form answers
with grounded gold passages from Wikipedia, focusing on generating cohesive long-form
answers from non-contiguous text segments.
•NovelQA [34]: NovelQA is a benchmark designed to evaluate large language models on
deep narrative understanding through complex questions based on English novels.
•FiQA [35]: A financial question answering dataset comprising 500 QA pairs, where short
answers are extended to long-form using GPT-4, filtered to remove hallucinations.
•KIWI [36]: A dataset of knowledge-intensive writing instructions for answering research
questions, comprising 71 QA pairs with long-form answers validated for quality.
B Implementation Details
All experiments were conducted on 8 NVIDIA H800 GPUs using bfloat16 precision to ensure efficient
memory usage and stable training. We trained for no more than 2 epochs with a batch size of 64.
The initial learning rate was set to 1×10−6, with a cosine learning rate schedule and a warm-up
phase covering 10% of the total training steps. No minimum learning rate ratio was specified. During
reinforcement learning, we used a rollout temperature of 0.9 and performed 8 rollouts per input to
stabilize exploration. These settings were chosen to balance convergence speed and policy diversity
under high-performance hardware constraints.
13
# Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data

**Authors**: Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hengyuan Zhang, Dongmei Zhang

**Published**: 2025-03-07 17:11:34

**PDF URL**: [http://arxiv.org/pdf/2503.05587v1](http://arxiv.org/pdf/2503.05587v1)

## Abstract
Robustness has become a critical attribute for the deployment of RAG systems
in real-world applications. Existing research focuses on robustness to explicit
noise (e.g., document semantics) but overlooks spurious features (a.k.a.
implicit noise). While previous works have explored spurious features in LLMs,
they are limited to specific features (e.g., formats) and narrow scenarios
(e.g., ICL). In this work, we statistically confirm the presence of spurious
features in the RAG paradigm, a robustness problem caused by the sensitivity of
LLMs to semantic-agnostic features. Moreover, we provide a comprehensive
taxonomy of spurious features and empirically quantify their impact through
controlled experiments. Further analysis reveals that not all spurious features
are harmful and they can even be beneficial sometimes. Extensive evaluation
results across multiple LLMs suggest that spurious features are a widespread
and challenging problem in the field of RAG. The code and dataset will be
released to facilitate future research. We release all codes and data at:
$\\\href{https://github.com/maybenotime/RAG-SpuriousFeatures}{https://github.com/maybenotime/RAG-SpuriousFeatures}$.

## Full Text


<!-- PDF content starts -->

Quantifying the Robustness of Retrieval-Augmented Language Models
Against Spurious Features in Grounding Data
Shiping Yang1 2 *Jie Wu2Wenbiao Ding2Ning Wu2Shining Liang2Ming Gong2Hengyuan Zhang3
Dongmei Zhang2
Abstract
Robustness has become a critical attribute for
the deployment of RAG systems in real-world
applications. Existing research focuses on ro-
bustness to explicit noise (e.g., document seman-
tics) but overlooks spurious features (a.k.a. im-
plicit noise). While previous works have explored
spurious features in LLMs, they are limited to
specific features (e.g., formats) and narrow sce-
narios (e.g., ICL). In this work, we statistically
confirm the presence of spurious features in the
RAG paradigm, a robustness problem caused by
the sensitivity of LLMs to semantic-agnostic fea-
tures. Moreover, we provide a comprehensive
taxonomy of spurious features and empirically
quantify their impact through controlled experi-
ments. Further analysis reveals that not all spu-
rious features are harmful and they can even be
beneficial sometimes. Extensive evaluation re-
sults across multiple LLMs suggest that spurious
features are a widespread and challenging prob-
lem in the field of RAG. We release all codes and
data at: https://github.com/maybenotime/RAG-
SpuriousFeatures.
1. Introduction
Retrieval-Augmented Generation (RAG) has emerged as a
promising paradigm to mitigate LLMs hallucinations (Gao
et al., 2023; Yang et al., 2023b), integrating relevant external
knowledge to improve the factuality and trustworthiness
of LLM-generated outputs (Zhou et al., 2024). However,
Retrieval-Augmented Language Models (RALMs) still face
substantial robustness issue due to the presence of noise in
retrieved documents (Liu et al., 2023; Li et al., 2024c).
Recent research aims to explore the characteristics that af-
*Work done during internship at Microsoft1Simon Fraser Uni-
versity2Microsoft3Tsinghua University. Correspondence to: Jie
Wu<jiewu1@microsoft.com >.
Question :The atomic  number  of 
indium  which  belongs  to 5th period  
is
Same Content but Different Spurious Features
Search and Retrieve
{"title" : "Period  5 element",  "text" : “…49 In Indium  
Post-transition  metal  ( Kr ) 4d 5s 5p…54 Xe Xenon  
Noble  gas ( Kr ) 4d 5s 5p."}FormatStyle
JSON
MD
title: Period  5 element  
text: …49 In Indium  Post-transition  metal  ( Kr ) 4d 5s 
5p…54 Xe Xenon  Noble  gas ( Kr ) 4d 5s 5p.YAML···# Period  5 element  
…49 In Indium  Post-transition  metal  ( Kr ) 4d 5s 
5p…54 Xe Xenon  Noble  gas ( Kr ) 4d 5s 5p.
Inconsistent Response
Style
Format
JSON
 49
MD
 59
YAML
 59
···
Figure 1. An example from the SURE framework (Sec. 3), illus-
trating the sensitivity of RAG systems to spurious features within
grounding data. MD represents the Markdown format. The origi-
nal retrieved document is fed into the LLMs in different formats,
leading to inconsistent responses.
fect the robustness of RAG systems from the perspective of
grounding data construction (Cuconasu et al., 2024). These
studies examine various factors, including the type (Wu
et al., 2024a), number (Xu et al., 2024), and position of doc-
uments (Liu et al., 2024) within the prompt context. A more
detailed discussion of related work is in Appendix A. How-
ever, previous analyses primarily focus on explicit noise
(i.e., causal features) that significantly alter the semantic
information of grounding data(Wu et al., 2024b; Cuconasu
et al., 2024), while neglecting spurious features (a.k.a. im-
plicit noise) that make slight modifications but still preserve
the meaning of contexts. This limitation extends to existing
evaluation benchmarks, which simulate complex noise sce-
narios to assess the robustness of RAG systems (Chen et al.,
2024a; Wu et al., 2024a), yet lack available benchmarks and
metrics to measure the robustness of LLMs against spurious
features.
Contemporary RAG systems typically employ production-
level retrievers, such as Bing and Google, to collect relevant
information from the internet. Unlike a single corpus, the
internet encompasses diverse data with distinct features. For
1arXiv:2503.05587v1  [cs.CL]  7 Mar 2025

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
any given query, there may exist numerous golden docu-
ments that contain the correct answer but differ in style,
format, or other attributes. As shown in Figure 1, we have
observed that LLMs may fail to consistently derive the cor-
rect answer from golden documents with different formats.
A similar phenomenon is reported in Sclar et al. (2024)
and He et al. (2024), which demonstrate that LLMs are
extremely sensitive to the format of prompts (i.e., spurious
features). For more related work, see Appendix A. Unfortu-
nately, there is no statistic and empirical evidence to support
the existence of spurious features in the RAG paradigm.
This highlights the urgent need to redefine spurious fea-
tures in RAG and systematically quantify the robustness of
RALMs against them.
To address these challenges, we first design a preliminary
experiment to demonstrate that RALMs are sensitive to
semantic-agnostic features in the grounding data, thereby
extending the definition of spurious features to RAG sys-
tems. Building on findings from our preliminary experiment
and recent studies, we identify five types of spurious fea-
tures that may appear in RAG scenarios. Then, we propose
a novel framework, SURE , for automating the process of ro-
bustness evaluation. This framework follows a perturb-then-
evaluate approach, offering great scalability. In SURE , auto-
mated perturbations are applied on the original instances to
inject the corresponding spurious features. The perturbed in-
stances are then examined to ensure that the causal features
remain intact. After these steps, we employ tailored met-
rics to quantify the robustness of RALMs against spurious
features. Further analysis reveals that not every spurious fea-
tures is harmful and they can even be beneficial sometimes.
We also find that our proposed metrics are more suitable for
evaluating spurious features in golden documents. Based on
this finding, we distill the most challenging instances from
the synthetic data generated by our framework to create
a lighter benchmark, SIG, enabling more efficient robust-
ness evaluation. We evaluate 12 representative LLMs with
varying architectures and scales, and the results show that
maintaining robustness against spurious features remains a
significant challenge, especially for open-source LLMs.
Our contribution can be summarized as follows: 1)We ex-
tend the definition of spurious features to RAG systems
through a preliminary experiment. To the best of our knowl-
edge, this is the first comprehensive study to define and
evaluate spurious features from RAG perspective. 2)We
propose a novel evaluation framework, SURE , to assess the
robustness of RALMs against spurious features, which ef-
fectively simulate five types of spurious features that may
appear in real world scenarios. 3)Through extensive experi-
ments and analysis using SURE on two representative LLMs,
we provide valuable insights for future research. 4)Based
on these findings, we curate a lightweight yet challenging
evaluation dataset named SIG and benchmark the robust-ness of the current state-of-the-art LLMs against spurious
features.
2. Preliminary
In this section, we first define causal and spurious features in
the context of retrieval-augmented generation and then pro-
vide statistical evidence to support the existence of spurious
features. Specifically, we start by defining an oracle re-
triever to sample data according to the preferences of LLMs
and then apply statistical testing to validate that RALMs
exhibit biases toward semantic-agnostic features within the
grounding data.
2.1. Causal and Spurious Features in RAG
In general, causal features are input features that have a di-
rect causal effect on the output of predictive model (Yu et al.,
2020). Their relationship is rooted in causality, rather than
mere statistical correlation. When it comes to Large Lan-
guage Models, the meaning and intent of prompts serve as
causal features that directly influence the models’ responses.
In the context of RAG, causal features refer to the semantic
information of grounding data, including whether the cor-
rect answer exists, the amount of noise documents in the
grounding data, and the content of those noise documents.
In contrast, spurious features are input features that co-occur
with causal features and are erroneously captured by the
model (Neuhaus et al., 2023). These features exhibit a sta-
tistical correlation with the model’s output but lack a causal
relationship. Recent research has shown that LLMs are
sensitive to seemingly trivial features like prompt format-
ting, thereby extending the definition of spurious features to
LLMs (Sclar et al., 2024). Similarly, we hypothesize that
the semantic-agnostic features of the grounding data can
be defined as spurious features in RAG systems. However,
the conclusion drawn from LLMs may not be applicable
to RALMs. Unlike LLMs, the input prompts for RALMs
incorporate additional dynamic content—grounding data—
to augment their output. This grounding data is derived
from retrieved documents rather than static instructions pro-
vided directly by users. Therefore, we design a preliminary
experiment to validate whether RALMs are sensitive to
semantic-agnostic features within the grounding data.
2.2. Oracle Retriever
We aim to confirm the existence of spurious features in RAG
scenarios, i.e., to demonstrate the sensitivity of RALMs to
semantic-agnostic features in grounding data.
There are some challenges in revealing the sensitivity of
RALMs. Specifically, When retrieving from a single corpus,
it is difficult to mine semantically equivalent counterparts
with obvious differences in semantic-agnostic features. This
2

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
is because documents from the corpus often share similar
styles and formats. With only slight differences in these
features, we are difficult to observe significant performance
variations using traditional evaluation metrics like accuracy.
Thus, more fine-grained metrics are required to capture the
subtle performance changes. Inspired by the use of LLMs
as supervision signals for document utility (Izacard et al.,
2023; Gan et al., 2024), we propose the oracle score, which
measures fine-grained performance through calculating the
log probability of generating correct answers based on the
given documents. The oracle score is defined as follows:
Oracle (x, y,θ) =TX
t=1logp(yt|x, y<t,θ) (1)
where xis the input prompt for RALMs, including the in-
struction I, grounding data G, and query Q;yrepresents the
ground truth answer; θdenotes the model parameters; and
Tis the total length of the answer sequence. For cases with
multiple answers, we compute the final score by averaging
the corresponding oracle scores across all answers. Notably,
the oracle score is less effective at estimating the helpfulness
of documents when handling queries with long-sequence an-
swers1. This limitation arises from the growing expansion
of the solution space as the answer length increases.
We further define an oracle retriever that ranks documents
according to their oracle scores. Since the oracle score is
computed using LLMs as supervision signals, the oracle re-
triever can be regarded as a sampler that reflects the LLMs’
preferences for grounding data. With this sampler, we suc-
cessfully transform the target of demonstrating that RALMs
are sensitive to semantic-agnostic features within grounding
data into showing that the oracle retriever is biased toward
spurious features in documents.
2.3. Preliminary Experiment & Analysis
Using the oracle retriever, we recall 100 documents from
the Wikipedia dump for each query in the NQ-open dataset.
However, the computational cost of calculating oracle scores
across the entire corpus to select top-ranked documents is
prohibitive. Therefore, we introduce Contriever-msmarco , a
traditional dense retriever, for first-stage retrieval, followed
by reranking in descending order based on the oracle scores.
In addition to reducing the computational load, this initial
retrieval also ensures the semantic similarity of retrieved
documents. To further eliminate the effect of causal features,
documents without golden answers are filtered out, ensuring
that the remaining documents have roughly consistent causal
features.
We then select the first-ranked and last-ranked documents
1In our experiments, we only use queries where the answer
sequences is fewer than 5 tokens, ensuring the effectiveness of
oracle scores.for each query from the remaining documents, resulting
in two sets, each containing 2658 samples. By comparing
the differences in feature distributions between these two
sets, we can assess whether the oracle retriever exhibits
bias toward semantic-agnostic features. If these two sets
do not belong to the same feature distribution, this can be
attributed to the oracle retriever’s bias towards semantic-
agnostic features during sampling. To confirm that this
bias is not introduced by the dense retriever in first-stage
retrieval, we establish a control group by randomly sampling
two documents instead of selecting the first- and last-ranked
documents.
To evaluate whether the two distributions are same, we
employ the Kolmogorov-Smirnov (K-S) test. The following
semantic-agnostic features are measured in the experiments:
1) Flesh Score, 2) Distinct-1, 3) Dependency Tree Depth, 4)
PPL, and 5) Token Length. A detailed introduction of the
K-S test and these features can be found in the Appendix B.
We conduct experiments using Mistral-7B-Instruct-v0.3 to
implement the oracle retriever. The K-S statistic and P-value
are presented in Table 3. Furthermore, we visualize the
feature distributions for both the experimental and control
groups in Figure 6. For all tested features in the experimen-
tal group, the K-S test rejects the null hypothesis, concluding
that the distribution of the two sets are significantly different.
In contrast, for the control group, the K-S test fails to reject
the null hypothesis. The results for Llama-3.1-8B-Instruct
are also provided in Appendix B. According to these results,
we can conclude that RALMs exhibit bias toward spurious
features in documents.
The preliminary experiment provides statistical evidence
supporting the existence of spurious features in RAG sys-
tems. Nevertheless, it does not offer empirical evidence or
quantitative analysis. Inspired by previous data synthesis
studies (Tan et al., 2024b; Tong et al., 2024; Li et al., 2024b;
Wang et al., 2024a), we use a data synthesis approach to
better control feature variables and quantify the robustness
of RALMs against spurious features.
3. Proposed Framework
In this section, we detail our proposed evaluation framework,
SURE (Spurious Feat UresRobustness Evaluation), which
designed specifically for assessing the robustness of RALMs
against spurious features in grounding data. As illustrated
in Figure 2, this framework comprise four components: 1)
Comprehensive Taxonomy. We identify five distinct types of
spurious features that may arise in the context of Retrieval
Augmented Generation. 2)Spurious Features Injection. We
design a data synthesis pipeline to automate the injection of
spurious features, utilizing both model-based and rule-based
methods to construct counterparts of the original document
3

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Simple
Format
StyleTaxonomy
Complex
LLM -
Gen
Source
Self-Gen
Logic
Reverse RandomLLM
Rank
HTML Markdown
YAML JSON
Meta Data
Source TimeTestbed Construction
Instance Preparation  
Query Source: NQ-Open
Document Source: Wiki dump
Spurious Features Injection
Model -based Perturbations
   Rule -based Modification
Causal Features Preservation
➢Maintain Semantic Equivalence
Bidirectional entailment algo
➢Preserve Ground Truths
String Match Strategy
RAG Systems
➢Retriever
       
➢LLMsContriever -msmarco
Llama3.1 -8B-Instruct
Mistral -7B-Instruct
Evaluation Metrics
Robustness Rate
 Win Rate
Accuracy
 Lose RateMistral-7B-Instruct Llama-3.1-8B-InstructRobustness Rate
Figure 2. Overview of our SURE framework. We provide a Comprehensive Taxonomy that includes five types of spurious features, further
divided into 13 subtypes of perturbations (left section). To construct the testbed, we prepare raw instances initially and then synthesize the
modified instances through a workflow consisting of Spurious Features Injection andCausal Features Preservation (middle section). By
applying carefully tailored metrics for Robustness Evaluation , we quantify the robustness of target RAG systems (right section).
with varying spurious features. 3)Causal Features Preser-
vation. We employ a bidirectional entailment algorithm and
a string matching strategy to ensure that the causal features
of grounding data remain unchanged. 4)Robustness Evalua-
tion. We report the win, lose and robustness rate at instance
level to facilitate a fine-grained evaluation.
3.1. Problem Formulation
Given a query q, the retriever Rreturns a list of relevant
documents from a corpus D={di}N
i=1. The relevance be-
tween document dand query qcan be measured by various
methods. In this work, we use a BERT-based dense retriever
to obtain the embedding of query and documents, respec-
tively. The relevance score is calculated by computing their
dot-product similarity:
s(q, di) =E(q)·E(di). (2)
Then, the Top-k documents with the highest similarity
scores are retrieved:
Dretrieve =argtop- k{s(q, di)|di∈D}. (3)
To formally quantify the robustness of RAG systems against
spurious features, we define the input prompt for the LLM-
based reader as P= (I, G, Q ), where Irepresents instruc-
tion,Grefers to the grounding data, constituted by a subset
ofDretrieve , andQis the query. A perturbation is introducedto investigate the impact of spurious features by applying
a semantic-agnostic modification to the original grounding
data, while preserving its causal features. We define g(.)
to automate this process, transforming Gtog(G)and pro-
ducing a counterpart ˆP= (I, g(G), Q). The outputs of
LLM-based reader for PandˆPare compared to evaluate
the impact of the introduced perturbation:
y=LLM (P),ˆy=LLM (ˆP). (4)
3.2. Taxonomy of Spurious Features
Based on the biased spurious features identified in our pre-
liminary experiment and prior works, we introduce five
types of spurious features and their corresponding perturba-
tions in detail as follows.
Style Perturbations The same content can be expressed
in different styles, using varying tones, words and sentence
structures. As shown in Section 2.3, LLMs exhibit biases
towards readability-related features. Similarly, for humans,
the readability of a text can significantly influence its ac-
cessibility to the audience (Yang et al., 2023a). Therefore,
we define two perturbations from the perspective of read-
ability style: Simple andComplex . The former simplifies
the grounding data by using basic vocabulary and simple
sentence structure, while the latter employs professional
vocabulary and a formal academic tone to complex the doc-
uments.
4

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Source Perturbations LLM-generated content, including
both misinformation and correct claims, infiltrates every cor-
ner of the internet. Recent studies have shown that neural
retrievers are biased towards LLM-generated content, lead-
ing to the marginalization of human-authored content (Dai
et al., 2024; Chen et al., 2024b). Moreover, our preliminary
experiments demonstrate that LLMs are biased towards the
Perplexity (PPL) of text. Thus, we define two types of
source perturbations: LLM-generated andSelf-generated .
Specifically, the LLM-generated perturbation paraphrases
the original document using a powerful LLM, while the self-
generated perturbation employs the same backbone model
used as the generator in the RAG system.
Logic Perturbations The arrangement of sentences
within a passage typically follows a logical order, ensur-
ing the clarity and coherence in the narrative flow. Here, we
simulate scenarios where the intrinsic logical chain is dis-
rupted by three different perturbations: Random ,Reverse ,
andLLM-reranked , each representing a distinct sentence
ordering strategy.
Format Perturbations The internet contains various data
formats, including HTML ,Markdown ,YAML andJSON .
These formats are usually processed into plain text before
being fed to LLMs. To mitigate the loss of structural in-
formation during this process, some RAG studies propose
using the original format, rather than plain text, to augment
the generation (Tan et al., 2024a). However, as highlighted
in previous research, the prompt format is recognized as a
spurious feature that can significantly impact model perfor-
mance (Sclar et al., 2024; He et al., 2024). Therefore, we
perturb the original document with four common formats to
explore the impact of grounding data format in the context
of RAG.
Metadata Perturbations Metadata is often included in
the HTML results returned by search engines. In our frame-
work, we consider two main types of metadata: Timestamp
andData source . The timestamp indicates the time when
the data was created, and data source identifies the origin of
the data.
3.3. Spurious Features Injection
The automation of spurious features injection is essential
for automating the entire evaluation framework. We detail
the process of collecting the original instances and describe
how the automated perturbation was implemented.
Instance Preparation An instance is the dynamic compo-
nent of the prompt P, consisting of a query Qand grounding
dataG. The queries are drawn from the NQ-open dataset,
while our data source is English Wikipedia dump as of 20
December 2018. To construct the original instances, we firstselect 1,000 queries based on the close-book QA results of
Mistral-7B-Instruct-v0.3 on NQ-open dataset. This subset
includes 500 queries that can be answered directly using
parametric knowledge ( Known ) and 500 queries that require
external knowledge for answering ( Unknown ). We then
retrieve 100 documents for each query from the Wikipedia
dump to serve as the grounding data. Overall, we have a
total of 100,000 original instances for the following pertur-
bation step.
Automated Perturbation As introduced in Section 3.1,
the perturbation g(.)injects spurious features by modify-
ing the grounding data. For style and source perturbations,
g(.)is implemented using an LLM2prompted by carefully
crafted guidelines to modify the raw document, producing
counterparts of the original instances. For logic and format
perturbations, we develop g(.)as a heuristic method based
on a set of predefined rules3. To simulate metadata that
may appear in the real world, we first synthesize pseudo
Wikipedia or Twitter links for the raw instances, and then or-
ganize them into HTML format using a rule-based g(.). The
complete implementation details for automated perturbation
are provided in Appendix C.
3.4. Causal Features Preservation
To eliminate the effect of causal features, it is essential to
follow the principle of controlled experiments by keeping
causal features constant while systematically manipulat-
ing spurious features. This approach isolates the impact
of spurious features from that of causal features, enabling
an accurate quantification of robustness against spurious
features. In our framework, we introduce two methods to
ensure the stability of causal features in the grounding data.
Maintain Semantic Equivalence For models capable of
following human instructions, we directly instruct them
to maintain semantic equivalence when injecting spurious
features. Nonetheless, it’s impossible to completely avoid
semantic shift during the perturbation process. To ensure
the semantic consistency before and after introducing per-
turbation, we employ a bidirectional entailment algorithm
to filter out instance pairs (raw instance, perturbed instance)
with semantic inequivalence. Specifically, for document G
and its modified counterpart g(G), we use a Natural Lan-
guage Inference (NLI) system to detect whether the latter
can be inferred from the former, and vice versa. The NLI
system classifies predictions into one of: entailment ,neu-
tral,contradiction . We compute both directions, and the
algorithm returns equivalent if and only if both directions
2Unless otherwise specified, all model-based g(.)are imple-
mented using Llama-3.1-70B-Instruct .
3One exception is that we implement the LLM-reranked pertur-
bation using an LLM-based g(.).
5

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
are predicted as entailment.
In general, this algorithm can be implemented by any NLI
system. However, in our case, the concatenation of Gand
g(G)sometimes exceeds the context limitation of a Bert-
based NLI model. Hence, we apply an LLM-based NLI
system4to implement the bidirectional entailment algo-
rithm. The prompt is included in Appendix D.
Preserve Ground Truths While semantic equivalence
protects causal features to the greatest extent, the perturba-
tion may lead to the correct answer being paraphrased into
an alias (e.g., ”President Roosevelt” to ”Roosevelt”). These
variations in the grounding data are likely to result in false
negatives when determining response correctness, despite
the NQ-Open dataset providing multiple potential answer
variants for each query. To address this issue, we employ a
simple string-matching strategy to filter out documents that
undergo unexpected modifications. For Golden documents
that originally contained the correct answers, we keep them
only if they preserve the ground truths after perturbation.
ForNoise documents that did not contain the correct an-
swers, we discard them if they unexpectedly acquire the
ground truths due to perturbations.
3.5. Overview of the Synthetic Dataset
Through the steps of spurious features injection andcausal
features preservation , we derive the final dataset avail-
able for robustness evaluation. The synthetic dataset gen-
erated by the SURE framework is divided into four subsets
based on the categories of queries and documents within
the instances. Notably, the distribution of the dataset is
model-specific, as the classification of Known andUnknown
queries is determined by the intrinsic knowledge of the tar-
get LLM. Table 1 presents the dataset statistics for assessing
Mistral-7B-Instruct-v0.3 . And the distribution for Llama-
3.1-8B-Instruct is shown in Appendix E.
3.6. Robustness Evaluation
We employ an evaluation method Y(.), in line with Liu
et al. (2024); Cuconasu et al. (2024), to measure the cor-
rectness of responses generated by RAG systems. This
approach checks whether any of the correct answers is con-
tained within the response produced by the LLM and then
derives a binary label. Previous researches use accuracy as
the primary metric and report it at dataset level to assess the
robustness of RALMs, which is quantified by calculating
the variations in the models’ accuracy across different types
of noise. However, dataset-level metrics has certain limita-
4Farquhar et al. (2024) confirms the effectiveness of the LLM-
based NLI system through human annotation, demonstrating that
its performance is on par with the DeBERTa-large model used in
Kuhn et al. (2023).K-G K-N U-G U-N Total
Style 7766 31152 2593 37692 79203
Source 9249 32435 3228 39101 84013
Logic 9724 35537 3587 41990 90838
Format 11037 38018 4141 45518 98714
Meta 11104 38018 4255 45420 98797
Table 1. Statistics of the evaluation dataset for Mistral-7B-Instruct-
v0.3. K-G denotes the instances composed of ( Known query,
Golden Document), while U-N refers to the instances consist-
ing of ( Unknown query, Noise Document). The values represents
the number of instance pairs for each type of perturbations within
the category of spurious features.
tions, as it may fail to capture fine-grained variations that
occur at the instance level. As shown in Figure 3, RALMs
may appear robust at dataset-level evaluations but exhibit
significant sensitivity at the instance level.
To quantify whether a RAG system is robust and unbiased at
the instance level, we assign a ternary label to each instance
by comparing the correctness of the LLM’s response before
and after introducing the perturbation. This comparison
process can be formulated as C=Y(yi)−Y( ˆyi), where
Clies in the set (−1,0,1). Based on the comparison out-
comes, we define three metrics: Robustness Rate (RR) ,
Win Rate (WR) , and Lose Rate (LR) . The RR is calculated
as follows:
RR=1
NNX
i=1I(C== 0) (5)
where Nis the total number of instances in the dataset;
yiandˆyirepresent the outputs of LLM for the original
and perturbed instances. RR measures the proportion of
instances where the RALM’s answer remains consistent
(0) before and after introducing the perturbation. Similarly,
WR and LR quantify the proportions of instances where
the correctness of the RALM’s response changes after the
perturbation, either from incorrect to correct ( C==−1) or
from correct to incorrect ( C== 1 ).
4. Experiments
4.1. Experimental Setup
We assess the robustness of RAG systems to spurious fea-
tures by evaluating them on their most popular application—
the Question Answering (QA) task, following the standard
”retrieve-read” setting of the RAG paradigm. We employ
Contriever-msmarco as the default retriever and conduct our
main experiments using the synthetic dataset generated by
theSURE framework. Due to space limitations, additional
6

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Mistral-7B-Instruct-v0.3
Spurious Features PerturbationsKnown-Golden Known-Noise Unknown-Golden Unknown-Noise Total
LR RR WR LR RR WR LR RR WR LR RR WR LR RR WR
StyleSimple 7.33 85.00 7,67 4.45 91.64 3.90 7.87 82.95 9.18 0.70 98.76 0.54 3.06 94.09 2.85
Complex 6.05 87.42 6.53 3.85 92.03 4.12 6.90 85.92 7.17 0.62 98.82 0.56 2.63 94.61 2.76
SourceLLM-Generated 5.91 87.62 6.47 3.57 92.27 4.16 6.41 86.52 7.06 0.59 98.75 0.66 2.55 94.56 2.90
Self-Generated 6.30 87.06 6.64 3.94 92.02 4.04 6.26 86.80 6.94 0.61 98.77 0.62 2.74 94.42 2.85
LogicReverse 5.44 89.34 5.22 2.99 94.10 2.92 5.97 88.54 5.49 0.48 99.04 0.48 2.21 95.65 2.14
Random 4.47 91.87 3.66 2.43 95.15 2.42 4.18 91.44 4.38 0.36 99.27 0.37 1.76 96.56 1.68
LLM-Ranked 3.52 93.15 3.33 2.07 95.84 2.09 3.57 92.89 3.54 0.34 99.30 0.36 1.48 97.04 1.48
FormatJSON 7.96 88.53 3.51 5.15 92.68 2.17 6.95 88.92 4.13 0.65 99.02 0.33 3.46 94.98 1.55
HTML 9.30 87.03 3.67 5.89 92.36 1.74 8.36 87.39 4.25 0.74 99.01 0.26 4.00 94.62 1.38
YAML 4.75 90.90 4.35 3.88 93.24 2.87 5.05 90.53 4.42 0.51 99.06 0.44 2.47 95.55 1.98
Markdown 3.98 92.49 3.53 2.91 94.36 2.72 4.11 92.59 3.31 0.44 99.15 0.41 1.94 96.29 1.77
MetadataTimestamp (pre) 2.62 94.90 2.48 1.28 97.61 1.11 3.15 94.45 2.40 0.17 99.67 0.17 1.00 98.12 0.89
Timestamp (post) 2.74 94.87 2.40 1.16 97.63 1.21 3.45 94.41 2.14 0.17 99.68 0.15 0.98 98.12 0.89
Datasource (wiki) 3.78 92.31 3.91 1.5 96.66 1.84 3.69 92.95 3.36 0.26 99.48 0.26 1.28 97.31 1.41
Datasource (twitter) 2.68 93.59 3.73 1.3 97.22 1.48 2.04 94.90 3.06 0.20 99.59 0.21 0.98 97.80 1.22
Table 2. Robustness evaluation results of Mistral-7B-Instruct-v0.3 on the synthetic dataset. For timestamp perturbation, (pre) and (post)
represents whether the virtual timestamp we created is before or after the LLM’s knowledge cutoff date. For datasource perturbation,
(wiki) and (twitter) indicate the domains of the pseudo link we synthesize. We use Bold to mark the WR values that are higher than the
LR, suggesting that the perturbation is beneficial.
Raw Document
Modified DocumentInstance 1 Instance 2 Instance 3 Instance N…
……
Dataset Level Variation:
 Acc w/ Raw=Acc w/ Modified
Instance Level Variation:
 Robustness Rate = 50%
Figure 3. A comparison of dataset-level metric (Acc) and instance-
level metric (RR) for robustness evaluation. "and%indicate the
correctness of responses. In this example, the value of RR reflects
the unrobustness at the instance level, while dataset-level metric
overlook the sensitivity of RALMs to spurious features within
documents.
implementation details can be found in Appendix F.
4.2. Result Analysis
In this section, we analyze our main results by category
and offer insights from various perspectives. The results
ofMistral-7B-Instruct andLlama-3.1-8B-Instruct are pre-
sented in Table 2 and Table 6, respectively.
For Different Queries There is a significant difference
in robustness rates between known and unknown queries
when evaluated on noise documents (compare K-N and U-N), whereas no such gap is observed for golden documents
(compare K-G and U-G). This phenomenon arises from the
tug-of-war between an LLM’s internal prior and external
evidence. When the retrieved content is noise documents,
LLMs are likely to override it with their own correct prior
knowledge (Known). However, the robustness to explicit
noise may be affected by the spurious features in docu-
ments. In other words, implicit noise (spurious features)
can coexist with explicit noise , highlighting the challenge
of maintaining robustness to spurious features in complex
noise scenarios.
For Different Grounding Data Across all types of spu-
rious features, the robustness rate on noise documents is
consistently higher than that on golden documents, whether
for known or unknown queries. This derives from the limi-
tation of the RR metric we propose, which measures unro-
bustness by capturing changes in answer correctness rather
than the inconsistency of responses. When tested on noise
documents, regardless of which spurious features are in-
jected, LLMs always generate incorrect responses, as noise
documents lack ground truths. In this case, even though
the responses change, the RR does not decrease since all
responses remain incorrect. However, in real-world appli-
cations, our primary concern is whether the RAG system
can consistently generate correct answer when faced with
golden documents containing different spurious features.
Therefore, we primarily focus on the RR results for the
(K, G) and (U, G) subsets .
7

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
For Different Metrics If the win rate surpasses the lose
rate, it shows that more instances were corrected rather than
misanswered after introducing perturbations. Based on the
comparison of WR and LR in Table 2 and Table 6, we can
conclude that Not every spurious feature is harmful and
they can even be beneficial sometimes . For example, the
WR of source perturbations is consistently higher than the
LR on both models. This suggests that the performance of
RAG can be enhanced by introducing beneficial spurious
features (e.g., simply paraphrasing documents using LLMs).
For Different Perturbations We observe notable differ-
ences in the robustness rates among five types of spuri-
ous features. However, within each category, the robust-
ness rates across different perturbations are relatively close.
When further comparing perturbations within the same cat-
egory, we find that while their RR values are compara-
ble, their WR and LR can differ significantly, indicating
LLMs’ preference for certain sub-perturbations. In sum-
mary, there are variations in robustness across different
spurious features, and preference exists among the sub-
perturbations of each spurious feature .
4.3. SIG Benchmark & Further Analysis
The raw synthetic dataset is not ideal for extensive evalu-
ation due to its large size. Furthermore, the class imbal-
ance result in unfair comparisons across different types of
spurious features. To facilitate more efficient evaluation,
we extract the most challenging data from the subsets of
synthetic dataset to create a lightweight benchmark: SIG
(Spurious features InGolden document). Specifically, we
select instance pairs where both the Mistral and Llama mod-
els exhibit sensitivity and unrobustness in our main experi-
ments. An equal number of samples (100) are chosen from
the (K,G) and (U,G) subsets of each perturbations.
Robustness Comparison of SOTA LLMs We evaluate
a broader range of models on SIG benchmark, including
GPT-4O ,GPT-4O-mini ,Mistral-Large-Instruct5,Llama-
3.3-70B-Instruct ,Deepseek-v3 (671B,MoE), and Qwen2.5-
72B-Instruct . To better compare the robustness of different
models, we average the RR of each perturbation within a
category to derive the overall robustness for a specific type
of spurious feature. The performance of six SOTA LLMs
is then visualized using a radar chart, as shown in Figure
4. Notably, GPT-4O-mini achieved the best performance,
even surpassing GPT-4O by a large margin on the format
and meta types. Despite the impressive robustness of closed-
source models, they may still exhibit sensitivity to certain
specific perturbations. For instance, GPT-4o achieved only
an 89% robustness rate on the datasource(twitter) perturba-
tion.
5https://huggingface.co/mistralai/Mistral-Large-Instruct-2411
Figure 4. Comparison of Robustness rates across six SOTA LLMs.
Scaling Analysis for Different Model Sizes To investi-
gate the impact of parameter scale on RAG robustness, we
gradually increase the size of LLM-based readers (Qwen2.5
series, ranging from 0.5B to 72B) and evaluate their robust-
ness across five types of spurious features. As illustrated in
Figure 5, the robustness rate for all spurious features shows
a relatively upward trend as the model size increases. How-
ever, when we further scale the model from 32B to 72B, the
RR undergoes a significant decline (except for format and
meta). This indicates that robustness issues related to spuri-
ous features cannot be resolved simply by increasing model
size. Interestingly, for meta perturbations, while RALMs
demonstrate strong robustness across all scales (even for the
0.5B model), their performance receives little to no benefit
from scaling up.
Figure 5. Scaling analysis of robustness to spurious features.
8

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
5. Conclusion
In this work, we formally highlight the spurious features
problem in RAG system. Through preliminary experiments,
we provide statistical evidence to support the presence of
spurious features in RALMs. We also propose a novel
evaluation framework, SURE , to assess the robustness of
LLMs against spurious features. This framework includes
a comprehensive taxonomy of spurious features, carefully
designed metrics for robustness evaluation, and a data syn-
thesis pipeline. Extensive evaluation and further analysis
offers valuable insights for future research.
Impact Statement
We introduce a new type of noise, spurious features, that can
undermine the robustness and trustworthiness of contempo-
rary RAG systems. With the increasing diversity of golden
documents on the internet, this noise may become more
prevalent. Furthermore, traditional methods for improving
the robustness of RAG systems are ineffective for handling
spurious features, as they are implicit and can coexist with
golden documents. Therefore, harmful spurious features
could undermines user trust and contaminates Internet data.
Moving forward, we aim to explore methods for mitigat-
ing hallucinations and robustness issues caused by spurious
features within grounding data.
References
Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu,
X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T.,
et al. Ms marco: A human generated machine reading
comprehension dataset. arXiv preprint arXiv:1611.09268 ,
2016.
Bian, N., Lin, H., Liu, P., Lu, Y ., Zhang, C., He, B., Han, X.,
and Sun, L. Influence of external information on large
language models mirrors social cognitive patterns. IEEE
Transactions on Computational Social Systems , 2024.
Chen, J., Lin, H., Han, X., and Sun, L. Benchmarking large
language models in retrieval-augmented generation. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence , volume 38, pp. 17754–17762, 2024a.
Chen, X., He, B., Lin, H., Han, X., Wang, T., Cao, B., Sun,
L., and Sun, Y . Spiral of silences: How is large lan-
guage model killing information retrieval?–a case study
on open domain question answering. arXiv preprint
arXiv:2404.10496 , 2024b.
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Cam-
pagnano, C., Maarek, Y ., Tonellotto, N., and Silvestri, F.
The power of noise: Redefining retrieval for rag systems.
InProceedings of the 47th International ACM SIGIRConference on Research and Development in Information
Retrieval , pp. 719–729, 2024.
Dai, S., Zhou, Y ., Pang, L., Liu, W., Hu, X., Liu, Y ., Zhang,
X., Wang, G., and Xu, J. Neural retrievers are biased
towards llm-generated content. In Proceedings of the
30th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , pp. 526–537, 2024.
Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G.,
Mazar ´e, P.-E., Lomeli, M., Hosseini, L., and J ´egou, H.
The faiss library. arXiv preprint arXiv:2401.08281 , 2024.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y . Detecting
hallucinations in large language models using semantic
entropy. Nature , 630(8017):625–630, 2024.
Feldman, P., Foulds, J. R., and Pan, S. Ragged edges:
The double-edged sword of retrieval-augmented chatbots.
arXiv preprint arXiv:2403.01193 , 2024.
Gan, C., Yang, D., Hu, B., Zhang, H., Li, S., Liu, Z., Shen,
Y ., Ju, L., Zhang, Z., Gu, J., et al. Similarity is not all
you need: Endowing retrieval augmented generation with
multi layered thoughts. arXiv preprint arXiv:2405.19893 ,
2024.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., and Wang, H. Retrieval-augmented generation
for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2023.
He, J., Rungta, M., Koleczek, D., Sekhon, A., Wang, F. X.,
and Hasan, S. Does prompt formatting have any impact
on llm performance? arXiv preprint arXiv:2411.10541 ,
2024.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski,
P., Joulin, A., and Grave, E. Unsupervised dense infor-
mation retrieval with contrastive learning. arXiv preprint
arXiv:2112.09118 , 2021.
Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni,
F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., and
Grave, E. Atlas: Few-shot learning with retrieval aug-
mented language models. Journal of Machine Learning
Research , 24(251):1–43, 2023.
Kang, H., Ni, J., and Yao, H. Ever: Mitigating hallucination
in large language models through real-time verification
and rectification. arXiv preprint arXiv:2311.09114 , 2023.
Kong, A., Zhao, S., Chen, H., Li, Q., Qin, Y ., Sun, R.,
Zhou, X., Wang, E., and Dong, X. Better zero-shot
reasoning with role-play prompting. In Proceedings of
the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) , pp.
4099–4113, 2024.
9

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Kuhn, L., Gal, Y ., and Farquhar, S. Semantic uncer-
tainty: Linguistic invariances for uncertainty estima-
tion in natural language generation. arXiv preprint
arXiv:2302.09664 , 2023.
Li, C., Wang, J., Zhu, K., Zhang, Y ., Hou, W., Lian, J., and
Xie, X. Emotionprompt: Leveraging psychology for large
language models enhancement via emotional stimulus.
arXiv e-prints , pp. arXiv–2307, 2023a.
Li, D., Zhang, H., Li, Y ., and Yang, S. Multi-level con-
trastive learning for script-based character understanding.
arXiv preprint arXiv:2310.13231 , 2023b.
Li, D., Jiang, B., Huang, L., Beigi, A., Zhao, C., Tan, Z.,
Bhattacharjee, A., Jiang, Y ., Chen, C., Wu, T., et al. From
generation to judgment: Opportunities and challenges of
llm-as-a-judge. arXiv preprint arXiv:2411.16594 , 2024a.
Li, D., Tan, Z., Chen, T., and Liu, H. Contextualization dis-
tillation from large language model for knowledge graph
completion. arXiv preprint arXiv:2402.01729 , 2024b.
Li, D., Yang, S., Tan, Z., Baik, J. Y ., Yun, S., Lee, J.,
Chacko, A., Hou, B., Duong-Tran, D., Ding, Y ., et al.
Dalk: Dynamic co-augmentation of llms and kg to an-
swer alzheimer’s disease questions with scientific litera-
ture. arXiv preprint arXiv:2405.04819 , 2024c.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts. Transactions of the
Association for Computational Linguistics , 12:157–173,
2024.
Liu, Y ., Huang, L., Li, S., Chen, S., Zhou, H., Meng, F.,
Zhou, J., and Sun, X. Recall: A benchmark for llms
robustness against external counterfactual knowledge.
arXiv preprint arXiv:2311.08147 , 2023.
Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M.,
Hajishirzi, H., and Zettlemoyer, L. Rethinking the role of
demonstrations: What makes in-context learning work?
InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing , pp. 11048–
11064, 2022.
Neuhaus, Y ., Augustin, M., Boreiko, V ., and Hein, M. Spuri-
ous features everywhere-large-scale detection of harmful
spurious features in imagenet. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
pp. 20235–20246, 2023.
Pezeshkpour, P. and Hruschka, E. Large language models
sensitivity to the order of options in multiple-choice ques-
tions. In Findings of the Association for Computational
Linguistics: NAACL 2024 , pp. 2006–2017, 2024.Sclar, M., Choi, Y ., Tsvetkov, Y ., and Suhr, A. Quanti-
fying language models’ sensitivity to spurious features
in prompt design or: How i learned to start worrying
about prompt formatting. In The Twelfth International
Conference on Learning Representations , 2024.
Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J.
Retrieval augmentation reduces hallucination in conver-
sation. In Findings of the Association for Computational
Linguistics: EMNLP 2021 , pp. 3784–3803, 2021.
Tan, J., Dou, Z., Wang, W., Wang, M., Chen, W., and Wen,
J.-R. Htmlrag: Html is better than plain text for model-
ing retrieved knowledge in rag systems. arXiv preprint
arXiv:2411.02959 , 2024a.
Tan, Z., Li, D., Wang, S., Beigi, A., Jiang, B., Bhattacharjee,
A., Karami, M., Li, J., Cheng, L., and Liu, H. Large
language models for data annotation and synthesis: A sur-
vey. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pp. 930–957,
2024b.
Tong, Y ., Wang, S., Li, D., Wang, Y ., Han, S., Lin, Z.,
Huang, C., Huang, J., and Shang, J. Optimizing language
model’s reasoning abilities with weak supervision. arXiv
preprint arXiv:2405.04086 , 2024.
Wang, S., Tong, Y ., Zhang, H., Li, D., Zhang, X., and Chen,
T. Bpo: Towards balanced preference optimization be-
tween knowledge breadth and depth in alignment. arXiv
preprint arXiv:2411.10914 , 2024a.
Wang, Y ., Hernandez, A. G., Kyslyi, R., and Kersting, N.
Evaluating quality of answers for retrieval-augmented
generation: A strong llm is all you need. arXiv preprint
arXiv:2406.18064 , 2024b.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
E., Le, Q. V ., Zhou, D., et al. Chain-of-thought prompting
elicits reasoning in large language models. Advances in
neural information processing systems , 35:24824–24837,
2022.
Wu, J., Che, F., Zhang, C., Tao, J., Zhang, S., and Shao,
P. Pandora’s box or aladdin’s lamp: A comprehensive
analysis revealing the role of rag noise in large language
models. arXiv preprint arXiv:2408.13533 , 2024a.
Wu, K., Wu, E., and Zou, J. Clasheval: Quantifying the
tug-of-war between an llm’s internal prior and external
evidence. In The Thirty-eight Conference on Neural In-
formation Processing Systems Datasets and Benchmarks
Track , 2024b.
Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Sub-
ramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro,
B. Retrieval meets long context large language models.
10

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
InThe Twelfth International Conference on Learning
Representations , 2024.
Yang, S., Sun, R., and Wan, X. A new dataset and empirical
study for sentence simplification in chinese. In Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pp.
8306–8321, 2023a.
Yang, S., Sun, R., and Wan, X. A new benchmark and
reverse validation method for passage-level hallucination
detection. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2023 , pp. 3898–3908, 2023b.
Yu, K., Guo, X., Liu, L., Li, J., Wang, H., Ling, Z., and
Wu, X. Causality-based feature selection: Methods and
evaluations. ACM Computing Surveys (CSUR) , 53(5):
1–36, 2020.
Zhang, H., Shang, C., Wang, S., Zhang, D., Yao, F., Sun,
R., Yu, Y ., Yang, Y ., and Wei, F. Shifcon: Enhancing
non-dominant language capabilities with a shift-based
contrastive framework. arXiv preprint arXiv:2410.19453 ,
2024a.
Zhang, H., Wu, Y ., Li, D., Yang, Z., Zhao, R., Jiang, Y ., and
Tan, F. Balancing speciality and versatility: a coarse to
fine framework for supervised fine-tuning large language
model. arXiv preprint arXiv:2404.10306 , 2024b.
Zhou, Y ., Liu, Y ., Li, X., Jin, J., Qian, H., Liu, Z., Li, C.,
Dou, Z., Ho, T.-Y ., and Yu, P. S. Trustworthiness in
retrieval-augmented generation systems: A survey. arXiv
preprint arXiv:2409.10102 , 2024.
Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y .,
Yang, L., Ye, W., Zhang, Y ., Zhenqiang Gong, N., et al.
Promptbench: Towards evaluating the robustness of large
language models on adversarial prompts. arXiv e-prints ,
pp. arXiv–2306, 2023.
Zhuo, J., Zhang, S., Fang, X., Duan, H., Lin, D., and Chen,
K. Prosa: Assessing and understanding the prompt sen-
sitivity of llms. In Findings of the Association for Com-
putational Linguistics: EMNLP 2024 , pp. 1950–1976,
2024.
11

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
A. Related Work
A.1. Robustness Evaluation of Retrieval-Augmented Generation
RAG systems comprise two core components: a retriever and an LLM-based reader. Augmenting LLMs with retrieved
external knowledge has been proven to effectively reduce hallucinations (Shuster et al., 2021; Kang et al., 2023). However,
the retrieved contexts inevitably contains noise in addition to desirable knowledge, which may mislead LLMs to produce an
incorrect response (Bian et al., 2024; Feldman et al., 2024). Previous works have explored automated evaluation frameworks
to assess the robustness of RAG systems in various settings. For instance, Chen et al. (2024a) benchmarked four fundamental
capabilities required for RAG, including noise robustness, negative rejection, information integration and counterfactual
robustness. Some studies have provided a detailed taxonomy of noise documents to further simulate the complexity of
real-world scenarios and highlighted the potential positive effects of certain types of noise (Cuconasu et al., 2024; Wu
et al., 2024a). There are also some recent works that propose using LLM-as-a-judge (Li et al., 2024a) to evaluate the RAG
system (Wang et al., 2024b).
While these studies have identified several explicit noises that affect the robustness of RAG systems, they overlook implicit
noises. This type of noise, such as phrasing and formatting, is everywhere and unavoidable, as it coexists with the grounding
data without altering its semantic information. In this work, we define these semantic-agnostic noises as spurious features
and evaluate the robustness of RALMs to such noises.
A.2. Prompt Sensitivity of LLMs
Large Language Models take prompts as inputs and then generate response accordingly. Prompts are instructions provided
to an LLM to perform specific tasks automatically and ensure desired qualities in the generated output. However, it is known
that current LLMs are sensitive to the features of input prompts (Zhu et al., 2023). This sensitivity poses challenges for
researchers attempting to evaluate the model’s performance accurately and precisely (Zhuo et al., 2024).
Some existing works have investigated the impact of different prompt techniques on model performance, including chain-of-
thought (Wei et al., 2022), in-context learning (Min et al., 2022), and role-play prompting (Kong et al., 2024). Beyond these
causal features that significantly influence the meaning of prompts, other works have demonstrated that LLMs are highly
sensitive to spurious features (Sclar et al., 2024), e.g, prompt formatting (He et al., 2024), language style (Li et al., 2023a),
the order of options (Pezeshkpour & Hruschka, 2024).
Currently, there is no statistical or empirical evidence to support the existence of spurious features in RALMs. To address
this gap, we extend the definition of spurious features to RAG systems through statistical testing and empirical analysis.
B. Preliminary Experiment Results
We introduce several semantic-agnostic features we measured in our preliminary experiments. These features include:
•Flesch Score : A readability metric designed to evaluate text difficulty. It is calculated based on the average number of
syllables per word and the average number of words per sentence. The Flesch score is a number on a scale from 0 to
100, where a higher score indicates that the text is easier to read.
•Distinct-1 : A metric used to assess the diversity of generated text. It calculates the proportion of unique words (distinct
words) to the total number of words in the output. A higher Distinct-1 score indicates that the text contains a greater
variety of unique words, implying more diversity in the generated content.
•Dependency Tree Depth (DTD) : A syntactic complexity metric calculated by analyzing its dependency tree. Depen-
dency Tree Depth refers to the maximum depth of a sentence’s dependency parse tree. A deeper tree suggests more
complex sentence structures, while a shallower tree indicates simpler syntactic constructions.
•Perplexity (PPL) : A metric used for evaluating language models, measuring how well a probabilistic model predicts a
given text. It reflects the uncertainty of a language model when generating sequences of words. Lower PPL values
indicate better predictive performance, meaning the model assigns higher probabilities to the actual labels in the
sequence.
•Token Length : We compute the total number of tokens in a text as an alternative measure of text length, given that
12

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
the documents in our corpus have been pre-segmented into fixed 100-word chunks. The value is model-specific and
depends on the model’s vocabulary.
Kolmogorov-Smirnov (K-S) Test The K-S test is a non-parametric statistical test used to compare the distribution of two
datasets. It evaluate whether two samples come from the same underlying probability distribution. The null hypothesis of the
K-S test is that the two samples are drawn from the same distribution, while the alternative hypothesis is that the two samples
are drawn from different distributions. There are two key values provided by K-S test: the K-S Statistic quantifies the largest
difference between the two sample distributions, and the p-value assess the statistical significance of that difference. If the
p-value is lower than a chosen significance level (0.05 in our experiments), we reject the null hypothesis, concluding that
the two distributions are significantly different. Otherwise, we fail to reject the null hypothesis, suggesting that there is
no significant difference between the two distributions. We provide the K-S test results in Table 3 and Table 4, with the
corresponding distribution visualizations plots shown in Figure 6 and Figure 7.
Experimental Group Control Group
K-S statistic P-value K-S statistic P-value
Flesch score 0.0677 1.01×10−5∗∗∗0.0301 0.1799
Distinct-1 0.0756 4.95×10−7∗∗∗0.0203 0.6431
DTD 0.0636 4.29×10−5∗∗∗0.0124 0.9866
PPL 0.0722 1.88×10−6∗∗∗0.0162 0.8776
Token Length 0.1708 2.91×10−34∗∗∗0.0256 0.3493
Table 3. K-S test results for Mistral-7B-Instruct-v0.3 as the oracle retriever.
Experimental Group Control Group
K-S statistic P-value K-S statistic P-value
Flesch score 0.0305 0.1694 0.0173 0.8210
Distinct-1 0.0798 8.94×10−8∗∗∗0.0327 0.1159
DTD 0.0474 0.0051∗∗0.0203 0.6431
PPL 0.0538 0.0009∗∗∗0.0181 0.7791
Token Length 0.1275 2.99×10−19∗∗∗0.0188 0.7349
Table 4. K-S test results for Llama-3.1-8B-Instruct as the oracle retriever.
C. Implementation Details for Injecting Spurious Features
We provide detailed prompts for LLM-based perturbations in Figure 8. For rule-based perturbations, placeholder template is
presented in Figure 9.
D. Implementation Details for Preserving Causal Features
We employ a bidirectional entailment algorithm to ensure the semantic equivalence before and after introducing spurious
features. The prompts for its core component, NLI model, are shown in Figure 10.
E. Statistics of the Synthetic Dataset
We present the dataset statistics for evaluating Llama-3.1-8B-Instruct in Table 5.
13

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
(a) Feature distributions of the experimental group
 (b) Feature distribution of the control group
Figure 6. Visualization of feature distributions for Mistral-7B-Instruct-v0.3
14

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
(a) Feature distributions of the experimental group
 (b) Feature distribution of the control group
Figure 7. Visualization of feature distributions for Llama-3.1-8B-Instruct
15

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Style Perturbations
[Simple]
Please simplify the following text while preserving its original meaning. Use shorter sentences, basic
vocabulary, and clear language. Avoid complex structures, technical terms, or ambiguous expressions.
Here is the passage to simplify: {Document }
[Complex]
Please complexify the following text while preserving its original meaning. Use longer sentences, intricate
sentence structures, and advanced vocabulary. Avoid contractions, informal language, and colloquial
expressions, ensuring the text maintains a professional and authoritative tone throughout.
Here is the passage to complexify: {Document }
Source Perturbations
Please rewrite the following passage. Ensure that the overall meaning, tone, and important details remain
intact. Avoid any significant shifts in style or focus. The aim is to create a fresh version while faithfully
conveying the original content.
Here is the passage to paraphrase: {Document }
Logic Perturbations
[LLM-Ranked]
Rearrange the following list of sentences in your preferred logical order and provide only the indices of the
sentences. Please do not include any explanations.
Example: {Example }
Sentences List: {Sentences List }
The length of the Sentences List is {Length of Sentences List }. Therefore, the indices must contain {Length
of Sentences List }elements, and the index values cannot exceed {Length of Sentences List - 1 }.
[Reverse] [Pyhton Code]
[Random] [Python Code]
Figure 8. Prompt templates for LLM-based perturbations.
16

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Format Perturbations
[JSON]
{
"title": "{Title}",
"text": "{Document}"
}
[HTML]
<html lang="en">
<head>
<meta charset="UTF-8">
{Title}
</head>
<body> {Document} </body>
</html>
[YAML]
Title: {Title}
Text: {Document}
[Markdown]
# {Title}
{Document}
Metadata Perturbations
[Timestamp]
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name=’timestamp’ content=’{timestamp}’>
{Title}
</head>
<body> {Document} </body>
</html>
[Datasource]
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name=’datasource’ content=’{datasource}’>
{Title}
</head>
<body> {Document} </body>
</html>
Figure 9. Placeholder templates for rule-Based perturbations.
17

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Consider the two passages below.
Premise: {raw text }
Hypothesis: {perturbated text }
Does the premise semantically entail the hypothesis? Answer with ’entailment’ if they are paraphrases,’contradiction’
if they have opposing meanings, or ’neutral’ if they are neither.
Response:
Figure 10. Prompts for LLM-based NLI system.
K-G K-N U-G U-N Total
Style 7321 28975 3038 39869 79203
Source 8768 30145 3709 41391 84013
Logic 9229 33294 4082 44233 90838
Format 10481 35616 4697 47920 98714
Meta 10563 35451 4796 47987 98797
Table 5. Distribution of the synthetic dataset for Llama-3.1-8B-Instruct .
F. Experimental Setup Details
Dataset We conduct our main experiments using the synthetic dataset generated by the SURE framework. The queries are
sourced from the NQ-open dataset6, and the documents are derive from the English Wikipedia dump.
Models We test two representative LLMs in our main experiments: Mistral-7B-Instruct-v0.3 andLlama-3.1-8B-Instruct .
Prompts The instruction Iin the RAG prompt P= (I, G, Q ), shown in Figure 11, is derived from Cuconasu et al. (2024),
with slight modifications to better adapt to our setting.
Implementation Details We follow the typical ”retrieve-read” setting of RAG paradigm. For the retrieval module, we
useContriever-msmarco7, a BERT-based dense retriever, as the default retriever. It is finetuned on the MS MARCO
dataset (Bajaj et al., 2016) after unsupervised pretraining via contrastive learning (Izacard et al., 2021). To optimize the
efficiency of vector similarity searches, we employ the Faiss library (Douze et al., 2024). For the read module, we deploy
LLMs on NVIDIA A100 GPUs and accelerate inference with vllm8. We follow (Li et al., 2023b; Zhang et al., 2024a;b) in
using a greedy decoding strategy and set the temperature to 0.1 to ensure stable outputs and strong reproducibility.
You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from the provided
document. If the document does not contain the answer, respond with NO-RES.
Figure 11. Instruction Iused for the QA task.
6https://huggingface.co/datasets/google-research-datasets/nq_open
7https://huggingface.co/facebook/contriever-msmarco
8https://github.com/vllm-project/vllm
18

Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
Llama-3.1-8B-Instruct
Spurious Features PerturbationsKnown-Golden Known-Noise Unknown-Golden Unknown-Noise Total
LR RR WR LR RR WR LR RR WR LR RR WR LR RR WR
StyleSimple 7.79 83.04 9.18 1.70 95.80 2.50 8.43 82.88 8.69 0.27 99.45 0.28 1.80 95.96 2.24
Complex 6.00 85.60 8.40 1.91 96.59 1.50 6.71 84.86 8.43 0.23 99.57 0.20 1.63 96.62 1.75
SourceLLM-Generated 5.89 86.43 7.69 1.43 96.83 1.74 6.20 85.71 8.09 0.22 99.56 0.22 1.51 96.60 1.89
Self-Generated 6.55 85.01 8.44 1.55 96.37 2.09 6.52 86.36 7.12 0.19 99.57 0.24 1.62 96.32 2.06
LogicReverse 5.06 90.82 4.12 1.13 97.82 1.06 5.73 89.71 4.56 0.21 99.67 0.13 1.29 97.64 1.07
Random 3.91 93.16 2.93 0.86 98.31 0.83 4.21 91.67 4.12 0.14 99.72 0.14 0.97 98.18 0.85
LLM-Ranked 3.24 93.93 2.83 0.82 98.43 0.74 3.58 93.36 3.06 0.13 99.76 0.11 0.85 98.39 0.75
FormatJSON 7.01 88.25 4.74 1.70 97.25 1.05 5.92 89.63 4.45 0.25 99.61 0.14 1.76 97.08 1.16
HTML 11.85 84.46 3.69 2.70 96.90 0.40 9.33 86.78 3.90 0.35 99.61 0.04 2.85 96.41 0.74
YAML 5.26 89.94 4.80 1.26 97.41 1.33 4.79 90.80 4.41 0.17 99.67 0.16 1.32 97.40 1.28
Markdown 2.32 92.23 5.45 0.60 96.89 2.51 2.34 93.46 4.19 0.07 99.61 0.32 0.61 97.55 1.84
MetadataTimestamp (pre) 2.08 95.81 2.11 0.28 99.42 0.29 2.54 95.56 1.90 0.02 99.95 0.03 0.46 99.10 0.44
Timestamp (post) 2.04 95.86 2.10 0.25 99.43 0.32 2.81 95.56 1.63 0.02 99.95 0.03 0.46 99.11 0.43
Datasource (wiki) 2.11 93.45 4.44 0.23 98.96 0.81 3.25 92.47 4.27 0.03 99.86 0.11 0.48 98.50 1.03
Datasource (twitter) 2.27 94.11 3.62 0.31 99.25 0.43 2.77 93.97 3.25 0.02 99.91 0.07 0.50 98.77 0.73
Table 6. Robustness evaluation results of Llama-3.1-8B-Instruct on the synthetic dataset. We use Bold to mark the WR values that are
higher than the LR, suggesting that the perturbation is beneficial.
19
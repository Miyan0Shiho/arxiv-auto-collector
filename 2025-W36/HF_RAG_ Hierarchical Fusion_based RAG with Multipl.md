# HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources and Rankers

**Authors**: Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri, Sudip Kumar Naskar

**Published**: 2025-09-02 21:03:40

**PDF URL**: [http://arxiv.org/pdf/2509.02837v1](http://arxiv.org/pdf/2509.02837v1)

## Abstract
Leveraging both labeled (input-output associations) and unlabeled data (wider
contextual grounding) may provide complementary benefits in retrieval augmented
generation (RAG). However, effectively combining evidence from these
heterogeneous sources is challenging as the respective similarity scores are
not inter-comparable. Additionally, aggregating beliefs from the outputs of
multiple rankers can improve the effectiveness of RAG. Our proposed method
first aggregates the top-documents from a number of IR models using a standard
rank fusion technique for each source (labeled and unlabeled). Next, we
standardize the retrieval score distributions within each source by applying
z-score transformation before merging the top-retrieved documents from the two
sources. We evaluate our approach on the fact verification task, demonstrating
that it consistently improves over the best-performing individual ranker or
source and also shows better out-of-domain generalization.

## Full Text


<!-- PDF content starts -->

HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources
and Rankers
Payel Santra∗
IACS
Kolkata, India
payel.iacs@gmail.comMadhusudan Ghosh∗
IACS
Kolkata, India
madhusuda.iacs@gmail.comDebasis Ganguly
University of Glasgow
Glasgow, United Kingdom
debasis.ganguly@glasgow.ac.uk
Partha Basuchowdhuri
IACS
Kolkata, India
partha.basuchowdhuri@iacs.res.inSudip Kumar Naskar
Jadavpur University
Kolkata, India
sudip.naskar@gmail.com
Abstract
Leveraging both labeled (input-output associations) and unlabeled
data (wider contextual grounding) may provide complementary ben-
efits in retrieval augmented generation (RAG). However, effectively
combining evidence from these heterogeneous sources is challeng-
ing as the respective similarity scores are not inter-comparable. Ad-
ditionally, aggregating beliefs from the outputs of multiple rankers
can improve the effectiveness of RAG. Our proposed method first
aggregates the top-documents from a number of IR models using a
standard rank fusion technique for each source (labeled and unla-
beled). Next, we standardize the retrieval score distributions within
each source by applying z-score transformation before merging
the top-retrieved documents from the two sources. We evaluate
our approach on the fact verification task, demonstrating that it
consistently improves over the best-performing individual ranker
or source and also shows better out-of-domain generalization.
CCS Concepts
•Information systems →Information retrieval ;Retrieval
models and ranking .
Keywords
Fact Verification, RAG, IR Fusion
ACM Reference Format:
Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowd-
huri, and Sudip Kumar Naskar. 2025. HF-RAG: Hierarchical Fusion-based
RAG with Multiple Sources and Rankers. In Proceedings of the 34th ACM
International Conference on Information and Knowledge Management (CIKM
’25), November 10–14, 2025, Seoul, Republic of Korea. ACM, New York, NY,
USA, 6 pages. https://doi.org/10.1145/3746252.3760942
1 Introduction
While social media platforms enable individuals to access, con-
tribute to, and disseminate information, they also facilitate the rapid
∗Both authors contributed equally to this research.
This work is licensed under a Creative Commons Attribution 4.0 International License.
CIKM ’25, Seoul, Republic of Korea
©2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2040-6/2025/11
https://doi.org/10.1145/3746252.3760942
Figure 1: Our proposed approach HF-RAG leverages both labeled and
unlabeled data to provide sub-topic–specific contextual information.
and widespread propagation of misinformation and fake news [ 6,
23]. As such, computational models for automated fact checking, i.e.,
methods to automatically examine the veracity of claims by retriev-
ing and analyzing supporting or refuting evidence [ 3,18,38,39],
are of high practical importance. Fact verification approaches in-
clude supervised fine-tuning (SFT), in-context learning (ICL), and
retrieval augmented generation (RAG). SFT adapts model parame-
ters with labeled data for task-specific learning, while instead of
updating model parameters ICL leverages labeled exemplars to
control predictions [ 4,30] and RAG includes relevant contextual
information from external unlabeled corpora [16, 26, 36].
We hypothesize that for fact verification, both labeled and un-
labeled data may serve as complementary sources of information,
each providing potentially relevant context for different aspects or
sub-topics of an input claim. Figure 1 illustrates this using a sample
claim from the Climate-FEVER dataset: Global warming causing ex-
tinction of polar bears . In this example, one sub-topic (red highlight)
pertains to the adverse effects of global warming, while the other
(blue highlight) concerns species extinction more broadly, not lim-
ited to polar bears. The first retrieved example in Figure 1, sourced
from the labeled FEVER training set, presents evidence suggesting
that brown bears are nearing extinction. Although this does notarXiv:2509.02837v1  [cs.IR]  2 Sep 2025

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Santra et al.
directly confirm the claim, it supports a plausible hypothesis that
polar bears might face a similar threat. This hypothesis is further
strengthened by additional contextual information retrieved from
Wikipedia, which provides relevant (unlabeled) evidence regarding
the broader risks posed by global warming [5, 37].
Novel Contributions .First, we propose to combine information
from two distinct sources –labeled and unlabeled data–to jointly cap-
ture both the topic-specific likelihood of a claim being true or false,
and the broader contextual information relevant to the input claim.
Second, we propose that rather than relying on a single ranking
model to retrieve topically relevant labeled or unlabeled examples,
it is potentially more effective to aggregate the outputs of multiple
rankers . This approach–commonly used in IR to improve perfor-
mance [ 7,10,11]–allows for the fusion of diverse ranking signals.
An overview of our proposed method, which involves a hierarchical
combination strategy–first performing intra-ranker fusion within
each source, followed by inter-source fusion–is presented in Fig-
ure 2. Based on this hierarchical fusion mechanism, we refer this
approach as Hierarchical Fusion-based RAG (HF-RAG ).
2 Proposed Hierarchical Fusion-based RAG
Combining Labeled and Unlabeled Contexts in RAG .Gen-
erally speaking, both RAG and ICL can be viewed as mechanisms
for incorporating additional contextual information, the former
relying on unlabeled documents retrieved from a corpus, while
the latter utilizing labeled instances from a training dataset. As a
consistent naming convention towards unifying these perspectives,
we refer to the former as Unlabeled RAG (U-RAG ) and the latter
asLabeled RAG (L-RAG ). In our proposed approach, we integrate
both sources of contextual information – unlabeled documents and
labeled examples – to leverage their complementary strengths of
topical relevance, and task-specific semantics, respectively. We hy-
pothesize that such a combined approach is likely to generalize
better to new domains, likely because while L-RAG provides the
necessary grounding to capture task-specific semantics (input-label
associations) required for effective predictions, the inclusion of
U-RAG prevents too much overfitting on a particular task itself
by capturing a broader task-agnostic semantics thus potentially
enabling better generalization to new domains and tasks.
Intra-Source Inter-Ranker Combinations by RRF .For a
specific source (labeled or unlabeled) 𝐶, an input claim x, and each
IR model𝜃∈Θ(where, Θis the set of retrievers) is first invoked to
obtain a top- 𝑘list of documents 𝐿𝐶,𝜃
𝑘. Next, we merge each of these
top-𝑘lists obtained from each ranker into a single ranked list by
the reciprocal rank fusion (RRF) [ 7] technique – a standard fusion
method in IR, which computes the overall score of a document as
its aggregated reciprocal ranks across each ranked list. Formally,
𝐿𝐶
𝑘=arg max
𝑘{𝜃𝐶(𝑑):𝑑∈Ø
𝜃∈Θ𝐿𝐶,𝜃
𝑘},𝜃𝐶(𝑑)=∑︁
𝜃∈Θ1
rank(𝐿𝐶,𝜃
𝑘,𝑑),
(1)
where arg max𝑘denotes a selection of the top- 𝑘documents with the
highest𝜃𝐶(𝑑)scores,𝜃𝐶(𝑑)denotes the RRF scores from source 𝐶,
andrank(𝐿𝐶,𝜃
𝑘,𝑑)denotes the rank of a document 𝑑in the list𝐿𝐶,𝜃
𝑘;
if𝑑∉𝐿𝐶,𝜃
𝑘then rank(𝐿𝐶,𝜃
𝑘,𝑑)is set to a large number 𝑀(≫𝑘).
Sparse
Bi-encoder
Cross-
encoder
Late-
interaction
Collection
ClaimRetrievers
Retrieved ContextIntra-sour ce,
Inter -ranking
Inter -sour ce
LLM Predicted LabelFused List
Score
Rank Position- -    
 - -   . . . 
- -    
 - -   - -    
 - -   RRF
Z-scor e
- -    
 - -   - -    
 - -   - -    
 - -   : Rank of a document d   
- -    
 - -   
(labeled) (unlabeled)...
...RRF
...
25
......
...11 ...
...34Figure 2: Schematic overview of our proposed method HF-RAG.
For a given claim, multiple retrievers are employed to obtain top-
ranked documents from labeled and unlabeled sources. These top-
documents for each source are combined via reciprocal rank fusion
(RRF). These fused lists of non-overlapping documents from the two
sources are then merged with a z-score transformation.
Equation 1 is applied for each source, 𝐶∈{𝑙,𝑢}(labeled and
unlabeled), to combine the information from multiple rankers into
two lists, respectively denoted by 𝐿𝑙
𝑘and𝐿𝑢
𝑘. Note that this way of
combining the outputs, from multiple rankers before triggering the
generative task, is different from: a) the FiD (Fusion-in-Decoder)
family of approaches [ 12,15] which merge the output from different
ranked lists into the context for task-specific tuning of the decoder,
and b) the RAG-Fusion [ 33,34] family of approaches, which modify
input queries with an objective to generate diverse lists of top-
documents. In contrast to FiD, our method involves only inference-
time computations, and different from RAG-Fusion, the objective is
not to increase inter-document diversity but rather to improve the
relevance of documents retrieved from each source.
Z-score for Inter-Source Combination .As the retrieved doc-
uments across the two information sources are non-overlapping,
it is not possible to apply RRF to compute the expected reciprocal
ranks of documents across the document lists 𝐿𝑙
𝑘and𝐿𝑢
𝑘. Since
the problem is similar to that of preference elicitation in a dueling
bandits setup [ 46], a standard technique is to employ probabilistic
comparisons to select the next candidate document from one of the
two lists. For these stochastic comparisons, it is a standard practice
to assume that the document scores in each list follow a Gaussian
distribution [ 46]. The difference of this problem of inter-source
combination of ranked lists with a standard dueling bandit problem
is that in our case no rewards are available to improve the selection
policy. As such, we simply use the z-score statistic, i.e., standardize
the scores of each document in the two lists, and use these scores
to induce a total ordering across the two lists. Formally speaking,
𝐿𝑘=arg max
𝑘{𝜙(𝜃𝐶(𝑑)):𝑑∈Ø
𝐶∈{𝑙,𝑢}𝐿𝐶
𝑘}, 𝜙(𝜃𝐶(𝑑))=𝜃𝐶(𝑑)−𝜇𝐶
𝜎𝐶,
(2)

HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources and Rankers CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
where𝜇𝐶and𝜎𝐶are the average and standard deviations of the
respective lists, i.e., labeled ( 𝐿𝑙
𝑘) and unlabeled ( 𝐿𝑢
𝑘). Intuitively
speaking, Equation 2 maps the document scores from the respective
sources to a standard normal scale N(0,1), removing collection-
specific bias [ 2,8] enabling a fairer comparison between labeled
and unlabeled documents.
To understand the connection between Equations 1 and 2 and
the schematic depicted in Figure 2, observe that we first aggregate
the ranked lists 𝐿𝐶,𝜃
𝑘produced by different retrieval models 𝜃for
each source 𝐶∈𝑙,𝑢(labeled and unlabeled), resulting in two fused
lists:𝐿𝑙
𝑘and𝐿𝑢
𝑘. These two source-specific lists are then further
combined in the final stage of the hierarchical fusion process1.
3 Experiment Setup
Our experiments are conducted to answer the following research
questions (RQs): a) RQ-1 : Does combining information from dif-
ferent sources and rankers in a hierarchical manner lead to better
out-of-domain generalization? b) RQ-2 : What is the relative con-
tribution of multiple rankers vs. multiple sources in an HF-RAG
setup? c) RQ-3 : How strongly does retrieval effectiveness correlate
with downstream gains? d) RQ-4 : How sensitive is HF-RAG to its
hyper-parameters, i.e., the number of examples in the context?
Datasets .We conduct our experiments on the fact verification
task [ 31], where the objective is to predict if an input claim can be ei-
ther supported or refuted with evidences retrieved from a collection
of documents, or there is not enough information in the collection to
do either. For our supervised and L-RAG-based approaches, we use
theFEVER training set [ 43] constituting claim-evidence pairs. As
the unlabeled data in U-RAG, we use the Wikipedia 2018 dump (the
underlying document collection for the FEVER dataset with avail-
able relevance assessments). For out-of-domain (OOD) evaluation
of models trained on the FEVER dataset we employ the test-splits
of the following: a) Climate-FEVER [9,40], comprising climate-
related claims (we removed the ‘disputed’ category to maintain
a consistent experiment setup), and b) SciFact [40,45], compris-
ing scientific claims. Similar to the FEVER dataset, the claims in
both these OOD datasets are also labeled as: ‘support’, ‘refute’, or
‘not-enough-information’.
Retrievers and Generators .We employ the following ranking
models in our experiments to retrieve the top-similar candidates
either from the FEVER training set (labeled data source), or from the
Wikipedia collection (unlabeled data source): a) BM25 [35]: a sparse
lexical model with prescribed settings of its hyper-parameters, i.e.,
(𝑘1,𝑏)=(1.2,0.75), b)Contriever [14]: a dense end-to-end bi-
encoder model, c) ColBERT [21]: a dense end-to-end late interac-
tion model, and d) MonoT5 [32]: a retrieve-rerank pipeline based
on a cross-encoder model (initial ranker set to BM25). For each IR
model, we retrieved the top- 50candidates for further processing
via the RRF pipeline (Equation 1).
We employ two LLMs of differing scales for the prediction: (a)
LLaMA 2.0 (70B) [ 41,44], representing a relatively large model, and
(b) Mistral (7B) [17, 42], a much smaller counterpart.
1Code available at: https://github.com/payelsantra/HF-RAGMethods Investigated .We compare our proposed method, HF-
RAG, against both parametric baselines that involve supervised
fine-tuning (SFT) and non-parametric RAG-based methods, which
may utilize labeled and/or unlabeled data. Among SFT-based meth-
ods, we employ the following: a) RoBERTa [29] – a common ap-
proach, reported in many studies [ 4,20,25], involving fine-tuning
a standard encoder model RoBERTa [ 28] on the FEVER training
dataset as a 3-way classifier mapping claim-evidence pairs to the
labels; b) LoRA [24] – an LLM decoder model is fine-tuned (specif-
ically, Llama-2-7B [ 1] for our experiments) as a 3-way classifier on
FEVER train claim-evidence pairs via the low-rank domain adapta-
tion (LoRA) technique [ 13]; and c) CORRECT [47] – which first
learns an evidence-conditioned prompt embedding by means of
noise contrastive loss on the FEVER training set of claim-evidence
pairs, and then uses this supervised prompt encoder for few-shot
inference with labeled data only (L-RAG).
In addition to the SFT-based methods, we also compare HF-RAG
with the following non-parametric RAG-based methods.
•0-shot [22,24,27]: This method predicts the class of a claim
(support/refute/not-enough-info) without relying on any addi-
tional context (labeled or unlabeled information sources) by lever-
aging the inherent knowledge stored in an LLM.
•L-RAG [27,29]: A standard in-context learning (ICL) workflow
that makes use of the labeled data from the FEVER training
data to predict the veracity of a claim. Out of the four available
rankers, we select the one that yields the best performance on
the FEVER dev set, which, in our experiment setup, turned out to
be Contriever. Contriever was then employed to retrieve a list of
similar claims (with their corresponding labels) from the FEVER
training set during inference on the test set.
•U-RAG [16,24,26]: This uses the unlabeled data source (Wikipedia
collection) for contextual generation via an LLM. Similar to L-
RAG, the ranker model was the best performing one on the
FEVER dev set, which turned out to be Contriever for Llama
and ColBERT for Mistral. The optimal ranker for a particular
LLM was then used to retrieve potentially relevant contextual
information from Wikipedia during inference on the test set.
•L-RAG-RRF : Instead of applying L-RAG on the optimized ranker,
here we apply all rankers to retrieve 4 ranked lists of top-50
candidates, following which, we merge them into a single list by
RRF (Equation 1 with the labeled data source, i.e., 𝐶={𝑙}).
•U-RAG-RRF : Similar to L-RAG-RRF, except this uses the unla-
beled data source to obtain the 4 different ranked lists, which are
then combined via RRF to yield 𝐿𝑢
𝑘(Equation 1 with 𝐶={𝑢}).
•LU-RAG-𝛼: This is an ablation for the z-score based combination
strategy - a part of our proposed method HF-RAG. Here, we apply
a different strategy to combine the top-lists retrieved from the
labeled and the unlabeled sources. Specifically, we use a linear
combination (parameterized by 𝛼) that controls the relative pro-
portion of top-documents to be selected from 𝐿𝑢
𝑘- the remaining
(1-𝛼) selected from the labeled source, 𝐿𝑙
𝑘. A grid search on the
FEVER train set was used to optimize 𝛼.
•RAG-OptSel : This acts as an upper bound on the performance
achievable by any single-ranker, single-source RAG configuration
selected from the 8 possible combinations in our setup (4 rankers
×2 sources). The best result among these 8 predictions is chosen

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Santra et al.
In-Domain Out-Domain
Predictor FEVER Climate-FEVER SciFact
RoBERTa 0.3010 0.2291 0.2371
LoRA 0.3959 0.3571 0.3489
CORRECT 0.3276 0.3295 0.3643
Llama Mistral Llama Mistral Llama Mistral
0-shot 0.4260 0.4623 0.4126 0.3724 0.3297 0.3258
L-RAG 0.4880 0.4890 0.4602 0.3901 0.3518 0.3347
U-RAG 0.4889 0.4880 0.4072 0.5083 0.3719 0.4168
L-RAG-RRF 0.5418 0.5583 0.4755 0.4468 0.3948 0.3665
U-RAG-RRF 0.4803 0.5185 0.4798 0.5249 0.4012 0.3963
LU-RAG-𝛼 0.4880 0.3955 0.4815 0.3703 0.3623 0.3178
HF-RAG 0.5744 0.5628 0.4838 0.5019 0.4320 0.4341
RAG-OptSel 0.5468 0.5584 0.4717 0.5001 0.3953 0.4246
Table 1: Performance of HF-RAG relative to the baselines. The best
results for a particular experiment setting are bold-faced, and the
second-best results are underlined. RAG-OptSel results are grayed
out to indicate that it is only a performance bound (using the test
labels). The table reports macro F1 scores, obtained with a context
size of 10, i.e., 𝑘=10in Equation 2.
using ground-truth labels from the corresponding test sets. The
goal is to assess whether the proposed combination method can
outperform this upper bound.
4 Results
Table 1 compares our proposed approach and the baselines for
in-domain and OOD evaluation. First, for RQ-1 (OOD general-
ization ), we observe that HF-RAG mostly outperforms both para-
metric and non-parametric baselines not only for OOD but also for
in-domain evaluation. Particularly encouraging are the large im-
provements observed for scientific claims (SciFact results in Table
1), as the results show that combining information sources poten-
tially mitigates overfitting a model to a particular domain, e.g., the
FEVER model generalizing well for the scientific domain.
In relation to RQ-2 (multi-rankers vs. multi-sources ), Table
1 shows that fusion with multiple rankers improves RAG effec-
tiveness with both labeled and unlabeled sources ( L/U-RAG-RRF
results, in general, better than L/U-RAG ). Eventually combining in-
formation across the two sources further improves results ( HF-RAG
results outperforming L/U-RAG-RRF ones ). Combination via z-score
is better than the proportional mixture of information from labeled
and unlabeled sources ( HF-RAG outperforming LU-RAG- 𝛼), which
indicates that z-score transformation is able to better capture the
relative preference between the documents from the two sources.
In relation to RQ-3 (correlation between retriever and gen-
erator performance ), Figure 3 demonstrates a positive correlation
between retrieval quality–measured by the relevance of evidence
retrieved from the unlabeled source–and downstream task perfor-
mance. The plots indicate that combining multiple rankers consis-
tently improves nDCG@10 across all three datasets. This ranker
fusion also results in gains in F1 score, further supporting the ben-
efit of enhanced retrieval quality on end-task performance.
In relation to RQ-4 (parameter sensitivity of HF-RAG ), we
observe from Figure 4a that HF-RAG exhibits greater stability with
BM25MonoT5ContrieverColBert Fusion
Retrieval Model0.00.10.20.30.40.50.60.70.8MetricF1-Score nDCG(a) FEVER dataset
BM25MonoT5ContrieverColBert Fusion
Retrieval Model0.00.10.20.30.40.5MetricF1-Score nDCG (b) Climate-FEVER
BM25MonoT5ContrieverColBert Fusion
Retrieval Model0.00.10.20.30.40.50.60.7MetricF1-Score nDCG (c) SciFact Dataset
Figure 3: Comparison between IR (nDCG@10) and claim verification
performance (F1) for U-RAG with various models, and U-RAG-RRF.
5 10 15 20
Number of Examples0.320.340.360.380.400.42F1 Score
0-shot
L-RAGU-RAG
L-RAG-RRFU-RAG-RRF
HF-RAG
(a) Sensitivity on SciFact
FEVER Climate-FEVER SciFact4.04.55.05.5CountAvg #Labeled
Avg #Unlabeled (b) Labeled vs. Unlabeled
Figure 4: (a) Parameter sensitivity of the RAG methods on SciFact
predictions; (b) Relative proportion of labeled and unlabeled data in
HF-RAG with 10 examples.
respect to context size (i.e., the number of retrieved examples), con-
sistently outperforming both L-RAG and U-RAG as well as their
inter-ranker combinations. Furthermore, in connection with RQ-2 ,
Figure 4b shows that HF-RAG effectively leverages appropriate
proportions of data from labeled and unlabeled sources. Among
the two OOD datasets, Climate-FEVER is more similar to FEVER in
terms of claim length and linguistic style. In contrast, the scientific
claims in SciFact are less aligned with the FEVER domain. Con-
sequently, HF-RAG tends to utilize more information from the la-
beled dataset—particularly veracity labels of the related claims—for
Climate-FEVER. For SciFact, however, it relies more heavily on ex-
ternal knowledge sources, which are likely to be more informative
than the veracity labels from FEVER, due to the domain shift.
5 Concluding Remarks
We proposed a multi-source multi-ranker RAG approach that first,
for each source, combines the top-retrieved documents obtained
from multiple ranking models and then combines the information
from the two sources of data–labeled and unlabeled–into a merged
context for RAG. Our experiments on the fact verification task
demonstrated that our method consistently outperforms several
baselines, and also improves over the best RAG performance achiev-
able with an individual ranker or source. Moreover, our method
was observed to generalize better on out-of-domain datasets. In the
future, we plan to extend this setup of hierarchical fusion involving
multiple sources and multiple rankers to multi-agent RAG with a
reasoner component, e.g., search-R1 [19].

HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources and Rankers CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
GenAI Usage Disclosure
Generative AI tools were not used for core idea generation or exper-
imental design. Its use was limited to minor writing and formatting.
References
[1]Meta AI. 2023. LLaMA 2: Open Foundation and Chat Models. https://huggingface.
co/meta-llama/Llama-2-7b. Accessed: 2025-06-03.
[2]Avi Arampatzis and Stephen Robertson. 2011. Modeling score distributions in
information retrieval. Information Retrieval 14 (2011), 26–46.
[3] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. 2023. Retrieval-based
language models and applications. In Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 6: Tutorial Abstracts) . 41–46.
[4]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In
Proceedings of the 34th International Conference on Neural Information Processing
Systems (Vancouver, BC, Canada) (NIPS ’20) . Curran Associates Inc., Red Hook,
NY, USA, Article 159, 25 pages.
[5]Manish Chandra, Debasis Ganguly, and Iadh Ounis. 2025. One size doesn’t
fit all: Predicting the number of examples for in-context learning. In European
Conference on Information Retrieval . Springer, 67–84.
[6]Xinran Chen, Sei-Ching Joanna Sin, Yin-Leng Theng, and Chei Sian Lee. 2015.
Why do social media users share misinformation?. In Proceedings of the 15th
ACM/IEEE-CS joint conference on digital libraries . 111–114.
[7]Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. 2009. Reciprocal
rank fusion outperforms condorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR conference on Research and
development in information retrieval . 758–759.
[8]Ronan Cummins. 2014. Document score distribution models for query perfor-
mance inference and prediction. ACM Transactions on Information Systems (TOIS)
32, 1 (2014), 1–28.
[9]Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bulian, Massimiliano Ciaramita,
and Markus Leippold. 2020. Climate-fever: A dataset for verification of real-world
climate claims. arXiv preprint arXiv:2012.00614 (2020).
[10] Mohamed Farah and Daniel Vanderpooten. 2007. An outranking approach for
rank aggregation in information retrieval. In Proceedings of the 30th annual
international ACM SIGIR conference on Research and development in information
retrieval . 591–598.
[11] Edward Fox and Joseph Shaw. 1994. Combination of multiple searches. NIST
special publication SP (1994), 243–243.
[12] Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and Hamed Zamani. 2023. Fid-
light: Efficient and effective retrieval-augmented text generation. In Proceedings
of the 46th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 1437–1447.
[13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large
language models. arXiv preprint arXiv:2106.09685 (2021).
[14] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv preprint arXiv:2112.09118
(2021).
[15] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with
generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020).
[16] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. In Proceedings of the
16th Conference of the European Chapter of the Association for Computational
Linguistics: Main Volume , Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty (Eds.).
Association for Computational Linguistics, Online, 874–880. doi:10.18653/v1/
2021.eacl-main.74
[17] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https:
//arxiv.org/abs/2310.06825
[18] Kelvin Jiang, Ronak Pradeep, and Jimmy Lin. 2021. Exploring listwise evidence
reasoning with t5 for fact verification. In Proceedings of the 59th Annual Meeting
of the ACL and the 11th IJCNLP (Volume 2: Short Papers) . 402–410.
[19] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL]https://arxiv.org/abs/2503.09516
[20] Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bin Liu.
2023. Continual Pre-training of Language Models. In International Conference on
Learning Representations . https://api.semanticscholar.org/CorpusID:258079422
[21] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd
International ACM SIGIR Conference on Research and Development in Information
Retrieval (Virtual Event, China) (SIGIR ’20) . Association for Computing Machinery,
New York, NY, USA, 39–48. doi:10.1145/3397271.3401075
[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in
neural information processing systems 35 (2022), 22199–22213.
[23] Srijan Kumar, Robert West, and Jure Leskovec. 2016. Disinformation on the web:
Impact, characteristics, and detection of wikipedia hoaxes. In Proceedings of the
25th international conference on World Wide Web . 591–602.
[24] Tiziano Labruna, Jon Ander Campos, and Gorka Azkune. 2024. When to Re-
trieve: Teaching LLMs to Utilize Information Retrieval Effectively. arXiv preprint
arXiv:2404.19705 (2024).
[25] Eunchan Lee, Changhyeon Lee, and Sangtae Ahn. 2022. Comparative Study of
Multiclass Text Classification in Research Proposals Using Pretrained Language
Models. Applied Sciences (2022). https://api.semanticscholar.org/CorpusID:
248471302
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[27] Xiaoqian Li, Ercong Nie, and Sheng Liang. 2023. From classification to generation:
Insights into crosslingual retrieval augmented icl. arXiv preprint arXiv:2311.06595
(2023).
[28] Yinhan Liu. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv
preprint arXiv:1907.11692 (2019).
[29] Quanyu Long, Wenya Wang, and Sinno Jialin Pan. 2023. Adapt in Contexts:
Retrieval-Augmented Domain Adaptation via In-Context Learning. arXiv preprint
arXiv:2311.11551 (2023).
[30] Andrew Parry, Debasis Ganguly, and Manish Chandra. 2024. In-Context Learning"
or: How I learned to stop worrying and love" Applied Information Retrieval. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 14–25.
[31] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani,
Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
et al.2020. KILT: a benchmark for knowledge intensive language tasks. arXiv
preprint arXiv:2009.02252 (2020).
[32] Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. 2021. The expando-mono-duo
design pattern for text ranking with pretrained sequence-to-sequence models.
arXiv preprint arXiv:2101.05667 (2021).
[33] Zackary Rackauckas. 2024. Rag-fusion: a new take on retrieval-augmented
generation. arXiv preprint arXiv:2402.03367 (2024).
[34] Zackary Rackauckas, Arthur Câmara, and Jakub Zavrel. 2024. Evaluating
rag-fusion with ragelo: an automated elo-based framework. arXiv preprint
arXiv:2406.14783 (2024).
[35] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (April 2009), 333–389.
doi:10.1561/1500000019
[36] Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri, and
Sudip Kumar Naskar. 2024. “The Absence of Evidence is Not the Evidence of
Absence”: Fact Verification via Information Retrieval-Based In-Context Learn-
ing. In Big Data Analytics and Knowledge Discovery: 26th International Confer-
ence, DaWaK 2024, Naples, Italy, August 26–28, 2024, Proceedings (Naples, Italy).
Springer-Verlag, Berlin, Heidelberg, 381–387. doi:10.1007/978-3-031-68323-7_34
[37] Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri,
and Sudip Kumar Naskar. 2025. The “Curious Case of Contexts” in Retrieval-
Augmented Generation With a Combination of Labeled and Unlabeled Data.
Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 15, 2
(2025), e70021.
[38] Tal Schuster, Adam Fisch, and Regina Barzilay. 2021. Get your vitamin C! robust
fact verification with contrastive evidence. arXiv preprint arXiv:2103.08541 (2021).
[39] Tal Schuster, Darsh J Shah, Yun Jie Serene Yeo, Daniel Filizzola, Enrico Santus,
and Regina Barzilay. 2019. Towards debiasing fact verification models. arXiv
preprint arXiv:1908.05267 (2019).
[40] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of
information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
[41] TheBloke. 2023. Llama-2-70B-Chat-AWQ. https://huggingface.co/TheBloke/
Llama-2-70B-Chat-AWQ. Accessed: 2025-06-01.
[42] TheBloke. 2023. Mistral-7B-Instruct-v0.2-AWQ. https://huggingface.co/
TheBloke/Mistral-7B-Instruct-v0.2-AWQ. Accessed: 2025-06-01.
[43] James Thorne, Andreas Vlachos, Oana Cocarascu, Christos Christodoulopoulos,
and Arpit Mittal. 2018. The Fact Extraction and VERification (FEVER) Shared Task.

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Santra et al.
InProceedings of the First Workshop on Fact Extraction and VERification (FEVER) .
Association for Computational Linguistics, Brussels, Belgium, 1–9. doi:10.18653/
v1/W18-5501
[44] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models. arXiv
preprint arXiv:2302.13971 (2023).
[45] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific
Claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , Bonnie Webber, Trevor Cohn, Yulan He, andYang Liu (Eds.). Association for Computational Linguistics, Online, 7534–7550.
doi:10.18653/v1/2020.emnlp-main.609
[46] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. 2012. The
K-armed dueling bandits problem. J. Comput. Syst. Sci. 78, 5 (2012), 1538–1556.
[47] Delvin Ce Zhang and Dongwon Lee. 2025. CORRECT: Context- and Reference-
Augmented Reasoning and Prompting for Fact-Checking. In Proceedings of the
2025 Conference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) ,
Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for Computational
Linguistics, Albuquerque, New Mexico, 3007–3019. https://aclanthology.org/
2025.naacl-long.154/
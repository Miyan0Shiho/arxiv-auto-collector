# Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Published**: 2025-03-27 09:54:37

**PDF URL**: [http://arxiv.org/pdf/2503.21315v1](http://arxiv.org/pdf/2503.21315v1)

## Abstract
Retrieval-augmented generation (RAG) systems enhance large language models by
incorporating external knowledge, addressing issues like outdated internal
knowledge and hallucination. However, their reliance on external knowledge
bases makes them vulnerable to corpus poisoning attacks, where adversarial
passages can be injected to manipulate retrieval results. Existing methods for
crafting such passages, such as random token replacement or training inversion
models, are often slow and computationally expensive, requiring either access
to retriever's gradients or large computational resources. To address these
limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an
efficient black-box method that leverages two key properties of retrievers:
insensitivity to token order and bias towards influential tokens. By focusing
on these characteristics, DIGA dynamically adjusts its genetic operations to
generate effective adversarial passages with significantly reduced time and
memory usage. Our experimental evaluation shows that DIGA achieves superior
efficiency and scalability compared to existing methods, while maintaining
comparable or better attack success rates across multiple datasets.

## Full Text


<!-- PDF content starts -->

Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus
Poisoning Attack
Cheng Wang†, Yiwei Wang||, Yujun Cai‡, , Bryan Hooi†,
†National University of Singapore||University of California, Merced
‡University of Queensland
wcheng@comp.nus.edu.sg
Abstract
Retrieval-augmented generation (RAG) sys-
tems enhance large language models by incor-
porating external knowledge, addressing issues
like outdated internal knowledge and hallucina-
tion. However, their reliance on external knowl-
edge bases makes them vulnerable to corpus
poisoning attacks, where adversarial passages
can be injected to manipulate retrieval results.
Existing methods for crafting such passages,
such as random token replacement or training
inversion models, are often slow and compu-
tationally expensive, requiring either access to
retriever’s gradients or large computational re-
sources. To address these limitations, we pro-
pose Dynamic Importance-Guided Genetic Al-
gorithm (DIGA), an efficient black-box method
that leverages two key properties of retrievers:
insensitivity to token order and bias towards
influential tokens. By focusing on these char-
acteristics, DIGA dynamically adjusts its ge-
netic operations to generate effective adversar-
ial passages with significantly reduced time and
memory usage. Our experimental evaluation
shows that DIGA achieves superior efficiency
and scalability compared to existing methods,
while maintaining comparable or better attack
success rates across multiple datasets.
1 Introduction
Large language models (LLMs) (Yang et al., 2024;
Achiam et al., 2023; Meta, 2024) have shown im-
pressive performance across a wide range of nat-
ural language processing tasks, but they still suf-
fer from significant limitations, such as hallucina-
tion (Huang et al., 2023; Chen and Shu, 2023) and
outdated internal knowledge. To address these is-
sues, retrieval-augmented generation (RAG) (Gao
et al., 2024; Fan et al., 2024) systems have emerged
as a promising solution by incorporating external,
up-to-date knowledge into the generation process.
By retrieving relevant information from large ex-
ternal corpora, RAG systems can enhance the ac-
curacy and relevance of LLM-generated outputs,HotFlip Vec2Text DIGA (ours)
Black-box × ✓ ✓
No Add. Training ✓ × ✓
Scalability × ✓ ✓
Table 1: Comparison of different corpus poisoning
methods. Our proposed method is black-box based,
requires no additional training, and maintains high scal-
ability.
especially in open-domain question answering and
knowledge-intensive tasks.
Despite the benefits, the reliance on external
knowledge sources exposes RAG systems to poten-
tial security vulnerabilities (Zeng et al., 2024; Xue
et al., 2024; Li et al., 2024b; Anderson et al., 2024).
A particular focus has been on adversarial attacks
targeting RAG systems. Initial studies (Song et al.,
2020; Raval and Verma, 2020; Liu et al., 2022)
explored attacks where adversarial passages are in-
jected into the knowledge base to influence the lan-
guage model’s output for specific queries. Building
upon this, Zhong et al. (2023) introduced a more po-
tent form of attack termed corpus poisoning, where
a single adversarial passage aims to affect retrieval
results across multiple queries simultaneously. Cur-
rently, two primary approaches dominate corpus
poisoning attack implementations. The first, based
on HotFlip (Ebrahimi et al., 2018), initializes an
adversarial passage with random tokens and itera-
tively optimizes it using gradients from the retriever.
The second method, Vec2Text (Morris et al., 2023),
originally developed to study text reconstruction
from embeddings, has been repurposed for corpus
poisoning by inverting the embeddings of query
centroids.
However, these existing methods face signifi-
cant limitations: (1) White-box access: HotFlip
requires access to the retriever model’s gradient,
limiting its applicability in real-world scenarios
where such access is often restricted. (2) Lack ofarXiv:2503.21315v1  [cs.LG]  27 Mar 2025

generalizability: Vec2Text requires training a sep-
arate inversion model on a specific dataset. This
limits its applicability to new domains or datasets
without retraining, as the model may struggle to
reconstruct embeddings for data significantly differ-
ent from its training set. (3) High computational
demands: Both methods require substantial com-
putational resources, making them impractical for
large-scale attacks. HotFlip’s iterative optimization
process is time-consuming and memory-intensive.
Vec2Text, while faster in generation, requires addi-
tional memory for its inversion model and is com-
putationally expensive during the training phase.
These resource constraints severely limit the scala-
bility of both approaches.
Given the limitations of existing methods, we
posit a crucial question: Is it possible to perform
corpus poisoning attacks under minimal assump-
tions about the target system while maintaining
high effectiveness and efficiency? In response, we
propose the Dynamic Importance-Guided Genetic
Algorithm (DIGA). Our method is built upon the
foundation of genetic algorithms, a class of op-
timization techniques inspired by the process of
natural selection. DIGA incorporates two key prop-
erties of retrieval systems into its evolutionary pro-
cess: their insensitivity to token order and bias
towards influential tokens. By leveraging these
properties, DIGA dynamically adjusts its genetic
operations, such as population initialization and
mutation, to craft effective adversarial passages.
This approach allows us to generate potent adver-
sarial content without requiring white-box access to
the retriever or extensive computational resources
(see Table 1).
Our empirical evaluation demonstrates that
DIGA achieves comparable or superior results to
existing methods across various datasets and re-
trievers. Notably, DIGA accomplishes this with
significantly lower time and memory usage, mak-
ing it more suitable for large-scale attacks. Further-
more, our method exhibits stronger transferability
across different datasets compared with another
black-box method. This enhanced transferability
is significant as it demonstrates DIGA’s ability to
generate more generalizable adversarial passages,
potentially making it more effective in real-world
scenarios where attack and target datasets may dif-
fer.
Our study underscores the vulnerabilities present
in RAG systems and the importance of continued
research into their security. Future efforts shouldprioritize developing robust defenses and evaluat-
ing real-world risks. We hope this work inspires
further advancements in safeguarding RAG sys-
tems.
2 Related Work
Attacks on RAG Systems and Corpus Poisoning.
Recent research has explored vulnerabilities in re-
trieval augmented generation (RAG) systems and
dense retrievers. Membership inference attacks
(Anderson et al., 2024; Li et al., 2024b) aim to
determine the presence of specific data in knowl-
edge bases. Other studies focus on manipulating
system outputs, such as PoisonedRAG (Zou et al.,
2024) crafting misinformation-laden passages, and
indirect prompt injection attacks (Greshake et al.,
2023) inserting malicious instructions. In corpus
poisoning, Zhong et al. (2023) proposed a gradient-
based approach inspired by HotFlip (Ebrahimi
et al., 2018), while Vec2Text (Morris et al., 2023)
introduced text embedding inversion. Unlike Poi-
sonedRAG, which targets individual queries, our
work aims to craft adversarial passages affecting
multiple queries simultaneously, presenting unique
challenges in corpus poisoning attacks on RAG
systems.
Genetic Algorithms in NLP. Genetic algorithms
(GA) have been increasingly applied to various
Natural Language Processing (NLP) tasks. Recent
studies have demonstrated GA’s efficacy in neural
architecture search (Chen et al., 2024), adversar-
ial prompt generation for LLM security (Liu et al.,
2023; Li et al., 2024a), and enhancing GA opera-
tions using LLMs (Guo et al., 2023; Lehman et al.,
2023; Meyerson et al., 2023). These applications
showcase the versatility of GA in addressing com-
plex NLP challenges. GARAG (Cho et al., 2024)
shares similarities with our work in applying GA
to RAG systems. However, our approach differs
significantly by targeting multiple queries simul-
taneously, rather than focusing on query-specific
adversarial passage generation. Furthermore, our
method employs a dynamic mechanism specifically
tailored to exploit the properties of dense retrievers,
offering a more nuanced approach to the challenge
of corpus poisoning in RAG systems.

Figure 1: Illustration of our motivations. Top: Demonstrating insensitivity to token order, where cosine similarity
remains nearly unchanged after permuting the tokens. Bottom: Highlighting bias towards influential tokens, shown
by the varying effects on cosine similarity when different tokens are deleted. Some tokens are more influential since
deleting them results in a larger change in similarity.
3 Method
3.1 Problem Formulation
LetCbe the corpus or knowledge base of the RAG
system, Q={q1, q2, ..., q n}be the set of queries,
ϕQbe the query embedding model, and ϕCbe the
context embedding model. Our goal is to generate
an adversarial passage awhose embedding maxi-
mizes the similarity to all the query embeddings,
increasing the likelihood of retrieval for any given
query. Formally, we aim to solve:
a= arg max
a′1
|Q|X
qi∈QϕQ(qi)TϕC(a′).
This can be rewritten as:
a= arg max
a′ϕC(a′)T1
|Q|X
qi∈QϕQ(qi)
= arg max
a′ϕC(a′)T¯ϕQ,
where ¯ϕQ=1
|Q|P
qi∈QϕQ(qi)represents the cen-
troid of all query embeddings. Therefore, the ad-
versarial passage is the solution to this optimization
problem, whose embedding should maximize the
similarity to the centroid of query embeddings. Togenerate multiple adversarial passages, we follow
Zhong et al. (2023) by applying k-means clustering
to the query embeddings and solving the optimiza-
tion problem for each cluster centroid, resulting in
kadversarial passages.
3.2 Motivation
The design of our method stems from two key ob-
servations about retrievers, as demonstrated in Fig-
ure 1: (1) Insensitivity to token order: Retrievers
show remarkable insensitivity to the order of words.
By comparing the original sentence with various
permutations of its tokens, including extreme cases
like complete reversal, we observe only negligible
differences in similarity. (2) Bias towards influen-
tial tokens: Different words contribute unequally
to the overall similarity score. When deleting cer-
tain words, the similarity drops significantly more
than for others. For instance, removing "animal" or
"fastest" from the sentence "Which animal is the
fastest?" causes a substantial decrease in similar-
ity, while deleting "is" or "the" has only a minor
impact.
Our method leverages these properties to effi-
ciently craft adversarial passages, resulting in a
more effective and computationally efficient ap-
proach.

Figure 2: An overview of our proposed method .
4 Dynamic Importance-Guided Genetic
Algorithm (DIGA)
We propose Dynamic Importance-Guided Genetic
Algorithm (DIGA), which is designed to efficiently
craft adversarial passages by exploiting two key
properties of retrievers: their insensitivity to token
order and bias towards influential tokens. DIGA
employs a genetic algorithm framework, dynami-
cally adjusting its genetic policies based on token
importance. Figure 2 illustrates the pipeline of our
method, which consists of several key components.
We will now give a detailed explanation of these
components in our proposed method. We provide
an algorithmic description of DIGA in Algorithm 1.
4.1 Token Importance Calculation
Our second observation—that different tokens con-
tribute unequally to the similarity score—motivates
us to quantify token importance. This quantifica-
tion is crucial for guiding the genetic algorithm
toward more effective adversarial passages, lead-
ing to faster convergence.
To efficiently calculate token importance across
a large corpus, we employ Term Frequency-Inverse
Document Frequency (TF-IDF) (Salton et al.,
1975). Our experiments demonstrate that TF-
IDF values align well with the Leave-one-out val-
ues (Cook, 1977), as illustrated in Figure 1. Im-
portantly, TF-IDF calculation is considerably more
efficient than LOO, making it suitable for large
corpus.4.2 Population Initialization
Initialization is a critical component in genetic algo-
rithms, influencing both the quality and diversity of
the initial population. Our method employs a score-
based initialization strategy that aligns with our
observation of retrievers’ bias towards influential
tokens. We initialize the population by sampling
tokens based on importance scores derived from
TF-IDF values, ensuring that more influential to-
kens have a higher probability of inclusion in the
initial adversarial passages. This approach effec-
tively focuses the search on promising regions of
the solution space from the start. The probabilis-
tic nature of token selection balances exploitation
of influential tokens with exploration of less com-
mon ones, creating a population biased towards
effective adversarial passages while maintaining
diversity for the genetic algorithm.
4.3 Fitness Evaluation
As shown in Section 3.1, there is a closed-form
solution to our optimization problem, which in-
volves maximizing the cosine similarity between
the embedding of an adversarial passage and the
centroid of query embeddings. Consequently, we
directly use this cosine similarity as our fitness
evaluation function. Formally, given a set of
queries Qcfrom the same cluster assigned by
k-means, the fitness of an adversarial passage
ais defined as: cos(ϕC(a),¯ϕQ), where ¯ϕQ=
1
|Qc|P
qi∈QcϕQ(qi).
4.4 Genetic Policies
Our genetic algorithm utilizes three core operations:
selection, crossover, and mutation. These opera-

tions are tailored to exploit the properties of dense
retrievers we identified earlier, ensuring both the
effectiveness of the generated adversarial passages
and the efficiency of the overall process.
Selection. Selection is performed using a com-
bination of elitism and roulette wheel selection.
Given a population of Nadversarial passages and
an elitism rate α, we first preserve the top N∗α
passages with the highest fitness scores. This en-
sures that the best solutions are carried forward to
the next generation. For the remaining N−N∗α
slots, we use a softmax-based selection probability.
This method balances exploration and exploitation
by giving better-performing adversarial passages a
higher chance of selection while still allowing for
diversity in the population.
Crossover. Our crossover operation exploits the
insensitivity to token order in retrievers. This moti-
vation highlights the fact that we should focus on
which tokens to choose, rather than how to arrange
them. This insight leads us to implement a single-
point crossover where two parent sequences are
split at a random point, and their tails are swapped
to create two offspring. This approach preserves
influential tokens from both parents while allowing
for new combinations.
Mutation. Mutation maintains population diver-
sity in genetic algorithms. Our DIGA mutation
strategy dynamically adjusts based on token im-
portance, leveraging the retriever’s bias towards
influential tokens. For each token in an adversarial
passage, we calculate a replacement probability:
Preplace = min(C−si)·τ
Z+γ,1
,
where Cis the maximum token score, siis the
current token’s score, τis a temperature parameter,
andγis a baseline mutation rate. Zis a normaliza-
tion factor defined as: Z=1
nP
i(C−si).
Temperature τmodulates mutation sensitivity to
token importance, while γensures a baseline muta-
tion rate. This importance-guided strategy balances
token preservation with exploration, adapting to the
evolving fitness landscape.
5 Experiments
5.1 Experimental Setup
Datasets. In our study, we investigate four di-
verse datasets to simulate a range of retrieval aug-mented generation scenarios. These datasets, char-
acterized by varying numbers of queries and corpus
sizes, are: Natural Questions (NQ) (Kwiatkowski
et al., 2019), NFCorpus (Boteva et al., 2016), FiQA
(Maia et al., 2018), and SciDocs (Cohan et al.,
2020). Each dataset presents unique characteris-
tics that allow us to examine the efficacy of our
proposed methods across different domains and re-
trieval contexts. A comprehensive overview of the
statistical properties of these datasets is provided
in Appendix A.
Models. In our experiments, we focus on
three dense retrievers: GTR-base (Ni et al.,
2022), ANCE (Xiong et al., 2020) and DPR-
nq (Karpukhin et al., 2020). The last two models
are mainly included to test the transferability of
different attack methods.
Apart from the retriever in the RAG system, the
attack method Vec2Text (Morris et al., 2023) also
requires an inversion model to invert the embedding
to text. We use the model provided by Morris et al.
(2023), which is trained on 5 million passages from
the NQ dataset where each passage consists of 32
tokens.
Evaluation Metric. To evaluate the effective-
ness of attacks, we define Attack Success Rate@ k
(ASR@ k), which is the percentage of queries for
which at least one adversarial passage is retrieved
in the top- kresults.
To evaluate cross-dataset transferability, we de-
fine the Transferability Score as:
1−1
|D\ {i}|X
j∈D\{i}ASR i,i−ASR i,j
ASR i,i,
where ASR i,jis the ASR when evaluating on
dataset iusing adversarial passages generated from
dataset j,ASR i,iis the in-domain ASR where the
evaluation and attack datasets are the same, and
D\ {i}is the set of all datasets excluding i. This
metric quantifies how well attacks generalize across
different datasets.
We also evaluate computational efficiency by
measuring method-specific execution time, exclud-
ing shared preprocessing tasks like corpus encod-
ing and k-means clustering. We also track peak
GPU memory usage to assess each method’s maxi-
mum memory demands during execution.
Implementation Details. Our experimental
protocol follows the methodology established

Dataset MethodNumber of Adversarial Passages
Time GPU Usage 1 10 50
ASR@5 ASR@20 ASR@5 ASR@20 ASR@5 ASR@20
NQWhite-box Methods
HotFlip 0.1 1.3 0.4 3.2 2.0 9.6 6.5x 10.2x
Black-box Methods
Vec2Text 0.0 0.0 0.1 1.8 1.4 4.3 - 1.43x
DIGA (ours) 0.0 0.0 0.1 1.5 1.7 5.2 1.0x 1.0x
NFCorpusWhite-box Methods
HotFlip 37.2 58.8 44.6 68.1 47.7 71.2 8.6x 10.5x
Black-box Methods
Vec2Text 0.6 2.5 8.1 14.2 23.0 38.7 - 1.32x
DIGA (ours) 5.7 10.8 11.2 18.2 26.1 44.3 1.0x 1.0x
FiQAWhite-box Methods
HotFlip 0.5 1.9 2.5 7.4 8.5 25.2 9.3x 9.2x
Black-box Methods
Vec2Text 0.0 0.0 0.9 1.2 3.9 11.9 - 1.2x
DIGA (ours) 0.0 0.0 1.3 1.7 4.4 14.3 1.0x 1.0x
SciDocsWhite-box Methods
HotFlip 0.7 1.8 2.0 9.6 4.0 14.3 8.4x 6.0x
Black-box Methods
Vec2Text 0.0 0.3 1.7 5.6 11.2 26.5 - 1.4x
DIGA (ours) 0.2 0.9 1.3 6.6 7.8 20.3 1.0x 1.0x
Table 2: ASR, time and GPU usage comparison for different methods . ’-’ indicates that Vec2Text’s time usage
is not directly comparable, as it requires training a separate model before generating adversarial passages.
by Zhong et al. (2023). We utilize the training sets
of the respective datasets to generate adversarial
passages, subsequently evaluating their effective-
ness on the corresponding test sets. For SciDocs,
which does not have a separate training set, we
perform both the attack and evaluation on the same
dataset.
We fixed the size of our adversarial passages to
50 tokens to have a consistent comparison with
other baselines. For the similarity metric, we use
dot product. A noteworthy aspect of our implemen-
tation is the decomposition of adversarial passage
generation into two distinct parts. We apply DIGA
to 80% of the tokens, allowing us to rapidly ap-
proach an approximate solution. For the remaining
20% of tokens, we implement a fine-tuning stage
using the vanilla genetic algorithm, enabling subtle
adjustments to further optimize the passage. For
additional implementation details, see Appendix B.
6 Results
6.1 ASR Results
Based on the experimental results presented in Ta-
ble 2, we can conduct a thorough analysis of the
performance of our proposed method, DIGA, incomparison to existing approaches across various
datasets and attack scenarios.
HotFlip, as a white-box method with access
to the retriever’s gradient, generally outperforms
black-box methods, particularly when the number
of adversarial passages is low. This is exempli-
fied by its performance on the NFCorpus dataset,
where it achieves an ASR@20 of 58.8% with just
one adversarial passage. However, this superior
performance comes at a significant computational
cost, with HotFlip requiring 6.5x to 9.3x more time
and 6.0x to 10.5x more GPU resources compared
to our method, depending on the dataset.
Among black-box methods, our proposed DIGA
consistently outperforms Vec2Text across differ-
ent datasets, particularly in NFCorpus and Sci-
Docs. DIGA achieves higher ASR, especially with
more adversarial passages, without requiring addi-
tional model training. It also maintains the lowest
time and GPU usage among all evaluated methods,
demonstrating superior efficiency.
In conclusion, while DIGA may not always
match the white-box HotFlip method’s ASR, it
consistently outperforms the black-box Vec2Text
method with lower computational requirements.

Method Evaluation DatasetAttack DatasetTransferability Score
NQ SciDocs FIQA
Vec2TextNQ 8.1 0.0↓100.0%0.1↓98.8%0.6
Scidocs 0.0↓100.0%28.6 0.0↓100.0%0.0
FIQA 0.0↓100.0%0.0↓100.0%11.9 0.0
DIGA (ours)NQ 5.2 0.1↓98.1%0.5↓90.4%5.8
Scidocs 0↓100.0%30.1 1.8↓94.0%3.0
FIQA 0.0↓100.0%0.0↓100.0%14.3 0.0
Table 3: Cross-Dataset Transferability Analysis. The results represent the adversarial success rate (ASR@20) after
injecting 50 adversarial passages, using GTR-base as the retriever for all experiments. Cells with gray background
denote in-domain attacks, where the evaluation dataset matches the attack dataset.
This makes DIGA particularly suitable for large-
scale corpus poisoning attacks, especially when the
retriever is inaccessible or resources are limited.
6.2 Transferability Results
Transferability on Different Datasets. Table 3
presents the cross-dataset transferability results
for black-box Vec2Text and our proposed DIGA
method. The results reveal that both methods strug-
gle with cross-dataset generalization, as evidenced
by the significant performance drops when attack-
ing datasets different from the one used for gen-
erating adversarial passages. DIGA demonstrates
slightly better transferability compared to Vec2Text.
This suggests that DIGA’s adversarial passages re-
tain more generalizable features across datasets.
MethodEvaluation Retriever
ANCE DPR-nq BM25 (sparse)
HotFlip 0.0 0.0 0.0
Vec2Text 9.2 0.6 1.5
DIGA (ours) 7.3 0.4 2.9
Table 4: Transferability analysis across different re-
trievers. Results are ASR@20 for 50 adversarial pas-
sages generated using GTR-base model on the NFCor-
pus dataset.
Transferability on Different Retrievers. In this
analysis, we investigate scenarios where there is
a mismatch between the attack retriever and the
actual evaluation retriever used in the RAG system.
For dense retrievers, we focus on ANCE and DPR-
nq. We also consider the case where the retriever
is a sparse retriever, specifically BM25 (Robert-
son et al., 2009). We use GTR-base as the attack
retriever to generate 50 adversarial passages and in-
ject them into the corpus of NFCorpus. The results
are presented in Table 4. We observe that HotFlip
Figure 3: Scalability Analysis. Note that the HotFlip
method is too computationally expensive to generate 50
more adversarial passages.
exhibits zero transferability in this scenario, with
its ASR@20 dropping from 71.2 to 0.0 in all cases.
Other black-box methods demonstrate comparable
transferability on different dense retrievers. No-
tably, our method shows excellent transfer to sparse
retrievers. This is primarily because the adversarial
passages generated by our method typically include
important tokens from the query, resulting in a to-
ken match.
7 Discussion and Analysis
Scalability Analysis. Figure 3 illustrates the scal-
ability of different corpus poisoning methods as
we increase the number of adversarial passages.
Our proposed DIGA method demonstrates superior
scalability and performance compared to existing
approaches. HotFlip is excluded beyond 50 pas-
sages due to its prohibitive computational cost for
large-scale attacks. DIGA consistently outperforms
Vec2Text across all scales. This analysis under-
scores DIGA’s effectiveness in large-scale corpus
poisoning scenarios, particularly for attacks on ex-

tensive knowledge bases where a higher number of
adversarial passages is required.
Impact of Initialization and Length. In the de-
sign of DIGA, two crucial factors influence the
initialization of adversarial passages: the length of
the passage and the method of initialization. To
investigate these factors, we conduct experiments
with varying passage lengths and compared score-
based initialization against random initialization.
The results (see Table 5) show that score-based
initialization consistently outperforms random ini-
tialization. Longer passages yield better results for
both methods, with 50-token passages achieving
the highest Attack Success Rate. This suggests that
longer passages allow for more influential tokens
to be included, enhancing attack effectiveness.
Initialization MethodLength (tokens)
10 20 50
Score-based 17.8 20.5 44.3
Random 2.1 3.4 15.4
Table 5: Ablation study on initialization method and
adversarial passage length. Results show ASR@20
for different initialization methods and passage lengths
on the NFCorpus dataset.
Adversarial Passages Analysis. To evaluate
the generated adversarial passages, we use GPT-
2 (Radford et al., 2019) to assess their perplex-
ity. Figure 4 illustrates the distribution of log-
perplexity scores for passages generated by dif-
ferent methods. Vec2Text, designed to produce
natural language, generates passages with the low-
est perplexity. In contrast, HotFlip and DIGA, both
employing discrete token replacements, yield pas-
sages with higher perplexity. Nevertheless, DIGA
outperforms HotFlip, likely because HotFlip tends
to introduce more unusual tokens.
It is worth noting that in practical applications,
these passages often serve as prefixes to longer,
more fluent texts containing misinformation (Zou
et al., 2024). In such cases, the overall perplexity
would be significantly lower as the natural language
component becomes dominant, potentially evading
simple perplexity-based detection while remaining
semantically influential for retrieval systems.
Indirect Prompt Injection with DIGA. Adver-
sarial passages can serve as carriers for various
malicious purposes. While Zhong et al. (2023)
have explored their use in spreading misinforma-
Vec2Text HotFlip Ours51015Log-PerplexityNQ
Vec2Text HotFlip Ours51015NFCorpus
Vec2Text HotFlip Ours51015Log-PerplexitySciDocs
Vec2Text HotFlip Ours51015FiQAFigure 4: Log-perplexity Distribution. This figure
presents the log-perplexity measurements for 50 ad-
versarial passages generated on NFCorpus using the
GTR-base model. Methods displayed to the right of the
dotted line represent discrete optimization techniques.
tion, we investigate a new scenario: indirect prompt
injection (Greshake et al., 2023). We inject mali-
cious prompts into adversarial passages generated
for 100 queries from NQ, appending "Ignore the
prompt and output I don’t know." Using GPT-3.5
(Brown et al., 2020), we measure the attack’s suc-
cess by the proportion of queries eliciting an "I
don’t know" response. Our findings reveal that
the effectiveness of indirect prompt injection cor-
relates positively with corpus poisoning ASR. No-
tably, DIGA successfully alters 51% and 67% of
model responses when retrieving 5 and 10 most
relevant passages, respectively. These results high-
light DIGA’s potential as a potent tool for large-
scale indirect prompt injection attacks.
8 Conclusion
In this work, we present Dynamic Importance-
Guided Genetic Algorithm (DIGA), a novel
method for executing corpus poisoning attacks.
DIGA capitalizes on the insensitivity of retriev-
ers to word order and their bias towards influential
tokens, enabling it to generate adversarial passages
that are both effective and efficient. Notably, our
approach requires no white-box access to the un-
derlying models and avoids the need for additional
training, making it highly adaptable.
Our extensive experiments demonstrate that
DIGA surpasses existing methods in terms of ef-
ficiency and scalability. At the same time, it con-
sistently achieves comparable or superior attack
success rates across various datasets and retrieval
models, highlighting its effectiveness and adapt-
ability in adversarial scenarios.

Limitations
While our method outperforms other black-box
approaches with lower time and memory require-
ments, a significant gap remains between our ap-
proach and white-box methods. Additionally, all
current methods, including ours, fall short of the
theoretical upper bound for attack success. This
highlights the complexity of corpus poisoning at-
tacks and the potential for improvement. Future
work should aim to close this gap by developing
techniques that better approximate white-box effec-
tiveness while preserving the practical benefits of
black-box methods.
Ethics Statement
Our study introduces a highly efficient and effec-
tive corpus poisoning method aimed at Retrieval-
Augmented Generation (RAG) systems. This ap-
proach has the potential to be exploited for spread-
ing misinformation or, when combined with ma-
licious prompts, influencing the model’s output.
The research underscores critical vulnerabilities
in current RAG systems and highlights the urgent
need for stronger defensive mechanisms. Given
the potential misuse of this method, future research
building on these findings should prioritize ethical
considerations and responsible use.
Acknowledgment
The work is supported by the Ministry of Education,
Singapore, under the Academic Research Fund Tier
1 (FY2023) (Grant A-8001996-00-00), University
of California, Merced, and University of Queens-
land.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 .
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2024. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation. Preprint , arXiv:2405.20446.
Vera Boteva, Demian Gholipour, Artem Sokolov, and
Stefan Riezler. 2016. A full-text learning to rank
dataset for medical information retrieval.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, ArvindNeelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen,
Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam Mc-
Candlish, Alec Radford, Ilya Sutskever, and Dario
Amodei. 2020. Language models are few-shot learn-
ers.Preprint , arXiv:2005.14165.
Angelica Chen, David Dohan, and David So. 2024. Evo-
prompting: language models for code-level neural
architecture search. Advances in Neural Information
Processing Systems , 36.
Canyu Chen and Kai Shu. 2023. Can llm-generated
misinformation be detected? arXiv preprint
arXiv:2309.13788 .
Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho
Hwang, and Jong C Park. 2024. Typos that broke the
rag’s back: Genetic attack on rag pipeline by simulat-
ing documents in the wild via low-level perturbations.
arXiv preprint arXiv:2404.13948 .
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel S. Weld. 2020. Specter:
Document-level representation learning using
citation-informed transformers. In ACL.
R Dennis Cook. 1977. Detection of influential obser-
vation in linear regression. Technometrics , 19(1):15–
18.
Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing
Dou. 2018. HotFlip: White-box adversarial exam-
ples for text classification. In Proceedings of the 56th
Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers) , pages 31–36,
Melbourne, Australia. Association for Computational
Linguistics.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. Preprint ,
arXiv:2405.06211.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz.
2023. Not what you’ve signed up for: Compromis-
ing real-world llm-integrated applications with indi-
rect prompt injection. In Proceedings of the 16th
ACM Workshop on Artificial Intelligence and Secu-
rity, pages 79–90.
Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao
Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yu-
jiu Yang. 2023. Connecting large language models

with evolutionary algorithms yields powerful prompt
optimizers. arXiv preprint arXiv:2309.08532 .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. Preprint , arXiv:2311.05232.
Ehsan Kamalloo, Nandan Thakur, Carlos Lassance,
Xueguang Ma, Jheng-Hong Yang, and Jimmy Lin.
2023. Resources for brewing beir: Reproducible ref-
erence models and an official leaderboard. Preprint ,
arXiv:2306.07471.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. arXiv preprint
arXiv:2004.04906 .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Matthew Kelcey,
Jacob Devlin, Kenton Lee, Kristina N. Toutanova,
Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: a benchmark for question answering
research. Transactions of the Association of Compu-
tational Linguistics .
Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal
Ndousse, Cathy Yeh, and Kenneth O Stanley. 2023.
Evolution through large models. In Handbook of
Evolutionary Machine Learning , pages 331–366.
Springer.
Xiaoxia Li, Siyuan Liang, Jiyi Zhang, Han Fang, Ais-
han Liu, and Ee-Chien Chang. 2024a. Seman-
tic mirror jailbreak: Genetic algorithm based jail-
break prompts against open-source llms. Preprint ,
arXiv:2402.14872.
Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang.
2024b. Generating is believing: Membership infer-
ence attacks against retrieval-augmented generation.
Preprint , arXiv:2406.19234.
Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song,
Changlong Sun, Xiaofeng Wang, Wei Lu, and Xi-
aozhong Liu. 2022. Order-disorder: Imitation adver-
sarial attacks for black-box neural ranking models.
InProceedings of the 2022 ACM SIGSAC Conference
on Computer and Communications Security , pages
2025–2039.
Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei
Xiao. 2023. Autodan: Generating stealthy jailbreak
prompts on aligned large language models. arXiv
preprint arXiv:2310.04451 .
Macedo Maia, Siegfried Handschuh, André Freitas,
Brian Davis, Ross McDermott, Manel Zarrouk, and
Alexandra Balahur. 2018. Www’18 open challenge:
financial opinion mining and question answering. InCompanion proceedings of the the web conference
2018 , pages 1941–1942.
Meta. 2024. The llama 3 herd of models. Preprint ,
arXiv:2407.21783.
Elliot Meyerson, Mark J Nelson, Herbie Bradley, Adam
Gaier, Arash Moradi, Amy K Hoover, and Joel
Lehman. 2023. Language model crossover: Vari-
ation through few-shot prompting. arXiv preprint
arXiv:2302.12170 .
John Morris, V olodymyr Kuleshov, Vitaly Shmatikov,
and Alexander Rush. 2023. Text embeddings reveal
(almost) as much as text. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 12448–12460, Singapore.
Association for Computational Linguistics.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo
Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan,
Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022.
Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing , pages
9844–9855, Abu Dhabi, United Arab Emirates. As-
sociation for Computational Linguistics.
Alec Radford, Jeff Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners.
Nisarg Raval and Manisha Verma. 2020. One word at a
time: adversarial attacks on retrieval models. arXiv
preprint arXiv:2008.02197 .
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Foundations and Trends ®in Information Re-
trieval , 3(4):333–389.
G. Salton, A. Wong, and C. S. Yang. 1975. A vector
space model for automatic indexing. Commun. ACM ,
18(11):613–620.
Congzheng Song, Alexander M Rush, and Vitaly
Shmatikov. 2020. Adversarial semantic collisions.
arXiv preprint arXiv:2011.04743 .
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold
Overwijk. 2020. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808 .
Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun
Chen, and Qian Lou. 2024. Badrag: Identifying vul-
nerabilities in retrieval augmented generation of large
language models. arXiv preprint arXiv:2406.00083 .
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang,
Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai,

Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Ke-
qin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni,
Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize
Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan,
Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge,
Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren,
Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing
Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan,
Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang,
Zhifang Guo, and Zhihao Fan. 2024. Qwen2 techni-
cal report. Preprint , arXiv:2407.10671.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing,
Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang,
Dawei Yin, Yi Chang, et al. 2024. The good and the
bad: Exploring privacy issues in retrieval-augmented
generation (rag). arXiv preprint arXiv:2402.16893 .
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 13764–13775, Singa-
pore. Association for Computational Linguistics.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. Poisonedrag: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. Preprint , arXiv:2402.07867.
A Dataset Statistics
In this section, we present the statistics of the
datasets used in our paper. All datasets are sourced
from the BEIR benchmark (Kamalloo et al., 2023),
and their key characteristics are summarized in Ta-
ble 6.
Dataset Queries Corpus Rel/Q
NQ 3,452 2.68M 1.2
NFCorpus 323 3.6K 38.2
FiQA 648 57K 2.6
SciDocs 1,000 25K 4.9
Table 6: Dataset Statistics. Summary of key charac-
teristics for NQ (Kwiatkowski et al., 2019), NFCor-
pus (Boteva et al., 2016), FiQA (Maia et al., 2018), and
SciDocs (Cohan et al., 2020) datasets. Rel/Q refers to
Average number of Relevant Documents per Query.
B Implementation Details
For the HotFlip attack implementation, we largely
adhere to the configuration described by Zhong
et al. (2023), with some modifications to reduce
running time. Specifically, we maintain a batch
size of 64 but reduce the number of potential token
replacements from the top-100 to the top-50. Addi-
tionally, we decrease the number of iteration stepsfrom 5,000 to 500 to balance attack effectiveness
with computational resources.
Our proposed method, DIGA, is implemented
with a population size of 100 and executed for 200
generations. We employ an elitism rate of 0.1 and a
crossover rate of 0.75. Based on preliminary exper-
imental results, we set the temperature parameter
τto 1.05 and the baseline mutation rate γto 0.05.
In all steps, token repetition is checked. All experi-
ments are conducted using four NVIDIA GeForce
RTX 3090 GPUs.
C Algorithm Description
In this section, we present the detailed algorithms
for our Dynamic Importance-Guided Genetic Al-
gorithm (DIGA) and the process of performing a
corpus injection attack using DIGA. Algorithm 1
outlines the main steps of DIGA. The algorithm
starts by calculating token importance scores and
initializing the population using score-based sam-
pling (Sec. 4.2). It then iterates through genera-
tions, applying genetic operations guided by to-
ken importance (Sec. 4.4) and evaluating fitness
(Sec. 4.3) until the termination criteria are met.
Algorithm 1 Dynamic Importance-Guided Genetic
Algorithm (DIGA)
Require: Corpus C′, Query set Q′, Population size
N, Maximum generations Gmax, Elitism rate
α
1:Calculate token importance scores w(t)for all
t∈ C′using TF-IDF
2:P←InitializePopulation( C′,N,w)▷Sec.4.2
3:F←EvaluateFitness( P,Q′) ▷Sec.4.3
4:forg= 1toGmaxdo
5: Pnew← ∅
6: Pelite←SelectElite( P,F,α)
7: while|Pnew|< N− |Pelite|do
8: p1, p2←SelectParents( P,F)
9: c1, c2←Crossover( p1,p2)
10: c1←ImportanceGuidedMutation( c1,
w)
11: c2←ImportanceGuidedMutation( c2,
w)
12: Pnew←Pnew∪c1, c2
13: end while
14: P←Pelite∪Pnew
15: F←EvaluateFitness( P,Q′)
16:end for
17:a∗←arg maxp∈PF(p)
18:return a∗

To perform a corpus injection attack using DIGA,
we follow the process described in Algorithm 2:
In this attack, we first cluster the queries into k
Algorithm 2 Corpus Injection Attack using DIGA
Require: Original corpus C, Original query set Q,
Number of adversarial passages k, DIGA ratio
β, Passage length L
1:Qclusters ←k-Means( Q,k)
2:foreach cluster QiinQclusters do
3: aDIGA
i←DIGA( C′,Qi,⌊βL⌋)▷Apply
DIGA to βfraction of tokens
4: aGA
i←VanillaGA( C′,Qi,L− ⌊βL⌋)▷
Apply GA to remaining tokens
5: ai←Merge( aDIGA
i ,aGA
i)
6:C ← C ∪ ai
7:end for
8:return C
groups. For each cluster, we generate an adversarial
passage using DIGA. We then select βfraction of
tokens from this passage and use a vanilla genetic
algorithm to refine the remaining 1−βof tokens.
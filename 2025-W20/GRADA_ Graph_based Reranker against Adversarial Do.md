# GRADA: Graph-based Reranker against Adversarial Documents Attack

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Published**: 2025-05-12 13:27:35

**PDF URL**: [http://arxiv.org/pdf/2505.07546v1](http://arxiv.org/pdf/2505.07546v1)

## Abstract
Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large
language models (LLMs) by integrating external knowledge from retrieved
documents, thereby overcoming the limitations of models' static intrinsic
knowledge. However, these systems are susceptible to adversarial attacks that
manipulate the retrieval process by introducing documents that are adversarial
yet semantically similar to the query. Notably, while these adversarial
documents resemble the query, they exhibit weak similarity to benign documents
in the retrieval set. Thus, we propose a simple yet effective Graph-based
Reranking against Adversarial Document Attacks (GRADA) framework aiming at
preserving retrieval quality while significantly reducing the success of
adversaries. Our study evaluates the effectiveness of our approach through
experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b,
Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with
results from the Natural Questions dataset demonstrating up to an 80% reduction
in attack success rates while maintaining minimal loss in accuracy.

## Full Text


<!-- PDF content starts -->

arXiv:2505.07546v1  [cs.IR]  12 May 2025GRADA: Graph-based Reranker against Adversarial Documents Attack
Jingjie Zheng1, Aryo Pradipta Gema2, Giwon Hong2, Xuanli He3,
Pasquale Minervini2,Youcheng Sun4,Qiongkai Xu5
1University of Melbourne,2University of Edinburgh,3University College London
4University of Manchester,5Macquarie University
jingjzheng@student.unimelb.edu.au ,zodiachy@gmail.com
{aryo.gema, giwon.hong, p.minervini}@ed.ac.uk
youcheng.sun@manchester.ac.uk ,qiongkai.xu@mq.edu.au
Abstract
Retrieval Augmented Generation (RAG) frame-
works improve the accuracy of large language
models (LLMs) by integrating external knowl-
edge from retrieved documents, thereby over-
coming the limitations of models’ static intrin-
sic knowledge. However, these systems are sus-
ceptible to adversarial attacks that manipulate
the retrieval process by introducing documents
that are adversarial yet semantically similar to
the query. Notably, while these adversarial doc-
uments resemble the query, they exhibit weak
similarity to benign documents in the retrieval
set. Thus, we propose a simple yet effective
Graph-based Reranking against Adversarial
Document Attacks (GRADA) framework aim-
ing at preserving retrieval quality while sig-
nificantly reducing the success of adversaries.
Our study evaluates the effectiveness of our ap-
proach through experiments conducted on five
LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b-
Instruct, Llama3.1-70b-Instruct, and Qwen2.5-
7b-Instruct. We use three datasets to assess per-
formance, with results from the Natural Ques-
tions dataset demonstrating up to an 80% reduc-
tion in attack success rates while maintaining
minimal loss in accuracy.
1 Introduction
Large Language Models (LLMs; Brown et al.,
2020) have demonstrated remarkable performance
across a wide range of natural language process-
ing tasks, including question answering (Fourrier
et al., 2024), text summarization (Graff et al., 2003;
Rush et al., 2015), and information retrieval (Yates
et al., 2021). However, LLMs inherently rely on
the static knowledge embedded in their training
data, limiting their adaptability to new and domain-
specific information. Retrieval-Augmented Gen-
eration (RAG; Lewis et al., 2020) was introduced
to bridge this gap by integrating external retrieval
modules, allowing LLMs to access and incorporate
relevant, up-to-date knowledge.
Question : "Who is the current CEO of Apple?"
Retriever Poisoned
CorpusAdversarial RAG
attacks exploit
query-document
similarity
⚠  Document #1  ⚠ 
"Who is the current CEO of Apple?  The current CEO of Apple is Elon Musk "Figure 1: An example of adversarial RAG attack which
exploits query-document similarity by prepending the
poisonous document with the query.
While RAG enhances the flexibility of LLMs, it
also introduces new vulnerabilities. Adversaries
can exploit retrieval mechanisms by injecting ma-
nipulated documents into the corpus (Zhong et al.,
2023; Clop and Teglia, 2024; Greshake et al., 2023;
Pasquini et al., 2024), subtly altering rankings to
mislead LLM outputs. As shown in Figure 1, these
adversarial documents mimic query-relevant pat-
terns, making them difficult to detect while degrad-
ing the reliability of retrieval-based LLM systems.
In practice, LLMs are used in the search engine
to produce direct answers to user’s query, called
Answer Engine Optimization (AEO) (Yalçın and
Köse, 2024). Adversarial documents can be created
using the previous methods to achieve higher rank-
ings in the search results. Also, guide the search
engine to produce malicious context that can poten-
tially harm the user (Hammond, 2024; Venkit et al.,
2024).
Existing noise filtering methods, such as Hy-
brid List Aware Transformer Reranking (HLATR,
Zhang et al., 2022) and BAAI General Embed-
dings (BGE-reranker, Xiao et al., 2023), focus
on improving document relevance by filtering out
generic noise or low-quality content. However,

GRADA
1) Similarity score (construct a graph)Vanilla RAG pipeline Retrieved Documents
Retriever
Corpus"Who is the current CEO of Apple?"
LLM Elon Musk
Answer is corrupted by the
malicious passages
LLM Tim CookQuery
 2) Ranking (Lower malicious docs)
Filtering 3
84
95
101
62
7Figure 2: An overview of GRADA. A vanilla RAG pipeline concatenates all retrieved documents along with the
question as the input to the LLM. However, the accuracy of this pipeline can be easily harmed by malicious passages.
In contrast, GRADA uses a graph-based approach to isolate and filter out malicious passages before passing the
retrieved documents as the LLM input.
these methods are ineffective against adversarial
attacks that exploit query-document similarity pat-
terns to evade detection. In addition, a recent study
has reviewed current graph-based reranking meth-
ods. It shows a potential path of using graphs in fu-
ture information retrieval (Zaoad et al., 2025) tasks
but the effects on adversarial documents remains
unknown. On the other hand, specialized adversar-
ial defenses, such as keyword filtering and decod-
ing aggregation (Xiang et al., 2024), can success-
fully remove adversarial content but at the cost of
discarding valuable benign documents, ultimately
weakening retrieval performance. This trade-off
highlights the need for a more nuanced defense
mechanism that can distinguish between adversar-
ial and benign documents without compromising
retrieval quality.
To address this challenge, we propose Graph-
based Reranking against Adversarial Document
Attacks (GRADA), a novel and effective defense
framework designed to protect RAG systems from
adversarial retrieval manipulations. Our key insight
is that adversarial documents, while optimized for
high query similarity, exhibit weaker semantic co-
herence with genuinely relevant documents in the
retrieval set. Leveraging this property, we con-
struct a graph where each retrieved document is
represented as a node, and edges capture document-
document similarity relationships. By propagat-
ing ranking scores through this graph structure,
our approach prioritizes clusters of semanticallyconsistent documents while suppressing adversar-
ially crafted outliers. As illustrated in Figure 2,
our method significantly enhances the robustness
of RAG-based LLMs, mitigating adversarial in-
fluences while preserving the integrity of benign
retrieval results.
We conducted comprehensive experiments on
Natural Questions (NQ), MS-MARCO, and Hot-
potQA across five different models. Our method
has shown at least a 30% decrease in reducing the
Attack Success Rate (ASR), with improvements of
up to 80% across various adversarial attack strate-
gies.
We summarize our contributions as follows:
•We introduce GRADA, a weighted similarity
graph among retrieved documents iteratively
propagates scores to mitigate the impacts of
adversarial passages.
•We introduce a novel scoring function that si-
multaneously captures both query-document
and document-document correlations, thereby
improving robustness against adversarial at-
tempts to mimic the query.
•Conducted comprehensive experiments across
three different datasets against four chosen
attacks. Showing GRADA’s advantages over
the current defense baselines.

2 Related Work
The IR community has long examined how mali-
cious actors distort ranking functions. (Gyongyi
and Garcia-Molina, 2005) propose a taxonomy
of web-spam strategies, categorizing attacks into
content-based, link-based, and behavior-based ma-
nipulations. Content-based spam involves manip-
ulating the textual content of web pages to boost
rankings, such as stuffing keywords or inserting
irrelevant but popular terms. Link-based spam,
including techniques like link farms, exploits the
graph structure of the web to artificially inflate
a page’s authority. (Ntoulas et al., 2006) focus
on detecting such content-based spam using sta-
tistical features extracted from term distributions
and document structure. Their work highlights
how spam pages often exhibit unnatural language
models and disproportionately high keyword densi-
ties. (Castillo and Davison, 2011) provide a com-
prehensive survey of adversarial techniques in web
search, including cloaking (serving different con-
tent to users and crawlers), redirection, and decep-
tive anchor texts. These traditional adversarial IR
strategies, although originally developed to mis-
lead search engines, reveal core vulnerabilities in
retrieval systems—vulnerabilities that persist, al-
beit in more complex forms, in modern neural IR
and RAG systems.
When RAG systems came out, Corpus poisoning
attacks (Zhong et al., 2023) show a possible new at-
tack surface on LLMs. However, this method does
not directly affect the accuracy of the LLM; instead,
it focuses on the retriever. Later, prompt injection
attacks were introduced to bypass the retriever and
affect the generator successfully (Greshake et al.,
2023; Pasquini et al., 2024). However, compared
to the prior work, these methods are unstable in
ensuring the retriever retrieves the adversarial pas-
sage every time. While these attacks are situated in
modern LLM -based retrieval, adversarial manipu-
lation of information -retrieval systems has a much
longer history that is instructive for our setting.
More recently, PoisonedRAG (Zou et al., 2024)
was proposed as a more stable attack. It uses two
passages concatenated together, with one of them
appended to guarantee the retrieval of the adver-
sarial passage and one to achieve a given adver-
sarial goal on the generator. The goal is to let the
LLM output the answer the attacker wants. Poi-
sonedRAG inspired a lot of the new attacks. Phan-
tom (Chaudhari et al., 2024), which introduces a
0.00.20.40.60.81.0
BM25 Similarity Matrix
Similarity012345678910
109876543210Figure 3: BM25 similarity matrix among retrieved doc-
uments, where D0-D4 are poisoned, and D5-D10 are
clean.
trigger to the question and achieves the adversarial
goal only when the trigger is shown in the query.
Another Prompt Injection Attack (PIA, Clop and
Teglia, 2024) makes use of the passage that guaran-
tees the retrieval in PoisonedRAG and focuses on
the adversarial goal beyond misinformation.
A recent study proposed a defense mechanism
that generates responses independently and pro-
duces an output based on the majority vote (Xiang
et al., 2024). However, this method initiates its de-
fense at the generator stage, which can impact the
accuracy of the system, especially when multiple
documents are required. GRADA addresses this
issue by focusing on the stage before generation,
specifically the reranking process.
3 GRADA
A defining characteristic of recent poisoning at-
tacks on RAG (Zou et al., 2024) is their focus on
ensuring semantic similarity to the query while
introducing anomalous similarity patterns among
poisoned documents. Adversarial documents are
very similar to the query but less to the legitimate
documents, forming isolated patterns within the
retrieval set, as illustratedFigures 2 and 3. Graph
structures naturally capture these complex inter-
document relationships by representing documents
as nodes and similarities as edges. Leveraging
this intuition, we propose a graph-based reranking
method that utilizes document-document similarity
to enhance retrieval robustness. In Section 3.1, we
detail the graph construction process, followed by a

description of our reranking system in Section 3.2.
3.1 Graph Construction
We construct a weighted, undirected graph G=
(V, E), where each node vi∈Vcorresponds to
a document doci, and each edge eij∈Eis an
undirected edge connecting node viandvj. Each
edge is assigned a weight wij∈R+, which quanti-
fies the similarity between the corresponding doc-
uments, i.e.,sim(vi, vj). The graph is undirected
because document relationships are not inherently
directional; rather, the connectivity structure de-
fines their associations. The edge weight wijcan
be computed using two different approaches:
•Doc-to-Doc Similarity (D2DSIM): The weight
is directly determined by the similarity between
documents.
•Hybrid Relevance Similarity (HRSIM): A
function fthat integrates both document-
document similarity and query-document rele-
vance:
wij=f 
sim(vi, vj),sim(vi, q),sim(vj, q)
The second approach assigns edge weights that
not only reflect direct document-to-document sim-
ilarity but also incorporate each document’s rele-
vance to an external query. This dual consideration
leads to a more nuanced representation of docu-
ment relationships.
To mitigate the influence of adversarial pas-
sages—documents that mimic the query qto gain
higher rankings—we introduce a function f, which
adjusts the similarity score by applying a penalty
based on the document-to-query similarities. First,
we define the combined query relevance for a pair
of documents viandvjas follows:
sim sum=sim(vi, q) +sim(vj, q)
Then, the edge weight wijbetween viandvjis
computed by subtracting a penalty term from their
direct similarity, ensuring that the weight remains
non-negative:
wij= max ( sim(vi, vj)−α·sim sum,0)
Here, αis a penalty coefficient that controls the
influence of query similarity. If sim(vi, vj)< α·
[sim(vi, q) +sim(vi, q)], the edge weight is set to
zero, effectively removing the connection between
viandvj.
Regarding the similarity function, we explore
two popular methods:•BM25: we use BM25 (Robertson and Zaragoza,
2009) to calculate sim(vi, vj). Since BM25 is an
asymmetric metric, we adopt the following ap-
proach to compute the similarity score, ensuring
symmetry in the process:
wij=1
2(BM25 (vi, vj) +BM25 (vj, vi))
•Embedding-based Distance (EBD): we trans-
form the documents xiandxjinto dense vectors
viandvjand compute their cosine distance:
wij=sim(vi, vj) =xi·xj
∥xi∥∥xj∥
3.2 Reranking
Inspired by PageRank (Page et al., 1999), we refine
document rankings through an iterative score prop-
agation process after constructing the graph. This
approach prioritizes well-connected nodes while
mitigating the influence of adversarial documents,
ensuring a more robust and reliable ranking.
Initially, each node viis assigned a score
s∗
i, forming the initial score vector s∗=
[s∗
1, s∗
2, . . . , s∗
n]⊤. The scores are then iteratively
updated at each step tvia:
s(t)
i= (1−d)s∗
j+dX
vj∈N(i)wijP
vk∈N(j)wjks(t−1)
j
(1)
where N(i)represents the set of neighbor nodes
connected by vianddis the damping factor, typi-
cally set to 0.85. The initial score vector s∗is set
by uniform initialization s∗=h
1
|V|,1
|V|, ...,1
|V|i
.
For experiments comparing different initializa-
tion methods, please refer to Appendix C.2.
The framework works as follows: The retriever
identifies Mdocuments most similar to the query,
withnbeing the number of documents originally
intended for retrieval and M≥n. We retrieve
additional documents to maintain consistency in
the number of documents in the non-defended sce-
nario.
By ensuring that poisoned documents do not
form the majority in the retrieved set (with M≥
2n), we prevent adversarial documents, which may
exploit the query for high relevance scores, from
dominating. For example, if the original set of n
documents contains all poisoned ones ( e.g.,n= 5),
adding ≥nbenign documents ensures the major-
ity is non-poisoned. This strategy guarantees that

non-poisoned documents remain a significant por-
tion of the final selection, enhancing the system’s
robustness against adversarial manipulation.
After the algorithm reaches a stationary score
distribution, the top ndocuments are retained,
while the remaining documents are discarded.
These top ndocuments are then provided as the
context of the model.
4 Experiments
This section begins by detailing the experimental
setup, followed by a comparison of our approach
with multiple baseline methods. Finally, we ana-
lyze the effectiveness of our approach across differ-
ent settings.
4.1 Experimental Setup
Attack setup. We conduct experiments on
three widely used english datasets: Natu-
ral Question (Kwiatkowski et al., 2019), MS-
MARCO (Nguyen et al., 2016) and Hot-
potQA (Yang et al., 2018). The victim models
chosen for this study are GPT-3.5-Turbo (version
0125) (Brown et al., 2020), GPT-4o (version 2024-
08-06) (OpenAI et al., 2024), Qwen2.5 (Qwen
et al., 2025) and LLaMA-3 (Grattafiori et al.,
2024). The prompts used to generate answers are
detailed in Appendix A. Contriever (Izacard et al.,
2021), is a dense retriever model used to find rel-
evant documents by calculating similarity scores
between the query and the knowledge base. It was
selected for this study due to its efficiency and abil-
ity to handle large datasets. In this work, we inves-
tigate four distinct attack strategies on RAG. Two
of them are Black-box attacks that have no knowl-
edge about the retriever: PoisonedRAG (Zou et al.,
2024) and PIA (Greshake et al., 2023; Pasquini
et al., 2024; Perez and Ribeiro, 2022). The re-
maining two are white-box attacks, in which the
attacker has access to the victim’s retriever: Poi-
sonedRAG(Hotflip) (Zou et al., 2024) and Phan-
tom (Chaudhari et al., 2024)
Under default settings without any defense, as
in Zou et al. (2024), we retrieve the five most sim-
ilar documents from the knowledge database to
serve as the context for each question. We select
100 close-ended questions from each dataset, yield-
ing 300 questions in total per attack-defense run.
Additionally, this process is repeated using 3 ran-
dom seeds, meaning each attack-defense pair is
evaluated on 900 questions in total.However, in contrast to Zou et al. (2024), where
five poisoned texts are generated and injected into
the knowledge base, To provide a more realistic
assessment of the attack’s effectiveness, we mod-
ify the experiment to inject only a single poisoned
document into the database. The original setup,
which retrieved only poisoned documents, resulted
in a 100% Attack Success Rate (ASR), making it
impractical to evaluate the true impact of the at-
tack. As shown in Figure 3, a similarity cluster
of poisoned documents appears in the top-left cor-
ner. By applying a clustering algorithm, we can
identify and merge redundant information, effec-
tively removing repetitive poisoned entries. This
adjustment ensures that only one poisoned docu-
ment is retrieved, allowing for a more meaningful
evaluation of the attack’s success.
Defense setup. We explore three similarity score
combinations for GRADA: Embedding-based Dis-
tance, BM25, and Hybrid Relevance Similarity
with BM25 as the similarity function.1Here, we
utilize Contriever to encode both documents and
queries, while for BM25, we adopt the imple-
mentation provided by Lù (2024). We compare
GRADA against two reranking models and one de-
fense method: HLATR (Zhang et al., 2022), which
achieved first place in the MS-MARCO Passage
Ranking Leaderboard, BGE-reranker (Xiao et al.,
2023), which achieves a high precision score in
ranking tasks, and Keyword Aggregation (Xiang
et al., 2024), the only existing defense specifically
designed for RAG-based adversarial attacks, as a
baseline.
We evaluate the effectiveness of these defense
methods by integrating them into our two-stage
retrieval system described in Section 3. We ini-
tially retrieve M= 10 documents, which are
then reranked using the aforementioned methods
(except for Keyword Aggregation). The top five
ranked documents are subsequently provided as the
context for the model to answer the query. This
ensures that, regardless of the defense configura-
tion, the model always receives a fixed number of
five context documents to respond to the question.
For Keyword Aggregation, which does not perform
reranking, the model directly generates the output
based on the algorithm’s selection.
Evaluation metrics. In our experiments, we em-
ploy Attack Success Rate (ASR) and Exact Match
1We examine other similarity functions in Section 4.3

(EM) as metrics. ASR is defined as the ratio of suc-
cessful attacks to the total number of attacks con-
ducted. An attack is considered successful if the
intended poisoned answer appears as a substring
within the generated response from the model. This
definition accommodates attack strategies like PIA,
which aim to introduce harmful links into the out-
put of the model, allowing for some tolerance to
semantically equivalent responses. A higher ASR
indicates a more successful attack. This evaluation
methodology follows the approach used in previous
work (Zou et al., 2024).
To assess the question-answering accuracy of the
models, we adopt EM score. EM requires that the
predicted answer of the model matches the ground
truth answer exactly. This strict criterion ensures
that the response of the model is precise and fol-
lows the need for exact wording specified in the
query, as outlined in Appendix A.
4.2 Results and Discussions
Attacking without defense. As shown in Table 1,
including a single poisoned document in the re-
trieval process results in a high ASR score. For
instance, PoisonedRAG achieves an ASR of 50%
across three datasets on both GPT-3.5-Turbo and
Llama3.1-8b-Instruct. PIA achieves at least 69%
ASR on Llama3.1-8b-Instruct and up to 100% ASR
in GPT-3.5-Turbo. These findings emphasize that
even minimal adversarial input can achieve very
high ASR and degrade the model’s accuracy.
Effectiveness of GRADA. The impact of
GRADA on mitigating adversarial attacks is
demonstrated in Tables 1 and 2. As shown in Ta-
ble 1, on the NQ and MS-MARCO datasets us-
ing GPT-3.5-Turbo, the ASR for PIA decreases
from 98.0% and 88.0% to 2.0% and 3.0% by using
D2DSIM-EBD. With D2DSIM-EBD, GRADA is
also effective against PoisonedRAG, effectively re-
ducing the ASRs from 55.7% and 46.5% to 26.1%
and 29.0%. However, the reduction of ASR against
PoisonedRAG is more modest than against the
other attacks. On this attack, D2DSIM-BM25
and HRSIM led to significant improvements com-
pared to D2DSIM-EBD, where D2DSIM-BM25
achieved an extra 13% decrease in ASR to 13.5%
and 16.5%. Beyond that, HRSIM which introduces
penalties for excessive similarity to the query, final-
izes the ASR to 3% and 8.5%.
The defense methods demonstrate consistent
effectiveness across the NQ and MS-MARCOdatasets, achieving ASR reductions of over 30% in
most cases. However, performance on HotpotQA
is less stable, particularly for D2DSIM-EBD and
D2DSIM-BM25, which achieve only around a 10%
reduction in ASR against PoisonedRAG attacks. In
contrast, HRSIM maintains its effectiveness, deliv-
ering ASR reductions exceeding 30%, comparable
to its performance on other datasets. This discrep-
ancy likely stems from HotpotQA’s multi-hop rea-
soning requirements, which pose challenges for
single-document similarity metrics.
In Table 1, HLATR and BGE-reranker exhibit
limited ability to filter poisoned documents, with
ASR remaining largely unchanged compared to sce-
narios without any defense mechanisms. Although
BGE-reranker occasionally outperforms HLATR,
its overall performance remains inferior to GRADA
in handling adversarial cases. This discrepancy
underscores a critical limitation in contemporary
reranking systems, which are primarily optimized
for question relevance but insufficiently equipped
to address adversarial attacks with high question
relevance.
Keyword Aggregation is able to reduce ASR
significantly, especially for attacks like PIA and
Phantom. Keyword Aggregation works by extract-
ing keywords from the answers of each passage
to generate the final response, effectively neutral-
izing attack payloads designed to manipulate or
deny answers, such as producing advertisements.
However, while it reduces ASR effectively, its EM
scores are lower than those of GRADA. For exam-
ple, on Llama3.1-8b-Instruct in Table 1, GRADA’s
EM scores dominate Keyword Aggregation with at
most 21% difference as some critical information
may be lost during keyword extraction. This shows
the ability of GRADA to perform well on normal
answers even after mitigating adversarial contents.
Similar results to those presented in Table 1 can
also be observed in Table 2. Notably, GRADA
combined with HRSIM consistently outperforms
all other approaches, demonstrating that HRSIM
is a strong similarity scoring function compared to
the alternatives used in GRADA.
Table 3 highlights the impact of different de-
fense mechanisms on benign inputs. On GPT-3.5-
Turbo, both HLATR and BGE-reranker demon-
strate strong performance, outperforming GRADA
and enhancing the model’s overall accuracy. these
reranking systems yield at least a 2% improvement
in EM scores, suggesting their effectiveness in mit-
igating noise unrelated to the posed questions.

DefensePoisonedRAG PIA
HotpotQA NQ MS-MARCO HotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-3.5-Turbo
None 59.0±1.4 / 32.3±0.5 55.7±1.2 / 33.3±1.1 46.5±1.5 / 41.0±0.0 100.0±0.0 / 0.0±0.0 98.0±0.0 / 2.0±0.0 88.0±0.0 / 7.7±0.5
HLATR 62.3±0.5 / 30.3±0.5 51.5±0.5 / 35.5±0.5 36.5±1.5 / 52.0±1.0 100.0±0.0 / 0.0±0.0 92.0±0.0 / 4.0±0.0 84.0±0.0 / 9.0±0.0
BGE-reranker 56.6±0.9 / 36.3±1.2 46.5±0.5 / 43.5±0.5 34.0±0.0 / 55.0±0.0 98.0±0.0 / 2.0±0.0 43.0±0.0 / 35.7±0.5 43.0±0.0 / 43.0±0.8
Keyword Aggregation 11.0±2.0 / 62.5±2.5 2.0±0.0 / 54.0±0.0 3.0±0.0 / 60.0±2.0 0.0±0.0 / 59.0±1.0 0.0±0.0 / 48.0±0.0 0.0±0.0 / 57.5±0.5
GRADA (D2DSIM-EBD) 48.6±1.2 / 39.0±0.8 26.1±1.0 / 50.9±1.0 29.0±1.0 / 55.0±1.0 33.0±0.0 / 42.3±0.5 2.0±0.0 / 58.3±0.5 3.0±0.0 / 70.5±0.5
GRADA (D2DSIM-BM25) 45.0±0.8 / 40.0±0.5 13.5±0.7 / 55.0±1.4 16.5±0.5 / 65.5±0.5 42.0±0.0 / 33.0±0.8 12.0±0.0 / 55.3±0.5 2.0±0.0 / 69.7±0.9
GRADA (HRSIM) 10.0±0.0 / 51.0±0.8 3.0±0.6 / 58.0±1.1 8.5±0.5 / 71.5±0.5 27.0±0.0 / 41.7±1.2 2.0±0.0 / 61.7±2.1 1.0±0.0 / 74.3±0.5
Llama3.1-8b-Instruct
None 50.7±0.5 / 37.0±0.0 49.0±0.8 / 33.0±0.8 40.7±0.5 / 40.0±0.0 88.3±0.5 / 3.0±0.0 82.0±0.0 / 8.0±0.0 69.0±0.0 / 14.0±0.0
HLATR 52.3±0.5 / 35.7±0.5 39.0±0.8 / 41.3±0.5 35.7±0.5 / 43.3±0.5 91.3±0.5 / 2.7±0.5 71.7±0.5 / 15.3±0.5 50.0±0.8 / 19.7±0.5
BGE-reranker 51.7±0.5 / 36.0±0.0 42.0±0.8 / 40.7±1.2 33.7±0.5 / 42.0±0.8 79.7±0.5 / 9.7±0.9 30.0±0.0 / 40.3±0.5 19.7±0.9 / 44.7±1.2
Keyword Aggregation 6.7±1.9 / 39.0±0.8 3.0±0.0 / 39.0±0.0 6.7±0.5 / 38.3±1.2 0.0±0.0 / 35.0±0.0 0.0±0.0 / 39.0±0.0 0.0±0.0 / 36.0±0.8
GRADA (D2DSIM-EBD) 42.0±0.0 / 36.7±0.5 24.0±0.0 / 46.7±0.5 31.7±0.5 / 40.0±0.8 30.7±0.5 / 35.3±0.90 1.0±0.0 / 55.3±0.5 2.0±0.0 / 56.0±0.0
GRADA (D2DSIM-BM25) 30.0±0.0 / 39.3±0.5 8.0±0.0 / 52.3±0.5 19.3±0.5 / 49.7±0.9 39.0±0.0 / 28.7±0.5 7.7±0.5 / 48.3±0.9 0.0±0.0 / 55.0±0.0
GRADA (HRSIM) 7.0±0.0 / 44.0±0.8 2.3±0.5 / 55.7±0.5 12.0±0.0 / 52.3±0.5 23.0±0.0 / 36.7±0.5 2.0±0.0 / 55.0±0.8 0.0±0.0 / 59.3±0.5
Table 1: ASR and EM (%) for various defense methods on the black-box attack methods on GPT-3.5-Turbo and
Llama3.1-8b-Instruct. The results of other models can be found in Tables 9 to 13. We highlight the top-2 lowest
ASR results in blue cells.
DefensePoisonedRAG(Hotflip) Phantom
HotpotQA NQ MS-MARCO HotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-3.5-Turbo
None 62.0±0.8 / 29.3±0.5 55.0±0.0 / 31.5±0.5 42.5±0.5 / 47.5±0.5 99.0±0.0 / 1.0±0.0 88.7±0.5 / 5.7±0.9 67.7±1.9 / 25.7±1.7
HLATR 60.7±0.5 / 30.3±0.5 49.6±0.9 / 36.0±0.8 31.3±2.1 / 55.0±2.2 97.3±0.5 / 2.7±0.5 90.7±0.5 / 7.0±0.8 64.7±9.6 / 27.3±8.2
BGE-reranker 56.6±0.5 / 34.3±1.2 43.0±0.8 / 40.7±0.5 27.3±1.2 / 59.7±0.5 94.0±0.0 / 6.0±0.0 70.7±4.7 / 17.3±0.5 57.3±9.4 / 30.7±7.4
Keyword Aggregation 12.0±0.8 / 62.3±2.1 2.0±0.0 / 52.0±4.0 4.0±0.8 / 57.0±2.6 0.0±0.0 / 50.0±0.8 0.0±0.0 / 44.0±0.0 0.0±0.0 / 57.0±1.0
GRADA (D2DSIM-EBD) 44.7±0.9 / 39.3±1.2 14.0±3.5 / 52.7±2.5 10.7±1.2 / 69.0±0.0 60.7±0.5 / 19.7±0.5 14.0±0.0 / 45.3±0.5 13.0±0.0 / 59.0±2.2
GRADA (D2DSIM-BM25) 37.0±0.8 / 44.0±0.0 9.0±0.0 / 59.3±0.5 7.3±0.9 / 70.7±0.9 27.0±0.0 / 33.0±0.8 5.7±0.5 / 50.0±0.8 0.3±0.5 / 66.0±2.2
GRADA (HRSIM) 7.3±0.5 / 52.7±0.9 4.0±0.0 / 58.3±1.2 6.3±0.9 / 72.3±1.2 23.0±0.0 / 37.3±1.2 0.0±0.0 / 48.5±0.5 0.0±0.0 / 70.0±0.5
Llama3.1-8b-Instruct
None 53.0±2.8 / 32.7±1.2 50.0±1.4 / 30.0±2.2 49.0±0.0 / 32.0±1.6 99.7±0.5 / 0.3±0.5 89.3±2.1 / 9.3±1.2 73.0±1.6 / 20.3±1.7
HLATR 53.3±2.9 / 32.7±2.1 43.7±2.1 / 37.7±2.4 36.0±1.4 / 37.7±1.7 96.7±1.2 / 3.0±0.8 92.7±1.2 / 6.0±1.4 72.3±1.2 / 18.0±1.6
BGE-reranker 50.0±3.7 / 34.3±0.5 42.3±0.5 / 36.3±1.2 27.3±1.2 / 59.7±0.5 95.3±1.2 / 3.0±0.8 72.0±1.6 / 21.7±1.7 62.0±0.8 / 26.0±1.6
Keyword Aggregation 12.0±0.8 / 62.3±2.1 2.0±0.0 / 52.0±4.0 4.0±0.8 / 57.0±2.6 0.0±0.0 / 32.0±0.0 0.0±0.0 / 36.0±0.0 0.0±0.0 / 39.7±0.9
GRADA (D2DSIM-EBD) 39.7±2.5 / 35.7±2.6 13.0±0.0 / 50.7±2.1 14.7±1.2 / 52.3±1.9 57.7±2.6 / 22.7±2.1 10.7±1.9 / 48.7±1.2 11.3±1.2 / 51.3±1.2
GRADA (D2DSIM-BM25) 32.0±0.8 / 38.0±0.0 8.7±0.9 / 52.0±0.8 13.3±0.9 / 53.0±0.8 26.7±0.5 / 37.0±2.2 4.3±0.5 / 53.7±1.2 1.0±0.0 / 56.0±0.0
GRADA (HRSIM) 8.7±1.7 / 44.7±1.2 2.0±0.8 / 53.3±2.6 6.3±0.9 / 72.3±1.2 10.3±2.1 / 41.3±1.9 0.3±0.5 / 53.7±0.9 0.0±0.0 / 60.3±1.7
Table 2: ASR and EM (%) for various defense methods on White-box attacks.
GRADA with D2DSIM-EBD effectively pre-
serves model performance on benign inputs across
all datasets, with EM score deviations remaining
within 4%. Notably, the use of D2DSIM-BM25
leads to an 6% improvement in EM scores on
NQ, matching the performance of BGE-reranker,
which achieves the highest EM overall. However,
on HotpotQA, HRSIM resulted in an 14% reduc-
tion in EM scores when handling benign inputs.
While this trade-off is significant, it corresponds
to HRSIM’s remarkable ASR reduction. Striking
a balance between retrieval quality and defense
robustness remains a crucial challenge for future
research.
Keyword Aggregation has a much lower perfor-
mance also in EM scores on benign input compared
to GRADA. For example, in MS-MARCO, it re-
sults in 40% compared to 57% on Llama3.1-8b-
0 1 2 3 4 5 6 7 8 9
Position of the T ext020406080Number of T extNQ(HRSIM) Poisoned T ext Distribution
Before Rerank
After RerankFigure 4: Distribution of poisoned document positions
after applying GRADA (HRSIM) in the NQ dataset.
Documents positioned below rank 5 are effectively
mitigated by the ranking algorithm. Other results are
showed in Figure 11 and Tables 6 to 8

Defense HotpotQA NQ MS-MARCO
GPT-3.5-Turbo
No-RAG 16.3±1.7 23.7±1.3 11.7±0.5
None 64.3±0.5 58.6±1.2 76.0±0.0
HLATR 70.0±0.8 62.0±0.8 77.7±0.5
BGE-reranker 68.0±1.4 64.7±1.2 78.3±0.5
Keyword Aggregation 68.3±0.5 48.0±0.0 59.0±0.0
GRADA (D2DSIM-EBD) 64.0±0.8 61.0±0.8 74.3±0.5
GRADA (D2DSIM-BM25) 57.3±0.5 64.7±0.5 75.0±1.6
GRADA (HRSIM) 50.0±0.5 62.0±0.0 75.3±0.5
Llama3.1-8b-Instruct
No-RAG 4.3±0.5 3.0±0.0 3.7±1.2
None 56.7±0.5 50.0±0.0 55.0±0.0
HLATR 56.0±0.8 51.0±0.0 56.3±0.5
BGE-reranker 58.0±0.8 54.0±0.8 59.3±0.9
Keyword Aggregation 34.0±0.0 39.0±0.0 36.0±0.8
GRADA (D2DSIM-EBD) 52.0±1.4 54.7±0.5 58.3±0.5
GRADA (D2DSIM-BM25) 47.0±0.8 52.7±0.5 54.3±0.9
GRADA (HRSIM) 43.3±0.9 54.7±0.5 57.0±0.8
Table 3: EM scores of defense methods when presented
with benign inputs.
Instruct and 59% compared to 75.3% on GPT-3.5-
Turbo. Indeed showing the cost of discarding valu-
able information when facing benign documents.
Using GRADA, we demonstrate that it is pos-
sible to defend against the chosen attacks without
compromising the model’s overall performance on
EM. While reranking methods such as HLATR
and BGE-reranker show promise in reducing noise,
their limited effectiveness in countering adversarial
attack noise highlights a critical gap in existing de-
fenses. Similarly, Keyword Aggregation presents
a valuable strategy for mitigating attack payloads
but comes with trade-offs in EM scores.
Why GRADA works. For the attack to be ef-
fective, the attackers must ensure that the retriever
selects the poisoned documents. To achieve this,
they primarily focus on making these documents
resemble the queries, as most retrieval models pri-
oritize query-document similarity when selecting
relevant results. Additionally, poisoned documents
typically exhibit only weak similarity to other doc-
uments in the corpus. This characteristic makes
them less susceptible to detection by defense mech-
anisms that compare retrieved documents against
one another.
GRADA exploits this by constructing a docu-
ment similarity graph where documents effectively
"vote" for other documents they strongly resem-
ble semantically. Benign documents, which share
common relevant content, naturally have strong
mutual similarities (e.g., averaging 0.82) and thus
reinforce each other. In contrast, adversarial doc-
uments—designed to mislead—tend to be more
isolated, receiving fewer "votes" due to their sig-
BM25 SBERTPercentage of Poisoned Documents
PoisonedRAG
PIA
EBDDifferent sim score functions on HRSIM
BGE-reranker020406080Figure 5: HRSIM performance with different similarity
functions selection on MSMARCO dataset. The figure
illustrates the proportion of test instances in which poi-
soned documents remain among the top five retrieved
results.
nificantly lower similarities (average 0.35) with
genuine documents. As a result, GRADA promotes
densely interconnected benign documents while re-
ducing the influence of sparse adversarial content.
A running example is also shown in Figure 14
4.3 Additional Studies
Ranking distribution. We have demonstrated
the effectiveness of our approach in enhancing de-
fense performance. To gain a deeper understanding
of its impact, we further analyze how our method
systematically lowers the ranking of poisoned docu-
ments. As illustrated in Figure 4, the position distri-
bution of poisoned documents within the retrieval
set shifts significantly after applying GRADA with
D2DSIM-BM25. Notably, over 70% of poisoned
documents are relegated beyond the top five posi-
tions, substantially reducing their influence. These
findings confirm that GRADA is both robust and
effective in mitigating adversarial attacks.
Selections of HRSIM. Thus far, our focus has
primarily been on utilizing BM25 for HRSIM. In
this section, we explore other similarity functions
for HRSIM. As shown in Figure 5, we extend our
analysis by incorporating SBERT (Reimers and
Gurevych, 2019), alongside the three previously
discussed methods, to better capture document-
to-document similarity. Our results indicate that
both EBD and SBERT exhibit strong overall per-
formance against PIA and PoisonedRAG attacks.
In contrast, BGE-Reranker struggles to effectively
filter out poisoned documents, likely due to its
primary training objective of computing query-
to-document similarities rather than document-to-
document relationships. HRSIM, when combined
with BM25, effectively minimize the presence of

Alpha ValueResults
Impact of Alpha Value
ASR
EM
Number of Poisoned docs
1.0 0.8 0.6 0.4 0.2 0.00.10.20.30.40.50.6Figure 6: Impact of the αvalue as it increases with three
metrics (ASR, number of poisoned documents, and EM)
on NQ dataset with GPT-3.5-Turbo.
ASR EM Poisoned Doc
CategoriesPercentageImpact of M values
Benign
M=n
M2n
020406080100
Figure 7: Impact of the Mvalue as it changes with three
metrics (ASR, number of poisoned documents, and EM)
on MSMARCO dataset with GPT-3.5-Turbo.
poisoned documents, reducing them to just 14 out
of 100 test instances. This outcome underscores
its remarkable effectiveness in filtering malicious
content.
Impact of αandM.As shown in Figure 6, the
number of poisoned documents in the context de-
creases as αincreases, reaching a minimum at
α= 0.3before starting to rise again after α= 0.8.
The ASR follows a similar trend to the number of
poisoned documents after α= 0.3. Conversely,
the EM score exhibits a minimum at α= 0.7.
We selected α= 0.4because it strikes a balance,
avoiding excessive penalization for query similar-
ity, which could otherwise result in fewer query-
related documents. When α= 0.4, all three met-
rics (ASR, number of poisoned documents, and
EM) are within an acceptable range, approaching
the optimal performance values for α.
Figure 7 illustrates the effect of selecting M=n.
It shows that, regardless of how documents are re-
ranked, poisoned documents can still remain within
the context provided to the model. However, this
approach results in a 17% decrease in ASR and a9% increase in EM, indicating that simply adjust-
ing document positions can significantly impact
model performance. This aligns with our observa-
tions in Table 5, and the specific positions of the
documents are detailed in Figure 4. By including
additional documents for reranking and then re-
trieving only the top nresults, the ASR is further
reduced from 21% to 10%, with only 14% of poi-
soned documents remaining in the context provided
to the model. This demonstrates the importance
of including extra documents during reranking to
remove poisoned content and achieve better overall
performance effectively.
Computational Complexities. The overall com-
plexity of GRADA consists of two main compo-
nents:
•Similarity matrix construction: O(N2),
where N is the number of retrieved documents.
This step can incur additional costs depend-
ing on the chosen similarity function. For
example, using D2DSIM-EBD (embedding-
based document similarity), the complexity
becomes O(N2·d), where dis the embedding
dimension. BM25-based similarity: the com-
plexity is O(N2·L), where L is the average
document length. This is efficient due to the
sparsity of token overlaps and inverted index
optimizations. Here, since we are reranking
the documents after the retrieval step. The
retrieved documents set is usually constrained
with limited amounts of data, making this a
viable solution.
•Graph-based reranking (e.g., PageRank):
O(n+m), where n is the number of nodes
(documents) and m is the number of edges in
the constructed similarity graph.
The only defense Keyword Aggregation requires
querying the language model N times—once per
document—to collect individual answers before
aggregating: O(N∗CLM)(where CLMrefers to
the language model’s cost). This incurs signifi-
cantly higher costs in terms of API calls and model
generation time, especially with large models.
GRADA, by comparison, does not require any
model calls. The only required model call is after
GRADA to query the final answer, making it more
efficient and scalable for large-scale or production
RAG deployments.

Defense Total Time (s) Processing Time (s) Defense-Only Time (s) Defense-Only Processing (s)
Keyword Aggregation 12.61 11.11 11.59 9.21
GRADA (D2DSIM-EBD) 1.56 1.12 0.62 0.62
GRADA (D2DSIM-BM25) 0.97 0.52 0.02 0.02
GRADA (HRSIM) 1.05 0.61 0.05 0.05
Table 4: Runtime Comparison (on GPT-3.5-Turbo, average per query): Total Time (s) and Processing Time (s)
represent the complete runtime for answering one question, including retrieval, defense method, and LLM response
generation. In contrast, Defense-Only Time (s) measures exclusively the runtime of the defense methods themselves.
Total Time is recorded using Python’s time.time() function, whereas Processing Time is measured with Python’s
time.process_time() function.
5 Conclusion
The study examines the robustness challenges faced
by RAG systems. We identify a critical vulnerabil-
ity in current adversarial attacks, which focus on
increasing semantic similarity to the query with-
out accounting for the relationships between the
retrieved documents. Our proposed graph-based
filtering framework, GRADA, enhances the robust-
ness of RAG systems by leveraging document sim-
ilarities and effectively mitigating adversarial im-
pacts through information flow. Experimental re-
sults on datasets such as MS-MARCO and NQ,
demonstrate at least 30% reductions in ASR across
various adversarial strategies. Overall, this work
presents a promising direction for developing more
secure and reliable RAG systems.
Limitations
Despite its effectiveness, our approach has limita-
tions. First, it struggles with multi-hop reasoning
tasks, facing attacks like PIA and Phantom. As the
number of poisoned documents increases, system
robustness deteriorates. Second, our method as-
sumes poisoned documents are a minority. When
they form the majority, their effectiveness declines,
and future work should explore adaptive retrieval
strategies to counter adversarial dominance.
Ethics Statement
Our study focuses on improving the robustness of
RAG systems, thereby enhancing their reliability
and minimizing harmful manipulations. We evalu-
ated our proposed method, GRADA, using publicly
available datasets as detailed in Appendix F. We
do not engage in harmful data practices.
Acknowledgement
Aryo Pradipta Gema was supported by the
United Kingdom Research and Innovation (grantEP/S02431X/1), UKRI Centre for Doctoral Train-
ing in Biomedical AI at the University of Edin-
burgh, School of Informatics. Giwon Hong was
supported by the ILCC PhD program (School of
Informatics Funding Package) at the University of
Edinburgh, School of Informatics. Xuanli He was
funded by an industry grant from Cisco. Pasquale
Minervini was partially funded by ELIAI, an in-
dustry grant from Cisco, and a donation from Ac-
centure LLP. This work was supported by the Ed-
inburgh International Data Facility (EIDF) and the
Data-Driven Innovation Programme at the Univer-
sity of Edinburgh.
References
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, and 12 others. 2020. Language
models are few-shot learners. In Proceedings of the
34th International Conference on Neural Information
Processing Systems , NIPS ’20, Red Hook, NY , USA.
Curran Associates Inc.
Carlos Castillo and Brian D. Davison. 2011. Adversar-
ial Web Search . Now Foundations and Trends.
Harsh Chaudhari, Giorgio Severi, John Abascal,
Matthew Jagielski, Christopher A. Choquette-Choo,
Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea.
2024. Phantom: General trigger attacks on re-
trieval augmented language generation. Preprint ,
arXiv:2405.20485.
Cody Clop and Yannick Teglia. 2024. Backdoored
retrievers for prompt injection attacks on retrieval
augmented generation of large language models.
Preprint , arXiv:2410.14479.
Clémentine Fourrier, Nathan Habib, Alina Lozovskaya,
Konrad Szafer, and Thomas Wolf. 2024. Open
llm leaderboard v2. https://huggingface.

co/spaces/open-llm-leaderboard/open_llm_
leaderboard .
David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda.
2003. English gigaword. Linguistic Data Consor-
tium, Philadelphia , 4(1):34.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models. Preprint , arXiv:2407.21783.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz.
2023. Not what you’ve signed up for: Compromis-
ing real-world llm-integrated applications with indi-
rect prompt injection. In Proceedings of the 16th
ACM Workshop on Artificial Intelligence and Secu-
rity, AISec ’23, page 79–90, New York, NY , USA.
Association for Computing Machinery.
Zoltan Gyongyi and Hector Garcia-Molina. 2005. Web
spam taxonomy. In First International Workshop on
Adversarial Information Retrieval on the Web (AIR-
Web 2005) .
Kristian Hammond. 2024. The risk of google’s shift
from search engine to answer machine. CENTER
FOR ADVANCING SAFETY OF MACHINE INTEL-
LIGENCE .
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense infor-
mation retrieval with contrastive learning.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.Xing Han Lù. 2024. Bm25s: Orders of magnitude faster
lexical search via eager sparse scoring. Preprint ,
arXiv:2407.03618.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng
Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016. MS MARCO: A human gener-
ated machine reading comprehension dataset. CoRR ,
abs/1611.09268.
Alexandros Ntoulas, Marc Najork, Mark Manasse, and
Dennis Fetterly. 2006. Detecting spam web pages
through content analysis. In Proceedings of the 15th
International Conference on World Wide Web , WWW
’06, page 83–92, New York, NY , USA. Association
for Computing Machinery.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Lawrence Page, Sergey Brin, Rajeev Motwani, and
Terry Winograd. 1999. The pagerank citation rank-
ing: Bringing order to the web. Technical Report
1999-66, Stanford InfoLab. Previous number = SIDL-
WP-1999-0120.
Dario Pasquini, Martin Strohmeier, and Carmela Tron-
coso. 2024. Neural exec: Learning (and learning
from) execution triggers for prompt injection attacks.
InProceedings of the 2024 Workshop on Artificial
Intelligence and Security , AISec ’24, page 89–100,
New York, NY , USA. Association for Computing
Machinery.
Fábio Perez and Ian Ribeiro. 2022. Ignore previous
prompt: Attack techniques for language models.
Preprint , arXiv:2211.09527.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report. Preprint ,
arXiv:2412.15115.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP) , pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Found. Trends Inf. Retr. , 3(4):333–389.

Alexander M. Rush, Sumit Chopra, and Jason Weston.
2015. A neural attention model for abstractive sen-
tence summarization. Proceedings of the 2015 Con-
ference on Empirical Methods in Natural Language
Processing .
Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou,
Yixin Mao, and Chien-Sheng Wu. 2024. Search
engines in an ai era: The false promise of fac-
tual and verifiable source-cited responses. Preprint ,
arXiv:2410.22349.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wag-
ner, Danqi Chen, and Prateek Mittal. 2024. Certifi-
ably robust rag against retrieval corruption. Preprint ,
arXiv:2405.15556.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding. Preprint ,
arXiv:2309.07597.
Nursel Yalçın and Utku Köse. 2024. The future of seo
is answer engine optimization (aeo). Forbes .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. In Conference on Empirical Methods in Natural
Language Processing (EMNLP) .
Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. 2021.
Pretrained transformers for text ranking: BERT and
beyond. In Proceedings of the 2021 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies: Tutorials , pages 1–4, Online. Association
for Computational Linguistics.
Md Shahir Zaoad, Niamat Zawad, Priyanka Ranade,
Richard Krogman, Latifur Khan, and James Holt.
2025. Graph-based re-ranking: Emerging tech-
niques, limitations, and opportunities. Preprint ,
arXiv:2503.14802.
Yanzhao Zhang, Dingkun Long, Guangwei Xu, and
Pengjun Xie. 2022. Hlatr: Enhance multi-stage text
retrieval with hybrid list aware transformer reranking.
ArXiv , abs/2205.10569.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 13764–13775, Singa-
pore. Association for Computational Linguistics.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. Poisonedrag: Knowledge poisoning at-
tacks to retrieval-augmented generation of large lan-
guage models. Preprint , arXiv:2402.07867.

Prompts given to LLMs
Prompt 1: You are a helpful assistant, be-
low is a query from a user and some relevant
contexts. Answer the question given the
information in those contexts / keywords.
Your answer should be short ,concise and
without explanations ,just answer with one
or two words like yes or no. If you cannot
find the answer to the question, just say "I
don’t know". Contexts / Keywords: [con-
text / keywords] Query: [question] Answer:
Prompt 2: You are a helpful assistant, be-
low is a query from a user and some relevant
contexts. Answer the question given the in-
formation in those contexts. If you cannot
find the answer to the question, just say "I
don’t know". Contexts: [context] Query:
[question] Answer:
Prompt 3: You are a helpful assistant, An-
swer the query with no more than ten words.
If you do not know the answer confidently,
just say "I don’t know". Query: [question]
Answer:
Figure 8: Example of prompts given to LLM, Prompt 1
is used in all of the defense methods to generate the final
output. Prompt 2 is only used in the phase to generate
keywords and Prompt 3 is used for No-RAG from the
paper (Xiang et al., 2024).
A Prompts to LLM
Figure 8 shows the two prompts we use to query
the LLMs.
B Deconstructing PoisonedRAG
PoisonedRAG (Zou et al., 2024) is an adversarial
attack on RAG systems that operates in two stages.
The first part enhances the semantic similarity of
the adversarial passage to the query, increasing the
likelihood of it being retrieved. The second part
introduces adversarial content to mislead the model
into generating a specific incorrect response.
While the approach used to achieve the first part
of the attack is effective, it is also simple. Specif-
ically, the adversarial passage is constructed by
prepending the query into the poisonous passage.
Despite its simplicity, PoisonedRAG degrades the
accuracy of the LLMs significantly. As shown in
Table 5 (first row), the attack achieves an ASR ofAttack Method HotpotQA NQ MS-MARCO Average
Normal retrieved 59.0 56.0 48.0 54.3
w/o question 66.0 61.0 51.0 59.3
Poisoned in the middle 59.0 54.0 37.0 50.0
w/o question 63.0 51.0 34.0 49.3
Table 5: PoisonedRAG Attack Success Rate (%) where
the retrieval part is removed, and the poisoned docu-
ments are placed in the middle.
54.3% on average across three datasets with just
one adversarial passage retrieved as the most simi-
lar to the query.
Our analysis reveals that the prepended query in
the adversarial passage does not significantly affect
the ASR. As shown in Table 5 (second row), remov-
ing the prepended query leads to an increase in the
ASR. This shows that the query was prepended only
to ensure that the retriever retrieves the adversarial
document, but not affecting the accuracy signifi-
cantly. Furthermore, Table 5 (third and fourth row)
shows that the position of the poisoned document
within the retrieved documents set influences the
ASR significantly, with a decrease in average ASR
of 10%. This phenomenon is similar to the lost-
in-the-middle effect (Liu et al., 2024), where the
position of the document impacts its effectiveness
in influencing the output of the reader model.
Due to its straightforward approach of prepend-
ing the query to the adversarial documents, Poi-
sonedRAG attacks can be easily identified. As
demonstrated in Figure 3 and Figure 9, the attacks
injected into the database often exhibit consider-
able similarity to one another. By focusing on the
similarities between the documents in the retrieved
set, we can filter out adversarial passages and de-
crease the ASR.
C Ablation Study
C.1 Number of poisoned documents increase
As shown in Figure 10a, the effectiveness of
GRADA reduces as the proportion of poisoned
documents increases. When using D2DSIM-EBD,
the ASR achieved by GRADA approaches that of
an undefended system. However, HRSIM remains
effective, achieving a 27% reduction in ASR even
when half of the retrieved documents are adversar-
ial. This is further supported by Figure 10b, which
shows that 38% of poisoned documents are still
successfully filtered.

PoisonedRAG Example
Question: "When did the apple iphone se come out?"
Benign documents 1: "March 21, 2017 iPhone SE shipped with iOS 9.3, supporting iPhone 6
and 6S-exclusive features such as Apple Pay, Live Photos, Retina Flash and always-on Siri voice
activation.[6][10][16]"
Benign documents 2: "On launch, it was released in models with either 16 or 64 GB of internal
storage; on March 21, 2017, Apple announced that these models would be replaced by new
improved models at the same launch price points, releasing on March 24, 2017.[11][12]"
Poisoned Document 1: "When did the apple iphone se come out. The Apple archive, housing
important launch events, mentions a significant one on April 15, 2015. On this day, the compact,
budget-friendly iPhone SE hit the market, impressing many gadget enthusiasts."
Poisoned Document 2: "When did the apple iphone se come out. The iPhone series has released
several models. Notable amongst them is the iPhone SE, introduced to the market on April 15,
2015, attracting considerable consumer attention."
Figure 9: Example of PoisonedRAG attacks. Poisoned documents injected into the database are all very similar to
each other and focus solely on ensuring similarity to the query, the similarities among the retrieved documents are
never considered.
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Number of poisoned documents in retrieved M documents020406080ASR
Effect of GRADA as number of poisoned documents increase
None
D2DSIM-EBD
D2DSIM-BM25
HRSIM
(a) ASR of GRADA as poisoned documents increase.
Number of poisoned documents in each retrieved M documentsT otal Number of DocsEffect of GRADA as number of poisoned documents increase
None
D2DSIM-EBD
D2DSIM-BM25
HRSIM
5 4 3 2 10100200300400500 (b) Total poisoned documents remain after filtering.
Figure 10: Impact of increasing poisoned documents on GRADA’s performance in NQ dataset (GPT-3.5-Turbo,
M= 10 ).
C.2 Different initial score vector
Different initial score vectors can have a sig-
nificant impact on the final distribution of
documents in certain cases. For instance,
we experimented with initializing the score
vector with query-document similarity s∗=h
sim (q,v0)Pn
j=0sim (q,vj),sim (q,v1)Pn
j=0sim (q,vj), ...,sim (q,vn)Pn
j=0sim (q,vj)i
.
As shown in Figure 12a, using a query-document
initialization results in more documents being
positioned between rank 5 and 8, rather than lower.
We hypothesize that this is because adversarial
documents may receive disproportionately high
initial scores compared to benign documents.
Such an imbalance gives adversarial documents a
substantial advantage, particularly when the edge
weights between documents are relatively small.In these scenarios, the graph-based reranking
process may struggle to compensate for this
initial disparity, as illustrated in Figure 14. From
the analysis in Figure 12b, we observe that this
phenomenon is more prevalent in datasets like
HotpotQA.
D Different initial score vector
demonstration
Figure 13 shows the documents we used in Fig-
ure 14.
E Computational Resources
The cost of a single defense run on GPT-3.5-Turbo
is $0.50, identical to a standard query since the
method does not introduce additional API calls.

0 1 2 3 4 5 6 7 8 9
Position of the T ext02468Number of T ext
Before Rerank
After RerankNQ(EBD) Ground Truth Distribution(a) D2DSIM-EBD
0 1 2 3 4 5 6 7 8 9
Position of the T ext02468Number of T extNQ(BM25) Ground Truth Distribution
Before Rerank
After Rerank (b) D2DSIM-BM25
0 1 2 3 4 5 6 7 8 9
Position of the T ext02468Number of T ext
Before Rerank
After RerankNQ(HRSIM) Ground Truth Distribution (c) HRSIM
Figure 11: Distribution of Ground Truth document positions after applying GRADA in the NQ dataset with different
ranking methods.
0 1 2 3 4 5 6 7 8 9
Position of the T ext020406080Number of T extPoisoned T ext Distribution with different initialization
Before Rerank
q-doc initialization
Uniform initialization
(a) Distribution of Poisoned document positions after apply-
ing GRADA (HRSIM) with different initialization in the NQ
dataset.
NQ HotpotQA MS-MARCO020406080100120Number of Poisoned DocumentsPoisoned T ext Distribution with different initialization
Before Rerank
q-doc initialization
Uniform initialization(b) Total number of poisoned documents after applying
GRADA (HRSIM) with different initialization in the NQ
dataset.
Figure 12: Impact of different initialization score vectors on GRADA’s performance ( M= 10 ).
Experiments for LLaMA-3 and Qwen2.5 were con-
ducted on a single NVIDIA A100 80GB GPU, with
each defense run taking one hour to complete.
F License and Distribution Terms
The dataset used in our experiments is publicly
available under Creative Commons Attribution 4.0
International (MS-MARCO) and Apache License
2.0 (NQ, HotpotQA). The code used in our exper-
iments is publicly available under MIT License
(BM25s, PoisonedRAG) https://github.com/
agrzheng/GRADA

Initial Score Example in Figure 14
Question: "Are Random House Tower and 888 7th Avenue both used for real estate?"
Documents 1: "The former Bertelsmann Building, now known as 1540 Broadway, is a 44-story,
733 foot (223 m) office tower in Times Square in Manhattan..."
Documents 2: "The Random House Tower, also known as the Park Imperial Apartments, is a
52-story mixed-use tower in New York City, United States, that is..."
Documents 3: "888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown
Manhattan which was completed in 1969 and has 46 floors. Emery Roth & Sons designed..."
Documents 4: "What do the estates of film stars Vincent Price and Glenn Ford have in common?
And what do each of these estates have in common with valuables owned by Laugh-In’s Arte..."
Documents 5: "750 Seventh Avenue is a 615 ft (187m) tall Class-A office skyscraper in New York
City. It was completed in 1989 in the postmodern style and has 36 floors..."
Documents 6: "The Fisk Towers is a front for the Kingpin (Wilson Fisk)’s public ventures as well
as a base of operations for his criminal activities, until..."
Document 0: "Are Random House Tower and 888 7th Avenue both used for real estate?.Random
House Tower is occupied by a publishing company, not devoted to real estate. 888 7th Avenue is
primarily used for law firms, again not real estate operations."
Figure 13: Document examples used to generate Figure 14 to demonstrate different initial score vector and their
results when the adversarial documents receive significantly higher initial scores compared to benign documents.
Red Documents indicates the poisoned document.
Defense Method PoisonedRAG PoisonedRAG(Hotflip) PIA Phantom
No Defense 99.0 99.0 96.0 94.0
HLATR 100.0 100.0 93.0 89.0
BGE-reranker 100.0 98.0 47.0 58.0
GRADA (D2DSIM-EBD) 55.0 20.0 6.0 5.0
GRADA (D2DSIM-BM25) 25.0 16.0 6.0 4.0
GRADA (HRSIM) 13.0 8.0 7.0 2.0
Table 6: The percentage of poisoned documents in the given context to LLM before and after different defense
methods on the NQ dataset. A lower value is better. Method Keyword not included as it does not conduct reranking.
Defense Method PoisonedRAG PoisonedRAG(Hotflip) PIA Phantom
No Defense 98.0 99.0 89.0 65.0
HLATR 98.0 96.0 85.0 70.0
BGE-reranker 98.0 98.0 48.0 53.0
GRADA (D2DSIM-EBD) 69.0 22.0 10.0 10.0
GRADA (D2DSIM-BM25) 34.0 15.0 2.0 2.0
GRADA (HRSIM) 19.0 8.0 1.0 2.0
Table 7: The percentage of poisoned documents in the given context to LLM before and after different defense
methods on the MS-MARCO dataset. A lower value is better. Method Keyword not included as it does not conduct
reranking.

Uniform Initialization
0.10.20.030.05
0.030.11
0.080.210.030.16
doc 1
0.16
doc 60.16
doc 50.16
doc 30.16
doc 2
0.16
doc 40.16
0.16
doc 0Uniform Initialization
0.10.20.030.05
0.030.11
0.080.210.030.26
doc 1
0.12
doc 60.21
doc 50.24
doc 30.12
doc 2
0.03
doc 40.16
0.03
doc 0Query-doc Initialization
0.10.20.030.05
0.030.11
0.080.210.030.08
doc 1
0.05
doc 60.08
doc 50.08
doc 30.18
doc 2
0.0
doc 40.16
0.54
doc 0Query-doc Initialization
0.10.20.030.05
0.030.11
0.080.210.030.23
doc 1
0.10
doc 60.18
doc 50.24
doc 30.13
doc 2
0.0
doc 40.16
0.15
doc 0Figure 14: A demonstration on different initial score vector and their results when the adversarial documents receive
significantly higher initial scores compared to benign documents. This is an example from the HotpotQA dataset
with the question:" Are Random House Tower and 888 7th Avenue both used for real estate?". The top 4 ranked
documents are listed with bold final values.
Defense Method PoisonedRAG PoisonedRAG(Hotflip) PIA Phantom
No Defense 100.0 100.0 100.0 100.0
HLATR 100.0 100.0 100.0 99.0
BGE-reranker 98.0 100.0 98.0 98.0
GRADA (D2DSIM-EBD) 84.0 66.0 52.0 49.0
GRADA (D2DSIM-BM25) 64.0 53.0 35.0 32.0
GRADA (HRSIM) 19.0 18.0 26.0 20.0
Table 8: The percentage of poisoned documents in the given context to LLM before and after different defense
methods on the HotpotQA dataset. A lower value is better. Method Keyword not included as it does not conduct
reranking.

Model Defense HotpotQA NQ MS-MARCO
GPT-4oNo-RAG 26.0±0.0 20.0±0.0 14.0±0.0
None 60.7±1.2 58.7±0.5 65.0±0.0
HLATR 63.3±2.6 61.3±0.5 67.3±0.5
BGE-reranker 62.7±1.2 66.0±0.8 70.3±1.9
Keyword Aggregation 61.7±0.9 47.0±0.0 47.0±0.0
GRADA (D2DSIM-EBD) 57.3±0.9 55.0±0.8 62.0±0.0
GRADA (D2DSIM-BM25) 56.3±0.9 58.7±0.5 66.0±0.0
GRADA (HRSIM) 51.0±1.4 62.0±0.0 65.0±0.0
Llama3.1-70b-InstructNo-RAG 26.0±1.6 21.3±1.2 15.3±0.5
None 60.0±0.8 66.7±0.5 53.7±0.5
HLATR 62.7±0.5 60.7±0.5 53.0±0.8
BGE-reranker 59.0±0.0 66.0±0.0 55.7±0.5
Keyword Aggregation 24.3±1.20 22.3±0.5 15.3±0.5
GRADA (D2DSIM-EBD) 47.7±0.5 55.7±0.5 46.3±0.5
GRADA (D2DSIM-BM25) 39.0±0.8 58.3±0.5 52.7±0.5
GRADA (HRSIM) 32.0±0.8 54.7±0.9 53.0±0.8
Qwen2.5-7b-InstructNo-RAG 6.0±0.0 11.0±0.0 4.7±0.5
None 46.0±0.0 50.7±1.2 51.0±0.0
HLATR 47.3±0.5 48.0±0.0 44.0±1.4
BGE-reranker 44.7±0.5 49.0±0.0 47.3±0.5
Keyword Aggregation 13.0±0.0 16.0±0.0 23.0±0.8
GRADA (D2DSIM-EBD) 40.7±0.5 45.7±0.5 44.0±0.8
GRADA (D2DSIM-BM25) 37.7±0.5 48.3±0.5 46.0±0.8
GRADA (HRSIM) 30.0±0.0 44.7±1.2 48.7±0.9
Table 9: EM scores of defense methods on GPT-4o, Llama3.1-70b-Instruct and Qwen2.5-7b-Instruct when presented
with benign inputs.

Model DefenseHotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-4oNone 42.0±0.0 / 40.0±0.8 29.3±0.9 / 38.0±1.4 24.0±0.0 / 46.0±0.0
HLATR 38.0±0.8 / 44.7±2.0 27.0±0.8 / 48.3±1.2 21.0±0.0 / 53.0±0.0
BGE-reranker 37.3±1.2 / 45.7±1.7 24.7±0.5 / 54.7±0.5 20.0±0.0 / 54.0±0.0
Keyword Aggregation 6.7±0.5 / 58.3±1.9 1.0±0.0 / 46.0±0.0 5.7±0.5 / 45.7±0.5
GRADA (D2DSIM-EBD) 37.0±0.0 / 42.0±0.8 9.7±0.5 / 46.3±0.9 19.0±0.0 / 51.0±0.0
GRADA (D2DSIM-BM25) 23.7±0.5 / 43.3±0.9 5.0±0.0 / 60.0±0.0 10.0±0.0 / 64.0±0.0
GRADA (HRSIM) 5.0±0.0 / 50.0±1.4 1.0±0.0 / 64.0±1.4 4.0±0.0 / 67.0±0.0
Llama3.1-70b-InstructNone 57.7±0.9 / 37.3±0.9 56.7±0.5 / 29.7±0.5 54.3±1.2 / 28.3±0.9
HLATR 53.0±0.8 / 43.3±0.5 49.0±0.0 / 39.0±0.0 40.7±1.2 / 37.0±0.8
BGE-reranker 53.3±0.5 / 41.7±0.5 49.3±0.5 / 38.0±0.0 37.0±0.8 / 37.0±0.8
Keyword Aggregation 4.7±0.5 / 26.0±0.8 3.0±0.0 / 22.3±0.5 3.0±0.0 / 58.0±2.2
GRADA (D2DSIM-EBD) 45.3±0.5 / 37.3±0.5 26.0±0.0 / 44.0±0.0 34.3±0.5 / 39.3±0.5
GRADA (D2DSIM-BM25) 36.0±0.0 / 37.7±0.5 11.0±0.0 / 56.3±0.5 15.0±0.0 / 50.7±0.5
GRADA (HRSIM) 8.3±0.5 / 37.7±0.5 2.7±0.5 / 53.0±0.0 9.0±0.0 / 54.0±0.0
Qwen2.5-7b-InstructNone 62.0±0.0 / 24.0±0.0 50.3±0.5 / 26.7±0.5 49.0±0.0 / 28.7±0.5
HLATR 60.0±0.0 / 28.7±0.5 42.7±0.5 / 31.3±0.5 41.0±0.0 / 28.0±0.8
BGE-reranker 60.0±0.0 / 30.7±0.5 47.7±0.5 / 29.3±0.5 42.0±0.8 / 29.3±0.5
Keyword Aggregation 4.7±0.5 / 6.0±0.0 3.0±0.0 / 11.0±0.0 9.3±0.9 / 25.0±0.8
GRADA (D2DSIM-EBD) 57.0±0.0 / 24.3±0.5 24.3±0.5 / 35.3±1.2 37.3±0.5 / 31.3±0.5
GRADA (D2DSIM-BM25) 42.3±0.5 / 27.7±0.5 12.7±0.5 / 45.0±0.8 23.7±0.5 / 38.0±1.4
GRADA (HRSIM) 7.7±0.5 / 34.0±0.8 5.3±0.5 / 41.0±0.0 12.3±0.5 / 39.0±0.8
Table 10: ASR and EM (%) for various defense methods on PoisonedRAG on GPT-4o, Llama3.1-70b-Instruct and
Qwen2.5-7b-Instruct. Blue cells indicate top-two lowest ASR.
Model DefenseHotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-4oNone 45.3±0.5 / 41.7±0.5 32.3±0.5 / 39.0±1.4 24.7±0.9 / 46.0±1.4
HLATR 42.0±0.8 / 45.0±0.8 28.3±0.9 / 48.7±1.9 19.7±2.4 / 53.0±1.4
BGE-reranker 40.0±0.0 / 41.3±0.5 27.0±0.0 / 49.0±0.0 20.0±0.0 / 53.7±0.9
Keyword Aggregation 8.7±0.5 / 59.3±1.9 1.0±0.0 / 46.0±0.0 4.0±0.0 / 48.0±1.4
GRADA (D2DSIM-EBD) 31.7±0.5 / 45.3±1.2 5.0±0.8 / 55.3±1.2 11.3±0.5 / 56.0±1.4
GRADA (D2DSIM-BM25) 21.0±0.0 / 46.3±0.9 5.0±0.0 / 61.3±0.5 7.3±0.5 / 67.0±1.4
GRADA (HRSIM) 5.0±0.0 / 49.3±1.9 1.0±0.0 / 63.3±2.4 4.0±0.0 / 66.3±0.5
Llama3.1-70b-InstructNone 56.7±0.9 / 33.3±0.9 54.7±2.1 / 26.7±1.7 47.7±0.5 / 29.0±0.8
HLATR 52.0±2.2 / 37.3±1.2 47.3±2.1 / 35.7±0.9 32.3±0.5 / 37.0±0.8
BGE-reranker 48.3±1.2 / 44.3±1.9 42.7±1.2 / 41.3±1.2 35.7±1.9 / 33.3±0.9
Keyword (Xiang et al., 2024) 4.7±0.5 / 26.0±0.8 3.0±0.0 / 22.0±0.0 3.0±0.0 / 57.0±0.8
GRADA (D2DSIM-EBD) 37.0±0.0 / 40.3±1.7 11.0±1.6 / 48.7±2.1 15.3±1.7 / 45.7±0.5
GRADA (D2DSIM-BM25) 33.3±0.9 / 37.7±2.1 6.7±0.5 / 56.0±0.8 10.3±0.5 / 51.7±1.2
GRADA (HRSIM) 8.7±0.5 / 36.7±2.6 1.0±0.0 / 54.0±0.8 6.7±0.5 / 52.3±2.4
Qwen2.5-7b-InstructNone 58.7±0.9 / 30.7±1.2 58.0±2.2 / 22.3±2.1 51.0±1.4 / 31.3±2.9
HLATR 55.7±0.9 / 33.7±2.1 51.0±0.0 / 29.0±0.8 36.3±3.3 / 33.0±3.3
BGE-reranker 54.0±1.6 / 33.7±2.6 51.0±0.8 / 29.3±0.5 37.3±4.0 / 33.3±3.9
Keyword 4.7±0.5 / 6.0±0.0 3.0±0.0 / 11.0±0.0 10.3±0.5 / 23.7±0.5
GRADA (D2DSIM-EBD) 45.7±0.9 / 31.0±1.6 14.7±1.7 / 41.0±3.6 19.0±1.6 / 36.3±0.5
GRADA (D2DSIM-BM25) 38.3±0.5 / 31.7±1.2 12.0±2.2 / 42.0±0.8 14.7±1.2 / 40.7±0.5
GRADA (HRSIM) 6.0±0.0 / 33.0±0.0 4.3±0.9 / 45.3±0.5 10.7±0.5 / 39.0±1.4
Table 11: ASR and EM (%) for various defense methods on PoisonedRAG(Hotflip). Blue cells indicate top-two
lowest ASR.

Model DefenseHotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-4oNone 99.0±0.0 / 0.3±0.5 95.7±0.5 / 3.7±0.5 80.0±0.0 / 11.0±0.0
HLATR 97.6±0.9 / 1.3±0.9 78.0±0.0 / 15.0±0.0 53.0±0.0 / 32.0±0.0
BGE-reranker 87.3±0.5 / 7.0±1.4 36.0±1.4 / 39.7±0.9 24.0±0.0 / 51.0±0.0
Keyword Aggregation 0.0±0.0 / 53.7±2.4 0.0±0.0 / 44.0±0.0 0.0±0.0 / 45.7±0.5
GRADA (D2DSIM-EBD) 30.7±0.5 / 42.3±0.9 2.0±0.0 / 57.3±0.5 2.0±0.0 / 60.0±0.0
GRADA (D2DSIM-BM25) 40.0±1.4 / 36.3±0.9 10.7±0.9 / 57.3±0.9 0.0±0.0 / 68.0±0.0
GRADA (HRSIM) 25.0±0.0 / 42.7±0.5 1.0±0.0 / 63.7±0.9 0.0±0.0 / 68.0±0.0
Llama3.1-70b-InstructNone 100.0±0.0 / 0.0±0.0 98.0±0.0 / 2.0±0.0 88.0±0.0 / 8.0±0.0
HLATR 100.0±0.0 / 0.0±0.0 91.7±0.5 / 5.3±0.5 84.0±0.0 / 8.7±0.5
BGE-reranker 98.0±0.0 / 2.0±0.0 42.3±0.5 / 38.7±0.5 43.0±0.0 / 30.3±0.5
Keyword Aggregation 0.0±0.0 / 26.7±0.5 0.0±0.0 / 23.0±1.4 0.0±0.0 / 59.3±0.9
GRADA (D2DSIM-EBD) 33.0±0.0 / 29.0±0.0 2.0±0.0 / 55.3±0.5 3.0±0.0 / 49.0±0.8
GRADA (D2DSIM-BM25) 42.0±0.0 / 25.0±0.0 12.0±0.0 / 52.0±0.8 2.0±0.0 / 54.3±1.2
GRADA (HRSIM) 26.0±0.0 / 32.0±0.8 1.3±0.5 / 55.3±0.5 1.0±0.0 / 54.7±0.5
Qwen2.5-7b-InstructNone 5.3±0.5 / 22.7±0.5 5.7±0.5 / 17.0±0.0 6.0±0.0 / 27.0±0.8
HLATR 14.0±0.8 / 24.0±1.4 17.7±0.9 / 12.7±0.9 18.0±0.0 / 20.7±0.5
BGE-reranker 25.0±0.0 / 17.0±0.0 23.0±0.0 / 31.7±0.5 18.3±0.5 / 32.0±0.0
Keyword Aggregation 0.0±0.0 / 6.0±0.0 0.0±0.0 / 11.0±0.0 0.0±0.0 / 21.3±0.5
GRADA (D2DSIM-EBD) 12.0±0.0 / 34.7±0.9 2.0±0.0 / 47.0±0.8 3.0±0.0 / 41.7±0.5
GRADA (D2DSIM-BM25) 15.0±0.0 / 28.0±0.0 8.0±0.0 / 43.7±0.5 1.0±0.0 / 44.0±1.4
GRADA (HRSIM) 8.7±0.5 / 35.3±0.5 2.0±0.0 / 46.3±0.9 1.0±0.0 / 47.3±1.7
Table 12: ASR and EM (%) for various defense methods on PIA. Blue cells indicate top-two lowest ASR.
Model DefenseHotpotQA NQ MS-MARCO
ASR ↓/ EM ↑ ASR ↓/ EM ↑ ASR ↓/ EM ↑
GPT-4oNone 57.3±0.5 / 25.3±0.9 37.0±0.0 / 18.7±0.5 21.0±0.0 / 45.0±0.0
HLATR 47.0±1.4 / 27.3±0.5 36.3±0.5 / 22.3±0.5 18.0±0.0 / 53.0±0.0
BGE-reranker 35.7±0.9 / 29.7±0.5 20.3±0.5 / 32.3±0.5 19.0±0.0 / 53.0±0.0
Keyword Aggregation 0.0±0.0 / 57.0±0.0 0.0±0.0 / 48.0±0.0 0.0±0.0 / 45.0±0.0
GRADA (D2DSIM-EBD) 30.0±1.4 / 35.3±0.5 3.7±0.5 / 43.3±1.9 2.0±0.0 / 53.0±0.0
GRADA (D2DSIM-BM25) 7.3±0.9 / 40.0±1.4 2.0±0.0 / 51.0±0.0 0.3±0.5 / 63.0±0.0
GRADA (HRSIM) 3.3±0.9 / 41.3±0.9 0.0±0.0 / 50.0±1.4 0.0±0.0 / 63.7±0.5
Llama3.1-70b-InstructNone 98.7±0.5 / 1.3±0.5 90.7±1.2 / 7.3±1.2 74.3±1.2 / 19.7±0.5
HLATR 98.0±0.8 / 0.7±0.5 93.7±0.9 / 5.3±0.5 78.0±1.6 / 13.3±0.9
BGE-reranker 96.3±0.5 / 3.7±0.5 75.7±0.9 / 14.0±0.8 70.7±0.9 / 20.3±1.7
Keyword Aggregation 0.0±0.0 / 18.7±0.5 0.0±0.0 / 17.3±0.5 0.0±0.0 / 51.3±1.2
GRADA (D2DSIM-EBD) 60.3±2.9 / 16.3±1.2 12.7±2.6 / 41.7±1.7 13.7±2.4 / 45.3±2.1
GRADA (D2DSIM-BM25) 27.0±1.4 / 27.7±1.2 5.3±0.5 / 49.3±0.5 1.3±0.5 / 55.3±0.5
GRADA (HRSIM) 11.3±0.9 / 27.3±1.2 0.7±0.5 / 50.7±1.2 0.0±0.0 / 56.0±0.8
Qwen2.5-7b-InstructNone 58.7±3.8 / 18.3±1.2 56.0±2.9 / 12.0±2.2 40.0±1.4 / 25.3±2.1
HLATR 63.0±1.4 / 17.7±2.1 71.0±1.6 / 9.3±1.7 48.3±2.1 / 18.7±2.1
BGE-reranker 62.3±4.1 / 19.7±0.5 57.7±2.6 / 19.7±1.2 50.3±0.9 / 25.7±2.5
Keyword Aggregation 0.0±0.0 / 1.0±0.0 0.0±0.0 / 5.0±0.0 0.0±0.0 / 5.0±0.0
GRADA (D2DSIM-EBD) 41.0±2.8 / 17.0±3.7 11.0±2.8 / 32.0±0.8 11.7±1.7 / 40.7±2.1
GRADA (D2DSIM-BM25) 24.0±0.0 / 27.7±2.1 5.3±0.5 / 35.3±1.2 0.3±0.5 / 45.7±0.9
GRADA (HRSIM) 14.0±2.4 / 27.3±0.9 0.7±0.5 / 36.3±0.5 0.0±0.0 / 48.7±1.7
Table 13: ASR and EM (%) for various defense methods on Phantom. Blue cells indicate top-two lowest ASR.
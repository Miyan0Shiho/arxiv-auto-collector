# Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering

**Authors**: Xukai Liu, Ye Liu, Shiwen Wu, Yanghai Zhang, Yihao Yuan, Kai Zhang, Qi Liu

**Published**: 2025-05-19 03:25:18

**PDF URL**: [http://arxiv.org/pdf/2505.12662v1](http://arxiv.org/pdf/2505.12662v1)

## Abstract
Recent advances in large language models (LLMs) have led to impressive
progress in natural language generation, yet their tendency to produce
hallucinated or unsubstantiated content remains a critical concern. To improve
factual reliability, Retrieval-Augmented Generation (RAG) integrates external
knowledge during inference. However, existing RAG systems face two major
limitations: (1) unreliable adaptive control due to limited external knowledge
supervision, and (2) hallucinations caused by inaccurate or irrelevant
references. To address these issues, we propose Know3-RAG, a knowledge-aware
RAG framework that leverages structured knowledge from knowledge graphs (KGs)
to guide three core stages of the RAG process, including retrieval, generation,
and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval
module that employs KG embedding to assess the confidence of the generated
answer and determine retrieval necessity, a knowledge-enhanced reference
generation strategy that enriches queries with KG-derived entities to improve
generated reference relevance, and a knowledge-driven reference filtering
mechanism that ensures semantic alignment and factual accuracy of references.
Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG
consistently outperforms strong baselines, significantly reducing
hallucinations and enhancing answer reliability.

## Full Text


<!-- PDF content starts -->

arXiv:2505.12662v1  [cs.CL]  19 May 2025Know³-RAG: A Knowledge-aware RAG Framework
with Adaptive Retrieval, Generation, and Filtering
Xukai Liu1,2, Ye Liu1,2, Shiwen Wu3, Yanghai Zhang1,2,
Yihao Yuan1, Kai Zhang1,2, Qi Liu1,2
1School of Computer Science and Technology, University of Science and Technology of China
2State Key Laboratory of Cognitive Intelligence, Hefei, Anhui, China
3The Hong Kong University of Science and Technology
{chthollylxk, yhzhang0612, yh0319}@mail.ustc.edu.cn, yeliu.liuyeah@gmail.com
swubs@connect.ust.hk, {kkzhang08, qiliuql}@ustc.edu.cn,
Abstract
Recent advances in large language models (LLMs) have led to impressive progress
in natural language generation, yet their tendency to produce hallucinated or un-
substantiated content remains a critical concern. To improve factual reliability,
Retrieval-Augmented Generation (RAG) integrates external knowledge during
inference. However, existing RAG systems face two major limitations: (1) un-
reliable adaptive control due to limited external knowledge supervision, and (2)
hallucinations caused by inaccurate or irrelevant references. To address these is-
sues, we propose Know ³-RAG , a knowledge-aware RAG framework that leverages
structured knowledge from knowledge graphs (KGs) to guide three core stages of
the RAG process, including retrieval, generation, and filtering. Specifically, we in-
troduce a knowledge-aware adaptive retrieval module that employs KG embedding
to assess the confidence of the generated answer and determine retrieval necessity,
a knowledge-enhanced reference generation strategy that enriches queries with KG-
derived entities to improve generated reference relevance, and a knowledge-driven
reference filtering mechanism that ensures semantic alignment and factual accuracy
of references. Experiments on multiple open-domain QA benchmarks demonstrate
that Know ³-RAG consistently outperforms strong baselines, significantly reducing
hallucinations and enhancing answer reliability.
1 Introduction
Large language models (LLMs) have demonstrated remarkable performance across a wide range of
natural language processing (NLP) tasks, such as machine translation, text generation, and question
answering [ 28]. However, they might generate content that is factually incorrect or unsubstantiated —
a phenomenon known as hallucination. As LLMs become increasingly integrated into high-stakes
domains, reducing hallucinations has emerged as a central challenge for building reliable language
generation systems [17].
To address this issue, Retrieval-Augmented Generation (RAG), a paradigm that integrates external
knowledge during inference, has emerged as a popular method to mitigate hallucinations [ 8]. Recent
studies [ 9,23,50] have explored various forms of LLM-generated content as external knowledge
to further enhance the performance of RAG. Specifically, these efforts can be broadly categorized
into two directions. The first (Figure 1(a)) explores self-adaptive RAG, where the model learns to
determine whether external evidence is needed. Typical approaches involve model self-ask [ 42,1]
or training predictive modules to estimate retrieval necessity [ 16,53]. The second (Figure 1(b))
investigates context-augmentation RAG, which aims to improve the quality of the information
Preprint. Under review.

(a) Self-Adaptive RAG Framework (b) Context -Augmentation RAG Framework
(c) Our Know3-RAG Framework
What are Michael Jordan's 
contributions to AI research?What are Michael Jordan's 
contributions to AI research?Michael Jordan, a basketball 
player , popularised Bayesian 
networks in the machine 
learning… 
Michael I. Jordan is an American scientist… 
researcher in machine learning...
Michael Jordan is an American businessman 
and former professional basketball player……
Michael I. Jordan is a prominent 
computer scientist whose work in 
machine learning , probabilistic 
graphical models , and Bayesian 
networks has significantly influenced 
the field of AI…What are Michael Jordan's 
contributions to AI research?: I got confused 
  about Jordan.
：
：I know the answer!Artificial 
Intelligence 
ResearcherMichael I. Jordan
Computer 
ScientistKG knowledge
Michael Jordan, a basketball 
player  of NBA, has great 
contributions to AI research
… 
Do you need more 
information?
No
Query 1
Query n…
There are
 too many 
 Jordan.
(Response of  LLM)
Michael I. Jordan is an American 
scientist… researcher in machine 
learning...
Michael Jordan is an American 
businessman and former 
professional basketball player……
(c.1) Knowledge -aware Adaptive Retrieval
(c.2) Knowledge -enhanced
         Reference Generation Michael I. Jordan is an American 
scientist… researcher in machine 
learning...
(c.3) Knowledge -driven 
 Reference Filter
Figure 1: Comparison of three different RAG frameworks. Our proposed Know ³-RAG Framework
employs knowledge graphs to facilitate adaptive retrieval and reference processing.
acquired via LLMs. This is achieved by enhancing query expressiveness through LLMs [ 41,39] or
synthesizing high-quality reference content directly from the LLMs’ internal knowledge [34, 9].
Despite their effectiveness, existing methods still encounter two major limitations. Firstly, for the
self-adaptive RAG, their adaptive retrieval mechanisms are typically guided by internal model signals
without knowledge verification, making them susceptible to training data biases and thus limiting
their generalization ability. In fact, studies have shown that self-ask signals often fail to accurately
reflect true retrieval needs [ 52]. An example can be seen in Figure 1(a), where the self-ask part refuse
to explore more information about Michael Jordan , resulting in its confusion in the answer.
Secondly, for context-augmentation RAG, current pipelines tend to emphasize relevance during
retrieval while overlooking the quality control, leading to the inclusion of noisy or misleading content.
On the one hand, retrievers may select factually incorrect content due to a preference for fluent LLM-
generated passages even if they are incorrect [ 6]. On the other hand, topically irrelevant information
may also be retrieved, thereby confusing the LLM’s understanding of the context. As shown in
Figure 1(b), the basketball player Michael Jordan is incorrectly included in the retrieval results of
computer scientists, thereby misleading the model generation process. Such failures highlight the
need for external knowledge supervision throughout the RAG pipeline, from adaptive control to
reference filtering.
To address these limitations, we propose Know ³-RAG , a Knowledge-aware RAG framework that
leverages structured knowledge graphs (KGs) as external supervision across three core stages of
the RAG process, including retrieval, generation and filtering. Specifically, as shown in Figure 1
(c), to address the problem of unreliable adaptive retrieval, we first proposed the Knowledge-aware
Adaptive Retrieval, which employs KG representation to assess the confidence of current generated
answer and determine whether additional retrieval is necessary. Subsequently, to boost the quality
of references, we introduce Knowledge-enhanced Reference Generation, which injects KG-related
entities into the query formulation, thus generating more relevant and specific reference documents.
Further, a Knowledge-driven Reference Filtering mechanism is designed to evaluate the retrieved
references for both semantic relevance and factual consistency with the given input query.
Finally, extensive experiments on multiple open-domain QA benchmarks demonstrate that Know ³-
RAG achieves state-of-the-art performance while significantly improving interpretability and reducing
hallucinations. The code is available at https://github.com/laquabe/Know3RAG .
2 Related Works
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has emerged as a key strategy for mitigating hallucinations
in large language models (LLMs) by integrating external knowledge into inference [ 8]. Initial
2

methods, such as In-Context RALM [ 29], directly incorporated retrieved text into the prompts, while
IRCoT [ 36] integrated iterative retrieval with reasoning steps. Previous RAG methods often retrieve
a fixed number of documents, risking either inadequate coverage (if too few are retrieved) or noise
introduction (if too many are retrieved), both of which can harm model effectiveness [ 15]. To address
this, adaptive retrieval techniques were developed. For instance, SKR [ 42] introduced a self-ask
mechanism to guide retrieval. Adaptive RAG [ 16] and Thrust [ 53] leveraged a classifier to determine
retrieval necessity, which was based on text or embedding, respectively. Self-RAG [ 1] trained the
LLM to emit special tokens indicating retrieval actions. However, their reliance on the LLM itself or
the training data makes their generalization susceptible to data biases.
Meanwhile, research also focused on optimizing the content provided to the LLMs. Some methods
aimed to retrieve more relevant information by enhancing query expressiveness. Query2Doc [ 41] and
HyDE [ 10] expanded the query by rewriting LLMs. ToC [ 21] employs a tree-structured search to
explore diverse evidence. BlendFilter [ 39] combined internal and external query augmentations with
a LLM-based knowledge filtering mechanism to improve retrieval quality. Other works concentrate
on generating high-quality references directly from LLMs’ internal knowledge. Recitation [34] and
GenRead [ 50] utilize LLMs to synthesize pseudo-documents as reference documents. LongLLM-
Lingua [ 18] compresses documents via a trainable model to extract key information. DSP [ 24]
and BGM [ 20] further optimize the model through reinforcement learning based on LLM feedback.
Knowledge Card [ 9] suggested fine-tuning domain knowledge models and allowing the LLM to
select the useful domain. Nevertheless, these approaches often do not integrate explicit verification
mechanisms to ensure the quality of external knowledge. In contrast, by providing explicit verifica-
tion powered by knowledge graphs, our proposed training-free method precisely addresses this gap,
significantly enhancing the relevance and reliability of the references.
2.2 LLM Reasoning with Knowledge Graph
Knowledge Graphs (KGs), like DBpedia [ 22], YAGO [ 30], and Wikidata [ 38], explicitly structure
rich factual knowledge. Integrating this structured knowledge provides a promising approach to
enhance LLM reasoning capabilities [ 27]. Some methods focus on adapting KG information for
LLM reasoning. For example, Mindmap [ 44] converted KGs into mind maps via designed prompts to
facilitate model comprehension. StructGPT [ 19] and React [ 49] leveraged the Tool-augmented LLM
paradigm, defining APIs to enable LLM interaction with KGs. Furthermore, Think-on-Graph [ 33]
regarded the LLM as an agent exploring KGs, while KnowledgeNavigator [ 13] identifies similar
questions to provide diverse starting points.
Other methods focus on integrating internal model knowledge with external KG evidence. For
instance, CoK [ 40] assumed that LLMs have internalized the KG knowledge KG and prompts it to
generate relevant triples before answering the question. RoG [ 26] and Readi [ 5] generated relational
paths via fine-tuning or prompting, and leverage these paths to reason over KGs. R3 [ 35] utilized
KGs to validate individual reasoning steps performed by the LLM. GraphRAG [ 7] employed LLMs
to construct KGs from unstructured documents and leverages the KGs to enhance retrieval. However,
many existing methods primarily engage with local, textual information from the KG, overlooking
the rich global structural information inherent in the graph. Our framework, in contrast, addresses this
limitation by employing Knowledge Graph Embedding (KGE) techniques. This allows us to encode
global structural and semantic information from the KG and integrate it throughout the retrieval and
generation pipeline, thereby improving the relevance and consistency of the generated references.
3 Problem Definition
We consider the task of open-domain question answering, which aims to generate an answer to a
given question without providing any predefined context in advance. Formally, given a question Q,
the model seeks to produce an answer aby leveraging any available external knowledge sources.
Following the setting of existing paradigm [ 34,50,9], we adopt the LLM-generated content as the
external knowledge, which is motivated by the observation that LLM outperforms traditional retrieval
methods in handling complex queries [ 23]. Speically, we decomposes the task into two stages: (1)
generating a set of references D={d1, d2, . . . , d k}relevant to Q, and (2) generating the answer a
based on the question and the references. The incorporation of external knowledge helps mitigate the
limitations of parametric memory in models, enabling more accurate answer generation.
3

Question
Question
Answer 
Answer
Knowledge Graph
Triple 
Factual 
Check
Final
Answer
Query 
Generation
Query
Entity -enhanced
Query
Knowledge
Model 1 Knowledge Generation Models  
… Knowledge
Model n 
Pseudo
Reference
2
Pseudo
Reference
2n
…Pseudo References
Triple 
Factual 
CheckLLM
Relevance  
Check
References  FilterUseful
Reference
1
Useful
Reference
k
…Useful ReferencesNext Turn
Pseudo
Reference
1
(b) Knowledge -enhanced
Reference Generation(c) Knowledge -driven
Reference Filter(a) Knowledge -aware Adaptive RetrievalFigure 2: Model Architecture of Know ³-RAG, which contains: (a) Knowledge-aware Adaptive
Retrieval, (b) Knowledge-enhanced Reference Generation, (c) Knowledge-driven Reference Filter.
4 Know³-RAG Method
4.1 Overview
As illustrated in Figure 2, our Know ³-RAG framework comprises three core components: (a)
Knowledge-aware Adaptive Retrieval, (b) Knowledge-enhanced Reference Generation, and (c)
Knowledge-driven Reference Filter. These modules are organized in a closed-loop architecture
to enable iterative refinement of answers. In detail, the Knowledge-aware Adaptive Retrieval (a)
receives the answer from the previous QA iteration and verifies the reliability of extracted triples in
the answer using a knowledge graph (KG), thereby determining the necessity for additional retrieval.
If further retrieval is required, the Knowledge-enhanced Reference Generation (b) augments the
query with relevant KG entities and then generates pseudo-references via knowledge generation
models. Subsequently, the Knowledge-driven Reference Filter (c) evaluates the pseudo-references
for relevance and factuality and retains the useful references among them. The filtered references
are then combined with the original question and previous references to inform the QA model to
generate the next round answer, thus completing one full process cycle.
4.2 Knowledge-aware Adaptive Retrieval
As discussed in the Introduction, the adaptive iterative mechanism determines whether to invoke
retrieval based on the reliability of the current answer. The central challenge lies in evaluating
the reliability of the generated answer without human annotations. To address this, we employ
the Knowledge Graph Embedding (KGE) model to quantify the reliability of the generated text.
Specifically, for the answer atgenerated at the t-th iteration for the question Q, we first extract a set
of evidence triples Tri(at)={tri(at)
1, . . . , tri(at)
i}using a large language model. Each triple is then
scored using a relative triple scoring function, and the overall reliability score stfor the answer is
computed by aggregating the individual triple scores. Formally:
at=LLM ans(Q, Dc
t),Tri(at)=LLM tri(at), st=X
iS(tri(at)
i), (1)
where LLM ansdenotes the QA model, and Dc
tis the set of references available at iteration t.LLM tri
refers to the triple extraction LLM, and S(·)is the relative triple scoring function.
Relative Triple Scoring. Scores produced by KGE models lack absolute interpretability and are
meaningful only in a comparative sense, as they are primarily designed for ranking tasks [ 3,37].
To leverage these scores for reliability assessment, we propose a relative triple score. Specifically,
for each extracted triple tri(at)
ifrom the generated answer at, we retrieve a set of reference triples
Tri(at)
refi={tri(at)
refi1, . . . , tri(at)
refij}from the KG that share the same head entity h(at)
i. We then compute
4

the triple relative score by measuring the absolute value between its KGE score and the average KGE
score of its reference triples:
S(tri(at)
i) =KGE(tri(at)
i)−1
|Tri(at)
refi|X
jKGE(tri(at)
refij), (2)
where KGE(·)denotes the KGE scoring function, and | · |is the absolute value function. Intuitively,
a lower relative score indicates higher consistency between the generated content and knowledge
in the KG, thus reflecting greater reliability of the answer. During inference, our adaptive retrieval
employs an iterative process with a dynamically increasing threshold θt[46,32]. At each iteration
t, we compare the relative triple score stagainst θtto decide whether to terminate. The threshold
is defined as θt=θ0
c
1+e1−θ0t
, where θ0is a hyperparameter dependent on the QA model and
dataset, and cis a constant. To prevent infinite iteration, a maximum number of iterations, denoted by
T, is set. If st< θtor ift=T, the answer is output; otherwise, a reference query is triggered for
further refinement.
4.3 Knowledge-enhanced Reference Generation
When retrieval is triggered, the Knowledge-enhanced Reference Generation introduces additional
references to support answer generation. Specifically, we first generate a new query qtfor the current
iteration, defined as:
qt=Q, t = 1
LLM query(Q, Dc
t, at), t > 1,(3)
where LLM query denotes a LLM for query generation. After obtaining the query qt, we further
enhance it by incorporating entity information from the knowledge graph (KG) to enrich the query.
Specifically, we get a set of relevant entities Et, including three types:
•Query linked entities Eq
t. Entities directly linked to the query via entity linking, representing
knowledge most relevant to the query.
•Local neighbor entities El
t. Entities that are one-hop neighbors of Eq
tin the KG. We filter these
neighbors based on the semantic relevance between their relations and the query.
•Global predicted entities Eg
t. Tail entities predicted by the KGE model along relations connected
toEq
t, supplementing El
tto mitigate knowledge graph incompleteness.
Incorporating these relevant entities enables more effective knowledge infusion, improving the effi-
ciency of the iterative generation process. The detailed entity enhancement procedure is presented
in Algorithm 1 (refer to Appendix A). The entity-enhanced query qKG
tis then constructed by con-
catenating the relevant entity set and the original query: qKG
t= [Et;qt]. Then, using both qtand
qKG
t, we generate candidate reference documents through knowledge models, which are any kind
of generative models. Unlike traditional retrieval methods, knowledge models can generate more
targeted documents for complex queries [ 34,50,23]. The document generation process is formalized
as:
d=K(q), (4)
where K(·)denotes a knowledge model. To further enrich the references, we adopt a multi-source
strategy, invoking ndifferent knowledge models to generate documents based on both qtandqKG
t,
resulting in an initial reference candidate set Dm:
Dm={dt1, . . . , d tn, dKG
t1, . . . , dKG
tn}, (5)
4.4 Knowledge-driven Reference Filter
As discussed in Introduction, noisy or misleading content may be included to influence the generation
process. To mitigate the risk of introducing irrelevant or inaccurate documents into the final reference
set, we design the Knowledge-driven Reference Filtering module, which consists of two core
components: (1) LLM relevance check and (2) triple factual check.
Specifically, the LLM relevance check assesses the semantic relevance between the generated
document dand the original question Q. To improve the robustness of the relevance assessment, we
5

incorporate entity information EQextracted from the question as additional context. Formally, the
relevance check is defined as:
c=LLM rel(d, Q, EQ), (6)
where LLM reldenotes a LLM used for relevance judgment, and c∈ {False,True}represents the
binary relevance decision. The triple factual check follows the approach introduced in the Knowledge-
aware Adaptive Retrieval module: we extract triples from the document and score the reliability of
the document using the relative triple score.
Finally, we integrate the results from both modules to filter useful references. Specifically, among
documents judged as relevant, we select the top kdocuments ranked by their relative triple scores,
forming the next iteration’s reference set Dc
t+1:
Dc
t+1=Dc
t∪ {dc
t1, . . . , dc
tk},
at+1=LLM ans(Q, Dc
t+1),(7)
where dc
tkdenotes the documents selected in the current round. In this way, our Know ³-RAG
framework completes a full circle.
5 Experiments
5.1 Datasets
Table 1: The Statistics of Test Datasets
Dateset Nums.
HotpotQA 7,405
2WikiMultiHopQA 12,576
PopQA 14,267We evaluate our approach on three widely used open-
domain QA benchmarks: HotpotQA [ 48], 2WikiMulti-
HopQA [ 14], and PopQA [ 48]. Following the setup in
Wang et al. [ 39], we report results on the publicly available
development sets for HotpotQA and 2WikiMultiHopQA,
and on the official test set for PopQA. We use Exact Match
(EM) and F1 as evaluation metrics. Dataset statistics are
summarized in Table 1. It is important to note that al-
though these datasets may provide auxiliary information (e.g., supporting passages), we do not use
any of it during evaluation. All methods rely solely on the input question, and all additional context
is obtained by the method itself.
5.2 Baselines
We adopt following state-of-the-art baselines to evaluate our proposed method:
•Direct Prompting [4]: A simple approach where the LLM directly answers questions without
reasoning steps. We evaluate both directly answering with and without retrieval.
•Chain-of-Thought [43]: This method instructs the LLM to generate reasoning steps before
answering. Similar to Direct Prompting, we evaluate both CoT with and without retrieval.
•Recite [34]: This method leverages the LLM’s parametric knowledge by prompting it to first recite
relevant factual content, then LLMs respond based on that recitation.
•SKR [42]: A self-adaptive RAG that queries the LLM to decide whether retrieval is necessary.
•Knowledge Card (KC) [9]: This approach uses domain-finetuned knowledge models for doc-
ument generation. We evaluate the three KG-related knowledge models provided by the au-
thors—Wikidata, Wikipedia, and YAGO—as external knowledge sources.
•Chain of Knowledge (CoK) [40]: Based on the hypothesis that LLMs internalize knowledge
graph, this method prompts the LLM to output relevant triples before generating an answer.
•BlenderFilter (BF) [39]: A hybrid retrieval framework that combines internal knowledge of LLMs
and external knowledge of retrieval. Retrieved content is filtered by an LLM to discard irrelevant
information, aiming to broaden retrieval coverage while maintaining quality.
Following Wang et al. [ 39], for all retrieval-based methods, we use ColBERTv2 [ 31] as the retriever
and employ the 2017 Wikipedia abstract dump as the retrieval corpus [ 48]. We exclude the KBQA
methods such as Think-on-Graph [ 33], as they need to access the gold topic entities and structured
queries, which are incompatible with our open-domain QA setting.
6

Table 2: Main Results
GLM4-9b Qwen2.5-32b GPT4o-mini
Method HotPot 2Wiki Pop HotPot 2Wiki Pop HotPot 2Wiki Pop
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Direct 0.181 0.242 0.252 0.278 0.185 0.222 0.237 0.322 0.277 0.322 0.181 0.237 0.220 0.294 0.226 0.272 0.327 0.378
CoT 0.190 0.262 0.245 0.308 0.185 0.235 0.233 0.331 0.243 0.317 0.190 0.237 0.315 0.426 0.383 0.443 0.355 0.401
RAG-D 0.219 0.281 0.233 0.300 0.279 0.326 0.248 0.335 0.269 0.324 0.299 0.352 0.291 0.383 0.301 0.360 0.316 0.362
RAG-C 0.196 0.262 0.193 0.242 0.257 0.302 0.243 0.321 0.262 0.304 0.273 0.327 0.307 0.404 0.337 0.410 0.373 0.428
Recite 0.215 0.318 0.209 0.274 0.183 0.240 0.243 0.345 0.273 0.339 0.193 0.248 0.328 0.447 0.347 0.434 0.331 0.386
SKR 0.218 0.280 0.260 0.312 0.286 0.332 0.253 0.336 0.232 0.280 0.232 0.282 0.303 0.402 0.232 0.280 0.366 0.420
KC 0.183 0.262 0.229 0.261 0.171 0.208 0.221 0.295 0.167 0.195 0.279 0.309 0.313 0.410 0.172 0.208 0.297 0.331
CoK 0.205 0.288 0.269 0.302 0.192 0.230 0.251 0.353 0.274 0.333 0.200 0.245 0.328 0.443 0.379 0.441 0.337 0.387
BF 0.221 0.308 0.232 0.273 0.287 0.332 0.257 0.343 0.207 0.252 0.296 0.349 0.339 0.448 0.272 0.339 0.380 0.439
Know³-RAG 0.235 0.351 0.261 0.362 0.337 0.399 0.270 0.387 0.343 0.424 0.332 0.406 0.335 0.453 0.382 0.457 0.396 0.462
5.3 Implementation Details
We evaluate our framework using three types of LLMs: GLM4-9B [ 11], Qwen2.5-32B [ 47], and
GPT-4o-mini, representing two open-source models of different sizes and one closed-source model.
For entity linking, we use spaCy to efficiently identify entities. We choose ComplEX [ 37] as the KGE
model. For the knowledge models, we consider two categories: (1) Non-fine-tuned models, including
LLaMA3-8B [ 12], Qwen2.5-7B [ 47], and the QA model itself; (2) Fine-tuned domain models from
Knowledge Card [ 9], based on OPT-1.3B [ 51] and trained separately on Wikidata, Wikipedia, and
YAGO. To maintain efficiency, each knowledge model outputs only one document per query. All
LLM-based modules apart from the QA model use lightweight open-source LLMs like LLaMA3-8B.
The hyperparameter θ0is in Table 6 in Appendix B.1, the constant c is 128, and the maximum number
of iteration Tis 2. At each iteration, we retain k+tdocuments as useful references, with kset to 5.
All QA models are prompted using a 3-shot in-context learning setup and set the temperature to 0.
Detailed prompts are provided in Appendix D.
5.4 Experimental Results
Table 2 presents the main results across three models and three open-domain QA datasets. Our
Know³-RAG framework consistently achieves the best or near-best performance in both EM and F1
scores, demonstrating robust generalizability across models and datasets.
We also highlight two interesting observations. First, both intrinsic and external knowledge are
essential. For example, Chain-of-Knowledge (CoK), which relies on intrinsic knowledge, outperforms
BlenderFilter (BF), which leverages external knowledge, on the 2WikiMultiHopQA (2Wiki) dataset,
while the opposite holds for PopQA. In contrast, our method integrates both to achieve superior
performance across most models and datasets. Second, we find that naive self-ask strategies, such as
SKR, may misjudge the actual retrieval need, leading to an inferior result compared to direct retrieval
(RAG) in some scenarios. Our Know ³-RAG framework addresses this issue through our triple-based
adaptive RAG, which more accurately identifies the knowledge gaps of the model and iteratively
refines the quality of the answers.
5.5 Ablation StudyTable 3: Ablation Analysis on GLM4-9b
MethodHotPot 2Wiki Pop
EM F1 EM F1 EM F1
Know³-RAG 0.235 0.351 0.261 0.362 0.337 0.399
➀w/o Reference 0.190 0.262 0.245 0.308 0.185 0.235
w/o KG query 0.212 0.321 0.253 0.341 0.228 0.284
w/o raw query 0.193 0.287 0.207 0.267 0.328 0.385
➁w/o Filter 0.221 0.335 0.229 0.322 0.314 0.380
w/o Tri-check 0.230 0.347 0.231 0.324 0.331 0.395
w/o Rel-check 0.211 0.318 0.228 0.315 0.296 0.359
➂w/o Adaptive
Retrieval0.227 0.339 0.259 0.357 0.330 0.391Table 3 reports the results of the ablation
study conducted on GLM4-9b across three
QA datasets. The results confirm the effec-
tiveness of all three proposed modules➀➁➂.
We further provide a detailed analysis of
two components. For the Knowledge-
enhanced Reference Generation module,
intrinsic knowledge proves more valuable
than external knowledge. This observa-
tion suggests that large language models
increasingly internalize factual knowledge
as they develop. Moreover, it implies that
7

Figure 3: Reference Used of Various Knowledge Models in Turn 1
incorporating knowledge graphs may introduce noise, such as linking to the wrong entities, which
can misguide reference generation. As for the Knowledge-driven Reference Filter module, we find
that relevance check yields greater improvements than factual check. This indicates that factually
correct yet contextually irrelevant content poses a greater risk of misleading the model’s reasoning,
highlighting the importance of semantic alignment in the reference selection process.
5.6 Comparison of Knowledge ModelsTable 4: Performance of Partial Knowledge Models
ModelHotPot 2Wiki Pop
EM F1 EM F1 EM F1
Llama3-8b 0.216 0.296 0.248 0.295 0.233 0.276
Qwen2.5-7b 0.182 0.249 0.246 0.287 0.159 0.195
GLM4-9b 0.190 0.262 0.245 0.308 0.185 0.235
Qwen2.5-32b 0.233 0.331 0.243 0.317 0.190 0.237
GPT4o-mini 0.315 0.426 0.383 0.443 0.355 0.401
Know³-RAG
GLM4-9b 0.235 0.351 0.261 0.362 0.337 0.399
Qwen2.5-32b 0.270 0.387 0.343 0.424 0.332 0.406
GPT4o-mini 0.335 0.453 0.382 0.457 0.396 0.462To demonstrate how Know ³-RAG uti-
lizes knowledge across models, we first
analyze the relevance and usage of ref-
erences generated by each model. Rel-
evance denotes whether a reference is
semantically aligned with the question,
while usage indicates whether it is in-
cluded in the final reference set. As
shown in Figure 3, larger models (e.g.,
LLaMA3, Qwen2.5) exhibit high rele-
vance and usage, reflecting their strong
inherent knowledge, consistent with pre-
vious ablation results. KG-enhanced
queries on these models also yield substantial usage rates. In contrast, smaller models yield low
relevance and usage when queried directly, as also observed in the main results of Knowledge Card
(KC). However, incorporating KGs significantly boosts their usage, underscoring the role of KGs
in stimulating the knowledge of weaker models. To validate that Know ³-RAG selectively exploits
correct model knowledge rather than simply aggregating it, Table 4 reports QA performance of partial
knowledge models. Smaller models are omitted due to fine-tuning limitations. As shown, all models
perform worse than Know ³-RAG, suggesting effective knowledge selection of Know ³-RAG. The
impact of the maximum reference is further analyzed in Appendix B.2.
5.7 Analysis of Adaptive Retrieval Control
To intuitively show how our knowledge-aware adaptive retrieval controls the output decision, Table 5
presents the percentage of output answers for each turn. We observe that only a small percentage
of questions are resolved without retrieval at t= 0, highlighting the necessity of incorporating
8

Table 5: Percentage of Output Answers in Different Turns
TurnGLM4-9b Qwen2.5-32b GPT4o-mini
Hotpot 2Wiki Pop Hotpot 2Wiki Pop Hotpot 2Wiki Pop
t = 0 0.0952 0.0166 0.0001 0.0042 0.0009 0.0000 0.3554 0.0180 0.0001
t = 1 0.3011 0.8384 0.0390 0.7494 0.2253 0.0036 0.5286 0.8611 0.0020
t = 2 0.6037 0.1450 0.9609 0.2464 0.7738 0.9964 0.1160 0.1209 0.9979
Bernhard Heiden was born in 
Baar, Switzerland …What was the nationality of 
Bernhard Heiden's teacher?
Turn 0
Turn 1…Bernhard Heiden was a notable composer and 
music educator, and his most well -known teacher 
was Arnold Schoenberg … Schoenberg is primarily 
identified as Austrian…The answer is Austrian.
QA 
LLM
QuestionTurn0 Answer Knowledge -aware Adaptive Retrieval(Bernhard Heiden, student of, Arnold Schoenberg)
(Arnold Schoenberg, country of citizenship, Austria)
Evidence TriplesNext Turn
What was the nationality of 
Bernhard Heiden's teacher?
Entity -enhanced
QueryQuestion
Bernhard Heiden
Paul HindemithRelated EntityQuery
Knowledge
Model 1 
Knowledge Generation Models  …
Knowledge
Model n Knowledge
Model 2 
Bernhard Heiden was a 
German -American composer 
and conductor. His most 
noted teacher was Ferruccio 
Busoni , an Italian composer, 
pianist, and conductor…Bernhard Heiden was a 
German composer who 
studied under the guidance 
of another renowned 
German composer, Paul 
Hindemith …
Pseudo References
Triple 
Factual 
CheckLLM
Relevance  
Check
Knowledge -driven 
Reference FilteringBernhard Heiden 
was a German 
composer who 
studied under the 
guidance of 
another renowned 
German 
composer, Paul 
Hindemith …Question
What was the 
nationality of 
Bernhard Heiden's 
teacher?
…
References
…
QA  LLM
Turn 1Question -Answer
…According to the 
references, 
Bernhard Heiden 
was a German 
composer…Heiden 
studied under Paul 
Hindemith …The 
references confirm 
that Paul Hindemith 
was a German  
composer…The 
answer is German.
Turn1 Answer
Triple 
Factual 
Check
Triple 
Factual 
Check
(Bernhard Heiden, 
student of,
Paul Hindemith)
(Paul Hindemith,
country of 
citizenship,
German)
Knowledge -aware 
Adaptive RetrievalEvidence TriplesFinal Answer
German
Knowledge -enhanced 
Reference Generation
Figure 4: A case study of Know³-RAG, which utilize knowledge graph for open-domain question.
external knowledge. At t= 1, the majority of questions from HotpotQA and 2WikiMultiHopQA
are successfully answered, suggesting that a single retrieval iteration is typically sufficient for these
datasets as we fuse the related entities. In contrast, a considerable number of PopQA questions
continue to the next turn. We attribute this to the long-tailed nature of PopQA, which makes it
challenging for the KGE model to assess the reliability of the responses. We also explore the impact
of the maximum number of iterations on performance in Appendix B.3.
5.8 Case Study
To provide an intuitive illustration of our framework, we present a representative case in Figure 4.
At turn 0, the QA model initially produces an incorrect answer, incorrectly assuming that Bernhard
Hayden studied under Arnold Schoenberg . Our Knowledge-aware Adaptive Retrieval catches the error
triple (Bernhard Heiden, student of, Arnold Schoenberg) , thereby triggering the retrieval. In turn 1, our
framework uses the Knowledge-enhanced Reference Generation module to construct a KG-enhanced
query. Specifically, Bernhard Hayden andPaul Hindemith are identified as related entities. After
generating multiple candidate references using the query, our Knowledge-driven Reference Filter
filters the irrelevant or incorrect references. With the filtered references, the QA model re-generates
an answer, correctly identifying that Bernhard Heiden studied under Paul Hindemith , aGerman
national. The corresponding factual triples (Bernhard Heiden, student of, Paul Hindemith) and(Paul
Hindemith, country of citizenship, German) are successfully validated, and the correct answer is
accepted by our framework.
6 Conclusions
In this work, we present Know ³-RAG, a knowledge-aware retrieval-augmented generation framework
that introduces structured knowledge supervision from knowledge graphs across three key stages:
adaptive retrieval, reference generation, and reference filtering. By incorporating external knowledge
signals into each component, Know ³-RAG enables more reliable control over retrieval behavior,
improves the relevance and coverage of retrieved content, and enhances the factual consistency of
final outputs. Extensive experiments on multiple open-domain QA benchmarks demonstrate that
Know ³-RAG consistently outperforms strong baselines in terms of both accuracy and hallucination
reduction. We hope our work will lead to more future studies.
9

References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag:
Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations .
[2]Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, and Andrea Pierleoni.
2022. ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking. In
Proceedings of the 2022 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies: Industry Track . 209–220.
[3]Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
2013. Translating embeddings for modeling multi-relational data. Advances in neural informa-
tion processing systems 26 (2013).
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al .2020. Language
models are few-shot learners. Advances in neural information processing systems 33 (2020),
1877–1901.
[5]Sitao Cheng, Ziyuan Zhuang, Yong Xu, Fangkai Yang, Chaoyun Zhang, Xiaoting Qin, Xiang
Huang, Ling Chen, Qingwei Lin, Dongmei Zhang, et al .2024. Call Me When Necessary:
LLMs can Efficiently and Faithfully Reason over Structured Environments. In Findings of the
Association for Computational Linguistics ACL 2024 . 4275–4295.
[6]Sunhao Dai, Chen Xu, Shicheng Xu, Liang Pang, Zhenhua Dong, and Jun Xu. 2024. Bias and
Unfairness in Information Retrieval Systems: New Challenges in the LLM Era. In Proceedings
of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Barcelona,
Spain) (KDD ’24) . Association for Computing Machinery, New York, NY , USA, 6437–6447.
https://doi.org/10.1145/3637528.3671458
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to
global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024).
[8]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining . 6491–6501.
[9]Shangbin Feng, Weijia Shi, Yuyang Bai, Vidhisha Balachandran, Tianxing He, and Yulia
Tsvetkov. 2024. Knowledge Card: Filling LLMs’ Knowledge Gaps with Plug-in Specialized
Language Models. International Conference on Learning Representations.
[10] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval
without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) . 1762–1777.
[11] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego
Rojas, Guanyu Feng, Hanlin Zhao, et al .2024. Chatglm: A family of large language models
from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793 (2024).
[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al .2024.
The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
[13] Tiezheng Guo, Qingwen Yang, Chen Wang, Yanyi Liu, Pan Li, Jiawei Tang, Dapeng Li, and
Yingyou Wen. 2024. Knowledgenavigator: Leveraging large language models for enhanced
reasoning over knowledge graph. Complex & Intelligent Systems 10, 5 (2024), 7063–7076.
10

[14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing
A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings
of the 28th International Conference on Computational Linguistics , Donia Scott, Nuria Bel,
and Chengqing Zong (Eds.). International Committee on Computational Linguistics, Barcelona,
Spain (Online), 6609–6625. https://doi.org/10.18653/v1/2020.coling-main.580
[15] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A survey on hallucination in
large language models: Principles, taxonomy, challenges, and open questions. ACM Transac-
tions on Information Systems 43, 2 (2025), 1–55.
[16] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-
RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question
Complexity. In Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers) . 7029–7043.
[17] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation.
Comput. Surveys 55, 12 (2023), 1–38.
[18] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili
Qiu. 2024. LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios
via Prompt Compression. In Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 1658–1677.
https://doi.org/10.18653/v1/2024.acl-long.91
[19] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. 2023.
StructGPT: A General Framework for Large Language Model to Reason over Structured Data.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing .
9237–9251.
[20] Zixuan Ke, Weize Kong, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky.
2024. Bridging the Preference Gap between Retrievers and LLMs. In Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) .
10438–10451.
[21] Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, and Jaewoo Kang. 2023. Tree
of clarifications: Answering ambiguous questions with retrieval-augmented large language
models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing . 996–1009.
[22] Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N Mendes,
Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, et al .2015. Dbpedia–a
large-scale, multilingual knowledge base extracted from wikipedia. Semantic web 6, 2 (2015),
167–195.
[23] Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. 2024. Retrieval
Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing:
Industry Track , Franck Dernoncourt, Daniel Preo¸ tiuc-Pietro, and Anastasia Shimorina (Eds.).
Association for Computational Linguistics, Miami, Florida, US, 881–893. https://doi.
org/10.18653/v1/2024.emnlp-industry.66
[24] Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, and Xifeng Yan. 2023.
Guiding large language models via directional stimulus prompting. Advances in Neural Infor-
mation Processing Systems 36 (2023), 62630–62656.
[25] Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao
Dong, and Jie Tang. 2023. WebGLM: towards an efficient web-enhanced question answering
system with human preferences. In Proceedings of the 29th ACM SIGKDD conference on
knowledge discovery and data mining . 4549–4560.
11

[26] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on Graphs:
Faithful and Interpretable Large Language Model Reasoning. In The Twelfth International
Conference on Learning Representations .
[27] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying
large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge
and Data Engineering 36, 7 (2024), 3580–3599.
[28] Mohaimenul Azam Khan Raiaan, Md Saddam Hossain Mukta, Kaniz Fatema, Nur Mohammad
Fahad, Sadman Sakib, Most Marufatul Jannat Mim, Jubaer Ahmad, Mohammed Eunus Ali, and
Sami Azam. 2024. A review on large language models: Architectures, applications, taxonomies,
open issues and challenges. IEEE access 12 (2024), 26839–26874.
[29] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown,
and Yoav Shoham. 2023. In-context retrieval-augmented language models. Transactions of the
Association for Computational Linguistics 11 (2023), 1316–1331.
[30] Thomas Rebele, Fabian Suchanek, Johannes Hoffart, Joanna Biega, Erdal Kuzey, and Gerhard
Weikum. 2016. YAGO: A multilingual knowledge base from wikipedia, wordnet, and geonames.
InThe Semantic Web–ISWC 2016: 15th International Semantic Web Conference, Kobe, Japan,
October 17–21, 2016, Proceedings, Part II 15 . Springer, 177–185.
[31] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In
Proceedings of the 2022 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies . 3715–3734.
[32] Haitian Sun, Tania Bedrax-Weiss, and William Cohen. 2019. PullNet: Open Domain Question
Answering with Iterative Retrieval on Knowledge Bases and Text. In Proceedings of the 2019
Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , Kentaro Inui, Jing Jiang,
Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong,
China, 2380–2390. https://doi.org/10.18653/v1/D19-1242
[33] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni,
Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep and Responsible Reasoning
of Large Language Model on Knowledge Graph. In The Twelfth International Conference on
Learning Representations .
[34] Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2023. Recitation-
Augmented Language Models. In The Eleventh International Conference on Learning Repre-
sentations .
[35] Armin Toroghi, Willis Guo, Mohammad Mahdi Abdollah Pour, and Scott Sanner. 2024. Right
for Right Reasons: Large Language Models for Verifiable Commonsense Knowledge Graph
Question Answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing . 6601–6633.
[36] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleav-
ing Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.
InProceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) . 10014–10037.
[37] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard.
2016. Complex embeddings for simple link prediction. In International conference on machine
learning . PMLR, 2071–2080.
[38] Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase.
Commun. ACM 57, 10 (2014), 78–85.
[39] Haoyu Wang, Ruirui Li, Haoming Jiang, Jinjin Tian, Zhengyang Wang, Chen Luo, Xianfeng
Tang, Monica Cheng, Tuo Zhao, and Jing Gao. 2024. BlendFilter: Advancing Retrieval-
Augmented Large Language Models via Query Generation Blending and Knowledge Filtering.
12

InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing .
1009–1025.
[40] Jianing Wang, Qiushi Sun, Xiang Li, and Ming Gao. 2024. Boosting Language Models
Reasoning with Chain-of-Knowledge Prompting. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers) , Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok,
Thailand, 4958–4981. https://doi.org/10.18653/v1/2024.acl-long.271
[41] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query Expansion with Large
Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing . 9414–9423.
[42] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023. Self-Knowledge Guided Retrieval
Augmentation for Large Language Models. In Findings of the Association for Computational
Linguistics: EMNLP 2023 . 10303–10315.
[43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems 35 (2022), 24824–24837.
[44] Yilin Wen, Zifeng Wang, and Jimeng Sun. 2024. MindMap: Knowledge Graph Prompting
Sparks Graph of Thoughts in Large Language Models. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) . 10370–
10388.
[45] Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2019.
Scalable zero-shot entity linking with dense entity retrieval. arXiv preprint arXiv:1911.03814
(2019).
[46] Yi Xu, Lei Shang, Jinxing Ye, Qi Qian, Yu-Feng Li, Baigui Sun, Hao Li, and Rong Jin. 2021.
Dash: Semi-supervised learning with dynamic thresholding. In International conference on
machine learning . PMLR, 11525–11536.
[47] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin
Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li,
Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXiv
preprint arXiv:2412.15115 (2024).
[48] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop
Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural
Language Processing . 2369–2380.
[49] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan
Cao. 2023. React: Synergizing reasoning and acting in language models. In International
Conference on Learning Representations (ICLR) .
[50] Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, S Sanyal, Chenguang Zhu,
Michael Zeng, and Meng Jiang. 2023. Generate rather than Retrieve: Large Language Models
are Strong Context Generators. In International Conference on Learning Representations .
[51] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen,
Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al .2022. Opt: Open pre-trained
transformer language models. arXiv preprint arXiv:2205.01068 (2022).
[52] Zihan Zhang, Meng Fang, and Ling Chen. 2024. RetrievalQA: Assessing Adaptive Retrieval-
Augmented Generation for Short-form Open-Domain Question Answering. In Findings of the
Association for Computational Linguistics ACL 2024 . 6963–6975.
13

[53] Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, and Jianshu Chen. 2023.
Thrust: Adaptively propels large language models with external knowledge. Advances in Neural
Information Processing Systems 36 (2023), 69930–69948.
14

A Related Entity Search Algorithm
In section 4.3, we propose to augment the query with relevant entities from the knowledge graphs. Due
to space limitations, the detailed algorithm is presented in the Appendix. As illustrated in Algorithm 1,
the process begins by identifying entities Eq
twithin the query via entity linking. Subsequently, based
on entities Eq
t, we retrieve related entities from the knowledge graph. Specifically, we first filter
relations related to the query based on semantic similarity. Based on these relevant relations, we
obtain the local neighbor entities El
tthrough direct KG querying and global predicted entities Eg
tvia
KGE model prediction. Finally, these retrieved entities are incorporated into the query to enrich its
context.
Algorithm 1: Related Entity Search.
Input: Query qt.
Output: Related entities Et
El
t, Eg
t← ∅,∅
// Get the entities in the query through entity linking
Eq
t←EL(qt)
// Get the relevant neighboring entities
foreq
tinEq
tdo
// Query relations around eq
tin KG
Req
t←KG(eq
t)
// Obtain the most related tail entity from KG.
req
t←maxr∈Req
t(cos([eq
t;r], qt))
el
t←KG(eq
t, req
t)
El
t←El
t∪ {el
t}
// Obtain the most related tail entity based on KG embedding.
Req
t←Topkr∈Req
t(cos([eq
t;r], qt))
eg
t←maxr∈Req
t(KGE (eq
t, r))
Eg
t←Eg
t∪ {eg
t}
Et←Eq
t∪El
t∪Eg
t
B Experimental Supplement
B.1 Hyperparameter settings
As discussed in Section 4.2, our adaptive retrieval employs an iterative process with a dynamically
increasing threshold θtto control answer output. θtis determined by θ0and a constant c. Due to
space limitations, we report the hyperparameter θ0in Table 6 in the Appendix.
Table 6: θfor Different Models Across Datasets
Dataset GLM4-9b Qwen2.5-32b GPT4o-mini
HotpotQA 10 1 13
2WikiMultiHopQA 2 0.2 2
PopQA 0.1 0.02 0.01
B.2 Impact of maximum reference number
To assess the influence of the maximum number of references on model performance, we vary the
number of documents provided to the QA model during the first turn and report results across three
datasets in Table 7. Performance generally improves as the number of references increases, but the
15

Table 7: Performance with Varied References
GLM4-9b
Method HotPot 2Wiki Pop
EM F1 EM F1 EM F1
1 0.196 0.296 0.256 0.329 0.320 0.376
3 0.215 0.328 0.231 0.320 0.325 0.389
5 0.227 0.340 0.260 0.358 0.331 0.391
7 0.217 0.333 0.216 0.309 0.333 0.396
9 0.225 0.337 0.217 0.311 0.332 0.396
0.200.250.300.350.40Score
GLM4-9bHotPotQA
0.200.250.300.350.40
2WikiMultiHopQA
0.200.250.300.350.40
PopQA
0.200.250.300.350.40Score
Qwen2.5-32b
0.200.250.300.350.40
0.200.250.300.350.40
=0
 =1
 =2
Maximum /glyph1197umber of Iterations0.3250.3500.3750.4000.4250.450Score
GPT4o-mini
=0
 =1
 =2
Maximum /glyph1197umber of Iterations0.3250.3500.3750.4000.4250.450
=0
 =1
 =2
Maximum /glyph1197umber of Iterations0.3250.3500.3750.4000.4250.450
EM
F1
Figure 5: Performance of Know³-RAG with Varying Maximum Number of Iterations
gains reduce or even decline beyond a certain point. This suggests that although reference filtering is
applied, introducing too many documents can still lead to irrelevant or redundant content, ultimately
hindering answer quality.
B.3 Impact of the Maximum Number of Iterations
To explore the impact of the maximum number of turns on model performance, Figure 5 shows the
performance with the varying maximum number of iterations. We notice that all models demonstrate
higher EM and F1 scores at T= 2compared to T= 0, indicating that iterative retrieval positively
impacts answer quality. We also find that the growth from T= 0toT= 1is significantly greater
than from T= 1toT= 2, indicating that most questions can be effectively addressed after the first
retrieval turn. This finding is consistent with Table 5 and further demonstrates the effectiveness of
our knowledge-aware adaptive retrieval.
C Limitation
While our framework demonstrates promising results, several limitations deserve discussion. First, the
integration of knowledge graphs introduces non-negligible noise. For efficiency, we adopt relatively
simple methods for triple extraction and entity linking, which may introduce noise into the retrieved
16

entity knowledge. This noise, in turn, reduces the relative contribution of external knowledge in the
overall performance. Future work could incorporate more accurate extraction and linking techniques
to enhance knowledge quality [ 45,2]. Second, the source of the retrieved text can be further expanded.
Our current framework primarily relies on knowledge graphs due to their high reliability. However,
knowledge graphs are often updated with some delay, which may hinder the timeliness of retrieved
facts. To address this, incorporating search engine-based retrieval as a supplementary source presents
a viable direction [ 25,9]. Nevertheless, such sources often include irrelevant information and require
advanced denoising, which we leave for future exploration.
D Prompts
In this section, we show the prompt we used for the question answer, reference generation, query
generation, triple extraction, and relevance check.
Given the following question, references (may or may not be available), explain your reasoning step -by-
step based on the references and then provide your best possible answer. If there is no reference or you 
find the reference irrelevant, please provide an answer based on your knowledge.
Reference: {Reference}
Question: {Question}
Your response should end with "The answer is [ your_answer_text ]", where the [ your_answer_text ] should 
be yes, no, or a few words directly answering the question. \n Let \'s think step by step.' 'Prompt for Question -Answer 
Figure 6: Prompt for Question Answer
I have a list of open -ended questions, and I \'d like you to write a reference paragraph for each question. 
These paragraphs should provide sufficient background, key concepts, or context to guide the next person 
in answering the question effectively. You do not need to provide an answer directly, just enough 
information to help the next person frame their answer concisely and accurately.
To make your reference passages more accurate, I \'m going to provide you with some entities inside the 
question that you can refer to them, but they \'re not necessarily accurate.
Question: {Question}
Related Entities:{Entities}
You should just output "[ reference_paragraph ]", where the [ reference_paragraph ] is the reference you 
write.Prompt for Reference Generation 
Figure 7: Prompt for Reference Generation
17

You will be given references, a question, and an answer. The answer may be incomplete or incorrect. 
Identify the most critical missing or incorrect information in the references and the answer. Formulate one 
most important new question that will most effectively help retrieve the necessary information to answer 
the original question.
Reference :{Reference}
Question: {Question}
Answer: {Answer}
Please directly output the new question:Prompt for Query GenerationFigure 8: Prompt for Query Generation
I have a task to extract relationships between entities from a given text. 
You will be provided with a list of possible entities as a reference. Your task is to extract triples 
representing relationships between entities in the text. Each triple should include a subject, predicate, and 
object directly from the text. The entities in the triples may or may not match the provided reference list, 
but use the list to guide your extraction process. If no meaningful relationships are found, return None.
Instructions:
- Extract the subject, predicate, and object exactly as they appear in the text.
- If no valid relationships are found, return None.
- Output only the extracted triples in the format: [list of triples].
Text: Albert Einstein was born in Ulm, Germany in 1879.
Entities: Albert Einstein, Ulm, Germany
[{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, {"subject": "Albert Einstein", 
"predicate": "was born in", "object": "Germany"}]
Text: She is a member of the organization.
Entities: the organization
None
Text: {Text}
Entities: {Entities}Prompt for Triple Extraction 
Figure 9: Prompt for Triple Extraction
18

I need your help determining the reliability of a passage in the context of its ability to answer a specific 
question. I might provide some entities to help you better understand the problem. Here are the key 
considerations:
1. Passage Relevance: Check if the passage provides information relevant to answering the question. Even 
if the passage does not directly mention the entities, it should address the key concepts or ideas related to 
the question. If the passage does not contribute meaningfully to answering the question, it may be 
unreliable.
2. Entity Accuracy: The entities provided are from the question and may not appear in the passage. These 
entities are meant to help you understand the context of the question. If the passage conflicts with the 
entities provided (e.g., incorrect descriptions or relationships), this could affect its reliability.
3. Overall Reliability: Based on the relevance of the passage to the question and the accuracy of the entities, 
assess whether the passage is reliable for answering the question. If there are doubts or inconsistencies, 
provide a clear explanation.
Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and 
end your response with: "The reliability of the passage is [yes or no]."
Question: What is the nutritional value of an apple?
Entities: Apple: A fruit known for its nutritional benefits, such as fiber and vitamins.
Passage: An apple is a nutritious fruit rich in fiber, vitamins, and antioxidants.
The passage provides relevant information about the nutritional value of an apple, aligning with the 
question. The entity "Apple" refers to the fruit, which matches the context of the question. The reliability 
of the passage is yes.
Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and 
end your response with: "The reliability of the passage is [yes or no]."
Question: What is the CEO of Apple Inc.?
Entities: 1. Apple Inc.: A technology company, known for products like the iPhone and Mac computers.
Passage: Apples are widely consumed fruits that come in different varieties, including Granny Smith and 
Red Delicious.
The passage discusses apples as a fruit, which is unrelated to the question about the CEO of Apple Inc. The 
passage does not address the company or its leadership. The reliability of the passage is no.'
Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and 
end your response with: "The reliability of the passage is [yes or no]."
Question: {Question}
'Entities: {Entities}'Prompt for Relevance Check Figure 10: Prompt for Relevance Check
19
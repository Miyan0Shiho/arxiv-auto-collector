# NeocorRAG: Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains

**Authors**: Shiyao Peng, Qianhe Zheng, Zhuodi Hao, Zichen Tang, Rongjin Li, Qing Huang, Jiayu Huang, Jiacheng Liu, Yifan Zhu, Haihong E

**Published**: 2026-04-30 13:37:01

**PDF URL**: [https://arxiv.org/pdf/2604.27852v1](https://arxiv.org/pdf/2604.27852v1)

## Abstract
Although precise recall is a core objective in Retrieval-Augmented Generation (RAG), a critical oversight persists in the field: improvements in retrieval performance do not consistently translate to commensurate gains in downstream reasoning. To diagnose this gap, we propose the Recall Conversion Rate (RCR), a novel evaluation metric to quantify the contribution of retrieval to reasoning accuracy. Our quantitative analysis of mainstream RAG methods reveals that as Recall@5 improves, the RCR exhibits a near-linear decay. We identify the neglect of retrieval quality in these methods as the underlying cause. In contrast, approaches that focus solely on quality optimization often suffer from inferior recall performance. Both categories lack a comprehensive understanding of retrieval quality optimization, resulting in a trade-off dilemma. To address these challenges, we propose comprehensive retrieval quality optimization criteria and introduce the NeocorRAG framework. This framework achieves holistic retrieval quality optimization by systematically mining and utilizing Evidence Chains. Specifically, NeocorRAG first employs an innovative activated search algorithm to obtain a refined candidate space. Then it ensures precise evidence chain generation through constrained decoding. Finally, the retrieved set of evidence chains guides the retrieval optimization process. Evaluated on benchmarks including HotpotQA, 2WikiMultiHopQA, MuSiQue, and NQ, NeocorRAG achieves SOTA performance on both 3B and 70B parameter models, while consuming less than 20% of tokens used by comparable methods. This study presents an efficient, training-free paradigm for RAG enhancement that effectively optimizes retrieval quality while maintaining high recall. Our code is released at https://github.com/BUPT-Reasoning-Lab/NeocorRAG.

## Full Text


<!-- PDF content starts -->

NeocorRAG: Less Irrelevant Information, More Explicit Evidence,
and More Effective Recall via Evidence Chains
Shiyao Peng
psy200104@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaQianhe Zheng
zhengqianhe@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaZhuodi Hao
jodiehao@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaZichen Tang
TangZichen@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, China
Rongjin Li
lirongjin@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaQing Huang
huangqing@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaJiayu Huang
jyxhuangjiayu@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaJiacheng Liu
Liujiacheng@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, China
Yifan Zhu
yifan_zhu@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, ChinaHaihong E∗
ehaihong@bupt.edu.cn
Beijing University of Posts
and Telecommunications
Beijing, China
Abstract
Although precise recall is a core objective in Retrieval-Augmented
Generation (RAG), a critical oversight persists in the field: improve-
ments in retrieval performance do not consistently translate to
commensurate gains in downstream reasoning. To diagnose this
gap, we propose theRecall Conversion Rate (RCR), a novel
evaluation metric to quantify the contribution of retrieval to rea-
soning accuracy. Our quantitative analysis of mainstream RAG
methods reveals that as Recall@5 improves, the RCR exhibits a
near-linear decay. We identify the neglect of retrieval quality in
these methods as the underlying cause. In contrast, approaches
that focus solely on quality optimization often suffer from inferior
recall performance. Both categories lack a comprehensive under-
standing of retrieval quality optimization, resulting in a trade-off
dilemma. To address these challenges, we proposecomprehensive
retrieval quality optimization criteriaand introduce theNeo-
corRAGframework. This framework achieves holistic retrieval
quality optimization by systematically mining and utilizingEv-
idence Chains. Specifically, NeocorRAG first employs an inno-
vative activated search algorithm to obtain a refined candidate
space. Then it ensures precise evidence chain generation through
constrained decoding. Finally, the retrieved set of evidence chains
guides the retrieval optimization process. Evaluated on benchmarks
including HotpotQA, 2WikiMultiHopQA, MuSiQue, and NQ, Neo-
corRAG achievesSOTAperformance on both 3B and 70B param-
eter models, while consuming less than 20% of tokens used by
∗Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW ’26, Dubai, United Arab Emirates.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792093comparable methods. This study presents an efficient, training-
free paradigm for RAG enhancement that effectively optimizes
retrieval quality while maintaining high recall. Our code is released
at https://github.com/BUPT-Reasoning-Lab/NeocorRAG.
CCS Concepts
•Information systems →Information retrieval;Question an-
swering;Document filtering.
Keywords
Retrieval-Augmented Generation, Evidence Chains Mining, Re-
trieval Quality Optimization, Reasoning Performance
ACM Reference Format:
Shiyao Peng, Qianhe Zheng, Zhuodi Hao, Zichen Tang, Rongjin Li, Qing
Huang, Jiayu Huang, Jiacheng Liu, Yifan Zhu, and Haihong E. 2026. Neo-
corRAG: Less Irrelevant Information, More Explicit Evidence, and More
Effective Recall via Evidence Chains. InProceedings of the ACM Web Confer-
ence 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates.ACM,
New York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792093
1 Introduction
Retrieval-Augmented Generation (RAG) [ 1,7,11] has emerged as
a standard, non-parametric approach to provide Large Language
Models (LLMs) with up-to-date, factual information, thus mitigating
hallucinations [22, 25, 48, 49].
Standard RAG systems [ 4,18,31,32], which rely on simple vector-
based retrieval mechanisms, have demonstrated excellent recall per-
formance in basic Question Answering (QA) tasks [ 21]. Structure-
enhanced RAG methods, such as HippoRAG [ 13] and RAPTOR [ 32],
have further achieved impressive results on more challenging QA
benchmarks [ 17,35,46]. Among them, HippoRAG2 [ 14] stands
out by reaching a passage Recall@5 of 96.3% on the HotpotQAarXiv:2604.27852v1  [cs.IR]  30 Apr 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
7075 8085 90957688
86
84
82
80
78GTR
（T5-
base）
BM25
Contriever
HippoRAG
RAPTOR
HippoRAG2Simple Model
Structure-
augmented RAG
Trend Line
（Slope： -0.272 ）Recall Conversion Rate@5
Reacll@5What is one of the stars of The Newcomers known for ？
 Query：
IrrelevantThe Newcomers……starring Christopher McCoy, 
Kate Bosworth, Paul Dano and Chris Evans…
Chris Evans…Evans is known for his superhero 
roles as the Marvel…
Anna Camp\nAnna Ragsdale Camp (born Sep…
Recall@3=1
The Newcomers...starring Christopher McCoy, 
Kate Bosworth, Paul Dano and Chris Evans…
…starring…Paul Dano…The score was admired…
Chris Evans…Evans is known for his superhero 
roles as the Marvel…
Recall@3=1
Wrong
TrueInterference
Irrelevant
Figure 1: The inadequacy of recall metrics in capturing re-
trieval quality misleads RAG optimization and hinders rea-
soning performance improvement. (Left) The increase in
theRecall@5 score of retrieval methods is accompanied by
a decrease in the Recall Conversion Rate (RCR). (Right) For
retrieved texts both with a Recall@3=1 score, the upper exam-
ple contains misleading information for the model, while the
lower one contains only completely irrelevant information,
leading to different QA results. This indicates that Recall@n
can only measure “whether relevant documents are hit” but
fails to accurately assess retrieval quality.
benchmark [ 46], seemingly proving that RAG systems can already
retrieve relevant content accurately.
However, its corresponding QA F1 score of only 75.5% presented
a stark contrast. This discrepancy suggests that existing retrieval
metrics fail to capture how retrieved content is actually utilized
during downstream reasoning. Motivated by this observation, we
introduce theRecall Conversion Rate (RCR), a novel evaluation
metric designed to quantify the contribution of retrieved content to
reasoning. Based on this evaluation metric, we quantitatively eval-
uated the main RAG methods. Our analysis reveals that improve-
ments in retrieval recall do not correlate with gains in reasoning
accuracy. As shown in the left panel of Figure 1, the RCR exhibits a
near-linear decay as Recall@5 increases, exposing a “high recall,
low conversion” phenomenon. As illustrated in Figure 1, even with
optimal recall scores, the retrieved content can still contain noisy
text that severely interferes with the model’s reasoning.
This observation exposes a fundamental limitation of commonly
used retrieval metrics, such as Recall@n [3] and NDCG [38], which
prioritize surface-level matching of relevant snippets while over-
looking whether the retrieved content truly provides faithful, non-
misleading evidence that supports downstream reasoning. Con-
sequently, improvements in retrieval metrics often fail to yield
corresponding gains in reasoning performance.
Reasoning-enhanced methods have recognized the importance
of retrieval quality for reasoning performance, yet they fall into a
“trade-off” dilemma.Evidence-based retrieval methodsattempt
to suppress noise by extracting a small set of explicit evidence: they
either identify title-anchored evidence via document triples [8] or
directly synthesize evidence sentences from external knowledge
sources via generative retrieval [ 23]. While effective at filtering
interference, these approaches often bias the retained context to-
ward shallow, explicitly expressed facts and fail to surface deeper
or latent associations within the retrieved texts, which cause somecorrectly retrieved information to be discarded.Dynamic retrieval
methods, such as CoRAG [ 39], aim to expand evidence coverage
through multi-step retrieval chains but rely on learned retrieval
policies that are constrained by the training data, leading to limited
generalization and unstable recall behavior across model scales.
In fact, both types of methods lack a systematic analysis of the
intrinsic components of retrieval quality. From the perspective of
how information supports reasoning, retrieved texts can be clas-
sified into four categories:Irrelevant Information,Interfering
Information,Explicit Evidence, andHidden Evidence. Irrele-
vant information has a negligible impact on the reasoning process;
interfering information can mislead the model’s judgment; whereas
explicit and hidden evidence are the critical pillars for achieving
correct reasoning.
Therefore, we propose more systematic criteria for retrieval
quality optimization:
(1)More Comprehensive:Ensuring broad coverage of the re-
trieved information.
(2)Less Noise Information:Suppressing the inclusion of inter-
fering noise.
(3)Evidence Visualization:Enhancing the ability to identify and
associate hidden evidence.
Using these criteria, we perform a more systematic analysis of
the deficiencies in the two main categories of existing methods
in Figure 2. Neither of these approaches can fully satisfy all the
optimization standards. To fully address the challenge of optimiz-
ing retrieval quality, we proposeNeocorRAG, a framework that
enhances retrieval quality by deeply mining evidence chains within
retrieved texts. Evidence chains mining is realized through two
key strategies: (1) we design a novelactivated path search algo-
rithmthat starts from the key entities of the question to identify a
candidate evidence space within the subgraph of the texts, thereby
thoroughly mining potential evidence chains while substantially ex-
cluding irrelevant information; and (2) we then utilizeconstrained
decodingto precisely search for faithful evidence chains within
the candidate set. Relying on the resulting set of evidence chains,
our approach removes interfering noise and explicates hidden asso-
ciative evidence, all while ensuring no loss in recall performance.
We validate NeocorRAG on four widely used QA benchmarks,
including NQ [ 21], MuSiQue [ 35], 2WikiMultiHopQA [ 17], and
HotpotQA [ 46]. In our experiments, using lightweight models from
the Llama 3 [ 12] and Qwen2.5 [ 28] series for constrained decoding,
NeocorRAG demonstrates consistent state-of-the-art (SOTA) per-
formance across all benchmarks in both 3B (Llama-3.2-3B-Instruct)
and 70B (Llama-3.3-70B-Instruct) QA settings, while consuming
less than 20% tokens used by comparable methods. In identical QA
prompt settings, compared to baseline before retrieval quality opti-
mization, the F1 score improves by up to9.4 percentage points
and the RCR increases by up to9.6 percentage points.
As a universalplug-and-playframework, NeocorRAG offers
a novel perspective on retrieval optimization for RAG systems. In
summary, our core contributions are threefold:
(1)We reveal the limitations of existing retrieval optimization meth-
ods in enhancing retrieval quality, propose more comprehensive
criteria for retrieval quality optimization, and introduce the Neo-
corRAG framework to effectively address these issues through

NeocorRAG : Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
      d1->The Newcomers  (film) ... starring      
      Christopher McCoy , Kate Bosworth , Paul
Dano  and Chris Evans . Christopher McCoy ...Query：What is one of the stars  of The
Newcomers  known for ？
      d3->Chris Evans  …Evans  is known for  his
superhero roles as the Marvel …      d2->…starring …Paul Dano …The score  was
widely admired …
Structural Enhancement Methods After
48 20 10More Comprehensive
Less Noise Information
Evidence Visualization
Irrelevant InformationComposition before  Retrieval  
48 2010
Interfering Information22
100%
Explicit Evidence Information
Hidden Evidence InformationReasoning Enhancement Methods - Evidence-based RetrievalAfter
28 67More Comprehensive
Less Noise Information
Evidence Visualization
After
33 14 8More Comprehensive
Less Noise Information
Evidence Visualization
Ours (NeocorRAG)After
28 69More Comprehensive
Less Noise Information
Evidence VisualizationReasoning Enhancement Methods - Dynamic Retrieval
6
10
18
Figure 2: Comparison of existing methods based on proposed retrieval quality optimization criteria, evaluating two method
categories against three standards (More Comprehensive,Less Noise Information,Evidence Visualization). Structure Enhancement
Methods (top) excel in comprehensiveness but introduces significant noise and lacks evidence visualization. Reasoning
Enhancement Methods (middle), which include evidence-based retrieval and dynamic retrieval, show varied improvements
in noise suppression and evidence visualization but are deficient in comprehensiveness. In contrast, NeocorRAG (bottom)
simultaneously satisfies all three criteria, achieving a balanced and superior optimization of retrieval quality.
evidence-chain-driven optimization, providing a clear direction
for RAG system enhancement.
(2)We design a novel deep evidence mining method that optimizes
retrieval quality by thoroughly mining associative evidence
while controlling irrelevant information.
(3)NeocorRAG achieves SOTA performance on four classic QA
benchmarks, being lightweight, efficient, training-free, and plug-
and-play.
2 Related Work
RAG integrates retrieval modules with language models to im-
prove factual accuracy and reduce hallucinations, establishing itself
as a core paradigm for knowledge-intensive tasks [ 48]. Existing
approaches follow two main optimization directions:structural
enhancement, incorporating structural information to improve
recall, andreasoning enhancement, aligning retrieval with the
model’s reasoning needs.
The first category focuses onStructural Enhancement[ 6,16,
30,33,40,45]. These methods aim to address the lack of mean-
ing construction [ 20] and associative linking [ 34] in traditional
RAG recall by leveraging rich structural information. For instance,
GraphRAG [ 15] manages documents and entities with relations
through graph community detection algorithms; HippoRAG [ 13]
simulates the mechanism of hippocampal memory to perform asso-
ciative retrieval; and RAPTOR [ 32] builds a hierarchical summary
through recursive embeddings, using a Gaussian Mixture Model
(GMM) to detect clusters of documents for summarization. These
methods excel in comprehensiveness and achieve high recall. How-
ever, they often lack effective noise reduction and do not support
evidence visualization, limiting their ability to convert high recall
into improved reasoning performance.
The second category focuses onReasoning Enhancement[ 2,
5,19,27]. These methods concentrate on optimizing retrieval basedon the model’s reasoning needs. For example, IRCoT [ 36] uses
Chain-of-Thought prompting to guide alternating retrieval and
reasoning steps; CoRAG [ 39] trains a model for dynamic retrieval
queries by designing a retrieval chain data generation strategy;
Trace [ 8] iteratively retrieves reasoning chains from logically con-
nected knowledge triples to answer questions; and RetroLLM [ 23]
proposes a unified retrieval-generation framework that enables the
LLM to directly generate fine-grained evidence from the knowl-
edge base. These methods improve noise reduction and evidence
visualization to some extent. However, by focusing on localized
reasoning paths, they often compromise completeness and incur
significant computational overhead.
In general, existing methods exhibit complementary limitations
in the optimization of retrieval quality, which continue to hinder
further improvements in downstream reasoning performance. More
broadly, the challenge of organizing and utilizing retrieved informa-
tion to better support complex reasoning has long been recognized
in information retrieval research, including classical digital library
and knowledge organization systems such as Greenstone [ 41] and
KEA [ 42]. These observations motivate the need for more system-
atic criteria for optimizing retrieval quality and form the basis for
the problem setting explored in this work.
3 Methodology
In this section, we introduce NeocorRAG, a novel framework to
enhance model reasoning in QA tasks by addressing the dual needs
of maintaining high recall and optimizing retrieval quality. First,
we define the RCR to quantify the contribution of retrieved texts
to reasoning and use this evaluation metric to guide retrieval opti-
mization (Section 3.1). Second, we propose a scheme to achieve this
optimization by thoroughly mining deep evidence chains within
the retrieved texts (Section 3.2). Finally, we detail the implementa-
tion of evidence chains mining and retrieval optimization, which

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
Query：What is one of the  stars  of 
The Newcomers  known for ？
      d4->Chris Evans  … is known for his ...      d1->The Newcomers  (film) ... starring ...   
      d2->... It stars  ... Kate Bosworth
      d3->…starring …Paul Dano …The score...
      d5->Anna Camp,Anna Ragsdale Camp …
Documents       activated node
      candidate node
      evidence edge      query node
      next step
      Turples
"The Newcomers"
"Christopher McCoy"
"The Newcomers"
"Chris Evans"..."stars"
"stars"
Graph After Graph Before
LLM
stars Chris  Evans known for the
Newcomers stars for < M > Chris  Evans knownJohny  StormSteve Roger（a）Candidate Evidence Space Identification
The Newcomers - stars -> Chris Evans
- known for role -> Steve Rogers /
Captain America - part of -> Marvel
Cinematic Universe
The Newcomers - stars -> Chris Evans
- known for role -> Johnny Storm /
Human Torch - part of -> Fantastic
Four
Evidence Chains      d4->Chris Evans  … is known for his ...      d1->The Newcomers  (film) ... starring ...   
LLM
True and CompleteAnswer
Chris Evans is known
for his super-hero
roles as the Marvel
Comics
Tree
（b）Precise Evidence Chains Generation （c）Retrieved Context Optimization
Figure 3: Overall framework of NeocorRAG. Our approach enhances retrieval quality through three sequential stages. (a)
Candidate Evidence Space Identification: An activation-based search algorithm dynamically explores paths in the document
subgraph to identify candidate evidence chains (orange paths), scoring them with a comprehensive metric. (b) Precise Evidence
Chains Generation: The candidate chains are structured into a prefix tree. An LLM then performs a fine-grained, constrained
decoding within this space to generate precise evidence chains. (c) Retrieved Context Optimization: The generated evidence
chains are used to filter the retrieved documents, removing noise (e.g., 𝑑3,𝑑5) and explicitly presenting the most relevant
documents (𝑑 1,𝑑4) along with the chains to the LLM.
includes the identification of the candidate evidence space, the con-
strained generation of evidence chains, and the refinement guided
by evidence chains of the retrieved contexts (Section 3.3).
3.1 Necessity of RCR for Retrieval Optimization
Addressing the critical shortcoming of existing retrieval metrics,
which overlook the actual contribution of retrieved texts to reason-
ing performance, we pioneer theRecall Conversion Rate (RCR)
to quantify this contribution. Its core premise is that the value of
retrieval lies not only in “whether relevant information is retrieved”,
but also, more importantly, in “whether the retrieved information
can be effectively converted into correct answers”. To unify existing
evaluation metrics, RCR is defined as the ratio of the F1 score of
the generated answer to the retrieval coverage, with the formula as
follows:
RCR(Q,D 𝑔,𝐴,𝐴∗)= 
F1(𝐴,𝐴∗)
Recall(D 𝑔,Q),ifRecall(D 𝑔,Q)>0
0,otherwise(1)
Here,Qdenotes the set of queries, and 𝐴and𝐴∗represent the
generated answer and the ground-truth answer for each query,
respectively. F1(𝐴,𝐴∗)measures the accuracy of the answer, while
Recall(D 𝑔,Q)measures the retrieval coverage, which can be in-
stantiated as Recall@𝑛 . Unless otherwise specified, we instantiate
Recall asRecall@5 in all experiments, and denote the resulting
metric as RCR@5 . Specifically, for each query 𝑞∈Q , letD𝑔(𝑞)de-
note its supporting passages; Recall @𝑛is computed as the average
fraction ofD 𝑔(𝑞)that appears in the top-𝑛retrieved passages.
Based on the strong recall baseline of HippoRAG2, our approach
improves retrieval efficacy by specifically targeting an improvement
in RCR while minimizing degradation in recall performance. Thisallows our method to effectively navigate the fundamental trade-off
between high recall and high utility in retrieved contexts.
3.2 Evidence Chains Mining
To optimize retrieval quality without sacrificing recall, it is essential
to thoroughly mine the evidence information within the retrieved
texts that supports reasoning. We argue that the reasoning value
of the retrieved texts is not uniformly distributed but concentrated
in chains that connect entities, relationships, and facts related to
the query. Therefore, we define such structured sequences of key
information in retrieved texts that support reasoning as “evidence
chain” and have implemented a recall optimization algorithm based
on these evidence chains.
The crux of this optimization lies in a faithful method for mining
these evidence chains. We design a high-efficiency mining strategy
characterized by “broad candidate generation followed by strict fil-
tering”. A novel activation-based path search algorithm ensures effi-
cient exploration of the potential evidence space, while constrained
decoding leverages the global representation power of generative
models to precisely identify valid evidence chains within this space.
The specific framework, illustrated in Figure 3, consists of three
main stages:
(1)Candidate Evidence Space Identification: Lock the candi-
date evidence space within the subgraph constructed from the
retrieved documents.
(2)Precise Evidence Chains Generation: Distill faithful and
valid evidence chains from the candidate space.
(3)Retrieved Context Optimization: Filter out noisy texts and
explicitly surface hidden relational evidence based on the set of
evidence chains.

NeocorRAG : Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
3.3 Evidence-Aware Retrieval Optimization
3.3.1 Candidate Evidence Chains Search.This stage aims to mine
potential evidence chains by conducting node identification and
path exploration over the subgraph 𝐺=(𝑉,𝐸) constructed from
retrieved texts. The process progresses from initializing relevant
nodes to searching and scoring candidate evidence chains, compris-
ing the following steps:
Step 1: Initial Node Identification.Given a query 𝑞, we extract a
set of keywords entities𝐸 𝑞={𝑒 1,𝑒2,...,𝑒 𝑚}using an LLM. Based
on these entities, we identify the relevant initial nodes by selecting
those whose semantic embeddings are sufficiently close to the
aggregated embeddings of𝐸 𝑞. Formally:
𝑉init=
𝑣∈𝑉𝜙sim(Emb(𝑣),Emb(𝐸 𝑞))≥𝜏 node	
(2)
where Emb(·) denotes a pretrained embedding model, 𝜙sim(·,·) is
the semantic similarity function, and 𝜏node∈[0,1]is the threshold
for identifying query-relevant nodes.
Step 2: Dynamic Path Exploration.To efficiently discover poten-
tial evidence chains, we employ a dynamic exploration algorithm
based on Depth-First Search (DFS). Starting from the initial node
set𝑉init, the algorithm recursively explores the paths under a maxi-
mum hop constraint 𝐾. At each exploration step, for a current node
𝑣𝑖and its neighbor 𝑛𝑗connected through the relation 𝑟, we form a
triple𝑡=(𝑣 𝑖,𝑟,𝑛 𝑗)and compute its local semantic similarity to the
query:
𝑆(𝑣𝑖,𝑛𝑗)=cos(Emb(𝑡),Emb(𝑞)) (3)
The search continues to expand only if the triple score satisfies the
edge-level threshold condition. Specifically, low-confidence neigh-
bors are pruned and only those satisfying the following constraint
are included in the search frontier:
𝑛𝑗∈Expanded Nodes⇐⇒𝑆(𝑣 𝑖,𝑛𝑗)≥𝜏 edge (4)
Step 3: Activated Comprehensive Path Scoring.Local similarity
scores alone are insufficient to evaluate the overall quality of an evi-
dence chain. We therefore design a comprehensive scoring function
that incorporates both penalties and rewards. Given a candidate
path𝑃=[𝑡 1,𝑡2,...,𝑡|𝑃|], the final score is calculated as:
score(𝑃)=𝛼ReLU(|𝑃|−𝐿)
|        {z        }
Length Penalty·cos 
1
|𝑃||𝑃|∑︁
𝑘=1Emb(𝑡 𝑘),Emb(𝑞)!
|                                   {z                                   }
Overall Semantic Relevance·
𝛽Í|𝑃|
𝑘=1I(cos(Emb(𝑡 𝑘),Emb(𝑞))≥𝜏boost)
|                                    {z                                    }
High-Confidence Activation Reward(5)
The comprehensive scoring function balances three core com-
ponents. Thelength penaltyterm 𝛼ReLU(|𝑃|−𝐿)applies a penalty
to paths whose length exceeds the expected maximum 𝐿, where
𝛼∈( 0,1)controls the penalty strength and ReLU(𝑥)=max( 0,𝑥)
ensures that no penalty is applied when the length is within the
limit. Theoverall semantic relevanceis measured by the cosine
similarity between the average embedding of all triples in the path
and the query embedding, capturing the global alignment between
the candidate chain and the original question. We set 𝜏boost≥𝜏edge,
so that only triples that pass the exploration threshold can furthercontribute to activation rewards. Thehigh-confidence activation
rewardterm uses an exponential form with base 𝛽>1, rewarding
paths that contain more triples whose semantic similarity to the
query exceeds a predefined confidence threshold 𝜏boost, thus boost-
ing the score of chains enriched with highly relevant information.
Here, I(·)is an indicator function that returns 1 if the condition is
true and 0 otherwise.
3.3.2 Precise Evidence Chains Generation.To generate precise evi-
dence chains from the candidate set obtained in the previous stage,
we organize the candidate chains into a tree structure of prefixes [ 9].
This tree-based organization facilitates constrained decoding for
the LLM, ensuring that the next generated token is always a child
of the current token’s corresponding node in the prefix tree. This
guaranties the precision of the evidence chains generated [24, 44].
Given the input query 𝑞and the set of retrieved documents T, we
design an instruction prompt to guide the LLM’s decoding process
under the constraints imposed by the prefix tree Tprefix. This process
effectively leverages the LLM’s representational power to search
within the candidate chain set, and can be formalized as:
𝑃𝜃(e|𝑞,T)=𝑃 𝜃(e|𝑞)
|  {z  }
Vanilla Decoding
·|e|Ö
𝑖=1𝑃𝜃(𝑒𝑖|𝑞,𝑒 1,...,𝑒 𝑖−1)·𝐶Tprefix(𝑒𝑖|𝑒1,...,𝑒 𝑖−1)
|                                                         {z                                                         }
Evidence-Constrained Decoding(6)
where𝜃represents the LLM parameters,e =[𝑒 1,𝑒2,...,𝑒|e|]de-
notes the generated evidence chain (a sequence of tokens), and
𝐶Tprefix(𝑒𝑖|𝑒1,...,𝑒 𝑖−1)is a constraint function that verifies whether
the partial sequence [𝑒1,...,𝑒 𝑖]is a valid prefix of any candidate
chain in the prefix treeT prefix:
𝐶Tprefix(𝑒𝑖|𝑒1,...,𝑒 𝑖−1)= 
1if∃𝑝∈Psuch that
[𝑒1,...,𝑒 𝑖]is a prefix of𝑝
0otherwise(7)
Here,Pis the set of candidate evidence chains. The constraint
function𝐶Tprefixensures that all generated tokens correspond to
nodes in the prefix tree Tprefix, thereby preventing hallucinations.
This approach not only guarantees the precision of the generated
chains but also bridges the gap between the LLM’s parametric
knowledge and the factual knowledge in the retrieved texts, helping
to uncover hidden information.
In practice, answering a question often relies on multiple evi-
dence chains, and different chains can help the model reason more
comprehensively [ 37]. Therefore, we employ beam search [ 10] to
obtain multiple evidence chains, which serve as supplementary
evidence to the initially retrieved documents.
3.3.3 Evidence-Guided Document Filtering.In the final stage, we
use the set of precise evidence chains to guide the refinement of
the retrieved context. We first partition the initial document set 𝐷
according to its retrieval ranking into a high-confidence set 𝐷topN,
which is preserved by default, and a low-confidence set 𝐷low. The
filtering process is then applied exclusively to 𝐷low. Specifically, we
retain low-confidence documents that contain at least one triple

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
Table 1: Overall performance on classical QA benchmarks. We report passage recall@5 (R5), answer F1 score (F1), and RCR@5
(RCR). Best results are highlighted inbold, * denotes statistical significance (p < 0.05) against comparable baselines Trace.
RetrievalSimple QA Multi-Hop QA Avg.
NQ MuSiQue 2Wiki HotpotQA
R5 F1 RCR R5 F1 RCR R5 F1 RCR R5 F1 RCR R5 F1 RCR
Llama-3.3-70B
Simple Baselines
BM25 56.1 59.0 105.2 43.5 28.8 66.2 65.3 51.2 78.4 74.8 62.3 83.3 59.9 50.3 83.3
Contriever 54.6 58.9 107.9 46.6 31.3 67.2 57.5 41.9 72.9 75.3 63.4 84.2 58.5 48.9 83.1
GTR (T5-base) 63.4 59.9 94.5 49.1 34.6 70.5 67.9 52.8 77.8 73.9 62.8 85.0 63.6 52.5 82.0
Structure-Enhanced RAG
RAPTOR 68.3 50.7 74.2 57.8 28.9 50.0 66.2 52.1 78.7 86.9 69.5 80.0 69.8 50.3 70.7
HippoRAG 44.4 55.3 124.5 53.2 35.1 66.0 90.4 71.8 79.4 77.3 63.5 82.1 66.3 56.4 88.0
HippoRAG2 78.0 63.3 81.2 74.7 48.6 65.1 90.4 71.0 78.5 96.3 75.5 78.4 84.9 64.6 75.8
Reasoning-Enhanced RAG
CoRAG - 54.5 - - 52.9 - - 75.1 - - 75.1 - - 64.4 -
Trace (Llama-3.2-3B) 77.9 48.7 62.6 73.3 33.4 45.6 90.5 59.9 66.1 96.4 68.9 71.5 84.5 52.7 61.4
Trace (Llama-3-8B) 77.9 50.1 64.3 73.3 39.1 53.4 90.5 65.4 72.2 96.4 73.1 75.9 84.5 56.9 66.4
Ours (Llama-3.2-3B)77.9 64.9* 83.3 73.3 50.8* 69.3 90.5 71.6* 79.1 96.4 77.1* 79.9 84.5 66.1 77.9
Ours (Llama-3-8B)77.965.6*84.3 73.352.6*71.7 90.576.1*84.1 96.478.3*81.2 84.5 68.1 80.3
Llama-3.2-3B
HippoRAG2 78.0 49.9 63.9 74.7 30.4 40.7 90.4 46.6 51.5 96.3 55.7 57.8 84.9 45.7 53.5
CoRAG - 37.8 - - 28.1 - - 42.3 - - 56.0 - - 41.1 -
Trace (Llama-3.2-3B) 77.9 47.5 61.0 73.3 24.6 33.5 90.5 40.1 44.3 96.4 61.9 64.2 84.5 43.5 50.8
Trace (Llama-3-8B) 77.9 46.9 60.3 73.3 27.4 37.4 90.5 45.7 50.5 96.4 64.5 66.9 84.5 46.1 53.8
Ours (Llama-3.2-3B)77.9 57.0* 73.1 73.3 33.5* 45.6 90.5 49.9* 55.1 96.4 66.8* 69.3 84.5 51.8 60.8
Ours (Llama-3-8B)77.957.2*73.5 73.334.8*47.4 90.551.4*56.8 96.467.0*69.5 84.5 52.6 61.8
from the union of all chains ( Tunion), forming a supplementary set:
𝐷supplement ={𝑑∈𝐷 low|∃𝑡∈T union,𝑡⊆𝑑} (8)
where𝑡=(𝑣 𝑖,𝑟,𝑣 𝑗)denotes a triple in an evidence chain, and
𝑡⊆𝑑 indicates that the linearized triple string appears as a sub-
string of the document 𝑑. The final refined result is 𝐷refined =
𝐷topN∪𝐷 supplement . This reduction in the total document count,
observed in our experiments, occurs because the set of salvaged
documents ( 𝐷supplement ) is typically much smaller than the origi-
nal low-confidence set ( 𝐷low). This targeted approach preserves
core information while salvaging valuable evidence and eliminating
substantial noise.
4 Experiments
This section details our experimental setup, results, and analyzes,
designed to answer the following four key research questions:
•RQ1:Does NeocorRAG demonstrate superior efficacy as a re-
trieval optimization method over SOTA baselines?
•RQ2:Is NeocorRAG a low-cost, plug-and-play, and highly adapt-
able framework?
•RQ3:Are all constituent modules within the NeocorRAG frame-
work indispensable for its overall performance?
•RQ4:How does the evidence chain mechanism in NeocorRAG
align with and materialize the proposed retrieval quality stan-
dards?4.1 Experimental Setup
Datasets.To comprehensively evaluate the performance of Neo-
corRAG, we select four QA datasets covering varying levels of
complexity:Simple QA:Natural Questions (NQ) [ 21], used to test
single-hop fact retrieval capabilities.Multi-Hop QA:MuSiQue [ 35],
2WikiMultiHopQA [ 17], and HotpotQA [ 46]. These datasets require
the model to integrate multiple pieces of information to infer the
answer, evaluating its associative and reasoning abilities. In partic-
ular, we use the same corpus and datasets as HippoRAG [ 13], with
further details in Appendix A.
Baselines.To fully validate the performance advantages of Neo-
corRAG, we select three categories of retrieval methods as base-
lines:Simple Baselines, including BM25 [ 29], Contriever [ 18], and
GTR [ 26].Structure-Enhanced RAG, including RAPTOR [ 32],
HippoRAG [ 13], and HippoRAG2 [ 14].Reasoning-Enhanced RAG,
including CoRAG [ 39] and Trace [ 8]. To ensure a fair comparison,
all baseline methods were evaluated under a unified experimental
configuration to the greatest extent possible. See Appendix B for
more details.
Evaluation Metrics.Consistent with previous studies [ 13,14,
32], we use theRecall@5andF1 scoreto evaluate retrieval ca-
pacity and reasoning performance, respectively. Additionally, to
measure the efficiency of converting retrieval performance into
reasoning effectiveness, we use theRCR.

NeocorRAG : Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Trace (Llama-3.2-3b)Trace (Llama-3-8b)CoRAGOurs (Llama-3.2-3b)Ours (Llama-3-8b)Input tokens Output tokens
Total tokens
1.85k
1.90k
9.35k
64.26k63.55k
1 1k 1M
Figure 4: Token efficiency comparison (tokens per question
across four QA benchmarks).
Table 2: Ablation study results of NeocorRAG components.
Method EM F1 RCR@5
NeocorRAG 53.7 66.8 69.3
w/ Naive Chain Search 35.8 (↓17.9) 49.5 (↓17.3) 51.3 (↓18.0)
w/o Constrained Decoding 45.1 (↓8.6) 58.7 (↓8.1) 61.0 (↓8.3)
w/o Document Filtering 43.3 (↓10.4) 57.0 (↓9.8) 59.1 (↓10.2)
Implementation Details.NeocorRAG is based on the retrieval
results of HippoRAG2 [ 14], using the same initial retrieval con-
figuration. For our core retrieval quality optimization stage, we
employ lightweight models from the Llama [ 12], Qwen [ 28] series
as the constrained decoding model for mining evidence chains and
the bge-large-en-v1.5 model [ 43,47] as the general-purpose en-
coding model. To rigorously assess the generalizability and model-
scale invariance of retrieval quality enhancements, our evaluation
employs a dual-model framework utilizing both Llama-3.3-70B-
Instruct (large-scale) and Llama-3.2-3B-Instruct (small-scale) as
downstream answer generators. All experiments were conducted
on 8 NVIDIA A40 GPUs (48GB VRAM each). We report the average
results over 5 independent runs with a fixed random seed of 42. See
Appendix B for more details.
4.2 Main Results (RQ1)
As shown in Table 1, NeocorRAG comprehensively outperforms all
baselines on all benchmarks and model sizes (3B and 70B), validating
its superior effectiveness in QA tasks.
Compared to HippoRAG2 [ 14], with identical recall rates and
nearly identical QA prompts, NeocorRAG achieves average im-
provement of the F1 score of 6.9% and 3.5%, and average RCR im-
provements of 8.3% and 4.5% in the 3B and 70B settings, respectively.
On HotpotQA, NeocorRAG achieves an improvement in F1 of up
to 9.4% and an increase in RCR of 9.6%. This performance confirms
that optimizing retrieval quality enhances reasoning performance.
Furthermore, NeocorRAG exceeds the reasoning-enhanced base-
lines. The average performance gap in the F1 score against Trace
[8] increases from 6.5% on 3B models to 11.2% on 70B models. Thisis because its shallow, title-based evidence tracking can erroneously
discard correct documents. This strategy becomes counterproduc-
tive with more powerful foundation models. Similarly, CoRAG [ 39]
underperforms in the 3B model by 11.5% in the F1 score, as its
dynamic retrieval relies too heavily on the models’ capabilities.
The significant performance gaps on both multi-hop and simple
question answering also indicate its poor generalization ability. In
contrast, NeocorRAG follows a more systematic guideline to op-
timize retrieval quality by deeply mining hidden evidence chains.
This approach ensures greater comprehensiveness while reducing
information noise and enabling evidence visualization, leading to
stable and significant performance gains.
4.3 Applicability and Efficiency (RQ2)
Adaptability to Different Base LLMs.To thoroughly validate the
adaptability of NeocorRAG, we conducted experiments using two
mainstream open-source model families for evidence chain mining:
Llama 3 and Qwen2.5, with parameter sizes ranging from 1B to 14B.
Across different model families and scales, NeocorRAG exhibits
stable performance with only minor variations, indicating that its
effectiveness is not tied to a specific backbone model. In particular,
for models below the 3B parameters, increasing the number of can-
didate evidence chains may lead to performance degradation due
to limited model capacity, whereas models beyond 3B consistently
benefit from expanded candidate evidence chains. Additional diag-
nostic results and detailed comparisons between model families and
evidence chain counts are provided in Appendix D. These findings
indicate that after Candidate Evidence Space Identification, the re-
fined candidate space requires only lightweight models for Precise
Evidence Chain Generation, eliminating the need for excessively
powerful models.
Token Cost Analysis.NeocorRAG significantly reduces token
consumption on all benchmarks compared to reasoning-enhanced
baselines. On average, our method uses only 20.1% of the tokens
required by CoRAG [ 39] and less than 2.94% of the tokens used by
Trace [ 8]. This high efficiency is attributed to our effective evidence
chain mining process, as further illustrated in Figure 4.
Summary.In summary, NeocorRAG is adaptable to various
model families and scalable from 1B to 14B parameters, maintain-
ing stable performance without complex adjustments. Its design,
centered on lightweight auxiliary models, ensures both efficiency
and low computational cost. Together, these characteristics answer
RQ2: NeocorRAG is a broadly applicable and highly efficient frame-
work with practical value. We further report runtime latency and
GPU memory overhead on HotpotQA in Appendix E.
4.4 Ablation Study (RQ3)
To answer RQ3, we conducted a series of ablation studies on the
HotpotQA benchmark using the Llama-3.2-3B-Instruct model. The
results presented in Table 2, evaluate the effectiveness of each core
component of NeocorRAG. Our key observations are as follows:
(1)Removal of any single component leads to a significant per-
formance drop, confirming the necessity of each part of our
design.
(2)Replacing our activated search with a naive chain search results
in the most substantial performance degradation, even below

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
405060708090
NQ MuSiQue 2Wiki HotpotQAF1 Score (%)HippoRAG (70B) HippoRAG (3B)
NeocorRAG (70B) NeocorRAG (3B)
+1.26
+5.70+0.51
+4.74+0.49
+1.26+0.86
+4.41
Figure 5: Contribution of evidence chains to the F1 score
when using ground truth (golden) documents. This setup
isolates the benefit of explicating hidden information from
the task of filtering irrelevant documents.
the variant without document filtering. This underscores the
critical importance of a high-quality candidate space for the
subsequent generation of evidence. Additional robustness and
sensitivity diagnostics under variations in candidate evidence
space construction are reported in Appendix F.
(3)Removal of the prefix tree constraints allows the generation
of numerous unfaithful chains from the refined candidate set.
Using these for optimization still causes a significant decline in
performance, highlighting the necessity of constrained decod-
ing.
(4)Forgoing document filtering allows a large amount of noise to
interfere with the model’s reasoning. The resulting performance
drop demonstrates the importance of actively removing this
interference.
4.5 Effectiveness Validation of Evidence Chains
against Retrieval Quality Standards (RQ4)
To evaluate the capability of evidence chains to discover latent
evidence and verify the role of evidence visualization in improving
reasoning performance, we conducted experiments using ground
truth documents with 3B and 70B parameter models. The results in
Figure 5 show that even in a completely noise-free environment,
the evidence chains still improve the average performance of the
3B model by 4.03%. In contrast, the 70B model achieves only a
limited gain of 0.78% in noisy scenarios. These findings demonstrate
that NeocorRAG effectively uncovers hidden evidence. Moreover,
explicit evidence is particularly critical for smaller-scale models
and remains highly valuable in the presence of increased noise.
Furthermore, we conducted an ablation study on the 3B model to
verify the effectiveness of the evidence chain in filtering irrelevant
documents. We introduce two metrics to quantify the filtering
process’s efficacy:Average Total Documents (ATD)andAverage
Invalid Documents (AID). Detailed definitions and formulas for
these metrics are provided in Appendix G.Table 3: Impact of evidence chains on filtering redundant
documents on four benchmarks. We report Average Total
Documents (ATD) and Average Irrelevant Documents (AID)
before and after applying evidence chain filtering. Last row
shows the average change across benchmarks, where ↑and↓
indicate performance increases and decreases, respectively.
Filter Recall@5 ATD AID RCR@5
Before
NQ 77.90 5.00 2.33 63.90
MuSiQue 73.30 5.00 3.12 40.70
2Wiki 90.50 5.00 2.80 51.50
HotpotQA 96.40 5.00 3.08 57.80
Avg. 84.53 5.00 2.83 53.48
After
NQ 68.56 3.76 1.35 73.47
MuSiQue 70.33 3.61 1.81 47.48
2Wiki 89.10 3.44 1.28 56.78
HotpotQA 94.45 3.28 1.13 69.51
Avg. 80.61 3.52 1.39 61.81
Change
Avg.↓3.92↓1.48↓1.44↑8.33
As shown in Table 3, evidence-chain-based filtering reduces ATD
by 1.48 and AID by 1.44 on average. This filtering procedure suc-
cessfully filtered over half of the irrelevant content. Critically, this
filtering process only slightly decreased the recall rate to 80.61%.
This figure is substantially higher than the 46.85% recall achieved
by the comparable Trace [ 8] filtering method. These experimental
results demonstrate that our method effectively enhances retrieval
quality while safeguarding recall performance. This further vali-
dates the effectiveness of our proposed criteria for retrieval quality
optimization.
5 Conclusion
This paper identifies limitations in existing retrieval methods through
the lens of retrieval quality optimization. We establish finer-grained
standards for high-quality retrieval and chart a new direction of im-
provement. To address current challenges, we propose an evidence-
chain-based algorithm that refines retrieved texts to better support
generative models’ reasoning needs. By thoroughly mining evi-
dence chains, we significantly reduce retrieval noise while explicitly
exposing hidden evidence without recall loss, substantially improv-
ing reasoning performance. The SOTA results of NeocorRAG on
four QA benchmarks validate our retrieval optimization approach.
Acknowledgments
This work is supported by the National Natural Science Founda-
tion of China (Grant No. 62473271), and the Fundamental Research
Funds for the Beijing University of Posts and Telecommunications
(Grant No. 2025AI4S03). This work is also supported by the En-
gineering Research Center of Information Networks, Ministry of
Education, China. We would like to thank Yuanze Li for assistance
with figure design and visualization in this paper. We would also like
to thank the anonymous reviewers and area chairs for constructive
discussions and feedback.

NeocorRAG : Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
References
[1]Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. 2024.
A Survey on RAG with LLMs.Procedia Computer Science246 (2024), 3781–3790.
doi:10.1016/j.procs.2024.09.178 28th International Conference on Knowledge
Based and Intelligent information and Engineering Systems (KES 2024).
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
Reflection. InThe Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=hSyW5go0v8
[3]Michael Buckland and Fredric Gey. 1994. The relationship between Recall and
Precision.Journal of the American Society for Information Science45, 1 (1994),
12–19. doi:10.1002/(SICI)1097-4571(199401)45:1<12::AID-ASI2>3.0.CO;2-L
[4]Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao,
Hongming Zhang, and Dong Yu. 2024. Dense X Retrieval: What Retrieval Granu-
larity Should We Use?. InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung
Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA,
15159–15177. doi:10.18653/v1/2024.emnlp-main.845
[5]Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and Stéphane CLIN-
CHANT. 2025. Provence: efficient and robust context pruning for retrieval-
augmented generation. InThe Thirteenth International Conference on Learning
Representations. https://openreview.net/forum?id=TDy5Ih78b4
[6]Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F. Yang, and Anton Tsitsulin.
2024. Don’t Forget to Connect! Improving RAG with Graph-based Reranking.
arXiv:2405.18414 [cs.CL] https://arxiv.org/abs/2405.18414
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models. InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining(Barcelona, Spain)
(KDD ’24). Association for Computing Machinery, New York, NY, USA, 6491–6501.
doi:10.1145/3637528.3671470
[8]Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2024. TRACE the Evidence:
Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented
Generation. InFindings of the Association for Computational Linguistics: EMNLP
2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association
for Computational Linguistics, Miami, Florida, USA, 8472–8494. doi:10.18653/v1/
2024.findings-emnlp.496
[9]Edward Fredkin. 1960. Trie memory.Commun. ACM3, 9 (Sept. 1960), 490–499.
doi:10.1145/367390.367400
[10] Markus Freitag and Yaser Al-Onaizan. 2017. Beam Search Strategies for Neural
Machine Translation. InProceedings of the First Workshop on Neural Machine
Translation, Thang Luong, Alexandra Birch, Graham Neubig, and Andrew Finch
(Eds.). Association for Computational Linguistics, Vancouver, 56–60. doi:10.
18653/v1/W17-3207
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, Amy Yang, Angela Fan, et al .2024. The Llama 3 Herd of Models.
arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
[13] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Lan-
guage Models. InAdvances in Neural Information Processing Systems, A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37.
Curran Associates, Inc., 59532–59569. doi:10.52202/079017-1902
[14] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su.
2025. From RAG to Memory: Non-Parametric Continual Learning for Large
Language Models. InProceedings of the 42nd International Conference on Ma-
chine Learning (Proceedings of Machine Learning Research, Vol. 267), Aarti Singh,
Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan
Maharaj, Kiri Wagstaff, and Jerry Zhu (Eds.). PMLR, 21497–21515. https:
//proceedings.mlr.press/v267/gutierrez25a.html
[15] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi
He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and
Jiliang Tang. 2025. Retrieval-Augmented Generation with Graphs (GraphRAG).
arXiv:2501.00309 [cs.IR] https://arxiv.org/abs/2501.00309
[16] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrieval-Augmented Gener-
ation for Textual Graph Understanding and Question Answering. InAdvances
in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave,
A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates,
Inc., 132876–132907. doi:10.52202/079017-4224
[17] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on ComputationalLinguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International
Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625.
doi:10.18653/v1/2020.coling-main.580
[18] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning.Transactions on Machine Learning
Research(2022). https://openreview.net/forum?id=jKN1pXi7b0
[19] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and
Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City,
Mexico, 7036–7050. doi:10.18653/v1/2024.naacl-long.389
[20] Gary Klein, Brian Moon, and Robert R. Hoffman. 2006. Making Sense of Sense-
making 1: Alternative Perspectives .IEEE Intelligent Systems21, 04 (July 2006),
70–73. doi:10.1109/MIS.2006.75
[21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. doi:10.1162/tacl_a_00276
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.),
Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/
paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[23] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yongkang Wu, Zhonghua Li, Ye Qi, and Zhicheng
Dou. 2025. RetroLLM: Empowering Large Language Models to Retrieve Fine-
grained Evidence within Generation. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang
Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.).
Association for Computational Linguistics, Vienna, Austria, 16754–16779. doi:10.
18653/v1/2025.acl-long.819
[24] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and
Shirui Pan. 2025. Graph-constrained Reasoning: Faithful Reasoning on Knowl-
edge Graphs with Large Language Models. InProceedings of the 42nd Interna-
tional Conference on Machine Learning (Proceedings of Machine Learning Research,
Vol. 267), Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix
Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (Eds.). PMLR, 41540–
41565. https://proceedings.mlr.press/v267/luo25t.html
[25] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, Toronto, Canada, 9802–9822. doi:10.
18653/v1/2023.acl-long.546
[26] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma,
Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large
Dual Encoders Are Generalizable Retrievers. InProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing, Yoav Goldberg, Zornitsa
Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu
Dhabi, United Arab Emirates, 9844–9855. doi:10.18653/v1/2022.emnlp-main.669
[27] Tong Niu, Shafiq Joty, Ye Liu, Caiming Xiong, Yingbo Zhou, and Semih Yavuz.
2024. JudgeRank: Leveraging Large Language Models for Reasoning-Intensive
Reranking. arXiv:2411.00142 [cs.CL] https://arxiv.org/abs/2411.00142
[28] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[29] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu,
Mike Gatford, et al .1995.Okapi at TREC-3. British Library Research and Devel-
opment Department.
[30] Diego Sanmartin. 2024. KG-RAG: Bridging the Gap Between Knowledge and
Creativity. arXiv:2405.12035 [cs.AI] https://arxiv.org/abs/2405.12035
[31] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies,
Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
(Eds.). Association for Computational Linguistics, Seattle, United States, 3715–
3734. doi:10.18653/v1/2022.naacl-main.272
[32] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InThe Twelfth International Conference on Learning
Representations. https://openreview.net/forum?id=GN921JHCRw
[33] Sara Sherif Daoud Saad and Stephanie Silva. 2025. Graph-Enhanced RAG: A
Survey of Methods, Architectures, and Performance.
[34] Wendy A. Suzuki. 2005. Associative Learning and the Hippocampus. doi:10.1037/
e400222005-005
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
doi:10.1162/tacl_a_00475
[36] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational
Linguistics, Toronto, Canada, 10014–10037. doi:10.18653/v1/2023.acl-long.557
[37] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan
Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Im-
proves Chain of Thought Reasoning in Language Models. InThe Eleventh Inter-
national Conference on Learning Representations. https://openreview.net/forum?
id=1PL1NIMMrw
[38] Yining Wang, Liwei Wang, Yuanzhi Li, Di He, and Tie-Yan Liu. 2013. A Theoretical
Analysis of NDCG Type Ranking Measures. InProceedings of the 26th Annual
Conference on Learning Theory (Proceedings of Machine Learning Research, Vol. 30),
Shai Shalev-Shwartz and Ingo Steinwart (Eds.). PMLR, Princeton, NJ, USA, 25–54.
https://proceedings.mlr.press/v30/Wang13.html
[39] Ziting Wang, Haitao Yuan, Wei Dong, Gao Cong, and Feifei Li. 2024. CORAG:
A Cost-Constrained Retrieval Optimization System for Retrieval-Augmented
Generation. arXiv:2411.00744 [cs.DB] https://arxiv.org/abs/2411.00744
[40] Qikai Wei, Huansheng Ning, Chunlong Han, and Jianguo Ding. 2025. A Query-
Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-
Augmented Generation in Large Language Models. arXiv:2507.16826 [cs.IR]
https://arxiv.org/abs/2507.16826
[41] Ian H. Witten, Stefan J. Boddie, David Bainbridge, and Rodger J. McNab. 2000.
Greenstone: a comprehensive open-source digital library software system. InPro-
ceedings of the Fifth ACM Conference on Digital Libraries(San Antonio, Texas, USA)
(DL ’00). Association for Computing Machinery, New York, NY, USA, 113–121.
doi:10.1145/336597.336650
[42] Ian H. Witten, Gordon W. Paynter, Eibe Frank, Carl Gutwin, and Craig G. Nevill-
Manning. 1999. KEA: practical automatic keyphrase extraction. InProceedings
of the Fourth ACM Conference on Digital Libraries(Berkeley, California, USA)
(DL ’99). Association for Computing Machinery, New York, NY, USA, 254–255.
doi:10.1145/313238.313437
[43] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and
Jian-Yun Nie. 2024. C-Pack: Packed Resources For General Chinese Embed-
dings. InProceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval(Washington DC, USA)(SI-
GIR ’24). Association for Computing Machinery, New York, NY, USA, 641–649.
doi:10.1145/3626772.3657878
[44] Xin Xie, Ningyu Zhang, Zhoubo Li, Shumin Deng, Hui Chen, Feiyu Xiong, Mosha
Chen, and Huajun Chen. 2022. From Discrimination to Generation: Knowledge
Graph Completion with Generative Transformer. InCompanion Proceedings of
the Web Conference 2022(Virtual Event, Lyon, France)(WWW ’22). Association
for Computing Machinery, New York, NY, USA, 162–165. doi:10.1145/3487553.
3524238
[45] Derong Xu, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Maolin Wang, Qidong Liu,
Xiangyu Zhao, Yichao Wang, Huifeng Guo, Ruiming Tang, Enhong Chen, and
Tong Xu. 2026. Align-GRAG: Anchor and Rationale Guided Dual Alignment
for Graph Retrieval-Augmented Generation. arXiv:2505.16237 [cs.CL] https:
//arxiv.org/abs/2505.16237
[46] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[47] Peitian Zhang, Zheng Liu, Shitao Xiao, Zhicheng Dou, and Jian-Yun Nie. 2024. A
Multi-Task Embedder For Retrieval Augmented LLMs. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for
Computational Linguistics, Bangkok, Thailand, 3537–3553. doi:10.18653/v1/2024.
acl-long.194[48] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2026. Retrieval-
Augmented Generation for AI-Generated Content: A Survey.Data Science and
Engineering(02 Jan 2026). doi:10.1007/s41019-025-00335-5
[49] Tolga Şakar and Hakan Emekci. 2025. Maximizing RAG efficiency: A comparative
analysis of RAG methods.Natural Language Processing31, 1 (2025), 1–25. doi:10.
1017/nlp.2024.53
A Dataset Details
In our experiments withNeocorRAG, we adopt the same Wikipedia
snapshot and retrieval pipeline asHippoRAG2. Each sampled ques-
tion is matched against the identical document corpus using consis-
tent retrieval parameters, token limits, and filtering heuristics. This
strict alignment ensures that any observed performance differences
can be attributed to the reasoning capability of the model rather
than retrieval variations or corpus drift.
Table 4: Detailed statistics of QA datasets used in our ex-
periments. All samples are drawn from the hard subset to
promote reasoning-intensive evaluation.
Dataset QA Type Samples Used
HotpotQA Multi-hop 1,000 (hard subset)
2WikiMultiHopQA Multi-hop 1,000 (hard subset)
MuSiQue Multi-hop 1,000 (hard subset)
NaturalQuestions (NQ) Simple (Open) 1,000 (nq-rear)
B Implementation Details
B.1 Configuration of Other Baseline Methods
We provide detailed implementation configurations for all baseline
methods, with particular attention toCoRAGandTrace. For a
fair comparison, we standardized theretriever encoder model
across these methods toe5-large-v2. This unification guaranties
a consistent foundation for retrieval input, enabling a more objec-
tive evaluation of the effectiveness of their respectiveReasoning-
Chain-Enhanced RAGframeworks.
ForTrace, similar to ourNeocorRAG, we utilized the retrieval
results provided by theHippoRAGframework as input. This setup
allows for a direct comparison of context optimization performance
starting from the same initial retrieved document set. In contrast,
CoRAGemploys its own retrieval module.
B.2 Configuration of Our Method
Here, we detail the parameter settings used to optimize retrieved
texts inNeocorRAG. Our central objective is to efficiently and
comprehensively extract faithful evidence chains to guide retrieved
text optimization.
During the finalQuestion Answering (QA)stage, we adopted
distinct prompting strategies depending on the scale of the em-
ployed Large Language Model (LLM):
•When using a3B-scale LLM,NeocorRAGemployed thesame
prompt template as Traceto ensure consistency during the
QA phase with comparable model sizes.
•When using a70B-scale LLM,NeocorRAGadopted thesame
QA-stage prompt template as HippoRAG2. This alignment
ensures comparability with a strong RAG baseline using the same
retrieval input and a high-capacity model.

NeocorRAG : Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Table 5: Hyperparameter settings of NeocorRAG.
Stage Parameter NQ MuSiQue 2Wiki HotpotQA
Candidate
Evidence Space
IdentifcationNode Relevance Threshold (𝜏 node) 0.90 0.90 0.90 0.90
Max Path Length (𝐿) 10 10 10 10
Activation Coefficient (𝛽) 1.10 1.10 1.10 1.10
Attenuation Coefficient (𝛼) 0.90 0.90 0.90 0.90
Pruning Score Threshold (𝜏 edge) 0.45 0.45 0.45 0.45
Precise
Evidence Chains
GenerationSearch Strategy Beam Beam Beam Beam
Beam Width (𝑘) 3 5 5 5
Number of GCD Docs 5 5 5 5
Number of Candidate Evidence Chains 60 60 60 60
Retrieved
Context
OptimizationReflection Top-N 2 2 2 2
Table 5 presents the hyperparameter settings of theNeocor-
RAGframework on four benchmarks: NQ, MuSiQue, 2WikiMulti-
HopQA, and HotpotQA. These parameters span three key process-
ing stages—Candidate Evidence Space Identification,Precise Evidence
Chain Generation, andRetrieved Context Optimization—each con-
tributing to the effective enhancement of retrieved text quality.
C LLM Prompts
Context Passages:
•Wikipedia Title: The Newcomers (film)
The Newcomers is a 2000 American family drama film directed by James
Allen Bradley and starring Christopher McCoy, Kate Bosworth, Paul
Dano and Chris Evans. Christopher McCoy plays Sam Docherty, a boy
who moves to Vermont with his family, hoping to make a fresh start
away from the city. It was filmed in Vermont, and released by Artist
View Entertainment and MTI Home Video.
•Wikipedia Title: Chris Evans (actor)
Christopher Robert Evans (born June 13, 1981) is an American actor and
filmmaker. Evans is known for his superhero roles as the Marvel Comics
characters Steve Rogers / Captain America in the Marvel Cinematic
Universe and Johnny Storm / Human Torch in "Fantastic Four" and .
# Question: what is one of the stars of The Newcomers known
for?
# Information association path: [’The Newcomers - stars
-> Chris Evans - known for role -> Steve Rogers / Captain
America - part of -> Marvel Cinematic Universe - based on
-> Marvel Comics’, ’The Newcomers - stars -> Chris Evans -
known for role -> Johnny Storm / Human Torch - part of ->
Fantastic Four’, ’The Newcomers - stars -> Chris Evans -
profession -> actor’]
Thought:Chris Evans, a star in The Newcomers, is known for su-
perhero roles as Marvel Comics characters like Captain America and
Human Torch.
Answer:superhero roles as the Marvel Comics.
The following is the prompt template used for the 70B-scale
LLM:
As an advanced reading comprehension assistant, your task is to ana-
lyze text passages and corresponding questions meticulously.Note!:
Information association paths serve as reference pointers to surface
latent connections. Your response start after "Thought:", where you willmethodically break down the reasoning process, illustrating how you
arrive at conclusions. Conclude with "Answer:" to present a concise,
definitive response, devoid of additional elaborations.
The following is the prompt template used for the 3B-scale LLM:
Given some contexts, candidate evidence chains, and a question, the
chain be used to help find potential associative information. please only
output the answer to the question.
Input:
Wikipedia Title: {}
Question: {}
Information association paths: {}
Thought:{}
D Backbone Model Adaptability
This appendix provides an additional diagnostic analysis on the
adaptability of NeocorRAG to different base language models and
evidence chain counts, supplementing the discussion in Section 4.3.
Figure 6 reports the performance of F1 between the model families
(Llama and Qwen) and the model sizes under different constraints
in the evidence chain of the candidate evidence. The results further
confirm that NeocorRAG maintains stable performance across the
backbone models and that lightweight models benefit from care-
fully controlled evidence chain counts, while larger models can
effectively exploit expanded candidate spaces. These observations
support our choice of design to use lightweight auxiliary models
for the generation of evidence chains.
E Efficiency and Overhead
Scope.In the main paper, we evaluate the efficiency of NeocorRAG
primarily through token cost analysis on four benchmarks, showing
that it uses only 20.1% of the tokens required by CoRAG and less
than 2.94% of those used by TRACE. Although token consumption is
a strong proxy for computational cost, it does not fully capture prac-
tical efficiency. We therefore further characterize runtime latency
and GPU memory overhead.
Experimental Setup.Following the protocol in Section 4.1, we
perform the efficiency analysis onHotpotQA, our most challenging
multi-hop benchmark. All experiments adopt identical retrieval
results from HippoRAG2 and the same decoding configuration. We

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Shiyao Peng et al.
F1 score (50 chains) F1 score (70 chains)
0.560.5650.570.5750.580.585
0 5 10 15
0.560.5650.570.5750.580.585
0 5 10 15
NQ
0.310.320.330.340.35
0 5 10 15
0.310.320.330.340.35
0 5 10 15
MuSiQue
0.470.480.490.50.510.520.53
0 5 10 15
0.470.480.490.50.510.520.53
0 5 10 15
2Wiki
0.6550.660.6650.670.6750.68
0 5 10 15
0.6550.660.6650.670.6750.68
0 5 10 15
HotpotQALlama Series Qwen Series
Figure 6: F1 Score vs. evidence chain count across model families and sizes. This figure uses Llama-3.2-3B as the QA model
and presents the performance comparison under two retrieval constraints (50 and 70 candidate evidence chains) on four QA
benchmarks.
useLlama-3.2-3B-Instructas the constrained-decoding model for
the generation of precise evidence chain and the final QA. Results
are averaged over5 runswith a fixed random seed of42.
Latency and Memory.In single-instance QA, NeocorRAG achieves
aspeedup of 3.73 ×over TRACE and aspeedup of 1.27 ×over
CoRAG, consistent with token cost trends. The maximum usage
of GPU memory during pathfinding and inference is10,380 MiB.
Our constrained decoding further reduces peak memory by13.7%
compared to a standard vLLM launch.
Graph Overhead.The graph is constructed once during corpus
ingestion. At query time, required subgraphs are retrieved via lo-
cal index lookups, which introduces no additional GPU memory
overhead. In general, NeocorRAG achieves high efficiency and low
computational cost.
F Robustness and Sensitivity
Scope.We provide additional diagnostics to examine the robust-
ness and sensitivity of NeocorRAG under variations in graph con-
struction and hyperparameters. All experiments are conducted on
HotpotQA, using the same protocol, random seed and five-run
averaging as in Section 4.1. We useLlama-3.2-3Bfor evidence
chain mining and report changes in F1 score.
Robustness to Candidate Evidence Space Variations.We
evaluate robustness to variations in triple quality, graph expan-
sion depth, and retriever encoders. Injecting 50% noise into triples
decreases F1 by≈0.43 points, while randomly dropping 10% in-
creases F1 by≈0.8 points. Setting max_path_length =7 decreases
F1 by≈0.43, whereas expanding to 13 improves F1 by ≈0.8. Replac-
ingbge-large with bge-base orbge-small reduces F1 by≈0.39and≈0.49 points, respectively. In general, NeocorRAG exhibits
predictable performance variations without brittle behavior.
Sensitivity to Evidence Chain Generation Hyperparam-
eters.We further study the sensitivity of NeocorRAG to key hy-
perparameters in the evidence chain generation stage. Varying the
number of candidate evidence chains from 40 to 70 results in F1
changes of less than 0.2 percentage points, indicating low sensitivity
to this hyperparameter and demonstrating that NeocorRAG does
not rely on narrowly tuned parameter choices to achieve strong
performance.
G Metric Definitions
This section provides detailed definitions of the metrics used to
quantify the efficacy of the filtering process in our ablation study
(Section 4.5).
•Average Total Documents (ATD): The average number of doc-
uments provided to the reasoning model:
ATD=1
𝑛𝑛∑︁
𝑖=1𝐷′
𝑖 (9)
where𝐷′
𝑖denotes the final set of documents used by the model
for the𝑖-th question, and𝑛is the total number of questions.
•Average Invalid Documents (AID): The average number of
irrelevant documents among those provided:
AID=1
𝑛𝑛∑︁
𝑖=1{𝑑|𝑑∈𝐷′
𝑖,𝑑∉𝐷𝑐
𝑖} (10)
where𝐷𝑐
𝑖denotes the ground truth document set for the 𝑖-th
question.
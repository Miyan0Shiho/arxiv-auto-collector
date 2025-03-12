# Path Pooling: Train-Free Structure Enhancement for Efficient Knowledge Graph Retrieval-Augmented Generation

**Authors**: Hairu Wang, Yuan Feng, Xike Xie, S Kevin Zhou

**Published**: 2025-03-07 07:48:30

**PDF URL**: [http://arxiv.org/pdf/2503.05203v1](http://arxiv.org/pdf/2503.05203v1)

## Abstract
Although Large Language Models achieve strong success in many tasks, they
still suffer from hallucinations and knowledge deficiencies in real-world
applications. Many knowledge graph-based retrieval-augmented generation
(KG-RAG) methods enhance the quality and credibility of LLMs by leveraging
structure and semantic information in KGs as external knowledge bases. However,
these methods struggle to effectively incorporate structure information, either
incurring high computational costs or underutilizing available knowledge.
Inspired by smoothing operations in graph representation learning, we propose
path pooling, a simple, train-free strategy that introduces structure
information through a novel path-centric pooling operation. It seamlessly
integrates into existing KG-RAG methods in a plug-and-play manner, enabling
richer structure information utilization. Extensive experiments demonstrate
that incorporating the path pooling into the state-of-the-art KG-RAG method
consistently improves performance across various settings while introducing
negligible additional cost. Code is coming soon at
https://github.com/hrwang00/path-pooling.

## Full Text


<!-- PDF content starts -->

Path Pooling: Train-Free Structure Enhancement for Efficient Knowledge
Graph Retrieval-Augmented Generation
Hairu Wang1,3*, Yuan Feng1,3*, Xike Xie2,3†, S Kevin Zhou2,3
1School of Computer Science, University of Science and Technology of China, China
2School of Biomedical Engineering, USTC, China
3Data Darkness Lab, MIRACLE Center, Suzhou Institute for Advanced Reasearch, USTC, China
{hrwang00,yfung}@mail.ustc.edu.cn ,{xkxie,skevinzhou}@ustc.edu.cn
Abstract
Although Large Language Models achieve
strong success in many tasks, they still suffer
from hallucinations and knowledge deficien-
cies in real-world applications. Many knowl-
edge graph-based retrieval-augmented genera-
tion (KG-RAG) methods enhance the quality
and credibility of LLMs by leveraging struc-
ture and semantic information in KGs as exter-
nal knowledge bases. However, these methods
struggle to effectively incorporate structure in-
formation, either incurring high computational
costs or underutilizing available knowledge. In-
spired by smoothing operations in graph repre-
sentation learning, we propose path pooling,
a simple, train-free strategy that introduces
structure information through a novel path-
centric pooling operation. It seamlessly inte-
grates into existing KG-RAG methods in a plug-
and-play manner, enabling richer structure in-
formation utilization. Extensive experiments
demonstrate that incorporating the path pool-
ing into the state-of-the-art KG-RAG method
consistently improves performance across var-
ious settings while introducing negligible ad-
ditional cost. Code is coming soon at https:
//github.com/hrwang00/path-pooling .
1 Introduction
Large Language Models (LLMs), pre-trained on
vast corpora, have excelled in various natural lan-
guage processing tasks (Shen et al., 2024; Naveed
et al., 2023; Ge et al., 2023). However, outdated
information or missing domain-specific knowledge
in public training corpora often lead to hallucina-
tions in real-world applications (Wang et al., 2023;
Hong et al., 2023). To mitigate this, many meth-
ods (Pan et al., 2024; Peng et al., 2024; Edge et al.,
2025) integrate high-quality Knowledge Graphs
(KGs) for retrieval-augmented generation (RAG),
enhancing credibility.
*Equal contribution.
†Corresponding author.KGs organize large collections of knowledge
triples in a well-structured graph form and serve
as core knowledge bases across various domains
(Chein and Mugnier, 2008; Robinson et al., 2015).
Compared to traditional text-based knowledge
bases, KGs not only capture rich semantic infor-
mation but also offer a clear structure organization,
enabling efficient knowledge management and up-
dates. Therefore, KG-RAG provides higher-quality
domain knowledge in many applications, signif-
icantly mitigating LLMs’ hallucinations (Jiang
et al., 2023; Baek et al., 2023).
However, the complex knowledge structures in
KGs pose challenges for effective knowledge re-
trieval in KG-RAG systems. Current mainstream
KG-RAG methods can be categorized into path-
based KG-RAG andtriple-based KG-RAG based
on their retrieval paradigms. Earlier path-based
KG-RAG paradigm employs LLMs for path traver-
sal on KGs (Sun et al., 2024; Ma et al., 2024), sub-
sequent path-based works try to fine-tune LLMs
with KG information to retrieve knowledge graph
paths (Luo et al., 2024; Mavromatis and Karypis,
2024). Although these methods effectively lever-
age the structure information of KGs, they face
significant efficiency challenges, such as the high
cost of path retrieval due to the vast number of pos-
sible path combinations in graph structures and the
computational burden of multiple LLM calls dur-
ing path traversal (Sun et al., 2024; Ma et al., 2024),
which limits their real-world applications. To ad-
dress this efficiency issue, triple-based KG-RAG
paradigm (Hu et al., 2024; He et al., 2024a; Li et al.,
2025) focuses on the smallest unit in KGs—the
triple. Compared to the exponential number of
paths in the graph, these methods significantly im-
prove deployment efficiency. They typically use
a lightweight retriever to directly retrieve query-
relevant triples, followed by a single LLM call
to generate responses. Notably, the latest method,
SubgraphRAG (Li et al., 2025), trains a simple mul-arXiv:2503.05203v1  [cs.AI]  7 Mar 2025

tilayer perceptron (MLP) for triple retrieval based
on semantic and structure information, achieving
state-of-the-art performance. Particularly in terms
of efficiency, SubgraphRAG overwhelmingly out-
performs previous path-based KG-RAG methods,
being over 100 times faster than the representative
RoG in the retrieval process while requiring only a
single LLM call for generation.
However, current triple-based KG-RAG meth-
ods, despite attempt to embed graph structure infor-
mation into triple representations, primarily treat
structure information as an auxiliary enhancement
to semantics (He et al., 2024a; Hu et al., 2024; Li
et al., 2025). Many of these methods introduce
complex structure embedding models, which sig-
nificantly increase training complexity and cost.
At the same time, due to the inherent nature of
triples as the atomic units of a knowledge graph,
they inevitably fall short in capturing structure in-
formation compared to path-based KG-RAG meth-
ods, where paths inherently encode richer structure
context. This raises a critical question: Can we
leverage graph structure for KG-RAG in a low-
cost, train-free manner within the triple-based
paradigm?
Recent studies have shown that many graph rep-
resentation learning techniques (Hamilton et al.,
2018; Ying et al., 2018; Yang et al., 2024) in-
herently perform a form of Laplacian smoothing
(Dong et al., 2016), which aggregates features from
neighboring structures. This suggests that struc-
turally adjacent triples often share similar prop-
erties. Inspired by this structure locality pattern,
we propose a path pooling strategy that leverages
this smoothing effect to refine triple-based scoring,
enabling richer structure information to be incorpo-
rated in a plug-and-play manner.
Our path pooling strategy addresses the limi-
tations of current triple-based KG-RAG methods
by introducing structure information through a
lightweight and efficient workflow. The strategy
consists of two key steps: path kernel search and
smoothing along kernels. First, we identify key
paths in the KGs using a graph search algorithm,
such as Dijkstra’s algorithm, to extract local struc-
ture patterns while avoiding over-smoothing. Then,
we apply an average pooling operation along these
paths to smooth triple scores and incorporate a
position score to further preserve the structure in-
formation within the path structures. This process
enhances the representation of triples by incorporat-
ing information from their neighbors in the graph,without requiring additional training or complex
embedding models. Finally, we use this smoothed
triple score to optimize KG-RAG methods through
reranking or reselection mechanisms, effectively
improving their generation quality. This simple
yet effective strategy significantly improves the
accuracy of existing triple-based KG-RAG meth-
ods while introducing minimal computational over-
head. By combining the efficiency of triple-based
retrieval with the structure richness of path-based
methods, our path pooling strategy bridges the gap
between these methods, offering a novel solution
for incorporating graph structure into KG-RAG
systems.
2 Preliminary
Knowledge Graphs (KGs). KGs organize triples
in a structured form, encompassing a wealth of
domain-specific knowledge. A KG can be repre-
sented by G={(h, r, t )|h, t∈V, r∈E}, where
Vdenotes the set of entities and Erepresents the
set of relations. Each triple τ= (h, r, t )is a fact,
corresponding to an edge in the KGs, which indi-
cates the association between entity hand entity t
through r.
KG-RAG. KGs contain amount semantic informa-
tion and clear structure information that can be
served as high-quality external knowledge bases.
Existing research on KG-RAG mainly involves
two stages: retrieval and generation. For a given
query q, the KG-RAG methods first retrieve knowl-
edge Kfrom KGs relevant to qin the retrieval
phase: retrieve (q, G)→K. Retrieved knowl-
edge Kaugments q, forming an enriched query
q′:prompt (q, K)→q′. This q′then is used as
prompt1to enhance LLMs in the generation phase:
Gen(q′)→answer , improving the accuracy and
credibility of the generated content. Based on the
retrieved knowledge Krepresented in the form of
paths or triples, KG-RAG can be categorized into
path-based KG-RAG and triple-based KG-RAG.
Path-based KG-RAG retrieves paths from KGs,
which contain multi-hop reasoning relations be-
tween question entities and potential answer en-
tities. Retrieved path set can be represented as
P={p}, where each path is an alternating se-
quence of relations and entities: p=⟨h1→r1→
h2→r2→, . . . ,→tn⟩. Path-based KG-RAG ef-
fectively leverages the topology structure of KGs
1The prompt used in our experiments is detailed in the
Appendix A.4

and can provide more coherent contextual informa-
tion to the LLMs.
Triple-based KG-RAG focuses on retrieving the
smallest unit, triple from KGs. The triple-based
KG-RAG methods typically assign a similarity
score to each triple, thereby generating a retrieved
triple sequence T={(hi, ri, ti, si)}. They or-
ganize the higher-scoring triples in Tinto the
form of ⟨(h1, r1, t1),(h2, r2, t2), . . . , (hn, rn, tn)⟩
to enhance the generation of LLMs. Compared to
path-based KG-RAG, triple-based KG-RAG offers
higher efficiency, as well as greater flexibility and
scalability.
3 Methodology
3.1 Overview
In this section, we introduce our path pooling strat-
egy in detail, which seamlessly bridges the retrieval
and generation stages of triple-based KG-RAG.
Path pooling contains two key steps: searching path
kernels and smoothing triples along kernels. Then,
we propose reranking and reselection mechanisms
to further refine the smoothed triple sequence. The
overall framework is illustrated in Figure 1.
3.2 Triple-based KG-RAG
Triple-based KG-RAG methods retrieve a triple se-
quence Tfrom KGs by leveraging semantic and
structure information of KGs jointly. In terms
of semantic information, triple-based KG-RAG
methods initiate the retrieval process by encoding
both the query and information of KGs into a uni-
fied vector space. They typically apply an off-the-
shelf text encoder (e.g., SentenceBert (Reimers and
Gurevych, 2019); gte-large-en-v1.5 (Zhang et al.,
2024c)) to embed the textual attributes of entities
and relations within the KGs, yielding the corre-
sponding embeddings. The same encoding strategy
is employed to the query in order to ensure consis-
tency. Next, to extract the most relevant entities
and relations in the KGs, triple-based KG-RAG
methods employ a scorer (e.g., cosine similarity;
MLP) to match semantic similarity between them
and the given query. In terms of structure infor-
mation, GNNs and its variant are widely used to
effective capture topology structure of KGs. Sub-
graphRAG utilizes structure information as feature
(Li et al., 2025) to facilitate the aforementioned se-
mantic matching process, while other methods (He
et al., 2024a; Hu et al., 2024) align structure infor-
mation with extracted text information to augmentAlgorithm 1: Path Pooling Workflow
Data: Triple sequence
T={τ= (h, r, t, s )}
1Construct subgraph Gk= (Vk, Ek)fromT;
2Search path kernels on Gk;
3P={p{e}→v|v∈Vk} ∪ {pv→{e}|v∈Vk};
4foreach pathp= (τ1, τ2, . . . , τ l)∈Pdo
5 Calculate path kernel score by pooling;
6 sp=pooling (p) =1
lPl
i=1si;
7end
8Assign positional score pitoτi;
9pi=min({sj|τj∈T})
i·a, S(τi) =sp+pi;
10s′
i= max {S(τi)|τi∈p, p∈P};
11return Smoothed Triple sequence T′;
Algorithm 2: Triple-based KG-RAG
Data: KGG={(h, r, t )}, given query q
1Get subgraph from Gcentered on query
entities;
2Retrieve triple sequence Ton the subgraph;
3ifenhanced with path pooling then
4 T′←Algorithm 1( T);
5 Rerank or Reselect the refined T′toT′′;
6end
7q′←prompt (q, T′′),answer ←Gen(q′);
8return answer ;
the LLMs. The above retrieval process essentially
functions as a pruning operation, preserving a com-
pact triple sequence Tmost relevant to given query
for subsequently generation process. Tcan comple-
ment the missing knowledge of LLMs. Algorithm 2
describes the workflow of triple-based KG-RAG,
where path pooling effectively enhance graph struc-
ture information in a plug-and-play manner.
3.3 Path Pooling
Existing KG-RAG methods struggle to fully uti-
lize knowledge graph (KG) structures. Path-based
KG-RAG paradigm introduces structure through
traversal and scoring but incur high computational
costs, while triple-based KG-RAG paradigm en-
codes structure information yet primarily serve
as semantic enhancements, leaving the graph’s
structure underutilized. Inspired by the common
smoothing operation in graph representation learn-
ing, we propose a path pooling strategy that refines
triple-based scoring by leveraging this smoothing
effect. We first identifies key paths using a search
algorithm, then smooths scores along these paths

KGs
Reranking
Reselection
LLMs
Response
...
Subgraph
 described by triples
Searching Path Kernels
Treat path as kernel, search paths on subgraph.1
2
Further Adding KG
 Structure 
 Information
Smoothing Along Kernels.
S
emantic Information
(entity, relation, entity’, weight)
Refining
 Triple Sequence
Path Set 
 P
pooling triples along path, 
geting base score 
 s
p 
.
adding positional score, 
smoothing triples, geting 
new triple score.
+p
3
 +p
2
 +p
1●
●
...
+ question
entity
1
 entity
3
entity
5
s
1
s
2
s
3
entity
2
entity
1
entity
3
entity
5
s
1
’
s
2
’
s
3
’
entity
2
...
triple =Figure 1: Framework of Path Pooling
to enhance structure information. Surprisingly,
this simple yet effective strategy significantly im-
proves the accuracy of existing triple-based KG-
RAG methods while introducing minimal addi-
tional overhead. Algorithm 1 describes the details
of path pooling’s workflow.
Searching Path Kernels. To capture the inher-
ent locality of graph structures, we apply graph
search algorithms (e.g., Dijkstra; BFS; Random
Walk) on Gkconstructed from triple sequence T,
to extract paths composed of triples. Given the
common issue of over-smoothing in graph learning
(Chen et al., 2020; Bo et al., 2021; Rusch et al.,
2023), we employ Dijkstra’s algorithm to find the
shortest paths starting from the query entities. Com-
pared to more complex path search algorithms,
shortest path search algorithm is more efficient
and keeps computational costs manageable2. For
triples unreachable from or to query entities, we
treat them as single-hop paths for subsequent uni-
fied processing. So far, we obtain the path kernel
setP.
Smoothing Along Kernels. Next, we apply
a smoothing operation to the triples along the
searched path kernel. In traditional graph rep-
resentation learning, this process often relies on
learnable weighted aggregation, which provides
finer-grained information smoothing but typically
incurs high training costs. Given the efficiency
concerns in KG-RAG, we adopt mean smoothing,
similar to traditional average pooling. Specifically,
2In experiments Section 4.3.1, we also explored other
graph algorithms for path kernel search, all of which led to
performance improvements within our path pooling strategy.we take the average score along the path kernel
as the base score for all triples. Average score in-
tegrally considers all triples on a path, providing
excellent smoothing effect, demonstrating good nu-
merical stability, and exhibiting strong resistance to
the influence of outliers. The smoothing operation
compensates the KGs’ structure information of the
retrieved triples to a certain extent, allowing triples
in the same path to have more similar scores.
Additionally, we introduce a fine-grained posi-
tional score to better preserve positional informa-
tion along the path. For T, we first calculate the
minimum sminof all triples. Based on the order
triples appear in the path, we sequentially assign
additional score to them, the first triple adding smin
divided by a, the second smindivided by 2a, and
so forth, where a is a nonzero constant. In practice,
different paths in the path set Pmay contain same
triples. To address this issue, we adopt maximum
as the final triple score. Through this process, a
triple’s representation is smoothed via its neigh-
bors’ representations, since each triple is able to
receive additional information from neighboring
triples, while maintaining the relative position or-
der within the path.
3.4 Boosting KG-RAG with Path Pooling
After path pooling which effectively incorporates
structure information, we obtain a smoothed triple
sequence T′={τ′
i|τ′
i= (hi, ri, ti, s′
i)}. We then
demonstrate two ways to further refine the triple
sequence T′to enhance triple-based KG-RAG: Po-
sition Reranking and Triple Reselection.
Position Reranking. Due to the position encod-

ing, LLMs exhibit significant sensitivity to token
positions, a phenomenon known as position bias
(Tang et al., 2023; Hsieh et al., 2024; Zhang et al.,
2024a). This bias typically manifests in two cases:
1)Lost in the Middle (Liu et al., 2023; Xu et al.,
2024; Cao et al., 2024; Jin et al., 2025): This typi-
cally occurs in models with limited long-sequence
capabilities, where tokens in the middle of the input
are often overlooked during inference. 2) Recency
Bias: (Qin et al., 2023; Peysakhovich and Lerer,
2023) This is prevalent in most models and is pri-
marily attributed to the inherent properties of pop-
ular rotary position encoding. Tokens appearing
later in the sequence tend to receive higher atten-
tion weights. Many RAG studies (Peysakhovich
and Lerer, 2023; He et al., 2024b) have shown
that reranking retrieved content based on position
bias can effectively improve the quality of model-
generated responses. Following this insight, we
leverage the smoothed triple scores from path pool-
ing to reorder retrieved triples, ensuring a better
optimal input sequence T′′. Specifically, we po-
sition triples with higher scores closer to the final
question, adapting the widely observed Recent Bias .
This mechanism enhances existing KG-RAG meth-
ods by optimizing the input order of triples without
altering their content, solely relying on smoothed
scores for reordering.
Triple Reselection. Another key challenge in
KG-RAG is balancing high-quality retrieval con-
text while minimizing the inclusion of question-
irrelevant tokens. Removing such irrelevant tokens
not only reduces noise interference but also lowers
inference costs. Inspired by the widely adopted
coarse-to-fine selection paradigm (Yu et al., 2024;
Zhang et al., 2024b; Zhao et al., 2025), we pro-
pose Triple Reselection to enhance performance.
Specifically, we first apply the naive triple-based
KG-RAG scorer for a coarse-grained selection, re-
taining the top- khigh-scoring triples. Then, we
refine this selection by applying path pooling to
obtain more precise scores, further filtering down
to the top- k′triples. Finally, these fine-grained se-
lected triple sequence T′′is fed into the LLM in
an order aligned with the Recent Bias . This mech-
anism not only optimizes position ranking using
smoothed scores but also filters out irrelevant infor-
mation, leading to better performance.4 Experiments
4.1 Experiment Setup
Dataset. To evaluate the effectiveness of our path
pooling strategy in multi-hop knowledge-intensive
tasks, all of our experiments are based on the
widely-used CWQ dataset, which is constructed
from the Freebase (Bollacker et al., 2008). Free-
base is an rich knowledge graph, containing 126
million triples. CWQ (Talmor and Berant, 2018) is
a challenging benchmark for KGQA that includes
reasoning questions involving up to four hops.
Evaluation Metrics. Consistent with previous
works (Luo et al., 2024; Mavromatis and Karypis,
2024; Li et al., 2025), we adopt Macro-F1 and
Hit@1 as our evaluation metrics. Hit@1 mea-
sures whether the correct answer appears as the
top-ranked result. Macro-F1, which accounts for
both precision and recall in scenarios with multi-
ple valid answers, provides a more comprehensive
assessment of reasoning performance
Implementation Details. We enhance the state-
of-the-art triple-based KG-RAG method, Sub-
graphRAG, using path pooling through positional
reranking and triple reselection. For brevity, we ab-
breviate SubgraphRAG as S-RAG in the following
experiments. To evaluate the performance of our
path pooling strategy, we employ a diverse selec-
tion of open-source LLMs: Llama 3.1-8B, Llama
3.1-70B (Llama Team, 2024), Qwen 2.5-7B, and
Qwen 2.5-72B (Qwen et al., 2025). For all mod-
els, we use a one-shot prompt for reasoning, as
detailed in Appendix A.4. To ensure reproducibil-
ity, we set the temperature to 0 and the maximum
token length for generation to 4000, aligning with
previous research (Li et al., 2025).
4.2 Main Results
Qwen2.5-7b Llama3.1-8b Qwen2.5-72b Llama3.1-70b
Models404550556065Scores
Qwen2.5-7b Llama3.1-8b Qwen2.5-72b Llama3.1-70b
Models4045505560Scores
(a) Hit@1 (b) F1-ScoreS-RAG S-RAG w/ Our Reranking S-RAG w/ Our Reselection
Figure 2: Results of Different Models with 50 Triples
The experimental results of the reranking and
reselection mechanism, which refine the triple se-
quence using path pooling, are presented in Ta-
ble 3. Overall, reranking introduces minimal addi-

Triple Num 25 +1.6ms/query( PR) +18.4ms/query( TR)
Methods F1-Score Hit@1
w/o ours w/PRw/TRw/o ours w/PR w/TR
Llama-8b 45.4 46.0 48.4 48.7 49.2 52.4
Llama-70b 52.3 54.0 55.8 56.0 58.7 61.2
Qwen-7b 40.7 42.2 45.0 43.8 46.2 49.1
Qwen-72b 49.3 51.3 53.6 52.5 54.9 57.0
Triple Num 50 +2.1ms/query( PR) +18.4ms/query( TR)
Methods F1-Score Hit@1
w/o ours w/PRw/TRw/o ours w/PR w/TR
Llama-8b 46.8 47.8 50.5 49.6 51.7 54.0
Llama-70b 53.6 54.5 56.5 57.5 58.9 60.9
Qwen-7b 41.9 43.3 46.0 45.2 47.1 50.1
Qwen-72b 51.2 53.1 55.6 54.3 56.4 59.0
Triple Num 100 +3.2ms/query( PR) +18.5ms/query( TR)
Methods F1-Score Hit@1
w/o ours w/PRw/TRw/o ours w/PR w/TR
Llama-8b 47.7 48.8 51.0 50.2 52.7 54.4
Llama-70b 53.0 54.2 56.5 57.2 58.9 61.2
Qwen-7b 42.8 44.0 46.1 45.8 47.4 49.9
Qwen-72b 52.1 53.5 55.4 55.3 57.0 59.1
Triple Num 200 +6.3ms/query( PR) +18.5ms/query( TR)
Methods F1-Score Hit@1
w/o ours w/PRw/TRw/o ours w/PR w/TR
Llama-8b 47.2 49.5 50.2 50.1 53.2 54.2
Llama-70b 52.8 54.5 55.5 57.6 59.6 60.7
Qwen-7b 41.8 43.4 45.1 44.9 46.9 48.7
Qwen-72b 51.9 53.7 54.6 55.6 57.3 58.7
Table 1: Results of Positional Reranking and Triple
Reselection with Path Pooling. The best scores are
highlighted with bold and the second-best scores are
highlighted with underline .PRdenotes our positional
reranking mechanism with path pooling and TRdenotes
our triple reselection mechanism with path pooling.
tional retrieval time, as it only requires reorder-
ing triples based on the smoothed scores from
path pooling. For instance, even when process-
ing up to 200 triples, the additional retrieval time
is only 3.2 ms per query, while achieving a 3.1%
improvement in Hit@1 and a 2.3% increase in F1-
Score on Qwen2.5-7b. Furthermore, when apply-
ing path pooling for triple reselection, the perfor-
mance gains become even more pronounced. We
conduct experiments by first retrieving 500 triples
using SubgraphRAG and then selecting 25 to 200
triples based on their path pooling scores. Notably,
with just 25 triples, the reselection mechanism
boosts Hit@1 by 5.2% on Llama3.1-70b and 5.3%
on Qwen2.5-7b while introducing only 18.4ms per
query. Remarkably, using only 25 triples, the rese-
lection mechanism even outperforms the original
SubgraphRAG utilizing 200 triples, demonstrating
its high effectiveness.Figure 2 provides a detailed visualization of this
phenomenon in triple number 50. Our path pool-
ing strategy consistently enhances the performance
of SubgraphRAG across various settings. Sim-
ply leveraging the structure information from path
pooling to rerank triples in the prompt based on
positional bias already leads to notable improve-
ments. Moreover, refining triple reselection using
path pooling scores to filter out irrelevant noise re-
sults in further performance gains. Figure 3 further
visualizes the performance of Llama 3.1-8B3as the
number of triples increases from 25 to 200. When
using path pooling for reranking, performance con-
sistently improves with more triples, achieving sub-
stantial gains at 200 triples, with Hit@1 and Macro-
F1 increasing by. This aligns with prior findings
on positional bias: the longer the sequence, the
more pronounced the bias (Liu et al., 2023). In con-
trast, when applying triple reselection, path pooling
achieves significant improvements with as few as
25 triples, boosting Hit@1 and Macro-F1 by. How-
ever, as the number of triples increases further, the
performance gains gradually diminish.
2550 100 200
Triple Nums5055Scores
(a) Llama3.1-8b Hit@12550 100 200
Triple Nums4550Scores
(b) Llama3.1-8b F1-ScoreS-RAG S-RAG w/ Our Reranking S-RAG w/ Our Reselection
Figure 3: Results of Varying Triples
4.3 Further Analysis
We conduct additional experiments to evaluate the
impact of different path search and pooling algo-
rithms in our path pooling strategy. For demon-
stration, we apply the triple reselection mechanism
with path pooling on Llama3.1-8b. And the re-
sults of positional reranking mechanism with path
pooling are reported in Appendix A.
4.3.1 Different Path Search Algorithms
Beyond the widely used shortest path algorithm, Di-
jkstra, other graph traversal methods such as BFS
and Random Walk can also be employed to extract
structure information. We evaluate the impact of
different path search algorithms within our path
3In the paper of SubgraphRAG, the Hit of Llama3.1-8b
with 100 triples is 56.98.

Triple Num 25 50 100 200 500
Dijkstra 1.6 2.1 3.2 6.3 18.1
Random Walk 3.1 3.6 4.5 5.7 9.1
BFS 1.9 3.1 7.4 29.4 232.2
Table 2: Time Statistics of Different Path Search Algo-
rithms (ms/query)
pooling strategy on Llama3.1-8b, as shown in Fig-
ure 4. To prevent excessive smoothing of structure
information, we set the maximum path length for
BFS and Random Walk to 4 , and Random Walk
samples 256 times. Overall, all path search algo-
rithms contribute to performance improvements
when integrated into path pooling. However, Ran-
dom Walk performs significantly worse than the
other two, likely because its stochastic nature fails
to capture key structure information. While BFS
and Dijkstra achieve similar performance, BFS in-
troduces a significantly larger number of paths,
leading to increased computational overhead, as
shown in Table 2. For instance, with 500 triples,
BFS requires 232.2 ms, whereas Dijkstra only takes
18.1 ms, making it 12.8x times more efficient. Thus,
we recommend using Dijkstra in the path pooling
strategy to achieve a balance between effectiveness
and efficiency.
25 50 100 200
Triple Nums45505560Scores
(a) Llama3.1-8b Hit@125 50 100 200
Triple Nums455055Scores
(b) Llama3.1-8b F1-ScoreS-RAG
S-RAG w/ Our Reselection (Random Walk)S-RAG w/ Our Reselection (BFS)
S-RAG w/ Our Reselection (Dijkstra)
Figure 4: Reselection Results of Different Path Search
Algorithms on Llama3.1-8b
4.3.2 Different Pooling Strategies
In Figure 5, we further compare the performance of
different smoothing operations in our path pooling
strategy. We evaluate two commonly used pool-
ing operation for smoothing: max pooling, which
retains the maximum score along the path, and av-
erage pooling, which computes the mean score of
triples on the path as the base score. Both operation
could capture local graph structure information, en-
hancing the reasoning performance of LLMs. How-
ever, average pooling consistently outperforms max
pooling, especially when the number of triples is
small. For instance, with 50 triples, average pool-
ing shows a significant advantage over max pooling.Using max pooling provides a increase of 1.27%,
from 50.86 to 49.59, while leveraging average pool-
ing brings a substantial improvement of 4.42%,
from 54.01 to 49.59. This suggests that average
pooling better preserves neighborhood information,
making it the preferred choice for smoothing in
path pooling.
25 50 100 200
Triple Nums455055Scores
(a) Llama3.1-8b Hit@125 50 100 200
Triple Nums40455055Scores
(b) Llama3.1-8b F1-ScoreS-RAG
S-RAG w/ Our Reselection (Max Pooling)S-RAG w/ Our Reselection (Average Pooling)
Figure 5: Reselection Results of Different Pooling
Strategies on Llama3.1-8b
4.3.3 Lost in the Middle Positional Bias
25 50 100 200
Triple Nums404550Scores
(a) Qwen2.5-7b Hit@125 50 100 200
Triple Nums4045Scores
(a) Qwen2.5-7b F1-Score
25 50 100 200
Triple Nums455055Scores
(c) Llama3.1-8b Hit@125 50 100 200
Triple Nums404550Scores
(d) Llama3.1-8b F1-ScoreS-RAG S-RAG w/ Our Reranking (Lost in Middle) S-RAG w/ Our Reranking (Recency Bias)
Figure 6: Reranking Results of Lost in the Middle on
Qwen2.5-7b and Llama3.1-8b
We further explore the effect of reranking triples
based on lost in the middle positional bias, as
shown in Figure 6. Starting from the triples
with the highest smoothed scores, we iteratively
place them in the head and tail of the input se-
quence. Overall, both Lost in the Middle and Re-
cency Bias demonstrate significant improvements
over SubgraphRAG. Specifically, Lost in the Mid-
dle shows slightly weaker performance on Qwen,
while achieving comparable results on Llama. This
discrepancy is primarily due to differences in posi-
tional bias across models. Future work could fur-
ther optimize performance by investigating these
positional biases in greater depth. Given its simplic-
ity and robustness, we recommend incorporating
Recency Bias into path pooling.

5 Related Work
Path-based KG-RAG retrieves top- kknowledge
graph paths relevant to the query, fully utilizing the
structure information of KGs. Existing research has
focused on retrieves paths to provide updated or
domain-specific contextual knowledge for LLMs.
For instance, GNN-RAG (Mavromatis and Karypis,
2024) trains a GNN to retrieve answer candidates
for a given query from a dense KG subgraph and
extracts the shortest paths between the question en-
tities and the answer candidates. RoG (Luo et al.,
2024) utilizes a planning-retrieval-reasoning frame-
work to fine-tune the LLMs for generating relation
paths, and then retrieves complete paths from the
KG for LLMs reasoning. Nevertheless, GNN-RAG
and RoG both face significant efficiency challenges,
as they fine-tune LLMs, resulting in high training
overhead and retrieval latency. Subsequent, agent-
based method, ToG (Sun et al., 2024) attempt to
leverage LLMs to dynamically retrieve paths from
KGs and make decisions accordingly. ToG 2.0
(Ma et al., 2024) proposes a hybrid RAG paradigm
that effectively combines unstructured text infor-
mation with structured graph information, enabling
depth and comprehensive retrieval processes. How-
ever, ToG and ToG 2.0 tightly couple LLMs with
retrieval process, requiring multiple calls, which
leads to high computitional burden.
Triple-based KG-RAG retrieves the top- ktriples
relevant to the query, which is more efficient com-
pared to path-based KG-RAG. GRAG (Hu et al.,
2024) employs a divide-and-conquer strategy to
retrieve subgraphs that are most relevant to the
query.During the reasoning phase, GRAG provides
the LLMs with both text and graph views of in-
formation, enabling a deeper understanding of the
relations between entities. G-retrieval (He et al.,
2024a) integrates the advantages of LLM, GNN
and RAG, employing cosine similarity to retrieve
entities and relations, while the PCST algorithm
manages the size of the subgraph. The textual repre-
sentations of the subgraph, along with the structure
information outputted by the GNN, are fed into the
LLM to generate answers. Unlike previous work,
currently SubgraphRAG (Li et al., 2025) trains a
lightweight multilayer perceptron, which enables
scalable and flexible retrieval process, achieving
higher retrieval efficiency and better performance.
Compared to path-based KG-RAG, none of the
above methods require fine-tuning or multiple calls
to the LLMs. Our path pooling strategy can beseamlessly pluged into the retrieval and generation
phases of the triple-based KG-RAG in a low-cost,
train-free manner. It enhances the performance of
LLMs by introducing graph structure information
in the retrieved triple sequence.
6 Conclusion
In this paper, we address a key challenge in KG-
RAG: integrating rich structure information from
knowledge graphs without incurring excessive com-
putational costs. Current triple-based KG-RAG
methods are efficient but fail to capture the full
graph structure, while path-based KG-RAG meth-
ods, though more expressive, are computationally
expensive. To bridge this gap, we propose the
path pooling strategy to enhance existing triple-
based KG-RAG in a plug-and-play manner by
leveraging graph smoothing effects. Our method
identifies key paths using graph search and applies
smoothing to refine triple scores along these paths,
effectively enriching the structure representation in
triple-based KG-RAG. Experiments demonstrate
that path pooling consistently improves KG-RAG
performance with minimal overhead. This work
introduces a novel train-free enhancement strat-
egy for KG-RAG by leveraging smoothing effects
widely used in graph learning. Future work could
explore more effective path searching and finer-
grained smoothing techniques to capture richer
structure information.
References
Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023.
Knowledge-augmented language model prompting
for zero-shot knowledge graph question answering.
Preprint , arXiv:2306.04136.
Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen.
2021. Beyond low-frequency information in graph
convolutional networks. Proceedings of the AAAI
Conference on Artificial Intelligence , 35(5):3950–
3957.
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim
Sturge, and Jamie Taylor. 2008. Freebase: a collabo-
ratively created graph database for structuring human
knowledge. In Proceedings of the 2008 ACM SIG-
MOD international conference on Management of
data, pages 1247–1250.
Yukun Cao, Shuo Han, Zengyi Gao, Zezhong Ding,
Xike Xie, and S. Kevin Zhou. 2024. Graphin-
sight: Unlocking insights in large language mod-
els for graph structure understanding. Preprint ,
arXiv:2409.03258.

Michel Chein and Marie-Laure Mugnier. 2008. Graph-
based knowledge representation: computational
foundations of conceptual graphs . Springer Science
& Business Media.
Deli Chen, Yankai Lin, Wei Li, Peng Li, Jie Zhou, and
Xu Sun. 2020. Measuring and relieving the over-
smoothing problem for graph neural networks from
the topological view. Proceedings of the AAAI Con-
ference on Artificial Intelligence , 34(04):3438–3445.
Xiaowen Dong, Dorina Thanou, Pascal Frossard, and
Pierre Vandergheynst. 2016. Learning laplacian ma-
trix in smooth graph signal representations. IEEE
Transactions on Signal Processing , 64(23):6160–
6173.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summarization.
Preprint , arXiv:2404.16130.
Yingqiang Ge, Wenyue Hua, Kai Mei, Jianchao Ji,
Juntao Tan, Shuyuan Xu, Zelong Li, and Yongfeng
Zhang. 2023. Openagi: When LLM meets domain
experts. In NeurIPS .
William L. Hamilton, Rex Ying, and Jure Leskovec.
2018. Representation learning on graphs: Methods
and applications. Preprint , arXiv:1709.05584.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V . Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson,
and Bryan Hooi. 2024a. G-Retriever: Retrieval-
Augmented Generation for Textual Graph Un-
derstanding and Question Answering. Preprint ,
arXiv:2402.07630.
Zhiyuan He, Huiqiang Jiang, Zilong Wang, Yuqing
Yang, Luna Qiu, and Lili Qiu. 2024b. Posi-
tion engineering: Boosting large language mod-
els through positional information manipulation.
Preprint , arXiv:2404.11216.
Ruixin Hong, Hongming Zhang, Hong Zhao, Dong
Yu, and Changshui Zhang. 2023. Faithful question
answering with monte-carlo planning. ACL2023 .
Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li,
Zifeng Wang, Long T Le, Abhishek Kumar, James
Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Kr-
ishna, et al. 2024. Found in the middle: Calibrating
positional attention bias improves long context uti-
lization. arXiv preprint arXiv:2406.16008 .
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. GRAG:
Graph Retrieval-Augmented Generation. Preprint ,
arXiv:2405.16506.
Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin
Zhao, and Ji-Rong Wen. 2023. StructGPT: A general
framework for large language model to reason overstructured data. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing , pages 9237–9251, Singapore. Associa-
tion for Computational Linguistics.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2025. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. In The
Thirteenth International Conference on Learning
Representations .
Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is effec-
tive: The roles of graphs and large language models
in knowledge-graph-based retrieval-augmented gen-
eration. In International Conference on Learning
Representations .
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2023. Lost in the middle: How language mod-
els use long contexts. Preprint , arXiv:2307.03172.
AI Meta Llama Team. 2024. The llama 3 herd of
models. Preprint , arXiv:2407.21783.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2024. Reasoning on Graphs: Faithful
and Interpretable Large Language Model Reasoning.
Preprint , arXiv:2310.01061.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li,
Huaren Qu, and Jian Guo. 2024. Think-on-Graph
2.0: Deep and Interpretable Large Language Model
Reasoning with Knowledge Graph-guided Retrieval.
Preprint , arXiv:2407.10805.
Costas Mavromatis and George Karypis. 2024. GNN-
RAG: Graph Neural Retrieval for Large Language
Model Reasoning. Preprint , arXiv:2405.20139.
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muham-
mad Saqib, Saeed Anwar, Muhammad Usman, Nick
Barnes, and Ajmal Mian. 2023. A comprehensive
overview of large language models. arXiv preprint
arXiv:2307.06435 .
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
apu Wang, and Xindong Wu. 2024. Unifying large
language models and knowledge graphs: A roadmap.
IEEE Transactions on Knowledge and Data Engi-
neering , 36(7):3580–3599.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. Preprint , arXiv:2408.08921.
Alexander Peysakhovich and Adam Lerer. 2023. At-
tention sorting combats recency bias in long context
language models. Preprint , arXiv:2310.01427.
Guanghui Qin, Yukun Feng, and Benjamin Van Durme.
2023. The NLP task effectiveness of long-range
transformers. In Proceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics , pages 3774–3790, Dubrovnik,
Croatia. Association for Computational Linguistics.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin,
Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang,
Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang,
Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li,
Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji
Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang
Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru
Zhang, and Zihan Qiu. 2025. Qwen2.5 technical
report. Preprint , arXiv:2412.15115.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP) , pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Ian Robinson, Jim Webber, and Emil Eifrem. 2015.
Graph databases: new opportunities for connected
data. " O’Reilly Media, Inc.".
T. Konstantin Rusch, Michael M. Bronstein, and Sid-
dhartha Mishra. 2023. A survey on oversmoothing in
graph neural networks. Preprint , arXiv:2303.10993.
Siqi Shen, Lajanugen Logeswaran, Moontae Lee,
Honglak Lee, Soujanya Poria, and Rada Mihalcea.
2024. Understanding the capabilities and limitations
of large language models for cultural commonsense.
pages 5668–5680.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-
Graph: Deep and Responsible Reasoning of Large
Language Model on Knowledge Graph. Preprint ,
arXiv:2307.07697.
Alon Talmor and Jonathan Berant. 2018. The web as
a knowledge-base for answering complex questions.
InProceedings of the 2018 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long Papers) , pages 641–651, New Or-
leans, Louisiana. Association for Computational Lin-
guistics.
Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy
Lin, and Ferhan Ture. 2023. Found in the mid-
dle: Permutation self-consistency improves listwise
ranking in large language models. arXiv preprint
arXiv:2310.07712 .
Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li,
Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang
Xiong. 2023. Knowledge-driven cot: Exploring faith-
ful reasoning in llms for knowledge-intensive ques-
tion answering. arXiv preprint arXiv:2308.13259 .
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee,
Chen Zhu, Zihan Liu, Sandeep Subramanian, EvelinaBakhturina, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Retrieval meets long context large lan-
guage models. In The Twelfth International Confer-
ence on Learning Representations .
Leshanshui Yang, Clément Chatelain, and Sébastien
Adam. 2024. Dynamic graph representation learn-
ing with neural networks: A survey. IEEE Access ,
12:43460–43484.
Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang
Ren, Will Hamilton, and Jure Leskovec. 2018. Hier-
archical graph representation learning with differen-
tiable pooling. In Advances in Neural Information
Processing Systems , volume 31. Curran Associates,
Inc.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan
You, Chao Zhang, Mohammad Shoeybi, and Bryan
Catanzaro. 2024. RankRAG: Unifying context rank-
ing with retrieval-augmented generation in LLMs.
InThe Thirty-eighth Annual Conference on Neural
Information Processing Systems .
Zhang et al. 2024a. Attention instruction: Amplifying
attention in the middle via prompting. arXiv preprint
arXiv:2406.17095 .
Jinxu Zhang, Yongqi Yu, and Yu Zhang. 2024b.
CREAM: Coarse-to-fine retrieval and multi-modal
efficient tuning for document VQA. In ACM Multi-
media 2024 .
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li,
and Min Zhang. 2024c. mGTE: Generalized long-
context text representation and reranking models for
multilingual text retrieval. In Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing: Industry Track , pages 1393–1412,
Miami, Florida, US. Association for Computational
Linguistics.
Xinping Zhao, Yan Zhong, Zetian Sun, Xinshuo
Hu, Zhenyu Liu, Dongfang Li, Baotian Hu, and
Min Zhang. 2025. Funnelrag: A coarse-to-fine
progressive retrieval paradigm for rag. Preprint ,
arXiv:2410.10293.

A Appendix
A.1 Triple Reselection on Qwen2.5-7b
A.1.1 Different Path Search Algorithms
25 50 100 200
Triple Nums4550Scores
(a) Qwen2.5-7b Hit@125 50 100 200
Triple Nums4045Scores
(b) Qwen2.5-7b F1-ScoreS-RAG
S-RAG w/ Our Reselection (Random Walk)S-RAG w/ Our Reselection (BFS)
S-RAG w/ Our Reselection (Dijkstra)
Figure 7: Reselection Results of Path Search Algorithms
on Qwen2.5-7b
Triple reselection mechanism with all path
search algorithms enhances the start-of-the-art
method SubgraphRAG on Qwen2.5-7b, as shown
in Figure 7. Employing Random Walk as path
search algorithm shows a smaller improvement
with the number of triples being 100 and 200. This
may be due to the limited number of walks, which
fails to incorporate sufficient graph structure in-
formation. The other two path search algorithms,
BFS and Dijkstra, result in considerable improve-
ment on CWQ, while Dijkstra spends less time for
path searching. For example, in the case of 50
triples, BFS can achieve a substantial improvement
of 4.93%, from 45.23 to 50.16, and Dijkstra can
provide a similar increase of 4.9%, from 45.23 to
50.13 on Hit@1.
A.1.2 Different Pooling Strategies
25 50 100 200
Triple Nums4550Scores
(a) Qwen2.5-7b Hit@125 50 100 200
Triple Nums4045Scores
(b) Qwen2.5-7b F1-ScoreS-RAG
S-RAG w/ Our Reselection (Max Pooling)S-RAG w/ Our Reselection (Average Pooling)
Figure 8: Reselection Results of Pooling Strategies on
Qwen2.5-7b
On Qwen2.5-7b, triple reselection mechanism
with different pooling strategies still improves Sub-
graphRAG, as shown in Figure 8. Notably, Sub-
graphRAG with max pooling can achieve better
results than the original SubgraphRAG across all
triple numbers, both Hit@1 and F1-score. Max
pooling can provide a smoothing effect for triples
along the path. However, since it retains only themaximum along that path, it may loss some im-
portant neighborhood information. In contrast, av-
erage pooling demonstrates better improvement
compared to max pooling across various parame-
ter settings. For example, max pooling provides a
1.84% improvement over not using path pooling
from 45.23 to 47.07, while average pooling brings
3.06% more improvement than max pooling from
47.07 to 50.13 with 50 triples on Hit@1, demon-
strating its significant effectiveness.
A.2 Positional Reranking on Llama3.1-8b
A.2.1 Different Path Search Algorithms
25 50 100 200
Triple Nums5055Scores
(a) Llama3.1-8b Hit@125 50 100 200
Triple Nums4550Scores
(b) Llama3.1-8b F1-ScoreS-RAG
S-RAG w/ Our Reranking (Random Walk)S-RAG w/ Our Reranking (BFS)
S-RAG w/ Our Reranking (Dijkstra)
Figure 9: Reranking Results of Path Search Algorithms
on Llama3.1-8b
Figure 9 presents the performance of positional
reranking mechanism with different path search al-
gorithms on Llama3.1-8b. Overall, Random Walk,
BFS and Dijkstra exhibit effectiveness in the knowl-
edge graph question answering scenario. Similar
to the previous triple reselection mechanism, BFS
delivers a better performance boost at the cost of
longer path search times. Since the number of
paths searched by Random Walk remains consis-
tent across different triple numbers, it performs
better when the number of triples is low, while its
performance does not significantly improve when
the number of triples increase. This indirectly high-
lights the importance of incorporating sufficient
structure information of KGs.
A.2.2 Different Pooling Strategies
25 50 100 200
Triple Nums455055Scores
(a) Llama3.1-8b Hit@125 50 100 200
Triple Nums404550Scores
(b) Llama3.1-8b F1-ScoreS-RAG
S-RAG w/ Our Reranking (Max Pooling)S-RAG w/ Our Reranking (Average Pooling)
Figure 10: Reranking Results of Pooling Strategies on
Llama3.1-8b

Triple Num 500 +18.1ms/query
Methods F1-Score Hit@1
w/o ours w/ ours ∆w/o ours w/ ours ∆
Llama-8b 45.5 48.3 +2.8 49.2 53.3 +4.1
Llama-70b 51.8 53.6 +1.8 56.9 58.7 +1.8
Qwen-7b 41.7 43.5 +1.8 44.9 47.1 +2.2
Qwen-72b 52.2 54.7 +2.5 55.8 58.8 +3.0
Table 3: Reranking Results with 500 triples
Figure 10 presents the results of positional
reranking mechanism with different pooling strate-
gies on Llama3.1-8b. Both positional reranking
with max pooling and positional reranking with
average pooling show promising results. Notably,
positional reranking with average pooling outper-
forms with max pooling in the majority of cases,
whether in terms of Hit@1 or F1-Score. For exam-
ple, with 200 triples, position reranking mechanism
with average pooling improves SubgraphRAG by
2% on Hit@1 and by 1% on F1-Score.
A.3 Positional Reranking on Qwen2.5-7b
Next, we present the experimental results of the
positional reranking mechanism with path pooling.
When the number of triples reaches 500, the in-
crease in token counts results in a long sequence
reasoning problem. However, SubgraphRAG en-
hanced with our positional reranking mechanism
still demonstrates substantial advantages. For in-
stance, this is particularly evident in the perfor-
mance of Llama3.1-8b, as shown in Table 3. Lever-
aging our positional reranking mechanism achieves
a 4.1% improvement in Hit@1 and a 2.8% increase
in F1-Score.
A.3.1 Different Path Search Algorithms
25 50 100 200
Triple Nums404550Scores
(a) Qwen2.5-7b Hit@125 50 100 200
Triple Nums4045Scores
(b) Qwen2.5-7b F1-ScoreS-RAG
S-RAG w/ Our Reranking (Random Walk)S-RAG w/ Our Reranking (BFS)
S-RAG w/ Our Reranking (Dijkstra)
Figure 11: Reranking Results of Path Search Algorithms
on Qwen2.5-7b
Figure 11 summarizes Hit@1 and F1-Score of
positional reranking mechanism with different path
search algorithms on Qwen2.5-7b. Consistent with
previous results, all three path search algorithms
are effective in enhancing the performance of Sub-graphRAG. As the number of triples increases, the
Hit@1 and F1-Score of SubgraphRAG show a ini-
tially rising and then declining trend. However,
when enhanced with a positional reranking mech-
anism based on BFS path search algorithm, both
metrics demonstrate a consistently upward trend.
For instance, with 100 triples, the Hit@1 of Sub-
graphRAG is 45.77, while it is 47.52 with BFS,
which brings about an increase of 1.75%. The re-
sult emphasizes the effectiveness of incorporating
the structure information of KGs.
A.3.2 Different Pooling Strategies
25 50 100 200
Triple Nums404550Scores
(a) Qwen2.5-7b Hit@125 50 100 200
Triple Nums4045Scores
(b) Qwen2.5-7b F1-ScoreS-RAG
S-RAG w/ Our Reranking (Max Pooling)S-RAG w/ Our Reranking (Average Pooling)
Figure 12: Reranking Results of Pooling Strategies on
Qwen2.5-7b
As illustrated in Figure 12, positional rerank-
ing mechanism with different pooling strategies
all enhance SubgraphRAG on Qwen2.5-7b. Sim-
ilar benefits are observed when leveraging triple
reselection mechanism and on Llama3.1-8b. As
commonly used pooling strategies, both max pool-
ing and average pooling demonstrate significant
roles here. However, surprisingly, when the number
of triples increases from 50 to 200, the positional
reranking mechanism using max pooling achieves
better enhancement effect than using average pool-
ing. For example, with 50 triples, the Hit@1 of
SubgraphRAG is 45.23, while it increases to 47.52
when using max pooling, resulting in an improve-
ment of 2.23%. In contrast, using average pooling
only yields a 1.9% increase.

A.4 Prompts
The following is the detailed prompt template used in SubgraphRAG for all experiments. Due to Recency
Bias that tokens appearing later in the sequence tend to receive higher attention weights, we place triples
in ascending order , aligned with all other experiments.
SubgraphRAG Prompt
System:
Based on the triplets retrieved from a knowledge graph, please answer the question. Please return
formatted answers as a list, each prefixed with "ans:".
User:
Triplets:
(m.011zsc4_, organization.leadership.organization, San Francisco Giants)
(m.0crtd80, sports.sports_league_participation.league, National League West)
. . .
(San Francisco Giants, time.participant.event, 2014 Major League Baseball season)
(San Francisco Giants, time.participant.event, 2012 Major League Baseball season)
(AT&T Park, location.location.events, 2010 World Series)
(San Francisco Giants, sports.professional_sports_team.owner_s, Bill Neukom)
(San Francisco Giants, time.participant.event, 2010 Major League Baseball season)
(San Francisco Giants, sports.sports_team.championships, 2010 World Series)
(San Francisco Giants, time.participant.event, 2012 World Series)
(Crazy Crab, sports.mascot.team, San Francisco Giants)
(San Francisco Giants, time.participant.event, 2010 World Series)
(San Francisco Giants, sports.sports_team.championships, 2012 World Series)
(San Francisco Giants, sports.sports_team.team_mascot, Crazy Crab)
(San Francisco Giants, sports.sports_team.championships, 2014 World Series)
(Lou Seal, sports.mascot.team, San Francisco Giants)
Question: What year did the team with mascot named Lou Seal win the World Series?
Assistant:
To find the year the team with mascot named Lou Seal won the World Series, we need to find the
team with mascot named Lou Seal and then find the year they won the World Series. From the
triplets, we can see that Lou Seal is the mascot of the San Francisco Giants. Now, we need to find
the year the San Francisco Giants won the World Series. From the triplets, we can see that San
Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series. So,
the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012,
and 2014. Therefore, the formatted answers are:
ans: 2014 World Series
ans: 2012 World Series
ans: 2010 World Series
User:
Triplets:
Question:
The following is the detailed prompt template used in our reranking and reselection mechanism with path
pooling for all experiments. Retrieved triple sequence has a large number of repeating entities. In order to
form more coherent contextual and to save inference overhead, if, after reranking, the tail entity of the

previous triple is the same as the head entity of the following triple, we merge the duplicate ones and
using→to connect entities and relations.
Path Pooling Prompt
System:
Based on the paths retrieved from a knowledge graph, please answer the question. Please return
formatted answers as a list, each prefixed with "ans:".
User:
Paths:
m.011zsc4_->organization.leadership.organization->San Francisco Giants
m.0crtd80->sports.sports_league_participation.league->National League West . . .
San Francisco Giants->sports.sports_team.location->San Francisco
San Francisco Giants->time.participant.event->2014 Major League Baseball season
San Francisco Giants->time.participant.event->2012 Major League Baseball season
San Francisco Giants->sports.professional_sports_team.owner_s->Bill Neukom
San Francisco Giants->time.participant.event->2010 Major League Baseball season
San Francisco Giants->time.participant.event->2012 World Series
San Francisco Giants->time.participant.event->2010 World Series
San Francisco Giants->sports.sports_team.team_mascot->Crazy Crab
Lou Seal->sports.mascot.team->San Francisco Giants->sports.sports_team.championships->2014
World Series
Question: What year did the team with mascot named Lou Seal win the World Series?
Assistant:
To find the year the team with mascot named Lou Seal won the World Series, we need to find
the team with mascot named Lou Seal and then find the year they won the World Series. From
the paths, we can see that Lou Seal is the mascot of the San Francisco Giants. Now, we need to
find the year the San Francisco Giants won the World Series. From the paths, we can see that San
Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series. So,
the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012,
and 2014. Therefore, the formatted answers are:
ans: 2014 World Series
ans: 2012 World Series
ans: 2010 World Series
User:
Paths:
Question:
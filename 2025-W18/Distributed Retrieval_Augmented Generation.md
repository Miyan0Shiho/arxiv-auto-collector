# Distributed Retrieval-Augmented Generation

**Authors**: Chenhao Xu, Longxiang Gao, Yuan Miao, Xi Zheng

**Published**: 2025-05-01 10:37:06

**PDF URL**: [http://arxiv.org/pdf/2505.00443v1](http://arxiv.org/pdf/2505.00443v1)

## Abstract
As large language models (LLMs) become increasingly adopted on edge devices,
Retrieval-Augmented Generation (RAG) is gaining prominence as a solution to
address factual deficiencies and hallucinations by integrating external
knowledge. However, centralized RAG architectures face significant challenges
in data privacy and scalability. For instance, smart healthcare services often
rely on collecting sensitive patient data and building a centralized knowledge
base to provide better diagnosis and treatment advice, while privacy concerns
significantly impede this process. Besides, maintaining a comprehensive and
continuously updated knowledge base is costly, particularly in response to
regional epidemics and rapidly mutating viruses. To address these challenges,
this paper introduces Distributed Retrieval-Augmented Generation (DRAG), a
novel framework that improves data privacy by eliminating the need for a
centralized knowledge base and restoring data control to owners. DRAG
incorporates a Topic-Aware Random Walk (TARW) algorithm that leverages LLMs to
extract query topics and facilitate targeted peer discovery within a
peer-to-peer network, enabling efficient knowledge retrieval in decentralized
environments. Extensive experiments across three diverse datasets and LLMs
demonstrate that DRAG with TARW achieves near-centralized RAG performance by
using half as many messages as flooding. The code is available at
https://github.com/xuchenhao001/DRAG.

## Full Text


<!-- PDF content starts -->

Distributed Retrieval-Augmented Generation
Chenhao Xu
chenhao.xu@vu.edu.au
Victoria University
Melbourne, AustraliaLongxiang Gao
gaolx@sdas.org
Qilu University of Technology
Jinan, China
Yuan Miao
yuan.miao@vu.edu.au
Victoria University
Melbourne, AustraliaXi Zheng
james.zheng@mq.edu.au
Macquarie University
Sydney, Australia
ABSTRACT
As large language models (LLMs) become increasingly adopted
on edge devices, Retrieval-Augmented Generation (RAG) is
gaining prominence as a solution to address factual deficien-
cies and hallucinations by integrating external knowledge.
However, centralized RAG architectures face significant chal-
lenges in data privacy and scalability. For instance, smart
healthcare services often rely on collecting sensitive patient
data and building a centralized knowledge base to provide
better diagnosis and treatment advice, while privacy con-
cerns significantly impede this process. Besides, maintaining
a comprehensive and continuously updated knowledge base
is costly, particularly in response to regional epidemics and
rapidly mutating viruses. To address these challenges, this
paper introduces Distributed Retrieval-Augmented Genera-
tion (DRAG), a novel framework that improves data privacy
by eliminating the need for a centralized knowledge base and
restoring data control to owners. DRAG incorporates a Topic-
Aware Random Walk (TARW) algorithm that leverages LLMs
to extract query topics and facilitate targeted peer discovery
within a peer-to-peer network, enabling efficient knowledge
retrieval in decentralized environments. Extensive experi-
ments across three diverse datasets and LLMs demonstrate
that DRAG with TARW achieves near-centralized RAG per-
formance by using half as many messages as flooding. The
code is available at https://github.com/xuchenhao001/DRAG.
Permission to make digital or hard copies of all or part of this work for
personal or classroom use is granted without fee provided that copies
are not made or distributed for profit or commercial advantage and that
copies bear this notice and the full citation on the first page. Copyrights
for components of this work owned by others than the author(s) must
be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific
permission and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
©2025 Copyright held by the owner/author(s). Publication rights licensed
to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06. . . $15.00
https://doi.org/XXXXXXX.XXXXXXXCCS CONCEPTS
•Computing methodologies →Natural language pro-
cessing ;Distributed artificial intelligence .
KEYWORDS
Large Language Model, Retrieval-Augmented Generation,
Distributed Computing, Edge Computing
ACM Reference Format:
Chenhao Xu, Longxiang Gao, Yuan Miao, and Xi Zheng. 2025. Dis-
tributed Retrieval-Augmented Generation. In Proceedings of Make
sure to enter the correct conference title from your rights confirmation
emai (Conference acronym ’XX). ACM, New York, NY, USA, 11 pages.
https://doi.org/XXXXXXX.XXXXXXX
1 INTRODUCTION
Large Language Models (LLMs) have revolutionized natu-
ral language processing, enabling a wide range of applica-
tions in edge computing [ 1]. From voice assistants like Siri
and Google Assistant to personalized recommendation sys-
tems, LLMs are integral to enhancing user experience on
edge devices [ 2]. In healthcare, LLMs are being explored for
symptom checking and medical advice [ 3]. In driver assis-
tance, LLMs can improve navigation and real-time traffic
updates [ 4]. However, a significant challenge with LLMs is
their propensity to generate factually incorrect information,
a phenomenon known as hallucination [ 5], which hampers
reliability in critical applications.
To mitigate this, Retrieval Augmented Generation (RAG) [ 6]
has emerged as a promising approach. RAG combines LLM
generative capabilities with external knowledge sources, re-
trieving relevant information before generating responses
to ensure factual grounding. In edge computing, RAG can be
particularly beneficial for providing up-to-date and accurate
information directly to users on their devices [2].
However, centralized RAG architectures present consider-
able hurdles, particularly concerning data privacy and scala-
bility [ 7–12]. Consider a smart healthcare application: whilearXiv:2505.00443v1  [cs.DC]  1 May 2025

Conference acronym ’XX, June 03–05, 2025, Woodstock, NY XXX et al.
leveraging LLMs to provide personalized treatment recom-
mendations, these systems often require access to sensitive
patient data to build and maintain a comprehensive knowl-
edge base. However, privacy concerns and regulations usu-
ally restrict the sharing of such protected health information.
Similarly, in autonomous driving systems, accessing the lo-
cations and traffic plans of other vehicles can improve navi-
gation efficiency and safety. However, aggregating such data
into a centralized server raises significant privacy concerns.
In addition, the huge amount of data generated by regional
epidemics, rapidly mutating viruses, and vehicle movements
pose centralized RAG a scalability challenge, requiring effi-
cient management of the computational demands for both
retrieval and generation. Beyond these technical challenges,
maintaining a continuously updated and comprehensive cen-
tralized knowledge base is inherently costly, requiring sub-
stantial resources for data acquisition, storage, and process-
ing.
To address the aforementioned challenges, this paper intro-
duces Distributed Retrieval-Augmented Generation (DRAG),
a novel framework that improves data privacy and scalability
of RAG. DRAG eliminates the need for a central knowledge
base or direct data sharing by enabling peer-to-peer (P2P)
knowledge retrieval and generation across a distributed net-
work. The decentralized design mitigates privacy risks by
allowing users to have full control over their data while
distributing the computational burden of knowledge base
maintenance and retrieval. To facilitate efficient knowledge
discovery within the DRAG network, a novel Topic-Aware
Random Walk (TARW) algorithm is proposed. TARW lever-
ages LLMs to extract query topics and guide peer discovery
while utilizing a local cache to further enhance performance.
The key contributions of this paper are summarized as fol-
lows:
•A decentralized RAG framework tailored for edge com-
puting environments, improving data privacy and scal-
ability by eliminating direct data sharing and distribut-
ing the knowledge retrieval burden among peers.
•A novel Topic-Aware Random Walk (TARW) algorithm
that leverages LLMs to extract query topics and guide
efficient peer discovery within the DRAG network
through a local cache.
•Extensive experiments across diverse datasets and LLMs
validate that DRAG with TARW achieves performance
comparable to centralized RAG with significantly re-
duced communication overhead than flooding.
The rest of the paper is organized as follows: Section 2
reviews related work on Retrieval-Augmented Generation
(RAG) scalability, privacy, and security. Section 3 details theproposed DRAG framework and the TARW algorithm. Sec-
tion 4 presents the experimental evaluation. Finally, Section 5
concludes the paper.
2 RELATED WORK
This section reviews related work on DRAG, focusing on
RAG scalability, privacy, and security.
2.1 RAG Scalability
Retrieval-Augmented Generation (RAG) has emerged as a
pivotal approach to mitigate hallucinations in LLMs [ 13].
Lewis et al. [ 14] pioneered this field with their seminal work
introducing RAG as a framework that combines dense re-
trieval with sequence-to-sequence models for knowledge-
intensive NLP tasks. Despite recent advancements in re-
trieval mechanisms [ 15–21], centralized RAG architectures
inherently face challenges regarding computational scalabil-
ity and knowledge base maintenance.
Some researchers have proposed query routing [ 22,23]
and hierarchical retrieval [ 24,25] methods that primarily
focus on efficiently assigning queries or aggregating results
across multiple LLM candidates. While these approaches mit-
igate RAG scalability issues to some extent, they still over-
look the fundamental scalability challenges associated with
centralized knowledge base storage and processing within
each LLM candidate.
Recent research has also explored federated RAG archi-
tectures [ 26,27], which enable federated search and collab-
orative knowledge base development. However, their ap-
proaches still rely on a centralized request mediator, which
inherently introduces vulnerability points and scalability
bottlenecks.
2.2 RAG Privacy and Security
Privacy and security concerns within RAG systems have
become increasingly critical as these technologies are de-
ployed in sensitive domains [ 28]. For instance, studies have
demonstrated the feasibility of privacy attacks that can ex-
tract confidential data from the RAG knowledge base [ 29].
Research has also revealed vulnerabilities of RAG to mem-
bership inference attacks [ 30–32] and poisoning attacks [ 33].
Consequently, various privacy-preserving techniques have
been explored in the context of RAG, including secure multi-
party computation [ 8], homomorphic encryption [ 34], wa-
termarking [ 35], and differential privacy [ 9,11,36]. While
these advancements represent significant progress, they typ-
ically rely on centralized knowledge bases or processing
nodes. This inherent reliance introduces potential points of
vulnerability and can compromise data sovereignty.

Distributed Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
By contrast, the DRAG framework proposed in this paper
addresses these limitations by empowering users with con-
trol over their data. DRAG allows individuals to determine
what information is shared within the network and what
remains private, thereby ensuring data privacy.
3 PROPOSED MODEL
This section explains the proposed model, covering the Dis-
tributed Retrieval-Augmented Generation framework, the
Topic-Aware Random Walk algorithm, and further discus-
sion.
3.1 Distributed RAG
Distributed RAG represents a paradigm shift from traditional
centralized RAG systems. Unlike conventional approaches
that rely on a centralized knowledge base, DRAG distributes
both the knowledge and computation across a network of
edge devices, creating a P2P framework for knowledge shar-
ing and query resolution. This architecture enables knowl-
edge sharing and collaborative query resolution, potentially
improving scalability, resilience, and privacy compared to
centralized approaches.
Figure 1 illustrates the architecture of DRAG. Each edge
device is referred to as a peer. In healthcare, peers could be
smartphones, wearable devices, provider workstations, or
hospital servers sharing anonymized insights. In autonomous
driving, peers could be individual vehicles, roadside units, or
smart cameras exchanging traffic information. Each peer is
equipped with a local knowledge base, a local LLM instance,
and a communication module facilitating direct peer-to-peer
interaction. Upon receiving a query, a peer (querying peer)
first attempts to answer it using its local knowledge base. If
the query cannot be adequately answered locally, the query-
ing peer initiates a distributed search across the network,
leveraging the TARW algorithm. Once receiving relevant
knowledge from other peers, the querying peer uses LLM to
aggregate and generate a final answer.
When a user issues a query, DRAG processes it through
the following steps:
(1)Local Query Analysis : The query processing flow in
DRAG begins with a local query analysis phase. The
local LLM on the querying peer analyzes the query to
extract key topics and determine if external knowledge
is required. This determination can be made through
several methods. One approach involves calculating
the semantic similarity between the query and the
content in the local knowledge base. If the highest sim-
ilarity score falls below a predefined threshold, exter-
nal knowledge is deemed necessary. Alternatively, the
system can incorporate user feedback signals, where
explicit or implicit signals, such as low satisfactionscores on initial local responses, trigger the need for
external knowledge.
(2)Knowledge Discovery : If external information is
deemed necessary, the system enters the knowledge
discovery phase, initiating the TARW algorithm to
discover peers possessing relevant knowledge. The
TARW algorithm intelligently explores the network,
prioritizing peers that have demonstrated expertise
in the topic areas identified during the query analysis
phase.
(3)Distributed Retrieval : Following the discovery of
relevant peers, DRAG proceeds to the distributed re-
trieval phase. Rather than retrieving raw data, DRAG
gathers user-filtered knowledge snippets from peers
to ensure data privacy. In particular, these filters can
be rule-based, relying on predefined patterns and key-
words, or LLM-driven, utilizing machine learning mod-
els to automatically identify and mask sensitive data.
By integrating these privacy-preserving mechanisms,
DRAG facilitates secure and efficient knowledge shar-
ing while minimizing the risks of data leakage and
privacy violations.
(4)Local Generation : At this stage, the querying peer
combines the retrieved knowledge snippets with its
local context to generate an accurate and coherent
response using its local LLM.
(5)Result Caching : Finally, query results, consisting
of the generated response and the source knowledge
snippets, are cached locally to improve response time
for future similar queries. Besides, the system also
caches information about the expertise areas of neigh-
boring peers. This can be achieved by storing topics
of their successful contributions to previous queries.
This cached expertise information enables the sys-
tem to intelligently route future queries directly to
the peers most likely to possess relevant knowledge,
further reducing network traffic and improving overall
efficiency.
The DRAG architecture offers several advantages over
centralized RAG systems. By keeping sensitive data on local
devices, DRAG enhances privacy, making it particularly suit-
able for domains like healthcare where data confidentiality
is crucial. The decentralized approach also improves scala-
bility by distributing the computational load across multiple
nodes, reducing bottlenecks associated with a centralized
knowledge base. Additionally, result caching minimizes the
latency of peer discovery and query processing, ensuring
faster responses while reducing redundant computations.
Finally, the distributed nature of this framework increases
robustness against single points of failure, ensuring uninter-
rupted service even if some nodes become unavailable.

Conference acronym ’XX, June 03–05, 2025, Woodstock, NY XXX et al.
Peer 1
LLMKnowledge Base
Communication
ModuleUser 1
QueryPeer 2
Knowledge Base
Communication
Module
LLMPeer 4
Knowledge Base
Communication
Module
LLM
Peer 3
Knowledge Base
Communication
Module
LLMResponse
User 2
 User 4
User 3
Step ① : Local Query Analysis
Step ②: Knowledge Discovery
Step ③: Distributed Retrieval
Step ④: Local Generation
Step ⑤: Result Caching
② ②②①②
③ ④⑤
Figure 1: Overview of the Distributed RAG (DRAG) Architecture. DRAG employs a peer-to-peer network where
each peer maintains a local knowledge base and utilizes the Topic-Aware Random Walk (TARW) algorithm for
distributed knowledge retrieval. The query process involves local query analysis, knowledge discovery, distributed
retrieval, local generation, and result caching.
3.2 Topic-Aware Random Walk
The knowledge discovery phase in DRAG leverages a novel
routing algorithm called TARW, designed to efficiently navi-
gate the distributed network and identify peers possessing
relevant knowledge. Unlike traditional random walk algo-
rithms, which explore the network randomly, TARW incor-
porates topic awareness to prioritize the exploration of peers
more likely to hold the desired information. The algorithm
is outlined in Algorithm 1.
TARW begins by extracting key topics Tfrom the input
query𝑞using the local LLM of the querying peer 𝑝0(Line
1). This process provides a semantic understanding of the
information being sought. A set of visited peers Vand a
queueQare initialized, with the querying peer and a hop
count of 0 added (Lines 2-3). Additionally, a set Eis initialized
to cache peer expertise (Line 4). The algorithm then iterates
while the queue is not empty (Line 5).
In each iteration, the algorithm dequeues a peer 𝑝𝑖and its
associated hop count ℎfrom the queue (Line 6). If the hop
count exceeds a predefined maximum number of hops, 𝐻𝑚𝑎𝑥,
the algorithm skips to the next iteration to prevent exces-
sively long paths (Lines 7-8). The peer then attempts to an-
swer the query locally by consulting its local knowledge base
(Line 10). If the relevance score of the locally retrieved knowl-
edgeK𝑖to the query 𝑞, as determined by a relevance function
(e.g., semantic similarity), exceeds a predefined threshold 𝜃,
the algorithm adds the peer 𝑝𝑖and the extracted topics TAlgorithm 1 Topic-Aware Random Walk (TARW)
Input: Query𝑞, Max hops𝐻𝑚𝑎𝑥, Neighbor selection count
𝑘, Relevance threshold 𝜃.
Output: Privacy-filtered knowledge Kor∅.
1:T← ExtractTopic(𝑞) ⊲Use local LLM
2:V←{𝑝0} ⊲𝑝0is the querying peer
3:Q←{(𝑝0,0)} ⊲Queue of (peer, hop count) pairs
4:E←∅ ⊲Cache of peer expertise
5:whileQ≠∅do
6:(𝑝𝑖,ℎ)← Dequeue(Q)
7: ifℎ≥𝐻𝑚𝑎𝑥then
8: continue
9: end if
10:K𝑖←QueryLocal(𝑝𝑖,𝑞)
11: ifRelevance(K𝑖,𝑞)≥𝜃then
12:E←E∪{( 𝑝𝑖,T)}
13: return PrivacyFilter(K𝑖)
14: end if
15:N𝑖←Neighbors(𝑝𝑖)\V
16:S← RelevanceScore(N𝑖,E,T)
17:P𝑛𝑒𝑥𝑡←SelectTopK(N𝑖,S,𝑘)
18: for𝑝𝑗∈P 𝑛𝑒𝑥𝑡do
19:V←V∪{ 𝑝𝑗}
20: Enqueue(Q,(𝑝𝑗,ℎ+1))
21: end for
22:end while
23:return∅

Distributed Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
to the set of cached peer expertise E(Line 12). This dynami-
cally updates the system’s knowledge of peer specializations.
The algorithm then applies a privacy filter to the retrieved
knowledge and returns the filtered result (Line 13). This en-
sures that only relevant and privacy-preserving knowledge
is shared.
If the local query does not yield sufficiently relevant in-
formation, the algorithm identifies the neighbors N𝑖of the
current peer 𝑝𝑖that have not yet been visited (Line 15). It
then calculates relevance scores Sfor all neighbors based
on their historical expertise Eand the extracted topics T
(Line 16). This allows the algorithm to prioritize neighbors
that have previously demonstrated expertise in the topics
relevant to the current query. The algorithm selects the top
𝑘neighbors (P𝑛𝑒𝑥𝑡) with the highest relevance scores (Line
17). These neighbors are then added to the visited set Vand
enqueued inQwith an incremented hop count (Lines 18-21).
If the algorithm exhausts all possible paths within the de-
fined hop limit without finding sufficiently relevant knowl-
edge, it returns an empty set (Line 23). The parameters 𝐻𝑚𝑎𝑥
and𝑘are crucial for balancing the exploration and exploita-
tion of the network, with 𝐻𝑚𝑎𝑥controlling the search depth
and𝑘determining the breadth of the search at each hop. By
dynamically caching peer expertise and using this informa-
tion to guide the random walk, TARW significantly improves
the efficiency and effectiveness of knowledge discovery in
DRAG, while enabling the system to adapt to the evolving
knowledge landscape of the distributed network.
3.3 Further Discussion
Beyond the core mechanics of DRAG and the TARW algo-
rithm, several critical aspects that may impact the feasibility
of the proposed model are discussed in this section.
Privacy: While DRAG inherently addresses some privacy
concerns by eliminating the centralized knowledge base, it
introduces new challenges that need careful consideration.
First, the user-filtered knowledge snippets shared between
peers may still inadvertently leak sensitive information. Ad-
vanced techniques, such as differential privacy [ 9,36], could
be integrated to further anonymize the shared knowledge
and protect against potential inference attacks. Second, the
TARW algorithm itself could be vulnerable to deanonymiza-
tion attacks if an adversary can observe the network traffic
and infer the query topics and the expertise of individual
peers. Defenses against such attacks might involve adding
noise to the topic extraction process, employing mix net-
works to obfuscate the routing paths, or implementing ac-
cess control mechanisms to restrict peer visibility. While this
risk exists in general, deploying DRAG within a controlled
environment, such as a consortium network of hospitals
and clinics, can offer a layer of enhanced privacy. In suchscenarios, all participating parties sign privacy-preserving
agreements that establish clear guidelines for data sharing
and usage, reducing the risk of malicious actors within the
network. Furthermore, the defined roles and responsibilities
within the consortium allow for better auditing and account-
ability, facilitating the detection and prevention of privacy
breaches.
Scalability: DRAG offers promising scalability benefits
compared to centralized RAG due to its distributed architec-
ture. The computational load is distributed across multiple
peers, avoiding the bottleneck of a central server. However,
the scalability of DRAG depends on the efficiency of the
TARW algorithm and the connectivity of the P2P network.
In densely connected networks, TARW can quickly discover
relevant peers. However, in sparsely connected networks,
the algorithm may require more hops to reach the desired
knowledge, increasing latency and network traffic. Employ-
ing network clustering techniques or implementing super-
peers can help improve connectivity and reduce the average
path length. Fault tolerance is another key advantage of
DRAG. Since the knowledge is distributed across multiple
peers, the system can continue to function even if some peers
become unavailable. The robustness of DRAG is further en-
hanced by the result caching mechanism, where knowledge
snippets are replicated across multiple peers after queries
to provide redundancy. The choice of caching strategy de-
pends on the trade-off between storage overhead and fault
tolerance requirements.
Knowledge Reliability: A major challenge in DRAG is
ensuring the reliability and consistency of the shared knowl-
edge. Unlike centralized knowledge bases that can be care-
fully curated and maintained, the knowledge in DRAG is dis-
tributed across multiple peers with varying levels of expertise
and data quality. This can lead to issues such as misinforma-
tion, stale knowledge, and biased information. To address
these challenges, DRAG can incorporate mechanisms for
knowledge validation and ranking. Peers can rate the qual-
ity and relevance of the knowledge snippets they receive,
and this feedback can be used to weigh the contributions of
different peers. Additionally, consensus mechanisms can be
employed to ensure that conflicting knowledge snippets are
resolved in a consistent manner. The trustworthiness of indi-
vidual peers is also an important factor. Reputation systems
can be used to track the historical performance of peers and
reward those who consistently provide high-quality knowl-
edge.
Incentive Mechanisms: The success of DRAG depends
on the willingness of peers to contribute their knowledge
and computational resources to the network. However, peers
may be reluctant to share their resources if they do not re-
ceive adequate compensation. Therefore, incentive mecha-
nisms are needed to encourage participation and ensure the

Conference acronym ’XX, June 03–05, 2025, Woodstock, NY XXX et al.
long-term sustainability of the system [ 37]. Several incentive
mechanisms could be employed in DRAG. One approach
is to use a token-based system, where peers are rewarded
with tokens for contributing knowledge, answering queries,
and validating information. These tokens can then be ex-
changed for services within the network, such as priority
access to information or increased computational resources.
Another approach is to use a reputation-based system, where
peers with high reputations are given preferential treatment.
This can incentivize peers to maintain high-quality knowl-
edge and provide accurate responses. Finally, a knowledge
marketplace can be established, where peers can sell their
knowledge to other peers in exchange for payment. This can
create a financial incentive for peers to contribute valuable
knowledge to the network. The design of effective incentive
mechanisms is a complex task that requires careful consider-
ation of the trade-offs between participation, fairness, and
efficiency.
4 EXPERIMENTS
This section presents the experimental evaluation, cover-
ing the experimental setup, performance comparison, and
sensitivity analysis.
4.1 Experimental Setup
This section details the experimental setup used to evalu-
ate the performance of DRAG. DRAG is compared against
two baseline systems: CRAG (Centralized RAG) and NoRAG
(LLM only, without RAG). Within the DRAG framework,
the effectiveness of the proposed TARW algorithm is evalu-
ated against two classic resource search algorithms in P2P
networks: Random Walk (RW) and Flooding (FL).
Network Configuration: Peer-to-peer networks are sim-
ulated using the NetworkX1library in Python. Specifically,
the Barabási-Albert model [ 38] is utilized to mimic real-world
network topologies like the Internet or social media. Network
sizes are explored with varying numbers of peers, specifi-
cally 20,40,60,80, and 100. The default number of peers
was set to 20unless otherwise specified. The connectivity of
the network, representing the number of peers each peer is
directly connected to, was also varied across values of 2,4,
6, and 8, with a default connectivity of 4.
Algorithm Parameters: The TARW algorithm’s perfor-
mance is influenced by several parameters. The maximum
number of hops ( 𝐻𝑚𝑎𝑥) is fixed to 6. Different values for the
neighbor selection count ( 𝑘), which determines the number
of neighbors considered for forwarding the query in each
hop, are explored, specifically using values of 2,4,6, and
8, with a default value of 4. The relevance threshold ( 𝜃),
1https://networkx.orgwhich dictates the minimum relevance score required for a
knowledge snippet to be considered relevant, is set to 0.8.
Large Language Models: To evaluate the generalizability
of DRAG, three open-source LLMs of comparable size are
employed: Llama 3.2-3B2(by default), Gemma 2 2B3, and
Qwen2.5 3B4. This allows for assessing the performance
of DRAG across different model architectures and training
datasets.
Datasets: The performance of DRAG is evaluated using
three diverse datasets, each presenting unique challenges:
•MMLU5:A multiple-choice question answering bench-
mark designed to evaluate general knowledge of LLM
across a wide range of subjects.
•Medical Extended6:A synthetic dataset comprising
generated patient questions and corresponding doctor
answers on fabricated symptoms, diseases, and treat-
ments. This dataset simulates a healthcare knowledge
domain where expertise is paramount.
•News7:A dataset containing approximately 210,000
news headlines from HuffPost, spanning the years 2012
to 2022. In the experiments, the LLM is tasked with
identifying the author of a news article based solely
on its headline, requiring the retrieval of relevant con-
textual information.
Evaluation Metrics: The following metrics are used to
evaluate the performance of RAG:
•# Hops: The average number of hops taken by the
query to retrieve relevant knowledge. This measures
the efficiency of the knowledge retrieval process in the
distributed environment.
•# Messages: The average number of messages sent
during the knowledge retrieval process. This is a metric
for evaluating the communication overhead of DRAG.
•Hit Rate: The percentage of queries for which relevant
knowledge was successfully retrieved. This indicates
the overall effectiveness of the knowledge retrieval
process.
•EM (Exact Match): The percentage of predicted to-
kens that exactly match their corresponding ground
truth tokens, providing a strict measure of accuracy.
•F1 Score: The harmonic mean of precision and re-
call, providing a balanced measure of LLM-generated
output.
2https://ollama.com/library/llama3.2:3b
3https://ollama.com/library/gemma2:2b
4https://ollama.com/library/qwen2.5:3b
5https://huggingface.co/datasets/cais/mmlu
6https://huggingface.co/datasets/sarus-tech/medical_extended
7https://huggingface.co/datasets/heegyu/news-category-dataset

Distributed Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
MMLU Medical News
Dataset020406080F1 (%)DRAG-TARW
CRAGCRAG-0.7S
CRAG-0.5SCRAG-0.7T
CRAG-0.5T
Figure 2: Comparative F1 Scores of DRAG and Cen-
tralized RAG Variants Under Varying Knowledge
Base Completeness. CRAG-0.7S means the centralized
knowledge base contains 70%of knowledge snippets,
while CRAG-0.7T means it includes 70%of topics.
•Precision: The ratio of correctly predicted tokens to
the total number of predicted tokens, measuring the
quality of the generated output.
•Recall: The ratio of correctly predicted tokens to the
total number of ground truth tokens, measuring the
completeness of the generated output.
4.2 Performance Comparison
The experimental results, presented in Table 1, provide a
comparison of the proposed DRAG framework with TARW
against several baselines: NoRAG (no retrieval), Central-
ized RAG (CRAG), DRAG with Random Walk (DRAG-RW),
and DRAG with Flooding (DRAG-FL). A key finding is that
DRAG-TARW achieves performance comparable to CRAG
across all three datasets (MMLU, Medical Extended, and
News) in terms of Exact Match (EM), F1 Score, Precision, and
Recall, demonstrating that the distributed approach main-
tains accuracy without a centralized knowledge base. For
instance, on the MMLU dataset, DRAG-TARW achieves an
EM of 83.90%compared to CRAG’s 85.73%, and an F1 score of
83.92%compared to CRAG’s 85.75%. Critically, DRAG-TARW
significantly reduces communication overhead compared to
DRAG-FL, as evidenced by the lower number of messages
required for knowledge retrieval. On MMLU, DRAG-TARW
uses an average of 6.87messages compared to DRAG-FL’s
10.91, and on News, DRAG-TARW uses 7.82messages com-
pared to DRAG-FL’s 10.99. This highlights the efficiency
of the TARW algorithm in identifying relevant peers and
routing queries effectively within a distributed environment.
DRAG-RW performs substantially worse than all other RAG
variants, particularly in terms of Hit Rate, indicating thatundirected exploration of the network struggles to locate rel-
evant knowledge. For example, the Hit Rate of DRAG-RW on
MMLU is only 23.46%, compared to 98.11%for DRAG-TARW.
While DRAG-TARW achieves lower message counts than
DRAG-FL, there is a slight trade-off in accuracy, with DRAG-
FL generally exhibiting marginally higher EM, F1, Precision,
and Recall scores. The choice between DRAG-TARW and
DRAG-FL depends on the specific application context and
the relative importance of accuracy versus communication
efficiency. Finally, the NoRAG baseline clearly demonstrates
the necessity of retrieval augmentation, as the LLM’s per-
formance without external knowledge is significantly lower
than any RAG configuration, highlighting the benefits of
integrating external knowledge to enhance accuracy and
completeness. For example, on the MMLU dataset, NoRAG
achieves an F1 score of only 17.41%, demonstrating the sub-
stantial improvement provided by retrieval-augmented gen-
eration.
The experimental results presented in Figure 2 demon-
strate the performance of the DRAG framework against Cen-
tralized RAG and its variants with incomplete knowledge
bases. DRAG consistently achieves F1 scores comparable to
CRAG when a complete knowledge base is available. For
instance, on the Medical Extended dataset, DRAG achieves
an F1 score of 90.58%compared to CRAG’s 94.46%. Notably,
DRAG significantly outperforms CRAG when the central-
ized knowledge base is incomplete, either due to a reduc-
tion in the number of samples (CRAG-0.7S, CRAG-0.5S) or
a reduction in the number of topics covered (CRAG-0.7T,
CRAG-0.5T). On the MMLU dataset, when only 70%of the
samples are available (CRAG-0.7S), CRAG’s F1 score drops
to61.54%, significantly lower than DRAG’s 83.92%. This re-
silience to incomplete knowledge highlights a key advantage
of DRAG: its ability to leverage a distributed knowledge base
to compensate for gaps in any single node’s information. This
advantage is especially noticeable in real-world situations
when it is infeasible to maintain a complete and up-to-date
centralized knowledge base, particularly in light of privacy
concerns.
4.3 Sensitivity Analysis
Figure 3 presents a sensitivity analysis of the DRAG frame-
work to variations in network size, examining the impact
on both F1 score and message overhead. As the number of
peers increases from 20to100, DRAG-TARW experiences
a modest decrease in F1 score. Specifically, on the MMLU
dataset, the F1 score decreases from 83.92%to79.72%, while
on the Medical Extended dataset, it decreases from 90.58%
to86.66%. Despite this slight reduction in F1 score, DRAG-
TARW consistently outperforms DRAG-RW, which exhibits a
far more significant drop as the network grows. For example,

Conference acronym ’XX, June 03–05, 2025, Woodstock, NY XXX et al.
Table 1: Comparative Performance Analysis of DRAG Variants with Centralized and Non-Augmented Baselines.
Scheme LLM Dataset # Hops # Messages Hit Rate EM F1 Precision Recall
NoRAG Llama 3.2-3B MMLU - - - 17.30% 17.41% 17.38% 17.51%
CRAG Llama 3.2-3B MMLU - - 99.81% 85.73% 85.75% 85.74% 85.76%
DRAG-RW Llama 3.2-3B MMLU 5.13 5.36 23.46% 19.97% 19.97% 19.97% 19.97%
DRAG-FL Llama 3.2-3B MMLU 1.68 10.91 100.00% 85.43% 85.45% 85.44% 85.46%
DRAG-TARW Llama 3.2-3B MMLU 1.72 6.87 98.11% 83.90% 83.92% 83.92% 83.93%
NoRAG Llama 3.2-3B Medical - - - 0.00% 32.28% 29.39% 40.56%
CRAG Llama 3.2-3B Medical - - 99.81% 82.80% 94.46% 99.49% 93.43%
DRAG-RW Llama 3.2-3B Medical 4.94 5.22 27.62% 19.66% 24.32% 25.81% 24.02%
DRAG-FL Llama 3.2-3B Medical 1.62 9.72 99.86% 77.78% 91.58% 96.56% 90.54%
DRAG-TARW Llama 3.2-3B Medical 1.88 8.82 98.67% 77.07% 90.58% 95.66% 89.51%
NoRAG Llama 3.2-3B News - - - 0.08% 0.67% 0.42% 3.25%
CRAG Llama 3.2-3B News - - 99.16% 69.07% 76.66% 80.08% 75.82%
DRAG-RW Llama 3.2-3B News 5.16 5.38 21.96% 28.97% 16.83% 17.56% 16.65%
DRAG-FL Llama 3.2-3B News 1.74 10.99 99.20% 68.95% 76.54% 79.95% 75.69%
DRAG-TARW Llama 3.2-3B News 1.88 7.82 96.86% 67.62% 74.73% 78.18% 73.86%
20 40 60 80 100
# Peers020406080F1 (%)
(a) Llama 3.2 3B - MMLU20 40 60 80 100
# Peers20406080F1 (%)
(b) Llama 3.2 3B - Medical20 40 60 80 100
# Peers020406080F1 (%)
(c) Llama 3.2 3B - News
20 40 60 80 100
# Peers2040# Messages
(d) Llama 3.2 3B - MMLU20 40 60 80 100
# Peers10203040# Messages
(e) Llama 3.2 3B - Medical20 40 60 80 100
# Peers1020304050# Messages
(f) Llama 3.2 3B - NewsDRAG-TARW DRAG-RW DRAG-FL
Figure 3: Impact of network size on F1 score (first row) and message overhead (second row) in DRAG.
on MMLU, the F1 score of DRAG-RW plummets from 19.97%
at20peers to just 3.57%at100peers. Simultaneously, while
the average number of messages for both DRAG-TARW and
DRAG-FL increases with network size, DRAG-TARW consis-
tently maintains a significantly lower message overhead than
DRAG-FL. Quantitatively, at 100peers on MMLU, DRAG-
TARW requires only 27.67messages compared to DRAG-FL’s
53.34, and on News, it requires 23.71messages versus 48.83for DRAG-FL, representing approximately a 50%reduction in
message overhead. This represents a substantial advantage in
resource-constrained edge environments. In addition, these
results suggest a potential trade-off between network size
and individual peer performance in DRAG, and emphasize
the need for further optimization of the knowledge searching
algorithm for larger networks.

Distributed Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
2000 4000 6000 8000 10000
# Queries101520253035# Messages
20 40 60 80 100
Figure 4: Convergence of average message counts with
increasing query number in DRAG-TARW. Each line
represents a different network size, with 20,40,60,80,
and 100indicating the number of peers in the network.
20 40 60 80 100
# Peer020406080F1 (%)
(a)20 40 60 80 100
# Peer0102030# Messages
(b)2 4 6 8
Figure 5: Impact of peer connectivity on performance
(a) and communication cost (b) in DRAG-TARW. Peer
connectivity is defined as the number of edges attached
from a new node to existing nodes ( 2,4,6, and 8), con-
sistent with the Barabási-Albert model.
MMLU Medical News
Dataset020406080F1 (%)
(a)MMLU Medical News
Dataset0246810# Messages
(b)Llama 3.2 3B Gemma 2 2B Qwen2.5 3B
Figure 6: Influence of large language model on DRAG-
TARW performance (a) and communication cost (b).
20 40 60 80 100
# Peers020406080F1 (%)
(a)20 40 60 80 100
# Peers0102030# Messages
(b)2 4 6 8Figure 7: Influence of the neighbor selection count,
𝑘, on performance (a) and communication cost (b) in
DRAG-TARW. 𝑘represents the number of query neigh-
bors considered for forwarding the query in each hop
(𝑘=2,4,6,8).
Figure 4 examines the relationship between the number
of queries processed and the average number of messages
required by DRAG-TARW on the MMLU dataset, revealing
a beneficial convergence effect. The data demonstrates that
as the number of queries increases, the average number of
messages per query gradually decreases, eventually stabiliz-
ing at a consistent level. For example, in a network with 100
peers, the average number of messages starts at 34.73for
the first 500queries but converges to 27.67after processing
10000 queries, representing a significant reduction of over
20%. This convergence is observed across all network sizes
(20to100peers), indicating that DRAG-TARW becomes in-
creasingly efficient as peers progressively cache more topics
and accumulate knowledge about their neighbors’ expertise
in specific topics over time. This suggests that DRAG-TARW
is particularly well-suited for scenarios with sustained query
loads or prior knowledge of neighbors’ expertise.
Figure 5 presents a sensitivity analysis on the impact of
peer connectivity within the DRAG-TARW framework on
the MMLU dataset. The results reveal that increasing peer
connectivity does not consistently lead to either improved
F1 scores or reduced message overhead. Despite varying the
number of connections per peer from 2to8, the F1 score
remains relatively stable, fluctuating between approximately
80%and 83%. Moreover, while one might expect increased
connectivity to facilitate more direct routing and fewer mes-
sages, the average number of messages could increase with
the number of attached edges, particularly in larger net-
works. For instance, in a network of 100peers, increasing
the connectivity from 2to8results in an increase in the
average number of messages from 26.64to29.37. This lack
of consistent improvement in both F1 and message count
likely stems from the inherent randomness in knowledge
distribution across the peer-to-peer network and the com-
plex interplay with the underlying network topology. With

Conference acronym ’XX, June 03–05, 2025, Woodstock, NY XXX et al.
randomly distributed knowledge, more connections do not
guarantee access to more relevant knowledge, and the spe-
cific arrangement of connections can influence routing paths
in unpredictable ways. These results highlight the challenges
of optimizing network topology in DRAG systems and under-
score the importance of considering the interplay between
knowledge distribution, network structure, and routing al-
gorithm design.
Figure 6 presents a sensitivity analysis evaluating the im-
pact of different LLMs on the performance and communi-
cation cost of DRAG-TARW across the MMLU, Medical Ex-
tended, and News datasets. The results underscore that the
choice of LLM significantly influences the overall perfor-
mance of DRAG-TARW, as evidenced by the variability in
F1 scores across different models for a given dataset. For
instance, on the MMLU dataset, Llama 3.2 3B achieves an
F1 score of 83.92%, while Gemma 2 2B scores 77.13% and
Qwen2.5 3B achieves 82.86%. This performance variation can
be attributed, in part, to the LLM’s ability to effectively ex-
tract the underlying topic from the query, which directly im-
pacts the efficiency of the knowledge search process within
DRAG-TARW. However, the data also reveals that a higher
F1 score does not necessarily correlate with a higher aver-
age number of messages. For example, on the News dataset,
Llama 3.2 3B, despite achieving a moderate F1 score ( 74.73%),
requires 7.82messages, while Gemma 2 2B, with a slightly
higher F1 score ( 76.12%), only requires 6.90messages. This
decoupling of F1 score and message overhead is likely due to
a combination of factors, including the inherent randomness
in the network topology and the stochastic nature of LLM
generation itself.
Figure 7 presents a sensitivity analysis examining the im-
pact of the number of query neighbors ( 𝑘) on the perfor-
mance and communication cost of DRAG-TARW. The results
demonstrate a clear trade-off between F1 score and message
overhead. Increasing the number of query neighbors initially
leads to a significant improvement in F1 score, particularly
when transitioning from 𝑘=2to𝑘=4. For instance, in a
network of 20peers, increasing 𝑘from 2to4results in a
jump in F1 score from 56.27%to83.92%. However, beyond
𝑘=4, further increases in the number of query neighbors
yield only marginal improvements in F1 score, while sub-
stantially increasing the average number of messages. In a
network of 100peers, the F1 score increases from 79.72%at
𝑘=4to only 85.33%at𝑘=8, while the average number of
messages increases from 27.67to28.71. This suggests that
while considering more neighbors initially helps to broaden
the search and improve retrieval accuracy, the benefits di-
minish as𝑘increases, and the additional communication
overhead outweighs the gains. These findings underscore
the importance of carefully selecting the number of queryneighbors to optimize the balance between performance and
efficiency in DRAG-TARW.
5 CONCLUSION
This paper introduced Distributed Retrieval-Augmented Gen-
eration, a novel framework that enhances data privacy and
scalability by eliminating the need for a centralized knowl-
edge base and empowering data owners with control over
their information. DRAG leverages a Topic-Aware Random
Walk algorithm to efficiently discover relevant peers and
retrieve knowledge within a P2P network. Extensive experi-
ments across diverse datasets and LLMs demonstrated that
DRAG with TARW achieves near-centralized RAG perfor-
mance while significantly reducing communication overhead
compared to flooding-based approaches. The results high-
light DRAG as a promising scheme in privacy-sensitive and
resource-constrained edge environments. Future work will
focus on further refining the knowledge searching algorithm
to improve efficiency and scalability in very large and dy-
namic networks.
REFERENCES
[1]Y. Zheng, Y. Chen, B. Qian, X. Shi, Y. Shu, and J. Chen, “A review
on edge large language models: Design, execution, and applications,”
ACM Computing Surveys , 2024.
[2]G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, “Mobile edge
intelligence for large language models: A contemporary survey,” IEEE
Communications Surveys & Tutorials , 2025.
[3]B. Wang, Q. Xie, J. Pei, Z. Chen, P. Tiwari, Z. Li, and J. Fu, “Pre-trained
language models in biomedical domain: A systematic survey,” ACM
Computing Surveys , vol. 56, no. 3, pp. 1–52, 2023.
[4]X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer, H. Cao, and A. C.
Knoll, “Vision language models in autonomous driving: A survey and
outlook,” IEEE Transactions on Intelligent Vehicles , 2024.
[5]Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto,
and P. Fung, “Survey of hallucination in natural language generation,”
ACM computing surveys , vol. 55, no. 12, pp. 1–38, 2023.
[6]Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang,
“Retrieval-augmented generation for large language models: A survey, ”
arXiv preprint arXiv:2312.10997 , 2023.
[7]W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li,
“A survey on rag meeting llms: Towards retrieval-augmented large
language models,” in Proceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining , 2024, pp. 6491–6501.
[8]G. Zyskind, T. South, and A. Pentland, “Don’t forget private retrieval:
distributed private similarity search for large language models,” in Pro-
ceedings of the Fifth Workshop on Privacy in Natural Language Process-
ing. Bangkok, Thailand: Association for Computational Linguistics,
Aug. 2024, pp. 7–19.
[9]T. Koga, R. Wu, and K. Chaudhuri, “Privacy-preserving retrieval
augmented generation with differential privacy,” arXiv preprint
arXiv:2412.04697 , 2024.
[10] A. Muhamed, P. Thaker, M. T. Diab, and V. Smith, “Cache me if you can:
The case for retrieval augmentation in federated learning,” in Privacy
Regulation and Protection in Machine Learning , 2024.

Distributed Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2025, Woodstock, NY
[11] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu, H. Liu, Y. Xing,
and J. Tang, “Mitigating the privacy issues in retrieval-augmented gen-
eration (rag) via pure synthetic data,” arXiv preprint arXiv:2406.14773 ,
2024.
[12] A. Purwar et al. , “Evaluating the efficacy of open-source llms in
enterprise-specific rag systems: A comparative study of performance
and scalability,” arXiv preprint arXiv:2406.11424 , 2024.
[13] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation, ” in Proceedings of the AAAI
Conference on Artificial Intelligence , vol. 38, no. 16, 2024, pp. 17 754–
17 762.
[14] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küt-
tler, M. Lewis, W.-t. Yih, T. Rocktäschel et al. , “Retrieval-augmented
generation for knowledge-intensive nlp tasks,” Advances in neural
information processing systems , vol. 33, pp. 9459–9474, 2020.
[15] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval aug-
mented language model pre-training,” in International conference on
machine learning . PMLR, 2020, pp. 3929–3938.
[16] V. Karpukhin, B. Oguz, S. Min, L. Wu, S. Edunov, D. Chen, and W.-t.
Yih, “Dense passage retrieval for open-domain question answering,”
inProceedings of the 2020 Conference on Empirical Methods in Natu-
ral Language Processing (EMNLP) . Association for Computational
Linguistics, Nov. 2020, pp. 6769–6781.
[17] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning to
retrieve, generate, and critique through self-reflection,” in The Twelfth
International Conference on Learning Representations , 2023.
[18] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and
J. Larson, “From local to global: A graph rag approach to query-focused
summarization,” arXiv preprint arXiv:2404.16130 , 2024.
[19] Y. Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, and L. Zhao, “Grag: Graph
retrieval-augmented generation, ” arXiv preprint arXiv:2405.16506 , 2024.
[20] C. Mavromatis and G. Karypis, “Gnn-rag: Graph neural retrieval for
large language model reasoning,” arXiv preprint arXiv:2405.20139 , 2024.
[21] Y. Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi, and
B. Catanzaro, “Rankrag: Unifying context ranking with retrieval-
augmented generation in llms,” Advances in Neural Information Pro-
cessing Systems , vol. 37, pp. 121 156–121 184, 2025.
[22] T. Shnitzer, A. Ou, M. Silva, K. Soule, Y. Sun, J. Solomon, N. Thompson,
and M. Yurochkin, “Large language model routing with benchmark
datasets,” in First Conference on Language Modeling , 2024.
[23] F. Mu, L. Zhang, Y. Jiang, W. Li, Z. Zhang, P. Xie, and F. Huang, “Un-
supervised query routing for retrieval augmented generation,” arXiv
preprint arXiv:2501.07793 , 2025.
[24] X. Zhang, M. Wang, X. Yang, D. Wang, S. Feng, and Y. Zhang, “Hi-
erarchical retrieval-augmented generation model with rethink for
multi-hop question answering,” arXiv preprint arXiv:2408.11875 , 2024.
[25] S. Wang, Y. Fang, Y. Zhou, X. Liu, and Y. Ma, “Archrag: Attributed
community-based hierarchical retrieval-augmented generation,” arXiv
preprint arXiv:2502.09891 , 2025.
[26] S. Wang, E. Khramtsova, S. Zhuang, and G. Zuccon, “Feb4rag: Evaluat-
ing federated search in the context of retrieval augmented generation, ”
inProceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval , 2024, pp. 763–773.
[27] P. Addison, M.-T. H. Nguyen, T. Medan, M. T. Manzari, B. McElrone,
L. Lalwani, A. More, S. Sharma, H. R. Roth, I. Yang et al. , “C-fedrag: A
confidential federated retrieval-augmented generation system,” arXiv
preprint arXiv:2412.13163 , 2024.
[28] S. Zeng, J. Zhang, P. He, Y. Xing, Y. Liu, H. Xu, J. Ren, S. Wang, D. Yin,
Y. Chang et al. , “The good and the bad: Exploring privacy issues in
retrieval-augmented generation (rag),” arXiv preprint arXiv:2402.16893 ,
2024.[29] C. Jiang, X. Pan, G. Hong, C. Bao, and M. Yang, “Rag-thief: Scalable
extraction of private data from retrieval-augmented generation ap-
plications with agent-based attacks,” arXiv preprint arXiv:2411.14110 ,
2024.
[30] M. Anderson, G. Amit, and A. Goldsteen, “Is my data in your retrieval
database? membership inference attacks against retrieval augmented
generation,” arXiv preprint arXiv:2405.20446 , 2024.
[31] M. Liu, S. Zhang, and C. Long, “Mask-based membership infer-
ence attacks for retrieval-augmented generation,” arXiv preprint
arXiv:2410.20142 , 2024.
[32] Y. Li, G. Liu, C. Wang, and Y. Yang, “Generating is believing: Member-
ship inference attacks against retrieval-augmented generation,” arXiv
preprint arXiv:2406.19234 , 2024.
[33] J. Xue, M. Zheng, Y. Hu, F. Liu, X. Chen, and Q. Lou, “Badrag: Identify-
ing vulnerabilities in retrieval augmented generation of large language
models,” arXiv preprint arXiv:2406.00083 , 2024.
[34] D. Zhao, “Frag: Toward federated vector database management for col-
laborative and secure retrieval-augmented generation,” arXiv preprint
arXiv:2410.13272 , 2024.
[35] P. Lv, M. Sun, H. Wang, X. Wang, S. Zhang, Y. Chen, K. Chen, and
L. Sun, “Rag-wm: An efficient black-box watermarking approach
for retrieval-augmented generation of large language models,” arXiv
preprint arXiv:2501.05249 , 2025.
[36] N. Grislain, “Rag with differential privacy,” arXiv preprint
arXiv:2412.19291 , 2024.
[37] C. Xu, Y. Qu, T. H. Luan, P. W. Eklund, Y. Xiang, and L. Gao, “A light-
weight and attack-proof bidirectional blockchain paradigm for internet
of things,” IEEE Internet of Things Journal , vol. 9, no. 6, pp. 4371–4384,
2021.
[38] R. Albert, H. Jeong, and A.-L. Barabási, “Error and attack tolerance of
complex networks,” nature , vol. 406, no. 6794, pp. 378–382, 2000.
Received 20 February 2007; revised 12 March 2009; accepted 5 June
2009
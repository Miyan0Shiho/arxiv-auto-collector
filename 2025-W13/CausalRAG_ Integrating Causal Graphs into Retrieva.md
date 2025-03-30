# CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation

**Authors**: Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, Vipin Chaudhary

**Published**: 2025-03-25 17:43:08

**PDF URL**: [http://arxiv.org/pdf/2503.19878v1](http://arxiv.org/pdf/2503.19878v1)

## Abstract
Large language models (LLMs) have revolutionized natural language processing
(NLP), particularly through Retrieval-Augmented Generation (RAG), which
enhances LLM capabilities by integrating external knowledge. However,
traditional RAG systems face critical limitations, including disrupted
contextual integrity due to text chunking, and over-reliance on semantic
similarity for retrieval. To address these issues, we propose CausalRAG, a
novel framework that incorporates causal graphs into the retrieval process. By
constructing and tracing causal relationships, CausalRAG preserves contextual
continuity and improves retrieval precision, leading to more accurate and
interpretable responses. We evaluate CausalRAG against regular RAG and
graph-based RAG approaches, demonstrating its superiority across several
metrics. Our findings suggest that grounding retrieval in causal reasoning
provides a promising approach to knowledge-intensive tasks.

## Full Text


<!-- PDF content starts -->

CausalRAG: Integrating Causal Graphs into Retrieval-Augmented
Generation
Nengbo Wang1,2, Xiaotian Han1, Jagdip Singh2, Jing Ma1, Vipin Chaudhary1
1Department of Computer and Data Sciences, Case Western Reserve University
2Department of Design and Innovation, Case Western Reserve University
nengbo.wang@case.edu
Abstract
Large language models (LLMs) have revolu-
tionized natural language processing (NLP),
particularly through Retrieval-Augmented Gen-
eration (RAG), which enhances LLM capabil-
ities by integrating external knowledge. How-
ever, traditional RAG systems face critical limi-
tations, including disrupted contextual integrity
due to text chunking, and over-reliance on se-
mantic similarity for retrieval. To address these
issues, we propose CausalRAG , a novel frame-
work that incorporates causal graphs into the
retrieval process. By constructing and tracing
causal relationships, CausalRAG preserves con-
textual continuity and improves retrieval preci-
sion, leading to more accurate and interpretable
responses. We evaluate CausalRAG against reg-
ular RAG and graph-based RAG approaches,
demonstrating its superiority across several
metrics. Our findings suggest that grounding re-
trieval in causal reasoning provides a promising
approach to knowledge-intensive tasks.
1 Introduction
The rapid advancements in large language models
(LLMs) have revolutionized the field of natural lan-
guage processing (NLP), enabling a wide range of
applications (Anthropic, 2024; Google, 2024; Ope-
nAI, 2024). However, their reliance on pre-trained
knowledge limits their ability to integrate and rea-
son over dynamically updated external information,
particularly in knowledge intensive domains such
as academic research. Retrieval-Augmented Gen-
eration (RAG) has emerged as a promising frame-
work to address this limitation (Lewis et al., 2021),
combining retrieval mechanisms with generative
capabilities to enhance contextual understanding
and response quality.
Recent research has focused on improving RAG
along two primary directions: 1) enhancing re-
trieval efficiency and integration mechanisms by de-
signing more adaptive and dynamic retrieval frame-
works (Gan et al., 2024; Ravuru et al., 2024; Zhanget al., 2024a); 2) improving the representation of
external knowledge to facilitate retrieval and rea-
soning, with graph-based RAGs being a dominant
approach (Edge et al., 2024; Guo et al., 2024; Potts,
2024). Despite these advancements, existing RAG
architectures still face critical limitations that im-
pact retrieval quality and response accuracy, pri-
marily due to three key issues: 1) disruption of
contextual integrity caused by the text chunking
design; 2) reliance on semantic similarity rather
than causal relevance for retrieval; and 3) a lack of
accuracy in selecting truly relevant documents.
Through a combination of theoretical analysis
and empirical evaluation, we rethink the limitations
of current RAG systems by introducing a novel
perspective based on context recall and precision
metrics. Our findings reveal that both regular and
graph-based RAGs struggle not only to retrieve
truly grounded context but also to accurately dis-
cern the relationship between retrieved content and
the user query. We identify this fundamental issue
as one primary reason why LLMs in RAG frame-
works often generate seemingly relevant yet shal-
low responses that lack essential details .
To address these gaps, we introduce Causal-
RAG , a novel RAG framework that integrates causal
graphs to enhance retrieval accuracy and reason-
ing performance. Unlike regular and graph-based
RAGs, CausalRAG constructs a causal graph from
uploaded documents, preserving contextual rela-
tionships while capturing cause-effect dependen-
cies. By ensuring that retrieved documents are both
relevant and causally grounded, CausalRAG en-
ables the generation of more contextually rich
and causally detailed responses . This approach
not only improves retrieval effectiveness but also
mitigates hallucinations and enhances answer faith-
fulness.
We evaluate CausalRAG on datasets from di-
verse domains and across varying context lengths,
comparing its performance with regular RAG and
1arXiv:2503.19878v1  [cs.CL]  25 Mar 2025

GraphRAG, a graph-based RAG framework from
Microsoft (Edge et al., 2024). Our experiments as-
sess performance across three key metrics: answer
faithfulness, context recall, and context precision.
Results demonstrate that CausalRAG achieves su-
perior performance across different contexts. Addi-
tionally, we conduct a case study and a parameter
analysis to further examine our framework, ana-
lyzing and providing insights that contribute to on-
going research in RAG. The contributions of this
work are threefold:
•We systematically identify the inherent lim-
itations of RAG’s retrieval process through
analytical and experimental study. More im-
portantly, we uncover one major reason why
LLMs in RAG tend to generate superficial,
generalized answers that lack the grounded
details expected by users.
•We propose CausalRAG , a framework that en-
hances both retrieval and generation quality
by incorporating causality into the RAG, ef-
fectively addressing these limitations.
•Our work further mitigates hallucination is-
sues and significantly improves the inter-
pretability of AI systems. We summarize key
findings and insights in both retrieval and gen-
eration process, contributing to research in
RAG.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG enhances LLMs’ ability to handle knowledge-
intensive tasks by integrating external knowledge
retrieval (Lewis et al., 2021). Existing research pri-
marily advances RAG along two key dimensions:
1) improving retrieval efficiency and integration
mechanisms; 2) enhancing the representation of
external knowledge to facilitate reasoning and in-
terpretability.
Optimizing retrieval flow and interaction. The
first stream focuses on improving the interaction
flow within the RAG system to enhance output
quality. Approaches have introduced pre-retrieval,
retrieval, and post-retrieval refinements to mitigate
redundancy and computational overhead (Wang
et al., 2024). Modular RAG architectures fur-
ther advance this by enabling iterative retrieval-
generation cycles, allowing dynamic interactions
between retrieval and content creation. For exam-
ple, CAiRE-COVID (Su et al., 2020) demonstratedthe effectiveness of iterative retrieval in multi-
document summarization, while some work (Feng
et al., 2023) extended this approach to multi-hop
question answering. Recent innovations include
METRAG (Gan et al., 2024), which integrates
LLM supervision to generate utility-driven retrieval
processes, and RAFT (Zhang et al., 2024a), which
trains models to disregard distractor documents
while improving citation accuracy through chain-
of-thought reasoning.
Structuring external knowledge for efficiency.
The second direction explores improved methods
for structuring external knowledge to achieve better
retrieval efficiency. For example, GraphRAG (Edge
et al., 2024), proposed by Microsoft, treats exter-
nal knowledge as interconnected nodes, capturing
causal and thematic relationships to enhance re-
trieval depth and reasoning. LightRAG introduced
a dual-level retrieval mechanism for incremental
knowledge updates (Guo et al., 2024). More re-
cently, Lazy GraphRAG has been developed as a
continuous version of GraphRAG, deferring com-
putationally expensive operations until query time
and leveraging lightweight indexing techniques for
efficiency (Potts, 2024). Despite these advance-
ments, ensuring the quality and relevance of re-
trieved documents remains a significant concern
in current RAG systems, as it directly impacts the
coherence of the final response (Gupta et al., 2024).
2.2 Causal Graphs and RAG
The combination of causal graphs and RAG has
emerged as a promising approach to enhancing
knowledge retrieval and reasoning. As causal-
ity provides a structural understanding of depen-
dencies within data, it enables more interpretable
and reliable AI outputs (Ma, 2024). Existing re-
search in this domain primarily advances causal
discovery with RAG and LLMs. For instance,
some work proposed an LLM-assisted breadth-
first search (BFS) method for full causal graph
discovery, significantly reducing time complexity
(Jiralerspong et al., 2024). Additionally, some fur-
ther introduced a correlation-to-causation inference
(Corr2Cause) task to evaluate LLMs’ ability to in-
fer causation from correlation, revealing their lim-
itations in generalization across different datasets
(Jin et al., 2024).
Despite these advancements, most studies focus
on utilizing RAG or LLMs for causal discovery or
causal effect estimation (Ma, 2024; Kıcıman et al.,
2024), whereas the direct integration of causal
2

graphs into RAG architectures remains largely
unexplored . Our work aims to be a pioneer in this
direction. A few existing studies have touched upon
this concept but differ in scope. One approach inte-
grates causal graphs within the LLM architecture
itself, structuring the transformer’s internal token
processing using causality rather than enhancing
RAG retrieval (Chen et al., 2024b). Another em-
ploys causal graphs in RAG systems but focuses
on the pre-retrieval stage and largely reduces the
core process into a single embedding model with-
out deeper exploration (Samarajeewa et al., 2024).
GraphRAG is a well-regarded and influential work
in this area, as it introduces a graph-based structure
into the RAG system, leveraging graph commu-
nity detection and summarization techniques for
retrieval (Edge et al., 2024). While it does not incor-
porate causality, it has significantly improved RAG
performance. Therefore, we adopt GraphRAG as
the baseline for our work.
In the following sections, we first analyze the
nature of regular RAG and grah-based RAGs from
a novel perspective, identifying their inherent limi-
tations. We then introduce a causal graph structure
to address these gaps within the RAG system and
present our framework— CausalRAG .
3 Why Regular RAG Fails in Providing
Accurate Responses
In this section, through both analytical and experi-
mental investigations, we identify three fundamen-
tal limitations of regular RAG and rethink its design
by examining its three core elements—user query,
retrieved context, and response—through a novel
perspective based on precision and recall.
3.1 Limitations of regular RAG
The first limitation arises from RAG’s common
practice of chunking texts into minimal units (as
illustrated in Figure 1a). This process disrupts the
natural linguistic and logical connections in the
original text. These connections are crucial for
maintaining contextual integrity, and if they are lost,
an alternative mechanism must be implemented to
restore them.
The second limitation lies in the semantic search
process. RAG typically retrieves the semantically
closest documents from a vector database based on
query similarity. However, in many cases, critical
information necessary for answering a query is not
semantically similar but rather causally relevant.A classic example is the relationship between di-
apers and beer—while they are not semantically
related, they may exhibit a causal connection in
real worlds. This limitation suggests that RAG’s
reliance on semantic similarity may lead to the re-
trieval of contextually irrelevant but superficially
related information.
The third limitation is that even when RAG re-
trieves a relevant context, this does not necessarily
guarantee an accurate response. To formalize this
issue, we used two key metrics: context recall and
context precision , defined as follows:
Context Recall =PN
i=1I(Ci∈R)
|R|(1)
HereRis the reference set of all relevant refer-
ence documents. Ciis the ithretrieved document.
I(Ci∈R)is an indicator function that returns 1 if
Cibelongs to the reference set R, otherwise 0. It
should always return 1 if no hallucination occurs
in the LLM.
Context Precision =PN
i=1IQ(Ci∈R)PN
i=1I(Ci∈R)(2)
HereIQ(Ci∈R)is an indicator function that
returns 1 if the context retrieved is causally related
to user’s query, otherwise 0.
Recall-Precision Perspective. Context recall mea-
sures how much of the correct contextual informa-
tion can be retrieved from the user’s uploaded docu-
ments given a query. While RAG primarily focuses
on retrieving related documents, adjusting retrieval
parameters to increase the number of retrieved doc-
uments can generally lead to higher recall in reality.
However, context precision presents a challenge.
It measures the proportion of retrieved documents
that are actually correct given the user query. As
discussed earlier, RAG’s reliance on semantic simi-
larity rather than causal relevance often leads to the
retrieval of superficially similar but logically irrel-
evant content. In summary, while RAG can recall
numerous answers from reference materials, the
proportion of correct responses remains low, ulti-
mately reducing its precision. This recall-precision
perspective provides a new lens to see limitations
of regular RAG.
3.2 Rethinking Graph-based RAGs
Applying this perspective, we can better understand
why Graph-based RAGs serve as improved variants
3

(a) Analysis of Limitations in regular RAG System (b) Experimental study on context recall and precisionUploaded 
DocumentsUploaded 
Documents
…… ……Chunk and 
embedding Vector Store
…… …… 
…… …… 
…… …… Query
…… ……QueryEmbeddingStore
Semantic search
Retrieved Context
…… ……RAG template:
{Query}  {Context}GenerationInputLimitations 1. 
Disrupted linguistic coherence
Limitations 2. 
Semantic search introduces biasLimitations 3. 
Bias amplification in generation
Figure 1: Analytical and experimental studies reveal limitations in regular RAG and GraphRAG. (a) identifies three
key retrieval and generation issues in regular RAG; (b) evaluates RAG via context precision and recall, showing
regular RAG excels in recall but lacks precision. GraphRAG improves precision but trades off some recall.
of RAG. By summarizing and ranking the impor-
tance of graph communities before retrieval, they
largely enhance the quality of retrieved context,
thereby improving context precision. However, it
only partially addresses the identified limitations,
as its summarization process does not entirely fil-
ter out irrelevant information. More critically, its
reliance on community-based summarization for
retrieval may adversely impact recall. Based on
these analytical insights, we further conducted an
experimental study to validate our analysis.
Experimental study. As shown in Figure 1(b),
we conducted an experimental study to empirically
verify our analysis. Specifically, we employed an
LLM to evaluate the performance of regular RAG
and GraphRAG using the RAG evaluation frame-
work Ragas (Es et al., 2023). It takes as input
the three key elements—user query, retrieved con-
text, and reference—along with predefined metric
definitions, and prompts the LLM to assign numer-
ical ratings. This LLM-based evaluation has been
widely used in recent RAG research and has also
been adopted in RAG evaluations (Samarajeewa
et al., 2024; Edge et al., 2024).
The results indicate that both global and local
versions of GraphRAG achieve higher context pre-
cision compared to regular RAG. However, its
recall performance remains unsatisfactory, even
slightly lower than that of RAG, due to its graph
community-based retrieval process.
By combining our precision-recall-based analyt-
ical analysis with empirical findings, we clearly
outline the inherent limitations of both RAG and
its graph-based extensions. In the next section, we
introduce our proposed framework, CausalRAG ,designed to address these issues.
4 Methodology
In this section, we introduce our novel frame-
work— CausalRAG —which integrates RAG with
causality to overcome the limitations of existing
RAG systems. Overall, CausalRAG utilizes a
graph-based approach to represent uploaded doc-
uments, enhancing robustness against discontinu-
ities caused by text chunking (as shown in Figure
2). More importantly, by expanding and tracing
nodes in the graph through causal relationships,
CausalRAG is able to retrieve a more diverse set
of contextual information while maintaining causal
groundedness. This enables CausalRAG to achieve
both high recall and strong precision. We now dis-
cuss each step in detail.
4.1 Indexing
At the outset, once the system receives the user’s
uploaded documents and query, we first index these
inputs into our vector database. For the uploaded
documents, we employ a text-based graph construc-
tion method to transform the text into a structured
graph before storing the nodes and edges in the
vector database. Specifically, we leverage an LLM
to construct the graph following the approach by
LangChain (Chase, 2022), where the LLM scans
the text to identify graph nodes and determine re-
lationships between them. Although LLM-based
graph construction has shown strong performance
(Chen et al., 2024a) and is widely adopted in graph-
based RAG research , we further validate our con-
structed graph using expert knowledge, which we
elaborate on in the case study. After construct-
4

LLMResponseUpload DocIndexing
Vector StoreAsk QueryText-based 
Graph 
Construction
Discovering & 
Estimating
Causal Path (Context)Embedding 
SearchGraph
Extended 
Nodes (s)
Query 
EmbeddingInitial 
Nodes (k)Causal GraphQuery
Causal Summary
(Grounded Context)User 
QueryRAG 
PromptUser 
Retrieving 
Context 
CausallyFigure 2: Overview of CausalRAG ’s architecture. Documents are indexed as graphs, and queries retrieve causally
related nodes. A causal summary is generated and combined with the query to ensure grounded responses.
ing and indexing the base graph, we embed the
user query into the vector database, preparing it for
subsequent search and matching. Notably, this in-
dexing process occurs independently of query time,
allowing for efficient retrieval during inference.
4.2 Discovering and Estimating Causal Paths
This is the first step during query time. We begin by
matching the user query to nodes in the graph based
on their embedding distance. The knodes with the
smallest distances are selected, representing the
most relevant information directly related to the
query. Notably, kis a tunable parameter—higher
values retrieve more relevant information at the
cost of increased computational complexity.
After selecting the initial knodes, we expand the
search along the base graph’s edges by a step size
ofs, thereby broadening the retrieved context. This
step is crucial as it preserves causal and relational
connections within the text, allowing CausalRAG
to retrieve more context while maintaining high
recall. The parameter scontrols the depth of ex-
pansion, where higher values lead to more diverse
information retrieval.
Once the relevant nodes and edges are collected,
we employ an LLM to identify and estimate causal
paths within them, constructing a refined causal
graph. As discussed, LLMs have demonstrated effi-
ciency in discerning and analyzing causal relation-
ships (Zhang et al., 2024b; Zhou et al., 2024). This
step ensures that CausalRAG prioritizes causally
relevant information, improving precision.Furthermore, the derived causal graph serves
two key purposes: 1) It preserves causally relevant
information that traditional retrieval methods strug-
gle to capture. More importantly, by adjusting the
parameter s, this approach can capture long-range
causal relationships within the text, particularly
when the text is lengthy; 2) It filters out seman-
tically related but causally irrelevant information.
Without this filtering, responses may contain unnec-
essary or even hallucinated content, compromising
answer faithfulness.
4.3 Retrieving Context Causally
After constructing the causal graph, we summa-
rize the retrieved information and generate a causal
summary. Notably, the input at this stage consists
of information that is not only highly relevant but
also causally grounded in the user’s query, ensuring
greater validity. This approach contrasts with tradi-
tional retrieval methods, which often rely purely on
semantic similarity and may retrieve contextually
related yet causally irrelevant information.
The causal summary is derived by tracing key
causal paths within the graph, prioritizing nodes
and relationships that contribute directly to answer-
ing the query. This ensures that the retrieved in-
formation maintains logical coherence and factual
consistency while filtering out spurious or weakly
related context. Additionally, by leveraging causal
dependencies, our method reduces the risk of re-
trieving semantically similar but misleading evi-
dence.
5

Once the causal summary is generated, it is com-
bined with the user query to construct a refined
prompt for CausalRAG . This structured final in-
put allows RAG to focus on reasoning through
causal relationships rather than merely aggregating
loosely related text spans.
5 Experiment
To evaluate the effectiveness of CausalRAG , we
conduct a series of experiments comparing it with
regular RAG and GraphRAG baselines. Our evalu-
ation spans multiple datasets, retrieval settings, and
performance metrics, ensuring a comprehensive
analysis of retrieval quality and answer faithfulness.
We systematically explore different parameter set-
tings and retrieval strategies to assess their impact
on model performance.
5.1 Experimental Setup
Parameters & Baselines. We set the default pa-
rameters for CausalRAG ask=s= 3and use the
same kvalue for GraphRAG’s community-based
retrieval and regular RAG’s document retrieval to
ensure fairness. We evaluate four RAG variants:
regular RAG (Lewis et al., 2021), GraphRAG with
local and global search (Edge et al., 2024), and our
proposed CausalRAG . Local search retrieves from
raw document graphs, making it ideal for passage-
specific queries, while global search relies on graph
community summaries for broader context under-
standing. We use GPT-4o-mini as the base LLM
for all models.
Datasets. We evaluate on publicly available re-
search papers from various domains, ranging from
100 to around 17,000 tokens, reflecting typical user-
uploaded documents. An LLM will generate n
grounded questions per document, ensuring they
are explicitly answerable. We use the Ragas frame-
work (Es et al., 2023) and assess models on three
key metrics: answer faithfulness, context precision,
and context recall. Faithfulness measures factual
consistency on a scale from 0 to 1, with higher
scores indicating better alignment with reference
documents.
5.2 Performance Comparison
To comprehensively evaluate the effectiveness
ofCausalRAG , we conducted experiments on a
diverse dataset comprising abstracts from 100
randomly selected research papers using Ope-
nAlex(Priem et al., 2022). All evaluations wereperformed using the OpenAlex public academic
dataset, which spans multiple disciplines, includ-
ing applied mathematics, art history, library sci-
ence, and psychology, ensuring a comprehensive
assessment across different domains. For each
document, we constructed graph and generated
n= 5 questions per paper for assessing retrieval
and response quality. As mentioned, we then
compared CausalRAG ’s performance against three
baselines—regular RAG, GraphRAG-Local, and
GraphRAG-Global—across three key metrics: an-
swer faithfulness, context recall, and context preci-
sion.
Figure 3 presents the comparative results. Over-
all,CausalRAG consistently outperforms all other
models across the three evaluation metrics, demon-
strating its ability to generate more factually accu-
rate responses while maintaining both high recall
and precision.
Answer Faithfulness. This metric measures the ex-
tent to which the generated response aligns with the
reference information, ensuring factual correctness.
As shown in Figure 3, regular RAG achieves rela-
tively strong performance in this metric, suggesting
that despite its reliance on purely semantic retrieval,
it still retrieves some relevant context. However,
GraphRAG-Local still slightly outperforms regular
RAG by leveraging structured graph-based retrieval
locally, which supports its grounded answer. Natu-
rally, GraphRAG-Global CausalRAG supported by
high-level community summary performs worse in
this metric. Lastly, CausalRAG holds its answer
faithfulness at a good level by retrieving causally
grounded context, reducing hallucinations, and en-
suring responses are not only relevant but also jus-
tified by retrieved evidence.
Context Recall. Context recall evaluates how
much of the correct reference information is re-
trieved during the RAG process. As expected, reg-
ular RAG exhibits high recall, given its broad re-
trieval approach, which tends to maximize the in-
clusion of potentially relevant content. GraphRAG-
Local and GraphRAG-Global show a slight reduc-
tion in recall due to their community-based sum-
marization, which, while improving retrieving pro-
cess, sacrifices some context diversity. CausalRAG ,
by contrast, strikes a balance between precision
and recall and its recall is still higher than regular
RAG. It is ensured by that retrieved information is
causally relevant, avoiding excessive retrieval of
loosely related but non-causal information.
Context Precision. Precision is where Causal-
6

Figure 3: Performance comparison of Regular RAG,
GraphRAG-Global, GraphRAG-Local, and CausalRAG
on the OpenAlex dataset across three key metrics: an-
swer faithfulness, context recall, and context precision.
RAG demonstrates the most significant advantage.
While regular RAG achieves relatively low pre-
cision—often retrieving semantically similar yet
contextually irrelevant content—GraphRAG-Local
and GraphRAG-Global significantly improve upon
this by leveraging graph structures to organize
knowledge more effectively. However, both still
struggle with retrieving causally relevant content
and are lower than CausalRAG on this metric. We
can see CausalRAG still maintain a great precision,
as it inherently filters retrieved information through
causal graph.
5.3 Case Study
We also conducted a case study to evaluate the
four RAG variants on a research paper with long
text, but testing them on its abstract, introduction,
and full text respectively. This approach ensures
content consistency while varying document length
(ranging from 255 to 16,475 tokens), allowing us
to assess the scalability of CausalRAG .
For each of the three text materials, we con-
structed a separate graph, tested n= 20 questions,
and evaluated all four RAG variants across three
key metrics. As mentioned earlier, we employed an
LLM-based approach for graph construction, and
the validity of these graphs—built from a business
school marketing paper—was further verified by
experts.
This case study also examines the retrieval dif-
ferences among regular RAG, GraphRAG, and our
proposed CausalRAG (as shown in Figure 4). In
this example, a user uploads a long paper and
asks: " How do different combinations of influence
tactics impact the likelihood of winning a sales
contract? " Expert validation confirms a clear an-
swer—salespeople use influence tactics such ascompliance-based and internalization-based strate-
gies to attract buyers’ attention, which inherently
increases their chances of securing a contract.
Analyzing the retrieval process, we observe that
regular RAG retrieves semantically related content
but fails to capture key information, leading to a
vague and uninformative response. GraphRAG cor-
rectly retrieves relevant nodes like attention, but
its community report process dilutes their impor-
tance by incorporating extraneous nodes, introduc-
ing bias despite factual grounding. In contrast,
CausalRAG retrieves the initial relevant nodes and
expands upon them by identifying causal paths es-
sential for answering the query, ensuring a precise
and causally grounded response.
While this example clarifies CausalRAG ’s inter-
nal process, our full experimental results (Figure
5) demonstrate its scalability. CausalRAG con-
sistently outperforms other RAG variants across
different document lengths. Additionally, regular
RAG performs well on shorter documents, even sur-
passing GraphRAG. However, as document length
increases—such as with full papers—GraphRAG
overtakes regular RAG, aligning with our analytical
analysis.
CausalRAG exhibits robustness across varying
context lengths by preserving access to discrete in-
formation while leveraging causal estimation for
grounded retrieval. More importantly, an additional
advantage—only partially reflected here—is that
with a larger step size s,CausalRAG can trace long
and complex causal relationships hidden in lengthy
documents, generating well-founded responses—a
capability that is challenging for existing RAG sys-
tems.
5.4 Parameter Study
We also tested the impact of different parameter
combinations of kandsonCausalRAG (as shown
in Figure 6). Using the average of our evaluation
metrics, we observe a consistent trend: the perfor-
mance of CausalRAG improves as kandsincrease.
Specifically, the performance rises from 0.534 at
k=s= 1 to 0.824 at k=s= 5, aligning with
intuitive expectations.
Notably, the improvement is more pronounced
when increasing kfrom 1 to 3, suggesting that
retrieving additional context enhances reasoning
quality. However, when k≥4, performance gains
become less significant, indicating possible satu-
ration due to information redundancy. Similarly,
while increasing sgenerally leads to better results,
7

Causal RAG -Retrieving process :
Retrieved Context (Causal Graph 
Summary):
“…… Causal Analysis Report ……
……. Major Causal Pathways: Buyer Attention 
→ Sales Contract Award ……”
Response:
“…… Key Combinations of Influence 
Tactics … Direct Persuasion and Relationship 
Building…
…… Mediating Role of Buyer Attention ……”
[Correct: Clearly identifies the mediating 
role of buyer attention and influence tactic 
combinations.
Analysis: Key context is correctly identified 
and preserved via the causal graph.]
GraphRAG -Retrieving process:
Retrieved Context (Community Report):
“…… Community Report: Buyer Attention, 
Influence Tactics, and Sales Contract Award…….
…… decision -makers can develop more effective 
strategies to engage buyers, influence their 
decisions…. ”
Response:
“…… The likelihood of winning a sales contract 
is influenced various influence tactics… 
persuasive communication, emotional appeals, 
and logical reasoning to secure a contract ……”
[Partly Correct: Salespeople attract buyer 
attention, leading to contract wins.
Analysis: GraphRAG recalled buyer attention 
correctly, but diverse concepts caused bias.]Regular RAG:
Retrieved context:
“...... tactics on the likelihood of sales contract 
award……
…… the concurrent use of sales influence 
tactics….“
Response:
“…… Different combinations of influence 
tactics can impact the likelihood of winning 
……”
[Wrong: Salespeople use specific influence 
tactics and mechanisms.
Analysis: Retrieved context is relevant but 
too general, lacking a concrete answer.]User Uploaded Document 
(Paper. Singh et al., 2020)
“sales negotiations are increasingly … ”
User’s question
(one of the questions generated by LLM)
“How do different combinations of 
influence tactics impact the likelihood 
of winning a sales contract?”Figure 4: Case Study comparing retrieving process between different RAG systems.
Figure 5: Performance comparison of four RAGs across
documents of varying lengths (Abstract, Introduction,
Full Paper) based on mean values of answer faithfulness,
context recall, and context precision.
Figure 6: Parameter study showing how different pa-
rameter choices (k and s) affect model performance.its effect diminishes at higher values of k, where
retrieval is already extensive.
These results suggest an optimal trade-off be-
tween performance and computational efficiency.
While the highest values ( k= 5, s= 5) yield the
best results, moderate settings such as k= 3, s= 3
still achieve competitive performance with lower
retrieval costs. Future work could explore adaptive
strategies to adjust these parameters dynamically
based on query complexity.
5.5 Conclusion and Future work
We introduced CausalRAG , a framework that inte-
grates causality into RAG to enhance performance.
Our analytical and experimental analysis identified
key limitations of regular RAG—loss of contex-
tual integrity, reliance on semantic similarity over
causal relevance, and low precision. By leverag-
ing causal graphs, CausalRAG retrieves causally
grounded context, outperforming baseline RAGs
across diverse text lengths and domains. Addition-
ally, it improves LLM explainability, enhances an-
swer groundedness, and reduces hallucinations in
generative responses. Future work includes bench-
marking CausalRAG on domain-specific tasks.
8

Limitations
While CausalRAG enhances retrieval effectiveness
by integrating causality, it has certain limitations.
First, constructing causal graphs from unstructured
text relies on LLM-based extraction, which may
introduce extra costs, particularly in complex or
ambiguous cases. Second, the computational cost
of expanding and analyzing causal paths increases
with larger documents, potentially impacting re-
trieval efficiency in extreme cases with a large num-
ber of tokens.
References
Anthropic. 2024. Claude 3.5 sonnet.
Harrison Chase. 2022. LangChain.
Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi Ni,
and Jieping Ye. 2024a. SAC-KG: Exploiting Large
Language Models as Skilled Automatic Construc-
tors for Domain Knowledge Graphs. arXiv preprint .
ArXiv:2410.02811 [cs].
Haotian Chen, Lingwei Zhang, Yiran Liu, and Yang Yu.
2024b. Rethinking the Development of Large Lan-
guage Models from the Causal Perspective: A Legal
Text Prediction Case Study. Proceedings of the AAAI
Conference on Artificial Intelligence , 38(19):20958–
20966.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From Local to Global:
A Graph RAG Approach to Query-Focused Summa-
rization. arXiv preprint . ArXiv:2404.16130 [cs].
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2023. Ragas: Automated eval-
uation of retrieval augmented generation. Preprint ,
arXiv:2309.15217.
Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Mao-
jin Yang, and Bing Qin. 2023. Retrieval-generation
synergy augmented large language models. Preprint ,
arXiv:2310.05149.
Chunjing Gan, Dan Yang, Binbin Hu, Hanxiao Zhang,
Siyuan Li, Ziqi Liu, Yue Shen, Lin Ju, Zhiqiang
Zhang, Jinjie Gu, Lei Liang, and Jun Zhou. 2024.
Similarity is not all you need: Endowing retrieval
augmented generation with multi layered thoughts.
Preprint , arXiv:2405.19893.
Google. 2024. Our next-generation model: Gemini 1.5.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. LightRAG: Simple and Fast
Retrieval-Augmented Generation. arXiv preprint .
ArXiv:2410.05779.Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A Comprehensive Survey of Retrieval-
Augmented Generation (RAG): Evolution, Current
Landscape and Future Directions. arXiv preprint .
Version Number: 1.
Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrin-
maya Sachan, Rada Mihalcea, Mona Diab, and Bern-
hard Schölkopf. 2024. Can Large Language Models
Infer Causation from Correlation? arXiv preprint .
ArXiv:2306.05836 [cs].
Thomas Jiralerspong, Xiaoyin Chen, Yash More, Vedant
Shah, and Yoshua Bengio. 2024. Efficient Causal
Graph Discovery Using Large Language Models.
arXiv preprint . ArXiv:2402.01207 [cs, stat].
Emre Kıcıman, Robert Ness, Amit Sharma, and Chen-
hao Tan. 2024. Causal Reasoning and Large Lan-
guage Models: Opening a New Frontier for Causality.
arXiv preprint . ArXiv:2305.00050.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2021. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. arXiv preprint .
ArXiv:2005.11401 [cs].
Jing Ma. 2024. Causal Inference with Large
Language Model: A Survey. arXiv preprint .
ArXiv:2409.09822 [cs].
OpenAI. 2024. Hello GPT-4o.
Brenda Potts. 2024. LazyGraphRAG sets a new stan-
dard for quality and cost.
Jason Priem, Heather Piwowar, and Richard Orr. 2022.
Openalex: A fully-open index of scholarly works,
authors, venues, institutions, and concepts. Preprint ,
arXiv:2205.01833.
Chidaksh Ravuru, Sagar Srinivas Sakhinana, and
Venkataramana Runkana. 2024. Agentic retrieval-
augmented generation for time series analysis.
Preprint , arXiv:2408.14484.
Chamod Samarajeewa, Daswin De Silva, Evgeny Os-
ipov, Damminda Alahakoon, and Milos Manic. 2024.
Causal Reasoning in Large Language Models using
Causal Graph Retrieval Augmented Generation. In
2024 16th International Conference on Human Sys-
tem Interaction (HSI) , pages 1–6. ISSN: 2158-2254.
Dan Su, Yan Xu, Tiezheng Yu, Farhad Bin Siddique,
Elham Barezi, and Pascale Fung. 2020. CAiRE-
COVID: A question answering and query-focused
multi-document summarization system for COVID-
19 scholarly information management. In Proceed-
ings of the 1st Workshop on NLP for COVID-19 (Part
2) at EMNLP 2020 , Online. Association for Compu-
tational Linguistics.
9

Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven
Zheng, Swaroop Mishra, Vincent Perot, Yuwei
Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang,
Chen-Yu Lee, and Tomas Pfister. 2024. Speculative
RAG: Enhancing Retrieval Augmented Generation
through Drafting. arXiv preprint . ArXiv:2407.08223
[cs].
Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gon-
zalez. 2024a. Raft: Adapting language model to
domain specific rag. Preprint , arXiv:2403.10131.
Yuzhe Zhang, Yipeng Zhang, Yidong Gan, Lina
Yao, and Chen Wang. 2024b. Causal Graph
Discovery with Retrieval-Augmented Generation
based Large Language Models. arXiv preprint .
ArXiv:2402.15301 [cs, stat].
Yu Zhou, Xingyu Wu, Beicheng Huang, Jibin Wu,
Liang Feng, and Kay Chen Tan. 2024. Causal-
Bench: A Comprehensive Benchmark for Causal
Learning Capability of LLMs. arXiv preprint .
ArXiv:2404.06349 [cs].
10
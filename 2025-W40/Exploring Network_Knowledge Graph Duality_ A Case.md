# Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis

**Authors**: Evan Heus, Rick Bookstaber, Dhruv Sharma

**Published**: 2025-10-01 17:02:14

**PDF URL**: [http://arxiv.org/pdf/2510.01115v1](http://arxiv.org/pdf/2510.01115v1)

## Abstract
Large Language Models (LLMs) struggle with the complex, multi-modal, and
network-native data underlying financial risk. Standard Retrieval-Augmented
Generation (RAG) oversimplifies relationships, while specialist models are
costly and static. We address this gap with an LLM-centric agent framework for
supply chain risk analysis. Our core contribution is to exploit the inherent
duality between networks and knowledge graphs (KG). We treat the supply chain
network as a KG, allowing us to use structural network science principles for
retrieval. A graph traverser, guided by network centrality scores, efficiently
extracts the most economically salient risk paths. An agentic architecture
orchestrates this graph retrieval alongside data from numerical factor tables
and news streams. Crucially, it employs novel ``context shells'' -- descriptive
templates that embed raw figures in natural language -- to make quantitative
data fully intelligible to the LLM. This lightweight approach enables the model
to generate concise, explainable, and context-rich risk narratives in real-time
without costly fine-tuning or a dedicated graph database.

## Full Text


<!-- PDF content starts -->

Exploring Network-Knowledge Graph Duality: A Case Study in
Agentic Supply Chain Risk Analysis
Evan Heus∗
University of California, Berkeley
Berkeley, CA, USA
MSCI
New York, NY, USA
evheus@berkeley.eduRick Bookstaber
MSCI
New York, NY, USA
rick.bookstaber@msci.comDhruv Sharma
MSCI
New York, NY, USA
dhruv.sharma@msci.com
Abstract
Large Language Models (LLMs) struggle with the complex, multi-
modal, and network-native data underlying financial risk. Standard
Retrieval-Augmented Generation (RAG) oversimplifies relation-
ships, while specialist models are costly and static. We address this
gap with an LLM-centric agent framework for supply chain risk
analysis. Our core contribution is to exploit the inherent duality
between networks and knowledge graphs (KG). We treat the supply
chain network as a KG, allowing us to use structural network sci-
ence principles for retrieval. A graph traverser, guided by network
centrality scores, efficiently extracts the most economically salient
risk paths. An agentic architecture orchestrates this graph retrieval
alongside data from numerical factor tables and news streams. Cru-
cially, it employs novel “context shells” — descriptive templates that
embed raw figures in natural language — to make quantitative data
fully intelligible to the LLM. This lightweight approach enables
the model to generate concise, explainable, and context-rich risk
narratives in real-time without costly fine-tuning or a dedicated
graph database.
1 Introduction
Large language models (LLMs) are increasingly embedded in critical,
high-stakes decision processes, from medical diagnostic triage [ 3] to
financial advice[ 12]. However, most domain deployments still rely
onfine -tuninga specialist model on a hand -curated corpus or on
straightforward retrieval augmented generation (RAG) that proxies
relationships with vector distance. Both paths leave value on the
table for supply chain risk analysis: the former is costly to update
with current events and freezes knowledge at training time, while
the latter ignores the richnetwork semanticshidden in structured
data. In addition, they typically operate on a single modality, despite
the fact that domain -specific tasks require seamless reasoning over
text, tables, graphs, and time-series data.
We address this gap with an LLM -centric system that usesnet-
work science principles to uncover and expose semantically mean-
ingful supply -chain paths, for example,Apple ↔Smartphones
↔Integrated Circuits. These distilled sub-graphs are fed into
the LLM’s context at inference time. The paths are first retrieved
by traversing the supply chain network from seed nodes that are
semantically similar to key entities in a user’s input. Using struc-
tural centrality metrics to determine traversal distance, econom-
ically relevant sub -networks are retrieved and then formatted to
expose semantic significance to the LLM. Beyond graph paths, our
∗This research was conducted while Evan Heus was an MFE intern at MSCI.framework ingests numerical factor tables and curated news snip-
pets, fusing these modalities into a single prompt so the model can
weigh quantitative signals alongside narrative context. This turns
the graph into an “explainable retrieval engine” rather than a static
data source. By importing only the highest -salience paths, we give
the LLM a concise, interpretable scaffold on which to reason, while
freeing it to generate natural-language risk narratives.
Why a network lens?Vector similarity treats every fact as an
isolated point; supply -chain risk lives in thelinks. By casting the
KG as a graph 𝐺=(𝑉,𝐸) whose edges denote economic rela-
tions (Produces,Has Input,Manufactured In), we use network
analysis to avoid over-flooding the LLM’s context window. There-
fore, each retrieved path is (i)interpretable—a concise Company
→Product →Location narrative the agent can inspect—and (ii)
actionable: the same traversal provides quantitative signals such as
edge weights for the revenue generated by a particular product for
a given company. The result is a prompt that carries built -in eco-
nomic meaning, enabling the LLM to explain hidden dependencies
rather than guess relationships from token proximity.
Our approach is built on a foundational insight: the duality of the
supply chain network and the knowledge graph. A network is a set
of nodes and edges, whereas a knowledge graph is a set of entities
and semantic relationships. For supply chains, these are one and the
same. The economic edge (Company A)-[PRODUCES]->(Product
B) is both a structural link in a network and a semantic triple in a
knowledge graph. This duality allows us to reframe the complex
problem of KG traversal into an efficient network-science problem,
using well-established centrality metrics to identify salient paths
for LLM reasoning.
Contributions.We pose four main advances:
(1)Network -science path discoverythat extracts relevant
sub-graphs.
(2)Knowledge-graph semantic encodingtransforms a net-
work into inference -time prompts exposing built -in eco-
nomic meaning.
(3)Agent -orchestrated, multi -modal retrieval loopwhere
a triage agent selects between graph traversal, factor data,
and news tools prior to synthesis.
(4)Context shells for numerical datathat wrap each fig-
ure in descriptive language, letting the LLM reason over
context-rich quantitative risk metrics.
Paper organization.Section 2 reviews graph-aware LLM curricula,
GraphRAG, and graph database traversal. Section 3 sketches the
overall agentic architecture and loop. Section 4 describes the three
data channels: risk factors, curated news, and synthetic supply chain
1arXiv:2510.01115v1  [cs.AI]  1 Oct 2025

Heus et al.
graph. We also describe the tools that retrieve them, while Section
5 zooms in on the factorcontext-shelltemplate. Section 6 explains
how retrieved evidence is merged into the prompt and the rank-
then-traverse algorithm that extracts salient supply chain paths.
A live dialogue in Section 7 shows the retrieval chain in action,
Section 8 concludes, and the Appendix A provides full relationship
tables and the complete KG diagram.
2 Background and Related Work
Bottom-Up KG curricula.Recent work onbottom -up domain -specific
super -intelligence(BDSI) shows how multi -hop KG paths can be ver-
balized into 24,000 reasoning tasks that supervise a 32 B -parameter
model, yielding state -of-the-art scores on ICD -Bench [ 3]. Their cur-
riculum transforms the graph structureinto training data, pushing
the model to compose primitives into higher-order concepts.
Graph -aware retrieval.GraphRAG [ 5] first extracts an entity
knowledge graph from a document corpus, then offline generates
hierarchical community summaries using a Leiden -based clustering;
at query time, it assembles partial answers from those summaries
and fuses them into a global response. This excels on corpus -wide
“sense -making” questions but its pre -processing and multi -stage
summarization pipeline are compute -intensive. Lighter variants
such as Neural -KB, GNN -RAG[ 8] and temporal -aware RAG[ 13]
likewise keep the base LM frozen while injecting graph structure
to boost recall and freshness. For real-time applications such as
finance, retraining such a model daily is even less feasible.
KG traversal in existing engines.Platforms such as Neo4j,[ 10]
and LangChain’s Graph Retriever,[2] execute a pattern like
MATCH (a)-[:SUPPLIES*1..3]->(b)
by expanding a frontier of node IDs from a seed vertex, applying the
edge labels, hop range, property filters, and optionaltop- 𝑘limit that
the user specifies. Because the traversal runs inside a live Neo4j
server, whose page cache keeps hot neighborhoods in memory
while the full adjacency list stays on disk, results arrive in millisec-
onds. Nevertheless, they do require a running graph service with
enough RAM to cache frequently accessed portions of the graph.
Our method retains the same query vocabulary, yet materializes
only the path segments needed at inference, avoiding the opera-
tional overhead of a dedicated graph database while still supporting
fast, parameter-controlled look-ups.
Positioning of this paper.Our proposed agentic framework bridges
the gap: it agrees with the insight in [ 3] that KG paths encode
domain reasoning, but applies that insightat inference timewith
lightweight network traversal, avoiding the cost and staleness of
specialist fine -tuning while remaining faster than on -the-fly graph
construction methods like GraphRAG.
3 System Overview
Upon initiating a chat, the user supplies our system with the port-
folio constituents and their weights. We make sure that the LLM
holds this at the top of the context window to always be able to
refer to it, since LLMs are known to get “lost in the middle” and
not pay attention to data in the middle of the context window[7].Figure 1 gives a bird’s -eye view of our framework. A user (typ-
ically a CIO) poses a query; two lightweight agents decidewhat
to retrieve andhowto format before a ‘frozen’ LLM generates the
final answer. Here and in what follows, by ‘frozen’ we mean a
pre-trained LLM that is not being fine-tuned as new data arrive.
Figure 1: End -to-end pipeline. TheTriage Agentdecides
whether the query can be answered from memory. If not,
aRerouting Agentselects one or more tools (Factors, News,
Supply -Chain KG), whose outputs are stored in a temporary
database and injected into the LLM prompt.
At the start of every turn, the user’s message is appended to
the ongoing dialogue and sent, together with a snapshot of the
portfolio, to a lightweightTriage Agent. When that agent finds the
answer in memory, it returns it straight away; otherwise it flags the
query for augmentation and delegates to a downstreamRerouting
Agentthat decides which retrieval tool (factor exposures, curated
news or a supply-chain walk) to invoke and with what parameters.
The selected tools tap separate FAISS indices[ 4], one per modal-
ity, so semantic relationships inside each data type remain intact
and compound queries1can be resolved cleanly across stores. Each
tool call is executed through the OpenAI function-calling inter-
face; the language model emits a JSON stub, the back-end runs
the call, and the retrieved snippets flow into a transient database
before being re-injected into the prompt. Finally, a frozen GPT-4o
model synthesizes those multi-modal fragments with the original
question, streams anAugmented Responseto the UI, and stores the
turn in memory, readying the loop for whatever the user asks next.
Responses are fast, as the tool calling system is a series of quick
API calls.
This modular design balancesinterpretability(the network paths
are passed verbatim),flexibility(new tools can be dropped in with-
out re -training), andefficiency(no fine -tuning loop). The next sec-
tion details the construction of each data mode.
4 Data Modes And Tools
Our system operates on three external data channels that comple-
ment the conversation context and the user’s portfolio. Each is
1A request is termed acompound querywhen it references two or more distinct sub-
topics or entities. Embedding the full sentence typically positions its vector between
the relevant clusters in space, so a 𝑘-nearest search can miss evidence dispersed across
those clusters. For example, in the query “factor exposures and cobalt” there may
be no single source covering both topics together, while each topic is well covered
individually at a significant vector distance from the other; 𝑘-nearest retrieval therefore
fails to return comprehensive coverage.
2

Network-KG Duality for Supply Chain Risk
stored in its own FAISS index and is retrieved through a dedicated
tool.
Multi -Asset -Class (MAC) factors.For every security in the user’s
portfolio, we ingest MSCI MAC[ 11] factor scores and the accompa-
nying methodology text. Before embedding, each record is wrapped
in acontext shell(see section 5): a short paragraph that “tokenizes”
the numerical z -scores within natural -language sentences so the
LLM can reason about their meaning rather than seeing them as
opaque numbers. The resulting embeddings power the get_factors
tool.
Curated news.Two news streams feed the system.Macro arti-
cleson long -horizon risks (demographics, climate, geopolitics) and
stock -specific newscomes from theLexisNexisarchive, restricted
to the single trading day. All articles are chunked to roughly a page,
embedded once, and served by the get_news tool; the vector store
preserves outlet and timestamp metadata to support recency filters.
Supply -chain knowledge graph.The final mode is a multi -entity
knowledge graph that linksCompany,Product,Input Product,
Input to Input Product,Industry, andLocationnodes via re-
lations such asProduces,Has Input,Manufactured Inand so
forth (the full list of edge types are given in 1 in A. Although illus-
trative rather than fully industrial, the graph is constructed with a
pipeline akin to AIPNet [ 6], to query ChatGPT’s intrinsic knowl-
edge base to create a dataset of the desired form. These paths enter
the prompt as restructured text within a context shell, exposing the
semantic relationships between nodes on the traversed graph.
Taken together, MAC factors, curated news, and the supply -chain
graph give our system the multi -modal context required for real -time,
portfolio risk analysis.
5 Context Shell
Domain risk analysis is based onnumbers: z-scored factor exposures,
portfolio weights, valuation ratios. Large language models, how-
ever, treat raw figures as opaque tokens and rarely attach contextual
economic meaning to them. To retain a frozen, generic LLM while
still letting it reason quantitatively, our proposed systemwraps ev-
ery table row from the MSCI Multi-Asset-Class (MAC) factor
model in acontext shell: a short paragraph that “tokenizes” each
figure inside an explanatory sentence so that neighboring words
endow the number with semantics. The surrounding text becomes
part of the embedding; the number itself is now a first-class token
that via attention relationships with pertinent context becomes
easily interpretable. A related but different idea has been explored
in [1].
Illustrative shell.Below is an excerpt from a shell; the highlighted
values are place holders for numerical figures inserted for every
security whose factor exposures we have:The position in the portfolio is associated with the security
[‘Security Name’] represented by the ticker [‘Ticker’] . This
position constitutes [‘Weight’] % of the total portfolio. Each of
the following factors is given az-score (mean 0, sd 1) for this
equity relative to all other equities.
[‘Security Name’] Equity Beta: [‘Equity Beta’]
Description:captures market risk beyond the baseline Market
factor.
When High:portfolio tilts toward high-beta stocks, amplifying
risk.
When Low:portfolio tilts toward low-beta stocks, partially
offsetting risk.
[‘Security Name’] Book-to-Price: [‘Book-to-Price’]
Description:book value divided by market capitalization.
When High:stock may be undervalued or distressed.
When Low:stock may be overvalued or considered a growth
stock.
When the shell is embedded and stored in a dedicated FAISS
index, both the verbal context and the numeric tokens contribute to
semantic weighting. At inference time, a get_factors tool fetches
the most similar shells for the query; the LLM then attends towhat
the numbers meanrather than merely copying them, enabling factor-
aware narratives without any task-specific fine-tuning or database
querying, giving the LLM a lightweight yet principled bridge be-
tween structured factor data and natural language reasoning.
6 Network-Science Path Discovery
Here, we operationalize the network-KG duality. Instead of complex
queries over a knowledge graph database, we use network traversal
algorithms to retrieve economically significant sub-graphs.
When a user query arrives, theTriage Agentinspects both the
conversation history and the current portfolio, deciding whether the
answer is already in memory or whether fresh data are needed. For
queries that could benefit from the supply chain context, it delegates
to agraph-traversertool chosen by the downstream Rerouting
Agent.
The traverser first extracts every mention of the company, prod-
uct, or location in the user’s text, embeds those strings with the same
model used to pre-compute node embeddings for the knowledge
graph, and retrieves the closest vectors; the matched vertices be-
comeseed nodesfor traversal. Because each vertex is stored inside a
context-shell that exposes its business metadata during embedding,
similarity search is guided by both entity names and surrounding
economic meaning, yielding semantically faithful starting points.
Traversal depth automatically adapts to the structural role of
each seed node. We use three network centrality measures to cap-
ture different facets ofstructuralnode importance.Degreeis a
local score that counts immediate connections of a node, highlight-
ing popular “hubs”.Closenessgauges how near a node lies, on
average, toallothers, rewarding globally well-connected vertices,
whilebetweennesstallies how often a node sits on shortest paths,
identifying nodes that link otherwise distant regions of the graph.
3

Heus et al.
These unweighted scores are utilized in this prototype as an effi-
cient first-pass filter to identify structurally important nodes (e.g.,
major hubs or bottlenecks for information flow), which often proxy
for economic relevance in large networks. The explicit economic
weights are only applied in the subsequent semantic encoding step.
Averaging the three yields a single salience value that reflects
both the influence of a node on its neighborhood and its role in the
global network structure, giving the traversal algorithm a balanced
way to decide how far to expand from each seed node. For all nodes,
this centrality metric is pre-computed. As seen when contrasting
the one hop network in Figure 2 and the two hop network in Figure
3, the number of nodes retrieved increases exponentially with each
additional hop.
Figure 2: Apple’s network with immediately adjacent Product
nodes.
In practice supply chain networks follow a scale-free, power-
law degree distribution [ 9]: hubs such asElectronic Integrated
Circuitsin the middle of Figure 3 in the Apple network expose a
wealth of information in just one hop, whereas peripheral nodes
require additional traversal hops for useful coverage.
The subtree gathered from this traverse step is then distilled into
short narrative paths. A network, strictly defined, is a collection of
connected objects. Each object has minimal metadata, such as an
identifier, and edges may have weights. Underneath the hood, in
its most basic form, this is what a supply chain is.
For example, if traversal identifies weighted path asApple Inc.
→Desktop Computers →Integrated Circuits →Shanghai,
Chinawith the corresponding edge weights as10% ,19%,13%re-
spectively, our system verbalizes this structured data into an in-
tuitive narrative. By treating the network as a knowledge graph,
we leverage its inherent semantic structure, bringing it closer to
natural language. Add node types and edge types that can be en-
coded into context shells, and transform this information into the
following:Applegenerates 10% of its revenue from sellingDesktop Com-
puters, which spends 19% of its production budget onInte-
grated Circuits, 13% of which are produced inShanghai,
China.
The semantic text surrounding the injected entities and data
is predefined by the edge type between two types of nodes. The
complete set of edge types and metadata is provided in Appendix
A.
Each edge type corresponds to a semantically meaningful edge
phrase, so the language model receives a compact but economically
meaningful scaffold on which to reason. These supply chain snip-
pets let AI-CRO explainwhya node matters, linking downstream
cash flows to upstream inputs and vice versa depending on the
CIO’s query.
7 Sample Chat
The dialogue below illustrates how AI-CRO chains triage, tool calls,
and multi-modal retrieval in real time. The CIO manages a portfolio
of the top-50 constituents of the MSCI USA Index, where Apple is
the largest weight, and Tesla a significant name in electric vehicles
(EVs).
User:
I read about problems in the DRC with coltan.
System:
Issues in the Democratic Republic of Congo (DRC) involv-
ingcoltan, a key source of tantalum for electronics, overlap
with risks of cobalt supply that affect battery producers.
Apple and Tesla rely on both minerals, so disruptions in
eastern DRC represent potential operational and reputa-
tional vulnerabilities for the portfolio.
Reference:Supply-chain paths forcoltan.
Our system mapped “coltan” to a product node, located the up-
stream path into Apple’s and Tesla’s battery lines, and surfaced a
portfolio-level risk narrative.
User:
What has been in the news on this?
Console (log):
Tool called:get_news
Query embedding:“recent news on coltan & cobalt
supply-chain issues in the DRC”
Retrieved:News article, pages 3, 4 and 1
4

Network-KG Duality for Supply Chain Risk
Figure 3: Two-hop supply-chain network for Apple.
5

Heus et al.
System:
Recent coverage documents coltan mined by the M23 mili-
tia in eastern DRC, smuggled via Rwanda into global sup-
ply chains. This makes it harder for Apple, Tesla, and
peers to guarantee conflict-free sourcing, heightening both
supply-disruption and ESG-reputation risk.
Reference:News Article.
The Rerouting Agent embedded a semantic description, retrieved
the top-three matching articles, and the LLM wove their content
into the conversation.
User:
Can you walk me through the various ways this could hurt
Apple?
System:
(i) Supply-chain delays— shortages of tantalum capaci-
tors could postpone iPhone and Mac launches.
(ii) Reputational risk— perceived reliance on conflict
minerals may erode brand trust.
(iii) Margin pressure— higher input costs compress hard-
ware margins and may force price rises that dampen de-
mand.
Reference:Apple supply-chain paths.
Across three turns, our system triaged queries, selected tools,
retrieved factor-relevant evidence, and produced narrative answers
that integrate operational, ethical, and financial dimensions, with-
out any bespoke fine-tuning.
8 Conclusion and Next Steps
Summary.Our proposed framework demonstrates that a frozen
general purpose LLM can operate as a domain expert when paired
with three carefully engineered data modes. MSCI MAC factor
models, curated news, and a synthetic supply chain knowledge
graph, along with an agent loop that decides when and how to pull
each modality. By turning high-salience graph paths into natural-
language snippets and wrapping numeric factors in context shells,
the system surfaces hidden dependencies and quantitative signals
that ordinary vector-only RAG pipelines overlook, all while running
in real time without specialist fine-tuning.
Although our system demonstrates a powerful new approach,
we acknowledge several limitations that point to avenues for future
work. First, the supply chain graph was constructed using an LLM
pipeline and is inherently limited in scope. While effective for this
case study, it may contain inaccuracies or omissions. Future work
should focus on validating and enriching this graph with structured
trade data from sources such as customs declarations or bill of
landing records.
Second, our path discovery relies on unweighted topological cen-
trality scores to guide the traversal. While this effectively identifies
structurally important paths, it can overlook economically criticalpaths involving peripheral nodes with a high financial weight. Fu-
ture work will investigate integrating edge weights directly into the
traversal algorithm (e.g., using weighted shortest path algorithms or
value-weighted centrality measures) to improve economic fidelity.
Finally, the system currently uses static edge weights. A key
next step is to integrate real-time financial data to dynamically
update these weights, reflecting changing revenue dependencies or
production costs.
References
[1]Anthropic. 2024. Introducing Contextual Retrieval. Anthropic Blog. https:
//www.anthropic.com/news/contextual-retrieval Accessed: 2025-08-04.
[2]Tomaz Bratanic. 2024. Enhancing RAG-based Application Accuracy
by Constructing and Leveraging Knowledge Graphs. LangChain Blog.
https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-
constructing-and-leveraging-knowledge-graphs/ Accessed August 2, 2025.
[3] Bhishma Dedhia, Yuval Kansal, and Niraj K Jha. 2025. Bottom-up Domain-specific
Superintelligence: A Reliable Knowledge Graph is What We Need.arXiv preprint
arXiv:2507.13966(2025). https://arxiv.org/abs/2507.13966
[4] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The Faiss library.arXiv preprint arXiv:2401.08281(2024). arXiv:2401.08281 [cs.LG]
[5]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A Graph
RAG Approach to Query-Focused Summarization.arXiv preprint arXiv:2404.16130
(2024). https://arxiv.org/abs/2404.16130
[6]Thiemo Fetzer, Peter John Lambert, Bennet Feld, and Prashant Garg. 2024.
AI-generated Production Networks: Measurement and Applications to Global
Trade. https://aipnet.io/wp-content/paper/ai_generated_production_networks_
aipnet_v20241118.pdf Working paper, AIPNet project.
[7] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2023. Lost in the Middle: How Language Models
Use Long Contexts. arXiv:2307.03172 [cs.CL] https://arxiv.org/abs/2307.03172
[8] Costas Mavromatis and George Karypis. 2024. Gnn-rag: Graph neural retrieval
for large language model reasoning.arXiv preprint arXiv:2405.20139(2024).
[9] Luca Mungo, Alexandra Brintrup, Diego Garlaschelli, and François Lafond. 2023.
Reconstructing Supply Networks.INET Oxford Working Paper2023-19 (Oct.
2023). arXiv:2310.00446 [physics.soc-ph] https://arxiv.org/abs/2310.00446
[10] Neo4j Inc. 2025.Neo4j Cypher Manual. Version 5.x. https://neo4j.com/docs/
cypher-manual/.
[11] P. Shepard, A. DeMond, L. Xiao, C. Zhou, and J. Ahlport. 2020.The MSCI Multi-
Asset Class Factor Model. Technical Report. MSCI. MSCI Research.
[12] Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie,
and Iadh Ounis. 2025. Are Generative AI Agents Effective Personalized Financial
Advisors?. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 286–295.
[13] Fengbin Zhu, Junfeng Li, Liangming Pan, Wenjie Wang, Fuli Feng, Chao
Wang, Huanbo Luan, and Tat-Seng Chua. 2025. FinTMMBench: Benchmarking
Temporal-Aware Multi-Modal RAG in Finance.arXiv preprint arXiv:2503.05185
(2025).
6

Network-KG Duality for Supply Chain Risk
A Knowledge-Graph Details
Table 1: Relationships by Entity Type for the knowledge graph.
Entity Type Possible Relationships Entity Metadata
Company —Produces→Product
—Produces→Input ProductTicker
Total Revenue
Product —Sold By→Company
—Belongs To→Industry
—Has Input→Product (Upstream)
—Input To→Product (Downstream)
—Manufactured In→Location
—Sourced From→Location
—Made With→Input ProductHS Code
Total Revenue Share
Production Cost Percentage
Industry —Includes Product→Product
—Includes Product→Input ProductNAICS Code
Location —Production Location For→Product Longitude
Latitude
Production Share
Figure 4: Knowledge-graph relationship diagram.
7
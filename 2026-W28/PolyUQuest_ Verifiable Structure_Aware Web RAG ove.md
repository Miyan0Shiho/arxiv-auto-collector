# PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs

**Authors**: Ying Liu, Yi Ye, Quanyu Feng, Mingxi Ye, Mingtao Zhang, Haoyang Li, Chen Jason Zhang, Qing Li

**Published**: 2026-07-09 09:10:52

**PDF URL**: [https://arxiv.org/pdf/2607.08269v1](https://arxiv.org/pdf/2607.08269v1)

## Abstract
Existing retrieval-augmented generation (RAG) systems treat web pages as flat text, losing the structural and semantic signals encoded in HTML. We present PolyUQuest, a verifiable, structure-aware web RAG framework built on a heterogeneous graph that unifies hyperlink topology between pages, DOM hierarchy within pages, and entity-relation knowledge across pages. A two-tier router dispatches each query to one of three retrieval modes matched to its structural need, including direct block retrieval, cross-page graph traversal, and multi-hop entity reasoning. Every answer is fully verifiable, as each cited block carries its source page, heading path, and entity links so that users can trace any claim back to its structural evidence. We evaluate on the official websites of the Hong Kong Polytechnic University (PolyU), comprising 4,240 pages, 31,086 DOM blocks, 29,119 entities, and 37,680 relations, together with a multi-type evaluation benchmark. PolyUQuest outperforms existing RAG systems in answer correctness, coverage, and faithfulness, while consuming significantly fewer LLM tokens per query. The demonstration provides an interactive interface for inspecting cited answers, comparing retrieval traces across routing modes, and exploring evidence graph paths. PolyUQuest is being prepared for deployment as a student-facing QA service at PolyU.

## Full Text


<!-- PDF content starts -->

PolyUQuest: Verifiable Structure-Aware Web RAG over
Heterogeneous Graphs
Ying Liu, Yi Ye, Quanyu Feng, Mingxi Ye, Mingtao Zhang, Haoyang Li*, Chen Jason Zhang, Qing Li
The Hong Kong Polytechnic University, Hong Kong SAR, China
{yarden.liu, yi000.ye, quanyu.feng, mingxi.ye, mingtao.zhang}@connect.polyu.hk
{haoyang-comp.li, jason-c.zhang, qing-prof.li}@polyu.edu.hk
Abstract
Existing retrieval-augmented generation (RAG) systems treat web
pages as flat text, losing the structural and semantic signals encoded
in HTML. We present PolyUQuest, a verifiable, structure-aware
web RAG framework built on a heterogeneous graph that unifies
hyperlink topology between pages, DOM hierarchy within pages,
and entity–relation knowledge across pages. A two-tier router dis-
patches each query to one of three retrieval modes matched to its
structural need, including direct block retrieval, cross-page graph
traversal, and multi-hop entity reasoning. Every answer is fully ver-
ifiable, as each cited block carries its source page, heading path, and
entity links so that users can trace any claim back to its structural
evidence. We evaluate on the official websites of the Hong Kong
Polytechnic University (PolyU), comprising 4,240 pages, 31,086
DOM blocks, 29,119 entities, and 37,680 relations, together with a
multi-type evaluation benchmark. PolyUQuest outperforms exist-
ing RAG systems in answer correctness, coverage, and faithfulness,
while consuming significantly fewer LLM tokens per query. The
demonstration provides an interactive interface for inspecting cited
answers, comparing retrieval traces across routing modes, and ex-
ploring evidence graph paths. PolyUQuest is being prepared for
deployment as a student-facing QA service at PolyU. A demo video
is available at https://youtu.be/thKWYaL_4rw.
Keywords
Retrieval Augmented Generation, Question Answering, Heteroge-
neous Graphs, Web Search
1 Introduction
Web pages are among the most widely used knowledge sources for
retrieval-augmented generation (RAG) [ 10,16,25], where retrieving
accurate and current evidence is key to curbing hallucination [ 4,
11,27,31]. Unlike plain documents, web content is organized in
three complementary layers: hyperlinks between pages, a DOM
hierarchy within each page, and named entities that recur across
pages. Consider a prospective student asking, “Which professors in
the Department of Computing conduct NLP research and also teach
related courses?” No single page holds the answer: it is scattered
across faculty profiles and course pages. Answering it requires
following hyperlinks and connecting entities across pages, beyond
the reach of plain similarity retrieval.
Existing RAG approaches cannot handle questions over struc-
turally complex websites, especially those that require cross-page
navigation. Plain-text and chunk-based RAG [ 15,21] flattens each
page into a bag of chunks and retrieves by similarity, discarding
∗Corresponding author.structure entirely, while agentic web search can recover it by brows-
ing, only at prohibitive token cost and latency. Graph RAG sys-
tems [ 5,8,17,23,24], such as LightRAG [ 5], instead capture entity-
relation structure [ 19], yet they still flatten HTML into text and in-
ject large global contexts that inflate the cost of every query [ 12,32].
HTML- and document-structure-aware RAG [ 13,18,22], such as
HtmlRAG [ 22], in contrast, preserves structure within a page or
documents, but ignores the hyperlinks that connect pages and so
cannot follow evidence across a site. In summary, existing systems
fail to reason over hyperlink topology, DOM hierarchy, and cross-
page entities together, and those closest to structure-awareness pay
heavily in tokens and latency.
PolyUQuest(Section 2). In response, we have developed PolyUQuest,
an interactive structure-aware RAG system over heterogeneous
web graphs. PolyUQuest unifies hyperlink topology, DOM hierar-
chy, and cross-page entity associations under a single graph, and
exploits them through query routing that addresses the tension be-
tween structural fidelity, cross-page reasoning, and retrieval costs
that limit prior approaches. Compared to existing RAG systems,
PolyUQuest offers the following:
Three-layer web graph. PolyUQuest models a website as a single
heterogeneous graph whose webpage, evidence-block, entity, and
topic nodes are connected across all three structural layers. Unlike
prior systems that usually emphasize one structural layer, this
unified graph allows the system to move from an entity to the
block where it appears, then to the containing page and its linked
neighbors when an answer spans several pages.
Structure-driven retrieval. Different questions draw on different lay-
ers of the graph, so PolyUQuest classifies each query and dispatches
it to one of three retrieval modes: direct block retrieval for single-
hop facts, cross page navigation for comparison and aggregation,
and multi-hop entity reasoning for evidence scattered across pages.
By retrieving only the structurally relevant evidence, this routing
avoids the large global contexts that make graph RAG costly.
Verifiable answer provenance. Every cited block carries its source
page, heading path, and entity links. Users can click a citation to
inspect the exact page context behind a claim, follow graph paths
connecting people, programmes, courses, and pages, and see at a
glance whether an answer rests on one page or several.
Demonstration(Section 3). We give a guided tour of PolyUQuest
over institutional webpages, covering cited-answer inspection, guided
retrieval traces, and graph-based evidence exploration. Participants
can ask questions, inspect the source blocks and pages behind each
answer, compare retrieval traces across modes, and explore the
graph paths that justify a result.
arXiv:2607.08269v1  [cs.AI]  9 Jul 2026

CIKM ’26, November 7–11, 2026, Rome, Italy Liu et al.
Offline Indexing
Structure & Entity Indexing
…h3 ...h2 ...h1 ...
Blocks, Entities, IndexesWeb Crawling
Pages & Hyperlinks
Three -Level Heterogeneous Web Graph
Web Pages
Evidence 
Blocks
Entities &
Topics
Vector
IndexesOnline Answering
Evidence Assembly
Page Title
Heading Path
Source URL
Citation IDsTwo -Tier 
Routing
Verifiable Answer
AnswerInteractive Interface
Retrieval Trace
QUERY ROUTING
TRAVERSAL
ANSWER GENERATIONStructure Aware Retrieval
Mode A：Direct Block Retrieval
Mode B：Navigation Retrieval
Mode C：Entity ReasoningrUser
Query
Figure 1: Framework Overview of PolyUQuest
2 PolyUQuest System
This section describes offline indexing (§2.1), online retrieval (§2.2),
and the full workflow (§2.3).
2.1 Offline Indexing
Three-Layer Heterogeneous Graph.PolyUQuest models web-
siteWas a heterogeneous graph GW=(V,E) , whereV=
V𝑃∪V𝐵∪V𝐸∪V𝑇contains webpage, evidence block, entity,
and topic nodes [ 10,19,25]. The edge setEcaptures four families
of connections: page–page hyperlinks, page–block containment,
block–entity mentions, and semantic associations among entities
and topics. Each evidence block node 𝑏∈V𝐵is attached to its
source page 𝑝(𝑏), heading path ℎ(𝑏), block embeddingx 𝑏, length
ℓ(𝑏), and mentioned entity subset V𝐸(𝑏)⊆V𝐸.Layer 1 (Site Graph):
hyperlinks capture how webpages point to each other.Layer 2 (Block
Tree):page content is organized into heading-aware text blocks that
preserve the local document hierarchy.Layer 3 (Entity Graph):ex-
tracted entities are connected to their source blocks, related entities,
and topics. These layers allow the system to move from an entity
to the evidence block where it appears, then to the containing page
and neighboring pages when cross-page evidence is needed.
Block Tree Construction.Following HTML-aware studies [ 18,
22], our system removes non-content regions such as scripts, style
definitions, embedded frames, and comments from raw HTML while
retaining useful textual, hyperlink, and accessibility information.
A bottom-up algorithm traverses the cleaned DOM tree, merging
sibling subtrees until a block reaches a configurable word threshold,
adopting the granularity-adjustable block-tree construction. Weset this word size to 150 words, following [ 5,22]. Tables, lists, and
preformatted code are treated as atomic units that cannot be split
in the middle. Each retained block inherits its surrounding section
headings, a compact representation that captures page hierarchy
without storing the full DOM tree. Repeated boilerplate regions,
such as navigation bars, footers, and cookie notices, are filtered out
before block embedding.
Entity Extraction and Resolution.An LLM extracts entities
and relations from each page’s blocks using a small set of website-
relevant entity types, such as people, programmes, departments,
courses, and research topics [ 6,7,9]. Each extraction is linked back
to the source block, enabling fine-grained provenance. Because
the same person, programme, or department may appear under
abbreviations, aliases, or slightly different page titles, PolyUQuest
resolves extracted mentions before adding them to the graph. It first
normalizes obvious aliases and uses authority pages as anchors,
then merges remaining candidates with a combined semantic and
string similarity score; an LLM is used to decide borderline cases.
2.2 Online Structure-Aware Retrieval
PolyUQuest organizes online retrieval into three modes according
to the structure needed by the query: Mode A for direct factual
lookup, Mode B for navigation across linked pages, and Mode C
for entity-based multi-hop reasoning [ 28,29]. The router selects
among these modes before the chosen retrieval pathway assembles
cited evidence for answer generation.
Two-Tier Router.Given a user query 𝑞with embeddingx 𝑞, the
router classifies it in two tiers. A first tier of lightweight rules han-
dles unambiguous structural signals: “which professors” triggers
Mode C, “admission requirements for” triggers Mode B, and com-
parison patterns such as “differences between X and Y” map to
Mode B for cross-page aggregation. Queries that match no heuristic
pass to a second-tier LLM classifier, which assigns the query to
one of the three modes with a confidence score. This design keeps
routing latency low on common patterns while delegating long-tail
queries to the LLM.
Mode A: Direct Block Retrieval. For single-hop factual questions
such as “What is the tuition fee for MSc in AI?”, the system retrieves
candidates in two stages. The first stage builds a candidate block
setB𝑞by combining dense approximate nearest-neighbor search
overx𝑏with BM25 sparse retrieval, capturing both semantic and
lexical signals [ 15,30]. This mode treats each heading-aware block
as the basic evidence unit, so it is suitable when the answer is likely
localized within one page section. A cross-encoder reranker then
rescores blocks inB𝑞by jointly attending to the query, the block
text, its source page 𝑝(𝑏), and heading path ℎ(𝑏). The top blocks
are enriched with parent block content before answer generation,
which helps preserve local context for citations.
Mode B: Navigation Retrieval. For aggregation queries across pages,
such as comparing two programmes, the system decomposes the
query into subqueries via the LLM. For each subquery, it first re-
trieves relevant pages and then expands to nearby linked pages in
the site graph. The candidate set B𝑞is then assembled from evi-
dence blocks on the discovered pages. This expansion is important
for university websites and other structured web collections, where
overview pages often link to separate pages for fees, admission,

PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs CIKM ’26, November 7–11, 2026, Rome, Italy
deadlines, or contacts [ 1,2,25]. A holistic cross-encoder reranking
scores all candidates across subqueries before the final synthesis,
allowing the answer generator to compare evidence from multiple
pages under a shared context.
Mode C: Entity Reasoning. For queries requiring multiple hops,
such as “Which professors research NLP and teach related courses?”,
the system first extracts topic keywords via the LLM and then
follows two complementary evidence paths. The first path retrieves
candidate entities and expands through entity relations. The second
path retrieves topic keywords and expands to entities associated
with those topics. Together, these paths identify query-related entity
nodesV𝐸,𝑞⊆ V𝐸, and the system traces them back to source
blocks that mention them [ 14,16,19,20]. This mode is designed
for questions whose evidence is scattered across multi-pages Using
the candidate blocksB 𝑞, Mode C ranks each block𝑏by:
𝑠(𝑏,𝑞)=cos(x 𝑞,x𝑏)+min(|V𝐸(𝑏)∩V𝐸,𝑞|,𝜅)
𝜅
+ 
1−ℓ(𝑏)
max𝑏′∈B𝑞ℓ(𝑏′)!
.(1)
The three terms measure query–block similarity, capped entity
coverage, and block conciseness. The cap 𝜅=30prevents blocks
with many incidental entities from dominating the ranking, tuned
on validation sets. A same-page penalty further encourages source
diversity when many retrieved blocks come from the same page.
2.3 Workflow
PolyUQuest has two phases: an offline phase that builds a three-
layer graph from crawled pages, and an online phase that routes
queries to structure-aware retrieval pathways.
Offline Indexing Pipeline.
•Web Crawling. PolyUQuest crawls selected PolyU webpages and
records hyperlinks between pages as the site graph.
•DOM Evidence Construction. It parses each page into heading-
aware evidence blocks and preserves section context for later
citation and evidence inspection.
•Entity Graph Construction. It extracts entities and relations from
evidence blocks, resolves duplicate entities, and links each entity
back to its source evidence.
Online Query Pipeline.
•Query Routing. The router assigns the query to factual lookup,
navigation across pages, or entity reasoning.
•Structure-Aware Retrieval. The selected mode retrieves evidence
from DOM blocks, linked pages, or entity paths.
•Evidence Assembly. The system reranks retrieved blocks and as-
sembles cited evidence with page titles and heading paths.
•Answer Generation. The LLM generates an answer while the in-
terface displays citations and the retrieval trace.
3 Demonstration
Demo Setup.The demo runs on 4,240 official English pages from
the PolyU website, covering programme, staff, department, and
research sections, with 31,086 DOM blocks, 29,119 resolved entities,
and 37,680 relations. The system provides a Chat view for arbitraryquestion answering with cited evidence and a Graph view for ex-
ploring the underlying webpage, DOM, and entity structures. The
source code is in https://github.com/13-pieces-teen/PolyUQuest.
Demonstration Scenarios.We will guide users through three sce-
narios: answer provenance inspection, guided retrieval trace, and
evidence graph exploration. Figure 2 illustrates the three demon-
stration scenarios: the Chat view with cited answers and follow-up
questions (left), the retrieval traces for each routing mode (top),
and the evidence inspection panels showing source pages, retrieved
blocks, and knowledge graph paths (bottom right).
Scenario 1: Answer Provenance Inspection. This scenario covers the
primary question answering use case. Users first enter a programme
or staff-related question in the Chat view. PolyUQuest returns a con-
cise answer with inline citations, while the evidence panel shows
the source page, heading path, and DOM block behind each cited
claim. Users can click a citation to highlight the corresponding
evidence block and inspect its surrounding page context. The in-
terface also groups evidence by source page, so users can quickly
see whether an answer is grounded in one page or assembled from
several pages. This scenario shows that users can verify the answer
without manually opening and comparing many webpages.
Scenario 2: Guided Retrieval Trace. Users then try three represen-
tative queries: a single page factual question, a cross-page com-
parison, and an entity-centric question. For each query, the trace
panel shows which retrieval mode is selected and summarizes the
main interaction steps from routing to evidence assembly. Users
can compare how the displayed trace changes across factual lookup,
page navigation, and entity reasoning. The trace also reports the
retrieved pages and evidence blocks at each stage, making rout-
ing decisions and retrieval failures visible during the demo. This
scenario demonstrates how different user intents lead to different
retrieval behaviors while keeping the interface understandable to
non-expert users.
Scenario 3: Evidence Graph Exploration. Finally, users open the Graph
view from a cited answer. The interface highlights the webpage,
DOM block, entity, and topic nodes that support the current answer.
Users can expand neighboring nodes to explore related pages or
entities. Hovering over nodes reveals page titles, block snippets, or
entity descriptions, while selected paths remain highlighted against
the cited answer. This scenario shows how PolyUQuest turns an
answer into an inspectable evidence path.
Performance Highlights.In this demo, we evaluate PolyUQuest
on 300 PolyU-Web questions against ChunkRAG [ 21], HtmlRAG [ 22],
FastGraphRAG [ 3], and LightRAG [ 5]. The questions are constructed
to cover the three retrieval needs targeted by PolyUQuest: single-
page factual lookup, cross-page navigation, and entity-centric rea-
soning. For each question, we manually annotate the relevant offi-
cial PolyU pages and reference evidence blocks. All baselines use the
same generator LLM, prompt, and embedding model as PolyUQuest.
Following [ 5,21,22,26], we report three metrics: Answer Correct-
ness (Corr.) for factual accuracy, Coverage (Cov.) for completeness,
and Faithfulness (Faith.) for whether generated claims are sup-
ported by retrieved evidence. Query Token (Q. Tok.) reports the
average LLM tokens per query, and Build Token (B. Tok.) reports
the total tokens consumed during offline graph construction.

CIKM ’26, November 7–11, 2026, Rome, Italy Liu et al.
What subjects are in the AIDA secondary major for 
BEng Mechanical Engineering?PolyUQuest
Based solely on the provided context, the subjects for 
the Secondary Major in Artificial Intelligence and Data
Analytics (AIDA) within the BEng (Hons) in Mechanical
Engineering are as follows [1][3]:
**Core Subjects (30 credits):**
*Mathematics I for AIDA: AMA1110 Basic 
Mathematics I – Calculus and Probability & Statistics 
(3 credits) [1][3].
*Mathematics II for AIDA: AMA2111 Mathematics I 
(3 credits) [1][3].
*Programming I: Programming Fundamentals: ENG2002
Computer Programming (3 credits) [1][3]......
(1) single-hop factualMode A
QUERY ROUTING
mode_a · heuristic
BLOCK ANN + BM25
top-50 hits
CROSS-ENC. RERANK
50 → 10 · top 0.93
ANSWER GENERATION
General LLMs
(3) multi-page entityMode C
QUERY ROUTING
mode_c · heuristic
TOPIC EXTRACTION
LLM keywords
ENTITY_ANN+2-HOP KG
RELATES_TO HAS_TOPIC
BLOCK STORE+ANSWER
Normalised Eq.(1)
(2) cross-pageMode B
QUERY ROUTING
mode_b · 55% conf
QUERY DECOMPOSITION
4 sub-queries
WEBPAGE + LINKS_TO
20 → 32 pages
BLOCK RETRIEVAL + BM25
602 hits · top50
RERANK + ANSWER
121 → 10 · top914
"What RPg programmes does PolyU 
offer?""Differences between Exchange & 
Short-term?""Which LSGI profs do remote 
sensing?"
Who should I contact for more details about the AIDA 
major's credit requirements?
Based solely on the provided context, you should contact 
the Programme Leader(s) for the specific programme you 
are inquiring about. The relevant contacts mentioned in the 
context are:
*For the BEng in Mechanical Engineering with a 
Secondary Major in AIDA , contact Prof. CHENG Song 
(songryan.cheng@polyu.edu.hk) [1].
*For the BBA in Management with a Secondary
Major in AIDA......
Ask a follow-up... Source PageRetrieved Block
Knowledge Graph
Figure 2: Interactive demo interface of PolyUQuest.
Table 1: Main results on PolyU-Web (300 questions). Html-
RAG and ChunkRAG have no offline build phase, so build
tokens are marked as —.
System Corr. Cov. Faith. Q. Tok.↓B. Tok.↓
ChunkRAG [21] 0.532 0.479 0.710 2,947 —
HtmlRAG [22] 0.453 0.448 0.804 4,009 —
FastGraphRAG [3] 0.295 0.469 0.737 4,484 28.1M
LightRAG [5] 0.610 0.612 0.559 29,825 37.4M
PolyUQuest0.644 0.649 0.921 2,968 17.5M
Table 2: Ablation study.
Variant Correct. Cover. Faithful.
PolyUQuest (full)0.644 0.649 0.921
w/o DOM blocks 0.563 0.510 0.916
w/o cross-page mode 0.624 0.633 0.905
As shown in Table 1, PolyUQuest achieves the best Correctness
(0.644), Coverage (0.649), and Faithfulness (0.921) among all com-
pared systems, with a 36-point Faithfulness gain over the next-best
baseline. Meanwhile, it consumes only2,968 tokens per query,
10×fewer than LightRAG and 26% fewer than HtmlRAG, because
the router dispatches each query to a targeted retrieval mode instead
of injecting a large global context. LightRAG’s lower Faithfulness
stems from its injecting LLM-generated community summaries
rather than raw source text, so the model often produces claims
not grounded in any retrieved passage.Ablation Study.Table 2 reports two ablations that isolate the
contribution of DOM-aware blocks and cross-page navigation. Re-
placing DOM blocks with fixed-size chunks causes the largest drop:
Correctness falls by 8.1 points and Coverage by 13.9 points, con-
firming that structure-aware segmentation is critical for retrieving
coherent evidence. Removing cross-page navigation has a smaller
effect, with Correctness down 2.0 and Coverage down 1.6 points.
Faithfulness stays above 0.90 in both cases because the generation
prompt remains unchanged. These results show that DOM-aware
blocks are the primary driver of PolyUQuest’s quality advantage.
Generalizability Discussion.PolyUQuest is most directly appli-
cable to organizational websites whose knowledge is distributed
across linked pages, hierarchical sections, and recurring named
entities, such as universities, government agencies, hospitals, and
enterprises. Porting to a new domain requires a fresh crawl and
a domain-specific entity schema, while the indexing, routing, re-
trieval, and provenance mechanisms remain unchanged.
4 Conclusion
We present PolyUQuest, a verifiable structure-aware web RAG
system that models hyperlink topology, DOM hierarchy, and cross-
page entity relations as a unified heterogeneous graph. By routing
queries to structure-matched retrieval modes, PolyUQuest retrieves
targeted evidence for factual lookup, cross-page navigation, and
entity-centric reasoning while maintaining traceable provenance.
The demo lets attendees ask questions, inspect source blocks and
graph traces behind each answer, and compare retrieval modes
across different structural needs. We are preparing PolyUQuest for
deployment as a QA service at PolyU.

PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs CIKM ’26, November 7–11, 2026, Rome, Italy
GenAI Usage Disclosure
Generative AI tools (Anthropic Claude) were used solely to improve
the grammar, spelling, and clarity of author-written text. They were
not used to generate any substantive content, and the authors take
full responsibility for the entire work.
References
[1]Qi Chen, Xiubo Geng, Corby Rosset, Carolyn Buractaon, Jingwen Lu, Tao Shen,
Kun Zhou, Chenyan Xiong, Yeyun Gong, Paul Bennett, et al .2024. Ms marco
web search: A large-scale information-rich web dataset with millions of real click
labels. InCompanion Proceedings of the ACM Web Conference 2024. 292–301.
[2]Nayoung Choi, Grace Byun, Andrew Chung, Ellie S Paek, Shinsun Lee, and
Jinho D Choi. 2025. Reference-Aligned Retrieval-Augmented Question Answering
over Heterogeneous Proprietary Documents. InProceedings of the 34th ACM
International Conference on Information and Knowledge Management. 5626–5633.
[3]Circlemind. 2024. Fast GraphRAG. https://github.com/circlemind-ai/fast-
graphrag
[4]Wenqi Fan, Pangjing Wu, Yujuan Ding, Liangbo Ning, Shijie Wang, and Qing Li.
2025. Towards Retrieval-Augmented Large Language Models: Data Management
and System Design. In2025 IEEE 41st International Conference on Data Engineering
(ICDE). IEEE, 4509–4512.
[5]Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang. 2024. Lightrag:
Simple and fast retrieval-augmented generation.arXiv preprint arXiv:2410.05779
2, 3 (2024).
[6]Chen Han, Yuanyuan Li, and Xijin Tang. 2025. DocPolicyKG: A Lightweight
LLM-Based Framework for Knowledge Graph Construction from Chinese Policy
Documents. InProceedings of the 34th ACM International Conference on Informa-
tion and Knowledge Management. 4753–4757.
[7]Farnoosh Hashemi and Laks VS Lakshmanan. 2025. KRAFT: A Knowledge Graph-
Based Framework for Automated Map Conflation. InProceedings of the 34th ACM
International Conference on Information and Knowledge Management. 802–812.
[8]Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025.
Grag: Graph retrieval-augmented generation. InFindings of the Association for
Computational Linguistics: NAACL 2025. 4145–4157.
[9]Peng Huang, Meihui Zhang, Ziyue Zhong, Chengliang Chai, and Ju Fan. 2024.
Representation learning for entity alignment in knowledge graph: A design
space exploration. In2024 IEEE 40th International Conference on Data Engineering
(ICDE). IEEE, 3462–3475.
[10] Yiqian Huang, Shiqi Zhang, and Xiaokui Xiao. 2025. Ket-rag: A cost-efficient
multi-granular indexing framework for graph-rag. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 1003–1012.
[11] Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, and Jiawei
Han. 2025. Retrieval and structuring augmented generation with large language
models. InProceedings of the 31st ACM SIGKDD Conference on Knowledge Discov-
ery and Data Mining V. 2. 6032–6042.
[12] Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, and Tim Kraska.
2025. Piperag: Fast retrieval-augmented generation via adaptive pipeline paral-
lelism. InProceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery
and Data Mining V. 1. 589–600.
[13] Da Li, Keping Bi, Jiafeng Guo, and Xueqi Cheng. 2025. Bridging Queries and
Tables through Entities in Open-Domain Table Retrieval. InProceedings of the
34th ACM International Conference on Information and Knowledge Management.
1540–1550.
[14] Mingyang Li, Shisong Chen, Shengkun Tu, Ziyi Du, Jinghao Zhang, Zhixu Li,
and Yanghua Xiao. 2025. KUG: Joint Enhancement of Internal and External
Knowledge for Retrieval-Augmented Generation. InProceedings of the 34th ACM
International Conference on Information and Knowledge Management. 1603–1612.
[15] Peizheng Li, Chaoyi Chen, Hao Yuan, Zhenbo Fu, Hang Shen, Xinbo Yang, Qiange
Wang, Xin Ai, Yanfeng Zhang, Yingyou Wen, et al .2025. NeutronRAG: Towards
Understanding the Effectiveness of RAG from a Data Retrieval Perspective. In
Companion of the 2025 International Conference on Management of Data. 163–166.
[16] Rui Li, Quanyu Dai, Zeyu Zhang, Xu Chen, Zhenhua Dong, and Ji-Rong Wen.
2025. Knowtrace: Bootstrapping iterative retrieval-augmented generation with
structured knowledge tracing. InProceedings of the 31st ACM SIGKDD Conference
on Knowledge Discovery and Data Mining V. 2. 1470–1480.
[17] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao,
Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, et al .2025. Kag:
Boosting llms in professional domains via knowledge augmented generation. In
Companion Proceedings of the ACM on Web Conference 2025. 334–343.
[18] Yiming Lin, Madelon Hulsebos, Ruiying Ma, Shreya Shankar, Sepanta Zeighami,
Aditya G Parameswaran, and Eugene Wu. 2025. Querying templatized document
collections with large language models. In2025 IEEE 41st International Conference
on Data Engineering (ICDE). IEEE, 2422–2435.
[19] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey.ACM Transactions on Information Systems44, 2 (2025), 1–52.
[20] Xingrun Quan, Yongkang Zhou, and Junjie Yao. 2025. ThoughtForest-KGQA: A
Multi-Chain Tree Search for Knowledge Graph Reasoning. InProceedings of the
34th ACM International Conference on Information and Knowledge Management.
5156–5160.
[21] Ishneet Sukhvinder Singh, Ritvik Aggarwal, Ibrahim Allahverdiyev, Muhammad
Taha, Aslihan Akalin, Kevin Zhu, and Sean O’Brien. 2024. Chunkrag: Novel
llm-chunk filtering method for rag systems.arXiv preprint arXiv:2410.19572
(2024).
[22] Jiejun Tan, Zhicheng Dou, Wen Wang, Mang Wang, Weipeng Chen, and Ji-
Rong Wen. 2025. Htmlrag: Html is better than plain text for modeling retrieved
knowledge in rag systems. InProceedings of the ACM on Web Conference 2025.
1733–1746.
[23] Yubo Wang, Haoyang Li, Fei Teng, and Lei Chen. 2025. AGRAG: Advanced Graph-
based Retrieval-Augmented Generation for LLMs.arXiv preprint arXiv:2511.05549
(2025).
[24] Yubo Wang, Haoyang Li, Fei Teng, and Lei Chen. 2026. GORAG: Graph-based
Online Retrieval Augmented Generation for Dynamic Few-shot Social Media
Text Classification. InProceedings of the ACM Web Conference 2026(United Arab
Emirates)(WWW ’26). Association for Computing Machinery, New York, NY,
USA, 9148–9159. doi:10.1145/3774904.3793011
[25] Yu Wang, Nedim Lipka, Ruiyi Zhang, Alexa Siu, Yuying Zhao, Bo Ni, Xin Wang,
Ryan Rossi, and Tyler Derr. 2024. Topology-aware retrieval augmentation for text
generation. InProceedings of the 33rd ACM International Conference on Information
and Knowledge Management. 2442–2452.
[26] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong,
Xiao Huang, and Jinsong Su. 2025. When to use graphs in rag: A compre-
hensive analysis for graph retrieval-augmented generation.arXiv preprint
arXiv:2506.05690(2025).
[27] Yuming Xu, Mingtao Zhang, Zhuohan Ge, Haoyang Li, Nicole Hu, Yongqi Zhang,
Zhiyuan Wen, Jason Chen Zhang, Qing Li, and Lei Chen. 2026. Securing Retrieval-
Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions.
arXiv:2604.08304 [cs.CR] https://arxiv.org/abs/2604.08304
[28] Dezhi Ye, Ye Qin, Bowen Tian, Jiabin Fan, Jie Liu, Haijin Liang, and Jin Ma. 2025.
Can LLMs Really Help Query Understanding In Web Search? A Practical Per-
spective. InProceedings of the 34th ACM International Conference on Information
and Knowledge Management. 5433–5438.
[29] Runjie Yu, Weizhou Huang, Shuhan Bai, Jian Zhou, and Fei Wu. 2025. AquaPipe:
A Quality-Aware Pipeline for Knowledge Retrieval and Large Language Models.
Proceedings of the ACM on Management of Data3, 1 (2025), 1–26.
[30] Shuo Yu, Mingyue Cheng, Qi Liu, Daoyu Wang, Jiqian Yang, Jie Ouyang, Yucong
Luo, Chenyi Lei, and Enhong Chen. 2025. Multi-source knowledge pruning for
retrieval-augmented generation: A benchmark and empirical study. InProceed-
ings of the 34th ACM International Conference on Information and Knowledge
Management. 3931–3941.
[31] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting
Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al .2025. Siren’s Song in the AI
Ocean: A Survey on Hallucination in Large Language Models.Computational
Linguistics51, 4 (2025), 1373–1418.
[32] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He,
Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, et al .2025. In-Depth Analysis
of Graph-Based RAG in a Unified Framework.Proceedings of the VLDB Endowment
18, 13 (2025), 5623–5637.
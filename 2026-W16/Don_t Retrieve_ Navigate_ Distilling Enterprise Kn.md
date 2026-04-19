# Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh

**Published**: 2026-04-16 03:05:37

**PDF URL**: [https://arxiv.org/pdf/2604.14572v1](https://arxiv.org/pdf/2604.14572v1)

## Abstract
Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics.

## Full Text


<!-- PDF content starts -->

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge
into Navigable Agent Skills for QA and RAG
Yiqun Sun Pengfei Wei Lawrence B. Hsieh
Magellan Technology Research Institute (MTRI)
{duke.sun, pengfei.wei, lawrence.hsieh}@mtri.co.jp
Abstract
Retrieval-Augmented Generation (RAG) grounds
LLM responses in external evidence but treats
the model as a passive consumer of search re-
sults: it never sees how the corpus is organized
or what it has not yet retrieved, limiting its abil-
ity to backtrack or combine scattered evidence.
We present CORPUS2SKILL, which distills a doc-
ument corpus into a hierarchical skill directory
offline and lets an LLM agent navigate it at serve
time. The compilation pipeline iteratively clusters
documents, generates LLM-written summaries at
each level, and materializes the result as a tree
of navigable skill files. At serve time, the agent
receives a bird’s-eye view of the corpus, drills
into topic branches via progressively finer sum-
maries, and retrieves full documents by ID. Be-
cause the hierarchy is explicitly visible, the agent
can reason about where to look, backtrack from
unproductive paths, and combine evidence across
branches. On WixQA, an enterprise customer-
support benchmark for RAG, CORPUS2SKILL
outperforms dense retrieval, RAPTOR, and agen-
tic RAG baselines across all quality metrics.
1. Introduction
Enterprise knowledge bases contain thousands of hetero-
geneous documents: support articles, product guides, pol-
icy documents, and FAQs (Gao et al., 2024). Large lan-
guage models alone cannot answer questions about these
corpora reliably: they lack access to private, frequently
updated content and are prone to hallucinating plausible
but incorrect details (Lewis et al., 2020; Gao et al., 2024).
Retrieval-Augmented Generation (RAG; Lewis et al., 2020)
was introduced to solve these problems by grounding LLM
responses in retrieved evidence. By embedding the query,
fetching similar passages from a vector index, and condi-
Preprint. April 17, 2026.
Navigate (Ours)Retrieve (RAG)
Retrieval Pipeline
EmbedAnswer
Skill C Skill A Skill B
idx-3 idx-4User
QueryLLM
LLM
Agent
doc doc docVectorDBTop-k
Docs
AnswerUser
QueryFigure 1.Retrieve vs. Navigate.Traditional RAG passively feeds
fixed passages to the LLM. CORPUS2SKILLdistills the corpus
into a navigable skill hierarchy that the agent actively explores,
backtracks, and drills into to locate evidence.
tioning the model’s answer on the retrieved context, RAG
enables LLMs to leverage enterprise-specific knowledge
they were never trained on, providing factual, up-to-date
answers while reducing hallucination. This pipeline has
become the dominant paradigm for enterprise QA, customer
support, and compliance workflows (Gao et al., 2024; Singh
et al., 2025). Yet it presents a structural limitation for more
complex queries:the LLM never sees the forest for the
trees.
Consider a support agent faced with the query “How do
I switch my business from sole proprietorship to LLC on
Wix?” This query spans multiple topics: account types, pay-
ment provider configuration, and legal entity management,
each covered in different sections of the knowledge base. A
flat retrieval system returns the top- kpassages closest to the
query embedding, which are likely to match on surface-level
terms (“sole proprietorship,” “LLC”) but miss the opera-
1arXiv:2604.14572v1  [cs.IR]  16 Apr 2026

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
tionally critical article explaining that account types cannot
be changed directly and require contacting support. The
LLM receives these fragments but has no understanding of
how the corpus is organized, what topic areas exist beyond
the retrieved set, or whether the best-matching documents
were even in the retrieval window. It cannot reason about
where else to look, whether its evidence is complete, or
how the retrieved passages relate to the broader knowledge
structure. In short, conventional RAG treats the LLM as
a passive consumer of search results, leaving the model’s
planning and reasoning capabilities largely unused.
Agentic RAG systems (Yao et al., 2023; Schick et al., 2023)
partially address this by allowing the LLM to issue mul-
tiple search queries iteratively. But even with tool access,
the agent operates without a map: it must guess productive
search terms, has no way to survey the corpus structure, and
cannot systematically narrow its focus from broad topics to
specific documents. Each search query is a shot in the dark,
guided only by the agent’s prior knowledge and whatever
fragments previous queries returned. For the business-type
query above, an agentic retrieval system would need to in-
dependently discover that the answer involves both the pay-
ments ecosystemandaccount verification policies — topics
whose connection is not apparent from keyword similarity
alone.
Hierarchical approaches such as RAPTOR (Sarthi et al.,
2024) and GraphRAG (Edge et al., 2024) organize docu-
ments into trees or graphs with LLM-generated summaries
at internal nodes. These summaries capture thematic struc-
ture that flat embeddings miss. However, these systems still
rely on embedding-based similarity search at query time:
RAPTOR indexes all tree nodes in a flat vector store, while
GraphRAG requires a graph database and entity extraction
pipeline. The hierarchy enriches the retrieval candidates
but remains invisible to the LLM at query time; the model
still receives search results without understanding the orga-
nizational structure that produced them. More recent work
pushes further: StructRAG (Li et al., 2025) restructures re-
trieved information into tables or graphs at inference time,
and BookRAG (Wang et al., 2025) builds hierarchical in-
dices with agent-based planners. Yet these systems still
invoke embedding-based search at query time and do not
give the agent direct access to the organizational structure
itself.
Meanwhile, a parallel line of work on LLM agent skills
(Wang et al., 2023; Jiang et al., 2026) has shown that agents
perform complex tasks more effectively when given struc-
tured, reusable knowledge modules that they can selectively
load and follow. Agent platforms now support filesystem-
based skill packages1that the agent discovers via metadata
1https://docs.anthropic.com/en/docs/
agents-and-tools/agent-skillsand reads on demand, a mechanism known as progressive
disclosure (Section 2.3). This suggests an alternative to re-
trieval: rather than searching for passages,distillthe corpus
into a navigable skill hierarchy and let the agent explore it.
We propose exactly this: make the hierarchy itself the pri-
mary interface for the LLM. Rather than searchingovera
hierarchy, the agent navigatesthroughit. This is the core
idea behind CORPUS2SKILL(Figure 1): a one-time offline
compilation phase transforms a document corpus into a nav-
igable skill tree, and at serve time, the LLM agent browses
this tree to find information, much like a human navigating
a well-organized file system or table of contents.
We evaluate CORPUS2SKILLon WixQA (Cohen et al.,
2025), an enterprise customer-support benchmark designed
for evaluating RAG systems over real-world knowledge
bases. CORPUS2SKILLachieves the highest quality across
all metrics, outperforming flat retrieval, RAPTOR, and a
multi-turn agentic RAG system, while requiring no em-
bedding index or vector database at serve time. These re-
sults suggest a complementary design point for knowledge-
grounded systems: invest computation at compile time to
make the corpusnavigable, rather than solely at query time
to make it searchable.
We summarize our contributions as follows.
1.We identify corpus navigability as an underexplored
dimension in current RAG systems: LLMs lack a struc-
tured overview of the knowledge base, which limits their
ability to systematically locate the best evidence.
2.We propose CORPUS2SKILL, a compile-then-navigate
framework: an offline pipeline transforms a raw corpus
into a hierarchical skill directory via iterative clustering
and LLM-based summarization, and at serve time an
LLM agent navigates this hierarchy through file brows-
ing and document lookup tools, requiring no embedding
index or vector database at serve time.
3.We evaluate on WixQA (Cohen et al., 2025) against
five baselines and conduct ablation studies on hierarchy
structure, exploration budget, and serving model choice.
2. Background: Skills in LLM Agents
2.1. From Tools to Skills
Modern LLM agents interact with external systems through
tools: atomic, stateless functions such as web search or
database queries (Schick et al., 2023; Yao et al., 2023).
Tools give an agent the ability toactbut do not tell ithowto
approach a complex task.Agent skillsfill this gap: reusable
modules that package procedural knowledge or multi-step
workflows into a form the agent can load and follow (Jiang
et al., 2026). Unlike a tool, a skill shapes reasoning over mul-
tiple steps, persisting across tool calls to guide the agent’s
2

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
decisions throughout a task. This distinction is increasingly
important as agents tackle longer-horizon tasks where choos-
ing the right sequence of tool calls requires domain-specific
strategy, not just tool availability.
Formally, Jiang et al. (2026) define a skill as a tuple S=
(C, π, T, R) : an applicability condition C, an execution
policy π, a termination condition T, and a reusable interface
R. This distinguishes skills from one-off plans (no reusable
interface), episodic memories (no execution policy), and
atomic tools (no multi-step structure).
2.2. Skill Representations
Skills take several forms in practice:
•Code-as-skill.V oyager (Wang et al., 2023) stores suc-
cessful action sequences as executable functions in a
growing library.
•Natural-language playbooks.SkillX (Wang et al., 2026)
and EvoSkill (Alzubi et al., 2026) distill agent experience
into structured textual instructions.
•Filesystem-based packages.A directory containing
a structured instruction file ( SKILL.md ) alongside op-
tional scripts and data assets. This is the pattern adopted
by recent agent platforms and the one CORPUS2SKILL
builds on.
Code-based skills offer precision but require execution en-
vironments; natural-language skills are portable and inter-
pretable at the cost of ambiguity. Filesystem-based packages
balance the two by combining structured text with auxiliary
resources that the agent can access on demand.
2.3. Progressive Disclosure
Loading all skill content at startup would overwhelm the
context window; loading nothing would leave the agent
unable to choose. The filesystem pattern resolves this via
progressive disclosure(Shneiderman, 1996):
1.Metadata(at startup): skill name and one-line descrip-
tion (<100 tokens each).
2.Instructions(on demand): the full SKILL.md is read
when the agent selects a skill.
3.Resources(as needed): supporting files remain on disk,
consuming zero tokens until accessed.
The agent thus maintains awareness of many skills while
paying the token cost only for those actually used.
2.4. From Procedural Skills to Informational Skills
In prior work, skills encodehow to perform a task: work-
flows, decision trees, or code patterns. CORPUS2SKILL
repurposes the same mechanism to encodewhat a corpus
contains. Each skill directory represents a topical parti-tion, withSKILL.md andINDEX.md files that summarize
its contents at increasing levels of detail. The agent fol-
lows the same progressive disclosure workflow (description
→SKILL.md→INDEX.md→ document) but navigates
an information hierarchy rather than executing procedu-
ral instructions. This reinterpretation—from procedural to
informational skills—is the central design insight of COR-
PUS2SKILL.
3. CORPUS2SKILLFramework
CORPUS2SKILLoperates in two phases: a one-time offline
compile phasethat transforms a raw document corpus into
a browsable skill tree, and a per-query onlineserve phase
where an LLM agent navigates this tree with full visibil-
ity into the corpus structure. Figure 2 illustrates the full
architecture.
3.1. Problem Setup
LetD={d 1, . . . , d N}be a corpus of Ndocuments. Given
a natural-language query q, the task is to produce an answer
agrounded in a subset Dq⊆ D of relevant documents.
Standard RAG systems select Dqvia a retrieval function
f(q,D)→ D qthat operates as a black box from the LLM’s
perspective: the model receives Dqbut has no visibility into
whatDcontains or whyD qwas chosen.
CORPUS2SKILLreplaces this opaque retrieval step with
agent-driven navigation. The compile phase constructs a
hierarchical skill forest S={s 1, . . . , s K}where each skill
skis a navigable directory tree of summaries covering a top-
ical partition of D. Following the agentic skills formalism
of Jiang et al. (2026), each skill can be viewed as a tu-
ple(C, π, T, R) : theapplicability condition Cis the skill’s
summary (the agent reads it to decide relevance); theexecu-
tion policy πis the navigation workflow (descend through
summaries, then retrieve documents); thetermination con-
dition Tis the agent’s judgment that sufficient evidence
has been gathered; and thereusable interface Ris the stan-
dardized file structure ( SKILL.md ,INDEX.md ) and the
getdocument tool. At serve time, the agent selects and
navigates skills autonomously, making the document selec-
tion process transparent and steerable.
3.2. Compile Phase
The compilation pipeline consists of four stages (implemen-
tation details and full prompts in Sections A and C).
Document loading and embedding.The compiler ac-
cepts directories containing standard document formats
(.md,.txt ,.json ,.jsonl ). Each document is as-
signed a content-hash-based unique identifier, ensuring de-
terministic, collision-free IDs that are independent of the
3

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Compile Phase
Serve Phase
Agent Envrionment
-> "To transfer your domain, first unlock it at your current registrar 
and obtain the authorization code ... "Compiled Agent Skills
DocumentsLLM Assisted Hierarchical Clustering
-> a1244: "T ransferring a Domain to W ix"
    a2124: "Costs of Domain T ransfers"
    a5211: "T roubleshooting Failed T ransfers"Turn 2: View dolnain-management/INDEX.md
Turn 3: get_document("a1244")-> "Covers domains, SEO, ads ... "
     Sub-groups: domain-management/, analytics/, seo-performance/ ...Document Store
get_document(id)
Turn 4: -> Grounded answerUser Querycode_execution(INDEX.md)Root Clusters
Embedding
Model(SKILL.md)
Meta Clusters
Document
CorpusGroups
Grounded
AnswerSummarizeSKILL.md
INDEX.mdSKILL.md
ClusteringSKILL.md
Clustering
ClusteringSkills
Clusters
Compiled
SkillsEmbed LLMINDEX.md INDEX.md
SummarizeEmbed LLM
a1244 a2124 a5211 a2453 a4312 a3123 a3243 a3223 a1235
Agent Interaction T race
Turn 1: view marketing-seo/SKILL.md
Auxiliary
ToolsINDEX.md INDEX.md INDEX.mdSummarizeLLM
(INDEX.md)Groups
Figure 2.Corpus2Skill system architecture. The compile phase (top) embeds documents, builds a multi-level cluster hierarchy with
LLM summarization at each level, and materializes the result as a forest of navigable skills. Root clusters map to skill directories
(SKILL.md ), sub-clusters map to group directories ( INDEX.md ), and leaf documents are stored in a separate document store. The serve
phase (bottom) uses an LLM agent with pre-compiled skills and two tools: file navigation ( codeexecution ) and document lookup
(getdocument).
source system’s naming scheme. Documents are truncated
to a maximum character length and embedded using a sen-
tence embedding model (Reimers & Gurevych, 2019), pro-
ducing a dense vector for each document. These embed-
dings capture semantic similarity between documents and
serve as the foundation for the subsequent clustering step.
Iterative hierarchical clustering.We build a multi-level
hierarchy through an iterative bottom-up process controlled
by two parameters: the branching ratio p(target number of
children per cluster) and the maximum number of top-level
clusters K. At each level, the current set of nembeddings
is partitioned into ⌈n/p⌉ groups via K-Means (Lloyd, 1982)
on L2-normalized vectors. Each resulting cluster is then
summarized by an LLM that reads representative member
texts and produces a concise description of the cluster’s top-
ical coverage, the types of questions the documents answer,
and key terms or features mentioned. These summaries are
re-embedded using the same sentence embedding model,
and the resulting embeddings become the input to the next
round of K-Means. This process repeats until the number
of remaining clusters falls below K, yielding a set of root
clusters that form the top level of the hierarchy.
This iterative embed-cluster-summarize loop is inspired by
the recursive tree construction in RAPTOR (Sarthi et al.,
2024), but adapted for our navigability goal. We use K-Means with hard assignments so that each document maps to
exactly one path in the directory tree, and we materialize the
hierarchy as browsable files rather than indexing it in a flat
vector store. These two choices, deterministic partitioning
and filesystem materialization, are what enable agent-driven
navigation at serve time. Small clusters below a minimum
size are merged into the nearest centroid to avoid orphan
groups.
Labeling.After the clustering hierarchy stabilizes, the
LLM generates a short (2 to 5 word) filesystem-safe label
for each non-leaf node. These labels serve a dual purpose:
they provide human-readable directory names for inspection
(e.g.,wix-commerce-monetization ) and, more im-
portantly, they give the agent semantically meaningful entry
points for topic routing. When the agent reads a SKILL.md
listing subgroups such as “wix-payments-ecosystem” or
“domain-dns-management,” the labels alone often suffice to
identify the relevant branch without reading full summaries.
Skill tree construction.The hierarchy is materialized into
a filesystem structure designed for agent navigation:
•Root clustersbecome top-level skill directories, each
containing a SKILL.md with the cluster summary and a
listing of child groups.
•Sub-clustersbecome subdirectories with INDEX.md
4

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
files that list either further subgroups or leaf document
identifiers alongside brief summaries of each document.
•Full document textis stored externally in a
documents.json file, referenced by document ID
from within the navigation files.
This separation between navigation metadata and document
content is a deliberate architectural choice. Navigation files
(SKILL.md ,INDEX.md ) remain small (typically under
2 KB each), containing only summaries and document iden-
tifiers. This keeps the agent’s per-step token cost low: read-
ing anINDEX.md to survey 20 documents costs far fewer
tokens than reading the documents themselves. Full docu-
ment content is accessed only when the agent has narrowed
its search, via a dedicated getdocument tool call. The
result is a forest of Kindependent skill trees, each rep-
resenting a topical domain within the corpus. The forest
structure allows the agent to reason about topic relevance at
the coarsest level before committing to a branch.2
Skill file design.Each SKILL.md andINDEX.md fol-
lows a uniform template with YAML frontmatter ( name ,
description ,level ,numdocuments ) that the
Skills API reads for progressive disclosure. Compilation
prompts producerouting-orientedsummaries: the LLM
identifies the topic area, question types, and key terms, so
that each file functions as a decision point for the navigating
agent. Each file also embeds actionable next-step instruc-
tions (e.g., “Use getdocument with the doc id”), so the
agent always knows which tool to invoke next.
Figure 3 shows abridged examples of both file types. The
SKILL.md (top) gives the agent a single-glance overview
of all subgroups within a skill; the leaf INDEX.md (bottom)
lists individual document IDs with concise titles for targeted
retrieval. Unabridged versions at all three hierarchy levels
are in Section B.
Navigation complexity.The hierarchy depth grows as
O(logpN): with branching ratio pandNdocuments, the
agent traverses at most L=⌈logpN⌉levels to reach a
leaf. At each level ℓ, the agent reads one navigation file
summarizing O(p) children. A single decision at level ℓ
reduces the candidate set from N/pℓ−1toN/pℓ, yielding a
multiplicative information funnel:
N÷p− − →N
p÷p− − →N
p2···÷p− − →N
pL≈1(1)
The total summaries inspected scale as O(p·logpN)rather
thanO(N) . For WixQA ( N=6,221 ,p=10 ), this gives L=3
levels and approximately 30 summaries to locate a single
2Anthropic’s current Skills API imposes limits of 8 skills per
request, 200 files and 30 MB per skill. These constraints informed
our default parameters but are not fundamental to the framework
design.SKILL.md(root level, abridged)
---
name: skill-01-wix-commerce-monetization
description: >
Full Wix commerce and monetization
ecosystem, from storefronts to payments.
level: 3
numdocuments: 1513
---
## Contents
### Sub-groups (directories)
-group-00-wix-restaurants/(159 docs):
Wix Restaurants orders, menus...
-group-01-wix-payments-ecosystem/(87):
Payment processing, account mgmt...
-group-02-wix-pos-systems/(110 docs)
[...16 subgroups total]
INDEX.md(leaf level, verbatim)
---
name: L1-C298
description: >
Fraud prevention and payment security
within Wix Payments.
level: 1 num documents: 5
---
### Documents (5 items)
-`56ae76`: Best Practices for Accepting
Card Payments
-`448d79`: About the ‘Liability Shift’
-`bb73d9`: Declined Payment Due to
Suspected Fraud
-`c3b01a`: About Suspicious Payments
-`d22b8c`: Double Charge on Statement
Figure 3.Abridged SKILL.md (top) and leaf INDEX.md (bot-
tom) from the WixQA compilation. The SKILL.md provides a
routing overview of 1,513 documents; the INDEX.md lists indi-
vidual document IDs with titles for targeted retrieval.
document among thousands. Note that this bound counts
summaries inspected under an idealized single-path traver-
sal; in practice, the actual token cost is higher because the
Skills API includes previously loaded navigation files in
each subsequent call, and the agent may explore multiple
branches. The per-query costs reported in Section 4 reflect
these real-world overheads. A 100,000-document corpus
withp=10would require only one additional level (L=5).
3.3. Serve Phase
At serve time, the pre-compiled skill directories are up-
loaded to Anthropic’s Skills API. The API implements pro-
gressive disclosure: skill names and one-line descriptions
are pre-loaded into the agent’s context, but full file content
is loaded only when the agent explicitly reads a file. This
design is critical for token efficiency. Pre-loading all naviga-
tion files would flood the context with thousands of tokens
of irrelevant summaries; loading nothing would leave the
agent blind to the corpus. Progressive disclosure strikes a
balance: the agent sees just enough (skill names and descrip-
tions, approximately 200 tokens total) to make an informed
first decision, then incrementally loads detail as it descends.
Tools.The agent receives two tools: (1)
codeexecution , which allows browsing SKILL.md
5

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
andINDEX.md files through view commands, providing
the agent with full visibility into the hierarchy; and (2)
getdocument(doc id), a custom tool that retrieves
the full text of a document from the document store given
its identifier. The two-tool design reflects the two types
of information the agent needs:navigational(where to
look) andevidential(what to cite). Keeping them separate
means the agent can survey many document summaries
cheaply before committing to the token cost of reading full
documents.
Navigation workflow.Because skill descriptions are pre-
loaded, the agent begins each query with a bird’s-eye view
of the corpus, knowing what topics exist and roughly how
many documents each topic covers. A typical query pro-
ceeds in 2 to 3 turns:
1.The agent identifies the most relevant skill from pre-
loaded descriptions and reads its SKILL.md to under-
stand the subgroup structure.
2.The agent navigates into the relevant subgroup’s
INDEX.md , which lists document identifiers with brief
titles and summaries.
3.The agent calls getdocument to retrieve the full text
of one or more promising documents and synthesizes a
grounded answer.
At each step, the agent makes an informed decision guided
by summaries describing each branch. If a SKILL.md
lists ten subgroups and the agent has only explored one, it
knows nine remain—an awareness that flat retrieval cannot
provide. This structural visibility enables two behaviors:tar-
geted backtracking(abandoning an unpromising branch for
a more relevant one) andcross-branch synthesis(combining
evidence from multiple subgroups within or across skills).
The system prompt enforces agrounding rule(every claim
must trace to a retrieved document, not a summary file) and
aconcise-answer format(word limits, no preamble); see
Section D.
4. Experiments
4.1. Setup
Dataset.We evaluate on WixQA (Cohen et al., 2025), a
customer support QA benchmark built from the Wix knowl-
edge base. The corpus contains 6,221 support articles cov-
ering website building, e-commerce, SEO, marketing, and
platform features. The evaluation set consists of 200 expert-
written questions with gold-standard answers and gold arti-
cle IDs.
Compilation.We compile the WixQA corpus with
branching ratio p=10 and maximum top-level clusters K=7 .
This produces a 3-level hierarchy with 6 top-level skills, 665navigation files, and a 13 MB document store. Compilation
takes 6.5 minutes on a 32-CPU server.
Baselines.All methods use the same LLM for answer
generation and the same evaluation protocol. We compare
five baselines spanning three retrieval paradigms:BM25
(sparse keyword retrieval (Robertson & Zaragoza, 2009)),
Dense(Qwen3-Embedding-0.6B (Zhang et al., 2025) with
a FAISS index (Johnson et al., 2019)),Hybrid(Reciprocal
Rank Fusion (Cormack et al., 2009) of BM25 and Dense),
RAPTOR(hierarchical tree retrieval (Sarthi et al., 2024)
with UMAP+GMM+BIC clustering), andAgentic(an LLM
agent with iterative access to BM25, Dense, and Hybrid
tools). The first three are single-shot (top-5 passages, one
LLM call); RAPTOR uses collapsed tree retrieval; Agentic
is allowed up to 10 interaction rounds. Full implementation
details are in Section H.
Metrics.We report four lexical overlap metrics:Token F1
(harmonic mean of token-level precision and recall against
the gold answer),BLEU(Papineni et al., 2002) (n-gram
precision with brevity penalty),ROUGE-1andROUGE-2
(Lin, 2004) (unigram and bigram recall). We additionally re-
port two LLM-judged metrics following the LLM-as-judge
paradigm (Zheng et al., 2023), inspired by RAGAS (Es
et al., 2024):Factualitymeasures how factually correct the
generated answer is relative to the gold answer (1–5 scale,
normalized to 0–1), andContext Recallmeasures how well
the retrieved context covers the essential claims in the gold
answer (same scale). We also report input token usage and
per-query cost. Details on metric computation and fairness
standardization across methods are in Section I.
4.2. Main Results
Quality.Table 1 presents the main results.COR-
PUS2SKILLachieves the highest scores across all six
quality metrics.On Token F1, CORPUS2SKILLscores
0.460, a 19% relative improvement over Agentic (0.388)
and 27% over Dense retrieval (0.363). On Factuality, COR-
PUS2SKILL(0.729) exceeds both Agentic (0.724) and RAP-
TOR (0.675). On Context Recall, CORPUS2SKILLleads
by a clear margin (0.652 vs. 0.616 for RAPTOR and 0.481
for Agentic), indicating that the navigation-based approach
surfaces more relevant content than either vector search or
multi-turn agentic retrieval.
Hierarchical methods outperform flat retrieval.Both
RAPTOR and CORPUS2SKILLsubstantially outperform the
three flat retrieval baselines (BM25, Dense, Hybrid), con-
firming that hierarchical document organization improves
answer quality. RAPTOR’s collapsed tree retrieval achieves
Factuality of 0.675 and Context Recall of 0.616, substan-
tially above flat methods (Dense: 0.536/0.450, BM25:
6

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Table 1.Main results on WixQA (200 expert queries). CORPUS2SKILLachieves the highest quality across all metrics. Quality cells
shaded green (darker = better); cost cells shaded red (darker = higher).
MethodQuality Metrics Cost
F1↑BLEU↑R-1↑R-2↑Fact.↑CtxR↑In tok.↓$/query↓
BM25 0.342 0.060 0.364 0.119 0.470 0.386 708 0.007
Dense 0.363 0.074 0.382 0.141 0.536 0.450 656 0.008
Hybrid 0.360 0.071 0.380 0.137 0.524 0.410 698 0.008
RAPTOR 0.389 0.109 0.406 0.189 0.675 0.616 1,807 0.012
Agentic 0.388 0.099 0.404 0.186 0.724 0.481 25,807 0.098
CORPUS2SKILL 0.460 0.137 0.476 0.231 0.729 0.65253,487 0.172
Query:“I need to switch my business type from sole prop to
LLC in order to use an EIN. ”
Step 1: view wix-commerce-monetization/SKILL.md
-> 10 subgroups; picks "wix-payments-
ecosystem" (87 docs)
Step 2: view wix-payments-ecosystem/INDEX.md
-> 8 leaf groups; picks group covering
account management
Step 3: view group-05/INDEX.md
-> doc d56cc79: "Changing Your Wix
Payments Account Type"
Step 4: get_document("d56cc797742e")
-> Retrieves full article
Answer: "You cannot change your account
type directly. Contact Wix Customer Care."
Figure 4.Trace 1: Direct navigation. The agent descends through
a single skill branch to find the relevant document in 4 steps.
0.470/0.386). Cluster summaries surface thematic connec-
tions that flat embeddings miss, allowing the retrieval system
to match queries to relevant document groups rather than
individual passages.
Cost analysis.CORPUS2SKILLcosts $0.17 per query, ap-
proximately 1.75 ×Agentic and 14 ×RAPTOR (Table 1).
The cost is dominated by input tokens (53K per query),
as the Skills API includes navigation file content in each
call. Despite this, CORPUS2SKILLproduces only 752 out-
put tokens per query—roughly half of Agentic (1,391)—
suggesting that navigating to the right documents yields
more targeted answers.
4.3. Case Study: Navigation in Action
We present two representative traces showing how COR-
PUS2SKILLnavigates the hierarchy (Figures 4 and 5).
Patterns.These traces illustrate two common navigation
patterns:direct descent(Figure 4: skill →subgroup →
leaf→document) andcross-branch exploration(Figure 5:
the agent visits multiple subgroups within the same skill to
combine evidence). In both cases, the agent’s decisions are
guided by summaries at each level. Full traces with raw tool
calls are provided in Section E.Query:“I want to change the currency for my course. ”
Step 1: view wix-commerce-monetization/SKILL.md
-> 15 subgroups; "wix-online-programs"
and "wix-billing-documents" relevant
Step 2: view wix-online-programs/INDEX.md
-> 5 leaf groups; picks group covering
program settings (18 docs)
Step 3: view group-03/INDEX.md
-> doc 7a9dc6: "Online Programs:
Editing Your Program Settings"
Step 4: get_document("7a9dc6f2de90")
-> Program settings doc; mentions
currency is a site-level setting
Step 5: view wix-billing-documents/INDEX.md
-> picks group covering currency
management (18 docs)
Step 6: view group-00/INDEX.md
-> doc 29139b: "Setting Your Currency
for Accepting Payments"
Step 7: get_document("29139bd7c379")
Answer: "Go to Settings > Language &
Region > Currency. This site-level setting
applies to all Wix products including
Online Programs."
Figure 5.Trace 2: Cross-branch navigation. The agent visits
two subgroups within the same skill to combine evidence from
different topic branches.
Table 2.Ablation: cluster structure. Narrow trees match or exceed
the default; wide trees degrade quality.
VariantF1↑Fact.↑CtxR↑$/q↓
Narrow (p=5) 0.461 0.736 0.674 0.186
Default (p=10) 0.460 0.729 0.652 0.172
Wide (p=20) 0.361 0.410 0.355 0.242
4.4. Ablation Studies
We conduct three ablation studies to characterize how
compile-time and serve-time design choices affect the
quality-cost tradeoff.
Cluster structure.We vary the branching ratio pand top-
cluster count Kto produce hierarchies of different shapes
(Table 2). With p=5, the narrow branching ratio creates a
4-level hierarchy with 3 top-level skills. With deeper hierar-
chies or larger corpora, the file count per skill can exceed
the API’s 200-file limit. We introduce acompact modethat
merges near-leaf INDEX.md files into their parent: instead
7

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Table 3.Ablation: agent exploration budget. Quality is stable
across budgets, indicating effective hierarchy organization.
Max roundsF1↑Fact.↑CtxR↑$/q↓
5 0.453 0.721 0.636 0.170
10 0.461 0.748 0.667 0.172
20 0.460 0.729 0.652 0.172
Table 4.Ablation: serving LLM choice. A cheaper model achieves
comparable quality at roughly half the cost.
ModelF1↑Fact.↑CtxR↑$/q↓
Claude Sonnet 4.6 0.460 0.729 0.652 0.172
Claude Haiku 4.5 0.423 0.645 0.705 0.088
of creating a separate subdirectory for each lowest-level
group, the parent’s INDEX.md contains labeled sections
listing each sub-group’s documents inline. This reduces
file count by up to 80% while preserving all navigational
information. The narrow hierarchy slightly outperforms the
default on all quality metrics (F1 0.461 vs. 0.460, Factuality
0.736 vs. 0.729, Context Recall 0.674 vs. 0.652) at a modest
8% cost increase ($0.186 vs. $0.172 per query). The deeper
hierarchy provides more fine-grained topic groupings, and
the compact merged files give the agent a denser view of
the leaf-level documents in fewer navigation steps. In the
wide-shallow variant ( p=20 ), F1 drops 21% and Factuality
drops 44% while costing 41% more. With only 2 extremely
broad skills covering approximately 3,000 documents each,
theSKILL.md summaries become too generic for effective
topic routing.
Agent exploration budget.We vary the maximum al-
lowed interaction rounds while keeping the default skill tree
(Table 3). The exploration budget has minimal impact on
quality: even a budget of 5 achieves F1 of 0.453, only 1.5%
below the full-budget configuration. Factuality is nearly
identical across variants (0.721 to 0.748), and the marginal
cost difference is negligible.The hierarchical structure
is sufficiently well-organized that the agent rarely needs
extended exploration.
Serving LLM choice.We swap only the serving model
while keeping the same compiled skill tree (Table 4). A
smaller, cheaper model achieves comparable quality at
roughly half the cost ($0.088 vs. $0.172 per query). While
F1 drops by 8% and Factuality by 12%, Context Recallin-
creasesfrom 0.652 to 0.705, suggesting the cheaper model
navigates the hierarchy more thoroughly, retrieving more rel-
evant documents even if its answer synthesis is slightly less
polished.At $0.088 per query, the cheaper model offers
a compelling cost-quality tradeoff.This also demonstrates
that the compiled skill tree is robust to the choice of serving
model: the quality of the hierarchy, not the sophistication ofthe navigator, is the primary driver of retrieval performance.
5. Limitations
Cost.The per-query cost of CORPUS2SKILLis dominated
by input tokens: the Skills API includes navigation file con-
tent in every call, so even relatively small SKILL.md and
INDEX.md files accumulate significant token counts across
turns. CORPUS2SKILLis best suited for high-value queries
where answer quality justifies the cost premium, such as
complex multi-hop support questions or compliance-critical
lookups. For high-volume, low-stakes queries, single-shot
retrieval or RAPTOR remain cost-effective. Prompt caching
could substantially reduce this overhead by reusing repeated
skill file content across queries within the same session.
API constraints.The framework relies on Anthropic’s
Skills API, which imposes limits of 8 skills per request, 200
files per skill, and 30 MB per skill. These constraints shape
the hierarchy design: the 8-skill limit caps top-level clusters,
and the 200-file budget constrains tree depth before requir-
ing compact merging (Section 4.4). The WixQA corpus fits
comfortably (6 skills, ≤133 files each), but larger corpora
would require compact merging or multi-level partitioning.
As Skills APIs evolve and constraints relax, even narrower
hierarchies could improve navigation precision.
Hard single-path clustering.Each document is assigned
to exactly one cluster path, a deliberate choice that en-
ables filesystem materialization and deterministic naviga-
tion. However, documents that span multiple topics (e.g.,
billingandsubscriptions) can only appear under one branch,
creating a routing bottleneck when queries target the other
topic. Our failure analysis (Section G) confirms that top-
level routing errors account for the majority of failures. Soft
or multi-parent assignment could mitigate this at the cost of
duplicated content and more complex navigation; we leave
this exploration to future work.
Compilation.The one-time compilation costs $5–$10 in
LLM calls plus 6.5 minutes of wall-clock time, negligi-
ble for enterprise deployments handling thousands of daily
queries. The pipeline is fully automated and deterministic
given fixed random seeds, but does not yet support incremen-
tal updates—adding documents requires recompilation.
6. Related Work
Cluster-based document organization.Scatter/Gather
(Cutting et al., 1992) introduced the paradigm of navigating
a pre-computed cluster organization rather than matching
by similarity, grounded in the cluster hypothesis (van Rijs-
bergen, 1979; Liu & Croft, 2004). Where Scatter/Gather
required users to manually select and refine clusters in an
8

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
interactive loop, CORPUS2SKILLdelegates this navigation
entirely to an LLM agent that reasons about cluster sum-
maries at each level.
Hierarchical summarization for retrieval.RAPTOR
(Sarthi et al., 2024) recursively clusters and summarizes
text chunks into a tree, retrieving via layer-by-layer traver-
sal or collapsed-tree vector search. CORPUS2SKILLshares
the compile-time embed-cluster-summarize loop but mate-
rializes the hierarchy as navigable files and replaces fixed
traversal with agentic navigation. GraphRAG (Edge et al.,
2024) builds a knowledge graph with community-level sum-
maries; unlike CORPUS2SKILL, it requires a graph database
at serve time. HiRAG (Huang et al., 2025) augments RAG
with a hierarchical knowledge graph that interleaves entity
and community levels; it still relies on graph retrieval at
query time rather than agent-driven file navigation.
Retrieval-augmented generation.RAG (Lewis et al.,
2020) grounds LLM responses in retrieved passages, with
dense passage retrieval (Karpukhin et al., 2020) and late-
interaction models (Santhanam et al., 2022) pushing re-
trieval quality. Self-RAG (Asai et al., 2024) teaches the
model to critique its own retrieval; IRCoT (Trivedi et al.,
2023) interleaves retrieval with chain-of-thought reasoning
for multi-hop questions, using intermediate reasoning steps
to formulate better queries. Recent surveys (Gao et al., 2024;
Singh et al., 2025) organize RAG along axes of granularity,
timing, and agentic control. CORPUS2SKILLoccupies a dis-
tinct point on this spectrum: rather than improving retrieval,
it replaces embedding-based search with agent-driven file
navigation over a pre-compiled hierarchy.
Structure-aware and agentic RAG.StructRAG (Li et al.,
2025) dynamically restructures retrieved information into
tables, graphs, or catalogues at inference time, letting the
LLM reason over structured views. BookRAG (Wang et al.,
2025) builds a hierarchical index mirroring a document’s
table of contents and uses an agent-based planner to query it.
A-RAG (Du et al., 2026) exposes hierarchical retrieval in-
terfaces (keyword, semantic, chunk-read) to an LLM agent,
and SPD-RAG (Akay et al., 2026) assigns a dedicated sub-
agent per document in a multi-agent pipeline. NaviRAG
(Dai et al., 2026) builds a hierarchical view of documents
and uses an LLM agent to actively navigate it, detecting in-
formation gaps and retrieving at the appropriate granularity.
HCAG (Wu & Deng, 2026) constructs a multi-resolution
knowledge base offline and performs hierarchical retrieval
with a multi-agent discussion at query time. All of these sys-
tems retain some form of retrieval or search infrastructure
at serve time. CORPUS2SKILLdiffers by materializing the
hierarchy as a static filesystem of skill files and navigating it
with a single agent using file browsing and document lookup
rather than embedding-based search.Tool-use agents.Toolformer (Schick et al., 2023) and
ReAct (Yao et al., 2023) established that LLMs can au-
tonomously select and use external tools. CORPUS2SKILL
extends this paradigm with domain-specific, read-only tools
(file browsing and document lookup) over a static hierarchy,
making navigation deterministic and reproducible given the
same compiled skill tree.
Agent skill construction.V oyager (Wang et al., 2023)
builds a skill library from successful action sequences as
code; SkillX (Wang et al., 2026) and EvoSkill (Alzubi
et al., 2026) distill agent trajectories into reusable knowl-
edge. These create skills fromtrajectories, whereas COR-
PUS2SKILLcreates skills fromdocument corporavia clus-
tering and summarization, a fundamentally different input
modality.
Positioning.CORPUS2SKILLcombines offline hierarchi-
cal clustering, materialization as navigable skill files, and
LLM-driven top-down navigation into a system that replaces
embedding-based retrieval with agent-driven file browsing
at serve time. While NaviRAG and HCAG share the mo-
tivation of active hierarchical navigation, CORPUS2SKILL
is distinguished by its filesystem-based skill representation,
progressive disclosure via SKILL.md /INDEX.md files,
and reliance on a standard code-execution tool rather than
custom retrieval APIs. Table 6 in the Appendix provides a
detailed feature comparison.
7. Conclusion
CORPUS2SKILLdemonstrates that giving an LLM agent a
structured, browsable overview of a document corpus leads
to higher-quality answers than providing it with retrieved
passages alone.The key insight is that modern LLM
agents are capable navigators when given the right map:
a pre-compiled hierarchy of summaries and topic clusters
enables informed, top-down exploration that exploits the
agent’s reasoning abilities rather than treating it as a pas-
sive consumer of search results. Because the agent reads
summaries describing each branch, it can make informed
routing decisions and combine evidence across branches
when queries span multiple topics—an advantage that flat
retrieval, which offers no corpus overview, cannot match.
While the approach incurs higher per-query cost due to nav-
igation overhead, it achieves the highest quality on WixQA
and suggests a complementary design point for knowledge-
grounded systems: invest computation at compile time to
make the corpus navigable, rather than at query time to
make it searchable. Hierarchy depth grows logarithmically
(O(logpN)), so the framework scales to larger corpora.
Future work includes evaluation on more domains, incre-
mental compilation, and prompt caching to cut query cost.
9

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Impact Statement
CORPUS2SKILLis designed for enterprise knowledge-
grounded QA, a setting where incorrect answers can lead
to real-world harm (e.g., wrong compliance guidance or
billing instructions). Like all LLM-based systems, it can
produce confidently wrong answers when the agent nav-
igates to irrelevant documents or misinterprets retrieved
content. Hierarchical summaries produced during compi-
lation may also expose corpus structure that organizations
consider sensitive. Hard clustering means that documents
spanning multiple topics are routed to a single branch, cre-
ating blind spots that may disproportionately affect certain
query types. Finally, because compilation is a batch process,
the system cannot reflect real-time corpus updates, risking
stale or outdated answers until recompilation occurs. We
recommend human-in-the-loop verification for high-stakes
deployments.
References
Akay, Y . C., Kartal, M. Y ., Alparslan, E., Ortakoyluoglu,
F., and Akpinar, A. SPD-RAG: Sub-agent per docu-
ment retrieval-augmented generation.arXiv preprint
arXiv:2603.08329, 2026.
Alzubi, S., Provenzano, N., Bingham, J., Chen, W., and Vu,
T. EvoSkill: Automated skill discovery for multi-agent
systems.arXiv preprint arXiv:2603.02766, 2026.
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
RAG: Learning to retrieve, generate, and critique through
self-reflection. InInternational Conference on Learning
Representations, 2024.
Cohen, D., Burg, L., Pykhnivskyi, S., Gur, H., Kovynov, S.,
Atzmon, O., and Barkan, G. WixQA: A multi-dataset
benchmark for enterprise retrieval-augmented generation.
arXiv preprint arXiv:2505.08643, 2025.
Cormack, G. V ., Clarke, C. L. A., and Buettcher, S. Recip-
rocal rank fusion outperforms Condorcet and individual
rank learning methods.Proceedings of the 32nd Interna-
tional ACM SIGIR Conference, pp. 758–759, 2009.
Cutting, D. R., Karger, D. R., Pedersen, J. O., and Tukey,
J. W. Scatter/Gather: A cluster-based approach to brows-
ing large document collections. InProceedings of the
15th Annual International ACM SIGIR Conference, pp.
318–329, 1992.
Dai, J., Wu, D., Chen, Y ., Zeng, Z., Yan, Y ., Liu, Z., and
Sun, M. NaviRAG: Towards active knowledge naviga-
tion for retrieval-augmented generation.arXiv preprint
arXiv:2604.12766, 2026.Du, M., Xu, B., Zhu, C., Wang, S., Wang, P., Wang, X., and
Mao, Z. A-RAG: Scaling agentic retrieval-augmented
generation via hierarchical retrieval interfaces.arXiv
preprint arXiv:2602.03442, 2026.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody,
A., Truitt, S., Metropolitansky, D., Osazuwa Ness, R.,
and Larson, J. From local to global: A Graph RAG
approach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
Es, S., James, J., Espinosa Anke, L., and Schockaert, S.
RAGAS: Automated evaluation of retrieval augmented
generation. InProceedings of the 18th Conference of the
European Chapter of the Association for Computational
Linguistics: System Demonstrations, pp. 150–163, 2024.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., Wang, M., and Wang, H. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2024.
Huang, H., Huang, Y ., Yang, J., Pan, Z., Chen, Y ., Ma, K.,
Chen, H., and Cheng, J. Retrieval-augmented generation
with hierarchical knowledge. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2025, pp.
6044–6060, 2025.
Jiang, Y ., Li, D., Deng, H., Ma, B., Wang, X., Wang, Q.,
and Yu, G. SoK: Agentic skills – beyond tool use in LLM
agents.arXiv preprint arXiv:2602.20867, 2026.
Johnson, J., Douze, M., and J ´egou, H. Billion-scale similar-
ity search with GPUs.IEEE Transactions on Big Data, 7
(3):535–547, 2019.
Karpukhin, V ., O ˘guz, B., Min, S., Lewis, P., Wu, L., Edunov,
S., Chen, D., and Yih, W.-t. Dense passage retrieval
for open-domain question answering.Proceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing, pp. 6769–6781, 2020.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., Riedel, S., and Kiela, D. Retrieval-augmented gener-
ation for knowledge-intensive NLP tasks. InAdvances
in Neural Information Processing Systems, volume 33,
2020.
Li, Z., Chen, X., Yu, H., Lin, H., Lu, Y ., Tang, Q., Huang,
F., Han, X., Sun, L., and Li, Y . StructRAG: Boosting
knowledge intensive reasoning of LLMs via inference-
time hybrid information structurization. InInternational
Conference on Learning Representations, 2025.
Lin, C.-Y . ROUGE: A package for automatic evaluation
of summaries. InText Summarization Branches Out, pp.
74–81, 2004.
10

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Liu, X. and Croft, W. B. Cluster-based retrieval using lan-
guage models.Proceedings of the 27th Annual Interna-
tional ACM SIGIR Conference, pp. 186–193, 2004.
Lloyd, S. P. Least squares quantization in PCM.IEEE Trans-
actions on Information Theory, 28(2):129–137, 1982.
McInnes, L., Healy, J., and Melville, J. UMAP: Uniform
manifold approximation and projection for dimension
reduction.arXiv preprint arXiv:1802.03426, 2018.
McLachlan, G. J. and Peel, D.Finite Mixture Models. Wiley,
2000.
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. BLEU:
A method for automatic evaluation of machine transla-
tion. InProceedings of the 40th Annual Meeting of the
Association for Computational Linguistics, pp. 311–318,
2002.
Reimers, N. and Gurevych, I. Sentence-BERT: Sentence
embeddings using Siamese BERT-networks. InProceed-
ings of the 2019 Conference on Empirical Methods in
Natural Language Processing, pp. 3982–3992, 2019.
Robertson, S. and Zaragoza, H. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends
in Information Retrieval, 3(4):333–389, 2009.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C.,
and Zaharia, M. ColBERTv2: Effective and efficient
retrieval via lightweight late interaction. InProceedings
of the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics, pp.
3715–3734, 2022.
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A.,
and Manning, C. D. RAPTOR: Recursive abstractive
processing for tree-organized retrieval. InInternational
Conference on Learning Representations, 2024.
Schick, T., Dwivedi-Yu, J., Dess `ı, R., Raileanu, R., Lomeli,
M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Tool-
former: Language models can teach themselves to use
tools.Advances in Neural Information Processing Sys-
tems, 36, 2023.
Schwarz, G. Estimating the dimension of a model.The
Annals of Statistics, 6(2):461–464, 1978.
Shneiderman, B. The eyes have it: A task by data type
taxonomy for information visualizations. InProceedings
of the IEEE Symposium on Visual Languages, pp. 336–
343, 1996.
Singh, A., Ehtesham, A., Kumar, S., Khoei, T. T., and Vasi-
lakos, A. V . Agentic retrieval-augmented generation: A
survey on agentic RAG.arXiv preprint arXiv:2501.09136,
2025.Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions. InProceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
10014–10037, 2023.
van Rijsbergen, C. J.Information Retrieval. Butterworth-
Heinemann, 2nd edition, 1979.
Wang, C., Yu, Z., Xie, X., Yao, W., Fang, R., Qiao, S.,
Cao, K., Zheng, G., Qi, X., Zhang, P., and Deng, S.
SkillX: Automatically constructing skill knowledge bases
for agents.arXiv preprint arXiv:2604.04804, 2026.
Wang, G., Xie, Y ., Jiang, Y ., Mandlekar, A., Xiao, C., Zhu,
Y ., Fan, L., and Anandkumar, A. V oyager: An open-ended
embodied agent with large language models. InAdvances
in Neural Information Processing Systems, 2023.
Wang, S., Zhou, Y ., and Fang, Y . BookRAG: A hierarchi-
cal structure-aware index-based approach for retrieval-
augmented generation on complex documents.arXiv
preprint arXiv:2512.03413, 2025.
Wu, Y . and Deng, X. HCAG: Hierarchical abstraction and
retrieval-augmented generation on theoretical repositories
with LLMs.arXiv preprint arXiv:2603.20299, 2026.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K., and Cao, Y . ReAct: Synergizing reasoning and act-
ing in language models. InInternational Conference on
Learning Representations, 2023.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang, B.,
Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou,
J. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176, 2025.
Zheng, L., Chiang, W.-L., Sheng, Y ., Zhuang, S., Wu, Z.,
Zhuang, Y ., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H.,
Gonzalez, J. E., and Stoica, I. Judging LLM-as-a-judge
with MT-Bench and chatbot arena. InAdvances in Neural
Information Processing Systems, 2023.
11

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
A. Compilation Details
Top-level skill labels.The 6 skills produced by the de-
fault configuration ( p=10 ,K=7 ) on the WixQA corpus are
shown in Table 5.
Table 5.Top-level skills produced by CORPUS2SKILLon WixQA.
Skill Label Docs Coverage
business-tools 931 Scheduling, CRM
commerce 1,513 Store, payments
content-media 1,074 Blog, galleries
platform-guide 1,161 Editor, domains
marketing-seo 553 SEO, analytics
platform-overview 989 Members, apps
Embedding.Documents are embedded using
Qwen/Qwen3-Embedding-0.6B, a 622M-parameter
model producing 1024-dimensional embeddings. We use
the document prompt template with batch size 32. This
model is chosen for its strong performance-to-size ratio and
ability to run on CPU.
Summarization.The compilation pipeline uses direct
LLM API calls (not an agent loop) for summarization and
labeling. Level-1 summaries condense up to 15 document
texts (600 characters each); deeper-level summaries con-
dense up to 20 child summaries (300 characters each). Sum-
marization is performed concurrently using async calls with
a semaphore of 20 for batch efficiency.
File counts.The compilation produces 665 nav-
igation files ( SKILL.md +INDEX.md ) and a
documents.json file storing the full text of all
6,221 documents (13 MB). Total output size is 16 MB.
Document-level summarization prompt.Used at the
lowest level to summarize a cluster of raw documents.
You are summarizing a cluster of related
documents from a knowledge base. Write a
2-3 sentence summary that captures:
1. The common TOPIC area these documents
cover
2. The types of QUESTIONS these documents
answer
3. Key TERMS or features mentioned across
documents
Be specific and concrete. Name actual
features, products, or processes.
Documents ({N} total, showing up to 15):
--- Document 1 ---
{doc_text[:600]}--- Document 2 ---
{doc_text[:600]}
[...]
Summary:
Cluster-level summarization prompt.Used at higher
levels to summarize a group of sub-cluster summaries into
a broader overview.
You are summarizing a level-{L} grouping
of {N} sub-groups from a knowledge base.
Each sub-group already has a summary below.
Write a 2-3 sentence overview that captures:
1. The broad DOMAIN these sub-groups cover
2. The range of TOPICS within this domain
3. What types of user QUESTIONS this group
can answer
Be specific -- name the main product areas,
features, or workflows.
Sub-group summaries:
- Sub-group 1: {summary[:300]}
- Sub-group 2: {summary[:300]}
[...]
Overview:
Labeling prompt.Used to generate a filesystem-safe di-
rectory name from a cluster summary. The response is
post-processed: lowercased, non-alphanumeric characters
replaced with hyphens, truncated to 50 characters.
Generate a short (2-5 word) filesystem-safe
label for this cluster. Use lowercase,
hyphens instead of spaces. No quotes.
Summary: {summary[:500]}
B. Skill Structure Examples
We show the full content of representative navigation files
from the WixQA compilation (default configuration: p=10 ,
K=7 ). These files are what the agent reads during naviga-
tion.
Root-level SKILL.md .This file sits at the top of the
skill-01-wix-commerce-monetization skill di-
rectory, the largest skill covering 1,513 documents across
16 subgroups. The agent reads this file in a single tool call
and uses the subgroup summaries to decide which branch
to explore.
---
name: skill-01-wix-commerce-monetization
description: >
This group covers the full Wix commerce
12

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
and monetization ecosystem, spanning every
major product vertical and workflow
available to Wix merchants and business
owners. Topics range from storefront and
product management (Wix Stores,
Restaurants, POS, Print/Fulfillment) to
financial operations (payments, taxes,
refunds, invoices, pricing plans,
subscriptions, and promotional tools like
coupons and gift cards)...
level: 3
num_documents: 1513
---
## Overview
This group covers the full Wix commerce and
monetization ecosystem, spanning every major
product vertical and workflow. Topics range
from storefront and product management (Wix
Stores, Restaurants, POS, Print/Fulfillment)
to financial operations (payments, taxes,
refunds, invoices, pricing plans,
subscriptions, and promotional tools), as
well as loyalty programs, partner ecosystems,
and learning/course delivery via Wix Online
Programs. Users can get answers about setting
up and running an online or in-person
business on Wix.
## Contents
### Sub-groups (directories)
Read the INDEX.md in each sub-group to
understand what it covers.
-**group-00-wix-restaurants-platform/ **
(159 docs): Covers the Wix Restaurants
platform in its entirety, spanning both
the original and newer versions of Wix
Restaurants Orders and Menus apps.
-**group-01-wix-payments-ecosystem/ **
(87 docs): Covers Wix’s payment ecosystem,
spanning payment processing, account
management, and financial operations.
-**group-02-wix-pos-systems/ **(110 docs):
Covers Wix’s Point of Sale (POS)
ecosystem, both tablet-based Retail POS
and smartphone-based Mobile POS.
-**group-03-wix-stores-product-management/ **
(99 docs): Covers Wix Stores product and
storefront management, from setup to
customization.
-**group-04-wix-stores-shipping-config/ **
(57 docs): Covers shipping, pricing, and
checkout configuration.
-**group-05-wix-payment-providers/ **
(289 docs): Covers payment processing and
provider integration, both first-party
and third-party gateways.
-**group-06-wix-subscription-management/ **
(133 docs): Covers subscription and pricing
plan management across Premium, Studio,
Editor X, Ascend, and App Market plans.-**group-07-wix-print-fulfillment/ **
(55 docs): Covers print and fulfillment-
related commerce features.
-**group-08-wix-commerce-fulfillment/ **
(74 docs): Covers order management,
fulfillment, shipping, and inventory.
-**group-09-wix-online-programs/ **
(63 docs): Covers Wix Online Programs for
building and delivering online courses.
-**group-10-refund-cancellation-policies/ **
(25 docs): Covers refund and cancellation
policies across Wix products.
-**group-11-wix-pricing-plans/ **
(113 docs): Covers Wix Pricing Plans and
monetization features lifecycle.
-**group-12-wix-partner-loyalty-programs/ **
(75 docs): Covers partner, loyalty, and
rewards ecosystems.
-**group-13-taxes-fees-config/ **
(53 docs): Covers tax configuration,
tips, and fees/surcharges.
-**group-14-wix-billing-documents/ **
(87 docs): Covers financial document
management: invoices, quotes, proposals,
receipts.
-**group-15-wix-payments-promotions/ **
(34 docs): Covers promotional tools:
discount coupons, gift cards, and saved
payment methods.
Mid-level INDEX.md .This file sits
one level below the root, inside
group-01-wix-payments-ecosystem/ , cov-
ering 87 documents across 10 leaf groups. The agent
reads this after deciding from the SKILL.md above that
payments are relevant.
---
name: wix-payments-ecosystem
description: >
Covers Wix’s payment ecosystem, spanning
payment processing, account management,
and financial operations across Wix
Payments and related commerce tools.
level: 2
num_documents: 87
---
## Overview
These sub-groups collectively cover Wix’s
payment ecosystem, spanning payment
processing, account management, and
financial operations across Wix Payments
and related commerce tools (Wix Stores,
Bookings, Restaurants, and Forms). Topics
range from setting up and verifying Wix
Payments accounts, managing payouts and
transactions, handling fraud, chargebacks,
and restricted products, to workflows like
collecting donations, processing manual
card entries, and understanding split
payment limitations.
13

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
## Contents
### Sub-groups (directories)
Read the INDEX.md in each sub-group to
understand what it covers.
-**group-00/ **(6 docs): Wix account
billing, payments management, and
role-based permissions.
-**group-01/ **(14 docs): Wix Payments
payment processing, payout management,
and account administration.
-**group-02/ **(15 docs): Setup, mgmt,
and compliance requirements of Wix
Payments accounts.
-**group-03/ **(7 docs): Accepting and
managing online donations through Wix.
-**group-04/ **(8 docs): Payment
flexibility limitations across Wix
products (split, defer, partial pay).
-**group-05/ **(5 docs): Fraud prevention
and payment security within Wix Payments.
-**group-06/ **(8 docs): Wix Payments
dashboard and transaction management.
-**group-07/ **(12 docs): Credit and debit
card payment processing within Wix.
-**group-08/ **(4 docs): Payment provider
compliance and restricted/prohibited
products.
-**group-09/ **(8 docs): Chargebacks and
fees within Wix Payments.
Leaf-level INDEX.md .This file sits at the bottom of the
hierarchy, inside group-05/ under the payments ecosys-
tem. It lists individual document IDs with titles. The
agent reads this to select which documents to retrieve via
getdocument.
---
name: L1-C298
description: >
Fraud prevention and payment security
within Wix Payments, addressing merchant-
facing and customer-facing concerns
related to credit card transactions.
level: 1
num_documents: 5
---
## Overview
These documents cover fraud prevention and
payment security within Wix Payments,
addressing both merchant-facing and
customer-facing concerns related to credit
card transactions. They answer questions
about how to identify and respond to
suspicious or declined payments, understand
liability for fraudulent chargebacks, and
interpret unusual bank statement activity.
Key terms: liability shift, chargeback
mechanisms, chip card readers, suspiciouspayment flags, fraud detection models.
## Contents
### Documents (5 items)
Use get_document tool with the doc_id
to read full content.
- ‘56ae7642b2a8c2ea‘: Best Practices for
Accepting Card Payments
- ‘448d7947c5a22489‘: Wix Payments:
About the ’Liability Shift’
- ‘bb73d9cb90385f12‘: Wix Payments:
Declined Payment Due to Suspected Fraud
- ‘c3b01aac19482333‘: Wix Payments:
About Suspicious Payments
- ‘d22b8cf6e599e0b8‘: Double Charge
Appearing on Your Bank Statement
C. Compilation Prompts
The compilation pipeline uses three LLM prompts, each
issued as a single-turn API call (no agent loop or tool use).
Document-level summarization.Used at the lowest level
to summarize a cluster of raw documents.
You are summarizing a cluster of related
documents from a knowledge base. Write a
2-3 sentence summary that captures:
1. The common TOPIC area these documents
cover
2. The types of QUESTIONS these documents
answer
3. Key TERMS or features mentioned across
documents
Be specific and concrete. Name actual
features, products, or processes.
Documents ({N} total, showing up to 15):
--- Document 1 ---
{doc_text[:600]}
--- Document 2 ---
{doc_text[:600]}
[...]
Summary:
Cluster-level summarization.Used at higher levels to
summarize a group of sub-cluster summaries into a broader
overview.
You are summarizing a level-{L} grouping
of {N} sub-groups from a knowledge base.
Each sub-group already has a summary below.
Write a 2-3 sentence overview that captures:
1. The broad DOMAIN these sub-groups cover
14

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
2. The range of TOPICS within this domain
3. What types of user QUESTIONS this group
can answer
Be specific -- name the main product areas,
features, or workflows.
Sub-group summaries:
- Sub-group 1: {summary[:300]}
- Sub-group 2: {summary[:300]}
[...]
Overview:
Labeling.Used to generate a filesystem-safe directory
name from a cluster summary. The response is post-
processed: lowercased, non-alphanumeric characters re-
placed with hyphens, truncated to 50 characters.
Generate a short (2-5 word) filesystem-safe
label for this cluster. Use lowercase,
hyphens instead of spaces. No quotes.
Summary: {summary[:500]}
D. Serving System Prompt
The serving agent receives the following system prompt:
# Corpus-Grounded Support Agent
# (Hierarchical Navigation)
You are a knowledge agent that answers
questions by navigating a hierarchical
skill directory. You explore a structured
file tree where documents are organized
into topic clusters.
## Hard Rules
- Every factual claim must trace to a
document you retrieved via get_document.
- SKILL.md files are NAVIGATION AIDS --
they tell you where to look, not what
to say.
- Never fabricate steps, URLs, prices, or
specifics not found in documents.
- Never guess. If you cannot find relevant
content after thorough exploration,
say so.
## Navigation Strategy
Your skills directory has this structure:
skill-XX-topic/
SKILL.md <- top-level summary
+ index of contents
group-YY-subtopic/
INDEX.md <- summary + index of
sub-contents
group-ZZ/INDEX.md <- summary + document
ID listings
Skill names and descriptions are already
available to you. Follow this workflow:
1. Read the SKILL.md of the 1-2 most
relevant skills for your query.
2. Drill into the most relevant sub-group:
Read its INDEX.md.
3. At the leaf level, INDEX.md lists
document IDs with brief titles. Pick
the most relevant document IDs.
4. Call get_document with each relevant
doc_id to retrieve the full text.
5. Read at least one full document before
answering.
## Tools
- Code execution: Use ‘ls‘ and ‘cat‘ to
navigate the skills hierarchy.
- get_document(doc_id): Retrieve the full
text of a document by its ID. The doc_id
values are listed in leaf-level INDEX.md
files.
## Answer Format
- First sentence = direct answer.
No preamble.
- Factual questions: 1-3 sentences
(˜80 words max).
- Procedural questions: numbered steps
only (˜150 words max).
- Plain text. No bold, headers, or
dividers.
- One approach only. Do not present
alternatives.
- Never add "contact support" or closing
remarks.
E. Full Navigation Traces
We present complete navigation traces with raw tool calls
for four queries from the evaluation set.
Trace 1: Direct descent (Factuality 1.0, CtxR 1.0).
Query:“I need to switch my business type from sole prop to
LLC in order to use an EIN. ”
[Context: agent sees 6 skill descriptions]
--- Turn 1 ---
Agent thought: The query is about changing
business type for Wix Payments. The skill
"wix-commerce-monetization" covers payments,
taxes, and business operations.
Tool: text_editor_code_execution
view skill-01-wix-commerce-monetization/
SKILL.md
15

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Result: 10 subgroups listed, including:
- group-01-wix-payments-ecosystem/
(87 docs): "payment processing,
account management, financial
operations"
--- Turn 2 ---
Tool: text_editor_code_execution
view group-01-wix-payments-ecosystem/
INDEX.md
Result: 8 leaf groups. Agent identifies
group-05 covering Wix Payments account
setup and verification.
Tool: text_editor_code_execution
view group-05/INDEX.md
Result: 12 documents listed, including:
- d56cc797742e: "Changing Your Wix
Payments Account Type"
--- Turn 3 ---
Tool: get_document
{"doc_id": "d56cc797742e7d0f"}
Result: Full article text (1,247 chars)
describing that account type cannot be
changed directly; must contact Wix
Customer Care.
--- Answer ---
"You cannot change your account type
directly within Wix Payments. To switch
from sole proprietorship to LLC, contact
Wix Customer Care with your EIN and LLC
formation documents ready."
Navigation: 6221 -> 1513 -> 87 -> 12 -> 1
Trace 2: Multi-document retrieval (Factuality 0.8, CtxR
1.0).Query:“How do I sync the hotel app with my calen-
dars iCal link to allow visitors to book available dates?”
[Context: agent sees 6 skill descriptions]
--- Turn 1 ---
Agent thought: Hotels and scheduling fall
under "wix-business-tools" which covers
scheduling, CRM, and business management.
Tool: text_editor_code_execution
view skill-00-wix-business-tools/SKILL.md
Result: 8 subgroups listed, including:
- group-02-wix-hotels-platform/
(82 docs): "hotel management, bookings,
reservations, calendar sync"
--- Turn 2 ---
Tool: text_editor_code_execution
view group-02-wix-hotels-platform/INDEX.md
Result: 7 leaf groups. Agent identifiesgroup-05 covering external calendar
integration and iCal sync.
Tool: text_editor_code_execution
view group-05/INDEX.md
Result: 17 documents listed, including:
- 5f02fc2f8937: "Importing Reservations
from External Calendar (iCal)"
- 741830b917b4: "Connecting Airbnb iCal
to Your Site"
--- Turn 3 ---
Tool: get_document
{"doc_id": "5f02fc2f8937d16a"}
Result: Step-by-step iCal import guide
for original Wix Hotels.
Tool: get_document
{"doc_id": "741830b917b47d07"}
Result: Airbnb iCal connection guide for
Wix Hotels by HotelRunner.
--- Answer ---
"For original Wix Hotels: Go to Hotels
tab > Reservations > iCal, select room
type, paste iCal URL, click Save. For
HotelRunner: go to dashboard, navigate
to channel connections, add Airbnb iCal
link for each property..."
Navigation: 6221 -> 931 -> 82 -> 17 -> 2
Trace 3: Mobile quick action (Factuality 1.0, CtxR 1.0).
Query:“I need to temporarily remove the quick action
button from the mobile view. ”
--- Turn 1 ---
Agent thought: Mobile site editing falls
under "wix-platform-guide" which covers
Editor, site building, and customization.
Tool: text_editor_code_execution
view skill-03-wix-platform-guide/SKILL.md
Result: 10 subgroups. Agent picks
"wix-mobile-site-building" covering
mobile editor features.
--- Turn 2 ---
Tool: text_editor_code_execution
view wix-mobile-site-building/INDEX.md
Result: Leaf groups covering mobile UI
elements. Agent finds group with docs
on Quick Action Bar.
Tool: text_editor_code_execution
view group-03/INDEX.md
Result: 8 documents including:
- bce5e3d829fe: "Adding and Setting Up
16

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Your Quick Action Bar"
--- Turn 3 ---
Tool: get_document
{"doc_id": "bce5e3d829fe6eeb"}
Tool: get_document
{"doc_id": "186ad4a2da9f1f30"}
--- Answer ---
"There is no hide option, only deletion.
Right-click the Quick Action Bar in the
mobile Editor and select Delete. To re-add
later, go to Mobile Tools and click Add."
Navigation: 6221 -> 1161 -> 142 -> 8 -> 2
Trace 4: Cross-branch exploration (Factuality 1.0, CtxR
0.6).Query:“I want to change the currency for my
course. ”
[Context: agent sees 6 skill descriptions]
--- Turn 1 ---
Agent thought: "Course" suggests Online
Programs, and "currency" is a payments/
billing topic. The skill "wix-commerce-
monetization" covers both online programs
and payment configuration.
Tool: text_editor_code_execution
view skill-01-wix-commerce-monetization/
SKILL.md
Result: 15 subgroups listed, including:
- group-09-wix-online-programs/
(107 docs): "online courses, program
management, pricing, delivery"
- group-14-wix-billing-documents/
(68 docs): "currency settings, invoice
automation, financial documents"
--- Turn 2 ---
Agent thought: Start with Online Programs
since the query is about course currency.
Tool: text_editor_code_execution
view group-09-wix-online-programs/INDEX.md
Result: 5 leaf groups. Picks group-03
covering program settings (18 docs).
Tool: text_editor_code_execution
view group-03/INDEX.md
Result: 18 documents including:
- 7a9dc6f2de90: "Online Programs:
Editing Your Program Settings"
--- Turn 3 ---
Tool: get_document
{"doc_id": "7a9dc6f2de90db61"}
Result: Program settings article. Mentionsthat currency is configured at the site
level via Language & Region settings,
not per-program.
Agent thought: Need the actual currency
configuration steps. Navigate to billing.
Tool: text_editor_code_execution
view group-14-wix-billing-documents/
INDEX.md
Result: 5 leaf groups. Picks group-00
covering currency management (18 docs).
Tool: text_editor_code_execution
view group-00/INDEX.md
Result: 18 documents including:
- 29139bd7c379: "Setting Your Currency
for Accepting Payments"
Tool: get_document
{"doc_id": "29139bd7c379ff77"}
Result: Step-by-step currency configuration
guide via Settings > Language & Region.
--- Answer ---
"To change the currency for your course,
update the site-level currency setting:
1. Go to Settings in your dashboard.
2. Click Language & Region.
3. Scroll to Currency and select your
desired currency.
4. Click Save.
This applies to all Wix products including
Online Programs. Your currency must match
your payment provider’s currency."
Navigation paths:
6221 -> 1513 -> 107 -> 18 -> 1 (programs)
6221 -> 1513 -> 68 -> 18 -> 1 (billing)
This trace demonstrates cross-branch navigation: the agent
first explored the online-programs subgroup to understand
the relationship between courses and currency, then nav-
igated to the billing-documents subgroup to find the au-
thoritative currency configuration steps. The two branches
within the same skill provided complementary information
that neither alone would have answered the query fully.
F. Positioning Comparison
Table 6 highlights three axes on which CORPUS2SKILL
differs from prior work.
Input modality.Scatter/Gather, RAPTOR, GraphRAG,
and CORPUS2SKILLall start from documents or text
chunks, but V oyager and SkillX build skills from agent
trajectories—successful action sequences or distilled expe-
rience. CORPUS2SKILLis the only system that converts
17

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Table 6.Positioning of CORPUS2SKILLrelative to prior work.
Aspect Scatter/Gather RAPTOR GraphRAG Voyager/SkillX CORPUS2SKILL
Input Documents Text chunks Documents Trajectories Documents
Organization Flat clusters Hier. tree Knowledge graph Skill library Hier. skill tree
Navigation Human user Algorithm Graph search Vector sim. LLM agent
Serve infra Cluster index Vector index Graph DB Vector DBLLM only
a static document corpus into the skill format typically re-
served for procedural knowledge.
Navigation mechanism.Scatter/Gather relies on human
users to iteratively select clusters; RAPTOR and GraphRAG
delegate navigation to fixed algorithms (vector similarity or
graph traversal). CORPUS2SKILLis unique in delegating
navigation to an LLM agent that can reason about which
branch to explore, backtrack when a path is unproductive,
and combine evidence across branches. This agent-driven
navigation is what enables the trial-and-error behavior illus-
trated in Figures 4 and 5.
Serve-time infrastructure.All prior systems require ded-
icated infrastructure at query time: a cluster index, a vector
database, or a graph database. CORPUS2SKILLeliminates
this dependency entirely—the only runtime component is
the LLM itself, which receives pre-compiled skill files via
the API. This “LLM only” property simplifies deployment
but shifts the cost burden to input tokens, as discussed in
Section 5.
G. Failure Analysis
We examine the 62 queries (31%) on which CORPUS2SKILL
scores Factuality ≤0.4 or Context Recall ≤0.2. By compar-
ing each query’s retrieved documents against the gold article
IDs (using prefix matching, since CORPUS2SKILLuses trun-
cated content hashes), we identify three failure modes:
Navigation miss (38 queries).The agent retrieves doc-
uments but none match the gold article, indicating it de-
scended into the wrong branch of the hierarchy. For ex-
ample, a query about reordering CMS collection items is
routed to a general site-editing skill instead of the CMS-
specific one; a query about blocking time slots in a staff
schedule navigates to a generic booking group rather than
the staff-management leaf. These misroutes occur at thetop
levelof the hierarchy, where broad skill summaries fail to
distinguish closely related topics.
Partial navigation (19 queries).The agent’s retrieved
set includes the gold document, yet scores remain low. In
these cases the gold article is retrieved alongside several
non-relevant documents, diluting the context provided tothe answer synthesis step. The agent correctly identified the
general topic area but over-retrieved from neighboring leaf
groups.
Synthesis error (3 queries).The agent retrieves contextu-
ally relevant documents (Context Recall ≥0.6) but produces
a factually incorrect answer, typically by over-generalizing
across multiple articles or misinterpreting conditional in-
structions as universal.
No documents retrieved (2 queries).Two queries (“How
do I clear my browser cache?” and a Gmail password reset
question) resulted in zero retrieved documents. Both have
gold articles in the corpus (Wix publishes help articles for
these topics), but the agent could not locate a matching
skill branch and correctly declined to answer rather than
hallucinating.
Takeaway.The dominant bottleneck is initial topic rout-
ing: 38 of 62 failures stem from the agent choosing the
wrong top-level branch. The compact-mode ablation (Sec-
tion 4.4), which produces narrower top-level clusters, par-
tially mitigates this by providing finer-grained topic separa-
tion at the first navigation step.
H. Baseline Implementation Details
All baselines use the same LLM (Claude Sonnet) for an-
swer generation and the same evaluation prompts for fair
comparison.
BM25.We implement BM25 sparse keyword retrieval us-
ing therankbm25 library (Robertson & Zaragoza, 2009).
Documents are tokenized and indexed at the full-article
level. At query time, we retrieve the top-5 articles by BM25
score and pass them as context to a single LLM call.
Dense.We encode all documents using Qwen3-
Embedding-0.6B (Zhang et al., 2025), producing
1024-dimensional embeddings stored in a FAISS flat index
(Johnson et al., 2019). We use the document prompt
template for encoding articles and the query prompt
template for encoding questions. At query time, we retrieve
the top-5 nearest neighbors by cosine similarity and pass
them as context to a single LLM call.
18

Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills
Hybrid.We combine BM25 and Dense scores using Re-
ciprocal Rank Fusion (Cormack et al., 2009) with k=60
and equal weighting ( α=0.5 ). The fused ranking selects the
top-5 passages, which are passed to a single LLM call.
RAPTOR.We faithfully implement the RAPTOR algo-
rithm (Sarthi et al., 2024): UMAP (McInnes et al., 2018) for
dimensionality reduction, Gaussian Mixture Model (GMM)
soft clustering (McLachlan & Peel, 2000) with BIC-based
cluster count selection (Schwarz, 1978), and recursive sum-
marization of clusters. We use the same embedding model
(Qwen3-Embedding-0.6B) as the Dense baseline for con-
sistency. At query time, collapsed tree retrieval searches
across all hierarchy levels simultaneously, returning the top-
5 passages from any level.
Agentic.An LLM agent is given access to three
search tools ( searchbm25 ,searchdense ,
searchhybrid ), each returning top- kresults. The
system prompt instructs the agent to search first, then
answer based on accumulated evidence, choosing BM25
for exact terms and product names, dense for conceptual
queries, and hybrid as a default strategy. The agent can
issue up to 10 rounds of iterative retrieval, reformulating
sub-queries based on intermediate results. Evidence is
accumulated across all turns; when the agent stops, it
produces a structured JSON response containing the answer
and curated top-5 references selected from the full evidence
pool. For evaluation fairness, context recall is computed
over the last 5 retrieved documents (matching the context
window of single-shot baselines). This baseline tests
whether agentic multi-turn retrieval over traditional search
infrastructure can match navigation-based exploration.
I. Evaluation Metrics
Lexical metrics.Token F1tokenizes both the predicted
and gold answers into word tokens and computes the har-
monic mean of token-level precision and recall.BLEU
(Papineni et al., 2002) measures modified n-gram precision
(up to 4-grams) with a brevity penalty to penalize overly
short outputs.ROUGE-1andROUGE-2(Lin, 2004) mea-
sure unigram and bigram recall respectively, capturing how
much of the gold answer’s content appears in the prediction.
LLM-judged metrics.BothFactualityandContext Re-
calluse an LLM judge (Zheng et al., 2023) following the
RAGAS framework (Es et al., 2024), scoring on a 1–5 scale
normalized to 0–1. Factuality evaluates whether the claims
in the generated answer are factually consistent with the
gold answer, independent of the retrieved context. Context
Recall evaluates whether the retrieved context contains suf-
ficient information to answer the question, by checking how
many essential claims in the gold answer can be attributedto the provided context.
Context Recall standardization.A critical design deci-
sion is the standardization of Context Recall measurement
across different retrieval paradigms (single-shot, multi-turn,
and navigation-based). For fair comparison, all methods
follow the same protocol: (1) collect the last 5 documents
or passages retrieved by the method; (2) join them with
separator tokens; (3) truncate to a uniform 8,000-character
window. The LLM judge receives this standardized context.
This ensures that single-shot baselines contribute their top-5
retrieved passages, Agentic contributes the last 5 documents
from its multi-turn retrieval, and CORPUS2SKILLcon-
tributes the last 5 documents retrieved via getdocument
tool calls.
Cost metrics.Input token count is the total number of
tokens sent to the LLM across all turns. Per-query cost is
computed using the model’s published per-token pricing (in-
put and output tokens combined). For single-shot methods,
cost reflects one LLM call; for multi-turn methods (Agen-
tic, CORPUS2SKILL), it reflects all turns including tool-call
reasoning.
19
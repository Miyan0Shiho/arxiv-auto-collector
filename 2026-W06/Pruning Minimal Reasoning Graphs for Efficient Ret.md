# Pruning Minimal Reasoning Graphs for Efficient Retrieval-Augmented Generation

**Authors**: Ning Wang, Kuanyan Zhu, Daniel Yuehwoon Yee, Yitang Gao, Shiying Huang, Zirun Xu, Sainyam Galhotra

**Published**: 2026-02-04 08:48:11

**PDF URL**: [https://arxiv.org/pdf/2602.04926v1](https://arxiv.org/pdf/2602.04926v1)

## Abstract
Retrieval-augmented generation (RAG) is now standard for knowledge-intensive LLM tasks, but most systems still treat every query as fresh, repeatedly re-retrieving long passages and re-reasoning from scratch, inflating tokens, latency, and cost. We present AutoPrunedRetriever, a graph-style RAG system that persists the minimal reasoning subgraph built for earlier questions and incrementally extends it for later ones. AutoPrunedRetriever stores entities and relations in a compact, ID-indexed codebook and represents questions, facts, and answers as edge sequences, enabling retrieval and prompting over symbolic structure instead of raw text. To keep the graph compact, we apply a two-layer consolidation policy (fast ANN/KNN alias detection plus selective $k$-means once a memory threshold is reached) and prune low-value structure, while prompts retain only overlap representatives and genuinely new evidence. We instantiate two front ends: AutoPrunedRetriever-REBEL, which uses REBEL as a triplet parser, and AutoPrunedRetriever-llm, which swaps in an LLM extractor. On GraphRAG-Benchmark (Medical and Novel), both variants achieve state-of-the-art complex reasoning accuracy, improving over HippoRAG2 by roughly 9--11 points, and remain competitive on contextual summarize and generation. On our harder STEM and TV benchmarks, AutoPrunedRetriever again ranks first, while using up to two orders of magnitude fewer tokens than graph-heavy baselines, making it a practical substrate for long-running sessions, evolving corpora, and multi-agent pipelines.

## Full Text


<!-- PDF content starts -->

Pruning Minimal Reasoning Graphs for Efficient Retrieval-Augmented
Generation
Ning Wang*1Kuanyan Zhu*2Daniel Yuehwoon Yee*3
Yitang Gao4Shiying Huang1Zirun Xu5Sainyam Galhotra†1
1Cornell University2University of Cambridge3The University of Hong Kong4HKUST
5University of British Columbia
nw366@cornell.edu kz345@cam.ac.uk u3636035@connect.hku.hk sg@cs.cornell.edu
Abstract
Retrieval-augmented generation (RAG) is
now standard for knowledge-intensive LLM
tasks, but most systems still treat every query
as fresh, repeatedly re-retrieving long pas-
sages and re-reasoning from scratch, inflating
tokens, latency, and cost. We presentAuto-
PrunedRetriever, a graph-style RAG system
thatpersiststhe minimal reasoning subgraph
built for earlier questions andincrementally
extends it for later ones. AutoPrunedRetriever
stores entities and relations in a compact, ID-
indexed codebook and represents questions,
facts, and answers as edge sequences, enabling
retrieval and prompting over symbolic struc-
ture instead of raw text. To keep the graph
compact, we apply a two-layer consolidation
policy (fast ANN/KNN alias detection plus
selectivek-means once a memory threshold is
reached) and prune low-value structure, while
prompts retain only overlap representatives
and genuinely new evidence. We instantiate
two front ends: AUTOPRUNEDRETRIEVER-
REBEL, which uses REBEL (Huguet Cabot
and Navigli, 2021) as a triplet parser, and
AUTOPRUNEDRETRIEVER-LLM, which
swaps in an LLM extractor. On GraphRAG-
Benchmark (Medical and Novel), both
variants achievestate-of-the-art complex
reasoning accuracy, improving over Hip-
poRAG2 (Jim ´enez Guti ´errez et al., 2025) by
roughly 9–11 points, and remain competitive
on contextual summarize and generation.
On our harder STEM and TV benchmarks,
AutoPrunedRetriever again ranks first, while
using up to two orders of magnitude fewer
tokens than graph-heavy baselines, making it
a practical substrate for long-running sessions,
evolving corpora, and multi-agent pipelines.
1 Introduction
Retrieval-augmented generation (RAG) grounds
LLMs in external knowledge, reducing halluci-
*Equal contribution.
†Corresponding author.nations, enabling citation, and allowing updates
without full retraining. Recent advances in dense
retrieval and generation have yielded strong per-
formance on open-domain question answering and
tool-augmented assistants (Lewis et al., 2020;
Karpukhin et al., 2020a; Izacard et al., 2022; Ram
et al., 2023). However, moving from retriev-
ing relevant text to solving complex, knowledge-
intensive tasks still requires multi-hop reasoning:
composing evidence across documents, enforcing
temporal or structural constraints, and maintaining
consistency across repeated or related queries.
In practice, most RAG systems treat each query
independently. Even when multiple questions are
closely related—or arise sequentially in agentic
workflows, systems repeatedly re-retrieve overlap-
ping passages and re-reason from scratch. This
leads to substantial redundancy in retrieved con-
text, inflated token usage, higher latency, and in-
creased cost. These inefficiencies are especially
pronounced in long-running sessions and multi-
agent settings (e.g., planner–researcher–verifier
pipelines), where similar reasoning chains are re-
visited many times (Yao et al., 2023; Wu et al.,
2024)..
Graph-based RAG methods address some of
these issues by lifting retrieval from flat text
passages to structured representations over enti-
ties and relations (Han et al., 2024; Sun et al.,
2022; Baek et al., 2023; Wang et al., 2023).
By explicitly modeling compositional structure,
GraphRAG systems improve multi-hop reasoning
and disambiguation. Yet existing approaches still
face three fundamental bottlenecks when deployed
over evolving corpora and long reasoning chains.
We demonstrate these challenges with an example.
Example 1.Consider a small corpus containing
five documents describing (i) a corporate acquisi-
tion, (ii) regulatory status under the EU Digital
Services Act, (iii) GDPR incident histories, (iv)arXiv:2602.04926v1  [cs.DB]  4 Feb 2026

2024 vendor contracts, and (v) subsidiary rela-
tionships (Fig. 1). From this corpus, we ask three
related questions:
Q1:Which subsidiaries acquired after Jan. 1,
2021 are subject to the EU Digital Services Act?
Q2:For those subsidiaries, did GDPR incident
rates decrease post-acquisition?
Q3:Which 2024 vendor contracts involve those
same subsidiaries? A standard GraphRAG
pipeline constructs an entity–relation graph over
the corpus and answers each query via neighbor-
hood expansion. Even in this minimal setting,
three limitations emerge.(M1) Graph construc-
tion and maintenance:entity aliases and nam-
ing variants require global checks and relinking
as new evidence arrives.(M2) Reasoning gran-
ularity:neighborhood-based expansion retrieves
broad subgraphs around central entities (e.g., the
parent company or regulator), rather than the few
edges that realize each reasoning chain.(M3) Re-
dundant retrieval:when Q1–Q3 are issued se-
quentially or by multiple agents, largely overlap-
ping subgraphs are repeatedly retrieved and seri-
alized, compounding token and latency costs.
Notably, the answers to Q2 and Q3 reuse much
of the reasoning structure required for Q1. How-
ever, existing systems fail to exploit this overlap,
repeatedly reconstructing context instead of per-
sisting and extending prior reasoning.
These observations suggest a shift in perspec-
tive: retrieval should not aim to recover all poten-
tially relevant context, but instead identify, cache,
and reuse the minimal reasoning structure needed
to answer a query, and incrementally extend that
structure as new questions arrive. These limita-
tions motivate three design principles: local incre-
mental structure, path-centric retrieval, and exact
symbolic reuse, which guide the design of Auto-
PrunedRetriever.
Our Approach.We introduce AutoPrune-
dRetriever, a structure-first RAG system that treats
reasoning paths, rather than passages or neighbor-
hoods, as the primary retrieval unit. AutoPrune-
dRetriever converts text into symbolic triples and
represents questions, facts, and answers as com-
pact sequences of entity–relation edges. The sys-
tem persists only the minimal subgraphs that sup-
port successful reasoning and reuses them across
later queries, avoiding repeated re-retrieval and re-
prompting.
To keep memory compact and stable over time,AutoPrunedRetriever applies a two-layer consol-
idation policy: a lightweight, continuous alias-
detection pass using approximate nearest neigh-
bors, and a periodic, budget-triggered consolida-
tion step that merges aliases and prunes low-value
structure. Retrieval is explicitly path-centric, scor-
ing candidate reasoning chains rather than ex-
panding broad neighborhoods. Prompts are con-
structed from a compact symbolic codebook that
includes only novel or non-redundant evidence,
substantially reducing token usage while preserv-
ing grounding in source text.
We instantiate AutoPrunedRetriever with
two front ends: AutoPrunedRetriever-
REBEL, which uses a fixed triplet parser,
and AutoPrunedRetriever-LLM, which replaces
it with an LLM-based extractor. Across the
GraphRAG benchmark as well as harder STEM
and TV reasoning datasets, both variants achieve
state-of-the-art complex reasoning accuracy
while using up to two orders of magnitude fewer
tokens than graph-heavy baselines. These results
demonstrate that pruned, persistent reasoning
structure, not larger graphs or longer prompts, is
the key substrate for efficient, long-running, and
agentic RAG systems.
1.1 Design Principles
The design of AutoPrunedRetriever is guided by
three principles, each directly addressing one of
the limitations illustrated in Example 1.
P1. Local, incremental structure (addresses
M1). To avoid the cost and brittleness of
global graph maintenance, AutoPrunedRetriever
builds reasoning structure locally and incremen-
tally. Text is encoded into symbolic triples and
grouped into small, coherent graphs that can be ex-
tended over time. Entity consolidation is applied
selectively at the symbol level, allowing aliases to
be merged without re-extracting text or relinking
global structure.
P2. Path-centric retrieval (addresses M2). Rea-
soning is realized by short chains of entities and
relations, not broad neighborhoods. AutoPrune-
dRetriever therefore treats edge sequences as the
primary retrieval unit and scores candidate reason-
ing paths directly, avoiding the retrieval of large
subgraphs that do not contribute to the required in-
ference.
P3. Exact symbolic reuse (addresses M3). To
prevent repeated serialization of overlapping con-
text, reuse across queries is exact and symbolic

Corpus Triples Graphs
Meta 
Codebook
...
Retrieved 
Triples
Pruned
Retrieved 
TriplesQueryAI
Language 
Model
Answer
Figure 1:AutoPrunedRetrieverpipeline: (1) en-
code into symbols and edges, (2) build chunked small
graphs, (3) coarse→fine retrieval, (4) selector + com-
pact prompt packing, (5) entity-only consolidation with
a DPO wrapper to trade accuracy vs. tokens.
rather than textual. AutoPrunedRetriever caches
reasoning subgraphs as compact sequences of en-
tity–relation identifiers and constructs prompts
that include only novel or non-redundant evidence,
ensuring that token usage scales with new reason-
ing rather than repeated context.
We discuss more details about these principles
in Section 2.
2 Method
2.1 Overview
Free text is a noisy, length-biased substrate for
reasoning: it repeats the same facts in many sur-
face forms, conflates content with string realiza-
tion, and penalizes reuse by charging tokens re-
peatedly for identical information. These proper-
ties make it ill-suited for persistent, multi-hop rea-
soning across related queries.
AutoPrunedRetriever addresses these issues
with a symbol-first pipeline that implements the
three design principles introduced in Section 1.1.
Specifically, we: (1) encode questions, answers,
and facts into a shared symbolic codebook of
entities and relations (Sec. 2.2); (2) build local,
coherent reasoning graphs that serve as retrieval
atoms (Sec. 2.3); (3) retrieve paths rather than
neighborhoods via coarse-to-fine scoring in sym-
bol space (Sec. 2.4); (4) select and package only
non-redundant structure into compact prompts
(Secs. 2.5, 2.6); and (5) consolidate entities incre-
mentally to keep the persistent graph compact over
time (Sec. 2.7). Finally, we adapt compression
behavior to accuracy–efficiency tradeoffs using a
lightweight DPO-trained policy (Sec. 2.8).
Each component is designed to ensure that re-
trieval, reasoning, and prompting scale with new
reasoning rather than repeated context.2.2 Step 1: Symbolic Encoding
Intuition.We normalize free-form text into a
small set of symbols and the edges they instanti-
ate. This exposes shared structure across questions
and facts and makes reuse exact via IDs rather
than brittle string matches. Because natural lan-
guage is heavy-tailed, most informational mass is
carried by a small “core” vocabulary; encoding
those into a codebook yields large compression
gains (see Lemma 1). In App. A.2 (Lemma 14,
Proposition 16) also shows that such E–R–E code-
books mitigate the “token dilution” effect of long
sequence embeddings.
Formulation.For a text spany(question, an-
swer, fact), a parser (REBEL or an LLM) produces
triples
τ(y)⊆ U E× UR× UE.
We maintain a meta-codebook
C= (E, R,M,Q,A,F, E emb, Remb),
whereE, Rare entity and relation dictionaries,
M⊆E×R×Eis the sparse set of unique edges,
andQ,A,Fstore sequences of edge IDs for ques-
tions, answers, and facts.
An INDEXIFYmap
y=INDEXIFY 
τ(y);E, R,M
∈M⋆
serializes text into edge indices, extending
(E, R,M)only when new symbols/edges appear.
We appendyto the appropriate store inQ,A,F.
The embedding tablesE emb :E→Rdand
Remb:R→Rdgive us E–R–E embeddings for
each edge(e, r, e′)∈M.
2.3 Step 2: Chunked Small Graphs
(Local-First Construction)
Intuition.Instead of inserting every triple into
a global graph, we buildsmall, locally coherent
graphsin a single pass. The question is no longer
“where in the global graph does this triple live?”
but “does this triplefitthe current small graph?”
This keeps working sets small and updates lo-
cal, while preserving global consistency via shared
IDs inM.
Runs and modalities.For each modalityy∈
{Q,A,F}we build a run repository
Runs(y)∈ 
M⋆⋆,
a list of edge-ID sequences (runs) that correspond
to small graphs.

Streaming construction.We maintain a current
small graphG k= (V k, Ek)with an embedding
centroid. For each incoming triple, we compute a
fit scorethat combines:
1.Semantic cohesion: cosine similarity be-
tween the triple embedding and the centroid ofG k.
2.Structural continuity: a bonus when the
triple continues a path (e.g., tail prev=head new) or
reuses nodes inV k.
If the (bonus-adjusted) score exceeds a thresh-
oldτ, we append toG k; otherwise we closeG k,
linearize its edges intog k∈M⋆, storeg k∈
Runs(y), and startG k+1. This yields maximal lo-
cally coherent segments (Lemma 2).
Boundary refinement.A single pass can over-
cut near boundaries, so we run a localmerge-if-
not-a-true-cuttest on adjacent runs: we re-embed
the concatenation of the boundary region, re-run
the same segmenter, and merge if the new segmen-
tation doesnotplace a cut at the original bound-
ary. Surviving boundaries are self-consistent fixed
points of the segmenter (Lemma 3).
The result is a sequence of compact, coherent
runs that serve as retrieval atoms. We quantify
their intra-run cohesion and the induced working-
set reduction for retrieval in Lemmas 4 and Lem-
mas 5.
2.4 Step 3: Coarse to Fine Path Retrieval
Intuition.Naively embedding every query
against every run is slow and favors long, noisy
chunks. We insteadfirstwork in symbol space to
get a small shortlist andthendo a more detailed
triple-level check on that shortlist. This two-layer
scheme keeps cost nearO(k)rather thanO(n)
while preserving precision (Lemma 7).
Coarse stage (symbol-space recall).Each run
decodes to triples(h, ρ, t). We pool entity embed-
dings intoE(·)and relation embeddings intoR(·).
For a queryqand candidate runfwe compute a
simplemax-pairscore over entities and relations,
scoarse(q, f) =w entmax
i,jcos 
E(q) i, E(f) j
+w relmax
p,rcos 
R(q) p, R(f) r
,
and keep the top-kruns as a high-recall shortlist
Ik. This stage only touches small entity/relation
sets and is therefore very fast.Fine stage (triple-aware re-ranking).For each
candidate inI k, we linearize triples (h ρ t) into
short lines, embed them, and build a similarity ma-
trixSbetween query and candidate lines. The final
score is a weighted sum of five simple terms com-
puted onS: a top-tmean over the best entries (re-
lational strength), a coverage term counting how
many query triples are well matched, a soft many-
to-many overlap term, a greedy 1:1 match encour-
aging parsimonious alignment, and a small whole-
chunk bonus gated by full-text similarity to avoid
“longer is better”. We then take the globalTopM
runs across all queries and channels (answers /
facts / prior questions). Exact formulas and hy-
perparameters are deferred to App. A.3.
2.5 Step 4: Knowledge Selection
Intuition.Real corpora have heavy-tailed reuse
of motifs, so naive “pull everything similar” either
misses paraphrased support or floods the prompt
with near-duplicates. We therefore operate on runs
and, for each channel (answers, facts, related ques-
tions), choose an action
s∈ {include all,unique,not include}.
Hereinclude allkeeps all surviving runs (e.g.,
for safety/audit regimes),uniquekeeps one repre-
sentative per semantic cluster to avoid echoing and
token bloat, andnot includedrops the channel
when it adds little beyond context. Semantic clus-
ters are defined in run-embedding space; we pick
a consensus representative per cluster, and show
in Lemma 8 that this representative stays close to
paraphrastic variants.
2.6 Step 5: Compact prompt Construction
Given the selected runs, we assemble the minimal
symbolic state the LLM needs:
U=q∪a∪f,
E′={h:t: (h, ρ, t)∈U},
R′={ρ: (h, ρ, t)∈U}.
We maintain two equivalent encodings and
choose the cheaper at query time: (1)word
triples, which list(h, ρ, t)directly for low-
redundancy regimes; and (2)compact indices,
which mapE′, R′to short IDs and represent
query/answer/fact sequences as ID triples. The
prompt payload isΠ = (E′, R′,q,a,f,rules),
with a brief textual header explaining the ID for-
mat. Token cost scales with|E′|+|R′|+|q|+|a|+
|f|, typically far below concatenated passages.

2.7 Step 6: Entity-Only Consolidation
Intuition.Long-running deployments accumu-
late aliases, misspellings, and near-duplicates
(IBMvs.International Business Machines). We
consolidateentities only, then remap all edges and
sequences.
•Layer 1 (ANN+KNN).Continuously build
an ANN-backedk-NN graph over entity embed-
dings; connect pairs with cosine above a con-
servative thresholdτ Eand form provisional alias
groups.
•Layer 2 (on-demandk-means).When
memory or|E|exceeds a budget, refine groups
withk-means and choose medoid representatives
mE(·), which minimize within-cluster distortion
(Lemma 11). Each edge(u, r, v)is remapped to
(mE(u), r, m E(v)), and duplicates are removed.
Because questions/answers/facts are stored as
edge-ID sequences, this remap automatically
cleans them as well. This quotienting can only re-
duce edge and sequence cardinalities (Lemma 9)
and does not increase the number of sentence-level
text encodings (Lemma 10).
2.8 Step 7: Adaptive Compression via DPO
The “right” selector choice depends on the query
(ambiguity, hops), model (context length, robust-
ness), domain (redundancy), and user goals (accu-
racy vs. latency/tokens). We learn a small categor-
ical policyπ θover the selector actions per chan-
nel, conditioned on features such as query length,
ambiguity scores, model ID, and token budget.
Offline, for each query we evaluate several se-
lector configurationsyand compute a utility
U(x, y) =α·Acc+δ·Faithfulness−
β·Tokens−γ·Latency.
We form preference pairs(x, y+, y−)whenever
U(x, y+)> U(x, y−), and trainπ θ(y|x)with
DPO against a fixed reference policy. Under a
Bradley–Terry preference model, DPO aligns pol-
icy log-odds with utility differences up to a scal-
ing and reference correction (Lemma 12), and a
simple action lattice yields monotone token con-
trol under a budget constraint (Lemma 13).
At inference time, the policy picks (include
all,unique, ornot include) per channel, steer-
ing AutoPrunedRetriever to the appropriate oper-
ating point—e.g., overlap-heavy scaffolds for am-
biguous multi-hop questions vs. aggressive dedu-plication under tight budgets—while retaining the
same symbolic infrastructure.
3 Experiments
We study:
•RQ1(§3.2): complex reasoning performance
on Medical, Novel, STEM, TV .
•RQ2(§3.3): efficiency (tokens, latency,
workspace) on STEM/TV .
•RQ3(§3.4): overall performance on the full
GraphRAG benchmark.
3.1 Experimental Settings
Devices and models.GraphRAG experiments
(§3.4) run on an Intel i9-13900KF, 64 GB RAM,
RTX 4090 (24 GB); STEM/TV experiments (§3.2,
§3.3) use an A100 (40 GB). All LLM-backed pars-
ing/judging uses thegpt-4o-miniAPI.
Datasets. MedicalandNovelare from the
GraphRAG-Benchmark (Xiang et al., 2025) with
Fact Retrieval, Complex Reasoning, Contextual
Summarize, and Creative Generation. We ad-
ditionally constructTVandSTEMfrom the
HotpotQA Wikipedia pipeline (HotpotQA Team,
2019), grouped into 47 TV and 32 STEM micro-
corpora; TV questions focus on character/episode
relations, STEM on cross-sentence scientific infer-
ence.
Evaluation and parsers.We follow the LLM-
judge protocol of Xiang et al. (Xiang et al.,
2025). AUTOPRUNEDRETRIEVERis evaluated
with two front ends: a REBEL-based triplet
parser (Huguet Cabot and Navigli, 2021) and
an LLM-based parser (gpt-4o-mini); both use
the same pruning and indices-only prompting
pipeline.
3.2 Full Complex-Reasoning Evaluation
We aggregate all complex-reasoning sources:
Medical-CR,Novel-CR,STEM,TV, to test a
single pruned, symbolic pipeline across technical,
narrative, and pop-culture reasoning.
Quantitative results.Across all four sets, AU-
TOPRUNEDRETRIEVERis consistently strongest
(Fig. 2). OnMedical-CRandNovel-CR, the
REBEL variant reaches72.49%and63.02%
ACC vs. HIPPORAG2 (61.98%, 53.38%), i.e.,
+10.51/+9.64 points. The LLM-parser vari-
ant is close (71.59%, 62.80%), indicating that
the gain mainly comes from symbolic pruning

and retrieval. OnSTEM, REBEL/LLM obtain
81.4%/78.1% vs. HIPPORAG2 (69.9%); onTV,
68.2%/65.2% vs. HIPPORAG2 (59.5%). The or-
dering is the same on all four: APR-REBEL>
APR-llm>HippoRAG2>LightRAG.
Medical Novel STEM TV0.00.20.40.60.8Answer Correctness0.725
0.6300.814
0.6820.716
0.6280.781
0.650
0.620
0.5340.699
0.5950.613
0.491
0.1710.465
Medical Novel STEM TV0.00.10.20.30.4ROUGE-L0.3080.3120.308
0.1840.3110.354
0.304
0.1910.370
0.334
0.317
0.1950.250
0.242
0.0000.117AutoPrunedRetriever-REBEL AutoPrunedRetriever-llm HippoRAG2 LightRAG
Figure 2: Average answer correctness on all complex-
reasoning sets (Medical-CR, Novel-CR, STEM, TV)
for HippoRAG2, LightRAG, AutoPrunedRetriever-
REBEL, AutoPrunedRetriever-llm.
Case study (STEM): reasoning over ecologi-
cal chains.Consider the STEM question“How
does the variability in the size of brown bears
across different regions serve as evidence for un-
derstanding their adaptability in various environ-
ments?”In our run this was a question where AU-
TOPRUNEDRETRIEVERbeat both HIPPORAG2
and LIGHTRAG. What our retriever actually sur-
faced was acompact symbolic subgraphcentered
on three functional edges: (1) region→resource
availability, (2) resources→body size, and (3)
body size→environmental adaptability. Because
those three edges were present together, the LLM
could reconstruct the full causal chain:
region⇒food⇒size⇒adaptability.
The generated answer therefore explained that
large coastal/Kodiak bears reflect high salmon
(high calories), whereas smaller inland bears re-
flect limited resources, and that thisvariation
itselfis evidence of species-level adaptability.
By contrast, HIPPORAG2 retrieved a broader,
taxonomy-oriented context about brown-bear sub-
species (“the taxonomy of brown bears remains
somewhat bewildering”), which led the model to
produce adescriptiveanswer (“there are many va-
rieties, so sizes differ”) but not amechanisticone
(no resource→size link). LIGHTRAG retrieved
topic-level chunks like “Brown bears vary greatly
in size depending on where they live,” which was
enough for correlation but not for causation. This
illustrates the core advantage: our pruning keepsonly the minimal butfunctionalintermediates, so
the model can follow the hops in order instead of
guessing them.
Case study (TV): retrieving the exact two
pieces.A similar pattern appears on TV-style, en-
tangled questions, e.g.,“In The Simpsons minor-
character descriptions, what in-universe line ex-
plains the Yes Guy’s stretched-out ‘Ye-e-e-s?!’,
and what does the entry say about Wiseguy not
actually having a single fixed name?”This ques-
tion is hard not because the language is long, but
because the answer lives intwoseparate men-
tions: one that gives the in-universe justification
(“I had a stro-o-o-oke”) and another that clarifies
the meta-labeling of the Wiseguy character. AU-
TOPRUNEDRETRIEVERretrieved precisely those
two pieces as separate edges/nodes and presented
them together, so the LLM could output both the
in-universe gagandthe meta-level note about the
character not having a fixed proper name. HIP-
PORAG2, which tends to over-expand its graph,
pulled a broader “recurring jokes / minor charac-
ters” context and produced a generic “it’s a run-
ning gag” answer that failed to name the stroke
line. LIGHTRAG, which collapses to topic-level
chunks, also stayed at the descriptive level (“re-
curring jokes create humor”) and missed the ex-
act line. This shows that for entangled narrative
questions, the benefit is not “more graph,” but “the
righttwo edges at once.”
Overall, complex reasoning is where the
pruned, indices-only pipeline shows the clearest
advantage over prior GraphRAG variants.
3.3 Efficiency on Instrumented Corpora
(STEM, TV)
We measure efficiency onSTEMandTV, where
we fully control build-time and storage logging.
Retrieval prompt tokens and latency.Fig-
ure 3 shows average query-time input tokens.
AUTOPRUNEDRETRIEVER-REBEL is most
compact (about 1,090 tokens on STEM and 523
on TV), followed by AUTOPRUNEDRETRIEVER-
LLM(3,027 / 592). HIPPORAG2 and LIGH-
TRAG send much larger contexts (1,898/1,589
and 8,846/2,964). End-to-end latency (Fig. 4)
roughly tracks this token ordering: methods with
longer prompts are slower, while APR-REBEL
remains competitive.
Build-time graph/prompt tokens and
workspace.Figure 5 reports serialized graph-side
payloads and workspace size. APR-REBEL is

STEM TV0200040006000800010000Avg Input T okens
1090.161
523.1963026.610
591.8631897.680
1589.0008846.083
2964.392
STEM TV0102030405060Avg Output T okens53.195
50.59856.212
49.44152.375
48.890
3.25056.287AutoPrunedRetriever-REBEL
AutoPrunedRetriever-llmHippoRAG2 LightRAGFigure 3: Input and output token usage on STEM and
TV .
STEM TV0.00.51.01.52.02.53.03.5Avg gen Latency (s)3.091
2.6072.955
2.601
0.180 0.1991.653
0.687
STEM TV05101520Avg Retrieval Latency (s)8.94814.254
11.52418.062
0.214 0.2123.744
1.210AutoPrunedRetriever-REBEL
AutoPrunedRetriever-llmHippoRAG2 LightRAG
Figure 4: End-to-end latency on STEM and TV .
smallest on both corpora; APR-llm stores more
LLM-extracted triples (≈1.2×106graph/prompt
tokens on STEM and2.84×106on TV), but still
below HIPPORAG2 and LIGHTRAG. Figure 6
shows that APR’s codebook / graph-size growth
has plateaus where new items merge into existing
entities, reflecting the two-layer entity-pruning
step.
STEM TV0.00.20.40.60.81.01.2Graph Prompt T okens1e7
0 01,204,7572,835,689
2,212,0665,016,9414,822,57711,161,008
STEM TV0.00.51.01.52.02.5Graph Completion T okens1e6
0 0761,8162,494,504
624,0781,957,095
764,0441,980,769
STEM TV05001000150020002500Workspace Size (MB)
685427052,292
259443
154370AutoPrunedRetriever-REBEL
AutoPrunedRetriever-llmHippoRAG2 LightRAG
Figure 5: Build-time graph/prompt tokens and
workspace usage on STEM and TV .
3.4 Full GraphRAG Benchmark
We now evaluate on the full GraphRAG bench-
mark (Xiang et al., 2025), i.e.,MedicalandNovel
across Fact Retrieval, Complex Reasoning, Con-
textual Summarize, and Creative Generation (Ta-
ble 1).
OnContextual Summarize, APR transfers
well: on Medical it reaches68.78%/70.14%
ACC (REBEL/LLM), slightly above FAST-
GRAPHRAG (67.88%), and on Novel it reaches
0 20 40 60 80 100 120
Question Index0.00.51.01.52.02.53.0 MB from first
0 20 40 60 80 100
Question Index0.00.51.01.52.0 MB from first
AutoPrunedRetriever-REBEL AutoPrunedRetriever-llmFigure 6: APR codebook / graph-size evolution (left:
STEM, right: TV).
82.55%/83.10%, far above MS-GRAPHRAG
(LOCAL) (64.40%). OnFact Retrieval, APR-
REBEL matches or exceeds strong baselines in
ROUGE-L (e.g.,38.02on Novel) while keep-
ing ACC competitive with classic RAG and
Hippo-style systems. ForCreative Genera-
tion, APR-llm attains62.97%ACC on Novel
and 65.02% on Medical, near or above other
graph-based methods.
Token usage on GraphRAG (Fig. 7) remains
low: APR-REBEL uses1,110tokens on Novel
and1,341on Medical; APR-llm uses956and
2,234, placing both among the most compact
graph-aware systems while staying competitive or
SOTA on complex reasoning and summarization.
Novel Medical050000100000150000200000250000300000350000Avg T okens
1,110 1,341 956 2,243 879 95438,707 39,821331,375 332,881
1,008 1,020100,832 100,310
4,204 4,298 3,441 3,510 7,208 7,342AutoPrunedRetriever-REBEL
AutoPrunedRetriever-llm
V-RAG
MS-GraphRAG(local)MS-GraphRAG(global)
HippoRAG2
LightRAGFast-GraphRAG
RAPTOR
HippoRAG
Figure 7: Average input token usage on the GraphRAG
benchmark (Medical, Novel).
4 Related Work
Retrieval-Augmented Generation (RAG).
RAG grounds LLMs in external knowledge via
retriever–reader pipelines such as DrQA (Chen
et al., 2017), DPR (Karpukhin et al., 2020b), and
the original RAG model (Lewis et al., 2020).
REALM integrates retrieval into pre-training with
a non-parametric memory (Guu et al., 2020),
while FiD performs passage-wise encoding and
fusion for strong open-domain QA at higher
decoding cost (Izacard and Grave, 2021). Atlas
and follow-ups show that retrieval-augmented

Table 1: GraphRAG benchmark results onNovelandMedical.
Category Model Fact Retrieval Complex Reasoning Contextual Summarize Creative Generation
ACC ROUGE-L ACC ROUGE-L ACC Cov ACC Cov FS Cov
Novel DatasetRAG (w/o rerank) 58.76 37.35 41.35 15.12 50.08 82.53 41.52 47.46 37.84
RAG (w/ rerank)60.9236.08 42.93 15.39 51.30 83.64 38.26 49.2140.04
MS-GraphRAG (local) (Edge et al., 2025) 49.29 26.11 50.93 24.09 64.40 75.58 39.10 55.44 35.65
HippoRAG (Guti ´errez et al., 2025) 52.93 26.65 38.52 11.16 48.7085.5538.8571.5338.97
HippoRAG2 (Jim ´enez Guti ´errez et al., 2025) 60.14 31.35 53.38 33.42 64.10 70.84 48.28 49.84 30.95
LightRAG (Guo et al., 2025a) 58.62 35.72 49.07 24.16 48.85 63.05 23.80 57.28 25.01
Fast-GraphRAG (CircleMind AI, 2024) 56.95 35.90 48.55 21.12 56.41 80.82 46.18 57.19 36.99
RAPTOR (Sarthi et al., 2024) 49.25 23.74 38.59 11.66 47.10 82.33 38.01 70.85 35.88
Lazy-GraphRAG (Edge et al., 2024a) 51.65 36.97 49.22 23.48 58.29 76.94 43.23 50.69 39.74
AutoPrunedRetriever-REBEL49.2538.02 63.0231.25 82.55 83.95 59.94 25.78 21.21
AutoPrunedRetriever-llm45.99 26.99 62.80 35.35 83.1083.8662.9734.40 22.13
Medical DatasetRAG (w/o rerank) 63.72 29.21 57.61 13.98 63.72 77.34 58.94 35.88 57.87
RAG (w/ rerank) 64.73 30.75 58.64 15.57 65.7578.5460.61 36.74 58.72
MS-GraphRAG (local) (Edge et al., 2025) 38.63 26.80 47.04 21.99 41.87 22.98 53.11 32.65 39.42
HippoRAG (Guti ´errez et al., 2025) 56.14 20.95 55.87 13.57 59.86 62.73 64.4369.21 65.56
HippoRAG2 (Jim ´enez Guti ´errez et al., 2025)66.2836.69 61.9836.9763.08 46.1368.0558.78 51.54
LightRAG (Guo et al., 2025a) 63.3237.1961.32 24.98 63.14 51.16 - - -
Fast-GraphRAG (CircleMind AI, 2024) 60.93 31.04 61.73 21.37 67.88 52.07 65.93 56.07 44.73
RAPTOR (Sarthi et al., 2024) 54.07 17.93 53.20 11.73 58.73 78.28 - - -
Lazy-GraphRAG (Edge et al., 2024a) 60.25 31.66 47.82 22.68 57.28 55.92 62.22 30.95 43.79
AutoPrunedRetriever-REBEL61.28 32.9672.4930.79 68.78 40.15 64.04 32.19 11.12
AutoPrunedRetriever-llm61.25 34.69 71.59 31.11 70.1440.59 65.02 33.06 28.62
LMs with updatable indices and reranking can
reach strong few-shot performance (Izacard
et al., 2022), but these text-first systems still treat
queries independently and remain passage-centric
and token-heavy for multi-hop or long-context
reasoning.
Vector Retrieval and Re-ranking.Dense
retrieval embeds queries and documents into a
shared space (Lee et al., 2019; Karpukhin et al.,
2020b) and is typically served by ANN in-
dexes such as FAISS (Johnson et al., 2017) or
HNSW (Malkov and Yashunin, 2016). Two-
stage pipelines then apply stronger rerankers:
BERT-based cross-encoders (Nogueira and Cho,
2019) and late-interaction models like Col-
BERT/ColBERTv2 (Khattab and Zaharia, 2020;
Santhanam et al., 2022) improve ranking quality
while trading off latency and indexability.
Graph-based RAG (GraphRAG).Graph-
based RAG replaces flat passages with entity–
relation structure. MS-GraphRAG builds graphs
over private corpora and retrieves summarized
subgraphs, with a local variant that builds per-
session graphs (Edge et al., 2024c). Lazy-
GraphRAG defers graph construction to query
time (Edge et al., 2024b). HippoRAG and
HippoRAG2 introduce neuro-inspired long-term
memory that consolidates and reuses reasoningpaths (Guti ´errez et al., 2025a,b). LightRAG and
Fast-GraphRAG simplify graphs into key–value
forms for low-latency retrieval (Guo et al., 2025b;
CircleMind AI, 2024), while RAPTOR uses a
tree of recursive summaries instead of an explicit
graph (Sarthi et al., 2024). Together, these il-
lustrate the shift from passage-centric to struc-
tured retrieval, but still pay substantial graph-
serialization and token cost.
Memory-Augmented Architectures.
External-memory LMs maintain persistent,
updatable stores alongside parameters, exploring
scalable memory (Wu et al., 2022), efficient
retrieval (Liu et al., 2024), and continual con-
solidation (Dao et al., 2023). These build on
differentiable memory architectures such as mem-
ory networks (Weston et al., 2014) and neural
Turing machines (Graves et al., 2014).
Efficient Model Deployment.LLM efficiency
is improved by quantization and low-rank adap-
tation (e.g., QLoRA) (Dettmers et al., 2023),
mixed-precision training (Micikevicius et al.,
2018), and hardware-aware optimization (Ai et al.,
2024). Adaptive computation and dynamic rout-
ing (Schuster et al., 2022; Li et al., 2023) further
trade depth and routing complexity against accu-
racy, complementing retrieval- and memory-side
efforts to cut tokens and latency.

5 Limitations
Our study has several limitations. First, we
evaluate AUTOPRUNEDRETRIEVERprimarily on
English, knowledge-intensive QA benchmarks
(GraphRAG-Benchmark, STEM, TV); it is un-
clear how well the same pruning and symbol-
ization scheme transfers to other languages, do-
mains, or noisy user logs. Second, our pipeline
depends on upstream triple extractors (REBEL or
an LLM); systematic extraction errors or missing
relations can still harm downstream reasoning, and
we do not jointly train extraction and retrieval. Fi-
nally, we focus on text-only corpora and single-
turn question answering in agentic settings, leav-
ing multimodal inputs, tool-use workflows, and
human-in-the-loop updates to future work.
References
Qinbin Ai, Mingxuan Chen, Wenhu Chen, Tri Dao,
Daniel Fu, Albert Gu, and 1 others. 2024. Effi-
cient inference for large language models: A sur-
vey.Foundations and Trends in Machine Learning,
17(2):145–387.
Jinheon Baek, Myeongho Jeong, Sung Ju Hwang, and
Jungwoo Park. 2023. Knowledge graph grounded
question answering with graph neural networks. In
Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP
’23.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading wikipedia to answer open-
domain questions. InACL.
CircleMind AI. 2024. Fast graphrag: High-speed
graph-based retrieval-augmented generation. Ana-
lytics Vidhya Blog.
Tri Dao, Daniel Y . Fu, Stefano Ermon, Atri Rudra, and
Christopher R ´e. 2023. Hungry hungry hippos: To-
wards language modeling with state space models.
InInternational Conference on Learning Represen-
tations, ICLR ’23.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. Qlora: Efficient finetuning
of quantized llms. InAdvances in Neural Informa-
tion Processing Systems, volume 36 ofNeurIPS ’23.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summariza-
tion.Preprint, arXiv:2404.16130.
Darren Edge, Ha Trinh, and Jonathan Larson. 2024a.
LazyGraphRAG: Setting a New Standard for Qual-
ity and Cost. Microsoft Research Blog.Darren Edge, Ha Trinh, and Jonathan Larson. 2024b.
LazyGraphRAG: Setting a New Standard for Qual-
ity and Cost. Microsoft Research Blog.
Darren Edge, Ha Trinh, Jonathan Larson, Alex Chao,
and Robert Osazuwa Ness. 2024c. From local to
global: A graph rag approach to query-focused sum-
marization. ArXiv:2404.16130 [cs.CL].
Alex Graves, Greg Wayne, and Ivo Danihelka. 2014.
Neural turing machines.Preprint, arXiv:1410.5401.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025a. Lightrag: Simple and fast retrieval-
augmented generation.Preprint, arXiv:2410.05779.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025b. Lightrag: Simple and fast retrieval-
augmented generation. ArXiv:2410.05779 [cs.IR].
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025a. HippoRAG:
Neurobiologically Inspired Long-Term Memory
for Large Language Models. ArXiv:2405.14831
[cs.CL].
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025b. From rag to mem-
ory: Non-parametric continual learning for large
language models. ArXiv:2502.14802 [cs.CL].
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025. Hipporag: Neu-
robiologically inspired long-term memory for large
language models.Preprint, arXiv:2405.14831.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-
supat, and Ming-Wei Chang. 2020. REALM:
Retrieval-augmented language model pre-training.
InICML.
Haoyu Han, Yue Wang, Hagai Shomer, Kai Guo,
Jiawei Ding, Yang Lei, Bo Li, and Jie Tang.
2024. Retrieval-augmented generation with graphs
(graphrag).Preprint, arXiv:2404.16130.
HotpotQA Team. 2019. Preprocessed wikipedia
for hotpotqa.https://hotpotqa.github.io/
wiki-readme.html.
Pere-Llu ´ıs Huguet Cabot and Roberto Navigli. 2021.
REBEL: Relation extraction by end-to-end language
generation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2021, pages 2370–
2381, Punta Cana, Dominican Republic. Associa-
tion for Computational Linguistics.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open-
domain question answering. InICLR.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022. Atlas: Few-shot learning
with retrieval augmented language models.Journal
of Machine Learning Research, 23(251):1–43.

Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to mem-
ory: Non-parametric continual learning for large
language models.arXiv preprint arXiv:2502.14802.
Poster at ICML 2025.
Jeff Johnson, Matthijs Douze, and Herv ´e J´egou.
2017. Billion-scale similarity search with GPUs.
arXiv:1702.08734.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020a. Dense passage retrieval for
open-domain question answering. InProceedings of
the 2020 Conference on Empirical Methods in Nat-
ural Language Processing (EMNLP), pages 6769–
6781, Online. Association for Computational Lin-
guistics.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020b. Dense passage retrieval for
open-domain question answering.arXiv preprint
arXiv:2004.04906.
Omar Khattab and Matei Zaharia. 2020. ColBERT: Ef-
ficient and effective passage search via contextual-
ized late interaction over BERT. InSIGIR.
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova.
2019. Latent retrieval for weakly supervised open-
domain question answering. InACL.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. InAdvances in Neu-
ral Information Processing Systems, volume 33 of
NeurIPS ’20, pages 9459–9474.
Yuhong Li, Daniel Fu, Wenhu Chen, Manoj Kumar,
Noah Smith, Ming-Wei Chen, and Christopher R ´e.
2023. Efficient streaming language models with at-
tention sinks. InProceedings of the 40th Interna-
tional Conference on Machine Learning, ICML ’23.
Zhenghao Liu, Jianfeng Wang, Jie Tang, Jie Zhou, and
Ashwin Kalyan. 2024. Memory-efficient large lan-
guage models via dynamic memory allocation. In
International Conference on Learning Representa-
tions, ICLR ’24.
Yury A. Malkov and Dmitry A. Yashunin. 2016. Ef-
ficient and robust approximate nearest neighbor
search using hierarchical navigable small world
graphs.arXiv:1603.09320.
Paulius Micikevicius, Sharan Narang, Jonah Alben,
Gregory Diamos, Erich Elsen, David Garcia, Boris
Ginsburg, Michael Houston, Oleksii Kuchaiev,
Ganesh Venkatesh, and Hao Wu. 2018. Mixed preci-
sion training. InInternational Conference on Learn-
ing Representations, ICLR ’18.Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage
re-ranking with BERT. InarXiv:1901.04085.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Keshav Santhanam, Omar Khattab, and et al. 2022.
Colbertv2: Effective and efficient retrieval via
lightweight late interaction. InNAACL.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InProceedings of the Inter-
national Conference on Learning Representations
(ICLR). ArXiv:2401.18059 [cs.CL].
Tal Schuster, Adam Fisch, Tommi Jaakkola, and
Regina Barzilay. 2022. Confident adaptive language
modeling. InAdvances in Neural Information Pro-
cessing Systems, volume 35 ofNeurIPS ’22, pages
17456–17472.
Zeyu Sun, Hongyang Yang, Rui Zhou, Chen
Wang, Xiaodong Liu, and Xuanjing Huang. 2022.
Graphene: Retrieval-augmented generation for effi-
cient knowledge-intensive reasoning. InAdvances
in Neural Information Processing Systems, vol-
ume 35 ofNeurIPS ’22.
Xiaorui Wang, Zhiyuan Zhang, Yizheng Hao, Lei Li,
Lei Hou, Zhiyuan Liu, Xuan Song, and Maosong
Sun. 2023. Graph-based memory for large language
models. InProceedings of the 40th International
Conference on Machine Learning, ICML ’23.
Jason Weston, Sumit Chopra, and Antoine Bordes.
2014. Memory networks. InInternational Confer-
ence on Learning Representations, ICLR ’15.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang,
Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadal-
lah, Ryen W White, Doug Burger, and Chi Wang.
2024. Autogen: Enabling next-gen LLM applica-
tions via multi-agent conversation.
Yuhuai Wu, Markus Rabe, DeLesley Hutchins, and
Christian Szegedy. 2022. Memorizing transformer.
InInternational Conference on Learning Represen-
tations, ICLR ’22.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2025. When to use graphs in rag: A com-
prehensive analysis for graph retrieval-augmented
generation.Preprint, arXiv:2506.05690.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models.Preprint, arXiv:2210.03629.

A appendix
A.1 Theoretical Properties
A.1.1 Encoding and Heavy-Tailed Repetition
Lemma 1(Concentration of repetition).LetVbe
the corpus vocabulary andf:V →Ntoken fre-
quencies. In typical language corporafis heavy-
tailed: there is a core setV core⊂ Vsuch that
X
v∈Vcoref(v)≈Θ(|C|),
where|C|is the total token count. Thus, indexing
recurrent entities/relations captures most informa-
tional mass while reducing redundancy.
Sketch.Empirical token distributions in large cor-
pora follow Zipf-like laws; a small set of types
accounts for most occurrences. Mapping these to
IDs and sharing them across queries/facts removes
repeated surface forms without losing the domi-
nant mass.
A.1.2 Chunked Small Graphs (Local-First
Construction)
Lemma 2(Maximal local-coherence partition).
Fix a thresholdτ, a bounded continuity bonus
0≤b <∞, and a fit rule that is monotone
nonincreasing as the centroid drifts away from a
triple. The one-pass rule “append if fit-score≥τ,
else cut” yields a partitionG= (G 1, . . . , G K)in
which eachG kismaximal: no additional triple
can be appended without violating the fit test.
Sketch.Within a segment, appending triples can
only decrease (or leave unchanged) the fit of future
triples because the centroid moves away from any
fixed candidate. Once a triple fails the test, any
larger graph containing it would also fail. Thus,
each cut point defines a maximal prefix w.r.t. the
local rule.
Lemma 3(Boundary-consistency merge).LetΘ
denote the segmenter parameters (threshold, con-
tinuity bonus, etc.). For adjacent runs(L, R), de-
fineMERGE(L, R)as true iff the segmenter ap-
plied to the concatenationL∥Reither (i) pro-
duces a single chunk or (ii) places its first cut away
from the original boundary at|L|. Then a bound-
ary is removediffthe segmenter, when given both
sides at once, would not cut at that location. Sur-
viving boundaries are fixed points of the segmenter
under local re-evaluation.Sketch.The merge test re-runs exactly the same
algorithm and hyperparameters on the local win-
dow. If the “true” segmentation prefers a differ-
ent cut, we merge; otherwise we keep the bound-
ary. This is equivalent to requiring boundary self-
consistency under the same rule.
Lemma 4(Intra-chunk cohesion bound).Assume
unit-normalized triple embeddings(v i)and an ac-
ceptance rulecos(¯c, v i) +δ i≥τwith0≤δ i≤b,
where¯cis the running centroid. For any completed
small graphG kwith|G k| ≥2,
2
|Gk|(|G k|−1)X
p<qcos(v p, vq)≥τ−b.
Sketch.The centroid always lies in the convex hull
of the triple embeddings. The acceptance condi-
tion ensures each new triple is not too far (in co-
sine) from the current centroid, up to the continu-
ity bonusb. Averaging pairwise cosines and using
triangle-type inequalities yields the bound.
Lemma 5(Working-set reduction for retrieval).
LetM=|M|be the number of distinct edges,
and suppose runs have lengthsℓ 1, . . . , ℓ Kwith
L= max kℓk. If symbolic pre-filtering selectsH
candidate runs for re-ranking, then the fine stage
touches at mostH·Ledges. Compared to scan-
ning all edges, this yields an asymptotic reduction
factorΩ(M/(H·L)).
Sketch.Each candidate run contributes at mostL
edges to be scored. BoundingHandLindepen-
dently ofM(corpus size) gives sublinear depen-
dence onM.
Corollary 6(Precision–recall / segmentation
tradeoff).Increasingτ(or decreasing the continu-
ity bonusb) shortens runs, improves cohesion, and
reduces retrieval latency at the cost of recall. De-
creasingτlengthens runs, improves recall, but in-
creases candidate sizes. The boundary merge step
counteracts over-segmentation by removing unsta-
ble cuts.
Sketch.Higher thresholds cause earlier cuts;
lower thresholds allow more heterogeneous con-
tent in each run. The merge rule prunes cuts
that the full-window segmenter would not repro-
duce.

A.1.3 Coarse retrieve
Lemma 7(Efficiency of Coarse→Fine with Max–
Pair Filtering).Letnbe corpus size,dthe em-
bedding dimension, andk≪nthe coarse
shortlist size. The coarse stage computes, per
query–candidate pair, constants over small en-
tity/relation sets; the fine stage evaluates onlyk
items with sentence embeddings. Thus cost drops
fromO(n·d)toO(k·d)while preserving precision
providedkretains high-overlap candidates.
A.1.4 Semantic Selection and Consensus
Lemma 8(Semantic consensus is close to its
members).Let runsr 1, . . . , r mhave embed-
dingsψ(r i)and cosine similaritysim(r a, rb) =
cos(ψ(r a), ψ(r b)). Let¯rmaximize the average
similarity
¯r∈arg max
rmX
i=1sim(r, r i).
If all runs in the cluster are mutually similar, i.e.,
sim(r a, rb)≥θfor someθclose to 1, then¯ris
also close to every member:
1−sim(r a,¯r)≤ε(θ)for alla,
for a functionε(θ)→0asθ→1.
Sketch.On the unit sphere, cosine similarity de-
fines a bounded distance. If¯rwere far from some
memberr awhile all runs are mutually close, mov-
ing¯rtowardr awould increase its average simi-
larity to the cluster, contradicting maximality. The
boundε(θ)follows from standard geometric argu-
ments.
A.1.5 Consolidation of Entities and Edge Se-
quences
Lemma 9(Quotient consolidation reduces edge
cardinality).LetG= (V, R, E)be a directed
multigraph with labeled edgesE⊆V×R×V.
Let∼be the equivalence relation onVinduced
by entity consolidation andπ:V→V/∼the
projection. Defineϕ:E→E′byϕ(u, r, v) =
(π(u), r, π(v))and letE′= uniq(ϕ(E)). Then:
1.|E′| ≤ |E|, with equality iffϕis injective on
E.
2. For any edge sequenceσ= (e i1, . . . , e iT),
the remapped-and-deduped sequenceσ′=
uniq(ϕ(σ))satisfies|σ′| ≤ |σ|.Sketch.E′is the image ofEunderϕfollowed by
deduplication, so it cannot have more elements.
Sequences inherit this property pointwise; collaps-
ing equal edges cannot increase length.
Lemma 10(Vectorization cost is preserved).
Letv sbe a schema vectorizer depending only
on symbolic indices and precomputed vectors
(Eemb, Remb). Then the number of sentence-
level text encodings required by the pipeline is un-
changed by applyingϕand deduplicating edges.
Sketch.Consolidation changes only which indices
point to which vectors. All sentence encodings are
done before consolidation (for triples and chunks),
so later index manipulation reuses them.
Lemma 11(Medoid representatives minimize
within-cluster distortion).LetC⊆Vbe a
cluster of entities and define cosine dissimilarity
d(x,y) = 1−cos(x,y). A medoidr⋆∈Csatis-
fies
r⋆∈arg min
r∈CX
i∈Cd(ei,er),
and for any other representativer∈C,
X
i∈Cd(ei,er⋆)≤X
i∈Cd(ei,er).
Sketch.This is the standard property of medoids:
by definition they minimize total dissimilarity
within the cluster.
A.1.6 DPO Wrapper and Policy Behavior
Lemma 12(DPO aligns policy log-odds with util-
ities).Assume preferences follow a Bradley–Terry
model:
Pr(y+≻y−|x) =σ 
λ[U(x, y+)−U(x, y−)]
for someλ >0, whereUis a latent utility. Let
πrefbe fixed. Then any minimizerπ θof the DPO
objective satisfies, up to a normalization constant
Cx,
logπ θ(y|x)−logπ θ(y′|x)
=λ
βdpo
U(x, y)−U(x, y′)
+ ∆ ref(y, y′|x) +C x.
where∆ refdepends only onπ ref.
Sketch.DPO maximizes the conditional log-
likelihood of observed pairwise preferences under

a logistic link, with a reference correction. At op-
timum, gradient stationarity enforces proportion-
ality between policy log-odds and utility differ-
ences, offset by the fixed reference.
Lemma 13(Monotone token control via ac-
tion lattice).Suppose each channel’s action set
{include all,unique,not include}forms a
lattice under⪰with
include all⪰unique⪰not include,
such that Tokens(x, y)is monotone nonincreasing
down the lattice and Acc(x, y)is Lipschitz in a
task metric. Then there exists a Lagrange mul-
tiplierη⋆≥0such that the DPO-trained policy
with penaltyη⋆·Tokens attains a target budgetB,
and tighter budgetsB′< Bcan be achieved by
increasingη(eventually collapsing to alwaysnot
include).
Sketch.On a finite action set, the mixed policy
over actions yields a convex set of achievable (to-
kens, accuracy) pairs. A standard Lagrangian ar-
gument with a monotonically ordered action set
gives existence of a multiplier realizing each feasi-
ble budget, and increasing the penalty pushes mass
toward cheaper actions.
A.2 Effect of Token Length and Benefits of
Entity–Relation Factorization
Sequence embeddings.For simplicity, we ap-
proximate the embedding of a text spanSof
lengthntokens by the mean of its token embed-
dings:
z(S)≈1
nnX
i=1xi∈Rd.
Each token embedding decomposes into
xi=si+εi,
wheres iis the semantic signal andε iis zero-mean
noise withE[ε i] = 0andVar(⟨q, ε i⟩) =σ2for
any unit query vectorq.
We partition the tokens into: (i) relevant tokens
R(carry information the query cares about) and
(ii) irrelevant tokensI(background, boilerplate,
narration), with|R|=mand|I|=n−m.
Lemma 14(Token dilution).Letqbe a unit query
vector aligned with the average relevant signal
µrel:=1
mX
i∈Rsiwith⟨q, µ rel⟩=α >0.Assume irrelevant tokens have no systematic
alignment withq, i.e.,E[⟨q, s i⟩] = 0fori∈I.
Then the signal-to-noise ratio (SNR) of the pas-
sage embedding in the directionqdecays as
SNR(S) := 
E[⟨q, z(S)⟩]2
Var(⟨q, z(S)⟩)∝1
n.
Proof.We have
z(S) =1
nnX
i=1xi=1
nX
i∈R(si+εi)+1
nX
i∈I(si+εi).
Taking inner product withqand expectation, and
usingE[⟨q, ε i⟩] = 0for alliandE[⟨q, s i⟩] = 0for
i∈I,
E[⟨q, z(S)⟩] =1
nX
i∈R⟨q, s i⟩=m
nα.
Thus, for fixedmandα, the expected signal scales
asmα/n.
For the variance, by independence and identical
variance of the noise:
Var(⟨q, z(S)⟩) = Var1
nnX
i=1⟨q, ε i⟩
=1
n2nX
i=1Var(⟨q, ε i⟩) =σ2
n.
Therefore
SNR(S) =(mα/n)2
σ2/n=m2α2
σ2·1
n,
which showsSNR(S)∝1/nas claimed.
Corollary 15(Length bias).Consider two pas-
sagesS shortandS longthat contain the samemrel-
evant tokens (same fact, sameα) but have lengths
nsandn ℓwithn ℓ> ns. Then
E[⟨q, z(S short)⟩] =m
nsα >m
nℓα=E[⟨q, z(S long)⟩]
Thus, even for equally relevant content, longer
passages tend to produce lower expected similar-
ity toqand are systematically disadvantaged in
retrieval.
Implication.Lemma 14 and Corollary 15 for-
malize a “token dilution” effect: when we em-
bed entire passages, the representation of a fact is
weakened by irrelevant tokens, and the SNR de-
creases as1/nwith passage length. Consequently,

retrieval quality depends not only on what is said,
but also on how long and where it is written.
Entity–relation factorization.In our system,
we instead represent knowledge as graph edges
(e, r, e′)∈M⊆E×R×Eand store embed-
dings for entities and relations via the codebook
Eemb:E→Rd, R emb:R→Rd.
Thus each factf= (e, r, e′)is encoded by the
triple
Eemb(e), R emb(r), E emb(e′),
derived from short token sequences for entity
names and relation labels, rather than from whole
passages.
Proposition 16(Advantages of E–R–E embed-
dings).LetE, R,M, E emb, Remb be as in Sec-
tion 2.2. Under the averaging+noise model of
Lemma 14, the following hold:
1. (High-SNR micro-embeddings) There exist
constantsc 1, c2>0, independent of the pas-
sage lengthn, such that for any factf=
(e, r, e′)∈M,
c1≤SNR(v)≤c 2
∀v∈ {E emb(e), R emb(r), E emb(e′)}.
In particular, E–R–E embeddings donotsuf-
fer the1/ndecay of Lemma 14.
2. (Compositional query scoring) Let a queryq
induce componentsq E, qR∈Rd. For suit-
able nonnegative weightsλ s, λr, λtwe can
score an edge(e, r, e′)∈Mvia
score(e, r, e′|q) =λ s⟨qE, Eemb(e)⟩+
λr⟨qR, Remb(r)⟩+λ t⟨qE, Eemb(e′)⟩,
i.e., as an inner product between a structured
query and short, high-SNR E–R–E embed-
dings, instead of a single inner product with
a noisy passage embeddingz(S).
3. (Localized interference and updates) Each
factfhas its own edge(e, r, e′); inter-
ference between facts arises only through
sharedE emb(·)orR emb(·). Updating a sin-
gle fact changesO(1)vectors instead of re-
embedding entire passages containing many
unrelated facts.Proof sketch.(1) Let the surface string fore∈E
haven Etokens, of whichm Eare semantically rel-
evant. By constructionn Eis bounded by a small
constant (e.g.,1–3), som E≈n E=O(1). Ap-
plying the same calculation as in Lemma 14 with
n=n Egives
SNR 
Eemb(e)
∝m2
E
nE= Θ(1),
with constants independent of the passage length
nin whichfappears. The same argument applies
toR emb(r)andE emb(e′), yielding the claimed
boundsc 1, c2.
(2) Because we storeE emb(e),R emb(r), and
Eemb(e′)separately, any queryqthat decomposes
into components(q E, qR)admits the factorized
score above. Algebraically, this is a weighted
sum of inner products between short E–R–E vec-
tors and corresponding query components, rather
than a single inner product⟨q, z(S)⟩with a length-
dependent passage embedding.
(3) The representation of a factfis the triple
(Eemb(e), R emb(r), E emb(e′)). Adding, remov-
ing, or modifyingfonly affects these embed-
dings and other edges sharinge,r, ore′. No
re-embedding of unrelated tokens is required, in
contrast to passage-level embeddings that entan-
gle many facts in the samez(S).
A.3 Retrieval Details
For completeness we summarize the exact scoring
terms used in the coarse and fine stages.
Coarse score.An indexed runydecodes to
triplesS(y) ={(h, ρ, t)} ⊆M. We collect en-
tity and relation embeddings into matricesE(y)∈
Rne×dandR(y)∈Rnr×d. For a queryqand can-
didate runf,
scoarse(q, f) =w entmax
i,jcos 
E(q) i, E(f) j
+w relmax
p,rcos 
R(q) p, R(f) r
,
and we takeI k= TopKfscoarse(q, f).
Fine score from triple lines.For each candidate
f∈I k, we linearize triples to short lines “h ρ t”,
embed query and candidate lines intoQ∈Rnq×d
andC∈Rnc×d, and form the cosine matrix
S=bQbC⊤∈[−1,1]nq×nc.
All fine-stage terms are computed onS:

•RelTopT:flattenS, take the top-tentries, and
average.
•Coverage:Cov(τ cov) =P
i1[max jSij≥
τcov].
•Many-to-many (MP):applyσ 
(Sij−
τpair)/Tpair
elementwise and normalize by√nqncorlog(1 +n qnc).
•Distinct 1:1:greedily select the largest un-
used entries aboveτ distand average them
with a1a)Raw−textprompt/√mfactor.
•Whole-chunk gate:compute a full-chunk
cosine between concatenated query and can-
didate text, normalize by a length term, and
gate with a sigmoid so very long but off-topic
chunks do not get extra credit.
The final semantic score is a weighted sum
sfine= RelTopT +λ covCov +λ mpMP
+λ 1:1Distinct +λ wholeWholeGate.
with one set of weights and thresholds per
dataset/model, reused across all experiments.
A.4 Prompt Format and Input Encoding
We encode each input as a compact, graph-
structured prompt rather than a long text block.
Concretely, a prompt consists of:
• A codebook of entities and relations,E′and
R′(either as words or short IDs).
• Edge sequences for the query, prior knowl-
edge, and facts: query edgesq, knowledge
edgesk, and fact edgesf.
• A short instruction block describing how to
interpret each tuple(h, ρ, t)or ID triple.
Figure 8 contrasts a conventional raw-text
prompt with our ID-based and word-based encod-
ings. The ID variants use a JSON-style schema:
One of three formats depends on which one has
fewer tokens.(a) Raw-text prompt
Q: Which subsidiaries acquired since 2021
are exposed to new EU rules?
Context:
- In 2022, AlphaCorp acquired BetaLtd...
- EU Regulation 2024/12 applies to...
- Post-merger reports indicate ...
(plus additional retrieved passages ...)
(b) ID-referenced codebook with edge matrix
{
"e": ["AlphaCorp","BetaLtd",
"EUReg2024_12","2021+"],
"r": ["acquired_in","exposed_to",
"subject_to"],
"edge_matrix": [[0,0,3],
[1,1,2],
[0,2,2]],
"questions(edges[i])":[0,1],
"facts(edges[i])": [0,2]
"rules":"<KB schema string>"
}
(c) ID-referenced compact triples
{
"e": ["AlphaCorp","BetaLtd",
"EUReg2024_12","2021+"],
"r": ["acquired_in","exposed_to",
"subject_to"],
"questions([[e,r,e], ...]):": [[0,0,3], [1,1,2]],
"facts([[e,r,e], ...]):": [[0,0,3], [0,2,2]],
"rules":"<KB schema string>"
}
(d) Word-level triples (no IDs)
{
"questions(words)": [[AlphaCorp,acquired_in,2021+],
[BetaLtd,exposed_to,EUReg2024_12]],
"facts(words)": [[AlphaCorp,acquired_in,2021+],
[AlphaCorp,subject_to,EUReg2024_12]],
"rules":"<KB schema string>"
}
Figure 8:AutoPrunedRetriever input encodings.
Panel (a) shows a conventional long-context prompt;
(b) encodes the same information via an entity/relation
codebook and an edge matrix; (c) uses explicit triple
lists with IDs; (d) uses full-word triples.
(a) Edge-matrix JSON schema
---Knowledge Base---
[JSON format]
- e: list of entities (e[i] = entity string)
- r: list of relations (r[j] = relation string)
- edge_matrix: [[head_e_idx, r_idx, tail_e_idx]]
* NOTE: edges[i] is just shorthand for edge_matrix[i]
- questions(edges[i]): questions linked by edge i
- given knowledge(edges[i]): prior answers linked by edge i
- facts(edges[i]): facts linked by edge i
(b) ID-based triple JSON schema
---Knowledge Base---
[JSON format]
- e: list of entities (e[i] = entity string)
- r: list of relations (r[j] = relation string)
- [e,r,e]: triple [head_e_idx, r_idx, tail_e_idx]
- questions([[e,r,e], ...]): question triples
- given knowledge([[e,r,e], ...]): prior answer triples
- facts([[e,r,e], ...]): fact triples
(c) Word-level triple schema
---Knowledge Base---
[JSON format]
- questions(words): question triples
- given knowledge(words): prior answer triples
- facts(words): fact triples
Figure 9:Knowledge-base JSON specifications
(“rules”) used by AutoPrunedRetriever.The con-
crete encodings in Fig. 8 all instantiate one of these
schemas.
•e: entity vocabulary (either strings or IDs).

•r: relation vocabulary.
•edge matrixor triple lists:[h, r, t]indices
intoe/rorE/R.
•questions,knowledge,facts: subsets of
edges tagged as questions, prior knowledge,
or background facts.

A.5 Cross-domain qualitative comparison on STEM and TV
Domain /
IDQuestion
(abridged)AutoPrunedRetriever Result
Analysis (abridged)HippoRAG2 Result Analysis
(abridged)LightRAG Result Analysis
(abridged)
STEM-
5c755e96Brown bear size
variation and adapt-
abilityContext:“Kodiak bears are
largest due to high salmon avail-
ability; inland bears smaller with
limited resources.”
Reasoning:geography→food
abundance→body-mass shift→
adaptability.
Answer:Larger coastal/Kodiak
bears reflect rich caloric in-
take; smaller inland bears re-
flect scarcity⇒size variance ev-
idences environmental adaptabil-
ity.
Error:— (correct)Context:“The taxonomy of
brown bears remains bewilder-
ing; multiple subspecies identi-
fied.”
Reasoning:taxonomy→mor-
phological variation (no environ-
mental cause).
Answer:Size variability indi-
cates subspecies diversity.
Error:Misses causal driver
(resources)⇒descriptive, not
mechanistic.Context:“Brown bears vary
greatly in size depending on
where they live.”
Reasoning:region→size→
adaptation (shallow).
Answer:Bears adapt to local
conditions, so sizes differ.
Error:Correlation only; lacks re-
source/metabolic link.
STEM-
4e26ae6dHistorical range→
ecological roleContext:“Mexican grizzly /
Kodiak / Himalayan subspecies;
apex predators affecting vegeta-
tion and prey.”
Reasoning:historical range→
diversification→habitat adapta-
tion→modern trophic role.
Answer:Past range shaped re-
gional lineages whose adapta-
tions underwrite today’s grizzly
apex role.
Error:— (correct)Context:“Pleistocene lineage
prior to demise.”
Reasoning:lineage timeline→
extinction (no present ecology).
Answer:Historical divergence
explains current bears (vague).
Error:Lacks link to present-day
ecological function.Context:“Ecological dynamics
of predator–prey systems.”
Reasoning:ecosystem complex-
ity→generic role.
Answer:Grizzlies play roles in
ecosystems (generic).
Error:No entity-level or causal
path from range to role.
STEM-
1b8f5662Physical adapta-
tions→hunting
success (moose)Context:“Charge and scent-
based ambush tactics; terrain af-
fects prey choice.”
Reasoning:morphology + habi-
tat→tactic→success vs large
prey.
Answer:Bears’ strength/claws
+ terrain-leveraged tactics raise
success on moose.
Error:— (correct)Context:“Brown bears as apex
omnivores in ecosystems.”
Reasoning:apex predator→sur-
vival (no tactics).
Answer:As apex predators they
can hunt large prey.
Error:Omits behavioral mecha-
nism/tactical link.Context:“Bears interact with
diverse ecosystems.”
Reasoning:environment→
adaptation (broad).
Answer:Environmental adapta-
tion enables hunting.
Error:High-level summary; no
tactic/terrain edge.
TV-
29d2f5b1Caboose and
Omega possession
(Red vs Blue)Context:“Caboose’s abnormal
behavior linked to Omega pos-
session and oxygen deprivation
after suit reboot.”
Reasoning:possession + hypoxia
→erratic acts→friendly-fire.
Answer:Caboose; accidents
stem from AI control + hypoxia.
Error:— (correct)Context:“Carolina’s body taken
by Omega; personality changes.”
Reasoning:AI possession→
behavior change (host misat-
tributed).
Answer:Carolina behaves abnor-
mally due to Omega.
Error:Entity confusion; tem-
poral mismatch; misses hypoxia
factor.Context:“Omega AI causes ag-
gression.”
Reasoning:AI influence→ab-
normal behavior (partial).
Answer:Omega explains erratic
acts.
Error:Omits oxygen-deprivation
component; partial causality.
TV-
33a6bd74Sarge’s Season 15
depression and re-
demptionContext:“Sarge creates fake en-
emies, betrays Reds/Blues, later
saves them.”
Reasoning:depression→be-
trayal→redemption (temporal).
Answer:Depression triggers be-
trayal; later redemption by saving
team.
Error:— (correct)Context:“Sarge experiences
burnout.”
Reasoning:depression→low
morale (truncated).
Answer:Sarge acts poorly due to
burnout.
Error:Misses be-
trayal–redemption arc.Context:“Acts irrationally after
long missions.”
Reasoning:fatigue→misbehav-
ior (truncated).
Answer:Irrational actions due to
fatigue.
Error:Omits betrayal + later
change-of-heart.
TV-
1e87f0a2The Simpsons –
Yes Guy / Wiseguy
meta humorContext:“Yes Guy’s ‘Ye-e-e-
s?!’ explained by ‘I had a stro-o-
o-oke’; Wiseguy labeled stereo-
type.”
Reasoning:stroke gag→speech
quirk→meta-reference.
Answer:The gag is justified in-
universe; Wiseguy isn’t a fixed
name.
Error:— (correct)Context:“Running joke across
episodes.”
Reasoning:repetition→humor
(no causal quote).
Answer:It’s a recurring joke.
Error:Lacks textual evidence
explaining the quirk.Context:“Minor characters re-
curring jokes.”
Reasoning:trope repetition→
humor (generic).
Answer:Recurring jokes create
humor.
Error:Descriptive only; misses
explicit line/second part.
Table 2: Cross-domain qualitative comparison on STEM and TV questions. Each model column includes its
retrieved context, reconstructed reasoning chain, final answer, and an error note (if any). AutoPrunedRetriever pre-
serves minimal but functional causal paths; HippoRAG2 tends to over-expand associatively; LightRAG collapses
to topic-level summaries.
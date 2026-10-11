# NativeScope: Relation-Localized Retrieval over Native Topology with a Correct Anchor

**Authors**: Long Wang

**Published**: 2026-10-08 16:21:51

**PDF URL**: [https://arxiv.org/pdf/2610.12243v1](https://arxiv.org/pdf/2610.12243v1)

## Abstract
Dense retrieval usually ranks text chunks by their semantic similarity to a question. This ignores structure that many data systems already store, including section membership, session boundaries, and native order. We propose NativeScope, a scope-then-rank method for queries with a known anchor and relation. It represents a query as q -> (A, r, B). The anchor A and relation r select native units through belonging, before, or after operators, and the target term B ranks only chunks that overlap the selected scope. An internal variant, NS-FullQ, ranks the same candidates with the full question. We evaluate both methods on 200 controlled document and memory records derived from QASPER and LongMemEval under a 1,024-token budget. NativeScope attains native-unit recall of 89.28 percent for documents and 72.50 percent for memories, improving over instance-wide Dense RAG by 42.75 and 22.00 percentage points. NS-FullQ reaches 87.78 percent and 68.50 percent; its differences from NativeScope are inconclusive, locating the primary gain in relational scoping rather than the shorter ranking query. With automatic Top-1 anchors, memory recall falls to 35.50 percent. NativeScope is therefore effective when anchor coordinates and native relations are reliable, but hard scoping inherits errors from the localization interface.

## Full Text


<!-- PDF content starts -->

NativeScope: Relation-Localized Retrieval over Native T opology
with a Correct Anchor
Long W ang
alueas@126.com
ORCID: 0009-0006-8371-1562
Abstract
Dense retrieval usually ranks text chunks by their semantic similarity to a question.
This ignores structure that many data systems already store, including section membership,
session boundaries, and native order. W e propose NativeScope, a scope-then-rank method
for queries with a known anchor and relation. It represents a query as q→(A, r, B ): the
anchor Aand relation rselect native units through belonging, before, or after operators, and
the target term B ranks only chunks that overlap the selected scope. An internal variant,
NS-F ullQ, ranks the same candidates with the full question. W e evaluate both methods on
200 controlled document and memory records derived from QASPER and LongMemEval
under a 1,024-token budget. NativeScope attains native-unit recall of 89.28% for documents
and 72.50% for memories, improving over instance-wide Dense RAG by 42.75 and 22.00
percentage points. NS-F ullQ reaches 87.78% and 68.50%; its differences from NativeScope
are inconclusive, locating the primary gain in relational scoping rather than the shorter
ranking query . With automatic T op-1 anchors, memory recall falls to 35.50%. NativeScope
is therefore effective when anchor coordinates and native relations are reliable, but hard
scoping inherits errors from the localization interface.
Keywords: relational retrieval; native topology; retrieval-augmented generation; long-term
memory; conditional evaluation
1 Introduction
Retrieval-augmented generation typically finds evidence by comparing a question with text
chunks. Some questions, however, constrain location as well as content: the answer may lie
in the section containing a paragraph or before a particular dialogue turn. Documents and
memory systems often preserve these relations as section identifiers, session boundaries, and
native order. T reating the collection as flat text discards information that can directly restrict
the search space.
W e ask a conditional question: how much does relational scoping help once the query anchor
is known and the stored topology is trustworthy? The anchor may come from a user selection,
a stable record identifier, or a validated upstream locator. Our main evaluation supplies the
correct coordinate to isolate retrieval; a separate automatic-localization experiment measures
how the method degrades when that condition fails.
NativeScope writes a relational query as q→(A, r, B ).A identifies a location, rspecifies
a scope operation over native structure, and B identifies what should be retrieved within that
scope. The key step is to use the relation as a candidate-domain constraint before semantic
ranking. The full question can also rank candidates within the same scope; we call this internal
variant NS-F ullQ. Relational scoping and the choice between qandB as a ranking input are
therefore evaluated separately .
NativeScope connects native coordinates, relation operators, and semantic ranking in a single
retrieval interface. Unlike systems that infer a new hierarchy or graph, it executes relations
1
arXiv:2610.12243v1  [cs.IR]  8 Oct 2026

already present in the source data. W e implement the same pipeline over document paragraphs
and dialogue turns, with shared definitions of coordinates, token budget, and evidence coverage.
The evaluation separates the value of relational scoping from the choice of ranking query and
from the reliability of anchor localization.
2 Related W ork
2.1 Dense retrieval and retrieval-augmented generation
Dense retrieval and RAG rank semantically related passages and then expose them to a generator
[1,2,3]. Iterative systems additionally decide when to retrieve again or how retrieval should
interact with reasoning and generation [ 4,5,6]. NativeScope addresses a different decision inside
each retrieval step: which candidates are eligible when the data source already supplies a relation
to an anchor.
Long context does not eliminate candidate selection. Models may use relevant information
poorly when it appears in the middle of a context [ 7], and LongRAG organizes long-document
retrieval from both global-information and local-fact perspectives [ 8]. Our evaluation keeps the
encoder and token budget fixed and isolates the effect of candidate scope rather than generation
quality .
2.2 Hierarchy , graph indexes, and retrieval granularity
Structured retrievers organize candidates through document hierarchies, recursive summaries,
entity graphs, or discourse trees [ 9,10,11,12,13]. These systems show that flat similarity
search can benefit from structured candidate organization, but generally construct or infer an
index during preprocessing.
Retrieval quality also depends on the indexed unit and its context: prior work compares
document, passage, sentence, and proposition indexes, preserves global context during chunk
encoding, or expands hits by semantic continuity [ 14,15,16]. NativeScope is orthogonal to
these choices. It executes section, session, and order relations already stored by the source, then
maps the resulting native units to whatever chunks the index provides.
2.3 Conversational retrieval, anchors, and long-term memory
Conversational retrieval must recover an information need from multi-turn context, often
through learned conversation representations, query rewriting, or salient entities [ 17,18,19].
This work suggests a useful separation between locating a relevant context and ranking content
within it. NativeScope makes that separation explicit through anchor, relation, and target
inputs.
Long-term memory systems add storage policies, hierarchy , temporal organization, and con-
textual reconstruction to retrieval [ 20,21,22,23,24]. CueMem is especially close to anchor-
centered retrieval because it locates a source turn from a cue and restores related context [ 25].
Our experiments hold the decomposition, topology , and encoder fixed to isolate the effect of
executing an explicit relation over stored coordinates.
3 Problem F ormulation and Method
Figure 1shows the complete NativeScope pipeline. A relation is executed over native units
before eligible units are mapped to text chunks. Semantic similarity ranks only within the
resulting relation-localized scope, separating where to search from what to search for .
2

Figure 1: NativeScope pipeline. A decomposed query and correct anchor drive a native relation
operator. Eligible units are mapped to candidate text chunks by character-interval overlap.
NativeScope and NS-F ullQ share the same scoping stage and differ only in within-scope ranking
input. Gold target locations and answers are not exposed to the ranker.
3.1 Input conditions and native topology
Let a runtime instance be X= (U, TX), where U={u1, . . . , u n}contains native units and TX
stores their order, container identifiers, and character intervals. Documents use paragraphs as
units and sections as containers; memories use dialogue turns as units and sessions as containers.
Leto(u)be unit order, c(u)its container, and I(u)its character interval. The text-chunk index
Cis produced from the original text, and every chunk retains its interval I(C)and a mapping
to overlapping native units.
The main task receives query q, a frozen decomposition (A, r, B ), correct anchor coordinate
a, and reliable topology TX. Gold target locations and answers are not provided to the ranker.
The anchor is additional structural input and is held constant across all anchor-dependent com-
parisons. Automatic parsing and localization can be connected upstream, but their reliability is
evaluated separately from the main retrieval condition.
3.2 Why observed topology matters
Pretrained representations may encode common structural semantics, but they can use an
instance-specific relation only when the retrieval interface exposes it. Consider two instances
with the same observable query and text but different section or session assignments. A retriever
that sees only the shared text must return the same candidates for both, even when the relation
points to different target units. W e refer to this mismatch between represented content and
unavailable runtime structure as a local topology-availability gap.
This observation motivates the interface rather than a universal impossibility claim. A
suﬀiciently large budget could return the union of all plausible targets, and some questions
remain answerable from content alone. NativeScope instead asks whether explicitly executing
an available relation can improve retrieval under a fixed budget.
3

3.3 Relation-localized scoping and target ranking
W e implement three operators. The before and after relations use native unit order rather than
general temporal reasoning:
Γbelonging (a, TX) ={u:c(u) =c(a)}, (1)
Γbefore (a, TX) ={u: 1≤o(a)−o(u)≤8}, (2)
Γafter(a, TX) ={u: 1≤o(u)−o(a)≤8}. (3)
Directional scopes follow global native order and are not truncated at the current container. Let
S= Γr(a, TX). Eligible chunks are
C(S) ={C∈C:∃u∈S,|I(C)∩I(u)|>0}. (4)
This overlap mapping does not trim chunks at a scope boundary , so an eligible boundary chunk
may contain a small amount of out-of-scope text. Scoping constrains candidate eligibility rather
than enforcing character-level clipping.
With normalized encoder e, NativeScope ranks eligible chunks by
sB(C) =e(B)⊤e(C), C∈C(S). (5)
NS-F ullQ replaces B with the full question qwhile keeping the anchor, relation operator, and
candidate set identical. The two variants isolate the effect of the ranking input within a common
relation-localized framework.
Figure 2illustrates the operators on a common native-unit sequence. Relational selection
precedes chunk scoring: only chunks whose character intervals intersect selected units become
candidates, regardless of the semantic scores of out-of-scope chunks.
Figure 2: Native relation operators and unit-to-chunk mapping. Belonging selects the anchor’s
container, while before and after select up to eight units in the respective direction under global
native order. Colored bars denote chunks that may span several native units.
3.4 Budget and chunking
Chunks are ranked by decreasing score, with ties broken by start character, end character,
and chunk identifier. Context is assembled in rank order while already returned overlapping
characters are deduplicated. The BGE tokenizer counts the actual concatenated text. Retrieval
stops when adding the next chunk would exceed 1,024 tokens; it neither skips that chunk to
seek a shorter one nor truncates a prefix to fill the remaining budget.
Proposition 1 (Conditional chunking invariance of the native scope) .If two chunkings leave
native units, topology, and the anchor coordinate unchanged, and Γrdoes not inspect chunk
boundaries, both chunkings yield the same native retrieval scope S.
4

Algorithm 1 NativeScope under a correct anchor
Require: Correct anchor a, relation r, target term B, topology TX, chunksCwith character
intervals, budget L
Ensure: Ranked returned chunks R; budget accounting uses their deduplicated context
1:S←Γr(a, TX);R←∅
2: ConstructC(S)by native character-interval overlap
3: RankC(S)bysBwith deterministic coordinate-based tie breaking
4: for each ranked chunk C do
5: H←DedupeAndJoin( R+ [C])
6: ifTokenCount( H)> L then
7: break
8: end if
9: R←R+ [C]
10: end for
11: return R
Proof. All inputs to Γrare unchanged, so its output is unchanged.
Native-unit ranking would also be invariant if it did not depend on chunking. Our imple-
mentation encodes chunks and selects them under a budget, so chunking can change candidates,
representations, and complete-evidence coverage even when Sis stable. Container-to-unit and
unit-to-chunk mappings can be precomputed to avoid re-inferring stored relations; we do not
claim a measured latency or memory advantage from this implementation choice.
4 Data and Experimental Protocol
4.1 Controlled data construction and review
The document domain uses QASPER v0.3 test [ 26], and the memory domain uses LongMemEval-
S cleaned [ 27]; oracle and M splits are not mixed in. W e derive relational queries from real source
text and gold evidence rather than report leaderboard results on the original benchmarks. Each
domain contains 100 records allocated as 50 belonging, 25 before, and 25 after queries, for 200
records total. Each source question contributes at most one accepted record, and calibration
and formal pools are separated.
Gold evidence is mapped to native target units before eligible anchors are enumerated.
Belonging pairs share a container and are separated by at least four units; before and after pairs
place the target four to eight units from the anchor in the specified direction. QASPER evidence
must map unambiguously to paragraphs, while LongMemEval uses the answer-session identifier
and turn-level has_answer marker. F rozen hash ordering and rule-based validation enforce the
relation quotas, coordinate constraints, evidence support, and leakage checks.
An instruction-following model assisted question wording at temperature zero, followed by
agent review and local rule validation. This process is recorded as agent-assisted construction,
not independent human annotation. T arget-term refinement preserves phrases that can be
located in the source. Some targets remain identifiable from B alone, so the dataset tests
whether relational scoping is useful, not whether topology is counterfactually necessary for
every question. Model identity , prompts, review fields, and provenance are retained with the
accompanying artifacts; Appendix A summarizes data identity and run provenance.
Source versions, licenses, derived diagnostic material, and checksums appear in Appendix A.
Diagnostic records are not added to the 200-example main set.
5

4.2 Methods and shared settings
All methods use 768-dimensional normalized BGE-base-en-v1.5 embeddings and cosine similarity
[28], with a shared instruction prefix on the query side. Encoding length is explicitly capped
at 512 tokens. Long native units are truncated only for automatic localization embeddings;
original text used for coordinates and budget accounting is not truncated. The main evaluation
forms nonoverlapping chunks from fixed 256 source-token spans ( fixed_256 ) with reversible
character mappings and a common 1,024-token return limit. Because tokenization can differ
across processing stages, a chunk re-encoded by BGE need not contain exactly 256 BGE tokens.
T able 1distinguishes the information and role of each comparison. Dense RAG searches
every chunk within a source instance. Anchor Window is an undirected ±8native-unit window
ordered by distance to the anchor, with ties broken by native order and chunk identifier; it
does not semantically rerank for the target. Because constructed targets are at least four units
from the anchor and long units can exhaust the token budget, this is a generic distance-window
baseline rather than a scope-matched implementation of the three relations.
T able 1: Interfaces of the main comparisons. The last three methods receive the same correct
anchor; all methods share chunks, encoder, and return budget.
Method Candidate scope Ranking input/basis Role
Dense RAG F ull instance F ull question q Flat dense baseline
Anchor Window Anchor ±8units Native distance Distance-window control
NS-F ullQ Chunks overlapping
Γr(a, TX)F ull question q Internal ranking variant
NativeScope Same as above T arget term B T arget-ranking implementation
Dense RAG does not receive the gold anchor coordinate, so comparison with scoped methods
measures the value of both structural input and its use. NS-F ullQ holds the candidate scope
fixed and isolates ranking with qversus B.
4.3 Primary metric and statistics
LetGibe the gold native-unit set for question i, and let Hibe the union of character intervals
from returned chunks. W e define
Ri=1
|Gi|∑
u∈Gi1[I(u)⊆Hi], Recall@1024 =1
NN∑
i=1Ri. (6)
The metric requires complete coverage of each gold native unit; returning only an answer span
or part of a paragraph is insuﬀicient. Results are macro-averaged over questions rather than
weighted by the number of evidence units. A long dialogue turn may require multiple chunks
for full coverage, making this metric stricter than answer-string matching. It is neither QA
accuracy nor an oﬀicial score from the source benchmarks.
F or paired method differences on the same questions, we perform 10,000 paired bootstrap
replications within each domain. Using NumPy’s default_rng with seed 20260929, each repli-
cate independently samples with replacement 50 belonging, 25 before, and 25 after questions,
recomputes the mean paired difference, and enters an ordinary 2.5/97.5 percentile interval.
Relation-specific descriptions accompany aggregate results. Questions are the resampling units;
intervals are not clustered by source paper or history , and no multiple-comparison correction is
applied.
6

4.4 Post-observation revision and result reuse
The original frozen protocol used BGE T op-1 automatic anchoring. After observing localization
errors and severe memory-domain degradation, we revised the main question to the conditional
setting in which a correct anchor is supplied. This post-observation analysis uses all 200 records
and leaves the questions, decompositions, evidence, encoder, chunking, and budget unchanged.
The original automatic-localization results remain in Section 5.3 as an applicability boundary .
Existing outputs are reused only when their inputs and ranking rule match the revised con-
dition; all other anchor-dependent results are recomputed. This decision depends on anchor
correctness, not retrieval performance. Exact reuse counts, hashes, and method-specific prove-
nance appear in Appendix A. The revision isolates relational retrieval under a correct anchor
but does not convert the analysis into a preregistered test; the earlier chunking, mechanism, and
generative-QA runs remain outside the main evidence.
5 Results
5.1 Main results with correct anchors
T able 2reports results over all base records. NativeScope reaches 89.28% in documents and
72.50% in memories, improving over Dense RAG by 42.75 and 22.00 percentage points. Both
paired intervals in T able 3lie above zero, supporting the retrieval value of relational scoping on
this controlled dataset under correct anchors and a fixed budget, rather than an unrestricted
end-to-end RAG advantage.
T able 2: Recall@1024 with correct anchors (percent). Each domain has N= 100 under fixed
256-token chunking. Bold marks only the largest point estimate.
Method Document Memory
Dense RAG 46.53 50.50
Anchor Window 45.36 2.00
NS-F ullQ (internal variant) 87.78 68.50
NativeScope 89.28 72.50
Figure 3places all four results on a common scale. Both relation-localized implementations
substantially exceed instance-wide Dense RAG, while NativeScope and NS-F ullQ are close. The
main evidence therefore points to candidate scoping rather than the necessity of B-only ranking.
T able 3: Paired NativeScope-minus-control differences and 95% intervals, in percentage points.
Bold marks comparisons whose interval lies entirely above zero; no advantage is claimed when
an interval includes zero.
Domain Control Difference 95% interval
Document Dense RAG +42 .75 [33 .25,52.50]
Document Anchor Window +43 .92 [34 .75,53.34]
Document NS-F ullQ +1.50 [−2.00,5.50]
Memory Dense RAG +22 .00 [12 .00,32.00]
Memory Anchor Window +70 .50 [61 .50,79.00]
Memory NS-F ullQ +4.00 [−3.00,11.00]
The memory-domain window control reaches 2.00%. As specified in the protocol, it prioritizes
nearby units even though constructed targets begin four units away , and long turns can exhaust
7

Dense RAGAnchor WindowNS-FullQNativeScope020406080100Gold native-unit Recall@1024 (%)46.53 45.3687.7889.28Document domain (QASPER-derived)
Dense RAGAnchor WindowNS-FullQNativeScope50.50
2.0068.5072.50Memory domain (LongMemEval-derived)Figure 3: T wo-domain main results under correct anchors ( N= 100 per domain). The vertical
axis is gold native-unit Recall@1024; all methods use the same chunks, encoder, and return
budget.
the budget before the target is covered. The comparison therefore shows that relation-aware
scoping is preferable to this fixed distance policy under the present design.
5.2 Relation types and the internal ranking variant
NS-F ullQ and NativeScope share anchors, native scopes, and candidate chunks on all 200 ques-
tions. Rankings differ on 171 records and returned chunk identifiers on 165, so the implementa-
tions are behaviorally distinct. The aggregate benefit of B overqis nevertheless only 1.50 and
4.00 points, and both intervals include zero. The present results support relational scoping but
neither an independent advantage from B-only ranking nor statistical equivalence between the
two variants.
T able 4: Recall@1024 by relation under correct anchors (percent). These groups partition the
main evaluation. Bold marks the largest point estimate in each row; ties are included.
Domain Relation N Dense Window NS-F ullQ NativeScope
Document belonging 50 46.06 53.56 86.06 92.06
Document before 25 55.00 37.00 92.00 86.00
Document after 25 39.00 37.33 87.00 87.00
Memory belonging 50 39.00 2.00 53.00 61.00
Memory before 25 60.00 4.00 80.00 84.00
Memory after 25 64.00 0.00 88.00 84.00
NativeScope’s point estimate exceeds Dense RAG in all six groups, but this does not establish
a significant improvement in every group: the memory before and after intervals are [0,48] and
[0,40] percentage points. The relative ranking input also varies by relation: B is six points below
the full question for document before and four points below for memory after. These results
favor a relation-localized framework with a selectable ranking input rather than a requirement
to discard the full question.
5.3 Degradation under automatic localization
T able 5preserves the original automatic-localization results for default chunking. T op-1 anchor
accuracy is only 61% in documents and 37% in memories; the proportions retaining all gold
targets in the relation scope are 74% and 53%. Dense RAG does not consume the predicted
8

Given correct anchor Automatic T op-1 anchor020406080100Gold native-unit Recall@1024 (%)
87.78
65.4489.28
66.94Document
Given correct anchor Automatic T op-1 anchor
68.50
33.5072.50
35.50MemoryNS-FullQ NativeScopeFigure 4: Effect of the anchor condition on relation-localized recall. Correct-anchor and original
automatic-localization results share default chunking and Recall@1024, while the main condi-
tional evaluation was introduced after the automatic degradation was observed.
anchor; memory-domain NativeScope reaches 35.50%, below Dense RAG at 50.50%, because an
upstream coordinate error can become a hard candidate exclusion.
T able 5: Default-chunking results under the original automatic-localization protocol (percent).
Localization and scope coverage are diagnostics for the relation-localized pipeline. Within the
retrieval-method rows, bold marks the largest recall in each domain.
Metric/method Document Memory
Anchor T op-1 accuracy 61.00 37.00
All gold targets in relation scope 74.00 53.00
Dense RAG recall 46.53 50.50
Anchor Window recall 43.69 3.00
NS-F ullQ recall 65.44 33.50
NativeScope recall 66.94 35.50
Figure 4directly compares the two shared-scope implementations under correct and auto-
matic anchors. The document domain drops by about 22 points, and the memory domain by
more than 33, making anchor reliability a central condition for safe hard scoping.
On memory questions under automatic localization, NativeScope records 14 wins, 57 ties,
and 29 losses relative to Dense RAG. T wenty-six of the losses have an incorrect anchor, and
25 exclude at least one gold target from the relation scope. One representative error maps a
generic assistant opening to a similar turn in another session, thereby excluding the target before
ranking begins. The pattern links most losses to the localization interface, although it does not
assign a separate causal mechanism to every record.
6 Discussion and Limitations
The experiments support a simple division of labor: native relations determine where retrieval
should search, and semantic similarity orders the candidates inside that scope. Under correct
anchors, this interface substantially improves complete-evidence recall in both domains. The
small and uncertain difference between NativeScope and NS-F ullQ locates the contribution in
executable relational scoping, while leaving the best within-scope ranking query open.
Three limitations determine the scope of this evidence. First, the correct-anchor condition
was introduced after automatic localization failed, making the main result conditional and post-
9

observation. Second, each domain contains 100 agent-assisted controlled queries without inde-
pendent human reannotation or fully natural external-query evaluation; literal phrase matching,
minimum anchor distance, and complete-unit recall all shape the task. Third, the comparison
set omits stronger lexical retrievers, rerankers, and external structured-retrieval systems. The
study consequently establishes the utility of native relations in this controlled interface, not their
necessity or global optimality for question answering.
The correct-anchor evaluation covers only the default chunking. Earlier chunking and mech-
anism runs use a different anchor condition and are retained as exploratory records rather than
combined with the main analysis. Generative QA stopped after 36 of 800 planned responses,
and latency , memory , and API cost were not evaluated. These omissions leave cross-chunking
robustness, end-to-end answer quality , and systems eﬀiciency for future work.
In deployment, reliable coordinates may come from user selection, stable record identifiers, or
a validated upstream locator. When only unreliable automatic localization is available, calibra-
tion, verification, or fallback is necessary because hard scoping can underperform instance-wide
retrieval. Errors in container metadata, cross-session events, and ambiguous relation parsing
remain outside the current interface guarantees.
7 Conclusion
NativeScope executes stored relations to restrict the candidate domain before semantic ranking.
On 200 controlled records with correct anchors, it improves native-unit recall over instance-wide
Dense RAG by 42.75 points for documents and 22.00 points for memories. F ull-question ranking
performs similarly within the same scope, while automatic-anchor failures sharply reduce recall.
The central result is therefore conditional but actionable: when a system can supply reliable
coordinates and native relations, executing that structure can materially improve retrieval under
a fixed context budget.
A Reproduction Information and Evidence Inventory
A.1 V ersions and data identity
Raw datasets are not redistributed with the derived package, and API keys do not enter the
paper or artifacts. QASPER is recorded under CC BY 4.0, LongMemEval cleaned under MIT,
and BGE weights under MIT. F ull raw-file SHA-256 values, derived-set hashes, and model-file
inventories appear in experiments/E/data_manifest.json . Key identities are:
QASPER revision: fdc9d8214fbab5dd782958601db4d678e6934a54 ; v0.3 test.
LongMemEval revision: 98d7416c24c778c2fee6e6f3006e7a073259d48f ; S cleaned only .
BGE revision: a5beb1e3e68b9ab74eb54cfd186867f64f240e1a .
Base 200-record SHA-256: 5de99299a64db82ef3b8aa5277745e93dc1693185d06b226
e452e84a8162b678 .
Native parsing retains paragraph or turn text, container identifiers, and global unit order. F rozen
records contain source-question identifiers, source revisions, candidate hashes, prompt versions,
and reviewer types. Derived data and audit logs are not equivalent to human gold annotation;
reusers should retain this provenance statement.
A.2 Run scope and result provenance
The original automatic-localization experiment contains 40 configurations and 4,000 question–
method results spanning two domains, four methods, and five chunkings. The mechanism
batch contains 102 configurations and 9,280 results, including ten-seed topology shuffling, full-
question gold-anchor diagnostics, three semantic conditions, a positive lexical-matching subset,
10

and role/reverse checks. Except for explicitly labeled gold-anchor diagnostics, these runs use
automatic localization. T able 6states how they contribute to the paper.
T able 6: Executed material and its evidential role. Completed runs do not establish all earlier
claims.
Batch Results Anchor condition Use in this paper
Original chunking main run 4,000 Automatic Default-chunking
boundary result; other
runs retained as
exploratory records
Original mechanism batch 9,280 Automatic and separate diagnostics Audit retained; not
evidence for mechanisms
under the revised
condition
Correct-anchor revision 800 Given correct coordinate Main evidence in
T ables 2–4
Generative QA 36 Original protocol, incomplete Paused; no method
comparison reported
The revision note is experiments/E/GIVEN_ANCHOR_REVISION.md . The independent result
directory experiments/E/results/given_anchor_v1/ contains per-record results, reuse prove-
nance, run status, and statistics. analysis.json stores unrounded means, paired differences,
and intervals; table percentages and percentage points are these values multiplied by 100 and
rounded. Across all four methods, the document domain’s 400 results comprise 322 reused and
78 recomputed, while the memory domain’s 400 comprise 274 reused and 126 recomputed; these
reused counts include the 100 Dense RAG results in each domain. The anchor intervention
replaces only the localization return value. NativeScope continues to rank with B and uses
unchanged scope and budget code.
A.3 V erification boundary
Data validation checks coordinates, containers, directional distances, evidence mappings, unique-
ness, duplicates, and quotas. Retrieval records preserve the native scope, candidate chunks,
ranking, and returned chunks so coverage and budget can be recomputed. Consistency checks
between full-question and B ranking must compare candidate sets rather than infer identical
behavior from similar recall. The original gold-anchor diagnostic ranks with the full question
and must be interpreted as NS-F ullQ when reproduced.
References
[1] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell W u, Sergey Edunov,
Danqi Chen, and W en tau Yih. Dense passage retrieval for open-domain question answer-
ing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing , pages 6769–6781, 2020.
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, F abio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K”uttler, Mike Lewis, W en tau Yih, Tim Rockt”aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems , volume 33, pages 9459–9474, 2020.
[3] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models
for open domain question answering. In Proceedings of the 16th Conference of the European
11

Chapter of the Association for Computational Linguistics: Main V olume , pages 874–880,
2021.
[4] Harsh T rivedi, Niranjan Balasubramanian, T ushar Khot, and Ashish Sabharwal. Interleav-
ing retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.
In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics ,
pages 10014–10037, 2023.
[5] Zhengbao Jiang, F rank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Y u, Yiming
Y ang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
pages 7969–7992, 2023.
[6] Akari Asai, Zeqi W u, Yizhong W ang, A virup Sil, and Hannaneh Hajishirzi. Self-RAG:
Learning to retrieve, generate, and critique through self-reflection. In The Twelfth Interna-
tional Conference on Learning Representations , 2024.
[7] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, F abio
Petroni, and Percy Liang. Lost in the middle: How language models use long contexts.
T ransactions of the Association for Computational Linguistics , 12:157–173, 2024.
[8] Qingfei Zhao, Ruobing W ang, Y ukuo Cen, Daren Zha, Shicheng T an, Y uxiao Dong, and
Jie T ang. LongRAG: A dual-perspective retrieval-augmented generation paradigm for long-
context question answering. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing , pages 22600–22632, 2024.
[9] Y e Liu, Kazuma Hashimoto, Yingbo Zhou, Semih Y avuz, Caiming Xiong, and Philip Y u.
Dense hierarchical retrieval for open-domain question answering. In Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2021 , pages 188–200, 2021.
[10] Parth Sarthi, Salman Abdullah, Aditi T uli, Shubh Khanna, Anna Goldie, and Christo-
pher D. Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval.
In The Twelfth International Conference on Learning Representations , 2024.
[11] Darren Edge, Ha T rinh, Newman Cheng, Joshua Bradley , Alex Chao, Apurva Mody , Steven
T ruitt, and Jonathan Larson. F rom local to global: A graph RAG approach to query-focused
summarization. arXiv preprint arXiv:2404.16130 , 2024.
[12] W ensheng Lu, Keyu Chen, Zhifeng Shen, Ruizhi Qiao, and Xing Sun. Hichunk: Evaluating
and enhancing retrieval augmented generation with hierarchical chunking. In Proceedings
of the 64th Annual Meeting of the Association for Computational Linguistics , pages 29738–
29753, 2026.
[13] Huiyao Chen, Yi Y ang, Yinghui Li, Meishan Zhang, Baotian Hu, and Min Zhang. Beyond
chunking: Discourse-aware hierarchical retrieval for long document question answering. In
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics ,
pages 18176–18198, 2026.
[14] T ong Chen, Hongwei W ang, Sihao Chen, W enhao Y u, Kaixin Ma, Xinran Zhao, Hongming
Zhang, and Dong Y u. Dense x retrieval: What retrieval granularity should we use? In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 15159–15177, 2024.
[15] Michael Günther, Isabelle Mohr, Daniel James Williams, Bo W ang, and Han Xiao. Late
chunking: Contextual chunk embeddings using long-context embedding models. arXiv
preprint arXiv:2409.04701 , 2024.
12

[16] Nathanaël Langlois. SCAR: Semantic continuity-aware retrieval for eﬀicient context expan-
sion in RAG. arXiv preprint arXiv:2606.16661 , 2026. Concurrent preprint.
[17] Kelong Mao, Chenlong Deng, Haonan Chen, F engran Mo, Zheng Liu, T etsuya Sakai, and
Zhicheng Dou. ChatRetriever: Adapting large language models for generalized and robust
conversational dense retrieval. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing , pages 1227–1240, 2024.
[18] Nirmal Roy , Leonardo F. R. Ribeiro, Rexhina Blloshmi, and Kevin Small. Learning when
to retrieve, what to rewrite, and how to respond in conversational QA. In Findings of the
Association for Computational Linguistics: EMNLP 2024 , pages 10604–10625, 2024.
[19] Hassan Shavarani and Anoop Sarkar. Entity retrieval for answering entity-centric questions.
In Proceedings of the 4th International W orkshop on Knowledge-Augmented Methods for
Natural Language Processing , pages 1–17, 2025.
[20] Charles Packer, Sarah W ooders, Kevin Lin, Vivian F ang, Shishir G. Patil, Ion Stoica,
and Joseph E. Gonzalez. MemGPT: T owards LLMs as operating systems. arXiv preprint
arXiv:2310.08560 , 2023.
[21] W anjun Zhong, Lianghong Guo, Qiqi Gao, He Y e, and Y anlin W ang. MemoryBank: En-
hancing large language models with long-term memory . In Proceedings of the AAAI Con-
ference on Artificial Intel ligence , volume 38, pages 19724–19731, 2024.
[22] Zihe Y e, Jingyuan Huang, W eixin Chen, and Y ongfeng Zhang. H-mem: Hybrid multi-
dimensional memory management for long-context conversational agents. In Proceedings
of the 19th Conference of the European Chapter of the Association for Computational
Linguistics , pages 7756–7775, 2026.
[23] Zihao T ang, Xin Y u, Ziyu Xiao, Zengxuan W en, Zelin Li, Jiaxi Zhou, Hualei W ang, Haohua
W ang, Haizhen Huang, W eiwei Deng, F eng Sun, and Qi Zhang. Mnemis: Dual-route
retrieval on hierarchical graphs for long-term llm memory . In Proceedings of the 64th Annual
Meeting of the Association for Computational Linguistics , pages 23914–23928, 2026.
[24] Kaixiang W ang, Yidan Lin, Jiong Lou, Zhaojiacheng Zhou, Bunyod Suvonov, and Jie Li.
E-mem: Multi-agent based episodic context reconstruction for llm agent memory . arXiv
preprint arXiv:2601.21714 , 2026.
[25] Changjian W ang, Rongzhen Li, W eili Guan, Shuming Shi, Quan Lu, and Ning Jiang. Cue-
Mem: Cue-guided context reconstruction for long-term conversational memory . arXiv
preprint arXiv:2609.12354 , 2026. Concurrent preprint.
[26] Pradeep Dasigi, Kyle Lo, Iz Beltagy , Arman Cohan, Noah A. Smith, and Matt Gardner.
A dataset of information-seeking questions and answers anchored in research papers. In
Proceedings of the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language T echnologies , pages 4599–4610, 2021.
[27] Di W u, Hongwei W ang, W enhao Y u, Y uwei Zhang, Kai-W ei Chang, and Dong Y u. Long-
MemEval: Benchmarking chat assistants on long-term interactive memory . In The Thir-
teenth International Conference on Learning Representations , 2025.
[28] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-Pack: Packaged re-
sources to advance general chinese embedding. arXiv preprint arXiv:2309.07597 , 2023.
13
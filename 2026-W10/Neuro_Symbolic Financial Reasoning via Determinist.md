# Neuro-Symbolic Financial Reasoning via Deterministic Fact Ledgers and Adversarial Low-Latency Hallucination Detector

**Authors**: Pedram Agand

**Published**: 2026-03-04 22:55:16

**PDF URL**: [https://arxiv.org/pdf/2603.04663v1](https://arxiv.org/pdf/2603.04663v1)

## Abstract
Standard Retrieval-Augmented Generation (RAG) architectures fail in high-stakes financial domains due to two fundamental limitations: the inherent arithmetic incompetence of Large Language Models (LLMs) and the distributional semantic conflation of dense vector retrieval (e.g., mapping ``Net Income'' to ``Net Sales'' due to contextual proximity). In deterministic domains, a 99% accuracy rate yields 0% operational trust. To achieve zero-hallucination financial reasoning, we introduce the Verifiable Numerical Reasoning Agent (VeNRA). VeNRA shifts the RAG paradigm from retrieving probabilistic text to retrieving deterministic variables via a strictly typed Universal Fact Ledger (UFL), mathematically bounded by a novel Double-Lock Grounding algorithm. Recognizing that upstream parsing anomalies inevitably occur, we introduce the VeNRA Sentinel: a 3-billion parameter SLM trained to forensically audit Python execution traces with only one token test budget. To train this model, we avoid traditional generative hallucination datasets in favor of Adversarial Simulation, programmatically sabotaging golden financial records to simulate production-level ``Ecological Errors'' (e.g., Logic Code Lies and Numeric Neighbor Traps). Finally, to optimize the Sentinel under strict latency budgets, we utilize a single-pass classification paradigm with optional post thinking for debug. We identify the phenomenon of Loss Dilution in Reverse-Chain-of-Thought training and present a novel, OOM-safe Micro-Chunking loss algorithm to stabilize gradients under extreme differential penalization.

## Full Text


<!-- PDF content starts -->

Neuro-Symbolic Financial Reasoning via
Deterministic Fact Ledgers and Adversarial
Low-Latency Hallucination Detector
Pedram Agand∗
FactAI Lab,
Vancouver, BC, Canada
pagand@gofactai.com
Abstract
Standard Retrieval-Augmented Generation (RAG) architectures fail in high-stakes
ﬁnancial domains due to two fundamental limitations: the inherent arithmetic in-
competence of Large Language Models (LLMs) and the distributional semantic
conﬂation of dense vector retrieval (e.g., mapping “Net Income” to “Net Sales”
due to contextual proximity). In deterministic domains, a 99% accuracy rate yields
0% operational trust. To achieve zero-hallucination ﬁnancial reasoning, we intro-
duce the Veriﬁable Numerical Reasoning Agent (VeNRA). VeNRA shifts the RAG
paradigm from retrieving probabilistic text to retrieving deterministic variables via
a strictly typed Universal Fact Ledger (UFL), mathematically bounded by a novel
Double-Lock Grounding algorithm. Recognizing that upstream parsing anomalies
inevitably occur, we introduce the VeNRA Sentinel: a 3-billion parameter SLM
trained to forensically audit Python execution traces with only one token test bud-
get. To train this model, we avoid traditional generative hallucination datasets
in favor of Adversarial Simulation , programmatically sabotaging golden ﬁnancial
records to simulate production-level “Ecological Errors” (e.g., Logic Code Lies
and Numeric Neighbor Traps). Finally, to optimize the Sentinel under strict la-
tency budgets, we utilize a single-pass classiﬁcation paradigm with optional post
thinking for debug. We identify the phenomenon of Loss Dilution in Reverse-
Chain-of-Thought training and present a novel, OOM-safe Micro-Chunking loss
algorithm to stabilize gradients under extreme differential penalization.
1 Introduction
Retrieval-Augmented Generation (RAG) has become the standard for open-domain Question An-
swering (QA) and is widely regarded as the leading paradigm for grounding AI outputs in source
material Dadopoulos et al. [2025 ]. However, its deployment in high-stakes, deterministic domains,
speciﬁcally ﬁnancial auditing, remains stalled by a phenomenon we term Stochastic Inaccuracy . In
these environments, an accuracy rate of 99% yields 0% operational trust if the 1% error constitutes
a sign-convention inversion on a balance sheet or a temporal hallucination in a compliance report.
In fact, as PHANTOM demonstrates, ﬁnancial QA demands exact numerical precision; even minor
errors can critically undermine trust Ji et al. [2025 ].
This friction arises from two fundamental architectural mismatches. First, dense vector retrieval
models rely on distributional semantics . Consequently, they frequently map mathematically oppo-
site terms (e.g., “Net Income” vs. “Net Loss”) to proximate vectors due to their identical linguistic
contexts, leading to catastrophic retrieval conﬂation. Basic dense retrieval often fails to fetch the
correct evidence in ﬁnancial documents Dadopoulos et al. [2025 ], and general-purpose embeddings
∗Research conducted in FactAI Lab available at https://gofactai.comarXiv:2603.04663v1  [cs.LG]  4 Mar 2026

Figure 1: The VeNRA Neuro-Symbolic Paradigm. (Top) Standard RAG suffers from “Stochastic
Inaccuracy” due to probabilistic arithmetic and vector conﬂation. (Bottom) VeNRA parses PDFs
into a Universal Fact Ledger (UFL). A Query Navigator retrieves speciﬁc variables via a Lexical
Gate. An Architect LLM generates a deterministic Python trace ( T). Crucially, the 3B Sentinel
model forensically audits the logic trace before surfacing the answer.
struggle with ﬁnance-speciﬁc terms (e.g., “short” vs. “long”) Anderson and Koudas [2024 ], con-
sistent with our observation of conﬂating semantically opposite terms. Second, Large Language
Models (LLMs) operate as probabilistic next-token predictors, not arithmetic engines. When tasked
with ﬁnancial reasoning, they simulate the syntax of calculation without preserving its mathemati-
cal invariants, resulting in plausible-sounding but factually baseless ﬁgures Wang et al. [2025 ]. As
noted in recent work, LLMs inherently struggle with tasks requiring exact computation Chang et al.
[2025 ], producing outputs that sound plausible but violate arithmetic invariants.
Current methodologies to mitigate these hallucinations rely on two fragile pillars: parametric scal-
ing (utilizing frontier models with higher parameters) and generative evaluation benchmarks (e.g.,
HaluEval). Both are insufﬁcient for production ﬁnancial systems. Frontier models introduce pro-
hibitive latency ( >2000 ms), rendering them unusable for real-time guardrails. In practice, GPT-4
with retrieval often fails on ﬁnance questions (81% error/refusal rate) Islam et al. [2023 ], and feed-
ing longer contexts only yields marginal gains with unacceptable latency Islam et al. [2023 ]. More
critically, current evaluation benchmarks rely almost exclusively on Generative Noise , prompting
an LLM to invent an error. For example, the HaluEval benchmark generates false answers via an
LLM perturbation framework Li et al. [2023 ], producing semantically obvious, structurally lazy lies
(“Strawman Hallucinations”). Real-world ﬁnancial agents do not fail by inventing ﬁctitious corpo-
rate acquisitions; they fail via Ecological Errors : precise mechanical failures such as selecting an
adjacent temporal column in a table (Numeric Neighbor Traps) or executing correct Python logic
on incorrect variable extractions (Logic Code Lies). Indeed, ﬁnancial hallucinations often follow
speciﬁc patterns (numerical miscalculations, temporal shifts, etc.) Tan et al. [2025 ], which generic
benchmarks do not capture.
In this paper, we argue that reliable ﬁnancial AI cannot be achieved purely through parametric scal-
ing or generative evaluation. Instead, it requires Neuro-Symbolic Cognitive Ofﬂoading andForensic
Auditing . We introduce the VeNRA2(Veriﬁable Numerical Reasoning Agent) framework, an end-
to-end ecosystem designed to prevent, simulate, and detect ﬁnancial hallucinations. To mitigate
hallucinations, VeNRA replaces unstructured text retrieval with the Universal Fact Ledger (UFL).
By strictly extracting text into typed variables backed by Double-Lock Grounding (requiring both
character-offset alignment and semantic schema validation), and gating semantic retrieval with deter-
ministic lexical constraints, we relegate the LLM entirely to the role of a deterministic Python code
architect. To detect slipped hallucinations, we introduce the VeNRA Sentinel , a locally deployable
3-B parameter SLM capable of auditing complex mathematical traces in <50ms, zero-shot, zero
2The code is available in Github: https://github.com/pagand/VeNRA
2

test-time compute. To train the Sentinel, we pioneer Adversarial Simulation , rejecting generative
noise in favor of VeNRA-Data : a 10,000-sample dataset constructed by programmatically sabotag-
ing golden ﬁnancial records to recreate the exact mechanical failures of production RAG systems.
Tooptimize the Sentinel for zero-latency inference, we adopt a “Label-First” generation paradigm.
We identify the phenomenon of Loss Dilution in Reverse-Chain-of-Thought training, where the gra-
dient signal of the verdict token is overwhelmed by the subsequent reasoning tokens. To solve this,
and to prevent Vocabulary Out-Of-Memory (OOM) errors caused by massive differential weighting,
we present a novel Micro-Chunking Trainer . This algorithm bounds VRAM usage to O(c·|V|)
rather than sequence length, enabling stable training of highly discriminative judges on consumer
hardware. In summary, our main contributions are:
•Neuro-Symbolic Architecture: The Universal Fact Ledger (UFL) with Double-Lock
Grounding and Lexical Pre-Filtering, virtually eliminating extraction hallucination and dis-
tributional vector conﬂation.
•Adversarial Simulation (VeNRA-Data)3:A novel methodology for generating rigorous
hallucination benchmarks via deterministic sabotage (e.g., Logic Code Lies), evaluated via
a strict Hybrid-Family splitting protocol to prevent dataset leakage.
•Forensic Auditing Systems Engineering4:The VeNRA Sentinel, a 3B parameter judge
achieving low latency veriﬁcation. We contribute the OOM-safe Micro-Chunking Trainer,
mathematically solving the gradient dilution problem in Reverse-CoT architectures.
2 Background and Related Work
Neural RAG has become the standard architecture for knowledge-intensive tasks Lewis et al. [2020 ].
However, its application in high-stakes, deterministic domains such as ﬁnancial auditing reveals sys-
tematic mismatches between probabilistic language modeling and strict numerical reasoning. While
existing literature addresses ﬁnancial QA benchmarks, hallucination detection, and neuro-symbolic
execution in isolation, VeNRA integrates these streams to address the speciﬁc failure modes of pro-
duction ﬁnancial systems.
2.1 Failures of Dense Retrieval in Finance
Classical RAG architectures rely on dense vector retrieval (e.g., DPR, ColBERT) to map queries
to evidence Karpukhin et al. [2020 ],Khattab and Zaharia [2020 ]. While effective for encyclopedic
semantic matching, these models suffer from distributional conﬂation in ﬁnancial contexts. Embed-
dings optimized for semantic similarity frequently cluster mathematically opposite terms (e.g., “A
owes B” vs. “B owes A”) due to identical linguistic contexts Thakur et al. [2021 ]. Furthermore, re-
cent studies on FinanceBench demonstrate that generic embedding models struggle with the precise
temporal and entity alignment required for 10-K ﬁlings, where the answer often hinges on distin-
guishing “2022” from “2021” within dense tabular structures Islam et al. [2023 ]. VeNRA addresses
this by shifting from pure semantic retrieval to a hybrid Lexical-Semantic Gate, prioritizing deter-
ministic token overlap for discriminative ﬁnancial variables.
2.2 Numerical Reasoning and Neuro-Symbolic Agents
Large Language Models (LLMs) act as probabilistic next-token predictors, often failing to maintain
arithmetic invariants during text generation Patel et al. [2021 ]. To mitigate this, recent work has
pivoted toward neuro-symbolic methods. Chain-of-Thought (CoT) prompting Wei et al. [2022 ] im-
proves reasoning but remains prone to calculation errors. Program-Aided Language Models (PAL)
andProgram of Thoughts (PoT) advance this by delegating computation to an external Python inter-
preter Gao et al. [2023 ],Chen et al. [2022 ].
However, benchmarks like FinQA Chen et al. [2021 ] and TAT-QA Zhu et al. [2021 ] reveal that even
code-generating models struggle with grounding ; frequently executing correct logic on hallucinated
input variables. VeNRA extends the PAL paradigm by enforcing a stricter separation of concerns:
3https://huggingface.co/datasets/pagand/venra
4https://huggingface.co/spaces/pagand/VeNRA_halDet
3

the LLM acts solely as a Code Architect operating over a strictly typed Universal Fact Ledger (UFL),
preventing the injection of unsupported scalars into the computation graph.
2.3 Hallucination Benchmarks: Generative vs. Ecological
The evaluation of hallucinations has largely relied on datasets generated via LLM perturbation.
Benchmarks like HaluEval Li et al. [2023 ] and TruthfulQA Lin et al. [2022a ] use strong models to
rewrite factual sentences into plausible falsehoods. While scalable, this method produces “Semantic
Noise”, hallucinations that are often linguistically distinct from the source. In contrast, ﬁnancial
hallucinations are often mechanical : a column shift in a table or a unit swap (e.g., millions to bil-
lions). These “Ecological Errors” are underrepresented in generative benchmarks. New benchmarks
like FinTextQA and FINDER have been proposed to capture more realistic, expert-level ﬁnancial
queries Dadopoulos et al. [2025 ],Chen et al. [2024 ],Choi and Kim [2025 ], yet they focus primar-
ily on QA accuracy rather than auditing execution traces. VeNRA-Data diverges from prior work
by using Adversarial Simulation , deterministic programmatic sabotage of execution traces, rather
than generative rewriting, aligning closer to Red Teaming methodologies Ganguli et al. [2022 ] than
traditional QA dataset creation.
2.4 LLM-as-a-Judge and Low-Latency Auditing
Using LLMs as judges to evaluate RAG outputs is a burgeoning ﬁeld Zheng et al. [2024 ]. However,
most “Judge” architectures rely on frontier models (e.g., GPT-4) which incur multi-second laten-
cies, associated cost, privacy concern, inconsistency, prompt sensitivity, bias, etc rendering them
unsuitable for real-time guardrails Kim et al. [2023 ]. Furthermore, standard instruction tuning for
these judges optimizes for ﬂuent rationales rather than calibrated classiﬁcation, leading to “syco-
phancy” Sharma et al. [2023 ]. VeNRAs Sentinel model contributes to the growing literature on
efﬁcient Small Language Models (SLMs) Gunasekar et al. [2023 ] and single-pass classiﬁcation. By
utilizing a “Label-First” paradigm to tether gradients and a novel Micro-Chunking loss to manage
vocabulary memory, we demonstrate that a 3B parameter model can achieve frontier-level auditing
performance with single token test-time budget, overcoming the latency-accuracy trade-off inherent
in prior LLM-as-a-Judge work.
2.5 The Long-Context Fallacy
Finally, VeNRA challenges the assumption that expanding context windows (e.g., to 128k tokens)
solves retrieval challenges. Consistent with the “lost-in-the-middle” phenomenon, LLMs often over-
look information buried deep in long contexts Liu et al. [2024 ], which is critical for footnotes in
10-K ﬁlings. VeNRAs Double-Lock Grounding andPrompt Bookending strategies explicitly miti-
gate this by structuring the context via the UFL rather than relying on the models raw attention span
over unstructured text.
3 The VeNRA Architecture: Grounded Neuro-Symbolic RAG
Standard RAG architectures model high-stakes QA as a purely probabilistic sequence generation
task. In deterministic domains such as ﬁnance, this paradigm induces stochastic inaccuracy : LLMs
simulate the syntax of mathematics rather than computing it, and dense vector embeddings fre-
quently conﬂate mathematically opposite but contextually proximate terms (e.g., “Deferred Rev-
enue” vs. “Realized Revenue”).
To mitigate these fundamental ﬂaws, we introduce VeNRA to shift the RAG paradigm from “Retriev-
ing Text” to “Retrieving Variables”. Rather than forcing a single LLM to simultaneously act as a
semantic search engine, an unstructured parser, and an arithmetic calculator, VeNRA strictly decou-
ples these cognitive burdens. Unstructured documents are deterministically parsed into a strongly
typed Universal Fact Ledger (UFL), mathematically grounded, and subsequently queried by an LLM
constrained strictly to Python code generation.
4

3.1 Structure-Preserving Ingestion and the Universal Fact Ledger
The Universal Fact Ledger (UFL) serves as the foundational data contract for the VeNRA ecosystem,
acting simultaneously as a computational table for arithmetic and a relational graph for semantic
traversal. To populate the UFL from unstructured 10-K ﬁlings without succumbing to “Context
Rot” Hong et al. [2025 ], the loss of structural lineage and coreference, we employ a deterministic
ingestion pipeline.
Trailing-Buffer Semantic Chunking. Standard recursive chunking mechanisms frequently sever
cross-boundary coreferences (e.g., a table row referencing “See Note 12”, where the note text resides
in the subsequent chunk). To preserve localized context, we developed a newline-preference trailing-
buffer algorithm. Let Cibe the i-th chunk of text, bounded by a character threshold τ= 3000 .
Rather than splitting at an arbitrary character index, the algorithm splits at the most recent newline
character. For every generated text block Ci, a trailing buffer bi−1consisting of the ﬁnal k= 300
characters of the preceding block Ci−1is injected as a preﬁx:
C′
i= [Previous Context: bi−1]⊕Ci (1)
Crucially, table structures also update this rolling buffer. Because ﬁnancial table headers carry es-
sential scalar context (e.g., “in millions”), propagating the table’s raw tail into the subsequent prose
chunk allows the extraction model to successfully resolve anaphoric scalar references.
Deterministic Table Melting. Generative LLMs exhibit severe unreliability when extracting 2D
grid data Zhu et al. [2021 ]. Consequently, VeNRA bypasses LLM extraction for tables entirely, em-
ploying deterministic Pandas routines. Raw markdown tables are parsed using a hierarchical disam-
biguation stack. Indentation markers (e.g., &nbsp; ) are computed to generate fully qualiﬁed schema
paths (e.g., Assets > Current Assets > Inventory ), preventing the collapse of distinct sub-categories
into ambiguous metrics like “Other.” Furthermore, row-level overrides dynamically intercept global
column scale factors; metrics containing substrings such as “per share” enforce a 1.0multiplier and
a unit classiﬁcation of USD/Share , ensuring mathematical invariants are preserved.
3.2 Double-Lock Grounding and Post-Hoc Alignment
The primary vulnerability of JSON-constrained LLM extraction is the generation of unsupported
numerical values. To solve the “Generator vs. Extractor” dilemma, we force the generative model to
provide a grounding_quote (q), the verbatim substring from the source chunk ( C) that justiﬁes the
extracted scalar ( v). Every extracted fact undergoes a rigorous Double-Lock Grounding protocol.
Lock 1: Mechanical Quote Grounding. The algorithm attempts to physically map the quote qto
exact character offsets in C. Standard normalized sequence matching algorithms (e.g., difflib )
over entire text chunks were discarded; we observed that sequence matching against a large denom-
inator (e.g., a 3000-character chunk) asymptotically approaches 0, rendering it empirically useless
for isolated numeric veriﬁcation. Instead, we use a three-tier fallback:
•Tier 1 (Exact Match): AnO(n)character-exact substring search, followed by a whitespace-
normalized pass (collapsing \s+to a single space) to account for LLM-induced spacing
artifacts.
•Tier 2 (Partial Match): The aligner extracts the longest contiguous numeric token qnum
fromq(e.g., “615” from “$615 million”) and searches for it verbatim, verifying the scalar
value independently of surrounding text hallucination.
•Tier 3 (Fuzzy Sliding-Window): A sliding window Wof tokens is passed over C. We
deliberately avoid Jaccard Similarity, which heavily penalizes the intersection due to the
large union of the document window ( |W| ≫ | q|). Instead, we compute a fast token
intersection based on Recall: R=|q∩W|
|q|. IfR > 0.55, the window is reﬁned with a
localized character-matching algorithm.
If all three tiers fail, the fact is stamped UNALIGNED , conﬁdence is set to 0.0, and the scalar is
restricted from downstream execution.
Lock 2: Semantic Metric Grounding. Mechanical grounding prevents invented numbers, but it
does not prevent “Phantom Metrics”, a hallucination where an LLM assigns a veriﬁably real number
5

to a fabricated metric name. Let Mbe the set of core tokens in the generated metric name, excluding
general ﬁnancial stop-words (e.g., “net”, “total”, “ratio”). Let TCbe the set of tokens in the source
chunk. The semantic overlap score is deﬁned as s=|M∩TC|
|M|. Ifs < τ sem(where τsem= 0.30),
the fact is rejected as a semantic hallucination, successfully preventing the model from assigning
orphaned numbers to hallucinatory schemas.
3.3 Schema-Driven Cognitive Ofﬂoading
A persistent challenge in applied LLM engineering is “prompt overﬁtting”, attempting to handle
inﬁnite edge cases via increasingly complex system instructions. In ﬁnance, edge cases include
dynamic formulas (e.g., “ﬂoating rate equal to SOFR plus 2.25%” ) and covenants (e.g., “leverage
ratio not exceeding 3.50” ). A naive prompt instructing the LLM to differentiate these leads to
downstream mathematical catastrophes when the LLM extracts “2.25” as a ﬁxed interest rate.
VeNRA introduces Schema-Driven Cognitive Ofﬂoading . Rather than relying on conditional logic
in the prompt, we leverage the LLM’s inherent pre-training in taxonomy by introducing a Pydantic
classiﬁcation enum: fact_type∈{ACTUAL, LIMIT, FORMULA} . Once the model classiﬁes the
fact via constrained decoding, deterministic Python post-processing handles the operational logic.
ForFORMULA types, the system forcefully nulliﬁes the ﬂoat ( num_value = None ). This creates a
critical safety boundary: when the Code Agent later queries the UFL, it encounters a None type and
reads the formula string in the text_nuance metadata, preventing it from executing arithmetic on
a partial ﬂoating rate. For LIMIT types, the system automatically appends the sufﬁx [Limit] to the
metric name, preserving the strict ontological boundary between an actual performance metric and
a legal constraint.
3.4 Hybrid Lexical-Semantic Retrieval
Once the Universal Fact Ledger (UFL) is populated, the system must retrieve the precise variables
required to satisfy a user query. Pure semantic vector search is fundamentally ill-suited for ﬁnancial
retrieval because dense embedding models (e.g., bge-m3 ) rely on distributional semantics. Conse-
quently, they frequently group mathematically opposite terms by contextual proximity. A query for
“Net Sales” will consistently retrieve chunks discussing “Net Income” with high cosine similarity,
leading to cascading arithmetic failures downstream.
To overcome this, VeNRA introduces a Hybrid Lexical-Semantic Retriever , governed by an LLM-
based Query Navigator and a deterministic Lexical Gate.
Query Navigation and Schema Mapping. Upon receiving a query, a Navigator SLM evaluates the
user intent against a dynamically injected schema summary, which maps canonical entity IDs and
high-frequency metric names. The Navigator outputs a bipartite RetrievalPlan containing: (1)
a structured UFL ﬁlter (e.g., Entity ID, Year, Metric Keywords, and Nuance conditions), and (2) a
“Vector Hypothesis”, an idealized text snippet theoretically containing the answer.
Lexical Pre-Filtering (The Semantic Gate). To prevent distributional semantic conﬂation, all vec-
tor retrieval candidates (and fuzzy UFL metric matches) must pass a deterministic Lexical Gate
before entering the context window.
LetQbe the multiset of normalized tokens in the query, and Cbe the multiset of tokens in the
candidate chunk. Standard Jaccard Similarity ( J=|Q∩C|
|Q∪C|) is strictly avoided; because ﬁnancial text
chunks are highly verbose ( |C|≫| Q|), the massive union denominator forces Jtoward zero even
for perfect semantic matches. Instead, we compute token intersection Recall:
Rlex=∑
t∈set(Q)min(#( t, Q),#(t, C))
|Q|(2)
where #(t, X)denotes the multiplicity of token tin multiset X. IfRlex< τlex(where τlex= 0.30),
the candidate is dropped.
Crucially, two pre-processing steps guarantee the gate’s efﬁcacy:
1.Domain-Speciﬁc Stop-Word Filtering: Standard ﬁnancial modiﬁers (e.g., “net”, “total”,
“per”) are strictly classiﬁed as non-discriminative stop-words and removed from Q. If
6

“net” were retained, a query for “Net Income” ( Q={net,income}) evaluated against a
candidate discussing “Net Sales” ( C={net,sales}) would yield Rlex= 0.50, falsely
passing the threshold. Filtering “net” isolates the discriminative tokens, yielding Rlex=
0.0and successfully blocking the semantic hallucination.
2.Preﬁx Stripping: The trailing context buffer ( [Previous Context: bi−1]) injected during
the chunking phase (Section 3.1) is forcefully stripped from Cprior to tokenization. This
prevents cross-chunk token contamination, ensuring that an irrelevant candidate does not
pass the gate simply because its injected preﬁx shares tokens with the query.
Relational Graph Expansion. Because the UFL acts as a relational graph, retrieval operates along
three concurrent expansion pathways:
1.Structured Expansion: The UFL DataFrame is ﬁltered sequentially. If a nuance_focus
is generated by the Navigator (e.g., “Restated”), it acts as a boolean AND-gate over the
text_nuance column, deterministically isolating adjusted ﬁgures.
2.Entity Pivoting: If retrieved UFL rows contain a related_entity_id (e.g., a supplier),
the system pivots and queries the vector database for chunks mentioning that speciﬁc entity,
using a relaxed lexical threshold ( τlex= 0.20) to account for sparse entity mentions in
prose.
3.Frequency-Based Chunking: To ensure the reasoning agent has access to the qualitative
text (the “data hub”) that generated the numbers, the system isolates the most frequent
source_chunk_id s from the retrieved UFL rows and forcefully includes their raw text
chunks in the context window.
3.5 Program-Aided Execution and Context Assembly
The ﬁnal architectural phase extends the Program-Aided Language Models (PAL) paradigm Gao
et al. [2023 ]. The LLM is stripped of its role as a probabilistic calculator and relegated to the role of
a deterministic software architect.
Context Assembly and Priority Scoring. To prevent “Context Rot”, where an LLM loses attention
over an expansive context window, the Context Assembler deduplicates all retrieved UFL rows and
text chunks. Chunks are ranked via a priority scoring heuristic: a chunk creceives +5points if it
is physically linked to a retrieved UFL row (via source_chunk_id ), and +1point for every BM25
keyword it contains. The context is strictly truncated to the top k= 5 chunks and presented as a
fused Markdown representation containing both the structured UFL table and the unstructured raw
text.
Two-Pass Deterministic Orchestration. VeNRA executes reasoning via a dual-pass neuro-
symbolic ﬂow:
•Pass 1 (The Architect): A reasoning model evaluates the assembled context. It generates
a step-by-step logic plan and writes deterministic Python code to compute the answer. The
code is executed in an isolated local Python subprocess. If a formula constraint was previ-
ously ofﬂoaded (Section 3.3) and the UFL num_value isNone , the code naturally halts or
explicitly requests the missing variable from the text, preventing blind arithmetic.
•Pass 2 (The Synthesizer): A low-latency SLM ingests the mathematical output from the
Python subprocess and formulates the ﬁnal natural language response, ensuring every stated
number is cited directly to a UFL row_id or Chunk ID.
Runtime Fallback Protocol: “Trust Text Over Data”. Despite rigorous ingestion grounding, op-
tical character recognition (OCR), failure in UFL underlying LLM, and PDF parsing anomalies in-
evitably occur. VeNRA implements a programmatic fault-tolerance mechanism. The Architect agent
is explicitly instructed via its system prompt to cross-reference the structured UFL rows against the
provided raw text chunks. If the agent detects that the UFL data is malformed (e.g., an unaligned
temporal column) or mathematically contradicts the raw text, it programmatically discards the UFL
variable. Instead, the Python code is written to extract and compute the values directly from the
provided source text string, guaranteeing system resilience.
7

4 VeNRA-Data: Adversarial Simulation of RAG Failures
To effectively train a low-latency Sentinel model to audit the outputs of the VeNRA architecture, we
require a dataset that accurately reﬂects the failure modes of ﬁnancial RAG systems. We introduce
VeNRA-Data , a heavily curated, 10,000-sample dataset comprising contrastive pairs of grounded
and ungrounded ﬁnancial reasoning traces.
To prevent the classiﬁcation model from becoming overly dependent on contextual presence, a com-
mon ﬂaw where models ﬂag true ﬁnancial axioms as hallucinations simply because they are omitted
from the immediate text, we avoid the binary True/False paradigm. Instead, VeNRA-Data adopts a
strict tripartite taxonomy: SUPPORTED (backed by text or valid computation), UNFOUNDED (contra-
dicts evidence, uses ungrounded inputs, or commits logic errors), and GENERAL (axiomatic ﬁnancial
truths independent of the provided context).
4.1 The Flaw in Generative Hallucinations
Current hallucination detection benchmarks (e.g., HaluEval) primarily rely on LLM-generated se-
mantic noise. Dataset creators typically prompt a frontier model to intentionally rewrite a factual
sentence into a hallucination. This methodology fundamentally lacks ecological validity . Genera-
tive hallucinations tend to be linguistically obvious, structurally lazy, and linguistically distinct from
the source text.
Financial RAG systems in production do not fail by inventing linguistically anomalous sentences;
they fail via precise mechanical, calculative, and mapping errors. An LLM extracting data from a
10-K is far more likely to select an adjacent temporal column (e.g., 2022 instead of 2023 ) or divide
by a hallucinated denominator than it is to invent a ﬁctitious corporate acquisition.
To overcome the limitations of generative noise, we shift the paradigm to Adversarial Simulation .
We utilize ﬁve high-quality, human-annotated ﬁnancial QA datasets including FinQA Chen et al.
[2021 ], TAT-QA Zhu et al. [2021 ], FinanceBench Islam et al. [2023 ], truthfulQA Lin et al. [2022b ],
and PHANTOM Ji et al. [2025 ] as our “Golden” baseline. Rather than asking an LLM to rewrite
them, we employ a deterministic Saboteur Engine to systematically mutate speciﬁc informational
pathways, algorithmically recreating the exact cognitive failures observed in production architec-
tures.
4.2 The Saboteur Engine: Programmatic Injections
The Saboteur Engine consists of modular, deterministic perturbation algorithms. For a given Golden
record containing a Query ( Q), Context ( C), execution Trace ( T), and Target Sentence ( S), the
engine attempts to generate an UNFOUNDED “Hard Negative” clone via one of four speciﬁc attack
vectors.
1. The Logic Code Lie. This vector targets records containing arithmetic traces (e.g., FinQA).
It simulates a scenario where the generated Python code is syntactically and mathematically valid,
but the input variables hallucinated by the agent are fundamentally incorrect. Let VTbe the set of
numeric scalars present in the execution trace T, andVCbe the set of numeric scalars present in the
context C. The engine isolates a target variable v∈VT(excluding structural indices such as 0or1).
It then samples a distractor scalar d∈{VC\VT}. The trace is mutated via functional substitution:
T′=T[v←d].
Crucially, the Saboteur then executes the hallucinated trace T′in a secure sandbox to produce a new
resultant scalar R′. The target sentence is updated to reﬂect R′(S←S′). The resulting record
presents a profound challenge to the Judge model: the sentence S′perfectly matches the output of
the trace T′, but the trace relies on an input dthat misaligns with the semantic text C.
2. The Numeric Neighbor Trap. This vector targets tabular extraction failures, simulating the “col-
umn/row shift” errors endemic to PDF optical character recognition (OCR) and LLM grid parsing.
Let the ﬁnancial table be represented as a 2D matrix M. The engine locates the ground-truth scalar
within the matrix at coordinates (i, j)such that Mi,j∈S. The algorithm deterministically shifts the
coordinate space to select a neighboring cell M′, where M′∈{Mi±1,j, Mi,j±1}.
8

By substituting M′intoS, the engine simulates two highly prevalent RAG failures: a Temporal Slip
(a horizontal column shift, retrieving the wrong ﬁscal year) or a Metric Slip (a vertical row shift,
retrieving an adjacent accounting category).
3. Irrelevancy and Time Warps. This vector simulates staleness and retrieval failure.
•The Time Warp: If the user query Qcontains a speciﬁc temporal bounding condition (e.g.,
the year 2020 ), the Saboteur mutates the query Q←Q2021. The context Cand the sentence
Sremain entirely unchanged. This trains the model to detect scenarios where the provided
answer is factually supported by the text, but fails to satisfy the temporal constraints of the
user’s prompt.
•Context Swap (Non-Existence): The engine retains QandS, but replaces the context C
entirely with text chunks sampled from a completely disjoint ﬁnancial document. This cre-
ates a “non-existence” hallucination, forcing the Sentinel model to detect when a generated
answer is entirely ungrounded by the provided retrieval context.
4. Semantic and Scale Drift. Finally, the engine applies deterministic regex-bounded substitutions
to simulate unit and entity mapping failures. Using a predeﬁned domain ontology, scale tokens are
inverted (e.g., “millions” ↔“billions”), and capitalized Named Entities are swapped with disjoint
entities found elsewhere in the text space (e.g., misattributing a subsidiary’s revenue to the parent
corporation).
4.3 The Teacher-Auditor Protocol and Noise Injection
While programmatic sabotage deterministically generates hard negatives, it is susceptible to “Acci-
dental Truths”, scenarios where an injected scalar coincidentally matches a valid contextual fact, or
where the original “Golden” record itself contains human annotation errors (e.g., “Dirty Gold”).
To guarantee the pristine quality of the training signals, VeNRA-Data employs a multi-tiered
Teacher-Auditor Protocol . The Auditor is tasked with independently evaluating the sabotaged tu-
ple(Q, C, T′, S′)to verify if it is genuinely unfounded.
Dual-Path Cause-Effect Validation. A critical algorithmic hurdle in automated auditing is the
decoupling of cause and effect. In a “Logic Code Lie” (Section 4.2), the Saboteur mutates an input
variable ( v←d). However, the Teacher model frequently ﬂags the output scalar (the ﬁnal computed
answer in S′) as the error span, rather than the injected input d. A naive string-matching validation
script falsely rejects these valid sabotages because the cause ( d) does not equal the effect ( S′).
We resolve this via a Dual-Path Validation algorithm. Let Eteacher be the error span detected by the
Teacher, and Θteacher be the internal reasoning trace (the <think> block). A sabotage injection d
is validated if it satisﬁes either of the following conditions:
1.Effect Match: The fuzzy token set ratio between dandEteacher exceeds a threshold
τfuzz>60.
2.Cause Match: The injected scalar dis explicitly present within the Teacher’s causal reason-
ing block Θteacher .
Axiomatic Noise Injection. To train the GENERAL class, we utilize TruthfulQA’s ﬁnance and eco-
nomics splits. Initially, these records contain an empty context ( C=∅). However, training an
autoregressive model on this representation induces a spurious correlation: the model learns the
structural shortcut C=∅=⇒GENERAL , bypassing semantic reasoning entirely. To force true
logic-awareness, we implement Axiomatic Noise Injection . For every general axiom, we sample a
subset of completely irrelevant ﬁnancial text chunks from a distractor pool ( Cdistractor ). The model
must learn to actively ignore the distractor context and rely on its parametric memory when the trace
signals GENERAL_KNOWLEDGE , preventing structural overﬁtting.
The “Virtual Context” Protocol and HITL Reﬁnement. For highly ambiguous samples where the
automated validation yields low conﬁdence, we utilize an AI Proxy Reﬁner. To minimize context-
window bloat and token costs, the Reﬁner operates under a “Virtual Context” protocol: it is de-
nied access to the raw context C. Instead, it relies entirely on the Teacher’s Θteacher block as its
“eyes.” Guided by a strict meta-prompt, the Reﬁner surgically corrects nomenclature confusions,
9

🧪 VeNRA Active Data Refinement Studio
Audit Rejection Reason: Teacher labeled as Supported 🤖 AI-Refined (Meta-Prompt)
⬅ PreviousSample1/ 344
Next (Skip) ➡
Audit Target:sabotaged_unfounded
Attack Type:semantic_driftIntended Lie:Framework Teacher Verdict:Supported
Confidence: N/A
📝 Evidence (Read Only)
User Question:
Were the company's internal control over financial reporting and
disclosure controls and procedures considered eﬀective as of
December 31, 2023?
Text Context:
Changes in and Disagreements with Accountants on Accounting and
Financial Disclosure. Not applicable. Item 9A. Controls and Procedures.
Disclosure Controls and ProceduresWe maintain disclosure controls and
procedures, as defined in Rules 13a-15(e) and 15d-15(e) under the Exchange
Act, that are designed to ensure that information required to be disclosed in
the reports we file with the SEC under the Exchange Act is recorded,
processed, summarized and reported within the time periods specified in
the SEC’s rules and forms, and that such information is accumulated and
communicated to our management, including our principal executive
oﬀicer and principal financial oﬀicer, as appropriate, to allow timely
decisions regarding required disclosure. As of December 31, 2023, we
carried out an evaluation, under the supervision and with the participation
of our management, including our principal executive oﬀicer and principal
financial oﬀicer, of the eﬀectiveness of the design and operation of our
disclosure controls and procedures. Based on this evaluation, our principal
executive oﬀicer and principal financial oﬀicer concluded that our
disclosure controls and procedures were eﬀective at a reasonable
assurance level as of December 31, 2023.105Internal Control over Financial
ReportingOur management is responsible for establishing and maintaining🛠 Refinement Zone (Editable)
🎯 Edit Target Sentence
Framework, the company's internal control over financial reporting and
disclosure controls and procedures were considered eﬀective as of
December312023
⚙ Edit Logic Trace
# VERIFICATION_TYPE: TEXT_ANAL YSIS
💭 Edit Internal Thinking
1. Entity Audit: The text concludes that controls were eﬀective but the
sentence starts with 'Framework', which is a sabotaged replacement for the
word 'Yes'. 2. Temporal Audit: Date (Dec 31, 2023) matches. 3. Trace Validation:
The trace is generic. The teacher's reasoning missed that 'Framework' is a
nonsensical entity swap that invalidates the sentence as a faithful extraction.
🔍 Edit Final Analysis
The target sentence is logically corrupted by a semantic swap where the
aﬀirmative conclusion ('Yes') was replaced by the word 'Framework'.
Although the underlying claim about eﬀectiveness is factually supported by
thetextthespecificphrasingintroducedbythesabotagecreatesan
Commit Decision
🔥 UNFOUNDED
 ✅ SUPPORTED
 🌐 GENERAL
 🗑 DISCARDData Pipeline
Total Audited
9335
Auto-Verified
8610
In Review Queue
344
Human Decisions
66
AI Refined
336
🔄 Refresh Data
VeNRA HITL v3.4 | Studio
ModeFigure 2: The VeNRA Reﬁnement UI. Streamlit UI to faciliate human annotation that are ambigu-
ous from teacher and the AI proxy reﬁner.
sign-convention inversions (e.g., treating a cash outﬂow as an inﬂow), and boilerplate evasions, ef-
fectively overruling the Teacher model when it exhibits “math-blindness.”. If the result was still
ambigious, we tag for human annotation via the speciﬁc UI shown in Fig. 2.
4.4 Contrast Pairs and Hybrid-Family Splitting
The ﬁnal architectural requirement for VeNRA-Data is the prevention of dataset leakage and the
establishment of a causal evaluation framework.
Standard randomized stratiﬁcation (e.g., an 80/10/10 train/val/test split) over a dataset containing
cloned records guarantees severe data leakage. If a Golden record (the Parent) is assigned to the
training set, and its sabotaged clone (the Child) is assigned to the validation set, the model evaluates
on a context it has already memorized, inﬂating performance metrics.
To eliminate leakage and maximize the learning signal, we implement Hybrid-Family Splitting .
Let a Family Fibe deﬁned as a set containing a golden Parent record and its corresponding sabo-
taged Children: Fi={Sparent , S′
child _1, . . .}. Families are derived programmatically via a lineage
heuristic ( family_id ).
The dataset is partitioned not by individual records, but by entire Families. Families are grouped
into four distinct semantic buckets:
1.Sabotage Pairs: Families containing both SUPPORTED andUNFOUNDED siblings.
2.Natural Failures: Families containing organically UNFOUNDED records (e.g., original Fi-
nanceBench failures) with no supported sibling.
3.Natural Supported: Families comprising only golden records not selected for sabotage.
4.Axioms: Families representing GENERAL truths.
A stratiﬁed split is performed across these buckets, ensuring exactly 10% of the Families from each
bucket are allocated to the Validation and Test sets respectively.
10

This strict topological isolation provides a dual beneﬁt: it mathematically guarantees zero contextual
leakage between splits, and it forces the model to learn the delta . By encountering identical contexts
and queries during training, where one answer is mathematically sound and the other contains a
subtle logic lie, the model’s loss landscape is optimized to detect minute reasoning derivations rather
than broad linguistic artifacts. Furthermore, preserving these pairs in the evaluation sets enables the
calculation of a strict “Flip Rate” (Sensitivity), measuring the model’s ability to correctly invert its
prediction when presented with a minimal adversarial perturbation.
5 VeNRA Sentinel: Forensic Auditing under Latency Constraints
To deploy VeNRA in a production ﬁnancial environment, the generated answers must be audited
in real-time. Frontier LLM-as-a-Judge models introduce prohibitive latency ( ≈2-4seconds) and
excessive operational costs. To solve this, we introduce the VeNRA Sentinel , a specialized 3-B
parameter Small Language Model (SLM) based on Qwen-2.5-Coder-3B-Instruct5.
The Sentinel is trained as a “White-Box” hallucination detector. It ingests the User Query ( Q), the
Text Context ( C), the Python Execution Trace ( T), and the Target Sentence ( S), outputting a strict
tripartite verdict: SUPPORTED ,UNFOUNDED , orGENERAL . To satisfy a strict inference latency budget
while maintaining reasoning capabilities, we had to fundamentally alter standard instruction-tuning
paradigms.
5.1 The Sytem 1.5 Paradigm and Token Orthogonality
Standard Chain-of-Thought (CoT) prompting dictates that reasoning must precede the ﬁnal answer
(e.g., [Analysis]→[Label] ). While this allows the model to utilize computational “test time”
to arrive at a conclusion, autoregressive generation of a 150-token analysis incurs ≈1500 ms of
latency on standard consumer hardware, violating our latency constraints.
To satisfy latency constrain, we inspired from SALSA (Single-pass Autoregressive LLM Structured
Classiﬁcation) framework, formatting the training targets as Reverse-CoT:
Label: [Verdict] \n Analysis: [Reasoning] .
In an autoregressive framework, tokens generated after the label cannot mathematically inﬂuence the
probability distribution of the label itself during a single forward pass. However, training the model
to predict the label and immediately explain it acts as a “Gradient Tether.” It forces the hidden
states of the Label token to become incredibly dense, requiring it to contain not just the binary
classiﬁcation decision, but the precise latent “seed” of the upcoming forensic explanation. During
production inference, we execute a single forward pass, extract the unnormalized logits for the label
tokens, and programmatically halt generation, achieving latency veriﬁcation. If a query is ﬂagged
asUNFOUNDED , generation can optionally resume to provide the debug analysis to the end-user.
To extract highly calibrated logits at the very ﬁrst decoding step, the target tokens must be se-
mantically and mathematically distinct. Naive binary labels (e.g., True /False orSupported /
Unfounded ) suffer from severe semantic conﬂation in the model’s pre-training data.
To select optimal classiﬁcation tokens, we extracted the input embedding matrix E∈R|V|×dfrom
the base Qwen model and computed pairwise cosine similarities:
sim(ei, ej) =ei·ej
∥ei∥∥ej∥(3)
We discovered that standard labels like True /False exhibited a high cosine similarity ( sim≈
0.68), meaning their initial vector representations are highly proximate, forcing the attention layers
to work harder to separate them. Through programmatic search, we identiﬁed a highly orthogonal
set of single-token labels: Found ,Fake , and General . This set yielded a maximum pairwise
cosine similarity of <0.25, ensuring that the linear classiﬁcation head operates on easily separable
latent vectors.
5https://huggingface.co/pagand/venra
11

5.2 The Effects of Loss Dilution
Training the Sentinel model using standard trainers (e.g. SFTTrainer ) resulted in catastrophic
failure. The model rapidly achieved a low validation loss but exhibited a near-random “Flip Rate”
accuracy on the adversarial contrast pairs. The model had learned to be a Sycophant : it minimized
global loss by defaulting to the majority class ( Found ) and generating highly ﬂuent, generic ﬁnancial
analyses.
In a standard autoregressive Cross-Entropy formulation over a sequence of length N, the loss is
averaged across all active completion tokens:
Lbatch =−1
NN∑
i=1logP(yi|x<i) (4)
In our Reverse-CoT prompt, the sequence consists of 1label token and approximately 150analysis
tokens. Consequently, the gradient contribution of the critical classiﬁcation decision is profoundly
diluted:∂L
∂θlabel∝1
151<0.7%. The model is overwhelmingly penalized for grammatical imperfec-
tions in its reasoning rather than catastrophic failures in its verdict.
To restore the gradient signal of the verdict, we must apply a large differential
weight to the label token. However, implementing a weighted loss via PyTorch’s
CrossEntropyLoss(reduction=’none’) on a 3-B parameter model triggers immediate
Out-Of-Memory (OOM) failures on a single GPU (with <50GB of VRAM)6. To solve this, we
employed a custom Micro-Chunking Trainer . We bypass the model’s internal loss calculation,
ﬂattening the shift-logits and shift-labels. We then iteratively slice the sequence into micro-chunks
of size c= 512 entirely within the loss computation loop. For a ﬂat sequence length T=B×S,
the weighted loss is accumulated safely:
Ltotal =∑⌈T/c⌉
k=0∑
i∈CkwiLCE(ˆyi, yi)
∑⌈T/c⌉
k=0∑
i∈Ckwi(5)
This architectural optimization bounds the VRAM consumption of the loss function to O(c×|V|)
rather thanO(B×S×|V|), maintaining exact mathematical equivalence to the full weighted Cross-
Entropy.
5.3 Dynamic Loss Clamping and Differential Weighting
While the Micro-Chunking Trainer enables weighted loss, multiplying the label gradient by 50×
introduces severe topological instability ( Gradient Seesaw ). Imposing a large penalty on a conﬁ-
dently wrong prediction (where P(ˆy)≈0) causes the cross-entropy loss to spike asymptotically7.
To stabilize convergence, we introduced Dynamic Loss Clamping , explicitly capping the maximum
penalization:
Lweighted = min( wi·LCE,5.0·wmax) (6)
Initially, we applied the penalty uniformly across all three classes. This induced a secondary failure
mode: the model began drastically over-predicting the minority General class. The model utilized
theGeneral token as a “safe harbor” to avoid the massive penalization of misclassifying the pri-
mary Found /Fake distribution. We resolved this via Asymmetric Differential Weighting : assigning
a50×penalty to Found /Fake to force hard binary logic, but reducing the General penalty to 10×,
preserving axiomatic knowledge retention without inducing class bleed.
To facilitate training at high adapter ranks ( r= 96,128) under these extreme gradient conditions,
we utilized 4-bit NF4 Quantization (QLoRA) paired with Rank-Stabilized LoRA (rsLoRA) Kala-
jdzievski [2023 ]. By scaling the adapter activations byα√rrather thanα
r, rsLoRA prevents gradient
vanishing in deeper layers, enabling stable logic-awareness acquisition.
6For a micro-batch size B= 2and sequence length S= 4096 , calculating the unreduced loss instantiates
a tensor of size [B×S× |V|]. At FP32 precision (required for loss stability), this single intermediate tensor
consumes ≈1.25Billion ﬂoats ( ≈5GB). Combined with gradients and optimizer states, this vastly exceeds
VRAM capacity.
7During early training, we observed massive gradient norm explosions ( ∥g∥>72,000), causing the model
to violently oscillate, updating from predicting 100% Fake in one step to 100% Found in the next.
12

5.4 Prompting Physics and Uncertainty Quantiﬁcation
Because the base model ( Qwen-2.5-Coder-3B-Instruct ) possesses a strong inductive bias for
code execution, it instinctively attempts to mathematically solve the Python trace provided in the
prompt. However, a 3B parameter model lacks the FLOP depth to reliably execute complex 6-digit
subtraction in a single forward pass. If it attempts arithmetic simulation, it hallucinates. We coun-
teract this via Algorithmic De-activation in the system prompt8to successfully redirects the model’s
attention heads from arithmetic simulation toward semantic set-intersection and string-matching.
Causal LLMs suffer from the “Lost in the Middle” phenomenon in large contexts An et al. [2024 ].
To prevent the model from forgetting the veriﬁcation target after processing 4,000 tokens of 10-K
text, we implement Selective Prompt Repetition . We state the Query and Trace at the very top of the
prompt, inject the massive context, and strictly repeat the Query and Trace at the bottom boundary
immediately preceding the Label: generation token. This aligns with the ﬁnding of Leviathan et al.
[2025 ] for non-reasoning tasks.
During inference, raw softmax probabilities are deceptive; a model can output Ptop1= 0.99simply
due to softmax temperature, despite high internal confusion Liu et al. [2025 ]. Drawing inspiration
from recent production safety frameworks such as HaluGate Park et al. [2025 ], VeNRA implements
a Logit Gap uncertainty threshold: ∆ = Ptop1−Ptop2. If∆<0.15, the Sentinel determines the
semantic entropy is too high, forcefully overrides the output to Uncertain , and routes the query to
a human auditor or a frontier System-2 model.
6 Discussion, Limitations, and Conclusion
6.1 Discussion: The Neuro-Symbolic Efﬁciency Frontier
The prevailing trajectory in Large Language Model research assumes that hallucination mitigation
is a function of parametric scaling. The VeNRA framework demonstrates an alternative, highly
efﬁcient paradigm: Neuro-Symbolic Cognitive Ofﬂoading . By acknowledging that LLMs are funda-
mentally probabilistic sequence generators, VeNRA surgically removes their responsibility to per-
form arithmetic, bounding them instead within strictly typed schemas (the UFL) and deterministic
execution environments (Python).
This architectural shift yields a dual beneﬁt. First, it achieves “Zero-Hallucination” arithmetic by
construction, provided the initial extraction is grounded. Second, the development of VeNRA-Data
proves that the bottleneck in creating reliable “LLM-as-a-Judge” models is not model size, but data
ecology. By shifting evaluation benchmarks from linguistically obvious semantic noise to struc-
turally precise Adversarial Simulation (e.g., Logic Code Lies), we empower locally deployable 3B
parameter SLMs to achieve forensic veriﬁcation capabilities previously restricted to frontier models,
effectively democratizing high-stakes ﬁnancial auditing.
6.2 Limitations
Despite its structural rigor, the VeNRA architecture possesses inherent limitations grounded in its
reliance on deterministic extraction:
1.Recursive Graph Blindness: The UFL ﬂattens ﬁnancial reports into a relational table.
While this enables rapid SQL-like ﬁltering, it fails to capture recursive second-order rela-
tionships (e.g., “Supplier of a Supplier”) inherent in complex supply chains. Currently, the
Naive Graph Expansion relies on single-step ﬁltering, potentially missing multi-hop risks.
2.Referential Nuance Loss: Financial footnotes frequently employ referential pointers (e.g.,
“See Note 12”). While our Trailing-Buffer Chunking captures immediate context, a footnote
extracted into the text_nuance column loses its hypertextual integrity. If the referenced
note resides in a disjointed chunk not retrieved by the Navigator, the qualitative context is
severed, potentially leading to misinterpretation of conditional covenants.
8Example prompt: “Do NOT recalculate the math. Assume the arithmetic evaluates to the Claimed Answer.
You must verify the EXTRACTION and LOGIC. ”
13

3.The Semantic Schema Bottleneck: The Lexical Pre-Filter effectively prevents vector con-
ﬂation for standard GAAP metrics. However, highly idiosyncratic corporate taxonomies
(e.g., non-GAAP “Adjusted EBITDAR excluding Stock-Based Comp”) may evade the Nav-
igator’s synonym mapping. If the Lexical Gate is too aggressive, it risks ﬁltering out valid
but non-standard metric terminologies.
6.3 Conclusion
In high-stakes domains such as ﬁnance, a 99% accuracy rate yields 0%operational trust. VeNRA
establishes a novel blueprint for grounded Retrieval-Augmented Generation. By pioneering the Uni-
versal Fact Ledger (a deterministic variable store), Double-Lock Grounding (a strict extraction
ﬁrewall), and the VeNRA Sentinel (a latency-optimized forensic judge), we provide a mathemati-
cally grounded architecture for reliable ﬁnancial reasoning. We release the VeNRA-Data adversarial
dataset and the OOM-safe Micro-Chunking Trainer to accelerate the development of specialized,
logic-aware veriﬁcation models across the open-source community.
14

A Universal Fact Ledger (UFL) 2.0 Schema
The UFL is implemented as a strict Pydantic v2 schema to ensure compatibility with the Python
execution engine. It acts as the API contract between the probabilistic extractor and the deterministic
solver.
class UFLRow(BaseModel):
row_id: str # md5(entity + metric + period)
canonical_entity_id: str # Normalized (e.g., ’ID_AAPL’)
metric_name: str # Semantic key (e.g., ’Revenue’)
# --- Computation & Grounding ---
num_value: Optional[float] # None for qualitative facts
grounding_quote: str # Verbatim substring for alignment
unit_normalized: str # e.g., "USD", "Ratio", "USD/Share"
scale: float # Multiplier (e.g., 1e6)
# --- Context & Filtering ---
period_end: Optional[str] # ISO-8601 Date
period_type: Optional[str] # ’FY’, ’Q3’, ’TTM’
doc_section: str # Breadcrumb path
source_chunk_id: str # ChromaDB Foreign Key
text_nuance: Optional[str] # Footnotes / formulas
# --- Alignment Metadata (Post-Hoc Aligner) ---
char_interval: Optional[Tuple[int, int]]
alignment_status: Literal["EXACT", "PARTIAL", "FUZZY", "UNALIGNED"]
confidence_score: float # 0.95 for Table Melts, 0.0 if Unaligned
B Adversarial Simulation Protocols (VeNRA-Data)
The Saboteur Engine generates “Hard Negatives” by mutating grounded reasoning traces via four
speciﬁc attack vectors.
1. Logic Code Lie (Variable Swap)
•Context: “...revenue was $100M, while operating expenses were $80M...”
•Golden Trace: step_1 = 100 - 80 →Target: “Proﬁt was 20.” (SUPPORTED)
•Sabotaged Trace: step_1 = 150 - 80 →Target: “Proﬁt was 70.” (UNFOUNDED)
•Task: Judge must detect that 150does not match the text context 100.
2. Numeric Neighbor Trap (Table Shift)
•Context: A matrix where M2022= 45.5andM2023= 52.1.
•Query: “What was the metric in 2022?”
•Sabotaged Sentence: “The metric in 2022 was 52.1.” (UNFOUNDED: Temporal slip).
3. Time Warp (Irrelevancy)
•Golden Query: “What was Apple’s revenue in 2021?”
•Sabotaged Query: “What was Apple’s revenue in 2020 ?”
•Note: Context and Answer remain entirely unchanged. The model must recognize that the
correct evidence answers the wrong temporal question.
The Teacher-Auditor Validation Protocol To prevent “Accidental Truths” (where a sabotage coin-
cidentally matches reality), we employ a Dual-Path Validation check using a GPT-4o Auditor:
•Effect Match: Does the Auditor’s detected error span fuzzy-match our injected value?
15

•Cause Match: Does the injected value appear explicitly in the Auditor’s internal reasoning
trace ( Θteacher )?
Only samples satisfying one of these conditions are accepted into the dataset.
C Micro-Chunking Differential Loss Algorithm
To prevent 1.25Billion ﬂoat Vocabulary OOMs when applying 50×differential penalties to label
tokens, we utilize the following algorithm to iteratively compute loss within a bounded VRAM
footprint.
Algorithm 1 OOM-Safe Micro-Chunking Weighted Loss
Input: Logits ˆY∈RB×S×|V|, Labels Y∈ZB×S, WeightsW, Chunk Size c
Output:Lfinal (Scalar)
1:Flatten shift-logits: ˆYflat←ˆY[...,:−1,:].view(−1,|V|)
2:Flatten shift-labels: Yflat←Y[...,1 :].view(−1)
3:Ltotal←0.0,Wtotal←0.0
4:T←length (Yflat)
5:fori= 0toTstepcdo
6: ˆychunk←ˆYflat[i:i+c]
7: ychunk←Yflat[i:i+c]
8: Mask←(ychunk̸=−100)
9: ifany(Mask )then
10: Lraw←CrossEntropy (ˆychunk , ychunk ,reduce =none)
11: wchunk←W [ychunk·Mask ] ▷Safe weight lookup
12: Lweighted←Lraw×wchunk
13: Lclamped←min(Lweighted ,5.0×max(W)) ▷Gradient Seesaw Fix
14: Ltotal←Ltotal+∑(Lclamped [Mask ])
15: Wtotal←Wtotal+∑(wchunk [Mask ])
16: end if
17:end for
18:return Ltotal/max( Wtotal,1e−9)
D VeNRA Prompting Physics (Bookending)
The Sentinel Reverse-CoT prompt relies on strict selective repetition to overcome the “Lost in the
Middle” phenomenon over 4,000tokens.
<|im_start|>user
### TASK:
Verify if the CLAIMED_ANSWER given the AGENT_TRACE is [Found, Fake, General].
### VERIFICATION TARGET:
**Query:** {query}
**AGENT TRACE:** {trace}
**Claimed Answer:** {statement}
### EVIDENCE (Source Document):
{context_chunks}
### VERIFICATION TARGET Recap:
**Query:** {query}
**AGENT TRACE:** {trace}
**Claimed Answer:** {statement}
### AUDIT ALGORITHM (CRITICAL):
16

Do NOT recalculate the math. Assume the arithmetic evaluates to the Claimed
Answer. You must verify the EXTRACTION and LOGIC. Output your label first.
<|im_end|><|im_start|>assistant
Label:
References
Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. Make your llm
fully utilize the context. Advances in Neural Information Processing Systems , 37:62160–62188, 2024.
Ross Anderson and Nick Koudas. Greenback bears and ﬁscal hawks: Finance is a jungle and text embeddings
must adapt. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 1120–1135, 2024.
Yupeng Chang, Xinyu Wang, and Jindong Zhang. Thor: Tool-integrated hierarchical optimization via rl for
mathematical reasoning. arXiv preprint arXiv:2501.04566 , 2025.
Wenhu Chen, Xueguang Ma, Xinyu Wang, and William W Cohen. Program of thoughts prompting: Disentan-
gling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588 , 2022.
Yilun Chen, Shuai Wang, and Yang Liu. Fintextqa: A dataset for ﬁnancial text question answering. arXiv
preprint arXiv:2406.08544 , 2024.
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Andre Dylan, William Yang Wang,
and Diyi Perrault. Finqa: A dataset of numerical reasoning over ﬁnancial data. In Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing , pages 3697–3711, 2021.
Eunsol Choi and Jinhyeon Kim. Finder: Financial numeric data extraction and reasoning benchmark. arXiv
preprint arXiv:2501.11022 , 2025.
Konstantinos Dadopoulos, Dimitra Koutsomiha, and Nikos Karacapilidis. Metadata-driven retrieval-augmented
generation for ﬁnancial question answering. arXiv preprint arXiv:2501.08200 , 2025.
Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan
Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods,
scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858 , 2022.
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham
Neubig. Pal: Program-aided language models. In International conference on machine learning , pages
10764–10799. PMLR, 2023.
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi,
Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need.
arXiv preprint arXiv:2306.11644 , 2023.
Kelly Hong, Anton Troynikov, and Jeff Huber. Context rot: How increasing input tokens impacts llm perfor-
mance. URL https://research. trychroma. com/context-rot, retrieved October , 20:2025, 2025.
Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Fi-
nancebench: A new benchmark for ﬁnancial question answering. arXiv preprint arXiv:2311.11944 , 2023.
Ziwei Ji, Tianshi Zhang, Nayeon Lee, and Gabriel Grand. Phantom: A benchmark for hallucination detection
in ﬁnancial long-context qa. arXiv preprint arXiv:2502.12490 , 2025.
Damjan Kalajdzievski. A rank stabilization scaling factor for ﬁne-tuning with lora. arXiv preprint
arXiv:2312.03732 , 2023.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 6769–6781, 2020.
Omar Khattab and Matei Zaharia. Colbert: Efﬁcient and effective passage search via contextualized late in-
teraction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and
Development in Information Retrieval , pages 39–48, 2020.
Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin,
Sungdong Kim, James Thorne, et al. Prometheus: Inducing ﬁne-grained evaluation capability in language
models. arXiv preprint arXiv:2310.08491 , 2023.
17

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Prompt repetition improves non-reasoning llms. arXiv
preprint arXiv:2512.14982 , 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Advances in Neural Information Processing Systems , volume 33, pages 9459–9474,
2020.
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucina-
tion evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing , pages 6449–6464, 2023.
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods.
InProceedings of the 60th Annual Meeting of the Association for Computational Linguistics , pages 3214–
3252, 2022a.
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods.
InProceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long
papers) , pages 3214–3252, 2022b.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the middle: How language models use long contexts. Transactions of the Association for Computa-
tional Linguistics , 12:157–173, 2024.
Xunzhuo Liu, Huamin Chen, and vLLM Semantic Router Team. Token-level truth: Real-time hallucination
detection for production llms. https://blog.vllm.ai/2025/12/14/halugate.html , dec 2025. vLLM
Semantic Router Blog.
Eunkyu Park, Minyeong Kim, and Gunhee Kim. Halloc: Token-level localization of hallucinations for vision
language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pages 29893–29903, 2025.
Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word
problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies , pages 2080–2094, 2021.
Mrinank Sharma, Meg Tong, Tomasz Korbak, David Rogers, Gabriel PACIFIC, and Ethan Perez. Towards
understanding sycophancy in language models. arXiv preprint arXiv:2310.13548 , 2023.
Bowen Tan, Zhexin Zhang, and Zhuoer Liu. Fred: Financial retrieval-enhanced detection and editing of hallu-
cinations in language models. arXiv preprint arXiv:2502.09112 , 2025.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heteroge-
neous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 ,
2021.
Dannong Wang, Jaisal Patel, Daochen Zha, Steve Y Yang, and Xiao-Yang Liu. Finlora: Benchmarking lora
methods for ﬁne-tuning llms on ﬁnancial datasets. arXiv preprint arXiv:2505.19819 , 2025.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and
Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural
Information Processing Systems , 35:24824–24837, 2022.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yanping Zhuang, Zi Lin, Zhuo-
han Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in
Neural Information Processing Systems , 36, 2024.
Fengbin Zhu, Wenqiang Lei, Chao You, Simeng Gu, Zifan Chen, Soujanya Wang, Soujanya Poria, and Tat-Seng
Chua. Tat-qa: A question answering benchmark on a hybrid of tabular and textual content in ﬁnance. In
Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics , pages 3276–3287,
2021.
18
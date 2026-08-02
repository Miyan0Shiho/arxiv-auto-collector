# When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations

**Authors**: Jinhan Kim, Samuele Pasini, Paolo Tonella

**Published**: 2026-07-29 12:32:04

**PDF URL**: [https://arxiv.org/pdf/2607.26843v1](https://arxiv.org/pdf/2607.26843v1)

## Abstract
Retrieval-Augmented Generation (RAG)-based LLM systems rely on external document corpora that can evolve and change over time. However, current evaluation methodologies (e.g., RAGAS) assess correctness against static snapshots, failing to detect faults when routine updates, factual changes, or noise alter the underlying data. We introduce a metamorphic testing framework that evaluates the consistency of RAG systems under corpus evolution. We formalise a fault taxonomy and 11 mutation operators that systematically perturb the system at both the pre-chunk (retrieval index) and post-chunk (retrieved context) levels. An empirical evaluation across five datasets and over 28k mutants reveals metamorphic violation rates of 4.9-10.2%. In a meta-evaluation against ground truth, our metamorphic oracle achieves F1 scores of 0.927-1.000, while the best RAGAS metric reaches only 0.570. Finally, we provide actionable insights into mitigating these faults through retrieval re-configuration, generator upgrades, and LLM-based reranking.

## Full Text


<!-- PDF content starts -->

When Knowledge Changes: Metamorphic Testing of RAG Systems
with Mutations
Jinhan Kim✉
Università della Svizzera italiana
Lugano, Switzerland
jinhan.kim@usi.chSamuele Pasini
Università della Svizzera italiana
Lugano, Switzerland
samuele.pasini@usi.chPaolo Tonella
Università della Svizzera italiana
Lugano, Switzerland
paolo.tonella@usi.ch
Abstract
Retrieval-Augmented Generation (RAG)-based LLM systems rely
on external document corpora that can evolve and change over
time. However, current evaluation methodologies (e.g., RAGAS)
assess correctness against static snapshots, failing to detect faults
when routine updates, factual changes, or noise alter the underlying
data. We introduce a metamorphic testing framework that evalu-
ates the consistency of RAG systems under corpus evolution. We
formalise a fault taxonomy and 11 mutation operators that system-
atically perturb the system at both the pre-chunk (retrieval index)
and post-chunk (retrieved context) levels. An empirical evaluation
across five datasets and over 28k mutants reveals metamorphic
violation rates of 4.9–10.2%. In a meta-evaluation against ground
truth, our metamorphic oracle achieves F1 scores of 0.927–1.000,
while the best RAGAS metric reaches only 0.570. Finally, we provide
actionable insights into mitigating these faults through retrieval
re-configuration, generator upgrades, and LLM-based reranking.
CCS Concepts
•Software and its engineering;
Keywords
Retrieval-Augmented Generation, Metamorphic Testing
ACM Reference Format:
Jinhan Kim, Samuele Pasini, and Paolo Tonella. 2026. When Knowledge
Changes: Metamorphic Testing of RAG Systems with Mutations. InProceed-
ings of the 41st IEEE/ACM International Conference on Automated Software
Engineering (ASE ’26), October 12–16, 2026, Munich, Germany.ACM, New
York, NY, USA, 12 pages. https://doi.org/10.1145/3832783.3837543
1 Introduction
Retrieval-Augmented Generation (RAG) systems have become a
dominant architectural pattern for deploying large language mod-
els (LLMs) in knowledge-intensive applications [ 7]. By retrieving
documents from an external corpus and conditioning generation on
the retrieved content, RAG systems aim to overcome the limitations
of LLMs’ parametric knowledge and provide answers grounded in
external sources. This architecture is widely adopted in settings
such as enterprise knowledge bases, compliance assistants, and
proprietary documentation systems, where correctness depends on
access to evolving document collections [10].
This work is licensed under a Creative Commons Attribution 4.0 International License.
ASE ’26, Munich, Germany
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2882-2/2026/10
https://doi.org/10.1145/3832783.3837543Existing evaluation methodologies for RAG systems primarily
assess snapshot performance [ 4,12]. Benchmarks typically mea-
sure answer accuracy, retrieval quality, or grounding with respect
to a fixed corpus representation. These evaluations are effective
for comparing model configurations under controlled conditions,
but they implicitly assume that the document corpus and retrieval
pipeline remain unchanged. In practice, this assumption does not
always hold: document corpora evolve continuously, facts are up-
dated, rules change, evidence is added or removed, and documents
are restructured, re-chunked, or re-indexed as part of system main-
tenance and evolution.
From a software engineering perspective, it is critical that a RAG
system remains sensitive to thestateof the corpora. The system
should not only provide correct answers for a static snapshot but
also demonstrate appropriate behavioural shifts (or stability) as
the corpus undergoes semantically significant (or insignificant)
changes. For example, when an irrelevant document is re-formatted,
the system’s output should remain unchanged; when a factual value
is updated, the output should reflect the update; when supporting
evidence is removed, the system should express uncertainty (e.g.,
abstain) rather than hallucinate. Whether current RAG systems
satisfy such expectations remains unexplored.
To overcome this problem, we introduce a metamorphic testing
paradigm specifically tailored for RAG systems. Rather than relying
on a fixed ground-truth answer, we evaluate correctness by observ-
ing how system outputs shift in response to controlled semantic
transformations of the corpus. We formalise these expectations as
a set of Metamorphic Relations (MRs). These relations define the
invariant properties of the system under corpus evolution, such
as maintaining behavioural consistency when encountering irrele-
vant noise, updating answers appropriately when factual premises
change, and explicitly acknowledging conflicts when contradictory
evidence is injected.
Systematically driving this evaluation requires formalising how
RAG systems fail. Based on a fault taxonomy that we derived by
synthesising failure modes from prior empirical studies of RAG
robustness [ 8,14,18–20], we design 11 mutation operators applied
across two operational scopes:pre-chunkmutations that perturb
the upstream document index, andpost-chunkmutations that alter
the downstream retrieved context directly. We then execute the
RAG pipeline across the original and mutated context, checking
the outputs against our defined MRs. Any violation serves as an in-
dicator of semantic inconsistency, pointing to a potential weakness
of the RAG system.
We evaluate our framework on five datasets, executing over
28k mutants across both scopes. Our study yields three principal
findings. First, the framework detects faults across all five datasets,
with violation rates ranging from 4.9% to 10.2%. The two mutation
arXiv:2607.26843v1  [cs.SE]  29 Jul 2026

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
scopes expose complementary faults (Jaccard overlap 12–45%), con-
firming the diagnostic value of testing at both the index and context
levels. Second, in a meta-evaluation against ground-truth correct-
ness, our MR oracle achieves F1 scores of 0.927–1.000, whereas the
best-performing RAGAS metric [ 4] reaches only 0.570 F1. Third,
repair effectiveness varies substantially across datasets. The union
of retrieval-budget increases, a generator upgrade, and LLM-based
reranking resolves 43.1% of evaluated faults. Union repair rates
range from 47.3% to 63.0% across the four financial datasets but
reach only 10.2% on RepLiQA, pointing to failure modes that neither
broader retrieval, stronger generation, nor LLM-based reranking
can fully address.
In summary, this paper makes the following contributions:
•We introduce a metamorphic testing framework for RAG
systems under corpus evolution, driven by 11 mutation op-
erators across two scopes.
•We conduct an empirical study across five datasets demon-
strating that our MRs expose silent failures that RAGAS
metrics overlook, validated by a human study and oracle
stability analysis.
•We analyse the repairability of detected faults, showing that
LLM-based reranking complements retrieval and generator
interventions and that their union repairs 43.1% of detected
faults.
2 Background
This section summarises established concepts and practices relevant
to this work.
2.1 Retrieval-Augmented Generation (RAG)
RAG refers to a class of systems that combine a language model
with an external document collection [ 5]. Given a user query, a
retrieval component selects a set of documents from the corpus, and
a generation component (e.g., LLM) produces an output conditioned
on both the query and the retrieved documents. RAG systems are
commonly used when the information required to answer queries
is not fully captured in the model’s parameters, or when access
to external, domain-specific, or up-to-date knowledge is required.
The document corpus is therefore treated as an integral part of the
system rather than as static training data.
The standard RAG procedure consists of two primary phases:
document indexing and output generation. During the offline in-
dexing phase, the corpus is prepared by parsing and segmenting
documents into smaller textual units calledchunks. These chunks
are transformed into high-dimensional vectors via an embedding
model [ 3] and stored in a vector database. This spatial arrangement
is fundamental to the system’s behaviour, as proximity within the
vector space indicates semantic similarity between pieces of in-
formation. In the subsequent online inference phase, the system
processes a user query by embedding it with the same model to
identify and retrieve thetop-kmost similar chunks using a distance
measure like cosine similarity. These chunks serve as the essential
context for the system’s decision-making and are integrated into
a structured prompt. The process concludes when the LLM gener-
ates a response that is specifically conditioned on both the original
query and this retrieved evidence.2.2 Evaluation of RAG Systems
Frameworks like RAGAS [ 4] and ARES [ 12] propose automated,
reference-free evaluation metrics that approximate human judge-
ments using LLM-as-a-judge. For example, at the retrieval level,
Context Precisioncalculates the fraction of retrieved chunks that
are genuinely relevant to the query. Similarly,Context Relevance
evaluates the context at the sentence level by measuring the ratio
of sentences strictly necessary to answer the query against the total
number of sentences retrieved. At the generation level,Response
Relevancyestimates how well the generated answer addresses the
original query. It achieves this by prompting an auxiliary LLM to
generate new questions based on the generated answer and cal-
culating their semantic similarity to the original query. Finally,
Faithfulnessevaluates whether the answer is actively supported by
the retrieved context. It decomposes the response into atomic state-
ments and calculates the proportion of these statements backed by
the retrieved chunks, which is crucial for identifying hallucinated
claims.
While these metrics are useful for comparing systems under
controlled conditions, they provide limited insight into testing how
a RAG system behaves as its underlying knowledge base evolves.
This highlights the need for evaluation frameworks that assess se-
mantic consistency upon knowledge evolution, ensuring the system
remains robust under realistic corpus perturbations.
2.3 Metamorphic Testing
Metamorphic Testing (MT) is a testing methodology developed for
systems where test oracles are unavailable, incomplete, or imprac-
tical [ 13]. Instead of checking individual outputs for correctness,
MT specifies relations that should hold between outputs produced
under related inputs or environments. Violations of Metamorphic
Relations (MRs) indicate unexpected or inconsistent system be-
haviour, even when individual outputs may appear plausible.
In this work, MT provides a basis for evaluating RAG systems by
comparing system behaviour across different corpus states without
assuming a fixed oracle. Instead of checking whether an output
is absolutely correct against a static ground truth, we define a set
of MRs that specify how the system’s outputs should relate when
the document corpus undergoes controlled semantic changes. This
relational approach allows us to define correctness as the satisfac-
tion of specific predicates across original and mutated corpora. By
checking these relationships, we expose structural faults and rea-
soning errors that may remain undetected by standard evaluation
methods.
3 System Model and Fault Taxonomy
In this section, we first formalise the RAG architecture as a com-
posite pipeline of retrieval and generation functions governed by a
set of system parameters (Section 3.1). We then introduce a fault
taxonomy that classifies robustness failures based on the system’s
inability to maintain semantic consistency under controlled pertur-
bations to its corpus or configuration (Section 3.2).
3.1 System Model
The overview of our approach is depicted in Figure 1. We treat
the RAG pipeline as a stateful process where the transition from

When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations ASE ’26, October 12–16, 2026, Munich, Germany
Original Execution 
Source Documents 
(Original) 
User Query Retriever 
Indexing 
Search Generator 
(LLM) 
Retrieved 
Context 
(Original) 
Output (y) 
Mutated Execution 
Source Documents 
(Mutated) 
User Query Retriever 
(Mutated) 
Indexing 
Search Retrieved 
Context 
(Mutated) 
Generator 
(LLM) 
Mutation 
Operator ( 𝛿)
Output (y') 
Metamorphic 
Evaluation 
Semantic 
Comparison 
MR Validation 
Fault 
Detected System 
Robust 
Figure 1: Overview of our metamorphic testing framework for RAG systems. The original execution (top) establishes a baseline
by processing source documents and a user query through standard indexing and search to produce an original answer ( 𝑦).
The mutated execution (bottom) introduces a mutation operator ( 𝛿) that perturbs the system. The metamorphic evaluation
component performs a semantic comparison between two outputs ( 𝑦and 𝑦′). A violation of the metamorphic relation ( ¬P𝛿)
indicates a detected fault, whereas semantic consistency suggests system robustness.
input query to output answer is mediated by the retrieval of evi-
dence from the corpus. By introducing systematic perturbations
to the source documents, retrieved context, or system settings, we
would (or would not) observe the resulting shifts in the output.
This differential execution allows us to isolate the causal chain of a
failure: a fault is not merely a wrong answer, but a deviation from
the expected semantic relationship between two executions.
We formalise a RAG system Sas a composite function of a
retrieverRand a generatorG, operating over a dynamic corpus
C. To capture the highly configurable nature of these pipelines, we
denote the overall system configuration as Θ=⟨𝜃R, 𝜃G⟩, where
𝜃Rrepresents retrieval hyperparameters (e.g., top- 𝑘, chunk size, or
embedding model) and 𝜃Grepresents generation hyperparameters
(e.g., LLM temperature, LLM model version).1Given a query 𝑞, the
system produces an output𝑦=S Θ(𝑞,C)=G 𝜃G 𝑞,R 𝜃R(𝑞,C)
In real-world deployments, neither CnorΘis static. The corpus
evolves through updates, segmentation changes, or noise accumula-
tion, while system configurations are frequently tuned to optimise
performance. This environmental and configurational volatility
renders traditional test oracles (fixed ⟨𝑞, 𝑎⟩ pairs) insufficient, as
the correct answer 𝑎may change or become invalid under different
system states.
To address this oracle problem, we adopt MT. We define correct-
ness not as an absolute match against a static ground truth, but as
the satisfaction of MR. Let 𝛿be a mutation operator that perturbs
1We distinguish the generator parameters ( 𝜃G) from the retrieval parameters ( 𝜃R)
because 𝜃Gcould theoretically be mutated to expose faults. However, our proposed
mutation operators do not alter 𝜃Gbecause this work primarily targets the retrieval
component of the RAG pipeline. Nevertheless, one of our research questions (RQ3)
explores adjusting 𝜃Gtofixa fault once it has been localised to the generator rather
than the retriever (see Section 6.3).the system state such that ⟨C′,Θ′⟩=𝛿(C,Θ) . A robust RAG sys-
tem must satisfy a specific metamorphic relation P𝛿between the
original output𝑦and the post-mutation output𝑦′:
∀𝑞,C,Θ:P 𝛿 SΘ(𝑞,C),S Θ′(𝑞,C′)
In this framework, afaultis defined as a violation of this predi-
cate (¬P𝛿). For example, if 𝛿injects irrelevant noise into the corpus,
P𝛿requires semantic invariance ( 𝑦≈𝑦′). Conversely, if 𝛿updates a
fact in the corpus or severely restricts 𝜃R, causing essential chunks
to drop,P𝛿dictates a corresponding output update or a fallback
response. The following taxonomy classifies these violations based
on the semantic nature of the failure.
3.2 Fault Taxonomy
To systematically characterise failures in RAG systems, we define a
fault model based on the system’s inability to maintain semantic
consistency under perturbations. Our taxonomy was constructed
by synthesising failure modes reported in prior empirical studies
of RAG robustness. We began from the RAG noise taxonomy of
Wu et al. [ 18], which identifies noise types that degrade LLM per-
formance, and extended it with findings from studies on irrelevant
context sensitivity [ 14], knowledge conflicts between retrieved and
parametric knowledge [ 19], positional bias in long contexts [ 8], and
retrieval configuration sensitivity [ 20]. We consolidated these into
three fault classes: (1)Noise Sensitivity Faults, extending the RAG
noise taxonomy established by Wu et al. [ 18]; (2)Format Intoler-
ance Faults; and (3)Architectural Brittleness Faults, a classification
addressing structural brittleness in retrieval pipelines. This taxon-
omy serves as the foundation for the mutation operators defined in
Section 4.

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
3.2.1 Noise Sensitivity Faults.This category encompasses faults
where the generator (i.e., LLM) fails to correctly filter or arbitrate
external information, leading to output corruption.
•Context Overfitting Fault: A logic fault where the system treats
topically related but information-poor padding as valid evidence.
In production environments, this manifests when retrieval sys-
tems surface content optimised for search engines or advertise-
ments that share keywords with the query but lack factual sub-
stance. Shi et al. [ 14] demonstrate that LLMs are highly sus-
ceptible to such irrelevant context, often prioritising retrieved
distractors over relevant knowledge.
•Knowledge Conflict Fault:A state inconsistency fault where the
system fails to resolve contradictory facts. This reflects the com-
mon real-world challenge ofstale data, where a knowledge base
contains both outdated legacy documents and recent updates
(e.g., v1 vs. v2 manuals). Wu et al. [ 19] demonstrate that sys-
tems frequently exhibit ‘context bias’, overriding correct internal
priors to align with contradictory retrieved evidence. A robust
system must exhibit explicit conflict awareness, signalling or
resolving the discrepancy.
•Input Robustness Fault: A failure to handle minor character-level
perturbations, resulting in significant degradation of retrieval pre-
cision or generation quality [ 18]. This fault is critical in pipelines
processing OCR-scanned legacy documents, noisy speech-to-
text transcripts, or user-generated content containing informal
spelling.
3.2.2 Format Intolerance Faults.Although recent studies suggested
that mixing data types should theoretically help models distinguish
between plain text and core data, many RAG systems still fail under
these conditions [ 18]. Such faults typically occur because the sys-
tem lacks the flexible parsing logic required to process unexpected
structural changes.
•Heterogeneous Data Fault: The system can fail or hallucinate
when processing mixed-modality content. This is a pervasive
issue in technical domains where documentation frequently in-
terleaves natural language with code snippets (Python, SQL),
JSON configurations, or tabular financial data.
•Syntactic Filtering Fault: The system attempts to reason over un-
intelligible, garbled token sequences instead of discarding them
as noise. This simulates the faults in data cleaning pipelines,
where HTML artefacts, encoding errors (mojibake), or truncated
data packets pollute the retrieval index, leading to garbage-in-
garbage-out behaviour.
3.2.3 Architectural Brittleness Faults.This category describes faults
where the system’s correctness is tightly coupled to specific im-
plementation details or parameters of the retrieval pipeline ( 𝜃R),
rather than the semantic content of the knowledge base. A robust
RAG system should exhibitoperational invariance, producing consis-
tent outputs regardless of choices in data discretisation or retrieval
configuration.2
•Segmentation Boundary Fault:A fault where the system’s output
is unstable with respect to the specific discretisation (chunking) of
2However, note that we scope this expectation to non-disruptive changes; radical
configuration changes that fundamentally alter the retrieved evidence would justifiably
produce different outputs (see Section 4.5 for details).the source text. In practice, fixed-size windowing often arbitrarily
severs semantic dependencies, splitting a disclaimer from its
clause or a subject from its verb. This sensitivity is empirically
linked to the “lost-in-the-middle” phenomenon identified by Liu
et al. [ 8], where models fail to retrieve relevant information solely
because the chunking strategy shifted the evidence away from
the edges of the context window.
•Configuration Overfitting Fault:This represents a dependency
fault where correctness is strictly tied to specific chunk align-
ments within the index or a fixed retrieval budget defined in
𝜃R. It demonstrates that the system is brittle and overfitted to
a narrow range of configurations. Xu et al. [ 20] highlight that
architectural choices in retrieval depth can unexpectedly degrade
performance, exposing this underlying lack of generalisation.
4 Mutation Operators and Test Oracles
We define a set of 11 mutation operators to realise the fault tax-
onomy defined in Section 3.2. In the following sub-sections, after
defining the scope of mutation, we introduce our novel mutation
operators, followed by a description of the metamorphic relations
used as oracles.
4.1 Mutation Scope
We evaluate RAG robustness across two distinct operational stages,
pre-chunking and post-chunking. By perturbing the system at dif-
ferent stages, we can assess how changes to the knowledge base
and the retrieved context independently influence the reliability of
the generated output.
4.1.1 Pre-chunk Scope (System-Level Mutation).This scope per-
forms system-level mutation by modifying the raw documents
or retrieval configurationsupstreamof the ingestion pipeline. For-
mally, we define the mutated system state as ⟨C′, 𝜃′
R⟩=𝛿 𝑝𝑟𝑒(C, 𝜃R),
resulting in a perturbed output 𝑦′=G 𝜃G(𝑞,R 𝜃′
R(𝑞,C′)). This per-
turbation alters the documents in the underlying corpus or the
retrieval parameters (such as embedding models or chunking strate-
gies), requiring a full re-indexing of the knowledge base. Unlike
simple input fuzzing for LLMs, this modifies the index structure
and knowledge representation, simulating the operational reality
of continuous document edits, policy updates, or architectural mi-
grations.
4.1.2 Post-chunk Scope (Input-Level Mutation).This scope per-
forms input-level mutation (analogous to standard input fuzzing)
targeting the retrieved context passed from the retriever to the
generator. While less realistic, it can provide a more efficient al-
ternative to pre-chunk mutation as it bypasses the re-indexing of
the corpus or the re-computation of embeddings. Formally, we de-
fine the perturbed output as 𝑦′=G 𝜃G 𝑞, 𝛿𝑝𝑜𝑠𝑡(R𝜃R(𝑞,C)), where
𝛿𝑝𝑜𝑠𝑡perturbs the retrieved context before it reaches the generator.
Mutations are injecteddownstreamof the retrieval step, modify-
ing only the transient prompt while keeping the retrieval index C
and configurations 𝜃Rand𝜃Gpristine. This approach serves as a
controlled experiment: by guaranteeing the mutation is present in
the prompt, we isolate the generator and stress-test its reasoning
capabilities independently of any retrieval variance. Consequently,

When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations ASE ’26, October 12–16, 2026, Munich, Germany
this scope simplifies fault localisation; if a failure occurs under post-
chunk mutations, the behavioural change can be attributed to the
generator.
4.2 Noise Mutation Operators
These operators inject textual perturbations to exposeNoise Sensi-
tivityandFormat Intolerancefaults.
Semantic Noise Operator (SeN).SeN targetsContext Overfit-
ting Faultsby injecting off-topic sentences from a static pool of
unrelated facts. Pre-chunk injection scatters distractors throughout
the document, shifting chunk boundaries and polluting the index
to test if the retriever maintains precision against irrelevant signals.
Post-chunk injection appends off-topic text to retrieved chunks,
testing the LLM’s ability to filter irrelevant trailing context.
Datatype Noise Operator (DN).DN targetsHeterogeneous Data
Faultsby injecting foreign fragments (code, URLs, JSON) into prose.
Fragments are drawn from a predefined pool of code snippets and
inserted at sentence boundaries. Pre-chunk injection embeds these
into the corpus structure, testing vector space robustness against
mixed modalities. Post-chunk injection injects fragments into re-
trieved chunks individually, validating if the LLM can parse mixed-
type inputs without hallucination.
Illegal Sentence Noise Operator (ISN).ISN targetsSyntac-
tic Filtering Faultsby inserting syntactically malformed ‘garbage’
sentences constructed by randomly sampling and concatenating
words from a predefined vocabulary of common English tokens
(e.g., articles, verbs, nouns), producing syntactically incoherent
text. Pre-chunk injection distributes this noise across the document,
potentially causing the retriever to index unintelligible chunks.
Post-chunk injection appends garbled text to retrieved chunks,
testing the LLM’s exception handling and garbage-in-garbage-out
resilience.
Counterfactual Noise Operator (CN).CN targetsKnowledge
Conflict Faultsby injecting statements that contradict ground truth.
To produce realistic contradictions, the operator prompts a separate
LLM to generate a plausible counterfactual statement given the orig-
inal fact with the user query. In pre-chunk injection, contradictions
are embedded in the corpus; critically, the retriever may return
the contradiction without the original fact, simulating stale data
scenarios. In post-chunk injection, a contradiction is generated for
and appended to the retrieved chunks, forcing the LLM to arbitrate
immediate logical conflicts between parametric priors and external
evidence.
Supportive Noise Operator (SuN).SuN targetsContext Over-
fitting Faultsby adding topically related but information-void fillers.
The operator heuristically extracts key terms (e.g., domain nouns)
from the source text and inserts them into predefined sentence tem-
plates, producing filler that sounds on-topic but contains no specific
facts. Pre-chunk injection dilutes the corpus, challenging dense re-
trievers that may rank filler-heavy chunks highly due to semantic
overlap. Post-chunk injection appends fillers to retrieved chunks,
testing if the LLM is distracted by relevant-sounding padding.
Orthographic Noise Operator (ON).ON targetsInput Robust-
ness Faultsby applying character-level typos. Pre-chunk injection
degrades the index, testing fuzzy matching between clean queriesand misspelled documents. Post-chunk injection preserves retrieval
but forces the LLM to reason over degraded surface forms.
4.3 Chunking Mutation Operators
These operators alter the segmentation strategy to exposeSegmen-
tation Boundary Faults.
Chunk Merge (CM) & Chunk Split (CSp) Operators.CM
aggregates adjacent chunks into fewer, larger index entries, altering
the granularity of the retrieval index. CM is only applicable in pre-
chunk mode; at the post-chunk stage, the retriever has already
returned individual chunks, so there are no adjacent chunks to
merge. Conversely, CSp subdivides chunks. Pre-chunk injection
fragments the index, increasing the risk of relevant content falling
below retrieval thresholds; post-chunk injection forces the LLM to
synthesise information across a highly fragmented context list.
Chunk Shift Operator (CSh).CSh re-chunks the document
starting from an offset (e.g., 50%). Pre-chunk injection performs
global re-chunking, creating entirely new boundary alignments
across the corpus. Post-chunk injection performs a local token shift,
trimming tokens from the start of a retrieved chunk and prepending
them to the next. This operator directly provokes the “lost-in-the-
middle” phenomenon [ 8] by arbitrarily shifting the position of key
evidence within the context window.
4.4 RAG Configuration Operators
These operators perturb the execution environment to evaluate
the system againstConfiguration Overfitting Faults. By directly
modifying the retrieval hyperparameters 𝜃Rrather than the corpus
text, these operators are exclusively pre-chunk: they require re-
indexing or re-retrieval, which has no analogue at the post-chunk
stage where the retrieved context is already fixed.
Embedding Model Swap (EM).EM replaces the retrieval hy-
perparameter in 𝜃R(e.g., swapping text-embedding-3-small for
large ) to evaluate whether system performance is coupled to a
specific vector space. A robust RAG implementation should exhibit
operational invariance across different embedding geometries.
Top- 𝑘Perturbation (TK).TK modifies the retrieval hyperpa-
rameter in 𝜃Rby altering the number of chunks passed to the
generator. Restricting 𝑘evaluates whether the generator can main-
tain precision when relying on fewer retrieved chunks, whereas
expanding 𝑘tests whether it can isolate relevant evidence from
redundant or distracting context. However, beyond its role as a
mutation operator, we note that changing 𝑘may alsorepaira fault.
As a repair operator, increasing 𝑘may recover missing evidence,
while decreasing 𝑘may filter out distractions. The same type of
top-𝑘change can therefore induce or repair a fault depending on
its context, consistent with the inversive relationship between bugs
and patches observed by Kim et al. [ 6]. This motivates our evalu-
ation of retrieval-budget changes as a repair intervention in RQ3
(see Section 6.3).
4.5 Metamorphic Relationships as Test Oracles
We define MRs that verify the logical consistency between RAG
system executions. Specifically, we implement the metamorphic
predicateP𝛿(defined in Section 3.1) to enforce principles ofseman-
tic invarianceorcontrolled varianceacross mutations. Our oracles

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
are categorised into two classes based on the expected transforma-
tion of the system’s response:
•Invariance Relations (Semantic-Preserving):For a mutation op-
erator 𝛿𝑝that preserves the semantic essence of the corpus to
answer the query, a robust system must exhibit operational
invariance. We expect 𝑦≈𝑦′, where≈denotes semantic equiv-
alence.
•Variance Relations (Semantic-Altering):For a mutation operator
𝛿𝑎, e.g., that injects counterfactual facts related to the query,
the system must update its output to reflect the new state of
the knowledge base. We expect 𝑦′≠𝑦, specifically where 𝑦′
aligns with the injected fact.
It is important to note that, in reality, the classification of an MR
as semantic-preserving or semantic-altering is a function of both
the mutation operator type and its specific parameters. For example,
with the ISN operator, a 5% noise injection ratio is intended to be
semantic-preserving, whereas a 95% injection (replacing most con-
tent with garbled text) is likely semantic-altering. Evaluating both
the mutation’s effect and the system’s response requires judgements
that go beyond simple string matching. We employ an LLM-as-a-
Judge [ 21] (detailed in Section 5.2.3) in two roles: (1) asemantic
preservation judgethat verifies, for each mutant, whether the factual
evidence required to answer the query remains intact after the per-
turbation, determining the actual semantics-preserving or -altering
status rather than relying on the operator’s intended category; and
(2) ananswer equivalence judgethat assesses whether𝑦and𝑦′are
semantically equivalent given the query and context, implementing
the≈/≠relations above.
When the RAG systemabstains, the verdict is assigned determin-
istically without invoking the LLM judge. For semantics-preserving
mutations, abstention constitutes an MR violation; for semantics-
altering mutations, abstention is treated as MR satisfaction.
5 Experimental Setup
5.1 Research Questions
We structure our empirical study around the following three re-
search questions:
RQ1 (Fault Detection):Can our metamorphic framework detect
RAG systems’ faults?We quantify the system’s susceptibility to
noise and structural perturbations by investigating the rates of MR
violations per mutation operator. Additionally, we conduct a com-
parative analysis between pre-chunk (system-level) and post-chunk
(input-level) mutations to assess their respective effectiveness and
identify the overlap in the faults detected.
Lastly, we conduct a human validation study to evaluate the
reliability of our automated oracle. By measuring the alignment
between human annotators and the LLM-as-a-Judge regarding both
the semantic preservation of the mutation and the equivalence
of the generated answers, we aim to prove that our automated
evaluation framework is trustworthy.
RQ2 (Comparison):Do our MRs reveal the faults that existing RAG
evaluation metrics overlook?We compare our MRs with four repre-
sentative snapshot metrics from RAGAS. This comparison does not
treat RAGAS as a robustness oracle: RAGAS evaluates an individual
answer and its retrieved context, whereas our MRs evaluate the
relationship between two executions before and after a controlledmutation. We therefore examine whether snapshot metrics detect
the same evolution-induced faults exposed by relational testing.
Using ground-truth correctness as the meta-evaluation oracle, we
compute precision, recall, and F1-score for each evaluator and anal-
yse their fault overlap using Venn diagrams. We also assess the
stability of the MR oracle on RepLiQA, where answer-equivalence
judgements require LLM-based evaluation. We stratify-sample 100
cases by mutation scope, operator, initial verdict, and semantic-
preservation label, then repeat the final MR judge verdict five times
per case.
RQ3 (Repair):Can identified faults be mitigated by retrieval, gen-
eration, or reranking interventions?We investigate the persistence
of identified faults under three repair interventions: increasing the
retrieval budget, upgrading the generator model, and inserting an
LLM-based reranking stage before generation. For pre-chunk faults,
we evaluate direct retrieval with 𝑘= 3where applicable and with
𝑘= 5, together with the reranking. For post-chunk faults, we eval-
uate an upgrade from GPT-5-mini to GPT-5.4 and reranking over
the mutated retrieved context. The reranking strategy partitions
its candidates into at most 30 evidence units, uses an LLM to score
each question-unit pair from 0 to 5, and selects up to six of the
highest-ranked units for answer regeneration.
5.2 Experimental Procedure
Our evaluation employs a multi-stage pipeline designed to iso-
late the causal impact of mutations on RAG behaviour. We utilise
datasets (detailed in Section 5.4) comprising a query 𝑞, a corpusC
(documents or relevant information), and a GT answer 𝑎𝑔𝑡. While
a primary advantage of our metamorphic approach is its indepen-
dence from GT labels in production, we leverage 𝑎𝑔𝑡in the study
to facilitate a rigorous meta-evaluation ofevaluators.
5.2.1 Filtering for Context Dependency.To ensure our test suite
targets the retrieval pipeline rather than the generator’s parametric
memory, we apply aclosed-book filter. We first execute the genera-
tor in isolation (without C):𝑦𝑐𝑙𝑜𝑠𝑒𝑑 =G(𝑞) . We retain only those
triplets⟨𝑞,C, 𝑎 𝑔𝑡⟩in the dataset where the closed-book response
𝑦𝑐𝑙𝑜𝑠𝑒𝑑 isincorrect, i.e., 𝑦𝑐𝑙𝑜𝑠𝑒𝑑 ≠𝑎𝑔𝑡, while withC, i.e., generating
the answer using the full RAG pipeline, 𝑦𝑏𝑎𝑠𝑒=S(𝑞,C) success-
fully produces the ground-truth answer 𝑎𝑔𝑡, i.e., 𝑦𝑏𝑎𝑠𝑒≈𝑎𝑔𝑡. This
step ensures that the retrieved evidence is anecessary condition
for correctness, allowing us to attribute subsequent failures to the
perturbation𝛿.
5.2.2 Metamorphic Execution.For each filtered instance, we treat
the successful baseline output 𝑦𝑏𝑎𝑠𝑒as our reference point ( 𝑦in
Figure 1). We then apply a mutation 𝛿to generate a post-mutation
output 𝑦′. Unlike traditional benchmarks that compare 𝑦′to a static
ground truth 𝑎𝑔𝑡, our metamorphic oracle evaluates the logical
consistency between 𝑦𝑏𝑎𝑠𝑒and𝑦′. We say a fault is detected by our
framework if the answer pair⟨𝑦 𝑏𝑎𝑠𝑒,𝑦′⟩violates the metamorphic
predicateP 𝛿.
5.2.3 Use of LLM-as-a-Judge.The complexity of evaluating answer
pairs (e.g., comparing 𝑎𝑔𝑡with 𝑦𝑏𝑎𝑠𝑒, or𝑦𝑏𝑎𝑠𝑒with 𝑦′) depends heav-
ily on the output format. For numerical outputs in our financial
dataset, T2-RAGBench (see Section 5.4), we perform direct com-
parison to determine correctness. However, when the generator

When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations ASE ’26, October 12–16, 2026, Munich, Germany
produces natural language outputs, assessing logical equivalence
becomes non-trivial.
To address this, we employ an LLM-as-a-Judge [ 21]. The judge
is provided with the query 𝑞, the retrieved contexts, information
about the perturbation 𝛿, and the respective answer pairs to de-
termine their relationship under P𝛿. In addition to evaluating the
system’s output, we utilise the LLM-as-a-Judge to verify the nature
of the mutation itself. Specifically, the judge determines whether
a perturbation is semantic-preserving (where the factual evidence
required to answer 𝑞remains intact) or semantic-altering. This
automated verification is critical because the disruptiveness of the
mutation operators is highly sensitive to their parameterisation.
For instance, operators such as SeN can vary from subtle stylistic
changes to highly disruptive perturbations if a high volume of noise
is injected into the document.
Although the final verdict of the judge is binary, we utilise Likert
scoring-based prompting to capture logical and linguistic nuances,
a method shown to be more effective than simple prompting [ 17].
For example, in the case of semantic-altering mutations, the judge
evaluates how effectively the system updates its response to reflect
new or counterfactual information on a 1–5 scale. High scores of 4–5
indicate successful adaptation, where 𝑦′clearly reflects the mutated
context or appropriately acknowledges conflicting information,
satisfying theP𝛿relation. Conversely, scores of 1–2 indicate a
failure to adapt, where 𝑦′remains essentially identical to 𝑦𝑏𝑎𝑠𝑒
despite the change in evidence, resulting in an MR violation.
We acknowledge that relying on the LLM-as-a-Judge introduces
a potential threat to validity due to hallucinations of the judges.To
assess this threat, we conduct two complementary analyses. First,
two authors independently annotated a random sample of 100
RepLiQA mutation pairs, for both semantic preservation and answer
equivalence. Second, we assessed oracle stability using 100 RepLiQA
cases stratified by mutation scope, operator, initial verdict, and
semantic-preservation label, repeating the final MR judgement five
times for each case. Section 6.1 reports the human agreement and
oracle stability results.
5.3 Meta-Evaluation for Enabling Comparison
To quantify the diagnostic accuracy of our framework, we introduce
a meta-evaluation procedure that assesses the effectiveness of our
framework against established RAG evaluation metrics, response
relevancy, faithfulness, context precision, and context recall, pro-
vided by the RAGAS framework [ 4]. This stage effectively serves to
evaluate the evaluators by measuring how reliably each oracle de-
tects system failures. While a core advantage of MR is its ability to
operate without a ground-truth oracle, we selectively re-introduce
𝑎𝑔𝑡here as an objective baseline to determine absolute factual cor-
rectness under mutation. With GT, we map the verdicts of both
RAGAS and our MR framework into a formal classification of True
Positives (TP), False Positives (FP), True Negatives (TN), and False
Negatives (FN), and corresponding recall, precision, and F1-score.
For our framework, we compare the metamorphic verdict ( P𝛿)
against the logical relationship between the post-mutation out-
put ( 𝑦′) and the ground truth ( 𝑎𝑔𝑡). For the RAGAS metrics, we
define a fault detection as any score falling below a set threshold.
This mapping allows us to identifysilent faults. For instance, if asemantic-preserving mutation results in an output 𝑦′that contra-
dicts 𝑎𝑔𝑡while a RAGAS metric maintains a high score, the metric
suffers an FN, failing to register a structural fragility that our MR
framework is designed to expose.
5.4 Datasets
We use two primary benchmarks to evaluate and compare our
metamorphic framework for testing RAG systems across different
domains and reasoning requirements.
5.4.1 T2-RAGBench.The financial question-answering scenario
is based on T2-RAGBench [ 16], a benchmark designed to evaluate
RAG over large collections of financial documents containing both
textual and tabular information.
T2-RAGBench includes four datasets: ConvFinQA [ 1], FinQA [ 1],
TAT-DQA [ 23], and VQAonBD [ 11] that cover heterogeneous finan-
cial sources such as annual reports, regulatory filings, and financial
statements. All four datasets provide predefined test queries and
verified answers. Test queries in these datasets typically require
multi-step reasoning over retrieved evidence, such as locating spe-
cific tables or passages, extracting numerical values, and performing
comparisons or arithmetic operations.
5.4.2 RepLiQA.We utilise the RepLiQA benchmark [ 9], which is
a question-answering dataset specifically curated to benchmark
LLMs on unseen reference content. It comprisessyntheticreference
documents crafted by human annotators, depicting imaginary sce-
narios, people, and places across 17 distinct categories. Because
these fictional entities and events are entirely absent from the inter-
net, they could not have been included in the pre-training corpora
of existing LLMs. Each sample in RepLiQA includes a reference
document, a specific question about the document’s topic, and a
ground-truth answer derived directly from the text. By relying
on purely synthetic content, RepLiQA guarantees that accurate
answers can only be generated if the RAG system successfully re-
trieves the correct document and the LLM accurately comprehends
the provided context, rather than relying on memorised facts.
5.4.3 Filtering, Sampling, and Cost.Although datasets in our eval-
uation were designed to preclude data contamination during LLM
training, we nevertheless subject all datasets to the filtering proce-
dure described in Section 5.2.1. We retain only the context depen-
dent triplets⟨𝑞,C, 𝑎 𝑔𝑡⟩. This filtering reduces the initial datasets to
a pool of qualified instances: 1,868 for ConvFinQA, 3,139 for FinQA,
3,412 for TAT-DQA, 7,791 for VQAonBD, and 6,888 for RepLiQA.
However, even with the refined pools, evaluating the entire pop-
ulation remains economically (or computationally) infeasible to us.
Each base instance yields 11 mutated executions. Applying these
across two mutation modes does not strictly double the count: since
three operators are inapplicable to post-chunk mode (Section 4), we
execute11+8=19total mutants per triplet. Every mutant requires
LLM API calls for both execution and evaluation. Our framework
uses 2–3 LLM calls per mutation test: one for RAG generation on
the mutated corpus, one for the semantic preservation judge, and
one for the answer equivalence judge (the latter is replaced by exact
numeric comparison on financial datasets, reducing the count to
2). For running RAGAS, each of its four metrics invokes the LLM

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
Table 1: Pre-chunk and Post-chunk MR Violation Rates (%) with Fault Overlap (Jaccard %)
RepLiQA ConvFinQA FinQA TAT-DQA VQAonBD
MO Pre Post J Pre Post J Pre Post J Pre Post J Pre Post J
CM 8.1 – – 5.0 – – 2.3 – – 4.8 – – 5.1 – –
CSh 9.0 8.3 41 8.0 6.0 11 8.3 6.0 10 7.3 5.3 6 11.0 1.3 6
CSp 5.0 9.0 27 9.3 4.3 17 7.7 2.0 4 5.0 2.3 10 14.7 2.0 6
CN 11.5 18.0 5 19.7 23.3 12 18.0 26.0 15 24.7 22.3 11 27.6 27.0 17
DN 7.4 9.0 69 4.1 4.3 25 4.0 2.7 6 4.7 3.3 35 1.3 2.3 25
EM 2.3 – – 13.6 – – 4.0 – – 0.0 – – 12.5 – –
ISN 8.2 9.3 72 6.3 4.3 12 7.6 2.7 19 3.4 3.3 12 1.9 2.0 22
ON 9.0 9.7 87 3.7 4.3 50 3.3 2.4 13 3.7 3.3 31 2.3 1.9 38
SeN 11.4 9.3 62 8.4 4.3 21 6.0 2.7 10 4.3 3.0 25 2.9 1.0 22
SuN 7.8 9.3 52 5.3 4.3 35 3.6 3.0 20 3.5 3.0 29 1.4 2.0 25
TK 9.3 – – 2.1 – – 2.8 – – 4.9 – – 3.8 – –
Avg. 8.1 10.2 45 7.8 6.9 18 6.2 5.9 12 6.0 5.8 15 7.7 4.9 15
Table 2: Mutation parameters used in the experiments.
MO Pre-chunk Post-chunk
SeN0.20𝑛 𝑠sentences 1 sentence/chunk
DN𝑛 𝑠/3fragments𝑛 𝑠/3fragments
ISN0.15𝑛 𝑠sentences 1 sentence/chunk
CN1 passage/document 1 passage/chunk
SuN0.20𝑛 𝑠sentences 1 sentence/chunk
ONError rate 0.01 Error rate 0.01
CMMerge factor 2 N/A
CSpSplit factor 2 Split factor 2
CShOffset 0.5 (128 words) 10-token shift
EMsmall→largeN/A
TK𝑘=1→3N/A
𝑛𝑠denotes the number of sentences. Sentence-based quantities are rounded down
with a minimum of one. EM changes between the text-embedding-3-small and
text-embedding-3-largemodels.
independently (Faithfulness: 2 calls for statement extraction and in-
ference;Answer Relevancy: 3 calls for question generation;Context
PrecisionandContext Recall: 1 call each), totalling 7 LLM calls per
evaluation point. For a campaign of 𝑁entries and 𝑀operators, we
use𝑁×𝑀× 3LLM calls, while RAGAS uses 𝑁×𝑀× 7. We note that
this cost breakdown is not a comparative efficiency claim between
ours and RAGAS, which serve different evaluation purposes; rather,
it quantifies the cumulative API cost that necessitated sampling
instead of exhaustive evaluation. To keep the experiment tractable
while preserving statistical validity, we restrict the scope of the eval-
uation. From each of the five filtered datasets, we sample 𝑛= 300
base instances. This yields300 ×19=5,700mutants per dataset,
and a total of 28,500 mutants across five datasets. All results are
reported based on these samples and mutants.
5.5 Configurations
We run closed-book filtering ten times, retaining only those triplets
that consistently satisfy these conditions across all trials. In the
study, the default top- 𝑘is set to1. Table 2 reports the parameter
settings used in the experiment. For pre-chunk mutations,SeNand
SuNinsert sentences corresponding to 20% of the original sentencecount,ISNuses 15%, andDNinserts one fragment per three sen-
tences. At post-chunk scope,SeN,ISN, andSuNinsert one sentence
per retrieved chunk, whileDNinserts one fragment for every three
sentences.CNadds one counterfactual passage per document or
retrieved chunk, andONapplies a character-level error probability
of 0.01. The structural operators use a merge or split factor of 2,
whileCShuses a 0.5 offset before chunking and a 10-token shift
after retrieval. Finally,EMchanges the embedding model from
text-embedding-3-small totext-embedding-3-large , andTK
increases 𝑘from 1 to 3. In the repair study, we do not evaluate
𝑘= 3as a repair for faults induced byTKas it reproduces the
mutated retrieval setting; we still evaluate these faults using 𝑘= 5
and LLM-based reranking. Sentence-derived quantities are rounded
down with a minimum of one.
Default RAG generation, counterfactual generation, closed-book
filtering, semantic-preservation judging, and answer-equivalence
judging use OpenAI’s GPT-5-mini3. As the GPT-5 family is com-
posed of reasoning models, the API does not support user con-
figurable sampling parameters such as temperature or top- 𝑝; we
therefore use the API defaults. We use text-embedding-3-small4
as the embedding model. We use a RAGAS fault threshold of0 .3.
While the artefact also includes results for thresholds of0 .0and0 .5,
these variations do not significantly alter the findings. All exper-
iments are conducted on a machine equipped with an Apple M3
Max and 36 GB of RAM, running Tahoe 26.2.
6 Results
6.1 RQ1 (Fault Detection)
Table 1 reports MR violation rates and fault overlap (Jaccard index,
J) by mutation operator and scope. Average violation rates range
from 6.0% to 8.1% for pre-chunk mutations, and from 4.9% to 10.2%
for post-chunk mutations.CNconsistently produces the highest
violation rates across all datasets and both scopes, reaching 27.6%
(pre) and 27.0% (post) on VQAonBD, showing that injecting counter-
factual content directly degrades both retrieval relevance and gen-
eration faithfulness. Chunking operatorsCShandCSpalso produce
3gpt-5-mini-2025-08-07
4https://platform.openai.com/docs/models/text-embedding-3-small

When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations ASE ’26, October 12–16, 2026, Munich, Germany
elevated pre-chunk rates (e.g., 14.7% forCSpon VQAonBD), because
altering chunk boundaries before indexing changes which content
is retrieved. In contrast, surface-level operators (ON,SuN,DN) yield
lower violation rates overall, indicating that RAG pipelines tolerate
minor formatting and typographic perturbations well.
To illustrate, consider two RepLiQA faults (i.e., True Positives).
When asked “What was a significant obstacle Emily faced during
her marathon training?”, the baseline system correctly answered
“a minor ankle sprain.” After a semantics-preservingCShmutation
that merely shifted chunk boundaries, the system instead answered
“balancing a full-time job with her running schedule,” retrieving
a different passage fragment and producing a factually different
answer despite no change to the underlying content. Conversely,
when a semantics-alteringCNmutation injected counterfactual
information about a yoga studio’s new offering, the system was
asked “What new offering did ZenSpace Yoga launch in early 2024?”
and still answered “virtual reality meditation classes” verbatim,
ignoring the contradictory evidence now present in the retrieved
context.
The Jaccard column (J) reveals that pre-chunk and post-chunk
mutations largely expose different faults. Average overlap is only
12–18% on four of five datasets, with RepLiQA being the exception
at 45%. Surface-level operators (ON,ISN,DN) show relatively high
overlap because these perturbations are too minor to meaningfully
alter retrieval. Both scopes end up probing the same generator-side
sensitivity, and the fault sets converge. In contrast,CNexhibits
very low overlap. Overall, the low average overlap confirms that
the two mutation scopes are complementary: using both would
broaden fault coverage beyond what either scope achieves alone.
RepLiQA’s higher overlap (45%) can be explained by its design,
where each query targets one short document, so the retriever is
likely to return similar chunks regardless of whether the mutation
is applied before or after chunking.
To validate the automated verdicts, two authors independently
labelled 100 randomly-sampled RepLiQA mutation pairs for answer
equivalence and semantic preservation. Inter-annotator agreement
was 98% ( 𝜅= 0.92) and 99% ( 𝜅= 0.96), respectively.5Each annota-
tor agreed with the LLM judge 93–95% of the time ( 𝜅= 0.68–0 .77).
Most disagreements (7 out of 8) occurred when the LLM judged
answers as equivalent but annotators deemed them different, typi-
cally involving partial matches where the mutated answer omitted
details or added precision (e.g., “late 2023” vs. “September 12, 2023”).
Annotator confidence on these disputed cases averaged 2.50 out of
5, compared to 4.58 on agreed items, indicating that disagreements
cluster at genuinely ambiguous boundaries rather than reflecting
systematic bias.
To assess whether stochastic variation in judging affects the
detected violations, we repeated the final MR verdict five times for
100 stratified RepLiQA cases. 94 cases received unanimous verdicts,
and the majority verdict agreed with the initial verdict in 97 cases.
SATISFIED/VIOLATED flips occurred in six cases, indicating that
judgement instability is limited but not absent.
5𝜅denotes Cohen’s kappa, a chance-corrected measure of inter-rater agreement [2].
2 0171
200
00
02
00
0
57
15
2913
0 5181
21
218
2511
162MR
Faithfulness
Answer Relevancy
Context Precision
Context Recall(a) RepLiQA
0 0000
000
00
00
00
0
1
0
10134
17 12160
00
078
1410
25MR
Faithfulness
Answer Relevancy
Context Precision
Context Recall (b) ConvFinQA
Figure 2: Fault-overlap Venn diagrams for RepLiQA and Con-
vFinQA.
Answer to RQ1:Our metamorphic framework detected faults
across all five datasets (violation rates 4.9–10.2%), with pre- and
post-chunk mutations exposing distinct faults (Jaccard overlap
12–45%), confirming both scopes are needed. A further human
study validated the automated verdicts. Human validation yielded
93%–95% agreement with the automated judge. Across repeated
oracle calls, 94% of cases received unanimous verdicts and 97%
retained the initial verdict under majority voting.
6.2 RQ2 (Comparison)
Table 3 compares our MR-based framework against four RAGAS
metrics in a meta-evaluation against ground-truth correctness. MR
achieves near-perfect F1 across all datasets: 0.927 on RepLiQA and
1.000 on others.6In comparison, the best-performing RAGAS metric
isContext Recall, which reaches its highest F1 of 0.570 on RepLiQA.
All other RAGAS metrics remain below 0.46 F1 across all datasets.
Figure 2 illustrates the fault-overlap structure for RepLiQA and
ConvFinQA. The remaining datasets exhibit similar patterns and are
provided in our artefact (see Section 9). The diagrams depict only
true-positive cases. Although metrics such asAnswer Relevancy
appear to offer broad coverage, Table 3 shows that all RAGAS
metrics suffer from low precision and recall, so their apparent
TP-only coverage overstates actual reliability. RAGAS metrics also
largely overlap with each other, while MRs capture additional faults
that RAGAS misses entirely.
These results demonstrate that metamorphic testing provides
a more reliable diagnostic signal than score-based evaluation. By
checking behavioural consistency under controlled perturbations
rather than estimating quality from a single response, MRs achieve
both high recall and high precision.
Answer to RQ2:Our metamorphic framework substantially
outperforms RAGAS metrics, achieving F1 scores of 0.927–1.000
compared to 0.083–0.570 for RAGAS. The fault-overlap analysis
shows that RAGAS metrics are largely redundant with each other,
whereas MRs additionally detect faults that RAGAS misses.
6Note that the perfect F1 on the financial datasets is expected: their answers are single
numeric values, so MR validation via exact match coincides with meta-evaluation with
GT, leaving no room for classification error (see Section 8).

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
Table 3: Meta-evaluation: MR vs. RAGAS metrics. Recall, Precision, and F1 computed from TP/TN/FP/FN classifications against
ground-truth correctness.
RepLiQA ConvFinQA FinQA TAT-DQA VQAonBD
Metric Recall Prec. F1 Recall Prec. F1 Recall Prec. F1 Recall Prec. F1 Recall Prec. F1
Faithfulness 0.575 0.379 0.456 0.373 0.077 0.128 0.625 0.085 0.150 0.605 0.103 0.176 0.492 0.086 0.147
Answer Relevancy 0.707 0.220 0.336 0.992 0.070 0.131 1.000 0.060 0.113 0.990 0.060 0.114 0.987 0.065 0.122
Context Precision 0.554 0.369 0.443 0.181 0.065 0.096 0.401 0.103 0.164 0.113 0.066 0.083 0.155 0.089 0.113
Context Recall 0.641 0.514 0.570 0.209 0.060 0.093 0.645 0.099 0.171 0.254 0.090 0.133 0.272 0.118 0.165
Ours 0.894 0.9630.927 1.000 1.0001.000 1.000 1.0001.000 1.000 1.0001.000 1.000 1.0001.000
6.3 RQ3 (Repair)
Table 4 reports all repair configurations. For pre-chunk faults, direct
retrieval with 𝑘= 3and 𝑘= 5repairs 32.5% and 32.5%, respectively,
while reranking repairs 30.8%. For post-chunk faults, upgrading
the generator repairs 32.6%, while reranking repairs 23.2%. Alto-
gether, the considered configurations repair 43.1% of faults. These
aggregate results mask substantial variation across datasets. Union
repair rates range from 47.3% to 63.0% across the four financial
datasets but reach only 10.2% on RepLiQA. For pre-chunk faults,
direct retrieval achieves 28.8% to 51.8% on the financial datasets,
while both retrieval budgets repair 4.0% on RepLiQA. Reranking
similarly achieves 30.2% to 49.1% on the financial datasets but only
6.5% on RepLiQA. For post-chunk faults, the generator upgrade
repairs 37.0% to 48.2% on the financial datasets but only 7.8% on
RepLiQA, while reranking achieves 26.1% to 35.5% and 5.8%, re-
spectively. The consistently low rates on RepLiQA indicate faults
that broader retrieval, stronger generation, and reranking cannot
adequately address.
Repairability also varies by mutation operator. Among pre-chunk
faults,EMis the most repairable through direct retrieval with 𝑘= 5
(49.1%), whileCSpbenefits most from reranking (49.6%). In contrast,
CNremains difficult for both interventions, with repair rates of
16.4% and 19.7%, respectively. For post-chunk faults,CNbenefits
most from the generator upgrade (37.3%) but least from reranking
(16.6%), whereasSuNachieves the highest reranking repair rate
(36.9%). These differences show that the considered interventions
address distinct failure modes rather than providing a uniformly
effective repair.
As an example of a successful repair, a semantics-preserving
SeNpost-chunk mutation caused the system to abstain (“I cannot
answer this question based on the provided context”); upgrading the
generator recovered the correct answer with full detail. Conversely,
a semantics-altering CN mutation that injected a counterfactual
about an app lacking currency conversion features persisted after
repair: the stronger model still adopted the contradictory evidence,
producing an answer that directly opposed the ground truth.
While the achieved repair rates remain partial, we note two
limitations of this analysis. First, we evaluate only three general-
purpose repair interventions, whereas developers may devise ad-
ditional strategies based on their system and domain knowledge.
Second, we do not know what proportion of the detected faults
practitioners would consider worth repairing in a real deployment.
Despite these limitations, the 43.1% union repair rate indicates thatthe evaluated interventions can address a substantial subset of the
detected faults.
Answer to RQ3:Repair is useful but partial. For pre-chunk faults,
setting 𝑘= 3,𝑘= 5, and reranking repair 32.5%, 32.5%, and 30.8%,
respectively. For post-chunk faults, the generator upgrade and
reranking repair 32.6% and 23.2%. The union repairs 43.1% of
evaluated faults, but the 10.2% repair rate on RepLiQA shows that
some corpus-evolution faults persist across all interventions.
7 Related Work
Frameworks such as RAGAS [ 4] and ARES [ 12] evaluate retrieval
and generation quality for individual query–answer pairs against
afixedcorpus snapshot. While effective for point-in-time bench-
marking, they evaluate outputs in isolation and cannot capture
whether a system behaves consistently as the corpus evolves. Our
work introduces a relational, corpus-evolution-aware paradigm
that surfaces failures invisible to these static metrics.
MetaRAG [ 15] is closer to our setting as it also applies metamor-
phic testing to RAG systems, but it places the testing boundary at
the response level: it mutates factoids extracted from a generated an-
swer and checks whether the variants are supported by the retrieved
context. In contrast, our system under test is the RAG pipeline under
corpus evolution. Pre-chunk mutations alter documents, chunking,
the index, or retrieval configuration before re-executing ingestion,
retrieval, and generation, while post-chunk mutations isolate gen-
erator behaviour after retrieval. Thus, MetaRAG checks whether
an existing answer is supported by its evidence, whereas our frame-
work tests whether a RAG pipeline behaves consistently when the
knowledge state changes.
Prior work shows that LLMs are easily distracted by irrelevant
retrieved passages [ 14], exhibit positional bias when key evidence
appears mid-context [ 8], and suppress correct parametric knowl-
edge when retrieved content contradicts it [ 19]. The most directly
related work is Wu et al. [ 18], which taxonomises RAG noise types
empirically. Our fault taxonomy (Section 3.2) extends theirs by
adding Format Intolerance and Architectural Brittleness classes,
and makes each class an executable mutation operator with an
automatic oracle, rather than a descriptive analysis.
A separate line of work investigates adversarial attacks on re-
trieval based systems. Zhong et al. [ 22] show that injecting a small
number of adversarial passages into a dense retrieval corpus can
mislead retrieval models with high success rates, even generalising
to unseen queries and domains. PoisonedRAG [ 24] extends this

When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations ASE ’26, October 12–16, 2026, Munich, Germany
Table 4: Fault repair success by dataset and scope. Each cell reports repaired/applicable faults (rate); the union counts a fault as
repaired if any applicable configuration succeeds.
Pre-chunk Post-chunk All
Dataset 𝑘=3𝑘=5Rerank GPT-5.4 Rerank Union
RepLiQA 10/248 (4.0%) 11/276 (4.0%) 18/276 (6.5%) 19/243 (7.8%) 14/243 (5.8%) 53/519 (10.2%)
ConvFinQA 102/212 (48.1%) 113/218 (51.8%) 107/218 (49.1%) 80/166 (48.2%) 59/166 (35.5%) 242/384 (63.0%)
FinQA 67/196 (34.2%) 59/205 (28.8%) 62/205 (30.2%) 61/142 (43.0%) 37/142 (26.1%) 164/347 (47.3%)
TAT-DQA 78/190 (41.1%) 78/204 (38.2%) 71/204 (34.8%) 51/138 (37.0%) 43/138 (31.2%) 182/342 (53.2%)
VQAonBD 86/208 (41.3%) 102/215 (47.4%) 86/215 (40.0%) 52/118 (44.1%) 34/118 (28.8%) 189/333 (56.8%)
Overall 343/1,054 (32.5%) 363/1,118 (32.5%) 344/1,118 (30.8%) 263/807 (32.6%) 187/807 (23.2%) 830/1,925 (43.1%)
to end-to-end RAG, achieving 90% attack success by injecting as
few as five crafted texts per target question into a knowledge base
of millions. These works consider adversarial robustness, whereas
our framework tests robustness under corpus evolution (reformat-
ting, updates, noise). The two perspectives are complementary:
poisoning attacks expose security vulnerabilities, while our meta-
morphic approach surfaces engineering faults that arise during
corpus maintenance.
8 Threats to Validity
Internal validity.The LLM-based judge may misclassify semantic
preservation or answer equivalence. Our human study (Section 6.1)
found agreement between 93%–95%, although we acknowledge
that the random sample did not cover all case categories evenly:
semantics-altering cases comprised 12.0% of the sample compared
with 12.3% of the evaluated RepLiQA population, while violated
cases comprised 4.0% compared with 9.1%. We do not control or
measure all sources of stochasticity across the complete pipeline:
repeating mutation generation, e.g., the generation of counterfac-
tuals, may produce a different execution. Our stability analysis
isolates one component: we held each sampled mutation, retrieved
context, and generated answer fixed and repeated only the final
MR judgement five times. We observed unanimous verdicts in 94%
of cases, while the majority verdict agreed with the initial verdict
in 97%. The four financial datasets use exact numeric comparison
for answer equivalence, reducing their reliance on the judge. The
closed-book filter may retain queries answerable by chance; we
mitigate this by running it ten times.
External validity.Our evaluation spans five datasets in two do-
mains but uses a single RAG architecture and LLM. Different re-
trieval strategies or models may yield different fault profiles. We
sample 300 base instances per dataset due to the cost of running
both our framework and RAGAS across all mutation operators
and scopes (28k mutants total). However, the consistent patterns
observed across five independently sampled datasets provide confi-
dence in the robustness of the findings. In future work, the cost of
mutation campaigns could be reduced by prioritising the most effec-
tive operators (e.g.,CN,CSh) based on the violation rates reported
in our results.
Construct validity.The RQ2 meta-evaluation uses ground-truth
to classify faults. For RepLiQA this again relies on the LLM judge,
but exact-match results on financial datasets corroborate the same
findings. A potential concern is circularity in the meta-evaluation:the filtering procedure (Section 5.2) already verifies that the baseline
answer 𝑦matches the ground truth 𝑎𝑔𝑡, so for semantics-preserving
mutations the MR check ( 𝑦′≈𝑦) and the GT check ( 𝑦′≈𝑎𝑔𝑡) are
correlated by transitivity. We note three mitigating factors. First,
the filtering is necessary for experimental validity, not to advantage
MR: without it, mutations would be applied to already-incorrect
baselines, making fault detection uninterpretable. Both MR and
RAGAS are evaluated on the same filtered instances. Second, the
correlation does not hold for semantics-altering mutations, where
the GT check verifies whether 𝑦′reflects the injected counterfactual,
independently of 𝑦≈𝑎 𝑔𝑡. Third, RepLiQA empirically demonstrates
non-trivial evaluation: MR achieves 0.927 F1, not 1.000, confirming
that the LLM-based equivalence judgement ( 𝑦′≈𝑦) and the GT
comparison ( 𝑦′≈𝑎𝑔𝑡) can genuinely disagree when answers are
natural language rather than single numeric values.
9 Conclusion
We presented a metamorphic testing framework for evaluating RAG
systems under corpus evolution. Rather than assessing snapshot
correctness, the framework defines metamorphic relations that ver-
ify behavioural consistency across 11 mutation operators applied
at two scopes: pre-chunk (index-level, exercising the full pipeline)
and post-chunk (context-level, isolating the generator). Across five
datasets, our framework detects faults (violation rates from 4.9% to
10.2%) that standard RAGAS metrics miss, with human validation
confirming the automated verdicts. The two mutation scopes ex-
pose complementary faults. Retrieval-budget increase, generator
upgrade, and LLM-based reranking together resolve 43.1% of de-
tected faults. While this is encouraging, their limited effectiveness
on RepLiQA points to future work on alternative repair strategies.
Data Availability
The implementation and the data are available.7
References
[1]Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al .
2021. Finqa: A dataset of numerical reasoning over financial data. InProceedings
of the 2021 Conference on Empirical Methods in Natural Language Processing.
3697–3711.
[2]Jacob Cohen. 1960. A coefficient of agreement for nominal scales.Educational
and psychological measurement20, 1 (1960), 37–46.
[3]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert:
Pre-training of deep bidirectional transformers for language understanding. In
7https://github.com/dbr7/RAG-metamorphic-mutation

ASE ’26, October 12–16, 2026, Munich, Germany Kim et al.
Proceedings of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1 (long and
short papers). 4171–4186.
[4]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas:
Automated evaluation of retrieval augmented generation. InProceedings of the
18th Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations. 150–158.
[5]Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei
Tong, and Guoping Hu. 2025. Retrieval Augmented Generation Evaluation in
the Era of Large Language Models: A Comprehensive Survey.arXiv preprint
arXiv:2504.14891(2025).
[6]Jinhan Kim, Jongchan Park, and Shin Yoo. 2023. The inversive relationship
between bugs and patches: An empirical study. In2023 IEEE International Confer-
ence on Software Testing, Verification and Validation Workshops (ICSTW). IEEE,
314–323.
[7]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[8]Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts.Transactions of the association for computational linguistics12
(2024), 157–173.
[9]Joao Monteiro, Pierre-Andre Noel, Etienne Marcotte, Sai Rajeswar, Valentina
Zantedeschi, David Vazquez, Nicolas Chapados, Christopher Pal, and Perouz
Taslakian. 2024. Repliqa: A question-answering dataset for benchmarking llms
on unseen reference content.Advances in Neural Information Processing Systems
37 (2024), 24242–24276.
[10] Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai
Zeng, Luna Dong, Yinglong Xia, Krishnaram Kenthapadi, et al .2025. Towards
trustworthy retrieval augmented generation for large language models: A survey.
arXiv preprint arXiv:2502.06872(2025).
[11] Sachin Raja, Ajoy Mondal, and CV Jawahar. 2023. Icdar 2023 competition on visual
question answering on business document images. InInternational Conference
on Document Analysis and Recognition. Springer, 454–470.
[12] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2023.
ARES: An Automated Evaluation Framework for Retrieval-Augmented Genera-
tion Systems. arXiv:2311.09476 [cs.CL]
[13] Sergio Segura, Gordon Fraser, Ana B Sanchez, and Antonio Ruiz-Cortés. 2016. A
survey on metamorphic testing.IEEE Transactions on software engineering42, 9
(2016), 805–824.[14] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi,
Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily
distracted by irrelevant context. InInternational Conference on Machine Learning.
PMLR, 31210–31227.
[15] Channdeth Sok, David Luz, and Yacine Haddam. 2025. MetaRAG: Metamor-
phic Testing for Hallucination Detection in RAG Systems.arXiv preprint
arXiv:2509.09360(2025).
[16] Jan Strich, Enes Kutay Isgorur, Maximilian Trescher, Chris Biemann, and Mar-
tin Semmann. 2025. T2-RAGBench: Text-and-Table Benchmark for Evaluating
Retrieval-Augmented Generation.arXiv preprint arXiv:2506.12071(2025).
[17] Victor Wang, Michael JQ Zhang, and Eunsol Choi. 2025. Improving llm-as-a-
judge inference with the judgment distribution.arXiv preprint arXiv:2503.03064
(2025).
[18] Jinyang Wu, Shuai Zhang, Feihu Che, Mingkuan Feng, Pengpeng Shao, and
Jianhua Tao. 2025. Pandora’s box or aladdin’s lamp: A comprehensive analysis
revealing the role of rag noise in large language models. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 5019–5039.
[19] Kevin Wu, Eric Wu, and James Zou. 2024. Clasheval: Quantifying the tug-of-
war between an llm’s internal prior and external evidence.Advances in neural
information processing systems37 (2024), 33402–33422.
[20] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu,
Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan
Catanzaro. 2023. Retrieval meets long context large language models.arXiv
preprint arXiv:2310.03025(2023).
[21] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .2023. Judging
llm-as-a-judge with mt-bench and chatbot arena.Advances in neural information
processing systems36 (2023), 46595–46623.
[22] Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. 2023. Poison-
ing retrieval corpora by injecting adversarial passages. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. 13764–13775.
[23] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng
Chua. 2022. Towards complex document understanding by discrete reasoning. In
Proceedings of the 30th ACM International Conference on Multimedia. 4857–4866.
[24] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. {PoisonedRAG}:
Knowledge corruption attacks to {Retrieval-Augmented }generation of large
language models. In34th USENIX Security Symposium (USENIX Security 25).
3827–3844.
Received 2026-03-26; accepted 2026-06-18
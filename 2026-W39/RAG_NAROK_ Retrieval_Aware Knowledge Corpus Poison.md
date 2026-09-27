# RAG-NAROK: Retrieval-Aware Knowledge Corpus Poisoning in RAG with Source-specific Refutation

**Authors**: Abdullahil Kafi, Alvi Ataur Khalil

**Published**: 2026-09-21 22:53:04

**PDF URL**: [https://arxiv.org/pdf/2609.25469v1](https://arxiv.org/pdf/2609.25469v1)

## Abstract
Retrieval augmented generation (RAG) systems have emerged as the dominant architecture for grounding large language model (LLM) outputs in verifiable external knowledge, yet their structural reliance on a dynamic retrieval pipeline introduces a largely unexplored class of adversarial vulnerability. Existing knowledge-base poisoning attacks are fundamentally static. Adversarial documents are pre-computed and injected without any awareness of what the victim system will actually retrieve for a given query, leaving the attack blind to the competitive documentary landscape that surrounds its payload in the generator's context window. Unlike traditional static poisoning attacks that are blind to the retrieved context, we introduce RAG-NAROK (Retrieval-Anchored Generation Negation And Response Quality Collapse), a RAG attack framework that adapts to the query text. RAG-NAROK exploits the transparency inherent in RAG pipeline to first extract the legitimate source identities, then generate Anchor-Specific Refutation documents that explicitly name and devalue retrieved sources while leveraging recency and authority biases to steer the text generation toward a target answer. Our results demonstrate that RAG-NAROK significantly outperforms static baselines across diverse domains, revealing a fundamental tension between RAG transparency and AI security.

## Full Text


<!-- PDF content starts -->

RAG-NAROK: Retrieval-Aware Knowledge Corpus
Poisoning in RAG with Source-specific Refutation
Abdullahil Kafi and Alvi Ataur Khalil
Transformative Innovation for Trustworthy AI and Network Security (TITANS) Lab,
Computer Science, Southern Illinois University, USA
abdullahil.kafi@siu.edu, a.khalil@siu.edu
Abstract—Retrieval augmented generation (RAG) systems have
emerged as the dominant architecture for grounding large
language model (LLM) outputs in verifiable external knowledge,
yet their structural reliance on a dynamic retrieval pipeline
introduces a largely unexplored class of adversarial vulnerability.
Existing knowledge-base poisoning attacks are fundamentally
static. Adversarial documents are pre-computed and injected
without any awareness of what the victim system will actually
retrieve for a given query, leaving the attack blind to the compet-
itive documentary landscape that surrounds its payload in the
generator’s context window. Unlike traditionalstaticpoisoning
attacks that are blind to the retrieved context, we introduce RAG-
NAROK(Retrieval-Anchored Generation Negation And Response
Quality Collapse), a RAG attack framework that adapts to the
query text. RAG-NAROKexploits the transparency inherent in
RAG pipeline to first extract the legitimate source identities, then
generate Anchor-Specific Refutation documents that explicitly
name and devalue retrieved sources while leveraging recency
and authority biases to steer the text generation toward a target
answer. Our results demonstrate that RAG-NAROKsignificantly
outperforms static baselines across diverse domains, revealing a
fundamental tension between RAG transparency and AI security.
Index Terms—Retreival augmented generation, large language
model, embedding, knowledge poisoning, adversarial RAG.
I. INTRODUCTION
The deployment of artificial intelligence systems for high-
stakes information retrieval, spanning clinical decision sup-
port, legal research, financial advisory, and cybersecurity-
related tasks, has undergone a foundational shift with the
progress of large language models (LLMs). LLMs, trained
on trillion-token corpora, acquire vast repositories of world
knowledge in their parameters, enabling them to generate
fluent, contextually coherent text across virtually unlimited
domains [1]. Despite their impressive capabilities, LLMs
trained purely on static corpora face a few limitations that
motivate the retrieval-augmented generation (RAG) architec-
ture. First, the staleness problem: model weights are fixed
at training time, which makes any factual claim about the
world subject to obsolescence as events unfold after the
training cutoff [2]. Second, the hallucination problem: LLMs
frequently generate fluent yet factually incorrect statements,
particularly on long-tail queries where the training signal is
sparse or conflicting [3]. Third, the provenance problem: the
model cannot reliably cite the source of its generated claims,
making factual verification and accountability infeasible in
high-stakes professional settings [4]. Each of these limitationsis addressed, at least partially, by grounding text generation in
externally retrieved documents, the core idea of RAG.
However, the reliability of a RAG system rests on an
assumption that is structurally fragile: that the external knowl-
edge corpus is trustworthy. Because the corpus is a mutable,
externally managed component [5], any entity capable of
contributing content to it, whether through an open submission
pipeline, a web-crawled ingestion process, or a compromised
upload interface, can introduce adversarial documents that
steer the system toward incorrect or misleading conclusions.
Unlike attacks on model weights, corpus-level interference
requires no gradient computation, making it a practical threat
in real-world deployments. As a result, the system is rendered
degraded: a RAG pipeline that confidently returns answers
grounded in corrupted evidence, undermining user trust be-
cause the failure is invisible.
Existing work on corpus poisoning [6], [7] has demonstrated
that adversarial documents can be crafted to rank highly for
target queries, and joint optimization approaches such as Joint-
GCG [8] have shown that simultaneously targeting the re-
triever and generator improves attack success rates. However,
most existing methods suffer from two key limitations. First,
many operate under unrealistic threat models that assume
white-box access to the embedding space. Second, they pri-
marily optimize for retrieval success, often generating passages
whose linguistic or embedding characteristics differ from those
of the underlying corpus. As a result, these passages may
be vulnerable to anomaly detection-based defenses proposed
in recent RAG security work [9]. In practice, the injected
document may be flagged as anomalous if it exhibits distribu-
tional shift relative to the legitimate corpus [9]. Moreover, an
injected adversarial document, even when retrieved, competes
withk−1legitimate passages that may actively contradict the
adversarial claim, effectively “outvoting” it and neutralizing
the attack [10]. To the best of our knowledge, no existing
attack seeks to evade both defense techniques simultaneously.
In this paper, we introduceRAG-NAROK(Retrieval-
Anchored Generation Negation And Response Quality Col-
lapse), a black-box, context-adaptive framework for degrad-
ing the response quality of RAG-based question answering
systems through knoowledge corruption. Rather than crafting
adversarial content in isolation,RAG-NAROKfirst performs a
reconnaissance phaseto fingerprint the target retriever char-
acteristics, then executes aShadow RAG observation loopthat
1
arXiv:2609.25469v1  [cs.AI]  21 Sep 2026

queries the target system and extracts the full top-kretrieval
context, including the factual claims, named sources, and
rhetorical register of each legitimate passage. This intelligence,
made available by the source transparency conventions of
trustworthy RAG deployments, is then exploited to synthesize
anchor-specific refutationdocuments.
The main contributions of this paper are three-fold:
•We introduceRAG-NAROK, a black-box, context-adaptive
corpus poisoning framework in RAG systems.
•We propose a heuristic Retriever Fingerprinting proce-
dure, using black-box query probing alone without access
to embeddings or index internals, to optimize crafted
adversarial documents for retrieval.
•We evaluate theretrieval dominance,defense evasion,
andgeneration influenceofRAG-NAROKacross three
heterogeneous domain corpora, legal, financial, and cy-
bersecurity, and observe first-position retrieval dominance
and anomaly detection evasion rate of up to 88.37%.
The remainder of this paper is organized as follows: Section II
provides background on LLMs and RAG. Section III reviews
related work, Section IV presents theRAG-NAROKframe-
work, Section VI evaluates its effectiveness, and Sections VII
and VIII discuss limitations and conclude the paper.
II. BACKGROUND
Understanding how modern knowledge-enhanced AI sys-
tems work requires familiarity with two core concepts: LLM
and RAG. This section provides background on these concepts.
A. Large Language Model
A Language Model (LM) is a probability distribution over
sequences of tokens drawn from a fixed vocabulary set. Given
a sequence of tokensx 1, x2, ..., x n, the language model assigns
a probability to the tokens of the vocabulary set:
P(x1, . . . , x n) =nY
t=1P(xt|x1, . . . , x t−1)
The model generates text ‘one token’ at a time, generating
each new token based on all previously generated tokens. The
token with the highest-probability is always selected. LLMs
are language models trained on a massive corpus, typically on
hundreds of billions to trillions of tokens of text, using the
self-supervised objective of next-token prediction [1].
B. Retrieval-Augmented Generation
RAGpipeline operates in three stages:
(i) Retrieval:Given a user queryq, a retrieverRmaps query
qto embeddinge qand returns the top-kcorpus documentsD,
from a knowledge corpus, the embeddings of which are the
closest toe q.
(ii) Context Assembly:The retrieved documentsD=
{d1, d2, . . . , d k}are concatenated withqinto a prompt
P(q, D), often with a faithfulness instruction directing the
generator to defer to retrieved context.
(iii) Generation:The generatorGproduces an outputyby
maximizing the conditional probability:
y= arg maxˆyP(ˆy| P(Q, D)).
RETRIEV AL
User
Knowledge Base
(Documents)
...1
1a2
Search
34
5Query
(question)Retriever
Top-k
Relevant
ChunksAugmented Prompt
System Instructions
+User Question
+Top-k Retrieved Docs
LLM
(Generator)
Answer
(Response)AUGMENTED GENERA TION
6
Vector
DatabaseEmbedding
ModelFig. 1. Retrieval Augmented Generation.
The knowledge corpus is stored as a collection of embed-
dings in a vector database. Givene q, the vector database re-
turns the top-kdocuments most similar toe qunder the chosen
distance metric. Widely deployed systems include FAISS [11],
Pinecone [12], and Chroma [13]. Figure 1 demonstrates how
the RAG pipeline works.
III. RELATEDWORK
The related literature spans four research areas: input-layer
attacks on LLM pipelines, corpus-layer poisoning attacks on
RAG systems, knowledge-conflict dynamics in RAG-assisted
LLMs, and defensive countermeasures for RAG integrity.
Early adversarial attacks on LLM-based systems targeted
the input layer rather than the knowledge base. Prompt
injection embeds malicious instructions in user queries or
documents, causing models to override system instructions
[14]. Indirect prompt injection hides adversarial instructions
inside retrieved documents [15], while jailbreaking uses ad-
versarial prompts and gradient-optimized suffixes to bypass
safety alignment [16]. However, input-layer attacks operate
only within single query sessions and do not persistently
corrupt the knowledge base across future queries.
The more consequential attack surface for RAG is the
external knowledge corpus. Retrieval-only poisoning attacks
craft adversarial passages optimized for embedding similarity
with target queries [6]. However, these fail when surrounded
by coherent legitimate passages. PoisonedRAG [7] advanced
this by jointly optimizing both retrieval and generation sub-
components, achieving attack success rates up to 97–99%.
Recent work includes Joint-GCG [8], a unified gradient-based
attack across both stages. Backdoor methods such as Trojan-
RAG [17] and BadRAG [18] embed conditional triggers in the
pipeline. However, most of these attacks either assume white-
box access to the knowledge corpus, which is less realistic
in practice, or prove ineffective when legitimate documents
outnumber the adversarial ones.
Knowledge-conflict research in LLM characterizes how
LLMs resolve contradictions between retrieved context and
parametric memory. Xu et al. [19] provide a taxonomy of
knowledge conflicts and show that LLMs exhibit strong para-
metric bias; however, they override it when external con-
text appears authoritative and corroborated. Jin et al. [20]
demonstrated the Dunning-Kruger effect in LLMs, where
models are paradoxically more susceptible to high-confidence
adversarial framing. Research on recency bias shows LLMs
2

Shadow Retrieval Stylometric Profiler
Target Query
(qT)RetrieverProbe Query
(qP)
AttackerTop-k Retrieved
DocumentsStatistics
Academic
Technical
Adversarial Document Synthesis EngineKnowledge Corpus
...Aa %Flesch
Index
Sentence
Length
Passive
Ratio
Legal
...
Retriever
TypeAnchor
SetStylometric
ProfileAnchor=specific
RefutationStylometric
MimicryAuthority
FramingCandidate
Refutation
Document D
(Ready for
Injection)
Refine Prompt and
Re-synthesize D
Matches
Stylometry?No
Yes≡Retriever Fingerprinting via Query Probing
N Synthetic
Probe Documents
D1PD2PD3P
   contains canary term
Ci and factual marker FiDiPDNP...
Heuristic Classifier
Dense Sparse HybridProbe T arget RAG
System
Retrieval Activation
Score
Accept D
for InjectionFig. 2.RAG-NAROKFramework Architecture.
preferentially adopt claims framed as temporal updates [21].
Both of these biases exhibited in LLMs create vulnerabilities
that can be leveraged to enhance the efficacy of knowledge-
corpus corruption attacks.
Most existing RAG knowledge-corruption attacks, particu-
larly under black-box retriever settings, do not generate adver-
sarial documents that both evade anomaly-detection defenses
and undermine conflict-resolution mechanisms. Additionally,
many existing attacks rely on unrealistic assumptions, such
as direct access to embedding manipulation.RAG-NAROK
addresses these limitations by operating under a realistic
black-box threat model while generating stealthy adversarial
documents that evade anomaly detection-based defenses and
exploit LLMs’ biases to maximize attack effectiveness.
IV. THERAG-NAROKFRAMEWORK
RAG-NAROKoperates under a realistic black-box threat
model: the adversary can submit queries to the target RAG
system and inject documents into its corpus; however, has
no access to embeddings, index internals, or model weights.
As illustrated in Figure 2, the pipeline proceeds in four
sequential phases. First, theRetriever Fingerprintingprobes
the target system with synthetic documents to infer whether
its retriever is sparse, dense, or hybrid. Then, theShadow
Retrievalsubmits the target query and extracts the top-k
retrieved documents alongside their source metadata, forming
the competitive context. Afterward, theStylometric Profil-
ingcharacterizes the linguistic register of those documents
to constrain synthesis. Finally, theAdversarial Document
Synthesis Engineconsumes the retriever type, anchor set,
and stylometric profile to produce a payload that reproduces
the majority of the legitimate factual content while reversing
the semantic conclusion in the remaining portion, wrapped
in authority-framing rhetoric that exploits LLM’s recency
and credibility biases. An iterative verification loop rejects
candidates that deviate from the corpus stylometric profile.
V. TECHNICALDETAILS
This section describes the technical components underlying
each phase of theRAG-NAROKpipeline, detailing the formal
mechanisms for retriever fingerprinting, anchor extraction,
stylometric profiling, and adversarial document synthesis.A. Retriever Fingerprinting
To infer retriever type heuristically without white-box
access, we injectNsynthetic probe documentsD P=
{DP
1, DP
2, . . . , DP
N}, each containing a set of unique canary
termsC iand verifiable factual markersF ithat serve as
unambiguous retrieval activation identifiers. For a given query
q, we define theretrieval activation score:
A(q) =X
t∈C i∪F i1
t∈Gen(q)
|Ci|+|F i|(1)
whereGen(q)denotes the generated output of the target
black-box system and1[·]is the indicator function.A(q)∈
[0,1]measures the fraction of probe identifiers surfaced in the
generation, serving as a soft proxy for retrieval activation.
We then evaluate each probe across five query families
(exact match, synonyms, paraphrases, lexical degradation,
token removal) and compute a fingerprint vector(S, R), where
semantic robustness:
S=¯A(Q para) +¯A(Q syn)
2·A(Q exact)(2)
andrare-token dependence:
R= 1−A(Q token-removed )
A(Q exact)(3)
Given the fingerprint vector(S, R), we classify the target
retriever according to the following heuristic threshold:
ˆτ=

DENSEifS >0.70∧R <0.30
SPARSEifS <0.30∧R >0.70
HYBRIDotherwise(4)
While these thresholds can be learned, we determine them
through experiments and observations where they separate
clear semantic robustness from lexical dependence.
B. Anchor Extraction
Given the top-kdocumentsD T={d 1, . . . , d k}returned
by shadow retrieval, we extract from eachd ia set ofanchor
objectsA i={a i,1, ai,2, . . .}, where each anchor object is
comprised of the atomic statementσ i,j, named entitye i,j, en-
tity typeτ i,j(e.g., RESEARCHINSTITUTION, LEGALBODY),
and temporal anchorρ i,j. The full anchor setA=Sk
i=1Ai
encodesthe factual structure of the competitive context window
and serves as an input to the document synthesis stage.
3

C. Stylometric Profiling
To evade perplexity-based and embedding-anomaly detec-
tors [9], we profileD Talong two tracks:
(i) Statistical Stylometry:Φ stat= 
Flesch, Ws, r passive
,
whereΦ statcaptures readability, mean sentence length, and
passive-voice ratio.
(ii) LLM-assisted prose classification:Φ prose∈ S prose, where
Sprose denotes a finite set of prose-style labels.
The combined stylometric profileΦ = (Φ stat,Φ prose), together
with the inferred retriever typeˆτand the anchor setA,
constitutes the complete input specification to the document
synthesis engine.
D. Adversarial Document Synthesis
The synthesis engine takes(A,Φ,ˆτ)as input and builds the
adversarial payload around three principles.
Manifold-Anchored Semantic Hijacking:the document
reproduces≈70%ofD T’s factual content to anchor the LLM’s
early-layer manifold commitment [22], [23], then steers the
remaining 30% toward the adversarial conclusionz, producing
a fluent, internally consistent, yet factually corrupted output.
Stylometric Mimicry:synthesis is prompted to matchΦ stat
within some tolerance and adoptΦ prose, making the document
distributionally indistinguishable from the corpus.
Authority Framing:an assistant-response prefill condi-
tioned one i,jandτ i,j(e.g.,“Pursuant to the 2026 amendment
supersedinge i,j. . . ”) exploits LLM instruction-following to
treat the adversarial content as an authoritative update [14].
Because instruction-following fidelity in LLMs is non-
deterministic, a single synthesis pass is not guaranteed to pro-
duce a document satisfying all stylometric constraints.RAG-
NAROKwraps the synthesis step in an iterative verification
loop. After each synthesis pass, the generated document ˆdis
evaluated against the target profile.
VI. RESULTS& EVALUATION
This section discusses the evaluation metrics used and
the impacts of the attack on three phases: retrieval ranking,
defense evasion, and influence on response generation.
A. Experimental Setup
We evaluateRAG-NAROKon three domain-specific cor-
pora spanning legal, technical, and financial text:CFR-
21, comprising the 2025 Code of Federal Regulations Ti-
tle 21;Cybersec-IT, from HuggingFace (ansulev/Trendyol-
Cybersecurity-Instruction-Tuning-Dataset); andFeds-2026,
based on Federal Reserve discussion papers [24], [25]. To-
gether, these corpora cover distinct prose styles targeted by
RAG-NAROK’s design. We compare against aNa ¨ıve Baseline
that generates adversarial passages via direct LLM prompting
to refute legitimate documents, without stylometric mimicry,
anchor extraction, or manifold-anchored framing. We evaluate
attack efficacy along three dimensions:Retrieval Dominance,
measuring whether adversarial documents outrank benign ones
in the top-kresults;Anomaly Evasion, measuring success
-0.4 -0.2 0.0 0.2 0.4 0.6 0.8
Efficacy Margin Score-10-50510152025Document Displacement (Slots)
Strategy
Naive (Baseline)
RAG-NarokFig. 3. Efficacy Margin vs. Benign Document Displacement Distribution.
against a dual-engine anomaly detector; andGeneration In-
fluence, measuring whether retrieved adversarial content alters
the final generated response.
B. Retrieval Dominance
1) Metrics:We quantify retrieval impact using four metrics
computed as means over multiple trials.Highest Adversarial
Rank Slotrecords the best (lowest) rank position achieved by
the injected document across trials.Efficacy Distance Margin
measures the similarity score advantage of the adversarial
document over the top benign competitor:
Margin=Score(d adv)−max
d∈D benignScore(d)(5)
Average Benign Document Rank Displacementmeasures
how many rank slots benign documents are pushed down
on average after injection.Reciprocal Rank Delta(∆RR)
quantifies the degradation in the effective retrieval quality of
benign documents:
∆RR=1
rank post(dbenign)−1
rank pre(dbenign)(6)
A negative∆RR indicates that benign documents were
pushed to worse rank positions post-injection.
2) Results:Table I summarizes the retrieval and evasion
results. Across all three corpora,RAG-NAROKconsistently
achieves near-complete retrieval dominance, placing adver-
sarial documents at or near the top rank while substantially
increasing the similarity gap over competing benign docu-
ments. In contrast, the Na ¨ıve Baseline exhibits limited retrieval
influence, with adversarial passages generally appearing much
lower in the ranking.
These retrieval gains translate into a pronounced degrada-
tion of benign document visibility. Compared to the base-
line,RAG-NAROKinduces substantially greater benign-rank
displacement and larger reductions in reciprocal rank, indi-
cating that legitimate evidence is systematically pushed out of
4

TABLE I
ADVERSARIALRETRIEVALPERFORMANCEMETRICS ACROSSCYBER-SECURITY ANDFINANCEDOMAINS.
MetricCybersec-IT Feds-2026 CFR-21
Naive Approach RAG-NAROKNaive Approach RAG-NAROKNaive Approach RAG-NAROK
Average Highest Poison Rank Slot7.30 1.00 8.60 1.00 10 1.20
Mean Efficacy Distance Margin0.0117 0.1767 -0.0406 0.3685 -0.0150 0.1426
Average Benign Document Rank Displacement↓0.84↓3.33↓0.94↓8.00↓0.55↓9.38
Mean Reciprocal Rank Delta (∆RR)-0.0313 -0.2466 -0.0324 -0.8889 -0.0212 -0.4056
Anomaly Detector Evasion Rate41.18% 68.93% 76.92% 80.95% 75.00% 88.37%
Feds-2026 CFR-21 Cybersec-IT02468101214Average Poisoned
Documents in Top-207.013.4
10.4
4.912.8
8.6
3.212.8
7.4
2.812.8
5.7Duplication Factor
1×
5×
10×
20×
Fig. 4. Average number of poisoned documents retrieved in the top-20 results
under different benign document duplication factors.
prominent retrieval positions. The strongest retrieval disruption
is observed on Feds-2026, while Cybersec-IT remains the most
challenging corpus despite still exhibiting clear adversarial
dominance. Figure 3 further illustrates the positive relationship
between Efficacy Margin and Benign Displacement, showing
that stronger adversarial ranking advantages are accompanied
by greater suppression of legitimate documents.
3) Robustness to Corpus Dilution:To evaluate resilience
against corpus inflation as a potential defense, we replicate
the benign corpus at 5, 10, and 20 times density and measure
the mean number of adversarial documents appearing in the
top-20 retrieved results. The Figure 4 summarizes the results.
While increasing corpus density reduces adversarial repre-
sentation in the retrieval set,RAG-NAROKremains effective
across all corpora. CFR-21 exhibits the strongest resilience
to dilution, maintaining a high concentration of adversarial
documents even under aggressive inflation, whereas Feds-2026
experiences the largest reduction. Nevertheless, adversarial
documents consistently persist within the top-20 retrieved
results across all settings, indicating that corpus inflation alone
is insufficient to prevent injected adversarial content from
reaching the generator context.
C. Anomaly Detection Evasion
1) Detector Architecture:We evaluate against a dual-
engine anomaly detector representative of defenses proposed
for RAG systems [9], [26]. One Isolation Forest operates on
document embeddings, while a second operates on stylometric
features. Both are trained on the legitimate corpus, and a doc-
ument is flagged if either detector identifies it as anomalous.
2) Results:Table I summarizes evasion performance.
Across all corpora,RAG-NAROKconsistently achieves higherevasion rates than the Na ¨ıve Baseline, demonstrating that sty-
lometric mimicry and manifold-anchored synthesis reduce de-
tectability. The largest improvement is observed on Cybersec-
IT corpora, suggesting that corpus-specific writing character-
istics make stylometric alignment particularly important in
technical domains.
2
 0 2 4 6 8
PC1 (73.0% variance)3
2
1
01234PC2 (21.7% variance)
Embedding-space Stylometric Manifold
Legitimate Corpus
Naïve Attack
RAGNarok
Fig. 5. Defense Evasion Analysis.
Figure 5 provides a geometric view of the stylometric
feature space for Feds-2026. Compared to the Na ¨ıve Baseline,
RAG-NAROK-generated documents cluster substantially closer
to legitimate corpus documents, indicating that the stylometric
verification loop successfully constrains synthesis toward the
benign corpus manifold.
D. Generation Influence
1) LLM-as-a-Judge:We use an LLM-as-a-Judge evalua-
tion [27] to measure whether retrieved adversarial documents
alter the semantic conclusions of generated responses. The
judge reports (1) a Payload Integration Rate indicating whether
the adversarial payload influenced the response and (2) a Mean
Cascade Rating (0–10) measuring the extent of that influence.
2) Results:Table II reports generation-stage influence.
RAG-NAROKmatches or exceeds the Na ¨ıve Baseline on
Payload Integration Rate across all corpora and consistently
achieves higher Mean Cascade Ratings, indicating deeper
propagation of adversarial content into generated responses.
The largest gain is observed on Feds-2026, mirroring the
strong retrieval dominance achieved on that corpus.
5

TABLE II
ADVERSARIALPAYLOADINTEGRATION ANDLLM-AS-A-JUDGE
CASCADERATINGS.
Dataset Attack Strategy Payload Integration Rate Mean Cascade Rating
Cybersec-ITBaseline 60% 6.50
RAG-NAROK75% 7.60
Feds-2026Baseline 70% 5.57
RAG-NAROK80% 8.00
CFR-21Baseline 80% 7.25
RAG-NAROK80% 8.00
These results suggest that retrieval success alone does not
fully explain generation influence. The improvements achieved
byRAG-NAROKindicate that manifold-anchored construction
and authority framing contribute to generation-stage persua-
sion beyond simply obtaining favorable retrieval ranks.
VII. LIMITATIONS
Several limitations constrain the current work. Firstly, the
heuristic-based retriever fingerprinting parameters(S, R)are
not learned, and the fingerprinting procedure itself war-
rants dedicated empirical study, particularly for re-ranking-
augmented retrievers. Secondly, LLM-as-a-Judge evaluations
are subject to positional and verbosity biases [27]; structured
output mitigates; however, it does not eliminate this concern.
Moreover, the threat model assumes the sources are transparent
in the RAG pipeline, and would fail under a model where
sources cannot be observed. Lastly, the document synthesis
and the iterative verification process for the synthesized doc-
ument is computationally expensive.
VIII. CONCLUSION
We introducedRAG-NAROK, a black-box, context-aware
corpus poisoning framework for RAG systems that adapts its
adversarial payload to the observed retrieval context before
injection.RAG-NAROKconsistently ranks high in retrieval,
evades anomaly detection in up to 88.37% of trials, and
corrupts the semantic conclusion of generated responses across
legal, financial, and cybersecurity corpora. We hope this work
motivates a new class of context-aware defenses, including
provenance-aware retrieval auditing, adaptive anomaly detec-
tion, and generation-layer conflict resolution, that match the
sophistication of the threat.
REFERENCES
[1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askellet al., “Language mod-
els are few-shot learners,”Advances in neural information processing
systems, vol. 33, pp. 1877–1901, 2020.
[2] Z. Kasner and O. Du ˇsek, “Neural pipeline for zero-shot data-to-text gen-
eration,” inProceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), 2022.
[3] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, and P. Fung, “Survey of hallucination in natural language
generation,”ACM computing surveys, vol. 55, no. 12, pp. 1–38, 2023.
[4] R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim,
C. Hesse, S. Jain, V . Kosaraju, W. Saunderset al., “Webgpt: Browser-
assisted question-answering with human feedback,”arXiv preprint
arXiv:2112.09332, 2021.
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.[6] Z. Zhong, Z. Huang, A. Wettig, and D. Chen, “Poisoning retrieval cor-
pora by injecting adversarial passages,” inProceedings of the Conference
on Empirical Methods in Natural Language Processing, 2023.
[7] W. Zou, R. Geng, B. Wang, and J. Jia, “{PoisonedRAG}: Knowledge
corruption attacks to{Retrieval-Augmented}generation of large lan-
guage models,” in34th USENIX Security Symposium (USENIX Security
25), 2025, pp. 3827–3844.
[8] H. Wang, R. Zhang, J. Wang, M. Li, Y . Huang, D. Wang, and
Q. Wang, “Joint-gcg: Unified gradient-based poisoning attacks on
retrieval-augmented generation systems,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 40, no. 42, 2026.
[9] H. Zhou, K.-H. Lee, Z. Zhan, Y . Chen, Z. Li, Z. Wang, H. Haddadi,
and E. Yilmaz, “Trustrag: enhancing robustness and trustworthiness in
retrieval-augmented generation,”preprint arXiv:2501.00879, 2025.
[10] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettle-
moyer, and W.-t. Yih, “Replug: Retrieval-augmented black-box language
models,” inProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), 2024, pp. 8371–8384.
[11] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,”IEEE transactions on big data, vol. 7, no. 3, 2019.
[12] Pinecone Systems, Inc., “Pinecone: The vector database for ai,”
accessed: 2026-06-20. [Online]. Available: https://www.pinecone.io/
[13] Chroma, “Chromadb,” open-source database. Accessed: 2026-06-20.
[Online]. Available: https://github.com/chroma-core/chroma
[14] F. Perez and I. Ribeiro, “Ignore previous prompt: Attack techniques for
language models,”arXiv preprint arXiv:2211.09527, 2022.
[15] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-world llm-
integrated applications with indirect prompt injection,” inProceedings
of the 16th ACM workshop on artificial intelligence and security, 2023.
[16] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson,
“Universal and transferable adversarial attacks on aligned language
models,”arXiv preprint arXiv:2307.15043, 2023.
[17] P. Cheng, Y . Ding, T. Ju, Z. Wu, W. Du, P. Yi, Z. Zhang, and G. Liu,
“Trojanrag: Retrieval-augmented generation can be backdoor driver in
large language models,”arXiv preprint arXiv:2405.13401, 2024.
[18] J. Xue, M. Zheng, Y . Hu, F. Liu, X. Chen, and Q. Lou, “Badrag:
Identifying vulnerabilities in retrieval augmented generation of large
language models,”arXiv preprint arXiv:2406.00083, 2024.
[19] R. Xu, Z. Qi, Z. Guo, C. Wang, H. Wang, Y . Zhang, and W. Xu, “Knowl-
edge conflicts for llms: A survey,” inProceedings of the Conference on
Empirical Methods in Natural Language Processing, 2024.
[20] Z. Jin, P. Cao, Y . Chen, K. Liu, X. Jiang, J. Xu, L. Qiuxia, and J. Zhao,
“Tug-of-war between knowledge: Exploring and resolving knowledge
conflicts in retrieval-augmented language models,” inProceedings of
the 2024 joint international conference on computational linguistics,
language resources and evaluation (LREC-COLING 2024), 2024.
[21] H. Fang, S. Tao, N. Chen, K.-X. Chang, and T. Sakai, “Do large language
models favor recent content? a study on recency bias in llm-based
reranking,” inProceedings of the 2025 Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval in
the Asia Pacific Region, 2025, pp. 85–94.
[22] K. Meng, D. Bau, A. Andonian, and Y . Belinkov, “Locating and editing
factual associations in gpt,”Advances in neural information processing
systems, vol. 35, pp. 17 359–17 372, 2022.
[23] M. Geva, J. Bastings, K. Filippova, and A. Globerson, “Dissecting
recall of factual associations in auto-regressive language models,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, 2023, pp. 12 216–12 235.
[24] “Federal reserve discussion series paper 2026-028,” Board of Governors
of the Federal Reserve System, FEDS Working Paper, 2026.
[25] “Federal reserve discussion series paper 2026-026,” Board of Governors
of the Federal Reserve System, FEDS Working Paper, 2026.
[26] H. Song, Y .-A. Liu, R. Zhang, J. Guo, M. de Rijke, Y . Fan, and X. Cheng,
“Adversarialcot: Single-document retrieval poisoning for llm reasoning,”
arXiv preprint arXiv:2604.12201, 2026.
[27] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu, Y . Zhuang, Z. Lin,
Z. Li, D. Li, E. Xinget al., “Judging llm-as-a-judge with mt-bench
and chatbot arena,”Advances in neural information processing systems,
vol. 36, pp. 46 595–46 623, 2023.
6
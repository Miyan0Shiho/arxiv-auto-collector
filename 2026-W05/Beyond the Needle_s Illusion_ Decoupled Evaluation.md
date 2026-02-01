# Beyond the Needle's Illusion: Decoupled Evaluation of Evidence Access and Use under Semantic Interference at 326M-Token Scale

**Authors**: Tianwei Lin, Zuyi Zhou, Xinda Zhao, Chenke Wang, Xiaohong Li, Yu Chen, Chuanrui Hu, Jian Pei, Yafeng Deng

**Published**: 2026-01-28 05:44:00

**PDF URL**: [https://arxiv.org/pdf/2601.20276v1](https://arxiv.org/pdf/2601.20276v1)

## Abstract
Long-context LLM agents must access the right evidence from large environments and use it faithfully. However, the popular Needle-in-a-Haystack (NIAH) evaluation mostly measures benign span localization. The needle is near-unique, and the haystack is largely irrelevant. We introduce EverMemBench-S (EMB-S), an adversarial NIAH-style benchmark built on a 326M-token MemoryBank. While the full MemoryBank spans 326M tokens for retrieval-based (RAG) evaluation, we evaluate native long-context models only at scales that fit within each model's context window (up to 1M tokens in this work) to ensure a fair comparison. EMB-S pairs queries with collision-tested near-miss hard negatives and gold evidence sets spanning one or more documents, validated via human screening and LLM verification. We also propose a decoupled diagnostic protocol that reports evidence access (document-ID localization) separately from end-to-end QA quality under full-context prompting. This enables consistent diagnosis for both native long-context prompting and retrieval pipelines. Across a reference-corpus ladder from domain-isolated 64K contexts to a globally shared 326M-token environment, we observe a clear reality gap. Systems that saturate benign NIAH degrade sharply in evidence access under semantic interference. These results indicate that semantic discrimination, not context length alone, is the dominant bottleneck for long-context memory at scale.

## Full Text


<!-- PDF content starts -->

Beyond the Needle’s Illusion: Decoupled Evaluation of Evidence Access and
Use under Semantic Interference at 326M-Token Scale
Tianwei Lin*,1,2Zuyi Zhou*,1,2Xinda Zhao1,2Chenke Wang1,2Xiaohong Li1,2
Yu Chen1,2Chuanrui Hu†,1,2Jian Pei†,3Yafeng Deng†,1,2
1EverMind2Shanda Group3Duke University
{tianwei.lin, zuyi.zhou, xinda.zhao, xiaohong.li,
yu.chen, chuanrui.hu, yafeng.deng}@shanda.com
cw4565@nyu.edu j.pei@duke.edu
Abstract
Long-context LLM agents must access the right
evidence from large environments and use it
faithfully. However, the popular Needle-in-
a-Haystack (NIAH) evaluation mostly mea-
sures benign span localization. The needle
is near-unique, and the haystack is largely ir-
relevant. We introduceEverMemBench-S
(EMB-S), an adversarial NIAH-style bench-
mark built on a326M-token MemoryBank.
While the full MemoryBank spans 326M to-
kens for retrieval-based (RAG) evaluation, we
evaluate native long-context models only at
scales that fit within each model’s context win-
dow (up to 1M tokens in this work) to ensure
a fair comparison. EMB-S pairs queries with
collision-tested near-miss hard negatives and
gold evidence sets spanningone or moredoc-
uments, validated via human screening and
LLM verification. We also propose adecou-
pled diagnostic protocolthat reportsevidence
access(document-ID localization) separately
from end-to-end QA quality under full-context
prompting. This enables consistent diagnosis
for both native long-context prompting and re-
trieval pipelines. Across a reference-corpus
ladder from domain-isolated 64K contexts to a
globally shared 326M-token environment, we
observe a clear reality gap. Systems that satu-
rate benign NIAH degrade sharply in evidence
access under semantic interference. These re-
sults indicate that semantic discrimination, not
context length alone, is the dominant bottleneck
for long-context memory at scale.
1 Introduction
Large language models (LLMs) are increasingly
deployed as reasoning layers over large document
collections, powering retrieval-augmented gener-
ation and tool-using agents in real-world search
*Equal contribution.
†Corresponding author.
IrrelevantNeedleIrrelevantIrrelevantIrrelevant
IrrelevantGold EvidenceIrrelevantIrrelevantIrrelevantHard NegativesGold EvidenceBenign NIAHEMB-SHard NegativesNeedle
Gold EvidenceGold EvidenceHard NegativesIrrelevantIrrelevantQuery
Semantic RelevanceHighLowWhat wasAlex's total incomein2023?
Document IntegrationQuery
NeedleNeedleGold EvidenceHard NegativesAlexalsoearned$30kfromconsultingin2023.Alex‘s2022total incomewas$140k.Gold EvidenceIn2023,Alexreceivedasalaryof$120kfromTechCorp.Figure 1:Benign NIAH vs. EMB-S.NIAH finds a near-
unique needle in mostly irrelevant content; EMB-S must
distinguish near-miss negatives and integrate evidence
that may span multiple documents.
and analytics systems (Yehudai et al., 2025; Ferrag
et al., 2025). In these settings, reliability hinges
on evidence-grounded behavior: given a query or
internal state, a system must not only access the
correct evidence from long contexts or large cor-
pora, but also use it faithfully to produce grounded
outputs and actions.
Evaluating this capability at scale, however, re-
mains challenging (Bai et al., 2023; An et al.,
2024; Bai et al., 2024). Fully end-to-end evalu-
ation over hundred-million-token environments is
computationally expensive and difficult to standard-
ize, which limits reproducibility and broad com-
parison. As a result, the community has widely
adoptedNeedle-in-a-Haystack(NIAH) tests as a
cost-controlled proxy for evidence access (Hsieh
et al., 2024; Kuratov et al., 2024; Gao et al., 2025).
In a typical NIAH setup, a short “needle” is inserted
into a long “haystack,” and the model is asked to
recover it in a single long-context invocation.
However, common NIAH benchmarks primarily
measurebenign span localizationrather than real-
1arXiv:2601.20276v1  [cs.CL]  28 Jan 2026

istic evidence access. In most setups, the needle is
(near-)unique and the vast majority of the haystack
is irrelevant, so success often reduces to matching
a low-entropy string signal (Gao et al., 2025). Mod-
ern long-context models can therefore saturate on
NIAH, creating the impression that evidence ac-
cess has been solved, even though the evaluation
rarely stresses semantic ambiguity or competing
evidence.
The dominant difficulty in real long-context
workloads arises from a different source. In realis-
tic corpora, evidence is rarely unique: documents
overlap, paraphrase one another, and partially sat-
isfy a query while violating a key constraint, such
as an incorrect entity, year, or numerical value.
Thesenear-missdocuments are semantically close
to the gold evidence and create dense interference
throughout the context. As a result, evidence ac-
cess becomes a problem ofsemantic discrimina-
tion under global interference, often requiring
the integration of multiple documents rather than
the retrieval of a single isolated span. Crucially,
this challenge is orthogonal to context length alone:
increasing scale primarily amplifies semantic in-
terference rather than introducing new forms of
difficulty.
To bridge this realism gap while preserving the
tractability of NIAH-style evaluation, we intro-
duceEverMemBench-S (EMB-S), anadversar-
ial NIAH-stylebenchmark. Conceptually, EMB-S
is a drop-in upgrade to standard NIAH. It retains
single-context evaluation within each model’s long-
context limits (up to 1M tokens in this work) but
replaces largely irrelevant haystacks with collision-
tested near-miss hard negatives and gold evidence
sets spanning one or more documents, validated
via human screening and LLM verification. As il-
lustrated in Figure 1, benign NIAH isolates a single
needle among irrelevant content. In contrast, EMB-
S deliberately mixes multiple gold documents with
a dense spectrum of semantically similar distrac-
tors. This forces models to discriminate based on
constraints rather than simple surface overlap.
Beyond increasing dataset difficulty, EMB-S is
paired with adecoupled diagnostic protocolde-
signed to disentangle distinct failure modes. Specif-
ically, we reportevidence accessseparately from
end-to-end QA quality under a shared document-ID
evidence interface. In theLocalizationtask, sys-
tems output the top- kdocument IDs they consider
relevant; inGenerative QA, we evaluate answer
quality under full-context prompting at scales thatfit within the model’s context window. This sepa-
ration helps distinguish access failures from down-
stream answer-quality degradation under semantic
interference, which are otherwise conflated in end-
to-end QA metrics (Friel et al., 2024; Krishna et al.,
2024; Liu et al., 2024).
We evaluate native long-context models and
retrieval-based pipelines across areference corpus
ladder, ranging fromeight domain-isolated 64K-
token corporato a326M-token MemoryBank.
This controlled scaling study reveals a consistent
pattern: systems that achieve near-perfect scores
on benign NIAH degrade sharply in evidence ac-
cess as semantic interference increases, even when
context length is held constant. These results indi-
cate that semantic discrimination—not raw context
length—is the dominant bottleneck for evidence
access at scale.
In summary, our contributions are threefold:
•Adversarial NIAH with semantic interfer-
ence.We introduce EverMemBench-S, which
augments NIAH with collision-tested near-
miss hard negatives and gold evidence sets
(single- or multi-document), transforming nee-
dle finding into semantic discrimination under
dense, constraint-violating distractors.
•A cost-feasible diagnostic protocol with a
shared evidence interface.EMB-S reports
evidence access via document-ID localization
metrics and end-to-end QA quality under full-
context prompting, enabling diagnosis of ac-
cess failures and robustness under semantic
interference without sacrificing scalability.
•Unified evaluation across long-context
prompting and retrieval pipelines.EMB-S
provides a shared document-ID–based eval-
uation framework for native long-context
prompting, embedding-based retrievers, and
retrieve-then-generate systems across corpora
ranging from 64K to 326M tokens (Li et al.,
2024; Jiang et al., 2024).
2 Related Work
Evidence Access under Global Interference.
Long-context evaluation suites such as Long-
Bench (Bai et al., 2023), L-Eval (An et al., 2024),
LV-Eval (Yuan et al., 2024), LongBench v2 (Bai
et al., 2024), RULER (Hsieh et al., 2024), and BA-
BILong (Kuratov et al., 2024) have driven progress
2

Table 1:Comparison of long-context and memory benchmarks.EMB-S uniquely enables decoupled diagnosis
of evidenceaccessvs.usein a globally shared environment with dense semantic interference.
Dimension NIAH FRAMES BEAM LC-Eval PRELUDE EMB-S
DIAGNOSTICSCOPE
Sparse Evidence Regime✗ ✗ ✗ ✗ ✗ ✓
Global Shared Env.✗ ✓ ✗ ✗ ✗ ✓
Semantic Discrimination✗ ✓ ✗ ✗ ✗ ✓
Multi-hop Reasoning✗ ✓ ✗ ✗ ✗ ✓
Decoupled Access/Use✗ ✗ ✗ ✗ ✗ ✓
MEMORYCOVERAGE
Working Memory (Context)✓ ✓ ✓ ✓ ✓ ✓
Parametric Memory✓ ✗ ✓ ✓ ✓ ✓
External Memory✗ ✓ ✓ ✗ ✗ ✓
Persistent/Latent✗ ✗ ✗ ✗ ✗ ✓
SCALE ANDREALISM
Verified Hard Negatives✗ ✓ ✗ ✗ ✗ ✓
Scale>1M Tokens✓ ✗ ✓ ✗ ✗ ✓
Multi-scale Eval.✓ ✗ ✓ ✓ ✗ ✓
Diverse Corpus✗ ✓ ✓ ✓ ✗ ✓
on end-to-end QA and reasoning over extended
contexts. These benchmarks primarily evaluate
end-to-end task success, implicitly assuming sparse
relevance and near-unique evidence, and therefore
do not stress semantic discrimination under glob-
ally shared, hard-negative-heavy contexts. A key
limitation is that end-to-end accuracy conflates (i)
failing to access the evidence, (ii) accessing it but
being diluted by interference, and (iii) accessing
the right evidence but making a reasoning/ground-
ing error. A key limitation in benchmark construc-
tion is that relevance is typically sparse and evi-
dence near-unique, without explicitly constructing
constraint-violating near-miss distractors, so these
suites largely measurelength, notdifficulty.
From Needle Localization to Semantic Discrimi-
nation.NIAH-style needle insertion tests offer a
cheap and scalable probe of long-context behavior
by planting a short target span in a long haystack
and asking models to recover it (Hsieh et al., 2024;
Kuratov et al., 2024; Gao et al., 2025). A key lim-
itation is that the needle is often a low-entropy,
near-unique signal, with few near-miss candidates,
no systematic constraint-violation design, and an
implicit single-needle assumption. Under dense
near-miss conditions, evidence access ceases to
be a localization problem and becomes semantic
discrimination under global interference.
Decoupling Access and Use at Scale.RAG
benchmarks such as RAGBench (Friel et al., 2024),
FRAMES (Krishna et al., 2024), CRUD-RAG (Lyuet al., 2024), and DomainRAG (Wang et al., 2024)
decouple retrieval and generation and provide
component-level metrics for pipeline-based sys-
tems. Recent work also revisits the long-context vs.
RAG trade-off and explores long-context-enhanced
retrieval pipelines (Li et al., 2024; Jiang et al.,
2024). These benchmarks decouple retrieval and
generation, but are architecturally tied to pipeline-
based systems, preventing unified evaluation of
native long-context prompting and retrieval-based
systems under the same evidence interface. A key
limitation is the lack of a shared document-ID ev-
idence space that supports architecture-agnostic
diagnosis of evidence access under large, globally
mixed corpora.
Positioning & Summary.A key gap is scal-
able evaluation of evidence access under dense
semantic interference, where answers require multi-
evidence aggregation and near-miss distractors are
both abundant and adversarial. EMB-S occupies
a previously underexplored regime: scalable eval-
uation of evidence access under dense semantic
interference, with explicit diagnostics that separate
access from use.
Table 1 summarizes how EMB-S differs along
these diagnostic dimensions.
3 Dataset Construction
Figure 2 provides an overview of EMB-S’s data
foundation and construction pipeline. Our key de-
sign choice is tofix the query setwhilesystemati-
3

Memory Bank
（326M tokens, 160, 280 docs）
Final
Queries
（Domain -Labeled）
Domain Data
（64K tokens, 8 Domains ）…
Domain -isolated Inter -domain mixing
64K
✖
8Component B
Reference 
Corpus 
LadderComponent A
Query 
Construction
Single
Domain
Automated AnnotationHuman Annotation
Atomization of ReferenceContextual Synthesis
(Query, Answer, RefDoc )
⇩
(Query, Answer, [RefList])Phase II
Reasoning Chain 
Topology Augmentation
Step 2
Context Validation
Filter out low -quality 
samplesPhase I
Initial Query  
Formulation
Phase III
Quality Assurance 
Pipeline
Step 2
Collision Resolution 
Check candidates :
A.Invalidating Conflict
B.Hard Negatives
C.False Negatives
Step 1
Candidate Discovery
Retrieve the semantically 
related items of the reference informationStep 1
Standardize
(Query, Answer, RefDoc)
Track I
Track IIQuery RefDoc
Mixing Multiple Domain128K
✖
8256K
✖
8512K
✖
1
…326M
✖
1
Mixing Memory BankMemory Bank（326M tokens, 160, 280 docs）1M
✖
1Figure 2:EMB-S overview.Component A builds 483 domain-labeled queries from a 326M-token MemoryBank
(standardization, multi-hop synthesis, and collision testing). Component B defines a reference-corpus ladder from
64K to 326M tokens with increasing inter-domain mixing and distractor injection.
cally varying the searchable evidence pool: Com-
ponent A constructs a compact set of rigorously val-
idated, domain-labeled queries with gold evidence
sets (RefList) spanning one or more documents and
collision-tested distractors (via human screening
and LLM verification), while Component B defines
a multi-scale reference-corpus ladder that increases
bothscaleandsemantic interferencein a controlled
manner. This decoupling enables apples-to-apples
comparisons across native long-context prompting
and retrieval-based pipelines under matched super-
vision.
3.1 Raw Data Sources
The construction of EverMemBench-S begins with
systematically collecting existing, publicly avail-
able long-context evaluation benchmarks. We ag-
gregate 9 diverse datasets spanning different scales,
task types, and evaluation objectives to form the
326M-token MemoryBank (160,280 documents);
the complete list of sources and their characteristics
is provided inAppendix A.4. All raw datasets are
downloaded, cleaned, and deduplicated before fur-
ther processing (Magar and Schwartz, 2022). To en-
able domain-conditioned evaluation at long-contextscale, we additionally assign documents (and down-
stream queries) to one of 8 broad domains; details
and guidelines are provided in Appendix A.1.
3.2 Data Standardization
To ensure cross-source consistency, we transform
all raw instances into a unified tuple:(Query, An-
swer, RefDoc). Here,RefDocdenotes the refer-
ence document that must be accessed to correctly
answer the query. In later stages, we form a gold
evidence setRefListspanning one or more docu-
ments; we call a querysingle-sourceif |RefList|=
1(i.e., it requires retrieving a single gold reference
document) andmulti-sourceotherwise. We also
collect a pool of hard negatives for adversarial eval-
uation under semantic interference.
3.3 Data Construction Overview
EverMemBench-S is constructed via a systematic
process with two tightly coupled components (Fig-
ure 2).Component Aapplies a three-stage query
construction pipeline that transforms 39,860 het-
erogeneous instances into483rigorously validated,
domain-labeled queries with gold evidence sets
(RefList) and collision-tested distractors (Table 2).
4

Table 2:Pipeline statistics.Quantitative overview of
the EverMemBench-S construction pipeline (instance
counts across stages; final benchmark contains483
queries).
Stage Step / Process #Inst.
STAGEI: STANDARDIZATION&HUMAN SCREENING
Raw aggregation 39,860
Human screening (seed pool) 9,621
STAGEII: REASONING-CHAIN SYNTHESIS
Seed pool (from Stage I) 9,621
Track 1: Retrieval-guided query rewriting 2,112
Track 2: RefDoc atomization 7,509
Synthesis output (merged) 3,457
Deduplication & difficulty calibration 882
STAGEIII: QUALITY CONTROL(COLLISION TESTING)
Collision testing & LLM verification (final)483
Component Bthen builds a multi-scalereference
corpus ladderthat starts from domain-isolated
long-context corpora and expands to the full 326M-
token MemoryBank via inter-domain mixing and
progressive distractor injection (1M/2M/. . . /326M).
We provide additional implementation details (e.g.,
stage-level criteria, domain labeling, and verifica-
tion guidelines) in Appendix A.1.
Stage I: Standardization and Human Screen-
ing.Stage I converts heterogeneous sources into
standardized triples(Query, Answer, RefDoc),
where RefDoc is the reference document required
to infer the answer from the query. We then per-
formhuman screeningto remove low-quality in-
stances (e.g., ambiguous questions, unsupported
answers, or noisy/mismatched references), yield-
ing a clean seed pool for downstream multi-hop
synthesis.
Stage II: Reasoning-Chain Synthesis (Two
Tracks).Stage II converts single-hop triples into
reasoning-intensive items and constructs the gold
evidence setRefList.RefListmay remain single-
document (single-source) or be expanded into mul-
tiple documents (multi-source). To create multi-
source items, we implement two complementary
expansion tracks:Track 2 (RefDoc atomization)de-
composes complex reference documents into mul-
tiple sub-documents, forcing cross-document ag-
gregation;Track 1 (retrieval-guided query rewrit-
ing)retrieves semantically similar documents from
the MemoryBank and rewrites the query so that
answering requires both the original RefDoc and
newly added supporting documents. We then cali-
brate difficulty and remove near-duplicates using
a strong dense retriever (Qwen3-Embedding-8B),producing 882 candidate queries for Stage III (Ta-
ble 2).
Stage III: Quality Control via Collision Test-
ing.Stage III improves scoring reliability by filter-
ing inconsistencies and validating negatives. For
each candidate triple, we retrieve the top- knearest
non-referencedocuments under dense embedding
similarity from the MemoryBank (collision candi-
dates), usingQwen3-Embedding-8B(excluding
documents inRefList). An LLM then verifies each
candidate against(Query, Answer, RefList)and
assigns one of three outcomes:(a) Conflict(contra-
dicts the query/answer/reference content; discard
the sample),(b) Hard Negative(semantically sim-
ilar but does not support the answer; retained as an
adversarial distractor), or(c) False Negative(pro-
vides additional valid evidence; added to RefList).
The final benchmark contains483queries with
LLM-verified reference sets and validated hard neg-
atives.
3.4 Reference Corpus Ladder
Beyond constructing queries, EMB-S defines aref-
erence corpus ladderthat enables controlled stress
tests under increasingscaleandcross-domain in-
terferencewhile keeping the query set fixed (Fig-
ure 2). Each query is assigned a domain label
d∈ {1, . . . ,8} based on the domains of its gold
reference documents, and we construct a family of
reference corpora {C}as follows.Scale as token
budget.We parameterize each corpus by a target
token budget S(e.g., 64K, 512K, 326M), i.e., the
total tokens of the searchable evidence pool CS
rather than the number of documents.1
Domain-isolated base corpora (64K).For each
domain d, we build a domain-specific corpus C64K
d
that (i) contains all gold reference documents for
queries in domain dand (ii) fits within a 64K long-
context input after accounting for prompt/query
overhead. This setting supportsdomain-isolated
evaluation, where each domain can be tested inde-
pendently.
Inter-domain mixing (128K/256K).To gradu-
ally introduce semantic interference while retaining
a domain-conditioned evaluation axis, we expand
each domain corpus by sampling documents from
the other 7 domains. Let S(D, B) denote sampling
documents from a collection Duntil the total token
1When serializingCSinto a single input for a model with
window N, we additionally enforce the constraint using the
model’s tokenizer (Appendix A.2).
5

budget reachesB:
CS
d=C64K
d∪ S
[
d′ ̸=dC64K
d′;S−64K

forS∈ {128K,256K},
where sampling is performed without replacement
and with document-level deduplication.
Shared mid-scale corpus (512K).We then
merge the 8 domain pools into a shared corpus
C512K= DedupS8
d=1C64K
d
, yielding a single
reference space shared by all domains.
Global distractor injection.Finally, we ex-
pand the shared corpus by injecting distractor doc-
uments sampled from the remaining MemoryBank
(denoted as M) to reach larger scales (e.g., 1M,
2M, . . . , up to 326M tokens):
CS=C512K∪ S 
M \ C512K;S−512K
,
with a fixed random seed to ensure reproducibility
across runs and scales.
3.5 Dataset Characteristics and Sparsity
Validation
We summarize two key properties of the con-
structed benchmark that directly impact evaluation
difficulty: (i) the distribution of query types and
context lengths (Figure 3), and (ii) whether the ref-
erence corpus ladder induces domain “silos” or a
globally mixed search space under scaling (Fig-
ure 4).
0 10 20 30 40
Percentage (%)Multi-question
Multi-choice
Multi-hop37.6%
31.4%
31.0%(a) Task Type Distribution
<200 200-400 400-1K 1K-2K 2K-4K 4K-8K
Reference Length (tokens)01020304050Percentage (%)45.3%(b) Multi-choice
<200 200-400 400-1K 1K-2K 2K-4K 4K-8K
Reference Length (tokens)0102030Percentage (%)30.8%(c) Multi-hop
<200 200-400 400-1K 1K-2K 2K-4K 4K-8K
Reference Length (tokens)0102030Percentage (%)28.8%(d) Multi-question
Figure 3: Distribution of task types in EverMemBench-
S and the distribution of context lengths for each task
type.
Validation of Interference Progression.Fig-
ure 4 shows that retrieval becomes more cross-
source as the corpus scales (e.g., 512K →326M):top-kresults increasingly span multiple datasets,
indicating a shift toward a more globally mixed
retrieval environment.
2 3 4 5 6 7
Number of Different Datasets in Top10 Retrieval Results050100150200Number of Queries
5.8% 5.6%28.4%29.0%38.9% 38.5%
22.2% 22.2%
3.7% 3.7%
1.0% 1.0%Dataset Diversity Distribution In Different Scales
(512K vs 326M Context)
512K Context
326M Context
Figure 4:Cross-source mixing (512K vs. 326M).Dis-
tribution of the number of distinct datasets in the top-10
retrieved documents per query.
4 Experiments
4.1 Research Questions
We structure our experiments as a diagnostic study
centered onevidence access under interferenceand
its downstream impact on long-context question an-
swering. Specifically, we investigate the following
research questions:
•RQ1 (Scaling): How doesevidence accessde-
grade as the searchable corpus scales and seman-
tic interference increases?
•RQ2 (Single vs. multi-source): How large is the
gap between retrieving a single required refer-
ence document and retrievingallrequired refer-
ence documents under interference?
•RQ3 (End-to-end QA): Under realistic long-
context usage, how well can long-context LLMs
answer questions when provided the full refer-
ence corpus within their context-window con-
straints?
4.2 Experimental Setup
We evaluate 483 queries over a reference cor-
pus ladder that scales from 64K (domain-isolated)
through 128K/256K and a shared 512K corpus,
up to a 326M-token MemoryBank with global dis-
tractor injection. Throughout, we distinguish the
environment scale S(the size of the searchable
evidence pool CS, up to 326M tokens) from the
model input scale N(a model’s maximum con-
text length per invocation). For evidence access,
we retrieve top- Kdocument IDs from CSand re-
port R@1 (single-source), SR@10, and FR@10
6

Table 3:Retrieval performance (RAG / evidence access).Results are reported on four representative scales on
the reference-corpus ladder: 64K, 512K, 30M, and 326M. We reportR@1(single-source queries only),SR@10
(standard recall@10 over all queries), andFR@10(strict recall@10 on multi-source queries requiring all references).
We additionally report reranking results (Qwen3-Reranker-8B) where available.
Model 64K 512K 30M 326M
R@1 SR@10 FR@10 R@1 SR@10 FR@10 R@1 SR@10 FR@10 R@1 SR@10 FR@10
Embedding models
SFR-Embedding-Mistral 0.858 0.874 0.648 0.845 0.864 0.621 0.750 0.755 0.436 0.554 0.607 0.209
BGE-M3 0.831 0.861 0.642 0.811 0.847 0.615 0.696 0.724 0.397 0.480 0.591 0.173
Snowflake-Arctic-M-v2.0 0.831 0.848 0.609 0.818 0.827 0.573 0.736 0.703 0.361 0.446 0.572 0.173
Qwen3-Embedding-8B 0.8720.930 0.8000.8650.928 0.7880.804 0.831 0.585 0.622 0.682 0.304
Qwen3-Embedding-4B 0.872 0.921 0.767 0.858 0.912 0.749 0.804 0.818 0.561 0.588 0.675 0.304
Qwen3-Embedding-0.6B 0.851 0.897 0.716 0.831 0.884 0.684 0.784 0.778 0.478 0.568 0.644 0.248
KaLM-Embedding-Gemma3-12B0.9050.908 0.7340.8920.899 0.713 0.818 0.806 0.540 0.622 0.657 0.263
Sparse retrieval
BM25 0.655 0.687 0.358 0.622 0.616 0.266 0.514 0.461 0.116 0.345 0.327 0.030
Rerank (Qwen3-Reranker-8B)
Qwen3-Embedding-8B 0.872 0.913 0.770 0.865 0.908 0.758 0.8180.846 0.600 0.689 0.745 0.400
KaLM-Embedding-Gemma3-12B 0.885 0.911 0.761 0.865 0.903 0.7400.8380.834 0.567 0.676 0.739 0.391
Table 4:Generative QA quality (LLM-as-a-Judge, 0–
5).Full-context results at 64K/256K/1M when feasible.
Model Context window 64K 256K 1M
Full-capacity Models
Qwen3-235B-A22B-Instruct 256K 3.35 3.20 -
Qwen3-Next-80B 256K 3.35 3.30 -
DeepSeek-V3.2 128K 3.40 - -
GLM-4.7 200K 3.18 - -
Gemini-3-Pro-Preview 1M3.55 3.45 3.28
Efficiency-oriented Models
GLM-4.7-Flash 200K 3.10 - -
MiniMax-M2.1 204K 3.38 - -
Grok-4.1-Fast 2M 3.45 3.36 3.01
Kimi-Linear-48B-A3B-Instruct 1M 3.42 3.38 3.22
Gemini-3-Flash-Preview 1M3.48 3.40 3.25
(strict multi-source). For end-to-end QA, we an-
swer queries given the full reference corpus as con-
text whenever the full input fits within N, scored by
LLM-as-a-Judge on a 0–5 scale (Grok-4; prompt
in Figure 6). We adopt adiagnosticprotocol with a
shared document-ID evidence interface that reports
evidence access separately from end-to-end QA
quality. Evidence access is evaluated via document-
ID localization/recall over CS, while QA scores
are reported under full-context prompting when
feasible. Appendix A.3 defines the document-ID
interface and LLM output canonicalization.
We define a query as single-source if its gold evi-
dence set contains exactly one reference document,
and multi-source otherwise.
Evaluation regimes. (i) For RAG-style sys-
tems, we evaluateevidence accessonly, measuring
whether the retriever surfaces the gold reference
documents as the searchable corpus scales along
the ladder. (ii) For native long-context LLMs, we
evaluate end-to-end QA by scoring answers pro-
duced with the full reference corpus provided as
64K 128K 256K 512K 1M 10M 30M 50M 100M 200M 326M
Corpus Scale (Number of Documents)BM25
SFR-Emb-Mistral
BGE-M3
Snowflake
Arctic-M-v2.0
Qwen3-Emb-0.6B
Qwen3-Emb-4B
Qwen3-Emb-8B
KaLM-Emb
Gemma3-12B
Qwen3-Emb-8B
(Rerank)
KaLM-Emb-Gemma3
12B (Rerank)0.69 0.68 0.64 0.62 0.60 0.51 0.46 0.44 0.39 0.36 0.33
0.87 0.87 0.87 0.86 0.85 0.81 0.76 0.73 0.68 0.64 0.61
0.86 0.86 0.86 0.85 0.84 0.78 0.72 0.70 0.66 0.61 0.59
0.85 0.84 0.84 0.83 0.81 0.75 0.70 0.68 0.64 0.60 0.57
0.90 0.90 0.89 0.88 0.88 0.83 0.78 0.75 0.71 0.67 0.64
0.92 0.92 0.92 0.91 0.91 0.86 0.82 0.80 0.76 0.71 0.68
0.93 0.93 0.93 0.93 0.92 0.87 0.83 0.80 0.76 0.72 0.68
0.91 0.91 0.91 0.90 0.89 0.85 0.81 0.77 0.73 0.69 0.66
0.91 0.91 0.91 0.91 0.90 0.87 0.85 0.83 0.80 0.76 0.74
0.91 0.91 0.91 0.90 0.90 0.86 0.83 0.82 0.78 0.76 0.74Retrieval SR@10 Results
0.000.250.500.751.00Figure 5: Retrieval heatmap across the reference corpus
ladder (64K–326M): evidence access degrades as the
searchable pool expands and semantic interference in-
creases.
context whenever it fits within the model’s context
window N. Appendix A provides full implementa-
tion details.
4.3 Evidence Access under Semantic
Interference (RQ1–RQ2)
This section evaluates how well retrievers localize
gold evidence as corpus scale and interference in-
crease. We report results for dense retrievers, a
sparse BM25 baseline, and a reranking-enhanced
variant as an auxiliary analysis.
Figure 5 visualizes SR@10 degradation across
the full corpus ladder, while Table 3 reports repre-
sentative scales (64K/512K/30M/326M) with R@1,
SR@10, and FR@10.
As corpus scale increases from 64K to 326M,
evidence access degrades substantially across all
retrievers. For example, a strong dense retriever
drops from SR@10 ≈0.93 at 64K to ≈0.68 at
326M, while FR@10 declines more sharply (e.g.,
7

≈0.80→0.30 ). In contrast, BM25 degrades
rapidly under interference, with FR@10 approach-
ing zero at large scale.
Across embedding models, Qwen3-Embedding-
8B attains the best SR@10 and FR@10 at all
four reported scales (e.g., SR@10 0.930 →0.682
and FR@10 0.800 →0.304 from 64K to 326M),
while KaLM-Embedding-Gemma3-12B yields the
strongest R@1 at 64K/512K (0.905/0.892) and
ties Qwen3-Embedding-8B at 326M (0.622). The
SR@10–FR@10 gap widens from 0.13 at 64K to
0.38 at 326M for Qwen3-Embedding-8B, under-
scoring multi-source brittleness.
Single-source vs. multi-source retrieval (RQ2).
Across all scales, FR@10 is consistently far be-
low SR@10, and the gap widens as interference
increases. This demonstrates that retrievingallre-
quired evidence documents is substantially harder
than retrieving a single relevant document, making
multi-source queries a strict stress test for evidence
access.
Auxiliary analysis: reranking as mitigation.
While not a primary benchmark axis, reranking pro-
vides a partial mitigation under heavy interference.
At 326M, reranking lifts Qwen3-Embedding-8B
from SR@10 0.682 to 0.745 and FR@10 0.304
to 0.400; for KaLM-Embedding-Gemma3-12B,
SR@10 improves from 0.657 to 0.739 and FR@10
from 0.263 to 0.391. This suggests that hard-
negative discrimination becomes increasingly valu-
able as corpus scale grows.
4.4 End-to-end Long-context QA under Full
Context (RQ3)
We evaluate end-to-end question answering using
LLM-as-a-Judge scores (0–5) under full-context
prompting. We report results on representative
scales (64K/256K/1M; Table 4) only when the
full input fits within each model’s context window
(up to 1M here); missing entries indicate that the
full input at that scale exceeds the model’s con-
text window and thus cannot be evaluated. The
best scores come from Gemini-3-Pro-Preview at
64K/256K/1M (3.55/3.45/3.28).
Two consistent patterns emerge. First, answer
quality can drop as the full-context scale increases
(e.g., Gemini-3-Pro-Preview 3.55 →3.28 from 64K
to 1M), reflecting the increasing difficulty of full-
context QA under heavier semantic interference.
Second, larger context windows alone do not guar-
antee better answer quality: models with compara-ble or larger windows can still trail behind the best-
performing model at the same evaluated scales.
4.5 Key Takeaways
Our experiments yield three takeaways aligned
with RQ1–RQ3. (1) Scaling induces access fail-
ures: evidence access degrades substantially as
corpus scale and interference increase. (2) Multi-
source retrieval is the critical stress point: the
widening SR@10–FR@10 gap shows that satis-
fying multiple evidence constraints is particularly
fragile. (3) Long-context capacity is not suffi-
cient: even when full-context evaluation is feasi-
ble, full-context QA quality can degrade with scale
and varies substantially across models. Together,
these findings demonstrate that success on benign
long-context settings does not predict robustness
in high-interference environments, and that effec-
tive long-context systems require robust evidence
access paired with strong reasoning.
Limitations
While this work establishes a rigorous benchmark
for large-scale memory, four limitations warrant
further discussion.First, EMB-S prioritizes high-
precision instance validation and hard-negative ver-
ification, which yields a relatively compact set;
with additional expert time and expanded source
pools, the same pipeline can scale to broader cov-
erage and more diverse failure modes.Second,
our document-ID localization protocol for native
LLMs (Appendix A.3) introduces an explicit evi-
dence interface that may not reflect typical long-
context usage; despite output canonicalization, this
setup can introduce formatting and calibration bi-
ases when comparing against dedicated retrievers.
Third, parts of our construction and evaluation rely
on specific tools and models (e.g., Grok-4 as the
judge; Qwen3-Embedding-8B for difficulty cali-
bration and collision testing); while we mitigate
this with human screening and human evaluation,
future work should replicate results with alterna-
tive judges and multi-retriever mining.Fourth,
we do not account for inference-time efficiency
gaps between full-context prompting and retrieval
pipelines (e.g., latency and GPU memory), which
can be substantial at million-token inputs; EMB-S
is intended to diagnose robustness under semantic
interference rather than to advocate one paradigm
over the other.
8

References
Chenxin An, Shansan Gong, Ming Zhong, Xingjian
Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and
Xipeng Qiu. 2024. L-eval: Instituting standardized
evaluation for long context language models. InPro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 14388–14411, Bangkok, Thailand.
Association for Computational Linguistics.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024.
Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks.arXiv
preprint arXiv:2412.15204.
Yushi Bai, Yutao Zheng, Jiawei Zhang, Yida Chen, Xin
Dong, Yu Su, and Shuo Wang. 2023. LongBench:
A Bilingual, Multitask Benchmark for Long Context
Understanding.arXiv preprint arXiv:2308.14508.
Mohamed Amine Ferrag, Norbert Tihanyi, and Mer-
ouane Debbah. 2025. From llm reasoning to au-
tonomous ai agents: A comprehensive review.arXiv
preprint arXiv:2504.19678.
Omar Friel, John Morris, Omar Khattab, Koustuv
Santhanam, Kartik Talamadupula, Cédric Glanois,
Felipe de Vassimon, Cristiano de Souza, Arash
Einolghozati, Sriram Subramanian, and 1 others.
2024. RAGBench: A Comprehensive Benchmark
for Retrieval-Augmented Generation.arXiv preprint
arXiv:2407.03224.
Yunfan Gao, Yun Xiong, Wenlong Wu, and 1 others.
2025. U-NIAH: Unified RAG and LLM evaluation
for long context needle-in-a-haystack.arXiv preprint
arXiv:2503.00353.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
and Boris Ginsburg. 2024. RULER: What’s the Real
Context Size of Your Long-Context Language Mod-
els?arXiv preprint arXiv:2404.06654.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024.
LongRAG: Enhancing retrieval-augmented gener-
ation with long-context LLMs.arXiv preprint
arXiv:2406.15319.
Shyam Krishna, Karthik Radhakrishnan, Urmish
Thakker, Bharath Karthik, Avinash Madasu, Karan
Aggarwal, Sufian Hossain, Jivi Shah, Sai Sreeharsha,
Joydeep Biswas, Ghazal Fazelnia, Maziar Sanjabi,
Drew Stokes, Tom Goldstein, and Vahid Alizadeh.
2024. FRAMES: A Benchmark for Few-shot Eval-
uation of RAG-in-the-loop Systems.arXiv preprint
arXiv:2409.12941.
Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rod-
kin, Dmitry Sorokin, Artyom Sorokin, and Mikhail
Burtsev. 2024. Babilong: Testing the limits of llms
with long context reasoning-in-a-haystack. InAd-
vances in Neural Information Processing Systems.Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. 2024.
Long context vs. RAG for LLMs: An evaluation and
revisits.arXiv preprint arXiv:2412.18050.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the Middle: How Language
Models Use Long Contexts.Transactions of the As-
sociation for Computational Linguistics.
Yuanjie Lyu, Zhiyu Li, Simin Niu, and 1 others. 2024.
CRUD-RAG: A comprehensive chinese benchmark
for retrieval-augmented generation of large language
models.arXiv preprint arXiv:2401.17043.
Inbal Magar and Roy Schwartz. 2022. Data Contami-
nation: From Memorization to Exploitation. InPro-
ceedings of the 60th Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 2: Short
Papers), pages 157–165, Dublin, Ireland. Association
for Computational Linguistics.
Shuting Wang, Jiongnan Liu, Shiren Song, and 1 oth-
ers. 2024. DomainRAG: A chinese benchmark for
evaluating domain-specific retrieval-augmented gen-
eration.arXiv preprint arXiv:2406.05654.
Weizhi Xiong, Yifei Li, Qian Cheng, Jian Huang, Zhi-
Hong Liu, Feida Huang, Bo Li, and Dong Li. 2023.
LLMs Cannot Find a Needle in a Haystack: The
Role of Long-Context Memory.arXiv preprint
arXiv:2311.07997.
Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun
Zhao, Roy Bar-Haim, Arman Cohan, and Michal
Shmueli-Scheuer. 2025. Survey on evaluation of llm-
based agents.arXiv preprint arXiv:2503.16416.
Tao Yuan, Xuefei Ning, Dong Zhou, Zhijie Yang,
Shiyao Li, Minghui Zhuang, Zheyue Tan, Zhuyu
Yao, Dahua Lin, Boxun Li, Guohao Dai, Shengen
Yan, and Yu Wang. 2024. Lv-eval: A balanced long-
context benchmark with 5 length levels up to 256k.
Preprint, arXiv:2402.05136.
Guanting Zhang, Xin Mao, Yiqun Zhang, Jia Chen,
Ruizhe Li, and Zhi Chen. 2024a. Loong: A Simple,
Versatile, and Unbiased Benchmark for Long Context
LLMs.arXiv preprint arXiv:2405.15786.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang
Xu, Zhaofeng He, Yujia Qin, Chao Yin, Fandong
Meng, Jie Zhou, Zhiyuan Liu, and Maosong Sun.
2024b. InfiniteBench: Extending Long-Context
Evaluation Beyond 1M Tokens.arXiv preprint
arXiv:2406.09697.
9

A Implementation Details
A.1 Construction Pipeline Details
Overview.Figure 2 (main paper) summarizes the
end-to-end construction pipeline, and Table 2 re-
ports the instance counts across stages. This ap-
pendix provides additional details omitted from
the main text, including stage-level criteria, human
screening guidelines, and LLM verification criteria.
A.1.1 Stage I: Standardization and Human
Screening
Stage I converts heterogeneous sources into stan-
dardized triples and removes low-quality items be-
fore multi-hop synthesis. We implement two steps:
•Step 1 (Standardization):convert raw instances
into(Query, Answer, RefDoc), where RefDoc
is the reference document that must be accessed
to derive the answer from the query.
•Step 2 (Human screening):validate that the query
is well-posed and that RefDoc supports the an-
swer; discard ambiguous, unsupported, or noisy
instances.
A.1.2 Stage II: Reasoning-Chain Synthesis
(Two Tracks)
Stage II constructs reasoning-intensive stress tests
by constructing the gold evidence setRefList,
which may contain one document (single-source)
or multiple documents (multi-source). To create
multi-source items, we expand RefDoc into a multi-
document RefList via two complementary tracks:
•Track 2 (RefDoc atomization):for RefDoc with
complex internal logic, weatomizeit into mul-
tiple sub-documents so that no single piece is
sufficient.
•Track 1 (Retrieval-guided query rewriting):re-
trieve semantically similar documents from the
MemoryBank, select related documents that pro-
vide complementary constraints, and rewrite the
query so that answering requires both the original
RefDoc and newly added supporting documents.
We then perform difficulty calibration and dedu-
plication using a strong dense retriever (Qwen3-
Embedding-8B) to produce 882 candidate queries
(Table 2).
A.1.3 Stage III: Collision Testing and LLM
Verification
Stage III performs adversarial quality control to (i)
remove inconsistent samples and (ii) collect val-
idated distractors. For each candidate query, weretrieve the top- knearestnon-referencedocuments
under dense embedding similarity from the 326M-
token MemoryBank usingQwen3-Embedding-8B
(excluding documents inRefList). An LLM then
verifies each retrieved candidate against(Query,
Answer, RefList)and assigns one of three out-
comes:
•Conflict:the candidate contradicts the query/an-
swer/reference content; we discard the entire
sample.
•Hard Negative:semantically similar but does
not support the answer; we retain it as an adver-
sarial distractor.
•False Negative:provides additional valid evi-
dence; we add it to RefList.
A.2 Evaluation Scenarios and Input Settings
To keep the main text focused on results, we sum-
marize the input settings used in Section 4 and
Table 4 here. All QA scores are produced by the
same LLM-as-a-Judge protocol (Appendix A).
Window-aware full-context evaluation.For a
model with maximum context window N, we only
run full-context experiments on ladder scales S
whosetokenizedinput (prompt + query + context)
fits withinN.2
We report one QA input setting at each scaleS:
•GLOBAL:provide the full reference corpus CS
as the model’s context (reported only when the
full input fits withinN; up to 1M in Table 4).
A.3 Document-ID Evidence Interface and
Localization Protocol
To support architecture-agnostic evaluation of evi-
dence access, EMB-S uses a shareddocument-ID
evidence space. Each document in the Memory-
Bank is assigned a unique integer ID, and each
ladder corpus CSis a subset of these ID-labeled
documents. Retrieval systems therefore return doc-
ument IDs directly, and are scored by exact ID
matching (Section A).
Localization (evidence access).Given a query
and a searchable corpus CS, a system outputs the
top-Kdocument IDs it considers relevant. For
native long-context LLMs, we include explicit
document-ID headers when serializing a corpus
into a single input (e.g., [DocID=123] preceding
2The ladder is constructed with fixed token budgets and
prompt overhead reserved (Section 3.4), but we still enforce
the constraint using the model’s tokenizer for safety.
10

each document) and apply a simple output canoni-
calization: we extract integer IDs from the model
output in order and keep the first Kunique IDs.
This reduces formatting variance while preserving
the underlying evidence ranking signal.
A.4 Raw Data Sources and Characteristics
Source Scale Task Scope Focus
LongBench (2023) 2M Retr. LLM Summ.
L-Eval (2024) 200K QA LLM Summ.
InfiniteBench (2024b) 2M QA LLM Compl.
BABILong (2024) 10M QA LLM Reas.
Loong (2024a) 200K QA LLM Reas.
NeedleBench (2023) 10K Retr. LLM Trace
RULER (2024) 128K QA LLM Trace
RAGBench (2024) 128K Both RAG M-hop
LV-Eval (2024) 256K QA LLM Bal.
EMB-S (Ours) 326M Both All All
Table 5: Raw data sources for EverMemBench-S.Scope:
target system (LLM/RAG/Memory). Sources span 10K–
10M tokens.
A.5 Retrieval Metric Calculation
Retrieval performance is evaluated using exact doc-
ument identifier matching. For a given query q, let
Gq={d 1, d2, ..., d m}be the set of ground-truth
reference document IDs (single-source queries
have|Gq|= 1 ; multi-source queries have |Gq|>
1), and Rq@K={r 1, r2, ..., r k}be the set of top-
Kdocument IDs retrieved by the model. We report
thestandard recall(denoted asSR@Kin Table 3):
SR@K=|Rq@K∩G q|
|Gq|(1)
For completeness, we also reportR@1on the
single-source subset, where the metric reduces to a
binary indicator of whether the unique gold docu-
ment is ranked at position 1.
We further reportFR@Kon the multi-source
subset, which requires retrievingallreference doc-
uments:
FR@K=⊮[G q⊆R q@K].(2)
We report FR@K averaged over queries with
|Gq|>1.
This strict ID matching ensures that the model
retrieves the exact evidence required for reasoning,
rather than merely semantically similar documents
which may be hard negatives.
A.6 LLM-as-a-Judge Prompt Template
For the Generative QA evaluation, we employ the
following prompt template for LLM-as-a-Judgescoring. The judge is instructed to outputa single
integer scorein{0, . . . ,5}(Figure 6).
LLM-as-a-Judge Scoring Prompt (0–5)
Instruction:Based on the accuracy, completeness, and
relevance of the predicted answer to the real answer in the
context of thequery, assign an objective score from0 to
5(5 being the highest, 0 the lowest). The final output can
only be a single number.
Scoring Criteria:
•5:The predicted answer is exactly the same as the real
answer and correctly answers the query. Differences in
wording do not affect factual accuracy.
•4:The predicted answer contains all the core informa-
tion of the real answer, with no errors, but includes a
small amount of non-critical redundant content.
•3:The predicted answer captures the core information
but differs from the real answer in some aspects. The
predicted answer is slightly incomplete or imprecise,
but contains no errors.
•2:The predicted answer is partially relevant to the real
answer but omits a significant amount of information or
deviates from the core topic of the query.
•1:The predicted answer attempts to address the query
(maintains basic relevance to the topic) but provides
factually incorrect information. It does not contradict
the core claim of the real answer, but shows incomplete
or inaccurate understanding of the topic.
•0:The predicted answer is completely unrelated to the
query, consists of gibberish, or is a pure hallucination
that shares no logical connection with the real answer.
Query:{query}
True Answer:{reference_answer}
Predicted Answer:{generated_answer}
Output only a single number (0, 1, 2, 3, 4, or 5):
Figure 6:LLM-as-a-Judge scoring prompt.The
prompt used to score generated answers with a single
integer score in {0,. . . ,5}.
A.7 Judge Model and Human Sanity Check
Judge model.Unless otherwise noted, all LLM-as-
a-Judge scores in Table 4 are produced by Grok-4
using the prompt in Figure 6. The judge input
contains only the {query, true answer, predicted
answer} triplet (no system identifiers) to reduce
potential preference biases.
Human sanity check.To complement auto-
mated scoring, we additionally perform a small-
scale human review on randomly sampled QAR in-
stances, rating completeness and correctness with
a 10-point rubric as a sanity check.
11
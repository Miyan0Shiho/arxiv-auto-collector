# Citation-Closure Retrieval and Per-Rule Attribution for Real-World Regulatory Compliance Question Answering

**Authors**: Yeong-Joon Ju, Seong-Whan Lee

**Published**: 2026-05-28 10:38:38

**PDF URL**: [https://arxiv.org/pdf/2605.29742v1](https://arxiv.org/pdf/2605.29742v1)

## Abstract
Deploying Large Language Models (LLMs) for regulatory compliance demands rigorous traceability via comprehensive citations across multi-tiered authority structures. Unlike traditional multi-hop or legal QA, this task requires structured procedural lookups and evidence-set closure rather than entity resolution or case-law reasoning. Existing RAG systems struggle here due to flattened citation edges, fragmented retrieval expansions, and fragile post-hoc attribution. We formalize Regulatory Compliance QA with RegOps-Bench, a novel benchmark featuring an Operational Knowledge Graph derived from complex national R\&D regulations. To address these bottlenecks, we propose RefWalk, a unified framework driven by a shared topic anchor. RefWalk traverses cross-document citations, fuses multi-view candidates via max-based aggregation, and enforces per-rule attribution to explicitly map claims to sources. We establish a strong baseline with substantial improvements in retrieval recall and citation accuracy. Finally, a contrastive evaluation on a U.S. health compliance dataset (HIPAA) reveals that existing systems exhibit saturation on flat-structure rules, underscoring the need for RegOps-Bench. Our code is available at https://github.com/yeongjoonJu/RefWalk.

## Full Text


<!-- PDF content starts -->

Citation-Closure Retrieval and Per-Rule Attribution for Real-World
Regulatory Compliance Question Answering
Yeong-Joon Ju1, Seong-Whan Lee1
1Department of Artificial Intelligence, Korea University
{yj_ju, sw.lee}@korea.ac.kr
Abstract
Deploying Large Language Models (LLMs)
for regulatory compliance demands rigorous
traceability via comprehensive citations across
multi-tiered authority structures. Unlike tradi-
tional multi-hop or legal QA, this task requires
structured procedural lookups and evidence-set
closure rather than entity resolution or case-law
reasoning. Existing RAG systems struggle here
due to flattened citation edges, fragmented re-
trieval expansions, and fragile post-hoc attribu-
tion. We formalize Regulatory Compliance QA
with RegOps-Bench, a novel benchmark featur-
ing an Operational Knowledge Graph derived
from complex national R&D regulations. To ad-
dress these bottlenecks, we propose RefWalk,
a unified framework driven by a shared topic
anchor. RefWalk traverses cross-document ci-
tations, fuses multi-view candidates via max-
based aggregation, and enforces per-rule at-
tribution to explicitly map claims to sources.
We establish a strong baseline with substantial
improvements in retrieval recall and citation
accuracy. Finally, a contrastive evaluation on
a U.S. health compliance dataset (HIPAA) re-
veals that existing systems exhibit saturation on
flat-structure rules, underscoring the need for
RegOps-Bench. Our code is available at https:
//github.com/yeongjoonJu/RefWalk.
1 Introduction
Regulated organizations operate under strict, lay-
ered frameworks of statutes, enforcement decrees,
ministerial rules, notices, and operational manuals.
Since non-compliance carries severe risks, compli-
ance staff routinely navigate complex procedural
questions whose grounds span these multiple au-
thority tiers (Arner et al., 2018). While Large Lan-
guage Models (LLMs) offer a promising avenue to
alleviate this burden, deploying them to assist with
such inquiries requires rigorous traceability (Ariai
et al., 2025; Liu et al., 2023). To establish a veri-
fiable audit trail, models should quote controllingarticles and explicitly map every generated claim
back to its source.
In this paper, we formalize this objective as Reg-
ulatory Compliance Question Answering (QA),
where a model addresses a practitioner’s query by
retrieving the exhaustive set of governing articles
and detailing the precise claims derived from each
rule. As modern regulatory systems evolve into
highly complex cross-reference networks across
multiple documents (Katz et al., 2020; Ruhl and
Katz, 2015), this task structurally diverges from
both standard multi-hop and legal QA paradigms.
Unlike standard multi-hop QA paradigms focusing
on entity resolution (Yang et al., 2018; Trivedi et al.,
2022), hop transitions in this QA task follow typed
citation rules, and search termination relies on com-
plete evidence-set closure rather than finding a sin-
gle final entity. Furthermore, while traditional legal
QA tasks (Guha et al., 2023; Li et al., 2024; Yang
et al., 2026) largely center on judicial interpretation
and case-law reasoning, recent regulatory NLP has
expanded into procedural clause retrieval (Louis
and Spanakis, 2022) and corporate sustainability
extraction (Ali et al., 2025). However, these ex-
isting benchmarks predominantly operate on flat
rule structures or isolated reports, failing to cap-
ture the mechanics of hierarchical compliance. In
contrast, Regulatory Compliance QA demands nav-
igating multi-tiered, cross-document delegations to
achieve deterministic evidence-set closure.
These rigorous demands expose two critical fail-
ure modes in existing RAG systems. The first is
the inability to execute precise structural retrieval.
Current graph-based RAG approaches (Edge et al.,
2024; Guo et al., 2025; Gutiérrez et al., 2025a;
Ma et al., 2025; Peng et al., 2025) typically
flatten explicit regulatory citations into generic
entity-relation edges, stripping away the seman-
tic distinctions necessary for procedural naviga-
tion. Because they rely on global signal propaga-
tion across surface-level entity overlaps, standard
1arXiv:2605.29742v1  [cs.AI]  28 May 2026

entity-centric graphs inevitably fail to resolve the
rigid, multi-tiered delegation pathways inherent
in regulatory networks. Moreover, because regu-
latory queries often omit specific situational con-
straints, systems must exhaustively identify all po-
tential conditional branches and legal exceptions.
Attempts to resolve this via query expansion or de-
composition (Gao et al., 2023b; Rackauckas, 2024;
Wang et al., 2023; Petcu et al., 2026) typically rely
on surface-level paraphrasing, produce disjoint sub-
queries that fragment rather than unify the required
evidence set.
The second major limitation emerges during
answer generation, where attribution is typically
treated as a post-hoc afterthought (Saxena et al.,
2025). In regulatory compliance, the risk of a hal-
lucinated citation outweighs the benefit of a broad
response. However, rather than enforcing a schema-
level binding between individual claims and their
sources, generative models generally append cita-
tions as free-form footers, leading to systemic attri-
bution failures (Bohnet et al., 2022; Liu et al., 2023;
Hou et al., 2025). Ultimately, this lack of structural
binding between claims and their sources fails to
provide the rigorous traceability required, leaving
a significant gap in the reliability of regulatory AI.
To systematically diagnose these limitations, we
introduce RegOps-Bench, an evaluation framework
serving as a testbed for multi-tier procedural navi-
gation, instantiated via a highly structured Korean
national R&D regulatory corpus. This corpus fea-
tures a five-tiered authority structure that exempli-
fies the nested complexities inherent in real-world
administrative regulations. We model this corpus
into an Operational Knowledge Graph (OKG) and
construct 250 high-quality QA pairs grounded in
real-world inquiries from the Institute of Informa-
tion & Communications Technology Planning &
Evaluation (IITP). By expanding these inquiries
using a novel axis-decoupling principle, we inde-
pendently control the substantive intent of a query
and the structural complexity of its required refer-
ence set, spanning from straightforward lookups
to exception-heavy procedural branching. This or-
thogonal design precisely isolates whether a system
fails due to navigating a dense regulatory hierarchy
or resolving complex procedural logic.
To overcome the identified retrieval and genera-
tion bottlenecks, we propose RefWalk, a structural
traversal framework that navigates regulatory ci-
tation pathways guided by a shared topic anchor.
RefWalk mitigates structural retrieval failures byexploring the OKG through three distinct semantic
views, restricting hop expansion strictly to cross-
document citation edges to eliminate internal con-
tainment noise. To preserve the specialized signals
required across different difficulty tiers, candidates
are fused using Reciprocal Rank MAX (RRM)
rather than standard sum-based aggregation (Cor-
mack et al., 2009), which otherwise dilutes crucial
specialist cues. During generation, RefWalk tackles
systemic attribution failures by injecting the same
anchor alongside a per-rule schema. Instead of ap-
pending citations as a post-hoc afterthought, we
structure the model to generate claims directly as
attributes of their governing rules. This approach
inherently mitigates attribution hallucinations and
binds generation to its source, advancing the trace-
ability required for professional practice. Finally,
we demonstrate the broader applicability of this
evaluation framework and RefWalk by validating
both on a HIPAA-derived dataset.
In summary, our main contributions are three-
fold. First, we formalize the task of Regulatory
Compliance QA and release RegOps-Bench, the
first benchmark for deterministic traversal of multi-
tier regulatory hierarchies. Second, we propose Re-
fWalk, a unified RAG framework that navigates
explicit structural delegations, establishing a robust
baseline for complex procedural lookups. Third,
we outperform state-of-the-art RAG methods on
complex cross-reference tasks. Furthermore, abla-
tion studies confirm the impact of our schema-level
binding, while contrastive experiments on HIPAA
validate the necessity of multi-tiered evaluation.
2 Related Work
2.1 Regulatory NLP and Structural Retrieval
Legal NLP has increasingly shifted from coarse
classification (Chalkidis et al., 2020; Guha et al.,
2023) toward reasoning- and retrieval-intensive
evaluation (Yang et al., 2026; Hou et al., 2025).
Within this space, regulatory compliance presents
unique challenges distinct from case-law anal-
ogy, requiring the navigation of complex, multi-
document networks (Katz et al., 2020; Ruhl and
Katz, 2015; Sleimi et al., 2018) to meet the de-
mands of RegTech (Arner et al., 2018). While De
Jure (Guliani et al., 2026) focuses on structuring
raw regulations into rule sets, we leverage its evalu-
ation pipeline for our HIPAA data generation. How-
ever, existing benchmarks still fail to formalize QA
across layered, cross-document procedural chains.
2

Standard multi-hop QA (Yang et al., 2018;
Trivedi et al., 2022; Ho et al., 2020) and graph-
based RAG (Edge et al., 2024; Guo et al., 2025;
Gutiérrez et al., 2025a; Ma et al., 2025; Wang et al.,
2025) typically attempt complex retrieval by prop-
agating through entity-centric relations. Similarly,
query decomposition (Gao et al., 2023b; Wang
et al., 2023; Rackauckas, 2024; Petcu et al., 2026;
Khot et al., 2023; Trivedi et al., 2023; Jiang et al.,
2023) divides queries into independent sub-facts.
2.2 Attributed Generation and Citation
Faithfulness
Verifiable grounding is widely recognized as a strict
deployment prerequisite for legal AI (Ariai et al.,
2025; Hou et al., 2025). Consequently, citation
faithfulness has emerged as a central evaluation
axis, driving the development of specialized attri-
bution benchmarks (Bohnet et al., 2022; Rashkin
et al., 2023; Liu et al., 2023; Gao et al., 2023c) and
methods that fold retrieval or revision decisions
into the generation policy (Asai et al., 2024; Gao
et al., 2023a). Despite these advances, audits of
off-the-shelf LLMs and domain-specific legal sys-
tems (Liu et al., 2023; Hou et al., 2025) consistently
reveal that a substantial fraction of claims remain
uncited or misattributed. This occurs because exist-
ing systems treat attribution as post-hoc annotation,
thereby lacking structural guarantees.
3 RegOps-Bench: Axis-Decoupled
Construction for Compliance QA
RegOps-Bench is designed around three core prop-
erties of this domain: (1) ground truth is a typed ci-
tation closure, (2) difficulty is defined by the struc-
tural complexity of the reference set rather than
lexical phrasing, and (3) evaluation operates at the
regulatory unit where authority delegates.
3.1 Operational Knowledge Graph (OKG)
Corpus.To anchor the benchmark in real-world
compliance scenarios, we curated a corpus of 12
Korean R&D regulatory documents covering the
scope of 56 seed FAQs from an official FAQ docu-
ment1of the Institute of Information & Communi-
cations Technology Planning & Evaluation (IITP).
These documents span five authority tiers (1: statute
– 5: manual), from statutory acts down to oper-
ational manuals (Table 1). The seven in-domain
1https://www.iitp.kr/web/lay1/bbs/S1T46C59/A/
13/view.do?article_seq=4331&sort=latest&cpage=1&
rows=10Table 1:RegOps-Bench corpus.12 Korean R&D regu-
lations spanning the statute-to-manual delegation chain.
#Art. counts articles; #Prov. counts paragraphs, unde-
fined (—) for the section-structured manual.
Document Authority Type Tier #Art. #Prov.
In-domain (7)
Innovation Act legal authority 1 42 161
Enforcement Decree executive decree 2 75 225
Enforcement Rules executive rule 3 4 7
Cost-Use Standards admin. notice 4 125 362
Mgmt. Regulation admin. notice 4 54 207
Standard Guide admin. notice 4 56 125
Practitioner’s Manual manual 5 40 —
Auxiliary (2)
V AT Act legal authority 1 85 294
V AT Act Rules executive rule 3 85 153
Distractor (3)
S&T Basic Act legal authority 1 71 208
S&T Basic Act Decree executive decree 2 72 209
S&T Basic Act Rules executive rule 3 9 16
Total 718 1967
documents form a largely self-contained delega-
tion network. While certain high-level mandates
inevitably delegate to external authorities, cross-
document references, including all FAQ lineages,
are mostly resolved within this set, where each
lower tier procedurizes the open-textured mandates
of the tier above it. A single compliance answer
therefore requires composing a statutory obliga-
tion with its decree-level conditions and notice-
level numeric thresholds. Two auxiliary V AT docu-
ments cover the tax-side cross-references recurring
in cost-eligibility questions. The remaining three
Science & Technology Basic Act documents serve
as distractors, sharing lexical overlap with the in-
domain corpus but remaining legally out of scope.
OKG Construction.We construct the OKG via
deterministic rule extraction. By leveraging the for-
mulaic citation patterns of legislation, this design
effectively mitigates the generative hallucinations
inherent in LLM-based indexing and provides de-
terministic edge typing—a prerequisite for eval-
uating closure. Furthermore, this approach elimi-
nates heavy LLM computational overhead, making
the index highly sustainable and easily updatable
under frequent regulatory amendments. Each arti-
cle (조) forms a node carrying its authority tier.
Inter-article relations are typed into six classes:
PART_OF (hierarchy), REFERENCES (citation), the
DELEGATES_TO /SPECIFIES pair (downward del-
egation/upward realization), DEFINES (term-to-
article), and REQUIRES_FORM (article-to-form). For
instance, “as prescribed by the Presidential Decree”
3

Table 2: Difficulty rubric for RegOps-Bench. Levels are
checked top-down: a question receives the first level
whose triggers it satisfies.
Level Characteristic Trigger
L1 single-anchor lookup|refs|= 1, not conditional
L2 conditional or 2-ref|refs| ≥2orconditional
L3 multi-hop / multi-doccross-doc, external-law,
|refs| ≥4, 4-institution parallel,
or multi-facet arity≥3
L4 conditional multi-hopconditionaland(cross-doc,
sanction, multi-institution,
or multi-facet arity≥3)
yields aDELEGATES_TO edge. Retrieval and cita-
tion are strictly evaluated at the article level ( 조),
the atomic unit of authority; finer paragraph ( 항)
structure is retained within nodes for condition eval-
uation, but is not a scoring target.
3.2 QA Construction with Axis Decoupling
Axis-Decoupled Augmentation via LLM.To
ensure the benchmark comprehensively evaluates
the structural complexity defined in our design
principles, we expand the 56 seed FAQs into a
final set of 250 expert-grounded questions using a
high-capacity LLM (Gemini-3-flash (Google Deep-
Mind, 2025)). Our augmentation follows an axis-
decoupling principle with two distinct axes:
•Question Type— The substantive intent of the query (e.g.,
single-clause lookup, exception-heavy, or sanction-bearing
condition).
•Difficulty Level— The structural complexity of the re-
quired reference set (L1 to L4), strictly bounded by the
quantitative triggers in Table 2.
The conditions are sampled and combined indepen-
dently, so that a single question type can realize
any of L1–L4 depending only on the anchor and
injected facets, minimizing the confounding effect
of surface-level phrasing.
For each seed, the LLM is given the governing
articles and instructed to re-synthesize a situational,
first-person practitioner inquiry matching the sam-
pled axes, injecting facets—actors, temporal con-
straints, and institutional variables—that necessi-
tate the target difficulty. For instance, to elevate a
straightforward seed to L4, the model constructs
a scenario in which an actor’s specific condition
triggers a cross-document citation to an external
disqualification provision. Difficulty thus remains
an intrinsic property of the regulatory logic (the
reference set) rather than an artifact of linguisticcomplexity. To verify that our augmentation in-
duces the targeted structural depth, we compare the
corpus characteristics of the LLM-generated splits
against the original human FAQs in Appendix A.5.
Reference Rules and Closure.To reduce human
annotator variance and ensure reproducibility, we
standardize the ground-truth reference mapping via
four deterministic expert rules, activated by formal
structural markers identified during synthesis:
•R1 (Domain Anchor):binds the query to the top-priority
controlling document of its domain—the Cost-Use Stan-
dards by default, or the domain-specific instrument other-
wise (the Mgmt. Regulation for institutional-IT queries, the
Standard Guide for facility/equipment, the V AT Act for
tax-side, the Innovation Act for statutory-procedure).
•R2 (Parallel & Deemed-Application Expansion):ex-
pands the set across the four institutional slots (government-
funded, university, non-profit, for-profit) via parallel groups,
and followsdeemed-application( 의제준용 ) forward edges
so that a deemed article additionally pulls in the provisions
it incorporates by reference.
•R3 (Pre-Approval Exception):binds exception-heavy in-
quiries to their governing pre-approval clause (Cost-Use
Standards Art. 73).
•R4 (Sanction Exhaustion):closes sanction-bearing ques-
tions onto their corresponding settlement and disqualifica-
tion provisions (Enforcement Decree Art. 26 and Cost-Use
Standards Art. 83).
The external-law case is handled by absorbing
it into the in-corpus V AT domain rather than as a
separate rule, since the V AT Act and its Rules are
already part of the corpus. This rule-based mapping
turns each QA pair from an ambiguous retrieval
task into a verifiable, deterministic traversal of the
regulatory hierarchy.
Quality Assurance Protocol.Each generated
QA pair passes a layered validation cascade before
merging. A regex runner gate first rejects forbidden
formats, such as multi-institution enumerations and
textbook-style summary or comparison requests.
Survivors are scored by an LLM judge against a
four-criterion rubric evaluating practitioner voice,
concrete context, situational framing, and a single-
institution viewpoint, yielding around a 50% ac-
ceptance rate. This is supplemented by heuristic
checks (script leakage, meta-patterns, degenerate
length) and a difficulty re-validation step that dis-
cards items whose realized reference set diverges
from the target level. A closure validator then ver-
ifies and auto-completes any missing R3 or R4-
triggered authorities. We also conduct a manual cu-
ration pass over the accepted pool to filter residual
low-quality items. Finally, ground-truth reference
4

V
OKG -Aware Retrieval
Anchor Extraction 𝝉𝒒Q. We are outsourcing research to overseas partners (University A 
 & Research Institute B) and need to increase the budget for local 
 payments. How can we manage fund transfers and documentation   
 in the Integrated Account and Information System while 
 maintaining compliance with the National R&D Innovation Act?
Topic Management of fund transfers …
Actor Institution
Temporal For budget increases
Magnitude Unspecified
Situational Joint outsourcing to University …𝑞𝑛𝑎𝑟𝑟𝑜𝑤
𝑞𝑚𝑖𝑑
𝑞𝑤𝑖𝑑𝑒DENSE SEED
3-View Queries
#1
S* - Act. 28…#4
S* - Act. 13…#8
S* - Act. 27…#25 (low)
E* - Act. 24…Dense Retrieval
E* - Act. 24
[DECREE]
N* - Act. 13
[STATUTE]SPECIFIESS* - Act. 27
[NOTICE]DELEGATES_TO
DELEGATES_TOOKG Expansion
S* - Act. 28S* - Act. 64
S* - Act. 69 S* - Act. 6S* - Act. 13Multi -View RRM Fusion Per-Rule Attribution Schema
E*-Act. 24 (Management of R&D Funds)
S*-Act. 28 (Common Usage Standards for 
International Joint R&D Funds)
S*-Act. 27 (Common Usage Standards for 
Commissioned R&D Funds)
S*-Act. 13 (Permitted Uses of International 
Joint R&D Funds)
Gold References
Node Dense 𝑞𝑛𝑎𝑟𝑟𝑜𝑤 𝑞𝑚𝑖𝑑 𝑞𝑤𝑖𝑑𝑒
E*-Act. 24 
S*-Act. 28 
S*-Act. 27 
S*-Act. 13 #25
#1
#8
#4#5
#3
#6
#9#1
#6
#4
#7#16
#2
#1
#4Rank for Per -View Rerank
E*-Act. 24 
[DECREE]
RRM  (max specialist)RRF (average of ranks)#6
#2
Attributed Rule Excerpts
S*-Acts. 28 (Standards for International Joint R&D Funds)
- [Paragraph 1] … only when the head of a central adminis…
- [Paragraph 3]  … or substituted into an account or balanc…
- [Paragraph 4]  … the foreign institution regarding the use …
- [Paragraph 5]  … who intends to change and use internati…
E*-Acts. 24 (Enforcement Decree of the National R&D …)
- [Paragraph 1]  When managing government -funded rese…
- [Paragraph 4]  The head of a research and development …
Answer
The transfer of funds to overseas co -trustees (University A 
and Research Institute B) and the … as follows:
First, the budget increase (modification) procedure . To 
increase and use international joint research and …
Second, fund transfer and accounting . Funds transferred 
or replaced from the research and development funds in …
Third, management of supporting documents . You must 
maintain supporting documents, such as contracts with …
Therefore , to ensure full compliance with the procedures 
under the Innovation Act, please increase the budget 
through the approval process of the head of the central 
administrative agency, transfer funds to the overseas 
institution’s account via a separate account …Figure 1:Overall framework of RefWalk.The examples are translated from the original Korean text for illustrative
purposes. Document abbreviations are defined in Table 1 (S*: Standards for the Use of Expenses,E*: Enforcement
Decree of Innovation Act,N*: Innovation Act).
sets undergo a deterministic post-merge audit that
corrects a clause-attribution artifact (cross-article
paragraph-token leakage) present in 38% of pre-
audit items. The ground-truth reference sets are
detailed in the Appendix A.4.
4 RefWalk: Traversing Reference Paths
for Structural Attribution
The rigorous properties of regulatory compli-
ance QA—typed citation closures and article-level
structural traversal—demand a framework where
retrieval and generation operate as a traceable
pipeline. RefWalk organizes this framework around
the topic anchor, a structured abstraction distill-
ing the query into its core procedural intent and
conditional facets. Rather than treating query pro-
cessing, graph expansion, and answer generation
as isolated, fragmented stages, this single anchor
propagates through the entire framework (Figure 1).
It simultaneously drives multi-view query construc-
tion, seeds OKG traversal, and re-enters the gener-
ation prompt to structurally constrain attribution.
4.1 Topic-Anchored Multi-View Retrieval
Anchor Extraction and Views.For a given
query q, a frozen LLM τ(q) extracts the topic an-
chor: a structured tuple of the core topic and facet
conditions (actor, temporal, magnitude, and situ-
ational context). Because regulatory QA requires
matching both explicit granular constraints and im-
plicit procedural exceptions, we construct three tar-
geted views from this shared anchor. qnarrow directly
uses the original question for dense semantic match-ing of explicit details. Conversely, qwidedrops the
raw question, relying solely on the abstracted topic
and conditions to capture structural multi-hop ref-
erences where surface phrasing is uninformative.
qmidcombines both for balanced retrieval.
OKG Expansion and RRM Fusion.Retrieval
begins by fetching an initial candidate pool Sof
sizeNusing qmidvia dense retrieval. We expand
this pool one hop along citation-bearing edges
(REFERENCES ,DELEGATES_TO ,SPECIFIES ) in the
OKG, isolating explicit delegations. To penalize
indirect evidence, 1-hop neighbors receive a de-
cay factor δ. For nodes discovered via multiple
pathways, we take the maximum decayed score to
preserve the strongest signal without inflating it
via sum-based aggregation. The expanded pool is
then scored by a cross-encoder across our three se-
mantic views and fused via Reciprocal Rank MAX
(RRM):
score(d) = max
v∈{narrow, mid, wide}1
k+rank v(d),
(1)
where kdenotes a smoothing constant. Unlike sum-
based aggregations (e.g., RRF) that demand consen-
sus, RRM ensures that candidates highly ranked by
a single specialist view retain their priority. Finally,
we introduce an authority-aware decay µfor candi-
dates sourced from lower-tier operational manuals.
Applied post-fusion, µinjects a domain-specific in-
ductive bias into the ranking: it intrinsically priori-
tizes primary statutes, yet permits strongly matched
manual passages to surface when they contain crit-
ical procedural details. This allows RefWalk to
5

maintain citation closure while navigating multi-
tiered delegations, circumventing the fragmenta-
tion of standard query decomposition.
4.2 Per-Rule Attribution Schema
In regulatory compliance, the risk of a halluci-
nated citation outweighs the benefit of a broad
response. Generative models typically treat attri-
bution as a post-hoc afterthought, appending free-
form footers that frequently mismatch the gener-
ated claims. RefWalk structurally mitigates this
vulnerability through a per-rule attribution schema.
The extracted topic anchor re-enters the framework
by being injected into the generation prompt along-
side the retrieved OKG passages and a strict JSON
schema that maps specific rule_id keys to arrays
of generated claims. Rather than generating free-
form text and retroactively appending citations, the
model must emit procedural claims exclusively as
array values bound to their governing rules. Under
this strict schema, every generated claim is inher-
ently bound to its source. This ensures that the
structural precision achieved during OKG retrieval
directly translates into the rigorous traceability re-
quired for regulatory compliance.
5 Experiments
5.1 Setup
Dataset & Models.To demonstrate the effect
of our framework, we conduct our primary eval-
uations on RegOps-Bench. We also evaluate on a
HIPAA-derived QA dataset ( n= 100 ) constructed
following the De Jure procedure (Guliani et al.,
2026). For retrieval, we use Qwen3-Embedding-
0.6B and Qwen3-Reranker-0.6B (Zhang et al.,
2025) to ensure our gains stem from our structural
design rather than brute-force embedding scale. For
generations, we employ Qwen3.5-4B (Qwen Team,
2026a) and Qwen3.6-35B (Qwen Team, 2026b)
to observe scale-dependent behaviors, alongside
Gemini-3.1-Pro (Google DeepMind, 2026) for fron-
tier model validation.
Metrics.For retrieval, we report Recall@10,
nDCG@10, and FullCov@10. We introduce Full-
Cov@10 (Full Coverage) as the fraction of queries
where the entire ground-truth reference set is suc-
cessfully retrieved within the top 10 candidates,
effectively capturing the exhaustiveness required
for compliance QA. End-to-end generation is eval-
uated using Claim F1 and Citation F1. Claim F1Table 3:Retrieval performance on RegOps-Bench
and HIPAA-derived QA (Top-10).
MethodRegOpsHIPAA
R FullCov nDCG R
BM25 33.8 23.6 26.4 94.0
Dense 54.4 35.6 45.3 90.0
Dense+Rerank 57.0 36.4 53.5 94.0
LightRAG 43.9 30.4 31.0100.0
HippoRAG-2 41.7 27.6 25.9 97.0
Query Decomp 50.1 31.2 42.7 92.0
PIKE-RAG 54.8 35.6 52.1 97.0
Ours 63.8 44.4 57.4 95.0
w/o multi-view 59.9 41.2 54.5 94.0
w/o OKG 60.3 41.2 54.8 98.0
w/o anchor 60.4 41.6 55.4 94.0
applies an LLM judge (Zheng et al., 2023) to la-
bel each predicted claim against the reference set
as {match, partial, none}, with partials weighted
0.5 before bipartite resolution (Min et al., 2023).
Citation F1 compares predicted and ground-truth
references after rolling sub-article ids ( 항/호) up to
their조-level ancestor, so credit is given for citing
the correct provision regardless of granularity (Gao
et al., 2023c). In regulatory compliance, Citation
Precision is prioritized. While a missed citation
merely results in an incomplete answer, hallucinat-
ing a legal citation poses severe operational risks.
Thus, our analysis focuses on a model’s ability to
mitigate attribution hallucinations.
Baselines.We compare RefWalk against strong
baselines across two stages. For retrieval, we
evaluate standard architectures (BM25, Dense,
Dense+Rerank) alongside state-of-the-art graph
and query-based retrievers, including LightRAG
(Guo et al., 2025), HippoRAG-2 (Gutiérrez et al.,
2025b), Query Decomposition (Petcu et al., 2026),
and PIKE-RAG (Wang et al., 2025). For end-
to-end RAG, we evaluate NativeRAG with a
Dense+Rerank pipeline as well as the four afore-
mentioned systems equipped with generation capa-
bilities. To ensure a fair comparison, all baselines
share the exact same embedding, reranking, and
generation backbones as RefWalk.
Hyperparameters.For RefWalk, we set the re-
trieval pool size to N= 50 , the OKG seed count to
M= 10 , and the RRM fusion constant to k= 60 .
Further details are represented in Appendix A.1.
6

Table 4:End-to-end RAG performance on RegOps-
Bench.w/oschema removes the per-rule attribution
schema, emitting free-form output.
Model MethodClaim Citation
F1 P R F1 P R
Qwen3.5
4BNativeRAG 33.5 30.548.936.0 38.1 44.4
LightRAG 30.0 26.0 48.3 26.4 23.9 40.1
HippoRAG-2 34.3 31.8 47.5 36.4 38.4 45.2
PIKE-RAG 34.9 39.0 38.4 44.0 54.6 44.2
Ours35.9 41.738.346.7 58.245.6
w/o schema 34.3 32.0 47.9 40.1 41.249.8
Qwen3.6
35BNativeRAG 37.0 36.1 49.2 46.1 52.7 49.4
LightRAG 36.3 33.850.940.1 39.2 50.1
HippoRAG-2 36.3 35.1 48.3 46.8 53.1 50.3
PIKE-RAG 37.2 42.6 39.7 43.7 56.9 43.8
Ours40.4 43.246.154.2 68.552.1
w/o schema 37.4 36.2 49.8 51.3 57.2 55.2
5.2 Retrieval Results
Table 3 presents the retrieval performance. Re-
fWalk establishes a state-of-the-art on RegOps-
Bench, achieving the most substantial gain in Full-
Cov@10. While these absolute metrics leave an
unsolved gap for future work, this steep difficulty
highlights the necessity of RegOps-Bench as a non-
saturated stress test rather than a limitation of our
method. Ablation studies confirm that removing
multi-view reranking, OKG expansion, or anchor
enrichment each degrades performance, demon-
strating that these components jointly address the
structural failures of dense retrieval. To address
potential concerns regarding extraction imperfec-
tions during OKG construction, we further validate
the robustness of our approach against graph-level
noise in Appendix A.8. On the HIPAA dataset, re-
trieval metrics generally saturate across most base-
lines. This suggests that while baseline retrieval
is highly effective on flat-structure benchmarks,
simple semantic matching is insufficient for the
complex cross-reference environments.
5.3 End-to-End RAG Results
Table 4 compares the overall end-to-end perfor-
mance on RegOps-Bench. While Claim F1 remains
relatively consistent across systems—since pow-
erful generators can infer similar claims once par-
tial evidence is retrieved—RefWalk achieves sub-
stantial gains in Citation F1 over Native RAG and
graph-based baselines. This improvement is struc-
turally enforced by our per-rule attribution schema.
We validate this through the w/o schema ablation,
which effectively isolates the generation bottleneck
by pairing RefWalk’s advanced retrieval with aTable 5:End-to-end RAG performance on HIPAA-
derived QA (cross-domain).
Model Method Claim F1 Citation F1
Qwen3.5-4BNativeRAG 62.3 72.8
LightRAG 47.4 49.8
HippoRAG-2 58.8 68.6
PIKE-RAG73.980.6
Ours 72.4 84.0
w/o schema 58.3 70.6
Qwen3.6-35BNativeRAG 61.8 73.7
LightRAG 50.0 48.1
HippoRAG-2 62.0 74.0
PIKE-RAG 71.9 82.2
Ours73.9 85.3
w/o schema 66.4 73.3
Table 6: Retrieval performance by query difficulty (L1–
L4) on the RegOps benchmark. We report R@10 and
FullCov@10 (%). Best per column is inbold, second-
best is underlined .
R@10 FullCov@10
MethodL1 L2 L3 L4 L1 L2 L3 L4
Dense 86.5 63.4 34.7 36.7 85.4 48.2 11.1 4.5
LightRAG 80.2 52.9 24.1 24.2 79.2 38.3 13.0 1.5
HippoRAG-2 77.1 49.6 22.7 22.2 75.0 32.1 11.1 1.5
Query Decomp 83.3 55.8 31.6 34.3 81.3 40.7 7.4 3.0
PIKE-RAG 75.0 64.6 37.6 42.2 75.0 49.4 13.0 8.9
Ours 95.8 70.8 42.6 45.8 95.8 56.8 20.4 11.9
Native RAG-style free-form prompt. When the
schema constraint is removed, Citation F1 drops
sharply. This result shows that without schema-
level structural binding, generative models treat
citations as post-hoc footers, failing to accurately
map claims to their corresponding rules even when
the correct evidence is retrieved.
5.4 Difficulty-Stratified Analysis
To understand exactly where existing systems fail,
we stratify performance by difficulty (L1–L4).
As shown in Table 6, all methods succeed at
single-anchor lookups (L1). However, at L3 (multi-
hop/cross-doc) and L4 (conditional multi-hop),
baseline retrieval significantly collapses. Lexical
or simple dense retrieval cannot navigate explicit,
multi-tiered delegations. This bottleneck directly
cascades into generation, as demonstrated in Table
7. While baseline models attempt to answer L3/L4
queries, their Citation F1 plummets because they
fail to bind claims to the correct cross-document
sources. In contrast, RefWalk demonstrates signif-
icant relative improvements in Citation F1 across
these complex tiers, effectively mitigating the se-
vere drop-offs caused by fragmented retrieval.
7

Table 7: End-to-end RAG performance by query dif-
ficulty (L1–L4) on the RegOps benchmark ( n=
48/81/54/67for L1/L2/L3/L4).
Claim F1 Citation F1
MethodL1 L2 L3 L4 L1 L2 L3 L4
Backbone: Qwen3.5-4B
NativeRAG 40.6 40.1 24.629.354.4 39.7 24.8 27.6
LightRAG 30.2 35.9 25.2 26.9 38.1 29.2 18.7 20.9
HippoRAG-2 42.3 39.1 26.1 29.357.8 40.1 22.7 27.5
PIKE-RAG45.443.2 22.1 27.5 67.0 55.4 25.0 28.8
Ours 43.1 46.0 27.825.268.4 57.8 29.0 31.9
Backbone: Qwen3.6-35B
NativeRAG 44.7 42.1 29.2 31.5 68.2 55.2 30.3 31.9
LightRAG 42.3 44.5 29.1 28.1 58.1 50.9 26.3 25.2
HippoRAG-2 48.5 41.8 28.5 27.2 75.0 53.7 29.4 32.1
PIKE-RAG 49.1 45.5 26.7 27.1 62.3 52.9 26.2 33.1
Ours49.3 48.5 30.6 32.0 79.0 66.7 34.2 37.2
Table 8:Mechanism Analysis of Multi-View Fusion.
RRM maximizes overall performance by acting as a
specialist selector.
VariantOverall Recall@10 by Difficulty
R@10 L1 L2L3L4
3-view RRM (Ours) 63.0 91.7 73.3 45.5 44.2
3-view wRRF (wide=2) 62.0 87.5 73.9 42.6 45.0
3-view RRF (uniform) 61.2 87.574.936.845.6
wide-only 61.3 89.6 72.2 42.9 42.5
narrow-only 61.0 90.6 72.8 36.3 45.4
5.5 Mechanism Analysis: RRM vs RRF
As shown in Table 8, RefWalk’s resilience at L3/L4
is driven by the RRM fusion. For complex cross-
document references, the qwide view carries the
structural signal. Standard sum-based aggregations
(e.g., uniform or weighted RRF) dilute this special-
ist signal by demanding consensus across all views,
which leads to a degradation in L3 performance.
RRM preserves the single strongest rank, allowing
the framework to maintain citation closure without
compromising the retrieved evidence set.
5.6 Mitigating Attribution Hallucination via
Schema Constraints
To validate whether per-rule schema alleviates attri-
bution hallucination (Table 9), we conduct a ceiling
analysis using an Oracle with Controlled Noise set-
ting (k= 10 ). While Native RAG’s scores are artifi-
cially inflated under a Pure Oracle setting (without
distractors), padding the context with hard nega-
tives from the 1-hop citation neighborhood exposes
a systemic eager-citing bias, causing a sharp de-
cline in Citation Precision. Conversely, RefWalk’s
conservative mapping strategy filters out these hard
negatives, securing a highly competitive CitationTable 9:Effect of Schema Constraint on Attribution
Reliability.We compare generation methods across
varying model capacities and retrieval settings. RefWalk
consistently pushes Precision and F1 higher by prevent-
ing the eager-citing behavior observed in Native RAG.
Model Retrieval MethodClaim Citation
P F1 P F1
Qwen3.6
35BTop-10Native 36.1 37.0 52.7 46.1
Ours43.2 40.4 68.5 54.2
Pure OracleNative 47.3 43.4100.0*82.2*
Ours49.7 44.099.2 75.6
Oracle
(k= 10)Native 40.7 40.1 82.169.9
Ours47.2 42.2 85.867.6
PIKE-RAG w/ schema 45.0 40.2 65.4 46.6
Gemini
3.1-ProTop-10Native 47.8 40.5 53.3 45.4
Ours53.0 40.6 65.9 54.1
*Artificially inflated due to the absence of distractors.
Precision. Applying our schema constraint to PIKE-
RAG also boosts its native Claim Precision and F1.
This highlights the transferability of our schema-
binding strategy in steering LLMs toward more pre-
cise operational claims. However, even with this
structural enforcement, PIKE-RAG’s Citation F1
remains substantially lower than RefWalk’s. Fur-
thermore, evaluating Gemini-3.1-Pro reveals that
even frontier models with advanced reasoning ca-
pabilities are highly susceptible to attribution hal-
lucination under Native RAG.
6 Conclusion
Assisting regulatory compliance with LLMs re-
quires a paradigm shift from broad answer gen-
eration to verifiable, structure-bound attribution. In
this work, we formalized this challenge through
RegOps-Bench, a benchmark designed to capture
the intricate, multi-tiered delegation networks in-
herent in real-world regulatory ecosystems. By
decoupling procedural complexity from surface-
level phrasing, our evaluations revealed that cur-
rent retrieval-augmented systems—despite their
proficiency on flat-structure tasks—struggle signifi-
cantly to navigate complex legal citations. To over-
come this, we introduced RefWalk, a framework
that unifies Operational Knowledge Graph traversal
with per-rule attribution. Our findings demonstrate
that preserving specialist signals via multi-view
RRM fusion effectively resolves cross-document
chains, while structural schema-binding curtails
the persistent attribution hallucinations prevalent
even in frontier LLMs. Ultimately, RefWalk suc-
cessfully mitigates the vulnerabilities of free-form
8

generation, providing a robust and generalizable
foundation for deploying fully traceable AI in high-
stakes regulatory domains.
7 Limitations
While RefWalk establishes a robust framework for
verifiable compliance QA, its design philosophy
fundamentally prioritizes safety over comprehen-
siveness, introducing an inherent precision-recall
trade-off in attribution. In high-stakes domains like
regulatory compliance, hallucinating a false cita-
tion is far more dangerous than missing a valid
one. Consequently, our per-rule attribution schema
enforces a highly conservative mapping strategy.
As demonstrated in our oracle experiments, this
strictness successfully pushes citation precision to
highly reliable levels under noisy conditions, but it
inevitably sacrifices recall compared to the eager-
citing behavior of Native RAG. Future work should
explore adaptive schema constraints that dynam-
ically balance recall without compromising strict
hallucination boundaries.
Beyond generation constraints, the framework’s
retrieval architecture relies on the deterministic
extraction of an Operational Knowledge Graph
(OKG). By leveraging the highly formulaic cita-
tion patterns of Korean regulatory documents, we
achieved high-precision, low-cost rule extraction
without the heavy LLM-based indexing overhead
seen in other graph-based RAGs. However, scaling
this purely deterministic graph-building process to
less structured jurisdictions, such as heavily case-
law-driven domains or entirely different languages,
remains a challenge. Adapting RefWalk to such
environments will likely require transitioning to hy-
brid (rule and LLM-assisted) extraction pipelines.
Finally, navigating the OKG accurately requires
preserving expert signals during complex multi-
hop retrieval, which RefWalk achieves via multi-
view cross-encoding and RRM fusion. While
highly effective for targeted candidate pools (e.g.,
N= 50 ), applying deep cross-attention indepen-
dently across three distinct semantic views scales
linearly with the pool size, which may eventually
encounter computational bottlenecks in massive-
scale deployments. Addressing these latency chal-
lenges—such as optimizing prompt topology for
cross-view KV cache sharing, or introducing dy-
namic view routing to conditionally bypass redun-
dant cross-encoder passes—remains an essential
direction for scaling RefWalk without sacrificingits rigorous matching precision.
References
Mohammed Ali, Abdelrahman Abdallah, and Adam
Jatowt. 2025. SustainableQA: A comprehensive
question answering dataset for corporate sustain-
ability and EU taxonomy reporting.arXiv preprint
arXiv:2508.03000.
Farid Ariai, Joel Mackenzie, and Gianluca Demartini.
2025. Natural language processing for the legal do-
main: A survey of tasks, datasets, models, and chal-
lenges.ACM Computing Surveys, 58:1–37.
Douglas W Arner, Jànos Barberis, and Ross P Buckley.
2018. Regtech: Building a better financial system. In
Handbook of blockchain, digital finance, and inclu-
sion, volume 1, pages 359–373.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe International Conference on Learning Repre-
sentations (ICLR).
Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aha-
roni, Daniel Andor, Livio Baldini Soares, Massimil-
iano Ciaramita, Jacob Eisenstein, Kuzman Ganchev,
Jonathan Herzig, and 1 others. 2022. Attributed
question answering: Evaluation and modeling for
attributed large language models.arXiv preprint
arXiv:2212.08037.
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malaka-
siotis, Nikolaos Aletras, and Ion Androutsopoulos.
2020. LEGAL-BERT: The muppets straight out
of law school. InFindings of the Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 2898–2904.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval (SIGIR), pages 758–759.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization.arXiv
preprint arXiv:2404.16130.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony
Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent
Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
Kelvin Guu. 2023a. RARR: Researching and re-
vising what language models say, using language
models. InProceedings of the Annual Meeting of
the Association for Computational Linguistics (ACL),
pages 16477–16508.
9

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023b. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the Annual Meet-
ing of the Association for Computational Linguistics
(ACL), pages 1762–1777.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023c. Enabling large language models to generate
text with citations. InProceedings of the Conference
on Empirical Methods in Natural Language Process-
ing (EMNLP), pages 6465–6488.
Google DeepMind. 2025. Gemini 3 flash - model card.
Google DeepMind. 2026. Gemini 3.1 pro model card.
Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré,
Adam Chilton, Alex Chohlas-Wood, Austin Peters,
Brandon Waldon, Daniel Rockmore, Diego Zam-
brano, and 1 others. 2023. Legalbench: A collabora-
tively built benchmark for measuring legal reasoning
in large language models.Advances in Neural In-
formation Processing Systems (NeurIPS), 36:44123–
44279.
Keerat Guliani, Deepkamal Gill, David Landsman,
Nima Eshraghi, Krishna Kumar, and Lovedeep
Gondara. 2026. De jure: Iterative llm self-refinement
for structured extraction of regulatory rules.arXiv
preprint arXiv:2604.02276.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Conference
on Empirical Methods in Natural Language Process-
ing (EMNLP), pages 10746–10761.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025a. From RAG to mem-
ory: Non-parametric continual learning for large lan-
guage models. InThe International Conference on
Machine Learning (ICML).
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025b. From RAG to mem-
ory: Non-parametric continual learning for large lan-
guage models.International Conference on Machine
Learning. ArXiv:2502.14802.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics (COLING),
pages 6609–6625.
Abe Bohan Hou, Orion Weller, Guanghui Qin, Eugene
Yang, Dawn Lawrie, Nils Holzenberger, Andrew
Blair-Stanek, and Benjamin Van Durme. 2025. Clerc:
A dataset for us legal case retrieval and retrieval-
augmented analysis generation. InFindings of the
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics
(NAACL), pages 7898–7913.Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 7969–7992.
Daniel Martin Katz, Corinna Coupette, Janis Beckedorf,
and Dirk Hartung. 2020. Complex societies and the
growth of the law.Scientific reports, 10:18737.
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao
Fu, Kyle Richardson, Peter Clark, and Ashish Sab-
harwal. 2023. Decomposed prompting: A modular
approach for solving complex tasks. InThe Inter-
national Conference on Learning Representations
(ICLR).
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of the ACM
SIGOPS Symposium on Operating Systems Princi-
ples (SOSP).
Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe
Zhang, and Yiqun Liu. 2024. Lexeval: A compre-
hensive chinese legal benchmark for evaluating large
language models.Advances in Neural Information
Processing Systems (NeurIPS), 37:25061–25094.
Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023.
Evaluating verifiability in generative search engines.
InFindings of the Conference on Empirical Methods
in Natural Language Processing (EMNLP), pages
7001–7025.
Antoine Louis and Gerasimos Spanakis. 2022. A statu-
tory article retrieval dataset in French. InProceed-
ings of the Annual Meeting of the Association for
Computational Linguistics (ACL), pages 6789–6803.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li,
Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo.
2025. Think-on-graph 2.0: Deep and faithful large
language model reasoning with knowledge-guided
retrieval augmented generation. InThe International
Conference on Learning Representations (ICLR).
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision in
long form text generation. InProceedings of the Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 12076–12100.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2025. Graph retrieval-augmented generation:
A survey.ACM Transactions on Information Systems
(TOIS), 44:1–52.
Roxana Petcu, Kenton Murray, Daniel Khashabi, Evan-
gelos Kanoulas, Maarten de Rijke, Dawn Lawrie, and
10

Kevin Duh. 2026. Query decomposition for RAG:
Balancing exploration-exploitation. InProceedings
of the Conference of the European Chapter of the
Association for Computational Linguistics (EACL),
pages 6857–6871.
Qwen Team. 2026a. Qwen3.5: Accelerating productiv-
ity with native multimodal agents.
Qwen Team. 2026b. Qwen3.6-35B-A3B: Agentic cod-
ing power, now open to all.
Zackary Rackauckas. 2024. RAG-Fusion: a new take
on retrieval-augmented generation.arXiv preprint
arXiv:2402.03367.
Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora
Aroyo, Michael Collins, Dipanjan Das, Slav Petrov,
Gaurav Singh Tomar, Iulia Turc, and David Reitter.
2023. Measuring attribution in natural language gen-
eration models.Computational Linguistics, 49:777–
840.
John B Ruhl and Daniel Martin Katz. 2015. Measuring,
monitoring, and managing legal complexity.Iowa L.
Rev., 101:191.
Yash Saxena, Raviteja Bommireddy, Ankur Padia, and
Manas Gaur. 2025. Generation-time vs. post-hoc
citation: A holistic evaluation of LLM attribution. In
Workshop on Neural Information Processing Systems
(NeurIPS Workshop).
Amin Sleimi, Nicolas Sannier, Mehrdad Sabetzadeh,
Lionel Briand, and John Dann. 2018. Automated
extraction of semantic legal metadata using natural
language processing. InIEEE International Require-
ments Engineering Conference (RE), pages 124–135.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics (TACL), 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of the
Annual Meeting of the Association for Computational
Linguistics (ACL), pages 10014–10037.
Jinyu Wang, Jingjing Fu, Rui Wang, Lei Song, and
Jiang Bian. 2025. PIKE-RAG: specialized knowl-
edge and rationale augmented generation.arXiv
preprint arXiv:2501.11551.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. InProceedings of the Conference on Em-
pirical Methods in Natural Language Processing
(EMNLP), pages 9414–9423.
Xinyu Yang, Chenlong Deng, Tongyu Wen, Binyu Xie,
and Zhicheng Dou. 2026. LawThinker: A deep re-
search legal agent in dynamic environments.arXiv
preprint arXiv:2602.12056.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for di-
verse, explainable multi-hop question answering. In
Proceedings of the Conference on Empirical Methods
in Natural Language Processing (EMNLP), pages
2369–2380.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, and 1 others.
2023. Judging LLM-as-a-judge with mt-bench and
chatbot arena.Advances in neural information pro-
cessing systems (NeurIPS), 36:46595–46623.
A Appendix
A.1 Implementation Details
All open-source models were served using
vLLM (Kwon et al., 2023) on RTX A6000 (48
GB) GPUs in bfloat16 precision with a maxi-
mum sequence length of 32,768 tokens. We em-
ployedQwen3.6-35B-A3B-FP8 andQwen3.5-4B
as the 35B and 4B backbones, respectively, for
the cross-scale comparison. Dense retrieval was
conducted using Qwen3-Embedding-0.6B , while
the cross-encoder utilized Qwen3-Reranker-0.6B
with the official yes/no template. For all text gen-
eration, the temperature, top- p, and top- kparame-
ters were set to 0.1, 0.95, and 20, respectively. We
disabled the thinking mode and fixed the random
seed to 42 across all runs. For the retrieval stage of
RefWalk, we utilized a 50-candidate rerank pool
over three views fused via Reciprocal Rank MAX
(k=60). This was further augmented by a 1-hop
OKG expansion over REFERENCES ,DELEGATES_TO ,
andSPECIFIES edge types, applying a decay fac-
tor of δ= 0.7 and up to M= 10 seed nodes.
Here, δdown-weights 1-hop neighbors as indirect
evidence; co-discovered neighbors take the max de-
cayed score, not the sum. In addition, an authority-
aware manual-node decay of µ= 0.7 is multiplied
into the fused rerank score of any candidate sourced
from a manual ( 매뉴얼 ) rather than a statute ( 법
령/시행령 /시행규칙 ). Because it is applied after
multi-view fusion, it acts as a soft tie-break that
favors higher-authority sources while still allowing
a strongly-matched manual passage to outrank a
weakly-matched statute. All hyperparameters were
11

kept fixed across both the RegOps and HIPAA
datasets. End-to-end generation is evaluated using
Claim F1 and Citation F1. Claim F1 is computed
via LLM-as-judge (Zheng et al., 2023) bipartite
matching between predicted and reference atomic
claims, where exact semantic matches receive full
credit and partial matches—those preserving the
core proposition but missing a condition, exception,
or numeric scope—receive half credit. Citation F1
measures the overlap between cited and ground-
truth references at the article ( 조) level, rolling up
sub-article citations ( 항/호) to their parent article
so that an answer is credited whenever it points to
the correct legal provision regardless of granularity.
A.2 Knowledge Source and OKG Properties
Statistics.The underlying corpus of RegOps-
Bench comprises 718 procedural articles contain-
ing roughly 478K subword tokens. As detailed in
Table 10, the article length distribution is heavily
right-skewed. However, because only a small frac-
tion of the articles exceed the standard 8,192-token
embedding context window, article-level indexing
remains lossless for the overwhelming majority
of the text. Furthermore, the corpus includes a
non-trivial amount of structured data, with approxi-
mately 10% of articles embedding tabular content
and 3% acting as appendix forms (별표).
OKG Composition.The Operational Knowl-
edge Graph (OKG) instantiated from this tex-
tual corpus consists of 2,572 nodes (primar-
ily operational article/provision units) and 3,942
typed edges. As Table 11 illustrates, the graph’s
edge distribution is dominated by the structural
containment hierarchy ( Part_Of ) and citation-
bearing relations. Notably, the upward real-
ization (Specifies ) and downward delegation
(Delegates_To ) edges are perfectly balanced. This
symmetry reflects the paired multi-tier legal struc-
ture of the corpus, forming the exact structural
signature that retrieval systems must traverse to
resolve complex regulatory queries.
Reference Topology of the OKG.Beyond local
edge distributions, we analyze the global connec-
tivity of the OKG to substantiate that the in-domain
documents form a largely self-contained delegation
network. Considering only the citation-bearing rela-
tions (excluding the within-document containment
hierarchy), the graph fragments into 1,878 compo-
nents but is dominated by a single referential back-
bone of 654 nodes spanning all twelve documents;the remaining components have a median size of
one, indicating that most articles delegate into the
shared backbone rather than isolated islands. When
the containment hierarchy is added, the graph con-
solidates into 148 components. This includes a gi-
ant component of 2,078 nodes that again spans all
twelve documents, leaving only 81 truly orphaned
nodes. Of the 148 components, only nine bridge
two or more documents, mostly representing aux-
iliary tax pairings. Because the giant component
co-mingles the containment hierarchy with citation
edges, we further apply greedy-modularity commu-
nity detection to it, recovering 48 distinct commu-
nities. The largest communities align tightly with
intuitive procedural themes: a statutory-procedure
cluster anchored on the Innovation Act (314 nodes
across 9 documents), a cost-use cluster anchored on
the Cost-Use Standards (300 nodes, 6 documents),
and a self-contained tax cluster anchored on the
V AT Act (264 nodes, 1 document). This empiri-
cal decomposition confirms that the benchmark’s
multi-hop reference closures are not artifacts of a
single dense hub but rather span semantically co-
herent, multi-document procedural neighborhoods.
Table 10: Per-article length distribution of the knowl-
edge source. Tokens are computed with the Qwen3-
Embedding tokenizer. Sub-items count the paragraph/-
subparagraph (항/호) units nested in each article.
Metric Mean p50 p75 p95 Max
Characters / article 993.1 422 847 2567 25226
Tokens / article 665.7 290 576 1753 16801
Sub-items / article 2.7 2 4 8 17
Table 11: Typed-edge composition of the OKG. The
balanced DELEGATES_TO/SPECIFIEScounts reflect the
paired delegation/realization structure of the corpus.
Edge type Count Ratio (%)
Part_Of(hierarchy) 1730 43.9
References(citation) 881 22.3
Specifies(upward) 557 14.1
Delegates_To(downward) 557 14.1
Requires_Form142 3.6
Defines75 1.9
Total 3942 100.0
A.3 Benchmark Details
RegOps-Bench comprises 250 question–answer
pairs, combining 194 instances generated via axis-
decoupled augmentation with 56 verbatim queries
12

Table 12: Composition of the 250 RegOps-Bench ques-
tions. All percentages are over the full set; under “Do-
main anchor” the 56 human seeds (22.4%) carry no
synthesized anchor and are omitted from the listing.
Axis Categoryn%
DifficultyL1 48 19.2
L2 81 32.4
L3 54 21.6
L4 67 26.8
Question typeIITP seed (human) 56 22.4
Multi-facet 53 21.2
Single-clause 50 20.0
General-principle 43 17.2
Exception-paired 26 10.4
Institution-specific 22 8.8
SourceAugmented (LLM) 194 77.6
IITP board (human) 56 22.4
Domain anchorCost-Use Standards (default) 122 48.8
Innovation-Act procedure 26 10.4
Institutional-IT 22 8.8
Facility/equipment 15 6.0
Tax (V AT) 9 3.6
retained from the IITP practitioner board. As out-
lined in Table 12, the benchmark spans diverse
substantive intents—ranging from single-clause
lookups to multi-facet institutional inquiries. The
Cost-Use Standards serves as the primary domain
anchor for nearly half of the generated questions,
supplemented by procedural chains from the Inno-
vation Act, institutional IT regulations, and auxil-
iary tax documents.
We balance the dataset toward higher complexity,
with the advanced L3 and L4 tiers jointly account-
ing for nearly half of the benchmark. This distri-
bution ensures the evaluation stresses multi-hop
and conditional reasoning rather than simple fact
retrieval. Table 13 validates this design by tracing
how structural complexity materializes across the
difficulty tiers. The profile exhibits a clear, mono-
tonic progression: as difficulty increases from L1
to L4, conditional reasoning transitions from en-
tirely absent to ubiquitous, and cross-document
references become highly prevalent.
Correspondingly, the mean size of the required
reference set and the citation-graph depth grow in
lock-step with the difficulty levels. It is important
to note that difficulty in RegOps-Bench is an in-
trinsic property of the reference closure rather than
mere surface phrasing. The overall reference-count
distribution is long-tailed: while roughly 45% of
the questions resolve at a single hop, over 16%
require navigating four or more hops along the ci-
tation graph, rigorously testing a system’s capacity
for deep reference traversal.Table 13: Structural profile of RegOps-Bench by diffi-
culty. “Cond.” = conditional, “X-doc” = cross-document
reference, “Chain” = reference set forms a proce-
dure chain; “Refs” and “Hops” are the mean realized
reference-set size and mean citation-graph depth.
Cond. X-doc Chain Refs Hops
Level % % % avg avg
L1 0.0 0.0 0.0 1.31 1.00
L2 87.7 0.0 33.3 2.19 1.43
L3 9.3 48.1 85.2 5.52 2.96
L4 100.0 58.2 94.0 5.90 2.99
Table 14: Five most-cited articles in the ground-truth
reference sets and their roles under the deterministic
reference rules.
Article (anchor) Role Cites
Cost-Use Standards Art. 73 Pre-approval (R3) 70
Cost-Use Standards Art. 28 Joint-R&D funds 40
Cost-Use Standards Art. 13 Fund usage 33
Enforcement Decree Art. 26 Settlement (R4) 33
Cost-Use Standards Art. 83 Disqualification (R4) 32
A.4 Ground-Truth Reference-Set Analysis
Across the 250 questions, the ground-truth ref-
erence sets contain 933 citations with multiplic-
ity (314 unique articles), averaging 3.73 refer-
ences drawn from 1.36 documents per question.
By annotation granularity, 71.6% of citations are
article-level (조), 27.0% are sub-clause ( 항/호), and
1.4% are appendix forms ( 별표 ), scoring collapses
sub-clause citations to their governing article, the
atomic unit of authority. Reference sets are well-
grounded in the indexed graph: 97.1% of citations
resolve to an OKG node, with only 2.9% point-
ing outside the constructed graph. The reference
distribution is sharply concentrated on the oper-
ational core of the corpus (Table 14): the single
most-cited article is the Cost-Use Standards pre-
approval clause (Art. 73, 70 citations), followed by
the international-joint-R&D and fund-management
standards, and by the two sanction-side authorities
(Enforcement Decree Art. 26 and Cost-Use Stan-
dards Art. 83). These dominant anchors correspond
directly to the deterministic expert rules R3 (pre-
approval) and R4 (sanction exhaustion).
A.5 Validating the Axis-Decoupling Principle
The augmentation is governed by the axis-
decoupling principle: substantive question type
and structural difficulty are sampled independently,
so a single question type can realize any diffi-
culty level depending only on the anchor and
13

Table 15: Question-type ×difficulty cross-tabulation, il-
lustrating that the same substantive intent can be realized
across multiple difficulty tiers under axis-decoupled aug-
mentation.
Question type L1 L2 L3 L4 Total
Single-clause 35 13 2 0 50
Exception-paired 0 26 0 0 26
Institution-specific 3 16 0 3 22
General-principle 0 0 20 23 43
Multi-facet 0 0 23 30 53
IITP seed (human) 10 26 9 11 56
Total 48 81 54 67 250
Table 16: Synthetic (LLM-generated) vs. human (IITP
QA board) splits of RegOps-Bench.
Source L1 L2 L3 L4 All
(a) Query distribution
# Queries LLM-Generated 38 55 45 56 194
IITP-Board 10 26 9 11 56
Share (%) LLM-Generated 19.6 28.4 23.2 28.9 100.0
IITP-Board 17.9 46.4 16.1 19.6 100.0
(b) Corpus characteristics by difficulty
Mean hop count LLM-Generated 1.00 1.62 3.29 3.04 2.29
IITP-Board 1.00 1.04 1.33 2.73 1.41
Cross-document (%) LLM-Generated 0.0 0.0 55.6 58.9 29.9
IITP-Board 0.0 0.0 11.1 54.5 12.5
Mean #GT refs LLM-Generated 1.29 2.42 6.04 6.34 4.17
IITP-Board 1.40 1.69 2.89 3.64 2.21
(c) RefWalk-35B Claim-F1
Claim-F1 LLM-Generated 52.6 57.2 30.5 30.7 42.5
IITP-Board 36.7 30.1 30.8 39.0 33.1
(d) RefWalk-35B Citation-F1
Citation-F1 LLM-Generated 87.3 71.6 30.7 34.1 54.4
IITP-Board 47.3 56.3 51.9 53.2 53.4
injected facets. Table 15 reports the empirical
type×difficulty cross-tabulation, which confirms
the intended spread: single-clause questions con-
centrate at L1, exception-paired questions at L2,
while the multi-facet and general-principle types
populate the L3/L4 tiers where deep reference clo-
sures are required. As a quantitative control, 69.4%
of augmented items realize exactly the targeted
difficulty level after generation; the difficulty re-
validation step discards the divergent remainder so
that the released set is internally consistent. The
facet injection that drives this spread draws on six
facet dimensions (transaction, actor-role, recipient-
role, geography, lifecycle, and domain) and 14 four-
institution parallel-group topics, providing the com-
binatorial breadth needed to decouple intent from
structural depth.
In Table 16, we report query distribution, cor-
pus characteristics, and end-to-end RefWalk perfor-
mance per difficulty. Synthetic queries are harder
and deeper than the human board, yet RefWalk per-
formance aligns closely with the hard tier (L3/L4)Table 17: Integrated mean query latency (at 35B back-
bone) and index-build cost on RegOps-Bench. All la-
tency metrics are reported in mean wall-clock seconds
per query. Bold text indicates the best performance in
each column.
Query 35B Latency (s) Index
Method Retrieval RAG LLM calls
NativeRAG0.02 7.46–
PIKE-RAG 0.33 14.64 718
LightRAG 1.46 8.48 2,329
HippoRAG-2 2.36 9.26 1,436
Query Decomp 4.52 – –
RefWalk (Ours)4.2 9.49 –
and on citation grounding overall, indicating the
augmentation is faithful and non-trivial.
A.6 Computational Cost
In Table 17, metrics were evaluated on the Re-
gOps dataset ( n= 250 ) under identical hardware
configurations using a 35B backbone. At retrieval
time, RefWalk incurs an overhead of +4.18 s com-
pared to the fastest baseline (NativeRAG). This
represents the marginal cost of executing the cross-
encoder across three distinct views ( qnarrow ,qmid,
qwide) alongside a 1-hop OKG expansion. Notably,
RefWalk is ∼1.5×faster than PIKE-RAG in RAG
latency (9.49 s vs. 14.64 s). On the indexing side,
RefWalk avoids generative LLM calls during index
construction, whereas other graph-based methods
incur substantial pre-computation overhead.
A.7 Sensitivity Analysis of the RRM
Smoothing Constantk
Table 18: Sensitivity analysis of RefWalk with respect to
the RRM smoothing constant kon the RegOps dataset
(n= 250, retrieval-only).
Overall Recall@10 by Difficulty
kR@10 FC@10 L1 L2 L3 L4
10 62.79 42.80 91.67 72.02 45.52 44.88
30 62.61 42.80 91.67 72.02 45.52 44.21
6063.01 43.20 91.67 73.25 45.52 44.21
100 62.61 42.80 91.67 72.02 45.52 44.21
200 62.69 42.80 91.67 72.02 45.52 44.51
As detailed in Table 18, the empirical variance
of the overall Recall@10 metric fluctuates within
a marginal range of just 0.40 percentage points
across a multi-order sweep of k∈[10,200] . This
stable behavior aligns with the structural mechan-
ics of Reciprocal Rank Max (RRM). Consequently,
14

the smoothing constant ksolely scales the numeri-
cal resolution of tie-breaks between identical rank
positions within the same view, rather than altering
the global relative weights across different views.
A.8 Robustness to Operational Knowledge
Graph Construction Noise
Table 19: Robustness of RefWalk to OKG construction
noise on the RegOps dataset (n= 250, retrieval-only).
ConditionOverall Recall@10 by Difficulty
R@10 FC@10 L1 L2 L3 L4
Clean 63.01 43.20 91.67 73.25 45.52 44.21
Drop10% 62.63 42.80 91.67 72.02 45.21 44.51
20% 62.44 42.80 91.67 72.02 45.21 43.81
Rewire10% 62.65 42.80 91.67 72.02 45.21 44.58
20% 62.44 42.80 91.67 72.02 45.21 43.81
As detailed in Table 19, RefWalk exhibits high
resilience to OKG construction noise, with over-
all Recall@10 degrading by at most 0.57 percent-
age points even under an aggressive 20% perturba-
tion rate. The near-identical performance drops ob-
served between edge dropping and rewiring ( ≤0.02
percentage points variance) demonstrate that the
cross-encoder reranker effectively filters out in-
jected false-positive connections, leaving missed
true-positives as the sole operative failure mode.
Furthermore, the multi-hop segment (L3) degrades
by only 0.31 percentage points under 20% noise.
This robust behavior underscores the structural role
of the OKG as a candidate pool augmentation chan-
nel rather than a direct ranking signal; a corrupted
edge only induces a recall failure if the correspond-
ing gold document is simultaneously absent from
the dense retrieval seed.
A.9 Prompt Template for RefWalk
Cross-Encoder Reranking.For the reranking
stage, each query-document pair (q, d) is evaluated
using a cross-encoder model to compute a determin-
istic relevance score. The model is optimized via a
binary classification objective, prompting it to gen-
erate a strict yesornotokens indicating whether
the document satisfies the query. This prompt struc-
ture is uniformly applied across all three retrieval
views within the RRM-fusion pipeline and is con-
sistently exercised throughout all experiments in
this study.
The unified ChatML prompt template, which em-
beds the structural instruction, query placeholder,
and target document context, is illustrated in Fig-
ure 2.Prompt for Cross-Encoder Reranker
<|im_start|>system
Judge whether the Document meets the
requirements based on the Query and the
Instruct provided. Note that the answer can
only be “yes” or “no”.<|im_end|>
<|im_start|>user
<Instruct>: Given a user question, retrieve
legal clauses or regulations that answer the
question.
<Query>: {query}
<Document>: {doc}<|im_end|>
<|im_start|>assistant
Figure 2: The unified ChatML prompt structure used for
binary relevance scoring in the cross-encoder reranking
stage.
Topic Anchoring τ(q).The detailed prompt tem-
plate for this task is illustrated in Figure 3. The
structured topic + facet extraction prompt used
by the frozen LLM call τ(q) produces the Topic
Anchortopic ,actor ,temporal ,magnitude ,
situational for every RegOps and HIPAA query.
The{domain} placeholder is “Korean R&D fund-
ing regulations” for RegOps and “U.S. health-
care privacy regulations (HIPAA)” for HIPAA;
{language} is “Korean” or “English” respectively.
The output is constrained to the QueryAnalysis
JSON schema.
Table 20: Multi-View Query Format structure.
View Format
qnarrow the original question text.
qmid[TOPIC]⟨topic⟩[Q]⟨question⟩
Conditions: [ACTOR] ⟨a⟩, [TEMPO-
RAL]⟨t⟩, [MAGNITUDE] ⟨m⟩, [SITUA-
TIONAL]⟨s⟩
qwide[TOPIC]⟨topic⟩
Conditions: [ACTOR] ⟨a⟩, [TEMPO-
RAL]⟨t⟩, [MAGNITUDE] ⟨m⟩, [SITUA-
TIONAL]⟨s⟩
Multi-View Tagged-Query Rendering.Given
the topic anchor produced by τ(q), the three re-
trieval views qnarrow ,qmid, and qwideare determin-
istically generated as shown in Table. 20. For in-
stance, the resulting rendered qmidview is struc-
tured as follows:
15

[TOPIC]장비구매사전승인
[Q] 3천만원이상GPU를구매할때사전승인이
필요한가요?
Conditions: [ACTOR] all, [TEMPORAL]구매전,
[MAGNITUDE] 3천만원이상,
[SITUATIONAL] unspecified
Schema-Constrained Generation.This section
details the system and user prompt templates
employed by the RefWalk framework to realize
the schema-constrained attribution mechanism de-
scribed in §4.2. Rather than generating free-form
prose followed by loose citation footers, the gen-
erative model is strictly constrained via a strict
JSON schema. The keys of the emitted JSON ob-
ject must correspond exactly to the node_id ele-
ments present within the retrieved context, and the
values are restricted to arrays of granular claims
bounded by paragraph-level annotations (e.g., [제O
항]).
This architectural constraint structurally pre-
vents post-hoc rationalization (i.e., “writing first,
citing later”). This setup corresponds directly to the
w/ schemaRefWalk variant evaluated in Table 5
of the main paper. The formalized Korean variant
of the system prompt and user template for the
RegOps domain are presented in Figure 4.
NativeRAG Free-Form Generation Baseline.
To isolate and evaluate the baseline generation qual-
ity under unconstrained conditions, we implement
a conventional free-form generation model paired
with an appended citation-footer instruction. This
setup serves as the generation standard for multi-
ple baselines, including NativeRAG, LightRAG,
HippoRAG-2, and PIKE-RAG, as well as thew/o
schemaablation configuration of RefWalk. The
prompt templates for this baseline are detailed in
Figure 7.
A.10 Qualitative Examples
To illustrate the operational behavior of RefWalk
(Qwen3.6-35B ) across different reasoning depths,
we present three qualitative case studies represent-
ing schema-bound exception surfacing (L2), ex-
haustive parallel-institution closure (L3), and cross-
document delegation-chain traversal (L4).
In each case, the model is required to emit a
structured per-rule attribution object containing
governing article identifiers paired with their spe-
cific claims, followed by a synthesized final answer.
Verbatim Korean originals are provided alongside
polished English translations marked as[EN].Analysis of Example 1.As illustrated in Figure 8,
the ground-truth rule closure requires targeting ex-
actly{Art. 28, Art. 73} . RefWalk achieves perfect
citation precision and high claim alignment (cita-
tion F1 = 1.00 , claim F1 = 0.89 ). Crucially, rather
than treating the exchange-rate clause as an unstruc-
tured text block, the attribution schema structurally
binds the exception directly to its governing statu-
tory articles, ensuring that the final counterfactual
synthesis is grounded in traceable textual support.
Analysis of Example 2.As demonstrated in Fig-
ure 9, this instance evaluates a multi-institutional
parallel rule structure. To construct a legally sound
answer, the model must map the entire parallel
web without missing the critical distinction be-
tween institution-level and project-level caps. Re-
fWalk successfully establishes full reference clo-
sure by identifying all four parallel anchors (cita-
tionF1= 1.00 ). Rather than compressing distinct
corporate entities into a single generic rule, it pre-
serves the formalmutatis mutandisrelationships as
discrete, attributable statutory claims.
Analysis of Example 3.As shown in Figure 10,
the legal reasoning path spans two distinct levels
of authority, successfully tracing a hierarchical del-
egation link (DELEGATES_TO/SPECIFIES) from
Enforcement Decree Art. 24 down to the interna-
tional joint R&D provisions in Cost-Use Standards
Art. 28. RefWalk successfully navigates this cross-
document dependency chain without introducing
hallucinated citations.
However, consistent with the precision-first be-
havior observed in our empirical evaluation, the
model selectively recovers the primary structural
anchors of the chain while missing surround-
ing contextual siblings (e.g., Cost-Use Standards
Art. 12, 13, and 27). This behavior accounts for
its lower article-level citation recall ( 0.40) on this
specific problem instance, directly exposing the un-
resolved recall headroom that REGOPS-BENCHis
tailored to isolate.
16

Prompt for Topic Anchoring
Extract a procedural topic and structured conditions from a given question for {domain}.
## Topic
- Output a phrase (15 under syllables)
- Express a generalized procedural category that matches chapter/article granularity in the
regulation corpus
- Include the cost category, item type, or document type implicit in the question
- Exclude specific numbers, amounts, or actor identities (these belong in conditions)
## Conditions
For each of the four dimensions below, extract the value mentioned in the question. If a
dimension is not mentioned, use "unspecified".
- Actor: the top entities whose institutional or organizational type determine which
rules apply (e.g., institution type, role, position)
- Magnitude: a quantitative threshold or scale that triggers different rules (e.g., monetary
amount, count, percentage; keep specific numbers)
- Temporal: a time point, period, or sequence relative to the action (e.g., before/after action,
deadline, period, frequency, duration)
- Situational: any contextual condition not captured by the other dimensions that affects rule
applicability (e.g., funding source, collaboration arrangement, employment status, planning
status, organizational relation, document type)
## Output format
Use JSON format. Topic and condition values in {language}. The parsed output should allow
reconstruction of the original question’s meaning.
## Examples
### Example 1
Question: Can a covered entity share a patient’s medical records with an external researcher
who is not affiliated with the entity?
{
"topic": "Disclosure of medical records to external researcher",
"actor": "Covered entity",
"magnitude": "unspecified",
"temporal": "unspecified",
"situational": "External researcher (not affiliated)"
}
### Example 2
Question: Can I pay the labor cost in cash to a new recruiter at a for-profit organization?
{
"topic": "Payment of cash labor costs",
"actor": "For-profit organization",
"magnitude": "unspecified",
"temporal": "unspecified",
"situational": "new recruit"
}
### Example 3
Question: 3천만원이상GPU를구매할때사전승인이필요한가요?
{
"topic": "장비구매사전승인",
"actor": "unspecified",
"magnitude": "3천만원이상",
"temporal": "구매전",
"situational": "unspecified"
}
Figure 3: Generalized prompt structure for procedural topic extraction and anchoring.
17

RefWalk System Prompt Template (RegOps-Bench, Korean)
당신은국가연구개발사업및관련법령/규정해석을지원하는전문가입니다.사용자가제공하는문맥
(Context),질문(Question),그리고질문의주요조건(주체,시점,규모,상황)을철저히분석하여정확한
답변을제공해야합니다.
# Instruction
1.문맥의존성:모든답변 및근거는반드시제공된’Context’내에서만추출해야합니다.제공된문서에
없는내용을 임의로지어내거나외부지식을개입하지마세요.
2.참조단위통일(조단위): JSON의Key값은Context에명시된문서의node_id (예: “국가연구개발사업_
연구개발비_사용기준_제22조”)형식과동일하게작성해야합니다.절대임의로Key의 이름을지어내거나
변형하지마세요.
3.근거(Claim)추출및세부단위표기:각참조 조항에서질문에답변하기위해 필요한구체적인사실이나
규정내용을추출하여배열(Array)형태로나열하세요.이때실제근거가되는세부단위(항,호)가
존재한다면내용앞에대괄호(예: [제O항])로표기하세요.
4.예외및단서조항 필수반영:제공된조건(주체,시점,규모,상황)의값이’all’(특정되지않음)
이거나포괄적인경우,존재하는예외조건이나단서조항(예: “다만,∼”, “∼의경우 예외로한다”,
“중앙행정기관의 장이 인정하는경우”등)을반드시탐색하여근거(Claim)에포함시켜야합니다.
5.최종답변(Answer)작성:
-추출한근거(Claim)와 예외조항을종합하여질문에부합하는명확한답변을 작성하세요.
-예외조건이 있는경우,답변내에“다만, [예외조건]의경우[예외결과]가 가능합니다/제한됩니다.”
의형태로명시적으로고지하세요.
-답변내에 어떤법령의몇조몇항에따른것인지명확히언급하여신뢰성과전문성을높이세요.
6.출력형식:반드시아래의JSON format으로만응답해야하며, JSON외의다른인사말이나부연설명은
절대포함하지마세요.
# Output format (JSON)
{
“OOO_제O조”: [
“[제1항]해당조항에서도출한원칙적인근거원문및내용”,
“[제2항]대상별로다른규정이적용되는경우의또다른원문내용”,
“[예외]해당조항에존재하는예외및단서조항내용”
],
“answer”: “원칙적인규정에대한설명.다만, [예외조건]에해당하는경우[예외사항]이적용됩니다.”
}
RefWalk User Template (RegOps-Bench, Korean)
## Context:
{ctx}
The below question is about {topic}. The conditions for the topic are as follows:
-주체: {actor}
-시점: {temporal}
-규모: {magnitude}
-상황: {situational}
## Question: {q}
Figure 4: The separated System Prompt and User Template structures used by RefWalk (RegOps-Bench).
18

RefWalk System Prompt Template (RegOps, English)
You are an expert who supports the interpretation of national research and development (R&D)
projects and related statutes/regulations. You must thoroughly analyze the context (Context),
question (Question), and the key conditions of the question (subject, timing, scale, situation)
provided by the user to deliver an accurate answer.
# Instruction
1. Context Dependency: All answers and grounds must be extracted strictly from within the provided
’Context’. Do not fabricate any content not present in the provided document or introduce external
knowledge.
2. Uniformity of Reference Unit (Article level): The key values of the JSON must be written in
the exact same format as the node_id specified in the Context (e.g., “국가연구개발사업_연구개발비_
사용기준_제22조”). Never arbitrarily invent or modify the names of the keys.
3. Ground (Claim) Extraction and Detailed Unit Notation: Extract the specific facts or regulatory
content required to answer the question from each referenced clause, and list them in an array
format. If a detailed unit (Paragraph, Subparagraph) that serves as the actual ground exists,
indicate it with square brackets (e.g., [제O항]) before the content.
4. Mandatory Reflection of Exceptions and Provisos: If the value of the provided conditions
(subject, timing, scale, situation) is ’all’ (unspecified) or comprehensive, you must search
for existing exceptional conditions or provisos (e.g., “다만,∼” [provided that,∼], “∼의경우
예외로한다” [except in the case of∼], “중앙행정기관의 장이 인정하는경우” [cases recognized by
the head of the central administrative agency], etc.) and include them in the grounds (Claim).
5. Drafting the Final Answer (Answer):
- Synthesize the extracted grounds (Claim) and exceptions to write a clear answer that aligns
with the question.
- If there is an exceptional condition, explicitly notify it in the answer in the form of
“다만, [예외조건]의경우[예외결과]가 가능합니다/제한됩니다.” [Provided that, in the case of
[Exceptional Condition], [Exceptional Outcome] is possible/restricted.]
- Clearly mention which Article and Paragraph of which statute the answer is based on to
enhance credibility and expertise.
6. Output Format: You must respond strictly in the JSON format below, and never include any other
greetings or additional explanations outside of the JSON.
# Output format (JSON)
{
“OOO_Article O”: [
“[Paragraph 1] Original text and details of the principle-based ground derived from the
relevant clause”,
“[Paragraph 2] Another original text content in cases where different regulations apply to
each subject”,
“[Exception] Content of exceptions and provisos existing within the relevant clause”
],
“answer”: “Explanation of the principle-based regulation. Provided that, in the case of
[Exceptional Condition], [Exceptional Matter] applies.”
}
Figure 5: The System Prompt Template structures used by RefWalk (RegOps-Bench).
19

RefWalk System Prompt Template (HIPAA)
You are an AI compliance expert in the U.S. healthcare privacy regulations (HIPAA, 45 CFR Parts
160 and 164).
# Instruction
1. Context dependency: Every claim and the final answer MUST be drawn from the provided ’Context’
only. Do not invoke outside knowledge or fabricate content.
2. Reference-key format: Each JSON key MUST be the ‘node_id‘ exactly as it appears between the
[ and ] of a Context passage header — without the surrounding brackets. Do not invent, modify,
or concatenate identifiers. (Example: header [spans-passthrough-candidates/sent_0012-0ec05553]→
key “spans-passthrough-candidates/sent_0012-0ec05553”.)
3. Cite minimally: include only the rule(s) that materially support the answer. The default is
one rule per question; cite a second only when the answer genuinely depends on a distinct second
rule. Do not echo every passage in the Context.
4. Claim extraction: Under each cited node_id, list the specific facts the rule contributes
toward the answer (JSON array of one-sentence English strings). Surface an exception or proviso
(“except”, “provided that”, “unless”) only when the cited rule itself explicitly contains one —
do NOT search for exceptions when none are present in the cited text.
5. Final ’answer’ field:
- Concise English synthesis grounded in the cited claims.
- Reference the regulation in-line by its CFR section (e.g., "§164.502(a)(1)") so the reasoning
is traceable.
- State any explicit exception when it applies.
6. Output format: Respond with ONLY the JSON object below. No greetings, no preface, no trailing
prose.
# Output format (JSON)
{
“<node_id>”: [
“Primary normative content derived from this rule”,
“Exception or proviso, only when the cited rule explicitly contains one”,
],
“answer”: “Final answer text. Reference the regulation in-line (e.g., §164.502(a)) and state
any explicit exception when it applies.”
}
RefWalk User Template (HIPAA)
## Context:
{ctx}
The question below is about {topic}. The conditions for the topic are as follows:
- actor: {actor}
- temporal: {temporal}
- magnitude: {magnitude}
- situational: {situational}
## Question: {q}
Figure 6: The separated System Prompt and User Template structures used by RefWalk (HIPAA).
20

NativeRAG System Prompt & Citation Instruction (RegOps-Bench, Korean)
당신은국가연구개발사업및관련법령/규정해석을지원하는전문가입니다.주어진조항(법령·시행령·고
시·기준)에근거하여사용자의질문에한국어로답변하세요.조항에명시되지않은사항은추측하지말고,
부족한경우그사실을명시하세요.
# Citation-Footer Instruction
답변의마지막줄에다음형식으로인용한조항의node_id목록을출력하라.
예: [참조]국가연구개발사업_연구개발비_사용기준_제48조,국가연구개발사업_연구개발비_사용기준_제48조_
제8항
node_id는위’Reference Document List’에표기된항목을그대로사용하라.
NativeRAG System Prompt & Citation Instruction (RegOps-Bench, English)
You are an expert who supports the interpretation of national research and development (R&D)
projects and related statutes/regulations. Based on the provided clauses (statutes, enforcement
decrees, notifications, standards), answer the user’s questions in Korean. Do not speculate on
matters not explicitly stated in the clauses; if information is insufficient, clearly state that
fact.
# Citation-Footer Instruction
On the very last line of your response, output a list of the node_ids of the cited clauses in
the following format.
Example: [참조]국가연구개발사업_연구개발비_사용기준_제48조,국가연구개발사업_연구개발비_사용기준_
제48조_제8항
For the node_id, use the exact items as listed in the ‘Reference Document List’ above.
NativeRAG System Prompt & Citation Instruction (HIPPA)
You are a legal/compliance assistant well-versed in U.S. healthcare privacy regulations (HIPAA).
Answer the user’s question in English, grounded strictly in the rules provided below. Do not
speculate beyond what the rules state; if the rules are insufficient to answer, say so explicitly.
# Citation-Footer Instruction
On the final line of your answer, list the node_ids of the rules you cited in the format below.
Example: [Citations] spans-passthrough-candidates/sent_0012-0ec05553,
spans-passthrough-candidates/sent_0010-99ea86fe
Use node_ids exactly as they appear in the ‘Reference Document List’ above; do not invent
identifiers.
NativeRAG User Template (English)
Question: {question}
Reference Document List:
[{i}] [{role}] {node_id}
{text}
{citation_instruction}
Figure 7: The free-form generation prompt topology consisting of separate system and user instruction blocks for
the baseline frameworks.
21

[Question]
[KO]해외파트너와공동수행중인과제의연구개발비집행을계획보다변경해야할경우,별도인가나절차가
필요한가요?또한원화대비외화가치가하락해예산잔액이부족해지는경우에도 동일한절차를밟아야
하나요?
[EN] If we need to modify the R&D budget execution plan for a project conducted jointly with
an overseas partner, is a separate authorization required? Furthermore, must we follow the same
procedure if the budget becomes insufficient due to a decrease in foreign currency value against
the Korean Won?
[Per-rule attribution]
Cost-Use Standards, Art. 28 (···_사용기준_제28조)
•[제5항]연구개발기관의 장은국제공동연구개발비를원래계획과다르게변경하여사용하려는때에는제73
조제1항제8호에따라중앙행정기관의 장의사전승인을거쳐협약을변경한후해당국제공동연구개발비를
계상하여야한다.
[EN] (5) When the head of a research and development institution intends to use international joint
R&D funds differently from the original plan, they shall appropriate the funds only after obtaining
prior approval from the head of the central administrative agency and amending the agreement
pursuant to Article 73(1)8.
•[예외]단,환율의변동만으로금액이달라지는경우에는제외한다.
[EN] [Exception] Provided, That this shall not apply where the amount changes solely due to
exchange-rate fluctuations.
Cost-Use Standards, Art. 73 (···_사용기준_제73조)
•[제1항제8호]국제공동연구개발비를원래계획과다르게변경하려는경우(단,환율의변동만으로금액이달
라지는경우에는제외한다)는사전승인대상이다.
[EN] (1)8. Any modification to international joint R&D funds deviating from the original plan is
subject to prior approval (excluding cases where the amount changes solely due to exchange-rate
fluctuations).
[Answer]
[KO]해외파트너와공동수행중인과제(국제공동연구개발사업)의연구개발비집행을계획보다변경해야할
경우,원칙적으로중앙행정기관의 장의사전승인을받아협약을변경한후계상해야합니다(사용기준제28
조제5항및제73조제1항제8호).다만,원화대비외화가치하락으로예산잔액이부족해지는등‘환율의
변동만으로금액이달라지는경우’에는별도의사전승인절차가면제됩니다.
[EN] In principle, changing the budget execution for a joint project with an overseas partner
(international joint R&D) requires prior approval from the head of the central administrative
agency and a subsequent agreement amendment before appropriation (Art. 28(5) and Art. 73(1)8 of the
Cost-Use Standards). However, if the budget shortfall stems solely from exchange-rate fluctuations,
such as a depreciation of the foreign currency against the Won, the requirement for a separate
prior approval is waived.
Figure 8: Qualitative example of schema-bound exception surfacing (L2). The model correctly isolates the tail
exception clause and binds it to its parent statutory anchor instead of emitting a free-form textual caveat.
22

[Question]
[KO]저희기관에서올해사업비집행중 직접비가부족하여간접비조정을받았고,이로인해간접비잔액이
늘었습니다.남은 잔액으로연구개발능률성과급을지급하려하는데,성과급으로계상할수있는상한비율이
어떻게되며주의할점이 있을까요?
[EN] Our institution received an indirect-cost adjustment this year due to a shortfall in direct
costs, which consequently increased our remaining indirect-cost balance. We plan to pay R&D
efficiency performance bonuses from this surplus. What is the maximum allowable ratio for this
bonus, and are there any critical compliance issues we should note?
[Per-rule attribution]
Art. 43 (Government-Funded Institute)
•[제6항]정부출연기관의 장은연구개발능률성과급을해당기관의회계연도간접비총액의10퍼센트이하로
계상하여야한다.
[EN] (6) The head of a government-funded research institute shall budget R&D efficiency performance
bonuses at no more than 10% of the institution’s total indirect costs for the relevant fiscal year.
•[준용]제43조는정부출연기관에대한규정이나,대학및기타비영리기관에대해준용됨(제51조제4항,제59
조).
[EN] [Mutatis Mutandis] Although Article 43 explicitly addresses government-funded institutes, it
applies mutatis mutandis to universities and other non-profit organizations via Articles 51(4) and
59.
Art. 51 (University)
•[제4항]대학이사용하는직접비의부족에따른간접비조정···및연구개발능률성과급계상에관하여는
제43조제3항,제4항,제6항을준용한다.이경우“정부출연기관”은“대학”으로본다.
[EN] (4) Regarding indirect-cost adjustments due to direct-cost shortfalls···and the budgeting of
R&D efficiency performance bonuses for universities, Article 43(3), (4), and (6) shall apply mutatis
mutandis. In such cases, “government-funded research institute” shall be read as “university.”
Art. 59 (Other Non-Profit)
•[본문]기타비영리기관이사용하는간접비의사용기준에관하여는제43조를준용한다.이경우“정부출연기관”
은“기타비영리기관”으로본다.
[EN] [Main Text] The standards for using indirect costs by other non-profit organizations shall
be governed mutatis mutandis by Article 43. In such cases, “government-funded research institute”
shall be read as “other non-profit organization.”
Art. 69 (For-Profit)
•[제3항]영리기관의 장은연구개발능률성과급을해당연구개발과제간접비의10퍼센트이하로계상하여야
한다.
[EN] (3) The head of a for-profit institution shall budget R&D efficiency performance bonuses at
no more than 10% of the indirect costs allocated to the relevant R&D project.
[Answer]
[KO]연구개발능률성과급의상한은기관유형에따라적용기준이다릅니다.첫째,정부출연기관·대학·기
타비영리기관은제43조(대학은제51조제4항,기타비영리기관은제59조준용)에따라‘해당기관의회계연도
간접비총액’의10퍼센트이하로계상해야합니다.즉,개별과제별잔액이아닌기관전체의간접비총액이
기준입니다.둘째,영리기관은제69조제3항에따라‘해당연구개발과제간접비’의10퍼센트이하로계상해야
합니다.간접비조정으로잔액이늘었더라도위10%상한을초과하여계상·지급할수없습니다.
[EN] The ceiling for R&D efficiency performance bonuses varies by institution type. First, for
government-funded institutes, universities, and other non-profits (governed by Art. 43; applied
mutatis mutandis to universities via Art. 51(4) and to non-profits via Art. 59), the bonus is capped
at 10% of the institution’s total indirect costs for the fiscal year. This means the baseline is
the aggregate indirect costs of the entire institution, not the individual project balance. Second,
for-profit institutions must cap the bonus at 10% of the relevant project’s specific indirect costs
pursuant to Article 69(3). Even if the indirect-cost balance expands due to adjustments, payments
cannot exceed these respective 10% statutory caps.
Figure 9: Qualitative example of exhaustive parallel-institution closure (L3). The model dynamically tracks legal
mutatis mutandis(deemed application) edges across distinct organizational frameworks to construct an aggregate
compliance answer.
23

[Question]
[KO]저희기관이해외의A대학교와B연구소두곳에공동으로개발업무를위탁하고있는데,현지지급을위해
기존예산을증액하려고합니다.이때해외공동수탁처에대한자금이관및증빙자료관리는통합계좌와
통합정보시스템상에서어떻게처리해야혁신법절차상무리가없을까요?
[EN] Our institution is jointly outsourcing development tasks to two overseas entities (University
A and Research Institute B), and we intend to increase the existing budget for local disbursements.
To ensure compliance with the procedural requirements of the Innovation Act, how should we handle
the transfer of funds and the management of supporting documentation for these overseas co-trustees
within the integrated account and the integrated information system?
[Per-rule attribution]
Enforcement Decree, Art. 24 (Tier 2)
•[제1항]연구개발기관의 장은···통합정보시스템에서연구개발과제마다별도의계정을생성하여···연구개
발비카드를발급받아야한다.
[EN] (1) The head of a research and development institution shall establish a separate account for
each R&D project within the integrated information system and be issued an R&D funds card···
•[제4항]연구개발기관의 장은···통합계좌를개설하여 연구개발비를운영할수있다.다만,제19조제1
항제1호,제2호또는제4호에해당하는···경우는제외한다.
[EN] (4) The head of a research and development institution may open an integrated account to manage
R&D funds; provided, that institutions falling under Article 19(1)1, 2, or 4 shall be excluded.
Cost-Use Standards, Art. 28 Hallmarks (Tier 4)
•[제3항]연구개발과제계정에서국외에소재한기관또는외국인의계좌로이체···된국제공동연구개발비는
사용된연구개발비로본다.
[EN] (3) International joint R&D funds transferred from an R&D project account to an account of an
overseas institution or a foreign national···shall be deemed as expended R&D funds.
•[제4항]국외기관간 계약서,계좌이체증명등증명자료를갖추어야한다.
[EN] (4) Supporting documentation, including contracts with the overseas entity and bank transfer
confirmations, must be maintained.
•[제5항]국제공동연구개발비를원래계획과다르게변경하여사용하려는때에는제73조제1항에따라사전
승인을거쳐협약을변경한후계상하여야한다.
[EN] (5) When intending to use international joint R&D funds differently from the original plan,
the funds shall be appropriated only after obtaining prior approval and amending the agreement
pursuant to Article 73(1).
[Answer]
[KO]첫째(예산증액),국제공동연구개발비를증액하려면사용기준제28조제5항에따라중앙행정기관의 장의
사전승인을받아협약을변경한후계상해야합니다(단순환율변동은예외).둘째(자금이관·회계),과제
계정에서해외기관계좌로이체된금액은제28조제3항에따라사용된연구개발비로보며,시행령제24조제1
항에따라통합정보시스템에서과제별 별도계정을생성해관리하고효율적관리를위해통합계좌를운영할
수있습니다(특정유형기관은제외).셋째(증빙),제28조제4항에따라계약서·계좌이체증명등증명자료를
갖추어야합니다.
[EN] First (Budget Increase): To increase international joint R&D funds, you must obtain prior
approval from the head of the central administrative agency and amend the agreement before
appropriation pursuant to Article 28(5) of the Cost-Use Standards (except for pure exchange-rate
fluctuations). Second (Fund Transfer & Accounting): Sums transferred from the project account to
an overseas institution’s account are deemed spent R&D funds under Article 28(3). In accordance
with Article 24(1) of the Enforcement Decree, a separate account per project must be managed within
the integrated information system, and an integrated account may be operated for administrative
efficiency (excluding certain types of institutions). Third (Evidentiary Proof): You must maintain
formal supporting evidence, such as contracts with the foreign entities and bank transfer records,
pursuant to Article 28(4).
Figure 10: Qualitative example of cross-document chain traversal (L4). The reasoning path spans from high-level
statutory frameworks (Tier 2 Enforcement Decree) down to micro-level procedural constraints (Tier 4 Cost-Use
Standards).
24
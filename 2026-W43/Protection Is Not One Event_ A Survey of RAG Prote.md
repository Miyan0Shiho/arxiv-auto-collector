# Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint

**Authors**: Peiyang Liu, Xi Wang, Di Liang, Youru Li, Zhi Cai, Wei Ye

**Published**: 2026-10-21 06:17:23

**PDF URL**: [https://arxiv.org/pdf/2610.30193v1](https://arxiv.org/pdf/2610.30193v1)

## Abstract
Retrieval-augmented generation (RAG) systems face security, privacy, and copyright risks including content leakage, source reconstruction, unauthorized access, data poisoning, and prompt injection. Yet RAG protection studies use incompatible evaluation metrics, datasets, and threat models. We present a systematic survey and human-coded evidence map of 118 full-text-confirmed RAG defense papers, organized by measured endpoint rather than claimed goal or defense mechanism. Our seven-family taxonomy covers content disclosure, formal privacy proxies, access control and suppression, corpus-use detection, and pipeline integrity and attribution. Thirty-two annotators double-coded the papers kappa=0.867. The survey identifies three gaps. First, retrieval reranking and context rewriting each span five or six endpoint families, so mechanisms do not identify protection outcomes. Second, only 17.8% of studies use content disclosure or reconstruction as their primary endpoint; under the frozen copyright/ownership relation label, 73.3% of records have corpus-use detection as their primary endpoint, and none have an emitted-expression primary endpoint. Third, most evaluations use general-purpose question-answering datasets and non-adaptive attacks rather than sensitive data and adaptive, multi-turn extraction. Our controlled RAG testbed shows that preventing verbatim copying does not prevent private-fact recovery: a normalized 8-gram guard reduces 20-token copying to 0.0%, but a three-query adaptive attacker recovers 48.3% of hidden facts on Qwen3-8B. We release the taxonomy, dataset, coding protocols, and testbed to enable comparable, claim-aligned RAG security evaluation.

## Full Text


<!-- PDF content starts -->

Protection Is Not One Event:
A Survey of RAG Protection Research
by Measured Endpoint
Peiyang Liu1,2, Xi Wang2, Di Liang3, Youru Li4, Zhi Cai4and Wei Ye1,*
1National Engineering Research Center for Software Engineering, Peking University, Beijing, China,
2School of Software and Microelectronics, Peking University, Beijing, China,
3Tencent Technology, Shenzhen, China,
4College of Computer Science, Beijing University of Technology, Beijing, China.
liupeiyang@pku.edu.cn
Abstract
Retrieval-augmented generation (RAG) sys-
tems face security, privacy, and copyright risks
including content leakage, source reconstruc-
tion, unauthorized access, data poisoning, and
prompt injection. Yet RAG protection studies
use incompatible evaluation metrics, datasets,
and threat models. We present a systematic sur-
vey and human-coded evidence map of 118 full-
text-confirmed RAG defense papers, organized
by measured endpoint rather than claimed goal
or defense mechanism. Our seven-family tax-
onomy covers content disclosure, formal pri-
vacy proxies, access control and suppression,
corpus-use detection, and pipeline integrity
and attribution. Thirty-two annotators double-
coded the papers (κ= 0.867).
The survey identifies three gaps. First, retrieval
reranking and context rewriting each span five
or six endpoint families, so mechanisms do
not identify protection outcomes. Second, only
17.8% of studies use content disclosure or re-
construction as their primary endpoint; under
the frozen copyright/ownership relation label,
73.3% of records have corpus-use detection
as their primary endpoint, and none have an
emitted-expression primary endpoint. Third,
most evaluations use general-purpose question-
answering datasets and non-adaptive attacks
rather than sensitive data and adaptive, multi-
turn extraction.
Our controlled RAG testbed shows that prevent-
ing verbatim copying does not prevent private-
fact recovery: a normalized 8-gram guard re-
duces 20-token copying to 0.0%, but a three-
query adaptive attacker recovers 48.3% of hid-
den facts on Qwen3-8B. We release the taxon-
omy, dataset, coding protocols, and testbed to
enable comparable, claim-aligned RAG secu-
rity evaluation.
*Corresponding author1 Introduction
Retrieval-augmented generation (RAG) connects
language models to external document collections
at inference time (Lewis et al., 2020; Gao, 2023).
Because this external datastore can be searched,
updated, and quoted directly, it creates new risks
around privacy, copyright, and system security.
Even if a document never appeared in pre-training,
a RAG system might still leak its private details,
copy its wording verbatim, or fall victim to poi-
soned entries.
A rapidly growing literature aims to “protect”
RAG systems against these risks, yet papers using
this label rarely measure the same thing. Consider
two studies that both claim to protect the copyright
of a retrieval database. The first plants a watermark
to prove after the fact that the database was queried,
while the second blocks unauthorized users from
retrieving restricted documents. A third study de-
fends against poisoned passages to keep answers
accurate, and a fourth rewrites retrieved text so the
generator cannot copy it word for word. All four
papers report successful “protection,” but they eval-
uate completely different technical events. Treating
them as interchangeable, or pooling their scores
to compare defenses, is like comparing a burglar
alarm to a fireproof safe.
Existing surveys organize this literature by threat
model, pipeline stage, or system architecture. How-
ever, knowing where a defense sits in the pipeline
tells us very little about what it actually achieves.
In this work, we take a different approach: we
survey RAG protection research based on the ex-
act empirical endpoint each study measures. We
define a frozen seven-family taxonomy of mea-
sured endpoints (Table 1 and Figure 1) that sep-
arates actual content disclosure from access con-
trol, usage detection, and pipeline integrity. Using
this taxonomy, 32 annotators double-coded 118
full-text-confirmed RAG protection papers, reach-
arXiv:2610.30193v1  [cs.CR]  21 Oct 2026

ing high agreement on what each paper measures
(κ= 0.867).
Looking at the literature through its measured
endpoints reveals three major blind spots. First,
defense mechanisms only weakly predict what a
paper tests (§5.1). Popular techniques such as re-
trieval reranking and context rewriting are evalu-
ated across five or six different endpoint families,
meaning readers cannot infer what was protected
just from how the system was built. Second, re-
search attention is heavily skewed away from ac-
tual data leakage (§5). Only 21 of the 118 papers
(17.8%) have a primary endpoint that measures
whether protected content is emitted to the user
or reconstructed from the system. Among studies
with the frozen copyright/ownership relation label,
73.3% have corpus-use detection as their primary
endpoint, whereas none have an emitted-expression
primary endpoint. Third, evaluation setups often
fail to stress-test the defenses they propose (§5.3).
Most studies rely on standard question-answering
datasets that contain little sensitive content to begin
with, and very few test against adaptive attackers
who can ask follow-up questions. Our own con-
trolled experiments confirm why this distinction
matters: defenses that drive word-for-word copy-
ing down to zero still allow attackers to recover the
underlying private facts (Appendix C).
To help future work build on clear empirical
foundations, we make four contributions:
•An endpoint-based surveythat organizes
RAG defense mechanisms, evaluation met-
rics, datasets, and threat models by the exact
technical event they measure (§4, §5.3).
•A validated seven-family event taxonomy
that makes cross-paper comparisons repro-
ducible (§3).
•A human-coded evidence mapof 118 direct
RAG protection papers, complete with item-
level annotations, operating points, and open-
source verification tools (§3, Appendix D.2).
•A critical audit and controlled testbedexam-
ining reporting quality, and the gap between
stopping surface copying and preventing fact
recovery (§5, §6, Appendix C).
2 Related Work
Prior surveys of RAG security and privacy orga-
nize the field around system architecture and at-
tack surfaces. Broad reviews of language modelprivacy cover threats like membership inference
and data poisoning (Das et al., 2025) alongside
cryptographic and differentially private defenses
(Chen et al., 2025a). Dedicated surveys on copy-
right and ownership focus on text and model wa-
termarking (Yang et al., 2025) or fingerprinting
techniques (Xu et al., 2025). Within the RAG lit-
erature specifically, existing reviews group studies
by threat model, pipeline stage, or defensive mod-
ule (Mu et al., 2026; Palanisamy et al., 2026; Xu
et al., 2026). While this architectural view shows
where a defense intervenes, it obscures what the
evaluation actually tests. Individual empirical stud-
ies already hint at this divergence: papers evaluat-
ing literal and non-literal text reproduction (Chen
et al., 2024), copyright-sensitive activation steering
(Zhang et al., 2026c), proprietary knowledge base
defense (Li et al., 2026a), and knowledge extraction
benchmarks (Qi et al., 2026) all define “protection”
through different operational outcomes. Further-
more, recent adaptation methods train models to
decide when to retrieve, critique context, or with-
hold answers (Asai et al., 2024; Lin et al., 2024),
blurring the boundary between parametric memory
and external datastores. Rather than grouping pa-
pers by where their mechanisms sit in the pipeline,
we human-code the exact technical events their ex-
periments measure.
Our approach builds on meta-scientific audits of
machine learning research (Lipton and Steinhardt,
2019; Kapoor and Narayanan, 2023) and classical
measurement theory, which requires clearly defin-
ing target concepts and validating how they are
measured (Cronbach and Meehl, 1955; Campbell
and Fiske, 1959; Messick, 1995). These principles
have shaped recent critiques of benchmark design
and fairness evaluation in natural language pro-
cessing (Jacobs and Wallach, 2021; Blodgett et al.,
2021; Raji et al., 2021; Bowman and Dahl, 2021).
However, while prior audits typically expose flaws
inside an experimental setup, we examine the link
between two separate components of a paper: its
high-level protection claim and its measured em-
pirical endpoint. This focus on claim-measurement
alignment parallels recent findings in question an-
swering, where standard lexical overlap metrics
often fail to capture true semantic accuracy (Bulian
et al., 2022; Kamalloo et al., 2023; Adlakha et al.,
2024).

Protectioneventa record measures
118human double-coded ( κ= 0.867 )
Content disclosure
27 recordsAccess
suppressionCorpus-use
detectionIntegrity &
attribution
A
emitted expression copy-
ing
8A-pii
emitted atomic identifier
6B
source recovery outside
ordinary generation
7C
likelihood, formal, or indi-
rect proxy
6D
access, retrieval, or policy
suppression
26E
watermark, membership,
or ownership detection
18F
attribution, integrity, re-
moval, or misuse
47
A.verbatim
contiguous verbatim
source wording
(Zeng et al., 2024a;
Wang et al., 2026a;
Zeng et al., 2025a;
Liang et al., 2026b)
A.near
near-verbatim or para-
phrase
(Li et al., 2026a; Zhu
et al., 2026; Zhang
et al., 2026b)A-pii.direct
direct identifiers:
names, addresses
(Li et al., 2026b; Zhao
et al., 2027; Cheng
et al., 2026; Moulton
et al., 2025)
A-pii.quasi
quasi-identifiers, link-
age attributesB.embed
from embeddings or a
vector index
(Jin et al., 2025; Li
et al., 2026c; Wanger,
2026; Ye et al., 2025)
B.state
from parameters,
cache, or state
(Chen et al., 2026a;
Mao et al., 2026)C.formal
DP, RDP, or a certified
bound
(Golatkar et al., 2024;
Koga et al., 2024; Wu
et al., 2025b; Leben-
sold et al., 2024)
C.stat
likelihood or perplexity
proxy
(Min et al., 2024)D.answer
answerability given
retrieval
(Agnew et al., 2024;
Wang et al., 2025a;
Pallerla et al., 2026;
Hu et al., 2024)
D.retrieve
retrieval or routing of
a document
(Yao et al., 2025a;
Namboothiri, 2026;
Arceo and Narsing,
2026; Budigi and Siri-
giri, 2026)E.signal
a planted watermark
or canary
(Chen et al., 2025b;
Guo et al., 2025; Jo-
vanovi ´c et al., 2025;
Liu et al., 2025c)
E.member
membership, no
planted signal
(Zeng et al., 2025b;
Choi et al., 2025;
Zhang et al., 2025c;
Burnat, 2026)
E.owner
an ownership or iden-
tity verdict
(Liu et al., 2026)F.poison
poisoning or corpus
corruption
(Zhang et al., 2025a;
Yang et al., 2024; Cer-
aku, 2026; Sharma,
2026)
F.attrib
attribution, prove-
nance, citation
(Qi et al., 2024; Zhang
et al., 2026a; Cui and
Liu, 2026; Wang et al.,
2025b)
F.inject
prompt injection or
hijacking
(Saleem et al., 2026;
Cohen et al., 2024;
Datta et al., 2026;
Alam et al., 2026)
F.remove
removal, unlearning, or
opt-outHow to read a cell.Counts arehuman-coded core. A record sits under the event itsreported primary endpoint
measures, not the event its abstract names.
Figure 1: The frozen taxonomy and its 17 subfamilies. A record is assigned by its primary measured endpoint, not
by its topic.
3 The Endpoint Taxonomy and
Human-Coded Evidence Map
To compare RAG protection studies fairly, we first
need a clear way to categorize what their experi-
ments actually measure. Treating “protection” as
a single outcome confuses very different techni-
cal events. Stopping a model from copying source
text word for word is not the same as hiding per-
sonal identifiers, preventing attackers from invert-
ing vector embeddings, proving a differential pri-
vacy bound, blocking unauthorized retrieval, de-
tecting watermarks, or filtering out poisoned docu-
ments.
3.1 A Seven-Family Taxonomy of Measured
Endpoints
We introduce a frozen seven-family taxonomy
(Families A, A-pii, and B–F) that groups studies by
the exact empirical quantity their primary metric
evaluates, rather than where their defense mech-
anism sits in the RAG pipeline (Table 1 and Fig-
ure 1). This design deliberately cuts across the
module-based categories used in prior security re-
views (§4). For example, if a paper proposes an
unlearning method but evaluates it solely by mea-
suring how often the system refuses to answer for-
bidden queries, our taxonomy places it under ac-
cess suppression (Family D) rather than removal.Conversely, if a paper tests whether deleted data
can still be reconstructed from residual index states,
it belongs to source recovery (Family B).
The seven families fall into four broader asset
groups based on what is being protected:
•Content Disclosure and Formal Proxies
(Families A, A-pii, B, and C):These end-
points concern the secrecy of the underly-
ing data. Family A measures whether gen-
erated answers copy protected wording verba-
tim or through close paraphrase. Family A-pii
tracks whether the output leaks specific per-
sonal identifiers like names or phone numbers.
Family B measures whether an attacker can re-
construct source documents outside ordinary
text generation, such as by inverting embed-
dings or inspecting index structures. Family
C groups studies that report a mathematical
privacy bound (such as differential privacy ε)
or a perplexity shift without measuring ac-
tual leaked text. Because Family C reports a
theoretical or statistical proxy rather than ob-
served leakage, only Families A, A-pii, and B
directly measure content disclosure to a user
or adversary.
•Access Suppression (Family D):Endpoints
here test whether the system blocks unautho-

rized documents from being retrieved, routed,
or used to answer a query, typically through
role-based filters or refusal mechanisms.
•Corpus-Use Detection (Family E):Rather
than preventing data exposure, these end-
points test whether an auditor can prove after
the fact that a document or database was used,
either by detecting planted watermarks and
canaries or by running membership inference
audits.
•Pipeline Integrity and Attribution (Fam-
ily F):The largest group in our map evalu-
ates whether the RAG system resists exter-
nal manipulation (such as corpus poisoning
or prompt injection) or correctly attributes
generated answers back to their source docu-
ments. Family G is reserved for underdeter-
mined cases where the reported packet does
not establish a specific technical event.
Fixing category boundaries by the measured met-
ric makes cross-paper comparison reproducible.
When 32 annotators double-coded all 118 core
records using this seven-family scheme, they
reached high agreement on the primary endpoint
family (κ= 0.867).
3.2 Literature Search, Human Coding, and
Corpus Scope
To build the evidence map, we searched arXiv, Ope-
nAlex, and the ACL Anthology through July 17,
2026, following standard scoping review protocols
(Tricco et al., 2018; Peters et al., 2021). Our queries
returned 11,972 total hits, which deduplicated to
8,978 unique records. Screening titles and abstracts
narrowed this pool to 925 candidate papers, for
which we recovered 824 full texts (Appendix B).
Thirty-two trained annotators with backgrounds
in natural language processing, machine learning,
security, privacy, and information science double-
coded all 824 recoverable full texts. Any dis-
agreements were resolved by an independent ad-
judicator under a frozen protocol. To verify that
our initial screening did not miss relevant work,
we also double-screened a stratified sample of
1,200 excluded records to test our stopping rule
(Callaghan et al., 2024). To qualify as a direct
RAG-protection study in our core map, a paper
had to evaluate a retrieval-augmented system, in-
tervene on its query, corpus, index, context, or out-
put, and report a source-verified protection or at-tribution metric. Pure attack papers, benchmarks,
surveys, and untested proposals were tracked sep-
arately, and reporting task utility alone was not
enough for inclusion. Each confirmed paper was as-
signed a single primary endpoint family to prevent
multi-module systems from being double-counted
as separate studies, while defense mechanisms and
intervention stages could receive multiple labels.
This process produced 118 full-text-confirmed
direct RAG-protection records (112 from the main
candidate pool and 6 recovered from the excluded-
record audit). For each record, coders annotated
34 structured fields covering system architecture,
protected assets, defense mechanisms, measured
outcomes, evidence relations, and publication sta-
tus, reaching a median agreement of κ= 0.94
across all released fields (range 0.56 to1.00). We
release the de-identified 8,024-row human label ma-
trix alongside verification scripts that recompute
agreement across every field (Appendix B), and we
verified every cited operating point directly against
the source PDF rather than external metadata (Ap-
pendix B.3).
Our design-based audit of excluded papers es-
timates that roughly 27.2 additional confirmable
studies may exist beyond our candidate pool under
the frozen screening rule. Throughout the paper,
we report proportions with 95% Wilson score con-
fidence intervals and permutation-test p-values as
exact descriptions of our 118-record analyzed pool,
rather than absolute population totals for the entire
field (Appendix B.3). As we show next, different
endpoint families not only measure distinct events,
but often move in opposite directions under the
very same defense (Table 4).
4 The Landscape of Defenses by
Measured Endpoint
Every empirical RAG protection study specifies
where it intervenes in the pipeline, what mecha-
nism it applies, and what outcome metric it mea-
sures. While prior surveys organize the literature
around the first two dimensions (Mu et al., 2026;
Palanisamy et al., 2026; Xu et al., 2026), we or-
ganize our analysis around the third. Below, we
examine each endpoint family in turn, showing
how the defenses tested against it work, where they
trade off utility or fail under pressure, and what
blind spots their metrics leave behind. All counts
refer to the 118 human-coded core records, and Ap-
pendix D.2 lists the exact operating points reported

by each study.
4.1 Emitted Text and Identifiers (Families A
and A-pii)
Families A and A-pii are the only endpoints in
our taxonomy that directly score the text a user
receives. Family A measures whether the gener-
ated response reproduces protected source word-
ing verbatim or through close paraphrase, while
Family A-pii tracks whether the response leaks
atomic personal identifiers such as names, phone
numbers, or email addresses. Despite sitting at the
heart of privacy and copyright concerns, these two
families together account for just 14 of the 118
records in our map. The defenses evaluated here
intervene at four distinct points along the genera-
tion path. First, retrieval-side thresholding drops
passages whose embedding distance to the query
falls outside a safe band, ensuring the generator
never sees quotable source text (Zeng et al., 2024a).
Second, context transformation rewrites or con-
denses retrieved passages before generation, using
either extractive compression (Zeng et al., 2025a)
or multi-agent semantic rewriting that paraphrases
sensitive spans while keeping the passage useful
for answering questions (Zhao et al., 2027). Third,
constrained decoding intervenes during token gen-
eration to block verbatim reproduction: privacy-
aware decoding adds calibrated noise to next-token
probabilities under a differential privacy budget
(Wang et al., 2026a), adaptive backtracking rolls
the decoder back whenever a partial span begins to
match a protected entity (Yao et al., 2026), and dual-
path generation synthesizes answers from a draft
model while consulting the protected source only
through a restricted reference channel (Li et al.,
2026a). Fourth, studies in Family A-pii typically
rely on a two-stage filter pair, combining an input
judge that blocks extraction-style prompts with an
output regular expression or named-entity filter that
redacts identifiers before returning the final answer
(Li et al., 2026b; Moulton et al., 2025).
Because emitted-content metrics penalize even
partial text overlap, defenses in Families A and
A-pii face a much stricter test than binary access
checks. As a result, they report the lowest rates of
unqualified success across the entire evidence map:
88% of records in Family A and only 50% in Fam-
ily A-pii report net improvements, compared with
94% to 96% in the access, detection, and integrity
families. They are also the two families where
adaptive attackers most frequently recover the pro-tected information, occurring in 38% of Family
A records and 33% of Family A-pii records, ver-
sus just 4% for access suppression, 6% for usage
detection, and 0% for pipeline integrity. Look-
ing closely at their operating points explains why
achieving complete protection is so difficult. When
interventions push text reproduction toward zero,
task utility often drops sharply: distance thresh-
olding cuts source copying in Zeng et al. (2024a)
at the cost of reducing task ROUGE-L from 0.39
to 0.13, entity-level anonymization can sacrifice
up to 35% in Recall@ k(Zhu et al., 2026), and
privacy-aware decoding at strict budgets ( ε= 0.01
to0.05) causes model perplexity to diverge (Wang
et al., 2026a). Conversely, when systems relax
these controls to preserve answer quality, notice-
able leakage remains. Adaptive backtracking still
leaves an average leakage rate (ALR) of 0.13 and
a context leakage rate (CLR) of 24.3% (Yao et al.,
2026), while dynamic anonymization leaves resid-
ual leakage rates of 0.17 on ChatDoctor and 0.33
on EnronQA (Zhu et al., 2026). Similarly, the mul-
tilingual input judge in Li et al. (2026b) blocks
English extraction prompts but still leaves 7.5%
residual leakage on Arabic queries and 17.5% on
Swahili.
These divergent numbers show that there is no
universal exchange rate between privacy and accu-
racy in RAG generation. An operating point that
looks safe on aggregate copying scores can still fail
on specific entity types, non-English queries, or
multi-turn extraction prompts. Consequently, stud-
ies evaluating Families A and A-pii must report
their residual leakage alongside exact workload
conditions and utility costs, rather than treating a
drop in average text overlap as proof that source
content cannot be extracted.
4.2 Source Reconstruction Outside
Generation (Family B)
Even if a RAG system never leaks a single word
through its generated answers, an attacker can still
reconstruct indexed documents through underlying
system channels. Family B captures these non-
generation reconstruction endpoints, accounting
for 7 records in our map (and every record that
reports a content-reconstruction metric). Research
here targets three distinct leakage channels, each
requiring its own defense style. The most widely
studied channel is embedding inversion, where
an adversary with access to stored vectors runs
a trained decoder such as Vec2Text or a fine-tuned

T5 model to recover the original text. Defenses
against inversion either add entropy-driven pertur-
bations to the vectors (Jin et al., 2025) or apply
distance-preserving and homomorphic encryption
so the index remains searchable without exposing
raw embeddings (Ye et al., 2025; Mao et al., 2026).
A second channel is index-structure and deletion
residue: PRAG demonstrates that a semi-honest
server observing HNSW graph traversal paths can
reconstruct the underlying document graph without
inspecting a single vector (Li et al., 2026c), while
Ghost Vectors reveals that soft-deleted embeddings
linger in HNSW indices and remain invertible until
the index is rebuilt (Chakraborttii et al., 2026). The
third channel is parametric residue, which arises
when retrieved knowledge is compressed into dis-
tilled LoRA weights rather than passed as text con-
text, turning the adapter weights themselves into
an inversion target (Chen et al., 2026a). Reversing
this threat model, one study treats the vector store
as an exfiltration medium and uses outlier detectors
to catch steganographically smuggled embeddings
(Wanger, 2026).
Interpreting reconstruction scores in Family B
requires paying close attention to both the reference
object and the strength of the attacker’s decoder.
Across the seven studies, defenses measure success
against very different targets: distance-preserving
encryption drives Vec2Text reconstruction BLEU
down from 83 to 12 (Ye et al., 2025), path-oblivious
retrieval cuts HNSW edge recovery from 18.3% to
6.8% alongside a measurable drop in retrieval recall
(Li et al., 2026c), and epoch-key rotation reduces
ROUGE-L recovery from soft-deleted vectors from
0.207 to 0.000 (Chakraborttii et al., 2026). Be-
cause recovering source wording, graph edges, and
deleted vectors are distinct events, these numbers
cannot be compared on a single scale. More im-
portantly, a near-zero reconstruction score is only
meaningful if the baseline attack actually works
on the unprotected system. In the clinical-note
experiment of Chakraborttii et al. (2026), for ex-
ample, the unprotected baseline yields a Vec2Text
ROUGE-L of just 0.009 because the off-the-shelf
inverter struggles with domain-specific medical
text; in such cases, a low defended score reflects
a weak attacker rather than a strong defense. Fi-
nally, Family B endpoints are strictly orthogonal
to generation endpoints: encrypting embeddings or
hiding HNSW traversal paths protects the backend
datastore, but it does nothing to stop the language
model from quoting retrieved passages verbatim toan end user (Family A).
4.3 Mathematical Bounds and Likelihood
Proxies (Family C)
Instead of measuring actual text leakage or docu-
ment reconstruction, the 6 records in Family C eval-
uate privacy through theoretical guarantees or sta-
tistical proxies, and every record in this family pairs
its privacy metric with a task utility score. These
studies fall into two distinct groups. The formal
sub-family injects calibrated noise into the RAG
pipeline and tracks a mathematical privacy bud-
get. Within this group, DP-RDM adapts retrieval-
augmented diffusion models to private domains
under a differential privacy budget of ε= 10 (Go-
latkar et al., 2024; Lebensold et al., 2024), while
Private-RAG uses individual Rényi privacy filters
to answer hundreds of queries at ε≈10 in settings
where naive composition would require ε≈1000
(Wu et al., 2025b). Other formal approaches apply
differential privacy directly to token voting during
text generation (Koga et al., 2024) or use pure ε-
DP SimHash bucketing when building the retrieval
datastore (Abouelenein and Torki, 2026). By con-
trast, the statistical sub-family measures privacy
through changes in model likelihood rather than
formal bounds: SILO evaluates nonparametric data-
stores by comparing the model’s perplexity on a
held-out book when related books are present in
the index versus when they are removed, treating
the likelihood shift itself as the primary endpoint
(Min et al., 2024).
While mathematical bounds and likelihood shifts
offer clean theoretical targets, they do not measure
what a deployed system actually emits to a user,
and their numbers cannot be pooled with empiri-
cal leakage rates. Even within Family C, reported
operating points depend heavily on the protected
unit, the composition rule, and the query workload.
For example, DP-RDM achieves its ε= 10 guar-
antee over T= 100 k queries with a generation
FID of 155.67 (Lebensold et al., 2024), whereas
Private-RAG maintains ε≈10 by permanently
retiring individual documents from retrieval once
their personal privacy budgets are exhausted (Wu
et al., 2025b). Meanwhile, the private datastore
approach in Abouelenein and Torki (2026) trans-
lates its SimHash bucketing guarantee into an em-
pirical audit, reducing membership-inference ac-
curacy from 70.22% to 53.60% at the cost of a
2.6% drop in task accuracy. Because query bud-
gets, neighboring-dataset definitions, and utility

trade-offs vary so widely across these setups, a pa-
per reporting ε= 10 and a paper reporting zero
verbatim copied spans (Family A) are providing
fundamentally different types of evidence, even
when evaluated on the exact same corpus.
4.4 Access and Retrieval Suppression (Family
D)
Rather than scoring the words in a generated an-
swer, Family D evaluates whether restricted doc-
uments are retrieved, routed, or allowed to be an-
swered in the first place. Containing 26 records,
this is the largest content-adjacent family in our
map, and 96% of its studies report net improve-
ments. Defenses here differ primarily in where they
enforce authorization along the pipeline, and their
results point to a clear engineering rule: enforcing
access checks late in the pipeline is far riskier than
enforcing them early. The common retrieve-then-
filter pattern, which fetches top- kpassages first and
checks permissions afterward, still exposes unau-
thorized context in 86.1% of base queries (Nam-
boothiri, 2026). By contrast, pushing authorization
down into the database engine through row-level
security or predicate pushdown eliminates cross-
tenant retrieval leakage entirely (Budigi and Si-
rigiri, 2026; Arceo and Narsing, 2026). Beyond
structural database rules, a second group of de-
fenses relies on semantic or learned policies, filter-
ing passages by inferred user role, governing access
at retrieval time, or placing a policy firewall around
the RAG pipeline (Lorenzo et al., 2025; Nutakki,
2026; Yao et al., 2025a). A third group modifies the
system architecture or the documents themselves,
either isolating the communication channel so raw
user prompts never meet protected sources inside
the generator (Cherubin and Paverd, 2025) or per-
turbing source texts so a retrieving model cannot
extract usable answers from them (Agnew et al.,
2024). Notably, RAG unlearning studies that mea-
sure success by how often the model refuses to
answer forbidden questions also fall into Family D
rather than a removal category (Wang et al., 2025a).
Comparing operating points across Family D
highlights a sharp divide between deterministic
permission checks and learned semantic filters.
When access policies are tied to clean database
metadata, engine-level row security drives tenant-
isolation leakage from 0.2% down to zero across
1,000 queries (Budigi and Sirigiri, 2026). However,
when permissions must be inferred from natural
language or when metadata is imperfect, residualleakage quickly reappears. For example, the AI-
driven role filter in ARBITER reaches 85% ac-
curacy and an F1 score of 0.89 over 389 queries,
which actually trails the 0.93 to 0.94 F1 achieved
by a simple static filter (Lorenzo et al., 2025). Sim-
ilarly, hybrid graph-retrieval defenses remain vul-
nerable when 5% to 10% of document metadata
labels are randomly corrupted (Thornton, 2026a),
and enterprise confidentiality filters see their recall
drop to 74.35% when user queries omit explicit
project names (Byun et al., 2025).
More fundamentally, suppressing access or an-
swerability is not the same as preventing content
disclosure. On the one hand, a refusal rate mea-
sures policy compliance rather than whether sen-
sitive knowledge was truly erased from the sys-
tem. On the other hand, achieving zero unautho-
rized retrieval says nothing about what happens
when the userisauthorized. In commercial or li-
censed RAG deployments, legitimate subscribers
are permitted to retrieve from the datastore, yet the
operator must still prevent the model from repro-
ducing copyrighted chapters verbatim. Protecting
against that risk requires evaluating emitted expres-
sion (Family A) alongside retrieval permissions.
4.5 Post-Hoc Corpus-Use Detection (Family
E)
Instead of blocking queries or suppressing gener-
ated text, the 18 records in Family E ask a foren-
sic question: can an auditor prove after the fact
that a specific document collection was indexed
or queried by a RAG system? Strikingly, most
studies framed around copyright and dataset owner-
ship concentrate in this family, even though usage
detection does not measure or prevent content dis-
closure. Research here follows four technical paths
depending on whether the auditor can modify the
corpus beforehand. The dominant design plants
a verifiable signal into the data and later probes
the suspect system to recover it. Researchers have
embedded these signals as statistical watermarks
in generated text (Jovanovi ´c et al., 2025), dual-
layer semantic and lexical watermarks (Liu et al.,
2025b), reasoning traces in chain-of-thought space
(Guo et al., 2025), multi-bit ownership payloads
(Yang et al., 2026), black-box knowledge base wa-
termarks (Lv et al., 2025), synthetic sentinel docu-
ments (Kwok et al., 2026), retrievability-optimized
canaries (Liu et al., 2025c), visual sentinel images
(Luo et al., 2026), and multimodal watermarks for
RAG-as-a-Service platforms (Chen et al., 2025b).

When planting signals in advance is impossible, a
second, signal-free group audits corpus member-
ship directly through black-box queries (Zeng et al.,
2025b). Flipping the roles, a third group builds de-
fenses to protect the datastoreagainstmembership
inference attacks (Choi et al., 2025; Fu et al., 2026),
while a fourth inspects internal model activations
or runtime execution games to catch unauthorized
extraction (Zhang et al., 2025c; Xie et al., 2026b).
Evaluating Family E endpoints requires sepa-
rating the reliability of the forensic signal from
the secrecy of the underlying text. First, detection
scores are highly sensitive to calibration choices,
query budgets, the fraction of target material actu-
ally indexed, and simple post-processing attacks.
For example, the watermark detector in Ward be-
gins to trigger false positives when only 20% of the
watermarked document set is present in the target
corpus (Jovanovi ´c et al., 2025). Under adversar-
ial paraphrasing, membership audit performance in
Zeng et al. (2025b) drops to an AUC of 79.6%. Sim-
ilarly, applying a detect-and-inpaint filter causes
visual sentinel detection to fall from an AUC of
0.99 to 0.62 (Luo et al., 2026), while injecting ex-
ternal knowledge into the database cuts RAG-WM
watermark integrity from 100% down to 46.67%
(Lv et al., 2025). Comparing detectors therefore re-
quires matching their false-positive thresholds and
adversarial post-processing budgets. Second, and
more fundamentally, a usage detector is an alarm
rather than a lock: proving that a document sits in-
side a remote index says nothing about whether its
protected wording was ever emitted to an end user
(Family A), nor whether an attacker could recon-
struct that wording without triggering the alarm.
4.6 Pipeline Integrity, Attribution, and
Recurring Failure Modes (Family F)
Accounting for 47 records (40% of our entire
map), Family F is by far the largest endpoint cate-
gory, covering defenses against corpus poisoning,
prompt injection, and provenance failures. Rather
than protecting the secrecy of indexed documents,
these endpoints test whether the RAG pipeline re-
turns trustworthy, uncorrupted answers and cor-
rectly credits its sources. Poisoning defenses inter-
vene at three stages of the pipeline. Admission-
time filters screen out malicious documents be-
fore they enter the index using perplexity scores,
Mahalanobis distance, vector hubness, or inter-
document similarity (Cheng et al., 2025b; Xian
et al., 2024; Pathak and Sharma, 2026; Jin et al.,2026). Retrieval-time defenses re-rank candidate
passages, partition the retrieved set, or analyze con-
text structure to isolate corrupted chunks (Path-
manathan et al., 2025; Saxena et al., 2025; Yao
et al., 2025b). Generation-time defenses aggre-
gate answers across multiple passages, sometimes
providing formal robustness certificates under a
bounded poisoning budget, or isolate untrusted
spans during synthesis (Tan et al., 2026; Sharma,
2026; Yu et al., 2026). Alongside poisoning de-
fenses, prompt-injection studies enforce runtime
control-flow integrity or deploy layered input scan-
ners (Saleem et al., 2026; Alam et al., 2026), while
a separate attribution literature traces poisoned out-
puts back to responsible documents, localizes cor-
rupted spans at the character level, or uses internal
model states to verify answer provenance (Zhang
et al., 2025a; Cui and Liu, 2026; Zhang et al.,
2026a; Qi et al., 2024).
Comparing evaluation practices in Family F
against the rest of the map reveals a telling con-
trast. Because poisoning and injection studies draw
heavily from adversarial machine learning, 51% of
Family F records receive a low-concern rating for
adaptive robustness testing, and 45% explicitly doc-
ument an adaptive attack among their failure modes.
For example, while a non-adaptive guardrail re-
ports a true-positive rate of 1.000 at a false-positive
rate of 0.015 (Cohen et al., 2024), adaptive white-
box attacks like PoisonedRAG-W push document-
detector false-negative rates up to roughly 96% (Jin
et al., 2026), and consistency collusion bypasses
70.3% of multi-agent audits (Yu et al., 2026). Yet
despite this strong tradition of adaptive testing, zero
records in Family F document an attacker recover-
ing protected source content against the proposed
defense. The reason is simple: in Family F, an
attack success rate (ASR) measures whether the at-
tacker tricked the system into outputting a targeted
wrong answer, executing a hijacked instruction, or
citing the wrong source. Keeping ASR at zero
guarantees that the pipeline behaves as the operator
intended, but it is completely compatible with a
system that answers every user query by quoting
its private or copyrighted sources verbatim.
Looking across all seven endpoint families (Fam-
ilies A, A-pii, and B–F), three recurring evalua-
tion patterns explain why reported operating points
rarely transfer from one setting to another (Ap-
pendix D). First, defenses face steep protection-
utility trade-offs whose magnitudes are seldom re-
ported on a shared scale (Zeng et al., 2024a; Wang

et al., 2026a). Second, claims of zero residual
leakage almost always depend on fragile assump-
tions, such as perfectly clean permission metadata
or attackers who explicitly name their target en-
tities in the query (Nutakki, 2026; Chakraborttii
et al., 2026). Third, moving from static queries to
adaptive attackers routinely overturns the apparent
success of a defense (Yao et al., 2026; Jin et al.,
2025). These recurring limitations point to deeper
structural gaps in how RAG protection is designed
and evaluated, which we audit next.
5 Three Blind Spots in RAG Protection
Research
Stepping back from individual endpoint families,
our evidence map allows us to audit how the RAG
protection literature as a whole connects methods,
claims, and evaluations. Reading the 118 human-
coded studies across families exposes three sys-
temic blind spots that explain why progress in this
field is so difficult to compare.
5.1 Blind Spot I: Mechanisms Do Not Dictate
Endpoints
When readers encounter a RAG defense that
“reranks retrieved passages” or “rewrites context be-
fore generation,” they naturally assume the mecha-
nism tells them what risk is being mitigated. To test
this assumption, we cross-tabulated the 13 coded
defense mechanism types against the primary end-
point family each paper actually measured (Ap-
pendix D). While mechanism type and endpoint
family are statistically associated (mutual informa-
tion of 0.624 bits versus 0.211 bits under 20,000
label permutations, p= 5×10−5), the link is sur-
prisingly weak. Knowing a paper’s defense mecha-
nisms accounts for only 27% of the uncertainty in
its endpoint distribution. In fact, a leave-one-out
classifier that sees a paper’s entire set of defense
mechanisms predicts its measured endpoint family
only 53.4% of the time, barely outperforming the
39.8% baseline of always guessing Family F.
Only two specialized mechanisms point reliably
to a single endpoint: poisoning and integrity con-
trols are evaluated on an integrity endpoint (Family
F) in 94% of their records, and watermarking tech-
niques are evaluated on usage detection (Family
E) in 92%. For every other mechanism in the lit-
erature, the modal endpoint family accounts for
just 28% to 68% of records. This dispersion is
most striking for the two most popular interven-tions in our map: retrieval reranking or threshold-
ing (40 records) and context transformation or sum-
marization (39 records), which span five and six of
the seven endpoint families, respectively. In prac-
tice, the exact same technical intervention, such as
rewriting retrieved passages before they reach the
generator, is presented as a verbatim-copying de-
fense in one paper, a PII redaction tool in a second,
an access-control filter in a third, and a poisoning
defense in a fourth, scored on a completely differ-
ent endpoint each time.
Relying on high-level metric names is almost as
ambiguous as relying on mechanism names (Ap-
pendix D). Only a handful of specialized metrics
identify the underlying protection event on their
own: content-reconstruction scores (7 records) ap-
pear exclusively in Family B, attribution metrics
appear only in Family F, and literal text-overlap
scores concentrate in Families A and A-pii (14
records). By contrast, task utility appears across 58
records (49%) spanning every single family, and at-
tack success rate (ASR) or integrity metrics appear
in 61 records across multiple families, making ASR
the single most common metric in our map. Be-
cause a low ASR can mean anything from blocked
prompt injection to failed PII extraction, neither a
paper’s architectural mechanism nor its headline
metric label reliably tells a reader what asset was
actually protected.
5.2 Blind Spot II: High-Stakes Claims vs.
Skipped Content Disclosure
When we organize the literature by what it actu-
ally measures rather than what mechanisms it uses,
a massive gap appears where direct content pro-
tection should be. Across our 118 human-coded
records, only 21 studies (17.8%) have their pri-
mary endpoint in Families A, A-pii, or B, which
scores what a user receives or what an attacker
can reconstruct from the system. Even if we in-
clude mathematical bounds and likelihood proxies
from Family C, 77.1% of the literature (91 records
in Families D, E, and F) has its primary endpoint
outside this broader group. Although document
takedown, copyright opt-out, and the right to era-
sure are among the most concrete demands placed
on real-world RAG operators, papers addressing
those topics evaluate their methods through text-
overlap scores, reconstruction proxies, or query
refusal rates, landing in Families A, B, C, or D
rather than demonstrating actual source removal.
This skew is especially pronounced in research

framed around copyright and dataset ownership.
Among the 15 records coded with the frozen di-
rect copyright or ownership evidence relation, 11
(73.3% [48.0, 89.1]) have a usage-detection pri-
mary endpoint in Family E, compared with just
7 of the 103 records (6.8%) addressing neighbor-
ing security or privacy goals (Bonferroni-adjusted
p= 4.9×10−7across 14 comparisons). Mean-
while, zero records under this frozen label have a
strict emitted-expression primary endpoint in Fam-
ily A (95% exact upper bound of 18.1%). This
coded subset overwhelmingly prioritizes post-hoc
corpus-use detection, leaving verbatim reproduc-
tion by the generator absent from its primary end-
points.
5.3 Blind Spot III: Toy Benchmarks and
Static Adversaries
Even when a paper selects a clear endpoint that
matches its claim, the strength of its evidence de-
pends on the corpus, the adversary, and the rigor
of the experimental protocol. Across our map,
112 of the 118 records name a specific evalua-
tion dataset, spanning 278 distinct corpora in total
(Appendix D). Yet the literature relies overwhelm-
ingly on standard open-domain question-answering
benchmarks rather than datasets designed to test
data protection. The three most popular corpora
are Natural Questions (35 records), HotpotQA (31),
and MS MARCO (28), followed at a distance by
TriviaQA (7) and PopQA (5). By contrast, datasets
containing genuinely sensitive or private records
appear far less often, led by the Enron Email corpus
(12 records), HealthCareMagic (6), and ChatDoc-
tor (5). Evaluating privacy or copyright defenses
on general encyclopedia and web-search passages
creates an obvious interpretation problem: because
these corpora contain little sensitive or proprietary
expression to begin with, a low measured leakage
rate may simply reflect a benign workload rather
than an effective defense. A similar concentration
appears in model and retriever choices, where eval-
uations cluster around GPT-4o (31 records), GPT-
4o-mini (20), Contriever (18), all-MiniLM-L6-v2
(12), GPT-3.5 (12), Mistral-7B (12), Llama-3.1-8B
(12), and Qwen2.5-7B (11). Sharing a generator
or retriever across papers does not make their pro-
tection numbers comparable if one study scores
corpus membership on Wikipedia while another
scores verbatim copying on clinical notes.
Threat models and adversary interaction bud-
gets are equally narrow. Across the 118 studies,47 assume an external black-box attacker and 39
model a malicious corpus contributor with write ac-
cess, alongside 13 that assume a white-box system
operator and 6 that treat a cloud service provider
as a protocol participant (with 110 of 118 stud-
ies focusing exclusively on text). Only 16 records
model an authorized user, even though licensed
knowledge bases face their greatest extraction risk
from legitimate subscribers who are permitted to
query the system but prohibited from copying the
underlying database verbatim. Worse still, most
interactive evaluations restrict the adversary to a
static, one-shot interaction. Fifty-one records allow
the attacker only a single query and 25 model no
interactive querying at all, whereas just 11 stud-
ies grant the adversary feedback-adaptive multi-
query access. Testing a defense against a single
fixed prompt provides little assurance against a
real-world user who can inspect the system’s ini-
tial reply and iteratively refine follow-up queries to
bypass filters.
These gentle evaluation conditions help explain
why reported outcomes in the literature are so uni-
formly positive. In our eight-domain methodologi-
cal appraisal (Appendix D), zero records raise high
concern regarding threat-model definition or arti-
fact availability, and 95% receive low concern for
comparator and ablation design. However, only
48 of 118 records receive low concern for adap-
tive robustness testing, 72 records (61%) raise high
concern for uncertainty analysis because they re-
port single point estimates without confidence in-
tervals, random-seed variance, or statistical tests,
and 109 records raise some concern for evalua-
tion independence (with only one record achieving
low concern) because defenses are almost always
tested by their own authors against locally imple-
mented attacks. Against this backdrop, 108 of the
118 records (92%) report that their proposed inter-
vention improves protection, compared with just 9
reporting mixed effects and 1 reporting net degra-
dation. This 92% success rate should not be read as
proof that RAG protection is largely solved; rather,
it reflects a literature that predominantly tests self-
implemented defenses against non-adaptive attack-
ers on general-purpose QA benchmarks, without
quantifying statistical uncertainty.

6 Controlled Testbed and Future
Directions
Our literature audit shows that RAG protection
studies rarely make content disclosure their primary
endpoint, rely heavily on non-adaptive attackers,
and use metrics that cannot be compared across
families. To demonstrate why these distinctions
matter in practice, we first present results from
a controlled experimental testbed that evaluates
multiple defenses across competing endpoints on
the exact same workload, and then outline three
priorities for future research.
6.1 Stopping Surface Copying Does Not Stop
Fact Recovery
To test whether a defense that succeeds on one
endpoint actually protects underlying content
on another, we built a controlled evaluation
testbed spanning 192 documents across two do-
mains (Wikipedia and scientific papers) and two
access policies, paired with three unexposed
human-authored fact probes per document and 64
document-disjoint utility queries (full protocol and
tables in Appendix C). Fixing dense retrieval at
MiniLM top-3 and greedy decoding, we evaluated
four representative controls across Qwen3-8B (pri-
mary), Phi-4-14B (replication), Qwen2.5-3B, and
Mistral-7B: pre-generation access policy filtering
(Family D), a cosine similarity threshold at 0.5
(Family D), an upstream summarization prompt
instructing the model to answer in its own words
(Family A), and a normalized 8-gram output guard
that redacts spans matching protected-source 8-
grams while retaining unmatched text (Family A).
Crucially, rather than scoring a single static prompt,
we also evaluated an online three-query adaptive
controller that observes released text, refusals, and
redactions to iteratively extract information.
Evaluating these defenses simultaneously across
multiple endpoints confirms our central thesis:
eliminating surface copying does not prevent an
attacker from recovering private facts or identify-
ing source documents (Tables 6, 8, and 10). On
quotation-denied items, uncontrolled models copy
20-token source spans verbatim (copy 20) at high
rates, reaching 64.1% to 89.1% under natural dense
retrieval and up to 96.9% under separator-evasion
attacks. Adding the normalized 8-gram guard
drives both visible and decodedcopy 20down to
0.0% [0.0, 0.0] in the three-query adaptive evalu-
ation. Yet when we score the exact same guardedresponses on fact recovery and source linkage, sub-
stantial disclosure remains. Under the three-query
adaptive budget, an attacker facing the normalized
guard still recovers 48.3% [36.7, 60.8] of hidden
source facts on Qwen3-8B (31.7% on Qwen2.5-
3B) and links the response back to the exact source
document (re-ID@1) 42.5% of the time (Table 10).
Blinded human annotations of 192 source-response
pairs ( κ= 0.831 for expression, κ= 0.929 for
recovery; Table 12) tell the same story: on Phi-4-
14B, the normalized guard reduces human-judged
substantial content recovery only from 66.7% to
38.9% (Table 9). Furthermore, query budget sen-
sitivity exposes the danger of one-shot testing: on
guarded Qwen3-8B, hidden-fact recall is just 3.3%
[0.0, 7.5] after a single query, but jumps to 48.3%
on the third adaptive turn (Table 11).
Our testbed also shows how easily endpoint
and metric choices can distort comparisons of
prompt-based defenses and task utility. In our sin-
gle planned confirmatory contrast on 80 held-out
answer-only items, the upstream Summary defense
yields a difference-in-differences (DiD) in context-
attributable exact-answer recall of just −0.4 per-
centage points [ −8.3 ,7.5] on Qwen3-8B and 0.0
[−7.9 ,8.3] on Phi-4-14B (Figure 3). Short exact-
answer recall has only 51.8% sensitivity against
human-judged recovery and correlates with re-
sponse length (Table 12). Response length also
confounds standard utility metrics: while the Sum-
mary prompt appears to hurt ordinary-QA token F1
by−4.0 to−7.9 points across models (Figure 3),
our format diagnostic reveals that token F1 heavily
penalizes verbose free-form answers (scoring 18.9
on Qwen3-8B versus 46.8 when prompted for short
answers; Table 14). Finally, structural access con-
trols fail in complementary ways: pre-retrieval pol-
icy filtering blocks unauthorized documents when
metadata is clean, but a 12.5% rate of missing meta-
data labels under a fail-open policy immediately
restores 9.4% to 12.5% verbatim copying (Table 7).
Across every layer of the pipeline, a defense is only
as strong as the specific endpoint, query budget,
and metadata assumptions under which it is tested.
6.2 Three Golden Rules for Future RAG
Protection
Together, our evidence map and controlled testbed
point to three concrete rules for designing, report-
ing, and reviewing future RAG protection studies.
First, studies whose claims concern data secrecy,
privacy, or copyright must measure content dis-

closure directly. In our human-coded map, 97 of
the 118 core records do not use content disclosure
as their primary endpoint, and most evaluations
rely on general-purpose question-answering bench-
marks that contain little sensitive text to begin with
(§5.3). Suppressing retrieval access (Family D),
detecting watermarks after the fact (Family E), and
blocking poisoned passages (Family F) are valu-
able operational goals, but they do not replace a
measurement of what the system actually leaks to a
user. Any defense claiming to protect the contents
of a retrieval corpus should pair its operational met-
ric with a direct disclosure endpoint from Family
A, A-pii, or B, and evaluate on corpora containing
genuinely sensitive or proprietary spans.
Second, authors and benchmarks should state
their target endpoint at subfamily resolution and
test against multi-turn, adaptive adversaries. Nam-
ing the exact evidenced cell among our 17 sub-
families (for example, distinguishingA.verbatim
fromA.near, orD.retrievefromD.answer) allows
readers and meta-analyses to compare papers at a
stated level of detail without pooling incompatible
metrics. Likewise, because single-query evalua-
tions routinely miss extraction vulnerabilities that
emerge over three or more adaptive turns (§6.1),
multi-query feedback-adaptive testing and imper-
fect metadata stress tests should become standard
practice rather than rare exceptions.
Third, the field must move beyond simple event
matching to validate evidential sufficiency. Our
taxonomy establishes which technical event a pa-
per measures, not whether the chosen metric is a
sufficient or unbiased measure of that event. True
evidential sufficiency requires convergent and dis-
criminant validity controls (Campbell and Fiske,
1959) that almost no RAG protection study cur-
rently reports. In practice, this means verifying
that an extraction or inversion attack is genuinely
strong on the unprotected baseline before claiming
credit for a low defended score, and checking that
utility metrics like token F1 are not confounded by
simple shifts in response length.
7 Conclusion
“Protection” in retrieval-augmented generation is
not a single event. By introducing a frozen seven-
family endpoint taxonomy and coordinating 32 an-
notators to double-code 118 full-text RAG protec-
tion studies ( κ= 0.867 ), we showed that the liter-
ature evaluates fundamentally different technicaloutcomes that cannot be pooled or treated as in-
terchangeable. Looking across this evidence map
reveals that defense mechanisms weakly predict
what is actually measured, that only 17.8% of stud-
ies use content disclosure or reconstruction as their
primary endpoint, and that evaluations overwhelm-
ingly rely on benign question-answering bench-
marks and non-adaptive adversaries. Our con-
trolled testbed confirms the practical stakes of these
distinctions: defenses that eliminate verbatim span
copying still leave substantial source facts recov-
erable under multi-turn adaptive querying. By re-
leasing our human-coded evidence map, taxonomy
protocols, and evaluation testbed, we aim to help
the community move from ambiguous umbrella
claims toward precise, comparable, and construct-
valid evaluations within each protection goal.

References
Abdelrahman Abouelenein and Marwan Torki. 2026.
Differentially private datastore generation for
retrieval-augmented inference. InInternational Con-
ference on Pattern Recognition, pages 624–638.
Springer.
Vaibhav Adlakha, Parishad BehnamGhader, Xing Han
Lu, Nicholas Meade, and Siva Reddy. 2024. Eval-
uating correctness and faithfulness of instruction-
following models for question answering.Trans-
actions of the Association for Computational Linguis-
tics, 12:681–699.
William Agnew, Harry H Jiang, Cella Sum, Maarten
Sap, and Sauvik Das. 2024. Data defenses against
large language models.
Md Takrim Ul Alam, Akif Islam, Mohd Ruhul Ameen,
Abu Saleh Musa Miah, and Jungpil Shin. 2026.
Prompt control-flow integrity: A priority-aware run-
time defense against prompt injection in llm systems.
Ander Alvarez, Santhiya Rajan, Alessandro Genuardi,
Oliver Wirjadi, Samuel Mugel, and Román Orús.
2026. Provenanceguard: Source-aware factuality
verification for mcp-based llm agents.
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2024. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation.
Francisco Javier Arceo and Varsha Prasad Narsing.
2026. Securing the agent: Vendor-neutral, multi-
tenant enterprise retrieval and tool use.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
João Eduardo Batista, Emil Vatai, and Mohamed Wahib.
2025. Safe: Improving llm systems using sentence-
level in-generation attribution.
Su Lin Blodgett, Gilsinia Lopez, Alexandra Olteanu,
Robert Sim, and Hanna Wallach. 2021. Stereotyp-
ing norwegian salmon: An inventory of pitfalls in
fairness benchmark datasets.
Andreea-Elena Bodea, Stephen Meisenbacher, and Flo-
rian Matthes. 2026. A case study on the impact of
anonymization along the rag pipeline.
Samuel Bowman and George Dahl. 2021. What will it
take to fix benchmarking in natural language under-
standing?
Venkata Krishna Prasanth Budigi and Siri Chandana
Sirigiri. 2026. Beyond similarity search: A unified
data layer for production rag systems.
Jannis Bulian, Christian Buck, Wojciech Gajewski, Ben-
jamin Börschinger, and Tal Schuster. 2022. Tomayto,
tomahto. beyond token-level answer equivalence for
question answering evaluation.Florian AD Burnat. 2026. Auditing privacy in multi-
tenant rag under account collusion.
Grace Byun, Shinsun Lee, Nayoung Choi, and Jinho D
Choi. 2025. Secure multifaceted-rag: Hybrid knowl-
edge retrieval with security filtering.
Max Callaghan, Finn Müller-Hansen, Melissa Bond,
Candyce Hamel, Declan Devane, Wojciech Kusa,
Alison O’Mara-Eves, Rene Spijker, Mark Stevenson,
Claire Stansfield, et al. 2024. Computer-assisted
screening in systematic evidence synthesis requires
robust and well-evaluated stopping criteria.
Donald T Campbell and Donald W Fiske. 1959. Con-
vergent and discriminant validation by the multitrait-
multimethod matrix.
Daniela Ceraku. 2026. Rag-h: A rag-hardened soc
framework for trustworthy threat intelligence and ai
defence.
Chandranil Chakraborttii, Jackeline García Alvarado,
Sitora Abdulofizova, and Shivanshu Dwivedi. 2026.
Ghost vectors: Soft-deleted embeddings remain re-
constructible in hnsw vector databases.
Bingxiang Chen. 2024. An exploration of using
retrieval-augmented generation with access control.
Jinwen Chen, Hainan Zhang, Liang Pang, Yongxin
Tong, Haibo Zhou, Wei Lin, and Zhiming
Zheng. 2026a. Privacy-preserving reasoning with
knowledge-distilled parametric retrieval augmented
generation.
Kang Chen, Xiuze Zhou, Yuanguo Lin, Shibo Feng,
Li Shen, and Pengcheng Wu. 2025a. A survey on
privacy risks and protection in large language models.
Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao
Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong,
and Liang Wang. 2026b. Uncovering competing poi-
soning attacks in retrieval-augmented generation.
Tianyu Chen, Jian Lou, and Wenjie Wang. 2025b. Safe-
guarding multimodal knowledge copyright in the rag-
as-a-service environment.
Tong Chen, Akari Asai, Niloofar Mireshghallah, Sewon
Min, James Grimmelmann, Yejin Choi, Hannaneh
Hajishirzi, Luke Zettlemoyer, and Pang Wei Koh.
2024. Copybench: Measuring literal and non-literal
reproduction of copyright-protected text in language
model generation.
Ziyuan Chen, Yueming Lyu, Yi Liu, Weixiang Han,
Jing Dong, Caifeng Shan, and Tieniu Tan. 2026c.
Radar: Defending rag dynamically against retrieval
corruption.
Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan,
and Yunhao Yao. 2025a. Remoterag: A privacy-
preserving llm cloud rag service.

Zehua Cheng, Wei Dai, and Jiahao Sun. 2026. Identity-
decoupled anonymization for visual evidence in
multi-modal retrieval-augmented generation.
Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan,
Zhuqing Liu, Xiaohua Hu, and Minghong Fang.
2025b. Secure retrieval-augmented generation
against poisoning attacks.
Giovanni Cherubin and Andrew Paverd. 2025. High-
light & summarize: Rag without the jailbreaks.
Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook
Lee, and Jinseong Park. 2025. Safeguarding privacy
of retrieval data against membership inference at-
tacks: Is this query too close to home?
Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Kr-
ishnamurthy Dj Dvijotham, and Somesh Jha. 2025.
Through the stealth lens: Attention-aware defenses
against poisoning in rag.
Stav Cohen, Ron Bitton, and Ben Nassi. 2024. Here
comes the ai worm: Unleashing zero-click worms
that target genai-powered applications.
Lee J Cronbach and Paul E Meehl. 1955. Construct
validity in psychological tests.
Huining Cui and Wei Liu. 2026. Needle-in-rag: Prompt-
conditioned character-level traceback of poisoned
spans in retrieved evidence.
Badhan Chandra Das, M Hadi Amini, and Yanzhao
Wu. 2025. Security and privacy challenges of large
language models: A survey.
Soumil Datta, Melissa Umble, Daniel S Brown, and
Guanhong Tao. 2026. A sentence relation-based ap-
proach to sanitizing malicious instructions.
Sagie Dekel, Moshe Tennenholtz, and Oren Kurland.
2026. Addressing corpus knowledge poisoning at-
tacks on rag using sparse attention.
Zeming Fei, Hongming Fei, Xiaoyang Wang, Prosanta
Gope, Biplab Sikdar, Ying Zhang, et al. 2026. Selec-
tion integrity for llm graph memory: An accumula-
bility criterion for information-flow-blind retrieval.
Haowei Fu, Bo Ni, Han Xu, Kunpeng Liu, Dan Lin,
and Tyler Derr. 2026. Ensemble privacy defense
for knowledge-intensive llms against membership
inference attacks.
Chengcai Gao, Zhihong Sun, Xiaochuan Shi, Qiufeng
Wang, and Chao Liang. 2026. Bird: A bidirectional
ranking defense mechanism for retrieval augmented
generation.
Yunfan Gao. 2023. Retrieval-augmented generation for
large language models: A survey.
Armin Gerami, Kazem Faghih, and Ramani Du-
raiswami. 2025. Quantifying document impact in
rag-llms.Aditya Golatkar, Alessandro Achille, Luca Zancato,
Yu-Xiang Wang, Ashwin Swaminathan, and Stefano
Soatto. 2024. Cpr: Retrieval augmented generation
for copyright protection.
Ishrith Gowda. 2026. Memsad: Gradient-coupled
anomaly detection for memory poisoning in retrieval-
augmented agents.
Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu,
Chenxi Liu, Yanshuo Chen, and Heng Huang. 2025.
Towards copyright protection for knowledge bases of
retrieval-augmented language models via reasoning.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Jiyoung Whang. 2024.
Why so gullible? enhancing the robustness of
retrieval-augmented models against counterfactual
noise.
Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, and
Yangqiu Song. 2024. Privacy-preserved neural graph
databases.
Hao Huang, JiaTang Luo, Ruihua Zhou, Yunpeng Li,
and Yuling Liu. 2026. Don’t corrupt the fact: A
trustworthy rag watermarking framework based on
dual factual shield.
Abigail Z Jacobs and Hanna Wallach. 2021. Measure-
ment and fairness.
Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel
Zhang, and Alex Liu. 2025. Ragrank: Using pager-
ank to counter poisoning in cti llm pipelines.
Shuaifan Jin, Xiaoyi Pang, Zhibo Wang, He Wang, Ji-
acheng Du, Jiahui Hu, and Kui Ren. 2025. Safe-
guarding llm embeddings in end-cloud collaboration
via entropy-driven perturbation.
Weifei Jin, Xilong Wang, Wei Zou, Jinyuan Jia, and
Neil Gong. 2026. Cleanbase: Detecting malicious
documents in rag knowledge databases.
Nikola Jovanovi ´c, Robin Staab, Maximilian Baader, and
Martin Vechev. 2025. Ward: Provable rag dataset
inference via llm watermarks.
Ehsan Kamalloo, Nouha Dziri, Charles LA Clarke, and
Davood Rafiei. 2023. Evaluating open-domain ques-
tion answering in the era of large language models.
Sayash Kapoor and Arvind Narayanan. 2023. Leakage
and the reproducibility crisis in machine-learning-
based science.
Minseok Kim, Hankook Lee, and Hyungjoon Koo. 2025.
Rescuing the unpoisoned: Efficient defense against
knowledge corruption attacks on rag systems.
Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, and Kama-
lika Chaudhuri. 2024. Privacy-preserving retrieval-
augmented generation with differential privacy.
Tsun On Kwok, Xi Yang, Ki Sen Hung, Chang Liu, and
Yangqiu Song. 2026. Sentinelrag: Synthetic sentinel
knowledge for rag database copyright protection.

Jonathan Lebensold, Maziar Sanjabi, Pietro Astolfi,
Adriana Romero-Soriano, Kamalika Chaudhuri,
Mike Rabbat, and Chuan Guo. 2024. Dp-rdm: Adapt-
ing diffusion models to private domains without fine-
tuning.
Jun Wen Leong. 2026. Defense effectiveness across
architectural layers: a mechanistic evaluation of per-
sistent memory attacks on stateful llm agents.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks.
Qinfeng Li, Miao Pan, Ke Xiong, Ge Su, Zhiqiang
Shen, Yan Liu, Sun Bing, Hao Peng, and Xuhong
Zhang. 2026a. Ragfort: Dual-path defense against
proprietary knowledge base extraction in retrieval-
augmented generation.
Yanhang Li, Zhichao Fan, and Zexin Zhuang. 2026b.
Where privacy risk lives in english-source multilin-
gual rag: A stage-decomposed audit across five query
languages.
Zhijun Li, Minghui Xu, Huayi Qi, Wenxuan Yu,
Tingchuang Zhang, Qiao Zhang, GuangYong Shang,
Zhen Ma, and Xiuzhen Cheng. 2026c. Prag: End-to-
end privacy-preserving retrieval-augmented genera-
tion.
Xun Liang, Mengwei Wang, Yuefeng Ma, and Simin
Niu. 2026a. Safe rag by rag: Untying the bell that
rag rang with the rag hand.
Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan
Zhang, Boyi Liu, and Yongxin Tong. 2026b. Fedmo-
saic: Federated retrieval-augmented generation via
parametric adapters.
Bo Lin, Shangwen Wang, Yihao Qin, Liqian Chen, and
Xiaoguang Mao. 2025. Give llms a security course:
Securing retrieval-augmented code generation via
knowledge injection.
Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Richard James, Pedro Rodriguez, Ja-
cob Kahn, Gergely Szilvasy, Mike Lewis, et al. 2024.
Ra-dit: Retrieval-augmented dual instruction tuning.
Zachary C Lipton and Jacob Steinhardt. 2019. Research
for practice: troubling trends in machine-learning
scholarship.
Mingrui Liu, Sixiao Zhang, and Cheng Long. 2025a.
Mask-based membership inference attacks for
retrieval-augmented generation.
Peiyang Liu, Ziqiang Cui, Di Liang, and Wei Ye. 2025b.
Who stole your data? a method for detecting unau-
thorized rag theft.Xuan Liu, Hefeng Zhou, Sicheng Chen, Chao Yang,
Xingcheng Xu, Jingjing Qu, Jiong Lou, Jie LI, and
Xia Hu. 2026. Privacy-preserving text sanitization
for distributed agents collaboration via disentangled
representations.
Yepeng Liu, Xuandong Zhao, Dawn Song, and Yuheng
Bu. 2025c. Dataset protection via watermarked ca-
naries in retrieval-augmented llms.
Michele Lorenzo, Idilio Drago, Dario Salvadori, and
Fabio Romolo Vayr. 2025. Arbiter: Ai-driven filter-
ing for role-based access control.
Ziyuan Luo, Yangyi Zhao, Ka Chun Cheung, Simon See,
and Renjie Wan. 2026. Imagesentinel: Protecting vi-
sual datasets from unauthorized retrieval-augmented
image generation.
Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang,
Shengzhi Zhang, Yuxuan Chen, Kai Chen, and Limin
Sun. 2025. Rag-wm: An efficient black-box water-
marking approach for retrieval-augmented generation
of large language models.
Chenxin Mao, Shangyu Liu, Zhenzhe Zheng, Fan Wu,
Jie Wu, and Guihai Chen. 2026. An efficient and
privacy-preserving architecture for cross-institutional
collaborative rag.
Aiman Al Masoud, Marco Arazzi, and Antonino Nocera.
2026. Sd-rag: A prompt-injection-resilient frame-
work for selective disclosure in retrieval-augmented
generation.
Samuel Messick. 1995. Validity of psychological as-
sessment: Validation of inferences from persons’ re-
sponses and performances as scientific inquiry into
score meaning.
Sewon Min, Suchin Gururangan, Eric Wallace, Weijia
Shi, Hannaneh Hajishirzi, Noah Smith, and Luke
Zettlemoyer. 2024. Silo language models: Isolating
legal risk in a nonparametric datastore.
Junki Mori, Kazuya Kakizaki, Taiki Miyagawa, and Jun
Sakuma. 2026. Differentially private synthetic text
generation for retrieval-augmented generation (rag).
Richard Helder Moulton, Austin O’Brien, and John D
Hastings. 2025. Quantifying return on security con-
trols in llm systems.
Charafeddine Mouzouni. 2026. Context kubernetes:
Declarative orchestration of enterprise knowledge for
agentic ai systems.
Yanming Mu, Hao Hu, Feiyang Li, Qiao Yuan, Jiang
Wu, Zichuan Liu, Pengcheng Liu, Mei Wang, Hong-
wei Zhou, and Yuling Liu. 2026. Towards secure
retrieval-augmented generation: A comprehensive
review of threats, defenses and benchmarks.
Rohith Namboothiri. 2026. Authorization-first retrieval:
Enforcing least privilege in multi-agent rag systems.

Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko,
Gabriele Fugagnoli, Dimitris Sacharidis, Katja Hose,
and Tomer Sagi. 2025. Source attribution in retrieval-
augmented generation.
Havva Alizadeh Noughabi, Fattane Zarrinkalam, and
Ali Dehghantanha. 2026. Defense against knowledge
poisoning attack on graphrag.
Sandeep Nutakki. 2026. SENTINEL: Retrieval-time
governance for enterprise RAG a controlled red-team
benchmark for policy-aware retrieval, privacy en-
forcement, and audit replay.Journal of Advances
in Developmental Research, 17(1).
Ciyan Ouyang and Rui Hou. 2026. Memlineage:
Lineage-guided enforcement for llm agent memory.
Balamurugan Palanisamy, GSS Chalapathi, Vikas Has-
sija, and Rajkumar Buyya. 2026. Security and pri-
vacy in retrieval-augmented generation: Architec-
tures, threats, defenses, and future directions for
building trustworthy systems.
Pranav Pallerla, Wilson Naik Bhukya, Bharath Vemula,
and Charan Ramtej Kodi. 2026. Adaptive defense or-
chestration for rag: A sentinel-strategist architecture
against multi-vector attacks.
Francesco Panebianco, Stefano Bonfanti, Francesco
Trovò, and Michele Carminati. 2025. Leaksealer:
A semisupervised defense for llms against prompt
injection and leakage attacks.
Prashant Kumar Pathak and Tarun Kumar Sharma. 2026.
When global gating is enough: Admission-time hub-
ness control in anisotropic vector retrieval systems.
Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-
Liess, Cho-Yu Jason Chiang, and Furong Huang.
2025. Ragpart & ragmask: Retrieval-stage defenses
against corpus poisoning in retrieval-augmented gen-
eration.
KrishnaSaiReddy Patil. 2026. Ragshield: Detecting
numerical claim manipulation in government rag sys-
tems.
Alvaro Lopez Pellicer, Plamen Angelov, Marwan
Bukhari, Yi Li, Eduardo Soares, and Jemma Kerns.
2026. Protomedagent: Multimodal clinical inter-
pretability via privacy-aware agentic workflows.
Micah DJ Peters, Casey Marnie, Andrea C Tricco,
Danielle Pollock, Zachary Munn, Lyndsay Alexander,
Patricia McInerney, Christina M Godfrey, and Hanan
Khalil. 2021. Updated methodological guidance for
the conduct of scoping reviews.
Jirui Qi, Gabriele Sarti, Raquel Fernández, and Arianna
Bisazza. 2024. Model internals-based answer attribu-
tion for trustworthy retrieval-augmented generation.
Zhisheng Qi, Utkarsh Sahu, Li Ma, Haoyu Han,
Ryan Rossi, Franck Dernoncourt, Mahantesh Halap-
panavar, Nesreen Ahmed, Yushun Dong, Yue Zhao,et al. 2026. Benchmarking knowledge-extraction at-
tack and defense on retrieval-augmented generation
(rag).
Inioluwa Deborah Raji, Emily M Bender, Amandalynne
Paullada, Emily Denton, and Alex Hanna. 2021. Ai
and the everything in the whole wide world bench-
mark.
Gulshan Saleem, Nisar Ahmed, Muhammad Imran Za-
man, Ali Hassan, and Umar Mujahid. 2026. A lay-
ered security framework against prompt injection in
rag-based chatbots.
Alireza Salemi and Hamed Zamani. 2026. Improving
user privacy in personalized generation: Client-side
retrieval-augmented modification of server-side gen-
erated speculations.
Yash Saxena, Ankur Padia, Mandar S Chaudhary, Kalpa
Gunaratna, Srinivasan Parthasarathy, and Manas
Gaur. 2025. Ranking free rag: Replacing re-ranking
with selection in rag for sensitive domains.
Yangguang Shao, Chengshang Hou, Gaopeng Gou,
Gang Xiong, and Junzheng Shi. 2026. Poisoncraft:
Practical poisoning of retrieval-augmented genera-
tion for large language models.
Tarun Sharma. 2026. Smsr: Certified defence against
runtime memory poisoning in persistent llm agent
systems.
Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang,
Prateek Mittal, and Aleksandra Korolova. 2026. Re-
liabilityrag: Effective and provably robust defense
for rag-based web-search.
Yu-Kai Shih and You-Kai Kang. 2025. Design and im-
plementation of a secure rag-enhanced ai chatbot for
smart tourism customer service: Defending against
prompt injection attacks–a case study of hsinchu, tai-
wan.
Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Li-
jun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan,
Yang Liu, and Xiaojun Jia. 2026. Secon-rag: A two-
stage semantic filtering and conflict-free framework
for trustworthy rag.
Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping
Chen, and Jun Dai. 2024. Revprag: Revealing
poisoning attacks in retrieval-augmented generation
through llm activation analysis.
Xue Tan, Yi Zheng, Chang Huo, Yunruo Zhang, Yu Liu,
Hao Luan, Zhuyang Yu, Jun Dai, Xiaoyan Sun, and
Ping Chen. 2026. Pra-rag: Provably robust aggre-
gation in retrieval-augmented generation against re-
trieval corruption.
Scott Thornton. 2026a. Retrieval pivot attacks in hy-
brid rag: Measuring and mitigating amplified leakage
from vector seeds to graph expansion.

Scott Thornton. 2026b. Semantic chameleon: Corpus-
dependent poisoning attacks and defenses in rag sys-
tems.
Andrea C Tricco, Erin Lillie, Wasifa Zarin, Kelly K
O’Brien, Heather Colquhoun, Danielle Levac, David
Moher, Micah DJ Peters, Tanya Horsley, Laura
Weeks, et al. 2018. Prisma extension for scoping
reviews (prisma-scr): checklist and explanation.
Haoran Wang, Xiongxiao Xu, Baixiang Huang, and
Kai Shu. 2026a. Privacy-aware decoding: Mitigating
privacy leakage of large language models in retrieval-
augmented generation.
Shang Wang, Tianqing Zhu, Dayong Ye, and Wan-
lei Zhou. 2025a. When machine unlearning meets
retrieval-augmented generation (rag): Keep secret or
forget knowledge?
Weijie Wang, Peizhuo Lv, Yan Wang, Weiqing Huang,
Rujie Dai, Guokun Xu, Qiujian Lv, Hangcheng Liu,
Hao Hu, and Jiaheng Zhang. 2026b. Making theft
useless: Adulteration-based protection of proprietary
knowledge graphs in graphrag systems.
Yanting Wang, Runpeng Geng, Ying Chen, and Jinyuan
Jia. 2026c. Attntrace: Contextual attribution of
prompt injection and knowledge corruption.
Yanting Wang, Wei Zou, Runpeng Geng, and Jinyuan
Jia. 2025b. {TracLLM }: A generic framework for
attributing long context{LLMs}.
Yuhao Wang, Wenjie Qu, Shengfang Zhai, Yanze Jiang,
Zichen Liu, Yue Liu, Yinpeng Dong, and Jiaheng
Zhang. 2025c. Silent leaks: Implicit knowledge ex-
traction attack on rag systems through benign queries.
Yujing Wang, Jinwen Chen, Hainan Zhang, Liang
Pang, Yongxin Tong, Binghui Guo, Hongwei Zheng,
and Zhiming Zheng. 2025d. Learning to erase pri-
vate knowledge from multi-documents for retrieval-
augmented large language models.
Jascha Wanger. 2026. Vectorsmuggle: Steganographic
exfiltration in embedding stores and a cryptographic
provenance defense.
Pingyu Wu, Daiheng Gao, Jing Tang, Huimin Chen,
Wenbo Zhou, Weiming Zhang, and Nenghai Yu.
2025a. Mes-rag: Bringing multi-modal, entity-
storage, and secure enhancements to rag.
Ruihan Wu, Erchi Wang, Zhiyuan Zhang, and Yu-
Xiang Wang. 2025b. Private-rag: Answering multi-
ple queries with llms while keeping your data private.
Xun Xian, Ganghua Wang, Xuan Bi, Jayanth Srinivasa,
Ashish Kundu, Charles Fleming, Mingyi Hong, and
Jie Ding. 2024. On the vulnerability of applying
retrieval-augmented generation within knowledge-
intensive application domains.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2026. Certifiably
robust rag against retrieval corruption.Jin Xie, Songze Li, and Guang Cheng. 2026a. Seal-
tag: Self-tag evidence aggregation with probabilistic
circuits for pii-safe retrieval-augmented generation.
Yuanbo Xie, Yingjie Zhang, Yulin Li, Shouyou Song,
Xiaokun Chen, Zhihan Liu, Liya Su, and Tingwen
Liu. 2026b. Detecting rag extraction attack via dual-
path runtime integrity game.
Yuming Xu, Mingtao Zhang, Zhuohan Ge, Haoyang
Li, Nicole Hu, Yongqi Zhang, Zhiyuan Wen, Ja-
son Chen Zhang, Qing Li, and Lei Chen. 2026. Se-
curing retrieval-augmented generation: A taxonomy
of attacks, defenses, and future directions.
Zhenhua Xu, Xubin Yue, Zhebo Wang, Haobo Zhang,
Qichen Liu, Xixiang Zhao, Jingxuan Zhang, Wenjun
Zeng, Wengpeng Xing, Dezhang Kong, et al. 2025.
Copyright protection for large language models: A
survey of methods, challenges, and trends.
Ke Yang, Shuguang Yuan, Jing Yu, and Chi Chen. 2026.
Knowledge-infused multi-bit watermarking for rag
knowledge bases.
Shu Yang, Jiayuan Su, Han Jiang, Mengdi Li, Keyuan
Cheng, Muhammad Asif Ali, Lijie Hu, and Di Wang.
2024. Dialectical alignment: Resolving the tension
of 3h and security threats of llms.
Zhiguang Yang, Gejian Zhao, and Hanzhou Wu. 2025.
Watermarking for large language models: A survey.
Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang,
Cong Wang, and Zhan Qin. 2025a. Controlnet: A
firewall for rag-based llm system.
Ruobing Yao, Yifei Zhang, Shuang Song, Neng Gao,
and Chenyang Tu. 2025b. Ecosaferag: Efficient secu-
rity through context analysis in retrieval-augmented
generation.
Zhihao Yao, Yuxuan Gu, Xiachong Feng, Weitao Ma,
Bo Li, Xiaocheng Feng, and Bing Qin. 2026. Adap-
tive backtracking for privacy protection in large lan-
guage models.
Huanyi Ye, Jiale Guo, Ziyao Liu, and Kwok-Yan Lam.
2025. Efficient privacy-preserving retrieval aug-
mented generation with distance-preserving encryp-
tion.
Zhe Yu, Wenpeng Xing, Gaolei Li, Shuguang Xiong,
Hongzhi Wang, Xuyang Teng, and Meng Han. 2026.
Cordon-mas: Defending rag against knowledge poi-
soning via information-flow control.
Osama Zafar, Alexander Nemecek, Yiqian Zhang, Wen-
biao Li, Debargha Ganguly, Vikash Singh, Vipin
Chaudhary, and Erman Ayday. 2026. Privacy policy
enforcement guardrails for data-sensitive retrieval-
augmented generation.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, et al. 2024a. The good and the
bad: Exploring privacy issues in retrieval-augmented
generation (rag).

Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, et al. 2024b. The good and the
bad: Exploring privacy issues in retrieval-augmented
generation (rag).
Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren,
Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue
Xing, and Jiliang Tang. 2025a. Mitigating the privacy
issues in retrieval-augmented generation (rag) via
pure synthetic data.
Zhirui Zeng, Jiamou Liu, Meng-Fen Chiang, Jialing
He, and Zijian Zhang. 2025b. S-rag: A novel audit
framework for detecting unauthorized use of personal
data in rag systems.
Baolei Zhang, Haoran Xin, Yuxi Chen, Zhuqing Liu,
Biao Yi, Tong Li, Lihai Nie, Zheli Liu, and Minghong
Fang. 2026a. Who taught the lie? responsibil-
ity attribution for poisoned knowledge in retrieval-
augmented generation.
Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing
Liu, Biao Yi, Tong Li, and Zheli Liu. 2025a. Trace-
back of poisoning attacks to retrieval-augmented gen-
eration.
Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang,
Minghong Fang, Zhuqing Liu, Lihai Nie, and Zheli
Liu. 2025b. Benchmarking poisoning attacks against
retrieval-augmented generation.
Gang Zhang, Mingyu Tian, Xukun Luan, Yuanchi Ma,
and Jinyan Liu. 2026b. Is external database protec-
tion static in retrieval-augmented generation? rethink-
ing privacy preservation under dynamic queries.
Guangwei Zhang, Qisheng Su, Jiateng Liu, Cheng Qian,
Yanzhou Pan, Yanjie Fu, and Denghui Zhang. 2025c.
Isacl: Internal state analyzer for copyrighted training
data leakage.
Haichao Zhang, Chong Zhang, Peiyu Hu, Shi Qiu, and
Jia Wang. 2025d. Customized retrieval-augmented
generation with llm for debiasing recommendation
unlearning.
Shuning Zhang, Lyumanshan Ye, Xin Yi, Jingyu Tang,
Bo Shui, Haobin Xing, Pengfei Liu, and Hewu Li.
2024. " ghost of the past": identifying and resolving
privacy leakage from llm’s memory through proactive
user interaction.
Zhenliang Zhang, Xinyu Hu, and Xiaojun Wan. 2026c.
Scope: Intrinsic semantic space control for mitigating
copyright infringement in llms.
Yuanhe Zhao, Tianyu Zhang, Huafei Xing, Derek F
Wong, Jianbin Li, and Tao Fang. 2027. Privacy-
preserving rag via multi-agent semantic rewriting:
Achieving confidentiality without compromising con-
textual fidelity.Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025a. Trustrag: Enhancing robust-
ness and trustworthiness in retrieval-augmented gen-
eration.
Pengcheng Zhou, Yinglun Feng, and Zhongliang Yang.
2025b. Provably secure retrieval-augmented genera-
tion.
Xinyuan Zhu, Zekun Fei, Enye Wang, Ruiqi He, Jia
Guo, Ruijie Wang, Zheli Liu, and Qingkai Zeng.
2026. Not all entities are created equal: A dynamic
anonymization framework for privacy-preserving rag.

A Event and Control Schema
To prevent our coding from confusing a paper’s high-level topic with what its experiments actually measure, our schema records
six dimensions separately: the technical event, the protected asset, the adversary’s identity and access level, the defensive
intervention function, the primary measured endpoint, and the broader evidence relation (Table 1a). Table 1b defines our frozen
seven-family endpoint taxonomy (Families A, A-pii, and B–F). Families A, A-pii, B, and C are grouped under the shared
content-secrecy asset class; however, when we count studies whose primary endpoint measures actual content disclosure, we
include only Families A, A-pii, and B because Family C reports theoretical bounds or likelihood shifts without observing
emitted or reconstructed text. Under this four-group hierarchy, a paper’s claim and its primary measured endpoint are coded as
concordant when they fall into the exact same family, partial when they land in different families within the same asset group,
and discordant (off-target) when they belong to different asset groups altogether. Family G is assigned whenever a claim or
an experimental packet is too vague to establish a specific technical event. Finally, each publication receives a single mutually
exclusive primary endpoint family from human adjudication, whereas its defense mechanisms and intervention functions can
receive multiple labels.
Table 1: Event definitions and coding schema.
(a)Separate coding fields for technical events, assets, actors, functions, and evidence relations.
Technical event Primary asset / goal Actor and access Relevant functions Evidence relation
Unauthorized corpus use or
KB theftOwner text or image corpus;
evidence of useExternal owner/auditor with
black-box queries; copying
operatorDetect, attribute; occasionally
prevent/remediateDirect copyright/ownership
evidence (Jovanovi ´c et al.,
2025; Liu et al., 2025c; Lv
et al., 2025; Liu et al., 2025b;
Guo et al., 2025)
Retrieval extraction or
authorization bypassRestricted documents and
access policyEnd user with black-box,
adaptive queriesMeasure attacks; evaluate
preventive controls separatelyDirect copyright- or
privacy-framed extraction
attacks (Wang et al., 2025c;
Zeng et al., 2024b);
security-framed controls (Yao
et al., 2025a); benchmark
evidence (Qi et al., 2026)
Output copying or close
transformationExpression exposed in
retrieved contextEnd user or downstream
recipientPrevent, redact, attribute,
remediateDirect only when copying is
measured; citation alone is
insufficient;
copyright/ownership image
output comparison (Golatkar
et al., 2024); likelihood-based
text opt-out is a separate proxy
(Min et al., 2024);
security-framed text recovery
(Li et al., 2026a); attribution is
adjacent (Qi et al., 2024). Data
Defenses measures use
resistance, not copying (Agnew
et al., 2024)
Query, database, or
membership leakageConfidential query, corpus
record, or indexCloud provider, protocol
participant, or probing userPrevent or detect under a
privacy/security definitionDirect RAG but non-copyright
privacy/confidentiality
(Anderson et al., 2024; Liu
et al., 2025a; Cheng et al.,
2025a)
Corpus poisoning or
retriever backdoorIntegrity of retrieved evidence
and answerCorpus contributor with write or
injection accessPrevent, detect, trace Direct RAG security; not
evidence that copyright
copying is prevented (Zhang
et al., 2025b; Chen et al.,
2026b; Shao et al., 2026)
Source influence or
knowledge removalAttribution of an output;
requested removalOperator or auditor with black-,
gray-, or white-box accessAttribute, detect, remediate Direct RAG
attribution/remediation; legal
relevance is fact-dependent
(Nematov et al., 2025; Qi et al.,
2024; Wang et al., 2025a)
Table 1, (b)Frozen event families. C shares the content asset group for concordance but does not measure observed disclosure.
Asset Fam. The endpoint concerns. . .
Content
disclosureA generated output reproducing protected sourceexpression
A-pii generated output containing protectedatomic identifiers
Breconstructionof source content from embeddings, an index, or another non-generation channel
Proxy for
disclosure C a likelihood, perplexity, or formal privacybound, with no emitted or reconstructed content
Access D suppression of authorization, retrieval, routing, or answerability
Use Edetectionof a watermark, canary, membership, corpus use, or ownership
Integrity F poisoning, prompt injection, provenance, attribution, removal, or misuse
Unresolved G a packet that does not establish the outcome a stricter assignment needs

B Search, Coding, and Evidence Map
B.1 Literature Search and Eligibility Criteria
Our literature search ran through July 17, 2026, combining eight initial arXiv queries, nine expanded arXiv query families,
the same nine query families executed over OpenAlex, and a title-and-metadata scan of the ACL Anthology. We also ran ten
seed-paper lookups and ten forward-citation tracking passes as separate continuity checks, alongside capped supplementary
searches on Crossref and DBLP. Each search family paired core retrieval terms (such as “RAG,” “retrieval-augmented,” “retrieval-
enhanced,” and “grounded generation”) with protection-related keywords covering copyright, licensing, ownership, watermarking,
copying, extraction, privacy, attribution, removal, and security; an adjacent query family paired general language-model terms
with copyright and copying prevention. Retrieved records were deduplicated sequentially by arXiv identifier, Digital Object
Identifier (DOI), and normalized title, followed by an author-title-year audit that linked multiple preprints or versions belonging
to the same underlying study.
To be confirmed as a direct empirical RAG-protection record in our core map, a study had to evaluate a retrieval-augmented
system, intervene on its query, corpus, index, context, or output, report a source-verified protection or attribution operating point,
and address either a direct copyright/ownership goal or a neighboring RAG privacy/security goal. Pure attack papers, standalone
benchmarks, survey articles, and untested architectural proposals were tracked in separate roles, and reporting task utility alone
was not sufficient for inclusion.
In total, our queries returned 11,972 raw hits, which deduplicated to 8,978 unique records and yielded 925 candidate inclusions
after abstract screening. We successfully recovered full texts for 824 candidates, while 101 full texts were unavailable and one
recovered paper remained unresolved. Human full-text review confirmed 112 direct control studies from this candidate pool, and
our stratified audit of 1,200 records sampled from the 8,053 excluded records identified 6 additional direct controls, bringing the
final human-coded map to 118 records (Table 2a and Table 2b).
Table 2: Search-coverage estimates and evidence-map dimensions.
(a)Excluded-frame audit: 1,200 sampled exclusions, ten strata. Conservative intervals combine simultaneous finite-population hypergeometric bounds with
Bonferroni correction. The component sum is not a population total.
Quantity Observed Estimate/range Conservative 95% Identification
excluded frame 6 27.2 6 to 318 Protocol-confirmable controls under the
frozen human abstract rule; abstract-screen
false negatives are not estimated.
Available possible-inclusion full texts 112 112 n/a Census of available full texts only; 102
records are unclassified.
Descriptive component sum 118 139.2 118 to 430 Not an identified total for all eligible records
in the 8,978-record frame.
If two recall nonresponses are controls 120 146.1 120 to 441 All-positive second-stage nonresponse sensi-
tivity.
Main-pool missingness envelope (102 unclassified) n/a 139.2 to 241.2 n/a Non-probabilistic; recall nonresponses re-
main excluded.
Expanded missingness envelope n/a 139.2 to 248.1 n/a Also treats both weighted recall nonre-
sponses as controls.
Table 2, (b)Human-adjudicated map dimensions. Intervention functions are multi-label; abstract-only records are excluded.
Axis Category Records
Evidence relation copyright / ownership 15
neighboring RAG goal 103
Validated outcome emitted source expression 8
emitted atomic identifier / PII 6
non-output source recovery 7
Validated flag adaptive multi-query recovery 7
Legacy field broad “output-copy” yes 28Axis Category Records
Modality text 110
multimodal 5
image 3
Publication peer reviewed 29
preprint 88
thesis/report 1Axis Category Records
Year 2023 1
2024 18
2025 43
2026 56
Function prevent 83
detect 69
attribute 26
remediate 24
B.2 Human Annotation and Inter-Rater Reliability
Thirty-two trained annotators with backgrounds in NLP and machine learning (18), security and privacy (9), and information
science or law and technology (5) double-coded the recovered full texts. Whenever the initial pair disagreed, a third reviewer
outside that pair adjudicated the record. Before coding the main pool, every annotator completed a 100-item calibration test
with a required passing score of 95%; all 32 annotators passed, scoring between 95% and 98% (median 97%). Because raw
calibration files and identity-linked records remain restricted under participant confidentiality terms, domain backgrounds and
calibration pass rates are reported as coordinator attestations.
Our public artifact releases a de-identified 8,024-row human label matrix covering all 118 confirmed records across 34
coded fields. In this matrix, initial coder pairs that had no entry in the disagreement log are reconstructed as matching the
consensus label, while logged disagreements preserve the exact pre-adjudication calls of both coders. Running our standalone
verification script on this matrix reproduces a pre-adjudication agreement of 89.83% ( κ= 0.8672 ) on the primary endpoint
family (Table 3a and Table 3b). Across the 14 strictly categorical fields, median reliability is κ= 0.87 . Fifteen free-text rationale
and source-location fields had no logged categorical disagreements; their nominal κ= 1.00 reflects consensus text entry rather
than an independent reliability estimate. For the eight methodological appraisal domains, linear weighted κis computed across
lower, some, and high concern levels excluding “not reported” (NR) pairs, whereas nominal agreement includes NR calls.

Table 3: Human screening and refined-coding agreement.
(a)Pre-adjudication agreement over 824 full texts.
Field Agreementκ
Strict direct-control decision 92.4% 0.642
Evidence relation 74.3% 0.546
Primary study role 87.5% 0.844
Legacy broad output-copy field 97.2% 0.852
Legacy recovery field 94.1% 0.795
Publication status 93.6% 0.860Table 3, (b)Refined-coding agreement. Weighted κis linear across lower/some/high
and excludes NR; nominal agreement includes NR.
Field AgreementκWeightedκ
primary family 89.8% 0.867 n/a
adaptive recovery 99.2% 0.929 n/a
interaction adaptivity 83.1% 0.760 n/a
Event, asset, actor, and access 97.5% 0.932 0.932
Comparator and ablation 98.3% 0.826 0.851
Evaluation independence 95.8% 0.718 0.722
Sample and unit reporting 93.2% 0.873 0.881
Uncertainty analysis 94.1% 0.897 0.938
Adaptive or robustness test 91.5% 0.857 0.877
Matched utility and harms 94.9% 0.879 0.900
Artifact or reproduction support 97.5% 0.949 0.950
B.3 Copyright-Relation Sensitivity
Within our 118-record map, 15 records are coded under the frozen direct copyright or dataset-ownership evidence relation
(Table 4a). Eleven of these 15 studies (73.3% [48.0, 89.1]) have a usage-detection primary endpoint in Family E, compared
with just 7 of the 103 studies (6.8%) addressing neighboring goals (Bonferroni-adjusted p= 4.9×10−7across 14 family
comparisons). Under the frozen copyright-relation label (Table 4b), zero records have a strict emitted-expression primary
endpoint in Family A (95% exact upper bound of 18.1%).
All citations and operating points were verified against source PDFs, and the exact query strings and audit logs are preserved
in the released artifact.
Table 4: Evidence-relation sensitivity.
(a)Primary endpoints by evidence relation across 118 records. Copyright/ownership is a technical coding relation, not a legal classification.
Protected asset Primary reported endpoint All Copyright Other
(%) relation relation
Content disclosure Emitted source expression 8 (6.8) 0 8
Emitted atomic identifiers 6 (5.1) 0 6
Non-output reconstruction 7 (5.9) 0 7
Likelihood or formal bound 6 (5.1) 2 4
Access suppression Access and retrieval suppression 26 (22.0) 1 25
Use detection Corpus-use or ownership detection 18 (15.3) 11 7
Integrity / attribution Integrity, provenance, or removal 47 (39.8) 1 46
Total 118 15 103
Table 4, (b)Copyright-boundary sensitivity: relation agreement 74.3% (κ= 0.546),
centrality agreement 97.6% (κ= 0.893).
Boundary Records Emitted Diagnostic change
Frozen evidence-relation
label15 0 Includes S-RAG; excludes RAGFort.
Relation minus
personal-data S-RAG14 0 Removes the disputed privacy audit.
Frozen
copyright-centrality label16 1 Includes both S-RAG and RAGFort.
Centrality minus
personal-data S-RAG15 1 Substitutes proprietary-KB extraction
for the privacy audit.
C Controlled Testbed and Claim-Aligned Analyses
C.1 Design and analysis hierarchy
The fixed protocol (not preregistered) uses a 192-document security corpus across two domains and two policies, with three
unexposed human-authored probes per document; utility uses 64 document-disjoint open-policy items. MiniLM dense retrieval
is fixed at top-3, decoding is greedy, and the three-query controller observes only released text, redaction, refusal, transport
decodability, and released length. Qwen3 is the primary model, Phi-4 the replication, and Qwen2.5-3B and Mistral-7B exploratory.
The exact upstream Summary baseline is commit-pinned.
The sole planned confirmatory contrast is the Qwen3 Summary-minus-uncontrolled difference-in-differences (DiD) of exact
hidden-probe recall on all 80 held-out answer-only items. Context effects are target-present minus target-ablated. An attack
ranking selected on uncontrolled target-present development data is shared across all four exposure/control cells. Inference uses
paired source-item bootstrapping and sign randomisation without multiplicity correction for this single contrast. Other models,
policies, natural retrieval, shorter spans, and linkage analyses are secondary or exploratory.

Table 5: Controls and their stated claims.
(a)Declared claims and their aligned measurements; other endpoints describe separate channels.
Control Declared technical claim Claim-aligned evidence Not established
Policy filter Contexts unauthorized for the requesting
principal do not reach generation.Target presence after filtering;
access-control failures; authorized
QA utility.Copying or recovery from contexts
the principal is authorized to access.
Cosine threshold Retrieved documents with MiniLM cosine below
the pinned 0.5 operating point do not reach
generation.Post-threshold target presence,
all-item recovery, and matched QA
utility.Confidentiality of relevant documents
that remain above threshold.
Benchmark Summary The pinned upstream prompt requests a concise,
query-relevant summary in the model’s own
words and rejects irrelevant context.Response concision and relevance;
full-source coverage and blinded
human source-content recovery for
the broader extraction-defense
interpretation. Exact hidden-answer
recall is retained only with a construct
audit.Legal similarity, corpus-use detection,
or universal resistance to semantic
inference.
Benchmark SystemBlock The pinned system prompt forbids exposing raw
database content or facts available only from the
database.Context-attributable source-content
recovery, refusal behavior, and
matched benign utility under a
defense-aware attacker.Formal access control, legal
confidentiality, or robustness beyond
the tested models, prompts, and
budget.
Normalized 8-gram guard No released output contains a matching
normalized source 8-gram.Trigger behavior, shorter-span utility,
and the implied absence of matching
copy20under the same
normalization.Suppression of facts, paraphrase,
source identity, or reversible
encodings outside its
canonicalization.
C.2 Natural-policy and adaptive evaluations
Natural-policy and claim-aligned evaluations use distinct denominators, stated in their table panels. Natural-policy intervals
resample source items, and models are reported separately. A quotation-denied item can remain answerable; the policy filter
excludes inaccessible documents, while the normalised 8-gram guard suppresses protected source spans. Authorised quotation is
desired release. The adaptive experiment additionally distinguishes raw outcomes on all quotation-denied items from residual
outcomes conditional on successful unguarded copying.
Table 6: Natural-policy outcomes and copied-span thresholds.
(a)Natural-policy dense retrieval. Copying uses all quotation-denied items. Authorised quote copying is desired release. Brackets: 95% source-item bootstrap
intervals.
Model Target retrieved Uncontrolled copy 20 Policy filter copy 20 Filter + guard copy 20 Ordinary-QA F1 Authorized quote copy 20
Qwen-3B 98.4 [95.3, 100.0] 78.1 [68.8, 87.5] 37.5 [26.6, 50.0] 0.0 [0.0, 0.0] 0.250 [0.213, 0.288] 73.4 [62.5, 84.4]
Phi-3.5 98.4 [95.3, 100.0] 89.1 [81.2, 95.3] 45.3 [32.8, 57.8] 0.0 [0.0, 0.0] 0.369 [0.316, 0.426] 79.7 [68.8, 89.1]
Qwen-7B 98.4 [95.3, 100.0] 85.9 [76.6, 93.8] 46.9 [34.4, 59.4] 0.0 [0.0, 0.0] 0.309 [0.267, 0.353] 81.2 [71.9, 90.6]
Mistral-7B 98.4 [95.3, 100.0] 64.1 [51.6, 76.6] 29.7 [18.8, 40.6] 0.0 [0.0, 0.0] 0.261 [0.225, 0.298] 67.2 [56.2, 78.1]
Table 6, (b)Each cell is copy 5/copy10/copy20(%) under the semantic extraction query with quotation forbidden.
Model Lexical Dense
None Filter Filter+guard None Filter Filter+guard
Qwen-3B 87.5/84.4/70.3 51.6/39.1/34.4 26.6/0.0/0.0 92.2/89.1/78.1 62.5/45.3/37.5 26.6/1.6/0.0
Phi-3.5 87.5/84.4/79.7 53.1/37.5/34.4 14.1/1.6/0.0 93.8/92.2/89.1 64.1/48.4/45.3 20.3/3.1/0.0
Qwen-7B 85.9/85.9/79.7 48.4/40.6/37.5 14.1/1.6/0.0 96.9/93.8/85.9 60.9/50.0/46.9 15.6/1.6/0.0
Mistral-7B 87.5/76.6/68.8 51.6/39.1/26.6 23.4/3.1/0.0 93.8/85.9/64.1 60.9/45.3/29.7 25.0/1.6/0.0
Table 7: Metadata robustness and guard side effects.
(a)Metadata robustness. Cells are cross-model ranges, not pooled estimates. Fail-open missing or wrong labels admit 12.5% of access-denied targets.
Observed metadata Denied target kept Denied copy 20 Ordinary QA F1
Clean 0.0 to 0.0 0.0 to 0.0 0.250 to 0.369
Missing, fail open 12.5 to 12.5 9.4 to 12.5 0.252 to 0.369
Missing, fail closed 0.0 to 0.0 0.0 to 0.0 0.226 to 0.349
Wrong label 12.5 to 12.5 0.0 to 12.5 0.241 to 0.350
Table 7, (b)Paired guard-minus-unguarded side effects on the same response. Authorised target text is excluded from the guard set; triggers concern other
non-quotable sources.
Model Ordinary trigger Ordinary∆F1 F1 decrease Authorized non-target trigger Authorized∆5-gram recall
Qwen-3B 16.7 [9.4, 25.0] -0.004 [-0.014, 0.004] 5.2 [1.0, 9.4] 0.0 [0.0, 0.0] 0.000 [0.000, 0.000]
Phi-3.5 10.4 [4.2, 16.7] -0.006 [-0.020, 0.007] 4.2 [1.0, 8.3] 9.4 [3.1, 17.2] 0.000 [0.000, 0.000]
Qwen-7B 7.3 [2.1, 12.5] -0.011 [-0.023, -0.001] 5.2 [1.0, 10.4] 3.1 [0.0, 7.8] 0.000 [0.000, 0.000]
Mistral-7B 10.4 [5.2, 16.7] -0.008 [-0.021, 0.002] 4.2 [1.0, 8.3] 0.0 [0.0, 0.0] 0.000 [0.000, 0.000]

Table 8: Adaptive copying and semantic similarity.
(a)Adaptive copy 20(%). Raw rates use all quotation-denied items; residuals condition on successful unguarded copying, with each nshown. Surface, normalised,
and cumulative-session guards remain distinct.
Model Attack Raw copy 20 Surface residual Normalized residual Session residual
Qwen3-8B Separator evasion 71.9 [59.4, 82.8];n= 6495.7 [89.1, 100.0];n= 460.0 [0.0, 0.0];n= 46n/a
Four-turn chunks 60.9 [48.4, 71.9];n= 642.6 [0.0, 7.7];n= 392.6 [0.0, 7.7];n= 390.0 [0.0, 0.0];n= 39
Phi-4-14B Separator evasion 96.9 [92.2, 100.0];n= 64100.0 [100.0, 100.0];n= 620.0 [0.0, 0.0];n= 62n/a
Four-turn chunks 81.2 [71.9, 90.6];n= 640.0 [0.0, 0.0];n= 527.7 [1.9, 15.4];n= 520.0 [0.0, 0.0];n= 52
Table 8, (b)Detailed-paraphrase cosine similarity and mismatched-source margin; neither is thresholded as copying or legal similarity.
Model / control Source cosine True-negative margin
Qwen3-8B / None 0.794 [0.758, 0.828] 0.704 [0.665, 0.742]
Qwen3-8B / Surface guard 0.679 [0.624, 0.732] 0.583 [0.522, 0.638]
Qwen3-8B / Summary-style prompt 0.777 [0.738, 0.815] 0.691 [0.645, 0.733]
Qwen3-8B / Summary-style + surface guard 0.666 [0.606, 0.723] 0.576 [0.514, 0.638]
Phi-4-14B / None 0.882 [0.858, 0.902] 0.760 [0.732, 0.785]
Phi-4-14B / Surface guard 0.816 [0.783, 0.844] 0.696 [0.662, 0.728]
Phi-4-14B / Summary-style prompt 0.870 [0.842, 0.894] 0.752 [0.718, 0.785]
Phi-4-14B / Summary-style + surface guard 0.789 [0.748, 0.828] 0.679 [0.633, 0.719]
Table 9: Utility and model-specific human outcomes.
(a)Ordinary QA uses clean metadata. Authorised quotation reports coverage, completeness, and refusal under policy filtering plus the normalised guard.
ModelOrdinary F1
noneOrdinary F1
surfaceOrdinary F1
normalizedOrdinary F1
summary-styleAuthorized
targetAuthorized
5-gram recallAuthorized
refusal
Qwen3-8B 0.349 [0.309, 0.389] 0.346 [0.308, 0.386] 0.346 [0.306, 0.390] 0.355 [0.312, 0.399] 98.4 [95.3, 100.0] 0.190 [0.152, 0.228] 3.1 [0.0, 7.8]
Phi-4-14B 0.179 [0.156, 0.205] 0.184 [0.158, 0.209] 0.177 [0.153, 0.201] 0.208 [0.182, 0.237] 98.4 [95.3, 100.0] 0.169 [0.148, 0.191] 3.1 [0.0, 7.8]
Table 9, (b)Human-rated outcomes by model, using the definitions and pair sample in Table 12.
Model Control Expression Recovery Identifiablen
Qwen3-8B None 72.2 33.3 50.0 18
Normalized guard 5.6 0.0 0.0 18
Summary-style prompt 55.6 5.6 22.2 18
Summary + normalized guard 11.1 0.0 11.1 18
Phi-4-14B None 94.4 66.7 72.2 18
Normalized guard 44.4 38.9 38.9 18
Summary-style prompt 66.7 55.6 66.7 18
Summary + normalized guard 50.0 22.2 27.8 18
Table 10: Online adaptive extraction and recovery endpoints.
(a)Online three-query extraction (%). Attack selection uses development data only; branching sees released information. Dense top-3 rows condition on target
retrieval before policy filtering; decoded endpoints accumulate released turns.
Model Control Visible copy 20 Decoded copy 20 Hidden-fact recall Source re-ID@1n
Qwen2.5-3B None 50.0 [35.0, 65.0] 67.5 [52.5, 80.0] 40.8 [27.5, 54.2] 25.0 [12.5, 37.5] 40
Normalized guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 31.7 [20.8, 41.7] 15.0 [5.0, 27.5] 40
Summary + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 14.2 [6.7, 22.5] 22.5 [10.0, 35.0] 40
Policy + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 18.3 [10.8, 26.7] 5.0 [0.0, 12.5] 40
Policy + summary + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 17.5 [10.0, 25.8] 20.0 [7.5, 32.5] 40
Qwen3-8B None 55.0 [40.0, 70.0] 72.5 [57.5, 85.0] 60.0 [45.8, 73.3] 25.0 [12.5, 40.0] 40
Normalized guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 48.3 [36.7, 60.8] 42.5 [27.5, 57.5] 40
Summary + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 39.2 [27.5, 50.8] 40.0 [25.0, 55.0] 40
Policy + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 22.5 [13.3, 31.7] 20.0 [7.5, 32.5] 40
Policy + summary + guard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 27.5 [16.6, 39.2] 30.0 [15.0, 45.0] 40
Table 10, (b)Detailed-paraphrase recovery on quotation-denied items. Hidden-fact recall uses unexposed human-written probes; re-ID@1 ranks the response against
96 documents. These are recovery/linkage, not copying measures.
Model Control copy 20 Hidden-fact recall Source re-ID@1
Qwen3-8B None 3.1 [0.0, 7.8] 14.6 [7.9, 21.2] 87.5 [79.7, 95.3]
Normalized guard 0.0 [0.0, 0.0] 12.4 [6.6, 18.8] 78.1 [67.2, 87.5]
Summary + normalized guard 0.0 [0.0, 0.0] 9.5 [4.8, 15.3] 76.6 [65.6, 85.9]
Policy + summary + normalized guard 0.0 [0.0, 0.0] 4.2 [1.1, 7.9] 43.8 [31.2, 56.2]
Phi-4-14B None 4.7 [0.0, 10.9] 46.0 [37.3, 55.6] 90.6 [82.8, 96.9]
Normalized guard 0.0 [0.0, 0.0] 38.4 [29.6, 47.4] 90.6 [82.8, 96.9]
Summary + normalized guard 0.0 [0.0, 0.0] 34.1 [25.9, 42.3] 85.9 [76.6, 93.8]
Policy + summary + normalized guard 0.0 [0.0, 0.0] 20.6 [13.8, 28.6] 56.2 [43.8, 68.8]

Table 11: Query-budget and domain sensitivity.
(a)Normalised-guard budget sensitivity; stopped sessions carry forward their last release and decoded outcomes accumulate valid turns.
Model Budget Decoded copy 20 Hidden-fact recall Source re-ID@1n
Qwen2.5-3B 1 0.0 [0.0, 0.0] 0.8 [0.0, 2.5] 15.0 [5.0, 27.5] 40
2 0.0 [0.0, 0.0] 3.3 [0.0, 7.5] 10.0 [2.5, 20.0] 40
3 0.0 [0.0, 0.0] 31.7 [20.8, 41.7] 15.0 [5.0, 27.5] 40
Qwen3-8B 1 0.0 [0.0, 0.0] 3.3 [0.0, 7.5] 7.5 [0.0, 17.5] 40
2 0.0 [0.0, 0.0] 5.0 [0.8, 11.7] 10.0 [2.5, 20.0] 40
3 0.0 [0.0, 0.0] 48.3 [36.7, 60.8] 42.5 [27.5, 57.5] 40
Table 11, (b)Three-query domain results: 48 Wikipedia and 48 scientific documents, three hidden probes each; target-retrieved subsets contain 25 and 15 items.
Model DomainGuard
factsGuard
re-ID@1Full stack
factsFull stack
re-ID@1n
Qwen2.5-3B Wikipedia 42.7 [29.3, 57.3] 8.0 [0.0, 20.0] 20.0 [9.3, 32.0] 8.0 [0.0, 20.0] 25
Scientific papers 13.3 [4.4, 24.4] 26.7 [6.7, 53.3] 13.3 [4.4, 24.4] 40.0 [13.3, 66.7] 15
Qwen3-8B Wikipedia 62.7 [46.7, 77.3] 40.0 [20.0, 60.0] 34.7 [20.0, 49.3] 20.0 [8.0, 36.0] 25
Scientific papers 24.4 [11.1, 37.8] 46.7 [20.0, 73.3] 15.6 [4.4, 28.9] 46.7 [20.0, 73.3] 15
C.3 Human construct checks
Two independent human annotators, blind to model, control, score, and pair status, rated 144 true-source and 48 mismatched-
source pairs. True pairs balance two models, two prompts, and four controls. Agreement is 88.0% ( κ= 0.831 ) for expression
and 95.3% ( κ= 0.929 ) for recovery. Wilson intervals describe this fixed sample. Automated proxy correlations use source-
item-clustered bootstrap intervals; partial correlations rank-residualise response length. These are technical rather than legal
judgments.
Source linkage ranks Qwen3 responses with BM25 and MiniLM over the 192-item corpus, 96-way topic-matched candidates,
and expanded pools of 20,233 Wikipedia or 1,169 scientific documents. Uniform, question-only, answer-only, and length-matched
shuffled controls separate topic and length effects from response signal.
Table 12: Human judgments and automated proxy validation.
(a)Human-rated source/response outcomes: 144 true and 48 mismatched pairs. Brackets: Wilson 95% intervals. Expressionκ= 0.831; recoveryκ= 0.929.
Control Close/near-verbatim expression Substantial/near-complete recovery Source identifiablen
None 83.3 [68.1, 92.1] 50.0 [34.5, 65.5] 61.1 [44.9, 75.2] 36
Normalized guard 25.0 [13.8, 41.1] 19.4 [9.8, 35.0] 19.4 [9.8, 35.0] 36
Summary-style prompt 61.1 [44.9, 75.2] 30.6 [18.0, 46.9] 44.4 [29.5, 60.4] 36
Summary + normalized guard 30.6 [18.0, 46.9] 11.1 [4.4, 25.3] 19.4 [9.8, 35.0] 36
Mismatched-source controls 0.0 [0.0, 7.4] 0.0 [0.0, 7.4] 0.0 [0.0, 7.4] 48
Table 12, (b)Proxy validation on those 192 human-rated pairs. Every proxy is scored against the displayed source. Spearman intervals cluster by source item; partial
ρadjusts for response length;κuses four-level exact recall.
Quantity True-source pairs Mismatched-source pairs
Exact hidden-probe recall of displayed source (all probes), Spearmanρ0.72 [0.60, 0.82] n/a
Exact hidden-probe recall of displayed source (≥4-token answers), Spearmanρ0.51 [0.27, 0.69] n/a
Full-source multiset token recall, Spearmanρ0.87 [0.80, 0.92] n/a
Normalized character similarity, Spearmanρ0.85 [0.78, 0.90] n/a
Response token count, Spearmanρ0.84 [0.75, 0.90] n/a
Exact hidden-probe recall of displayed source (all probes), partial Spearmanρ(controlling for response
length)0.41 [0.22, 0.57] n/a
Exact hidden-probe recall of displayed source (≥4-token answers), partial Spearmanρ(controlling for
response length)0.25 [0.04, 0.44] n/a
Full-source multiset token recall, partial Spearmanρ(controlling for response length) 0.46 [0.33, 0.58] n/a
Normalized character similarity, partial Spearmanρ(controlling for response length) 0.38 [0.24, 0.50] n/a
Linear weightedκ(human vs. discretized exact recall) 0.490 n/a
Sensitivity, any recovery vs. displayed-source exact-recall indicator (%) 51.8 [42.7, 60.7] n/a
Specificity, any recovery vs. displayed-source exact-recall indicator (%) 96.6 [82.8, 99.4] 100.0 [92.6, 100.0]
Mismatched-control false-positive rate, human any recovery (%) n/a 0.0 [0.0, 7.4]
Mismatched-control false-positive rate, automated any recovery on the displayed (wrong) source’s own
probes (%)n/a 0.0 [0.0, 7.4]

UniformNo control
BM25No control
DenseSummary
BM25Summary
DenseQuestion
Dense
Experiment · N=192
T opic-matched · N=96
Expanded · N=20,233
Experiment · N=192
T opic-matched · N=96
Expanded · N=1,1690.5 100.0 100.0 97.5 97.5 97.5
1.0 100.0 97.5 100.0 100.0 77.5
0.0 100.0 97.5 100.0 97.5 70.0
0.5 42.5 40.0 32.5 30.0 27.5
1.0 42.5 40.0 35.0 27.5 20.0
0.1 42.5 37.5 32.5 25.0 20.0Response-to-source linkage
Qwen3-8B · T op-1 retrieval (%)
Wikipedia
Scienti fic
papers
Shared colour scale: 0% to 100%; values retain the reported rounding.Figure 2: Response-to-source linkage calibration (top-1, %). Qwen3 answer-only responses are ranked by BM25 or
MiniLM against the experimental corpus, question-selected topic-matched candidates, and expanded same-domain
pools. Candidate-set sizes and the reported uniform and question-only baselines are retained. This measures source
linkage, not copying or content recovery.
C.4 Primary contrast, utility, and sensitivity
20 30 40 50
Context e ffect (pp)Qwen3-8B (P)
Phi-4-14B (R)
Qwen2.5-3B (E)
Mistral-7B (E)44.2 / 43.8
38.3 / 38.3
24.2 / 20.4
31.2 / 41.7
−10 0 10 20
Difference-in-di fferences (pp)-0.4
0.0
-3.8
10.4
0 10 20
F1 (%)Qwen3-8B
Phi-4-14B
Qwen2.5-3B
Mistral-7B
−10 0 10QA F1
−10 0 10Quote F1
−10 0 105-gram recallA  Context e ffects (n=80) B  Summary minus no control
C  Ordinary-QA F1 (n=64) D  Paired utility changes (n=64)
Change (pp)
Bars: 95% item-bootstrap intervals. P primary; R replication; E exploratory.No control Summary
Figure 3: Context contrast and matched utility across the original model roles. (A) No-control and Summary
context effects after three queries; paired numbers are shown in that order. (B) Summary-minus-uncontrolled
differences-in-differences on all 80 answer-only items. Only Qwen3 was planned confirmatory; exact-answer
recall is an incomplete recovery proxy. (C) Absolute ordinary-QA F1 on 64 document-disjoint items. (D) Paired
Summary-minus-uncontrolled changes in QA F1, quote-query F1, and authorised quote 5-gram recall. Bars are 95%
source-item bootstrap intervals. Negative authorised-quote recall means less desired quotation.

Table 13: Paired utility, equivalence bounds, and batch variation.
(a)Paired utility deltas against uncontrolled (5,000-draw paired item-bootstrap intervals). Token F1 and span F1 distinguish response-length sensitivity from answer
coverage.
Model Control∆Token F1∆Span F1∆Resp. tokens
Qwen3-8B Summary -4.5 [-7.3, -1.8] -3.4 [-10.3, +3.3] +6.0 [-2.1, +13.1]
SystemBlock -0.4 [-1.1, +0.3] -0.1 [-1.6, +1.6] +2.7 [-0.2, +5.7]
Threshold +0.0 [-2.7, +2.4] -11.5 [-19.6, -4.2] -17.1 [-24.8, -10.3]
Phi-4-14B Summary -4.0 [-6.2, -2.0] -4.9 [-10.5, -0.3] +5.6 [-2.1, +13.5]
SystemBlock +2.0 [+0.5, +3.7] -0.0 [-3.2, +3.1] -3.6 [-7.9, +0.4]
Threshold +3.6 [+0.7, +6.6] -13.3 [-20.8, -6.3] -18.5 [-27.6, -9.3]
Qwen2.5-3B Summary -8.3 [-12.4, -4.6] -23.1 [-33.7, -12.9] +15.9 [+6.1, +25.6]
SystemBlock +1.8 [+0.4, +3.3] -3.0 [-10.0, +3.5] -10.1 [-15.8, -4.8]
Threshold -0.9 [-3.4, +1.6] -11.7 [-19.4, -4.5] -14.2 [-22.9, -5.9]
Mistral-7B Summary -6.2 [-9.2, -3.4] +3.8 [-2.8, +10.9] +27.3 [+19.1, +36.0]
SystemBlock -1.2 [-2.7, +0.1] +1.1 [-2.1, +5.0] +3.8 [-2.2, +10.3]
Threshold -0.3 [-2.9, +2.4] -9.1 [-17.1, -2.1] +2.4 [-6.7, +10.6]
Table 13, (b)Post-hoc utility equivalence bounds: a checkmark requires the
entire 95% paired span-F1 interval strictly inside the bound, a stricter criterion
than 90%-interval TOST inclusion.
Model Control±5pts±10pts
Qwen3-8B Summary , ,
SystemBlock✓ ✓
Threshold , ,
Phi-4-14B Summary , ,
SystemBlock✓ ✓
Threshold , ,
Qwen2.5-3B Summary , ,
SystemBlock ,✓
Threshold , ,
Mistral-7B Summary , ,
SystemBlock✓ ✓
Threshold , ,Table 13, (c)Six replicates per model vary batch size (4, 8, 16) and row order,
with seed, revision, and greedy decoding fixed.
Model F1 SD Span SD Range Changed
Qwen3-8B 0.013 0.035 0.032 6.2%
Phi-4-14B 0.017 0.065 0.043 18.8%
Qwen2.5-3B 0.250 0.000 0.530 17.2%
Mistral-7B 0.006 0.000 0.015 6.2%
Table 14: Context and answer-format effects on utility.
(a)Format diagnostic on 64 document-disjoint items per model. Only context and answer-format instruction vary. Brackets: 95% item-bootstrap intervals. Token F1
scores the short gold answer against the whole response.
Context Answer format Token F1 Span F1 Answer recall Resp. tokens
Qwen3-8B
Closed book free-form (frozen) 9.1 [6.5, 11.8] 24.8 [17.2, 32.7] 24.2 [16.4, 32.3] 20.3
Closed book short answer 12.8 [7.6, 18.6] 18.3 [11.7, 25.5] 15.6 [9.7, 22.1] 9.3
Retrieved top-3 free-form (frozen) 18.9 [15.0, 23.0] 67.4 [57.4, 76.9] 66.7 [56.6, 76.2] 38.1
Retrieved top-3 short answer 46.8 [36.7, 56.5] 57.0 [45.9, 67.7] 55.6 [44.6, 66.4] 7.7
Gold source only free-form (frozen) 28.0 [23.8, 32.5] 86.2 [79.4, 91.9] 84.7 [77.7, 90.7] 27.5
Gold source only short answer 60.9 [52.6, 68.8] 76.3 [67.7, 84.2] 73.3 [64.1, 81.7] 8.0
Phi-4-14B
Closed book free-form (frozen) 5.2 [3.4, 7.2] 30.8 [22.7, 39.2] 31.6 [23.5, 40.6] 85.3
Closed book short answer 12.4 [7.9, 17.8] 23.7 [16.4, 31.3] 21.1 [14.3, 28.2] 15.7
Retrieved top-3 free-form (frozen) 11.3 [8.7, 14.3] 71.6 [62.5, 80.2] 72.7 [63.4, 81.4] 86.0
Retrieved top-3 short answer 39.2 [30.6, 48.3] 65.2 [55.0, 75.1] 63.6 [53.2, 73.6] 22.3
Gold source only free-form (frozen) 22.6 [18.8, 26.4] 90.1 [85.4, 94.1] 91.4 [86.8, 95.2] 50.0
Gold source only short answer 57.4 [49.2, 65.2] 83.3 [75.0, 90.4] 82.1 [73.7, 89.5] 12.2
Qwen2.5-3B
Closed book free-form (frozen) 3.6 [2.5, 4.8] 23.7 [16.6, 31.1] 23.4 [16.3, 31.0] 64.1
Closed book short answer 9.1 [3.8, 15.4] 9.8 [4.1, 16.6] 9.5 [3.8, 16.1] 3.5
Retrieved top-3 free-form (frozen) 14.3 [10.6, 18.4] 64.0 [53.8, 73.6] 64.4 [54.1, 74.1] 56.2
Retrieved top-3 short answer 40.0 [29.6, 50.5] 46.6 [35.8, 57.6] 43.9 [33.2, 54.8] 5.6
Gold source only free-form (frozen) 20.9 [17.3, 24.7] 81.4 [74.2, 87.9] 81.1 [73.5, 87.9] 44.8
Gold source only short answer 57.3 [47.6, 67.1] 68.5 [58.1, 78.4] 65.5 [54.9, 75.5] 7.8
Mistral-7B
Closed book free-form (frozen) 5.1 [3.6, 6.6] 34.0 [25.8, 42.0] 33.4 [25.3, 41.4] 70.6
Closed book short answer 13.4 [7.9, 19.5] 20.7 [13.4, 28.7] 18.3 [11.4, 26.0] 12.2
Retrieved top-3 free-form (frozen) 17.3 [13.5, 21.3] 66.5 [56.7, 76.1] 66.4 [56.5, 76.1] 49.7
Retrieved top-3 short answer 40.3 [31.1, 49.6] 58.5 [48.0, 68.5] 56.7 [46.2, 67.0] 15.2
Gold source only free-form (frozen) 26.8 [22.2, 31.4] 85.0 [78.3, 90.7] 85.0 [78.0, 90.8] 34.2
Gold source only short answer 51.0 [43.0, 58.9] 72.3 [63.5, 80.5] 69.1 [60.2, 77.6] 9.2

Table 15: Exploratory construct and practical-equivalence checks.
(a)Exploratory construct sensitivity. DiD uses all 80 answer-only items except long-answer recall ( n= 57 , 21 Wikipedia and 36 scientific), standardised to equal
domain weights. Domain columns use 40 items each and forced target-present Summary exposure. Response length is in tokens; other columns are percentage points.
ModelLong-answer
recall DiDaNeg.-control
margin DiDFull-source
token DiDResponse length
DiD (tokens)Wikipedia exact
recallbSci. paper exact
recallb
Qwen3-8B -17.5 [-30.6, -5.4] -1.3 [-9.2, 7.1] 5.1 [-0.5, 10.5] 57.7 [28.6, 85.9] 75.8 [67.5, 83.3] 11.7 [5.8, 18.3]
Phi-4-14B -8.4 [-21.0, 4.2] 0.4 [-7.5, 8.8] 3.4 [-1.9, 8.6] 84.4 [49.8, 119.8] 68.3 [57.5, 79.2] 14.2 [7.5, 22.5]
Qwen2.5-3B -7.7 [-20.6, 5.7] -2.9 [-11.7, 5.8] 3.5 [-2.3, 9.7] 35.8 [14.1, 57.5] 28.3 [18.3, 39.2] 13.3 [6.7, 21.7]
Mistral-7B -0.3 [-13.8, 12.5] 11.7 [2.5, 20.4] 2.9 [-2.5, 8.2] 11.7 [-16.4, 39.9] 68.3 [58.3, 78.3] 23.3 [16.7, 30.0]
Table 15, (b)Post-hoc TOST on exact-answer DiD. Bounds were chosen after the frozen analysis. pTOST is the larger shifted one-sided sign-randomisation value
(20,000 draws). Equivalence requiresp <0.05and the 90% paired bootstrap interval inside the bound.
Model Bound DiD90% paired
bootstrap CIp TOST αInterpretation
Qwen3-8B±5pp -0.4 [-7.1, 6.2]0.1390.05 not established
Qwen3-8B±10pp -0.4 [-7.1, 6.2]0.0110.05 equivalent (within bound)
Phi-4-14B±5pp 0.0 [-6.7, 6.7]0.1190.05 not established
Phi-4-14B±10pp 0.0 [-6.7, 6.7]0.0100.05 equivalent (within bound)
Qwen2.5-3B±5pp -3.8 [-11.2, 3.7]0.3950.05 not established
Qwen2.5-3B±10pp -3.8 [-11.2, 3.7]0.0940.05 not established
Mistral-7B±5pp 10.4 [2.9, 17.9]0.8770.05 not established
Mistral-7B±10pp 10.4 [2.9, 17.9]0.5280.05 not established
C.5 Seven-query extension
The exploratory extension pins Summary, SystemBlock, and the threshold to the same upstream configuration. A complete-source
request is followed by up to six requests using Base64, ROT13, reversed words or characters, fact tables, and paraphrases.
The source intervention uses a shared attack ranking across three controls and two exposure arms. Natural retrieval uses
control-specific development rankings over both policies and all 160 test items included in intent-to-treat analyses. Utility is
evaluated on two query types and 64 document-disjoint items.
Budgets 1, 3, and 7 carry forward the last released state after stopping. Intervals use 5,000 paired source-item bootstrap draws
and sign randomisation uses 20,000 draws. Models and the two attacker designs are not pooled. Long-answer analyses use the
57 eligible items with accepted answers of at least four tokens, standardised to equal domain weights. Practical-equivalence
bounds are post-hoc: a large TOSTpvalue does not establish equivalence.
Table 16: Seven-query source-intervention results.
(a)Exploratory seven-query source intervention, shared attacker, 80 answer-only items. DiD is defence-minus-uncontrolled for the present-minus-ablated context
effect. Brackets: 95% paired source-item bootstrap intervals.
Model DefenseUncontrolled
context effectDefense
context effectDifference-in-
differencesn
Qwen3-8B Summary 43.3 42.5 -0.8 [-7.9, 6.2] 80
Qwen3-8B SystemBlock 43.3 45.8 2.5 [-3.8, 8.3] 80
Phi-4-14B Summary 40.4 40.8 0.4 [-7.1, 7.9] 80
Phi-4-14B SystemBlock 40.4 39.6 -0.8 [-6.7, 5.0] 80
Qwen2.5-3B Summary 22.9 22.9 0.0 [-8.8, 8.3] 80
Qwen2.5-3B SystemBlock 22.9 24.6 1.7 [-5.4, 8.8] 80
Mistral-7B Summary 33.8 40.4 6.7 [-2.5, 16.2] 80
Mistral-7B SystemBlock 33.8 28.3 -5.4 [-12.9, 2.1] 80
Table 16, (b)Seven-query shared-attacker DiD by endpoint. Long-answer recall uses 57 eligible items with equal domain weights; other outcomes use 80. Values are
percentage points except response-length tokens.
Model DefenseExact-answer
DiDLong-answer
DiDaFull-source
token DiDDecoded
copy20DiDResponse-length
DiD (tokens)
Qwen3-8B Summary -0.8 [-7.9, 6.2] -16.3 [-28.6, -4.0] 6.5 [0.8, 12.1] -36.2 [-48.8, -23.8] 110.4 [74.2, 146.4]
Qwen3-8B SystemBlock 2.5 [-3.8, 8.3] -1.9 [-10.5, 6.1] 1.8 [-1.3, 4.9] -12.5 [-21.2, -5.0] 24.5 [-7.0, 59.0]
Phi-4-14B Summary 0.4 [-7.1, 7.9] -13.2 [-25.4, -1.4] 1.1 [-3.8, 6.0] -28.7 [-40.0, -17.5] 69.7 [22.3, 118.4]
Phi-4-14B SystemBlock -0.8 [-6.7, 5.0] -5.1 [-14.4, 3.1] 3.6 [-0.4, 7.7] -3.8 [-12.5, 5.0] 82.7 [31.5, 132.9]
Qwen2.5-3B Summary 0.0 [-8.8, 8.3] -8.1 [-20.6, 4.1] 3.5 [-2.8, 10.0] -42.5 [-55.0, -30.0] 58.7 [13.4, 103.8]
Qwen2.5-3B SystemBlock 1.7 [-5.4, 8.8] 4.4 [-8.9, 16.7] 1.7 [-2.7, 6.1] 11.2 [1.2, 21.2] -33.0 [-92.0, 26.3]
Mistral-7B Summary 6.7 [-2.5, 16.2] -2.0 [-14.8, 10.4] 0.4 [-5.0, 5.6] -32.5 [-45.0, -20.0] -54.6 [-99.9, -12.7]
Mistral-7B SystemBlock -5.4 [-12.9, 2.1] -10.7 [-22.4, 0.3] -2.3 [-7.2, 2.5] 3.8 [-6.2, 13.8] -7.5 [-56.7, 39.8]

Table 17: Seven-query natural retrieval and ordinary QA utility.
(a)Seven-query natural retrieval with control-specific attackers. Paired deltas use all answer-only test items, without conditioning on retrieval; target presence is
reported separately. Brackets: 95% paired source-item bootstrap intervals.
Model DefenseUncontrolled
recallDefense
recallPaired
∆Target
presence
Qwen3-8B Summary 42.5 40.8 -1.7 [-7.9, 4.6] 72.5
Qwen3-8B SystemBlock 42.5 43.8 1.2 [-4.6, 7.1] 72.5
Qwen3-8B Threshold 42.5 32.5 -10.0 [-18.3, -2.5] 46.2
Phi-4-14B Summary 40.8 45.4 4.6 [-2.9, 12.1] 72.5
Phi-4-14B SystemBlock 40.8 39.6 -1.3 [-6.7, 4.2] 72.5
Phi-4-14B Threshold 40.8 32.1 -8.8 [-15.8, -2.1] 46.2
Qwen2.5-3B Summary 22.5 12.5 -10.0 [-16.7, -3.8] 72.5
Qwen2.5-3B SystemBlock 22.5 27.9 5.4 [-1.2, 12.5] 72.5
Qwen2.5-3B Threshold 22.5 22.1 -0.4 [-7.5, 6.7] 46.2
Mistral-7B Summary 30.4 37.5 7.1 [-1.3, 15.8] 72.5
Mistral-7B SystemBlock 30.4 23.8 -6.7 [-13.8, -0.4] 72.5
Mistral-7B Threshold 30.4 29.2 -1.2 [-6.7, 5.0] 46.2
Table 17, (b)Single-turn, document-disjoint ordinary-QA token F1 (percentage points) with 95% item-bootstrap intervals; utility does not vary with attack budget.
Model Uncontrolled Summary SystemBlock Threshold
Qwen3-8B 18.6 [14.6, 22.8] 14.1 [10.9, 17.4] 18.1 [14.3, 22.3] 18.6 [14.7, 22.6]
Phi-4-14B 11.5 [8.8, 14.7] 7.6 [6.0, 9.4] 13.5 [10.2, 17.2] 15.1 [11.1, 19.4]
Qwen2.5-3B 14.9 [11.2, 19.1] 6.6 [4.8, 8.7] 16.7 [12.8, 21.0] 14.1 [10.3, 18.0]
Mistral-7B 17.2 [13.5, 21.1] 11.0 [8.7, 13.5] 16.0 [12.5, 19.6] 16.9 [12.8, 21.6]
1 3 7−15010Effect (pp)
-0.82.5Qwen3-8B
1 3 7
0.4
-0.8Phi-4-14B
1 3 7
01.7Qwen2.5-3B
1 3 7
6.7
-5.4Mistral-7B
1 3 7
Queries−15010Effect (pp)
-1.71.2
-10
1 3 7
Queries
4.6
-1.3
-8.8
1 3 7
Queries
-105.4
-0.4
1 3 7
Queries
7.1
-6.7-1.2Adaptive query-budget curves
A  Source-intervention DiD · shared attacker
B  Natural-retrieval paired change · defense-aware, all-item ITTSummary SystemBlock Threshold
Figure 4: Exploratory exact-answer proxy changes at query budgets 1, 3, and 7 (percentage points). Top: source-
intervention difference-in-differences under a shared attacker. Bottom: paired natural-retrieval changes under
control-specific attackers, using all-item intent-to-treat analysis. All three original point estimates are plotted.
Stopped interactions carry their last released state forward; models and attacker designs are not pooled. Intervals
were not reported in the original budget table.
D Survey Synthesis and Record-Level Operating Points
Mechanisms, intervention stages, and metrics are multi-label axes, so their counts need not sum to 118.
Interaction settings comprise 51 single-query, 31 fixed multi-query, 25 non-interactive, and 11 feedback-adaptive records.
Attacker access includes external black box (47), corpus-contributor write (39), authorised user (16), operator white box (13),
service-provider protocol party (6), corpus-owner auditor (4), and operator grey box (3). Intervention stages are generation (81),
context construction (64), retrieval (60), post-generation (39), corpus acquisition (27), index construction (26), session (8), and
removal/update (7). Mechanism counts are in Figure 5, with one additional “other” record; metric counts are in Figure 5.

AA-pii BCDEF
01 [R] Rerank / threshold
02 [G] Context transform
03 [R] Integrity control
04 [G] Output filter
05 [P] Provenance
06 [R] Access filter
07 [G] Component isolation
08 [G] DP / noise
09 [G] Input filter
10 [G] Constrained decoding
11 [G] Watermark / canary
12 [I] Secure computation
13 [U] Unlearning / deletion20029225
75207414
00002032
340012212
00213414
01011514
1120738
3225240
0110715
3102421
00000111
0050110
0011301n
40
39
34
33
24
22
22
18
15
13
12
7
6AA-pii BCDEF
Attack success / integrity
T ask utility
Access / disclosure
Membership / use detection
Literal overlap
Formal bound
Attribution / citation
Content reconstruction
Semantic similarity
Removal / forgetting012012244
432613525
46102222
00032182
7310111
0016030
0000008
0070000
3100000
0000200n
61
58
37
25
14
10
8
7
4
2
0 15 30 45 61
Records (shared scale)A  Mechanisms by endpoint B  Metrics by endpoint
Stage: R retrieval; G generation; P post-gen.; I index; U removal/update.
118 human-coded records. Multi-label rows overlap; n is the row total.Figure 5: Mechanisms and metrics by primary endpoint over 118 records. Cells and row totals share a count scale;
multi-label rows overlap. Stage codes and numbered representative works retain the original annotations.
Representative works by mechanism row:01 (Kim et al., 2025; Choi et al., 2025); 02 (Zeng et al., 2025b,a); 03 (Tan et al., 2024; Yang et al., 2024); 04 (Cohen
et al., 2024; Zafar et al., 2026); 05 (Qi et al., 2024; Cui and Liu, 2026); 06 (Budigi and Sirigiri, 2026; Chen, 2024); 07 (Namboothiri, 2026; Hong et al., 2024); 08
(Abouelenein and Torki, 2026; Hu et al., 2024); 09 (Zeng et al., 2025b; Li et al., 2026b); 10 (Golatkar et al., 2024; Wang et al., 2026a); 11 (Yang et al., 2026;
Jovanovi ´c et al., 2025); 12 (Li et al., 2026c; Wanger, 2026); 13 (Zhang et al., 2025d,a).
0 30 60 90 118
RecordsEvent / asset / actor / access
Comparator and ablation
Evaluation independence
Sample and unit reporting
Uncertainty analysis
Adaptive / robustness testing
Matched utility and harms
Artifact / reproduction support89
112
53
25
48
86
6129
109
59
21
55
24
5772
1589 / 29 / 0
112 / 5 / 1
1 / 109 / 8
53 / 59 / 6
25 / 21 / 72
48 / 55 / 15
86 / 24 / 8
61 / 57 / 0L / S / HReporting and design appraisal
n=118 per domain; NR=0.Lower concern Some concern High concern
Figure 6: Reporting and design appraisal over 118 records. Counts are lower/some/high concern; NR is zero
throughout. These ratings are not efficacy weights.
D.1 Artifact and provenance
D.2 Reported operating points by endpoint family
Fam. Record Primary endpoint as reported
A Zeng et al. (2024a) Repeat Prompt / Repeat Context / ROUGE Prompt+Context counts of retrieved chunks reproduced
in output; summarization, distance threshold, re-ranking drive counts to 0
A Wang et al. (2026a) Repeat Prompt/Context defined as response reproducing text verbatim from the retrieval corpus;
ROUGE-L above threshold vs corpus text (264->127 etc.)
A Zeng et al. (2025a) Repeat Prompt = at least 10 tokens exactly matching original dataset; ROUGE-L>0.5; targeted PII
count driven to 0
A Liang et al. (2026b) Repeat Prompts / Repeat Length / Target Info under targeted and prefix extraction attacks (59 pct
reduction in repeated prompts)
A Zhu et al. (2026) Privacy leakage rate under the ’privacy extraction attack’ of He et al. (ChatDoctor 0.23 -> 0.17;
EnronQA 0.63 -> 0.33, Table 3); BLEU/ROUGE-L reported as UTILITY
A Zhang et al. (2026b) Repeat Prompts = >=10 consecutive tokens from retrieval corpus; ROUGE Prompts/Contexts>0.5;
zero leakage across epsilon=0.1..10
A Yao et al. (2026) ALR/ASR/CLR privacy-entity leakage in generated output vs a GRPO-trained adaptive attacker
(NormAvep 1.00 -> 0.13)
A Li et al. (2026a) Chunk Recovery Rate (ROUGE-L>0.5 generated output vs original chunk plus cosine) under
Worm-Attack and RAG-Thief; relative mean 1x -> 0.51x

Fam. Record Primary endpoint as reported
A-pii Li et al. (2026b) Document-level any-success PII leak rate in outputs 0.875 -> 0.000 (en/zh/de) under output regex
filter plus input judge
A-pii Zhao et al. (2027) Repeat Prompt = outputs containing >=10 consecutive verbatim tokens from private text; ROUGE
Prompt>0.5 vs original private text; adaptive attack table
A-pii Cheng et al. (2026) De-anonymization Rate under a three-tier threat model (ArcFace/CosFace/AdaFace 99.4 -> 1.8-3.1
pct; T2 top-1 4.2 pct), plus VQA accuracy / FID / LPIPS utility
A-pii Moulton et al.
(2025)Garak PII-leak failures 50 -> 0; Laplace-adjusted success probability 0.980 -> 0.019; ABAC arm
renders the PII corpus unretrievable
A-pii Bodea et al. (2026) LLM-as-a-Judge privacy leakage score 0-100 comparing sensitive entities in the SOURCE text
against those present in the GENERATED text, PRE and POST anonymization (Table 5), plus
ROUGE-L/cosine/PPL utility
A-pii Mori et al. (2026) Count of patient-name leaks in model responses under 1000 benign and 100 targeted extraction
queries (Table 3)
B Jin et al. (2025) ROUGE-2/BLEU-2/EMR/BiNLI of embedding-inversion (EIA) reconstructions of the user’s input
text, driven to ~0
B Chen et al. (2026a) ROUGE-2 recall of document reconstruction from generated LoRA weights by a trained T5
decoder (<9 pct)
B Li et al. (2026c) Edge recovery rate from observed HNSW traversal paths 18.3 -> 6.8 pct with false-positive rate
and Recall@10 degradation at identifiable rho/E operating points
B Wanger (2026) Isolation-Forest / One-Class-SVM detector ROC-AUC (0.85-1.000) and TPR@1%/5% FPR
against steganographic embedding exfiltration; VectorPin provenance protocol
B Ye et al. (2025) Vec2Text reconstruction BLEU/precision/recall/F1/ROUGE from encrypted embeddings (BLEU
83 -> 12) plus retrieval accuracy and throughput
B Mao et al. (2026) Resistance to ICA and graph matching; kNN-overlap rate between plaintext and ciphertext spaces;
relative attention-output error 1.49-1.63 pct
B Chakraborttii et al.
(2026)Vec2Text ROUGE-L recovery from soft-deleted HNSW vectors 0.207 -> 0.000 under epoch key
rotation; ECDSA-signed erasure proof
C Wu et al. (2025b) Formal (epsilon approx 10) DP guarantee across hundreds of queries via individual Renyi privacy
filters, plus utility; no empirical leakage measurement in the packet
C Lebensold et al.
(2024)Shutterstock watermark visibly copied into generated images (qualitative, UNDEFENDED RDM
baseline, Fig.2b); protection outcome is DP epsilon=10 plus FID 10.9
C Abouelenein and
Torki (2026)Membership-inference attack accuracy 70.22 -> 53.60 pct (advantage Adv = Acc - 0.5) at a 2.6 pct
accuracy cost under pure epsilon-DP SimHash bucketing
C Golatkar et al.
(2024)CLIP similarity between CPR-generated images and retrieved private images (Syn-Ret) held low
while prompt similarity (Syn-Cap) stays high; empirical k_c / NAF Delta_max bit bound
C Koga et al. (2024) S2MIA TPR-FPR curves with AUC collapsing to ~0.5 at epsilon=10 (BLEU used as the
membership score), plus average match accuracy utility
C Min et al. (2024) Perplexity on held-out Harry Potter book with vs without HP books in the datastore (Table 6);
qualitative sentence-level attribution examples (Table 7)
D Agnew et al. (2024) Judge-rated fraction of defended texts the attacker LLM is UNABLE TO USE to answer (news
RAG benchmark n=75; >90 pct on many datasets; 75 pct transfer to unseen text)
D Pallerla et al. (2026) MBA exact mask-fill reconstruction leakage rate (5 masked tokens, membership signal) 29.5->0.0
pct; poisoning ASR 56->0 pct
D Hu et al. (2024) HR@K / MRR separately for public answers and privacy-threatening (private-link-derived)
answers; adversarial training suppresses the latter
D Namboothiri (2026) Structural leak rate (unauthorized chunks RETRIEVED) 86.1->0 pct; keyword-matched answer
leak 41.3/29.5 -> 0 pct; 100 pct -> 0 pct violation rate
D Panebianco et al.
(2025)Cluster purity, accuracy, precision, recall, F1 (0.77) and AUPRC 0.97 for DETECTING PII-leak
and jailbreak interactions
D Zhang et al. (2024) User study (N=36): GPT-4o-rated reduction of private information stored/input into LLM memory,
plus subjective/objective usability metrics (Table 1)
D Lorenzo et al.
(2025)Role-based access filtering accuracy 85 pct and F1 0.89 on 389 queries vs static-filter baseline (F1
0.93-0.94) and no-filter baseline (F1 0.32-0.43)
D Xie et al. (2026a) Attack Success Rate of PII extraction (Bad Query / Adversarial / CopyBreakRAG) 81.9 -> 9.5 pct;
detection P/R/F1 94.1/93.5/93.8
D Zafar et al. (2026) AUROC / FPR95 / FPR90 and abstain rate for DETECTING quasi-identifier cluster leakage in
generated RAG outputs across three domains
D Cherubin and
Paverd (2025)Precision/recall for declined answers, K-precision/recall and Elo utility; architectural isolation
means user text never reaches the generator
D Arceo and Narsing
(2026)Cross-Tenant Leakage Rate = fraction of cross-tenant probes returning at least one unauthorized
CHUNK (80->0 pct); A VR; injection block rate
D Zhou et al. (2025b) LASR/NA V/LK all 0.00; LK = leaked tokens / total private tokens with leakage confirmed by
ROUGE-L recall between generated output and private KB; MIA and Vec2Text/GEIA and PASR
D Budigi and Sirigiri
(2026)Tenant-isolation leakage rate 0.2 pct -> 0 pct over 1000 queries with engine-level row-level
security, plus latency
D Chen (2024) Access-control compliance verified per test case (retrieval restricted to permitted profiles) plus
relevance/correctness of retrieved information
D Salemi and Zamani
(2026)Attacker profile-identification accuracy (linkability) and attribute-inference accuracy: ~100 pct
with full context vs 20 pct client-side / 58.5 pct query-only

Fam. Record Primary endpoint as reported
D Wang et al. (2025d) Private-triple removal rate r_remove and de-anonymization QA-attack success rate r_attack, plus
PPL fluency and downstream RAG accuracy
D Masoud et al.
(2026)Privacy and completeness scores with and without prompt injection (Table I); up to 58 pct
privacy-score improvement
D Thornton (2026a) Retrieval Pivot Risk 0.95 -> 0.0 across three corpora with Leakage@k, Amplification Factor and
per-attack RPR=1.0 baselines
D Nutakki (2026) Unauthorized context exposure 3.84 -> 0.00 pct; sensitive answer leakage 2.91 -> 0.04 pct; 98.6
pct red-team attacks blocked
D Zhang et al. (2025d) Unlearning effectiveness on the forget set after excluding forgotten user data from the retrieval
index, with HR@K / NDCG@K utility on remaining data
D Mouzouni (2026) Unauthorized-access outcome across four governance baselines (200 queries): results are never
unauthorized regardless of 63-75 pct routing accuracy
D Byun et al. (2025) Confidentiality-filter recall 99.01 pct (82.31 pct accuracy / 74.35 pct recall with ambiguous queries)
preventing security-sensitive queries reaching external LLMs
D Wu et al. (2025a) 98 pct accuracy detecting malicious, document-extraction and hallucination attacks; extraction
requests denied
D Ouyang and Hou
(2026)ASR matrix driven to zero across three attack families with lineage/provenance signing and a tau x
K ablation, plus utility baseline
D Wang et al. (2025a) Unlearning Success Rate; ROUGE-L recall of output vs short reference ANSWER; TPR@1%FPR
residual memorization
D Yao et al. (2025a) AUROC / matching accuracy for risk detection across five risks (0.96-0.997) and
precision/recall/F1 for mitigation
E Chen et al. (2025b) Verification Success Rate, CGSR and TPR/FPR of AQUA watermark verification; filter-resistance
results
E Zeng et al. (2025b) Shadow-RAG membership audit AUC 0.989 / Acc 0.957 / F1 0.956; robustness under
prompt-modifying and paraphrasing defenses
E Choi et al. (2025) Detection accuracy/precision/recall/F1 for member attack queries plus adjusted attack accuracy
(adversary advantage) reduction
E Xie et al. (2026b) Chunk Recovery Rate: chunk counted recovered iff ROUGE-L(generated output, original
chunk)>0.5 AND cosine similarity; 97.4 -> 1.6; detection TPR
E Kwok et al. (2026) Effective Detection Rate with p<1e-5 at 0.1 pct sentinel injection; FDR<=1.0 pct across four
datasets
E Zhang et al. (2025c) Internal-state classifier ACC/F1 (up to 100/100) separating copyrighted-leakage from non-leakage
inputs; IS-w/RAG vs LLM-as-a-Judge baselines
E Liu et al. (2026) stylometric attribution F1 0.825 -> 0.221 (TF-IDF) and 0.691 -> 0.203 (BERT)
E Huang et al. (2026) Watermark detection AUC (up to 100) and spoofing-AUC; Knowledge Corruption Rate;
ROUGE-1/L and PPL reported as UTILITY of the watermarked answer
E Yang et al. (2026) Watermark recovery rate WRR=1.00 for multi-bit payload; robustness/detection outcomes under
attacks
E Burnat (2026) Coalition-detection TPR=1.00 over 2400 trials at FPR<=0.05, plus (eps_acc, delta_acc)-DP
retrieval guarantee and a matching membership-inference AUC trend in k
E Fu et al. (2026) AUC / ASR / TPR@1 reductions across seven membership-inference attacks (SPV , LiRA, Recall,
Log-Loss); up to 27.8 pct MIA-success reduction and 75 pct TPR@1 reduction
E Pellicer et al. (2026) Artifact-level disclosure risk on exported evidence: Membership-Inference Accuracy -9.8 pp,
Attribute-Inference Accuracy -8.4 pp, top-1 Linkage success rate (Table 2)
E Guo et al. (2025) CoT-space watermark ownership verification p-values (malicious p«0.01 vs independent p=1.00)
E Jovanovi ´c et al.
(2025)Watermark-propagation p-values and TP/FP rates on the FARAD benchmark
E Liu et al. (2025c) ROC-AUC and TPR@1%FPR = 100 pct at a 12-query operating point for watermarked canaries
E Liu et al. (2025b) Interrogator-Detective dual-layer watermark detection accuracy and false-positive rates across
LLMs and adversarial conditions
E Luo et al. (2026) Detection AUC / TPR from sentinel-image key queries against the RAIG reference database
E Lv et al. (2025) WSN / WIRR watermark verification success and integrity across five datasets and four LLMs,
including robustness to removal attacks
F Yang et al. (2024) Poisoned-context attack defense improvement ~20 pct; QA accuracy under varying
poisoned/factual context ratios
F Saleem et al. (2026) Macro-averaged ASR and FPR, detection rate and response-quality score for a three-layer
prompt-injection defense (Table 8)
F Cohen et al. (2024) Virtual Donkey detection TPR=1.000 / FPR=0.015 (BLEU, ROUGE-L, METEOR input-output
similarity used as classifier FEATURES); attack-side exfiltrated email-address F1
F Ceraku (2026) Poison Impact Rate reduction across ingestion/retrieval/generation layers on a 300-document CTI
corpus (90 poisoned), plus latency
F Cui and Liu (2026) Character-level traceback overlap and over-selection metrics for locating poisoned spans;
passage-level identification and ASR baselines
F Sharma (2026) Certified defense: ASR 93 pct -> 5.3 pct (n=150) for runtime memory poisoning, with provenance
signing
F Jin et al. (2026) Malicious-document detection performance plus end-to-end ASR and precision
F Kim et al. (2025) ASR 0.04 against Gemini on MS MARCO with highest accuracy among defenses

Fam. Record Primary endpoint as reported
F Wang et al. (2025b) Attribution precision/recall for identifying the responsible context texts, plus ASR with and
without the defense
F Noughabi et al.
(2026)True-positive detection rates for poisoned subgraph elements vs perplexity-based filtering baselines
F Datta et al. (2026) ASR reduced to near zero for sentence-relation-based sanitization of malicious instructions (from
82-83 pct baseline)
F Yao et al. (2025b) Answer accuracy and ASR across poisoning settings for context-analysis-based security
F Gerami et al. (2025) Influence Score identifies the poisoned document with 86 pct success over 3000 samples
(p<0.0001)
F Tan et al. (2026) Provably robust aggregation: ASR reduced to as low as 1 pct at 71 pct accuracy
F Wang et al. (2026b) Adulterant Retrieval Rate 100 pct; Hijack Success >94 pct (unauthorized system answers wrongly);
CDPA/CIRA 100 pct authorized fidelity; 80.2 pct robustness retention
F Thornton (2026b) Best-F1 detection comparison (QPD, keyword anomaly) and ROC curves for corpus-dependent
poisoning
F Cheng et al. (2025b) Chunk-wise perplexity filtering detection of poisoned chunks plus text-similarity metrics
F Lin et al. (2025) Security Rate of generated code under two poisoning scenarios (average across LLMs drops 60.84
-> 50.57 without defense; CodeGuarder consistently improves SR)
F Yu et al. (2026) ASR increase of 4-16x when the isolation component is removed; certified-synthesis
information-flow controls
F Tan et al. (2024) TPR/FPR tables for revealing poisoned responses via LLM activation analysis (metric definitions
elided in packet)
F Wang et al. (2026c) Attribution precision/recall for the injected/corrupted texts plus ASR with and without attack
F Gowda (2026) ASR-R / ASR-A under faithful evaluation with a certified detection radius for memory poisoning
F Pathak and Sharma
(2026)Adversarial-hub attack recall 1.000 (universal/concept) and 0.91 on HotFlip at AUROC ~1.0 with
~1 pct benign FPR at an admission-time gate
F Dekel et al. (2026) ACC and ASR across poisoning attacks using sparse-attention corpus defense
F Alam et al. (2026) Attack Pass-through Rate plus 0 pct FPR on the benign subset for priority-aware prompt
control-flow integrity
F Pathmanathan et al.
(2025)ASR (retriever returns at least one poisoned passage) plus utility under benign conditions for
RAGPart/RAGMask
F Choudhary et al.
(2025)Attention-aware detectability of stealthy poisoning across epsilon regimes vs perplexity filters;
end-to-end robust pipeline
F Liang et al. (2026a) Answer accuracy and ASR (0.00 across several settings) for a RAG-guards-RAG expert-module
defense
F Xian et al. (2024) Mahalanobis-distance detection rate >98 pct for universally poisoned documents plus retrieval and
attack success rates
F Saxena et al. (2025) Corpus-poisoning detection precision/recall and F1 across datasets and evidence sizes vs perplexity
baselines; 86 pct rationale accuracy
F Patil (2026) ASR 0 pct (95 pct CI [0,1]) for numerical-claim manipulation with 99.8 pct entity detection rate
F Jia et al. (2025) Answer accuracy on a poisoned MS MARCO corpus using PageRank-based trust propagation
F Batista et al. (2025) Sentence-level in-generation attribution accuracy per algorithm (Table 5) with pre-attribution
ground truth
F Zhou et al. (2025a) ACC / ASR across datasets for K-means conflict removal and knowledge consolidation against
corpus poisoning
F Si et al. (2026) Answer accuracy and ASR (percentage of poisoning queries/documents succeeding) for two-stage
semantic filtering
F Alvarez et al. (2026) Block F1 0.802 / 0.846 and top-1 source accuracy 0.858 (source-plus-relation 0.229) vs
MiniCheck, RAGAS, AlignScore baselines
F Gao et al. (2026) ASR reduced by up to 87 pct against PoisonedRAG with answer-accuracy preservation (Tables 1-2)
F Leong (2026) ASR by defense across nine models (360 runs): input-layer 77.8 pct, Memory Sandbox 0 pct for
eight of nine, one model inverts to 100 pct via the RAG pathway
F Shih and Kang
(2025)Attack-detection precision 1.00 with recall/accuracy 0.45 and F1 0.62 for the Reverse RAG version
over 674 attacks; 100 pct block on a 301-sample high-confidence subset
F Shen et al. (2026) Provably robust generation accuracy and ASR under prompt injection at different attack positions
F Xiang et al. (2026) 71 pct clean accuracy with 38 pct certifiable robust accuracy; ASR reduced from over 90 pct
F Chen et al. (2026c) Answer accuracy and ASR (frequency of attacker-specified output) across datasets
F Hong et al. (2024) Discriminator classification F1 and exact-match robustness under counterfactual/corrupted
retrieved context
F Fei et al. (2026) Harm driven to 0.000 by removing unauthenticated structural writes and rerunning the selector;
accumulability criterion
F Qi et al. (2024) Agreement with human answer attribution on XOR-AttriQA; citation precision/recall/F1
(TRUE/NLI-based)
F Zhang et al. (2025a) Traceback precision/recall for identifying poisoned texts in the knowledge database vs
perplexity-based detection
F Zhang et al. (2026a) Responsibility-attribution accuracy for poisoned knowledge vs PPL/norm baselines; RAG accuracy
under benign and poisoned settings
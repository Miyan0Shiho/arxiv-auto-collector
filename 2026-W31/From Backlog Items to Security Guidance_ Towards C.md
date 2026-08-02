# From Backlog Items to Security Guidance: Towards Continuous Security Compliance

**Authors**: Ignacio García Núñez, Florian Angermeir, Fabiola Moyón Constante

**Published**: 2026-07-29 18:31:01

**PDF URL**: [https://arxiv.org/pdf/2607.27374v1](https://arxiv.org/pdf/2607.27374v1)

## Abstract
Continuous software engineering in regulated domains requires engineering teams to address security throughout the development lifecycle. Yet making security requirements explicit in backlog items is still problematic. Engineers must instead infer security relevance of backlog items from brief, free-form descriptions and often lack timely guidance on applicable requirements. We present an NLP-based backlog enrichment system that detects security-relevant backlog items and links them to relevant security requirements. The approach combines a security-relevance classifier with a retrieval-augmented generation (RAG) pipeline over security requirements documents. The approach was developed and evaluated in the context of a large enterprise in highly regulated domains. We present three contributions. First, we release a dataset of 288 backlog items labeled for security relevance by nine security practitioners, with substantial agreement (Fleiss' $κ=0.787$). Second, a recall-oriented classifier achieving $F2=0.774$ in-distribution and mean zero-shot G-measure $\approx 0.65$ across five established benchmarks, matching or outperforming most published classical-ML and open-source GPT baselines. Third, we preliminarily evaluated a four-stage security requirements document-grounded RAG pipeline with two practitioners on industrial backlogs using company-internal security policies and CIS Benchmarks. Of the retrieved 24 clauses, 12 were rated at least 4/5 for relevance. Our findings provide first indicators that NLP-based product backlog enrichment can support engineers in identifying security requirements early in the development process. With this work we aim to facilitate continuous security compliance through proactive introduction of security requirements in continuous software engineering.

## Full Text


<!-- PDF content starts -->

From Backlog Items to Security Guidance: Towards Continuous
Security Compliance
Ignacio García Núñez
garcia.nunez@tum.de
Technical University of Munich
Munich, GermanyFlorian Angermeir
fortiss and Blekinge Institute of
Technology
Munich and Karlskrona, Germany
and SwedenFabiola Moyón Constante
Siemens and Technical University of
Munich
Munich, Germany
Abstract
Continuous software engineering in regulated domains requires
engineering teams to address security throughout the development
lifecycle. Yet making security requirements explicit in backlog items
is still problematic. Engineers must instead infer security relevance
of backlog items from brief, free-form descriptions and often lack
timely guidance on applicable requirements. We present an NLP-
based backlog enrichment system that detects security-relevant
backlog items and links them to relevant security requirements. The
approach combines a security-relevance classifier with a retrieval-
augmented generation (RAG) pipeline over security requirements
documents. The approach was developed and evaluated in the con-
text of a large enterprise in highly regulated domains. We present
three contributions. First, we release a dataset of 288 backlog items
labeled for security relevance by nine security practitioners, with
substantial agreement (Fleiss’ 𝜅=0.787). Second, a recall-oriented
classifier achieving 𝐹2=0.774in-distribution and mean zero-shot
G-measure≈0.65across five established benchmarks, matching
or outperforming most published classical-ML and open-source
GPT baselines. Third, we preliminarily evaluated a four-stage se-
curity requirements document-grounded RAG pipeline with two
practitioners on industrial backlogs using company-internal se-
curity policies and CIS Benchmarks. Of the retrieved 24 clauses,
12 were rated at least 4/5 for relevance. Our findings provide first
indicators that NLP-based product backlog enrichment can sup-
port engineers in identifying security requirements early in the
development process. With this work we aim to facilitate continu-
ous security compliance through proactive introduction of security
requirements in continuous software engineering.
CCS Concepts
•Security and privacy →Software security engineering;•
Computing methodologies →Natural language processing;•
Information systems →Information retrieval;•Software and
its engineering→Agile software development.
Keywords
Security compliance, product backlog, natural language process-
ing, retrieval-augmented generation, security requirements, agile
software development
1 Introduction
Security compliance is no longer confined to a small set of safety-
or security-critical domains [19]. Regulations such as the EU Cyber
Resilience Act [10] and a growing body of sector-specific standards(e.g., IEC 62443 for industrial automation and control system de-
velopment [16]) are making security a baseline expectation for
nearly every organization that builds or operates software. In orga-
nizations leveraging agile or DevOps, this circumstance requires
engineering teams to continuously address security throughout
the development process [13, 31]. In these settings the product
backlog is the central planning artifact [17, 11], yet backlog items
rarely carry explicit security or compliance annotations [31]. Se-
curity considerations are therefore mostly embedded implicitly
in short, free-form engineering tasks. This leaves product teams
without clear guidance on specific security matters and creates a
practical mismatch between continuous software engineering and
compliance efforts. Security planning and reviewing is still often
performed manually and outside the main backlog flow, which does
not scale to fast development cycles [19]. Moyón et al. [20] describe
this tension in secure continuous software engineering through
several open practical challenges, such as making security archi-
tecture visible in backlog artifacts and documentation, involving
security activities with minimal lead-time burden, and enabling
security knowledge and ownership within engineering teams.
This manuscript reports on building, deploying, and evaluat-
ing a natural-language backlog enrichment system that addresses
this gap directly in the artifact developers already use. The system
surfaces security-relevant context at the product planning stage
by attaching security requirements relevant to the organization to
flagged backlog items. The work was developed and evaluated at
a large enterprise in highly regulated domains. To enable repro-
duction, we additionally evaluate the approach on public security
guidelines.
The paper makes three contributions:
C1An expert-annotated dataset of288backlog items carrying
binary security-relevance labels, drawn from two public At-
lassian projects and multi-rated by9security practitioners.
C2A recall-oriented binary classifier that flags security-relevant
backlog items as a first-stage filter for the backlog enrich-
ment system.
C3A retrieval pipeline that links backlog items to applicable
security requirements from requirement documents.
We evaluate the three contributions from complementary per-
spectives: inter-rater reliability and dataset characterization for
C1, in-distribution performance of C2 on C1 and cross-distribution
transfer to the Wu et al. [32] datasets, and a preliminary evaluation
via two practitioner interviews on industrial backlogs for C3, which
covers both proprietary and public security policy corpora.
Across the three contributions, a single overarching finding
emerges. Security-relevance of backlog items can be evaluated
arXiv:2607.27374v1  [cs.SE]  29 Jul 2026

García Núñez et al.
based on the ordinary item text, and concrete security requirements
can be attached to those items in a form that engineers and security
experts can act on without leaving their existing workflows.
2 Background
Three threads of related work are relevant to the contributions of
this paper: secure continuous software engineering and the role of
the product backlog, detection of security-relevant development
artifacts, and the operationalization of normative requirements,
such as regulations or standards, into developer guidance.
2.1 Secure Continuous Software Engineering
Continuous software engineering shortens delivery cycles and em-
phasizes incremental, just-in-time work [13]. In contrast, security
compliance is traditionally organized as a parallel, formalized track
running alongside development, as it is composed of review check-
points, traceable evidence, and audit-oriented documentation [21,
3]. This parallelism is a practical bottleneck: through case studies
in multiple industrial contexts Moyón et al. [20] confirmed open in-
dustrial challenges blocking adoption at scale, whose consequences
share a pattern at the artifact layer. A lack of item-level security
visibility re-introduces compliance checks at release time as a block-
ing task, security activities with high lead time are deferred under
pressure and re-emerge as security debt, and security knowledge
concentrated on experts creates a review bottleneck for the rest of
the organization. The contribution of this paper handles these con-
sequences directly: making security relevance visible at the item
level (C1, C2) and attaching applicable requirements to backlog
items at the point of planning (C3).
2.2 Product Backlog
Continuous software engineering teams use product backlogs and
issue trackers as their central planning artifact [17]. Backlog items
are typically short, informal, and optimized for coordination rather
than for documentation of non-functional concerns, such as security
compliance [11]. As a consequence, when an item carries security
implications, those implications are usually embedded implicitly
in the free-text description rather than defined as labels, fields, or
linked requirements [23], which is the artifact-layer gap that this
paper targets.
In this manuscript, we use the definition of Backlog and Backlog
Item derived from the ISO/IEC/IEEE 26515:2018 [17]: “A collection
of agile features (3.7) or stories of both functional and nonfunc-
tional requirements that are typically sorted in an order based on
value priority”. Related to that definition is the definition of fea-
ture, also provided in that document: “functional or nonfunctional
distinguishing characteristic of a system”.
Since our focus is on delivering security-relevant information to
engineers during planning and development, this manuscript does
not differentiate between different backlog types such as product
backlogs or sprint backlogs.
2.3 Security Requirements
For this work we use the broad term security requirements to refer
to requirements related to security formalized in various documents,
such as company-internal security policies, security standards (e.g.,IEC 62443-4-1), contracts, or public best practice guidelines (e.g., CIS
Benchmarks). Depending on the document type, those requirements
might be called differently. In internal security policies, security
requirements are referred to as clauses, in security standards they
often are called controls, and in public guidelines they are often
coined recommendations. Hence, when we use the termsclauses,
controls, orrecommendations, we refer to security requirements
in the context of the respective document type formalizing those
requirements.
2.4 Automated Detection of Security-Relevant
Backlog Items
Another branch of prior work focuses on automatically identifying
security-relevant backlog items. Wu et al. [32] showed that widely
used datasets contain substantial label noise and released cleaned
datasets that have since become a benchmark for state-of-the-art
approaches [1, 14, 28]. Subsequent work has improved backlog
item-based security-relevance identification on those benchmarks
using lightweight neural text models [1], open-source GPT-based
approaches [14], and fine-tuned or prompted LLMs [28].
Relevant to our contributions are other detection approaches
focusing on the privacy domain. Sangaroonsilp et al. [27] classify
privacy requirements in backlog items by treating issue trackers as
the central agile artifact and aligning backlog items with privacy-
requirement categories to support compliance-oriented develop-
ment.
A second closely related line of work makes regulations ac-
tionable for engineers. Ayala-Rivera and Pasquale’s GuideMe ap-
proach [5] supports the operationalization of GDPR obligations
into solution requirements and privacy controls, and the SoCo ap-
proach [6] incorporates suitable privacy and security requirements
into software design models.
These approaches share the motivation of C3 but operate at the
regulation-to-requirement or design-model level, rather than on
the backlog level.
Finally, software traceability research helps explain why this
problem remains open in practice. Ruiz et al. [25] report that practi-
tioners still perceive traceability as valuable but costly, with substan-
tial manual effort and limited automated support. This strengthens
the motivation for tooling that links the artifacts developers already
use to the security requirements they are expected to satisfy.
3 Methodology
Figure 1 summarizes the three contributions as a chained workflow:
C1 produces the expert-labeled dataset on which C2 is trained.
C2 flags security-relevant items from a backlog, and C3 retrieves
applicable security requirements for each flagged backlog item. The
subsections below describe the artifacts, activities, and evaluation
protocol of each contribution.
C1: Annotated Dataset of Security-Relevant
Backlog Items
Contemporary approaches [1, 14, 28] evaluate against the Wu et
al. benchmark [32], whose security labels derive from the Jira
security field in the dataset rather than from expert annotation. C1
contributes a smaller but independently expert-annotated dataset.

From Backlog Items to Security Guidance: Towards Continuous Security Compliance
Figure 1: Overview of Contributions and Research Activities as a Chained Workflow
We draw from TAWOS [30], a public archive of over500 ,000
backlog items from44open-source agile projects, selected for its
scale, diversity, and redistributability.
Source Corpus.From this pool we selected two Atlassian projects,
Jira ServerandConfluence Server, whose items describe enterprise
software tooling and whose language resembles the industrial back-
logs targeted by C3. A filtering pipeline applies title-level and full-
text exact deduplication and removes near-duplicates via MinHash
LSH (128permutations,5-word shingles) at a Jaccard threshold of
0.8, yielding an84,878-item labeling pool.
Labeling protocol.Nine practitioners from a large enterprise
were recruited on a voluntary and uncompensated basis. A web ap-
plication distributes items for asynchronous, independent labeling
over a two-week window, and a kick-off session introduced the task,
boundary cases, and the tool. For each item, labelers record a binary
security-relevant judgment after reading the title and descrip-
tion. One of the initial hypotheses was that security-requirement
categories might improve the ultimately retrieved security require-
ments. For that purpose, practitioners additionally assigned a Fire-
smith security-requirement category [12] to each backlog item. As
this hypothesis did not hold, we only store the Firesmith tags as
metadata in the released artifact but did not use them further, e.g., as
training labels. Reliability and agreement checks are not performed
on this metadata.
Label aggregation and reliability.Items require at least two
independent assessments before release. As the labeling process
was distributed asynchronously across a pool of nine practitioners,
not every practitioner rated every backlog item. Reliability statistics
are therefore computed over the available overlapping ratings in
the released dataset. The published label is the majority vote of
labelers’ binary judgments. Following Artstein and Poesio [4], we
assess inter-rater reliability (IRR) with a generalized multi-rater
coefficient rather than averaged pairwise coefficients, and addition-
ally report observed agreement for transparency. Accordingly, for
C1 we report Fleiss’ 𝜅, which quantifies agreement beyond chanceacross all raters simultaneously, and raw observed agreement 𝑃𝑜,
which provides an interpretable baseline of direct rater consensus.
C2: Security-Relevance Detection Tool
C2 is a lightweight binary classifier that flags potentially security-
relevant backlog items as a first-stage filter for the downstream
backlog enrichment system. Because it is more costly for an en-
gineer to miss relevant security requirements for a backlog item
than to discard irrelevant security requirements attached to it, we
optimize C2 for recall-weighted performance and use F2 as the
primary evaluation criterion, but do not tune the decision threshold
as the validation split is too small for reliable threshold calibration.
The input is the concatenated title and description of a backlog
item. The output is a binary security-relevance prediction, with C1
as the gold standard.
Architecture.We pair a frozen sentence-transformer encoder
with a lightweight classification head, deliberately avoiding full
fine-tuning in order to retain cross-domain generalizability.
The reported C2 model space comprises one TF-IDF baseline
and eight sentence-encoder variants. The sentence encoders are
MiniLM-L6, MiniLM-L12, MPNet, DistilRoBERTa-v1, and RoBERTa-
Large, all used via the Sentence Transformers library [24]. All five
encoders are evaluated with an elastic-net logistic-regression head.
In addition, DistilRoBERTa-v1 and MPNet are also evaluated with
a small Multilayer Perceptron (MLP) head, and DistilRoBERTa-v1
is additionally evaluated with an ℓ2-regularized logistic-regression
head.
Training protocol.The protocol follows established practices
for recall-oriented binary classification on imbalanced software
engineering datasets [29]. We apply a stratified hold-out split of
C1 with class-balanced sample weighting during training. In the
used dataset, security-relevant backlog items have a prevalence of
27.8%. Regularization strength (for logistic regression) and MLP
width and depth are tuned via grid search over 5-fold stratified

García Núñez et al.
cross-validation on the training portion. All models are evaluated
at a fixed threshold of0.5.
For deployment selection, we treat the elastic-net and MLP vari-
ants as the transfer-oriented candidate set. The choice of an elastic-
net head over ℓ2-regularized logistic regression and MLP is moti-
vated by generalization risk on a small dataset: elastic-net’s com-
binedℓ1andℓ2penalty promotes sparsity and resists overfitting to
the two-project scope of C1, whereas ℓ2-only regularization and
unconstrained MLP heads are more susceptible to absorbing project-
specific lexical patterns [33]. We therefore report ℓ2results as an
in-distribution reference, but do not treat the ℓ2variant as deploy-
ment candidate.
Evaluation.In-distribution performance is reported on the held-
out C1 test set ( 𝑛=58and prevalence of the security-relevant class
consistent with a stratified split over C1). The primary metric is
F2 (the𝐹𝛽score at𝛽= 2, which weights recall twice as heavily
as precision). Secondary metrics are precision, recall, AUC, and
per-model confusion matrices.
To test whether the learned notion of security-relevant trans-
fers across projects, we evaluate the reported C2 variants zero-shot
on the five Wu et al. [32] datasets (Chromium, Ambari, Camel,
Derby, Wicket) and compare against four families of published base-
lines: Wu et al. [32]: (FARSEC variants and classical text classifiers),
Soltaniani et al. [28] (prompted proprietary LLMs and fine-tuned
LLMs), Alqahtani [1] (fastText), and França et al. [14] (open-source
GPTs and classical ML). The held-out C1 F2 ranking is used sepa-
rately to choose the variant integrated into C3.
The primary cross-distribution metric is G-measure (harmonic
mean of recall and1 −FPR ), which is insensitive to the strong class
imbalance present in both C1 and the Wu et al. datasets (27 .8%vs.
1.9–17.9%). All four baseline families report or allow recomputation
of G-measure, making it the single comparable metric across lit-
erature. We additionally report the per-project standard deviation
𝜎of G-measure as an indicator of cross-project stability, with the
caveat that França et al. [14] do not evaluate on Chromium; their 𝜎
is therefore computed over four projects rather than five and is not
directly comparable to the other baseline families on equal footing.
C3: Backlog Item Enrichment
Given a security-relevant backlog item, C3 surfaces the subset of
security requirements from relevant security requirements docu-
ments. This should help engineers and security experts to address
security and compliance proactively during backlog refinement and
planning, rather than through reactive checks later in the devel-
opment lifecycle, directly addressing the lead-time and workload
burdens reported by Moyón et al. [20].
We frame this contribution as a document-grounded information
retrieval problem: given the title and description of a backlog item,
retrieve and rank the most relevant requirements from a corpus of
segmented security documents.
The backlog enrichment pipeline intended for C3 is developed
for on-premises deployment and operation. It is designed as a four-
stage retrieval-augmented generation (RAG) [18] architecture ap-
plied per backlog item.
(1)Requirement extraction.An open-weights LLM decomposes
the backlog-item text into one or more requirements, somulti-concern items do not dilute their retrieval signal
across competing topics.
(2)Bi-encoder recall.A sentence-transformer bi-encoder [24]
embeds each extracted requirement and retrieves the top- 𝑘
candidate security requirements by exact cosine similarity
over precomputed embeddings of all segmented security
requirements.
(3)Cross-encoder reranking.A sentence-pair cross-encoder [22]
rescores the shortlist with direct query–security require-
ment attention, sharpening the ranking over the dense base-
line.
(4)LLM semantic filter.An open-weights LLM reviews the
reranked shortlist and retains only security requirements it
judges semantically applicable to the requirement, produc-
ing the final ranked list. The LLM acts as a precision filter,
not a reranker: its role is to retain or discard individual
security requirements, grounded in the bi-encoder shortlist
so identifiers cannot be hallucinated.
The architecture was reached iteratively.Version 1 (Dense IR only)
uses the bi-encoder and cross-encoder as a standard two-stage re-
trieval baseline; it produces stable shortlists but admits topically
adjacent false positives where lexical or vector similarity is high but
the security requirement does not impose the same requirement.
Version 2adds the LLM semantic filter to prune the reranked short-
list with a brief per-security requirement justification.Version 3adds
requirement extraction as a preprocessing step, introduced after we
observed that multi-concern backlog items (e.g., “add OAuth2 login
andrate-limit the public API”) systematically retrieved security
requirements relevant to only one of their concerns. Decompos-
ing into atomic requirements before retrieval recovers the missed
concerns.
Both the extractor and the filter use open-weights LLMs so the
pipeline can be deployed on-premises, satisfying a hard privacy
requirement for the industrial deployment context. Additionally,
pinning a specific open model version prevents silent behavior
drifts known from commercial models [7, 2]. For both require-
ment extraction and LLM-based clause scoring in C3, we used
alibayram/Qwen3-30B-A3B-Instruct-2507 , identified by the Ol-
lama model ID 408ee351bdc6 (architecture qwen3moe ,30.5B pa-
rameters,Q4_K_Mquantization).
Security Requirement Segmentation.Segmentation is a one-
time, document-specific preprocessing step that converts each se-
curity document into a flat list of atomic, self-contained security
requirements suitable for retrieval. We provide a segmentation
backend for CIS Benchmarks [8] exported as PDF files, and the
retrieval pipeline itself is segmentation-agnostic. The evaluation
reported in this paper uses the internal-policy and CIS Benchmark
backends.
Preliminary Evaluation Protocol.We report a preliminary evalua-
tion on the performance and workflow fit potential of C3. Insights
gained on this analysis are then used to further develop the tool and
ensure seamless and efficient integration into continuous software
engineering workflows.
The preliminary evaluation combines an artifact-level analysis
of the pipeline’s retrieval output with two semi-structured practi-
tioner interviews conducted on industrial backlogs. The interviews

From Backlog Items to Security Guidance: Towards Continuous Security Compliance
contextualize the retrieval results and help identify design require-
ments that may be necessary for broader adoption. The units of
analysis are retrieved clauses and controls in relation to their source
backlog items.
Evaluation criteria.We define success of the preliminary eval-
uation along three dimensions:relevance, whether a retrieved secu-
rity requirement applies to the backlog item, captured by the prac-
titioner rating;actionability, whether the requirement is concrete
enough to change the engineer’s next action on the item, captured
by the upper rating anchor and the free-text commentary, andwork-
flow fit, whether practitioners would intend to consume the output
within their existing tools, captured by the closing discussion ques-
tions. Redundancy and coverage gaps are recorded through the
free-text commentary rather than quantified. The ratings serve
as qualitative indicators for future design iteration. Quantifying
retrieval precision with more raters, items, and backlogs is deferred
to future work.
Source corpora.We evaluate on two complementary corpora:
internal security policies as the primary use case, and public CIS
Benchmarks (nginx, docker-ce configurations) as a probe for gen-
eralization to concrete best-practice guidelines that any reader can
reproduce against. Using both corpora in the same interview ses-
sion, on the same backlog items, allows a direct within-session
comparison of retrieval quality across security document types
without modifying the retrieval pipeline. Only the security require-
ment segmenter backend has to be swapped.
Item selection.Items for each session were drawn randomly
from the set of backlog items that the selected C2 classifier flagged
at confidence scores ranging from0 .55to0.93, spanning high-
confidence and low-confidence regions of the classifier’s output.
This selection ensures that evaluated items are representative of
what the pipeline would surface in normal operation.
Interview sessions.Two30-minute interview sessions were
conducted with practitioners from the product team owning the
evaluated backlogs. In preparation, the deployed pipeline was run
over each product’s backlog using both corpora, the internal policies
and CIS Benchmarks. Each session presents the resulting shortlists
and collects two structured signals per security requirement: a0–5
relevance rating (0 =not relevant,5 =directly actionable) and free-
text commentary capturing the reasoning. Item-level familiarity is
recorded as descriptive context, but is not used to weight findings;
the sessions target the scenario in which the tool is intended to
be used, where the consumer of a security requirement might or
might not be the author of the corresponding backlog item, as they
might be an engineer or a security expert.
Three open discussion questions close each session: (D1) work-
flow fit and actionability, (D2) missing security requirements or
concerns, and (D3) open feedback, including trust and integration
preferences.
A web application supported the sessions by loading the product
backlog, retrieving relevant security requirements, and presenting
them.
4 Results
We report the results contribution by contribution: the dataset
characteristics and inter-rater reliability of C1, the in-distribution
0 100 200 300 400 500 600
Word count051015202530ItemsNon-security (n=208)
Security (n=80)Figure 2: Frequency of Labels based on Text Length on C1, in
Groups of 25 Words.
Table 1: C1 Dataset Characteristics Relative to the Wu et al.
Benchmark Family. C1 is Smaller but Provides Multi-Rater
Expert Annotation with Reported IRR.
Dataset Source Items Sec. Prevalence
C1 (Ours) Atlassian Jira / Confluence 288 80 27.8%
Wu: Chromium Chromium bugs 41940 808 1.9%
Wu: Ambari Apache JIRA 1000 56 5.6%
Wu: Camel Apache JIRA 1000 74 7.4%
Wu: Derby Apache JIRA 1000 179 17.9%
Wu: Wicket Apache JIRA 1000 47 4.7%
and cross-distribution behavior of the C2 classifier, and the retrieval
quality and practitioner design feedback gathered for C3.
C1 Results
The C1 release contains288backlog items (80security-relevant,
27.8%), drawn from an84 ,878-item filtered TAWOS pool and an-
notated by a pool of nine security experts. Each released item re-
ceived at least two independent assessments, resulting in 972 binary
security-relevance judgments in total. Inter-rater reliability on the
released items is substantial: Fleiss’ 𝜅=0.787, and raw observed
agreement𝑃𝑜=0.800, which confirms a reliable but non-trivial
task.
Of the288items,249are unanimous (55security-relevant,194
non-security-relevant) and39(13 .5%) show at least one dissenting
assessor. Disagreement items are retained with the majority-vote
label. As Figure 2 highlights, text descriptions are long enough
to support text-classification methods: median124words (range
21–748); security-relevant backlog items are meaningfully longer
(median145) than non-security-relevant backlog items (median
119), suggesting implicit security concerns surface more often in
backlog items with richer contextual descriptions.
Relative to the Wu et al. family of benchmarks (1 ,000–41,940
items,1.9–17.9%prevalence, community-assigned labels), C1 is
smaller, but to the best of our knowledge, the only publicly released
backlog-security dataset with independent multi-rater expert anno-
tation and reported IRR, positioning it as a high-precision training
dataset that complements Wu’s larger-scale datasets. An overview
of the items and prevalence of the security-relevant label across C1
and the datasets from Wu et al. [32] is available at Table 1.

García Núñez et al.
0.0 0.2 0.4 0.6 0.8 1.0
Recall0.00.20.40.60.81.0Precision
F2=0.4F2=0.6F2=0.7F2=0.8
DistilRoBERT a + LogReg (L2)  (F2=0.824)
MPNet + LogReg (Elastic)  (F2=0.774)
DistilRoBERT a + LogReg (Elastic)  (F2=0.741)
MPNet + MLP  (F2=0.723)
DistilRoBERT a + MLP  (F2=0.688)
MiniLM-L12 + LogReg (Elastic)  (F2=0.663)
RoBERT a-Large + LogReg (Elastic)  (F2=0.655)
MiniLM-L6 + LogReg (Elastic)  (F2=0.625)
TF-IDF + LogReg  (F2=0.542)
Figure 3: In-Distribution (C1 Held-Out) Precision/Recall of
All C2 Variants at the Fixed Operating Threshold of0.5.
C2 Results
The best in-distribution F2 overall is achieved by DistilRoBERTa +
LogReg (L2). However, as determined in Section 3, ℓ2variants are
treated as in-distribution references rather than deployment candi-
dates because they are more exposed to project-specific overfitting
on the two-project scope of C1. Within the transfer-oriented de-
ployment candidate set,MPNet + elastic-net logistic regression
achieves the highest held-out F2 and is therefore selected at thresh-
old0.5for integration into C3. On the held-out C1 test set ( 𝑛=58)
it reaches F2 =0.774, precision0 .650, recall0 .812, and ROC-AUC
0.903. This poses an improvement over the TF-IDF+LogReg base-
line (F2 =0.542, AUC0.750) and comparable to the other sentence-
embedder-based variants explored. Across all8sentence-encoder
variants, F2 clusters between0 .625and0.824with AUC between
0.839and0.939, suggesting that the task is well-posed and the sig-
nal is recoverable with any competitive text representation (see
Figure 3).
Cross-Distribution Generalizability.Table 2 summarizes mean G-
measure and cross-project standard deviation across two of our
models and the literature baselines.
Applied to the Wu et al. benchmarks [32], MPNet + LogReg
(Elastic) achieves mean G-measure =0.649(per-dataset: Ambari
0.716, Camel0 .58, Chromium0 .80, Derby0 .5, and Wicket0 .64),
with mean per-project recall of0 .771(ranging from0 .66to0.91),
despite a prevalence shift from27 .8%(C1) to1 .9%–17.9%(Wu et al.
datasets [32]).
Against the published baselines on the same benchmarks, MP-
Net + LogReg (Elastic)’s mean G-measure (0 .65) is on par with
the strongest classical ML baseline from Wu et al. [32] (FARSEC,
mean0.64), with a mixed per-project picture: it outperforms all of
Wu et al.’s classical ML baselines (FARSEC + RF/NB/LR/KNN/MLP)
on Ambari and Camel, is competitive on Chromium and Wicket
without exceeding the best FARSEC variants, and trails Wu et
al.’s baselines on Derby. Against França et al. [14], MPNet + Lo-
gReg (Elastic) outperforms every open-source GPT zero-shot and
classical ML baseline on Ambari and Wicket, ties with França’sTable 2: Per-Project G-Measure Heatmap Across All Evalu-
ated Models on the Five Wu et al. Benchmarks.
Paper Group Method Chromium Ambari Camel Derby Wicket Mean𝜎
Wu et al. FARSEC baselines Farsec 0.79 0.65 0.48 0.60 0.67 0.64 0.10
Wu et al. FARSEC baselines Farsec+TunedLearner 0.80 0.70 0.38 0.62 0.67 0.63 0.14
Wu et al. FARSEC baselines Farsec+TunedSMOTE 0.86 0.60 0.47 0.67 0.55 0.63 0.13
Wu et al. Text classifiers RF 0.84 0.40 0.54 0.63 0.65 0.61 0.14
Wu et al. Text classifiers NB 0.79 0.60 0.46 0.66 0.46 0.59 0.13
Wu et al. Text classifiers MLP 0.66 0.54 0.26 0.55 0.36 0.47 0.14
Wu et al. Text classifiers LR 0.73 0.40 0.26 0.55 0.23 0.43 0.19
Wu et al. Text classifiers KNN 0.73 0.40 0.26 0.55 0.23 0.43 0.19
Soltaniani et al. Prompted LLMs GPT-4.1 0.48 0.40 0.23 0.41 0.29 0.36 0.09
Soltaniani et al. Prompted LLMs Gemini-2.5 0.86 0.74 0.73 0.73 0.81 0.77 0.05
Soltaniani et al. Fine-tuned LLMs BERT 0.83 0.40 0.33 0.57 0.23 0.47 0.21
Soltaniani et al. Fine-tuned LLMs DistilBERT 0.85 0.32 0.33 0.56 0.47 0.51 0.19
Soltaniani et al. Fine-tuned LLMs DistilGPT2 0.88 0.12 0.04 0.18 0.04 0.25 0.32
Soltaniani et al. Fine-tuned LLMs Qwen2.5B 0.87 0.22 0.26 0.43 0.30 0.42 0.24
Alqahtani fasttext (10-fold CV) fasttext 0.93 0.81 0.67 0.80 0.77 0.80 0.08
França et al. GPT zero-shot Falcon - 0.55 0.49 0.45 0.48 0.49 0.03
França et al. GPT zero-shot Instruct - 0.61 0.48 0.44 0.45 0.50 0.07
França et al. GPT zero-shot Openorca - 0.54 0.53 0.54 0.60 0.55 0.03
França et al. GPT zero-shot Wizard - 0.50 0.40 0.47 0.48 0.46 0.04
França et al. ML random 80/20 RF - 0.04 0.59 0.68 0.62 0.48 0.26
França et al. ML random 80/20 LR - 0.34 0.53 0.66 0.53 0.52 0.11
França et al. ML random 80/20 SVM - 0.06 0.19 0.64 0.35 0.31 0.22
Ours embed + LogReg MPNet + LogReg (Elastic) 0.80 0.72 0.58 0.50 0.64 0.65 0.10
Ours embed + LogReg MiniLM-L12 + LogReg (Elastic) 0.74 0.66 0.63 0.60 0.63 0.65 0.05
strongest classical baseline (RF) on Camel, and trails the same base-
line on Derby. It is competitive with the best fine-tuned small LLMs
(BERT, DistilBERT) from Soltaniani et al. [28] while being a frozen-
encoder with a linear head (no LLM fine-tuning, orders of magni-
tude cheaper). MPNet + LogReg (Elastic) does not outperform the
best-performing Gemini 2.5 configuration from Soltaniani et al. [28].
Alqahtani’s [1] fastText in-distribution10-fold CV numbers are
also higher, though that comparison is not cross-distribution and
therefore not directly informative.
Cross-distribution stability is high: G-measure standard devi-
ation across the five projects ( 𝜎≈ 0.102) is comparable to FAR-
SEC (𝜎= 0.101), the strongest within-project baseline from Wu
et al. [32]. MiniLM-L12 + LogReg (Elastic), the most stable of our
variants, achieves 𝜎=0.046, the lowest cross-project standard de-
viation of all reported methods, while having a mean G-measure of
0.65, which matches MPNet + LogReg (Elastic)’s mean G-measure
of0.649. G-measure across fine-tuned small LLMs from Soltaniani
et al., which train on each project’s own data, exhibits substantially
higher standard deviation (BERT 𝜎=0.211, DistilBERT 𝜎=0.194,
Qwen2.5B𝜎=0.238), suggesting that per-project fine-tuning am-
plifies sensitivity to distributional shifts rather than suppressing
it.
C3: Preliminary Evaluation
We evaluated the pipeline (requirement extraction →bi-encoder re-
call→cross-encoder rerank →LLM semantic filter) through semi-
structured interviews with industry practitioners, using internal
security policies as the primary source corpus and CIS Benchmarks
(nginx, docker-ce) as a secondary corpus to probe generalization to
concrete best-practice security guidelines. This evaluation served
to provide preliminary performance evidence and to surface design
requirements.
Two semi-structured practitioner interview sessions are reported.
Together they covered24scored clauses and recommendations,
drawn from both corpora, across six backlog items. Items were se-
lected from the set flagged by C2 at confidence scores between0 .55
and0.93, spanning high-confidence and low-confidence regions of
the classifier’s output, hence representative of what the pipeline
would surface in practice.

From Backlog Items to Security Guidance: Towards Continuous Security Compliance
Retrieval Quality.The analysis of the retrieval quality is reported
over the24rated clauses, hence summarizing24ratings from two
raters over six backlog items. They thus characterize the study
sample and their role is to contextualize the qualitative findings
and to highlight failure modes. As presented in Table 3, ratings clus-
ter bimodally:12/24at≥4/5,7/24at≤1/5, and5/24in the2–3
band, a distribution consistent with, but not sufficient to confirm,
a pipeline whose shortlists are confident and whose applicability
is determined by item-security requirement granularity alignment
rather than by rater noise. Internal-policy clauses dominated the
high-rated band on items with well-structured descriptions, while
the CIS Benchmark shortlists showed greater within-shortlist stan-
dard deviation, concentrated on a single infrastructure item.
Table 3: Distribution of Requirements Relevance Ratings
Across the 24 Security Requirements Scored in the Interview
Sessions.
Rating band Count Share
≥4/5(directly applicable)12 50%
2–3/5(partial)5 21%
≤1/5(poor/irrelevant)7 29%
Failure modes.Two failure modes account for the low-rated
security requirements and are architecturally informative.Surface-
keyword mismatch.A JWT-generation item retrieved a password-
handling clause because both topics involve authentication creden-
tials at the embedding level, but the clause’s normative content
(“passwords must not be logged”) is inapplicable to JWT issuance.
Shortlist heterogeneity.On one infrastructure item the CIS Bench-
mark retrieval returned a strongly applicable recommendation
(rated5/5) alongside weakly applicable recommendations on adja-
cent topics (rated1/5) on the same shortlist, with the internal-policy
retrieval on the same item rated more precisely.
Top-ranked examples.Two items illustrate the two most informative
patterns in the case.
Item A— C2 confidence0 .86. A logging configuration task. Three
out of four retrieved clauses were rated ≥4/5. Practitioner com-
mentary described them as directly usable without paraphrasing.
The last clause was rated2 /5and was thus not described as rele-
vant. This is the pipeline’s best-case output: concrete task, rule-level
corpus, precise lexical and normative overlap.
Item B— C2 confidence0 .93. A scan and event notification con-
figuration task. Only one of the retrieved clauses from the internal
policy was rated≥4/5, the rest of them were rated ≤2/5and
described as not quite relevant to the source topic.
Both items lie in the top tercile of C2 confidence, matching the
deployment scenario in which the tool would provide information
to an engineer.
Practitioner Design Feedback.Valuable insights were gained from
the semi-structured interview, which are reported here. These in-
sights serve to justify future design decisions to integrate the tool
into further workflows.
In-backlog-system integration.Both practitioners identified
integration into the existing issue tracker systems as the primary
precondition for practical adoption. A standalone interface addstool-fatigue. Retrieval output surfaced as issue comments or side-
panel content can be consumed at the point where security planning
actually occurs.
Security Requirements Verbosity.Retrieved security require-
ments from both corpora are often too long for inline reading. The
recommended mitigation, using the same LLM stage to generate
a short, concrete action-item summary per retained requirement
with a link back to the authoritative security requirement, would
address this directly without modifying the retrieval architecture.
Corpus coverage.The one recurring gap report was that the
source corpus occasionally omits cross-cutting requirements. Backwards-
compatibility obligations flagged in one session are not expressed
as security requirements in any of the evaluated documents, con-
firming that the tool cannot surface what is not in the corpus.
5 Discussion
We interpret the results contribution by contribution: the design
choices and scope of the C1 dataset, the classifier-design rationale
and cross-distribution behavior of C2, and the architectural ratio-
nale, abstraction-level findings, and workflow-fit signals of C3.
C1: Dataset Quality and Scope
Wu et al. [32] document substantial label noise in community-
assigned security fields across the widely used Jira-based bench-
marks, and subsequent classifier work [1, 14, 28] inherits that noise
even after Wu’s cleaning pass. C1 trades scale for annotation rigor:
288items fully multi-rated by9security experts produce972inde-
pendent assessments at Fleiss’ 𝜅=0.787. The design target is to
complement Wu et al. ’s dataset with a high-precision training signal
whose labels were assigned by practitioners whose day-to-day role
is to decide exactly the kind of question the annotation asks. Wu
et al.’s datasets remain the larger-scale evaluation. C1 is the clean
training supervision.
The13.5%of items with at least one dissenting assessor, despite
substantial inter-rater agreement overall, confirm that security
relevance is not a sharp classification boundary. A natural gener-
alizability question for any expert-annotated resource is whether
a different expert pool would converge on similar labels. The an-
notators share a common organizational context, but that context
is broad due to the size of the industrial partner, which operates
across multiple highly regulated domains and is therefore subject
to a wide cross-section of security standards and regulations. This
implies that the practitioners’ day-to-day decisions span much of
the compliance landscape that other security-regulated organiza-
tions also face, making their experience representative of many
industries rather than narrowly organization-specific. The working
definition ofsecurity-relevantthat they apply is in turn shaped by
the same public standards (e.g., IEC 62443, CIS Benchmarks) that
those organizations follow, so it is not idiosyncratic to a single
industry.
The source corpus is intentionally narrow: two Atlassian en-
terprise projects whose item language resembles the industrial
backlogs targeted by C3. Datasets whose item descriptions are
tersely phrased or centered on consumer-facing bug reports sit
at a different item granularity and are better served by the Wu
family. The80security-related items also bound what future work

García Núñez et al.
can extract from C1 alone: frozen-encoder classifiers trained as
in C2 reach the ceiling of this signal comfortably, but multi-class
Firesmith-category models [12] would require augmentation or a
second annotation round.
Using C1 in practice.Beyond its role as training supervision for
C2, we see three concrete uses of the released dataset for industry
practitioners. First, as expert-labeled seed data to bootstrap an
organization-internal security-relevance classifier before internal
labels exist, following the C2 procedure of a frozen encoder with
a lightweight head. Second, as an audit set. Organizations already
operating a triage classifier or LLM-based filter can evaluate it
against expert consensus labels rather than community-assigned
ones. Third, as realistic material for security training. The items
are industry-style backlog texts with expert consensus security-
relevance labels, which teams can use to align their own working
definition of security relevance.
C2: Classifier Design Decisions
With288annotated items, full fine-tuning of a contemporary trans-
former risks overfitting to the two source projects rather than
learning a transferable notion of security-relevance. Linear-head
coefficients are also inspectable: security-filtering decisions are
reviewed internally, and an auditable decision path matters for
trust and for error analysis. F2 as the selection criterion encodes
the corresponding cost asymmetry fairly, since a missed security
requirement is more expensive than a false alarm that C3 subse-
quently filters through its semantic stage.
Generalizability evaluation.The generalizability results should be
interpreted as evidence of transferability. The deployed C2 model
is evaluated fully zero-shot across domains: trained only on C1 and
then applied unchanged to the Wu et al. benchmarks, whereas most
published baselines are evaluated in-distribution, with training
and test data drawn from the same project. Matching or approach-
ing those baselines is therefore a comparatively demanding target
rather than a neutral head-to-head comparison.
The comparison against prompted LLMs should also be read
cautiously. Although they are evaluated in a nominally zero-shot
setting, their pretraining data may plausibly already contain parts
of these widely used benchmark projects, potentially resulting in
overly optimistic results [9, 7]. We therefore treat competitive per-
formance on Wu et al. as a positive result showing that C2 has
learned a transferable notion ofsecurity-relevant.
The reported superior performance of Gemini by Soltaniani et
al. [28], while impressive, cannot be perceived as a stable base-
line given recent insights on the non-reproducibility of research
performed with commercial LLMs [2].
The tight clustering of in-distribution F2 across eight sentence-
encoder-based combinations (F2 ∈[0.625,0.824]) suggests that the
task is well-posed and recoverable by any competitive text represen-
tation. The bottleneck here is label quality. DistilRoBERTa + LogReg
(L2) has the highest held-out F2 overall, but was not treated as a de-
ployment candidate because on a two-project training resource its ℓ2
head is more exposed to project-specific overfitting than the sparser
elastic-net alternatives. Among the transfer-oriented deployment
candidates, MPNet + LogReg (Elastic) has the highest held-out F2and was therefore selected for integration into the C3 retrieval
system. The cross-distribution comparison uses mean G-measure
as its primary transfer metric and across-project standard deviation
𝜎as a secondary stability metric. Under that criterion, MPNet +
elastic-net logistic regression remains a strong deployment choice,
but the stability comparison also surfaces a real trade-off: MiniLM-
L12 + elastic-net logistic regression achieves essentially identical
mean G-measure (0 .650vs.0.649) with half the cross-project stan-
dard deviation ( 𝜎=0.046vs.0.102). The fact that both variants are
competitive with the strongest within-project baselines on stability,
and that fine-tuned small LLMs are considerably less consistent
despite per-project training supervision, supports the interpreta-
tion that the classifier’s stability derives from the transferability of
the learned representation. In deployments where cross-domain
consistency is the primary concern, MiniLM-L12 + LogReg (Elastic)
would therefore be the preferred variant.
Probability calibration is deliberately deferred: C2 is used down-
stream as a hard filter into C3, where calibration does not affect
retrieval-pipeline behavior. This would matter only if C2 output
fed a downstream ranking, which is not the case in the current
deployment.
C3: Security Requirement Enrichment
We discuss C3 along its design rationale, the practical value to
developer workflows observed in the preliminary evaluation, and
four findings that emerged in individual walkthroughs.
Design rationale.Two-stage C2–C3 system.Separating the high-
recall relevance filter from the precision-oriented retrieval stage
matches the asymmetric cost of each stage and lets each subsystem
be tuned independently: C2 maximizes recall of security-relevant
items, C3 maximizes precision of clause grounding. The LLM work-
load is therefore proportional to the number of security-relevant
items, not to the full backlog size.
RAG over pure LLM IR.The bi-encoder shortlist imposes a document-
grounded candidate set that the LLM filter cannot hallucinate out
of, while the LLM filter rejects topically adjacent but non-applicable
candidates that dense similarity alone cannot distinguish.
Requirement extraction preprocessing.Introduced after we ob-
served a systematic failure mode where multi-concern backlog
items diluted the retrieval signal of any single concern. Decompos-
ing into atomic requirements before retrieval recovers concerns
that the undecomposed retrieval pipeline misses.
Open-weights LLMs.Using on-premises open-weights models for
both extraction and filtering satisfies two deployment constraints
simultaneously: backlog contents cannot be sent to third-party
inference services, and pinning a specific model version prevents
silent behavior drift as commercial models are updated. More of
the benefits of open LLMs can be found in the LLM guidelines by
Baltes et al. [7].
Practical value.The tool surfaces implicit security requirements that
would otherwise require manual expert review, delivering security
requirements at the point of planning rather than as an after-the-
fact compliance check, potentially addressing the lead-time burden
identified by Moyón et al. [20]. In the preliminary evaluation, both
practitioners responded positively to this framing. The tool was

From Backlog Items to Security Guidance: Towards Continuous Security Compliance
perceived as useful at the item level, provided it integrates into
existing workflows rather than introducing a parallel interface.
Abstraction-level mismatch.Across both interview sessions, the
granularity gap between security requirement documents and back-
log items appeared to be the strongest predictor of retrieval useful-
ness. Rule-level documents, such as organizational security policies
and implementation-level benchmarks such as CIS Benchmarks,
operate below the capability abstraction of standards such as IEC
62443-3-3 [15], and yielded actionable matches in both cases. How-
ever, the within-shortlist variance of CIS Benchmark retrievals was
higher, consistent with their narrower implementation scope mak-
ing them precise for specific infrastructure items but less broadly
applicable across item types. Higher-abstraction capability stan-
dards, such as IEC 62443-3-3, produced retrievals that were topically
correct but rarely directly actionable per item. The semantic gap
between such control and a backlog item describing a specific fea-
ture is too large for retrieval to bridge reliably. If confirmed by the
upcoming complete evaluation, this finding has a direct implica-
tion for deployment. The retrieval pipeline should be paired with
documents whose security requirements granularity matches the
typical specificity of the target backlog.
A secondary observation is that retrieval shortlist quality appears
to track backlog-item granularity itself. Coarse epics retrieve many
weakly relevant security requirements, while atomic tasks retrieve
fewer but more strongly relevant ones. This suggests a potential
secondary use of C2 + C3 output as a proxy signal for backlog-
structuring quality, though we do not validate this claim in this
paper.
Workflow fit.Both practitioners independently identified in-backlog-
system integration as the primary precondition for practical adop-
tion. Standalone interfaces were perceived to add tool-fatigue. Re-
trieval output surfaced as issue comments or side-panel content
can be consumed at the point where planning actually happens.
This is considered the most consistent qualitative finding of the
interview study.
Security requirement verbosity was the dominant usability con-
cern. Retrieved clauses from both internal policies and public bench-
marks are often too long for inline reading. A natural next engineer-
ing step is to use the same LLM stage to generate a short, concrete
action-item summary per retained security requirement, with a
link back to the authoritative text for verification, an addition that
has been implemented but must be evaluated. Perceived usefulness
also appeared to correlate with source-item quality. Items with
structured descriptions produced higher-rated retrievals than items
with terse, unstructured titles.
Missing items and concerns.The interviews produced few system-
atic gap reports. The one recurring finding was that the source
corpus occasionally omits cross-cutting requirements. Backwards-
compatibility obligations flagged in one session are not expressed
as security clauses in any of the evaluated standards, and the tool
cannot surface what is not in the corpus. The extractor continues
to have difficulties with multi-concern items, which is why further
multi-concern decomposition is an open direction. This limitation
may also be mitigated by establishing guidelines for backlog itemdescriptions, since more detailed items may provide better source
material for retrieving relevant security requirements.
Generalizability.The C2 classifier is trained entirely on open-source
TAWOS data and carries no organization-specific knowledge, so
adopters applying C3 to their own internal security policies need
no re-annotation or retraining. The strongest generalizability claim
for C3, that the retrieval pipeline performs consistently across
standards and organizations, is part of future work. The CIS Bench-
mark evaluation partially addresses this by demonstrating consis-
tent pipeline behavior on a public security requirements document
against the same items used for the internal-policy evaluation, but
a second independent industrial corpus would be needed to con-
vert the architectural document-agnosticism claim into a measured
result.
6 Threats to Validity
We discuss the threats to validity of our research following the
categorization of Runeson and Höst [26].
6.1 Construct validity
The constructsecurity-relevanceis operationalized through expert
judgment rather than as an approximation of an objective ground
truth. Different expert pools might apply a different decision bound-
ary, and the13 .5%of C1 items with at least one disagreeing assessor
confirm that the construct admits legitimate boundary cases, as
discussed in Section 5. We mitigate this by multi-rating every re-
leased item, and by reporting Fleiss’ 𝜅and raw observed agreement
rather than a single reliability number. For C3,retrieval usefulness
is characterized in two layers: an artifact-level analysis of shortlists
(applicability relative to the retrieving item’s requirement) and a
practitioner0–5rating used as calibration of that analysis.
6.2 Internal validity
C2 thresholds and hyperparameters are selected on a single C1
validation split. The tight spread of F2 across the sentence-encoder
variants bounds the risk of threshold-to-split overfitting but does
not eliminate it, and probability calibration is not performed be-
yond threshold choice because C2 is used downstream as a hard
filter into C3. The held-out C1 test set contains only58items, so
the reported in-distribution point estimates probably carry wide
confidence intervals and, taken alone, cannot rule out overfitting
to the two source projects. We mitigate this by not relying on the
in-distribution split for the headline claim. Deployment selection is
restricted to the transfer-oriented candidate set, and the zero-shot
evaluation on the five Wu et al. benchmarks [32], spanning several
thousand items from unrelated projects, provides the primary evi-
dence that the learned notion of security relevance generalizes. The
C3 evaluation is preliminary and performed over two industrial
product backlogs and two practitioner interviews. Observations are
descriptive and the apparent convergences across the two sessions
are consistent with, but do not establish generalizable empirical
evidence. The complete evaluation, involving more practitioners
per backlog, multiple backlogs, ideally across organizations, and a
study design capable of measuring workflow impact rather than
perceived relevance, is deferred to future work. Item-level famil-
iarity is recorded as descriptive context. The interviews target the

García Núñez et al.
engineer-consulting scenario in which the consumer of a security
requirement is not necessarily the author of the corresponding
backlog item. Prior familiarity with a specific backlog item does
not, however, disqualify a practitioner’s judgment of whether a
retrieved security requirement is applicable.
6.3 External validity
C1 is drawn from two Atlassian open-source projects whose item
language resembles industrial enterprise backlogs. It is a precision
resource by design and does not aim to cover the broader variance
of public issue trackers (Wu et al.’s benchmarks fill that role). The
nine C1 annotators share an organizational context, which could
bias the operational definition ofsecurity-relevant. We argue in
Section 5 that this context is operationally shaped by the same
standards other security-regulated organizations follow.
6.4 Reliability
Both the C3 requirement extractor and the semantic filter are open-
weights LLMs, and both stages introduce run-to-run variability in
the resulting shortlist. We mitigate this by fixing the model identi-
fier, sampling temperature, and prompts, and by reporting averaged
behavior. We do not claim bitwise reproducibility of the C3 output.
Although we do not quantify run-to-run variance formally, repeated
executions during development informally suggested that the C3
pipeline was practically consistent: for the same backlog item and
source corpus, it reliably returned the same clauses and only small
differences in relevance scores. C1 and the full C2 pipeline are fully
reproducible from the released artifacts and are deterministic up
to library-level floating-point variation. The C3 cross-distribution
probe on CIS Benchmarks is reproducible end-to-end against public
documents. The internal-policy evaluation is reproducible only on
the internal corpus.
7 Conclusion
We set out to reduce the security workload in continuous security
compliance by providing security requirements directly on the
backlog items engineers and security experts already use. The three
contributions of this paper address complementary parts of that
goal.
C1 provides an independently expert-annotated backlog-security
dataset:288items annotated by a pool of nine security practitioners,
resulting in972independent binary assessments, Fleiss’ 𝜅=0.787,
and additional Firesmith security-requirement metadata available
for many items. C1 is positioned as a high-precision complement to
the larger Wu et al. benchmark family [32], and is released publicly
to allow downstream work on security-relevance detection to train
and audit against expert supervision.
C2 is a recall-oriented binary classifier for security-relevance. A
frozen MPNet encoder paired with an elastic-net logistic-regression
head reaches F2 =0.774on the C1 held-out split and, applied zero-
shot to the five Wu cleaned benchmarks, achieves cross-distribution
G-measure≈0.65, competitive with the published classical-ML and
open-source-GPT baselines despite a prevalence shift from27 .8%
on C1 to1.9–17.9%on the Wu projects.
C3 is a four-stage retrieval-augmented generation pipeline that
maps a flagged backlog item to the subset of security requirementsfrom relevant security documents. The C3 retrieval pipeline was
evaluated preliminarily on two industrial product backlogs with
industrial internal policies as the primary corpus and CIS Bench-
marks as a public-standard probe. Semi-structured practitioner in-
terviews produced a positive preliminary signal on items with well-
structured descriptions, identified integration into backlog systems
and security requirement summarization as the most important
additions, and suggested a consistent lesson about the granularity
gap between security requirement documents and backlog items.
Rule-level documents yield more precise, actionable matches than
higher-abstraction capability standards.
Taken together, the results provide preliminary evidence that
security-relevance can be decided directly from ordinary backlog
text to support engineering work through concrete security re-
quirements attached to those items in a way that potentially makes
security both more visible and better integrated in the software
engineering lifecycle, without requiring engineers to leave their
existing workflow.
Open directions follow directly from the preliminary evalua-
tion. First, a complete evaluation of C3 is required, including more
practitioner interviews, multiple backlogs, ideally across organiza-
tions, and a controlled study design capable of measuring workflow
impact and adoption. This would be achieved through a longitu-
dinal study. Second, applying C3 to a second industrial corpus,
and ideally to a second organization’s policies, would convert the
current document-agnostic architectural claim into a measured gen-
eralizability result. Third, in-tracker integration and per-security
requirements action-item summarization are concrete engineering
steps that the practitioners identified as the dominant preconditions
for practical adoption.
8 Data Availability Statement
We provide the artifacts of this research in two parts: the final C1
dataset covering all 288 items with aggregated labels and associated
metadata at https://doi.org/10.5281/zenodo.21450745, and the code
and model artifacts for C2 and the sanitized C3 retrieval pipeline
at https://doi.org/10.5281/zenodo.21450906. Due to confidentiality
agreements, we do not release the segmenter for the internal policy
documents or the internal policies.
References
[1] Sultan S. Alqahtani. 2024. Security bug reports classification using fasttext.
International Journal of Information Security, 23, 2, 1347–1358. doi:10.1007/s102
07-023-00793-w.
[2] Florian Angermeir, Maximilian Amougou, Mark Kreitz, Andreas Bauer, Matthias
Linhuber, Davide Fucci, Fabiola Moyón C., Daniel Mendez, and Tony Gorschek.
2026. Reflections on the reproducibility of commercial llm performance in
empirical software engineering studies. InIEEE/ACM 48th International Confer-
ence on Software Engineering(ICSE ’26). Association for Computing Machinery,
New York, NY, USA. doi:10.1145/3744916.3773207.
[3] Florian Angermeir, Jannik Fischbach, Fabiola Moyón, and Daniel Mendez. 2024.
Towards automated continuous security compliance. InProceedings of the 18th
ACM/IEEE International Symposium on Empirical Software Engineering and
Measurement(ESEM ’24). Association for Computing Machinery, Barcelona,
Spain, 440–446.isbn: 9798400710476. doi:10.1145/3674805.3690748.
[4] Ron Artstein and Massimo Poesio. 2008. Survey article: inter-coder agreement
for computational linguistics.Computational Linguistics, 34, 4, 555–596. doi:10
.1162/coli.07-034-R2.
[5] Vanessa Ayala-Rivera and Liliana Pasquale. 2018. The Grace Period Has Ended:
an approach to operationalize GDPR requirements. In2018 IEEE 26th Interna-
tional Requirements Engineering Conference (RE), 136–146. doi:10.1109/RE.2018
.00023.

From Backlog Items to Security Guidance: Towards Continuous Security Compliance
[6] Vanessa Ayala-Rivera, Andres Omar Portillo-Dominguez, and Liliana Pasquale.
2024. GDPR compliance via software evolution: weaving security controls in
software design.Journal of Systems and Software, 216, 112144. doi:10.1016/j.jss
.2024.112144.
[7] Sebastian Baltes et al. 2025. Guidelines for Empirical Studies in Software Engi-
neering involving Large Language Models. (2025). https://arxiv.org/abs/2508.1
5503 arXiv: 2508.15503[cs.SE].
[8] Center for Internet Security. 2024. CIS Benchmarks. https://www.cisecurity.or
g/cis-benchmarks/. (2024).
[9] Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Co-
han. 2024. Investigating data contamination in modern benchmarks for large
language models. InProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers). Association for Computational Linguistics,
Mexico City, Mexico, (June 2024), 8706–8719. doi:10.18653/v1/2024.naacl-long
.482.
[10] European Parliamentary Research Service. 2022. EU Cyber Resilience Act. Tech.
rep. Briefing, European Parliamentary Research Service. European Parliament.
https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI%282022
%29739259.
[11] António Ferreira, Alberto Silva, and Ana Paiva. 2022. Towards the art of writing
agile requirements with user stories, acceptance criteria, and related constructs.
In17th International Conference on Evaluation of Novel Approaches to Software
Engineering. (Jan. 2022), 477–484. doi:10.5220/0011082000003176.
[12] Donald G. Firesmith. 2004. Specifying reusable security requirements.Journal
of Object Technology, 3, 1, 61–75. doi:10.5381/jot.2004.3.1.a3.
[13] Brian Fitzgerald and Klaas-Jan Stol. 2017. Continuous software engineering: a
roadmap and agenda.Journal of Systems and Software, 123, 176–189. doi:https:
//doi.org/10.1016/j.jss.2015.06.063.
[14] Higor França, Katerina Goseva-Popstojanova, Cássio Teixeira, and Nuno Laran-
jeiro. 2025. Gpts are not the silver bullet: performance and challenges of using
gpts for security bug report identification.Information and Software Technology,
185, 107778. doi:10.1016/j.infsof.2025.107778.
[15] 2013. Industrial communication networks – network and system security – part
3-3: system security requirements and security levels. Geneva, Switzerland:
International Electrotechnical Commission, (2013).
[16] International Electrotechnical Commission. 2018. Security for Industrial Au-
tomation and Control Systems – Part 4-1: Secure Product Development Lifecy-
cle Requirements. Tech. rep. IEC 62443-4-1:2018. International Electrotechnical
Commission. https://webstore.iec.ch/publication/33615.
[17] 2018. Iso/iec/ieee 26515:2018 systems and software engineering – developing
information for users in an agile environment. ISO/IEC/IEEE, (2018).
[18] Patrick Lewis et al. 2020. Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Information Processing Systems.
Vol. 33. Curran Associates, 9459–9474.
[19] Fabiola Moyón, Pamela Almeida, Daniel Riofrío, Daniel Mendez, and Marcos
Kalinowski. 2020. Security compliance in agile software development: a system-
atic mapping study. In2020 46th Euromicro Conference on Software Engineering
and Advanced Applications (SEAA), 394–401. doi:10.1109/SEAA51224.2020.0007
1.
[20] Fabiola Moyón, Florian Angermeir, and Daniel Mendez. 2024. Industrial chal-
lenges in secure continuous development. InProceedings of the 46th Interna-
tional Conference on Software Engineering: Software Engineering in Practice
(ICSE-SEIP ’24). Association for Computing Machinery, Lisbon, Portugal, 309–
311.isbn: 9798400705014. doi:10.1145/3639477.3639736.
[21] Fabiola Moyón, Kristian Beckers, Sebastian Klepper, Philipp Lachberger, and
Bernd Bruegge. 2018. Towards continuous security compliance in agile soft-
ware development at scale. InProceedings of the 4th International Workshop on
Rapid Continuous Software Engineering(RCoSE ’18). Association for Computing
Machinery, Gothenburg, Sweden, 31–34.isbn: 9781450357456. doi:10.1145/319
4760.3194767.
[22] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT.
(2019). arXiv: 1901.04085[cs.IR].
[23] Indra Kharisma Raharjana, Daniel Siahaan, and Chastine Fatichah. 2021. User
stories and natural language processing: a systematic literature review.IEEE
Access, 9, 53811–53826. doi:10.1109/ACCESS.2021.3070606.
[24] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: sentence embeddings
using siamese BERT-networks. InProceedings of the 2019 Conference on Em-
pirical Methods in Natural Language Processing(EMNLP ’19). Association for
Computational Linguistics, 3982–3992. doi:10.18653/v1/D19-1410.
[25] Marcela Ruiz, Jin Yang Hu, and Fabiano Dalpiaz. 2023. Why don’t we trace? a
study on the barriers to software traceability in practice.Requirements Engi-
neering, 28, 619–637. doi:10.1007/s00766-023-00408-9.
[26] Per Runeson and Martin Höst. 2009. Guidelines for conducting and reporting
case study research in software engineering.Empirical Software Engineering,
14, 2, 131–164. doi:10.1007/s10664-008-9102-8.
[27] Pattaraporn Sangaroonsilp, Morakot Choetkiertikul, Hoa Khanh Dam, Chaiy-
ong Ragkhitwetsagul, and Aditya Ghose. 2023. An empirical study of automatedprivacy requirements classification in issue reports.Automated Software Engi-
neering, 30, 20. doi:10.1007/s10515-023-00387-9.
[28] Farnaz Soltaniani, Shoaib Razzaq, and Mohammad Ghafari. 2026. Evaluating
large language models for security bug report prediction. (2026). https://arxiv
.org/abs/2601.22921 arXiv: 2601.22921[cs.CR].
[29] Chakkrit Tantithamthavorn, Ahmed E. Hassan, and Kenichi Matsumoto. 2020.
The impact of class rebalancing techniques on the performance and interpreta-
tion of defect prediction models.IEEE Transactions on Software Engineering, 46,
11, 1200–1219. doi:10.1109/TSE.2018.2876537.
[30] Vali Tawosi, Federica Sarro, Ned Petric-Gray, and Mark Harman. 2022. TAWOS:
a large-scale agile project dataset for software engineering research. InPro-
ceedings of the 19th International Conference on Mining Software Repositories
(MSR ’22). Association for Computing Machinery, New York, NY, USA, 75–79.
doi:10.1145/3524842.3528494.
[31] Sven Türpe and Andreas Poller. 2017. Managing security work in scrum: ten-
sions and challenges. InSecSE@ESORICS. https://api.semanticscholar.org/Corp
usID:4933742.
[32] Xiaoxue Wu, Wei Zheng, Xin Xia, and David Lo. 2022. Data quality matters: a
case study on data label correctness for security bug report prediction.IEEE
Transactions on Software Engineering, 48, 7, 2541–2556. doi:10.1109/TSE.2021.3
063727.
[33] Hui Zou and Trevor Hastie. 2005. Regularization and variable selection via the
elastic net.Journal of the Royal Statistical Society: Series B (Statistical Methodol-
ogy), 67, 2, 301–320. doi:10.1111/j.1467-9868.2005.00503.x.
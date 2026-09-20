# Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction

**Authors**: Yuanzhe Jia, Ali Anaissi

**Published**: 2026-09-17 03:05:17

**PDF URL**: [https://arxiv.org/pdf/2609.19615v1](https://arxiv.org/pdf/2609.19615v1)

## Abstract
Modern applications generate massive volumes of raw telemetry data, but translating those noisy, heterogeneous event streams into actionable business insights remains a fundamental challenge. Data engineers and analysts expend substantial effort reconciling semantic discrepancies, hand-crafting parsing logics, and maintaining fragile mappings between raw data and business KPIs. In this paper, we present an end-to-end framework that fully automates the construction of a business semantic layer from application raw logs. Our approach introduces a two-stage semantic abstraction: first, high-level business features are identified via LLM inference augmented with domain-specific industry knowledge; second, fine-grained business nodes are derived through a structured pipeline comprising data refinement, hybrid retrieval, multi-stage filtering, semantic clustering, and canonical naming. Evaluation on production-scale telemetry demonstrates that our system improves human-assessed semantic quality from 50 to 80+ on a 100-point scale, reduces maintenance effort by 80%, filters out 74% of noise, and achieves 0.87 Cohen's kappa via an integrated LLM-as-Judge evaluation, enabling continuous, scalable quality assurance. Overall, our work distinguishes itself from prior work by addressing the novel problem of business semantic layer induction from raw telemetry, operating without labeled training data or manual rule engineering.

## Full Text


<!-- PDF content starts -->

Semantic Layer Induction from Raw Telemetry
via Hierarchical LLM and RAG Abstraction
Yuanzhe Jia1, Ali Anaissi1,2
1University of Sydney, Australia
2University of Technology Sydney, Australia
yjia5612@uni.sydney.edu.au, ali.anaissi@uts.edu.au
Abstract.Modernapplicationsgeneratemassivevolumesofrawteleme-
try data, but translating those noisy, heterogeneous event streams into
actionable business insights remains a fundamental challenge. Data en-
gineers and analysts expend substantial effort reconciling semantic dis-
crepancies, hand-crafting parsing logics, and maintaining fragile map-
pings between raw data and business KPIs. In this paper, we present an
end-to-end framework that fully automates the construction of a busi-
ness semantic layer from application raw logs. Our approach introduces
a two-stage semantic abstraction: first, high-level business features are
identified via LLM inference augmented with domain-specific industry
knowledge; second, fine-grained business nodes are derived through a
structured pipeline comprising data refinement, hybrid retrieval, multi-
stage filtering, semantic clustering, and canonical naming. Evaluation
on production-scale telemetry demonstrates that our system improves
human-assessed semantic quality from 50 to 80+ on a 100-point scale,
reduces maintenance effort by 80%, filters out 74% of noise, and achieves
0.87 Cohen’s kappa via an integrated LLM-as-Judge evaluation, enabling
continuous,scalablequalityassurance.Overall,ourworkdistinguishesit-
selffrompriorworkbyaddressingthenovelproblemofbusinesssemantic
layer induction from raw telemetry, operating without labeled training
data or manual rule engineering. The relevant code is publicly available
on GitHub1.
Keywords:Semantic Layer, Hierarchical Abstraction, Retrieval Aug-
mented Generation, LLM-as-Judge
1 Introduction
Modernsoftwareplatformsgeneratepetabytesofrawtelemetrydataeveryday—
user interactions, backend events, and API calls—capturing the full spectrum
of system activity. It is the lifeblood of data-driven decision-making, powering
conversion funnels, product analytics, and KPI monitoring. However, organiza-
tions consistently struggle to extract reliable business insights from this kind of
data. The challenge lies not only in data volume but also in semantic heterogene-
ity. The same action—say, "users search for a product"—may be tracked across
1https://github.com/yuanzhe-jia/semantic-layer
arXiv:2609.19615v1  [cs.CL]  17 Sep 2026

2 Yuanzhe Jia1, Ali Anaissi1,2
platforms and versions assearch_click(Android),search_submit(IOS) and
button_click(Web), each with different parameters and schema. This frag-
mentation forces teams into an endless cycle of manual mapping, custom SQL
logic per dashboard, and cross-functional debate about "what the telemetry data
actually means". The operational cost is substantial: a typical enterprise data
platform may maintain thousands of custom parsers, consuming thousands of
engineer-hours monthly.
Existing solutions fall short. Manual parser construction, while precise, is
brittle and does not scale across heterogeneous log formats. Syntax-based pars-
ing methods can extract templates but lack business semantic understanding.
Supervised learning approaches require extensive labeled data for each tracking
point, making them impractical for rapidly evolving applications. Even recent
LLM-based approaches to log parsing focus on syntax-level template extraction
rather than semantic mapping to business concepts. We argue that what enter-
prises require is not merely structured logs, but a business semantic layer—a
stable, canonical mapping from raw telemetry to human-interpretable insights
that directly align with customer needs and business objectives. In this paper,
we present an LLM-powered framework for business semantic layer induction.
The rest of the paper is structured as follows: Section 2 reviews and critiques
existing approaches in log parsing, semantic data management, and RAG-based
structured extraction; Section 3 details our hierarchical abstraction framework
and its algorithmic implementation; Section 4 describes the dataset, experimen-
tal setup, evaluation metrics, and empirical results; and Section 5 concludes the
paper with a summary of contributions and directions for future work.
2 Related Work
2.1 Log Parsing and Telemetry Analysis
Automated log parsing has been extensively studied in the systems community.
Traditional approaches use clustering or frequent pattern mining to extract log
templates. Methods like Drain [3] and LogParser [11] achieve high template ex-
tractionaccuracybutfocusonsyntax-levelpatterns—theyidentifywhatchanges
across log lines but do not understand what those changes semantically repre-
sent. More recent LLM-based parsers, such as LogParser-LLM [13], demonstrate
superior performance by seamlessly blending semantic insights with statistical
nuances, obviating the need for hyper-parameter tuning and labeled training
data while ensuring rapid adaptability through online parsing. However, these
approaches still focus on template extraction and field naming rather than map-
ping logs to semantic concepts. Similarly, Matryoshka et al. [7] use LLMs to
generate semantically-aware log parsers by inferring log syntax, variable nam-
ing, and schema normalization. Although impressive, they focus on mapping log
fields to standardized security schema for threat detection. In process mining,
researchers have explored semantics-aware event log analysis using LLMs [8], but
these approaches typically analyze existing logs rather than constructing them
from raw telemetry.

Title Suppressed Due to Excessive Length 3
2.2 Semantic Data Management
Beyond log parsing and telemetry analysis, the data management community
has long investigated semantic enrichment of enterprise data assets. Hoseini et
al. [4] provide a comprehensive survey on semantic data management in data
lakes, covering ontology-based data access and semantic modeling approaches
that link metadata to knowledge graphs. In parallel, significant research has fo-
cused on knowledge graph construction as a means of structuring and organizing
business semantics. Bian et al. [1] survey LLM-empowered knowledge graph con-
struction, analyzing how LLMs reshape ontology engineering and knowledge ex-
traction pipelines, while Zhao et al. [12] review machine learning approaches for
entity and ontology learning. From a metadata management perspective, recent
surveys on data catalog tools by Kropshofer et al. [5] and Tonnarelli et al. [9] ex-
amine how technical metadata annotated with domain knowledge improves data
accessibility and interoperability. However, these approaches rely on pre-defined
ontologies and struggle with the heterogeneity of multi-platform naming conven-
tion, requiring additional translation to aggregate fine-grained graph relations
into business KPIs. None address the unique challenge of inducing hierarchical
business semantics directly from noisy, high-volume telemetry.
2.3 RAG for Structured Data Extraction
RAG has been increasingly applied to structured knowledge extraction and
domain-specificreasoningtasks.EventRAG[10]introducesanevent-centricRAG
framework that constructs event knowledge graphs from narrative documents to
enhance LLM generation with structured event semantics and temporal rea-
soning. TM-RAG [14] employs ontology-guided graph retrieval with a timeline
ontology for automated construction claim report generation, demonstrating the
effectiveness of structured knowledge organization in domain-specific RAG sys-
tems. GenDFIR [6] applies RAG to cyber incident timeline analysis, retrieving
relevant forensic events from a structured knowledge base to support investi-
gation. While these approaches share insights that event-centric organization
and structured retrieval improve reasoning, they are designed for task-specific,
one-off query answering. In contrast, our method uses RAG to retrieve can-
didate mapping rules for each business capability with the distinct objective
of constructing a reusable, generalizable semantic layer that can serve diverse
downstream analytics without task-specific re-engineering.
3 Methodology
3.1 Problem Formulation
We formalize the semantic layer induction problem as follows:
– Given: A raw telemetry event corpusE={e 1, . . . , e N}, where eache iis a
tracking event that consists of a name and a set of key-value conditions; and
an optional industry taxonomyT.

4 Yuanzhe Jia1, Ali Anaissi1,2
– Find: A semantic layerS={(f j,Nj)}, wheref jis a business feature, and
Nj={n j,1, . . . , n j,K}are business nodes, such thatSmaximizes a semantic
coherence objective while minimizing feature sparsity.
Algorithm 1The proposed framework
Require:raw telemetryE, industry priorT, top-kthreshold
Ensure:semantic layerS
1:Erefined ←RefineEvents(E,T){Stage 1}
2:F ←IdentifyFeatures(E refined ,T){Stage 2}
3:foreach featuref∈ Fdo
4:R f←RuleRetrieve(f,E refined ){Stage 3}
5:Rclean
f←FilterCandidateRules(R f){Stage 4}
6:N f←ClusterAndNameNodes(Rclean
f ){Stage 5}
7:end for
8:S ←S
f(f,N f)
9:returnS
Astraightforwardapproachtothisproblemistolearnaflatmappingdirectly
from raw telemetry data to business labels. However, such a strategy suffers
from several fundamental limitations: the mapping space is large; semantically
similar events may be scattered across unrelated labels; and the resulting map-
pings are brittle to platform-specific naming conventions. Our proposed model
solves this problem via two-level hierarchy (see Algorithm 1). This hierarchy
(Feature→Node) serves as an inductive bias that constrains the search space:
the framework first reasons about high-level business capabilities, and then re-
fines each capability into its constituent actions. By decoupling the problem into
two nested subproblems, the hierarchy reduces the effective complexity of the
mapping task and enables the system to leverage industry priors at the feature
level before committing to fine-grained node assignments. Additionally, unlike a
naive pipeline, the model framework maintains a shared latent semantic space
across stages: representations learned in data refinement directly influence fea-
ture identification, which in turn constrains the retrieval space via a feedback
loop, ensuring that downstream errors do not cascade catastrophically.
3.2 Stage 1: Data Preparation
Raw telemetry data often includes high-cardinality fields and semantically weak
columns that hinder effective retrieval. To address this, we perform a two-step
data refinement process.
Column Importance Scoring:We first identify columns that carry significant
business context from application telemetry, which is typically stored as tracking
event logs, where each event contains a name and numerous metadata columns
(e.g.,url_path,page_title,element_id).Foreachmetadatacolumn,weprompt

Title Suppressed Due to Excessive Length 5
an LLM with statistical profiles (e.g., null ratio, distinct count) and a predefined
business taxonomy to assign an importance score ranging from 1 to 5. Columns
falling below a calibrated threshold are excluded from subsequent processing, ef-
fectively pruning irrelevant attributes that would otherwise introduce noise into
semantic matching. This step constitutes a lightweight but effective schema-level
filter that prioritizes business-relevant dimensions over purely technical or tran-
sient fields.
Enumeration Normalization:For each column retained after the importance
scoring, we tokenize its enumeration values using common delimiters (e.g., "/",
whitespace)toobtainfine-grainedtokens.Wethenapplyarandomstringdetector—
a entropy-based classifier augmented with LLM-based pattern recognition—to
identify transient identifiers such as UUIDs, session tokens, and request IDs.
These detected strings are replaced with a uniform mask symbol ("*"), effec-
tively stripping away instance-specific noise. Subsequently, we consolidate enu-
merations that share identical masked patterns via a set of regular expression
rules, merging semantically equivalent variants into a canonical representation.
The entire process yields a clean, low-cardinality vector space. Finally, we aggre-
gate identical cleaned records to produce a compact set of raw telemetry data,
and the refined data will serve as the input for subsequent feature identification
and mapping retrieval stages.
3.3 Stage 2: Business Feature Identification
We leverage an LLM to generate a set of high-level business features from the re-
fined data. Given that the data volume after Stage 1 remains prohibitively large
for direct LLM consumption, we first perform a stratified sampling to obtain
a representative subset that preserves the diversity of data patterns and their
frequency distribution. The sampling strategy ensures the LLM operates within
its context window while maintaining sufficient coverage of the application’s be-
havioral landscape. To ensure the generated features are both comprehensive
and industry-relevant, we augment the LLM context with domain-specific ref-
erence materials. Specifically, for a given vertical (e.g., e-commerce), we supply
the LLM with a curated list of canonical business features that are typical for
thatindustry,suchasSignin,Search,Cart,Checkout,andOrder.Thisexternal
knowledgeactsasastrongprior,anchoringthegenerationtoestablishedbusiness
taxonomiesandpreventingtheLLMfromproducingoverlygranular,UI-focused,
or platform-specific labels. This dual-input strategy—combining sampled data
with external industry knowledge—enables the system to produce features that
arebothempiricallygroundedintheobservedtelemetryandsemanticallyaligned
with real-world logic.
Constraints:The LLM is prompted to produce feature names that:
–Use one or two nouns.
–Prefer general names (e.g.,Searchrather thanKeyword Search).
–Avoid UI-specific nouns (e.g.,Button,Form).
–Reuse industry-standard feature names when semantically matching.

6 Yuanzhe Jia1, Ali Anaissi1,2
Quality Check:After generation, we apply a gate that verifies:
–All supplied industry-standard features are covered.
–Each feature name contains fewer than three words.
–The semantic similarity among features remains below a threshold.
3.4 Stage 3: Candidate Rule Retrieval
We retrieve candidate SQL-like mapping rules for each business feature identi-
fied in Stage 2. We first encode the refined data (produced in Stage 1) into dense
embeddings using a pre-trained sentence transformer. For a given business fea-
ture, we generate a query embedding from its name (optionally augmented with
a brief description) and perform a similarity search over the vector index. Crit-
ically, each indexed unit represents not an entire tracking event, but a specific
key-value condition. The retrieval mechanism is designed to identify the top-k
most semantically aligned condition subsets with respect to the target business
feature. This design ensures that the search focuses on attribute-level semantics,
rather than being biased by surface-level naming conventions. To improve recall,
we complement dense retrieval with BM25 keyword matching and combine both
scores via a weighted linear fusion.
For each business feature, the retrieval process returns a diverse set of track-
ing event conditions that are semantically related but span different facets of the
feature. For instance, for the business featureSearch, the retrieved event condi-
tions may include abutton_clickevent indicating the initiation of a search ac-
tion (e.g.,element_id = "search_init") as well as apage_viewevent reflect-
ingthesubsequentdisplayofsearchresults(e.g.,page_title LIKE "%Search%").
Each retrieved event condition is translated into a SQL-like predicate based on
its structural conditions. The union of these predicates for a given business fea-
ture forms an initial candidate rule set, which encapsulates multiple semantic
variants underlying the same business feature. This broad coverage ensures se-
mantic comprehensiveness, while the subsequent filtering and clustering stages
are responsible for disentangling these variants and assigning them to distinct
business nodes.
3.5 Stage 4: Candidate Rule Filtering
Retrieved rules contain significant noise, thus we apply a two-phase filter:
Hard Rule Filtering: Blocked events (e.g.,ad_click,api_error).
LLM Semantic Filtering:Foreachcandidaterule,anLLMindependentlyjudges
whether it belongs to the corresponding business feature. The LLM returnsYes
(retain) orNo(reject). Rules are rejected if they are:
–Not semantically relevant to the business feature.
–Logs that do not reflect user interactions.
–Pure input without results.
–No meaningful API calls.

Title Suppressed Due to Excessive Length 7
3.6 Stage 5: Business Node Clustering and Naming
Rules retained after filtering often correspond to multiple distinct user actions or
system processes falling under the same business feature. To derive semantically
coherent business nodes, we first perform a clustering step over the retained rules
for each business feature. Specifically, we use the LLM to group rules that share
the same underlying business semantics and represent exactly the same stage of
a business module (e.g., for theSearchfeature, rules indicating the initiation of
a search are clustered together, while those indicating the viewing of results form
a separate cluster). This ensures that all mapping rules within a given cluster
are semantically equivalent.
Subsequently, for each cluster, the LLM assigns a canonical name that con-
cisely captures the common semantics of its member rules. The naming format
follows a structured pattern:[noun]+[verb], where the noun part consists of
one or two nouns that denote the parent business node (e.g.,Search Result,
Cart Item), and the verb part is a single base-form word that describes the
precise user action or state transition (e.g.,View,Add). Additional constraints
enforce name consistency: multiple rules reflecting the same behavior must share
an identical name, and when semantically similar candidates appear, the most
general name is preferred. This hierarchical clustering-then-naming strategy en-
sures that each business node corresponds to a unique business action/status,
maintaining a clean, interpretable, and platform-agnostic semantic layer.
3.7 Deterministic Output
To ensure reproducibility and avoid hallucination, we:
–Use fixed random seeds for LLM inference.
–Settemperature = 0.0for deterministic sampling.
–Constrain the output format to valid JSON objects with concrete examples.
–Parse LLM responses withjson.loads().
The final output is a JSON object keyed bynode_id, each containing a
canonical business node name, matching conditions, and relevant metadata for
downstream usage.
{
"n0": {
"node_name": "Search Result View",
"node_rule": [
{
"event_name": "page_view",
"conditions": [
{
"key": "page_title",
"value": "%search%",
"operator": "LIKE"

8 Yuanzhe Jia1, Ali Anaissi1,2
}
]
}
]
"feature_name": "Search",
"industry": "e-commerce"
}
}
4 Experiments
4.1 Experimental Setup
Dataset:We evaluated our system on a large-scale e-commerce dataset [2] de-
rivedfromreal-worlduserinteractionlogscollectedfromanonlineretailer’sweb-
site over a six-month period. With millions of user sessions spanning the full e-
commerceuserjourney—frombrowsingandsearchingtocartingandpurchasing—
this dataset directly mirrors the heterogeneous, multi-platform telemetry scenar-
ios targeted by our business semantic layer induction framework.
Evaluation Metrics:
–Human Assessment: Semantic correctness rated by human experts.
–Noise Reduction: Percentage of data filtered out by the pipeline.
–ManualEffortReduction:Hoursperweeksavedbyhumanexpertspreviously
maintaining custom semantic mappings.
–LLM-as-Judge Agreement: Cohen’s kappa between LLM-as-Judge and hu-
man experts on annotated data.
Baseline:Initial prompt engineering iteration (no business feature identification,
no RAG retrieval, and no LLM semantic filtering) versus the full pipeline.
Implementation:The framework is implemented in Python 3.10+ as a modular
CLI tool, with Milvus for vector indexing, OpenAI API for LLM inference and
evaluation, Apache Airflow for batch orchestration, and Docker for containerized
deployment.
4.2 Human Assessment
We conducted a blind data review (see Table 1). For each of 100 sampled seman-
ticmappings(businessfeatures↔businessnodes↔SQL-likerules),5humanex-
perts rated semantic correctness on a 100-point scale. The full pipeline achieved
a mean score of 82.3 (σ= 9.7), compared to the baseline of 51.6 (σ= 14.2)—a
statistically significant improvement (p <0.01, two-tailed t-test).

Title Suppressed Due to Excessive Length 9
Table 1.Semantic quality comparison
Metric Baseline Full Pipeline Improvement
Human Assessment (0-100) 51.682.3+59.5%
Business Coverage 62%98%+58.1%
Name Consistency 59%96%+62.7%
4.3 Noise Reduction
Hard-coded and LLM-based filtering collectively eliminated 74% of candidate
rules as noise (see Table 2). The retained 26% of rules account for>90% of
semantic coverage, confirming that a small number of critical semantics drive
the most business insights.
Table 2.Noise filtering effectiveness
Filter Stage Rules Rejected Remaining
Hard Rule Filter 28% 72%
LLM Semantic Filter 46% 26%
Total 74% 26%
4.4 Manual Effort Reduction
To quantify the efficiency gains of our system, we also conducted a controlled
experiment comparing a data science team with 5 human experts against the
automated pipeline on equivalent semantic mapping tasks. Human experts re-
quired an average of 10 hours per week to generate and correct semantic map-
pings, whereas our system reduced this effort to approximately 2 hours—an 80%
reduction. Moreover, for querying newly defined metrics, the manual workflow
consumed 4 hours per query (including SQL construction and data validation),
while the semantic layer enabled self-service answers in under 25 minutes. These
results demonstrate that automation substantially reduces the operational bur-
den on domain experts, allowing them to focus on higher-level analytical work.
4.5 LLM-as-Judge Agreement
Weconstructedagoldensetof500manuallyannotatedsemanticmappings(busi-
ness features↔business nodes↔SQL-like rules) and used an LLM-as-Judge
with a structured prompt to rate the quality of those mappings on a 1–5 scale.
Comparing the LLM-as-Judge ratings against human expert annotations on the
same set yielded a Cohen’s kappa of 0.87, indicating near-perfect agreement and
validating the framework’s reliability for automated quality assessment. This

10 Yuanzhe Jia1, Ali Anaissi1,2
result is particularly significant because it demonstrates that LLMs can detect
quality degradation early and trigger targeted refinement, ensuring long-term
stability in production environments. This closed-loop evaluation mechanism
fundamentally shifts the maintenance paradigm from reactive, expert-dependent
corrections to proactive, automated quality stewardship.
4.6 Ablation Study
We finally conducted an ablation study to isolate the contribution of each ma-
jor component (see Table 3). The experiment confirms that every component
contributes positively, with enumeration normalization, business feature identi-
fication and embedding search retrieval being the most critical.
Table 3.Ablation study with human assessment
Configuration Score (0-100) Drop from full
Full pipeline 82.3 –
- No column importance scoring 74.5 -7.8
- No enumeration normalization 66.0-16.3
- No business feature identification 64.1-18.2
- No embedding search retrieval (BM25 only) 57.8-24.5
- No LLM semantic filtering (hard rule filter only) 75.9 -6.4
5 Conclusion
In this paper, we formalize and address the problem of business semantic layer
induction from raw application telemetry. Unlike prior work on log parsing and
schema matching, our framework tackles the unique challenge of deriving hier-
archical, business-aligned semantics without manual curation or labeled training
data. Our contributions are fourfold: (1) a systematic data refinement pipeline
combining LLM-driven column importance scoring and enumeration normal-
ization to substantially reduce feature sparsity; (2) a hierarchical abstraction
framework that decomposes the problem into coarse-grained business feature
identification and fine-grained business node classification, mirroring the natu-
ral reasoning structure of business analytics; (3) a principled induction pipeline
integrating hybrid retrieval, multi-stage filtering, and contrastive clustering with
canonicalnaming;and(4)anintegratedLLM-as-Judgeevaluationachieving0.87
Cohen’s kappa with human experts, enabling scalable and continuous quality
monitoring. Extensive experiments on production-scale e-commerce telemetry
demonstrate that our approach reduces manual maintenance effort by 80%, im-
proves semantic quality from 50 to 80+ on a 100-point scale, and filters 74%
of noisy candidates. By shifting the burden from ad-hoc parser maintenance

Title Suppressed Due to Excessive Length 11
to principled semantic induction, our framework empowers organizations to fo-
cus on extracting actionable business insights rather than debating customized
logics. Future work includes incorporating causal reasoning and cross-domain
transfer learning to further enhance the generalization and analytical depth of
the induced semantic layer.
References
1. Bian, H.: Llm-empowered knowledge graph construction: A survey. arXiv preprint
arXiv:2510.20345 (2025)
2. Dąbrowski, J., Janicka, M., Sienkiewicz, Ł., Stomfai, G., Dietmar, J., Barile, F.,
Polignano, M., Pomo, C., Srivastava, A.: The synerise dataset: An e-commerce
dataset for sequential recommendation, universal behavior modeling and deep re-
lational learning. In: Proceedings of the Recommender Systems Challenge 2025,
pp. 1–6. ACM (2025)
3. He, P., Zhu, J., Zheng, Z., Lyu, M.R.: Drain: An online log parsing approach with
fixed depth tree. In: 2017 IEEE international conference on web services (ICWS).
pp. 33–40. IEEE (2017)
4. Hoseini, S., Theissen-Lipp, J., Quix, C.: A survey on semantic data management
as intersection of ontology-based data access, semantic modeling and data lakes.
Journal of Web Semantics81, 100819 (2024)
5. Kropshofer, J., Schrott, J., Wöß, W., Ehrlinger, L.: A survey on the functionalities
of data catalog tools. IEEE Access (2025)
6. Loumachi, F.Y., Ghanem, M.C., Ferrag, M.A.: Advancing cyber incident time-
line analysis through retrieval-augmented generation and large language models.
Computers14(2), 67 (2025)
7. Piet, J., Fang, V., Khare, R., Coull, S., Paxson, V., Popa, R.A., Wagner, D.:
Semantic-aware parsing for security logs. arXiv preprint arXiv:2506.17512 (2025)
8. Pyrih, V., Rebmann, A., van der Aa, H.: Llms that understand processes:
Instruction-tuning for semantics-aware process mining. In: 2025 7th International
Conference on Process Mining (ICPM). pp. 1–8. IEEE (2025)
9. Tonnarelli, M., Kumara, I., Driessen, S., Tamburri, D.A., Van Den Heuvel, W.J.,
Oor, P.: Data catalog tools: A systematic multivocal literature review. Journal of
Systems and Software p. 112584 (2025)
10. Yang, Z., Wang, Y., Shi, Z., Yao, Y., Liang, L., Ding, K., Yilmaz, E., Chen, H.,
Zhang, Q.: Eventrag: Enhancing llm generation with event knowledge graphs. In:
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics. pp. 16967–16979 (2025)
11. Zhang, C., Xu, W., Liu, J., Zhang, L., Liu, G., Guan, J., Zhou, Q., Zhou, S.:
Semanticlog: Towards effective and efficient large-scale semantic log parsing. IEEE
Transactions on Software Engineering (2025)
12. Zhao, Z., Luo, X., Chen, M., Ma, L.: A survey of knowledge graph construction
using machine learning. Computer Modeling in Engineering & Sciences139(1),
225 (2024)
13. Zhong, A., Mo, D., Liu, G., Liu, J., Lu, Q., Zhou, Q., Wu, J., Li, Q., Wen, Q.:
Logparser-llm: Advancing efficient log parsing with large language models. In: Pro-
ceedings of the 30th ACM SIGKDD. pp. 4559–4570 (2024)
14. Zhu, W., Li, X., Wang, L., Wang, J., Wei, Y.: Tm-rag: A tree-mapped retrieval-
augmented generation framework for construction claim report generation. Ad-
vanced Engineering Informatics69, 104092 (2026)
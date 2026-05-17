# Persistent and Conversational Multi-Method Explainability for Trustworthy Financial AI

**Authors**: Georgios Makridis, Georgios Fatouros, John Soldatos, George Katsis, Dimosthenis Kyriazis

**Published**: 2026-05-12 07:45:46

**PDF URL**: [https://arxiv.org/pdf/2605.11687v1](https://arxiv.org/pdf/2605.11687v1)

## Abstract
Financial institutions increasingly require AI explanations that are persistent, cross-validated across methods, and conversationally accessible to human decision-makers. We present an architecture for human-centered explainable AI in financial sentiment analysis that combines three contributions. First, we treat XAI artifacts -- LIME feature attributions, occlusion-based word importance scores, and saliency heatmaps -- as persistent, searchable objects in distributed S3-compatible storage with structured metadata and natural-language summaries, enabling semantic retrieval over explanation history and automatic index reconstruction after system failures. Second, we enable multi-method explanation triangulation, where a retrieval-augmented generation (RAG) assistant compares and synthesizes results from multiple XAI methods applied to the same prediction, allowing users to assess explanation robustness through natural-language dialogue. Third, we evaluate the faithfulness of generated explanations using automated checks over grounding completeness, hallucinated claims, and method-attribution behavior. We demonstrate the architecture on an EXTRA-BRAIN financial sentiment analysis pipeline using FinBERT predictions and present evaluation results showing that constrained prompting reduces hallucination rate by 36\% and increases method-attribution citations by 73\% compared to naive prompting. We discuss implications for trustworthy, human-centered AI services in regulated financial environments.

## Full Text


<!-- PDF content starts -->

Persistent and Conversational Multi-Method
Explainability for Trustworthy Financial AI
Georgios Makridis∗, Georgios Fatouros∗, John Soldatos∗, George Katsis∗, Dimosthenis Kyriazis†
University of Piraeus, Greece
∗ExpertAI-Lux S.à r.l,†Department of Digital Systems
{george.makridis, george.fatouros, john.soldatos, george.katsis}@expertai-lux.com, dimos@unipi.gr
Abstract—Financial institutions increasingly require AI expla-
nations that are persistent, cross-validated across methods, and
conversationally accessible to human decision-makers. We present
an architecture for human-centered explainable AI in financial
sentiment analysis that combines three contributions. First, we
treat XAI artifacts—LIME feature attributions, occlusion-based
word importance scores, and saliency heatmaps—as persistent,
searchable objects in distributed S3-compatible storage with
structured metadata and natural-language summaries, enabling
semantic retrieval over explanation history and automatic index re-
construction after system failures. Second, we enable multi-method
explanation triangulation, where a retrieval-augmented generation
(RAG) assistant compares and synthesizes results from multiple
XAI methods applied to the same prediction, allowing users to
assess explanation robustness through natural-language dialogue.
Third, we evaluate the faithfulness of generated explanations
using automated checks over grounding completeness, hallucinated
claims, and method-attribution behavior. We demonstrate the
architecture on an EXTRA-BRAIN financial sentiment analysis
pipeline using FinBERT predictions and present evaluation results
showing that constrained prompting reduces hallucination rate by
36% and increases method-attribution citations by 73% compared
to naive prompting. We discuss implications for trustworthy,
human-centered AI services in regulated financial environments.
Index Terms—Explainable AI, Financial Sentiment Analy-
sis, Retrieval-Augmented Generation, Faithfulness Evaluation,
Human-Centered AI, Trustworthy AI
I. INTRODUCTION
Explainable AI (XAI) has become a regulatory and practical
necessity in financial services, driven by the EU AI Act [1] and
growing institutional demand for auditable decision support [2].
Financial sentiment analysis—classifying news headlines as
positive, negative, or neutral—is a canonical use case where
transformer-based models such as FinBERT [3] achieve high
accuracy but operate as opaque classifiers. Prior work [4]
demonstrated that LLM-based approaches can substantially
outperform FinBERT in financial sentiment classification, yet
the resulting predictions remain equally opaque to end-users.
Post-hoc explanation methods including LIME [5], occlusion-
based word importance, and attention analysis are routinely
applied to such models [6], yet three fundamental challenges
remain unaddressed in practice.
Challenge 1: Explanation ephemerality.In current XAI
pipelines, explanations are generated on demand, rendered as
static visualizations, and discarded. They are not stored as
persistent, searchable records and cannot be queried across
sessions, compared over time, or reconstructed after systemfailures. This makes compliance auditing and longitudinal
explanation analysis impractical—a concern particularly acute
in digital finance, where regulatory requirements demand
auditable decision trails [1]. While provenance-aware frame-
works [7], [8] and XAI-as-a-service platforms [9], [10] have
begun treating explanations as managed artifacts, none provide
a fully persistent, semantically searchable explanation store
with automatic reconstruction.
Challenge 2: Single-method fragility.[11] formally demon-
strated that popular XAI methods—LIME, KernelSHAP, In-
tegrated Gradients, attention—routinely disagree on the same
instance. Recent FinBERT-specific evaluation confirms this on
financial text, finding that LIME, Integrated Gradients, and
attention rollout produce divergent attribution rankings [12].
If a user sees only one explanation method, they receive
an incomplete and potentially misleading picture of model
behavior.
Challenge 3: Conversational XAI faithfulness.Conversa-
tional XAI systems [13]–[16] allow users to query explanations
in natural language. However, [17] demonstrated that LLM-
driven explanation chat can amplify over-reliance through an
“illusion of explanatory depth”—the assistant sounds authori-
tative regardless of whether its claims are grounded in actual
XAI evidence. Without automated faithfulness verification [18],
[19], fluent but unfaithful explanation narratives erode trust
rather than building it.
Contributions.We present a microservice architecture that
addresses all three challenges simultaneously:
1)Persistent XAI artifact store: Explanation outputs are
stored as structured metadata in S3-compatible object
storage with natural-language summaries, enabling semantic
vector search over explanation history and automatic index
reconstruction after container restarts (Section III).
2)Multi-method explanation triangulation: LIME, occlu-
sion, and saliency are applied to the same prediction and
their results are jointly indexed, enabling a RAG chatbot
to compare, contrast, and synthesize them in response to
user questions (Section IV).
3)Faithfulness-constrained RAG: Generated explanations
are grounded in retrieved XAI artifacts through constrained
prompting and evaluated using automated faithfulness
metrics (Section V).
The system is developed within the EXTRA-BRAIN Hori-
zon Europe project for digital finance and demonstrated onarXiv:2605.11687v1  [cs.AI]  12 May 2026

FinBERT-based news sentiment classification with per-sample
and dataset-level XAI capabilities.
II. RELATEDWORK
XAI for Financial NLP.Post-hoc explanation of financial
transformer models has been extensively studied. In [20] SHAP
was used over FinBERT to automatically induce explainable
finance lexicons. In our prior work [6] we extended XAI to
time-series classification using image highlight methods (LIME
and Grad-CAM) applied to 2D plot representations—a cross-
modal approach that informs our saliency-based vision XAI.
In [4] the authors benchmarked ChatGPT against FinBERT
for financial sentiment, showing 35% improved classification
but highlighting the need for interpretability of LLM-based
predictions. [21] proposed VirtualXAI, a framework that
integrates quantitative XAI benchmarking with qualitative
user assessments through GPT-generated virtual personas—
establishing the user-centric evaluation methodology that
informs our design. A comprehensive study applying five XAI
methods—LIME, SHAP, attention, Integrated Gradients, and
Grad-CAM—in parallel to the same sentiment model [22]
provides the closest blueprint for multi-method triangulation.
The 2025 FinBERT faithfulness audit [12] quantified deletion-
curve faithfulness across LIME, Integrated Gradients, and
attention rollout, finding LIME most faithful on financial text.
Our work extends this line by exposing multi-method results
interactively rather than as static evaluation tables.
Conversational XAI.TalkToModel [13] pioneered interac-
tive natural-language explanations, outperforming dashboard
XAI in user studies. ConvXAI [14] unified heterogeneous
explanation types under conversational design principles, while
InterroLang [15] extended the approach to NLP models. More
recent systems such as LLMCheckup [16] and Explingo [23]
replaced rule-based intent parsing with LLM orchestration; a
comprehensive survey [24] maps the design space across 15
core systems. [25] proposed HumAIne-Chatbot, a personalized
conversational AI with reinforcement learning that adapts
explanation delivery to user profiles. The authors of [17] found
that LLM-driven explanation chat can amplify over-reliance—
motivating our faithfulness-constrained design.
RAG Faithfulness Evaluation.RAGAs [18] introduced
reference-free metrics for faithfulness, relevance, and con-
text quality. ARES [26] fine-tuned lightweight judges with
prediction-powered inference. RAGTruth [27] provided word-
level hallucination annotations for supervised detection.
RAGChecker [19] decomposed evaluation into claim-level
faithfulness and noise sensitivity metrics. FActScore [28]
established atomic-level groundedness scoring. We adapt these
principles to the XAI domain, where “faithfulness” means that
the chatbot does not invent features, misattribute explanation
methods, or fabricate confidence values.
Persistent XAI Artifacts.This is the sparsest area and
our strongest novelty claim. [7] proposed provenance-enabled
XAI (PXAI) using lineage graphs as queryable artifacts. [10]
proposed Explainability-as-a-Service with semantic-similarity
caching for edge AI. The principle of treating AI-generatedDashboard
(Flask SPA)XAI Service
(Flask)
AI Outputs
(RAG + VectorDB)XAI API
(FastAPI)
Model API
(FinBERT) RustFS
(S3 Storage)ingest
chat, storeXAI calls
predict
persist
store resultsrehydrate
Fig. 1. System architecture. Solid arrows show runtime data flow; the dashed
arrow represents automatic vector index reconstruction from persistent metadata
on container restart.
artifacts as persistent, searchable, and personalizable records
has been explored in adjacent domains such as education,
where [29] demonstrated persistent personalized content gen-
eration with LLMs, establishing the design pattern we extend
to XAI artifacts. Our approach differs from all prior work
by combining persistent S3 storage with automatic semantic
rehydration of an in-memory vector index, enabling both bulk
dataset explanations and per-sample XAI results to survive
system failures and be searched conversationally.
III. SYSTEMARCHITECTURE
The system consists of five Docker microservices orches-
trated via Docker Compose, communicating over a shared
network with a remote S3-compatible object store (RustFS)
for persistent storage (Fig. 1).
Dashboard(Flask + JavaScript SPA) serves the web UI with
two primary tabs:Dataset Analysisfor bulk model output
visualization across the full dataset, andPer-Sample XAIfor
individual prediction explanations. It proxies all API calls and
handles authentication via Keycloak OIDC or legacy mode.
XAI Service(Flask) performs bulk data ingestion, prepro-
cessing, and dataset-level statistical visualization (sentiment
distributions, keyword analysis, asset-level box plots). It holds
the ingested data frame in memory and provides a per-sample
browsing interface to the Dashboard.
XAI API(FastAPI) implements the per-sample explanation
algorithms: occlusion-based word importance, LIME [5] text
explanations, and occlusion-based saliency maps for vision-
based tasks. For each request it retrieves the relevant data from
the artifact store, calls the Model API—potentially hundreds
of times for perturbation-based methods—and persists the full
result as structured JSON or PNG back to the store.
Model APIserves modular prediction endpoints for the
supported domains: sentiment classification [3] and, as an
extensibility demonstration, object detection. This component
is designed for straightforward replacement with production
models.

AI Outputs(Flask) is the RAG engine. It maintains an
in-memory vector database (SimpleVectorDB with OpenAI
embeddings) and conversation history. It supports question
answering over stored explanation artifacts, indexing of newly
generated XAI summaries, and persistence of plot images with
associated metadata.
A. Persistent XAI Artifact Store
The core architectural innovation is treating every explana-
tion output as a persistent, searchable artifact rather than an
ephemeral visualization. The artifact store organizes outputs
across logically separated buckets for rendered visualization
images, structured metadata records, and uploaded or exported
datasets. Domain-specific results—text-based attribution scores
and vision saliency maps—are stored in dedicated result
buckets, keeping the provenance trail clean across modalities.
Each metadata record follows a consistent schema that
captures provenance fields (model identifier, XAI method,
timestamp) alongside storage references and a natural-language
summary object used for semantic retrieval:
{"plot_type": "text_occlusion",
"title": "Occlusion -- sample #5",
"summary_for_rag": {
"text": "Target: positive. Top words:
outperformer (+0.245), growth (+0.156)",
"keywords": ["occlusion", "positive"],
"numeric_facts": {"baseline": 0.912}},
"provenance": {"model": "finbert",
"xai_method": "occlusion"}}
Theplot_type field is set at ingestion time by the expla-
nation module and acts as a method tag used during retrieval;
any new XAI method can be integrated by registering a new
type value without modifying the storage schema.
B. Automatic Rehydration
The vector database resides in RAM and is lost on container
restart. A lightweight rehydration mechanism detects an empty
collection for the requesting user on the first conversational
query after restart: it scans all stored metadata records for that
user, extracts the natural-language summary field from each,
computes its embedding, and inserts it into the vector index.
This process is transparent to the user—the first query after a
restart incurs a one-time latency1after which all subsequent
searches operate from RAM at sub-millisecond latency.
IV. MULTI-METHODEXPLANATIONTRIANGULATION
A key insight motivating our design is that multiple XAI
methods applied to the same prediction often disagree [11].
Rather than resolving this disagreement computationally, we ex-
pose it to the user through conversational interaction, enabling
what we termexplanation triangulation.
For text sentiment classification, the system provides two
complementary per-sample methods:
1Measured on our deployment: 0.8 s for 15 artifacts, 1.4 s for 50 artifacts,
2.1 s for 100 artifacts (single-threaded, remote RustFS over HTTPS).Occlusion.Each word is masked individually and the model
re-queried; the importance of word wiis the drop in target-
class probability: imp(w i) =P(c∗|x)−P(c∗|x\i). This
produces a complete attribution over all words, sorted by
absolute importance.
LIME.A neighbourhood of perturbed texts is generated
by randomly removing words, the model is queried on all
perturbations, and a local linear model is fitted. The coefficients
of the top-kfeatures constitute the explanation [5].
Both methods are applied to the same sample, and their
results are independently indexed in the vector database with
method-specific type tags. When the user asks the RAG chatbot
a comparative question such as“Do the XAI methods agree
on the most important words?”, the retrieval step returns both
artifacts (matched by semantic similarity), and the constrained
prompt (Section V) generates a response that explicitly cites
which method produced which attribution, surfaces agreements
and disagreements, and avoids conflating the two.
For vision tasks, the system provides occlusion-based
saliency: a patch slides across the image, the model is re-queried
at each position, and the confidence drop is mapped to a heat
map. This component serves as an extensibility demonstration—
validating that the persistent-artifact and RAG architecture
generalises beyond NLP to vision tasks—and is not covered
by the quantitative evaluation, which focuses on the financial
sentiment use case.
An illustrative example from our financial sentiment dataset:
for the headline“BioNTech forecasts strong growth in oncology
pipeline”, occlusion identifiedstrong(importance +0.312 ) and
growth( +0.287 ) as the top contributors, while LIME ranked
growth( +0.289 ) first andstrong( +0.195 ) third, withforecasts
(+0.201 ) intervening. The chatbot synthesises this as:“Both
methods agree that ‘growth’ strongly supports the positive
prediction. However, occlusion assigns higher importance to
‘strong’ than LIME does, while LIME elevates ‘forecasts’ as
a contributing factor that occlusion considers less significant. ”
This cross-validation surfaces explanation robustness—or its
absence— through natural dialogue rather than requiring the
user to compare static tables.
V. FAITHFULNESSEVALUATION
Following the finding that LLM-driven XAI chat can amplify
over-reliance [17], we implement a faithfulness evaluation
pipeline that measures whether generated explanations are
grounded in the actual XAI artifacts retrieved by the RAG
system.
A. Constrained vs. Naive Prompting
We compare two prompting strategies:
Naive baseline:The retrieved XAI documents are placed
in the LLM context with a generic instruction:“Answer the
user’s question based on the following context. ”
Constrained prompt:The system prompt explicitly requires:
(i) cite specific XAI artifacts by method and explanation type;
(ii) distinguish between LIME and occlusion results when both
are present; (iii) include numeric values (confidence scores,

TABLE I
FAITHFULNESS EVALUATION:CONSTRAINED VS.NAIVE PROMPTING
(n= 30QUERIES)
Metric Naive Constrained
Hallucination rate (↓) 0.140.09
Citations per response (↑) 5.18.9
Grounding completeness (↑)0.490.37∗
∗Constrained prompting produces more specific numeric claims; with limited
ground-truth coverage, additional claims reduce the grounding ratio despite
being contextually appropriate. See discussion in text.
importance weights) directly from the retrieved documents;
(iv) state when evidence is insufficient rather than speculating.
B. Faithfulness Metrics
We adapt RAG faithfulness principles [18], [19], [28] to the
XAI domain with three automated rule-based checks applied
to each generated response:
•Grounding completeness: fraction of numeric claims in the
response that can be traced to a retrieved XAI artifact (word
importance scores, confidence values, or dataset statistics).
•Hallucination rate: fraction of feature names mentioned
as “important” in the response that do not appear in any
retrieved document.
•Citation behavior: number of explicit method attributions
(e.g., “according to LIME. . . ”, “the occlusion analysis
shows. . . ”) per response.
C. Results
Table I summarizes results from 30 evaluation queries
spanning single-method questions (10 queries, e.g., “What
were the most important words?”), comparative multi-method
questions (5 queries, e.g., “Do the methods agree?”), adversarial
probes (5 queries referencing non-existent features), and dataset-
level questions (10 queries about distributions and statistics).
The ground truth comprises 11 word importance scores (4 top
occlusion tokens, 7 top LIME tokens, with 2 overlapping) and
15 distinct explanation types from the Vector DB.
Constrained prompting reduces the hallucination rate from
14% to 9% (36% relative improvement), indicating that explicit
grounding instructions effectively prevent the LLM from
inventing feature names. Citation behavior shows the largest
effect: 8.9 explicit method attributions per response versus
5.1 for naive prompting (73% increase), confirming that the
constrained prompt encourages method-specific language.
Grounding completeness presents a nuanced result: naive
prompting (0.49) outperforms constrained (0.37). The con-
strained prompt instructs the LLM to cite exact numeric
values, producing more numeric claims per response. However,
with only 11 word importance scores in the ground truth
and no standalone numeric values, many additional claims
cannot be verified, lowering the ratio. This highlights that
faithfulness evaluation requires comprehensive ground-truth
coverage; future work should persist richer structured metadata
alongside natural-language summaries.We note that this evaluation uses rule-based automated
checks with a modest test set; a larger-scale evaluation with
LLM-based judges [26] and human annotation would provide
stronger statistical guarantees.
VI. DISCUSSION
Practical implications.The persistent artifact store enables a
new class of XAI workflows: longitudinal explanation auditing
(“how has the model’s reliance on word X changed over
the last month?”), cross-user explanation comparison, and
automated regulatory reporting. Building on prior risk-aware
financial AI work [30], which demonstrated probabilistic deep
learning for portfolio risk assessment, this architecture adds
the explainability layer that regulators increasingly demand—
ensuring that not only the predictions but also their explanations
are persistent, auditable, and recoverable.
Relation to the EU AI Act.Article 13 of the EU AI Act [1]
requires high-risk AI systems to provide “sufficiently trans-
parent” outputs. Our architecture operationalizes this through
persistent, method-attributed, searchable explanations with
provenance metadata, offering a concrete technical pathway
toward compliance.
Limitations.Our faithfulness evaluation uses rule-based
checks rather than learned judges [26]; extending to LLM-based
evaluation would improve coverage. The vector database uses
simple cosine-similarity search; production deployment would
benefit from approximate nearest-neighbor indices. While we
demonstrate triangulation between two text XAI methods, the
framework naturally extends to SHAP, Integrated Gradients, and
counterfactuals [31]. Finally, a user study following established
HCXAI protocols [32], [33] would provide complementary
evidence on user trust and comprehension.
Broader applicability.While demonstrated on financial sen-
timent, the architecture is domain-agnostic. The same persistent-
artifact, multi-method, faithfulness-constrained RAG pattern
applies to clinical NLP, manufacturing defect detection [34],
or any setting where explanation auditability, robustness, and
conversational accessibility are required.
VII. CONCLUSION
We presented a human-centered XAI architecture for finan-
cial AI that addresses three gaps in current practice: explanation
ephemerality, single-method fragility, and conversational faith-
fulness. By treating XAI artifacts as persistent, searchable
objects with structured metadata, enabling multi-method expla-
nation triangulation through a RAG chatbot, and constraining
generated explanations with automated faithfulness checks, the
system provides auditable, robust, and accessible explanations
suitable for regulated financial environments. Evaluation shows
that constrained prompting reduces hallucination by 36% and
increases method-attribution citations by 73% compared to
naive prompting, while also revealing that comprehensive
ground-truth coverage is essential for meaningful grounding
evaluation. Future work will extend the faithfulness evaluation
with LLM-based judges, conduct controlled user studies with

financial analysts, and enrich the persistent metadata schema
to improve ground-truth coverage for automated evaluation.
ACKNOWLEDGMENT
This work has been supported by the EXTRA-BRAIN
project, funded by the European Union’s Horizon Europe
programme under Grant Agreement No. 101135809.
REFERENCES
[1]“Regulation (EU) 2024/1689 laying down harmonised rules on artificial
intelligence (AI act),”Official Journal of the European Union, 2024.
[2]A. Wilson, “Explainable AI in finance: Addressing the needs of diverse
stakeholders,” 2025.
[3]D. Araci, “Finbert: Financial sentiment analysis with pre-trained language
models,”arXiv preprint arXiv:1908.10063, 2019.
[4]G. Fatouros, J. Soldatos, K. Kouroumali, G. Makridis, and D. Kyriazis,
“Transforming sentiment analysis in the financial domain with ChatGPT,”
Machine Learning with Applications, vol. 14, p. 100508, 2023.
[5]M. T. Ribeiro, S. Singh, and C. Guestrin, “Why should I trust you?:
Explaining the predictions of any classifier,” 2016.
[6]G. Makridis, G. Fatouros, V . Koukos, D. Kotios, D. Kyriazis, and
J. Soldatos, “XAI for time-series classification leveraging image highlight
methods,” inManagement of Digital EcoSystems (MEDES 2023), ser.
CCIS, vol. 2022. Springer, 2024.
[7]S. Zhang, J. Zhou, and B. Ujcich, “Provenance-enabled explainable AI,”
inSIGMOD, vol. 2, no. 6, 2024.
[8]G. Kale, D. Nguyen, R. Harris, C. Li, S. Zhang, and Y . Ma, “Provenance
documentation to enable explainable and trustworthy AI: A literature
review,”Data Intelligence, vol. 5, no. 1, pp. 139–162, 2023.
[9]Y . Wanget al., “XAIport: A service framework for the early adoption
of XAI in AI model development,” inICSE-NIER, 2024.
[10] R. Singh and S. Roy, “Scalable explainability-as-a-service (XaaS) for
edge AI systems,”arXiv:2602.04120, 2026.
[11] S. Krishna, T. Han, A. Gu, J. Pombra, S. Jabbari, S. Wu, and H. Lakkaraju,
“The disagreement problem in explainable machine learning: A practi-
tioner’s perspective,”Transactions on Machine Learning Research, 2024,
arXiv:2202.01602.
[12] “Fine-tuning and explaining FinBERT for sector-specific financial news:
A reproducible workflow,”Electronics, vol. 14, no. 23, p. 4680, 2025.
[13] D. Slack, S. Krishna, H. Lakkaraju, and S. Singh, “Explaining machine
learning models with interactive natural language conversations using
TalkToModel,”Nature Machine Intelligence, vol. 5, pp. 873–883, 2023.
[14] Z. Shen, Q. Huang, K. Wu, and T. Huang, “ConvXAI: Delivering
heterogeneous AI explanations via conversations to support human-AI
scientific writing,” inCSCW Companion, 2023.
[15] N. Feldhus, Q. Wang, N. Anikina, A. Chopra, T. Oguz, and S. Möller,
“InterroLang: Exploring NLP models and datasets through dialogue-based
explanations,” inFindings of EMNLP, 2023.[16] Q. Wang, N. Anikina, N. Feldhus, J. van Genabith, L. Hennig, and
S. Möller, “LLMCheckup: Conversational examination of large language
models via interpretability tools and self-explanations,” inNAACL
HCINLP Workshop, 2024.
[17] G. He, A. Aishwarya, and U. Gadiraju, “Is conversational XAI all you
need? human-AI decision making with a conversational XAI assistant,”
inProc. ACM IUI, 2025.
[18] S. Eset al., “RAGAs: Automated evaluation of retrieval augmented
generation,” inEACL Demos, 2024.
[19] D. Ruet al., “RAGChecker: A fine-grained framework for diagnosing
retrieval-augmented generation,” inNeurIPS Datasets and Benchmarks,
2024.
[20] M. Rizinskiet al., “Sentiment analysis in finance: From transformers
back to explainable lexicons (XLex),”IEEE Access, 2024.
[21] G. Makridis, V . Koukos, G. Fatouros, M. M. Separdani, D. Kyriazis, and
J. Soldatos, “VirtualXAI: A user-centric framework for explainability
assessment leveraging GPT-generated personas,” inProc. 21st Intl. Conf.
Distributed Computing in Smart Systems and the Internet of Things
(DCOSS-IoT). IEEE, 2025.
[22] “Explainable aspect-based sentiment analysis using transformer models,”
Big Data and Cognitive Computing, vol. 8, no. 11, p. 141, 2024.
[23] A. Zyteket al., “Explingo: Explaining AI predictions using large language
models,” inIEEE BigData, 2024, arXiv:2412.05145.
[24] “Beyond one-shot explanations: A systematic literature review of
dialogue-based xAI approaches,”Artificial Intelligence Review, vol. 58,
2025.
[25] G. Makridis, V . Fragiadakis, H. Oliveira, P. Saraiva, P. Mavrepis,
G. Fatouros, and D. Kyriazis, “HumAIne-Chatbot: Real-time personalized
conversational AI via reinforcement learning,”arXiv:2509.04303, 2025.
[26] J. Saad-Falconet al., “ARES: An automated evaluation framework for
retrieval-augmented generation systems,” inNAACL, 2024.
[27] C. Niu, H. Wuet al., “RAGTruth: A hallucination corpus for developing
trustworthy retrieval-augmented language models,” inACL, 2024.
[28] S. Minet al., “FActScore: Fine-grained atomic evaluation of factual
precision in long form text generation,” inEMNLP, 2023.
[29] G. Makridis, A. Oikonomou, and V . Koukos, “FairyLandAI: Personalized
fairy tales utilizing ChatGPT and DALL-E 3,”arXiv:2407.09467, 2024.
[30] G. Fatouros, G. Makridis, D. Kotios, J. Soldatos, M. Filippakis, and
D. Kyriazis, “DeepVaR: A framework for portfolio risk assessment
leveraging probabilistic deep neural networks,”Digital Finance, vol. 5,
pp. 29–56, 2023.
[31] Y . Rong, T. Leemannet al., “Towards human-centered explainable AI:
A survey of user studies for model explanations,”IEEE TPAMI, 2024.
[32] R. R. Hoffman, S. T. Mueller, G. Klein, and J. Litman, “Measures for
explainable AI: Explanation goodness, user satisfaction, mental models,
curiosity, trust, and human-AI performance,”Frontiers in Computer
Science, vol. 5, 2023.
[33] U. Ehsan, K. Passi, Q. V . Liao, L. Chan, I. Lee, M. Muller, and M. O.
Riedl, “The who in XAI: How AI background shapes perceptions of AI
explanations,” 2024.
[34] G. Makridiset al., “XAI enhancing cyber defence against adversarial
attacks in industrial applications,” inIEEE 5th Intl. Conf. Image
Processing Applications and Systems (IPAS), 2022, pp. 1–8.
# FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection

**Authors**: Daniel Berhane Araya, Duoduo Liao

**Published**: 2025-10-13 17:31:49

**PDF URL**: [http://arxiv.org/pdf/2510.11654v1](http://arxiv.org/pdf/2510.11654v1)

## Abstract
Financial markets face growing threats from misinformation that can trigger
billions in losses in minutes. Most existing approaches lack transparency in
their decision-making and provide limited attribution to credible sources. We
introduce FinVet, a novel multi-agent framework that integrates two
Retrieval-Augmented Generation (RAG) pipelines with external fact-checking
through a confidence-weighted voting mechanism. FinVet employs adaptive
three-tier processing that dynamically adjusts verification strategies based on
retrieval confidence, from direct metadata extraction to hybrid reasoning to
full model-based analysis. Unlike existing methods, FinVet provides
evidence-backed verdicts, source attribution, confidence scores, and explicit
uncertainty flags when evidence is insufficient. Experimental evaluation on the
FinFact dataset shows that FinVet achieves an F1 score of 0.85, which is a
10.4% improvement over the best individual pipeline (fact-check pipeline) and
37% improvement over standalone RAG approaches.

## Full Text


<!-- PDF content starts -->

FinVet: A Collaborative Framework of RAG and
External Fact-Checking Agents for Financial
Misinformation Detection
Daniel Berhane Araya, Duoduo Liao
College of Engineering and Computing
George Mason University
Fairfax, V A, USA
{dberhan4, dliao2}@gmu.edu
Abstract—Financial markets face growing threats from mis-
information that can trigger billions in losses in minutes. Most
existing approaches lack transparency in their decision-making
and provide limited attribution to credible sources. We introduce
FinVet, a novel multi-agent framework that integrates two
Retrieval-Augmented Generation (RAG) pipelines with external
fact-checking through a confidence-weighted voting mechanism.
FinVet employs adaptive three-tier processing that dynamically
adjusts verification strategies based on retrieval confidence,
from direct metadata extraction to hybrid reasoning to full
model-based analysis. Unlike existing methods, FinVet provides
evidence-backed verdicts, source attribution, confidence scores,
and explicit uncertainty flags when evidence is insufficient.
Experimental evaluation on the FinFact dataset shows that FinVet
achieves an F1 score of 0.85, which is a 10.4% improvement
over the best individual pipeline (fact-check pipeline) and 37%
improvement over standalone RAG approaches.
Index Terms—Financial misinformation detection, Retrieval-
Augmented Generation (RAG), fact-checking, large language
models (LLMs), confidence-based voting, explainable AI, Google
Fact Check API
I. INTRODUCTION
Financial markets are increasingly susceptible to misinfor-
mation, where a single false claim can trigger billions of
dollars in losses within minutes. For instance, a 2013 tweet
from a hacked Associated Press account falsely reporting a
White House explosion “wiped out $130 billion from the stock
market” [1]. More recently, in May 2023, a viral fake image of
a Pentagon explosion caused the Dow Jones Industrial Average
to drop 80 basis points (bps) and the S&P 500 to fall by 26 bps
[2]. These incidents underscore the urgent need for a robust
and transparent framework capable of detecting and mitigating
financial misinformation. Financial misinformation is defined
as false or misleading information related to economic or
financial matters that can influence investor decisions, market
behavior, or regulatory outcomes [3]. It can also include fraud-
ulent schemes aimed at defrauding individuals or institutions.
The advent of generative AI has triggered a wave of AI-
driven tools, escalating misinformation, disrupting financial
markets, and fueling a surge in fraud. Malicious actors now
exploit generative AI to create deepfake videos, synthetic
voices, and fraudulent documents—enabling large-scale de-ception that challenges financial institutions’ detection capabil-
ities. Social media platforms, central to the rapid dissemination
of misinformation, exacerbate this growing threat. According
to Deloitte [4], AI-enabled fraud losses in the United States
are projected to increase from $12.3 billion in 2023 to $40
billion by 2027.
Identifying and mitigating financial misinformation is cru-
cial for financial institutions, investors, and regulators. Left
unchecked, misinformation can manipulate stock prices, mis-
lead investors, and destabilize markets. To this end, this paper
introduces FinVet, a novel multi-agent framework designed to
address the aforementioned challenges through the integration
of two Retrieval-Augmented Generation (RAG) [5] pipelines
and a Fact-Check Pipeline. The RAG components utilize an
external knowledge source, such as a vector store, which can
be tailored to any domain-specific dataset. The Fact-Check
Pipeline combines an external fact-checking source with a fall-
back mechanism leveraging a Large Language Model (LLM)
when direct evidence is unavailable. The outputs of these
independent pipelines are integrated via a confidence-based
voting mechanism to deliver a final verdict. This integrated
approach provides a robust and adaptable misinformation
detection system, offering improved accuracy and explainabil-
ity. In addition to flagging misinformation, FinVet presents
evidence-backed justifications, clearly cites source origins, and
assigns a confidence score quantifying the system’s certainty
in its verdict.
The main contributions of this paper are:
1) A novel multi-agent verification framework that sys-
tematically integrates dual RAG pipelines with external
fact-checking orchestrated using a confidence-weighted
voting mechanism, addressing the limitations of single-
pipeline approaches in financial misinformation detec-
tion.
2) An adaptive three-tier processing strategy that dynami-
cally selects verification approaches based on retrieval
confidence scores: (i) direct metadata extraction for
high-confidence retrievals, (ii) hybrid context-model rea-
soning for moderate confidence, and (iii) pure model-
based analysis for low-confidence scenarios, optimizingarXiv:2510.11654v1  [cs.IR]  13 Oct 2025

both computational efficiency and verification accuracy.
3) A hierarchical confidence-weighted integration mecha-
nism that combines heterogeneous verification outputs
to provide a more robust verdict.
4) Comprehensive empirical validation demonstrating that
this collaborative approach achieves an F1 score of 0.85,
which is a 10.4% improvement over the best individual
pipeline.
The remaining sections of the paper are organized as
follows: Section II reviews related work; Section III provides
a summary of the contributions; Section IV describes the pro-
posed framework; Section V presents the experimental results
and discussion; Section VI discusses ethical considerations;
and Section VII concludes the paper.
II. RELATEDWORK
Financial misinformation detection is a complex task that
has been extensively researched using various techniques.
Although most existing approaches predominantly rely on
deep learning-based methods, recent research has increasingly
explored Large Language Models (LLMs) for this purpose.
In this paper, based on the objectives and methodologies of
the reviewed studies, we categorize financial misinformation
detection research into deep learning-based and LLM-based
techniques which reflect the evolving landscape of approaches
in this domain.
A. Deep Learning-Based Techniques
Several previous studies utilize machine learning and deep
learning techniques, for financial misinformation detection.
Dmonteet al.[6] instruction-tune LLaMA-3.1-8B using
claim–justification pairs and also explore a prompting-based
few-shot approach. Similarly, Purbeyet al.[7] use a sequential
fine-tuning strategy—first training LLMs for classification and
then for joint explanation generation—to optimize end-to-end
performance on the financial misinformation detection task.
Zhang and Liu [8] focus on identifying financial news in the
Chinese market with a deep learning approach. They propose
a dual sub-network model, employing BERT-Chinese-wwm
embeddings [9] [10] to generate rich word representations,
which feed into three parallel Convolutional Neural Networks
(CNNs) with varying kernel sizes to extract multi-granular
semantic features from article content.
Kamal, Mohankumar, and Singh [11] tackle financial
misinformation with their Fin-MisID model. They leverage
RoBERTa [12] for contextual representations, feeding them
into multi-channel networks of parallel Convolutional Neural
Networks (CNNs) with varying kernel sizes, Bidirectional
Gated Recurrent Units (BiGRU) for sequential patterns, and
attention mechanisms to emphasize critical tokens.
Nasir, Khan, and Varlamis [13] propose a hybrid deep
learning architecture combining CNN and Long Short-Term
Memory networks (LSTMs) for fake news detection. CNNs
are used to extract local textual patterns and n-gram features,
which are then passed to LSTMs to capture long-range de-
pendencies in the sequence. The model is trained end-to-end,enabling joint learning of spatial and temporal features. This
approach leverages the complementary strengths of CNNs and
RNNs for improved representation of news content.
X. Zhang, Q. Du, and Z. Zhang [14] design a theory-driven
machine learning system for financial disinformation detec-
tion, grounded in Truth-Default Theory (TDT). Their frame-
work operationalizes five key deception cues—communication
context and motive, sender demeanor, third-party information,
coherence, and correspondence—into quantifiable features.
These features are extracted from financial news articles,
propagation patterns across platforms, author metadata, and
firm-level financial indicators. The system integrates these het-
erogeneous signals into a supervised learning pipeline, offering
interpretable and domain-sensitive disinformation detection.
Zhi et al. [15] propose a multi-fact CNN-LSTM model that
integrates multiple dimensions of information for financial
fake news detection. The model combines news content,
market data, user comments, and source credibility into a
unified framework. Character-level CNNs are employed to
extract granular textual features, which are then encoded
through LSTM layers to capture temporal dependencies. An
attention mechanism models the interaction between news
content and user comments, and all components are fused
through weighted aggregation for the final prediction.
The deep learning-based approaches described above pri-
marily focus on classification-oriented architectures that rely
on features extracted from news text and propagation patterns.
Although these methods demonstrate strong performance in
identifying financial misinformation, they typically operate
with limited transparency in their decision-making process.
In contrast, FinVet introduces a novel multi-agent framework
that combines RAG pipelines with an external fact-checking
system, orchestrated through a confidence-based voting mech-
anism. Rather than relying solely on internal embeddings or
latent features, FinVet retrieves verifiable evidence from a
domain-specific vector store and external sources, providing
rationale-backed predictions with explicit confidence scores
and citation of information sources.
Furthermore, FinVet employs a tiered verification strategy:
highly relevant retrieved evidence leads to direct labeling;
moderately relevant evidence is combined with LLM reason-
ing; and in the absence of external context, the system relies
on structured, role-based prompts to elicit reasoned judgments
from LLMs. This adaptive methodology not only improves
classification performance but also improves explainability.
B. LLM-based Approaches
Large Language Models (LLMs) have recently been ex-
plored for misinformation detection. Leite, Razuvayevskaya,
Bontcheva, and Scarton [16] analyze whether LLMs can
generate weak labels based on 18 credibility signals, which are
then aggregated using weak supervision to predict content ve-
racity. Their approach demonstrates that zero-shot LLM-based
credibility labeling, combined with signal aggregation, can
outperform traditional classifiers without requiring ground-
truth labels.

Liu et al. [17] developed an instruction-tuning dataset,
the Financial Misinformation Detection Instruction Dataset
(FMDID), by combining two existing datasets: FinFact [3]
and FinGuard [18]. Using their developed dataset, the authors
fine-tuned LLaMa2-chat-7b [19] and Llama-3.1-8B-Instruct1.
They compared their model’s performance against baseline
models, including Pre-trained Language Models (PLMs) like
BERT and RoBERTa, as well as LLMs such as LLaMA2,
LLaMA3.1, and ChatGPT [20].
Wan et al. [21] propose DELL, a multi-stage misinformation
detection framework that strategically integrates LLMs to
improve veracity assessment. DELL begins by generating syn-
thetic user-news interaction networks through various LLM-
generated news reactions. It uses six explainable proxy tasks,
such as sentiment, stance, and framing to enrich contextual
understanding with LLM-generated explanations. Finally, it
employs an LLM-based expert ensemble, where task-specific
predictions and confidence scores are selectively aggregated
to produce calibrated, rationale-backed misinformation judg-
ments.
In contrast to the aforementioned research approaches, Fin-
Vet integrates two RAG pipelines with an online fact-checking
tool, using a confidence score-based voting mechanism, ensur-
ing claims are validated using multiple pathways. Moreover,
by leveraging a domain-specific knowledge base, FinVet’s
architecture can easily adapt to emerging financial misinfor-
mation patterns while maintaining decision transparency that
is crucial for financial stakeholders.
To the best of our knowledge, no prior work in financial
misinformation detection has combined dual RAG pipelines,
external fact-checking services, and confidence-based verdict
integration within a unified, transparent architecture.
III. SUMMARY OFCONTRIBUTIONS
This paper presents FinVet, a multi-agent verification frame-
work that integrates evidence retrieval, external fact-checking,
and confidence-based reasoning into a unified decision-making
architecture. By directly addressing the limitations of ex-
isting approaches, particularly their reliance on single ver-
ification pipelines and limited transparency, FinVet delivers
interpretable, adaptive, and evidence-based results tailored for
financial misinformation detection. The key contributions of
this study are summarized below:
•Multi-Agent Verification Architecture:FinVet employs
three independently operating verification pipelines—two
RAG pipelines and a Fact-Check Pipeline that integrates
an external fact-checking source with an LLM fallback.
This ensemble design enhances redundancy, promotes
model diversity, and improves verification reliability.
•Three-Tier Similarity-Based Evidence Processing:A
cosine-similarity–driven mechanism dynamically adjusts
the reasoning pathway based on the relevance of retrieved
evidence. High similarity prompts direct evidence extrac-
tion, moderate similarity invokes hybrid model reasoning,
1https://huggingface.co/meta-llama/Llama-3.1-8B-Instructand low similarity triggers expert-mimicking role-based
analysis.
•Confidence-Based Verdict Integration:A hierarchical ag-
gregation mechanism combines outputs from all verifica-
tion pipelines and weights them by confidence, ensuring
that high-certainty responses are prioritized in the final
verdict.
•Transparent Verdicts with Source Attribution:FinVet out-
puts not only claim labels but also supporting evidence
sources and confidence scores, enhancing interpretability
and aligning with ethical standards for decision-making
in high-stakes financial domains.
•Empirical Validation through Baseline and Ablation Stud-
ies:FinVet was evaluated on the FinFact dataset against
multiple baselines, including zero-shot LLMs, chain-of-
thought prompting, RAG-only models, and standalone
fact-checking systems. It achieved an F1 score of 0.85,
outperforming all baselines by 8–36%. Ablation experi-
ments further confirm the additive contribution of each
verification component.
•First Integrated Application in Financial Misinformation
Detection:To our knowledge, this is the first work to
combine RAG, external fact-checking, and confidence-
based voting within a unified system for financial misin-
formation detection.
These contributions collectively position FinVet as a scal-
able, interpretable, and empirically validated framework that
advances the state of the art in financial claim verification.
IV. METHODOLOGY
This research proposes FinVet shown in Figure 1 and elabo-
rated in Algorithm 1. FinVet is a novel framework that employs
a multi-pipeline architecture for financial claim verification.
The framework consists of four primary components: (A)
Data Processing and Vector Storage, (B) Claim Verification
Pipelines, (C) Results Normalization, and (D) Verdict Integra-
tion and Reporting. These components are described in the
following sections.
A. Data Processing and Vector Store
This component serves as FinVet’s knowledge base by
transforming financial information into semantically rich, re-
trievable representations. First, financial texts are decomposed
into contextually meaningful claim–evidence pairs, preserving
critical metadata relationships. This approach balances granu-
larity and context, thus improving retrieval precision. Second,
each text segment is embedded into a high-dimensional vector
space using semantic encodings that capture domain-specific
relationships. This enables nuanced concept-driven evidence
retrieval. Finally, the origin of each piece of evidence is
tracked by storing associated metadata throughout the veri-
fication pipeline, ensuring that FinVet’s outputs are grounded
in traceable source-backed justifications.

Fig. 1. FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection
B. Claim Verification Pipelines
The Claim Verification Pipelines represent the core analyti-
cal engines of FinVet, and it is comprised of two complemen-
tary verification approaches: two RAG Verification Pipelines
and a Fact-Check Pipeline. These pipelines process financial
claims through different methodologies, each bringing unique
analytical strengths to determine the veracity of a claim. Using
multiple verification strategies based on different approaches,
the system attempts to achieve a more robust and nuanced
assessment than any single approach could provide.
1) RAG Verification Pipelines:The RAG Verification
Pipelines implement a retrieval-augmented generation ap-
proach that grounds verification in contextual knowledge. Each
pipeline retrieves relevant financial information from the vec-
tor store and uses this context in its verification process. FinVet
employs two distinct RAG pipelines that are implemented us-
ing different underlying langugage models. The framework is
inherently extensible, allowing for the integration of additional
RAG pipelines to further enhance the verification process.
This modular design enables the incorporation of specialized
verification pathways tailored to particular financial domains
or claim types, such as market predictions, earnings reports, or
macroeconomic indicators. Moreover, while the current RAG
implementations use a shared vector store, the architecture
supports the integration of multiple domain-specific knowl-
edge bases that can be used for different financial sectors,
regulatory frameworks, or historical time periods, enabling thesystem to triangulate evidence using multiple sources.
Both RAG pipelines follow a similar workflow: prepro-
cessing the claim text, retrieving relevant context from the
vector store based on semantic similarity, and generating
structured verification outputs including label determination
(true/false/Not Enough Information (NEI)), evidence gener-
ation, confidence scoring, and source attribution. The dual
pipeline approach helps mitigate individual model biases
through model diversity.
Central to FinVet’s RAG pipeline is the three-tier processing
step that dynamically adapts the verification process based on
retrieval confidence scores and it is described in Algorithm
2. The system determines the relevance of the evidence using
cosine similarity. This adaptive approach works as follows:
•Case 1: When highly relevant evidence is found (similar-
ity score>=0.6), the system extracts the label, evidence,
and source directly from the retrieved metadata without
additional LLM reasoning. The evidence is taken directly
from the retrieved sentence and the source attribution is
preserved by directly passing through the original source
from the metadata’s records. The confidence score reflects
the similarity score, and higher values represent closely
matching evidence;
•Case 2: For moderately relevant evidence (0.4<=
Similarity Score<0.6), a hybrid approach is used by
combining the retrieved context with model reasoning,
providing both the retrieved information and claim to the

Algorithm 1:FinVet: Financial Misinformation Detec-
tion Framework
Input :Financial claimC, Vector storeV, External
fact-check APIF, Similarity thresholds
θhigh= 0.6,θ med= 0.4
Output:VerdictL(true/false/NEI), EvidenceE,
SourceS, Confidenceρ
/*RAG Pipeline Processing */
1fori←1to2do
2R i←RAGPipeline(C,V,M i);
/*M1: LLaMA-3.3-70B,M 2:
Mixtral-8x7B */
3end
/*Fact-Check Pipeline Processing */
4R 3←FactCheckPipeline(C,F);
/*Results Normalization */
5forj←1to3do
6 ˆRj←NormalizeResult(R j);
7end
/*Verdict Integration */
8ifR 3.source = “External Fact-Check”then
9returnR 3;
10else
11k∗←arg max kˆRk.confidence;
12if∀k: ˆRk.confidence = 0then
13return{label: “NEI”, evidence: “Insufficient
information”,
14source: “No evidence”, confidence:
0};
15else
16return ˆRk∗;
17end
18end
foundation model with explicit instructions to evaluate the
claim using the provided context. The evidence is gener-
ated by the foundation model and the source attribution
maintains the original retrieved sources when the model
validates their relevance; if the model determines the re-
trieved information is insufficient, it indicates ”Parametric
Knowledge” as the source. Confidence is calculated as the
average of the retrieval similarity and the model’s self-
reported confidence, balancing evidence relevance with
model certainty;
•Case 3: When no strongly relevant evidence is available
(similarity<0.4), the system falls back to a role-based
prompting that instructs the foundation model to analyze
the claim from multiple expert perspectives (Financial
Analyst, Political Misinformation Specialist, Government
Policy Analyst, and Investigative Journalist). In this
tier, evidence is entirely generated by the foundation
model’s reasoning process and explicitly identified asmodel-generated content. Source attribution is consis-
tently marked as ”Parametric Knowledge” to maintain
transparent distinction between retrieved and generated
content. Confidence is based entirely on calibrated as-
sessment of the foundation model.
The similarity cut-off values used in the above three-tier
approach were obtained from multiple rounds of tests that were
performed using the FinFact dataset.
This adaptive processing framework optimizes both com-
putational efficiency as well as cost, by invoking increasingly
complex verification only when necessary.
2) Fact-Check Pipeline:The Fact-Check Pipeline pro-
vides a complementary verification approach that augments
the context-based RAG methodology. This pipeline operates
through two distinct verification stages:
•External Verification Component: The system queries
an external fact-checking source to identify previously
analyzed claims. When matches are found, the system
extracts evidence from the fact-check response, priori-
tizing textual ratings and publisher conclusions. Source
attribution is preserved directly via the URL to the
original fact-checked source. For cases where a match
is available in the external source, the system assigns the
maximum confidence (1.0) to the verdict.
•LLM Analyzer: When external verification yields no re-
sults, the system uses role-based prompting to an LLM.
Evidence is generated through the structured analysis
of the claim by the model, and sources are explicitly
labeled as ”Parametric Knowledge”. Confidence scores
are extracted from the model’s self-reported confidence
rating.
C. Results Normalization
At this stage, FinVet has the verification results of a claim
using three independent pipelines: two RAG pipelines and
an external Fact-Check Pipeline. This component transforms
these diverse outputs into a consistent format. It normalizes
labels to lower case (”true”, ”false” or ”nei”) and standardizes
confidence score formatting which is crucial for the next and
final component.
D. Verdict Integration and Reporting
This component combines the verdict of the above three
pipelines through a confidence-based voting mechanism. The
voting process logic is as follows:
1) If the Fact-Check Pipeline returns results from an exter-
nal fact-checked source, these are automatically priori-
tized as the final verdict.
2) In typical cases, the system selects the result with the
highest confidence score across all pipelines.
3) If all pipelines return zero confidence scores, the system
defaults to a ”NEI” verdict, explicitly acknowledging
insufficient evidence rather than making an unwarranted
classification.
Following this decision process, the reporting component
generates an output that includes: the final verification label

Algorithm 2:RAG Processing Pipeline
Input :ClaimC, Retrieved documents{d 1, ..., d k},
Similarity scores{s 1, ..., s k}, Thresholds
θhigh= 0.6,θ med= 0.4, ModelM, Expert
rolesR
Output:ResultR={label, evidence, source,
confidence}
/*Check retrieved documents */
1ifk= 0then
/*No documents retrieved */
2result←RoleBasedAnalysis(M,C,R);
3label←result.label;
4evidence←result.evidence;
5source←“Parametric Knowledge”;
6confidence←ModelConfidence(M);
7else
/*Evaluate Similarity */
8s max←max{s 1, ..., s k};
9d best←document withs max;
10ifs max> θhigh then
/*Case 1 */
11label←ExtractLabel(d best);
12evidence←ExtractEvidence(d best);
13source←ExtractSource(d best.metadata);
14confidence←s max;
15else
16ifs max≥θmed then
/*Case 2 */
17context←
CombineDocuments({d 1, ..., d k});
18result←ModelReasoning(M,C, context);
19label←result.label;
20evidence←result.evidence;
21ifModelValidatesRetrieved(result)then
22source←
ExtractSource(d best.metadata);
23else
24source←“Parametric Knowledge”;
25end
26confidence←Average(s max,
result.confidence);
27else
/*Case 3 */
28result←RoleBasedAnalysis(M,C,R);
29label←result.label;
30evidence←result.evidence;
31source←“Parametric Knowledge”;
32confidence←ModelConfidence(M);
33end
34end
35end
36returnR={label, evidence, source, confidence};(true/false/nei), supporting evidence, source and confidence-
score.
V. EXPERIMENTALRESULTS ANDDISCUSSION
This section describes the experimental implementation and
evaluation of FinVet. It includes the models and tools used and
is followed by a comprehensive evaluation of its performance.
A. Experimental Setup
This subsection describes the technical components used to
implement FinVet, including vector embedding and storage
tools, LLMs, and external verification tools:
1) Vector Embedding and Storage:FinVet was imple-
mented using the text embedding model all-MiniLM-L6-v2
[22] [23]. This sentence transformer model generates compact
384-dimensional embeddings that effectively capture semantic
relationships between the concepts of a claim. The Facebook
AI Similarity Search (FAISS) library with InVerted File with
Flat storage (IVFFlat) architecture [24] was used to index and
retrieve the embeddings, enabling fast similarity search in large
volumes of financial statements.
2) RAG Pipeline Implementation:The dual RAG pipelines
were implemented using the following LLMs:
•RAG Pipeline-1was implemented using the LLaMA-3.3-
70B-Instruct model2[25], accessed through the Hugging
Face API. This transformer-based model was selected for
its strong general knowledge and reasoning capabilities.
•RAG Pipeline-2was implemented using the Mixtral-
8x7B-Instruct-v0.1 model3[26], accessed through the
Hugging Face API. This model was selected for its
complementary reasoning patterns through its Mixture-
Of-Experts (MOE) architecture.
3) Fact-Check Pipeline:This pipeline was implemented as
follows:
•The external verification component was interfaced with
the Google Fact Check Tools API [27].
•The LLM Analyzer component used the LLaMA-3.3-
70B-Instruct model.
B. Dataset and Evaluation Methodology
FinVet was evaluated using the FinFact dataset [3], a com-
prehensive collection of financial claims with expert-verified
labels and supporting evidence. The dataset was partitioned
into 85% training and 15% testing datasets. The training
dataset was used to populate the vector store, while the remain-
ing test dataset was used for performance evaluation purposes.
The approaches are evaluated using standard metrics: accuracy
(Acc.), precision (Prec.), recall (Rec.), and F1-score (F1).
2https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
3https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

C. Baseline Evaluation
FinVet is evaluated against three baseline approaches:
RAG (LLaMA-3.3-70B-Instruct), and Zero-shot and Chain-
of-Thought (GPT-3.5-turbo). Table I shows the results. FinVet
achieves an F1 score of 0.85, demonstrating substantial im-
provements over all baselines. The retrieval-augmented base-
line, RAG (LLaMA-3.3-70B), achieves an F1 of 0.62, with
notably higher precision (0.69) than recall (0.56), suggesting
conservative prediction behavior when using retrieval alone.
FinVet improves upon this by 37% in F1 score while achieving
more balanced precision (0.86) and recall (0.84).
The parametric baselines show further performance degra-
dation. Zero-shot GPT-3.5 achieves an F1 of 0.51, with Fin-
Vet outperforming it by 67%. Surprisingly, Chain-of-Thought
prompting performs worse, dropping to 0.45 F1 with the
lowest accuracy (0.37) despite moderate precision (0.56), indi-
cating that explicit reasoning without grounded evidence may
introduce errors in financial claim verification. FinVet’s 89%
improvement over CoT underscores the critical importance of
combining retrieval with external verification for this domain.
D. Ablation Study
To understand the contribution of individual components
within the FinVet architecture, we conducted a systematic
ablation study evaluating five configurations:
•FinVet (Full System): The complete pipeline integrat-
ing two RAG modules, a Fact-Check pipeline, and
confidence-weighted verdict integration.
•Fact-Check Pipeline: A hybrid system combining Google
Fact Check API with LLaMA-3.3-70B for fallback rea-
soning when there is no match in external fact-checks.
This configuration excludes RAG pipelines.
•Google Fact Check Only: Standalone external fact-
checking API without any LLM support or RAG pipelines
•RAG (LLaMA-3.3-70B): Single RAG pipeline using
LLaMA-3.3-70B with vector store retrieval, excluding
fact-checking components.
•RAG (Mixtral-8x7B): Single RAG pipeline using Mixtral-
8x7B with vector store retrieval, excluding fact-checking
components.
Table II shows the ablation results. The individual RAG
pipelines achieve F1 scores of 0.49 (Mixtral-8x7B) and 0.62
(LLaMA-3.3-70B), providing moderate baseline performance.
The Google Fact Check API alone also achieves 0.62, though
limited by its coverage of claims. Combining the API with
LLM reasoning in the Fact-Check Pipeline improves per-
formance to 0.77. Finally, FinVet’s full integration of both
RAG pipelines and fact-checking achieves 0.85, which is a
10.4% relative improvement over the best individual compo-
nent (Fact-Check Pipeline) and a 37% improvement over the
standalone RAG approaches. These empirical results demon-
strate that, for financial claim verification, the proposed hybrid
architecture integrating dual RAG pipelines with fact-checking
capabilities significantly outperform each constituent verifica-
tion method when deployed independently.As shown in Table I, the FinVet system outperforms the
zero-shot LLM baseline by a significant margin, improving
F1 score by 36%. This result underscores the limitations of
parametric-only reasoning in high-stakes, fact-sensitive finan-
cial contexts and highlights the importance of architectural
integration across diverse verification pathways.
TABLE I
PERFORMANCECOMPARISON OFFINVET ANDBASELINEAPPROACHES
Approach Acc. Prec. Rec. F1
FinVet 0.84 0.86 0.84 0.85
RAG (LLaMA-3.3-70B) 0.58 0.69 0.56 0.62
Zero-shot (GPT-3.5) 0.50 0.52 0.50 0.51
CoTa(GPT-3.5) 0.37 0.56 0.37 0.45
aCoT: Chain-of-Thought prompting.
TABLE II
ABLATIONSTUDY OFFINVETCOMPONENTS
Approach Acc. Prec. Rec. F1
FinVet 0.84 0.86 0.84 0.85
Fact-Check Pipelinea0.76 0.78 0.76 0.77
Google Fact Check Only 0.49 0.84 0.49 0.62
RAG (LLaMA-3.3-70B) 0.58 0.69 0.56 0.62
RAG (Mixtral-8x7B) 0.44 0.53 0.46 0.49
aGoogle Fact Check API + LLaMA-3.3-70B.
VI. ETHICALCONSIDERATIONS
Although FinVet is designed to enhance transparency in
financial misinformation detection, its implementation raises
several important ethical considerations.
1)Algorithmic and Data Bias: The LLM models used in
FinVet are subject to inherent algorithmic biases arising
from their pretraining objectives and training data. These
models often reflect and amplify societal, institutional,
or geographic biases present in the large-scale corpora
on which they are trained. As a result, the reasoning and
evidence generation steps in FinVet may inherit these
biases, potentially affecting the objectivity and fairness
of its outputs. The framework’s performance depends
on the quality and diversity of its knowledge base. If
training data contains biases or lacks representation of
certain financial domains, FinVet may perform inconsis-
tently across different types of financial claims. Lastly,
the reliance on external fact-checking APIs introduces
potential biases embedded in these services.
2)False Positives and Negatives: No detection system is
perfect. False positives could incorrectly flag legitimate
financial information, potentially harming honest ac-
tors. In contrast, false negatives could allow harmful
misinformation to spread. In this research, uncertainty
handling has been implemented through explicit ”NEI”
verdicts to mitigate unwarranted classifications, but users
should be aware of these limitations. To mitigate these
risks, transparency is advocated in the decision-making

process, including clear documentation of pipeline bi-
ases and confidence thresholds. For future deployments,
the incorporation of diverse sources of fact-checking
and regular audits is recommended to ensure fairness
and accountability. Interdisciplinary collaboration with
ethicists and financial regulators is also encouraged to
align the system with societal values and legal standards.
VII. CONCLUSION ANDFUTUREWORK
This research introduced FinVet, a novel framework for fi-
nancial misinformation detection that integrates RAG pipelines
with external fact-checking through a confidence-weighted
voting mechanism. Experimental results demonstrate that the
framework achieves an F1 score of 85%, which is an 8 %
improvement over the best individual pipeline. This substantial
performance gain validates three core design principles: the
complementary value of diverse verification pathways, the
importance of external verification source integration, and the
effectiveness of confidence-based verdict integration. More-
over, FinVet’s ability to generate evidence-based justifications,
confidence scores, and transparent source attributions ad-
dresses key requirements for financial institutions, regulators,
and oversight bodies that rely on auditable verification and
decision traceability.
Future research will incorporate additional financial knowl-
edge sources—such as regulatory filings, earnings reports,
market analyses, and financial news—to further enhance
domain-specific retrieval and coverage. In addition, semantic-
aware evaluation metrics will be explored to assess the quality,
credibility, and contextual relevance of the evidence supporting
each verdict.
To further improve the accuracy and adaptability of the
framework, the LLMs will be fine-tuned using financial-
domain instruction datasets, user feedback loops will be
implemented to continuously refine retrieval and verification
quality, and multilingual support will be extended to facilitate
the detection of cross-border financial misinformation more
effectively.
ACKNOWLEDGMENT
This project was supported by resources provided by the
Office of Research Computing at George Mason University
(Office of Research Computing, 2025). The authors also thank
the Department of Information Sciences and Technology for
providing priority access to GPU resources, which facilitated
our model training and experimentation.
REFERENCES
[1] F. Qian, C. Gong, and Y . Liu, “Neural user response generator: Fake
news detection with collective user intelligence,” inProceedings of the
Twenty-Seventh International Joint Conference on Artificial Intelligence
(IJCAI). Stockholm, Sweden: International Joint Conferences on
Artificial Intelligence Organization, 2018, pp. 3834–3840. [Online].
Available: https://doi.org/10.24963/ijcai.2018/533
[2] S. J. Parsan and M. Kim, “Canada’s approach to combating
fake news: An international analysis and recommendations,” 2024,
available at SSRN, Accessed: 2025-06-01. [Online]. Available:
https://papers.ssrn.com/sol3/papers.cfm?abstract id=4906538[3] A. Rangapur, H. Wang, L. Jian, and K. Shu, “Fin-fact: A benchmark
dataset for multimodal financial fact checking and explanation
generation,” 2024. [Online]. Available: https://arxiv.org/abs/2309.08793
[4] Deloitte, “The rise of digital fraud: Understanding and mitigating
emerging threats,” Deloitte US Advisory Services, Tech. Rep., 2023.
[Online]. Available: https://www2.deloitte.com/content/dam/Deloitte/us/
Documents/Advisory/us-advisory-digital-fraud.pdf
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
nlp tasks,” inAdvances in Neural Information Processing Systems
(NeurIPS), vol. 33. Curran Associates, Inc., 2020, pp. 9459–
9474. [Online]. Available: https://proceedings.neurips.cc/paper/2020/
file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[6] A. Dmonte, R. Oruche, M. Zampieri, E. Ko, and P. Calyam, “GMU-
MU at the financial misinformation detection challenge task: Exploring
LLMs for financial claim verification,” inProceedings of the Joint
Workshop of the 9th Financial Technology and Natural Language
Processing (FinNLP), the 6th Financial Narrative Processing (FNP),
and the 1st Workshop on Large Language Models for Finance and Legal
(LLMFinLegal). Singapore: Association for Computational Linguistics,
2025, pp. 308–312.
[7] J. Purbey, S. Gupta, N. Manali, S. Pullakhandam, D. Sharma, A. Srivas-
tava, and R. M. R. Kadiyala, “1-800-SHARED-TASKS at the financial
misinformation detection challenge task: Sequential learning for claim
verification and explanation generation in financial domains,” inPro-
ceedings of the Joint Workshop of the 9th Financial Technology and
Natural Language Processing (FinNLP), the 6th Financial Narrative
Processing (FNP), and the 1st Workshop on Large Language Models
for Finance and Legal (LLMFinLegal). Singapore: Association for
Computational Linguistics, 2025, pp. 302–307.
[8] N. Zhang and J. Liu, “Fake financial news detection with deep
learning: Evidence from china,”Procedia Computer Science, vol. 221,
pp. 154–160, 2023. [Online]. Available: https://doi.org/10.1016/j.procs.
2023.08.023
[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
of deep bidirectional transformers for language understanding,” 2018.
[Online]. Available: https://arxiv.org/abs/1810.04805
[10] Y . Cui, W. Che, T. Liu, B. Qin, and Z. Yang, “Pre-training with whole
word masking for chinese BERT,”IEEE/ACM Transactions on Audio,
Speech, and Language Processing, vol. 29, pp. 3504–3514, 2021.
[Online]. Available: https://doi.org/10.1109/TASLP.2021.3124365
[11] A. Kamal, P. Mohankumar, and V . K. Singh, “Financial misinformation
detection via RoBERTa and multi-channel networks,” inPattern
Recognition and Machine Intelligence (PReMI 2023), ser. Lecture
Notes in Computer Science, P. Maji, T. Huang, N. R. Pal, S. Chaudhury,
and R. K. De, Eds. Springer, 2023, vol. 14301, pp. 646–653. [Online].
Available: https://doi.org/10.1007/978-3-031-40716-5 72
[12] Y . Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V . Stoyanov, “RoBERTa: A robustly optimized
BERT pretraining approach,” 2019, Computing Research Repository.
[Online]. Available: https://arxiv.org/abs/1907.11692
[13] J. A. Nasir, O. S. Khan, and I. Varlamis, “Fake news detection: A
hybrid CNN-RNN based deep learning approach,”International Journal
of Information Management Data Insights, vol. 1, no. 1, p. 100007,
2021. [Online]. Available: https://doi.org/10.1016/j.jjimei.2020.100007
[14] X. Zhang, Q. Du, and Z. Zhang, “A theory-driven machine
learning system for financial disinformation detection,”Production and
Operations Management, vol. 31, no. 8, pp. 3160–3179, 2022. [Online].
Available: https://doi.org/10.1111/poms.13788
[15] X. Zhi, L. Xue, W. Zhi, Z. Li, B. Zhao, Y . Wang, and Z. Shen,
“Financial fake news detection with multi-fact CNN-LSTM model,”
in2021 IEEE 4th International Conference on Electronics Technology
(ICET). Chengdu, China: IEEE, 2021, pp. 1338–1341. [Online].
Available: https://doi.org/10.1109/ICET51757.2021.9451087
[16] J. A. Leite, O. Razuvayevskaya, K. Bontcheva, and C. Scarton,
“Detecting misinformation with LLM-predicted credibility signals and
weak supervision,”EPJ Data Science, vol. 13, no. 1, p. 34, 2024.
[Online]. Available: https://epjdatascience.springeropen.com/articles/10.
1140/epjds/s13688-025-00534-0
[17] Z. Liuet al., “FMDLlama: Financial misinformation detection based
on large language models,” 2024, computing Research Repository.
[Online]. Available: https://arxiv.org/abs/2409.16452

[18] C. G. Martin, “Financial truth guard,” 2021. [Online]. Available:
https://github.com/carlos-gmartin/Financial-Truth-Guard
[19] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez,
A. Joulin, E. Grave, and G. Lample, “Llama 2: Open foundation
and fine-tuned chat models,” 2023. [Online]. Available: https:
//arxiv.org/abs/2307.09288
[20] OpenAI, “Chatgpt (version 4),” 2024, Large Language Model, Accessed:
2025-06-01. [Online]. Available: https://www.openai.com/chatgpt
[21] H. Wan, S. Feng, Z. Tan, H. Wang, Y . Tsvetkov, and M. Luo, “DELL:
Generating reactions and explanations for LLM-based misinformation
detection,” 2024. [Online]. Available: https://arxiv.org/abs/2402.10426
[22] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “Minilm:
Deep self-attention distillation for task-agnostic compression of pre-
trained transformers,”Advances in Neural Information Processing Sys-
tems, vol. 33, pp. 5776–5788, 2020.
[23] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using siamese BERT-networks,” inProceedings of the 2019
Conference on Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP). Hong Kong, China: Association for
Computational Linguistics, 2019, pp. 3982–3992. [Online]. Available:
https://aclanthology.org/D19-1410
[24] H. Jegou, M. Douze, J. Johnson, L. Hosseini, and C. Deng, “Faiss:
Similarity search and clustering of dense vectors library,” 2022, software
Library, Astrophysics Source Code Library, ascl:2210.008, Accessed:
2025-06-01. [Online]. Available: https://github.com/facebookresearch/
faiss
[25] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian,
A. Al-Dahle, A. Letmanet al., “The Llama 3 Herd of Models,”
arXiv preprint arXiv:2407.21783, 2024. [Online]. Available: https:
//arxiv.org/abs/2407.21783
[26] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary,
C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand,
G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A.
Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao,
T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mixtral
of experts,” 2024. [Online]. Available: https://arxiv.org/abs/2401.04088
[27] G. Developers, “Rest resource: claims,” 2025, accessed: 2025-06-01.
[Online]. Available: https://developers.google.com/fact-check/tools/api/
reference/rest/v1alpha1/claims
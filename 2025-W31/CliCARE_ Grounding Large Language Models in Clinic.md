# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records

**Authors**: Dongchen Li, Jitao Liang, Wei Li, Xiaoyu Wang, Longbing Cao, Kun Yu

**Published**: 2025-07-30 10:02:16

**PDF URL**: [http://arxiv.org/pdf/2507.22533v1](http://arxiv.org/pdf/2507.22533v1)

## Abstract
Large Language Models (LLMs) hold significant promise for improving clinical
decision support and reducing physician burnout by synthesizing complex,
longitudinal cancer Electronic Health Records (EHRs). However, their
implementation in this critical field faces three primary challenges: the
inability to effectively process the extensive length and multilingual nature
of patient records for accurate temporal analysis; a heightened risk of
clinical hallucination, as conventional grounding techniques such as
Retrieval-Augmented Generation (RAG) do not adequately incorporate
process-oriented clinical guidelines; and unreliable evaluation metrics that
hinder the validation of AI systems in oncology. To address these issues, we
propose CliCARE, a framework for Grounding Large Language Models in Clinical
Guidelines for Decision Support over Longitudinal Cancer Electronic Health
Records. The framework operates by transforming unstructured, longitudinal EHRs
into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range
dependencies, and then grounding the decision support process by aligning these
real-world patient trajectories with a normative guideline knowledge graph.
This approach provides oncologists with evidence-grounded decision support by
generating a high-fidelity clinical summary and an actionable recommendation.
We validated our framework using large-scale, longitudinal data from a private
Chinese cancer dataset and the public English MIMIC-IV dataset. In these
diverse settings, CliCARE significantly outperforms strong baselines, including
leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The
clinical validity of our results is supported by a robust evaluation protocol,
which demonstrates a high correlation with assessments made by expert
oncologists.

## Full Text


<!-- PDF content starts -->

CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision
Support over Longitudinal Cancer Electronic Health Records
Dongchen Li1, Jitao Liang1, Wei Li1*, Xiaoyu Wang2†, Longbing Cao3‡, Kun Yu4
1College of Computer Science and Engineering, Northeastern University, Shenyang, China
2Liaoning Cancer Hospital & Institute, Shenyang, China
3Macquarie University, Sydney, Australia
4College of Medicine and Biological Information Engineering, Northeastern University, Shenyang, China
2490254@stu.neu.edu.cn, 2472127@stu.neu.edu.cn, liwei@cse.neu.edu.cn, wangxyz007@hotmail.com,
longbing.cao@mq.edu.au, yukun@bmie.neu.edu.cn
Abstract
Large Language Models (LLMs) hold significant promise for
improving clinical decision support and reducing physician
burnout by synthesizing complex, longitudinal cancer Elec-
tronic Health Records (EHRs). However, their implementa-
tion in this critical field faces three primary challenges: the
inability to effectively process the extensive length and mul-
tilingual nature of patient records for accurate temporal anal-
ysis; a heightened risk of clinical hallucination, as conven-
tional grounding techniques such as Retrieval-Augmented
Generation (RAG) do not adequately incorporate process-
oriented clinical guidelines; and unreliable evaluation metrics
that hinder the validation of AI systems in oncology. To ad-
dress these issues, we propose CliCARE, a framework for
Grounding Large Language Models in Clinical Guidelines
for Decision Support over Longitudinal CAncer Electronic
Health REcords. The framework operates by transforming
unstructured, longitudinal EHRs into patient-specific Tempo-
ral Knowledge Graphs (TKGs) to capture long-range depen-
dencies, and then grounding the decision support process by
aligning these real-world patient trajectories with a norma-
tive guideline knowledge graph. This approach provides on-
cologists with evidence-grounded decision support by gener-
ating a high-fidelity clinical summary and an actionable rec-
ommendation. We validated our framework using large-scale,
longitudinal data from a private Chinese cancer dataset and
the public English MIMIC-IV dataset. In these diverse set-
tings, CliCARE significantly outperforms strong baselines,
including leading long-context LLMs and Knowledge Graph-
enhanced RAG methods. The clinical validity of our results
is supported by a robust evaluation protocol, which demon-
strates a high correlation with assessments made by expert
oncologists.
Code — https://github.com/sakurakawa1/CliCARE
1 Introduction
Large Language Models (LLMs) are emerging as promis-
ing tools for clinical decision support, with current re-
*Corresponding author.
†Corresponding author.
‡Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: The shared challenges for clinicians and LLM in
handling complex longitudinal EHRs.
search demonstrating their potential to serve as collabora-
tive partners that augment expert workflows, alleviate clin-
ician workloads, and improve decision-making in complex
fields such as oncology (Hager et al. 2024; Rajashekar et al.
2024). However, their integration into high-stakes clinical
practice is far from straightforward. The reality of clinical
oncology involves physicians navigating immense cognitive
burdens from manually integrating fragmented data within
multi-year Electronic Health Records (EHRs), a key con-
tributor to professional burnout (Warner et al. 2020; Sinsky
et al. 2016). This chaotic, unstructured environment ampli-
fies the critical disparity between LLMs’ high performance
on standardized benchmarks and their actual capabilities
in the clinic. Indeed, systematic reviews indicate that their
performance in cancer decision-making is inconsistent with
critical safety aspects frequently unaddressed (Hao et al.
2025), while other recent studies confirm that even state-of-
the-art models struggle to adhere to treatment guidelines or
accurately interpret laboratory results (Hager et al. 2024).
Therefore, the frontier of this field is not merely the devel-
opment of more powerful models, but the creation of ro-
bust frameworks that ensure these technologies are reliable,
safe, and effectively grounded in expert medical knowledge
to truly augment, not supplant, the role of the physician.In
practice, augmenting the expert physician’s role means sup-arXiv:2507.22533v1  [cs.CL]  30 Jul 2025

porting their core clinical workflow: synthesizing a patient’s
multi-year history into a coherent Clinical Summary, and
from that summary, generating an actionable Clinical Rec-
ommendation for future treatment.
However, automating this expert workflow with existing
LLMs faces three fundamental challenges. First, LLMs ex-
hibit a significant inability to perform effective temporal
reasoning over the extensive data records typical of can-
cer EHRs. Our research addresses a corpus containing large
samples of patient records, where a single patient’s history
can span years, exceed 20,000 tokens, and even include
multilingual entries, making brute-force approaches ineffi-
cient (Liu et al. 2024). The second challenge is the unac-
ceptable risk of clinical hallucination, which undermines
the potential for reliable decision support. Factually incor-
rect recommendations pose a direct threat to patient safety,
a risk that is exacerbated by the limitations of standard
Retrieval-Augmented Generation methods. The retrieval of
fragmented text fails to capture the sequential dependencies
in a patient’s trajectory and cannot effectively bridge the gap
with process-oriented clinical guidelines (Li et al. 2024). Fi-
nally, the field confronts two interconnected barriers to real-
world adoption. The deployment dilemma centers on a fun-
damental trade-off. On one hand, powerful, closed-source
models offer state-of-the-art performance but raise signifi-
cant concerns regarding cost and patient data privacy. On the
other hand, open-source alternatives are more efficient and
easier to deploy locally, though they often lag in capability.
This trade-off is compounded by the significant challenge of
reliable evaluation, as the high-stakes nature of clinical con-
tent renders conventional automated metrics untrustworthy,
thereby hindering reliable progress and diminishing clinical
trust (Wang et al. 2023; Zheng et al. 2023).
To address these barriers, we propose CliCARE, a frame-
work for Grounding Large Language Models in Clinical
Guidelines for Decision Support over Longitudinal CAncer
Electronic Health REcords. CliCARE first tackles long-
context temporal analysis by structuring raw EHRs into
Temporal Knowledge Graphs (TKGs) to make temporal re-
lationships explicit (Sec 3.1). It then mitigates hallucina-
tions by grounding the model through a deep alignment of
patient trajectories with clinical guidelines (Sec 3.2). This
structured representation provides both gold-standard data
for fine-tuning specialist models and rich context for large
generalist models. Finally, we ensure reliable evaluation via
a human-validated LLM-as-a-Judge protocol whose ratings
highly correlate with expert judgments (Sec 4.1). Our source
code is provided in the supplementary material for repro-
ducibility.
Our contributions are summarized below:
• We introduce CliCARE, an end-to-end framework that
grounds LLMs by transforming EHRs into TKGs and
aligning them with clinical guidelines, featuring an
adaptable architecture for both generalist and specialist
models.
• We propose a reliable evaluation methodology using a
Human-Validated LLM-as-a-Judge, whose ratings show
strong Spearman’s correlation with expert oncologists,addressing the limitations of standard automated metrics.
• Extensive experiments on diverse datasets show Cli-
CARE significantly outperforms robust baselines, while
ablation studies confirm the contribution of each compo-
nent.
2 Related Work
2.1 LLMs with Long-Form EHRs
Applying LLMs to long-form EHRs for clinical decision
support is fundamentally constrained by the challenge of
long-context processing. The evaluation of these challenges
has been systematized through benchmarks such as Long-
Bench (Bai et al. 2024), with specialized medical bench-
marks like MedOdyssey emerging to assess these capabil-
ities in a clinical context (Fan et al. 2024). Prominent issues
include the ”lost-in-the-middle” problem and the degrada-
tion of performance during domain-specific fine-tuning (Liu
et al. 2024; Zhang et al. 2024; Yang et al. 2024). While
economical solutions like Parameter-Efficient Fine-Tuning
(PEFT) show promise, these technical advances alone are
often insufficient for achieving clinically meaningful out-
comes without structured knowledge to guide the model
(Dong et al. 2024; Zhang et al. 2023; Nazary et al. 2024).
This need has led to a primary strategy of transforming un-
structured data into structured formats, a critical step for the
robust temporal analysis necessary for summarizing patient
journeys. However, while methods such as Patient Journey
Knowledge Graphs (PJKGs) exist (Khatib et al. 2025), their
variable accuracy and efficiency present reliability chal-
lenges for downstream decision support tasks.
2.2 Knowledge Graph-enhanced LLMs and RAG
Augmenting LLMs with external Knowledge Graphs (KGs)
is a crucial strategy for mitigating factual errors and halluci-
nations, which is essential for safety in high-stakes domains
such as healthcare (Khan, Wu, and Chen 2024). This ap-
proach helps bridge the gap between general-purpose mod-
els and specialized clinical knowledge (Yu and McQuade
2025). However, standard Retrieval-Augmented Generation
(RAG) retrieves isolated text snippets, overlooking the com-
plex relational structures necessary for effective clinical
decision-making (Lewis et al. 2020; Liu et al. 2024). This
limitation has prompted the development of Graph-Aware
RAG, which retrieves structured subgraphs instead of dis-
connected text. Frameworks such as MedRAG, GNN-RAG,
and KG2RAG utilize this method to enhance model perfor-
mance with domain-specific KGs (Zhao et al. 2025; Feng
et al. 2024; e Shi et al. 2024). A more advanced frontier
moves beyond retrieval to the alignment and fusion of KGs
and LLMs at the representation level, projecting both into
a unified semantic space for more nuanced, topology-aware
outputs (Jiang et al. 2024). This synergy creates a virtuous
cycle, where KGs not only ground LLMs, but LLMs are
also increasingly used to construct and enrich KGs from text
(Maushagen et al. 2024).

Figure 2: A comparison of RAG approaches for long-form longitudinal clinical tasks. (a) Standard RAG often suffers from
missing key information and hallucinations. (b) KG-enhanced RAG struggles to model temporal dependencies in patient jour-
neys. (c) In contrast, our CliCARE framework transforms EHRs into Temporal Knowledge Graphs, aligns patient trajectories
with guidelines, and generates answers using a distilled specialist model, which are then assessed by our proposed evaluation
approach.
2.3 Assessment of Open-Ended Clinical
Generation Tasks
Evaluating open-ended generation from LLMs in high-
stakes medical domains presents a critical challenge. Tradi-
tional automated metrics, such as ROUGE and BLEU, are
widely regarded as inadequate because their emphasis on
lexical overlap fails to capture essential aspects like clini-
cal validity, factual accuracy, and safety (Wang et al. 2023;
Singhal et al. 2023). In response, research has increasingly
focused on more nuanced evaluation methods, including dy-
namic agent-based assessments (Tu et al. 2024; Tadevosyan
et al. 2025) and the scalable ”LLM-as-a-Judge” paradigm
(Zheng et al. 2023). However, the reliability of LLM judges
is compromised by known systematic biases like positional
bias and verbosity, raising significant safety concerns in a
field where deep domain expertise is essential (Zheng et al.
2023; Wang et al. 2024). This highlights the urgent need
for rigorous methodologies to validate automated judgments
against human expert reasoning.
Existing research has treated long-context processing,
knowledge grounding, and reliable evaluation as distinct
challenges. A significant research gap exists in developing
a solution that simultaneously addresses the long-context
limitations in real-world EHRs, provides deep grounding in
clinical guidelines that exceeds standard RAG, and guaran-
tees trustworthy assessment. CliCARE is designed to bridge
this gap by integrating these capabilities into a unified, co-hesive pipeline.
3 CliCARE
In this section, we present the CliCARE framework, as il-
lustrated in Figure 2. CliCARE is designed to systematically
analyze long-form, unstructured cancer EHRs to generate
a clinically grounded clinical summary and clinical recom-
mendations. A key feature of this design is its extensibility;
the guideline knowledge graph can be efficiently updated
and expanded to accommodate new clinical evidence and
corresponding guidelines.
3.1 EHR-to-TKGs Transformation
The initial stage of CliCARE transforms raw, multi-year
EHRs from unstructured text into patient-centric TKGs,
effectively addressing the fundamental challenge of long-
context temporal reasoning.
Event Extraction. The complete EHR for each pa-
tient pcan be formalized as a sequence of documents
Dp= (dτ1, dτ2, . . . , d τn)ordered by timestamps Tp=
{τ1, τ2, . . . , τ n}, where each document dτiis an unstruc-
tured or semi-structured clinical text at time τi. To man-
age this extensive text sequence, we developed an efficient
context processing pipeline, fpipeline , to systematically com-
press, refine, and structure the raw text before it is input into
the final pathway generation model.
Ep=fpipeline (Dp) (1)

Here, Eprepresents a structured sequence of key clini-
cal events. Specifically, the core of fpipeline is an extrac-
tive summarization module based on the Longformer model.
Given the computational cost of processing the entire Dp,
we partition the document sequence into the most re-
cent clinical note, dτn, and the historical records, Dhist
p=
(dτ1, . . . , d τn−1). We utilize a Longformer model, MLF,
pre-trained on clinical text, to process the extensive histori-
cal records Dhist
p:
Shist
p=MLF(Dhist
p;θLF) (2)
where θLFare the model parameters and Shist
pis a summary
text that includes the most informative sentences from the
historical records. This summary effectively functions as
the patient’s ”past medical history.” The most recent clini-
cal note, dτn, is regarded as the ”history of present illness.”
The final structured event sequence Epis created by con-
catenating these two components, thus providing a chrono-
logically coherent and condensed patient history. From this
combined text, key clinical facts—such as diagnostic con-
firmations, staging updates, treatment regimens, biomarker
trends, and imaging assessments—are identified and orga-
nized into discrete events (Yang, Wang, and Li 2021; Huang,
Altosaar, and Ranganath 2019).
TKG Instantiation. Extracted event sequences Epare
organized into a patient-centric Clinical TKG, denoted as
Gt= (Et, Rt, T), where Etis the set of entities, Rt
the set of relations, and Tthe set of timestamps. To en-
rich the TKG with standardized medical knowledge, we
first construct a general, static biomedical knowledge graph
GB= (EB,RB), where EBcontains standardized medi-
cal concepts and RBrepresents the relations between them.
For each patient, letting Epbe the set of raw clinical en-
tities extracted from the patient’s record, we instantiate a
personalized TKG Gtby linking extracted clinical entities
from the patient’s record to the concepts in GB. This is
achieved via an entity linking function ϕ:Ep→ E B, which
maps textual mentions in the EHR to their corresponding
canonical entries in the biomedical ontology. Each entity
e∈ Etin the TKG is a spatiotemporal instance represented
ase= (eB, τ, A ), where eB∈ EBis the linked standard
entity, τ∈ T is the event timestamp, and Ais a set of event-
specific attributes.
The TKG employs a hierarchical timestamp granularity
by assigning precise timestamps τ∈Tonly to macro-level
clinical encounters, while linking intra-encounter events
through relative temporal relations, thereby mirroring the
structure of real-world clinical records to designed to cap-
ture capture the dynamic evolution of a patient’s disease
course.
3.2 Trajectory-Guideline Alignment
To integrate real-world patient data with normative medi-
cal knowledge, this stage aligns the descriptive patient TKG
with a prescriptive guideline KG through a training-free fu-
sion pipeline, as illustrated in Figure 3.
Figure 3: Our Trajectory-Guideline Alignment workflow. It
fuses patient data with guidelines via semantic matching,
LLM-based re-ranking, and iterative bootstrapping expan-
sion to create a comprehensive, evidence-grounded map-
ping.
Knowledge Formalization. Our guideline knowledge
graph, Gg, is a normative, static graph formalized as Gg=
(Eg, Rg). It is constructed based on authoritative CPGs
(e.g., NCCN), where Egrepresents abstract medical con-
cepts such as Cancer ,ClinicalSituation , and Treatment . The
edges, Rg, represent logical and recommendation relation-
ships, collectively forming a graph that represents an ideal-
ized clinical workflow. We first perform a basic entity-level
alignment between the patient’s temporal graph, Gt, andGg
using medical ontologies. Subsequently, the clinical history
for each patient pis organized from Gtinto a time-ordered
sequence of Clinical Events, resulting in a Temporal Trajec-
toryTrp=⟨e1, e2, . . . , e m⟩. The set of all patient trajec-
tories is denoted as {Trp}P
p=1. Concurrently, we systemati-
cally enumerate all possible normative treatment workflows
from Ggto form a set of paths {Pak}K
k=1, where each path
Pak=⟨s1, s2, . . . , s l⟩represents a recommended sequence
of guideline steps.
Similarity Matching. We developed a global matching
strategy based on deep semantic representations to directly
assess the similarity between the entire patient trajectory
and each candidate guideline path. Specifically, the overall
matching score for a candidate path Pak=⟨s1, s2, . . . , s l⟩
with the patient trajectory Trpis computed as follows. First,
for each step sjin the guideline path, we identify the most

semantically similar event in the patient’s complete event
trajectory Trp. This similarity is calculated using a BERT
model, fBERT, pre-trained on biomedical text. Finally, we
aggregate the best-match similarities for all steps to derive
the total score for the path. Formally, the matching score is
defined as:
Score (Trp, Pak) =lX
j=1max
ei∈Trp
cossim(fBERT(desc(sj)),
fBERT(desc(ei)))
(3)
where the desc (·)function retrieves the text description of
a node, and cos sim computes the cosine similarity between
two vectors. A higher score indicates a better alignment be-
tween the normative path Pakand the patient’s experience.
The optimal matching path Pa∗is then determined by se-
lecting the candidate with the highest score:
Pa∗= arg max
PakScore (Trp, Pak) (4)
This method relies primarily on the deep semantic under-
standing provided by BERT, transcending simple lexical
matching to capture conceptual associations between clin-
ical events and guideline steps, thereby facilitating precise
trajectory-path alignment.
LLM-based Reranking. The aforementioned method
generates a ranked list of candidate alignment paths for
each patient trajectory. However, purely algorithmic match-
ing may fail to capture the complexities of clinical logic.
Therefore, we introduce an LLM as a Clinical Reasoner,
fLLM , to perform reranking. We provide the LLM with a
rich-context prompt in a zero-shot manner, which includes
the patient’s trajectory Trp, the top-N candidate normative
paths{Pa1, . . . , Pa N}, and their corresponding matching
scores {Score 1, . . . , Score N}. The LLM’s task is to evalu-
ate which candidate alignment is the most clinically plau-
sible and to output a reranked list: ⟨Pa′
1, . . . , Pa′
N⟩=
fLLM(Trp,{⟨Pak,Score k⟩}N
k=1).
Alignment Expansion. To further enhance the coverage
and accuracy of our alignments, we introduce an expansion
stage inspired by bootstrapping techniques (Sun et al. 2018).
After the LLM reranking, the top-ranked alignment path,
Pa′
1, and its corresponding aligned node pairs serve as a
high-confidence seed set, A′. We then iteratively expand this
set. For each unaligned event euin the patient trajectory, the
framework seeks to identify the best corresponding node ˆs
from the entire guideline path Pa′
1. The selection process is
not based solely on direct similarity; rather, it considers how
well the candidate pair (eu, sv)coheres with the entire set of
existing high-confidence alignments in A′. This is accom-
plished by selecting the guideline node svthat maximizes
the sum of consistency scores with all established pairs in
the seed set. The process is formalized as follows:
ˆs= arg max
sv∈Pa′
1X
(ei,sj)∈A′sim((eu, sv),(ei, sj)) (5)where sim is a function that measures the consistency be-
tween a candidate pair (eu, sv)and an existing seed pair
(ei, sj). This function utilizes the semantic representations
derived from the language model fBERT to compute the sim-
ilarity between the corresponding nodes within the pairs.
A high consistency score indicates that the semantic re-
lationship between the patient event euand the candidate
guideline node svis analogous to the established, high-
confidence relationship between the seed event eiand the
guideline node sj. This approach enables us to leverage es-
tablished strong associations to infer new alignment rela-
tionships, thereby expanding our alignment set A′. After de-
termining the final expanded alignment path, we employ a
principled fusion strategy to enrich the guideline knowledge
graph Ggwith evidence from the patient trajectory Trp. Ul-
timately, this alignment process produces a robust, evidence-
fused knowledge representation that serves as a direct, high-
quality context for an LLM to generate its final clinical sum-
mary and clinical recommendation.
4 Experiments
4.1 Evaluation Method
To assess the quality of generated text, we developed
aHuman-validated LLM-as-a-Judge component. This
component assesses two primary tasks: retrospective Clin-
ical Summary, referred to as TCS, and prospective Clinical
Recommendation, referred to as TCR. Our methodology em-
ploys a concise, four-dimensional rubric, co-designed with
senior oncologists (see Appendix A for details), to assess
Factual Accuracy ,Completeness & Thoroughness ,Clinical
Soundness , and Actionability & Relevance . A robust LLM
judge is prompted to assign a score ranging from 1 (poor) to
5 (excellent) for each dimension.
To address the known systematic biases of LLM judges,
including positional bias, verbosity, and self-enhancement
(Zheng et al. 2023; Wang et al. 2024), and after verifying
their presence in our specific context (Appendix E), we im-
plement a robust two-part mitigation protocol. First, to en-
sure rating stability and reduce model bias, we create a judg-
ing ensemble composed of three powerful LLMs—GPT-4.1,
Claude 4.0 Sonnet, and Gemini 2.5 Pro—using their aver-
aged score. Second, to eliminate ordering effects, all items
are presented in a randomly shuffled sequence during the
evaluation.
We validated the ratings of our LLM judge against those
of three experienced oncologists using a subset of data to
ensure reliability. We employed Spearman’s rank correlation
coefficient, denoted as ρ, a non-parametric measure that as-
sesses the monotonic relationship between the LLM’s and
the experts’ rankings. The coefficient is calculated as:
ρ= 1−6Pd2
i
n(n2−1)(6)
where nis the number of samples and diis the difference in
ranks for each sample. A high correlation provides strong
evidence that our LLM judge functions as a reliable and
scalable proxy for human expert assessment. This justifies
its application for large-scale evaluations throughout our ex-
periments.

4.2 Datasets
We evaluated our framework using two large-scale clin-
ical datasets, with detailed analyses provided in Appen-
dices B and C. The first dataset is a private Chinese col-
lection, referred to as CancerEHR. It contains longitudinal
records for 2,000 patients from Liaoning Cancer Hospital.
These records span extensive periods—some exceeding two
decades—resulting in inputs of up to 20,000 tokens. The
dataset includes a variety of data types, such as physicians’
orders, laboratory results, and surgical notes. The second
dataset is derived from the publicly available MIMIC-IV
dataset (Johnson et al. 2023), filtered to include only patients
with cancer-related diagnoses, which we refer to as MIMIC-
Cancer. This dataset provides a focus on disease progres-
sion similar to CancerEHR; however, the language and data
structure differ, offering a robust test of our method’s gen-
eralizability. For brevity, in the subsequent implementation
and results sections, we will refer to the two datasets as
DCEHR for CancerEHR and DMCfor MIMIC-Cancer. Sim-
ilarly, the two primary tasks will be abbreviated as TCSfor
Clinical Summary and TCRfor Clinical Recommendation.
4.3 Baselines
We compare our proposed CliCARE framework against
a variety of robust baseline methods. These include stan-
dard retrieval-augmented generation (RAG) pipelines imple-
mented with powerful open-source models such as Mistral-
7B and its instruction-tuned variant (Jiang et al. 2023),
Qwen3-8B (Yang et al. 2025), and the domain-specific
BioMistral-7B (Labrak et al. 2024). Additionally, we evalu-
ate more advanced KG-enhanced RAG techniques designed
for long-context or knowledge-intensive tasks. The selected
methods include BriefContext (Zhang et al. 2024), which
employs a Map-Reduce strategy, as well as several Graph-
Aware RAG frameworks, such as GNN-RAG (Feng et al.
2024), KG2RAG (e Shi et al. 2024), and the healthcare-
focused MedRAG (Zhao et al. 2025).
4.4 Implementation Details
In the knowledge graph alignment stage, the threshold is
set to 0.7 when using BERT to calculate semantic (cosine)
similarity in the initial step. During the fine-tuning stage,
we divided the 2,000-sample dataset into a training set of
1,800 samples and a test set of 200 samples, with 10% of
the training data reserved for validation. The key hyperpa-
rameters for training include a batch size of 1, a maximum
context length of 20,000 tokens, and an initial learning rate
of5×10−5with a cosine scheduler. We utilized BF16 for
mixed-precision training, set the maximum output length
to 4,096 tokens, and trained for 3 epochs. All experiments
were conducted using a configuration of four NVIDIA A800
GPUs.
4.5 Experimental Results
High Agreement with Clinician Judgments. Acknowl-
edging the limitations of traditional metrics for clinical
tasks, we validated our LLM-as-a-Judge protocol againstMethodDCEHR DMC
TCS TCR TCS TCR
Qwen-3-8B
StandardRAG 1.485 1.527 2.475 2.467
BriefContext 2.681 2.701 2.571 2.497
MedRAG* 2.333 2.366 2.495 2.462
KG2RAG* 2.595 2.558 2.317 2.166
GNN-RAG* 2.508 2.527 2.194 2.182
CliCARE 3.173 3.215 2.575 2.544
Gemini 2.5 Pro
StandardRAG 2.735 2.818 3.563 3.556
BriefContext 4.527 4.468 4.354 4.233
MedRAG* 4.470 4.576 4.476 4.323
KG2RAG* 3.845 3.942 3.747 3.797
GNN-RAG* 3.607 3.552 3.683 3.588
CliCARE 4.976 4.965 4.398 4.333
Table 1: CliCARE Outperforms RAG Baselines on Clini-
cal Generation Tasks. Scores are assigned by our Human-
validated LLM-as-a-Judge. The asterisk (*) denotes KG-
enhanced RAG variants.
three experienced oncologists. To ensure a feasible yet rep-
resentative assessment, we created a validation subset by
randomly sampling generated outputs from eight different
models. Our protocol minimized bias by evaluating these
outputs column-wise and presenting the Clinical Summary
and Recommendation tasks together for a comprehensive
review (details in Appendix D). The results demonstrate a
strong positive correlation between the automated ratings
and those of the experts. Specifically, the Spearman’s rank
correlation ( ρ) between our LLM judge’s scores and the
physicians’ mean scores was approximately 0.7, confirming
that our metric serves as a reliable proxy for human expert
judgment.
CliCARE Significantly Outperforms Baselines. As de-
tailed in Table 1, CliCARE demonstrates a clear perfor-
mance advantage over a suite of robust baselines, including
both context-aware and KG-enhanced RAG methods. The
benefits of our framework are most pronounced when paired
with a powerful model on complex datasets. With Gemini
2.5 Pro, CliCARE achieves impressive Clinical Summary
and Recommendation scores of 4.976 and 4.965, respec-
tively, on the challenging CancerEHR dataset. This perfor-
mance significantly surpasses that of other methods. For in-
stance, while BriefContext achieves a commendable score of
4.527, it does so through a costly Map-Reduce strategy that
involves multiple LLM calls, underscoring the efficiency
of CliCARE’s approach. Even when utilizing a smaller
model like Qwen-3-8B, CliCARE obtains scores of 3.173
and 3.215, substantially outperforming all baselines on the
same complex dataset. This success is attributed to Cli-
CARE’s TKG transformation, which effectively organizes
the chaotic, longitudinal patient records and overcomes the
fragmented retrievals that hinder other RAG pipelines.
Structured Knowledge is Key for Complex EHRs. As
demonstrated in Table 2, our framework’s knowledge struc-

MethodStandard RAG CliCARE
DCEHR DMC DCEHR DMC
TCS TCR TCS TCR TCS TCR TCS TCR
Mistral-v0.1-7B 1.120 1.164 2.505 2.505 1.407 (+0.287) 1.526 (+0.362) 2.575 (+0.070) 2.514 (+0.009)
Mistral-Instruct-v0.1-7B 1.054 1.070 2.183 2.115 1.274 (+0.220) 1.355 (+0.285) 2.231 (+0.048) 2.071 (-0.044)
Biomistral-7B 1.161 1.098 2.785 2.698 1.548 (+0.387) 1.529 (+0.431) 2.903 (+0.118 ) 2.742 (+0.044)
Qwen-3-8B 1.485 1.527 2.475 2.467 3.173 (+1.688 ) 3.215 (+1.688 ) 2.575 (+0.100) 2.544 ( +0.077 )
Gemini-2.5-Pro 2.735 2.818 3.563 3.556 4.976 (+2.241) 4.965 (+2.147) 4.398 ( +0.835 ) 4.333 ( +0.777 )
GPT-4.1 2.667 2.873 4.419 4.429 4.690 (+2.023) 4.703 (+1.830) 4.737 (+0.318) 4.676 (+0.247)
Deepseek-R1 2.667 2.878 4.016 4.000 4.946 ( +2.279 ) 4.935 ( +2.057 ) 4.409 (+0.393) 4.319 (+0.319)
Claude-4.0-Sonnet 2.417 2.624 3.898 3.868 3.893 (+1.476) 3.924 (+1.300) 4.183 (+0.285) 4.110 (+0.242)
Table 2: Model performance with standard RAG versus the CliCARE framework. Applying CliCARE provides a substantial
performance uplift for most models.
turing offers a significant advantage over standard RAG.
The performance uplift is most pronounced on the complex
CancerEHR dataset, where nearly all models exhibit sub-
stantial gains. Notably, the improvements for Qwen-3-8B
and Deepseek-R1 are the largest in their respective groups,
with their Clinical Summary scores increasing by a remark-
able +1.688 and +2.279, respectively. This underscores that
even advanced models require a coherent structure for effec-
tive reasoning on complex records. On the simpler MIMIC-
Cancer dataset, while the absolute gains are smaller, Cli-
CARE still delivers a distinct and consistent advantage. For
instance, it elevates the score of a strong baseline like GPT-
4.1 from 4.419 to 4.737, a gain of +0.318. While the uplift
is nearly universal, we do note a single case of minor perfor-
mance degradation, confirming the intricate nature of these
tasks.
MethodDCEHR DMC
TCS TCR TCS TCR
CliCARE (Q) 3.173 3.215 2.575 2.544
w/o Exp. 3.012 (-) 3.035 (-) 2.075 (-) 2.110 (-)
w/o Rerank 2.857 (-) 2.866 (-) 2.000 (-) 1.962 (-)
w/o Comp. 1.485 (-) 1.527 (-) 2.475 (+) 2.467 (+)
CliCARE (G) 4.976 4.965 4.398 4.333
w/o Exp. 4.619 (-) 4.630 (-) 3.737 (-) 3.786 (-)
w/o Rerank 4.542 (-) 4.628 (-) 3.774 (+) 3.824 (+)
w/o Comp. 2.735 (-) 2.818 (-) 3.563 (-) 3.556 (-)
Table 3: Ablation study on CliCARE framework compo-
nents. Q denotes Qwen-3-8B and G denotes Gemini-2.5-
Pro. Exp., Rerank and Comp. signify the removal of Align-
ment Expansion, LLM-based Reranking, and TKG-based
Compression, respectively. The symbols (+)/(-) indicate a
performance increase/decrease compared to the row above.
Ablation Study. Our ablation study, with results in Ta-
ble 3, reveals the nuanced role of each module. This is
most evident for the Qwen model on the simpler MIMIC-
Cancer dataset; removing TKG-based Compression para-
doxically boosts the scores to 2.475 and 2.467. This re-sult is substantially better than when LLM-based Reranking
is removed, which causes a drop to 2.000, suggesting ag-
gressive compression can be counterproductive for shorter
records. A similar, though less pronounced, positive effect is
observed for the Gemini model under the same conditions.
Conversely, on the complex CancerEHR dataset, the consis-
tent, significant performance drops from any ablation high-
light that the full, integrated CliCARE framework is crucial
for achieving optimal performance.
MethodDCEHR DMC
TCS TCR TCS TCR
CliCARE(Qwen-3-8B)
All (100%) 3.173 3.215 2.575 2.544
Short (0˜33%) 3.228 3.345 2.850 2.645
Medium(33%˜66%) 3.267 3.283 2.429 2.450
Long (66%˜100%) 2.976 2.983 2.460 2.533
CliCARE(Gemini-2.5-Pro)
All (100%) 4.976 4.965 4.398 4.333
Short (0˜33%) 4.982 4.937 4.362 4.311
Medium(33%˜66%) 4.962 4.976 4.365 4.317
Long (66%˜100%) 4.988 4.982 4.467 4.373
Table 4: Performance analysis by EHR length. Segments are
stratified by percentile (0-33%, 33-66%, 66-100%). Aver-
age token counts for CancerEHR segments are 4875, 6303,
9411; for MIMIC-Cancer, 4070, 5176, 6463.
Performance Analysis Based on EHR Length. Further
analysis of record length reveals distinct performance pat-
terns, as detailed in Table 4. The smaller model, Qwen-3-
8B, performs optimally on Short length records but exhibits
a significant decline in quality when processing the longest
records, particularly with the complex CancerEHR data. In
contrast, the more powerful Gemini-2.5-Pro model demon-
strates strong and consistent performance across all record
lengths. Notably, when guided by the CliCARE framework,
it achieves its highest scores on the longest record segments
for both datasets. This finding suggests that CliCARE ef-
fectively organizes extensive clinical histories enables ad-
vanced models to leverage richer context for enhanced rea-

soning.
5 Conclusion
We introduced CliCARE, a framework for reliable clinical
decision support that transforms cancer EHRs into Temporal
Knowledge Graphs and aligns them with clinical guidelines.
This approach addresses key challenges in long-context rea-
soning and hallucination, enabling both small specialist and
large generalist models to significantly outperform strong
baselines. We also validated a robust LLM-as-a-Judge pro-
tocol that correlates highly with expert oncologist assess-
ments, representing a significant advancement toward de-
ploying trustworthy AI in clinical practice.
A Human Evaluation Questionnaire and
Protocol
Using a custom-built online questionnaire platform, we in-
structed practicing oncologists to evaluate two generated
outputs: the Clinical Summary and the Clinical Recommen-
dation.
For each evaluation task, the clinician was presented with
the following materials:
• A patient’s complete longitudinal Electronic Health
Record (EHR) from the CancerEHR dataset, containing
multiple encounters ( record 1,record 2, etc.).
• A human-expert-authored, gold-standard Clinical Sum-
mary and a corresponding Clinical Recommendation
(collectively, the ”label”), which together provide the
ideal summary and recommendation based on the EHR.
• Anonymized outputs from a random sample of
eight models—including both locally deployed
(CliPAGE 8B,etc.) and API-based ( CliPAGE ,etc.)
systems—were compared against each other and the
gold-standard labels.
The clinicians were then instructed to provide the follow-
ing series of assessments:
• Overall Head-to-Head Comparison: The overall quality
of the 8 models was directly compared using a 5-point
Likert scale, ranging from 1 (”Very Poor”) to 5 (”Very
Good”).
• Task-Specific Head-to-Head Comparison: Separate
head-to-head comparisons were performed for two key
tasks: ”Clinical Summary” and ”Clinical Recommen-
dations.” This evaluation focused specifically on the
quality of these two components within the generated
output.
B CancerEHRs Dataset Details and
Demographics
The CancerEHR dataset is a unique, non-public collection
of Chinese Electronic Health Records sourced from a large,
specialized cancer hospital in China. The dataset underwent
a carefully designed, multi-step processing pipeline to ulti-
mately create formatted text suitable for LLM input.B.1 Data Processing Pipeline.
In practice, the raw EHR data underwent a series of
processing steps. The core objective was to consolidate
the heterogeneous and fragmented raw data from the
Hospital Information System (HIS) into a patient-centric,
chronologically organized, unified text format. We de-
signed and wrote a series of specialized Python pars-
ing scripts for different clinical data tables. For instance,
12CDROUTPATIENT ORDER.py processed outpatient
medical orders, 3CDRPATIENT DETAIL.py extracted
basic patient information, and other scripts handled inpa-
tient records, lab reports, and imaging descriptions. These
scripts accurately extracted key fields from their respective
CSV source files. After parsing, each script converted the
structured CSV information into semi-structured TXT files.
All text fragments extracted from different sources were ul-
timately sorted and aggregated by patient ID and timestamp.
Through this process, we generated a longitudinal text file
named patient inpatient.txt for each patient in the
database. This file comprehensively documents all relevant
clinical events for the patient, transforming the scattered,
structured data into a patient-centric, sequential, unstruc-
tured text, which provides a high-quality input for the sum-
marization and knowledge graph construction in Stage 1.
B.2 Demographics and Clinical Characteristics.
We randomly sampled the complete records of 2,000 pa-
tients from the CancerEHR dataset for our analysis. Table 6
and 7 displays the demographic details and the distribution
of major cancer types within this dataset.
Table 6 shows the top 10 cancer type distribution in the
CancerEHRs dataset.
Table 7 summarizes the text length statistics for the Can-
cerEHRs dataset.
C Processed MIMIC-Cancer Dataset Details
and Demographics
To validate the generalizability of our model across differ-
ent languages and data sources, we constructed the Pro-
cessed MIMIC-Cancer Dataset. The processing pipeline for
this dataset is similar to that of CancerEHR, but its source is
the public MIMIC-IV database.
C.1 Data Processing Pipeline.
We first filtered the MIMIC-IV database to select all patients
who had an ICD-9 or ICD-10 cancer diagnosis code in their
diagnoses icd.csv file. Subsequently, we removed ad-
mission events from these patients’ records that were not di-
rectly related to cancer to ensure that each record focuses on
the cancer diagnosis and treatment process. Next, we applied
a processing pipeline similar to the one described in Section
B. The main difference was that we wrote parsing scripts
adapted to the MIMIC-IV data schema, extracted data from
files such as admissions.csv ,chartevents.csv ,
andlabevents.csv , and integrated this information into
patient-centric clinical narratives in English. During the KG
construction phase for this dataset, we primarily relied on in-
ternational guidelines such as NCCN and ESMO, as they are

Table 5: Evaluation Rubric for Factual Accuracy, Completeness & Thoroughness, Clinical Soundness, and Actionability &
Relevance.
Evaluation Dimension & Scoring Criteria
Factual Accuracy
- 5: All key information is 100% accurate.
- 3: Contains errors in non-critical information or factual deviations that do not affect final treatment
decisions or patient safety .
- 1: Contains any major factual error that could affect treatment decisions or patient safety .
Completeness & Thoroughness
- 5: Perfectly covers all critical aspects of the patient’s situation, identifies all key data elements, and insight-
fully adds important potential risks.
- 3: Covers most core aspects and data elements but omits some minor details or has individual improper
handling of data.
- 1: Seriously lacks core content, or seriously omits or misunderstands key core data elements.
Clinical Soundness
- 5: All conclusions and recommendations are robust, safe, and reflect the clinical prudence of a senior
expert. They are implicitly or explicitly based on recognized clinical guidelines.
- 3: Core recommendations are reasonable, but may include some unimportant or slightly unusual minor
suggestions, or some recommendations lack a clear evidence-based foundation.
- 1: Contains any recommendations that could jeopardize patient safety, clearly violate clinical common
sense, or are based on misquotes.
Actionability & Relevance
- 5: Provides highly insightful, quantifiable, and personalized action plans that focus on solving the most
urgent current problems.
- 3: Offers some actionable advice, but some key parts are too general, or recommendations are mixed with
retrospective analysis not directly relevant to the immediate next steps.
- 1: Provides a list of invalid information with no guiding value, or the recommendations are entirely discon-
nected from the current core clinical problem.

Table 6: Top 10 Cancer Type Distribution in the Can-
cerEHRs Dataset.
Diagnosis Category Count
Breast Cancer 491
Malignant Tumor 271
Rectal Cancer 144
Lung Cancer 108
Gastric Cancer 93
Colon Cancer 85
Ovarian Cancer 57
Cervical Cancer 55
Lymphoma 33
Postoperative 32
Table 7: Text Length Statistics for Records in the Can-
cerEHRs Dataset.
Statistic Word Count Character Count Digit Count
Mean 5,698.96 10,103.73 799.37
Median 5,158.50 8,797.50 492.50
Maximum 21,915.00 42,154.00 6,803.00
more relevant to clinical practice in the United States. The
entity recognition and linking processes were also adapted
for English medical terminology. Ultimately, we obtained
formatted treatment trajectory texts for 2,000 patients, which
could be directly used for comparative experiments with the
CancerEHR dataset.
C.2 Demographics and Clinical Characteristics.
Table 8 and 9 presents the demographic and clinical de-
tails for the 2,000 cancer patients selected for our Processed
MIMIC-Cancer Dataset.
Table 8: Top 10 Cancer Type Distribution in the Processed
MIMIC-Cancer Dataset.
Diagnosis Category Count
Diffuse large b 96
Multiple myeloma 86
Acute myeloid leukemia 66
Acute myeloblastic leukemia 53
B 52
Non 44
Malignant neoplasm of bronchus 43
Liver cell carcinoma 37
Malignant neoplasm of prostate 36
Table 8 shows the top 10 cancer type distribution in the
processed MIMIC-Cancer Dataset.
Table 9 summarizes the text length statistics for the pro-
cessed MIMIC-Cancer Dataset.
As shown in Figure 4 and Figure 5, we present the distri-
bution of the number of hospitalizations and the distribution
of text length for patients in both the CancerEHR and pro-
cessed MIMIC-Cancer Datasets. The hospitalization distri-Table 9: Text Length Statistics for Records in the Processed
MIMIC-IV Dataset.
Statistic Word Count Character Count Digit Count
Mean 3,377.30 22,895.11 1,274.10
Median 1,646.00 11,538.50 526.50
Maximum 31,713.00 204,741.00 16,250.00
bution illustrates the number of patients with different hos-
pitalization frequencies, reflecting the real-world visit pat-
terns of cancer patients. The text length distribution shows
the range of clinical note lengths per patient, highlighting
the diversity in text scale within each dataset.
D Additional Experimental Details
This section provides additional experimental details not
fully discussed in the main text, including the process for
calculating the agreement between the LLM and physicians,
and some statistical results from the evaluation.
D.1 Spearman coefficient
To assess the agreement between a Large Language Model
(LLM) and multiple human experts on the two clinical scor-
ing tasks, we employed Spearman’s rank correlation coef-
ficient ( ρ) for statistical analysis. For each evaluation met-
ric—”Evaluation of Clinical Summary” and ”Evaluation of
Clinical Recommendations”—we collected the scores from
the LLM (denoted as LLM Ave) and the independent scores
from three experts (Exp 1, Exp 2, Exp 3). We also calculated
the mean of the three expert scores (Mean). For each pair of
raters, the Spearman’s correlation coefficient was calculated
using the following steps:
• The scores from each of the two groups are converted to
ranks (i.e., the original scores are replaced by their rank
order within their respective groups).
• The Pearson correlation coefficient is then calculated be-
tween these two sets of ranks to yield the Spearman’s
correlation coefficient, ρ.
The resulting correlation coefficient ranges from -1 to 1,
where 1 indicates a perfect positive correlation, -1 indicates
a perfect negative correlation, and 0 indicates no correlation.
The strength of the correlation is interpreted as follows: a
coefficient greater than 0.8 is considered very strong; 0.6 to
0.8 is strong; 0.4 to 0.6 is moderate; 0.2 to 0.4 is weak; and
0 to 0.2 is very weak or negligible.The detailed formula for
calculating the Spearman’s coefficient is provided below:
ρ= 1−6Pn
i=1(R(Xi)−R(Yi))2
n(n2−1)(7)
where nis the number of samples, XiandYiare the scores
of the two raters for the i-th sample,and R(Xi)andR(Yi)
are their respective ranks.
D.2 Evaluation of statistical analysis
Table 10 presents a detailed statistical comparison of perfor-
mance for different types of large language models across

Figure 4: Distribution of Hospitalizations in the CancerEHR Dataset(a) and MIMIC-Cancer Dataset(b).
Figure 5: Distribution of Text Length in the CancerEHR Dataset(a) and MIMIC-Cancer Dataset(b).
two key clinical tasks. For this analysis, the evaluation re-
sults are structured into two main sections. The first sec-
tion details the performance statistics for a locally deployed
open-source model, Qwen-3-8B. The second section, in con-
trast, presents the corresponding scores from several repre-
sentative closed-source models accessed via API. This com-
parative analysis was conducted on two distinct medical
datasets, CancerEHR and MIMIC-Cancer, and was further
broken down by two independent evaluation tasks: Clinical
Summary and Clinical Recommendation. To provide a com-
prehensive overview of the performance distribution and sta-
bility, five key descriptive statistics were calculated for each
scenario: the mean, standard deviation (Std), first quartile
(Q1), median, and third quartile (Q3). This granular pre-
sentation aims to thoroughly reveal the performance charac-
teristics of each model, including their advantages, central
tendencies, and the dispersion of their scores under specific
clinical tasks and data environments.
E Detailed Experimental Results on NLP
Metrics
This section presents the detailed scores of the locally
deployed open-source models on the following Natural
Language Processing (NLP) metrics: BLEU, ROUGE-1,
ROUGE-2, and ROUGE-L. The results are summarized in
the table 11 and 12 below.
Specifically, Table 11 compares the performance of vari-
ous models when utilizing a standard RAG setup versus our
proposed CliCARE framework. Furthermore, Table 12 de-
tails the results of an ablation study on the key components
of the CliCARE framework, quantifying the contribution ofeach module to the overall performance.
It is important to note that for an open-ended medical gen-
eration task like ours, standard NLP metrics such as BLEU
and ROUGE have inherent limitations. These metrics pri-
marily measure the lexical similarity between the generated
text and the reference answers. Consequently, while they can
provide a general indication of fluency and content overlap,
they cannot fully capture the clinical authenticity or factual
accuracy of the generated responses.
F Effects of Context Length on Performance
This section elaborates on the sensitivity analysis conducted
to evaluate the effect of different context lengths on model
performance. When processing long-form medical docu-
ments such as EHRs, the model’s context processing capa-
bility is crucial, as it directly determines the accuracy and
comprehensiveness of its information integration, clinical
summarization, and decision support. To quantify this ef-
fect, we selected the Qwen3-8B model and conducted exper-
iments on two medical datasets: CancerEHR and MIMIC-
Cancer. We systematically configured and tested three dif-
ferent context lengths: 2k, 8k, and 20k tokens. By compar-
ing the performance under these configurations, we aim to
reveal the relationship between model performance and con-
text length.
As shown in Figure 7, these two matrices illustrate the re-
sults of the performance ablation study under different con-
text length settings. We compare the performance of three
context lengths—2k, 8k, and 20k tokens—on two datasets:
CancerEHR (Figure a) and MIMIC-Cancer (Figure b). The
values in the matrix represent the win rate of the row model
over the column model in pairwise comparisons. Across

Figure 6: High Agreement Between LLM Judge and Physician Raters Validates Evaluation Methodology. The figure compares
ratings from our LLM judge and three physicians on two datasets and two clinical tasks. Violin plots show similar score
distributions, while heatmaps confirm high inter-rater agreement. The strong Spearman’s rank correlation demonstrates that our
automated evaluation is a reliable proxy for human expert judgment.
Table 10: Comparative Statistical Analysis of Performance for Different LLM Types on Clinical Tasks across the CancerEHR
and MIMIC-Cancer Datasets.
MethodCancerEHR MIMIC-Cancer
TCS TCR TCS TCR
mean std Q1 median Q3 mean std Q1 median Q3 mean std Q1 median Q3 mean std Q1 median Q3
Qwen-3-8B
StandardRAG 1.171 0.537 1 1 1 1.163 0.576 1 1 1 1.919 1.140 1 1 3 1.862 1.021 1 2 3
BriefContext 1.933 0.974 1 2 3 2.020 1.005 1 2 3 2.330 1.160 1 2 3 2.189 1.048 1 2 3
MedRAG 2.094 1.306 1 2 3 1.948 1.182 1 1 3 2.030 1.269 1 1 3 1.964 1.137 1 1 3
KG2RAG 2.011 1.159 1 2 3 2.021 1.191 1 1.5 3 2.193 1.408 1 2 3 2.226 1.396 1 2 3
GNN-RAG 1.244 0.575 1 1 1 1.284 0.615 1 1 1 1.447 0.835 1 1 2 1.383 0.752 1 1 2
CliPAGE 2.446 1.361 1 2 4 2.385 1.336 1 2 4 2.046 1.217 1 2 3 2.015 1.167 1 2 3
Gemini 2.5 Pro
StandardRAG 1.804 1.321 1 1 2 1.781 1.300 1 1 2 3.863 1.244 3 4 5 3.832 1.244 3 4 5
BriefContext 3.933 1.003 4 4 5 3.924 0.920 4 4 4 4.442 0.996 4 5 5 4.376 1.001 4 5 5
MedRAG 4.526 0.977 4 5 5 4.543 1.037 5 5 5 4.477 1.252 5 5 5 4.492 1.176 5 5 5
KG2RAG 3.855 1.299 3 4 5 3.858 1.348 3 4 5 4.381 1.295 5 5 5 4.396 1.304 5 5 5
GNN-RAG 2.026 1.045 1 2 3 2.010 0.995 1 2 3 2.792 1.468 1 3 4 2.716 1.478 1 3 4
CliPAGE 4.938 0.389 5 5 5 4.934 0.392 5 5 5 4.198 1.327 4 5 5 4.213 1.272 4 5 5
both datasets, we found that increasing the model’s con-
text length generally improves its performance. On the Can-
cerEHR dataset (Figure a), the 20k model has a win rate of
0.50 against the 2k model, and the 8k model has a win rate of
0.46 against the 2k model. The competition between the 20k
and 8k models is closer, with their respective win rates being
0.36 and 0.30, indicating a slight advantage for the 20k ver-sion. Similarly, on the MIMIC-Cancer dataset, the advantage
of the 20k model over the 8k model is also pronounced, with
a win rate of 0.44, while the win rate of the 8k model against
the 20k model is only 0.34. These results demonstrate the
importance of a long context for improving model perfor-
mance when processing complex Electronic Health Record
(EHR) data, suggesting that the model can more effectively

Table 11: Model performance with standard RAG versus the CliCARE framework. The table shows scores for BLEU (B),
ROUGE-1 (R1), ROUGE-2 (R2), and ROUGE-L (RL). Applying CliCARE provides a substantial performance uplift, especially
on the complex CancerEHR dataset.
MethodStandard RAG CliCARE
CancerEHR MIMIC-Cancer CancerEHR MIMIC-Cancer
B R1 R2 RL B R1 R2 RL B R1 R2 RL B R1 R2 RL
Mistral-v0.1-7B 42.46 54.52 28.52 44.69 68.71 54.80 31.90 40.18 38.42 50.34 23.25 34.55 68.04 53.20 30.53 38.99
Mistral-Instruct-v0.1-7B 41.83 53.45 27.86 44.58 67.80 54.11 31.25 39.16 37.90 49.79 22.86 34.33 67.54 52.34 29.95 38.67
Biomistral-7B 41.20 53.14 27.58 43.87 66.81 53.42 31.05 39.28 29.98 41.07 18.42 28.07 66.50 51.62 29.32 38.28
Qwen-3-8B 22.81 39.59 18.24 26.49 67.73 53.07 29.94 38.38 39.88 53.16 24.55 35.23 66.90 52.09 29.02 37.58
Table 12: Ablation study on CliCARE framework components using NLP metrics.
MethodCancerCEHR MIMIC-Cancer
B R1 R2 RL B R1 R2 RL
CliCARE (Qwen-3-8B) 39.88 53.16 24.55 35.23 66.90 52.09 29.02 37.58
w/o Alignment Expansion 42.34 58.07 31.08 42.40 62.33 50.19 28.04 44.57
w/o LLM-based Reranking 43.58 58.71 31.45 42.65 61.59 49.55 27.54 44.06
w/o TKG-based Compression 22.81 39.59 18.24 26.49 67.73 53.07 29.94 38.38
Figure 7: Ablation study results for the model at different
context lengths (2k, 8k, 20k). (a) Pairwise comparison ma-
trix on the CancerEHR dataset. (b) Pairwise comparison ma-
trix on the MIMIC-Cancer dataset.
utilize the extended contextual information to make higher-
quality judgments, leading to more significant performance
gains.
G Implementation Details of the Alignment
Method
This section provides the detailed pseudocode for our pro-
posed Guideline and Patient Data Alignment method. As
outlined in 1, the algorithm consists of five main stages.
First, it employs a Large Language Model (LLM) to con-
struct a knowledge graph (KG) from clinical guidelines and
a temporal knowledge graph (TKG) from patient records
(Step 1). Following this, an initial set of alignment can-
didates is generated using BERT-based semantic similarity
(Step 2) and subsequently reranked by an LLM to create a
high-quality seed set ( Aseed) based on clinical plausibility(Step 3). The core of the method is the iterative BootEA pro-
cess, which expands these seed alignments by calculating a
weighted score derived from both semantic and neighbor-
hood similarities for all unaligned pairs (Step 4). The pro-
cess concludes by merging the KG and TKG based on the
final alignment set to produce a single, unified graph (Step
5).
H PROMPT OF Answer Label
GENERATION
As shown in Figure 8, this is a sample prompt for the Cli-
PAGE method. The prompt is structured into five primary
components: Longitudinal Cancer EHRs, Current record,
Retrieval-Augmented Generation (RAG) content, a section
for the Clinical Summary, and a section for the Clinical
Recommendation. These elements combine to form a sin-
gle, comprehensive prompt that guides the model’s response
generation process.
References
Bai, Y .; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du,
Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y .; Tang, J.; and Li,
J. 2024. LongBench: A Bilingual, Multitask Benchmark for
Long Context Understanding. arXiv:2308.14508.
Dong, Y .; Rastogi, E.; Naik, G.; Rajagopal, S. P.; Goyal, S.;
Zhao, F.; Chintagunta, B.; and Ward, J. 2024. A Contin-
ued Pretrained LLM Approach for Automatic Medical Note
Generation. In Proceedings of the 2024 Conference of the
North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies: Short
Papers , 125–132. Mexico City, Mexico: Association for
Computational Linguistics.
e Shi, Y .; xin Wu, Z.; wei Li, J.; jun Zhao, L.; lin Liu, S.;
nan Wang, C.; qi Li, J.; yang Kang, Y .; long Sun, C.; wei

Algorithm 1: Alignment Algorithm
Require: Clinical Guidelines Dg; Patient Clinical Data Dp; Pre-trained BERT model fBERT; Large Language Model fLLM;
Bootstrapping iterations I; Confidence threshold θ; Weighting factor αfor scoring.
Ensure: An aligned Knowledge Graph Galigned .
Step 1: Knowledge Graph Extraction
1:Utilize fLLMto extract a static KG fromDgand a Temporal TKG fromDp.
Step 2: Initial Semantic Alignment with BERT
2:Initialize candidate set Ainitial← ∅.
3:foreach entity ep∈TKG do
4: foreach entity eg∈KG do
5: sim←cosine similarity (fBERT(desc(ep)), fBERT(desc(eg))).
6: Add(ep, eg, sim )toAinitial.
7: end for
8:end for
Step 3: LLM Reranking
9:Areranked ←fLLM(”Rerank candidates by clinical plausibility” , Ainitial).
10:Aseed←FilterHighConfidencePairs (Areranked ).
Step 4: Iterative Bootstrapping Alignment
11:Afinal←Aseed.
12:LetUbe the set of unaligned entities.
13:fori= 1→Ido
Initialize a set for newly found alignments in this iteration:
14: Anew← ∅.
15: foreach unaligned pair (ep, eg)∈U×Udo
Calculate semantic similarity:
16: Ssem←cosine similarity (fBERT(ep), fBERT(eg)).
Calculate neighborhood similarity based on already aligned neighbors:
17: Shood←CalculateNeighborhoodScore ((ep, eg), Afinal).
Compute the final weighted score:
18: Sweighted ←α·Ssem+ (1−α)·Shood.
19: ifSweighted > θthen
20: Anew←Anew∪ {(ep, eg)}.
21: end if
22: end for
23: ifAnewis empty then
Exit loop if no new alignments are found.
24: break
25: end if
Add newly found alignments to the final set:
26: Afinal←Afinal∪Anew.
Update the set of unaligned entities:
27: U←U\entities in Anew.
28:end for
Step 5: Construct Aligned Graph
29:Galigned←MergeGraphs (KG, TKG, A final).
30:return Galigned .

Figure 8: A Sample Prompt for the CliPAGE Method Highlighting Its Key Components.

Lin, Q.; and mei Zhang, D. 2024. KG2RAG: A Knowledge
Graph-to-Text Framework for Retrieval-Augmented Gener-
ation. arXiv preprint arXiv:2405.00688 .
Fan, Y .; Sun, H.; Xue, K.; Zhang, X.; Zhang, S.; and Ruan,
T. 2024. Medodyssey: A medical domain benchmark for
long context evaluation up to 200k tokens. arXiv preprint
arXiv:2406.15019 .
Feng, Y .; Zhang, W.; Wang, X.; and Chen, Q. 2024.
GNN-RAG: A Graph-based Retrieval Method for Retrieval-
Augmented Generation. arXiv preprint arXiv:2405.02152 .
Hager, P.; Jungmann, F.; Holland, R.; Bhagat, K.; Hubrecht,
I.; Knauer, M.; Vielhauer, J.; Makowski, M.; Braren, R.;
Kaissis, G.; and Rueckert, D. 2024. Evaluation and miti-
gation of the limitations of large language models in clinical
decision-making. Nature Medicine , 30(9): 2613–2622.
Hao, Y .; Qiu, Z.; Holmes, J.; L ¨ockenhoff, C. E.; Liu, W.;
Ghassemi, M.; and Kalantari, S. 2025. Large language
model integrations in cancer decision-making: a systematic
review and meta-analysis. npj Digital Medicine , 8(1): 450.
Huang, K.; Altosaar, J.; and Ranganath, R. 2019. Clinical-
bert: Modeling clinical notes and predicting hospital read-
mission. arXiv preprint arXiv:1904.05342 .
Jiang, A. Q.; Sablayrolles, A.; Mensch, A.; Bamford, C.;
Chaplot, D. S.; de las Casas, D.; Bressand, F.; Lengyel, G.;
Lample, G.; Saulnier, L.; et al. 2023. Mistral 7B. arXiv
preprint arXiv:2310.06825 .
Jiang, Z.; Han, X.; Liu, Z.; and Sun, M. 2024. Efficient
Knowledge Infusion via KG-LLM Alignment. In Find-
ings of the Association for Computational Linguistics: ACL
2024 .
Johnson, A. E. W.; Bulgarelli, L.; Shen, L.; Gayles, A.;
Shammout, A.; Horng, S.; Pollard, T. J.; Hao, S.; Moody,
B.; Gow, B.; wei H. Lehman, L.; Celi, L. A.; and Mark,
R. G. 2023. MIMIC-IV , a freely accessible electronic health
record dataset. Scientific Data , 10(1): 1.
Khan, A.; Wu, T.; and Chen, X. 2024. Data Management
Opportunities in Unifying Large Language Models+ Knowl-
edge Graphs. In Workshops at the 50th International Con-
ference on Very Large Data Bases . VLDB.org.
Khatib, H. S. A.; Mittal, S.; Rahimi, S.; Marhamati, N.; and
Bozorgzad, S. 2025. From Patient Consultations to Graphs:
Leveraging LLMs for Patient Journey Knowledge Graph
Construction. arXiv preprint arXiv:2503.16533 .
Labrak, Y .; Vidal, M.; Appert, L.; Ladhari, B.; Bonnet, P.;
Ghannay, S.; Gandon, F.; and Nader, F. 2024. BioMistral:
A Collection of Open-Source Large Language Models for
Medical Domains. arXiv preprint arXiv:2402.10374 .
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in Neural In-
formation Processing Systems , volume 33, 9459–9474.
Li, M.-H.; Huang, J. J.; Yeung, J.; Chen, J.; Liu, S.; Zhang,
Z.; Lee, E. K.; and Wang, S.-C. 2024. CancerLLM: A large
language model in cancer domain. arXiv:2406.10459.Liu, F.; Li, Z.; Zhou, H.; Yin, Q.; Yang, J.; Tang, X.; Luo,
C.; Zeng, M.; Jiang, H.; Gao, Y .; Nigam, P.; Nag, S.; Yin, B.;
Hua, Y .; Zhou, X.; Rohanian, O.; Thakur, A.; Clifton, L.; and
Clifton, D. A. 2024. Large Language Models Are Poor Clin-
ical Decision-Makers: A Comprehensive Benchmark. In Al-
Onaizan, Y .; Bansal, M.; and Chen, Y .-N., eds., Proceedings
of the 2024 Conference on Empirical Methods in Natural
Language Processing , 13696–13710. Miami, Florida, USA:
Association for Computational Linguistics.
Maushagen, J.; Hext, T.; Mero ˜no-Pe ˜nuela, A.; and Pelle-
grini, T. 2024. Populating CSV Files from Unstructured Text
with LLMs for KG Generation with RML. In SEMANTiCS
(Posters, Demos, Workshops & Tutorials) .
Nazary, F.; Deldjoo, Y .; Noia, T. D.; and di Sciascio, E. 2024.
XAI4LLM. Let Machine Learning Models and LLMs Col-
laborate for Enhanced In-Context Learning in Healthcare.
arXiv:2401.01353.
Rajashekar, N. C.; Shin, Y . E.; Pu, Y .; Chung, S.; You, K.;
Giuffre, M.; Chan, C. E.; Saarinen, T.; Hsiao, A.; Sekhon,
J.; Wong, A. H.; Evans, L. V .; Kizilcec, R. F.; Laine, L.; Mc-
call, T.; and Shung, D. 2024. Human-Algorithmic Interac-
tion Using a Large Language Model-Augmented Artificial
Intelligence Clinical Decision Support System. In Proceed-
ings of the 2024 CHI Conference on Human Factors in Com-
puting Systems , CHI ’24. New York, NY , USA: Association
for Computing Machinery. ISBN 9798400703300.
Singhal, K.; Azizi, S.; Tu, T.; Mahdavi, S. S.; Wei, J.; Chung,
H. W.; Scales, N.; Tanwani, A.; Cole-Lewis, H.; Pfohl, S.;
et al. 2023. Large language models encode clinical knowl-
edge. Nature , 620(7972): 172–180.
Sinsky, C.; Colligan, L.; Li, L.; Prgomet, M.; Reynolds, S.;
Goeders, L.; Westbrook, J.; Tutty, M.; and Blike, G. 2016.
Allocation of Physician Time in Ambulatory Practice: A
Time and Motion Study in 4 Specialties. Annals of Inter-
nal Medicine , 165.
Sun, Z.; Hu, W.; Zhang, Q.; and Qu, Y . 2018. Bootstrapping
Entity Alignment with Knowledge Graph Embedding. In
Proceedings of the Twenty-Seventh International Joint Con-
ference on Artificial Intelligence, IJCAI-18 , 4396–4402. In-
ternational Joint Conferences on Artificial Intelligence Or-
ganization.
Tadevosyan, M.; Nori, A.; Singh, A.; et al. 2025. Articulate
Medical Intelligence Explorer. Nature . In Press.
Tu, T.; Azizi, S.; Driess, D.; Schaekermann, M.; Amin, M.;
Chang, P.-C.; Carroll, A.; Aha, D.; Corrado, G. S.; Fleet,
D. J.; et al. 2024. Towards conversational diagnostic AI.
Nature Medicine , 1–13.
Wang, L. L.; Otmakhova, Y .; DeYoung, J.; Truong, T. H.;
Kuehl, B.; Bransom, E.; and Wallace, B. 2023. Automated
Metrics for Medical Multi-Document Summarization Dis-
agree with Human Evaluations. In Rogers, A.; Boyd-Graber,
J.; and Okazaki, N., eds., Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , 9871–9889. Toronto, Canada: As-
sociation for Computational Linguistics.
Wang, P.; Li, L.; Chen, L.; and Lin, B. Y . 2024. Large Lan-
guage Models are Inconsistent and Biased Evaluators. In

Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies , 461–472.
Warner, J. L.; Dymshyts, D.; Hripcsak, G.; and Miller, R. S.
2020. Data Fragmentation and the Case for a National Can-
cer Data Ecosystem: A Report From the American Society
of Clinical Oncology. JCO Clinical Cancer Informatics , 4:
1022–1031.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; and . . . . 2025. Qwen3
Technical Report. arXiv preprint arXiv:2505.09388 .
Yang, F.; Wang, X.; and Li, J. 2021. BERT model in Chinese
clinical natural language processing: exploration and re-
search. https://github.com/trueto/medbert. Accessed: 2024-
07-09.
Yang, Q.; Wang, R.; Chen, J.; Su, R.; and Tan, T.
2024. Fine-Tuning Medical Language Models for En-
hanced Long-Contextual Understanding and Domain Exper-
tise. arXiv:2405.02986.
Yu, H.; and McQuade, F. 2025. Rag-kg-il: A multi-agent
hybrid framework for reducing hallucinations and enhancing
llm reasoning through rag and incremental knowledge graph
learning integration. arXiv preprint arXiv:2503.13514 .
Zhang, A.; Li, X.; Sahoo, P.; and Yu, M. 2024. Leveraging
Long Context in Retrieval Augmented Language Models for
Medical Question Answering. arXiv:2402.13329.
Zhang, K.; Kang, Y .; Zhao, F.; and Liu, X. 2023. LLM-based
Medical Assistant Personalization with Short-and Long-
term Memory Coordination. arXiv:2309.11696.
Zhao, X.; Liu, S.; Yang, S. Y .; et al. 2025. Medrag: Enhanc-
ing retrieval-augmented generation with knowledge graph-
elicited reasoning for healthcare copilot. In Proceedings of
the ACM on Web Conference 2025 , 4442–4457.
Zheng, L.; Chiang, W.-L.; Sheng, Y .; Zhuang, S.; Wu,
Z.; Zhuang, Y .; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.;
Zhang, H.; Gonzalez, J. E.; and Stoica, I. 2023. Judg-
ing LLM-as-a-judge with MT-Bench and Chatbot Arena.
arXiv:2306.05685.
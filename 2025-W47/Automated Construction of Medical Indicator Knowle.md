# Automated Construction of Medical Indicator Knowledge Graphs Using Retrieval Augmented Large Language Models

**Authors**: Zhengda Wang, Daqian Shi, Jingyi Zhao, Xiaolei Diao, Xiongfeng Tang, Yanguo Qin

**Published**: 2025-11-17 16:00:42

**PDF URL**: [https://arxiv.org/pdf/2511.13526v1](https://arxiv.org/pdf/2511.13526v1)

## Abstract
Artificial intelligence (AI) is reshaping modern healthcare by advancing disease diagnosis, treatment decision-making, and biomedical research. Among AI technologies, large language models (LLMs) have become especially impactful, enabling deep knowledge extraction and semantic reasoning from complex medical texts. However, effective clinical decision support requires knowledge in structured, interoperable formats. Knowledge graphs serve this role by integrating heterogeneous medical information into semantically consistent networks. Yet, current clinical knowledge graphs still depend heavily on manual curation and rule-based extraction, which is limited by the complexity and contextual ambiguity of medical guidelines and literature. To overcome these challenges, we propose an automated framework that combines retrieval-augmented generation (RAG) with LLMs to construct medical indicator knowledge graphs. The framework incorporates guideline-driven data acquisition, ontology-based schema design, and expert-in-the-loop validation to ensure scalability, accuracy, and clinical reliability. The resulting knowledge graphs can be integrated into intelligent diagnosis and question-answering systems, accelerating the development of AI-driven healthcare solutions.

## Full Text


<!-- PDF content starts -->

Automated Construction of Medical Indicator
Knowledge Graphs Using Retrieval Augmented
Large Language Models
Zhengda Wang
The second hospital of Jilin University
Northeast Asia Active Aging Laboratory
Jilin, China
Email: wangzd9920@mails.jlu.edu.cnDaqian Shi
QMUL DERI
UCL Institute Of Health Informatics,
Lonodn, UK
Email: d.shi@qmul.ac.ukJingyi Zhao
The second hospital of Jilin University
Northeast Asia Active Aging Laboratory
Jilin, China
Email: zhaojingyi249@163.com
Xiaolei Diao
QMUL School of Electronic Engineering
and Computer Science
London, UK
Email: xiaolei.diao1@gmail.comXiongfeng Tang
The second hospital of Jilin University
College of Artificial Intelligence,
Jilin University, Jilin, China
Email: tangxf921@jlu.edu.cnYanguo Qin
The second hospital of Jilin University
Northeast Asia Active Aging Laboratory
Jilin, China
Email: qinyg@jlu.edu.cn
Abstract—Artificial intelligence (AI) is reshaping modern
healthcare by advancing disease diagnosis, treatment decision-
making, and biomedical research. Among AI technologies, large
language models (LLMs) have become especially impactful,
enabling deep knowledge extraction and semantic reasoning
from complex medical texts. However, effective clinical decision
support requires knowledge in structured, interoperable formats.
Knowledge graphs serve this role by integrating heterogeneous
medical information into semantically consistent networks. Yet,
current clinical knowledge graphs still depend heavily on manual
curation and rule-based extraction, which is limited by the com-
plexity and contextual ambiguity of medical guidelines and liter-
ature. To overcome these challenges, we propose an automated
framework that combines retrieval-augmented generation (RAG)
with LLMs to construct medical indicator knowledge graphs.
The framework incorporates guideline-driven data acquisition,
ontology-based schema design, and expert-in-the-loop validation
to ensure scalability, accuracy, and clinical reliability. The result-
ing knowledge graphs can be integrated into intelligent diagnosis
and question-answering systems, accelerating the development of
AI-driven healthcare solutions.
Index Terms—knowledge graphs, large language models,
retrieval-augmented generation, clinical guidelines
I. INTRODUCTION
Artificial intelligence (AI) has become a central driver
of innovation in modern medicine. Across domains such as
medical imaging analysis, disease risk prediction, precision
therapeutics, and personalized health management, AI has im-
proved diagnostic accuracy, optimized treatment strategies, and
enhanced healthcare delivery efficiency [1], [2]. Within these
technologies, large language models (LLMs) have introduced a
paradigm shift in processing and understanding medical texts.
Clinical guidelines, biomedical literature, and electronic health
records are inherently complex, heterogeneous, and largely
unstructured [3]. Traditional rule-based or dictionary-drivenmethods often fail to capture their semantic depth or adapt
to evolving knowledge. By contrast, LLMs enable context-
aware semantic reasoning, robust cross-domain generalization,
and generative capabilities, making them powerful tools for
medical knowledge extraction and utilization [4].
However, the effective application of such knowledge re-
quires structured, interoperable representations. Knowledge
graphs serve as semantic scaffolds that transform fragmented
information into interconnected networks of entities and re-
lationships, supporting applications such as clinical decision
support systems (CDSS), intelligent question answering, and
biomedical research [5]. Yet, clinical knowledge graphs still
rely heavily on manual curation and rule-based extraction
to ensure factual precision and semantic consistency. This
challenge stems from the nuanced and context-dependent
nature of medical guidelines and scientific literature, where
diagnostic criteria, therapeutic recommendations, and indicator
definitions are embedded in complex narratives. Consequently,
fully automated approaches remain uncommon, and existing
methods struggle to accommodate the dynamic and cross-
guideline characteristics of clinical indicators.
To address these limitations, we propose a framework that
integrates retrieval-augmented generation (RAG) with LLMs
for automated construction of medical indicator knowledge
graphs. Semantic retrieval grounds generative outputs in au-
thoritative sources, while LLM-based reasoning supports accu-
rate entity and relation extraction. Additionally, an expert-in-
the-loop (HITL) validation mechanism ensures clinical relia-
bility while allowing iterative refinement [6]. This combination
of automation, semantic rigor, and expert oversight provides
a scalable and adaptable solution for building high-quality
medical knowledge graphs, laying a robust foundation for
intelligent healthcare systems.arXiv:2511.13526v1  [cs.AI]  17 Nov 2025

II. RELATEDWORK
Recent advances in LLMs and biomedical knowledge
graphs have driven significant progress in automating medical
knowledge structuring and utilization. Gao et al. introduced
MDKG, a contextualized mental disorder knowledge graph
that addresses the lack of semantic context and indicator-level
granularity in traditional KGs [7]. Soman et al. developed a
KG-optimized prompt generation framework, where structured
biomedical knowledge informs LLM-based question answer-
ing [8]. In clinical applications, Yang et al. leveraged GPT-
4 to construct a sepsis knowledge graph, marking the first
integration of LLM reasoning into infectious disease modeling
[9]. To enhance factual precision, Wei et al. proposed MTL-
KGV , a multitask verification framework that improves the
reliability of automatically extracted biomedical triples [10].
Likewise, Xu et al. released PubMed Knowledge Graph 2.0,
linking scientific publications, patents, and clinical trials into
a scalable biomedical knowledge network [11], while Wang
et al. developed an end-to-end KG system supporting both
construction and semantic querying across domains [12].
Complementary research by Shi et al. has advanced se-
mantic representation and reasoning in KGs. Their multidi-
mensional KG framework structures entities via hierarchical
semantic relations to enable adaptive path reasoning and
personalized recommendation [13]. Building on this, Shi and
Wang et al. introduced MRP2Rec, which captures high-order
relational dependencies to improve interpretability and rec-
ommendation performance [14]. For multimodal consistency,
their CharFormer model applies attention-guided feature fu-
sion for image denoising while preserving structural semantics
[15]. Additionally, ZiNet, presented at ACL 2022, provides
the first diachronic KG covering 3,000 years of Chinese lin-
guistic evolution by integrating glyphs, radicals, and semantic
hierarchies [16]. Extending ZiNet, Diao et al. proposed a
glyph-driven restoration network for Oracle Bone Inscriptions
that leverages semantic priors to improve restoration under
complex degradation [17].
Despite these advancements, existing methods still face
limitations, including limited retrieval grounding, weak on-
tology integration, insufficient adaptability to multi-indicator
clinical guidelines, and inadequate human–AI quality control.
To address these issues, this study introduces a RAG-based
pipeline combining guideline-grounded retrieval, ontology-
driven schema design, and structured expert-in-the-loop vali-
dation, offering a scalable and semantically consistent solution
for medical indicator knowledge graph construction.
III. METHODOLOGY
A. Data Acquisition and Integration
As shown in Fig. 1, the proposed framework integrates RAG
with ontology-guided structuring to construct a medical knowl-
edge graph. We first collect clinical practice guidelines from
authoritative sources such as national health agencies, profes-
sional associations, and international healthcare organizations.
These guidelines, often presented in varied formats, undergoa standardized preprocessing pipeline that includes content
filtering, removal of non-informative elements, terminology
normalization using controlled vocabularies, and unification
of entity labels.
With expert collaboration, we define core clinical entity
categories—including diseases, symptoms, diagnostic exam-
inations, pharmacologic and surgical treatments, rehabilitation
indicators, and postoperative metrics. This domain-specific
schema supports subsequent ontology development and se-
mantic extraction. Table 1 summarizes representative clinical
indicator ranges across major systems, along with associated
disease categories, including both directly and indirectly re-
lated conditions.
B. Ontology Design
The ontology serves as the structural and semantic backbone
of the medical knowledge graph, ensuring logical consis-
tency, semantic alignment, and interpretability of the ex-
tracted knowledge. To develop a domain-specific and reusable
ontology applicable across clinical scenarios, we combine
LLM-based prompt engineering with iterative expert feedback.
This enables both top-down schema definition and bottom-up
refinement grounded in real guideline content. The resulting
ontology includes core entity types such as diseases, diag-
nostic procedures, treatment strategies, medications, clinical
indicators, and postoperative metrics. It also defines clinically
meaningful relation types, including links between indications
and treatment options, diagnostic procedures and threshold
values, and postoperative indicators and follow-up plans.
In addition to entities and relationships, the ontology in-
corporates essential attributes such as disease prevalence, test
frequencies, value ranges, risk classifications, and intervention
thresholds. Hierarchical structures are embedded to enable
nested classification of concepts, such as anatomical location
within disease categories or the dependence of follow up indi-
cators on treatment modalities. Logical constraints are applied
to enforce semantic coherence and ensure completeness, such
as requiring that every rehabilitation indicator be linked to a
relevant clinical procedure. The resulting ontology is aligned
with established biomedical standards, including SNOMED
CT and UMLS, facilitating semantic interoperability and
seamless integration with existing healthcare data systems and
decision support platforms.
C. Information Extraction
To extract structured knowledge from clinical guidelines,
we adopt a two-stage hybrid pipeline that integrates RAG with
LLM inference. In the first stage, a semantic retrieval module
performs document-level search using dense vector repre-
sentations derived from pretrained biomedical embeddings.
This enables efficient identification of contextually relevant
guideline segments for specific extraction intents, offering
greater robustness and relevance than rule-based or keyword-
based matching.
In the second stage, the retrieved text is processed by an
LLM to perform entity recognition, relation extraction, and

TABLE I
REPRESENTATIVECLINICALINDICATORS ANDTHEIRDISEASEASSOCIATIONS
System Guideline Indicator Reference Range Direct Disease Indirect Diseases
EndocrineAmerican Thyroid
AssociationThyroid Stimulating
Hormone2–10mU/L Thyroid diseases Secondary thyroid diseases
American College of
PhysiciansTestosterone Male:300–1000ng/L
Female:200–800ng/LPolycystic ovary syndrome Testicular dysgenesis
Chinese Society of
EndocrinologyGrowth Hormone Children:<20µg/L
Male:<2µg/L
Female:<10µg/LGigantism, acromegaly Pituitary dwarfism
Chinese Society of
EndocrinologyHuman chorionic
gonadotropinMale or non-pregnant
female:<5IU/L
Postmenopausal women:
<10IU/LHydatidiform mole Elevated hCG in early
pregnancy
Chinese Society of
CardiologyAntidiuretic hormone1.4–5.6pmol/L Nephrogenic diabetes
insipidusCentral diabetes insipidus
CirculatoryWorld Health
OrganizationBlood pressure<120/80mmHg Hypertension, hypotension Cardiovascular diseases
American Heart
AssociationCholesterol<200mg/dL Atherosclerosis Metabolic syndrome
Chinese College of
Cardiovascular
PhysiciansCreatine kinase Male:50–310U/L
Female:40–200U/LAtherosclerosis Myocarditis, rhabdomyolysis
European Society of
CardiologyHigh-density lipoprotein
(HDL)Male:>40mg/dL
Female:>50mg/dLCoronary heart disease Obesity, insulin resistance
European Society of
CardiologyLow-density lipoprotein
(LDL)<100mg/dL Coronary heart disease Diabetic vascular
complications
UrinaryAmerican College of
RheumatologyUric acid Male:3.0–7.0mg/dL
Female:2.5–6.5mg/dLGout Chronic kidney disease
American Society of
NephrologyUrinary red blood cells<3per HPF Urolithiasis, glomerular
diseaseLupus nephritis, diabetic
nephropathy
American Society of
NephrologyUrinary white blood
cells<5per HPF Urinary tract infection Chronic renal insufficiency
Kidney Disease:
Improving Global
OutcomesUrinary protein 24 h:<150mg Glomerular disease Hypertensive nephropathy
Kidney Disease:
Improving Global
OutcomesGlomerular filtration
rate90–120m2/1.73 Renal insufficiency, chronic
kidney diseaseCardiovascular diseases
DigestiveWorld Gastroenterology
OrganisationFecal occult blood test Negative Gastrointestinal bleeding Colorectal cancer
Chinese Society of
HepatologyTransaminase0–40U/L Hepatocellular injury Alcoholic liver disease
International
Association of
PancreatologyLipase13–60U/L Pancreatitis Renal insufficiency
American Cancer
SocietyCA19-9<37U/mL Pancreatic cancer Hepatobiliary diseases
American Cancer
SocietyCEA<5ng/mL Colorectal tumor Hepatic metastasis
attribute identification. The extracted results are organized
into structured representations, such as subject–relation–object
triples and attribute–value pairs, and aligned with the prede-
fined ontology to ensure semantic and structural consistency.
By combining targeted semantic retrieval with the flexible
reasoning capabilities of LLMs, the framework achieves ac-
curate, scalable, and semantically grounded knowledge graph
construction from unstructured clinical guideline text.
D. Knowledge Fusion and Graph Generation
After information extraction, the structured knowledge el-
ements are aligned with the predefined ontology to form
a coherent and semantically consistent medical knowledge
graph. This fusion process includes entity normalization to
resolve synonyms and lexical variations, relation disambigua-
tion to match extracted associations with ontology-definedsemantics, and attribute integration to standardize numerical
and categorical values.
To handle redundancy and conflict, duplicate or inconsis-
tent entries are reconciled through rule-based prioritization
and expert-guided resolution. These steps ensure semantic
clarity, structural integrity, and clinical relevance in the final
graph. Additionally, the framework supports interoperability
with established biomedical ontologies such as UMLS and
SNOMED CT, enhancing extensibility, reusability, and com-
patibility across diverse healthcare data environments and
downstream applications.
When inconsistencies, ambiguities, or missing information
are identified during expert review, the feedback is reinte-
grated to refine prompt templates, adjust extraction rules,
and improve LLM performance in subsequent cycles. This
iterative human–AI collaboration forms a continuous quality

DData Acquisition
（Clinical guidelines, preprocessing）
DOntology Design
（Entity types, relations, hierarchy）
DInformation Extraction
（RAG + LLM, entity & relation）
DKnowledge Fusion & Graph Generation
（Normalization, alignment, graph build）
Human-in-the-loop
（Expert validation & feedback）
Applications
（CDSS, QA, Biomedical research）
Fig. 1. Workflow of the proposed knowledge graph construction framework.
control loop that safeguards reliability, semantic accuracy, and
contextual relevance of the knowledge graph, while enabling
ongoing optimization of the extraction pipeline and underlying
knowledge representations.
E. Applications
The constructed medical knowledge graph supports a wide
range of downstream applications in both clinical and research
settings. In intelligent question-answering systems based on
GraphRAG, it enables multi-hop reasoning, causal inference,
and precise alignment with medical terminology, thereby
improving both accuracy and interpretability. Within CDSS,
it supports standardized diagnostic and treatment pathways,
dynamic retrieval of guideline content, and personalized rec-
ommendations tailored to patient-specific contexts.
For medical research, the structured knowledge base ac-
celerates hypothesis generation, enhances literature retrieval,
and enables comparative analyses of clinical indicators across
studies. By integrating RAG, ontology-driven schema mod-
eling, and expert-in-the-loop validation, the proposed frame-
work provides an efficient, scalable, and clinically meaningful
approach for constructing high-quality medical knowledge
graphs that align with the evolving needs of modern healthcare
systems.
IV. DISCUSSION
The proposed framework represents a substantial advance-
ment in medical knowledge graph construction by address-
ing the inherent limitations of traditional manual and rule-
based approaches [18]. Conventional methods are often labor-
intensive, narrowly scoped, and lack the scalability needed
to accommodate the rapid evolution of clinical knowledge
[19]. In contrast, our method integrates RAG with LLMs
to automate the extraction, normalization, and semantic inte-
gration of medical indicators from complex and unstructured
guideline texts, building on recent advances in domain-specific
generative model fine-tuning that have proven effective for
clinical summarization and knowledge extraction [20]. The
framework adopts a modular architecture consisting of data ac-
quisition, ontology design, information extraction, and knowl-
edge fusion, with each component independently optimizable
to enhance overall performance [21].
A key strength of the approach lies in the contextualized
extraction enabled by semantic retrieval and ontology-guidedstructuring. Vector-based retrieval grounds LLM inference
in clinically relevant source text, reducing hallucination and
improving factual reliability, while the ontology provides a
semantic scaffold that organizes extracted entities and relations
into clinically meaningful hierarchies. To provide an initial
quantitative check, we conducted an expert review of 240
extracted triples and confirmed 212 to be correct, resulting in
an overall precision of 88 percent, demonstrating stable extrac-
tion performance at this stage. At present, the framework has
standardized more than 120 clinical indicators derived from
38 authoritative guidelines spanning eight major physiolog-
ical systems: musculoskeletal, respiratory, urinary, digestive,
cardiovascular, endocrine, nervous, and immune–hematologic.
Each indicator is linked to its guideline source, contextual defi-
nition, and associated disease entities, forming a continuously
expanding and semantically aligned repository that supports
cross-system comparison and shared biomarker interpretation.
To ensure scientific rigor and clinical reliability, a human-
in-the-loop validation mechanism is embedded within the
pipeline. Clinical experts conduct structured review, refine
extraction prompts, and guide iterative improvement, forming
a continuous quality-control loop that enhances knowledge fi-
delity and system adaptability. The resulting knowledge graphs
are suitable for deployment in intelligent question-answering
systems, clinical decision-support tools, and biomedical re-
search platforms. While the framework demonstrates strong
interoperability with established biomedical ontologies such
as UMLS and SNOMED CT, future work will focus on
automated graph updating, domain-specific LLM calibration,
and maintaining a balanced integration of automation and
expert oversight to support a continuously evolving and clin-
ically reliable knowledge infrastructure. Sampled triples are
validated with a concise checklist and escalated when needed,
and expert feedback is used to iteratively refine prompts and
extraction rules.
V. CONCLUSION
In this study, we developed an automated framework in-
tegrating RAG with LLMs to construct medical indicator
knowledge graphs. The system has standardized over 120
indicators across eight physiological systems, demonstrating
strong scalability and semantic consistency. Future work will
focus on building a large-scale health visualization model
that combines clinical guidelines with real-world hospital data
to create personalized “health banks”, promoting precision
and intelligent healthcare through data-driven insights. In
addition, the framework will be expanded to support contin-
uous knowledge updating, cross-domain ontology alignment,
and multimodal data integration, paving the way toward an
adaptive and interpretable medical intelligence ecosystem.
VI. ACKNOWLEDGMENT
This work is supported by the Scientific Research
Project of the Department of Education of Jilin Province
(JJKH20250176KJ), and the Bethune Project of Jilin Univer-
sity (2025B37).

REFERENCES
[1] P. Chandak, K. Huang, and M. Zitnik, “Building a knowledge graph to
enable precision medicine,”Scientific Data, vol. 10, no. 1, p. 67, 2023.
[2] H. S. Al Khatib, S. Neupane, H. Kumar Manchukonda, N. A. Golilarz,
S. Mittal, A. Amirlatifi, and S. Rahimi, “Patient-centric knowledge
graphs: a survey of current methods, challenges, and applications,”
Frontiers in Artificial Intelligence, vol. 7, p. 1388479, 2024.
[3] X. Liang, Z. Wang, M. Li, and Z. Yan, “A survey of llm-augmented
knowledge graph construction and application in complex product
design,”Procedia CIRP, vol. 128, pp. 870–875, 2024.
[4] Y . Gao, R. Li, E. Croxford, J. Caskey, B. W. Patterson, M. Churpek,
T. Miller, D. Dligach, and M. Afshar, “Leveraging medical knowledge
graphs into large language models for diagnosis prediction: Design and
application study,”Jmir Ai, vol. 4, p. e58670, 2025.
[5] L. Murali, G. Gopakumar, D. M. Viswanathan, and P. Nedungadi,
“Towards electronic health record-based medical knowledge graph con-
struction, completion, and applications: A literature study,”Journal of
biomedical informatics, vol. 143, p. 104403, 2023.
[6] J. Wu, J. Zhu, Y . Qi, J. Chen, M. Xu, F. Menolascina, and
V . Grau, “Medical graph rag: Towards safe medical large lan-
guage model via graph retrieval-augmented generation,”arXiv preprint
arXiv:2408.04187, 2024.
[7] S. Gao, K. Yu, Y . Yang, S. Yu, C. Shi, X. Wang, N. Tang, and
H. Zhu, “Large language model powered knowledge graph construction
for mental health exploration,”Nature Communications, vol. 16, no. 1,
p. 7526, 2025.
[8] K. Soman, P. W. Rose, J. H. Morris, R. E. Akbas, B. Smith, B. Peetoom,
C. Villouta-Reyes, G. Cerono, Y . Shi, A. Rizk-Jacksonet al., “Biomed-
ical knowledge graph-optimized prompt generation for large language
models,”Bioinformatics, vol. 40, no. 9, p. btae560, 2024.
[9] H. Yang, J. Li, C. Zhang, A. P. Sierra, and B. Shen, “Large language
model–driven knowledge graph construction in sepsis care using multi-
center clinical databases: Development and usability study,”Journal of
Medical Internet Research, vol. 27, p. e65537, 2025.
[10] C.-P. Wei, P.-Y . Tsai, and J.-J. Li, “Biomedical knowledge graph
verification with multitask learning architectures,”Journal of Biomedical
Informatics, p. 104894, 2025.
[11] J. Xu, C. Yu, J. Xu, V . I. Torvik, J. Kang, M. Sung, M. Song, Y . Bu,
and Y . Ding, “Pubmed knowledge graph 2.0: Connecting papers, patents,
and clinical trials in biomedical science,”Scientific Data, vol. 12, no. 1,
p. 1018, 2025.
[12] L. Wang, H. Hao, X. Yan, T. H. Zhou, and K. H. Ryu, “From biomedical
knowledge graph construction to semantic querying: a comprehensive
approach,”Scientific Reports, vol. 15, no. 1, p. 8523, 2025.
[13] D. Shi, T. Wang, H. Xing, and H. Xu, “A learning path recommendation
model based on a multidimensional knowledge graph framework for e-
learning,”Knowledge-Based Systems, vol. 195, p. 105618, 2020.
[14] T. Wang, D. Shi, Z. Wang, S. Xu, and H. Xu, “Mrp2rec: Exploring
multiple-step relation path semantics for knowledge graph-based rec-
ommendations,”IEEE Access, vol. 8, pp. 134 817–134 825, 2020.
[15] D. Shi, X. Diao, L. Shi, H. Tang, Y . Chi, C. Li, and H. Xu, “Charformer:
A glyph fusion based attentive framework for high-precision character
image denoising,” inProceedings of the 30th ACM international con-
ference on multimedia, 2022, pp. 1147–1155.
[16] Y . Chi, F. Giunchiglia, D. Shi, X. Diao, C. Li, and H. Xu, “Zinet:
Linking chinese characters spanning three thousand years,” inFindings
of the association for computational linguistics: ACL 2022, 2022, pp.
3061–3070.
[17] X. Diao, D. Shi, W. Cao, T. Wang, R. Qi, C. Li, and H. Xu, “Oracle
bone inscription image restoration via glyph extraction,”npj Heritage
Science, vol. 13, no. 1, p. 321, 2025.
[18] H. Sch ¨afer, A. Idrissi-Yaghir, K. Arzideh, H. Damm, T. M. Pakull,
C. S. Schmidt, M. Bahn, G. Lodde, E. Livingstone, D. Schadendorf
et al., “Biokgrapher: Initial evaluation of automated knowledge graph
construction from biomedical literature,”Computational and Structural
Biotechnology Journal, vol. 24, pp. 639–660, 2024.
[19] R. Yang, Y . Ning, E. Keppo, M. Liu, C. Hong, D. S. Bitterman, J. C. L.
Ong, D. S. W. Ting, and N. Liu, “Retrieval-augmented generation for
generative artificial intelligence in health care,”npj Health Systems,
vol. 2, no. 1, p. 2, 2025.
[20] J. Wu, D. Shi, A. Hasan, and H. Wu, “Knowlab at radsum23: comparing
pre-trained language models in radiology report summarization,” inProceedings of the Annual Meeting of the Association for Computational
Linguistics. ACL, 2023, pp. 535–540.
[21] D. Shi and F. Giunchiglia, “Recognizing entity types via properties,” in
Formal Ontology in Information Systems. IOS Press, 2023, pp. 195–
209.
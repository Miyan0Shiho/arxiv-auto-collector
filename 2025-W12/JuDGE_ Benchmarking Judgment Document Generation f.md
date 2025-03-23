# JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System

**Authors**: Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu, Yiqun Liu

**Published**: 2025-03-18 13:48:18

**PDF URL**: [http://arxiv.org/pdf/2503.14258v2](http://arxiv.org/pdf/2503.14258v2)

## Abstract
This paper introduces JuDGE (Judgment Document Generation Evaluation), a
novel benchmark for evaluating the performance of judgment document generation
in the Chinese legal system. We define the task as generating a complete legal
judgment document from the given factual description of the case. To facilitate
this benchmark, we construct a comprehensive dataset consisting of factual
descriptions from real legal cases, paired with their corresponding full
judgment documents, which serve as the ground truth for evaluating the quality
of generated documents. This dataset is further augmented by two external legal
corpora that provide additional legal knowledge for the task: one comprising
statutes and regulations, and the other consisting of a large collection of
past judgment documents. In collaboration with legal professionals, we
establish a comprehensive automated evaluation framework to assess the quality
of generated judgment documents across various dimensions. We evaluate various
baseline approaches, including few-shot in-context learning, fine-tuning, and a
multi-source retrieval-augmented generation (RAG) approach, using both general
and legal-domain LLMs. The experimental results demonstrate that, while RAG
approaches can effectively improve performance in this task, there is still
substantial room for further improvement. All the codes and datasets are
available at: https://github.com/oneal2000/JuDGE.

## Full Text


<!-- PDF content starts -->

JuDGE: Benchmarking Judgment Document Generation for
Chinese Legal System
Weihang Su
swh22@mails.tsinghua.edu.cn
DCST, Tsinghua University
Beijing 100084, ChinaBaoqing Yue
DCST, Tsinghua University
Beijing 100084, ChinaQingyao Ai∗
aiqy@tsinghua.edu.cn
DCST, Tsinghua University
Beijing 100084, China
Yiran Hu
DCST, Tsinghua University
Beijing 100084, ChinaJiaqi Li
DCST, Tsinghua University
Beijing 100084, ChinaChangyue Wang
DCST, Tsinghua University
Beijing 100084, China
Kaiyuan Zhang
DCST, Tsinghua University
Beijing 100084, ChinaYueyue Wu *
wuyueyue1600@gmail.com
DCST, Tsinghua University
Beijing 100084, ChinaYiqun Liu
DCST, Tsinghua University
Beijing 100084, China
Abstract
This paper introduces JuDGE (Judgment Document Generation
Evaluation), a novel benchmark for evaluating the performance of
judgment document generation in the Chinese legal system. We
define the task as generating a complete legal judgment document
from the given factual description of the case. To facilitate this
benchmark, we construct a comprehensive dataset consisting of
factual descriptions from real legal cases, paired with their corre-
sponding full judgment documents, which serve as the ground truth
for evaluating the quality of generated documents. This dataset
is further augmented by two external legal corpora that provide
additional legal knowledge for the task: one comprising statutes
and regulations, and the other consisting of a large collection of
past judgment documents. In collaboration with legal professionals,
we establish a comprehensive automated evaluation framework to
assess the quality of generated judgment documents across vari-
ous dimensions. We evaluate various baseline approaches, includ-
ing few-shot in-context learning, fine-tuning, and a multi-source
retrieval-augmented generation (RAG) approach, using both gen-
eral and legal-domain LLMs. The experimental results demonstrate
that, while RAG approaches can effectively improve performance in
this task, there is still substantial room for further improvement1.
Keywords
Judgment Document Generation, Large Language Model, Domain-
Specific Evaluation, Retrieval Augmented Generation
1 Introduction
In recent years, the rapid development of large language mod-
els (LLMs) has enabled numerous applications in the legal do-
main [ 4,35,47,51], including contract analysis, case law research,
and legal document drafting [ 36,46,50,52]. Among these applica-
tions, the automated generation of judgment documents is particu-
larly promising. This task presents significant challenges, including
the accurate capture of factual details, the application of laws and
∗Corresponding author
1All the codes and datasets are available at: https://github.com/oneal2000/JuDGEstatutes, and the execution of legal reasoning. Moreover, it has the
potential to enhance judicial efficiency and reduce legal workloads.
Judgment documents are authoritative court records that encap-
sulate legal reasoning, procedural details, and final rulings [ 17,19].
As essential components of judicial proceedings, these documents
serve as authoritative records of legal decisions and reflect the
complexity of legal reasoning and statutory interpretation [ 53]. In
practice, drafting judgment documents is a highly labor-intensive
task that consumes substantial time and resources in courts. Specif-
ically, creating a judgment document requires the judge to gather
a large amount of legal information, including relevant statutes,
past case precedents, and fundamental legal knowledge. Once the
relevant legal information has been collected, it must be systemati-
cally organized and integrated with professional legal reasoning
to generate a well-structured and legally sound judgment docu-
ment. Given the inherently time-consuming and labor-intensive
nature of drafting judgment documents, there is significant po-
tential for using AI technologies to assist legal professionals. For
example, advanced IR techniques can systematically collect and
organize the relevant legal information, while generative AI can
support the drafting of well-structured legal texts. This combined
approach could substantially reduce manual effort and streamline
the document creation process.
Nonetheless, despite the rapid development of LLMs, their ap-
plication to judgment document generation remains largely unex-
plored. This gap arises primarily due to two critical factors. Firstly,
judgment document generation is a complex domain-specific task
that requires extensive legal expertise and knowledge. General-
purpose LLMs often lack the specialized legal knowledge necessary
to conduct complex legal reasoning and produce high-quality legal
texts. More importantly, the absence of standardized benchmarks
and automated evaluation methods significantly hinders progress
in this field. Currently, the quality of automatically generated judg-
ment documents can only be assessed through expert annotation,
which is both time-consuming and unscalable. Consequently, estab-
lishing robust benchmarks and automated evaluation frameworks
is crucial to overcoming the bottleneck in this field.arXiv:2503.14258v2  [cs.CL]  20 Mar 2025

Conference, Under Review, Su, et al.
To address these challenges and fill this research gap, we intro-
duce JuDGE (Judgment Document Generation Evaluation), a novel
benchmark for evaluating the performance of judgment document
generation. In collaboration with legal professionals, we further
establish a comprehensive, automated evaluation framework span-
ning four dimensions: penalty accuracy, convicting accuracy, refer-
encing accuracy, and documenting similarity with the ground truth.
Crucially, JuDGE is designed to mirror real-world judicial reasoning,
which typically follows a systematic syllogistic approach: the major
premise comprises relevant statutes, precedents, and jurisprudence,
while the minor premise is drawn from the factual circumstances
of the case. In practice, judges typically begin by establishing the
major premise comprising relevant statutes, precedents, and ju-
risprudence. Then summarize the minor premise from the case’s
factual circumstances. By synthesizing these elements through le-
gal reasoning, they conclude the judgment and draft the judgment
document detailing both the reasoning process and the final deci-
sion. To reflect this real-world process, the JuDGE benchmark focus
on generating the full judgment documents based on the factual de-
scriptions, which are sufficient as the minor premise for judgment
document generation. Specifically, the dataset incorporates two
specialized legal corpora (covering statutes, precedents, and other
domain-specific texts) to serve as the major premise, alongside a set
of publicly available factual descriptions from official sources that
function as the minor premise. We also provide the corresponding
ground-truth full judgment documents for each instance, offering
the reference for evaluation.
To demonstrate the use of our benchmark and facilitate future
research, we evaluate multiple baseline approaches using a variety
of LLMs. The baselines include standard methods that directly em-
ploy general-purpose and legal-domain LLMs to generate complete
judgment documents based on the provided case facts. Moreover,
we investigate more advanced strategies that leverage few-shot
in-context learning and supervised fine-tuning (SFT) to enhance
the models’ performance on this task. Beyond these conventional
approaches, we propose a more robust baseline based on multi-
source retrieval-augmented generation (Multi-source RAG), which
incorporates external knowledge from both statute corpus and judg-
ment document corpus. This strong baseline is intended to serve as
a reference point for future work on this task.
Experimental results on the Judgment Document Generation task
show that both general and domain-specific legal LLMs struggle
to produce high-quality legal documents when used directly or
with in-context learning. Augmenting these models via fine-tuning
and Multi-source RAG leads to performance improvements, but
there is still considerable room for improvement, underscoring the
challenging nature of this task. These findings highlight the need for
further exploration in this task and innovation at the intersection
of advanced information retrieval techniques and generative AI.
In summary, our contributions are threefold:
(1)We introduce JuDGE, a novel benchmark tailored for judgment
documents generation, incorporating a comprehensive dataset
designed to reflect real-world judicial reasoning processes.
(2)We establish a robust automated evaluation framework that
systematically evaluates the quality of generated judgment doc-
uments across multiple dimensions, facilitating scalable and
systematic evaluation for this task.
FactDescriptionUpon examination of the evidence, this court established thatin December 2017, defendant X1 defrauded victim X2 of a total of 81000 yuan in the name of helping his father to see a doctor in Qili Community, Beijing, on the grounds of doctor consultation and scientific research group treatment……StructureofJudgmentDocument
JudgmentResultAccordingly, the defendant X1 was sentenced in accordance with the provisions of Article 266, Article 52, Article 53, paragraph 1, and Article 67, paragraph 1 of the Criminal Law of the People's Republic of China.The defendant X1 committed the crime of defraud and was sentenced to one year and two months imprisonmentand a fine of 20000 yuan……HeadingSectionDaxingDistrictPeople'sCourtofBeijingMunicipalityCriminalJudgmentPaperNo.2019-Beijing-xxxPublicprosecutionauthority:DaxingDistrictProcuratorateDefendant X1, born on February 26, 1984, college educated,unemployed,male, with registered residence in XXX District, Beijing……
JudicialReasoningThis court holds that defendant X1 took advantage of the victim‘s trust in medical information to commit fraud. He deliberately deceived the victim by providing false information, resulting in significant financial losses. Although the defendant showed remorse during the trial, the large amount involved and the harm caused to society require punishment……
EndingIf you do not accept this judgment, you can file an appeal through this court or directly to the Beijing SecondIntermediate People's Court within 10 days from the second day of receiving the judgment……Figure 1: An illustration of the structure of judgment docu-
ment.
(3)We conduct experiments and analyses of various baseline ap-
proaches using different LLMs. Our findings serve as valuable
references for future research, while also illuminating key chal-
lenges and opportunities for advancement in this task.
2 Preliminaries and Task Definition
This section introduces the foundational concepts and formal defi-
nitions of the Judgment Document Generation task.
2.1 Judgment Document Structure and Notation
LetFdenote the space of fact descriptions extracted from legal
case documents. Each fact description 𝑓∈F provides a clear and
unbiased summary of the events and circumstances underlying a
legal dispute. The corresponding judgment document, denoted by
𝑗, belongs to the space J. A complete judgment document 𝑗∈J
typically comprises several interdependent sections, including but
not limited to:
•Heading Section : This section provides essential administrative
details that establish the jurisdiction and authenticity of the docu-
ment. It includes the court’s name, the case number, the presiding
judge’s name, and the date of judgment. Defendant information,
such as full name, gender, date of birth, and nationality, is also

JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System Conference, Under Review,
presented. These details formally identify the parties involved
and set the context for the proceedings.
•Fact Description : This section offers a clear and comprehensive
narrative of the events and circumstances that led to the legal
dispute. It outlines the key facts and actions in a concise and
objective manner. The description is crucial for establishing the
factual basis of the case. It provides the necessary context for
understanding the subsequent legal analysis.
•Judicial Reasoning : This section details the court’s evaluation
of both the factual and legal aspects of the case. It systematically
applies relevant laws, legal principles, and judicial precedents
to the established facts. The analysis discusses the evidence, the
offense’s nature, and the liability’s determination. It ensures that
the legal reasoning behind the decision is transparent and well-
founded.
•Judgment Result : This final section presents the court’s conclu-
sive decision along with its legal justification. It explicitly cites
the applicable statutes, regulations, and legal provisions that sup-
port the ruling. Additionally, the section outlines the specific
charges, sentencing details (including the prison term and fines),
and any other imposed penalties.
The interdependence between these sections ensures the judg-
ment document maintains logical consistency and legal coherence.
Each section builds upon the previous one, with the Judicial Anal-
ysis Process being generated based on the Fact Description while
simultaneously providing the legal foundation for the Judgment
Result .
2.2 Task Definition
The task of Judgment Document Generation is formalized as a con-
ditional text generation problem. Given an input fact description
𝑓∈F, the objective is to generate a complete judgment document
ˆ𝑗∈J that is both structurally coherent and legally sound. Formally,
the goal is to learn a mapping function:
M:F→J, (1)
such that for a given 𝑓, the generated document ˆ𝑗=M(𝑓)ap-
proximates the ground truth 𝑗in terms of content, structure, and
legal validity. In real-world applications, this mapping func-
tion is implemented through a domain-specific LLM or an
automated system.
In summary, the task of Judgment Document Generation in-
volves developing an automatic system that transforms a given
fact description 𝑓∈F into a complete and legally valid judgment
document ˆ𝑗∈J. Using our dataset, the fact description serves as
input to the system, and the output is the generated full judgment
document. The system’s performance is then automatically eval-
uated by comparing the generated document against the ground
truth across four perspectives, using twelve distinct metrics, as
detailed in Section §4.
3 Dataset Construction
This section outlines the construction process of the JuDGE dataset,
starting with the collection of criminal case documents and the
statute corpus. Next, it describes the pre-processing steps under-
taken to ensure high data quality, followed by expert annotationsto verify the dataset’s accuracy and consistency. The section then
presents key dataset statistics, offering insights into its composi-
tion, and concludes with a discussion of the ethical considerations,
highlighting our commitment to privacy, transparency, and regular
updates.
3.1 Data Collection
To construct the judgment document corpus for our dataset, we col-
lected publicly available criminal case documents from the China
Judgments Online platform, which is maintained by the Supreme
People’s Court of China, in line with the approach taken by Ma
et al.[25]. These documents span a 20-year period and were ran-
domly selected from an extensive database containing approxi-
mately six million legal cases. The selection process ensured that
the corpus included a diverse range of criminal cases, covering
various legal principles and judicial outcomes.
To construct the statutory articles corpus that serves as exter-
nal knowledge for our model, our legal team conducted detailed
annotations to identify the most relevant and up-to-date Chinese
statutory laws and regulations. These statutes were manually down-
loaded from official government websites to ensure accuracy and
currency. The laws were then segmented into individual articles
using automated scripts, facilitating easy retrieval and application
for downstream tasks. The complete list of statutes selected for
inclusion is publicly available on our official GitHub repository2.
3.2 Data Pre-processing
In the pre-processing phase, we implemented key filtering criteria
to ensure the quality and consistency of the dataset. The primary
filtering mechanisms are as follows: Firstly, we applied length re-
strictions to maintain document consistency with typical legal texts
and ensure manageable input sizes for model processing. Specifi-
cally, the fact description of each judgment document is limited to
a maximum of 1,000 Chinese characters, while the full judgment
text ranges from 1,000 to 3,000 Chinese characters. Secondly, we
filtered out case documents that don’t have key legal elements,
such as the Criminal Law articles, the crime type, and the sentence
duration, which are critical for establishing the legal validity of the
judgment. We also required the inclusion of standard legal phrases
like “this court believes” and “the judgment is as follows” to ensure
adherence to legal language conventions. Specifically, we designed
regular expressions to extract and validate each section3. For ex-
ample, the opening of the Fact Description section typically starts
with phrases such as “This court finds” or “It has been established
through trial”. The regular expressions can cover 98% of documents
in a dataset consisting of millions of judgment documents, with the
remaining 2% being non-standard or improperly formatted.
After filtering, we transform the Judgment Documents into struc-
tured representations, represented as a series of (key, value) pairs.
The specific fields for each document include CaseID, Fact Descrip-
tion Section, Judicial Reasoning Section, Judgment Result Section,
Sentence Length, Fine, Crime Type, and Referenced Legal Articles.
2https://github.com/oneal2000/JuDGE
3All the specific code implementations related to Data Pre-processing are available at:
https://github.com/oneal2000/JuDGE

Conference, Under Review, Su, et al.
Table 1: Fields and Descriptions of the JuDGE. This table
lists the key fields included in the dataset, along with their
respective descriptions.
Key Description
CaseID Unique identifier for the case
Fact Section summarizing the key facts of the case
Reasoning Section detailing the legal reasoning behind the judgment
Judgment Section outlining the court’s final decision and penalties
Sentence Duration of the prison sentence
Fine Monetary penalty imposed
Crime Type The type of crime involved in the case
Law Articles Legal statutes cited in the judgment
These fields are summarized in Table 1, which presents the keys
and their descriptions.
3.3 Expert Annotation
While the data pre-processing steps provided an initial quality
filter for judgment documents, we further enhanced the JuDGE
dataset through expert annotation. To achieve this, we recruited
law students to evaluate each judgment document instance based
on five critical aspects: document formatting, the correctness of
judicial reasoning, appropriate legal references, the accuracy of
crime classification, and the reasonableness of the sentencing. Each
document was annotated by two independent experts. If both an-
notators agreed that the document met the quality criteria, it was
classified as “usable.” If discrepancies arose between annotators,
the document was excluded from the dataset. The annotation team
consisted of seven members recruited from prominent law schools.
Our compensation plan, which was based on working hours, en-
sured that the annotation process was incentivized fairly, offering
an average hourly wage of 45 CNY, which is significantly higher
than the minimum wage requirements in Beijing.
To evaluate the reliability of the annotations, we applied Cohen’s
Kappa coefficient in a binary classification context. The analysis,
performed on 2,574 annotated instances, yielded a Kappa value
of 0.8012, indicating very high inter-annotator agreement. This
strong agreement reflects the consistency and reliability of the
expert annotations. After applying these filtering standards and
expert annotations, we obtained the final dataset with 2,505 legal
documents covering 182 different statutes and 142 distinct crime
types. This selection process ensured the dataset’s diversity in legal
references and crime types while also maintaining its high quality
for the benchmark.
3.4 Dataset Statistics
Table 2 summarizes the core numerical attributes of the JuDGE
dataset and its supplementary legal corpora. The dataset contains
2,505 fact-judgment pairs, covering a diverse range of criminal cases
with 142 unique charges and 182 distinct criminal law provisions.
This diversity suggests that the dataset is highly applicable for
modeling various legal scenarios.
The average lengths of the different sections further reflect the
dataset’s balanced level of detail. Fact descriptions average aroundTable 2: Basic Statistics of the JuDGE Dataset. “Total Fact-
Judgment Pairs” indicates the total number of paired entries
in the dataset; “Unique Charges” and “Unique Criminal Law
Provisions” refer to the number of distinct charges and law
articles covered in the test set. All length metrics are mea-
sured in Chinese characters.
Statistic Number
Total Fact-Judgment Pairs 2,505
Training Set Size 2,004
Test Set Size 501
Unique Charges 142
Unique Criminal Law Provisions 182
Avg. Fact Length 651.95
Avg. Reasoning Length 281.75
Avg. Judgment Result Length 207.06
Avg. Full Document Length 1,741.56
Avg. Charges per Document 1.26
Avg. Statutory Articles per Document 4.31
External Judgment Documents Corpus Size 103,251
External Statutory Articles Corpus Size 55,348
652 Chinese characters, providing a succinct yet informative ac-
count of case details. On a per-document basis, the inclusion of
roughly 1.26 charges and 4.31 statutory articles emphasizes each
case’s multifaceted legal context. Moreover, the dataset is supple-
mented by two extensive external corpora: one containing 103,251
judgment documents and another comprising 55,348 statutory
articles. These resources provide rich legal background informa-
tion that is particularly valuable for advanced IR technologies like
retrieval-augmented generation approaches.
3.5 Ethical Considerations
Throughout the construction of this benchmark, we have consis-
tently prioritized ethical considerations and taken essential mea-
sures to mitigate potential issues. To protect sensitive information,
all dataset entries have undergone strict anonymization, eliminating
any personally identifiable data. Furthermore, to enhance trans-
parency and facilitate reproducibility, we have publicly released
the JuDGE dataset, along with all the associated models and code,
on our official GitHub repository. This allows researchers to in-
dependently verify our work, replicate experiments, and extend
our contributions. In addition, JuDGE incorporates a Legal Corpus
comprising statutes, judicial interpretations, and other authorita-
tive legal texts. Given the evolving nature of legal frameworks,
we are committed to maintaining the dataset’s accuracy through
regular updates that reflect legislative changes and judicial devel-
opments. Finally, to ensure broad accessibility and foster further
research, JuDGE is distributed under the MIT license, which grants
researchers and developers unrestricted usage rights.
4 Evaluation Framework
Developing an effective evaluation framework for automatic judg-
ment generation requires identifying the key legal criteria that

JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System Conference, Under Review,
determine a judgment’s accuracy and quality. Through extensive
discussions with law students, legal scholars, and judges, we identi-
fied four critical dimensions that best capture the essential require-
ments of a well-reasoned legal judgment: penalty accuracy ,con-
victing accuracy ,referencing accuracy , and documenting similarity .
Our framework evaluates each generated judgment by comparing
it to an authoritative ground-truth judgment document, ensuring
alignment with actual case outcomes and established legal stan-
dards. In the following subsections, we first analyze the evaluation
criteria for each aspect and then define specific metrics to quantify
performance based on this analysis.
4.1 Penalty Accuracy.
In criminal proceedings, determining penalties such as prison sen-
tences and fines is both legally important and ethically sensitive.
Even minor inaccuracies in judgment can result in procedural er-
rors or unjust outcomes. To assess the precision of the penalties, we
compare each predicted penalty component with its correspond-
ing ground truth, aiming to quantitatively evaluate the document
generator’s performance in this aspect.
Metric Definition. To quantify how closely a predicted penalty
aligns with the ground truth, we define a normalized absolute dif-
ference for each penalty component:
𝑥=𝐿pred−𝐿true
max{𝐿pred,𝐿true}, (2)
where𝐿predand𝐿trueare the predicted and ground-truth values,
respectively. We then convert this difference into a score:
𝑆=1−𝑥. (3)
This formulation ensures that 𝑆remains in the range [0,1]and
provides a symmetric evaluation of prediction errors.
4.2 Convicting Accuracy.
Accurately identifying the charges is crucial in criminal cases to
ensuring that the judgment reflects the full scope of the offenses.
Incorrect classification or omission of charges can lead to legal inac-
curacies and undermine the fairness of the judgment. To assess the
accuracy of charge classification, we compare the predicted charges
in the generated judgment with the charges in the authoritative
ground-truth judgment. The goal is to evaluate the model’s ability
to identify all relevant charges correctly and comprehensively in
the case.
Metric Definition. We measure charge-level performance using
three standard classification metrics:
•Recall : The proportion of actual charges that the system cor-
rectly identifies.
•Precision : The proportion of predicted charges that are accurate.
•F1 Score : The harmonic mean of Precision and Recall, striking a
balance between completeness and correctness.
By capturing both completeness (Recall) and exactness (Precision),
these metrics provide a robust view of how well a system handles
diverse charges.4.3 Referencing Accuracy.
In jurisdictions following the civil law tradition, such as Germany,
Japan, and China, the accuracy of statutory citations is fundamen-
tal to the legal validity of a ruling. Unlike common law systems,
where judicial precedents play a dominant role, civil law systems
rely on codified statutes as the primary legal authority. As a result,
judgments must correctly and comprehensively reference the rele-
vant legal provisions to ensure their legitimacy and adherence to
statutory law. Given that this study focuses on the Chinese legal
framework, which exemplifies the civil law system, an evaluation
metric is introduced to assess the correctness of statutory citations
in generated judgments.
Metric Definition. To measure the accuracy of statutory citations,
the legal provisions cited in the generated judgment are systemati-
cally compared with those in the ground-truth judgment. Errors in
statutory references may occur in two forms: under-citation, where
essential legal provisions are omitted, and over-citation, where ir-
relevant or incorrect statutes are referenced. To assess these errors
quantitatively, three standard classification metrics are employed:
•Recall : The proportion of correctly cited ground-truth statutes
among all relevant statutes. A higher Recall indicates more com-
prehensive legal referencing and fewer omissions.
•Precision : The proportion of correctly cited statutes among all
citations in the generated judgment. A higher Precision score
reflects the system’s ability to avoid extraneous or incorrect legal
references.
•F1 Score : The harmonic mean of Precision and Recall, balancing
the completeness of legal references with their correctness.
These metrics collectively provide a rigorous evaluation of the
system’s ability to identify and accurately apply statutory provi-
sions in the generated judgments. Since the JuDGE dataset only
contains judgment documents related to criminal law, our proposed
metrics are currently focused on evaluating reference accuracy in
criminal law articles.
4.4 Documenting Similarity to Ground Truth.
Beyond evaluating specific elements such as penalties, charges, and
legal references, it is essential to assess the semantic consistency
of a generated judgment with the ground truth. Legal decisions
involve complex reasoning, where the Judicial Reasoning and Judg-
ment Result sections must align substantively with the ground truth
documents. Traditional lexical-based evaluations often fail to cap-
ture these deeper similarities, as legal reasoning can be conveyed
in different but equally valid ways. To address this, our evaluation
focuses on semantic alignment, ensuring that key arguments and
legal justifications are preserved even when expressed with varying
wording.
Metric Definition. We compare the Judicial Reasoning andJudg-
ment Result sections of the generated text against those of the
reference using two semantics-based metrics4:
•METEOR [1]: Captures semantic variations and paraphrases,
making it well-suited for longer and nuanced legal documents
4The intuitive significance of the numerical values presented in “METH. ” and “BERTS.”
is further clarified by a series of case studies, which can be accessed on our dataset’s
official websitehttps://github.com/oneal2000/JuDGE

Conference, Under Review, Su, et al.
where the same reasoning may be expressed in multiple, equally
valid ways.
•BERTScore [56]: Leverages contextualized embeddings to mea-
sure deep semantic alignment between generated and ground-
truth texts, ensuring that subtle differences in vocabulary do
not mask or inflate true alignment.
We opt against traditional metrics like BLEU or ROUGE, as these
n-gram-based measures can overlook the complex reasoning struc-
tures and nuanced legal vocabulary inherent in real judgments.
4.5 Automatic Evaluation Implementation
As described in the previous subsections, our evaluation framework
focuses on assessing penalties, charges, legal citations, and seman-
tic coherence in automatically generated legal documents. These
elements are not explicitly provided as separate outputs in the gen-
eration task (see Section 2.2). Instead, the models produce complete
legal documents, thus the automatic evaluation framework must
automatically extract the critical features required for our metrics,
such as charges, sentence lengths, fines, and cited statutes. Below,
we outline the methodology and rationale behind our automatic
extraction process.
Chinese legal judgments typically follow a highly standardized
structure mandated by the Supreme People’s Court, which enables
reliable extraction of relevant information. Each document contains
three main sections: Fact Description, Judicial Reasoning, and Judg-
ment Result. These sections are introduced by fixed phrases, such
as “After trial, it was found that...” (for facts), “The court holds
that. . . ” (for reasoning), and “According to [Law], the judgment is as
follows. . . ” (for results). We automatically parse the generated text
using these recurring lexical markers to isolate each section. There
are various ways these sections can be phrased across different
cases. Through extensive research and analysis, we have identified
a wide range of alternative expressions that fulfill the same struc-
tural role in each section. Similarly, for the Judgment Result section,
we also design and apply comprehensive regular-expression pat-
terns to extract key features, including specific charges, sentences,
fines, and cited legal articles.
Our extraction approach aligns with official guidelines and well-
established templates used in Chinese courts. Based on a large-scale
sampling of publicly available judgments, we verified that our rules
correctly handle over 99% of standard-format documents. Conse-
quently, if a system-generated judgment document cannot
be correctly parsed by our automatic evaluation framework,
this suggests a formatting error in this document. In such
cases, we set the corresponding fields (e.g., charges, sentences)
to empty or zero. This design choice ensures that documents that
do not conform to real-world standards receive a fair penalty in our
evaluation, as they do not fulfill the basic structural requirements
expected in a valid judgment document.
Although the development and implementation of robust extrac-
tion rules is crucial for the accuracy of our evaluation framework,
it primarily involves engineering-specific tasks. Since the imple-
mentation does not directly address the core scientific questions
relevant to the IR and NLP community, we do not elaborate on thetechnical details in this paper. For readers interested in implementa-
tion specifics, including all regular expression patterns and parsing
scripts, these are publicly available in our GitHub repository.
5 Experimental Setup
In this section, we present the experimental setup on the JuDGE
Benchmark. Section §5.1 covers the implementation details of our
baselines. In Section §5.2, we introduce the LLMs selected for this
task. Finally, Section §5.3 provides a detailed explanation of the
implementation, including how we train and fine-tune both the
retriever components and the LLMs.
5.1 Baselines
To support future research, we systematically evaluate several base-
line approaches on our proposed JuDGE benchmark using various
LLMs. These baselines include few-shot in-context learning and
fine-tuning techniques to adapt general LLMs to the judgment doc-
ument generation task. Additionally, we introduce an advanced
baseline, Multi-source Retrieval-Augmented Generation (MRAG),
which integrates knowledge from both statute and judgment docu-
ment corpora, establishing a strong baseline performance for future
reference.
5.1.1 Few-shot In-context Learning. In this baseline, we leverage
a few representative examples from the training set as in-context
demonstrations. Each prompt concatenates a brief instruction with
exemplar pairs comprising a fact description and its corresponding
full judgment document to illustrate the expected document struc-
ture and generation process. The LLM is then prompted to generate
a judgment document for a new fact based on the pattern provided
in the examples5.
5.1.2 Supervised Fine-tuning. For supervised fine-tuning, we di-
rectly optimize the model on the train set. Given a fact description 𝑓
and its corresponding judgment document 𝑗, the model is trained to
maximize the likelihood of generating 𝑗conditioned on 𝑓. Formally,
for a document of length 𝑇, the training objective is defined as:
LFT=−𝑇∑︁
𝑡=1log𝑃(𝑤𝑡|𝑤<𝑡,𝑓), (4)
where𝑤𝑡is the token at position 𝑡and𝑤<𝑡denotes the preceding
tokens. This loss encourages the model to produce structurally
coherent and legally valid judgment documents.
5.1.3 Retrieval Augmented Generation. In recent years, Retrieval-
Augmented Generation (RAG) has emerged as a key approach to mit-
igating hallucinations [ 27,42,45] in LLMs and enhancing their per-
formance on knowledge-intensive tasks [ 3,9,12,18,43,44,48,49].
The RAG paradigm follows the "Retrieval-then-Read" framework,
where a retriever [32, 54, 55] or a complex retrieval system [7, 34]
is adopted to search for relevant information, the retrieved informa-
tion is then incorporated into the input context of an LLM, enabling
it to generate responses based on external knowledge.
5Due to space constraints, we have not included the specific prompts in the main
text. However, all prompts used in this study are available at the following link:
https://github.com/oneal2000/JuDGE

JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System Conference, Under Review,
Building upon the RAG paradigm, we explore and propose an
RAG-based approach as a baseline for our task. In practical legal
applications, generating a complete judgment document requires
the integration of extensive legal information, including relevant
statutory regulations, historical case precedents, and fundamen-
tal legal principles. To address this requirement, we propose the
Multi-Source Retrieval-Augmented Generation (MRAG) baseline
that integrates external knowledge to enhance the generation pro-
cess. This baseline divides judgment generation into two phases:
Information Collection and Document Generation. In the Infor-
mation Collection phase, two distinct retrievers are employed to
retrieve relevant information from different sources. Specifically,
the Law Retriever targets relevant statutory regulations, while the
Case Retriever focuses on retrieving relevant judgment documents.
In the Document Generation phase, the LLM leverages the fact de-
scription from the dataset along with the information gathered in
the information collection phase to generate the complete judgment
document.
The Law Retriever is designed to retrieve relevant statutes based
on a case’s fact description. It employs a dual-encoder architecture,
where the relevance score between the factual description 𝑓and a
statute𝑠is defined as the dot product of their respective embeddings.
Formally, we define:
𝐸𝑚𝑏(𝑋)=𝑡𝑟𝑎𝑛𝑠𝑓𝑜𝑟𝑚𝑒𝑟[𝐶𝐿𝑆](𝑋), (5)
𝑆(𝑞,𝑠)=𝐸𝑚𝑏(𝑓)⊤·𝐸𝑚𝑏(𝑠), (6)
where𝑓denotes the factual description and 𝑠denotes the statute.
The function transformer[𝐶𝐿𝑆](·)produces a contextualized vector
for each token, and we select the [𝐶𝐿𝑆]token’s vector as the input’s
embedding. In Equation 6, the dot product of the embeddings is
used as the relevance score 𝑆. To train the Law Retriever, we employ
a contrastive learning strategy. Specifically, for each instance in the
JuDGE dataset, the model is trained to retrieve law articles relevant
to a given Fact. The law articles cited in the ground truth judgment
document serve as positive examples, while all other articles in the
corpus that are not cited are treated as negative examples. During
training, each fact is paired with its corresponding positive law
article𝑎+
𝑖. To construct the negative set, we randomly sample six
negative articles from the statute corpus, enabling the model to
distinguish relevant provisions from irrelevant ones. The model is
then optimized using the following loss function:
L(𝑓,𝑎+
𝑖,𝑁)=−logexp(𝑆(𝑓,𝑎+
𝑖))
exp(𝑆(𝑓,𝑎+
𝑖))+Í
𝑎−∈𝑁exp(𝑆(𝑓,𝑎−)),(7)
where𝑁represents the set of irrelevant law articles, and 𝑆is defined
in Equation 6 . This objective encourages the model to assign higher
relevance scores to the relevant law articles while minimizing the
scores of non-relevant ones.
In Table 3, we present the performance of our Law Retriever on
the statute retrieval task and compare it with traditional lexical-
based methods, TF-IDF [ 31] and BM25 [ 32]. The results show that
our Law Retriever significantly outperforms these baselines across
all reported metrics, highlighting its ability to capture deeper seman-
tic relationships between the factual descriptions and the statutoryTable 3: Comparison of the Law Retriever in the MRAG base-
line with traditional retrieval algorithms. Metrics include
MRR, Precision (P), and Recall (R) at various cutoffs (5 and
10). The best results are in bold.
MRR@100 P@5 P@10 R@5 R@10
TF-IDF 0.3605 0.1625 0.1200 0.2096 0.3074
BM25 0.4373 0.1501 0.1024 0.1958 0.2649
Law Retriever 0.8328 0.4535 0.3509 0.5529 0.8262
texts. Given that relevant legal provisions are often not lexically
identical to the facts, relying solely on exact term matching proves
insufficient. Consequently, these findings underscore the impor-
tance of training a specialized dense retrieval model.
For the Case Retriever, we employ the pre-trained dense retrieval
model SAILER [ 19], a structure-aware language model specifically
designed for legal document representation. The retrieval process
follows the standard approach used in the Legal Case Retrieval
task [ 20,25,26,39], a well-established and extensively studied
problem in the information retrieval community. Given that SAILER
has consistently demonstrated strong performance in this domain,
we adopt it directly as our retriever and do not further elaborate
on the retrieval process in this paper.
Following the retrieval stage, the base language model is fine-
tuned to generate the final judgment document. This fine-tuning is
conducted using a prompt template that concatenates the case fact
description with the top retrieved legal information—specifically,
the two most relevant cases from the Case Retriever and the ten
most pertinent statutes from the Law Retriever. The language model
is trained using a next-token prediction loss defined as:
LFT=−𝑇∑︁
𝑡=1log𝑃(𝑤𝑡|𝑤<𝑡,context), (8)
where𝑇is the length of the judgment document, and the con-
text comprises both the fact description and the retrieved legal
knowledge. During inference, the same prompt structure is em-
ployed, with dynamically retrieved cases and statutes replacing
their training-time counterparts. This retrieval-augmented strategy
enables the language model to generate judgment documents that
are not only structurally complete but also legally coherent and
grounded in external legal information.
5.2 Selected Large Language Models
To evaluate the performance of various LLMs on our proposed
JuDGE dataset, we conduct experiments using both general-purpose
and legal-domain models.
5.2.1 General-Purpose Models. We use four variants from the Qwen
2.5 series [ 29], consisting of both 3B and 7B parameter scales and
two model types (base vs. instruct). Specifically, the base models,
Qwen-2.5-3B and Qwen-2.5-7B are pre-trained on large-scale cor-
pora without alignment to human instructions. In contrast, the
instruct versions, Qwen-2.5-3B-Instruct and Qwen-2.5-7B-Instruct,
are fine-tuned with instruction tuning and related alignment meth-
ods to improve their ability to follow human instructions.

Conference, Under Review, Su, et al.
Table 4: Experimental results on multiple baselines across six different LLMs. QW-3B-Base refers to the Qwen-2.5-3B model,
while QW-3B-Chat refers to the Qwen-2.5-3B-Instruct model (the same naming convention applies to the 7B models). “METH.”
denotes METEOR, “BERTS.” for BERTScore, and “Prec.” for precision. The best results for each LLM are highlighted in bold.
Penalty Acc. Convicting Acc. Referencing Acc. Reasoning Section Judgment Section
Model Method Prison Fine Recall Prec. F1 Recall Prec. F1 METH. BERTS. METH. BERTS.
QW-3B-BaseSFT 0.5975 0.5149 0.9381 0.9375 0.9378 0.6915 0.6441 0.6669 0.6281 0.8400 0.7147 0.7408
RAG 0.6273 0.5132 0.9471 0.9511 0.9491 0.7569 0.6847 0.7190 0.5945 0.8412 0.7045 0.8307
QW-7B-BaseSFT 0.6380 0.5273 0.9651 0.9664 0.9657 0.7500 0.7187 0.7340 0.6523 0.8527 0.7548 0.8025
RAG 0.6489 0.5458 0.9411 0.9441 0.9426 0.7356 0.7649 0.7502 0.6020 0.8533 0.7047 0.8565
LexiLaw-6BDirect 0.0408 0.0261 0.6178 0.6228 0.6202 0.0010 0.0040 0.0016 0.0994 0.7259 0.3191 0.0043
ICL 0.0102 0.0126 0.4072 0.4112 0.4092 0.0071 0.0088 0.0078 0.3403 0.7179 0.1636 0.0058
SFT 0.5926 0.4942 0.9401 0.9288 0.9344 0.6030 0.6908 0.6439 0.6471 0.8399 0.6907 0.6715
Hanfei-7BDirect 0.4902 0.3541 0.8623 0.8683 0.8653 0.4624 0.5306 0.4941 0.5132 0.7643 0.5433 0.4860
ICL 0.0635 0.0282 0.1128 0.1118 0.1123 0.0209 0.0407 0.0276 0.3184 0.5404 0.2632 0.0434
SFT 0.6520 0.5526 0.9381 0.9318 0.9350 0.7238 0.6813 0.7019 0.6312 0.8389 0.7141 0.7865
QW-3B-ChatDirect 0.6117 0.4868 0.8713 0.8752 0.8732 0.5972 0.7300 0.6570 0.4620 0.7724 0.3658 0.7120
ICL 0.6278 0.4720 0.8832 0.8746 0.8789 0.4928 0.7092 0.5815 0.4245 0.6901 0.3699 0.6790
SFT 0.6673 0.5443 0.9401 0.9444 0.9423 0.7205 0.7503 0.7351 0.5836 0.8509 0.6803 0.8889
RAG 0.6527 0.5296 0.9511 0.9531 0.9521 0.7292 0.7484 0.7387 0.5807 0.8425 0.7095 0.8851
QW-7B-ChatDirect 0.6655 0.5075 0.9242 0.9242 0.9242 0.5045 0.8161 0.6235 0.5061 0.8098 0.4161 0.7823
ICL 0.6731 0.5095 0.9371 0.9385 0.9378 0.6537 0.7915 0.7161 0.5436 0.8245 0.6220 0.8523
SFT 0.6604 0.5506 0.9521 0.9518 0.9519 0.7787 0.7694 0.7740 0.6220 0.8597 0.7242 0.8961
RAG 0.6795 0.5512 0.9611 0.9604 0.9607 0.7955 0.7509 0.7726 0.6076 0.8513 0.7236 0.8887
5.2.2 Legal-Domain Models. We select HanFei-1.0 (7B) and Lex-
iLaw (6B) for domain-specific LLM experiments. HanFei-1.06is
trained on a broad corpus of law-related texts, including legal news,
forums, statutes, judicial interpretations, legal consultations, bar
exam questions, and court judgments. This model is designed to
enhance its reasoning and consultation capabilities in legal contexts.
LexiLaw7is built upon the ChatGLM-6B model and has been fur-
ther fine-tuned on specialized legal data to improve its performance
and knowledge specificity for legal tasks.
5.3 Implementation Details
For the retrieval module in the RAG baseline, we train the dense
retriever as follows. The Law Retriever are initialized using the
pre-trained Chinese-RoBERTa-WWM8. We use the AdamW opti-
mizer with a learning rate of 5×10−6, with mixed-precision ( 𝑓𝑝16)
training for enhanced efficiency. The models are trained for two
epochs. For fine-tuning the LLM, we use the Deepspeed framework
with Zero-2 optimization to enable efficient full-model fine-tuning.
This process employs the AdamW optimizer, a learning rate of
3×10−5, and mixed-precision ( 𝑏𝑓16) training for improved mem-
ory efficiency. The LLM is fine-tuned for four epochs on the dataset.
All training processes are conducted on a server equipped with
eight NVIDIA A100 GPUs, each with 40GB of memory. For the
generation phase of LLMs, all experiments are conducted using
the publicly available Hugging Face implementations. We use the
6https://github.com/siat-nlp/HanFei
7https://github.com/CSHaitao/LexiLaw
8https://huggingface.co/hfl/chinese-roberta-wwm-extdefault hyperparameters and chat template provided in their offi-
cial Hugging Face repository. All the prompt templates used in this
paper are available in our GitHub repository9.
6 Experimental Results
In this section, we present the main experimental results on the
JuDGE benchmark and provide an in-depth analysis of the findings.
As shown in Table 4, we compare several baselines across different
LLMs, including both base and chat-oriented models, as well as
legal LLMs. All the prompt templates used in the experiments are
detailed in our official GitHub repository10. We omit direct genera-
tion and in-context learning on the two base models (QW-3B-Base
and QW-7B-Base) because they are not instruction-tuned or aligned
with human preferences. Consequently, their default behavior is to
continue text generation rather than following a structured legal
drafting instruction. Below, we summarize the key observations:
(1)Legal LLMs (e.g., LexiLaw-6B andHanfei-7B ) show notably
poor performance when generating judgment documents directly
or via few-shot in-context learning. Even with in-context learn-
ing, the performance remains weak. A closer examination of the
generated outputs reveals that these models often reuse or par-
tially copy the exemplars provided in the prompt, neglecting to
incorporate the unique facts of each case. We suspect that this phe-
nomenon arises from catastrophic forgetting. Since these models
were continually pre-trained and fine-tuned on large-scale legal
9https://github.com/oneal2000/JuDGE
10https://github.com/oneal2000/JuDGE

JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System Conference, Under Review,
data (e.g., legal QA), their parameters have diverged significantly
from their initial state, resulting in the loss of previously acquired
knowledge and instruction-following capabilities. (2)Few-shot in-
context learning benefits larger chat models more significantly. For
QW-7B-Chat, introducing a few-shot example in the prompt con-
sistently enhances performance across almost all the metrics. In
contrast, QW-3B-Chat sees only marginal gains from in-context
learning. This difference highlights that larger models with more
capacity generally exhibit stronger in-context learning capabilities.
(3)Supervised fine-tuning substantially improves performance. Re-
gardless of the models’ initial capability, supervised fine-tuning
(SFT) leads to improvements over direct generation and in-context
learning. This trend is observed across both general-purpose models
(e.g., QW-3B-Chat and QW-7B-Chat) and legal LLMs (e.g., LexiLaw-
6B and Hanfei-7B). Notably, the improvement is particularly large
for legal LLMs: while their performance is almost unusable before
fine-tuning, it becomes acceptable and sometimes competitive after
the task-specific SFT. These findings reinforce the importance of su-
pervised adaptation for certain downstream tasks. (4)Multi-source
retrieval-augmented generation (RAG) offers competitive advan-
tages. For instance, MRAG consistently achieves the highest recall,
precision, and F1 score in convicting accuracy. However, while
MRAG achieves high recall in referencing accuracy, its precision
is relatively lower. We attribute this to our top- 𝑘retrieval strat-
egy (here𝑘=10), which occasionally supplies multiple irrelevant
statutes. As indicated by Table 2, only 4.31 criminal statutes are
truly relevant per case. Since current LLMs are not fully adept at
filtering out misleading articles, this results in a precision drop.
7 Related Work
7.1 Legal Information Retrieval
Legal information retrieval (IR) differs from open-domain IR as it re-
quires the retrieval system to incorporate extensive legal knowledge
and consider both semantic and legal perspectives when modeling
the text. A prominent research direction is Legal Case Retrieval ,
which focuses on retrieving past legal cases that are relevant to a
given query case, aiding legal professionals in case analysis and
decision-making [ 25,26,38]. Early work in this area primarily relied
on lexical matching methods to assess similarity [ 33]. Recently, with
the development of Dense Retrieval [ 10,15,37,41,55], researchers
have begun to explore methods to enhance retrievers with deeper
legal understanding via large-scale pretraining on unlabeled legal
corpora. For instance, Sailer [ 19] trains the dense retriever through
an autoencoder-like approach, where the embedding of a case’s
Fact section is used to reconstruct other sections of the document,
thereby infusing the fact embeddings with richer contextual in-
formation. Caseformer [ 38] proposes an unsupervised contrastive
learning approach for automatically measuring similarity between
legal documents. The method leverages legal case attributes such
as charges and cited legal provisions to automatically generate
positive and negative cases relative to a query case, thereby train-
ing dense retrieval models. This fully automated process enables
the generation of substantial training data from unlabeled cor-
pora, enhancing retrieval performance. Another major direction is
Statute Retrieval , which aims to locate and retrieve relevant statu-
tory articles for given a query. For instance, the annual COLIEEcompetitions [ 11,16,20,21,30] focus on Japanese legal bar exam
questions, where the goal is to retrieve relevant statutes from the
Japanese Civil Code. Similarly, the AILA [ 2] targets the Indian legal
system, using queries from Supreme Court of India judgments to
retrieve relevant statutes. Beyond professional use cases, statute
retrieval also caters to non-professional users who may lack formal
legal training; for example, STARD [ 40] is specifically designed
to address layperson queries, providing plain-language questions
paired with relevant statutory content. Beyond retrieval, Legal QA
tasks aim to satisfy the public’s need for accessible legal informa-
tion [ 8], such as the French long-form QA dataset LLeQA [ 23] and
GerLayQA [ 5], which contains laymen’s German legal questions
paired with authoritative law book paragraphs.
7.2 Legal Judgment Prediction
Legal Judgment Prediction (LJP) is a fundamental task in many
Civil Law jurisdictions. It aims to infer judicial outcomes based
on a case’s factual description, such as applicable legal articles,
charges, and penalty terms. For instance, Luo et al .[24] integrates
statutory details to enhance charge classification by leveraging the
inherent structure of legal codes. Similarly, Hu et al .[13] proposes
a few-shot learning approach to address data scarcity in charge
prediction tasks. Building on these efforts, researchers introduce
advanced neural architectures, including multi-channel attentive
networks and gating mechanisms to model the complex interactions
among legal facts, charges, and statutes [ 6,14,22]. More recently,
researchers have explored the applicability of large language models
for LJP, showing that LLMs can further enhance predictive accuracy
under realistic conditions [28].
Although previous efforts have notably advanced the accuracy
of LJP, they typically formulate the task as a classification problem
(i.e., predicting discrete labels for charges, articles, and sentences).
In contrast, our work focuses on the generative task, requiring the
model to produce the complete judgment document based on
the facts and relevant legal knowledge. This extends beyond merely
outputting labels: the generation process must logically organize
all necessary legal elements into a full judgment document. Such an
approach more closely aligns with practical legal scenarios, where
judges produce well-structured rulings that explicitly cite relevant
statutes, articulate charges, and determine sentencing outcomes.
As a result, the JuDGE benchmark broadens the scope of LJP from
classification to generation and bridges the gap between legal clas-
sification tasks and real-world judgment document drafting.
8 Conclusion
In this paper, we introduced JuDGE, a comprehensive benchmark
for evaluating the generation of judgment documents within the
legal domain. By formalizing the task and providing the bench-
mark dataset, we establish a solid foundation for further research in
this important yet underexplored area. Through collaboration with
legal professionals, we also developed an automated evaluation
framework that measures performance across four key dimensions:
penalty accuracy, convicting accuracy, referencing accuracy, and
similarity to ground truth. Our experimental results reveal that

Conference, Under Review, Su, et al.
current approaches struggle to produce high-quality judgment doc-
uments. Although supervised fine-tuning and our proposed Multi-
source RAG approach show improvements, the performance gap
indicates that judgment document generation remains challenging
and requires further research and innovation. We hope JuDGE will
inspire more advanced techniques in judgment document gener-
ation, ultimately helping the legal community improve efficiency
and reduce manual workloads.
References
[1]Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for
MT evaluation with improved correlation with human judgments. In Proceedings
of the acl workshop on intrinsic and extrinsic evaluation measures for machine
translation and/or summarization . 65–72.
[2]Paheli Bhattacharya, Kripabandhu Ghosh, Saptarshi Ghosh, Arindam Pal, Parth
Mehta, Arnab Bhattacharya, and Prasenjit Majumder. 2019. Fire 2019 aila track:
Artificial intelligence for legal assistance. In Proceedings of the 11th annual meeting
of the forum for information retrieval evaluation . 4–6.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, et al .2022. Improving language models by retrieving
from trillions of tokens. In International conference on machine learning . PMLR,
2206–2240.
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners. Advances in neural
information processing systems 33 (2020), 1877–1901.
[5]Marius Büttner and Ivan Habernal. 2024. Answering legal questions from laymen
in German civil law system. In Proceedings of the 18th Conference of the European
Chapter of the Association for Computational Linguistics (Volume 1: Long Papers) .
2015–2027.
[6]Huajie Chen, Deng Cai, Wei Dai, Zehui Dai, and Yadong Ding. 2019. Charge-based
prison term prediction with deep gating network. arXiv preprint arXiv:1908.11521
(2019).
[7]Xuesong Chen, Ziyi Ye, Xiaohui Xie, Yiqun Liu, Xiaorong Gao, Weihang Su,
Shuqi Zhu, Yike Sun, Min Zhang, and Shaoping Ma. 2022. Web search via an
efficient and effective brain-machine interface. In Proceedings of the fifteenth ACM
international conference on web search and data mining . 1569–1572.
[8] Phong-Khac Do, Huy-Tien Nguyen, Chien-Xuan Tran, Minh-Tien Nguyen, and
Minh-Le Nguyen. 2017. Legal question answering using ranking SVM and deep
convolutional neural network. arXiv preprint arXiv:1703.05320 (2017).
[9]Qian Dong, Qingyao Ai, Hongning Wang, Yiding Liu, Haitao Li, Weihang Su,
Yiqun Liu, Tat-Seng Chua, and Shaoping Ma. 2025. Decoupling Knowledge and
Context: An Efficient and Effective Retrieval Augmented Generation Framework
via Cross Attention. In Proceedings of the ACM on Web Conference 2025 .
[10] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun
Liu. 2024. Scaling Laws For Dense Retrieval. arXiv preprint arXiv:2403.18684
(2024).
[11] Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Juliano Rabelo, Ken Satoh, and
Masaharu Yoshioka. 2023. Summary of the competition on legal information, ex-
traction/entailment (COLIEE) 2023. In Proceedings of the Nineteenth International
Conference on Artificial Intelligence and Law . 472–480.
[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. In International conference on
machine learning . PMLR, 3929–3938.
[13] Zikun Hu, Xiang Li, Cunchao Tu, Zhiyuan Liu, and Maosong Sun. 2018. Few-shot
charge prediction with discriminative legal attributes. In Proceedings of the 27th
international conference on computational linguistics . 487–498.
[14] Liangyi Kang, Jie Liu, Lingqiao Liu, Qinfeng Shi, and Dan Ye. 2019. Creating
auxiliary representations from charge definitions for criminal charge prediction.
arXiv preprint arXiv:1911.05202 (2019).
[15] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. arXiv preprint arXiv:2004.04906 (2020).
[16] Mi-Young Kim, Juliano Rabelo, Randy Goebel, Masaharu Yoshioka, Yoshinobu
Kano, and Ken Satoh. 2022. Coliee 2022 summary: Methods for legal document
retrieval and entailment. In JSAI International Symposium on Artificial Intelligence .
Springer, 51–67.
[17] Sushanta Kumar, P Krishna Reddy, V Balakista Reddy, and Malti Suri. 2013.
Finding similar legal judgements under common law system. In Databases in
Networked Information Systems: 8th International Workshop, DNIS 2013, Aizu-
Wakamatsu, Japan, March 25-27, 2013. Proceedings 8 . Springer, 103–116.
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[19] Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue Wu, Yiqun Liu, Chong Chen,
and Qi Tian. 2023. SAILER: structure-aware pre-trained language model for legal
case retrieval. In Proceedings of the 46th International ACM SIGIR Conference on
Research and Development in Information Retrieval . 1035–1044.
[20] Haitao Li, Weihang Su, Changyue Wang, Yueyue Wu, Qingyao Ai, and Yiqun Liu.
2023. Thuir@ coliee 2023: Incorporating structural knowledge into pre-trained
language models for legal case retrieval. arXiv preprint arXiv:2305.06812 (2023).
[21] Haitao Li, Changyue Wang, Weihang Su, Yueyue Wu, Qingyao Ai, and Yiqun Liu.
2023. THUIR@ COLIEE 2023: more parameters and legal knowledge for legal
case entailment. arXiv preprint arXiv:2305.06817 (2023).
[22] Shang Li, Hongli Zhang, Lin Ye, Xiaoding Guo, and Binxing Fang. 2019. Mann:
A multichannel attentive neural network for legal judgment prediction. IEEE
Access 7 (2019), 151144–151155.
[23] Antoine Louis, Gijs van Dijck, and Gerasimos Spanakis. 2024. Interpretable
long-form legal question answering with retrieval-augmented large language
models. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38.
22266–22275.
[24] Bingfeng Luo, Yansong Feng, Jianbo Xu, Xiang Zhang, and Dongyan Zhao. 2017.
Learning to predict charges for criminal cases with legal basis. arXiv preprint
arXiv:1707.09168 (2017).
[25] Yixiao Ma, Yunqiu Shao, Yueyue Wu, Yiqun Liu, Ruizhe Zhang, Min Zhang, and
Shaoping Ma. 2021. LeCaRD: a legal case retrieval dataset for Chinese law system.
InProceedings of the 44th international ACM SIGIR conference on research and
development in information retrieval . 2342–2348.
[26] Yixiao Ma, Yueyue Wu, Weihang Su, Qingyao Ai, and Yiqun Liu. 2023. CaseEn-
coder: A Knowledge-enhanced Pre-trained Model for Legal Case Encoding. arXiv
preprint arXiv:2305.05393 (2023).
[27] Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023. Selfcheckgpt: Zero-
resource black-box hallucination detection for generative large language models.
arXiv preprint arXiv:2303.08896 (2023).
[28] Shubham Kumar Nigam, Aniket Deroy, Subhankar Maity, and Arnab Bhat-
tacharya. 2024. Rethinking Legal Judgement Prediction in a Realistic Sce-
nario in the Era of Large Language Models. arXiv:2410.10542 [cs.CL] https:
//arxiv.org/abs/2410.10542
[29] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[30] Juliano Rabelo, Randy Goebel, Mi-Young Kim, Yoshinobu Kano, Masaharu Yosh-
ioka, and Ken Satoh. 2022. Overview and discussion of the competition on legal
information extraction/entailment (COLIEE) 2021. The Review of Socionetwork
Strategies 16, 1 (2022), 111–133.
[31] Juan Ramos et al .2003. Using tf-idf to determine word relevance in document
queries. In Proceedings of the first instructional conference on machine learning ,
Vol. 242. Citeseer, 29–48.
[32] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends ®in Information Retrieval
3, 4 (2009), 333–389.
[33] Guilherme Moraes Rosa, Ruan Chaves Rodrigues, Roberto Lotufo, and Rodrigo
Nogueira. 2021. Yes, bm25 is a strong baseline for legal case retrieval. arXiv
preprint arXiv:2105.05686 (2021).
[34] Alireza Salemi and Hamed Zamani. 2024. Towards a search engine for machines:
Unified ranking for multiple retrieval-augmented large language models. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 741–751.
[35] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel
Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias
Gallé, et al .2022. Bloom: A 176b-parameter open-access multilingual language
model. arXiv preprint arXiv:2211.05100 (2022).
[36] Dong Shu, Haoran Zhao, Xukun Liu, David Demeter, Mengnan Du, and Yongfeng
Zhang. 2024. LawLLM: Law large language model for the US legal system.
InProceedings of the 33rd ACM International Conference on Information and
Knowledge Management . 4882–4889.
[37] Weihang Su, Qingyao Ai, Xiangsheng Li, Jia Chen, Yiqun Liu, Xiaolong Wu, and
Shengluan Hou. 2023. Wikiformer: Pre-training with Structured Information of
Wikipedia for Ad-hoc Retrieval. arXiv preprint arXiv:2312.10661 (2023).
[38] Weihang Su, Qingyao Ai, Yueyue Wu, Yixiao Ma, Haitao Li, and Yiqun Liu. 2023.
Caseformer: Pre-training for Legal Case Retrieval. arXiv preprint arXiv:2311.00333
(2023).
[39] WEIHANG SU, QINGYAO AI, YUEYUE WU, ANZHE XIE, CHANGYUE WANG,
YIXIAO MA, HAITAO LI, ZHIJING WU, YIQUN LIU, and MIN ZHANG. 2024.
Pre-training for Legal Case Retrieval Based on Inter-Case Distinctions. (2024).

JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System Conference, Under Review,
[40] Weihang Su, Yiran Hu, Anzhe Xie, Qingyao Ai, Quezi Bing, Ning Zheng, Yun Liu,
Weixing Shen, and Yiqun Liu. 2024. STARD: A Chinese Statute Retrieval Dataset
Derived from Real-life Queries by Non-professionals. In Findings of the Association
for Computational Linguistics: EMNLP 2024 , Yaser Al-Onaizan, Mohit Bansal, and
Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida,
USA, 10658–10671. https://doi.org/10.18653/v1/2024.findings-emnlp.625
[41] Weihang Su, Xiangsheng Li, Yiqun Liu, Min Zhang, and Shaoping Ma. 2023.
Thuir2 at ntcir-16 session search (ss) task. arXiv preprint arXiv:2307.00250 (2023).
[42] Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Zhijing Wu, and Yiqun
Liu. 2024. Mitigating entity-level hallucination in large language models. In
Proceedings of the 2024 Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval in the Asia Pacific Region . 23–31.
[43] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. Dragin:
Dynamic retrieval augmented generation based on the real-time information
needs of large language models. arXiv preprint arXiv:2403.10081 (2024).
[44] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric Retrieval Augmented
Generation. arXiv preprint arXiv:2501.15915 (2025).
[45] Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou,
and Yiqun Liu. 2024. Unsupervised real-time hallucination detection based on the
internal states of large language models. arXiv preprint arXiv:2403.06448 (2024).
[46] Weihang Su, Changyue Wang, Anzhe Xie, Qingyao Ai, Yiran Hu, and Yiqun
Liu. 2024. LegalAID: A Large Language Model for the Chinese Legal Field.
https://github.com/oneal2000/LegalAID.
[47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models. arXivpreprint arXiv:2302.13971 (2023).
[48] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. 2025. RbFT: Ro-
bust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects.
arXiv preprint arXiv:2501.18365 (2025).
[49] Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun Liu. 2024. Knowledge
Editing through Chain-of-Thought. arXiv preprint arXiv:2412.17727 (2024).
[50] Changyue Wang, Weihang Su, Hu Yiran, Qingyao Ai, Yueyue Wu, Cheng Luo,
Yiqun Liu, Min Zhang, and Shaoping Ma. 2024. LeKUBE: A Legal Knowledge
Update BEnchmark. arXiv preprint arXiv:2407.14192 (2024).
[51] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2. 5
Technical Report. arXiv preprint arXiv:2412.15115 (2024).
[52] Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun
Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, et al .2023. Disc-lawllm:
Fine-tuning large language models for intelligent legal services. arXiv preprint
arXiv:2309.11325 (2023).
[53] Michael Zander. 2015. The law-making process . Bloomsbury Publishing.
[54] ChengXiang Zhai. 2008. Statistical language models for information retrieval.
Synthesis lectures on human language technologies 1, 1 (2008), 1–141.
[55] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma.
2021. Optimizing dense retrieval model training with hard negatives. In Proceed-
ings of the 44th international ACM SIGIR conference on research and development
in information retrieval . 1503–1512.
[56] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav
Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint
arXiv:1904.09675 (2019).
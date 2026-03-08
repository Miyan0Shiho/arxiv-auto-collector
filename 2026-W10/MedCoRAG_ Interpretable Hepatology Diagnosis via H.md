# MedCoRAG: Interpretable Hepatology Diagnosis via Hybrid Evidence Retrieval and Multispecialty Consensus

**Authors**: Zheng Li, Jiayi Xu, Zhikai Hu, Hechang Chen, Lele Cong, Yunyun Wang, Shuchao Pang

**Published**: 2026-03-05 12:58:45

**PDF URL**: [https://arxiv.org/pdf/2603.05129v1](https://arxiv.org/pdf/2603.05129v1)

## Abstract
Diagnosing hepatic diseases accurately and interpretably is critical, yet it remains challenging in real-world clinical settings. Existing AI approaches for clinical diagnosis often lack transparency, structured reasoning, and deployability. Recent efforts have leveraged large language models (LLMs), retrieval-augmented generation (RAG), and multi-agent collaboration. However, these approaches typically retrieve evidence from a single source and fail to support iterative, role-specialized deliberation grounded in structured clinical data. To address this, we propose MedCoRAG (i.e., Medical Collaborative RAG), an end-to-end framework that generates diagnostic hypotheses from standardized abnormal findings and constructs a patient-specific evidence package by jointly retrieving and pruning UMLS knowledge graph paths and clinical guidelines. It then performs Multi-Agent Collaborative Reasoning: a Router Agent dynamically dispatches Specialist Agents based on case complexity; these agents iteratively reason over the evidence and trigger targeted re-retrievals when needed, while a Generalist Agent synthesizes all deliberations into a traceable consensus diagnosis that emulates multidisciplinary consultation. Experimental results on hepatic disease cases from MIMIC-IV show that MedCoRAG outperforms existing methods and closed-source models in both diagnostic performance and reasoning interpretability.

## Full Text


<!-- PDF content starts -->

MedCoRAG: Interpretable Hepatology Diagnosis via Hybrid Evidence Retrieval and
Multispecialty Consensus
Zheng Lia, Jiayi Xua, Zhikai Hua, Hechang Chenb, Lele Congc, Yunyun Wangd,∗, Shuchao Panga,e,∗
aSchool of Cyber Science and Engineering, Nanjing University of Science and Technology, Nanjing, 210094, China
bSchool of Artificial Intelligence, Jilin University, Changchun, 130015, China
cDepartment of Neurology, China-Japan Union Hospital of Jilin University, Changchun, 130033, China
dDepartment of Anesthesiology, China-Japan Union Hospital of Jilin University, Changchun, 130033, China
eSchool of Computing, Macquarie University, Sydney, NSW 2109, Australia
Abstract
Diagnosing hepatic diseases accurately and interpretably is critical, yet it remains challenging in real-world clinical settings. Ex-
isting AI approaches for clinical diagnosis often lack transparency, structured reasoning, and deployability. Recent efforts have
leveraged large language models (LLMs), retrieval-augmented generation (RAG), and multi-agent collaboration. However, these
approaches typically retrieve evidence from a single source and fail to support iterative, role-specialized deliberation grounded in
structured clinical data. To address this, we propose MedCoRAG (i.e., Medical Collaborative RAG), an end-to-end framework
that generates diagnostic hypotheses from standardized abnormal findings and constructs a patient-specific evidence package by
jointly retrieving and pruning UMLS knowledge graph paths and clinical guidelines. It then performs Multi-Agent Collaborative
Reasoning: a Router Agent dynamically dispatches Specialist Agents based on case complexity; these agents iteratively reason over
the evidence and trigger targeted re-retrievals when needed, while a Generalist Agent synthesizes all deliberations into a traceable
consensus diagnosis that emulates multidisciplinary consultation. Experimental results on hepatic disease cases from MIMIC-IV
show that MedCoRAG outperforms existing methods and closed-source models in both diagnostic performance and reasoning
interpretability.
Keywords:Clinical Decision Support, Large Language Models, Retrieval-Augmented Generation, Multiple Agents
1. Introduction
In the era of AI-driven precision medicine, accurate and in-
terpretable diagnosis of hepatic diseases from real-world Elec-
tronic Health Records (EHRs) is vital but challenging. Early
detection, which enables timely intervention, is crucial to pre-
vent irreversible damage and significantly improve outcomes.
However, these conditions often present with vague, overlap-
ping symptoms [1]. This clinical urgency demands not only re-
liable but also transparent methods that translate complex EHR
data into actionable and explainable diagnostic insights.
Large language models have demonstrated impressive ca-
pabilities on general medical benchmarks [2, 3] and offer a
promising foundation for clinical AI. However, when deployed
on real-world EHR data such as MIMIC-IV [4], they face sig-
nificant hurdles in the hepatology context: their knowledge is
static and potentially outdated, sometimes yielding confident
yet incorrect diagnoses [5, 6, 7]. More critically, their reasoning
processes lack traceable, step-by-step justification—making it
∗Corresponding author
Email addresses:lizheng050427@163.com(Zheng Li),
xujiayi041109@163.com(Jiayi Xu),huzhikai1115@163.com(Zhikai
Hu),chenhc@jlu.edu.cn(Hechang Chen),
congll18@mails.jlu.edu.cn(Lele Cong),wangyunyun@jlu.edu.cn
(Yunyun Wang),pangshuchao@njust.edu.cn(Shuchao Pang)difficult to align model outputs with the interpretability stan-
dards required for high-stakes liver disease diagnosis.
Retrieval-augmented generation (RAG) [8] has emerged as
a strategy to ground large language model outputs in external
evidence. While traditional RAG relies on unstructured text, it
struggles with multi-hop clinical inference due to the absence
of explicit medical relationships. Recent approaches integrate
medical knowledge graphs (KGs) [9, 10] to enable structured
reasoning, significantly outperforming classic RAG by leverag-
ing semantic paths between concepts. However, raw KG paths
often contain irrelevant or implausible links, and these methods
still fail to incorporate context-sensitive guidance from clinical
practice guidelines—such as diagnostic criteria or evidence hi-
erarchies. Meanwhile, emerging multi-agent frameworks aim
to emulate multidisciplinary consultation by deploying special-
ized agents that collaboratively debate differentials through it-
erative dialogue [3, 11, 12]. Yet most systems [13, 14] oper-
ate over loosely retrieved or internal knowledge, lacking deep
integration of both KGs and authoritative guidelines, and thus
fall short of evidence-based standards. Moreover, agent activa-
tion is typically static, with the same set of specialists engaged
regardless of case complexity [15, 12], resulting in either re-
dundant deliberation or insufficient expertise. Notably, even
advanced agent-based systems often prioritize end-task accu-
racy over interpretable, clinician-aligned diagnostic reasoning,arXiv:2603.05129v1  [cs.AI]  5 Mar 2026

(1) Traditional Medical RAG
(4) MedCoRAG (Ours)(2) KG-Augmented Medical RAG (3) Static RAG & Multi-Agent Systems
Diagnosis and ReasonsHybrid RAG
Pruned
Knowledge
GraphPruned
Mecica
GuideIterative 
RetrievalCollaborative 
Reasoning
ComplexEHR Data
EHR DataEHR Data EHR Data
Diagnosis
Simple Madical Guide
ReportsDiagnosis
Madical Guide Knowledge GraphDiagnosisFigure 1: Comparative Overview of Medical Diagnostic Reasoning Frame-
works
limiting trust in complex hepatobiliary decisions [16, 17, 18].
To address these gaps, we propose MedCoRAG (i.e., Medical
Collaborative RAG), a hybrid RAG and multi-agent framework
that grounds multidisciplinary clinical reasoning [19] in uni-
fied evidence synthesis. As shown in Figure 1, unlike prior ap-
proaches that rely on static agent teams or limited sources of ex-
ternal evidence, MedCoRAG integrates guideline-constrained
KG pruning with dynamic, complexity-aware specialist dis-
patch. We first transform structured EHR data into coherent
clinical narratives and generate an initial set of diagnostic hy-
potheses. For each hypothesis, the system retrieves clinical
guideline excerpts [20, 21, 22] and UMLS [23] knowledge
graph paths, then prunes the paths using an LLM that evalu-
ates their clinical coherence against the full narrative and guide-
lines, yielding a patient-tailored evidence package for all cases.
A Router Agent assesses case complexity based on the clinical
narrative and abnormal findings. In simple cases, a General-
ist Agent directly synthesizes a diagnosis based on the initial
evidence package. In complex cases, the system dynamically
dispatches relevant specialist agents such as Hepatology or On-
cology based on clinical context; these specialists perform iter-
ative reasoning over the evidence package and trigger targeted
retrievals whenever current evidence is insufficient. The Gen-
eralist Agent ultimately produces a single, traceable consensus
diagnosis through holistic adjudication of all specialist inputs,
deliberation history, and unresolved uncertainties.
We evaluate MedCoRAG on real-world hepatic cases from
MIMIC-IV and show that it generates precise and evidence-
grounded diagnoses. The framework’s dynamic routing and
evidence-pruning mechanisms suppress spurious associations
and redundant deliberation, yielding focused and interpretable
reasoning.
To summarize, our contributions are as follows:
•We propose MedCoRAG, a multi-specialty RAG–agent
framework that dynamically emulates hepatology MDT
consultations by coordinating on-demand specialist agents
in an iterative, evidence-constrained diagnostic loop over
shared, guideline-pruned multi-hop knowledge graphpaths.
•We introduce MDT-aligned hybrid reasoning, a method to
unify pruned KG paths and clinical guideline excerpts into
a single evidential space that is jointly interpreted through
role-specific specialist lenses, yielding interpretable and
hallucination-resistant consensus diagnoses grounded in
real-world clinical practice.
•We conduct comprehensive experiments on hepatic dis-
ease cases from the MIMIC-IV dataset. Experimental re-
sults demonstrate the effectiveness of MedCoRAG, which
achieves high performance across various diagnostic met-
rics.
2. Related Work
Medical Retrieval and Knowledge-Augmented Reasoning.
Standard RAG mitigates LLM hallucinations by grounding re-
sponses in medical literature or EHRs [24], but its similarity-
based retrieval often returns irrelevant passages and fails to
support iterative clinical reasoning [25]. Recent efforts ad-
dress this in complementary ways: MedGraphRAG [26] con-
structs a multi-tier knowledge graph from academic papers,
medical dictionaries and clinical guidelines, using hierarchi-
cal clustering to generate structured tag summaries for coher-
ent retrieval; MedRAG [27] improves diagnostic alignment
by grouping diseases via symptom similarity rather than di-
agnostic codes; KG-Rank [28] boosts answer quality by re-
ranking retrieved passages using KG-derived entity-path rele-
vance—achieving over 18% gain in ROUGE-L without mod-
ifying the LLM; and rationale-guided RAG [29] first gener-
ates a lightweight diagnostic rationale to steer single-step re-
trieval, improving precision without fine-tuning. Despite these
advances, each approach operates in isolation—none jointly in-
tegrates structured KG paths, full clinical guidelines, and adap-
tive retrieval. Our method bridges this gap by jointly retrieving
KG-derived reasoning paths and guideline excerpts, then apply-
ing domain-aware pruning to produce a focused, traceable, and
clinically coherent evidence package.
Multi-Agent Systems for Clinical Collaboration.Early multi-
agent frameworks established role-based clinical collaboration
through zero-shot role playing [11], argumentation-driven ex-
plainability [18], or simulated clinical environments [15, 30].
Recent work shifts toward evidence-grounded and optimized
workflows: ColaCare [14] and LINS [13] coordinate agents
over structured EHRs or citation-backed chains; TxAgent [31]
focuses on dynamic tool composition for therapy planning; and
MedAide [32] fuses intent-aware extractors for multifaceted
reasoning. Notably, MedAgent-Pro [33] introduces a rea-
soning agentic workflow that constructs traceable diagnostic
paths from multimodal inputs and clinical guidelines, while
MMedAgent-RL [34] leverages reinforcement learning to op-
timize agent collaboration policies for improved diagnostic ac-
curacy. Despite these advances, most systems either fix col-
laboration structures or decouple deliberation from a unified,
2

pruned evidence base—limiting adaptability and clinical fi-
delity. Our approach addresses this by dynamically routing spe-
cialists based on abnormal findings and coordinating their RL-
informed deliberation over a shared, guideline-anchored knowl-
edge graph.
3. Methodology
3.1. Overall Architecture
MedCoRAG implements an end-to-end diagnostic workflow
grounded in structured evidence synthesis, as shown in Fig-
ure 2. From the initial set of diagnostic hypotheses, the sys-
tem performs a first-round retrieval of clinical guideline ex-
cerpts and multi-hop knowledge graph paths, which are jointly
pruned using the full clinical narrative to form patient-specific
evidence packages. A Router Agent then assesses case com-
plexity based on the narrative and abnormal findings: for simple
cases, a Generalist Agent directly renders a diagnosis from the
initial evidence; for complex cases, it dynamically dispatches
specialty-specific agents. These specialists iteratively evalu-
ate hypotheses against the shared evidence, triggering agent-
guided re-retrieval when needed. The final diagnosis is pro-
duced by the Generalist through holistic adjudication of all de-
liberations—yielding a single, traceable, and clinically action-
able conclusion.
3.2. Core Components
3.2.1. Abnormal Findings and Preliminary Diagnosis
Abnormal Entity Recognition and Standardization.Given a pa-
tient’s case descriptionC, the system first invokes an LLM to
extract candidate abnormal entities:
Eraw=LLM NER(C),(1)
where LLM NERdenotes a large language model prompted to
identify diagnostic-relevant abnormalities from clinical text.
To align these entities with standardized medical terminol-
ogy, the system queries the knowledge graph UMLS for each
raw entitye∈E raw, obtaining a list of candidate standardized
entities:
S(e)=KMatch(e),(2)
where KMatch(·) denotes knowledge graph entity matching.
The raw entityeand its candidate matchesS(e) are then pre-
sented to the LLM, which selects the best-matching standard-
ized entity—or indicates no match:
estd=LLM align e,S(e)∈S(e)∪{∅}.(3)
The final set of standardized abnormal entities is defined as:
Eabn={e std|e∈E raw,estd,∅}.(4)
This ensures semantic consistency with the underlying
knowledge graph, enabling precise downstream evidence re-
trieval.Direct Generation of Candidate Diagnoses.The system gen-
erates the initial diagnostic hypotheses using both the complete
case descriptionCand the standardized abnormal findingsE abn.
These inputs are formatted into a structured clinical prompt, and
the LLM directly produces a concise list of plausible differen-
tial diagnoses.
Formally, the initial hypothesis list is generated as
Hinitial=LLM hypo(C,E abn).(5)
The output takes the form of a bounded sequence
[d1,d2,...,d K] withK≤K max(Kmax=4) to maintain
diagnostic focus, where eachd iis a standardized disease name.
3.2.2. Hybrid RAG
This module treats each candidate diagnosisd ias an anchor
and retrieves concurrently two complementary evidence types:
(1) authoritative statements from a clinical guideline corpus,
and (2) interpretable reasoning paths from the UMLS knowl-
edge graph.
Clinical Guideline Retrieval and Relevance Filtering.For each
candidate diagnosisd i, a composite query is constructed by
combiningd iwith all abnormal findings inE abn, forming a
semantically enriched representation that captures the clinical
context of the hypothesis. This query drives a two-stage re-
trieval process over a pre-indexed clinical guideline database.
In the first stage, a bi-encoder computes dense embeddings
for queries and guideline segments, and uses cosine similarity
to retrieve the Top-Kmost relevant segments (K=8). In the
second stage, a cross-encoder re-ranks these candidates based
on contextual alignment and selects the Top-Nsegments (N=
4). The resulting set is denoted as:
Gi=GRet(d i,Eabn;N=4),(6)
where GRet(·) denotes the two-stage guideline retrieval
pipeline.
Knowledge Graph Path Retrieval and Guideline-Informed
Pruning.For each pair (d i,ej) withe j∈ E abn, the system
queries the UMLS knowledge graph for semantic paths from
ejtod iwith at most 3 hops:
Pi j=KRet(e j,di;hmax=3),(7)
where KRet(·) denotes knowledge graph path retrieval.
Raw paths may contain irrelevant semantic links. Each path
pis first verbalized by LLM into a natural-language statement
linkinge jtod ivia intermediate concepts. To assess rele-
vance, verbalized paths ford i(denotedP i=S
ej∈EabnPi j) are
batched (8 per batch). An LLM evaluates each batch using
only the full case descriptionCand the top guideline excerpts
Gtop
i⊆G i, producing a binary judgment. Letϕ rel(p;C,Gtop
i)∈
{0,1}indicate whether pathpprovides a clinically coherent and
guideline-supported explanation for the patient’s presentation.
Then:
Pvalid
i=n
p∈P iϕrel(p;C,Gtop
i)=1o
.(8)
3

Diagnosis and
ReasonsElectronic 
Health RecordAbnormal Entities
Router AgentQuery
Complexity
 AssessmentComplex Case
ConfidenceInsufficient
New QueryMulti-Agent Collaborative Reasoning ModuleAbnormal Findings and Preliminary Diagnosis Module Hybrid RAG Module
Mutiple 
RetrievalSufficiencyHepatology
 AgentRadiology
 Agent
Oncology Agent
Simple CaseStandardization
UMLS
Madical Guide
Stance
ConfidenceSufficiencyStanceContext-based
 Pruning
Hypothetical Diagnosis
Hepatology
 AgentRadiology
 Agent
Oncology AgentReportsSufficient
Generalist Agentej
dir1
r2em
di        [e1,e2ek>DenseQueryHop≤3
Top-K [e1,e2ek>
[d1,d2ds>Evidence packet    
Pruned Guideline Excerpts    
[p1p2pm]    [g1g2gh]    
Pruned KG Paths Figure 2: Overall architecture of the MedCoRAG framework, comprising three core components. (1) Abnormal Findings and Preliminary Diagnosis: Abnormal
clinical findings are extracted from the patient narrative and standardized via UMLS to generate a focused set of initial diagnostic hypotheses. (2) Hybrid RAG:
For each hypothesis, the system retrieves clinical guideline excerpts and UMLS knowledge graph paths, then prunes them using the full clinical context to form a
coherent, patient-specific evidence package. (3) Multi-Agent Collaborative Reasoning: A Router Agent assesses case complexity to either activate relevant specialist
agents or delegate simple cases to the Generalist Agent; all agents iteratively reason over the shared evidence, trigger re-retrieval when needed, and converge on an
interpretable consensus diagnosis through the Generalist Agent.
Finally, for eachd i, the system aggregatesPvalid
iandG iinto
a structured evidence packageE(0)
i, which supports subsequent
multi-agent reasoning.
3.2.3. Multi-Agent Collaborative Reasoning
Complexity Assessment.The Router Agent processes the com-
plete patient case descriptionC—including free-text clinical
history, physical examination notes, and narrative laboratory
or imaging reports—together with standardized abnormal find-
ingsE abn. This holistic representation enables the detection of
subtle indicators of diagnostic complexity, such as contradic-
tory findings, multi-organ involvement, atypical presentations,
and diagnostic uncertainty. Based on this integrated context,
the Router applies a narrative-aware complexity discrimination
functionϕ compto yield a binary decision:
c=ϕ comp(C,E abn,H initial),(9)
wherec∈{0,1}. Whenc=0, the case is routed to a General-
ist Agent, which synthesizes a final diagnosis directly from the
pre-retrieved evidence packageE(0)
i, bypassing multi-agent de-
liberation. Whenc=1, the system initiates dynamic specialist
dispatch for collaborative reasoning.
For complex cases (c=1), the system performsdynamic
specialist dispatch.Instead of relying on a fixed specialty set,
the system dynamically selects relevant specialists based on the
clinical context. Specifically, a scheduling functionψidentifiesa contextually appropriate subset of specialists by jointly ana-
lyzing the semantic types of the abnormal findings and the full
clinical narrative:
Ai=ψ(C,E abn,di),(10)
whereA idenotes the set of specialists activated for evaluat-
ing diagnosisd i. For instance, a case describing “jaundice,
elevated creatinine, and maculopapular rash after drug initia-
tion” would activate Hepatology, Nephrology, and Dermatol-
ogy agents. To enable efficient deployment, we employ knowl-
edge distillation to transfer the clinical reasoning capabilities of
a large model (Qwen3-Max [35]) into a student model (Llama-
3.1-8B-Instruct [36]), which powers the specialist agents during
inference.
Evidence-Driven Specialist Reasoning.Each dispatched spe-
cialist agenta∈A ireceives an identical reasoning context:
Context(t)
i=
C,E abn,di,E(t)
i
,(11)
whereE(t)
idenotes the evidence package at iterationt(ini-
tialized att=0), comprising knowledge graph paths and
guideline excerpts retrieved via the hybrid RAG pipeline de-
scribed in Section 3.2.2. Guided by role-specific prompts, each
agent independently assesses candidate diagnosisd i, produc-
ing a stanceo(i)
a∈{S,N,O}(support, neutral, oppose), a confi-
dence scorec(i)
a∈[0,1], and an evidence sufficiency judgment
4

s(i)
a∈{Suf,Ins}, accompanied by a justification explicitly an-
chored to items inE(t)
i.
If any agent reportss(i)
a=Ins, the Coordinator Agent aggre-
gates these signals to decide whether to initiate an additional
retrieval round. Specifically, let
ρ(t)=1
|Ai|X
a∈A iI s(i)
a=Ins(12)
denote the proportion of agents deeming the current evidence
insufficient. Whenρ(t)> τ suff(τsuff=0.5) andt<T max
(Tmax=3), the system launches a targeted secondary retrieval:
agents collaboratively formulate queries that address specific
diagnostic uncertainties, such as requests for imaging charac-
teristics or guideline criteria tied to biomarker thresholds; these
queries are combined withC,E abn,di, and identified knowledge
gaps to construct refined retrieval inputs; the hybrid retrieval
pipeline (Section 3.2.2) is then re-executed to fetch supplemen-
tary knowledge graph paths and guideline excerpts; finally, the
new evidence is merged into an updated packageE(t+1)
ifor the
next reasoning iteration. This closed-loop mechanism allows
retrieval and deliberation to co-evolve until diagnostic confi-
dence stabilizes or the iteration limit is reached.
Consensus Formation and Confidence Calibration.At each
iterationt, a Generalist Agent synthesizes all specialist in-
puts—including stances, confidence scores, evidence-based
justifications, and sufficiency judgments—to generate an in-
terim consensus report for each candidate diagnosisd i. The cal-
ibrated confidence scores(t)
i, which quantifies diagnostic plau-
sibility, is computed as:
s(t)
i=1
|Ai|X
a∈A iI o(i)
a=S,(13)
whereI(·) denotes the indicator function. The process termi-
nates early ifs(t)
i> τ high(τhigh=0.9), signaling strong collec-
tive support ford i.
Final Diagnosis Selection and Output.Rather than selecting
the diagnosis with the highest calibrated score alone, the Gen-
eralist Agent conducts a holistic final adjudication. It in-
tegrates the complete deliberation history—including evolv-
ing evidence packages, inter-agent agreements and disagree-
ments, unresolved uncertainties, and patient-specific context
(C,E abn)—to produce a single, clinically coherent final diag-
nosis:
dfinal=F finaln
(di,s(T)
i,Reporti)o
di∈H initial
,(14)
whereF finaldenotes the Generalist’s adjudication function. This
decision is accompanied by a comprehensive, traceable con-
sensus report that synthesizes supporting and contradicting evi-
dence across all iterations, highlights key clinical uncertainties,
and recommends actionable next-step investigations.
The whole algorithm of MedCoRAG is shown in Algorithm
1.Algorithm 1MedCoRAG Diagnostic Workflow
Require:case descriptionC
Ensure:Final diagnosisd finalwith justification
1:E abn=n
LLM align e,KMatch(e)|e∈LLM NER(C)o
2:H=LLM hypo(C,E abn) {Top-Kfocused hypotheses (K≤
Kmax)}
3:foreachd i∈Hdo
4:G i=GRet(d i,Eabn;N=4)
5:Pvalid
i=n
p∈KRet(e j,di;hmax=3)LLM prune(p;C,Gtop
i)=1o
{for alle j∈E abn}
6:E(0)
i←(G i,Pvalid
i)
7:end for
8:ifϕ comp(C,E abn,H)=0then
9:d final←GeneralistAgent(C,E abn,E(0)
i)
10:returnd finalwith justification
11:end if
12:fort=1 toT maxdo
13:Specialistsa∈A i=ψ(C,E abn,di) outputo(i)
a∈{S,N,O}
ands(i)
a∈{Suf,Ins}
14:Computeρ(t)=1
|Ai|P
aI(s(i)
a=Ins)
15:ifρ(t)≤τ suffthen
16:break
17:else
18:UpdateE(t+1)
ivia agent-proposed queries and hybrid
retrieval
19:end if
20:end for
21:d final=F final {(di,Reporti)}i
22:returnd finalwith traceable consensus report
4. Experiments
We conduct a comprehensive evaluation of MedCoRAG on
a real-world hepatic disease diagnosis task to assess the effec-
tiveness of our abnormal-entity-driven reasoning framework,
multi-source knowledge integration, and multi-agent collabo-
ration mechanism.
4.1. Dataset
We curate a clinical dataset from the public MIMIC-IV
database [4], focusing on patients diagnosed with one of 13
common hepatic diseases, whose standardized abbreviations
are listed in Table 1. To reflect the chronic and progressive
nature of hepatic conditions, we retain all hospital admissions
per patient, thus we reconstruct longitudinal medical histories.
All data are fully de-identified.
Since MIMIC-IV primarily provides structured tabular
records lacking the narrative context, we synthesize realistic,
context-rich clinical narratives from each patient’s longitudi-
nal timeline using LLM. These narratives are then formatted as
medical question–answering pairs. The final dataset contains
3470 QA samples, split into training and test sets at a 7:3 ratio
during the distillation phase, stratified by disease category to
ensure balanced representation across all 13 classes.
5

Table 1: Disease abbreviations
Hepatic Disease Abbreviation
Hepatitis B HBV
Primary biliary cholangitis PBC
Secondary liver cancer SLC
Liver cyst LCyst
Hepatoblastoma HB
Liver cirrhosis LC
Hepatocellular carcinoma HCC
Hepatic hemangioma HH
Liver failure LF
Autoimmune hepatitis AIH
Drug-induced liver injury DILI
Non-alcoholic steatohepatitis NASH
Rupture and bleeding of esophagogastric varices EGVB
4.2. Evaluation Metrics
We evaluate diagnostic performance using four standard met-
rics: Recall, Precision, F1-score, and F0.5-score. All metrics
are reported as weighted averages across the 13 hepatic disease
classes to account for class imbalance.
4.3. Baseline Models
We compare MedCoRAG against a comprehensive set
of baselines spanning model scale and reasoning architec-
ture, all evaluated on the same test set using identical
clinical narratives. This includes medical-domain models
with up to 8B parameters—Qwen3-Medical-GRPO-4B [37],
OpenBioLLM-Llama3-8B [38], Bio-Medical-Llama3-8B [39],
and Llama3-Med42-8B [40]; large proprietary models includ-
ing DeepSeek-V3.1-Think [41], Gemini-2.5-Pro [42], GLM-
4.6 [43], and GPT-4o [44]; medium-sized models ranging from
14B to 32B parameters, including DeepSeek-R1-Distill-Qwen-
32B [45], GPT-OSS-20B [46], Gemma3-27B [47], Qwen-
QWQ-32B [48], and Phi-4-14B [49]; lightweight models un-
der 7B parameters, namely ChatGLM3-6B [50] and Mistral-
7B [51]; recent RAG & multi-agent diagnostic frameworks
—ColaCare [14], MedAgent-Pro [33], and MedAide [32]—
implemented using the same base model as MedCoRAG
(Llama-3.1-8B-Instruct [36]). All methods are evaluated under
the same protocol.
4.4. Implementation Details
For retrieval-augmented reasoning, we integrate two comple-
mentary external knowledge sources: (1) a structured biomed-
ical knowledge graph built upon the UMLS, and (2) an un-
structured corpus of 38 authoritative clinical guidelines on hep-
atic diseases issued by major professional societies such as
AASLD, EASL, APASL. For clinical guidelines, we perform
dense retrieval using Qwen3-Embedding-8B [52], with em-
beddings indexed in Milvus [53], followed by re-ranking with
Qwen3-Reranker-8B [52].
We use Llama-3.1-8B-Instruct as the backbone LLM. To
facilitate deployment, we distill the reasoning capability of
Qwen3-Max [35] into Llama-3.1-8B-Instruct. Specifically, theteacher model simulates specialist agent behavior within the
MedCoRAG framework to generate training data. The student
model is fine-tuned via supervised learning with LoRA [54] for
three epochs, using a cosine-decayed learning rate initialized at
5×10−5, an effective batch size of 8 via gradient accumulation,
and a maximum sequence length of 11,000 tokens, all inbf16
precision on a single A800 GPU.
4.5. Main Results
We conduct diagnostic classification across 13 hepatic dis-
ease categories on a standardized clinical test set. The over-
all performance of MedCoRAG is shown in Table 2. Among
all evaluated methods, MedCoRAG achieves the best weighted
Precision, Recall, F1-score, and F0.5-score. It performs bet-
ter than specialized medical models with up to 8B parameters,
large proprietary language models, medium- and small-scale
general-purpose models, as well as recent multi-agent diag-
nostic frameworks. These results indicate that MedCoRAG’s
approach—combining structured evidence synthesis with dy-
namic agent collaboration—can support accurate and reliable
clinical diagnosis across diverse model scales and reasoning
strategies.
5. Analysis
5.1. Diagnosis Accuracy and Misclassification Analysis
Table 3 presents per-disease diagnostic metrics for Med-
CoRAG, revealing consistently strong performance across a
range of hepatic conditions. The model achieves high precision
and recall for diseases with distinct clinical or radiological sig-
natures, including hepatic hemangioma, liver cyst, secondary
liver cancer, and drug-induced liver injury. The model achieves
perfect precision in predicting acute events such as rupture and
bleeding of esophagogastric varices, indicating high confidence
in critical diagnoses when they are issued. The confusion ma-
trix in Figure 3 shows that misclassifications primarily occur
among clinically related entities—such as cirrhosis-associated
complications and cholestatic disorders—reflecting known di-
agnostic similarities in hepatology. These results demonstrate
that MedCoRAG aligns its reasoning with established clinical
patterns while maintaining robust accuracy across diverse liver
diseases.
5.2. Diagnostic Complexity and Reasoning Patterns
We analyze how MedCoRAG adapts to varying diagnostic
demands by examining clinical complexity and reasoning depth
across diseases. As shown in Figure 4, conditions like PBC
and LF present with the highest numbers of abnormal entities,
reflecting intricate clinical profiles, while HH and HB involve
markedly simpler presentations. Correspondingly, Figure 5 re-
veals that diagnoses such as LCyst, LF, DILI, and SLC rely on
longer knowledge graph reasoning paths, indicating active in-
tegration of multi-hop evidence from guidelines and structured
medical knowledge. Together, these results illustrate that Med-
CoRAG tailors its inference process to the inherent complexity
of each case, leveraging deeper reasoning where clinical ambi-
guity is greatest.
6

Table 2: Comprehensive diagnostic performance (%) across model categories. The best result in each metric isbolded.
Category Model Precision Recall F1-score F0.5-score
Medical (≤8B)Qwen3-Medical-GRPO-4B [37] 69.68 58.93 61.07 59.06
OpenBioLLM-Llama3-8B [38] 62.15 51.48 54.21 52.24
Bio-Medical-Llama3-8B [39] 65.94 41.83 47.43 43.36
Llama3-Med42-8B [40] 64.57 61.41 60.28 60.51
General large (>100B)DeepSeek-V3.1-Think [41] 79.59 76.98 77.61 77.05
Gemini-2.5-Pro [42] 80.31 76.70 77.28 76.71
GLM-4.6 [43] 80.76 75.36 76.55 75.53
GPT-4o [44] 74.33 70.58 69.98 69.90
General medium (14–32B)DeepSeek-R1-Distill-Qwen-32B [45] 78.49 74.59 74.90 74.35
GPT-OSS-20B [46] 75.19 69.25 70.75 69.56
Gemma3-27B [47] 73.89 60.84 65.92 62.61
Qwen-QWQ-32B [48] 66.60 53.96 55.78 54.02
Phi-4-14B [49] 62.70 54.44 55.68 54.26
General small (≤7B)ChatGLM3-6B [50] 56.57 40.97 41.97 40.89
Mistral-7B [51] 60.86 40.02 41.52 39.12
Agent FrameworksColaCare [14] 78.01 72.66 73.35 72.53
MedAgent-Pro [33] 76.01 70.58 70.33 69.97
MedAide [32] 77.98 73.23 74.10 73.26
OursMedCoRAG81.32 79.18 79.12 78.99
Table 3: Per-disease diagnostic metrics (%) of MedCoRAG.
Disease Recall Precision F1-score F0.5-score
HBV 52.00 76.47 61.90 55.56
PBC 55.56 20.83 30.30 41.67
SLC 85.16 92.31 88.59 86.50
LCyst 87.76 93.48 90.53 88.84
HB 85.71 85.71 85.71 85.71
LC 67.44 62.59 64.93 66.41
HCC 90.45 74.19 81.52 86.65
HH 94.62 94.62 94.62 94.62
LF 15.79 75.00 26.09 18.75
AIH 69.49 54.67 61.19 65.92
DILI 82.00 88.74 85.24 83.27
NASH 69.35 87.76 77.48 72.39
EGVB 16.67 100.00 28.57 20.00
5.3. Ablation Study
We assess the contribution of each component in Med-
CoRAG to diagnostic performance (see Table 4) by evaluating
several ablated variants: (1) w/o CG (without Clinical Guide in-
tegration), where agents do not access structured clinical guides
during reasoning; (2) w/o MA (without Multi-Agent deliber-
ation), where diagnosis is generated by a single agent with-
out collaborative discussion; (3) w/o KG (without Knowledge
Graph grounding), where disease-entity relationships from the
medical knowledge graph are excluded; (4) w/o TD (without
Teacher Distillation), where the student model operates with-
out guidance from the teacher-based reasoning process; as wellTable 4: Ablation study results (%). Best in each column isbolded. CG: Clin-
ical Guide integration; MA: Multi-Agent deliberation; KG: Knowledge Graph
grounding; TD: Teacher Distillation.
Variant Precision Recall F1-score F0.5-score
w/o CG 78.08 72.75 73.43 72.62
w/o MA 76.30 69.72 69.70 69.14
w/o KG 77.96 73.14 73.81 73.03
w/o TD 78.20 73.71 74.48 73.73
w/o KG & CG 74.47 68.10 67.86 67.37
w/o TD & MA 76.83 68.74 69.93 68.62
w/o KG & CG & MA 69.30 57.50 55.32 55.62
MedCoRAG81.32 79.18 79.12 78.99
as combined ablations, including w/o KG & CG, w/o TD &
MA, and w/o KG & CG & MA.
The results show that removing any single component leads
to a performance drop across all metrics, with the most substan-
tial decline observed when both knowledge sources (KG and
CG) and multi-agent collaboration (MA) are absent (F1 drops
to 55.32). Notably, disabling multi-agent deliberation (w/o
MA) reduces F1 by 9.42 points compared to the full model,
indicating that agent interaction plays a critical role in refining
diagnostic hypotheses. Similarly, omitting clinical guides (w/o
CG) or the knowledge graph (w/o KG) results in F1 scores be-
low 74, suggesting that structured external evidence is essential
for accurate reasoning. The relatively stronger performance of
w/o TD (F1: 74.48) implies that while teacher-guided distilla-
tion further enhances performance, the core gains stem from the
integration of authoritative knowledge and collaborative agent
7

Figure 3: Confusion matrix of MedCoRAG on 13 hepatic disease classes.
dynamics.
5.4. Deployment Cost and Efficiency
The one-time cost to construct the teacher-generated train-
ing dataset using Qwen3-Max is $24.53, with no recurring ex-
penses thereafter. At inference time, MedCoRAG exhibits pre-
dictable latency: cases requiring multi-agent collaborative rea-
soning take an average of 33.36 seconds, while simpler cases
handled by a single generalist agent complete in just 9.95 sec-
onds.
5.5. Case Study: How Architecture Enables Expert-Level Di-
agnosis
We present a representative case that highlights the advan-
tages of MedCoRAG’s modular design. A 48-year-old female
presents with persistent fatigue, pruritus, and jaundice. Lab
findings reveal markedly elevated ALP (340 U/L), GGT (280
U/L), and IgM (3.8 g/L), with normal IgG levels. Abdominal
ultrasound shows no biliary obstruction but mild hepatomegaly.
She denies alcohol use, recent medication changes, or known
viral hepatitis exposure.
This presentation is diagnostically challenging due to overlap
among PBC, AIH, and DILI. A standard language model might
favor AIH or DILI based on fatigue and elevated transaminases,
overlooking the cholestatic pattern.
In contrast, MedCoRAG leverages its full architecture for
precise reasoning. First, abnormal-entity detection identi-
fies ALP_elevated, GGT_elevated, IgM_elevated, and pruritus,
which collectively trigger dynamic routing to the Autoimmune
Hepatology Agent while suppressing irrelevant specialists such
as Virology or Oncology.
Figure 4: Average number of abnormal entities per case across different hepatic
diseases. Higher values indicate more complex clinical presentations.
The activated agent then retrieves a clinical guideline ex-
cerpt from EASL stating that “persistent cholestasis with iso-
lated IgM elevation in middle-aged women is highly sugges-
tive of PBC, even in the absence of anti-mitochondrial antibod-
ies.” Concurrently, the knowledge graph yields a coherent 2-
hop path: pruritus→PBC→IgM_elevated, linking symptoms
to serological markers through established disease semantics.
Multi-agent collaboration further refines the diagnosis: the
Immunology Agent notes that normal IgG levels argue against
typical AIH, while the Hepatology Generalist cross-validates
the cholestatic enzyme profile and absence of drug exposure
to rule out DILI. The system converges on a high-confidence
diagnosis of PBC, accompanied by a traceable rationale and a
recommendation for anti-mitochondrial antibody testing.
This case illustrates how MedCoRAG’s components syner-
gize: abnormality-driven routing ensures clinical relevance; hy-
brid retrieval grounds reasoning in both guidelines and struc-
tured knowledge; and multi-agent deliberation enables nuanced
differential analysis—all without reliance on invasive findings.
This integrated workflow underpins the framework’s ability to
resolve ambiguous cases that mimic expert clinical judgment.
5.6. Limitations and Future Work
MedCoRAG demonstrates strong diagnostic performance
and interpretable reasoning on hepatic cases from MIMIC-IV ,
yet it has several limitations. The current implementation pro-
cesses only a single clinical snapshot, lacking modeling of lon-
gitudinal signals such as lab trends or imaging evolution. It also
depends on UMLS-aligned entities and static guidelines, which
can be sensitive to ambiguity in real-world clinical notes. More-
over, all evaluations are retrospective and have not been vali-
dated in live clinical workflows. To address these issues, future
8

Figure 5: Average number of hops in knowledge graph paths used during diag-
nosis. Higher values reflect greater reasoning complexity.
work will incorporate temporal EHR modeling for time-aware
diagnosis, extend the agent framework to broader clinical do-
mains beyond hepatology, enhance robustness to unstructured
text through improved natural language understanding, develop
lightweight mechanisms for efficient deployment, and conduct
prospective studies with clinical partners to assess real-world
impact on decision support and EHR integration.
6. Conclusion
We propose MedCoRAG, a hybrid retrieval-augmented gen-
eration and multi-agent framework for interpretable hepatic dis-
ease diagnosis. The method constructs a patient-specific evi-
dence package by jointly retrieving and pruning paths from a
medical knowledge graph and excerpts from clinical practice
guidelines. A router agent dynamically activates relevant spe-
cialist agents based on abnormal clinical findings, enabling iter-
ative, role-aware deliberation over the shared evidence packet.
Consensus is formed through holistic adjudication by a gen-
eralist agent, yielding a single, traceable diagnosis grounded
in both structured knowledge and expert guidance. Evaluated
on real-world hepatic cases from MIMIC-IV , MedCoRAG out-
performs a wide range of baselines, demonstrating superior di-
agnostic performance and reasoning explainability. This work
represents a step toward more transparent, evidence-grounded,
and clinically aligned AI for trustworthy medical decision sup-
port.
CRediT authorship contribution statement
Zheng Li: Conceptualization, Data curation, Formal analy-
sis, Investigation, Methodology, Project administration, Valida-tion, Visualization, Writing – original draft.Jiayi Xu: Investi-
gation, Writing – review & editing.Zhikai Hu: Investigation,
Writing – review & editing.Hechang Chen: Supervision, Val-
idation, Writing – review & editing.Lele Cong: Supervision,
Validation, Writing – review & editing.Yunyun Wang: For-
mal analysis, Project administration, Supervision, Validation,
Writing – review & editing.Shuchao Pang: Formal analysis,
Funding acquisition, Project administration, Supervision, Writ-
ing – review & editing.
Declaration of competing interest
The authors declare that they have no known competing fi-
nancial interests or personal relationships that could have ap-
peared to influence the work reported in this paper.
Data and code availability
The MIMIC-IV database is publicly available through Phy-
sioNet at:https://physionet.org/content/mimiciv/3.
1. The source code will be publicly available after acceptance.
Acknowledgements
This work is supported by the National Natural Sci-
ence Foundation of China (Grant No.62206128), National
Key Research and Development Program of China (Grant
No.2023YFB2703900)
References
[1] C. Gan, Y . Yuan, H. Shen, J. Gao, X. Kong, Z. Che,
Y . Guo, H. Wang, E. Dong, J. Xiao, Liver diseases: epi-
demiology, causes, trends and predictions, Signal Trans-
duction and Targeted Therapy 10 (1) (2025) 33.
[2] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn,
M. Amin, L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewis,
et al., Toward expert-level medical question answering
with large language models, Nature Medicine 31 (3)
(2025) 943–950.
[3] X. Tang, D. Shao, J. Sohn, J. Chen, J. Zhang, J. Xi-
ang, F. Wu, Y . Zhao, C. Wu, W. Shi, et al., Medagents-
bench: Benchmarking thinking models and agent frame-
works for complex medical reasoning, arXiv preprint
arXiv:2503.07459 (2025).
[4] A. E. Johnson, L. Bulgarelli, L. Shen, A. Gayles,
A. Shammout, S. Horng, T. J. Pollard, S. Hao, B. Moody,
B. Gow, et al., Mimic-iv, a freely accessible electronic
health record dataset, Scientific data 10 (1) (2023) 1.
[5] M. Griot, C. Hemptinne, J. Vanderdonckt, D. Yuksel,
Large language models lack essential metacognition for
reliable medical reasoning, Nature communications 16 (1)
(2025) 642.
9

[6] P. Hager, F. Jungmann, R. Holland, K. Bhagat,
I. Hubrecht, M. Knauer, J. Vielhauer, M. Makowski,
R. Braren, G. Kaissis, et al., Evaluation and mitigation
of the limitations of large language models in clinical
decision-making, Nature medicine 30 (9) (2024) 2613–
2622.
[7] Y . Zhu, J. Gao, Z. Wang, W. Liao, X. Zheng, L. Liang,
M. O. Bernabeu, Y . Wang, L. Yu, C. Pan, et al., Clini-
crealm: Re-evaluating large language models with con-
ventional machine learning for non-generative clinical
prediction tasks, arXiv preprint arXiv:2407.18525 (2024).
[8] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, H. Wang, H. Wang, Retrieval-augmented genera-
tion for large language models: A survey, arXiv preprint
arXiv:2312.10997 2 (1) (2023).
[9] Y . Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, L. Zhao, Grag:
Graph retrieval-augmented generation (2025).arXiv:
2405.16506.
URLhttps://arxiv.org/abs/2405.16506
[10] J. Wu, J. Zhu, Y . Qi, J. Chen, M. Xu, F. Menolascina,
V . Grau, Medical graph rag: Towards safe medical large
language model via graph retrieval-augmented generation,
arXiv preprint arXiv:2408.04187 (2024).
[11] X. Tang, A. Zou, Z. Zhang, Z. Li, Y . Zhao, X. Zhang,
A. Cohan, M. Gerstein, Medagents: Large language mod-
els as collaborators for zero-shot medical reasoning, in:
Findings of the Association for Computational Linguis-
tics: ACL 2024, 2024, pp. 599–621.
[12] Y . Kim, C. Park, H. Jeong, Y . S. Chan, X. Xu, D. McDuff,
H. Lee, M. Ghassemi, C. Breazeal, H. W. Park, Mdagents:
An adaptive collaboration of llms for medical decision-
making, Advances in Neural Information Processing Sys-
tems 37 (2024) 79410–79452.
[13] S. Wang, F. Zhao, D. Bu, Y . Lu, M. Gong, H. Liu, Z. Yang,
X. Zeng, Z. Yuan, B. Wan, et al., Lins: A general medi-
cal q&a framework for enhancing the quality and credibil-
ity of llm-generated responses, Nature Communications
16 (1) (2025) 9076.
[14] Z. Wang, Y . Zhu, H. Zhao, X. Zheng, D. Sui, T. Wang,
W. Tang, Y . Wang, E. Harrison, C. Pan, et al., Colacare:
Enhancing electronic health record modeling through
large language model-driven multi-agent collaboration,
in: Proceedings of the ACM on Web Conference 2025,
2025, pp. 2250–2261.
[15] S. Schmidgall, R. Ziaei, C. Harris, E. Reis, J. Jopling,
M. Moor, Agentclinic: a multimodal agent benchmark
to evaluate ai in simulated clinical environments, arXiv
preprint arXiv:2405.07960 (2024).
[16] T. Hellingman, M. de Swart, J. Joosten, M. Meijerink,
J. de Vries, J. de Waard, A. van Zweeden, B. Zonderhuis,G. Kazemier, The value of a dedicated multidisciplinary
expert panel to assess treatment strategy in patients suffer-
ing from colorectal cancer liver metastases, Surgical On-
cology 35 (2020) 412–417.
[17] E. N. Smith, M. R. Bashir, A. Fung, B. D. Cash, M. Dixon,
E. M. Hecht, B. M. McGuire, A. A. Pillai, G. K. Russo,
R. T. Shroff, et al., Acr appropriateness criteria®stag-
ing and follow-up of primary liver cancer, Journal of the
American College of Radiology 22 (11) (2025) S699–
S712.
[18] S. Hong, L. Xiao, X. Zhang, J. Chen, Argmed-agents: ex-
plainable clinical decision reasoning with llm disscusion
via argumentation schemes, in: 2024 IEEE International
Conference on Bioinformatics and Biomedicine (BIBM),
IEEE, 2024, pp. 5486–5493.
[19] B. Pillay, A. C. Wootten, H. Crowe, N. Corcoran, B. Tran,
P. Bowden, J. Crowe, A. J. Costello, The impact of mul-
tidisciplinary team meetings on patient assessment, man-
agement and outcomes in oncology settings: a system-
atic review of the literature, Cancer treatment reviews 42
(2016) 56–72.
[20] E. A. F. T. S. O. T. Liver, et al., Easl clinical practice
guidelines on tips, Journal of hepatology 83 (1) (2025)
177–210.
[21] T. H. Taddei, D. B. Brown, M. Yarchoan, M. Mendiratta-
Lala, J. M. Llovet, Critical update: Aasld practice guid-
ance on prevention, diagnosis, and treatment of hepato-
cellular carcinoma, Hepatology (2025) 10–1097.
[22] J. P. Arab, L. A. Díaz, J. Rehm, G. Im, M. Arrese, P. S.
Kamath, M. R. Lucey, J. Mellinger, M. Thiele, M. Thursz,
et al., Metabolic dysfunction and alcohol-related liver dis-
ease (metald): Position statement by an expert panel on
alcohol-related liver disease, Journal of hepatology 82 (4)
(2025) 744–756.
[23] O. Bodenreider, The unified medical language system
(umls): integrating biomedical terminology, Nucleic
acids research 32 Database issue (2004) D267–70.
URLhttps://api.semanticscholar.org/
CorpusID:205228801
[24] L. M. Amugongo, P. Mascheroni, S. Brooks, S. Doering,
J. Seidel, Retrieval augmented generation for large lan-
guage models in healthcare: A systematic review, PLOS
Digital Health 4 (6) (2025) e0000877.
[25] G. Xiong, Q. Jin, Z. Lu, A. Zhang, Benchmarking
retrieval-augmented generation for medicine, in: Findings
of the Association for Computational Linguistics ACL
2024, 2024, pp. 6233–6251.
[26] J. Wu, J. Zhu, Y . Qi, J. Chen, M. Xu, F. Meno-
lascina, Y . Jin, V . Grau, Medical graph RAG: Evidence-
based medical large language model via graph retrieval-
augmented generation, in: W. Che, J. Nabende,
10

E. Shutova, M. T. Pilehvar (Eds.), Proceedings of the
63rd Annual Meeting of the Association for Computa-
tional Linguistics (V olume 1: Long Papers), Association
for Computational Linguistics, Vienna, Austria, 2025, pp.
28443–28467.doi:10.18653/v1/2025.acl-long.
1381.
URLhttps://aclanthology.org/2025.acl-long.
1381/
[27] X. Zhao, S. Liu, S.-Y . Yang, C. Miao, Medrag: En-
hancing retrieval-augmented generation with knowledge
graph-elicited reasoning for healthcare copilot, in: Pro-
ceedings of the ACM on Web Conference 2025, 2025, pp.
4442–4457.
[28] R. Yang, H. Liu, E. Marrese-Taylor, Q. Zeng, Y . Ke, W. Li,
L. Cheng, Q. Chen, J. Caverlee, Y . Matsuo, et al., Kg-
rank: Enhancing large language models for medical qa
with knowledge graphs and ranking techniques, in: Pro-
ceedings of the 23rd Workshop on Biomedical Natural
Language Processing, 2024, pp. 155–166.
[29] J. Sohn, Y . Park, C. Yoon, S. Park, H. Hwang, M. Sung,
H. Kim, J. Kang, Rationale-guided retrieval augmented
generation for medical question answering, in: Pro-
ceedings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (V olume 1:
Long Papers), 2025, pp. 12739–12753.
[30] J. Li, Y . Lai, W. Li, J. Ren, M. Zhang, X. Kang, S. Wang,
P. Li, Y .-Q. Zhang, W. Ma, et al., Agent hospital: A sim-
ulacrum of hospital with evolvable medical agents, arXiv
preprint arXiv:2405.02957 (2024).
[31] S. Gao, R. Zhu, Z. Kong, A. Noori, X. Su, C. Ginder,
T. Tsiligkaridis, M. Zitnik, Txagent: An ai agent for thera-
peutic reasoning across a universe of tools, arXiv preprint
arXiv:2503.10970 (2025).
[32] D. Yang, J. Wei, M. Li, J. Liu, L. Liu, M. Hu, J. He, Y . Ju,
W. Zhou, Y . Liu, et al., Medaide: Information fusion and
anatomy of medical intents via llm-based agent collabora-
tion, Information Fusion (2025) 103743.
[33] Z. Wang, J. Wu, L. Cai, C. H. Low, X. Yang, Q. Li,
Y . Jin, Medagent-pro: Towards evidence-based multi-
modal medical diagnosis via reasoning agentic workflow,
arXiv preprint arXiv:2503.18968 (2025).
[34] P. Xia, J. Wang, Y . Peng, K. Zeng, X. Wu, X. Tang,
H. Zhu, Y . Li, S. Liu, Y . Lu, et al., Mmedagent-rl: Opti-
mizing multi-agent collaboration for multimodal medical
reasoning, arXiv preprint arXiv:2506.00555 (2025).
[35] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng,
B. Yu, C. Gao, C. Huang, C. Lv, et al., Qwen3 technical
report, arXiv preprint arXiv:2505.09388 (2025).[36] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle,
A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan,
et al., The llama 3 herd of models, arXiv e-prints (2024)
arXiv–2407.
[37] T. Z. XIONG, Qwen3_medical_grpo: A medical domain
llm fine-tuned with group relative policy optimization,
accessed: November 15, 2025 (Jun. 2025).
URLhttps://huggingface.co/lastmass/Qwen3_
Medical_GRPO
[38] M. S. Ankit Pal, Openbiollms: Advancing open-
source large language models for healthcare and
life sciences,https://huggingface.co/aaditya/
OpenBioLLM-Llama3-8B, accessed: November 15, 2025
(2024).
[39] ContactDoctor, Contactdoctor-bio-medical: A high-
performance biomedical language model, accessed:
November 15, 2025 (2024).
URLhttps://huggingface.co/ContactDoctor/
Bio-Medical-Llama-3-8B
[40] C. Christophe, P. K. Kanithi, T. Raha, S. Khan,
M. A. Pimentel, Med42-v2: A suite of clinical llms,
https://arxiv.org/abs/2408.06142, accessed: November
15, 2025 (2024).arXiv:arXiv:2408.06142.
[41] DeepSeek-AI, Deepseek-v3 technical report (2024).
arXiv:2412.19437.
URLhttps://arxiv.org/abs/2412.19437
[42] G. Gemini Team, Gemini 2.5: Pushing the frontier
with advanced reasoning, multimodality, long context,
and next generation agentic capabilities, arXiv preprint
arXiv:2507.06261 (2025).
[43] A. Zeng, X. Lv, Q. Zheng, Z. Hou, B. Chen, C. Xie,
C. Wang, D. Yin, H. Zeng, J. Zhang, et al., Glm-4.5:
Agentic, reasoning, and coding (arc) foundation models,
arXiv preprint arXiv:2508.06471 (2025).
[44] O. (2024), Gpt-4o system card (2024).arXiv:2410.
21276.
URLhttps://arxiv.org/abs/2410.21276
[45] DeepSeek-AI, Deepseek-r1: Incentivizing reasoning ca-
pability in llms via reinforcement learning (2025).
arXiv:2501.12948.
URLhttps://arxiv.org/abs/2501.12948
[46] OpenAI, gpt-oss-120b, gpt-oss-20b model card (2025).
arXiv:2508.10925.
URLhttps://arxiv.org/abs/2508.10925
[47] G. D. Gemma Team, Gemma 3 technical report (2025).
arXiv:2503.19786.
URLhttps://arxiv.org/abs/2503.19786
[48] Q. Team, Qwq-32b: Embracing the power of reinforce-
ment learning (March 2025).
URLhttps://qwenlm.github.io/blog/qwq-32b/
11

[49] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan,
S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi,
P. Kauffmann, et al., Phi-4 technical report, arXiv preprint
arXiv:2412.08905 (2024).
[50] T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin,
D. Zhang, D. Rojas, G. Feng, H. Zhao, et al., Chatglm: A
family of large language models from glm-130b to glm-4
all tools, arXiv preprint arXiv:2406.12793 (2024).
[51] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford,
D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel,
G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux,
P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, W. E.
Sayed, Mistral 7b (2023).arXiv:2310.06825.
URLhttps://arxiv.org/abs/2310.06825
[52] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang,
P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou,
Qwen3 embedding: Advancing text embedding and
reranking through foundation models, arXiv preprint
arXiv:2506.05176 (2025).
[53] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang,
X. Guo, C. Li, X. Xu, et al., Milvus: A purpose-built vec-
tor data management system, in: Proceedings of the 2021
international conference on management of data, 2021,
pp. 2614–2627.
[54] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang,
L. Wang, W. Chen, et al., Lora: Low-rank adaptation of
large language models., ICLR 1 (2) (2022) 3.
12
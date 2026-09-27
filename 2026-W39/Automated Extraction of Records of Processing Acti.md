# Automated Extraction of Records of Processing Activities (RoPA) Using Hybrid RAG and Locally Deployed Large Language Models

**Authors**: To Duy Hinh, Nguyen Le Quoc Anh, Phan Van Tri, Khuong Nguyen-An

**Published**: 2026-09-23 04:55:53

**PDF URL**: [https://arxiv.org/pdf/2609.27359v1](https://arxiv.org/pdf/2609.27359v1)

## Abstract
Vietnam's Personal Data Protection Law (Law No. 91/2025/QH15) and Decree No. 356/2025/ND-CP, effective January 1, 2026, require organizations to establish and maintain Records of Processing Activities (RoPA). Manual RoPA preparation is labor-intensive, while cloud-hosted large language models (LLMs) may conflict with data-sovereignty requirements. We propose RoPA Manager, a system for automated RoPA information extraction using hybrid retrieval that combines lexical ranking over tsvector, dense-vector search, Reciprocal Rank Fusion (RRF), and locally deployed LLMs. We introduce a Vietnamese RoPA benchmark with 32 organizations, 77 processing activities, 12 field groups, and 4,338 reference values. Evaluation is reported at three distinct levels. The automated scorer, tested on perturbed data without invoking an LLM, achieved F1 = 0.9493 [0.9436, 0.9548]; this measures scorer robustness rather than end-to-end extraction accuracy. End-to-end extraction achieved token coverage of 50.04-55.25% against the reference labels. Two independent experts reviewed 1,558 reference values (35.9% of the benchmark), found no incorrect values, and achieved 99.68% agreement with PABAK = 0.9936. Value-level precision was not measured. Across 32 paired scenarios on a 24 GB GPU, locally deployed Qwen3.5-27B-GPTQ-Int4 showed no statistically significant difference from cloud-based DeepSeek-V4-Flash (difference 0.20 percentage points in favor of DeepSeek, 95% CI [-0.93, 1.32], p = 0.72), while Gemma-4-31B performed significantly worse (p < 0.01).

## Full Text


<!-- PDF content starts -->

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
Automated Extraction of Records of 
Processing Activities (RoPA) Using Hybrid 
RAG and Locally Deployed Large Language 
Models  
To Duy Hinh1, Nguyen Le Quoc Anh1, Phan Van Tri1 and Khuong Nguyen -An2,3,* 
1 Academy of Cryptography Techniques, Ho Chi Minh City Campus, Government Cipher Committee, 
Vietnam  
2 Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT),  
268 Ly Thuong Kiet Street, Dien Hong Ward, Ho Chi Minh City, Vietnam  
3 Vietnam National University Ho Chi Minh City, Linh Xuan Ward, Ho Chi Minh City, Vietnam  
Emails: tdhinhit@gmail.com, haniz.cons@gmail.com, phanvantri@actvn.edu.vn, nakhuong@hcmut.edu.vn  
 
 
Abstract - Vietnam's Personal Data Protection Law 
(Law 91/2025/QH15) and Decree 356/2025/ND -CP, 
effective January 1, 2026, require organizations to 
create and maintain Records of Processing Activities 
(RoPA) under Article 31. Preparing RoPA manually 
requir es considerable staff time, while the strongest 
large language models (LLMs) run on foreign cloud 
services, which conflicts with data sovereignty 
requirements. This study proposes RoPA Manager, a 
system that extracts RoPA information by combining 
hybrid re trieval (lexical ranking over a tsvector index, 
dense vector search, and Reciprocal Rank Fusion, 
RRF) with locally deployed LLMs. The main 
contribution is the first Vietnamese RoPA benchmark, 
with 32 organizations, 77 processing activities, 12 field 
groups , and 4,338 reference values. Results are 
reported separately at three evaluation levels. The 
automated scorer, evaluated on noisy data without 
calling an LLM, reaches F1 = 0.9493 [0.9436; 0.9548]; 
this value measures the quality of the scorer, not the 
accuracy of end -to-end extraction. End -to-end 
extraction with LLM calls reaches 50.04 -55.25% token 
coverage against the reference labels. Two independent 
experts reviewed 1,558 reference values (35.9% of the 
benchmark); no value was rated Wrong, and 
agreement  was 99.68% with PABAK = 0.9936. Value -
level precision has not yet been measured. Paired tests 
on 32 scenarios show no significant difference between 
the local Qwen3.5 -27B-GPTQ -Int4 model on a 24 GB 
GPU and the cloud -based DeepSeek -V4-Flash (a 0.20 
percent age-point difference in favor of DeepSeek, 95% 
CI [ -0.93; +1.32], p = 0.72), while Gemma -4-31B 
performs significantly worse (p < 0.01).  Keywords - RoPA; personal data protection; RAG; hybrid 
retrieval; locally deployed large language models; 
Vietnamese benchmark.  
I. INTRODUCTION  
Vietnam's Personal Data Protection Law (Law 
91/2025/QH15) [2], effective from January 1, 2026, 
together with Decree 356/2025/ND -CP [1], which 
provides detailed guidance, requires every 
organization that processes personal data to create and 
maintain Record s of Processing Activities (RoPA) 
under Article 31 of Law 91/2025/QH15. Preparing 
RoPA manually requires strong legal expertise, takes 
considerable staff time, and is prone to errors, 
especially in organizations that run dozens of 
processing activities acr oss many business areas.  
Large language models (LLMs) combined with 
retrieval -augmented generation (RAG) [3] make it 
possible to automate the extraction of information 
from business documents into a standard RoPA 
structure. However, using cloud LLM application 
programming interfac es (APIs) to process personally 
identifiable information (PII) conflicts with the 
principle of data sovereignty. A solution is therefore 
needed that uses LLMs without sending personal data 
outside the organization's internal infrastructure.  
This paper presents RoPA Manager with three 
contributions. First, it provides the first benchmark 
for Vietnamese RoPA extraction, with 32 simulated 
organizations, 77 processing activities, 12 field 
groups, and 4,338 reference values, of which 35.9% 
were re viewed independently by two experts, together 
with an automated scorer. Second, it introduces a 
three -level reporting scheme that separates scorer 

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
performance, end -to-end coverage, and review of the 
reference labels. It is intended to avoid mixing three 
different kinds of metrics, a common problem in 
published evaluations of RAG systems. Third, it 
empirically examines the feasibility of local 
deploym ent: two open -source models completed all 
32 scenarios on a 24 GB GPU, and one of them 
reached coverage in the same range as the reference 
cloud model. The model -independent architecture is a 
design choice that serves the third contribution, not a 
separate  scientific contribution.  
RoPA directly supports the confidentiality, 
integrity, and availability of personal data, while the 
system itself must defend against LLM -specific 
threats such as prompt injection and data leakage 
through cloud models. RoPA Manager therefore 
builds securit y principles based on the STRIDE 
framework and the Zero Trust model into the system 
from the design stage, as described in Section III.D.  
II. RELATED  WORK  
Governance, risk, and compliance (GRC) tools 
such as OneTrust [6] and the ISO/IEC 29134:2023 
guidelines [7] support RoPA management but still 
require a data protection officer (DPO) to enter data 
manually. NLP -based compliance automation, such 
as checking the completeness of privacy policies 
against GDPR requirements [19], has been shown to 
be feasible but is tied to the European legal 
framework and to English. Legal information 
extraction benchmarks such as LexGLUE [20] also 
cover English only. Vietnamese pretrained models 
such as PhoBERT [8] have not been applied to the 
RoPA task. Work on domain -specific RAG [4], [5], 
[12] shows that retrieval quality and hallucination 
remain open challenges. A search of Google Scholar, 
IEEE Xplore, and the ACM Digital Lib rary for 2020 -
2026, combining "RoPA" or "records of processing 
activities" with "Vietnamese", "LLM", and "RAG", 
found no work on automated RoPA extraction for 
Vietnamese. To the best of the authors' knowledge, 
this is the gap that this paper addresses.  
At the technical level, the Transformer 
architecture with self -attention [14] is the basis of 
modern LLMs. LLMs are used instead of rule -based 
methods because of the nature of the input: the source 
documents for RoPA are internal procedures, 
contracts, and  impact assessment records. These 
documents are semi -structured and their layouts differ 
across organizations, so regular expressions or fixed 
extraction rules cannot cover them. This is the 
fundamental difference from tasks with a fixed input 
schema, wher e a rule engine is often enough. Hybrid 
retrieval is chosen because Vietnamese legal terms 
need exact keyword matching, in the spirit of the 
probabilistic relevance framework [21], in addition to semantic similarity. RRF [22] merges the two 
rankings using rank positions only, so it does not 
depend on the score scale of each retrieval branch. 
The multilingual -e5-base embedding model [23] is 
used instead of PhoBERT [8] because the system 
needs Vietna mese -English support and direct 
compatibility with pgvector [15].  
III. SYSTEM  DESIGN  AND  METHODS  
A. Overall architecture  
RoPA Manager uses a layered architecture with 
four layers: Presentation (single -page web 
application), Business logic (coordination of the 
extraction workflow), Artificial Intelligence (model -
independent RAG + LLM layer), and Data (relational 
database and vector database). A key design point is 
that the AI layer is endpoint -independent, so the 
system can switch between cloud and local LLMs 
through configuration alone.  
Architecture choices follow two goals: keeping 
personal data inside the internal infrastructure and 
minimizing the number of components to operate. 
The vector database uses pgvector [15] inside 
PostgreSQL instead of a separate dedicated vector 
system. The local inference server uses vLLM, whose 
paged attention reduces GPU memory use and whose 
OpenAI -compatible interface allows the model 
provider to be changed through endpoint 
configuration alone.  
B. Hybrid retrieval and extraction workflow  
The workflow has six stages (Figure 1): receiving 
and preprocessing documents (PDF/DOCX); 
chunking the text and creating embeddings; hybrid 
retrieval; calling the LLM for extraction; post -
processing and validation against the schema; and 
storing the result  for review. Semantic chunking uses 
a break threshold of 0.62. The sparse branch uses the 
ts_rank_cd lexical ranking function on a PostgreSQL 
tsvector index (simple configuration), while the dense 
branch uses cosine search through an HNSW index 
[13]. Each branch over -retrieves 30 candidates before 
fusion. The two branches are merged by a weighted 
variant of RRF [22] with k = 60 and a weight of 0.4 
for the lexical branch; both values are defaults and 
have not been systematically tuned. The top 8 chunks 
are k ept as context, and the evaluated version does not 
use cross -encoder reranking. The LLM receives a 
fixed prompt, identified by its SHA -256 checksum, 
together with the context. The output is constrained to 
JSON and validated against a Pydantic schema. 
Figur e 1 and Table I report the main parameters for 
reproducibility.  

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
 
Figure 1.  Three evaluation levels. Level 2 runs the full 
workflow with LLM calls; level 1 follows the dashed branch and 
passes noise -modified reference labels to the scorer without an 
LLM; level 3 reviews the reference labels themselves.  
Table I.  RETRIEVAL CONFIGURATION 
AND REPRODUCIBILITY PARAMETERS  
Component  Configuration used in experiments  
Vector index  pgvector / PostgreSQL 15; HNSW m = 
16, ef_construction = 64  
Generation  temperature = 0.2; max_tokens = 1800  
Output constraint  JSON + Pydantic 2.5.3 schema  
Local inference  vLLM, single 24 GB GPU  
Platform / seed  Python 3.11, FastAPI 0.109.0; seed 
20260705  
C. RoPA field groups  
RoPA information is organized into 12 
independent field groups based on Article 31 of Law 
91/2025/QH15 [2] and the detailed guidance in 
Decree 356/2025/ND -CP [1]. The groups are 
BASIC_INFO; PURPOSES; LEGAL_BASIS; 
DATA_SUBJECTS for data subjects and data ty pes; 
RECIPIENTS_TRANSFERS for recipients and 
cross -border transfers; RETENTION for retention 
periods and deletion methods; 
TECHNICAL_MEASURES and 
STORAGE_SECURITY; CONSENT_DETAILS for 
collecting and withdrawing consent; RISK_DPIA for 
impact assessment and risk level; REGULATORY 
for reporting to the regulator; and 
DOCUMENTATION. The 12 reporting groups have 
a many -to-many mapping to 12 query groups in the 
extraction layer: PURPOSES, DATA_SUBJECTS, 
and RECIPIENTS_TRANSFERS each combine two queries; STORAGE_SECURITY and 
TECHNICAL_MEASURES share one query; and 
DOCUMENTATION has no query of its own but is 
derived from the description field and the cited 
sources. The reporting group is the unit of both the 
reference labels and the measurements.  
D. Information security design  
Under the OWASP classification for LLM 
applications [18], prompt injection is the top risk for 
systems based on large language models. RoPA 
Manager applies the Zero Trust model of NIST SP 
800-207 [17] together with STRIDE threat analysis 
[16] to identify b oth traditional attack vectors 
(spoofing, data tampering, denial of service) and AI -
specific threats (prompt injection, data leakage 
through cloud models, and hallucinated false 
information). Three defense layers are built directly 
into the extraction work flow: Pydantic schema 
validation rejects any output that does not follow the 
JSON structure; the model's built -in safety 
mechanism refuses instructions outside the assigned 
task; and the JSON output constraint greatly limits 
information leakage through fre e text. The 
effectiveness of this design is tested experimentally in 
Section V.  
IV. EXPERIMENTAL  SETUP  
A. Infrastructure and models  
The system is evaluated with three LLMs: 
DeepSeek -V4-Flash through a cloud API as the 
reference, and Qwen3.5 -27B-GPTQ -Int4 (4 -bit 
quantized) and Gemma -4-31B (bf16), both deployed 
locally through vLLM on a single 24 GB GPU. The 
application layer runs on an AWS EC2 t3.large 
instance (2 vCPU, 8 GiB RAM, no GPU) because the 
embedding model runs efficiently on a CPU at this 
experimental scale. The experiments call the cloud 
API only with fully synthetic data, so they do not 
violate the data sovereignty principle ; with real data, 
the system must switch to local mode. All three 
models use the same source code, the same prompt 
template (with the same SHA -256 checksum), and the 
same scorer.  
B. Benchmark dataset  
The benchmark is built from controlled synthetic 
data in four steps: selecting seven sectors with heavy 
RoPA obligations (banking/finance, e -commerce, 
telecommunications, healthcare, education, 
information technology/software as a service, and 
aviation); d esigning 32 simulated organizations with 
1-5 processing activities each; writing unstructured 
Vietnamese source documents that also contain 
English terms; and labeling the 12 field groups 


Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
according to a standardized guide with a conservative 
rule, checked by a JSON schema validator and logical 
cross -checks. The documents and reference labels 
were written manually by the authors and were not 
generated by an LLM, which avoids the risk of mode l 
feedback loops. The difficulty distribution is 21 easy, 
33 medium, and 23 hard processing activities. Three 
measurement units are used in parallel: 924 group 
entities (77 x 12), 4,338 leaf values in identity mode, 
and 3,965 leaf values scored in noisy/an onymized 
mode after 373 fields (8.6%) were removed.  
C. Measurement method  
The study separates three evaluation levels 
because they use different inputs and measure 
different objects (Figure 1). Level 1 tests the scorer in 
identity mode and in noisy/anonymized mode; both 
modes compare the reference labels with versions of 
them mo dified in a controlled way, so no LLM is 
called. Level 2 runs the full workflow with LLM calls 
on 32 scenarios. The score of each field group is the 
share of reference -label tokens that are matched, so it 
is a recall -like measure at the token level. Becaus e the 
32 scenarios are paired observations, model 
comparisons use a paired t -test with a Wilcoxon 
check. Level 3 reviews the reliability of the reference 
labels themselves. The 95% confidence intervals are 
computed by nonparametric bootstrap (B = 10,000, 
seed = 20260712, N = 3,965); the KPI thresholds in 
Table II were set by the authors. The level 3 review 
sample contains 1,558 reference values (35.9% of the 
benchmark), drawn from 15 of the 32 scenarios and 
41 of the 77 processing activities, and covers all  7 
sectors and all 12 field groups. Samples were taken in 
scenario order within each sector block, not by 
stratified random sampling, so the sampling rate 
ranges from 33% to 100% across sectors. Two experts 
scored separate copies independently, without sha ring 
results and without knowing which model produced 
the labels, on a three -point scale: Correct / Partly 
correct / Wrong.  
D. Reproducibility  
Before each comparison, every component that 
can affect the result is frozen at a specific version: 
dataset v2.0, scorer v1.3, source code at git commit 
c38e18c for the cloud configuration and 045547f for 
the two local models, and the shared retrieval 
parameters and prompt template, with integrity 
checked by SHA -256. Each run records a run ID, 
provider, quantization method, temperature, seed, and 
execution timestamp. The source code, the 
benchmark, and the review records of the two experts 
are publicly avai lable at 
https://github.com/tdhinhit/RoPA -demo.  V. RESULTS  AND  DISCUSSION  
At the scorer level, identity mode gives 1.0 on 
every metric over 4,338 leaf values, which confirms 
that the scoring logic has no technical error. In 
noisy/anonymized mode with 3,965 leaf values, F1 = 
0.9493 [0.9436; 0.9548], Precision = 0.9470, and 
Recall  = 0.9517, with confidence intervals narrower 
than 0.02. Because this mode does not call an LLM, 
these values only show that the scorer tolerates 
imperfect (noisy) data; they do not measure extraction 
performance. Six of the seven level 1 metrics meet 
their thresholds. The exception is the correct refusal 
rate, measured on 795 fields whose reference label is 
null, which reaches only 0.7874 against a threshold of 
0.80 (Table II).  
Table II.  AUTOMATED SCORER RESULTS 
(NO LLM) AGAINST KPI THRESHOLDS  
Metric  Value (95% CI)  Threshold  Status  
Level 1 - scorer, noisy mode (N = 3,965, no LLM)  
F1  0.9493 [0.9436; 
0.9548]  >= 0.85  PASS  
Accuracy  0.9188 [0.9100; 
0.9271]  >= 0.85  PASS  
Precision  0.9470 [0.9390; 
0.9547]  >= 0.85  PASS  
Recall  0.9517 [0.9441; 
0.9591]  >= 0.80  PASS  
MCC [11]  0.7448  >= 0.55  PASS  
Hallucination 
(noisy data)  0.0426  <= 0.05  PASS  
Correct refusal 
rate (n = 795)  0.7874  >= 0.80  FAIL  
Analysis by the 12 field groups (Table III) shows 
that PURPOSES, DATA_SUBJECTS, and 
DOCUMENTATION reach almost perfect F1 scores. 
RETENTION is the lowest at 0.712, followed by 
STORAGE_SECURITY at 0.876 and 
REGULATORY at 0.886; across sectors, F1 stays 
within a narrow range of 0.939 -0.960. The simulated 
error pattern contains 169 false positives: 55% (93) 
are values inserted by the noise generator, mainly in 
RETENTION (46), RISK_DPIA (23), and 
STORAGE_SECURITY (23), and 45% (76) fall in 
pairs of groups with overlapping meanings. Of the 
153 false negatives, most are in BASIC_INFO (44), 
LEGAL_BASIS (23), and REGULATORY (18). 
Because these errors come from artificial noise rather 
than model output, this pattern is only a predictive 
hypothesis about the field gro ups where an LLM is 
likely to hallucinate or confuse values in operation; 
testing it on real level 2 output is future work.  

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
Table III.  SCORER RESULTS BY 12 FIELD 
GROUPS (NOISY MODE, NO LLM)  
Field group  TP FP FN P R F1 
PURPOSES  174 0 0 1.000  1.000  1.000  
DATA_SUBJECTS  566 0 8 1.000  0.986  0.993  
DOCUMENTATION  200 0 5 1.000  0.976  0.988  
BASIC_INFO  572 0 44 1.000  0.929  0.963  
TECHNICAL_MEASURES  303 24 0 0.927  1.000  0.962  
RECIPIENTS_TRANSFERS  202 18 0 0.918  1.000  0.957  
LEGAL_BASIS  235 0 23 1.000  0.911  0.953  
CONSENT_DETAILS  209 18 13 0.921  0.941  0.931  
RISK_DPIA  214 23 17 0.903  0.926  0.915  
REGULATORY  136 17 18 0.889  0.883  0.886  
STORAGE_SECURITY  138 23 16 0.857  0.896  0.876  
RETENTION  68 46 9 0.597  0.883  0.712  
At the end -to-end extraction level, Table IV 
compares the three models on the same dataset, 
source code, and prompt template. All three complete 
32/32 scenarios, which confirms that the model -
independent architecture works as designed. Token 
coverage is 55 .25% for DeepSeek -V4-Flash, 55.06% 
for the local Qwen3.5 -27B, and 50.04% for Gemma -
4-31B. Because the 32 scenarios are paired 
observations, the study uses a paired t -test with a 
Wilcoxon check. The difference between DeepSeek 
and Qwen is only +0.20 percent age points (95% CI [ -
0.93; +1.32], t(31) = 0.36, p = 0.72; Wilcoxon p = 
0.61). No significant difference is found, and the 
confidence interval also excludes differences larger 
than about 1.3 percentage points, so this result is 
evidence of equivalence with in about 1.3 percentage 
points, not only a lack of evidence for a difference. In 
contrast, Gemma is 5.21 percentage points below 
DeepSeek ([+2.03; +8.40], p = 0.0022) and 5.02 
percentage points below Qwen ([+2.10; +7.94], p = 
0.0014). The gap lies mainly i n LEGAL_BASIS, 
where Gemma reaches only 41.6 -56.1% coverage in 
four sectors, compared with 83.5 -100% for the other 
two models. The tests rely on the variation across 
scenarios within a single run.  Table IV.  COMPARISON OF THREE 
MODELS ON THE SAME DATASET AND 
PROMPT  
Criterion  DeepSeek -
V4-Flash  Qwen3.5 -
27B-Int4 Gemma -4-
31B 
Deployment  Cloud API  Local vLLM  Local vLLM  
Parameters  N/A (black 
box) 27B (4 -bit) 31B (bf16)  
Data location 
(design)  PII leaves 
infrastructure  PII internal  PII internal  
Completed 
scenarios  32/32  32/32  32/32  
Coverage (1 
run)  55.25%  55.06%  50.04%  
Standard 
deviation  4.43 4.63 9.30 
Chunking quality directly affects the retrieval 
stage. The operational workflow uses semantic 
chunking (Figure 1), with an average of 42 chunks per 
document. A separate control run uses recursive 
character -based chunking, which produces only 6.9 
chunks per  document on average. Across 32 
documents, both methods keep 100% of chunks 
within the 512 -token limit of the embedding model 
and do not break sentences; the semantic chunks also 
have high internal cohesion (cosine 0.786 -0.832). 
One consequence deserves at tention: with the policy 
of keeping 8 chunks as context, the current semantic 
setting puts only about 19% of each document into the 
context of each query, while the recursive setting 
would include almost the whole document. This is a 
significant competing hypothesis for the level 2 token 
coverage of 50.04 -55.25%, and an ablation study is 
needed to test it.  
For the retrieval stage, the study has not measured 
Recall@K, MRR, or NDCG, because no query -chunk 
relevance benchmark has been built yet. Without this 
separate measurement, the paper does not attribute the 
low coverage to either the retrieval stage or the  
generation stage.  
Regarding the reliability of the reference labels, 
no value was rated Wrong. The Correct rates of the 
two experts are 99.81% and 99.74%, and inter -expert 
agreement is 99.68% (1,553/1,558), with 5 
disagreements. All five concern the same issue: 
whether Arti cles 13 and 14 of Law 91/2025/QH15 
must be cited in cases of automated credit scoring, 
exam proctoring by face recognition, and AI -based 
staff assessment. This shows that the hardest part of 
the task is legal reasoning rather than text extraction. 
These ca ses are also rated hard, so extraction 
difficulty and labeling difficulty tend to go together. 
Combined with the high share of hard activities in the 
review sample (46.3% versus 29.9% in the full 
benchmark), this indicates that the sample is not 
biased tow ard easy cases. Because 99.7% of the labels 
fall in the same class, raw Cohen's kappa is only 0.28, 

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
which is the known prevalence paradox. The study 
therefore uses PABAK = 0.9936 [10] as the main 
agreement measure and reports raw kappa for 
transparency. Two limits should be noted: a high 
PABAK only shows that the two experts are 
consistent with each othe r, and both experts come 
from the same business ecosystem.  
For information security, the study uses two types 
of evaluation. First, a review of the operating 
configuration checks 12 HTTP response -header 
settings of the reverse proxy and gives 4 PASS, 6 
WARN, and 2 FAIL results. The two FAIL items are 
a missing HST S header and an API endpoint that does 
not require authentication; both must be fixed before 
operation. Second, an attack -resistance test shows 
that the system blocks all 8 prompt -injection 
scenarios: direct instruction injection, administrator 
role-play, forced system -prompt disclosure, indirect 
injection through a document field, personal -data 
extraction, JSON -schema override, multi -turn context 
poisoning, and Unicode encoding. This result comes 
from the three defense layers described in Section 
III.D. Wi th n = 8 and a single run, it is only a proof of 
concept and does not support a strong claim about 
attack resistance.  
The evaluation framework has three levels with 
different scopes. Level 1 evaluates the scorer, level 2 
measures token coverage against the reference labels 
in a recall -like way, and level 3 evaluates the 
reliability of the reference labels. None of the lev els 
measures value -level precision on real model output, 
that is, the share of generated values that are correct 
or the share of blank fields that are correctly left 
blank; level 1 measures these two quantities only for 
the scorer, without LLM calls. All c onclusions in this 
paper are therefore limited to technical feasibility and 
coverage.  
VI. CONCLUSION  AND  FUTURE  WORK  
This study sets a technical baseline for applying 
LLM + RAG to automate compliance with Vietnam's 
Personal Data Protection Law, and contributes the 
first Vietnamese RoPA benchmark and a three -level 
reporting scheme. As a cost reference, a 24 GB GPU 
costs a bout VND 50 -65 million, a full -time DPO 
about VND 180 -300 million per year, and a 
commercial GRC package about USD 10,000 -50,000 
per year. In order of priority, future work includes 
measuring value -level content accuracy on real drafts, 
with stratified sam pling, two independent reviewers, 
three labels (correct / wrong / extra generated value), 
and precision and recall reported by field group; 
repeating the multi -model experiments at least three 
times; building a relevance benchmark to measure 
Recall@K, MRR,  and NDCG, and running an 
ablation study for hybrid retrieval [5]; and measuring the calibration of model -generated confidence scores 
on real outputs.  
The main limitations are: (1) all data are synthetic 
and have not been tested on real RoPA records; (2) 
value -level precision has not been measured; (3) level 
1 metrics evaluate the scorer without LLM calls, so 
the error pattern by field group is only a pr edictive 
hypothesis; (4) the paired tests use a single run; (5) no 
ablation study has measured the separate contribution 
of RRF, and Recall@K, MRR, and NDCG have not 
been measured; (6) the reference -label review covers 
35.9% of the benchmark, samples scena rios in order 
rather than at random, and uses n = 2 experts; and (7) 
the calibration of model -generated confidence scores 
on real output has not been evaluated [9]. These 
limitations define the contribution as a technical 
baseline; the study does not claim  full operational 
legal compliance.  
ACKNOWLEDGMENTS  
The authors thank two independent compliance 
and legal experts for their single -blind review of 
1,558 reference values, which provides a reliable 
basis for all measurements in this study. The study 
was self -funded by the authors and received no 
funding fro m any organization or company. The 
authors have no commercial interest in the described 
system and no financial relationship with the model 
providers or compliance management tools 
mentioned in this paper, and declare no conflict of 
interest. Khuong Nguyen -An thanks Ho Chi Minh 
City University of Technology (HCMUT), VNU -
HCM, for supporting this research.  
 
 
REFERENCES  
[1] Government of the Socialist Republic of Vietnam, "Decree 
356/2025/ND -CP providing detailed rules for several articles 
of the Personal Data Protection Law," Hanoi, 2025. [in 
Vietnamese].  
[2] National Assembly of the Socialist Republic of Vietnam, 
"Personal Data Protection Law (Law 91/2025/QH15)," 
Hanoi: National Political Publishing House, 2025. [in 
Vietnamese].  
[3] P. Lewis, E. Perez, A. Piktus, et al., "Retrieval -augmented 
generation for knowledge -intensive NLP tasks," in 
Advances in Neural Information Processing Systems, vol. 
33, 2020, pp. 9459 -9474.  
[4] J. Chen et al., "Benchmarking large language models in 
retrieval -augmented generation," in Proc. AAAI Conf. 
Artificial Intelligence, 2024, pp. 17754 -17762.  
[5] S. Es et al., "RAGAS: Automated evaluation of retrieval -
augmented generation," in Proc. EACL (System 
Demonstrations), 2024, pp. 150 -158. 
[6] Gartner, "Market guide for privacy management tools," 
Gartner Research, 2025.  

Manuscript accepted for publication in The Proceedings of  t he 29th National Conference on Selected Issues of  
 Information and Communication Technology  (VNICT 2026)  - Hanoi, 7 -8/11/2026  
[7] ISO/IEC 29134:2023, "Information technology - Security 
techniques - Guidelines for privacy impact assessment," 
2023.  
[8] D. Q. Nguyen and A. T. Nguyen, "PhoBERT: Pre -trained 
language models for Vietnamese," in Findings of ACL: 
EMNLP, 2020, pp. 1037 -1042.  
[9] C. Guo et al., "On calibration of modern neural networks," 
in Proc. ICML, 2017, pp. 1321 -1330.  
[10] J. Byrt et al., "Bias, prevalence and kappa," J. Clin. 
Epidemiol., vol. 46, no. 5, pp. 423 -429, 1993.  
[11] B. W. Matthews, "Comparison of the predicted and observed 
secondary structure of T4 phage lysozyme," Biochim. 
Biophys. Acta, vol. 405, no. 2, pp. 442 -451, 1975.  
[12] S. Barnett et al., "Seven failure points when engineering a 
retrieval -augmented generation system," in Proc. 
IEEE/ACM Int. Conf. AI Engineering (CAIN), 2024, pp. 
194-199. 
[13] Y. A. Malkov and D. A. Yashunin, "Efficient and robust 
approximate nearest neighbor search using hierarchical 
navigable small world graphs," IEEE Trans. Pattern Anal. 
Mach. Intell., vol. 42, no. 4, pp. 824 -836, 2020.  
[14] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all 
you need," in Advances in Neural Information Processing 
Systems, vol. 30, 2017, pp. 5998 -6008.  
[15] A. Kane, "pgvector: Open -source vector similarity search for 
PostgreSQL," GitHub repository, 2023.  
[16] A. Shostack, Threat Modeling: Designing for Security. 
Indianapolis: John Wiley & Sons, 2014.  
[17] S. Rose et al., Zero Trust Architecture, NIST Special 
Publication 800 -207, 2020.  
[18] OWASP Foundation, "OWASP Top 10 for Large Language 
Model Applications," Version 1.1, 2025.  
[19] D. Torre et al., "An AI -assisted approach for checking the 
completeness of privacy policies against GDPR," in Proc. 
IEEE Int. Requirements Eng. Conf. (RE), 2020, pp. 136 -146. 
[20] I. Chalkidis et al., "LexGLUE: A benchmark dataset for legal 
language understanding in English," in Proc. ACL, 2022, pp. 
4310 -4330.  
[21] S. Robertson and H. Zaragoza, "The probabilistic relevance 
framework: BM25 and beyond," Found. Trends Inf. Retr., 
vol. 3, no. 4, pp. 333 -389, 2009.  
[22] G. V. Cormack et al., "Reciprocal rank fusion outperforms 
Condorcet and individual rank learning methods," in Proc. 
SIGIR, 2009, pp. 758 -759. 
[23] L. Wang, N. Yang, X. Huang, et al., "Multilingual E5 text 
embeddings: A technical report," arXiv preprint 
arXiv:2402.05672, 2024.

Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
Tự động hóa trích xu ất Hồ sơ ho ạt động xử 
lý dữ liệu cá nhân (RoPA) b ằng RAG lai và 
Mô hình ngôn ng ữ lớn triển khai t ại chỗ 
Tô Duy Hinh1, Nguy ễn Lê Qu ốc Anh1, Phan Văn Tr ị1 và Nguy ễn An Khương2,3,(✉) 
1 Học viện Kỹ thuật Mật mã, Cơ s ở TP. H ồ Chí Minh, Ban Cơ y ếu Chính ph ủ, Việt Nam  
2 Khoa Khoa h ọc và K ỹ thuật Máy tính, Trư ờng Đ ại học Bách khoa (HCMUT),  
268 Lý Thư ờng Ki ệt, Phư ờng Diên H ồng, TP. H ồ Chí Minh, Vi ệt Nam  
3 Đại học Qu ốc gia TP. H ồ Chí Minh, Phư ờng Linh Xuân, TP. H ồ Chí Minh, Vi ệt Nam  
Emails: tdhinhit@gmail.com, haniz.cons@gmail.com, phanvantri@actvn.edu.vn, nakhuong@hcmut.edu.vn  
 
 
Abstract  - Luật Bảo vệ dữ liệu cá nhân (Lu ật 
91/2025/QH15) và Ngh ị định 356/2025/NĐ -CP (hi ệu lực 
01/01/2026) bu ộc tổ chức lập và duy trì H ồ sơ ho ạt động 
xử lý dữ liệu cá nhân (Records of Processing Activities, 
RoPA) theo Đi ều 31. Vi ệc lập RoPA th ủ công t ốn nhi ều 
nhân l ực, trong khi các Mô hình ngôn ng ữ lớn (Large 
Language Model, LLM) m ạnh nh ất vận hành trên đám 
mây nư ớc ngoài, mâu thu ẫn với yêu c ầu chủ quyền dữ 
liệu. Nghiên c ứu đề xuất RoPA Manager, h ệ thống trích 
xuất thông tin RoPA b ằng truy xu ất lai k ết hợp xếp 
hạng từ vựng trên ch ỉ mục tsvector, vector m ật độ cao 
và dung h ợp th ứ hạng tương h ỗ (Reciprocal Rank 
Fusion, RRF) v ới LLM tri ển khai t ại chỗ. Đóng góp 
chính là b ộ kiểm chu ẩn RoPA ti ếng Vi ệt đầu tiên g ồm 
32 tổ chức, 77 ho ạt động xử lý, 12 nhóm trư ờng và 4338 
giá tr ị tham chi ếu. Kết quả được báo cáo tách b ạch theo 
ba tầng đo. B ộ chấm đi ểm tự động, đo trên d ữ liệu làm 
nhiễu và không g ọi LLM, đ ạt F1 = 0.9493 [0.9436; 
0.9548]; đây là ch ất lượng c ủa bộ chấm đi ểm, không 
phải độ chính xác c ủa hệ trích xu ất đầu-cuối. Trích xu ất 
đầu-cuối có g ọi LLM cho đ ộ phủ token so v ới nhãn 
chuẩn 50.04 -55.25%. Hai chuyên gia đ ộc lập thẩm định 
1558 giá tr ị nhãn chu ẩn (35.9% b ộ kiểm chu ẩn), không 
giá tr ị nào sai, đ ồng thu ận 99.68% v ới PABAK = 
0.9936. Nghiên c ứu chưa đo đ ộ chính xác ở cấp giá tr ị 
theo hư ớng precision. Ki ểm định ghép c ặp trên 32 k ịch 
bản cho th ấy mô hình t ại chỗ Qwen3.5 -27B-GPTQ -Int4 
không khác bi ệt so v ới mô hình đám mây DeepSeek -V4-
Flash (chênh l ệch 0.20 đi ểm nghiêng v ề DeepSeek, CI 
95% [ -0.93; +1.32], p = 0.72) trên GPU  24 GB, còn 
Gemma -4-31B th ấp hơn có ý nghĩa (p < 0.01).  
Keywords  - RoPA; bảo vệ dữ liệu cá nhân; RAG; truy 
xuất lai; mô hình ngôn ng ữ lớn triển khai tại chỗ; bộ 
kiểm chuẩn tiếng Việt. I. GIỚI THIỆU 
Luật Bảo vệ dữ liệu cá nhân (Lu ật 91/2025/QH15) 
[2] có hi ệu lực từ 01/01/2026, cùng Ngh ị định 
356/2025/NĐ -CP [1] hư ớng dẫn chi ti ết, yêu c ầu mọi 
tổ chức xử lý dữ liệu cá nhân l ập và duy trì H ồ sơ ho ạt 
động xử lý (RoPA) theo Đi ều 31 Lu ật 91/2025/QH15. 
Việc lập RoPA th ủ công đòi h ỏi chuyên môn pháp lý 
sâu, t ốn nhi ều nhân l ực và d ễ sai sót, nh ất là v ới tổ 
chức vận hành hàng ch ục hoạt động xử lý trải rộng 
nhiều lĩnh v ực. 
Mô hình ngôn ng ữ lớn (LLM) k ết hợp kiến trúc 
Tạo sinh tăng cư ờng bằng truy xu ất (RAG) [3] cho 
phép t ự động hóa trích xu ất thông tin t ừ tài liệu nghi ệp 
vụ sang c ấu trúc RoPA chu ẩn. Tuy nhiên, dùng LLM 
qua giao di ện lập trình ứng dụng (API) đám mây đ ể 
xử lý dữ liệu nhận dạng cá nhân (PII) mâu thu ẫn với 
nguyên t ắc chủ quyền dữ liệu. Do đó c ần giải pháp t ận 
dụng LLM mà d ữ liệu cá nhân không r ời khỏi hạ tầng 
nội bộ. 
Bài báo trình bày RoPA Manager v ới ba đóng góp. 
Thứ nhất là b ộ kiểm chu ẩn đầu tiên cho trích xu ất 
RoPA ti ếng Vi ệt, gồm 32 t ổ chức giả lập, 77 ho ạt 
động x ử lý, 12 nhóm trư ờng và 4338 giá tr ị tham 
chiếu, trong đó 35.9% đã qua th ẩm định độc lập của 
hai chuyên gia, kèm b ộ chấm điểm tự động. Th ứ hai 
là nguyên t ắc báo cáo ba t ầng, tách b ạch bộ chấm 
điểm, đ ộ phủ đầu-cuối và th ẩm định nhãn chu ẩn, 
nhằm tránh vi ệc gộp lẫn ba lo ại chỉ số vốn phổ biến 
khi công b ố hệ thống RAG. Th ứ ba là kh ảo sát th ực 
nghiệm tính kh ả thi của triển khai t ại chỗ, trong đó hai 
mô hình mã ngu ồn mở chạy trọn 32/32 k ịch bản trên 
GPU 24 GB và m ột mô hình đ ạt độ phủ cùng kho ảng 
với mô hình đám mây tham chi ếu. Ki ến trúc đ ộc lập 
mô hình là l ựa chọn thiết kế phục vụ đóng góp th ứ ba, 
không ph ải đóng góp khoa h ọc độc lập. 
RoPA ph ục vụ trực tiếp bộ ba Bảo mật, Toàn v ẹn 
và Sẵn sàng c ủa dữ liệu cá nhân, trong khi h ệ thống 

Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
phải phòng ch ống các m ối đe d ọa đặc thù c ủa LLM 
như tiêm nhi ễm câu l ệnh và rò r ỉ dữ liệu qua mô hình 
đám mây. Do đó, RoPA Manager tích h ợp ngay t ừ 
khâu thi ết kế các nguyên lý b ảo mật theo khung 
STRIDE và mô hình không tin c ậy mặc định (Zero 
Trust), trình bày t ại Mục III.D.  
II. NGHIÊN  CỨU LIÊN  QUAN  
Nhóm công c ụ quản trị tuân th ủ (GRC) như 
OneTrust [6] và hư ớng dẫn ISO/IEC 29134:2023 [7] 
hỗ trợ quy trình qu ản lý RoPA nhưng v ẫn yêu c ầu 
chuyên viên b ảo vệ dữ liệu (DPO) nh ập liệu thủ công. 
Nhóm t ự động hóa tuân th ủ bằng NLP, tiêu bi ểu là 
kiểm tra tính đ ầy đủ của chính sách quy ền riêng tư đ ối 
chiếu GDPR [19], kh ả thi nhưng g ắn với khung pháp 
lý châu Âu và ti ếng Anh; các b ộ kiểm chu ẩn trích xu ất 
thông tin pháp lý như LexGLUE [20] cũng ch ỉ phục 
vụ tiếng Anh. Mô hình ti ền huấn luy ện tiếng Vi ệt như 
PhoBERT [8]  chưa đư ợc áp d ụng cho bài toán RoPA. 
Nhóm RAG cho mi ền chuyên ngành [4], [5], [12] cho 
thấy chất lượng truy xu ất và ảo giác v ẫn là thách th ức 
mở. Khảo sát trên Google Scholar, IEEE Xplore và 
ACM Digital Library (RoPA / records of processing 
activities k ết hợp Vietnamese, LLM, RAG; 2020 -
2026) không tìm th ấy công trình nào v ề trích xu ất 
RoPA t ự động cho ti ếng Vi ệt; theo hi ểu biết của nhóm 
tác gi ả, đây là kho ảng trống bài báo hư ớng tới. 
Về nền tảng kỹ thuật, kiến trúc Transformer v ới 
cơ ch ế tự chú ý [14] là cơ s ở của các LLM hi ện đại. 
Việc dùng LLM thay cho phương pháp d ựa trên quy 
tắc xuất phát t ừ đặc thù đ ầu vào: tài li ệu ngu ồn phục 
vụ RoPA là quy trình n ội bộ, hợp đồng và h ồ sơ đánh 
giá tác đ ộng, ở dạng bán phi c ấu trúc và không đ ồng 
nhất về bố cục giữa các t ổ chức, nên bi ểu thức chính 
quy hay lu ật trích xu ất cố định không bao ph ủ được. 
Đây là khác bi ệt cơ b ản so v ới các bài toán có lư ợc đồ 
đầu vào c ố định, nơi máy lu ật thư ờng đủ. Truy xuất 
lai đư ợc chọn vì thu ật ngữ pháp lý ti ếng Vi ệt đòi h ỏi 
khớp chính xác theo t ừ khóa, theo tinh th ần của khung 
xếp hạng xác su ất [21], song song v ới tương đ ồng ng ữ 
nghĩa, và RRF [22] h ợp nhất hai b ảng xếp hạng ch ỉ 
dựa trên th ứ hạng nên không ph ụ thuộc thang đi ểm 
của từng lu ồng. Mô hình nhúng đa ngôn ng ữ 
multilingual -e5-base [23] đư ợc ch ọn thay vì 
PhoBERT [8] vì c ần hỗ trợ song ng ữ Việt-Anh và 
tương thích tr ực tiếp với pgvector [15].  
III. THIẾT KẾ HỆ THỐNG VÀ PHƯƠNG  
PHÁP  
A. Kiến trúc t ổng th ể 
RoPA Manager theo ki ến trúc phân l ớp gồm bốn 
lớp: Trình bày ( ứng dụng web đơn trang), Nghi ệp vụ 
(điều phối quy trình trích xu ất), Trí tu ệ nhân t ạo (lớp 
RAG + LLM đ ộc lập mô hình) và D ữ liệu (cơ s ở dữ 
liệu quan h ệ và cơ s ở dữ liệu vector). Đi ểm đáng lưu ý của thiết kế là lớp trí tu ệ nhân t ạo độc lập điểm cu ối, 
cho phép chuy ển đổi giữa LLM đám mây và LLM t ại 
chỗ chỉ qua c ấu hình.  
Các quy ết định ki ến trúc theo tiêu chí gi ữ dữ liệu 
cá nhân trong h ạ tầng nội bộ và tối thiểu số thành ph ần 
vận hành: cơ s ở dữ liệu vector dùng pgvector [15] tích 
hợp trong PostgreSQL thay vì m ột hệ vector chuyên 
dụng riêng; máy ch ủ suy lu ận tại chỗ dùng vLLM nh ờ 
paged attention giúp gi ảm bộ nhớ GPU và cung c ấp 
giao di ện tương thích OpenAI, cho phép chuy ển đổi 
nhà cung c ấp mô hình ch ỉ bằng cấu hình đi ểm cu ối. 
B. Quy trình truy xu ất lai và trích xu ất 
Quy trình g ồm sáu giai đo ạn (Hình 1): ti ếp nhận 
và tiền xử lý tài li ệu (PDF/DOCX); phân đo ạn và sinh 
vector nhúng; truy xu ất lai; g ọi LLM trích xu ất; hậu 
xử lý và ki ểm tra h ợp lệ theo lư ợc đồ; lưu tr ữ để rà 
soát. Phân đo ạn ngữ nghĩa dùng ngư ỡng ng ắt 0.62. 
Luồng thưa dùng hàm x ếp hạng từ vựng ts_rank_cd 
trên ch ỉ mục tsvector c ủa PostgreSQL (c ấu hình 
simple); lu ồng m ật độ cao dùng tìm ki ếm cosine qua 
chỉ mục HNSW [13]. M ỗi luồng lấy dư 30 ứng viên 
trước khi h ợp nhất. Hai lu ồng đư ợc hợp nhất bằng 
biến thể có trọng số của RRF [22] v ới hằng số k = 60 
và trọng số 0.4 cho nhánh t ừ vựng, hai giá tr ị lấy theo 
cấu hình m ặc định và chưa qua t ối ưu hóa có h ệ thống, 
rồi giữ 8 đoạn xếp hạng cao nh ất làm ng ữ cảnh; phiên 
bản được đo không b ật bước xếp hạng lại bằng bộ mã 
hóa chéo. LLM nh ận câu l ệnh cố định (checksum 
SHA -256) kèm ng ữ cảnh; đ ầu ra b ị ràng bu ộc JSON 
và xác th ực lược đồ Pydantic. Hình 1 và B ảng I công 
bố các tham s ố chính đ ể bảo đảm kh ả năng tái l ập. 
 
Hình 1.  Ba tầng đo c ủa khung đánh giá. T ầng 2 ch ạy trọn quy 
trình có g ọi LLM; t ầng 1 theo nhánh nét đ ứt, đối chiếu nhãn 
chuẩn đã làm nhi ễu với bộ chấm điểm mà không qua LLM; t ầng 
3 thẩm định chính nhãn chu ẩn. 


Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
Bảng I.  CẤU HÌNH TRUY XU ẤT VÀ 
THAM S Ố TÁI L ẬP 
Thành ph ần Cấu hình s ử dụng trong th ực nghi ệm 
Chỉ mục vector  pgvector / PostgreSQL 15; HNSW m = 
16, ef_construction = 64  
Sinh n ội dung  temperature = 0.2; max_tokens = 1800  
Ràng bu ộc đầu ra  JSON + lư ợc đồ Pydantic 2.5.3  
Suy lu ận tại chỗ vLLM, GPU đơn 24 GB VRAM  
Nền tảng / seed  Python 3.11, FastAPI 0.109.0; seed 
20260705  
C. Nhóm trư ờng RoPA  
Thông tin RoPA đư ợc tổ chức thành 12 nhóm 
trường độc lập bám sát Đi ều 31 Lu ật 91/2025/QH15 
[2] và hư ớng dẫn chi ti ết tại Ngh ị định 356/2025/NĐ -
CP [1], l ần lư ợt là BASIC_INFO, PURPOSES, 
LEGAL_BASIS, r ồi DATA_SUBJECTS cho ch ủ thể 
và lo ại dữ liệu, RECIPIENTS_TRANSFERS cho bên 
nhận và chuy ển d ữ liệu xuyên biên gi ới, 
RETENTION cho th ời hạn lưu tr ữ và phương th ức 
xóa, hai nhóm TECHNICAL_MEASURES và 
STORAGE_SECURITY, r ồi CONSENT_DETAILS 
cho vi ệc thu th ập và rút l ại đồng ý, RISK_DPIA cho 
đánh giá tác đ ộng và m ức rủi ro, REGULATORY cho 
việc báo cáo cơ quan qu ản lý, và sau cùng là nhóm 
DOCUMENTATION. Mư ời hai nhóm báo cáo ánh 
xạ nhiều-nhiều sang mư ời hai nhóm truy v ấn của lớp 
trích xu ất: PURPOSES, DATA_SUBJECTS và 
RECIPIENTS_TRANSFERS m ỗi nhóm g ộp từ hai 
truy v ấn; hai nhóm STORAGE_SECURITY và 
TECHNICAL_MEASURES dùng chung m ột truy 
vấn; riêng DOCUMENTATION không có truy v ấn 
mà suy ra t ừ trường mô t ả và ngu ồn trích d ẫn. Nhóm 
báo cáo là đơn v ị của nhãn chu ẩn và c ủa đo lư ờng. 
D. Thiết kế an toàn thông tin  
Theo phân lo ại của OWASP dành cho ứng dụng 
LLM [18], tiêm nhi ễm câu l ệnh (prompt injection) là 
rủi ro hàng đ ầu đối với hệ thống dựa trên mô hình 
ngôn ng ữ lớn. RoPA Manager áp d ụng mô hình không 
tin cậy mặc định (Zero Trust) theo hư ớng dẫn NIST 
SP 800 -207 [17], k ết hợp phân tích m ối đe d ọa theo 
khung STRIDE [16] đ ể nhận diện đồng th ời các 
vector t ấn công truy ền thống (gi ả mạo, can thi ệp dữ 
liệu, từ chối dịch vụ) và đ ặc thù AI (tiêm nhi ễm câu 
lệnh, rò r ỉ dữ liệu qua mô hình đám mây, ảo giác sinh 
thông tin sai  lệch). Ba l ớp phòng th ủ được tích h ợp 
trực tiếp vào lu ồng trích xu ất: xác th ực lược đồ 
Pydantic lo ại bỏ mọi đầu ra không đúng c ấu trúc 
JSON, cơ ch ế an toàn n ội tại của mô hình t ừ chối thực 
thi ch ỉ thị nằm ngoài ph ạm vi đư ợc giao, và ràng bu ộc 
đầu ra JSON h ạn chế đáng k ể khả năng rò r ỉ thông tin 
qua văn b ản tự do. Hi ệu quả của thiết kế này đư ợc 
kiểm ch ứng th ực nghi ệm tại Mục V. IV. THIẾT LẬP THỰC NGHI ỆM 
A. Hạ tầng và mô hình  
Hệ thống đư ợc đánh giá trên ba LLM: DeepSeek -
V4-Flash (API đám mây) làm m ốc so sánh, Qwen3.5 -
27B-GPTQ -Int4 (lư ợng tử hóa 4 -bit) và Gemma -4-
31B (bf16) tri ển khai t ại chỗ qua vLLM trên GPU đơn 
24 GB VRAM. T ầng ứng dụng ch ạy trên AWS EC2 
t3.large  (2 vCPU, 8 GiB RAM, không GPU) vì mô 
hình nhúng v ận hành hi ệu quả trên CPU ở quy mô th ử 
nghiệm. Giai đo ạn thực nghi ệm gọi API đám mây v ới 
dữ liệu hoàn toàn t ổng hợp nên không vi ph ạm nguyên 
tắc chủ quyền dữ liệu; khi v ận hành v ới dữ liệu thật, 
hệ thống bắt buộc chuy ển sang ch ế độ tại chỗ. Cả ba 
mô hình dùng chung mã ngu ồn, mẫu câu l ệnh (cùng 
checksum SHA -256) và b ộ chấm điểm. 
B. Bộ dữ liệu kiểm chu ẩn 
Bộ dữ liệu được sinh t ổng hợp có ki ểm soát qua 
bốn bư ớc: ch ọn 7 lĩnh v ực có nghĩa v ụ RoPA cao 
(ngân hàng/tài chính, thương m ại điện tử, viễn thông, 
y tế, giáo d ục, công ngh ệ thông tin/ph ần mềm dạng 
dịch vụ, hàng không); thi ết kế 32 tổ chức giả lập, mỗi 
tổ chức 1-5 hoạt động xử lý; so ạn tài li ệu ngu ồn phi 
cấu trúc ti ếng Vi ệt xen thu ật ngữ tiếng Anh; gán nhãn 
12 nhóm trư ờng theo hư ớng d ẫn chu ẩn hóa v ới 
nguyên t ắc bảo thủ, kiểm tra b ằng bộ xác th ực lược đồ 
JSON và ki ểm tra logic chéo. Tài li ệu và nhãn chu ẩn 
do nhóm tác gi ả soạn thủ công, không sinh b ằng 
LLM, nh ằm loại trừ rủi ro vòng l ặp mô hình. Phân b ố 
độ khó g ồm 21 d ễ, 33 trung bình và 23 khó. Ba đơn 
vị đo lư ờng dùng song song: 924 th ực thể nhóm (77 × 
12); 4338 giá tr ị lá ở chế độ đồng nh ất; và 3965 giá tr ị 
lá đư ợc chấm ở chế độ làm nhi ễu/ẩn danh, sau khi 
lược bỏ 373 trư ờng (8.6%).  
C. Phương pháp đo lư ờng 
Nghiên c ứu tách b ạch ba t ầng đo vì chúng có đ ầu 
vào và đ ối tượng khác nhau (Hình 1). T ầng 1 ki ểm tra 
bộ chấm điểm qua ch ế độ đồng nh ất và ch ế độ làm 
nhiễu/ẩn danh; c ả hai đ ối chiếu nhãn chu ẩn với phiên 
bản đã bi ến đổi có ki ểm soát nên không g ọi LLM. 
Tầng 2 ch ạy trọn quy trình có g ọi LLM trên 32 k ịch 
bản; điểm mỗi nhóm trư ờng đư ợc tính b ằng tỉ lệ token 
khớp với nhãn chu ẩn, tức một chỉ số dạng recall ở cấp 
token. Vì 32 k ịch bản là quan sát ghép c ặp, so sánh 
giữa các mô hình dùng ki ểm định t ghép c ặp kèm đ ối 
chứng Wilcoxon. T ầng 3 th ẩm định độ tin cậy của 
chính nhãn chu ẩn. Kho ảng tin c ậy 95% tính b ằng 
bootstrap phi tham s ố (B = 10000, seed = 20260712, 
N = 3965); ngư ỡng KPI t ại Bảng II do nhóm nghiên 
cứu tự đặt. Mẫu thẩm định ở tầng 3 g ồm 1558 giá tr ị 
nhãn chu ẩn (35.9% b ộ kiểm chu ẩn), thu ộc 15/32 k ịch 
bản và 41/77 ho ạt động xử lý, ph ủ đủ 7/7 lĩnh v ực và 
12/12 nhóm trư ờng. M ẫu đư ợc lấy theo th ứ tự kịch 
bản trong m ỗi khối lĩnh v ực, không áp d ụng thi ết kế 

Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
phân t ầng ng ẫu nhiên, nên t ỉ lệ lấy mẫu dao đ ộng 33 -
100% gi ữa các lĩnh v ực. Hai chuyên gia ch ấm độc lập 
trên hai b ản sao tách r ời, không trao đ ổi kết quả và 
không bi ết nhãn sinh t ừ mô hình nào, theo thang ba 
mức Đúng / Đúng m ột phần / Sai.  
D. Khả năng tái l ập 
Trước mỗi lượt so sánh, m ọi thành ph ần ảnh 
hưởng kết quả được đóng băng phiên b ản: bộ dữ liệu 
(v2.0), b ộ chấm điểm (v1.3), mã ngu ồn (git commit 
c38e18c cho c ấu hình đám mây và 045547f cho hai 
mô hình t ại chỗ), tham s ố truy xu ất và m ẫu câu l ệnh 
dùng chung, ki ểm soát toàn v ẹn bằng SHA -256. M ỗi 
lượt chạy ghi nh ận mã lư ợt chạy, nhà cung c ấp, 
phương pháp lư ợng tử hóa, temperature, seed và th ời 
điểm th ực thi. Mã ngu ồn, bộ kiểm chu ẩn và h ồ sơ 
thẩm định của hai chuyên gia đư ợc công khai t ại địa 
chỉ https://github.c om/tdhinhit/RoPA -demo.  
V. KẾT QUẢ VÀ THẢO LUẬN 
Ở tầng bộ chấm điểm, ch ế độ đồng nh ất đạt mọi 
chỉ số bằng 1.0 trên 4338 giá tr ị lá, xác nh ận logic 
chấm điểm không có l ỗi kỹ thuật. Trên ch ế độ làm 
nhiễu/ẩn danh (3965 giá tr ị lá), F1 = 0.9493 [0.9436; 
0.9548], Precision = 0.9470 và Recall = 0.9517, v ới 
biên kho ảng tin c ậy hẹp dư ới 0.02. Vì ch ế độ này 
không g ọi LLM, các con s ố trên ch ỉ chứng minh b ộ 
chấm điểm ch ịu đư ợc dữ liệu khi ếm khuy ết, không 
phải hiệu năng trích xu ất. Sáu trong b ảy chỉ số tầng 1 
đạt ngư ỡng; riêng đ ộ chính xác t ừ chối trả lời, đo trên 
795 trư ờng có nhãn chu ẩn null, ch ỉ đạt 0.7874 so v ới 
ngưỡng 0.80 (B ảng II).  
Bảng II.  KẾT QU Ả BỘ CHẤM ĐI ỂM TỰ 
ĐỘNG (KHÔNG G ỌI LLM) SO V ỚI NGƯ ỠNG 
KPI  
Chỉ số Giá tr ị (CI 
95%)  Ngưỡng Trạng thái  
Tầng 1 - bộ chấm đi ểm, ch ế độ làm nhi ễu (N = 3965, không 
gọi LLM)  
F1  0.9493 
[0.9436; 
0.9548]  ≥ 0.85  PASS  
Accuracy  0.9188 
[0.9100; 
0.9271]  ≥ 0.85  PASS  
Precision  0.9470 
[0.9390; 
0.9547]  ≥ 0.85  PASS  
Recall  0.9517 
[0.9441; 
0.9591]  ≥ 0.80  PASS  
MCC [11]  0.7448  ≥ 0.55  PASS  
Ảo giác (d ữ liệu 
nhiễu) 0.0426  ≤ 0.05  PASS  
Từ chối trả lời (n 
= 795)  0.7874  ≥ 0.80  FAIL  Phân tích theo 12 nhóm trư ờng (B ảng III) cho th ấy 
PURPOSES, DATA_SUBJECTS và 
DOCUMENTATION đ ạt F1 g ần tuy ệt đối, 
RETENTION th ấp nh ất với 0.712, ti ếp theo 
REGULATORY 0.886 và STORAGE_SECURITY 
0.876; theo lĩnh v ực, F1 dao đ ộng hẹp 0.939 -0.960. 
Cơ c ấu sai s ố trong ch ế độ mô ph ỏng gồm 169 dương 
tính gi ả, trong đó 55% (93) là giá tr ị được bộ tạo nhi ễu 
chèn thêm, t ập trung ở RETENTION (46), 
RISK_DPIA (23) và STORAGE_SECURITY (23), 
còn 45% (76) rơi vào các c ặp nhóm ch ồng lấn ngữ 
nghĩa. Trong 153 âm tính gi ả, phần lớn thu ộc về ba 
nhóm BASIC_INFO (44), LEGAL_BASIS (23) và 
REGULATORY (18). Vì đây là nhi ễu nhân t ạo chứ 
không ph ải đầu ra mô hình, cơ c ấu trên ch ỉ là gi ả 
thuyết dự báo v ề những nhóm trư ờng mà LLM nhi ều 
khả năng ảo giác ho ặc nhầm lẫn khi v ận hành; ki ểm 
chứng trên đ ầu ra th ật của tầng 2 là công vi ệc tiếp 
theo.  
Bảng III.  KẾT QU Ả BỘ CHẤM ĐI ỂM THEO 
12 NHÓM TRƯ ỜNG (CH Ế ĐỘ LÀM NHI ỄU, 
KHÔNG G ỌI LLM)  
Nhóm trư ờng TP FP FN P R F1 
PURPOSES  174 0 0 1.000  1.000  1.000  
DATA_SUBJECTS  566 0 8 1.000  0.986  0.993  
DOCUMENTATION  200 0 5 1.000  0.976  0.988  
BASIC_INFO  572 0 44 1.000  0.929  0.963  
TECHNICAL_MEASURES  303 24 0 0.927  1.000  0.962  
RECIPIENTS_TRANSFERS  202 18 0 0.918  1.000  0.957  
LEGAL_BASIS  235 0 23 1.000  0.911  0.953  
CONSENT_DETAILS  209 18 13 0.921  0.941  0.931  
RISK_DPIA  214 23 17 0.903  0.926  0.915  
REGULATORY  136 17 18 0.889  0.883  0.886  
STORAGE_SECURITY  138 23 16 0.857  0.896  0.876  
RETENTION  68 46 9 0.597  0.883  0.712  
Ở tầng trích xu ất đầu-cuối, Bảng IV so sánh ba mô 
hình trên cùng b ộ dữ liệu, mã ngu ồn và m ẫu câu l ệnh. 
Cả ba hoàn thành 32/32 k ịch bản, xác nh ận kiến trúc 
độc lập mô hình v ận hành đúng thi ết kế. Độ phủ token 
đạt 55.25% v ới DeepSeek -V4-Flash, 55.06% v ới 
Qwen3.5 -27B t ại chỗ và 50.04% v ới Gemma -4-31B. 
Vì 32 k ịch bản là quan sát ghép c ặp, nghiên c ứu áp 
dụng ki ểm định t ghép c ặp kèm đ ối chứng Wilcoxon. 
Giữa DeepSeek và Qwen, chênh l ệch ch ỉ +0.20 đi ểm 
với kho ảng tin c ậy 95% [ -0.93; +1.32], t( 31) = 0.36, 
p = 0.72 và Wilcoxon p = 0.61. Không phát hi ện khác 
biệt, đồng th ời kho ảng tin c ậy loại trừ mọi chênh l ệch 
lớn hơn kho ảng 1.3 đi ểm, nên đây là b ằng ch ứng ủng 
hộ tính tương đương ch ứ không ch ỉ là thi ếu bằng 

Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
chứng bác b ỏ. Ngư ợc lại, Gemma th ấp hơn DeepSeek 
5.21 đi ểm [+2.03; +8.40] v ới p = 0.0022 và th ấp hơn 
Qwen 5.02 đi ểm [+2.10; +7.94] v ới p = 0.0014; 
khoảng cách t ập trung t ại LEGAL_BASIS, nơi mô 
hình ch ỉ đạt 41.6 -56.1% đ ộ phủ ở bốn lĩnh v ực so v ới 
83.5-100% c ủa hai mô hình còn l ại. Kiểm định dựa 
vào bi ến thiên gi ữa các k ịch bản trong m ột lượt chạy. 
Bảng IV.  SO SÁNH BA MÔ HÌNH TRÊN 
CÙNG B Ộ DỮ LIỆU VÀ CÂU L ỆNH 
Tiêu chí  DeepSeek -
V4-Flash  Qwen3.5 -
27B-Int4 Gemma -4-
31B 
Triển khai  API đám mây  vLLM t ại chỗ vLLM t ại chỗ 
Tham s ố N/A (h ộp đen)  27B (4 -bit) 31B (bf16)  
Lưu trú d ữ 
liệu (thi ết kế) PII rời hạ tầng PII n ội bộ PII n ội bộ 
Kịch bản 
hoàn thành  32/32  32/32  32/32  
Độ phủ (1 
lượt chạy) 55.25%  55.06%  50.04%  
Độ lệch 
chuẩn 4.43 4.63 9.30 
Chất lượng phân đo ạn ảnh hư ởng tr ực tiếp đến 
tầng truy xu ất. Quy trình v ận hành dùng phân đo ạn 
ngữ nghĩa (Hình 1), trung bình 42 đo ạn/tài li ệu; 
nghiên c ứu chạy một đối chứng riêng v ới phân đo ạn 
đệ quy theo ký t ự, vốn chỉ tạo trung bình 6.9 đo ạn/tài 
liệu. Trên 32 tài li ệu, cả hai đ ều giữ 100% phân đo ạn 
trong gi ới hạn 512 token c ủa mô hình nhúng và không 
cắt gãy câu; phân đo ạn ngữ nghĩa gi ữ độ kết dính n ội 
bộ cao (cosine 0.786 -0.832). H ệ quả đáng lưu ý là v ới 
chính sách gi ữ 8 đoạn làm ng ữ cảnh, c ấu hình ng ữ 
nghĩa đang v ận hành ch ỉ đưa kho ảng 19% n ội dung 
mỗi tài li ệu vào ng ữ cảnh cho m ỗi truy v ấn, trong khi 
cấu hình đ ệ quy s ẽ đưa g ần như toàn b ộ. Đây là m ột 
giả thuyết cạnh tranh đáng k ể cho đ ộ phủ token 50.04 -
55.25% ở tầng 2 và c ần đối chứng lo ại trừ để kiểm 
chứng. 
Ở tầng truy xu ất, nghiên c ứu chưa đo Recall@K, 
MRR hay NDCG do chưa xây d ựng bộ đánh giá m ức 
độ liên quan ở cấp truy v ấn-đoạn; vì thi ếu phép đo 
tách t ầng, bài báo không quy k ết nguyên nhân c ủa độ 
phủ thấp cho t ầng truy xu ất hay t ầng sinh.  
Về độ tin cậy của nhãn chu ẩn, không giá tr ị nào b ị 
đánh giá Sai; t ỉ lệ Đúng c ủa hai chuyên gia là 99.81% 
và 99.74%, đ ồng thu ận liên chuyên gia 99.68% 
(1553/1558) v ới 5 trư ờng bất đồng. C ả 5 điểm bất 
đồng đều thu ộc cùng m ột chủ đề là nghĩa v ụ viện dẫn 
Điều 13 và Đi ều 14 Lu ật 91/2025/QH15, trong các 
tình hu ống ch ấm điểm tín d ụng tự động, giám th ị thi 
bằng nh ận dạng khuôn m ặt và đánh giá nhân s ự bằng 
AI, cho th ấy điểm khó nh ất của bài toán n ằm ở suy 
luận pháp lý ch ứ không ở trích xu ất văn b ản. Đây 
cũng là lo ại tình hu ống đư ợc xếp mức khó, nên đ ộ khó 
trích xu ất và đ ộ khó gán nhãn có xu hư ớng tương 
quan; k ết hợp với tỉ trọng ho ạt động khó trong m ẫu (46.3% so v ới 29.9% toàn b ộ), có th ể kết luận mẫu 
không thiên v ề phía d ễ. Do phân b ố nhãn c ực lệch với 
99.7% cùng m ột lớp, Cohen's Kappa thô ch ỉ đạt 0.28, 
đúng ngh ịch lý prevalence đã bi ết; nghiên c ứu dùng 
PABAK = 0.9936 [10] làm ch ỉ số chính và công b ố κ 
thô đ ể minh b ạch. Hai gi ới hạn cần nêu là PABAK 
cao ch ỉ chứng minh hai chuyên gia nh ất quán v ới 
nhau, và hai chuyên gia thu ộc cùng m ột hệ sinh thái 
doanh nghi ệp. 
Về an toàn thông tin, nghiên c ứu tách hai lo ại đánh 
giá. Th ứ nhất, rà soát c ấu hình v ận hành: 12 tiêu chí 
cấu hình tiêu đ ề phản hồi HTTP c ủa proxy ngư ợc cho 
4 PASS, 6 WARN và 2 FAIL, trong đó hai h ạng m ục 
FAIL là thi ếu HSTS và đi ểm cu ối API chưa yêu c ầu 
xác th ực, bắt buộc khắc phục trước khi v ận hành. Th ứ 
hai, đánh giá kháng t ấn công: h ệ thống ch ặn thành 
công c ả 8 kịch bản tiêm nhi ễm câu l ệnh (chèn ch ỉ thị 
trực tiếp, đóng vai qu ản trị, ép l ộ câu l ệnh hệ thống, 
tiêm gián ti ếp qua trư ờng tài li ệu, rút trích d ữ liệu cá 
nhân, ghi đè lư ợc đồ JSON, đ ầu độc ngữ cảnh đa lư ợt, 
mã hóa Unicode) nh ờ ba lớp phòng th ủ nêu t ại Mục 
III.D; v ới n = 8 và m ột lượt chạy, đây là ki ểm ch ứng 
khái ni ệm, chưa đ ủ cơ sở cho k ết luận mạnh về khả 
năng kháng t ấn công.  
Khung đo g ồm ba t ầng với phạm vi khác nhau. 
Tầng 1 đánh giá b ộ chấm đi ểm, tầng 2 đo đ ộ phủ 
token so v ới nhãn chu ẩn theo hư ớng recall, t ầng 3 
đánh giá đ ộ tin cậy của nhãn chu ẩn. Không t ầng nào 
đo độ chính xác ở cấp giá tr ị trên đ ầu ra th ật của mô 
hình, t ức tỉ lệ đúng trong s ố các giá tr ị hệ thống sinh 
ra và t ỉ lệ từ chối đúng trong s ố các trư ờng bỏ trống; 
tầng 1 ch ỉ đo hai đ ại lượng này trên b ộ chấm điểm, ở 
chế độ không g ọi LLM. M ọi kết luận trong bài vì v ậy 
giới hạn ở tính kh ả thi kỹ thuật và đ ộ phủ. 
VI. KẾT LUẬN VÀ HƯỚNG PHÁT  TRIỂN 
Nghiên c ứu thiết lập đường cơ s ở kỹ thuật cho ứng 
dụng LLM + RAG vào t ự động hóa tuân th ủ Luật Bảo 
vệ dữ liệu cá nhân t ại Việt Nam, đóng góp b ộ kiểm 
chuẩn RoPA ti ếng Vi ệt đầu tiên và nguyên t ắc báo cáo 
ba tầng. V ề chi phí tham kh ảo: GPU 24 GB kho ảng 
50-65 tri ệu VNĐ, m ột DPO chuyên trách 180 -300 
triệu VNĐ/năm, gói GRC thương m ại 10000 -50000 
USD/năm. Hư ớng phát tri ển theo ưu tiên g ồm đo đ ộ 
chính xác n ội dung ở cấp giá tr ị trên b ản nháp th ật, 
theo thi ết kế lấy mẫu phân t ầng với hai ngư ời chấm 
độc lập ba nhãn đ úng / sai / sinh thêm và báo cáo 
precision, recall theo nhóm trư ờng; lặp thực nghi ệm 
đa mô hình t ối thiểu ba lư ợt; xây d ựng bộ đánh giá 
mức độ liên quan đ ể đo Recall@K, MRR, NDCG và 
đối chứng lo ại trừ cho truy xu ất lai [5]; và đo m ức 
hiệu chu ẩn của điểm tin c ậy do mô hình t ự gán trên 
đầu ra th ật. 
Hạn chế chính g ồm: (1) d ữ liệu hoàn toàn t ổng 
hợp, chưa ki ểm ch ứng trên h ồ sơ RoPA th ực tế; (2) 
chưa đo đ ộ chính xác ở cấp giá tr ị theo hư ớng 

Bản thảo được nhận đăng trong K ỷ yếu Hội thảo quốc gia l ần thứ XXIX:  
Một số vấn đề chọn lọc của Công ngh ệ thông tin và truy ền thông  – Hà N ội, 7-8/11/2026 
precision; (3) ch ỉ số tầng 1 đo b ộ chấm điểm ở chế độ 
không g ọi LLM, nên cơ c ấu sai s ố theo nhóm trư ờng 
mới ở mức giả thuyết dự báo; (4) ki ểm định ghép c ặp 
dựa trên m ột lượt chạy; (5) chưa có đ ối chứng lo ại trừ 
nên đóng góp riêng c ủa RRF chưa đư ợc lượng hóa, và 
chưa đo Recall@K, MRR, NDCG; (6) m ẫu thẩm định 
nhãn chu ẩn phủ 35.9% b ộ kiểm chu ẩn, lấy theo th ứ tự 
kịch bản chứ không ng ẫu nhiên, v ới n = 2 chuyên gia; 
(7) chưa đánh giá m ức hiệu chu ẩn của điểm tin c ậy do 
mô hình t ự gán trên đ ầu ra th ật [9]. Các h ạn chế này 
xác l ập phạm vi đóng góp ở mức đường cơ s ở kỹ 
thuật; nghiên c ứu không tuyên b ố đạt mức tuân th ủ 
pháp lý v ận hành đ ầy đủ. 
LỜI CẢM ƠN 
Nhóm tác giả trân trọng cảm ơn hai chuyên 
gia tuân thủ và pháp chế độc lập đã thẩm 
định mù đơn 1558 về giá trị nhãn chuẩn, 
tạo cơ sở tin cậy cho toàn bộ phép đo trong 
nghiên cứu. Nghiên cứu được thực hiện 
bằng kinh phí tự túc của nhóm tác giả, 
không nhận tài trợ từ bất kỳ tổ chức hay 
doanh nghiệp nào. Nhóm tác giả không có 
lợi ích thương mại liên quan đến hệ thống 
được mô tả và không có quan hệ tài chính 
với các nhà cung cấp mô hình hoặc công cụ 
quản trị tuân thủ được nhắc đến trong bài; 
các tác giả tuyên bố không có xung đ ột lợi 
ích. Nguyễn An Khương  cảm ơn Trường 
Đại học Bách khoa, ĐHQG -HCM đã hỗ trợ 
cho nghiên cứu này.  
 
 
TÀI  LIỆU THAM  KHẢO 
[1] Chính ph ủ CHXHCN Vi ệt Nam, "Ngh ị định 356/2025/NĐ -
CP quy đ ịnh chi ti ết một số điều của Luật Bảo vệ dữ liệu cá 
nhân," Hà N ội, 2025.  
[2] Quốc hội CHXHCN Vi ệt Nam, "Lu ật Bảo vệ dữ liệu cá nhân 
(Luật 91/2025/QH15)," Hà N ội: NXB Chính tr ị Quốc gia, 
2025.  
[3] P. Lewis, E. Perez, A. Piktus, et al., "Retrieval -augmented 
generation for knowledge -intensive NLP tasks," in 
Advances in Neural Information Processing Systems, vol. 
33, 2020, pp. 9459 –9474.  [4] J. Chen et al., "Benchmarking large language models in 
retrieval -augmented generation," in Proc. AAAI Conf. 
Artificial Intelligence, 2024, pp. 17754 –17762.  
[5] S. Es et al., "RAGAS: Automated evaluation of retrieval -
augmented generation," in Proc. EACL (System 
Demonstrations), 2024, pp. 150 –158. 
[6] Gartner, "Market guide for privacy management tools," 
Gartner Research, 2025.  
[7] ISO/IEC 29134:2023, "Information technology — Security 
techniques — Guidelines for privacy impact assessment," 
2023.  
[8] D. Q. Nguyen and A. T. Nguyen, "PhoBERT: Pre -trained 
language models for Vietnamese," in Findings of ACL: 
EMNLP, 2020, pp. 1037 –1042.  
[9] C. Guo et al., "On calibration of modern neural networks," 
in Proc. ICML, 2017, pp. 1321 –1330.  
[10] J. Byrt et al., "Bias, prevalence and kappa," J. Clin. 
Epidemiol., vol. 46, no. 5, pp. 423 –429, 1993.  
[11] B. W. Matthews, "Comparison of the predicted and observed 
secondary structure of T4 phage lysozyme," Biochim. 
Biophys. Acta, vol. 405, no. 2, pp. 442 –451, 1975.  
[12] S. Barnett et al., "Seven failure points when engineering a 
retrieval -augmented generation system," in Proc. 
IEEE/ACM Int. Conf. AI Engineering (CAIN), 2024, pp. 
194–199. 
[13] Y. A. Malkov and D. A. Yashunin, "Efficient and robust 
approximate nearest neighbor search using hierarchical 
navigable small world graphs," IEEE Trans. Pattern Anal. 
Mach. Intell., vol. 42, no. 4, pp. 824 –836, 2020.  
[14] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all 
you need," in Advances in Neural Information Processing 
Systems, vol. 30, 2017, pp. 5998 –6008.  
[15] A. Kane, "pgvector: Open -source vector similarity search for 
PostgreSQL," GitHub repository, 2023.  
[16] A. Shostack, Threat Modeling: Designing for Security. 
Indianapolis: John Wiley & Sons, 2014.  
[17] S. Rose et al., Zero Trust Architecture, NIST Special 
Publication 800 -207, 2020.  
[18] OWASP Foundation, "OWASP Top 10 for Large Language 
Model Applications," Version 1.1, 2025.  
[19] D. Torre et al., "An AI -assisted approach for checking the 
completeness of privacy policies against GDPR," in Proc. 
IEEE Int. Requirements Eng. Conf. (RE), 2020, pp. 136 –
146. 
[20] I. Chalkidis et al., "LexGLUE: A benchmark dataset for legal 
language understanding in English," in Proc. ACL, 2022, pp. 
4310 –4330.  
[21] S. Robertson and H. Zaragoza, "The probabilistic relevance 
framework: BM25 and beyond," Found. Trends Inf. Retr., 
vol. 3, no. 4, pp. 333 –389, 2009.  
[22] G. V. Cormack et al., "Reciprocal rank fusion outperforms 
Condorcet and individual rank learning methods," in Proc. 
SIGIR, 2009, pp. 758 –759. 
[23] L. Wang, N. Yang, X. Huang, et al., "Multilingual E5 text 
embeddings: A technical report," arXiv preprint 
arXiv:2402.05672, 2024.  
 
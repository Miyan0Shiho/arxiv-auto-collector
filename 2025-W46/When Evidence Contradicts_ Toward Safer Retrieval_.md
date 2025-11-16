# When Evidence Contradicts: Toward Safer Retrieval-Augmented Generation in Healthcare

**Authors**: Saeedeh Javadi, Sara Mirabi, Manan Gangar, Bahadorreza Ofoghi

**Published**: 2025-11-10 03:27:54

**PDF URL**: [https://arxiv.org/pdf/2511.06668v1](https://arxiv.org/pdf/2511.06668v1)

## Abstract
In high-stakes information domains such as healthcare, where large language models (LLMs) can produce hallucinations or misinformation, retrieval-augmented generation (RAG) has been proposed as a mitigation strategy, grounding model outputs in external, domain-specific documents. Yet, this approach can introduce errors when source documents contain outdated or contradictory information. This work investigates the performance of five LLMs in generating RAG-based responses to medicine-related queries. Our contributions are three-fold: i) the creation of a benchmark dataset using consumer medicine information documents from the Australian Therapeutic Goods Administration (TGA), where headings are repurposed as natural language questions, ii) the retrieval of PubMed abstracts using TGA headings, stratified across multiple publication years, to enable controlled temporal evaluation of outdated evidence, and iii) a comparative analysis of the frequency and impact of outdated or contradictory content on model-generated responses, assessing how LLMs integrate and reconcile temporally inconsistent information. Our findings show that contradictions between highly similar abstracts do, in fact, degrade performance, leading to inconsistencies and reduced factual accuracy in model answers. These results highlight that retrieval similarity alone is insufficient for reliable medical RAG and underscore the need for contradiction-aware filtering strategies to ensure trustworthy responses in high-stakes domains.

## Full Text


<!-- PDF content starts -->

When Evidence Contradicts: Toward Safer
Retrieval-Augmented Generation in Healthcare
Saeedeh Javadi1⋆, Sara Mirabi2⋆, Manan Gangar2, and Bahadorreza Ofoghi2
1RMIT University, Melbourne, Australia
saeedeh.javadi@student.rmit.edu.au
2Deakin University, Melbourne, Australia
s222496341@deakin.edu.au, gangarmanan27@gmail.com, b.ofoghi@deakin.edu.au
Abstract.Inhigh-stakesinformationdomainssuchashealthcare,where
large language models (LLMs) can produce hallucinations or misinfor-
mation, retrieval-augmented generation (RAG) has been proposed as
a mitigation strategy, grounding model outputs in external, domain-
specific documents. Yet, this approach can introduce errors when source
documents contain outdated or contradictory information. This work
investigates the performance of five LLMs in generating RAG-based re-
sponses to medicine-related queries. Our contributions are three-fold: i)
the creation of a benchmark dataset using consumer medicine informa-
tion documents from the Australian Therapeutic Goods Administration
(TGA), where headings are repurposed as natural language questions,
ii) the retrieval of PubMed abstracts using TGA headings, stratified
across multiple publication years, to enable controlled temporal evalua-
tionofoutdatedevidence,andiii)acomparativeanalysisofthefrequency
and impact of outdated or contradictory content on model-generated re-
sponses,assessinghowLLMsintegrateandreconciletemporallyinconsis-
tent information. Our findings show that contradictions between highly
similar abstracts do, in fact, degrade performance, leading to inconsis-
tencies and reduced factual accuracy in model answers. These results
highlight that retrieval similarity alone is insufficient for reliable medical
RAG and underscore the need for contradiction-aware filtering strategies
to ensure trustworthy responses in high-stakes domains.
Keywords:Large language models·Retrieval-augment generation·
Contradiction detection·Medicine·Question answering.
1 Introduction
Large language models (LLMs) have demonstrated exceptional capabilities in re-
sponding to user information requests that require world knowledge. This knowl-
edge is inherent in the vast amounts of pre-training data they consume, which
is increasingly making these models a primary candidate for information seek-
ing, where they have achieved state-of-the-art performance on a wide range of
⋆Equal contribution.arXiv:2511.06668v1  [cs.IR]  10 Nov 2025

tasks, including medical question answering (QA) [30]. In domains like health,
however, knowledge of specific concepts is constantly being updated with new
findings, some of which are contradictory to previously known facts [4, 24]. In-
consistent and outdated sources used for training the generative models can
potentially result in unreliable generated information. To mitigate misinforma-
tion and hallucinations resulting from such inconsistencies in training medical
information where accuracy is paramount [19], retrieval-augmented generation
(RAG) is employed as a strategy [21, 30].
RAG effectively supplements the LLM’s internal knowledge with context in-
formation, and studies have shown that augmenting prompts with retrieved ev-
idence can reduce the incidence of hallucinations in practice [15]. The efficacy
of RAG is, however, predicated on the relevance and coherence of the retrieved
documents, thereby raising significant concerns regarding the model’s perfor-
mance and robustness should the retrieval process yield contradictory or irrele-
vantresults[32].AsignificantchallengeforRAGsystems,therefore,istheLLM’s
handling of conflicting or outdated information within retrieved documents [31].
Given that an LLM’s internal parameters are static, conflicts between its fixed
knowledge base and new, external context are an unavoidable issue. Resolving
such conflicts is a non-trivial task, as the model may incorrectly prioritize a par-
ticular source or even synthesize contradictory information. Despite this, there is
currently a lack of systematic research and established guidelines for managing
these issues in RAG systems, particularly within medical applications [30].
In this work, we investigate how well LLMs handle medicine-related queries
when grounded in authoritative reference documents. We focus on the effects
of contradictions and outdated content on the quality of RAG-powered LLM
outputs using a purpose-curated dataset from two prominent sources: the Aus-
tralian Therapeutic Goods Administration (TGA) and the PubMed repository.
The dataset includes medicine-related questions, linked PubMed abstracts, and
metadata such as publication year, ensuring a mix of recent and older sources.
The questions were used as information requests to several LLMs, where the
relevant PubMed abstracts were utilized for RAG, and the correct answers from
the TGA brochures were considered as ground-truth responses. Our main con-
tributions include:
–CuratingadatasetlinkingTGAconsumermedicineinformationwithPubMed
abstracts across multiple publication years, prioritizing temporal diversity.
–Implementation of a RAG pipeline using FAISS retrieval and the BAAI/bge-
small-en-v1.5 embedding model, enabling reproducible evaluation.
–EvaluationoffiveLLMs,includingFalcon3[25],Gemma-3[7],GPT-OSS[17],
Med-LLaMA3 [14], and Mixtral [10] on the dataset to assess their ability to
provide accurate, up-to-date, and non-contradictory answers using RAG.
–Analysis of how the time stamp of source information and potential contra-
dictions in source materials affect the quality of generated responses.
2

2 Related Work
SeveralstudieshavetakenRAGstrategiesinthedomaininrecentyears.MKRAG,
a RAG framework for medical QA, retrieves medical facts from an external
disease database and injects them into LLM prompts through in-context learn-
ing[22].OpenEvidenceandChatRWDweredevelopedtoapplyRAGtoliterature-
based clinical evidence and map queries into PICO study designs for real-world
evidence generation [16]. In a similar direction, a RAG-driven model was pro-
posed for health information retrieval that integrates PubMed Central with gen-
erative LLMs through a three-stage pipeline of passage retrieval, generative re-
sponse construction, and factuality checks via stance detection and semantic
similarity [26]. A multi-source benchmark for evidence-based clinical QA was
curated from Cochrane reviews [1], AHA guidelines [2], and narrative guidance,
underscoring the need for diversified and authoritative retrieval sources [27]. Ex-
tendingthisline,MedCoT-RAGcombinedcausal-awareretrievalwithstructured
chain-of-thought prompting for medical QA [28]. A two-layer RAG framework
leveraging Reddit data on emerging substances such as xylazine and ketamine
was introduced to generate query-focused summaries suitable for low-resource
settings [6]. BriefContext, a map-reduce framework that partitions long retrieval
contexts into shorter segments to mitigate the “lost-in-the-middle” problem and
improve the accuracy of medical QA in RAG systems, was introduced [33]. EX-
PRAG, a retrieval-augmented generation framework, leveraged electronic health
records by retrieving discharge reports from clinically similar patients through
a coarse-to-fine process (EHR-based report ranking followed by experience re-
trieval), enabling case-based reasoning for diagnosis, medication, and discharge
instruction QA [18]. MedRAG integrated electronic health records with a hier-
archical diagnostic knowledge graph for clinical decision support [34]. Discuss-
RAG, an agent-led framework, enhanced RAG by using multi-agent discussions
to construct context-rich summaries for retrieval and a verification agent to fil-
ter irrelevant snippets before answer generation [9]. Finally, a biomedical QA
system based on RAG was developed with a two-stage hybrid retrieval pipeline,
where BM25 [20] performed lexical retrieval over PubMed, and MedCPT’s cross-
encoder [12] reranked the top candidates to refine semantic relevance [23].
WhilepreviousstudiesdemonstratethebenefitsofRAGinmedicalQA,most
concentrate on retrieval pipelines, evidence structuring, or database integration.
Few explicitly examine the risks posed by outdated or contradictory evidence
within retrieved sources. This remains a critical gap for high-stakes medical
applications, where knowledge evolves rapidly.
3 Methodology
3.1 Problem Formulation and Objectives
ThisstudyaddressesthechallengeofevaluatingRAGinthemedicalQAdomain,
where information accuracy and temporal consistency are critical. The problem
is formally defined as follows. Let
M={m 1,...,m 1476},(1)
3

denote a set of1,476medicines regulated by the TGA. For each medicinem i∈
M, a fixed set of six standardized consumer-oriented queries,
Qi={q i,1,qi,2,...,q i,6},(2)
is defined, which covers common information needs: 1) therapeutic indications,
2) pre-use warnings, 3) drug-drug interactions, 4) dosage and administration, 5)
guidance while under treatment, including interactions/monitoring, and 6) ad-
verse effects.1These queries represent typical information needs that patients
and healthcare consumers might have when consulting medical information sys-
tems. Across the collection, a total of|Q|=/summationtext1476
i=1|Qi|= 6|M|= 8,856query
instances are obtained. For each queryq i,j∈Qiwithj∈{1,...,6}, a retrieval
process is performed against the PubMed biomedical literature database to ob-
tain a candidate set of abstracts,
Di,j={d 1,...,d ni,j},(3)
where each documentd∈ D i,jis characterized by its textt(d), publication
yearγ(d), citation countκ(d)∈N 0, and unique PubMed identifierPMID(d).
The retrieval process yields resulting in a total corpus containing approximately
400,000documents spanning publication years from1975to2025. The funda-
mental challenge lies in the fact that retrieved abstracts may contain outdated
recommendations, conflicting findings from different studies, or evolving medi-
cal consensus over time. The RAG system must therefore not only identify rel-
evant documents but also reconcile potentially contradictory information while
generating accurate, consistent responses. Our primary objectives are therefore
threefold.
Objective 1: Temporal Diversity in Evidence Selection.Given the evolv-
ing nature of medical knowledge, the system must select a subset,
Ri,j⊆Di,j,|R i,j|≤20,(4)
that maximizes temporal diversity while maintaining relevance. This requires
balancing recent findings with historical context, particularly when medical rec-
ommendations change over time.
Objective 2: Contradiction-Aware Retrieval.The system must identify
and quantify contradictions among retrieved abstracts to avoid incorrect or po-
tentially harmful responses. A contradiction function,
CNT:R i,j×R i,j→[0,1],(5)
1The sixth standardized phrasings in our corpus include, e.g.,“Why am I using
ABACAVIR?”(indications);“What should I know before I useABACAVIR?”(pre-use
warnings);“What if I am taking other medicines withABACAVIR?”(drug-drug inter-
actions);“How do I useABACAVIR?”(dosage/administration);“What should I know
while usingABACAVIR?”(on-treatment guidance); and“Are there any side effects of
ABACAVIR?”(adverse effects).
4

is defined such that, for any pair(d a,db), the valueCNT(d a,db)quantifies the
likelihood thatt(d a)contradictst(d b); this function is intended to support the
construction of “most-contradictory” and “least-contradictory” configurations.
Objective 3: RAG Performance Evaluation.Given a retrieved setR i,jand
anLLML∈Λ,thesystemmustgeneratearesponsea i,jthataddressesq i,jusing
the retrieved context. The quality of this response is evaluated against ground
truth answers extracted from TGA medicine information documents. Five LLMs
including Falcon3 [25], Gemma-3 [7], GPT-OSS [17], Med-LLaMA3 [14], and
Mixtral [10] are evaluated, collectively denoted asΛ.
3.2 Evidence Set Construction
Query Expansion and Search.The evidence acquisition employs a three-tier
query formulation strategy to balance precision and recall. For each(m i,qi,j)
pair, content termsT i,j={t 1,t2,...,t k}are extracted through part-of-speech
filtering using SpaCy [11], retaining nouns (NN, NNS), verbs (VB*), and proper
nouns (NNP, NNPS) while excluding pharmaceutical company names through
a curated exclusion list. Three query formulations are constructed: i) exact sen-
tence matchQ 1=/logicalandtext
t∈Ti,jsentence(t), ii) proximity-constrainedQ 2=NEAR 25
(Ti,j)∧m i[ti], and iii) full query proximityQ 3=NEAR 25(qi,j), where NEAR k
denotes proximity withinktokens and[ti]restricts to title field. Results are
deduplicated by PMID to formDraw
i,j.
Abstract Acquisition and Filtering.PMIDs are fetched in batches to mit-
igate API limitations. XML responses are parsed to extract(pmid,γ(d),t(d)).
Records without abstracts are discarded; all remaining years are preserved to
enable temporal analyses.
Temporal-CitationBalancedSelection.Acriticalinnovationinthismethod-
ology is the implementation of a selection algorithm that balances temporal di-
versity with citation impact. This approach addresses the challenge of capturing
evolving medical knowledge while prioritizing influential research within each
temporal stratum. Based on Algorithm 1, to construct a pool of up to 20 ab-
stractsperquerywhilepreservingtemporalcoverage,whenthenumberofunique
publication years exceeds 20, the system performs stratified sampling to select
years with approximately three-year intervals, ensuring representation across the
full temporal range. For datasets with fewer than 20 unique years, all years are
retained. Within each selected year, abstracts are ranked by citation count in
descending order, serving as a proxy for scientific impact and reliability.
The final selection employs a round-robin approach across years, iteratively
selecting the highest-cited unselected abstract from each year until either 20
abstracts are selected or all available abstracts are exhausted. This mechanism
ensures both temporal diversity and quality, preventing over-representation of
any single time period while maintaining citation-based quality signals.
5

Algorithm 1Temporal-citation balanced selection
Require:CandidatesDraw
i,j, citation functionκ:D→N, year functionγ:D→N
Ensure:Selected subsetR i,jwhere|R i,j|≤20
1:Y←{γ(d) :d∈Draw
i,j}
2:if|Y|≥20then
3:Y∗←STRATIFIED-SAMPLE(Y,20,gap= 3)
4:else
5:Y∗←Y
6:end if
7:for ally∈Y∗do
8: Sort{d∈Draw
i,j:γ(d) =y}byκ(d)descending
9:end for
10:Ri,j←∅
11:while|R i,j|<20and∃y∈Y∗with remaining candidatesdo
12:for ally∈Y∗in round-robin orderdo
13:ifcandidates remain for yearythen
14:R i,j←Ri,j∪{arg max d:γ(d)=yκ(d)}
15:end if
16:end for
17:end while
18:returnR i,j
3.3 Diversity-Aware Scoring for Retrieval Framework
Document Representation and Indexing.Document embeddings are com-
puted using an encoder functione:Σ∗→Rd, specifically the BAAI/bge-small-
en-v1.5 model [29]. The embeddings are indexed using Facebook AI Similarity
Search (FAISS) [13].
Maximal Marginal Relevance with Temporal Augmentation.The re-
trieval framework extends maximal marginal relevance (MMR) [3] by incorpo-
rating temporal diversity. For a given queryqand candidate setD, the MMR
score for documentd iis:
MMR(d i|q,D) =λ·cos(e(q),e(t(d i)))−(1−λ)·max
j̸=icos(e(t(d i)),e(t(d j)))(6)
where cosine similarity is defined ascos(x,y) =x⊤y
∥x∥ 2∥y∥ 2, in the embedding
space, andλcontrols the trade-off between relevance and diversity. This score is
then combined with a temporal diversity component. A temporal diversity score
is computed to favor documents spanning different time periods:
τ(di) =γ(di)−min d∈Dγ(d)
max d∈Dγ(d)−min d∈Dγ(d) +ϵ(7)
whereϵ= 10−5prevents division by zero. The final ranking score integrates
relevance, redundancy, and temporal components:
S(di|q) =α·MMR(d i|q,D) + (1−α)·τ(d i)(8)
6

The documents are re-ranked based on this score (Algorithm 2), and the
top-Kdocuments byS(·|q)are selected as context.
Algorithm 2MMR + year-aware ranking (per query)
1:Input:queryq, candidatesD={d 1,...,dn};n≤20, parametersλ,α
2:fori= 1tondo
3:si←cos/parenleftbig
e(q),e(t(d i))/parenrightbig
4:ri←maxj̸=icos/parenleftbig
e(t(di)),e(t(dj))/parenrightbig
5:MMR i←λsi−(1−λ)r i
6:end for
7: computeτ ifor alliusing (7)
8:Si←αMMR i+ (1−α)τ i
9:returnre-ranking documents inDbyS i(descending)
3.4 Contradiction Detection Scoring Framework
To quantify conflicting evidence in the retrieved abstract pool for a query, the
subsetR i,j={d 1,...,d n}⊆D i,jwas considered. Each documentd∈R i,jis
embedded with a scientific encodere sim(SPECTER [5]), producing represen-
tationh(d) =e sim(t(d)). For any abstract pair(d a,db)∈R i,j×R i,j, a coarse
similarity was computed as:
simabs(da,db) = cos/parenleftbig
h(da),h(d b)/parenrightbig
,(9)
serving as a coarse filter. Each abstractdwas segmented into sentencesS(d) =
{s(1)
d,...,s(Ld)
d}. Sentences were embedded with the same encoder,h(s) =
esim(s). For a pair(d a,db), candidate sentence pairs were retained when,
cos/parenleftbig
h(s),h(t)/parenrightbig
≥θ sent= 0.75,(s,t)∈S(d a)×S(d b).(10)
TheresultingcandidatesetisdenotedP(d a,db)⊆S(d a)×S(d b).Each(s,t)∈
P(dp,dq)ispassedtoabiomedicalnaturallanguageinference(NLI)classifierf nli
(PubMedBERT-MNLIMedNLI[8])thatreturns(P ent(s,t), P neu(s,t), P con(s,t)),
wherePdenotes the probability assigned to each relation:entailment(ent) in-
dicates that the hypothesis is supported by the premise,neutral(neu) denotes
no clear relation, andcontradiction(con) indicates conflict. The contradiction
functionCNT:R i,j×Ri,j→[0,1]is then defined at the document level by the
peak contradiction probability over candidate sentence pairs:
CNT(d a,db) = max
(s,t)∈P(d a,db)Pcon(s,t).(11)
The framework thus reports, for each document pair, i) the document-level
similarity, ii) the peak contradiction score, and iii) the most indicative sentence
pair with similarity evidence. A document-level contradiction salience within the
pool was defined as:
7

Text
1..1
0..*
1..1
0..*
1..*
0..1
1..*
0..*
1..1
0..1
Order
OrderID
OrderDate
HeadquartersID
OrderDetail
OrderDetailID
ProductID
OrderID
ProductQuantity
 
DeliveryDate
 
SupplierID
Headquarters
HeadquartersID
BranchID
Branch
BranchID
BranchName
Product
SupplierID
ProductID
Supplier
DeliveryID
DeliveryDate
SupplierID
Start
Get 
email 
address
Email 
valid 
& 
does 
not 
exist?
Yes
No
Password 
valid?
Get 
password
Create 
new 
account
Yes
No
End
Entity
Boundary
Control
Alternative
[Condition]
[Else]
Loop
[Condition]
Program
+ 
Main
- 
findAccount
(bank: 
Bank): 
Account
- 
doAddAccount
(bank: 
Bank)
- 
doWithdraw
(bank: 
Bank)
- 
doDeposit
(bank: 
Bank)
- 
doTransfer
(bank: 
Bank)
- 
doPrint
(bank: 
Bank)
- 
readUserOption
(): 
MenuOption 
<<enumeration>>
MenuOption
ADD_ACCOUNT
WITHDRAW
DEPOSIT
TRANSFER
PRINT
QUIT
Account
- 
_balance: 
decimal
- 
_name: 
string
<<property>> 
+ 
Name: 
string 
{readOnly}
+ 
Account(name: 
string, 
startingBalance: 
decimal
+ 
Deposit(amountToAdd: 
decimal): 
bool
+ 
Withdraw(amountToWithdraw: 
decimal): 
bool
+ 
Print
Bank
- 
_accounts: 
List<Account>
+ 
Bank
+ 
AddAccount(account: 
Account)
+ 
GetAccount(name: 
string): 
Account
+ 
ExecuteTransaction(transaction: 
WithdrawTransaction)
+ 
ExecuteTransaction(transaction: 
DepositTransaction)
+ 
ExecuteTransaction(transaction: 
TransferTransaction)
WithdrawTransaction
- 
_account: 
Account
- 
_amount: 
decimal
- 
_executed: 
bool
- 
_succeeded: 
bool
- 
_reversed: 
bool
<<property>> 
+ 
Executed: 
bool 
{readOnly}
<<property>> 
+ 
Succeeded: 
bool 
{readOnly}
<<property>> 
+ 
Reversed: 
bool 
{readOnly}
+ 
WithdrawTransaction(account: 
Account, 
amount: 
decimal)
+ 
Execute
+ 
Rollback
+Print
DepositTransaction
- 
_account: 
Account
- 
_amount: 
decimal
- 
_executed: 
bool
- 
_succeeded: 
bool
- 
_reversed: 
bool
<<property>> 
+ 
Executed: 
bool 
{readOnly}
<<property>> 
+ 
Succeeded: 
bool 
{readOnly}
<<property>> 
+ 
Reversed: 
bool 
{readOnly}
+ 
DepositTransaction(account: 
Account, 
amount: 
decimal)
+ 
Execute
+ 
Rollback
+Print
TransferTransaction
- 
_fromAccount: 
Account
- 
_toAccount: 
Account
- 
_amount: 
decimal
- 
_theWithdraw: 
WithdrawTransaction
- 
_theDeposit: 
DepositTransaction
- 
_executed: 
bool
- 
_reversed: 
bool
<<property>> 
+ 
Executed: 
bool 
{readOnly}
<<property>> 
+ 
Succeeded: 
bool 
{readOnly}
<<property>> 
+ 
Reversed: 
bool 
{readOnly}
+ 
TransferTransaction(fromAccount: 
Account, 
toAccount: 
Account, 
amount: 
decimal)
+ 
Execute
+ 
Rollback
+ 
Print
0..*
1..*
0..*
1..*
_account
_account
_fromAccount, 
_toAccount
1
1
_theWithdraw
_accounts
1
0..*
1
1
_theDeposit
fromAccount:
Account
toAccount:
Account
transferTransaction:
TransferTransaction
Text
theWithdraw:
WithdrawTransaction
theDeposit:
DepositTransaction
start 
transfer
start 
withdraw
withdraw
start 
deposit
deposit
success 
status
alt
[withdraw 
succeeded=true]
[else]
success 
status
success
status
Program
Terminal
run
alt
[deposit 
succeeded=true]
[else]
rollback 
withdraw
bank:
Bank
find 
from 
account
from 
account
find 
to 
account
to 
account
success 
status
Program
+ 
Main
- 
findAccount
(bank 
: 
Bank): 
Account
- 
doAdddAccount
(bank 
: 
Bank)
- 
readUserOption
(): 
MenuOption
...
<<enumeration>>
MenuOption
Add_Account
Withdraw
Deposit
Transfer
Print
Quit
Account
0..*
1..*
Terminal
Program
run
bank: 
Bank
find 
account
fromAccount
find 
account
toAccount
For 
Us
-
Indigenous 
peoples 
inclusion
-
Cultural 
preservation
-
Technology 
used 
for 
indigenous 
benefits
With 
Us
-
Co-design
-
Ethical 
indigous 
engagement
-
Shared 
authority 
in 
pedagogical 
activities
Like 
Us
-
Indigenous-led 
design
-
Epistemological 
embedding 
of 
cultural 
aspects
-
Deep 
enculturation 
of 
ICT 
pedagogies
Brochur 
Processing 
and 
Question/Answer 
Extraction
TGA 
CMI 
Brochurs
MedicineQuAD 
Dataset 
(json)
LLM
TGA
Questions
TGA 
Long 
Answers
LLM 
Responses
Evaluation 
Framework
Lexical 
[
ROUGE, 
BLEU, 
METEOR
]
Semantic 
[
BERTScore, 
Cosine 
Sim, 
Dot-Product 
Sim
]
Statistical 
Distribution 
[
KLD, 
JSD
]
Information 
Coverage
Preprocessing
[normalization,
tokenization,
stopword 
removal]
LLM 
Responses
TGA 
Long 
Answers
Systematic 
Stage 
& 
Typology 
Analysis 
of 
Online 
Abuse
time 
series 
of 
posts
Linguistic 
Analysis 
of 
Different 
Abusive 
Online 
Behaviors
predictive 
feature 
analysis
 
social 
media
Long-Term 
Harm 
& 
Impact 
Analysis 
of 
Online 
Abuse
affected 
users
Privacy-Preserved, 
Stage-Aware 
Language 
Modelling 
& 
Analysis 
of 
Online 
Textual 
Abuse 
aim-1
aim-2
aim-3
LLMs
aim-4
1,476 
medicines 
regulated 
by 
TGA
PubMed 
Medical 
Literature 
Database
Answer
Query 
Expansion 
& 
Search 
Temporal-Citation 
Balanced 
Selection
Embedding 
& 
Indexing 
FAISS
Generate
Diversity-Aware 
Scoring 
for 
Retrieval 
Framework
Contradiction 
Detection 
Scoring 
Framework
Query 
+ 
Top 
K 
Documents 
as 
Context 
Most 
Similar 
Documents
Vector 
Database
RAG 
with 
5 
LLMs
Falcon 
3
Gemma 
3
GPT-OSS
Med-LLaMA 
3
Mixtral
Most 
Contradictory 
Documents
Least 
Contradictory 
DocumentsFig. 1.Contradiction-aware medical RAG pipeline, showing data progression from
TGA queries through search, embedding, and three retrieval strategies to final LLM-
based generation and evaluation.
CNT(d) =1
|Ri,j|−1/summationdisplay
d′∈Ri,j
d′̸=dCNT(d,d′).(12)
GivenK, themost-contradictoryandleast-contradictorycontext sets used in
the retrieval variants were constructed as:
Cmost
i,j = arg topK
d∈Ri,j/parenleftbig
CNT(d)/parenrightbig
,Cleast
i,j= arg topK
d∈Ri,j/parenleftbig
−CNT(d)/parenrightbig
,(13)
respectively,whilethemost-similarconditionemployedthetop-Kbythediversity-
aware scoreS(·)in Equation (8).
3.5 Retrieval-Augmented Generation Pipeline
Given a queryq i,jand its ranked list obtained from Equation (8), a context set
Ci,j={d(1),...,d(K)}was formed by taking the top-Kdocuments. A grounded
prompting instruction was then applied so that an answera(ℓ)
i,jwas produced by
a modelℓ∈Λusing onlyC i,j; if the evidence was insufficient, the tokenInsuf-
ficient evidencewas to be emitted. Three retrieval conditions were instantiated,
each with a constantK: i) amost-similarcondition using the top-KbyS(·|q);
ii) amost-contradictorycondition selecting theKdocuments with largest con-
tradiction salience (Section 3.4); and iii) aleast-contradictorycondition selecting
theKsmallest by the same criterion. A schematic of the end-to-end pipeline is
provided in Figure 1, comprising evidence construction, indexing, and diversity-
aware ranking, retrieval variants, RAG inference, and evaluation.
4 Experimental Setup
Table 1 summarizes the dataset used in our experiments. Starting from nearly
400k retrieved PubMed documents, temporal–citation balanced selection re-
tained 91,662 PMIDs, of which 28,873 are unique. Of the 1,476 medicines, 1,074
8

Table 1.Statistics of the constructed TGA–PubMed dataset used in our experiments.
Statistic Value
Medicines (TGA) 1,476
Queries (6per medicine) 8,856
Time span of abstracts 1975–2025
Language English
PubMed documents retrieved (raw)∼400,000
Medicines with≥1 retrieved documents 1,074
PMIDs after Temporal-citation balanced selection 91,662
Unique PMIDs in filtered set 28,873
Average documents per query 13.96
have at least one retrieved document, among queries that retained evidence, the
mean context size is 13.96 documents per query.
The experiments employ the following configurations: MMR parameterλ=
0.7for balancing relevance and diversity, temporal weightingα= 0.7for the
final ranking score, and retrieval sizeK= 5documents per query. Document
embeddings use BAAI/bge-small-en-v1.5 (384 dimensions), indexed with FAISS
HNSW graphs (M=16, ef_construction=200). PubMed API calls use batch sizes
of 300 PMIDs per request. All language models in Table 4 use temperature=0
for deterministic generation with a 256 token limit. Implementation uses Python
3.10 with LangChain v0.1.0 for RAG orchestration.
Performance assessment employs multiple complementary metrics. Lexical
overlap is measured via ROUGE-1,2,L (R 1,R2,RL) scores. Semantic similarity
uses embedding cosine similarity with the same encoder as retrieval to minimize
metric/encoder mismatch. Vector similarity (VSIM), implemented with Gensim,
captures term-level relevance through Word2Vec embeddings. Distributional di-
vergence is quantified through Jensen-Shannon divergence (JSD) and Kullback-
Leibler divergence (KLD), where lower values indicate closer distributions. All
metrics are computed per query and macro-averaged across medicines.
5 Results and Discussion
5.1 Overall Performance
Table 2 presents comprehensive performance metrics across all models and the
three retrieval configurations. Several key patterns emerge from these results:
Model Performance Hierarchy.Mixtral consistently achieved the high-
est performance in the most-similar condition (R 1=0.163,BERT cosine=0.601),
followed closely by Med-LLaMA (R 1=0.156,BERT cosine=0.573) and Falcon
(R1=0.154,BERT cosine=0.589). The superior performance of Mixtral can be
attributed to its mixture-of-experts architecture, enabling more nuanced pro-
cessing of medical terminology and context.
Impact of Contradictions.Across models, performance consistently de-
graded when moving from most-similar to most-contradictory retrieval condi-
tions. The averageR 1score decreased by 18.2% when models were provided with
9

Table 2.Macro-averaged results (short references) for 5 LLMs across 3 retrieval
conditions. Higher is better except JSD/KLD (↓). Ret.=Retrieval, ms=most sim-
ilar, mc=most contradicted, lc=least contradicted, cos=cosine, dot=dot-product.
VSIM=vector similarity computed with Gensim.
Model Ret.ROUGE BERT VSIMJSD↓KLD↓
R1R2RLcos dot cos dot
Falcon3- ms 0.154 0.027 0.103 0.589 0.589 0.708 34.07 0.213 3.130
7B mc 0.148 0.025 0.099 0.571 0.570 0.689 32.52 0.220 3.420
lc 0.151 0.026 0.101 0.583 0.583 0.702 33.57 0.209 3.215
Gemma-3 ms 0.066 0.018 0.062 0.447 0.446 0.432 12.09 0.424 2.339
mc 0.060 0.018 0.060 0.430 0.430 0.421 11.64 0.438 2.290
lc 0.064 0.017 0.060 0.437 0.437 0.428 11.98 0.427 2.373
GPT-OSS- ms 0.114 0.016 0.087 0.559 0.559 0.676 18.40 0.243 3.394
20B mc 0.098 0.011 0.076 0.545 0.545 0.661 15.05 0.260 3.695
lc 0.100 0.016 0.079 0.549 0.549 0.668 16.37 0.257 3.514
Med- ms 0.156 0.032 0.109 0.573 0.573 0.645 28.69 0.256 2.772
LLaMA3- mc 0.134 0.027 0.097 0.529 0.529 0.607 24.75 0.283 2.988
8B lc 0.137 0.029 0.099 0.537 0.537 0.619 25.19 0.278 2.653
Mixtral- ms 0.163 0.029 0.118 0.601 0.601 0.706 32.74 0.214 3.208
8x7B mc 0.131 0.022 0.096 0.551 0.551 0.670 30.33 0.225 3.752
lc 0.140 0.023 0.099 0.579 0.579 0.690 30.84 0.223 2.813
contradictory documents. This was most pronounced for Mixtral, which showed
a 20% reduction inR 1, suggesting that larger models may be more susceptible
to conflicting information despite their generally superior performance. Based
on the results, performance under the least-contradictory condition was higher
than the most-contradictory but still fell short of the most-similar condition.
Semantic vs. Lexical Alignment.While ROUGE scores showed sub-
stantial variation across conditions, semantic similarity metrics (BERT cosine)
demonstrated greater stability, with an average decrease of only 8.3% between
most-similarandmost-contradictoryconditions.Thissuggeststhatmodelsmain-
tain conceptual understanding even when struggling with precise lexical genera-
tion under contradiction. VSIM showed a similar trend, with stronger alignment
under most-similar retrievals and moderate declines under contradictory evi-
dence. JSD values stayed stable across conditions, with only small differences
between most-similar and most-contradictory settings. In contrast, KLD showed
consistently higher values under contradictory retrievals.
Model-Specific Observations.Med-LLaMA, despite being fine-tuned for
medical applications, did not consistently outperform general-purpose models.
This suggests that domain-specific training may not fully compensate for the
challenges posed by contradictory retrieval contexts. GPT-OSS demonstrated
the most consistent performance across retrieval conditions, with only a 14%
degradation between best and worst conditions. Its transformer MoE architec-
10

ture appears to provide robustness against conflicting information, though at
the cost of lower peak performance.
5.2 Contradiction and Diversity-Aware Score Joint Distribution
To understand how document contradictions interact with the diversity-aware
scoreS(·), a 2-D frequency table was computed over all documents considered
during retrieval. Table 3 reveals critical insights about the relationship between
document diversity-aware score and contradiction likelihood. The highest fre-
quency of documents (5,492) falls in the intersection of high contradiction scores
(0.8-1.0) and moderate diversity-aware scores (0.2-0.4). This counterintuitive
finding suggests that documents with intermediate topical relevance are most
likely to contain conflicting information, potentially due to the same medical
concepts but from different temporal or methodological perspectives.
Table 3.Document frequency by binned diversity-aware similarity scoreS(columns)
and contradiction score CNT(rows). Zero values indicate that no documents fall into
those ranges.
CNTDiversity-Aware Similarity ScoreS
[0,0.2) [0.2,0.4) [0.4,0.6) [0.6,0.8) [0.8,1]
[0,0.2)695 1712 584 0 0
[0.2,0.4)816 2501 642 0 0
[0.4,0.6)1058 3435 982 0 0
[0.6,0.8)1328 4983 1295 0 0
[0.8,1]1617 5492 1249 0 0
5.3 Temporal Distribution of Contradiction Scores
To better understand how contradictions evolve over time, we aggregated doc-
uments into 5-year bins and computed proportions across contradiction score
bins. Figure 2 presents the resulting heatmap.
<2000 2000 05
 2005 10
 2010 15
 2015 20
 2020 25
Year[0,0.2)[0.2,0.4)[0.4,0.6)[0.6,0.8)[0.8,1]Contradiction Score 744 236 300 332 522 853861 344 424 507 705 1078947 444 640 783 1055 15791096 623 915 1121 1588 2220984 767 1117 1424 1794 2258
500100015002000
Document Count
Fig. 2.Normalized distribution of documents across contradiction score bins and 5-
year publication intervals.
11

Table 4.Models evaluated in our RAG experiments. We report the *released* model
characteristics and the *exact* inference variant used for reproducibility.
Model (ID) Family/Architecture Params Context License
Falcon3-7B [25] Decoder-only (dense) 7.46B 32k TII Falcon
gemma-3 (MLX) [7] Decoder-only (dense) 0.27B 32k Gemma
GPT-OSS-20b [17] Transformer MoE 21B 128k Apache-2.0
Med-LLaMA3-8B [14] Llama-3-8B (dense) 8.03B 8k Llama-3
Mixtral-8x7B [10] MoE (8×7B; top-2) 46.7B 32k Apache-2.0
The results reveal a clear temporal shift in the prevalence of contradictions.
Before 2000, most documents fall into the lower contradiction score bins, sug-
gesting relatively consistent findings across biomedical literature. From the early
2000s onwards, however, the share of documents in higher contradiction bins
([0.6–0.8)and[0.8–1])risessteadily.By2010–2025,thesebinsaccountforaround
half or more of all documents per interval, surpassing the lower bins. This in-
dicates that contradictions have become proportionally more prevalent in later
years, reflecting both the rapid expansion of biomedical research and the over-
turning of earlier clinical consensus. These findings underscore the importance
of contradiction-aware retrieval strategies that explicitly consider temporal dy-
namics when integrating evidence into RAG systems.
5.4 Limitations and Future Work
This study has several limitations. First, our contradiction detection relies on
sentence-level, which may miss document-level contradictions. Second, we eval-
uate only English-language documents, limiting generalization to multilingual
medical contexts. Future work should explore more sophisticated contradiction
resolution strategies, including argumentation mining and evidence synthesis
techniques. Additionally, incorporating clinical guidelines and expert knowledge
bases could help resolve contradictions by establishing authoritative sources. De-
veloping specialized medical RAG architectures that explicitly model temporal
evolution and evidence strength remains a direction for future research.
6 Conclusion
This work provides the first systematic evaluation of contradictory information
in medical RAG systems through a purpose-built TGA–PubMed dataset across
1,476 medicines. Our analysis shows that contradictions in retrieved evidence
consistently degrade model performance, with averageR 1scores declining by
18.2% when contradictory documents are present. Notably, even semantically
similar documents often contain conflicting information, with over 5,400 docu-
ment pairs exhibiting high contradiction scores.
12

These findings underscore contradictions in medical literature as a critical
vulnerability for RAG, where a 20% performance loss is unacceptable in high-
stakes applications. Future systems must therefore integrate contradiction de-
tection and resolution, leveraging temporal reasoning and uncertainty quantifi-
cation. Beyond technical metrics, our dataset establishes a benchmark for eval-
uating RAG robustness and highlights the urgent need for contradiction-aware
architectures to ensure factual accuracy in evolving healthcare contexts.
13

Bibliography
[1] Cochrane library. search | cochrane library.https://www.
cochranelibrary.com/cdsr/reviews(2025)
[2] American Heart Association: Statements search.https://professional.
heart.org/en/guidelines-statements-search(2025)
[3] Carbonell, J., Goldstein, J.: The use of mmr, diversity-based reranking for
reorderingdocumentsandproducingsummaries.ACMSIGIRForum51(2),
335–336 (1998)
[4] Carpenter, D., Geryk, L., AT, A.C., Nagler, R., Dieckmann, N., Han, P.:
Conflicting health information: A critical research need. Health Expect19,
1173–1182 (2016).https://doi.org/10.1111/hex.12438
[5] Cohan, A., Feldman, S., Beltagy, I., Downey, D., Weld, D.: Specter:
document-level representation learning using citation-informed transform-
ers. 2020. arXiv preprint arXiv:2004.07180 (2004)
[6] Das, S., Ge, Y., Guo, Y., Rajwal, S., Hairston, J., Powell, J., Walker,
D., Peddireddy, S., Lakamana, S., Bozkurt, S., et al.: Two-layer retrieval-
augmented generation framework for low-resource medical question answer-
ing using reddit data: proof-of-concept study. Journal of Medical Internet
Research27, e66220 (2025)
[7] DeepMind, G.: Gemma 3 270m instruction-tuned (mlx 8-bit).https://
huggingface.co/mlx-community/gemma-3-270m-it-8bit(2025)
[8] Deka, P.: Pubmedbert-mnli-mednli.https://huggingface.co/
pritamdeka/PubMedBERT-MNLI-MedNLI(2021)
[9] Dong, X., Zhu, W., Wang, H., Chen, X., Qiu, P., Yin, R., Su, Y., Wang,
Y.: Talk before you retrieve: Agent-led discussions for better rag in medical
qa. arXiv preprint arXiv:2504.21252 (2025)
[10] mradermacher (GGUF), M.A.: Mixtral-8x7b-instruct-v0.1 (gguf).https:
//huggingface.co/mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF
(2023), apache-2.0; 32k context
[11] Honnibal, M., Montani, I., Van Landeghem, S., Boyd, A.: spacy: Industrial-
strength natural language processing in python (2020).https://doi.org/
10.5281/zenodo.1212303
[12] Jin, Q., Kim, W., Chen, Q., Comeau, D.C., Yeganova, L., Wilbur, W.J., Lu,
Z.: Medcpt: Contrastive pre-trained transformers with large-scale pubmed
search logs for zero-shot biomedical information retrieval. Bioinformatics
39(11), btad651 (2023)
[13] Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with gpus.
IEEE Transactions on Big Data7(3), 535–547 (2019)
[14] Lab, Y.B.X.: Med-llama3-8b.https://huggingface.co/YBXL/
Med-LLaMA3-8B(2024)
[15] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.:
Retrieval-augmented generation for knowledge-intensive NLP tasks. In:

Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Ad-
vances in Neural Information Processing Systems. vol. 33, pp. 9459–9474.
Curran Associates, Inc. (2020)
[16] Low, Y.S., Jackson, M.L., Hyde, R.J., Brown, R.E., Sanghavi, N.M., Bald-
win, J.D., Pike, C.W., Muralidharan, J., Hui, G., Alexander, N., et al.:
Answering real-world clinical questions using large language model based
systems. arXiv preprint arXiv:2407.00541 (2024)
[17] OpenAI: gpt-oss-20b model card.https://huggingface.co/openai/
gpt-oss-20b(2025), apache-2.0; 21B total, 3.6B active; 128k context
[18] Ou, J., Huang, T., Zhao, Y., Yu, Z., Lu, P., Ying, R.: Experience retrieval-
augmentation with electronic health records enables accurate discharge qa.
arXiv preprint arXiv:2503.17933 (2025)
[19] Pal, A., Umapathi, L.K., Sankarasubbu, M.: Med-HALT: Medical domain
hallucination test for large language models. In: Jiang, J., Reitter, D., Deng,
S. (eds.) Proceedings of the 27th Conference on Computational Natural
Language Learning (CoNLL). pp. 314–334. Association for Computational
Linguistics, Singapore (Dec 2023).https://doi.org/10.18653/v1/2023.
conll-1.21,https://aclanthology.org/2023.conll-1.21/
[20] Robertson, S., Zaragoza, H., et al.: The probabilistic relevance framework:
Bm25 and beyond. Foundations and Trends®in Information Retrieval3(4),
333–389 (2009)
[21] Shi, Y., Xu, S., Yang, T., Liu, Z., Liu, T., Li, X., Liu, N.: MKRAG: Medical
knowledge retrieval augmented generation for medical question answering.
In: Proceedings of AMIA Annual Symposium (2025)
[22] Shi, Y., Xu, S., Yang, T., Liu, Z., Liu, T., Li, X., Liu, N.: MKRAG: Medical
knowledge retrieval augmented generation for medical question answering.
In: AMIA Annual Symposium Proceedings. vol. 2024, p. 1011 (2025)
[23] Stuhlmann,L.,Saxer,M.A.,Fürst,J.:Efficientandreproduciblebiomedical
question answering using retrieval augmented generation. arXiv preprint
arXiv:2505.07917 (2025)
[24] Taylor, T.: How to make sense of contradictory health
news,https://www.abc.net.au/news/health/2018-04-24/
making-sense-of-seemingly-contradictory-health-news/9343684,
accessed: 2025-09-16
[25] Technology Innovation Institute: Falcon3-7b-instruct.https:
//huggingface.co/tiiuae/Falcon3-7B-Instruct(2024), license:
TII Falcon-LLM 2.0
[26] Upadhyay, R., Viviani, M.: Enhancing health information retrieval with rag
by prioritizing topical relevance and factual accuracy. Discover Computing
28(1), 27 (2025)
[27] Wang, C., Chen, Y.: Evaluating large language models for evidence-based
clinical question answering. arXiv preprint arXiv:2509.10843 (2025)
[28] Wang, Z., Khatibi, E., Rahmani, A.M.: Medcot-rag: Causal chain-
of-thought rag for medical question answering. arXiv preprint
arXiv:2508.15849 (2025)
15

[29] Xiao, S., Liu, Z., Zhang, P., Muennighoff, N.: C-pack: Packaged resources to
advancegeneralchineseembedding.arXivpreprintarXiv:2309.07597(2023)
[30] Xiong, G., Jin, Q., Lu, Z., Zhang, A.: Benchmarking retrieval-augmented
generation for medicine. In: Ku, L.W., Martins, A., Srikumar, V. (eds.)
ACL (Findings). pp. 6233–6251. Association for Computational Linguistics
(2024)
[31] Xu, R., Qi, Z., Guo, Z., Wang, C., Wang, H., Zhang, Y., Xu, W.: Knowledge
conflicts for LLMs: A survey. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N.
(eds.) EMNLP. pp. 8541–8565. Association for Computational Linguistics
(2024)
[32] Yan, S.Q., Gu, J.C., Zhu, Y., Ling, Z.H.: Corrective retrieval augmented
generation (2024),https://arxiv.org/abs/2401.15884
[33] Zhang, G., Xu, Z., Jin, Q., Chen, F., Fang, Y., Liu, Y., Rousseau, J.F., Xu,
Z., Lu, Z., Weng, C., et al.: Leveraging long context in retrieval augmented
language models for medical question answering. npj Digital Medicine8(1),
239 (2025)
[34] Zhao, X., Liu, S., Yang, S.Y., Miao, C.: Medrag: Enhancing retrieval-
augmented generation with knowledge graph-elicited reasoning for health-
care copilot. In: Proceedings of the ACM on Web Conference 2025. pp.
4442–4457 (2025)
16
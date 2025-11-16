# A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain

**Authors**: Yining Lu, Wenyi Tang, Max Johnson, Taeho Jung, Meng Jiang

**Published**: 2025-11-10 19:40:30

**PDF URL**: [https://arxiv.org/pdf/2511.07577v1](https://arxiv.org/pdf/2511.07577v1)

## Abstract
Existing retrieval-augmented generation (RAG) systems typically use a centralized architecture, causing a high cost of data collection, integration, and management, as well as privacy concerns. There is a great need for a decentralized RAG system that enables foundation models to utilize information directly from data owners who maintain full control over their sources. However, decentralization brings a challenge: the numerous independent data sources vary significantly in reliability, which can diminish retrieval accuracy and response quality. To address this, our decentralized RAG system has a novel reliability scoring mechanism that dynamically evaluates each source based on the quality of responses it contributes to generate and prioritizes high-quality sources during retrieval. To ensure transparency and trust, the scoring process is securely managed through blockchain-based smart contracts, creating verifiable and tamper-proof reliability records without relying on a central authority. We evaluate our decentralized system with two Llama models (3B and 8B) in two simulated environments where six data sources have different levels of reliability. Our system achieves a +10.7\% performance improvement over its centralized counterpart in the real world-like unreliable data environments. Notably, it approaches the upper-bound performance of centralized systems under ideally reliable data environments. The decentralized infrastructure enables secure and trustworthy scoring management, achieving approximately 56\% marginal cost savings through batched update operations. Our code and system are open-sourced at github.com/yining610/Reliable-dRAG.

## Full Text


<!-- PDF content starts -->

A DECENTRALIZEDRETRIEVALAUGMENTEDGENERATIONSYSTEM WITH
SOURCERELIABILITIESSECURED ONBLOCKCHAIN
Yining Lu* 1Wenyi Tang* 1Max Johnson1Taeho Jung1Meng Jiang1
ABSTRACT
Existing retrieval-augmented generation (RAG) systems typically use a centralized architecture, causing a high
cost of data collection, integration, and management, as well as privacy concerns. There is a great need for a
decentralized RAG system that enables foundation models to utilize information directly from data owners who
maintain full control over their sources. However, decentralization brings a challenge: the numerous independent
data sources vary significantly in reliability, which can diminish retrieval accuracy and response quality. To
address this, our decentralized RAG system has a novel reliability scoring mechanism that dynamically evaluates
each source based on the quality of responses it contributes to generate and prioritizes high-quality sources during
retrieval. To ensure transparency and trust, the scoring process is securely managed through blockchain-based
smart contracts, creating verifiable and tamper-proof reliability records without relying on a central authority. We
evaluate our decentralized system with two Llama models (3B and 8B) in two simulated environments where six
data sources have different levels of reliability. Our system achieves a +10.7% performance improvement over its
centralized counterpart in the real world-like unreliable data environments. Notably, it approaches the upper-bound
performance of centralized systems under ideally reliable data environments. The decentralized infrastructure
enables secure and trustworthy scoring management, achieving approximately 56% marginal cost savings through
batched update operations. Our code and system are open-sourced at github.com/yining610/Reliable-dRAG.
1 INTRODUCTION
Retrieval Augmented Generation (RAG) (Lewis et al., 2020)
has been a popular technique for enhancing large language
models (LLMs) by providing them access to external knowl-
edge sources during inference. Most existing RAG systems
adopt a centralized architecture in which a single administra-
tor manages all data sources, including collection, cleaning,
de-duplication, and indexing. However, as the number of
data sources grows, this centralized architecture encounters
challenges in data management costs (Gao et al., 2024; Bar-
nett et al., 2024) and privacy concerns (Addison et al., 2024).
To address these challenges, we build a decentralized RAG
system ( dRAG ) that adopts the similar definition of data
providers introduced by Hecking et al. (2025). Our dRAG
system enables LLM services to retrieve documents directly
from sources maintained and controlled by individual data
owners, giving them the flexibility to decide what informa-
tion to share and what retrieval policies to implement.
However, building dRAG is not trivial because data sources
from different owners often vary notably in quality, mak-
*Equal contribution1Department of Computer Science and
Engineering, University of Notre Dame, USA. Correspondence to:
Yining Lu<ylu33@nd.edu>, Wenyi Tang<wtang3@nd.edu>.
Preprint. Under review.ing it impractical to treat all sources equally reliable. For
example, some data sources may contain outdated, mis-
information, or incomplete data (as shown in Figure 1).
The existing decentralized RAG systems (Hecking et al.,
2025; Xu et al., 2025; Zhou et al., 2025) typically assume
all data sources are fully reliable, which is unrealistic in
practice. To overcome this limitation, dRAG has a novel
scoring mechanism that dynamically evaluates the reliabil-
ity of data sources from the quality of responses they help
generate. These reliability scores enable dRAG to prioritize
high-quality sources during retrieval, thereby improving
both retrieval accuracy and response quality in noisy data
environments. Yet these reliability scores cannot be trusted
if managed by a centralized entity, as a centralized adminis-
trator could easily manipulate scores to favor certain data
sources, creating a single point of failure and compromising
system integrity. So, we leverage blockchain in dRAG to
secure the reliability scores and relevant logs. The scoring
mechanism is implemented within a smart contract, whose
execution and verification are collectively performed by the
majority of nodes in the blockchain network. This ensures
that all updates to the scores are executed properly, creat-
ing tamper-proof and transparent scoring records that all
participants can verify without relying on a central authority.
dRAG can benefit many domains and applications wherearXiv:2511.07577v1  [cs.CR]  10 Nov 2025

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
Data Source ACollect/Update/Clean... 
Data Source BData Source CDocuments
DocumentsDocumentsDocuments
AdminUserQuery
ResponseLLM Service
ManageManageManageQuery
Response
Data Owner A
Outdated
MisinformationIncompleteLLM Service
User
Data Owner BData Owner CManage
Figure 1. Overview of a centralized RAG system (left) and our
dRAG system (right), with their performance comparison (bottom)
on LLAMA-3.2-3B-INSTRUCT. In the unreliable data environ-
ments, our dRAG significantly outperforms the centralized sys-
tem with increasing query exposure, approaching the performance
upper-bound from the centralized system achieved in fully reliable
data environments.
data cannot be centrally managed, such as regulated enter-
prises (e.g., healthcare and public sector), cross-institutional
consortia, multi-tenant platforms, and open-source commu-
nities. dRAG offers reliable retrieval augmentation across
independently owned sources through two mechanisms: (1)
client-side reliability scoring for response quality, and (2)
a blockchain-based reliability management interface ensur-
ing transparency and trustworthiness. dRAG can be easily
deployed: data providers expose a standards-compliant re-
trieval endpoint and register with the smart contract registry,
while users install our lightweight, open-sourced client li-
brary, which connects to the deployed smart contract on-
chain for retrieval and reranking. Furthermore, the continu-
ously updated reliability scores on blockchain provide each
data owner with actionable feedback, encouraging proactive
monitoring and improvement of their data sources.
To demonstrate the effectiveness of dRAG , we simulate unre-
liable data environments in which multiple data sources pro-
vide Wikipedia documents of varying reliability to answer
questions from the Natural Questions dataset (Kwiatkowski
et al., 2019). Specifically, we inject noise into documents
by replacing ground-truth tokens with incorrect answers,
thereby polluting them to varying levels of reliability.Aneffective dRAG system should demonstrate robustness
in such unreliable environments by outperforming its
centralized counterpart(seeCentralized (unreliable)and
Decentralized (unreliable)lines in Figure 1) and exhibit
consistent growth as more user queries are processed. Re-
markably, our results also show that dRAG can even ap-
proach the performance of a centralized system with
fully reliable data sources(seeCentralized (reliable)with
no injected noise), an ideal configuration that is difficult
and costly to maintain in real-world settings. As previ-
ously discussed, centralized systems require administrators
to invest substantial effort in data management to ensure
data source reliability, whereas dRAG eliminates this burden
while achieving comparable performance. In summary, our
contributions are threefold:
•We introduce a reliability scoring mechanism for decen-
tralized RAG systems that dynamically evaluates source
reliability to improve retrieval and answer quality.
•We build our dRAG on blockchain to establish a decentral-
ized, transparent, and trustworthy scoring management
for data source reliability, using smart contracts to ensure
full traceability of score updates. We open-source our
system to facilitate future research.
•Through controlled experiments, we demonstrate that
dRAG can improve retrieval and response quality over
time, outperforming centralized systems under unreliable
data environments and achieving performance comparable
to those with ideal, fully reliable data sources.
2 RELATEDWORK ANDBACKGROUND
2.1 Centralized Retrieval Augmented Generation
Traditional RAG systems rely on centralized architectures in
which knowledge bases and retrieval mechanisms are man-
aged under unified control (Karpukhin et al., 2020; Lewis
et al., 2020; Li et al., 2022; Gao et al., 2024). This central-
ized design introduces several fundamental limitations that
hinder its scalability and practical deployment.
Centralized RAG architectures face inherent scalability lim-
itations stemming from their single-index design and mono-
lithic control (Wang et al., 2024; Douze et al., 2025). Even
with techniques such as query routing and hierarchical re-
trieval (Xu et al., 2025; Helmi, 2025), such systems struggle
to efficiently scale as data volumes and domains increase.
Beyond scalability, centralization leads to data management
complexity across diverse data sources (Chong et al., 2025),
high privacy risks (Zeng et al., 2024; 2025a), and gover-
nance challenges related to access control and compliance
(Jayasundara et al., 2024; Zeng et al., 2025b). By contrast,
ourdRAG inherently mitigates these issues by distributing

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
retrieval across independent data providers for better scala-
bility and data management.
2.2 Decentralized Retrieval Augmented Generation
Recent studies have explored decentralized solutions to
these limitations through different technical strategies
(Hecking et al., 2025; Yang et al., 2025; Zhou et al., 2025;
Chakraborty et al., 2025). The External Retrieval Interface
framework aims to decentralize the RAG system from a
software perspective by decoupling retrieval, augmentation,
and generation components through standardized protocols,
enabling data providers to maintain full control and keep
data at its source (Hecking et al., 2025). Another notable
approach is federated learning, which enables collaborative
training while preserving data locality. For example, Fe-
dRAG demonstrates privacy-preserving training of RAG
components without raw data exchange (Mao et al., 2025).
De-DSI applies federated learning to train differentiable
search indexes without centralized coordination (Neague
et al., 2024). Unlike these approaches, which require com-
plex training and coordinated parameter sharing, our dRAG
avoids training overhead by operating entirely at inference
time and emphasizes traceability through blockchain log-
ging of source contributions, providing a lightweight yet
trustworthy solution for quality control.
Blockchain is considered a natural choice for building de-
centralized applications, including decentralized RAG sys-
tems. It helps establish a decentralized, immutable ledger
through cryptographic hashing and peer-to-peer consensus
mechanisms, ensuring data integrity and transparency with-
out the need for a central authority. Smart contracts are
Turing-complete programs deployed on the ledger. It pro-
vides guaranteed execution integrity, as its code and state are
verified and validated by the entire decentralized network
via the underlying consensus mechanism, as long as the ma-
jority of the network is honest. The consensus ensures that
the contract logic executes exactly as programmed, with
the resulting state change being recorded on the tamper-
proof blockchain ledger only after achieving network-wide
agreement. This allows the development of robust, transpar-
ent, and tamper-proof decentralized applications (dApps)
that run on a decentralized network with public verifiability,
rather than relying on single/institutional trust. Existing
blockchain-based RAG approaches employ consensus pro-
tocols for decentralized knowledge validation. For example,
some frameworks introduce domain-specific validator net-
works (e.g., using distributed hash tables) (Yu & Sato, 2024)
or permissioned blockchains where expert nodes must re-
view and agree on new knowledge before it is integrated
(Andersen et al., 2025).
Unlike these mechanisms that rely on complex blockchain
protocols to coordinate knowledge propagation, our ap-proach uses smart contracts in a more lightweight manner.
It leverages smart contracts to dynamically evaluate and
prioritize data sources based on their reliability, eliminating
the need for complex protocols with specialized validator
committees or centralized content reindexing. Essentially,
we provide a transparent, tamper-proof trust layer on top of
decentralized retrieval, enabling robust performance even
when some sources are unreliable.
3 SYSTEMARCHITECTURE
3.1 System Components
Decentralized
Blockchain Network
LLM ServiceQuery
AnswerDecentralized Data
SourcesDocumentsScoresFeedback
Score
Update
Figure 2.System overview ofdRAG.
OurdRAG system can be abstracted into three components:
Decentralized Data Sources,LLM Service, andDecentral-
ized Blockchain Network(as shown in Figure 2).
Decentralized Data Sources.The decentralized data
sources in dRAG are maintained individually by each data
owner, and each may provide a standard-compatible API for
the LLM service to query for documents.
Decentralized Blockchain Network.The decentralized
blockchain network serves as the infrastructure for dRAG
to evaluate the reliability of each decentralized data source
(detailed in Section 4). The scoring management is imple-
mented via a smart contract on the Ethereum blockchain,
providing trustworthy external evaluation of decentralized
data sources. The smart contract, which maintains the score-
board for all data sources, is executed by a public, peer-to-
peer blockchain network. The integrity of its execution is
guaranteed by the decentralized nature of blockchain.
LLM Service.The LLM service responds to user queries,
interacting with other system components, including the
smart contract on the blockchain, and the decentralized data
sources to generate the answer. Notably, multiple LLM
services can share the same dRAG infrastructure to serve
different users by connecting and providing feedback to the
same smart contract on-chain.
3.2 System Workflow
Figure 2 presents the system overview of dRAG . Additional
implementation details are provided in Section 5. We as-
sume the LLM Service has direct access to individual data

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
sources, allowing it to retrieve the most relevant documents
through standard-compliant interfaces.
Similar to other RAG systems, users submit queries to the
LLM service to obtain answers from dRAG . The LLM ser-
vice will first sample a subset of data sources based on the
reliability evaluation from Section 4.1, and route the query
to them. The sampled data sources then use the locally
hosted retriever to find the documents most relevant to the
query and return them to the LLM service. The data source
sampling process can be customized to meet the query and
user needs. In dRAG , we present an effective sampling pro-
cess based on a parameter that quantifies the usefulness of
each data source in improving the LLM performance, as
described in Section 4.2. After retrieving the documents
from data sources, the documents will be reranked and con-
catenated with the query before being fed to the LLM. The
reranking process incorporates reliability scores to identify
the most reliable documents for answering the query. The
LLM then generates an answer, and the user provides feed-
back on it to update the reliability scores on the blockchain.
While our system accepts direct user feedback, we also sup-
port automatic evaluation via exact matching to ground-truth
answers (if available). We provide a detailed discussion of
reliability scoring, data source sampling, and reranking in
the next section.
4 EVALUATINGDATASOURCES
RELIABILITY
Algorithm 1Reliability-Guided RAG
1:Inputs:user queryq; data sourcesS={s i}
2:Hyperparameters:number of retrievers N; per-source fetch
M; reranked top-K
3:Initialization:usefulnessU iand reliabilityR ifor sources i
4:foreach queryqdo
5: SampleNsources{s i}proportionally toU i
6: Retrieve Mdocuments from each selected source Xwith
sizeM×N
7: RerankXwith Equation 4; keep top-KdocumentsX 1:K
8: Generate responseyusingX 1:K
9: For each document d∈X 1:Kand sentence x∈d , compute
f(x)via either Equation 1 or Equation 2
10: Compute document score:f(d)←Aggx∈df(x)
11: Update Ui, Rifor each source siwith documents in X1:K
usingf(d)following Equation 3
12:end for
We introduce how dRAG evaluates (§4.1) and updates (§4.2)
data source reliability to improve retrieval and response
quality. We present the whole procedure in Algorithm 1.
4.1 Sentence Importance Evaluation
The utility of retrieved documents varies across different
information needs and temporal contexts (Qian et al., 2024;Uddin et al., 2025). For instance, a Wikipedia article docu-
menting the 2020 U.S. presidential election provides limited
value when addressing queries about the 2024 election. We
consider a data sourcereliableif it consistently provides
informative content that improves LLM generation across
a diverse range of user queries. To quantify this notion of
reliability, we first compute the importance of sentences
from the retrieved document to LLM generation, which can
be evaluated under two scenarios.
Ground-truth answers are unavailable.We estimate the
importance of each sentence xin the retrieved document d
using Monte-Carlo Shapley (MC-Shapley) values (Gold-
shmidt & Horovicz, 2024). Formally, the reliability score
is defined as the expected marginal contribution of xto the
model output across sampled subsets of sentences:
fshapley (x) =E s⊆d\{x}
F(s∪ {x})−F(s)
,(1)
where F(·) is the utility function, which in our setting is
computed as the cosine similarity between the baseline re-
sponse (generated using the full retrieved document) and
the response conditioned on the sampled subset sors∪{x} .
This formulation quantifies the marginal contribution of
each sentence to the model’s output, featuringgreater gen-
eralizability but with increased computational overhead.
Ground-truth answers are available.We adopt an
information-theoretic formulation in which each sentence
is treated as a potential rationale. The informativeness of a
rationale, denoted as finfo(x), is defined as the conditional V-
information (Hewitt et al., 2021) capturing the reduction in
model predictive uncertainty conditioning on the rationale:
finfo(x) =H V(y∗|q)−H(y∗|q∪x),(2)
where y∗andqare the ground-truth answer and query. We
train evaluators followingRORA(Jiang et al., 2024) to
estimate H(·) as a multivariable predictive V-entropy. This
approachimproves computational efficiency but reduces
generalizability to out-of-domain sentences.
4.2 Data Source Reliability Update
Given sentence-level importance scores {f(x) :x∈d}
for a document dretrieved from data source si, we aggre-
gate these scores to obtain a document-level reliability esti-
mate f(d) . The aggregation strategy depends on the score
scale and the corresponding evaluation method: for Shapley-
based scores, we compute the arithmetic mean f(d) =
1
|d|P
x∈dfshapley (x), while for entropy-based scores, we
take the maximumf(d) = max x∈dfinfo(x).
We maintain a cumulative reliability score Rifor each data
source i, which is updated based on user feedback regarding

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
the correctness of the generated response y. The update rule
is straightforward: for documentdbelonging to sources i,
Ri←(
Ri+f(d),ifyis correct
Ri−f(d),ifyis incorrect(3)
This additive update scheme allows the system to accumu-
late evidence about source reliability over multiple interac-
tions, rewarding sources that contribute to correct answers
while penalizing those with errors.
While reliability measures source accuracy, our experiments
show that LLMs occasionally generate responses without
using the retrieved documents. In such cases, a source may
be inherently reliable yet fail to contribute usefully to the
generation. To address this gap, we introduce a comple-
mentary usefulness score Uithat quantifies the extent to
which documents from source iare actually used by the
LLM to generate responses. We evaluate whether generated
responses can be grounded in the retrieved documents and
update Uiaccordingly: (1) If the response isnot grounded,
we penalize the usefulness score as Ui←U i−f(d) , while
leaving Riunchanged. (2) If the responseis grounded,
we reward usefulness Ui←U i+f(d) and update reliabil-
ity score Rifollowing Equation 3. We provide empirical
justification forU iin Section 6.2.
Reliability-Aware Retrieval and Reranking.We incor-
porate these scores at different RAG stages to optimize
retrieval quality. Following the standard RAG pipeline
(Karpukhin et al., 2020; Li et al., 2025), we employ a dense
retrieval system implemented with FAISS (Douze et al.,
2025) for initial candidate selection, followed by a neural
reranker for final document ordering. Specifically, at the
retrieval stage, we use usefulness scores Uito sample N
dense retrievers and their corresponding data sources. At
the reranking stage, we leverage source reliability Rito re-
fine the final document ordering. For a candidate document
retrieved from data sources i, its final reranking score is:
score final= (1−α)·score rerank+α·σ(R i)(4)
where score rerank denotes the original reranking score and
σ(·) is a normalization function (implemented as min–max
normalization in our case). The hyperparameter α∈[0,1]
controls the influence of reliability on the final ranking.
5ON-CHAINRELIABILITYMANAGEMENT
SYSTEM WITHSMARTCONTRACT
In this section, we describe the detailed system implementa-
tion of dRAG with the reliability evaluation from Section 4.
LLM ServiceDecentralized Scoring Management
Reliability-Guided
Source SamplingDocuments
Retrieval
Reliability-Guided
RerankingLLM
Sentence Importance
EvaluationSmart 
Contract Scoreboar d
DS_1 Value_1
DS_2 Value_2
... ...Update
Verify signatures
against data source
Ground
Truth Query
AnswerFeedback
Log
DS_1
Database Retriever
Decentralized Data SourcesScores
(Query ,      )Sampled Results
([Docs],  sig)
([Docs],  sig)
...Feedback Log
Is Answer
Correct?False
Importance
ScoreDS_1: 0.72 
DS_2: 0.15
DS_3: 0.12
Data Source
SignaturesDS_1: 0xf8e92a1bcd45...
DS_2: 0x37a4c8fe019b...
DS_3: 0xd5b09c32a1f7...Figure 3. Detailed Workflow of dRAG , with a feedback log to up-
date the reliability scores. For brevity, state information linking
the update to a genuine query is omitted in the example. Ground
truth is optional for sentence importance evaluations; if it is un-
available, user feedback ( True/False ) will be used to update
the reliability scores.
5.1 On-Chain Trustworthy Reliability Management
with Smart Contract
By distributing the knowledge base and eliminating the
need for a centralized database manager, dRAG requires
decentralized management of source scores. Since the relia-
bility scores are used to evaluate the trustworthiness of data
sources, they must be managed on a public bulletin board
where no single party can tamper with them. Additionally,
the updating of these scores should allow public auditing to
ensure transparency. This design makes the scoring mecha-
nism indRAGmore trustworthy and transparent.
dRAG utilizes blockchain as an infrastructure for decen-
tralized score management and auditable logging. A smart
contract is deployed on the blockchain to facilitate the imple-
mentation of basic scoring functions, including initialization,
retrieval, and update. Every update of the scores will leave
a log on-chain, with traceable information that showcases
the reason for the modification. The blockchain serves as
a public bulletin board to keep the scores, and the smart
contract provides the interface to manage and update the
scores with decentralization-based execution integrity.
Score Initialization.In dRAG , every data source owner will
be registered as an account with an ECDSA (Elliptic Curve
Digital Signature Algorithm) compatible public/secret key
pair(pk, sk) on the blockchain. Every owner of the data
source needs to use the deployed dRAG contract to initialize
a scoring record for their data source on the blockchain. The
contract will create corresponding score records for each
data source, each with an initial score, which will be stored
as a state variable on-chain and be publicly accessible.
Reliability Score Update.After obtaining responses from

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
Algorithm 2Trustworthy Reliability Update with Signature
1:Inputs:State digest state , signatures sig[1..n] , source
IDsSource[1..n] , updated reliability scores R[1..n] , up-
dated usefulness scoresU[1..n], feedback infoinfo
2:t←current timestamp()
3:iflengths ofsig,Source,R,Udifferthen
4:abortwithInvalidUpdateCount
5:end if
6:fori= 1tondo
7:ifscore record forSource[i]does not existthen
8:abortwithDataSourceNotExists
9:end if
10:pki←scoreRecords[Source[i]].sourceAddress
11:if¬ECDSA.verify(state,sig[i],pki)then
12:abortwithInvalidSignature
13:end if
14: Update scoreRecords[Source[i]] :reliabilityScore←
R[i], usefulnessScore←U[i], timestamp←t
15: EmitScoreRecordUpdated event for Source[i]
with new scores,t, andinfo
16:end for
the LLM in dRAG , users can provide feedback on responses
along with ground-truth answers, and update the reliabil-
ity based on their contribution to the reliability records on
the blockchain via a smart contract. The LLM service will
first perform a sentence-level importance evaluation (de-
scribed in Section 4) and calculate score changes based on
the evaluation. Then, the LLM service needs to include the
evaluation in a feedback log transaction and submit it to the
blockchain network so the deployed smart contract can up-
date the scores for each source. To prevent arbitrary updates
to reliability records, the smart contract will perform basic
verification of the feedback log to ensure that updates can
be traced back to a legitimate query sent to the data sources.
Trustworthy Reliability Update with Signature.Al-
gorithm 2 presents the pseudocode for the signature ver-
ification and reliability score update logic of our smart
contract. We also provide a deployed smart contract in-
stance of this algorithm at the public testnet Sepolia at
2at6kd.short.gy/4R9Tbm.
When the user’s query is broadcast to the sampled data
sources to retrieve potentially relevant documents, the LLM
service also generates state information and attaches it to
the query. The state is a digital digest (secure hash) that
includes the query and the reliability scores of all sampled
data sources, which can be viewed as a summary binding
the current query to the current state of all sampled data
source scores. The sampled data sources can obtain the same
information (the query and the reliability scores of all data
sources) to generate and verify if the state matches their
current view. When returning the retrieved documents to theLLM service, each sampled data source must create a digital
signature sigusing its private key sk, indicating its consent
to acknowledge and confirm the state . This signature will
be considered as consent to allow the LLM service to update
the reliability score of the data sources. The LLM service
creates the feedback log, and the signatures obtained from
each data source will be included as part of the feedback log
transaction updating the scores, as shown in Figure 3.
The smart contract will retrieve all relevant reliability
records associated with the owners’ public keys and perform
batch verification of all signatures against the information
included in the transaction, including the query prompt, the
importance evaluation summary, and the reported scoring
state hash from the data sources. The batched verification
of the above information ensures that the feedback log trans-
action, which updates the scores, can be traced back to a
specific user query.
For further security and tracing purposes, like preventing
replay attacks by reusing the signature to maliciously update
the data source reliability scores, one may consider leaving a
log recording the query and sampled data source list (marked
as “unused”) on-chain before the LLM service broadcasts
the query and retrieves the documents from data sources.
When the data sources create the signature, the state should
also include the query log. The first feedback on the query
marks the query log as “used” and invalidates all subsequent
updates that attempt to exploit the same query.
5.2 Auditing Reliability Updates
Whenever a score update occurs for any data
source, the deployed smart contract will emit a
ScoreRecordUpdated event, including the nec-
essary information to trace the update back to the original
query included in the feedback log (as shown in line
14 of Algorithm 2). The event data will be part of the
blockchain but not directly accessible to the smart contract,
helping reduce the network’s computational overhead. The
event data can be queried and retrieved by any blockchain
gateway node, linked to the specific transaction in the
blockchain, allowing anyone to audit the entire history of
data source reliability updates and ensure the update can be
traced back to a genuine query. This information can be
used to reproduce the query for further investigation.
Additionally, our system can also be extended to support
asynchronous feedback and score updates from the users.
This could be useful in cases where ground truth for cer-
tain queries is not immediately available, allowing users
to provide evaluations and update scores at a later time.
Specifically, users still receive signatures from selected
data sources and obtain responses from the LLM. The
sentence-level importance evaluation will be conducted
without ground truth using Equation 1. Users can submit

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
Figure 4. Statistics of two simulated unreliable data environments.
Left: Token-level pollution shows the percentage of ground-truth
tokens replaced with random tokens across six data sources (A-F),
where different sources contain overlapping ground-truth docu-
ments.Right: Document-level pollution shows how 3197 ground-
truth documents are divided into six sources (A-F) and mixed with
polluted documents to create varying reliability levels, with no
document overlap between sources.
feedback with a non-determined importance report to the
blockchain. Once ground truth becomes available, users
can invoke the smart contract to report it on-chain and then
update the reliability scores accordingly.
6 EXPERIMENTS ANDRESULTS
To demonstrate the effectiveness of dRAG , we evaluate
it across two environmental configurations, each contain-
ing 6 data sources with varying pollution levels p∈
{0%,20%,···,100%} , where p= 0% indicates no pollu-
tion (i.e., the data source provides ground-truth documents
for the original Natural Questions dataset (Kwiatkowski
et al., 2019)), and higher values of pindicate increasingly
unreliable documents within the data source. For building
these two environments, we (1) inject noise into the doc-
uments by replacing ground-truth tokens with randomly
sampled tokens based on the given pollution level (token-
level pollution); (2) partition the original dataset into 6
disjoint data sources, where each source is populated with
randomly sampled polluted documents, such that all sources
maintain equal length and achieve their specified pollution
levels without overlap (document-level pollution).1Fig-
ure 4 shows statistics of these two data environments.
For polluted data environments, an effective reliability
evaluation should successfully identify and prioritize the
least polluted (i.e., most reliable) data sources throughout
the process, and therefore a well-designed dRAG system
should outperform its centralized counterpart under
1We use the words “polluted” and “unreliable” interchangeably
hereafter.polluted environments, while ideally approaching the
performance upper bound set by centralized systems in
unpolluted conditions.
We evaluate our dRAG system on LLAMA3.2-3B-
INSTRUCTand LLAMA-3.1-8B-INSTRUCT(Grattafiori
et al., 2024). We train our evaluators using invariant learning
with T5-base (Raffel et al., 2020) to compute Equation 2,
following the RORA approach (Jiang et al., 2024). We
use the following hyperparameters in our main experiments
(for results shown in Table 1): reliability weight α= 0.5
for token-level pollution environments and α= 0.2 for
document-level pollution environments.2We set the sam-
pling number to N= 5 , top-K= 2 , and per-source fetch
M= 3 . For LLM inference, we consistently set the tem-
perature to 0for better reproducibility. We initialize both
usefulness and reliability scores to 10.
We deploy the smart contract used in dRAG on the Sepolia
network, a public Ethereum testnet, together with an easy-
to-use package tool to test and evaluate the core functions of
dRAG . We provide a detailed cost evaluation for maintaining
thedRAGinfrastructure in Section 6.3.
6.1 Centralized versus Decentralized RAG System
dRAG outperforms centralized RAG system in unreliable
data environments.Figure 5a presents a performance
comparison between centralized and decentralized RAG
systems using LLAMA-3.1-8B-INSTRUCTon token-level
polluted data (left in Figure 4). Combined with the results
shown in Figure 1 using LLAMA-3.2-3B-INSTRUCT, these
findings demonstrate thatdRAGexhibits self-improvement
capabilities by progressively learning to prioritize the most
reliable data sources across user queries. The underlying
mechanism for this improvement is illustrated in Figure 5b.
Initially, all sources start with identical reliability scores.
However, data sourceAconsistently receives positive feed-
back as its high-quality documents are retrieved and con-
tribute to generating correct answers. As a result, its reli-
ability score increases while other sources remain largely
unchanged or slightly decline, makingAmore likely to be
retrieved in subsequent queries. This mechanism explains a
key behavioral pattern observed in dRAG when deployed in
unreliable data environments: the system initially performs
at a lower-bound level, equivalent to a centralized RAG
system using polluted data, but steadily improves over time
as reliable sources accumulate positive feedback and begin
to dominate retrieval decisions.
To further validate these improvements, we analyze the us-
age distribution of each data source across queries. Figure 5c
2The lower αin document-level settings accounts for the non-
overlapping document distribution, where a single high αvalue
would bias retrieval toward completely irrelevant data sources.
Further analysis is provided in Section 6.1.

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
200 400 600 800 1000 1200
Question Index30%35%40%45%50%55%60%AccuracyDecentralized (unreliable)
Centralized (unreliable)Centralized (reliable)
(a) System Performance Comparison
0 200 400 600 800 1000 1200 1400
Question Index10.012.515.017.520.022.525.0Reliability ScoresA B C D E F (b) Data Source Reliability
012345678910111213
Query Bins (Size=100)020406080100Usage (%)A B C D E F (c) Data Source Usage
Figure 5. (a)Performance comparison between dRAG and centralized RAG systems on LLAMA-3.1-8B-INSTRUCTacross reliable and
unreliable data environments;(b)Evolution of reliability score Ri(Equation 3) throughout the query process;(c)Percentage distribution
of data sources (A-F) across sequential query bins of 100 queries each.
Models Pollution StrategyCentralized
(Unreliable)dRAG(Unreliable) Centralized
(Reliable)MC-Shapley RORA
LLAMA-3.2-3B-INSTRUCTToken-level 30.56 42.69 49.9044.24Document-level 28.74 30.60 29.90
LLAMA-3.1-8B-INSTRUCTToken-level 37.37 46.58 43.3049.03Document-level 34.38 37.37 37.24
Table 1. Average accuracy of centralized and decentralized ( dRAG ) systems on reliable and unreliable data environments. For dRAG ,
performance is reported after 500 warmup queries to ensure reliability score convergence. Reliability scores are updated using sentence-
level importance estimation via MC-Shapley (Goldshmidt & Horovicz, 2024) or RORA (Jiang et al., 2024) methods. Unreliable data was
polluted using token-level or document-level strategies.
confirms that as dRAG digests more queries (organized into
bins of 100), it learns to prioritize more reliable sources.
Specifically, the usage percentage of data sourceAincreases
substantially from approximately 40% in the first query bin
to over 80% in later bins (bins 6-13). This increase is ac-
companied by a corresponding decrease in the utilization
of less reliable data sourcesB,C, andD, whose combined
usage drops from roughly 60% to under 20%. These results
highlight that dRAG effectively learns to distinguish data
quality and progressively favors the least polluted sources,
thereby improving retrieval performance over time.
dRAG achieves performance comparable to centralized
systems running in ideal, reliable data environments.
Table 1 presents the average accuracy of centralized and
decentralized systems across various models, scoring meth-
ods, and data environments. Our findings demonstrate that
dRAG consistently outperforms the centralized system under
unreliable data environments across all experimental config-
urations. More remarkably, under conditions of token-level
pollution, dRAG achieves performance levels comparable
to those of centralized systems operating on fully reliable
data sources. This aligns with the intuition that prioritizing
high-quality data sources during retrieval fundamentally en-
hances performance, with the upper bound constrained by
the model’s inherent capacity to process fully reliable data.However, we observe a notable performance disparity be-
tween the two pollution strategies. Specifically, unreli-
able environments constructed using document-level pollu-
tion yield lower accuracy compared to those using token-
level pollution. This difference stems from different doc-
ument distributions yielded by two strategies (Figure 4).
Document-level pollution creates data sources that cover
only a subset of the original documents (1066 out of 3197
to answer queries). Consequently, when dRAG promotes
the most reliable sources (e.g.,A) during reranking, it may
inadvertently favor irrelevant documents from these sources
simply based on their high reliability scores. In contrast,
token-level pollution creates a more balanced data environ-
ment where all data sources maintain identical document
coverage to the original corpus, with reliability varying at
the granular token level rather than through selective doc-
ument inclusion. We leave the challenge of handling these
incomplete and unreliable data sources to future work.
6.2 Sensitivity Analysis and Score Validation
Considering that dRAG uses reliability weights αfor bal-
ancing rerank and reliability scores for each data source,
and uses the number of retrievers Nfor sampling from the
usefulness score, we study dRAG ’s sensitivity towards the
choice of these two hyperparameters.

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
0.1 0.3 0.5 0.7 0.91.0
Reliability Weight28.0030.5632.0036.0040.0044.24Accuracy
(a) Reweight Weight Sensitivity
1 2 3 4 5 6
Number of Retrievers24.0030.5636.0042.0044.24Accuracy
 (b) Retriever Number Sensitivity
0 20 40 60 80 100
Pollution Level (%)5.007.5010.00Entropy
MC-Shapley RORA (c) Entropy of Polluted Data Sources
Figure 6. (a-b):Sensitivity test results of dRAG on reliability weight αand number of retrievers N. Results are obtained from a token-level
polluted environment where dRAG uses the MC-Shapley scoring method with LLAMA-3.2-3B-INSTRUCT. The — indicates the baseline
performance of a centralized system on unreliable data (30.56, from Table 1), and the — indicates the upper-bound performance of a
centralized system on reliable data (44.24).(c):Entropy in sentence importance estimation across different pollution levels.
Question:Who got the first Nobel prize in physics?Score
MC-SHAPLEY: ThefirstNobelPrize inPhysics wasawarded in1901 toWilhelm ConradR¨ontgen, ofGermany, who
received 150,782 SEK, which isequal to7,731,004 SEK inDecember2007. John Bardeen istheonly laureatetowin
theprize twice —in1956 and1972. Maria Sklodowska -Curie alsowon twoNobelPrizes, forphysics in1903 and
chem istry in1911.0.3626
0
0
RORA: ThefirstNobelPrize inPhysics wasawarded in1901 toWilhelm ConradR¨ontgen, ofGermany, who
received 150,782 SEK, which isequal to7,731,004 SEK inDecember2007. John Bardeen istheonly laureatetowin
theprize twice —in1956 and1972. Maria Sklodowska -Curie alsowon twoNobelPrizes, forphysics in1903 and
chem istry in1911.1.0806
-1.2246
-2.6737
Table 2. Example sentence-level scores for the first three sentences of the retrieved document, computed by MC-Shapley (using LLAMA-
3.2-3B-INSTRUCT) and RORA for the given question. The scores in the right column correspond to each sentence. The ground-truth
answer is “Wilhelm Conrad R ¨ontgen”. Darker highlighting denotes sentences assigned higher scores by each method.
Reliability weight α.We test dRAG using MC-Shapley
across six reliability weight values on LLAMA-3.2-3B-
INSTRUCTunder token-level polluted data, where a higher
reliability weight results in more aggressive reliability-
driven filtering. Results shown in Figure 6a indicate that
increasing the weight does induce stronger intervention (e.g.,
from 0.1 to 0.5), making the dRAG retrieves documents from
less polluted data sources and improves its performance to-
wards the upper bound. However, exceedingly large reliabil-
ity weights, such as 1.0, can cause the dRAG to over-rely on
reliability scores while overlooking the semantic relevance
captured by neural rerankers, resulting in accuracy decline
to approximately 34%. Notably, across all tested reliability
weights, dRAG consistently outperforms the centralized sys-
tem under identical conditions. This further demonstrates
dRAG ’s robustness in unreliable and noisy data environ-
ments, because accidental retrievals from highly polluted
sources can be overridden by documents from more reliable
sources, which typically receive higher reranking scores
according to Equation 4 and are consequently positioned
later in the context window where they can receive more
attention from LLMs.Number of retrievers N.The number of retrievers N
determines how many data sources are sampled based on
their usefulness scores. A higher value of Nforces dRAG to
retrieve documents from more data sources that have histor-
ically contributed meaningfully to LLM generation (see the
usefulness definition in Section 4.2). We evaluate Nranging
from a single data source to all six available sources, and the
results are shown in Figure 6b. Clearly, increasing the num-
ber of retrievers consistently improves accuracy, elevating
performance from below the lower bound to nearly reaching
the upper bound. This demonstrates that data sources with
higher reliability scores are not necessarily the most useful
ones for LLM generation. By retrieving documents from
multiple sources, dRAG learns from diverse information
across different reliability levels, enabling more accurate
and nuanced updates to the reliability scores. We also notice
a performance drop when we set N= 6 , where all data
sources are included and usefulness-based sampling is prac-
tically disabled. Therefore, this drop can be interpreted as
an ablation result, highlighting the practical importance of
usefulness-based sampling fordRAG.

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
Sentence importance score validation.We validate that
importance score estimation methods, MC-Shapley (Gold-
shmidt & Horovicz, 2024) in Equation 1 and RORA (Jiang
et al., 2024) in Equation 2 can effectively capture the contri-
bution of each retrieved sentence to LLM generation. While
these methods have been previously evaluated against hu-
man annotations in their original works, we test their utility
in our simulated unreliable data environments by analyz-
ing the entropy of the resulting importance scores across
sentences drawn from data sources with varying levels of
pollution. As illustrated in Figure 6c, both MC-Shapley and
RORA exhibit a consistent decrease in score entropy as the
data pollution level increases. This is because token-level
pollution progressively replaces ground-truth tokens with
incorrect ones, leading to uniformly low scores across sen-
tences, thereby reducing entropy. Importantly, this pattern
reflects the methods’ sensitivity to data quality degradation,
confirming their viability as robust indicators for identifying
unreliable information for our dRAG system. We further
provide a real example in Table 2 showing how the two
methods capture the relative importance of each sentence.
6.3 System Cost Evaluation
nTotal Gas Used (Cost) Per-Update Gas (Cost)
1 71,277 ($0.0258) 71,277 ($0.0258)
2 96,352 ($0.0349) 48,176 ($0.0174)
5 211,492 ($0.0766) 42,298 ($0.0153)
10 376,899 ($0.1364) 37,690 ($0.0136)
15 502,565 ($0.1820) 33,504 ($0.0121)
20 628,048 ($0.2274) 31,402 ($0.0114)
Table 3. Gas usage and USD cost for providing feedback and up-
dating the reliability scores at 0.09 gwei/gas and ETH = $4,022.14
(on Oct 29, 2025). nrepresents the number of data sources to
update in the feedback. This mainly reflects the cost of the signa-
ture. Actual gas consumption also depends on the query length
and other factors.
Utilizing decentralized infrastructure like a public
blockchain is typically not free, as it consumes the com-
putational power of the network. In the Ethereum network,
gasis the unit that measures the computational effort re-
quired to execute operations, like running a smart contract.
Each operation incurs a fixed amount of gas, and users pay
a fee based on this total, which compensates network partic-
ipants for the resources they use. Table 3 provides the cost
of the most important part of making dRAG a trustworthy
infrastructure, namely, on-chain verification of feedback
and score updates. As the number of score updates nin-
creases (i.e., the number of queries the dRAG has seen), the
per-update cost decreases, reflecting approximately 55.8%
of gas efficiency gains from our batching operations, drop-
ping from 2.6 cents per update (n=1) to 1.1 cents per update
(n=20). This demonstrates the potential for dRAG to uti-lize batched queries and feedback to update the reliability
evaluation while maintaining a reasonable cost range.
7 CONCLUSION ANDFUTUREWORK
We built dRAG , a decentralized RAG system that addresses
data reliability challenges in real-world settings. Our dRAG
dynamically computed and updated reliability scores for
each data source, securing these scores on a blockchain to
ensure both quality control and trust among data source
owners. Through controlled experiments, we demonstrated
thatdRAG consistently outperforms centralized systems in
unreliable data environments across different models, data
pollution strategies, and reliability scoring methods. No-
tably, our system can even achieve performance comparable
to that obtained in fully reliable environments. To the best
of our knowledge, dRAG is the first blockchain-based de-
centralized RAG system that incorporates quality control
mechanisms to handle noisy, real world-like data. We hope
this work opens new research directions at the intersection
of decentralized systems, RAG, and trustworthy AI.
While dRAG performs effectively when data sources pro-
vide full coverage for user queries (i.e., token-level pollu-
tion), it struggles under more challenging conditions. For
example, when pollution occurs at the document level, its
performance falls considerably short of the upper bound
achieved by centralized systems in fully reliable environ-
ments, as we briefly discussed in Section 6.1. This gap
arises from a fundamental challenge in the data distribution:
each data source provides only partial documents rather than
complete coverage of the query space, such that the most
reliable source may not contain the most relevant documents
for a given query. To address this limitation, a straightfor-
ward solution is to make reliability scores work adaptively
rather than through a single fusion at the reranking stage.
For instance, the system could first retrieve documents by
relevance, then use a learned gate (classifier) to determine
whether to apply reliability-based reranking or preserve the
original relevance ordering for each query. Such an adap-
tive, query-conditioned pipeline, paired with our iterative
source reliability updates, may close the performance gap
in challenging scenarios with fine-grained, non-overlapping
data sources.
Another promising direction for future research is to analyze
the convergence rate of reliability scores in dRAG . Specifi-
cally, it remains unclear how many queries are required for
thedRAG system to reach its performance upper bound, and
which factors (e.g., query diversity, number of data sources,
data source quality, etc.) most significantly influence the
convergence rate of these reliability scores. A systematic
investigation of these convergence dynamics would provide
valuable insights into the system’s scalability and practical
deployment. We leave this exploration to future work.

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
ACKNOWLEDGEMENTS
This work was partially supported by NSF IIS-2119531,
IIS-2137396, IIS-2142827, IIS-2234058, OAC-2312973,
NASA ULI 80NSSC23M0058 and Open Philanthropy. We
also appreciate the support from the Foundation Models
and Applications Lab of Lucy Institute and ND-IBM Tech
Ethics Lab.
REFERENCES
Addison, P., Nguyen, M.-T. H., Medan, T., Shah, J., Man-
zari, M. T., McElrone, B., Lalwani, L., More, A., Sharma,
S., Roth, H. R., Yang, I., Chen, C., Xu, D., Cheng, Y .,
Feng, A., and Xu, Z. C-fedrag: A confidential feder-
ated retrieval-augmented generation system, 2024. URL
https://arxiv.org/abs/2412.13163.
Andersen, T. E., Avalos, A. M., Dagher, G. G., and
Long, M. D-rag: A privacy-preserving framework for
decentralized rag using blockchain.Artificial Intelli-
gence, Soft Computing And Application Trends 2025,
2025. URL https://api.semanticscholar.
org/CorpusID:276622158.
Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., and
Abdelrazek, M. Seven failure points when engineering
a retrieval augmented generation system, 2024. URL
https://arxiv.org/abs/2401.05856.
Chakraborty, A., Dahal, C., and Gupta, V . Federated
retrieval-augmented generation: A systematic mapping
study, 2025. URL https://arxiv.org/abs/
2505.18906.
Chong, Z.-K., Ohsaki, H., and Ng, B. Llm-net: Democra-
tizing llms-as-a-service through blockchain-based expert
networks, 2025. URL https://arxiv.org/abs/
2501.07288.
Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G.,
Mazar ´e, P.-E., Lomeli, M., Hosseini, L., and J ´egou, H.
The faiss library, 2025. URLhttps://arxiv.org/
abs/2401.08281.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., Wang, M., and Wang, H. Retrieval-augmented
generation for large language models: A survey, 2024.
URLhttps://arxiv.org/abs/2312.10997.
Goldshmidt, R. and Horovicz, M. Tokenshap: Interpreting
large language models with monte carlo shapley value
estimation, 2024. URL https://arxiv.org/abs/
2407.10114.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn,
A., Yang, A., Mitra, A., Sravankumar, A., Korenev,
A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A.,
Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang,
B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra,
C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong,
C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D.,
Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary,
D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes,
D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan,
E., Smith, E. M., Radenovic, F., Guzm ´an, F., Zhang, F.,
Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail,
G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Ko-
revaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A.,
Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J.,
Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J.,
Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J.,
Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton,
J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia,
J., Alwala, K. V ., Prasad, K., Upasani, K., Plawiak, K., Li,
K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik,
K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary,
L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L.,
Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat,
L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh,
M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham,
M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M.,
Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N.,
Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N.,
Duchenne, O., C ¸elebi, O., Alrassy, P., Zhang, P., Li, P.,
Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan,
P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan,
R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic,
R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R.,
Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva,
R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S.,
Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang,
S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang,
S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S.,
Collot, S., Gururangan, S., Borodinsky, S., Herman, T.,
Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speck-
bacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V .,
Gupta, V ., Ramanathan, V ., Kerkez, V ., Gonguet, V ., Do,
V ., V ogeti, V ., Albiero, V ., Petrovic, V ., Chu, W., Xiong,
W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang,
X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Gold-
schlag, Y ., Gaur, Y ., Babaei, Y ., Wen, Y ., Song, Y ., Zhang,
Y ., Li, Y ., Mao, Y ., Coudert, Z. D., Yan, Z., Chen, Z.,
Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey,
A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand,
A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A.,
Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A.,
Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poul-
ton, A., Ryan, A., Ramchandani, A., Dong, A., Franco,

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A.,
Bharambe, A., Eisenman, A., Yazdan, A., James, B.,
Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola,
B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock,
B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B.,
Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C.,
Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C.,
Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty,
D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine,
D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang,
D., Le, D., Holland, D., Dowling, E., Jamil, E., Mont-
gomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T.,
Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun,
F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Cag-
gioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz,
G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov,
G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H.,
Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H.,
Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan,
H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I.,
Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli,
J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J.,
Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J.,
Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard, J.,
McPhie, J., Torres, J., Ginsburg, J., Wang, J., Wu, K., U,
K. H., Saxena, K., Khandelwal, K., Zand, K., Matosich,
K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh,
K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg,
L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L.,
Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M.,
Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso,
M., Groshev, M., Naumov, M., Lathi, M., Keneally, M.,
Liu, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel,
M., Vyatskov, M., Samvelyan, M., Clark, M., Macey,
M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari,
M., Bansal, M., Santhanam, N., Parks, N., White, N.,
Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta,
N., Laptev, N. P., Dong, N., Cheng, N., Chernoguz, O.,
Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P.,
Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P.,
Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P.,
Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R.,
Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan,
R., Battey, R., Wang, R., Howes, R., Rinott, R., Mehta,
S., Siby, S., Bondu, S. J., Datta, S., Chugh, S., Hunt, S.,
Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma,
S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay,
S., Feng, S., Lin, S., Zha, S. C., Patil, S., Shankar, S.,
Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe,
S., Chintala, S., Max, S., Chen, S., Kehoe, S., Satter-
field, S., Govindaprasad, S., Gupta, S., Deng, S., Cho,
S., Virk, S., Subramanian, S., Choudhury, S., Goldman,
S., Remez, T., Glaser, T., Best, T., Koehler, T., Robinson,
T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked,T., V ontimitta, V ., Ajayi, V ., Montanez, V ., Mohan, V .,
Kumar, V . S., Mangla, V ., Ionescu, V ., Poenaru, V ., Mi-
hailescu, V . T., Ivanov, V ., Li, W., Wang, W., Jiang, W.,
Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X.,
Wu, X., Gao, X., Kleinman, Y ., Chen, Y ., Hu, Y ., Jia, Y .,
Qi, Y ., Li, Y ., Zhang, Y ., Zhang, Y ., Adi, Y ., Nam, Y ., Yu,
Wang, Zhao, Y ., Hao, Y ., Qian, Y ., Li, Y ., He, Y ., Rait,
Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao,
Z., and Ma, Z. The llama 3 herd of models, 2024. URL
https://arxiv.org/abs/2407.21783.
Hecking, T., Sommer, T., and Felderer, M. An architec-
ture and protocol for decentralized retrieval augmented
generation. In2025 IEEE 22nd International Confer-
ence on Software Architecture Companion (ICSA-C),
pp. 31–35, 2025. doi: 10.1109/ICSA-C65153.2025.
00012. URL https://ieeexplore.ieee.org/
document/11014986.
Helmi, T. Decentralizing ai memory: Shimi, a semantic hier-
archical memory index for scalable agent reasoning, 2025.
URLhttps://arxiv.org/abs/2504.06135.
Hewitt, J., Ethayarajh, K., Liang, P., and Manning, C. Con-
ditional probing: measuring usable information beyond
a baseline. In Moens, M.-F., Huang, X., Specia, L., and
Yih, S. W.-t. (eds.),Proceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pp. 1626–1639, Online and Punta Cana, Domini-
can Republic, November 2021. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2021.emnlp-main.
122. URL https://aclanthology.org/2021.
emnlp-main.122/.
Jayasundara, S. H., Arachchilage, N. A. G., and Russello,
G. Ragent: Retrieval-based access control policy genera-
tion, 2024. URL https://arxiv.org/abs/2409.
07489.
Jiang, Z., Lu, Y ., Chen, H., Khashabi, D., Van Durme,
B., and Liu, A. RORA: Robust free-text rationale
evaluation. In Ku, L.-W., Martins, A., and Srikumar,
V . (eds.),Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 1070–1087, Bangkok, Thailand,
August 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.acl-long.60. URL https:
//aclanthology.org/2024.acl-long.60/.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense pas-
sage retrieval for open-domain question answering. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pp. 6769–6781, Online, November 2020. Association
for Computational Linguistics. doi: 10.18653/v1/

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
2020.emnlp-main.550. URL https://www.aclweb.
org/anthology/2020.emnlp-main.550.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M.,
Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and
Petrov, S. Natural questions: A benchmark for question
answering research.Transactions of the Association for
Computational Linguistics, 7:452–466, 2019. doi: 10.
1162/tacl a00276. URL https://aclanthology.
org/Q19-1026/.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., Riedel, S., and Kiela, D. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. In Larochelle, H.,
Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.),
Advances in Neural Information Processing Systems,
volume 33, pp. 9459–9474. Curran Associates, Inc.,
2020. URL https://proceedings.neurips.
cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.
pdf.
Li, H., Su, Y ., Cai, D., Wang, Y ., and Liu, L. A survey on
retrieval-augmented text generation, 2022. URL https:
//arxiv.org/abs/2202.01110.
Li, S., Stenzel, L., Eickhoff, C., and Bahrainian, S. A. En-
hancing retrieval-augmented generation: A study of best
practices. In Rambow, O., Wanner, L., Apidianaki, M.,
Al-Khalifa, H., Eugenio, B. D., and Schockaert, S. (eds.),
Proceedings of the 31st International Conference on Com-
putational Linguistics, pp. 6705–6717, Abu Dhabi, UAE,
January 2025. Association for Computational Linguis-
tics. URL https://aclanthology.org/2025.
coling-main.449/.
Mao, Q., Zhang, Q., Hao, H., Han, Z., Xu, R., Jiang, W.,
Hu, Q., Chen, Z., Zhou, T., Li, B., Song, Y ., Dong, J.,
Li, J., and Yu, P. S. Privacy-preserving federated embed-
ding learning for localized retrieval-augmented genera-
tion, 2025. URL https://arxiv.org/abs/2504.
19101.
Neague, P., Gregoriadis, M., and Pouwelse, J. De-
dsi: Decentralised differentiable search index. In
Proceedings of the 4th Workshop on Machine Learn-
ing and Systems, EuroMLSys ’24, pp. 134–143, New
York, NY , USA, 2024. Association for Computing
Machinery. ISBN 9798400705410. doi: 10.1145/
3642970.3655837. URL https://doi.org/10.
1145/3642970.3655837.
Qian, X., Zhang, Y ., Zhao, Y ., Zhou, B., Sui, X., Zhang,
L., and Song, K. TimeR4: Time-aware retrieval-augmented large language models for temporal knowl-
edge graph question answering. In Al-Onaizan, Y .,
Bansal, M., and Chen, Y .-N. (eds.),Proceedings of
the 2024 Conference on Empirical Methods in Nat-
ural Language Processing, pp. 6942–6952, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
394. URL https://aclanthology.org/2024.
emnlp-main.394/.
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang,
S., Matena, M., Zhou, Y ., Li, W., and Liu, P. J. Ex-
ploring the limits of transfer learning with a unified
text-to-text transformer.Journal of Machine Learning
Research, 21(140):1–67, 2020. URL http://jmlr.
org/papers/v21/20-074.html.
Uddin, M. N., Saeidi, A., Handa, D., Seth, A., Son, T. C.,
Blanco, E., Corman, S., and Baral, C. UnSeenTimeQA:
Time-sensitive question-answering beyond LLMs’ mem-
orization. In Che, W., Nabende, J., Shutova, E., and
Pilehvar, M. T. (eds.),Proceedings of the 63rd Annual
Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pp. 1873–1913, Vienna,
Austria, July 2025. Association for Computational Lin-
guistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/
2025.acl-long.94. URL https://aclanthology.
org/2025.acl-long.94/.
Wang, Z., Teo, S. X., Ouyang, J., Xu, Y ., and Shi, W.
M-rag: Reinforcing large language model performance
through retrieval-augmented generation with multiple
partitions, 2024. URL https://arxiv.org/abs/
2405.16420.
Xu, C., Gao, L., Miao, Y ., and Zheng, X. Distributed
retrieval-augmented generation, 2025. URL https://
arxiv.org/abs/2505.00443.
Yang, Y ., Chai, H., Shao, S., Song, Y ., Qi, S., Rui, R.,
and Zhang, W. Agentnet: Decentralized evolutionary
coordination for llm-based multi-agent systems, 2025.
URLhttps://arxiv.org/abs/2504.00587.
Yu, J. and Sato, H. Derag: Decentralized multi-source rag
system with optimized pyth network. In2024 IEEE In-
ternational Symposium on Parallel and Distributed Pro-
cessing with Applications (ISPA), pp. 106–115, 2024.
doi: 10.1109/ISPA63168.2024.00022. URL https://
ieeexplore.ieee.org/document/10885343.
Zeng, S., Zhang, J., He, P., Liu, Y ., Xing, Y ., Xu, H., Ren,
J., Chang, Y ., Wang, S., Yin, D., and Tang, J. The
good and the bad: Exploring privacy issues in retrieval-
augmented generation (RAG). In Ku, L.-W., Martins,
A., and Srikumar, V . (eds.),Findings of the Association

A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain
for Computational Linguistics: ACL 2024, pp. 4505–
4524, Bangkok, Thailand, August 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.
findings-acl.267. URL https://aclanthology.
org/2024.findings-acl.267/.
Zeng, S., Zhang, J., He, P., Ren, J., Zheng, T., Lu, H., Xu,
H., Liu, H., Xing, Y ., and Tang, J. Mitigating the privacy
issues in retrieval-augmented generation (rag) via pure
synthetic data, 2025a. URL https://arxiv.org/
abs/2406.14773.
Zeng, Z., Liu, J., Chiang, M.-F., He, J., and Zhang, Z.
S-RAG: A novel audit framework for detecting unau-
thorized use of personal data in RAG systems. In
Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T.
(eds.),Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pp. 10375–10385, Vienna, Austria, July
2025b. Association for Computational Linguistics. ISBN
979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.
512. URL https://aclanthology.org/2025.
acl-long.512/.
Zhou, W., Yan, Y ., and Yang, Q. Dgrag: Distributed
graph-based retrieval-augmented generation in edge-
cloud systems, 2025. URL https://arxiv.org/
abs/2505.19847.
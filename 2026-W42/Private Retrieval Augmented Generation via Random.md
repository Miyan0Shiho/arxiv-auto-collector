# Private Retrieval Augmented Generation via Random Projection

**Authors**: Dixi Yao, Tahseen Rabbani, Tian Li

**Published**: 2026-10-15 17:45:51

**PDF URL**: [https://arxiv.org/pdf/2610.22297v1](https://arxiv.org/pdf/2610.22297v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs) by querying external structured knowledge. However, it can also introduce privacy risks by leaking sensitive information from the retrieval datastore. We propose a simple method to preserve datastore privacy in RAG systems via random projection. By applying the same projection to both datastore embeddings and query embeddings, our method provably preserves semantic similarity between queries and retrieved items while substantially mitigating data extraction attacks. Across multiple RAG architectures and datasets, we show that this lightweight approach achieves superior retrieval and generation performance compared to prior methods which leverage formal differential privacy (DP) guarantees, while exhibiting comparable empirical privacy under strong attack models. Our results for the first time suggest that random projection can serve as a competitive and practical baseline for privacy-preserving RAG systems.

## Full Text


<!-- PDF content starts -->

PRIVATERETRIEVALAUGMENTEDGENERATION VIARANDOM
PROJECTION
A PREPRINT
Dixi Yao *Tahseen Rabbani Tian Li
University of Chicago
ABSTRACT
Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs)
by querying external structured knowledge. However, it can also introduce privacy risks by leaking
sensitive information from the retrieval datastore. We propose a simple method to preserve datastore
privacy in RAG systems via random projection. By applying the same projection to both datastore
embeddings and query embeddings, our method provably preserves semantic similarity between
queries and retrieved items while substantially mitigating data extraction attacks. Across multiple
RAG architectures and datasets, we show that this lightweight approach achieves superior retrieval
and generation performance compared to prior methods which leverage formal differential privacy
(DP) guarantees, while exhibiting comparable empirical privacy under strong attack models. Our
results for the first time suggest that random projection can serve as a competitive and practical
baseline for privacy-preserving RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) [ 1,2] enhances large language models (LLMs) by incorporating information
retrieved from external knowledge. However, recent studies show that carefully crafted prompts [ 3,4,5,6] can extract
sensitive information such as personal identity information from the external datastore.
Since attack queries can closely resemble legitimate user queries, simple access control may block ordinary users [ 3,4].
Recent work has begun to explore privacy risks in RAG, with a leading focus on differential privacy (DP) [ 6,7].
Although DP methods provide formal statistical guarantees, they often degrade output quality (e.g., leading to high
perplexity) due to DP’s fundamental properties of worst-case guarantees, and are usually complicated to implement in
practice. To this end, we propose private RAG via random projection to defend against data extraction attacks.
Our key idea is that differential privacy guarantees may be an overkill to prevent state-of-the-art attacks practically,
whereas (some specific) random projection matrices naturally preserve pairwise similarities of the inputs in the lower-
dimensional space. The extra randomness alters datastore embeddings so attackers may not recover the original text.
This enables users to obtain accurate answers from the RAG system while adversaries fail. We apply a single projection
step to datastore embeddings in the offline stage and the same matrix to user queries in the online stage.
In this work, we consider two RAG architectures: ‘KNN-LM’ [ 1] and ‘Standard RAG’ [ 2]2. In KNN-LM, the system
linearly combines the outputs of the language model and the k-nearest neighbour outputs (usually organized as a
softmax over cosine similarities between queries and the datastore embeddings), and generates next token. In Standard
RAG, the system concatenates retrieved texts with the query and feeds them to the LLM. In this work, we focus on the
settings where formal privacy guarantees are not desired, and thus evaluate privacy leakage by exploring the success
rates of strong, existing attacks empirically.
∗Correspondence to: Dixi Yao<dixi@uchicago.edu>
2We note that there exist various RAG architectures, and we use the naming ‘Standard RAG’ just to contrast with the less popular
KNN-LM architecture.
arXiv:2610.22297v1  [cs.AI]  15 Oct 2026

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
In Section 4, we empirically show that enforcing differential privacy guarantees may not provide better practical protec-
tion against data extraction and can instead harm utility. In summary, our RP technique significantly improves utility
(over 50% on average across datasets) over the baseline methods without degrading empirical privacy performance.
2 Related Works
Private RAG.Prior work shows that various RAG architectures can reveal sensitive personal data (e.g., emails,
phone numbers, and URLs) from the external datastore [ 3,5,8,9,10]. Existing defenses, such as DP-based sampling
and aggregation [ 6,11] and synthetic data [ 12], reduce efficiency on large datasets and often require extra validation
or training. Moreover, Zhao and Zhang [13] show that synthetic data alone does not fully prevent memorization or
leakage. He et al. [14] apply local differential privacy to sensitive content like addresses. However, privacy can involve
entire sentences or paragraphs beyond tokens, making simple masking insufficient. Thareja et al. [15] mix original and
sanitized documents, which requires identifying all private tokens and is impractical for large-scale RAG.
Privacy-Preserving Random Projection.Certain random projection operations are a common technique for re-
ducing dimensionality while preserving some similarity measures with high probability (e.g., based on the Johnson-
Lindenstrauss lemma [ 16]). Prior works have shown that methods built on top of these projections can achieve
differential privacy (DP) for specific applications [ 17,18,19,20,21,22,23,24,25]. In this work, we show that random
projection can be effective for a variety of RAG scenarios as well.
3 Private RAG via Random Projection
3.1 Threat Model
Online inference (API) Offline phase
Query q Datastore d1, …,dn
Tokenize & embed Tokenize & embed
Random projection R Random projection Rx(e1, …,en) same secret R
Similarity search: si= Similarity(Rx,Rei)Rx(Re1, …,Ren)
KNN-LMLLM q Softmax
λpLLM+ (1−λ)pkNN
Next tokensipLLMpkNNStandard RAG
Top-kdocument chunksLLM q
Answerconcatenate
Figure 1: Overview of Private RAG via RP.As illustrated in Figure 1, our algorithm is compatible
with both RAG architectures. In this work, we evaluate
our method as an empirical defense against data extrac-
tion attacks. We consider a black-box setting in which the
attacker has access only to model API responses, while
model parameters, the random projection matrix, logits,
embeddings, and latent representations remain inaccessi-
ble [26,27]. The attacker submits queries to the RAG sys-
tem using the API service and attempts to extract sensitive
information in the datastore from the generated response
through one round or multiple rounds. This is a common
protocol in modern RAG applications where companies
provide consulting service through LLM with an internal
datastore while only text answers are exposed to users.
The threat model also aligns with the threat scenarios
considered in prior work on RAG privacy [3, 4, 28].
In short, we assume the attacker knows that RP is used but does not have direct access to the specific random matrices,
with the projection matrix remaining undisclosed. Attackers are free to conduct repeated-query (Section 4.2) or
multi-round adaptive attacks (Section 4.3). As network- and system-level attacks are out of scope and access to
embeddings is restricted, we do not consider white-box attacks, embedding inversion, projection-matrix inversion, or
side-channel attacks requiring side information or access beyond the LLM’s text responses.
To provide a clearer understanding of what constitutes sensitive information, we show examples of leakage with and
without sensitive content. In Table 1, we observe that leakage of sensitive information exposes critical details such as
names, addresses, phone numbers, and email addresses. In contrast, non-sensitive information reveals no useful details.
For example, a generic URL or repeated mentions of “Queen St” are too common and lack identifying information.
3.2 Algorithm
LetD={d 1, . . . , d n}be the corpus of nnatural language documents (e.g., email threads or case paragraphs). Let f(·)
denote the tokenizer and embedding layers, which can be further fine-tuned to map these di’s (texts) to embeddings.
We assume a fixed pre-trained encoder throughout the paper for simplicity. We also assume that an attacker interacts
with the RAG system through API interactions and aims to extract data with constructed queries.
2

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Algorithm 1: Private RAG with Random Projection
(Standard RAG)
Require: Datastore Dwithndocuments; embedding func-
tionf:Text→Rd
Input:User queryq
Output:Generated answera
Parameter:Projection dimensionk
Offline preprocessing
1: SampleR∈Rk×d,Rij∼ N(0,1/k)
2: Compute embeddingsE D∈Rd×n,ei=f(d i)
3: Project: ˜ED=RE D
Online inference (per query)
4: Initializea← ∅
5: Compute query embedding:x=f(q)
6: Project: ˜x=Rx
7: Calculate similarity between ˜xand ˜ED
8: Retrieve most relevant text:y
9:q←q∥y
10:a←LM(q)
11:returnaWe sample an IID Gaussian matrix R∈Rk×d, where each
element follows N(0,1/k) . We precompute the projected data-
store embeddings offline by computing e′
i=Re ifor each
document embedding ei=f(d i)∈Rd. We perform this
preprocessing step once and store the results, introducing no
runtime overhead during inference. For any query/prompt,
we encode it as x=f(prompt) and compute its projection
x′=Rx , which requires only a single matrix-vector mul-
tiplication. We then perform retrieval in the k-dimensional
space by finding the nearest neighbors to x′among {e′
i}i∈[n]
and computing a softmax over their cosine similarities with x′.
Finally, we combine this softmax with the parametric LLM’s
logits for xin a weighted manner to select the next token. We
repeat generation until reaching the maximum token limit or
generating an end-of-sequence token.
Algorithm 2 (in Section D) and Algorithm 1 illustrate the im-
plementation of our random projection method. The overall
process follows the standard RAG pipeline: we first tokenize
and embed the datastore offline, and store the representations
in FAISS or another efficient indexing framework to enable fast
retrieval at inference time. During inference, the model performs autoregressive generation via iterative next-token
prediction.
For standard RAG (Algorithm 1), we compute the similarity between the projected query embedding ˜xand the projected
datastore embeddings ˜ED, and retrieve the most relevant text based on this similarity. The retrieved text is concatenated
with the original query to form an augmented prompt, which is then passed to a parametric LLM to generate the final
output. The algorithm of KNN-LM is summarized in Section D.
Privacy Implications.Random projection preserves privacy by altering the retrieval process: given a query, vanilla
RAG retrieves the highest-scoring document and generates a corresponding token, which may contain sensitive
information. After RP, embeddings that are close to the original top match can be selected with non-negligible
probability (Section C). As a result, alternative tokens may replace the originally retrieved token.
Utility Discussions.A natural question arises:if random projections alter the retrieved neighbors, how does the
model remain accurate?The reason is that random projection approximately preserves embedding distances, ensuring
retrieval still selects relevant information. This has been extensively studied in prior literature [ 16]. In Section B, for
completeness, we provide proof that our random projection preserves L2distance and cosine similarity. Section B.2
shows a study with synthetic data consistent with the theoretical analysis. For DP-based baselines, achieving privacy
guarantees sacrifices utility. DP introduces a fundamental tradeoff: a small clipping bound overly distorts embeddings,
while a large bound increases noise magnitude, both degrading retrieval utility. We illustrate this in Section 4.6.
Table 1: We focus on sensitive-information leakage and show examples of sensitive and non-sensitive information.
Leakage Information Sensitive Information Non-Sensitive Information
URL with Personal Informa-
tion (e.g., Token ID)If you prefer to receive a hardcopy of your pay stub, click on the link
below to complete a request form. Please click on the link below.
http://etcsurvey.corp.enron.com/wi/p0810709/i.asp?r=19690
&s=OUULGRTU&l=9Suggestions and feedback are welcome at https://www.reddit.com
Phone America Square, London, EC3N 1PQ Tel: +44(0)171 8394444 Fax:
+44(0)171 8394444If you have any comments or suggestions contact us at !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Personal Address If you have any questions, please do not hesitate to call. Salle E. Yoo
Davis Wright Tremaine One Embarcadero Center,Suite 600San
Francisco, CA94111The applicant, Mr Richard Owens, is a British national who was
born in 1944 and lives in Queen St.Queen St.Queen St.Queen St.
Name The applicant, MrMichael Boyce, is an Irish national, who lives in
Mullingar, Ireland. He was represented before the Court by MrP.J.
Groarke, a lawyer practising in County Longford, Ireland.The applicant, MrD.H., who was born in 1966 and lives in H.
3

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
2 3 4 5 6051015 Items Leaked
41424344210220230
2 3 4 5010 Items Leaked
0 50 100 1500255075 Privacy Leakage (%)
220 240 620 660 0 10002040 Privacy Leakage (%)
500 600 5000 7500260270
3 4 5 6 7050100 Items Leaked
350360
3 5 70100200 Items Leaked
24 28Perplexity
(a) KNN-LM (Enron)Perplexity
(b) Standard RAG (Enron)
Perplexity
(c) KNN-LM (ECHR)Perplexity
(d) Standard RAG (ECHR)Perplexity
(e) KNN-LM (C4)
Perplexity
(f) Standard RAG (C4)Parametric LLM
RAG
DP-RP-G, =1
DP-RP-G, =5
DP-RP-G, =10
DP-RP-G, =20
DPSparseVoteRAG, =1
DPSparseVoteRAG, =5
DPSparseVoteRAG, =10
DPSparseVoteRAG, =20
RP (Ours)
Figure 2: Utility/privacy comparisons of our method (RP) with the two DP baselines (DP-RP-G and DPSparseV oteRAG)
on two RAG architectures (denoted as KNN-LM and Standard RAG) and targeted attacks on Enron Email and ECHR
and untargeted attacks on C4. εdenotes various privacy budgets for the DP approaches. The x-axis is perplexity of
generated content, and the y-axis is percentage of data leakage. RP (Ours) achieves a superior utility–privacy trade-off,
with perplexity approaching the unprotected RAG baseline while maintaining empirically near-zero privacy leakage
comparable to the standalone parametric model without retrieval-augmentation.
Table 2: Full privacy-utility comparison across datasets and methods under the repeated-query attack. Utility is
measured using perplexity (PPL) and QA F1 scores. For Enron Email and C4, privacy leakage is reported as the number
of leaked instances, whereas for ECHR, it is reported as a percentage.
Method Dataset / Model Metric Param. RAGDP-RP-G DPSparseV oteRAGRP (Ours)
ϵ=1ϵ=5ϵ=10ϵ=20ϵ=1ϵ=5ϵ=10ϵ=20
KNN-LMEnron Email / GPT2PPL↓3.31 2.87 5.85 3.54 5.13 2.91 42.17 4.22 2.96 2.83 2.89
Leakage (#)↓0 13 0 0 0 0 0 0 0 2 0
ECHR / Llama3-1BPPL↓631.09 2.90 214.54 99.28 94.87 112.85 228.59 109.64 121.82 92.26 46.15
F1↑25.70 90.32 53.25 51.30 54.02 55.13 41.67 41.67 40.32 45.38 87.70
Leakage (%)↓32.82 82.72 2.76 2.41 2.81 2.69 1.75 37.86 38.68 38.82 2.26
ECHR / Qwen3-4BPPL↓1.47 1.55 2.26 2.28 2.28 2.30 3130.89 610.10 500.13 277.62 1.55
F1↑90.18 90.32 80.35 80.32 80.18 82.39 52.40 51.37 51.94 52.98 89.32
Leakage (%)↓2.03 19.08 13.45 13.21 14.85 15.39 2.44 2.46 2.21 2.24 1.94
C4 / Llama3-8BPPL↓2.76 3.16 5.98 5.47 4.12 4.38 6.54 5.91 5.39 4.51 2.98
Leakage (#)↓0 269 0 0 17 25 0 3 36 124 17
Standard RAGEnron Email / GPT2PPL↓– 1.58 3.42 3.23 5.24 2.91 3.34 3.71 3.61 1.62 2.05
Leakage (#)↓– 217 0 0 8 5 0 0 0 9 3
ECHR / Llama3-1BPPL↓– 9.84 512.66 104.70 95.72 79.72 4398.91 7780.70 4922.52 5264.40 47.92
F1↑– 97.18 51.39 54.31 54.12 54.79 1.72 43.72 42.42 43.33 66.70
Leakage (%)↓– 37.92 2.03 1.87 2.15 1.96 1.77 2.05 2.05 19.48 2.16
ECHR / Qwen3-4BPPL↓– 1.19 2.50 2.49 2.51 2.50 1.74 1.43 1.62 1.38 1.19
F1↑– 97.32 85.32 86.17 86.43 85.19 88.35 91.03 93.48 94.32 95.43
Leakage (%)↓– 21.30 25.43 14.36 17.07 15.09 2.84 2.88 2.86 3.32 2.05
C4 / Llama3-8BPPL↓– 2.83 28.41 27.32 26.54 23.12 5.73 5.46 4.65 4.12 2.96
Leakage (#)↓– 358 0 0 15 28 0 33 79 203 23
4 Evaluation
4.1 Settings
We focus on empirical data extraction attacks [ 28] rather than differential privacy bounds, considering both repeated
single-query and multi-round query attacks. We evaluate four datastore/LLM pairs: Enron Email [ 29] with GPT-2 [ 30],
European Court of Human Rights (ECHR) with LLAMA3-1B-INSTRUCT, C4 with LLAMA3-8B-INSTRUCT, and
HotpotQA [31] with QWEN2-7B-INSTRUCT. For utility, we measure perplexity [28] and task-specific performance.
4

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
For simplicity, we denote our random projection as RP, DP-guaranteed random projection with Gaussian noise as DP-RP-
G [17], and the work of Koga et al. [6]as DPSparseV oteRAG. We also report results for vanilla RAG and the parametric
model without a datastore. For differential privacy, we use a wide range of privacy budgets with ϵ∈ {1,5,10,20} . For
DP-RP-G, the original DP-RP method by Blocki et al. [17] assumes binary vector embeddings. To adapt this framework
to RAG and document retrieval, we first project both queries and datastore embeddings into a lower-dimensional space,
clip the projected embeddings, and then add Gaussian noise to obtain differential privacy guarantees. The complete
procedure is in Section D. We sweep the clipping bound from {0.1,0.5,1,5,10,15,25,50,100} under each setting
and report the results with optimal clipping bound. Remaining settings are in Section F. For DPSparseV oteRAG, we
sweep the subset numbermover{10,20,30,40,50,60}and choose the best value50(Section F.2).
In our methods, for KNN-LM, we set λas 0.1 and Kas 1024. For RP, the only tunable parameter is k(the projection
dimension). Larger kvalues better preserve utility while maintaining privacy protection. We choose k= 100 since it
achieves comparable utility to larger values (e.g., k= 1600 ) while providing better computational efficiency. More
detailed analysis is in Section 4.4.
4.2 Repeated-Query Attacks
Table 3: Number of leaked email addresses, URLs, and
phone numbers with different methods on Enron Email.
KNN-LM Standard RAG
Email URL Phone Email URL Phone
Parametric 0 0 0 0 0 0
RAG 0 13 0 36 56 125
RP (Ours) 0 0 0 0 3 0
DP-RP-G,ϵ= 10 0 0 0 0 0
DP-RP-G,ϵ= 50 0 0 0 0 0
DP-RP-G,ϵ= 100 0 0 0 8 0
DP-RP-G,ϵ= 200 0 0 0 1 4
DPSparseV oteRAG,ϵ= 10 0 0 0 0 0
DPSparseV oteRAG,ϵ= 50 0 0 0 0 0
DPSparseV oteRAG,ϵ= 100 0 0 0 0 0
DPSparseV oteRAG,ϵ= 200 2 0 0 8 1We launch repeated-query targeted attacks on Enron Email
for personal identity information (PII) and ECHR for legal
documentsand untargeted attacks on raw C4 content. At-
tackers can repeately inject the same query multiple times.
To make repeated-query extraction attack stronger, we in-
ject dataset-specific prior knowledge into malicious queries,
such as asking for emails when attacking the Enron Email
dataset. For each entry, we perform 10 times repeated ex-
traction attempts, and count any successful extraction as
leakage. We assume the system flags and blocks queries
repeated more than 10 times. Details, including query con-
struction, are provided in Section E.
Figure 2 gives an overview of the privacy/utility trade-off
of different methods under varying privacy budgets on three
datasets under repeated-query attack while Table 2 gives details. Standard RAG achieves higher utility but poses greater
risks of privacy leakage. Without protection, RAG leaks over 200 personal items in the Enron dataset. In contrast, our
methods maintain utility close to RAG while leaking almost no private information. For DP-RP-G, we achieve strong
privacy across all budgets, but utility remains unsatisfactory. DPSparseV oteRAG (DPSVR) shows that increasing the
privacy budget reduces perplexity but increases leakage, lacking a balanced trade-off point. Overall, our approach gives
strong privacy with minimal utility loss.
Table 4: Membership leakage rate. RP (Ours)
can achieve approximately the same perfor-
mance against MIA as parametric, which means
the attacker barely identify membership infor-
mation from RAG datastore.
Method Enron EmailECHR
(Qwen3-4B) C4 HotpotQA
Parametric 0.001 0.006 0.003 0.001
RAG 0.282 0.996 0.843 0.913
RP (Ours) 0.003 0.007 0.003 0.001Our evaluation covers both structured and unstructured sensitive
information to assess the method’s generalizability. On the Enron
dataset, we evaluate protection against structured data including email
addresses, phone numbers, and URLs. As shown in Table 3, random
projection effectively protects all three types, with leakage rates
below 3% across both KNN-LM and Standard RAG architectures.
We observe that URLs are the easiest to leak. For unstructured
text, we evaluate on the ECHR dataset, which contains legal case
narratives with sensitive personal information embedded in natural
language. Our method maintains strong privacy protection (2.19–
2.27% leakage) while preserving utility, demonstrating that random
projection works across both structured and unstructured sensitive
information. This suggests that the privacy-preserving mechanism operates at the embedding level, making it agnostic
to the format or structure of the sensitive content.
A further extension of the repeated-query attack is the membership inference attack (MIA). MIA is important in RAG
privacy, but it targets a different adversarial objective from the data extraction attacks studied in this paper: determining
whether a known record is contained in the datastore, so we do not focus on evaluating various MIA methods. We
evaluate this threat over the full private dataset and measure attack accuracy following E-MIA [ 32], which follows the
standard MIA protocol for RAG systems under our threat model. As shown in Table 4, RP provides strong protection
against E-MIA, achieving empirically near-zero membership leakage.
5

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
4.3 Multi-Round Adaptive Attacks
Table 5: Privacy results against multi-round attacks
on HotpotQA using the KNN-LM framework.
Jiang et al. [9] Yao et al. [33]
CRR SS EED Rounds CRR SS EED
Parametric 30% 0.17 0.030 2.18 37% 0.033 0.07
RAG 60% 0.16 0.028 2.05 75% 0.034 0.10
RP (Ours) 3% 0.16 0.88 2.05 0% 0.032 0.93
DP-RP-G (ϵ= 10) 3% 0.17 0.89 2.01 0% 0.031 0.90
DPSVR (ϵ= 10) 2% 0.17 0.88 2.34 0% 0.034 0.90Another strong attack is the multi-round adaptive attack [ 9],
where an attacker iteratively refines adversarial prompts based
on outputs from previous rounds and aggregates the extracted
information to infer private content. Yao et al. [33] further
constructs a knowledge graph from outputs across attack rounds
to generate future adversarial queries. The attack budget is set
to a maximum of 200 queries to the RAG system.
For both attack methods, we evaluate the reconstructed con-
tent using semantic similarity (SS) and extended edit distance
(EED). Following the experimental setup of Jiang et al. [9], we
construct the datastore using chunks of 1500 words with a maximum overlap of 300 words. We report the chunk
recovery rate (CRR), considering a chunk recovered if the reconstructed content overlaps with the original chunk by
more than 1200 words. For Jiang et al. [9], we report the average rounds required to satisfy the termination condition.
Table 6: Utility results of different algorithms on Hot-
potQA using the KNN-LM structure.
Ans Sup JointCos.EM F1 EM F1 EM F1
Parametric 7.48 28.74 21.37 49.62 7.13 28.55 0.73
RAG 12.48 29.88 25.38 59.74 10.84 32.06 0.75
RP (Ours) 11.32 28.57 25.22 57.14 9.12 34.09 0.75
DP-RP-G,ϵ= 10.08 7.10 9.54 30.42 0.15 9.03 0.54
DP-RP-G,ϵ= 50.21 9.32 10.35 27.82 0.16 9.85 0.53
DP-RP-G,ϵ= 103.58 10.84 9.71 28.93 1.19 7.34 0.54
DP-RP-G,ϵ= 204.56 9.81 14.95 31.23 1.67 8.32 0.61
DPSparseV oteRAG,ϵ= 10.02 8.12 13.98 23.12 0.00 13.12 0.51
DPSparseV oteRAG,ϵ= 53.31 11.23 12.88 30.64 1.31 14.29 0.52
DPSparseV oteRAG,ϵ= 107.48 18.18 14.39 28.85 1.54 16.67 0.54
DPSparseV oteRAG,ϵ= 207.48 18.18 14.38 28.85 1.54 15.38 0.54We launch both multi-round targeted attacks on HotpotQA for
document information. For meausing the utility of RAG system
on usual queries. We report exact match (EM) and F1 scores
between the generated output and the ground truth. For F1,
precision is the number of matching tokens divided by the
total generated tokens, and recall is the number of matching
tokens divided by the ground-truth tokens.Ansdenotes the final
generated answer, andSupdenotes the retrieved supporting
documents, measuing all tokens in the documents. For the
parametric model, we prompt it to generate supporting facts
to enable fair comparison, as the datastore may overlap with
pre-training data.Jointindicates cases where we put answer
and supporting documents together to measure.
From Table 5, we observe that the three privacy-preserving
methods achieve similar utility, while our method attains the
highest accuracy. All methods achieve similar semantic similarity between the original datastore content in each chunk
and the attacked results, which we attribute to LLMs attempting to generate correct answers. In contrast, a high chunk
recovery rate (i.e., how much datastore content is completely restored) and low EED indicate successful extraction
attacks. Examples of extracted content are shown in Table 7. As shown in Table 6, random projection preserves cosine
similarity between retrieved documents and ground truth, indicating successful retrieval. In contrast, differential privacy
methods degrade retrieval performance. Looser privacy budgets can improve utility but raise privacy concerns.
We present a case study of results under multi-round attacks. From Table 7, we can see thatPablo Traperoand
Aleksander Fordare originally mentioned in the question, so it is reasonable that the responses also contain these two
names. However, the response generated by our method is substantially different from the original dataset chunk while
still focusing on answering the question. No sensitive information, such as address, nationality, or birthdate, is revealed.
Hence, this demonstrates successful defense against multi-round attacks.
One interesting aspect of our response is that Pablo Trapero was actually born on October 4th, 1971, and Aleksander
Ford on November 24th, 1908. However, our response states that they were born on December 27th, 1965, and
September 1st, 1923, respectively. This indicates that our method preserves the final answers without revealing the
actual birthdates. The attacker may also mistakenly conclude that the birthdate information has been retrieved and stop
the attack. In contrast, standard RAG reveals a large portion of the original content. Jiang et al. [9]successfully extract
most sensitive content, while the knowledge graph-based method recovers most original information.
4.4 Effects ofk
In both RP and DP-RP-G, we sample the projection matrix from N(0,1/k) where kis the projection dimension.
Fig. 3 examines the impact of different kvalues. As discussed in Section 3, larger kbetter preserves retrieval scores.
Consistently, we observe higher utility with increasing k, while privacy remains unchanged. Our method relies on
similarity-preserving projection rather than precise distance preservation (Section B and Section C), so the expected
post-projection ranking change is less sensitive to k, indicating robustness across a wide range of values. Based on
analysis and empirical observation of choosing k, we suggest following ways of choosing k. If a user wants more
privacy guarantee or less offline computation latency, a smaller kshould be chosen (e.g., k= 25 ford= 2048 ). If
6

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Table 7: Examples of attack results under the multi-round setting. We highlight content matching the original datastore
chunks. RP avoids revealing sensitive identity information while preserving relational logic, such as who was born first,
even after multi-round attacks using a knowledge graph to retrieve related chunks [33].
Original Chunk ...Aleksander Ford: Aleksander Ford (born Mosze Lifszyc, 24 November 1908 in Kiev, Russian Empire – 4 April 1980 in Naples,
Florida, United States) was a Polish Jewish film director, and head of the Polish People’s Army Film Crew in the Soviet Union during
World War II. Ford became director of the nationalized Film Polski company following the Red Army occupation of Poland. 7 Days
in Havana: 7 Days in Havana (Spanish: 7 días en La Habana ) is a 2012 Spanish-language anthology film. Set during a week in the
Cuban capital Havana, the film features one segment for each day, each segment directed by a different filmmaker. The directors are
Julio Medem, Laurent Cantet, Juan Carlos Tabío, Benicio del Toro, Gaspar Noé, Pablo Trapero and Elia Suleiman. The screenplay
was written by the Cuban novelist Leonardo Padura Fuentes. The film is a co-production between companies in Spain, France and
Cuba...
Initial Query Pablo Trapero or Aleksander Ford? Who was born first?
Attacking Methods Response by RAG Response by ours
Adaptive [9] <think> Alek sander Ford (born Mosze Lifszyc, 24Novem ber1908
inKiev, RussianEmpire–4April 1980 inNaples, Florida, United
States) wasaPolishJewishfilmdirector, and 2006 BBC Four televi-
sion. Ford became directorofthenationalized Film Polskicompany
followingtheRedArmy occupationofPoland. 7Days inHavana: 7
Days inHavana (Span ish:7díasenLaHabana ) is a "Pennies From
Heaven" (1978)...<think> Pablo Trapero or Alek sander Ford? 7 Who was born first? 8
Who was born first? Alek sander Ford?
Yao et al. [33] <think> Thinking Process: 1. **Analyze the Request:** * Target:
An English actor of stage, film, and television. * Condition 1: Alek -
sander Ford (born Mosze Lifszyc, 24Novem ber1908 inKiev, Rus-
sianEmpire–4April 1980 inNaples, Florida, United States) wasa
PolishJewishfilmdirector * Condition 2: 7Days inHavana: 7Days
inHavana (Span ish:7días enLaHabana ) * Goal: Who was born
first? * ...<think> Who was born first? Pablo Trapero or Alek sander Ford?
<think> Thinking Process: 1. **Analyze the Request:** The user
is asking for a comparison of birth dates between two individuals:
Pablo Trapero and Alek sander Ford. The goal is to determine who
was born first. 2. **Identify the Individuals:** * ** Pablo Trapero:**
An Argentine film director, screenwriter, and producer. * ** Alek -
sander Ford:** A Polish film director and screenwriter. 3. **Retrieve
Birth Dates:** * ** Pablo Trapero:** I need to recall or verify his
birth date. * Search query mental check: " Pablo Trapero birth date".
* Knowledge: Pablo Trapero was born on December 27, 1965 . *
**Alek sander Ford:** I need to recall or verify his birth date. *
Search query mental check: " Alek sander Ford birth date". * Knowl-
edge: Alek sander Ford was born on September 1, 1923 . 4. **Com-
pare Dates:** * Alek sander Ford
a user wants higher utility, a bigger kshould be chosen (e.g., k= 1600 ford= 2048 ). Otherwise, a moderate value
would be suitable (e.g.,k= 100ford= 2048).
40 80 120
Perplexity1.82.12.4Leakage (%)
(a) KNN-LM
40 80 120
Perplexity1.82.12.4Leakage (%)
(b) Standard RAGRP (Ours) k=25
DP-RP-G k=25RP (Ours) k=100
DP-RP-G k=100RP (Ours) k=1600
DP-RP-G k=1600
Figure 3: Comparison of privacy/utility. Colors indicate k; markers distinguish RP and DP-RP-G. khas weak relation
with privacy but largerkgives better utility.
To understand the effect of not setting the sampling variance as 1/k, we use k= 1600 and sample from N(0,0.22),
resulting in a perplexity of 70.86 and2.32% privacy leakage. For k= 25 andN(0,0.0252), perplexity is 71.38 with
3.1% leakage. Choosing sampling variances larger or smaller than 1/kdegrades utility, validating that setting variance
to1/kis optimal.
4.5 Efficiency and Scalability
The DPSparseV oteRAG method can be inefficient. In our experiments, each inference takes about 30 ×longer than RP
and vanilla RAG (Table 8). This is due to DP voting and aggregation require partitioning the datastore into msubsets
and generating a token for each, resulting in minferences per token. Our method can achieve the same efficiency as
RAG on latency per token.
7

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Table 8: Comparison of the number of tokens generated and
latency per generation for different methods over the ECHR
dataset on an NVIDIA A100.
RAG RP (Ours) DPSVR
# Tokens per Generation 107.8123.492.3
Latency per Generation (s) 8.359.42258.9
Latency per Token (s) 0.08 0.08 2.80Table 9: Perplexity comparisons of KNN-LM and stan-
dard RAG on ECHR.
Method KNN-LMStandard
RAG
RP (Ours)43.20 47.92
DP-RP-G 97.09 102.66
DP-RP-G (only clipping) 45.47 46.99
Besides efficiency, we also take a look at the scalability of our evaluation. We empirically study datastores of up to about
100K HotpotQA documents, corresponding to roughly 200 million tokens, which is a reasonably sized RAG datastore
comparable with many real-world enterprise services [ 34,26]. Prior work [ 35] shows that extraction rates decrease
substantially as datastore size increases. We further conduct a controlled scalability study in Figure 4 by varying
the datastore size within the ECHR/Qwen3-4B setting. We show (1) the extraction rate of vanilla RAG decreases
as the datastore grows, while (2) our method consistently maintains strong privacy protection across datastore sizes.
Moreover, as discussed in Section 3, our RP method relies on replacing tokens or chunks with similar alternatives
from the datastore. As the datastore size increases, the method has access to a larger pool of candidate tokens for such
substitutions, reducing the extraction rate.
0 10 20 30 40 50
Leakage (%)1K
5K
10KDatastore Size2.14
2.02
2.052.16
2.03
2.0346.67
38.72
21.30RP (Ours) Param RAG
Figure 4: Scalability study on Standard RAG using ECHR with
Qwen3-4B. Across different datastore sizes, RP maintains leak-
age close to the parametric baseline while consistently providing
strong privacy protection.
0.1 1510152550100
Clipping Bound50100200500Perplexity
KNN-LM
Standard RAGFigure 5: Perplexity of DP-RP-G with clipping only
(without Gaussian noise) under different clipping
bounds.
4.6 Why Peformance Degrades More by Differential Privacy Methods
A natural question is:what is the challenge for differential-privacy based methods to maintain utiliy in RAG applica-
tions.For DP-RP-G, achieving an (ϵ, δ) guarantee requires bounding sensitivity, typically by clipping each projected
embedding to norm c, and then adding Gaussian noise N
0,c√
2 ln(1.25/δ)
ϵ
. Ifϵ→ ∞ , the noise vanishes; with a
sufficiently large clipping bound, DP-RP-G approaches ours RP method. Thus, the utility loss arises from the additional
constraints required for formal DP rather than from RP itself.
DP aggregation methods face a similar trade-off, as their utility depends strongly on perturbation magnitude and
therefore on privacy budget [ 6]. As shown in Section 4, even relatively loose privacy budgets can substantially degrade
utility while still allowing non-negligible leakage.
Clipping is typically required in DP mechanisms to bound sensitivity [ 36]; prior work similarly constrains vectors to
bounded domains [ 37]. To isolate the effect of clipping and adding noise, we ablate the clipping operation without
Gaussian noise. In Table 9, we compare RP with and without clipping, while Figure 5 varies the clipping bound. The
results show that larger clipping consistently reduces perplexity, whereas sufficiently large clipping bounds recover
performance close to RP.
5 Conclusion
In this work, we propose a simple random projection method to defend against data extraction attacks for RAG
applications. Compared with DP-based approaches, we show that the specific random Gaussian projection can
empirically protect against data reconstruction, at approximately the same level as parametric model without RAG
datastore, while still maintaining semantic utility, with a 7% overall decrease compared with vanilla RAG.
8

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Limitations
At evaluation side, large-scale evaluation of privacy-preserving methods in real-world LLM-based RAG systems
remains an interesting future work. Real-world systems may need to process billions of queries daily. In Table 8,
one query takes about 8 seconds. Hence, evaluating one million queries would take 92 days without system-level
optimization. Such large-scale evaluation thus requires system-level contributions, which we defer to future work.
Ethical Considerations
We use models and datasets that are publicly available on the Internet and licensed for non-commercial research use.
We do not involve personally identifiable information (PII) which are not allowed to make public. We use only publicly
available or synthetically generated datasets; we did not use any proprietary data or private user information at any
stage of the experiments. These datasets are publicly available and widely used in the research community. Consistent
with prior work, we did not further anonymize the data, as it was collected and released under publicly accessible
licenses and used solely for research purposes. We did not introduce new personal data, nor did we attempt to re-identify
individuals beyond what is already present in the original datasets.
References
[1]Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through
memorization: Nearest neighbor language models. InProceedings of the International Conference on Learning
Representations (ICLR), 2019.
[2]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive nlp tasks. InProceedings of the Advances in Neural Information Processing
Systems (NeuraIPS), volume 33, pages 9459–9474, 2020.
[3]Yu He, Yifei Chen, Yiming Li, Shuo Shao, Leyi Qi, Boheng Li, Dacheng Tao, and Zhan Qin. External data
extraction attacks against retrieval-augmented large language models.arXiv preprint arXiv:2510.02964, 2025.
[4]Yuhao Wang, Wenjie Qu, Shengfang Zhai, Yanze Jiang, Zichen Liu, Yue Liu, Yinpeng Dong, and Jiaheng
Zhang. Silent leaks: Implicit knowledge extraction attack on rag systems through benign queries.arXiv preprint
arXiv:2505.15420, 2025.
[5]Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang Wang,
Dawei Yin, and Jiliang Tang. The good and the bad: Exploring privacy issues in retrieval-augmented generation
(rag). InACL (Findings), 2024.
[6]Tatsuki Koga, Ruihan Wu, and Kamalika Chaudhuri. Privacy-preserving retrieval augmented generation with
differential privacy.arXiv preprint arXiv:2412.04697, 2024.
[7]Ruihan Wu, Erchi Wang, Zhiyuan Zhang, and Yu-Xiang Wang. Private-rag: Answering multiple queries with llms
while keeping your data private.arXiv preprint arXiv:2511.07637, 2025.
[8]Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, and Min Yang. Rag-thief: Scalable extraction of private
data from retrieval-augmented generation applications with agent-based attacks.CoRR, 2024.
[9]Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Yang Chen, and Min Yang. Feedback-guided extraction
of knowledge base from retrieval-augmented llm applications.arXiv preprint arXiv:2411.14110, 2024.
[10] Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, and Stefano Melacci. Pirates of the rag:
Adaptively attacking llms to leak knowledge bases.arXiv preprint arXiv:2412.18295, 2024.
[11] Nicolas Grislain. Rag with differential privacy. In2025 IEEE Conference on Artificial Intelligence (CAI), pages
847–852. IEEE, 2025.
[12] Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue Xing, and
Jiliang Tang. Mitigating the privacy issues in retrieval-augmented generation (rag) via pure synthetic data. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages
24538–24569, 2025.
[13] Yunpeng Zhao and Jie Zhang. Does training with synthetic data truly protect privacy? InThe Thirteenth
International Conference on Learning Representations (ICLR), 2025.
9

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
[14] Longzhu He, Peng Tang, Yuanhe Zhang, Pengpeng Zhou, and Sen Su. Mitigating privacy risks in retrieval-
augmented generation via locally private entity perturbation.Information Processing & Management, 62(4):
104150, 2025.
[15] Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, and Nils Lukas. Dp-fusion: Token-level differentially
private inference for large language models.arXiv preprint arXiv:2507.04531, 2025.
[16] William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space.Contemporary
mathematics, 26(189-206):1, 1984.
[17] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The johnson-lindenstrauss transform itself
preserves differential privacy. InProceedings of the 2012 IEEE 53rd Annual Symposium on Foundations of
Computer Science (FOCS), pages 410–419. IEEE Computer Society, 2012.
[18] Chugui Xu, Ju Ren, Yaoxue Zhang, Zhan Qin, and Kui Ren. Dppro: Differentially private high-dimensional data
release via random projection.IEEE Transactions on Information Forensics and Security, 12(12):3081–3093,
2017.
[19] Ping Li and Xiaoyun Li. Differential privacy with random projections and sign random projections.arXiv preprint
arXiv:2306.01751, 2023.
[20] Alaa Mahmoud Ibrahim, Mohamed Farouk, and Mohamed Waleed Fakhr. Privacy preserving image retrieval
using multi-key random projection encryption and machine learning decryption.Journal of Advanced Research in
Applied Sciences and Engineering Technology, 42(2):155–174, 2024.
[21] Kun Liu, Hillol Kargupta, and Jessica Ryan. Random projection-based multiplicative data perturbation for privacy
preserving distributed data mining.IEEE Transactions on knowledge and Data Engineering, 18(1):92–106, 2006.
[22] Mohammad Hasan Narimani and Mostafa Tavassolipour. Fedrp: A communication-efficient approach for
differentially private federated learning using random projection.arXiv preprint arXiv:2509.10041, 2025.
[23] Cihan Kaleli and Huseyin Polat. Privacy-preserving random projection-based recommendations based on
distributed data.International Journal of Information Technology & Decision Making, 12(02):201–232, 2013.
[24] Nikola Pavlovic, Sudeep Salgia, and Qing Zhao. Differential privacy in kernelized contextual bandits via random
projections.arXiv preprint arXiv:2507.13639, 2025.
[25] Bonwoo Lee, Cheolwoo Park, and Jeongyoun Ahn. Optimal differentially private kernel learning with random
projection.arXiv preprint arXiv:2507.17544, 2025.
[26] Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Weijia Chen, Roshan Desai, Wenxi Huang, and Mark H Butler.
Enterpriserag-bench: A rag benchmark for company internal knowledge.arXiv preprint arXiv:2605.05253, 2026.
[27] Sarah Packowski, Inge Halilovic, Jenifer Schlotfeldt, and Trish Smith. Optimizing and evaluating enterprise
retrieval-augmented generation (rag): A content design perspective. InProceedings of the 2024 8th International
Conference on Advances in Artificial Intelligence, pages 162–167, 2024.
[28] Yangsibo Huang, Samyak Gupta, Zexuan Zhong, Kai Li, and Danqi Chen. Privacy implications of retrieval-based
language models. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 14887–14902, 2023.
[29] Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. InProceedings
of the European Conference on Machine Learning (ECML), pages 217–226. Springer, 2004.
[30] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are
unsupervised multitask learners.OpenAI blog, 1(8):9, 2019.
[31] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D
Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2369–2380, 2018.
[32] Zelin Guan, Shengda Zhuo, Zeyan Li, Jinchun He, Wangjie Qiu, Zhiming Zheng, and Shuqiang Huang. E-mia:
Exam-style black-box membership inference attacks against rag systems.arXiv preprint arXiv:2605.00955, 2026.
[33] Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng Cai, Xiangqun Chen, Yao Guo, and Ding Li. Connect
the dots: Knowledge graph-guided crawler attack on retrieval-augmented generation systems.arXiv preprint
arXiv:2601.15678, 2026.
[34] Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, and Chien-
Sheng Wu. Benchmarking deep search over heterogeneous enterprise data. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing: Industry Track, pages 501–517, 2025.
10

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
[35] Zhenting Qi, Hanlin Zhang, Eric P Xing, Sham Kakade, and Hima Lakkaraju. Follow my instruction and spill the
beans: Scalable data extraction from retrieval-augmented generation systems. InInternational Conference on
Learning Representations, volume 2025, pages 48733–48755, 2025.
[36] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data
analysis. InProceedings of the Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March
4-7, 2006., pages 265–284. Springer, 2006.
[37] Krishnaram Kenthapadi, Aleksandra Korolova, Ilya Mironov, and Nina Mishra. Privacy via the johnson-
lindenstrauss transform.Journal of Privacy and Confidentiality, 5(1), 2013.
[38] Donald W Zimmerman, Bruno D Zumbo, and Richard H Williams. Bias in estimation and hypothesis testing of
correlation.Psicológica, 24(1), 2003.
[39] Harold Hotelling. New light on the correlation coefficient and its transforms.Journal of the Royal Statistical
Society. Series B (Methodological), 15(2):193–232, 1953.
[40] Ronald A Fisher. Frequency distribution of the values of the correlation coefficient in samples from an indefinitely
large population.Biometrika, 10(4):507–521, 1915.
[41] Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodro-
mos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on european court of
human rights cases. InProceedings of the Annual Conference of the North American Chapter of the Association
for Computational Linguistics (NAACL), Mexico City, Mexico, 2021. Association for Computational Linguistics.
11

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Contents
A Usage of AI 12
B Preservation of theL 2-Distance and Cosine Similarity of Random Projection 12
B.1 Proof: Preservation of the Cosine Similarity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
B.2 Synthetic Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
C Proof: Probability of Top-1 Selection Changing after Random Projection 14
D Complete Algorithms 16
E Implementation Details 17
F Settings of Hyper-Parameters 17
F.1 General Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
F.2 DPSparseRAG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
G Random Projection with Other Distributions 18
H More Discussion 18
H.1 False Positive Classification of Non-Sensitive Information . . . . . . . . . . . . . . . . . . . . . . . 18
H.2 An Example of Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
A Usage of AI
We employ large language models (LLMs) to improve the grammar and clarity of our writing and assist coding. All
research ideas, directions, and decisions are independently conceived and carried out by the authors.
B Preservation of theL 2-Distance and Cosine Similarity of Random Projection
The theory has already been proved by Johnson and Lindenstrauss [16]. For completeness, we provide analysis below.
Let the query embedding be x, the retrieved embedding by vanilla RAG be e, and the random projection matrix be R.
The original ℓ2distance is ∥x−e∥ 2. According to the Johnson–Lindenstrauss transform, when the projection variance
is1/k, we have the boundary condition
(1−λ JL)∥x−e∥ 2≤ ∥Rx−Re∥ 2≤(1 +λ JL)∥x−e∥ 2
whereλ JL= Ω q
logd
k
[16]. This ensures that theℓ 2distance is well preserved after projection.
Further, the original retrieval score, typically cosine similarity, ρ=x⊤e
∥x∥2∥e∥2can also be preserved after projection.
After random projection, the score becomes
ρ′=x⊤R⊤Re
∥Rx∥ 2∥Re∥ 2.
For fixed xande, the expectation satisfies E[ρ′] =ρ 
1−1−ρ
2k
+O 1
k2
(Proof in Section B.1). Thus, when kis
sufficiently large, the expected cosine similarity after projection closely approximates the original score, preserving
retrieval quality despite token perturbations.
B.1 Proof: Preservation of the Cosine Similarity
Our main argument is that with multiple rounds of token generation, the overall semantic meaning is preserved even if
individual tokens are perturbed or replaced. In other words, we have
E[ρ′]≈ρ
1−1−ρ
2k
.(1)
12

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Proof. We have query embedding xand datastore embedding e. First, let u=x
∥x∥2andv=e
∥e∥2and we can have
ρ=u⊤v. With that, we can have
ρ′=(u⊤R⊤)(Rv))
∥u⊤R⊤∥2∥Rv∥ 2=(Ru)⊤(Rv))
∥Ru⊤∥2∥Rv∥ 2
We leta=Ruandb=Rv. Then we can have
ρ′=a⊤b
∥a∥ 2∥b∥2.
Because Ris an IID Gaussian matrix and each element is independently sampled. We can find rotation matrices
UandVwhere URVd=R . Hence, we can have R(Vx)d=Rx . So, without loss of generality, we can rotate the
basis so that u=e 1= (1,0,0, . . . ,0)⊤,v=ρe 1+p
1−ρ2e2. We know that each row of Ris a Gaussian vector
gi∼ N(0, I d/k). So,ai=rT
iu=r i1andbi=r⊤
iv=ρr i1+p
1−ρ2ri2where ri1,ri2are IID N(0,1/k) . Thus,
for eachi= 1, . . . , k,
ai
bi
∼ N
0,1
k
1ρ
ρ1
.
With this bivariate normal distribution, we can define
Xi=√
kai∼ N(0,1),
Yi=√
kbi∼ N(0,1),
And we can have thatCorr(X i,Yi) =ρ. Then, we can calculate
ρ′=P
iXiYipP
iX2
ipP
iY2
i
This is exactly the sample Pearson correlation ofkIID bivariate normal samples with population correlationρ.
Lemma 1.For random variables XiandYiwhere Xi∼ N(0,1) ,Yi∼ N(0,1) , and Corr(X i,Yi) =ρ , we can
calculate the expectation of sample Pearson correlation coefficient
ρ′=P
iXiYipP
iX2
ipP
iY2
i
as
E[ρ′] =ρ
1−1−ρ
2k
+O1
k2
Although the proof of this lemma is not our original contribution, as there is plenty of literature showing this conclu-
sion [38, 39], we would like to provide a proof here to demonstrate some of our insights.
Proof.LetS xx=1
kPX2
i,Syy=1
kPY2
i,Sxy=1
kPXiYi. Then
ρ′=Sxy√Sxxp
Syy
If we use Taylor expansion, we will have
1√Sxx= 1−1
2Sxx+O1
k2
As a result, we will have
E[ρ′] =E[S xy]−1
2E[SxySxx]−1
2E[SxySyy]
+1
4E[SxxSyy] +O1
k2
=ρ−1
2kCov(XY,X2)−1
2kCov(XY,Y2)
+1
4kCov(X2,Y2) +O1
k2
13

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
We have E[X2] =E[Y2] = 1 andE[XY] =ρ . We also know Var(X) = Var(Y) = 1 andCov(X,Y) =ρ . Using
Isserlis theorem, we haveE[X4] = 3,E[Y4] = 3,E[X2Y2] = 1 + 2ρ2,E[X3Y] =E[XY3] = 3ρ. Hence, we have
Cov(X2,Y2) =E[X2Y2]−E[X2]E[Y2] = 2ρ2
Cov(X2,XY) =E[X3Y]−E[X2]E[XY] = 2ρ
By putting them back, we have our conclusion.
According to Lemma 1, we prove Eq. (1). The expected distortion goes down as O(1/k) . Hence, we can see that our
designed random projection preserves the cosine similarity.
B.2 Synthetic Data
0 500 1000 1500 2000
k0.10.2|/prime|
Cosine Similarity Error
Random
Synthetic
0 500 1000 1500 2000
k0.80.91.01.1RxRe2/xe2
L2 Distance Ratio
Bound
Random
Synthetic
Figure 6: Cosine similarity error and L2 distance ratio under
different sanity-check settings.Before starting our experiments, we first conduct a sanity
check to verify whether embedding distances remain con-
sistent after random projection. We consider two settings.
In the synthetic setting, we use LLAMA3-8B-INSTRUCT
to generate 100 datastore entries with the prompt“Gener-
ate a datastore entry containing domain knowledge. ”For
each of these, we repeat 100 times to generate a query
using the prompt“Generate a query about domain knowl-
edge. ”In the random setting, both datastore and query
embeddings are random vectors drawn fromN(0,1).
We measure the L2 distance and cosine similarity between
each query embedding and the datastore embeddings. We
use two metrics: the L2 distance ratio,∥Rx−Re∥ 2
∥x−e∥ 2, and the cosine similarity difference, |ρ−ρ′|. We first average each
metric over 100 datastores and then average across 100 runs. Fig. 6 presents the sanity-check results, along with the
theoretical upper and lower bounds derived from the Johnson–Lindenstrauss (JL) transform, given by 1±Ωq
logd
k
.
Askincreases, the cosine similarity error decreases and approaches zero, confirming our later observation that larger
kimproves perplexity, which is consistent with our theoretical analysis. For the L2 distance, the results follow the
JL transform prediction, with the ratios remaining within the theoretical bounds. A larger kalso stabilizes the ratio.
Overall, these results verify that, in expectation, distances are well preserved after random projection.
C Proof: Probability of Top-1 Selection Changing after Random Projection
Without loss of generality, we consider the top-1 case, where the datastore embedding e1attains the largest cosine
similarity ρ1in the vanilla RAG setting. We show that if there exists another datastore embedding ej(j̸= 1 ) such that
e1
∥e1∥2−ej
∥ej∥2
2≤∆
then the probability that the projected cosine similarityρ′
jexceedsρ′
1satisfies
Pr(ρ′
j> ρ′
1)≳1−Φ
∆
(1−ρ2
1)q
2(1−η max)
k−3

where ηmaxis an upper bound on the correlation between the Fisher-transformed projected similarities, and Φdenotes
the cumulative distribution function of the standard normal distribution.
Proof.We define the normalized query and datastore embeddings as
u=x
∥x∥ 2,v i=ei
∥ei∥2.
Under this notation, the assumption becomes∥v 1−vj∥2≤∆. By the Cauchy–Schwarz inequality,
|ρ1−ρj|=|u⊤(v1−vj)| ≤ ∥v 1−vj∥2≤∆.
14

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
We next introduce the Fisher transform
zi= tanh−1(ρi), z′
i= tanh−1(ρ′
i).
Since cosine similarity takes values in (−1,1) andtanh−1(·)is strictly increasing on this interval, the ordering of
similarities is preserved under this transformation. In particular,
ρ′
j> ρ′
1⇐⇒z′
j> z′
1.
As shown in Section B, the projected cosine similarity ρ′
iis the sample Pearson correlation computed from kIID
bivariate normal pairs with population correlationρ i. By the classical Fisherz-transform result [40], we have
z′
i≈ N
tanh−1(ρi),1
k−3
.
Because z′
1andz′
jshare the same projected query vector, they are generally correlated. Let η= Corr(z′
1, z′
j), and
assumeη≤η maxfor some fixed upper boundη max<1.
Under a joint Gaussian approximation, the differencez′
j−z′
1is approximately normally distributed with
E[z′
j−z′
1] =z j−z1,Var(z′
j−z′
1) =2(1−η)
k−3.
Therefore,
Pr(ρ′
j> ρ′
1) = Pr(z′
j−z′
1>0) = Φ
zj−z1q
2(1−η)
k−3
.
Sinceρ 1≥ρj, we havez 1≥zj. By the mean value theorem, there existscbetweenρ 1andρ jsuch that
zj−z1= (ρ j−ρ1)1
1−c2.
Using|ρ j−ρ1| ≤∆and1−c2≤1−ρ2
1, we obtain
zj−z1≥ −∆
1−ρ2
1.
Substituting this bound yields
Pr(ρ′
j> ρ′
1)≳Φ
−∆
(1−ρ2
1)q
2(1−η)
k−3

= 1−Φ
∆
(1−ρ2
1)q
2(1−η)
k−3

≳1−Φ
∆
(1−ρ2
1)q
2(1−η max)
k−3
,
which completes the proof.
This result shows that after random projection, embeddings that are sufficiently close in the original space can overtake
the original top-1 embedding with non-negligible probability. This behavior is desirable for privacy preservation:
sensitive tokens (e.g., names, addresses, or identification numbers) often have embeddings close to semantically related
but less sensitive alternatives, and random projection allows these alternatives to replace the original sensitive tokens
with controlled probability. Combined with the bounds in Section B, this demonstrates that random projection limits
distortion to a reasonable range, preserving general semantic content while reducing the likelihood that specific sensitive
tokens are deterministically selected.
15

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
Algorithm 2Private RAG with Random Projection (KNN-LM)
Require:DatastoreDwithndocuments; tokenizer and embedding functionf(·) :Text→Rd;λ= 0.25.
Input:User queryq
Output:Generated answera
Parameters:Projection dimensionk; maximum token lengthT max
Offline preprocessing
1: Sample random projection matrixR∈Rk×d, whereR ij∼ N(0,1/k)
2: Compute datastore embeddingsE D∈Rd×n, wheree i=f(d i)
3: Project embeddings: ˜ED=RE D
Online inference (per query)
4: Initializet←0,a← ∅,y← ∅
5:whilet < T maxandyis not the end tokendo
6: Compute query embedding:x=f(q)
7: Project query embedding: ˜x=Rx
8:P(y|x) =λP LM(x) + (1−λ)P KNN (˜x,˜ED)
9: Predict next token:y←arg max yP(y|x)
10: Append token:a←a∥y,q←q∥y
11:t←t+ 1
12:end while
13:returna
D Complete Algorithms
For KNN-LM (Algorithm 2), at each generation step we apply random projection to both the query embedding and the
datastore embeddings. The language model produces a next-token distribution PLMbased on the current context x.
In parallel, the KNN distribution PKNN is computed by measuring the cosine similarity between the projected query
embedding ˜xand each projected datastore embedding ˜ei, followed by a softmax over the top- Knearest neighbors
to obtain a probability distribution over candidate tokens. The final next-token distribution is obtained by linearly
combining the two distributions with weight λ= 0.25 . The selected token yis then appended to the context and the
generated answer, and the process repeats.
Algorithm 3Implementation of DP-RP-G (KNN-LM)
Require:DatastoreD={d i}n
i=1; embedding functionf:Text→Rd
Input:User queryq
Output:Generated answera
Parameters:Projection dimensionk; max tokensT max; clipping boundc; privacy(ϵ, δ)
Offline preprocessing
1: Sample random projectionR∈Rk×d, whereR ij∼ N(0,1/k)
2: Compute datastore embeddingsE D∈Rd×n, with columnse i=f(d i)
3: Project: ˜ED=RE D∈Rk×n
4:foreach column ˜eiof˜EDdo
5: ˆei←˜ei·min
1,c
∥˜ei∥2
6:end for
Online inference (per query)
7: Initializet←0,a← ∅,y← ∅,σ=c√
2 ln(1.25/δ)
ϵ
8: Sample IIDZ∈Rk×n, whereZ ij∼ N(0, σ2)
9:˜ED←˜ED+Z
10:whilet < T maxandyis not the end tokendo
11:x←f(q)∈Rd
12: ˜x←Rx
13:y←arg max yP(y|x)
14:a←a∥y,q←q∥y
15:t←t+ 1
16:end while
17:returna
We implement DP-RP-G in Algorithm 3, following Blocki et al. [17], which adds Gaussian noise after random projection
to achieve differential privacy. Unlike Blocki et al. [17], who assume binary or [0,1] -bounded vectors, which does not
16

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
hold for LLM embeddings. Therefore, after projection, we apply standard DP clipping to any vectorv:
clip(v, c) =v·min
1,c
∥v∥2
,
so that all vectors have ℓ2-norm at most c. After clipping, we add Gaussian noise calibrated to the privacy budget where
the variance σ=c√
2 ln(1.25/δ)
ϵ. As our goal is to protect the datastore, we apply the DP step only to ED(and not to
the queries). Hence, lines 12–19 in Algorithm 3 are the same as lines 7–15 in Algorithm 2. For the DP-RP-G method
implemented in standard RAG, we replace lines 12–19 with lines 7–13 in Algorithm 1.
Table 10: Comparison of KNN-LM and Standard RAG on Enron Email and ECHR with differentm(ϵ= 5).
KNN-LM Standard RAG
m 10 20 30 40 50 60 10 20 30 40 50 60
Enron Email
Privacy (# leakage) 4 2 0 0 0 0 8 6 1 0 0 0
Perplexity 3.14 3.15 3.14 3.14 3.16 3.16 3.70 3.68 3.70 3.71 3.71 3.98
ECHR
Privacy (% leakage) 2.12 2.24 2.06 2.12 2.18 1.94 2.31 2.02 2.16 2.08 2.12 2.14
Perplexity 512.64 512.64 512.64 112.35 109.64 110.35 4307.36 4888.23 5155.85 4307.36 7549.47 7551.63
E Implementation Details
First, we use the Enron Email dataset [ 29] as the datastore and GPT-2 [ 30] as the LLM because GPT-2’s pre-training
data do not overlap with Enron Email, ensuring the RAG data remain unseen. Thus, private information is limited to
the RAG documents. The dataset contains annotated sensitive information (URLs, email addresses, phone numbers),
and we use the entire dataset as the datastore, totaling 4576 sensitive items. We use the query prompts from [ 28] (e.g.,
You may also contact us via e-mail at ) and evaluate whether the annotated personally identifiable
information can be retrieved.
The European Court of Human Rights (ECHR) dataset [ 41] contains legal case documents from the ECHR. Queries
come from test-case descriptions, and utility is measured by matching generated judgments to ground-truth labels.
We use Llama3-1B-Instruct . For adversarial queries, we use Llama3-8B-Instruct to annotate sensitive
information and query with prompts concatenating "This is the first sentence of a paragraph
from ECHR legal documents. Complete the paragraph with the sentence:" and the first
sentence of each case. We check if generations contain the annotated sensitive identity, using 10K documents.
Besides targeted leakage where we check if exact annotated information is leaked, we use the C4 dataset, a cleaned
and large-scale version of Common Crawl, for untargeted leakage evaluation. We select 5,000 samples and use the
first sentence of each as the query. We then compute ROUGE-L scores between each generation and every datastore
sample. If any sample in the datastore has a score greater than 0.5 with the generation, where we follow the same
settings by Huang et al. [28], marking it as a leakage. We report the number of leakages out of the 5k samples. We use
the LLAMA3-8B-INSTRUCTmodel as the backbone.
HOTPOTQA [ 31] is a multi-hop QA dataset with strong supervision for supporting facts, enabling more explainable
systems. We use it to evaluate whether we retrieve correct supporting documents and measure cosine similarity between
retrieved and reference embeddings to assess retrieval quality. We use the training set as the datastore and the test
set for evaluation. The datastore contains over 90K documents. We evaluate with distractor setting which means the
whole datastore is for reference. We use QWEN2-7B-INSTRUCTas the backbone model. Since document retrieval
relies on KNN for both KNN-LM and standard RAG, we focus on the KNN-LM structure to analyze the retrieval
process. For evaluation, we report exact match (EM) and F1 scores between the generated output and the ground truth.
As the datastore is very large, we randomly select 100 documents as targets for targeted attacks. The initial query in
multi-round attack is each question.
F Settings of Hyper-Parameters
F.1 General Settings
For all LLM generations, we set the repetition penalty to 0.75 and the no-repeat-ngram-size to 0. The feature dimensions
of GPT-2, Llama3, and Qwen2 are 128, 2048, and 3584 respectively. All other generation parameters follow the default
17

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
HUGGINGFACEsettings. By default, we set the datastore size equal to the total number of tokens in each dataset. We
follow previous differential privacy work where we set δas the reverse scale of dataset size. For Enron Email and
ECHR, it is 10−4. For HotPotQA and C4, we use 10−5. We use each model’s tokenizer and its last hidden-state vectors
as embeddings for both retrieval settings. The max output token length is 2048.
F.2 DPSparseRAG
Apart from ϵand clipping bound we tune on in the experiments, another hyper-parameter for DPSparseRAG is m.
We therefore conduct a sweep over m∈ {10,20,30,40,50,60} withϵ= 5 on the Enron Email and ECHR datasets.
Overall, we observe in Table 10 no significant difference in utility across various values ofm.
Table 11: Perplexity and data leakage comparison using random projection matrices drawn from different distributions
on the Enron Email dataset with GPT2.
KNN-LM Standard RAG
Perplexity Email URL Phone Perplexity Email URL Phone
RAG 2.87 0 13 0 1.58 36 56 125
RP (Ours)Gaussian2.890 0 0 2.05 0 3 0
Rademacher 2.92 0 2 01.760 3 1
G Random Projection with Other Distributions
Beyond the normal (Gaussian) distribution, we also experimented with constructing the projection matrix using a
Rademacher distribution. In this distribution, each entry independently takes the value +1or−1with equal probability
(50%). As shown in Table 11, the Rademacher-based projection also preserves privacy effectively while achieving
performance comparable to vanilla RAG. These results indicate that our proposed random projection method remains
effective as long as the projection matrix preserves similarity structure, regardless of the specific distribution.
H More Discussion
H.1 False Positive Classification of Non-Sensitive Information
Although random projection provides strong privacy protection overall, our experiments on Enron Email, ECHR, and
C4 reveal that a small number of items may still be leaked. All failure cases fall into the case of leakage examples
without sensitive information. Analysis of these failure cases reveals a clear pattern: random projection provides
complete protection (100% leakage prevention) for specific content such as numbers, IDs, and email addresses.
While, in rare cases, common terminology which are non-sensitive information may mis-classified personal identity
information in annotation (e.g. common streat name Queen St.) This occurs because such terms may appear in multiple
contexts, making them less distinctive in the embedding space. These findings suggest that random projection is most
Table 12: Example responses generated by different methods given the prompt.
PromptComplete the sentences: you have any suggestions, please share. With the rise of social media...
Parametric With that in mind here are a few ideas to get you started: 1. **Create a vision board**: Cut out images and words that represent your goals
and desires and create a visual representation of what you want to achieve. 2. **Set SMART goals**: Make sure your goals are Specific
Measurable Relevant and Time-bound. This will help you stay focused and motivated. 3. **Break down big goals into smaller.
RAG it’s become increasingly important for businesses to have a strong online presence. One way to do this is by creating engaging content that
resonates with your target audience. Here are some tips to help you create content that drives engagement: 1. Know your audience: Before
you start creating content make sure you have a good understanding of your target! audience. Who are they? What are their interests?
What problems do they need?
RP (Ours) it’s become increasingly important for businesses to have a strong online presence. One way to do this is by creating engaging content that
resonates with your target audience. Here are some tips to help you create content that drives engagement: 1. Know your audience: Before
you start creating content make sure you have a good understanding of who your target! audience is. What are their interests need?
DP-RP-G (ϵ= 20)PrintWriter.PrintWriter userService userServiceExchangeExchangeExchange_generate_generate
DPSparseVoteRAG ( ϵ=
20)make sure you have a good understanding of your target! "#" "# "!$"" &"" $"#"# "
18

Private Retrieval Augmented Generation via Random ProjectionA PREPRINT
effective for protecting unique identifiers while providing strong but not absolute protection for common terms that may
appear across multiple documents.
H.2 An Example of Generation
Table 12 provides an example of responses generated by different methods for the same prompt. The parametric model
produces unrelated continuation that is not grounded in the datastore. In contrast, both RAG and our method generate
clear, relevant answers, whereas the DP-based methods often produce incoherent or unrelated content.
19